use std::ops::Range;

use crate::evidence::{EvidenceProvenance, EvidenceSource};
use crate::feature::{FeatureBundle, FeatureValue};
use crate::ids::{FeatureId, PhoneId};
use crate::phonology::PhoneToken;
use crate::prosody::{Stress, Syllable};
use crate::segment::{SegmentMatcher, SyllablePosition};
use crate::spec::Spec;
use crate::variety::LinguisticVariety;

pub fn syllabify_phones(phones: &[PhoneToken], variety: &LinguisticVariety) -> Vec<Syllable> {
    let mut syllables = Vec::new();
    for word in phone_words(phones) {
        syllables.extend(syllabify_word(word, variety));
    }
    syllables
}

fn phone_words(phones: &[PhoneToken]) -> Vec<&[PhoneToken]> {
    let mut words = Vec::new();
    let mut start = None;
    for (index, phone) in phones.iter().enumerate() {
        if is_word_boundary(phone) {
            if let Some(start_index) = start.take()
                && start_index < index
            {
                words.push(&phones[start_index..index]);
            }
            continue;
        }
        if !is_boundary_phone(phone) {
            start.get_or_insert(index);
        }
    }
    if let Some(start_index) = start {
        words.push(&phones[start_index..]);
    }
    words
}

fn syllabify_word(phones: &[PhoneToken], variety: &LinguisticVariety) -> Vec<Syllable> {
    let phones = phones
        .iter()
        .filter(|phone| !is_boundary_phone(phone))
        .cloned()
        .collect::<Vec<_>>();
    if phones.is_empty() {
        return Vec::new();
    }

    let nuclei = nucleus_spans(&phones);
    if nuclei.is_empty() {
        return vec![syllable(phones, Spec::Unspecified, Vec::new(), None)];
    }

    let mut syllables: Vec<Syllable> = Vec::with_capacity(nuclei.len());
    let mut prev_end = 0usize;
    for (syllable_index, nucleus) in nuclei.iter().enumerate() {
        let cluster = prev_end..nucleus.start;
        let (onset, coda) = if syllable_index == 0 {
            (cluster, 0..0)
        } else {
            split_maximum_onset(&phones, cluster, variety)
        };

        if syllable_index > 0 {
            let previous = syllables
                .last_mut()
                .expect("previous syllable exists after first nucleus");
            previous.phones.extend_from_slice(&phones[coda.clone()]);
            previous
                .phone_positions
                .extend(std::iter::repeat_n(SyllablePosition::Coda, coda.len()));
        }

        let mut syllable_phones = Vec::new();
        syllable_phones.extend_from_slice(&phones[onset.clone()]);
        syllable_phones.extend_from_slice(&phones[nucleus.clone()]);

        let mut positions = Vec::new();
        positions.extend(std::iter::repeat_n(SyllablePosition::Onset, onset.len()));
        positions.extend(std::iter::repeat_n(
            SyllablePosition::Nucleus,
            nucleus.len(),
        ));

        let stress = phone_stress(&phones[nucleus.start]);
        let nucleus_index = Some(onset.len());
        syllables.push(syllable(syllable_phones, stress, positions, nucleus_index));

        prev_end = nucleus.end;
    }

    if prev_end < phones.len() {
        let last = syllables
            .last_mut()
            .expect("at least one syllable exists after nucleus detection");
        last.phones.extend_from_slice(&phones[prev_end..]);
        last.phone_positions.extend(std::iter::repeat_n(
            SyllablePosition::Coda,
            phones.len() - prev_end,
        ));
    }

    syllables
}

fn split_maximum_onset(
    phones: &[PhoneToken],
    cluster: Range<usize>,
    variety: &LinguisticVariety,
) -> (Range<usize>, Range<usize>) {
    if cluster.is_empty() {
        return (cluster.clone(), cluster);
    }

    for split in cluster.start..=cluster.end {
        if is_legal_onset(&phones[split..cluster.end], variety) {
            return (split..cluster.end, cluster.start..split);
        }
    }

    (cluster.end..cluster.end, cluster.start..cluster.end)
}

fn is_legal_onset(cluster: &[PhoneToken], variety: &LinguisticVariety) -> bool {
    if cluster.is_empty() {
        return true;
    }
    if cluster.iter().any(is_nucleus) {
        return false;
    }
    if cluster.len() == 1 {
        return !is_illegal_single_onset(&cluster[0], variety);
    }

    variety.phonotactics.as_ref().is_some_and(|phonotactics| {
        phonotactics.constraints.iter().any(|constraint| {
            constraint.environment.syllable_position == Spec::Known(SyllablePosition::Onset)
                && constraint.id.contains(".legal_onset.")
                && phone_cluster_matches(cluster, &constraint.environment.before)
        })
    })
}

fn is_illegal_single_onset(phone: &PhoneToken, variety: &LinguisticVariety) -> bool {
    variety.phonotactics.as_ref().is_some_and(|phonotactics| {
        phonotactics.constraints.iter().any(|constraint| {
            constraint.environment.syllable_position == Spec::Known(SyllablePosition::Onset)
                && constraint.id.contains(".illegal_onset.")
                && segment_matcher_matches_phone(&constraint.matcher, phone)
        })
    })
}

fn phone_cluster_matches(phones: &[PhoneToken], matchers: &[SegmentMatcher]) -> bool {
    phones.len() == matchers.len()
        && phones
            .iter()
            .zip(matchers)
            .all(|(phone, matcher)| segment_matcher_matches_phone(matcher, phone))
}

fn segment_matcher_matches_phone(matcher: &SegmentMatcher, phone: &PhoneToken) -> bool {
    match matcher {
        SegmentMatcher::Any => true,
        SegmentMatcher::Phone(expected) => phone_matches_id(phone, expected),
        SegmentMatcher::FeatureBundle(expected) => {
            feature_bundle_matches(&phone.features, expected)
        }
        SegmentMatcher::Phoneme(_) | SegmentMatcher::Boundary(_) => false,
    }
}

fn nucleus_spans(phones: &[PhoneToken]) -> Vec<Range<usize>> {
    let mut spans = Vec::new();
    let mut index = 0usize;
    while index < phones.len() {
        if !is_nucleus(&phones[index]) {
            index += 1;
            continue;
        }

        let start = index;
        index += 1;
        while index < phones.len()
            && is_nucleus(&phones[index])
            && same_source(&phones[start], &phones[index])
        {
            index += 1;
        }
        spans.push(start..index);
    }
    spans
}

fn same_source(left: &PhoneToken, right: &PhoneToken) -> bool {
    left.span == right.span && phonology_base_symbol(left) == phonology_base_symbol(right)
}

fn is_nucleus(phone: &PhoneToken) -> bool {
    feature_bool(&phone.features, "syllabic") == Some(true)
}

fn phone_stress(phone: &PhoneToken) -> Spec<Stress> {
    match feature_category(&phone.features, "stress") {
        Some("primary") => Spec::Known(Stress::Primary),
        Some("secondary") => Spec::Known(Stress::Secondary),
        Some("unstressed") => Spec::Known(Stress::Unstressed),
        Some("reduced") => Spec::Known(Stress::Reduced),
        _ => Spec::Unspecified,
    }
}

fn phonology_base_symbol(phone: &PhoneToken) -> Option<&str> {
    feature_category(&phone.features, "base_symbol")
}

fn feature_bool(features: &FeatureBundle, name: &str) -> Option<bool> {
    match features
        .values
        .get(&FeatureId(format!("phonology.{name}")))?
    {
        Spec::Known(FeatureValue::Bool(value)) => Some(*value),
        _ => None,
    }
}

fn feature_category<'a>(features: &'a FeatureBundle, name: &str) -> Option<&'a str> {
    match features
        .values
        .get(&FeatureId(format!("phonology.{name}")))?
    {
        Spec::Known(FeatureValue::Category(value)) | Spec::Known(FeatureValue::Text(value)) => {
            Some(value)
        }
        _ => None,
    }
}

fn feature_bundle_matches(actual: &FeatureBundle, expected: &FeatureBundle) -> bool {
    expected.values.iter().all(|(feature, value)| {
        value == &Spec::Unspecified || actual.values.get(feature) == Some(value)
    })
}

fn phone_matches_id(phone: &PhoneToken, expected: &PhoneId) -> bool {
    matches!(&phone.phone, Spec::Known(id) if id == expected)
}

fn is_word_boundary(phone: &PhoneToken) -> bool {
    matches!(&phone.phone, Spec::Known(id) if id.as_str() == "boundary.word")
}

fn is_boundary_phone(phone: &PhoneToken) -> bool {
    matches!(&phone.phone, Spec::Known(id) if id.as_str().starts_with("boundary."))
}

fn syllable(
    phones: Vec<PhoneToken>,
    stress: Spec<Stress>,
    phone_positions: Vec<SyllablePosition>,
    nucleus_index: Option<usize>,
) -> Syllable {
    Syllable {
        phones,
        stress,
        phone_positions,
        span: None,
        nucleus_index,
    }
}

pub fn syllables_to_ipa(syllables: &[Syllable]) -> String {
    syllables
        .iter()
        .enumerate()
        .map(|(index, syllable)| {
            let mut text = String::new();
            let mut has_stress_mark = false;
            match syllable.stress {
                Spec::Known(Stress::Primary) => {
                    has_stress_mark = true;
                    text.push('ˈ');
                }
                Spec::Known(Stress::Secondary) => {
                    has_stress_mark = true;
                    text.push('ˌ');
                }
                _ => {}
            }
            if index > 0 && !has_stress_mark {
                text.insert(0, '.');
            }
            for phone in &syllable.phones {
                text.push_str(phone_ipa(phone));
            }
            text
        })
        .collect()
}

fn phone_ipa(phone: &PhoneToken) -> &str {
    match &phone.phone {
        Spec::Known(id) => id
            .as_str()
            .strip_prefix("ipa.phone.")
            .unwrap_or(id.as_str()),
        _ => "",
    }
}

pub fn syllabification_provenance() -> EvidenceProvenance {
    EvidenceProvenance {
        source: EvidenceSource::Rule,
        method: "maximum onset syllabification from variety phonotactics".into(),
        version: Some("0.1".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::english::variety;
    use crate::ids::VarietyId;
    use crate::phonemicize::{
        EnglishPhonemicizer, PhonemicizeRequest, Phonemicizer, phone_display_symbol,
    };

    fn syllables_for(text: &str) -> Vec<Syllable> {
        let output = EnglishPhonemicizer
            .phonemicize(&PhonemicizeRequest {
                text: text.into(),
                variety: VarietyId("en-US".into()),
                style: None,
            })
            .expect("phonemicize");
        syllabify_phones(&output.phones, &variety("en-US-GA"))
    }

    #[test]
    fn extra_uses_maximum_onset_for_legal_str_cluster() {
        assert_eq!(syllables_to_ipa(&syllables_for("extra")), "ˈɛk.st˭ɹə");
    }

    #[test]
    fn atlas_keeps_illegal_tl_split_out_of_onset() {
        assert_eq!(syllables_to_ipa(&syllables_for("atlas")), "ˈæt.ləs");
    }

    #[test]
    fn rhotic_vowels_do_not_add_coda_r_in_syllables() {
        assert_eq!(syllables_to_ipa(&syllables_for("current")), "ˈkʰɝ.ənt");
        assert_eq!(syllables_to_ipa(&syllables_for("derived")), "dɚˈaɪvd");
        assert_eq!(syllables_to_ipa(&syllables_for("surface")), "ˈsɝ.fəs");

        let current = syllables_for("current");
        assert_eq!(
            current[0].phone_positions,
            [SyllablePosition::Onset, SyllablePosition::Nucleus,]
        );
    }

    #[test]
    fn syllable_roles_mark_onset_nucleus_and_coda() {
        let syllables = syllables_for("atlas");
        assert_eq!(
            syllables[0].phone_positions,
            [SyllablePosition::Nucleus, SyllablePosition::Coda]
        );
        assert_eq!(
            syllables[1].phone_positions,
            [
                SyllablePosition::Onset,
                SyllablePosition::Nucleus,
                SyllablePosition::Coda
            ]
        );
        assert_eq!(
            syllables[1]
                .phones
                .iter()
                .filter_map(|phone| match &phone.phone {
                    Spec::Known(id) => Some(phone_display_symbol(id)),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            ["l", "ə", "s"]
        );
    }
}
