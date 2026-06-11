use crate::data::lexicons::cmudict::CmuStress;
use crate::data::notation::arpabet;
use crate::evidence::{EvidenceProvenance, EvidenceSource};
use crate::feature::{FeatureBundle, FeatureValue};
use crate::ids::{FeatureId, PhoneId, PhonemeId};
use crate::phonology::{PhoneToken, PhonemeToken};
use crate::prosody::Stress;
use crate::rules::{AllophoneRule, EpenthesisRule, RuleCondition};
use crate::segment::{SegmentMatcher, SyllablePosition, WordPosition};
use crate::spec::Spec;
use crate::syntax::SyntaxRuleContext;
use crate::variety::LinguisticVariety;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealizationOptions {
    pub careful_style: bool,
    pub phone_decomposition: PhoneDecompositionPolicy,
    pub syntax: SyntaxRuleContext,
}

impl Default for RealizationOptions {
    fn default() -> Self {
        Self {
            careful_style: false,
            phone_decomposition: PhoneDecompositionPolicy::KeepPhonemic,
            syntax: SyntaxRuleContext::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhoneDecompositionPolicy {
    KeepPhonemic,
    SplitForAcoustics,
    SplitForSinging,
}

pub fn realize_phonemes(
    variety: &LinguisticVariety,
    phonemes: &[PhonemeToken],
    options: &RealizationOptions,
) -> Vec<PhoneToken> {
    let mut phones = Vec::new();
    for index in 0..phonemes.len() {
        phones.push(realize_phoneme_at(variety, phonemes, index, options));
        phones.extend(epenthetic_phones_after(variety, phonemes, index));
    }
    phones
}

pub fn realize_phoneme_at(
    variety: &LinguisticVariety,
    phonemes: &[PhonemeToken],
    index: usize,
    options: &RealizationOptions,
) -> PhoneToken {
    let token = &phonemes[index];
    let default_phone = default_phone_token(variety, token);
    if let Some(rule) = variety
        .allophone_rules
        .iter()
        .find(|rule| rule_applies(rule, variety, phonemes, index, options))
    {
        phone_from_rule(variety, token, &default_phone, rule)
    } else {
        default_phone
    }
}

pub fn epenthetic_phones_after(
    variety: &LinguisticVariety,
    phonemes: &[PhonemeToken],
    index: usize,
) -> Vec<PhoneToken> {
    let Some(before) = phonemes.get(index) else {
        return Vec::new();
    };
    let Some(after) = phonemes.get(index + 1) else {
        return Vec::new();
    };

    variety
        .epenthesis_rules
        .iter()
        .filter(|rule| epenthesis_rule_applies(rule, variety, before, after))
        .map(|rule| phone_from_epenthesis_rule(variety, before, rule))
        .collect()
}

pub fn phoneme_features<'a>(
    variety: &'a LinguisticVariety,
    id: &PhonemeId,
) -> Option<&'a FeatureBundle> {
    variety
        .phonemes
        .phonemes
        .get(id)
        .map(|phoneme| &phoneme.features)
        .or_else(|| {
            let base_id = base_phoneme_id(id)?;
            variety
                .phonemes
                .phonemes
                .get(&base_id)
                .map(|phoneme| &phoneme.features)
        })
}

pub fn token_stress(token: &PhonemeToken) -> Option<Stress> {
    if let Some((_, stress)) = token_cmu_base_and_stress(token) {
        return match stress {
            Some(CmuStress::Primary) => Some(Stress::Primary),
            Some(CmuStress::Secondary) => Some(Stress::Secondary),
            Some(CmuStress::Unstressed) => Some(Stress::Unstressed),
            None => None,
        };
    }

    let Spec::Known(id) = &token.phoneme else {
        return None;
    };
    match phoneme_display_symbol(id).chars().last() {
        Some('1') => Some(Stress::Primary),
        Some('2') => Some(Stress::Secondary),
        Some('0') => Some(Stress::Unstressed),
        _ => None,
    }
}

pub fn token_is_vowel(variety: &LinguisticVariety, token: &PhonemeToken) -> bool {
    token_feature_matches(
        variety,
        token,
        &FeatureId("phonology.major".into()),
        &FeatureValue::Category("vowel".into()),
    )
}

fn token_syllable_position(
    variety: &LinguisticVariety,
    token: &PhonemeToken,
) -> Option<SyllablePosition> {
    if token_is_vowel(variety, token) {
        Some(SyllablePosition::Nucleus)
    } else {
        None
    }
}

fn rule_applies(
    rule: &AllophoneRule,
    variety: &LinguisticVariety,
    phonemes: &[PhonemeToken],
    index: usize,
    options: &RealizationOptions,
) -> bool {
    let Some(token) = phonemes.get(index) else {
        return false;
    };
    input_matches(rule, variety, token)
        && environment_matches(rule, variety, phonemes, index)
        && rule
            .conditions
            .iter()
            .all(|condition| condition_matches(condition, variety, phonemes, index, options))
}

fn input_matches(rule: &AllophoneRule, variety: &LinguisticVariety, token: &PhonemeToken) -> bool {
    (match &rule.input.phoneme {
        Spec::Known(expected) => phoneme_token_matches_id(token, expected),
        Spec::Unspecified => true,
        _ => false,
    }) && feature_bundle_matches(variety, token, &rule.input.features)
}

fn environment_matches(
    rule: &AllophoneRule,
    variety: &LinguisticVariety,
    phonemes: &[PhonemeToken],
    index: usize,
) -> bool {
    let token = &phonemes[index];
    let before_matches = rule.environment.before.is_empty()
        || index.checked_sub(1).is_some_and(|previous| {
            rule.environment
                .before
                .iter()
                .any(|matcher| segment_matches(variety, &phonemes[previous], matcher))
        });
    let after_matches = rule.environment.after.is_empty()
        || phonemes.get(index + 1).is_some_and(|next| {
            rule.environment
                .after
                .iter()
                .any(|matcher| segment_matches(variety, next, matcher))
        });
    let stress_matches = match &rule.environment.stress_context {
        Spec::Known(expected) => token_stress(token).is_some_and(|actual| actual == *expected),
        Spec::Unspecified => true,
        _ => false,
    };
    let syllable_position_matches = match &rule.environment.syllable_position {
        Spec::Known(expected) => {
            token_syllable_position(variety, token).is_some_and(|actual| actual == *expected)
        }
        Spec::Unspecified => true,
        _ => false,
    };
    let word_position_matches = match &rule.environment.word_position {
        Spec::Known(expected) => {
            token_word_position(phonemes, index).is_some_and(|actual| actual == *expected)
        }
        Spec::Unspecified => true,
        _ => false,
    };

    before_matches
        && after_matches
        && stress_matches
        && syllable_position_matches
        && word_position_matches
}

fn token_word_position(phonemes: &[PhonemeToken], index: usize) -> Option<WordPosition> {
    let token_word = phonemes.get(index).and_then(token_word_index);
    let previous_same_word = index.checked_sub(1).is_some_and(|previous| {
        token_word
            .zip(token_word_index(&phonemes[previous]))
            .is_some_and(|(current, previous)| current == previous)
    });
    let next_same_word = phonemes.get(index + 1).is_some_and(|next| {
        token_word
            .zip(token_word_index(next))
            .is_some_and(|(current, next)| current == next)
    });

    if token_word.is_some() {
        return match (previous_same_word, next_same_word) {
            (false, false) => Some(WordPosition::Isolated),
            (false, true) => Some(WordPosition::Initial),
            (true, false) => Some(WordPosition::Final),
            (true, true) => Some(WordPosition::Medial),
        };
    }

    match (index == 0, index + 1 == phonemes.len()) {
        (true, true) => Some(WordPosition::Isolated),
        (true, false) => Some(WordPosition::Initial),
        (false, true) => Some(WordPosition::Final),
        (false, false) => Some(WordPosition::Medial),
    }
}

fn token_word_index(token: &PhonemeToken) -> Option<usize> {
    let value = token
        .features
        .values
        .get(&FeatureId("orthography.word_index".into()))?;
    match value {
        Spec::Known(FeatureValue::Number(value)) if value.is_finite() && *value >= 0.0 => {
            Some(*value as usize)
        }
        _ => None,
    }
}

fn condition_matches(
    condition: &RuleCondition,
    variety: &LinguisticVariety,
    phonemes: &[PhonemeToken],
    index: usize,
    options: &RealizationOptions,
) -> bool {
    match condition {
        RuleCondition::PreviousMatches(matcher) => index
            .checked_sub(1)
            .is_some_and(|previous| segment_matches(variety, &phonemes[previous], matcher)),
        RuleCondition::NextMatches(matcher) => phonemes
            .get(index + 1)
            .is_some_and(|next| segment_matches(variety, next, matcher)),
        RuleCondition::PreviousHasFeature(feature, value) => {
            index.checked_sub(1).is_some_and(|previous| {
                token_feature_matches(variety, &phonemes[previous], feature, value)
            })
        }
        RuleCondition::NextHasFeature(feature, value) => phonemes
            .get(index + 1)
            .is_some_and(|next| token_feature_matches(variety, next, feature, value)),
        RuleCondition::PreviousStress(stress) => index
            .checked_sub(1)
            .and_then(|previous| token_stress(&phonemes[previous]))
            .is_some_and(|actual| &actual == stress),
        RuleCondition::PreviousStressIn(stresses) => index
            .checked_sub(1)
            .and_then(|previous| token_stress(&phonemes[previous]))
            .is_some_and(|actual| stresses.contains(&actual)),
        RuleCondition::NextStress(stress) => phonemes
            .get(index + 1)
            .and_then(token_stress)
            .is_some_and(|actual| &actual == stress),
        RuleCondition::NextStressIn(stresses) => phonemes
            .get(index + 1)
            .and_then(token_stress)
            .is_some_and(|actual| stresses.contains(&actual)),
        RuleCondition::CurrentWordHasSyntacticLink(_)
        | RuleCondition::PreviousWordHasSyntacticLink(_)
        | RuleCondition::NextWordHasSyntacticLink(_) => phonemes
            .get(index)
            .and_then(token_word_index)
            .is_some_and(|word_index| condition.matches_syntax(&options.syntax, word_index)),
        RuleCondition::NotCarefulStyle => !options.careful_style,
    }
}

fn segment_matches(
    variety: &LinguisticVariety,
    token: &PhonemeToken,
    matcher: &SegmentMatcher,
) -> bool {
    match matcher {
        SegmentMatcher::Any => true,
        SegmentMatcher::Phoneme(expected) => phoneme_token_matches_id(token, expected),
        SegmentMatcher::FeatureBundle(expected) => feature_bundle_matches(variety, token, expected),
        SegmentMatcher::Phone(_) | SegmentMatcher::Boundary(_) => false,
    }
}

fn epenthesis_rule_applies(
    rule: &EpenthesisRule,
    variety: &LinguisticVariety,
    before: &PhonemeToken,
    after: &PhonemeToken,
) -> bool {
    (rule.before.is_empty()
        || rule
            .before
            .iter()
            .any(|matcher| segment_matches(variety, before, matcher)))
        && (rule.after.is_empty()
            || rule
                .after
                .iter()
                .any(|matcher| segment_matches(variety, after, matcher)))
}

fn phoneme_token_matches_id(token: &PhonemeToken, expected: &PhonemeId) -> bool {
    let Spec::Known(actual) = &token.phoneme else {
        return false;
    };
    actual == expected || base_phoneme_id(actual).as_ref() == Some(expected)
}

fn feature_bundle_matches(
    variety: &LinguisticVariety,
    token: &PhonemeToken,
    expected: &FeatureBundle,
) -> bool {
    expected.values.iter().all(|(feature, value)| match value {
        Spec::Known(value) => token_feature_matches(variety, token, feature, value),
        Spec::Unspecified => true,
        _ => false,
    })
}

fn token_feature_matches(
    variety: &LinguisticVariety,
    token: &PhonemeToken,
    feature: &FeatureId,
    expected: &FeatureValue,
) -> bool {
    if token
        .features
        .values
        .get(feature)
        .is_some_and(|actual| actual == &Spec::Known(expected.clone()))
    {
        return true;
    }

    let Spec::Known(id) = &token.phoneme else {
        return false;
    };
    phoneme_features(variety, id)
        .and_then(|features| features.values.get(feature))
        .is_some_and(|actual| actual == &Spec::Known(expected.clone()))
}

fn phone_from_rule(
    variety: &LinguisticVariety,
    token: &PhonemeToken,
    default_phone: &PhoneToken,
    rule: &AllophoneRule,
) -> PhoneToken {
    let phone = rule.output.phone.clone();
    let features = match &phone {
        Spec::Known(id) => {
            let mut features = default_phone.features.clone();
            if let Some(phone) = variety.phones.phones.get(id) {
                features.values.extend(phone.features.values.clone());
            }
            features.values.extend(rule.output.features.values.clone());
            features
        }
        _ => rule.output.features.clone(),
    };

    PhoneToken {
        phone,
        span: token.span,
        features,
        acoustic_evidence: Vec::new(),
        confidence: token.confidence.min(rule.confidence),
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: format!("{} allophone rule {}", variety.id.0, rule.id),
            version: Some("0.1".into()),
        },
    }
}

fn phone_from_epenthesis_rule(
    variety: &LinguisticVariety,
    previous: &PhonemeToken,
    rule: &EpenthesisRule,
) -> PhoneToken {
    let features = match &rule.output.phone {
        Spec::Known(id) => variety
            .phones
            .phones
            .get(id)
            .map(|phone| phone.features.clone())
            .unwrap_or_default(),
        _ => rule.output.features.clone(),
    };

    PhoneToken {
        phone: rule.output.phone.clone(),
        span: None,
        features,
        acoustic_evidence: Vec::new(),
        confidence: previous.confidence.min(rule.confidence),
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: format!("{} epenthesis rule {}", variety.id.0, rule.id),
            version: Some("0.1".into()),
        },
    }
}

fn default_phone_token(variety: &LinguisticVariety, token: &PhonemeToken) -> PhoneToken {
    let phone = default_phone_id(variety, token);
    let mut features = match &phone {
        Spec::Known(id) => variety
            .phones
            .phones
            .get(id)
            .map(|phone| phone.features.clone())
            .or_else(|| phoneme_token_features(variety, token))
            .unwrap_or_default(),
        _ => FeatureBundle::default(),
    };
    merge_features(&mut features, &token.features);

    PhoneToken {
        phone,
        span: token.span,
        features,
        acoustic_evidence: Vec::new(),
        confidence: token.confidence,
        provenance: token.provenance.clone(),
    }
}

fn merge_features(target: &mut FeatureBundle, source: &FeatureBundle) {
    for (id, value) in &source.values {
        target.values.insert(id.clone(), value.clone());
    }
}

fn default_phone_id(variety: &LinguisticVariety, token: &PhonemeToken) -> Spec<PhoneId> {
    let Spec::Known(id) = &token.phoneme else {
        return match token.phoneme {
            Spec::Unknown => Spec::Unknown,
            Spec::Unspecified => Spec::Unspecified,
            Spec::NotApplicable => Spec::NotApplicable,
            _ => Spec::Unknown,
        };
    };

    if let Some(phone) = stress_aware_phone_id(token) {
        return Spec::Known(phone);
    }

    variety
        .phonemes
        .phonemes
        .get(id)
        .and_then(|phoneme| phoneme.default_phone.clone())
        .or_else(|| {
            let base_id = base_phoneme_id(id)?;
            variety
                .phonemes
                .phonemes
                .get(&base_id)
                .and_then(|phoneme| phoneme.default_phone.clone())
        })
        .or_else(|| {
            let base = phoneme_base_symbol(id);
            arpabet::entry(base).map(|entry| arpabet::phone_id_for_ipa(entry.phone_symbol))
        })
        .map(Spec::Known)
        .unwrap_or(Spec::Unknown)
}

fn stress_aware_phone_id(token: &PhonemeToken) -> Option<PhoneId> {
    let (base, stress) = token_cmu_base_and_stress(token).or_else(|| {
        let Spec::Known(id) = &token.phoneme else {
            return None;
        };
        let symbol = phoneme_display_symbol(id);
        let (base, stress) = arpabet::split_stress(symbol);
        Some((base.to_string(), stress.and_then(cmu_stress_from_digit)))
    })?;
    arpabet::reduced_phone_for_cmu(&base, stress)
}

fn token_cmu_base_and_stress(token: &PhonemeToken) -> Option<(String, Option<CmuStress>)> {
    let source_schema = token_category_feature(&token.features, "source_schema")?;
    if source_schema != "cmudict" && source_schema != "arpabet" {
        return None;
    }
    let base = token_category_feature(&token.features, "base_symbol")?.to_string();
    let stress = token_category_feature(&token.features, "stress").and_then(cmu_stress_from_name);
    Some((base, stress))
}

fn phoneme_token_features(
    variety: &LinguisticVariety,
    token: &PhonemeToken,
) -> Option<FeatureBundle> {
    if !token.features.values.is_empty() {
        return Some(token.features.clone());
    }
    let Spec::Known(id) = &token.phoneme else {
        return None;
    };
    phoneme_features(variety, id)
        .cloned()
        .or_else(|| arpabet::entry(phoneme_base_symbol(id)).map(arpabet::feature_bundle))
}

fn base_phoneme_id(id: &PhonemeId) -> Option<PhonemeId> {
    let (prefix, symbol) = id.0.rsplit_once(".phoneme.")?;
    let (base, stress) = arpabet::split_stress(symbol);
    stress.map(|_| PhonemeId(format!("{prefix}.phoneme.{base}")))
}

fn phoneme_display_symbol(id: &PhonemeId) -> &str {
    id.0.rsplit('.').next().unwrap_or(&id.0)
}

fn phoneme_base_symbol(id: &PhonemeId) -> &str {
    let symbol = phoneme_display_symbol(id);
    arpabet::split_stress(symbol).0
}

fn token_category_feature<'a>(features: &'a FeatureBundle, name: &str) -> Option<&'a str> {
    let value = features
        .values
        .get(&FeatureId(format!("phonology.{name}")))?;
    match value {
        Spec::Known(FeatureValue::Category(value)) => Some(value),
        Spec::Known(FeatureValue::Text(value)) => Some(value),
        _ => None,
    }
}

fn cmu_stress_from_name(name: &str) -> Option<CmuStress> {
    match name {
        "primary" => Some(CmuStress::Primary),
        "secondary" => Some(CmuStress::Secondary),
        "unstressed" => Some(CmuStress::Unstressed),
        _ => None,
    }
}

fn cmu_stress_from_digit(digit: char) -> Option<CmuStress> {
    match digit {
        '1' => Some(CmuStress::Primary),
        '2' => Some(CmuStress::Secondary),
        '0' => Some(CmuStress::Unstressed),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::lexicons::cmudict::CmuPhoneme;
    use crate::data::notation::arpabet;
    use crate::data::variety_by_code;
    use crate::ids::VarietyId;
    use crate::variety::VarietyImplementationStatus;

    fn phoneme(variety: &str, symbol: &str) -> PhonemeToken {
        let cmu = CmuPhoneme::parse(symbol);
        PhonemeToken {
            phoneme: Spec::Known(arpabet::phoneme_id(variety, symbol)),
            span: None,
            features: arpabet::cmu_token_features(&cmu),
            realized_as: Vec::new(),
            confidence: 1.0,
            provenance: EvidenceProvenance {
                source: EvidenceSource::Lexicon,
                method: "test".into(),
                version: None,
            },
        }
    }

    fn unknown_phoneme() -> PhonemeToken {
        PhonemeToken {
            phoneme: Spec::Unknown,
            span: None,
            features: FeatureBundle::default(),
            realized_as: Vec::new(),
            confidence: 0.0,
            provenance: EvidenceProvenance {
                source: EvidenceSource::Unknown,
                method: "test".into(),
                version: None,
            },
        }
    }

    fn underspecified_phoneme() -> PhonemeToken {
        PhonemeToken {
            phoneme: Spec::Unspecified,
            span: None,
            features: FeatureBundle::default(),
            realized_as: Vec::new(),
            confidence: 0.0,
            provenance: EvidenceProvenance {
                source: EvidenceSource::Unknown,
                method: "test".into(),
                version: None,
            },
        }
    }

    fn symbols(phones: &[PhoneToken]) -> Vec<String> {
        phones
            .iter()
            .map(|token| match &token.phone {
                Spec::Known(id) => id
                    .as_str()
                    .rsplit('.')
                    .next()
                    .unwrap_or(id.as_str())
                    .to_string(),
                Spec::Unknown => "?".into(),
                Spec::Unspecified => "_".into(),
                Spec::NotApplicable => "na".into(),
                Spec::Variable(_) | Spec::Gradient { .. } => "variable".into(),
            })
            .collect()
    }

    #[test]
    fn flapping_applies_between_stressed_and_unstressed_vowels() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variety,
            &[
                phoneme("en-US-GA", "AA1"),
                phoneme("en-US-GA", "T"),
                phoneme("en-US-GA", "ER0"),
            ],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["ɑ", "ɾ", "ɚ"]);
    }

    #[test]
    fn careful_style_blocks_flapping() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variety,
            &[
                phoneme("en-US-GA", "AA1"),
                phoneme("en-US-GA", "T"),
                phoneme("en-US-GA", "ER0"),
            ],
            &RealizationOptions {
                careful_style: true,
                ..Default::default()
            },
        );

        assert_eq!(symbols(&phones), ["ɑ", "t", "ɚ"]);
    }

    #[test]
    fn flapping_requires_stress_context() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variety,
            &[
                phoneme("en-US-GA", "AH0"),
                phoneme("en-US-GA", "T"),
                phoneme("en-US-GA", "ER0"),
            ],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["ə", "t", "ɚ"]);
    }

    #[test]
    fn voiceless_stops_aspirate_before_stressed_vowels_but_not_after_s() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let aspirated = realize_phonemes(
            &variety,
            &[phoneme("en-US-GA", "P"), phoneme("en-US-GA", "AY1")],
            &RealizationOptions::default(),
        );
        let s_cluster = realize_phonemes(
            &variety,
            &[
                phoneme("en-US-GA", "S"),
                phoneme("en-US-GA", "P"),
                phoneme("en-US-GA", "AY1"),
            ],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&aspirated), ["pʰ", "aɪ"]);
        assert_eq!(symbols(&s_cluster), ["s", "p˭", "aɪ"]);
    }

    #[test]
    fn d_flaps_between_stressed_and_unstressed_vowels() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variety,
            &[
                phoneme("en-US-GA", "AA1"),
                phoneme("en-US-GA", "D"),
                phoneme("en-US-GA", "ER0"),
            ],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["ɑ", "ɾ", "ɚ"]);
    }

    #[test]
    fn l_has_light_and_dark_contextual_models() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let light = realize_phonemes(
            &variety,
            &[phoneme("en-US-GA", "L"), phoneme("en-US-GA", "AY1")],
            &RealizationOptions::default(),
        );
        let dark = realize_phonemes(
            &variety,
            &[
                phoneme("en-US-GA", "B"),
                phoneme("en-US-GA", "AO1"),
                phoneme("en-US-GA", "L"),
            ],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&light), ["l", "aɪ"]);
        assert_eq!(symbols(&dark), ["b", "ɔ", "ɫ"]);
        assert_eq!(
            light[0]
                .features
                .values
                .get(&FeatureId("phonology.l_quality".into())),
            Some(&Spec::Known(FeatureValue::Category("light".into())))
        );
        assert_eq!(
            dark[2]
                .features
                .values
                .get(&FeatureId("phonology.l_quality".into())),
            Some(&Spec::Known(FeatureValue::Category("dark".into())))
        );
    }

    #[test]
    fn voiced_obstruents_can_partially_devoice_before_voiceless_obstruents() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variety,
            &[phoneme("en-US-GA", "B"), phoneme("en-US-GA", "T")],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["b", "t"]);
        assert_eq!(
            phones[0]
                .features
                .values
                .get(&FeatureId("phonology.partial_devoicing".into())),
            Some(&Spec::Known(FeatureValue::Bool(true)))
        );
    }

    #[test]
    fn unstressed_vowel_reduction_is_a_contextual_allophone_rule() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variety,
            &[phoneme("en-US-GA", "AH0")],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["ə"]);
        assert!(
            phones[0]
                .provenance
                .method
                .contains("unstressed_ah_nucleus_reduction")
        );
    }

    #[test]
    fn ah_defaults_to_schwa_and_uses_strut_in_stressed_syllables() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let default = realize_phonemes(
            &variety,
            &[phoneme("en-US-GA", "AH")],
            &RealizationOptions::default(),
        );
        let primary = realize_phonemes(
            &variety,
            &[phoneme("en-US-GA", "AH1")],
            &RealizationOptions::default(),
        );
        let secondary = realize_phonemes(
            &variety,
            &[phoneme("en-US-GA", "AH2")],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&default), ["ə"]);
        assert_eq!(symbols(&primary), ["ʌ"]);
        assert!(
            primary[0]
                .provenance
                .method
                .contains("stressed_ah_primary_strut_allophone")
        );
        assert_eq!(symbols(&secondary), ["ʌ"]);
        assert!(
            secondary[0]
                .provenance
                .method
                .contains("stressed_ah_secondary_strut_allophone")
        );
    }

    #[test]
    fn er_defaults_to_r_colored_schwa_and_uses_stressed_rhotic_vowel_in_stressed_syllables() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let default = realize_phonemes(
            &variety,
            &[phoneme("en-US-GA", "ER")],
            &RealizationOptions::default(),
        );
        let primary = realize_phonemes(
            &variety,
            &[phoneme("en-US-GA", "ER1")],
            &RealizationOptions::default(),
        );
        let secondary = realize_phonemes(
            &variety,
            &[phoneme("en-US-GA", "ER2")],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&default), ["ɚ"]);
        assert_eq!(symbols(&primary), ["ɝ"]);
        assert!(
            primary[0]
                .provenance
                .method
                .contains("stressed_er_primary_stressed_rhotic_allophone")
        );
        assert_eq!(symbols(&secondary), ["ɝ"]);
        assert!(
            secondary[0]
                .provenance
                .method
                .contains("stressed_er_secondary_stressed_rhotic_allophone")
        );
    }

    #[test]
    fn nasal_assimilation_applies_before_velar_stops() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variety,
            &[phoneme("en-US-GA", "N"), phoneme("en-US-GA", "K")],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["ŋ", "k"]);
    }

    #[test]
    fn nasal_assimilation_does_not_apply_before_non_velar_stops() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variety,
            &[phoneme("en-US-GA", "N"), phoneme("en-US-GA", "D")],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["n", "d"]);
    }

    #[test]
    fn removing_flapping_rule_disables_flapping() {
        let mut variety = variety_by_code("en-US-GA").expect("GA");
        variety
            .allophone_rules
            .retain(|rule| rule.id != "american_english_intervocalic_flapping");
        let phones = realize_phonemes(
            &variety,
            &[
                phoneme("en-US-GA", "AA1"),
                phoneme("en-US-GA", "T"),
                phoneme("en-US-GA", "ER0"),
            ],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["ɑ", "t", "ɚ"]);
    }

    #[test]
    fn removing_nasal_rule_disables_nasal_assimilation() {
        let mut variety = variety_by_code("en-US-GA").expect("GA");
        variety
            .allophone_rules
            .retain(|rule| rule.id != "alveolar_nasal_velar_assimilation");
        let phones = realize_phonemes(
            &variety,
            &[phoneme("en-US-GA", "N"), phoneme("en-US-GA", "K")],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["n", "k"]);
    }

    #[test]
    fn derived_stub_keeps_rule_behavior_and_stub_status() {
        let variety = variety_by_code("en-GB-RP").expect("RP");
        assert_eq!(
            variety.implementation_status,
            VarietyImplementationStatus::StubDerivedFrom(VarietyId("en-US-GA".into()))
        );
        let phones = realize_phonemes(
            &variety,
            &[
                phoneme("en-GB-RP", "AA1"),
                phoneme("en-GB-RP", "T"),
                phoneme("en-GB-RP", "ER0"),
            ],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["ɑ", "ɾ", "ɚ"]);
    }

    #[test]
    fn unknown_tokens_pass_through_without_panic() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variety,
            &[
                unknown_phoneme(),
                underspecified_phoneme(),
                phoneme("en-US-GA", "T"),
            ],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["?", "_", "t"]);
    }

    #[test]
    fn changed_phone_provenance_names_the_rule() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variety,
            &[phoneme("en-US-GA", "N"), phoneme("en-US-GA", "K")],
            &RealizationOptions::default(),
        );

        assert_eq!(phones[0].provenance.source, EvidenceSource::Rule);
        assert!(
            phones[0]
                .provenance
                .method
                .contains("alveolar_nasal_velar_assimilation")
        );
    }
}
