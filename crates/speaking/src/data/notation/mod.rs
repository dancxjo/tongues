pub mod arpabet;
pub mod openepd;

use crate::feature::{FeatureBundle, FeatureValue};
use crate::ids::{FeatureId, PhonemeId};
use crate::phonology::Phoneme;
use crate::spec::Spec;
use crate::variety::LinguisticVariety;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PronunciationNotation {
    Arpabet,
    Ipa,
}

#[derive(Debug, Clone)]
pub struct ParsedPronunciationToken {
    pub phoneme: PhonemeId,
    pub features: FeatureBundle,
}

pub fn pronunciation_notation(name: Option<&str>) -> Option<PronunciationNotation> {
    match name {
        Some("arpabet") | Some("cmudict") => Some(PronunciationNotation::Arpabet),
        Some("ipa") => Some(PronunciationNotation::Ipa),
        _ => None,
    }
}

pub fn parse_pronunciation_candidate(
    variety: &LinguisticVariety,
    candidate: &[String],
    notation: PronunciationNotation,
) -> Vec<ParsedPronunciationToken> {
    match notation {
        PronunciationNotation::Arpabet => candidate
            .iter()
            .map(|symbol| parsed_arpabet_token(variety, symbol))
            .collect(),
        PronunciationNotation::Ipa => candidate
            .iter()
            .flat_map(|symbol| parse_variety_ipa(symbol, variety))
            .collect(),
    }
}

fn parsed_arpabet_token(variety: &LinguisticVariety, symbol: &str) -> ParsedPronunciationToken {
    let cmu = crate::data::lexicons::cmudict::CmuPhoneme::parse(symbol);
    let raw_symbol = cmu.raw_symbol();
    let inventory_phoneme = variety.phonemes.phonemes.values().find(|phoneme| {
        phoneme.aliases.iter().any(|alias| {
            alias.system.eq_ignore_ascii_case("arpabet")
                && alias.symbol.eq_ignore_ascii_case(&cmu.base)
        })
    });
    let mut token = ParsedPronunciationToken {
        phoneme: inventory_phoneme
            .map(|phoneme| phoneme.id.clone())
            .unwrap_or_else(|| arpabet::phoneme_id(&variety.id.0, &raw_symbol)),
        features: inventory_phoneme
            .map(|phoneme| phoneme.features.clone())
            .unwrap_or_default(),
    };
    for (id, value) in arpabet::cmu_token_features(&cmu).values {
        let is_token_metadata = matches!(
            id.0.as_str(),
            "phonology.source_schema" | "phonology.base_symbol" | "phonology.stress"
        );
        let is_standard_reduction =
            id.0 == "phonology.reduced_vowel" && arpabet::entry(&cmu.base).is_some();
        if is_token_metadata || is_standard_reduction {
            token.features.values.insert(id, value);
        }
    }
    if let Some(phone) = arpabet::reduced_phone_for_cmu(&cmu.base, cmu.stress) {
        token.features.values.insert(
            FeatureId("phonology.default_phone".into()),
            Spec::Known(FeatureValue::Text(phone.as_str().to_string())),
        );
    }
    token
}

fn parse_variety_ipa(ipa: &str, variety: &LinguisticVariety) -> Vec<ParsedPronunciationToken> {
    let aliases = phoneme_aliases_by_length(variety);
    let chars = ipa
        .trim_matches('/')
        .chars()
        .filter(|ch| !matches!(ch, '.' | 'ˌ'))
        .collect::<Vec<_>>();
    let mut index = 0usize;
    let mut primary_stress_pending = false;
    let mut candidate = Vec::new();
    while index < chars.len() {
        if chars[index] == 'ˈ' {
            primary_stress_pending = true;
            index += 1;
            continue;
        }
        let rest = chars[index..].iter().collect::<String>();
        if let Some((phoneme, consumed)) = aliases.iter().find_map(|(alias, phoneme)| {
            rest.starts_with(alias)
                .then_some((*phoneme, alias.chars().count()))
        }) {
            let mut token = ParsedPronunciationToken {
                phoneme: phoneme.id.clone(),
                features: phoneme.features.clone(),
            };
            if primary_stress_pending && parsed_token_is_syllabic(&token) {
                token.features.values.insert(
                    FeatureId("phonology.stress".into()),
                    Spec::Known(FeatureValue::Category("primary".into())),
                );
                primary_stress_pending = false;
            }
            candidate.push(token);
            index += consumed;
        } else {
            index += 1;
        }
    }
    candidate
}

pub(crate) fn phoneme_aliases_by_length(variety: &LinguisticVariety) -> Vec<(String, &Phoneme)> {
    let mut aliases = Vec::new();
    for phoneme in variety.phonemes.phonemes.values() {
        aliases.push((phoneme_display_symbol(&phoneme.id).to_lowercase(), phoneme));
        for alias in &phoneme.aliases {
            aliases.push((alias.symbol.to_lowercase(), phoneme));
        }
    }
    aliases.sort_by(|left, right| right.0.len().cmp(&left.0.len()));
    aliases.dedup_by(|left, right| left.0 == right.0);
    aliases
}

fn parsed_token_is_syllabic(token: &ParsedPronunciationToken) -> bool {
    token
        .features
        .values
        .get(&FeatureId("phonology.syllabic".into()))
        == Some(&Spec::Known(FeatureValue::Bool(true)))
}

fn phoneme_display_symbol(id: &PhonemeId) -> &str {
    id.0.rsplit('.').next().unwrap_or(&id.0)
}
