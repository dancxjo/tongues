use std::collections::HashMap;

use crate::feature::{FeatureBundle, FeatureSystem, FeatureValue};
use crate::ids::{LanguageId, PhoneId, PhonemeId, VarietyId};
use crate::orthography::Orthography;
use crate::phonetics::{Phone, PhoneInventory};
use crate::phonology::{Phoneme, PhonemeInventory};
use crate::segment::{SegmentStatus, SymbolAlias};
use crate::spec::Spec;
use crate::variety::{LinguisticVariety, VarietyImplementationStatus, VarietyStatus};

const SEGMENTS: &[&str] = &[
    "a", "aː", "e", "ɛ", "i", "iː", "o", "oː", "u", "uː", "y", "ø", "œ", "ə", "ɐ", "aɪ̯", "aʊ̯",
    "ɔʏ̯", "b", "ç", "d", "f", "ɡ", "h", "j", "k", "l", "m", "n", "p", "r", "s", "ʃ", "t", "t͡s",
    "v", "x", "z",
];

pub fn variety() -> LinguisticVariety {
    LinguisticVariety {
        id: VarietyId("de-DE-Standard".into()),
        language: LanguageId("de".into()),
        name: "Standard German".into(),
        feature_system: FeatureSystem::default(),
        phonemes: phoneme_inventory(),
        phones: phone_inventory(),
        allophone_rules: Vec::new(),
        epenthesis_rules: Vec::new(),
        weak_forms: Vec::new(),
        orthographic_unit_pronunciations: Vec::new(),
        phonotactics: None,
        orthography: Some(Orthography {
            name: "German Latin orthography".into(),
            ..Default::default()
        }),
        morphology: None,
        acoustic_profile: None,
        prosody_profile: None,
        status: VarietyStatus::Attested,
        implementation_status: VarietyImplementationStatus::Complete,
    }
}

pub fn synthesize_ipa(word: &str) -> Option<String> {
    let chars = normalize(word)?;
    let mut ipa = String::new();
    let mut index = 0usize;
    while index < chars.len() {
        let rest = chars[index..].iter().collect::<String>();
        let (symbol, consumed) = if rest.starts_with("sch") {
            ("ʃ", 3)
        } else if rest.starts_with("ch") {
            (
                if previous_is_back_vowel(&chars, index) {
                    "x"
                } else {
                    "ç"
                },
                2,
            )
        } else if rest.starts_with("ei") || rest.starts_with("ai") {
            ("aɪ̯", 2)
        } else if rest.starts_with("eu") || rest.starts_with("äu") {
            ("ɔʏ̯", 2)
        } else if rest.starts_with("au") {
            ("aʊ̯", 2)
        } else if rest.starts_with("ie") {
            ("iː", 2)
        } else if rest.starts_with("sp") || rest.starts_with("st") {
            ("ʃ", 1)
        } else {
            (single(chars[index])?, 1)
        };
        ipa.push_str(symbol);
        index += consumed;
    }
    let ipa = add_initial_stress(&ipa);
    (!ipa.is_empty()).then_some(format!("/{ipa}/"))
}

fn single(ch: char) -> Option<&'static str> {
    Some(match ch {
        'a' => "a",
        'ä' => "ɛ",
        'b' => "b",
        'c' => "k",
        'd' => "d",
        'e' => "ə",
        'f' => "f",
        'g' => "ɡ",
        'h' => "h",
        'i' => "i",
        'j' => "j",
        'k' => "k",
        'l' => "l",
        'm' => "m",
        'n' => "n",
        'o' => "o",
        'ö' => "ø",
        'p' => "p",
        'q' => "k",
        'r' => "r",
        's' => "z",
        'ß' => "s",
        't' => "t",
        'u' => "u",
        'ü' => "y",
        'v' => "f",
        'w' => "v",
        'x' => "ks",
        'y' => "y",
        'z' => "t͡s",
        '-' | '\'' | '’' => "",
        _ => return None,
    })
}

fn normalize(word: &str) -> Option<Vec<char>> {
    let normalized = word.trim().to_lowercase();
    if normalized.is_empty()
        || normalized.chars().count() > 48
        || normalized
            .chars()
            .any(|ch| !(ch.is_alphabetic() || matches!(ch, '-' | '\'' | '’')))
    {
        return None;
    }
    Some(normalized.chars().collect())
}

fn previous_is_back_vowel(chars: &[char], index: usize) -> bool {
    index > 0 && matches!(chars[index - 1], 'a' | 'o' | 'u')
}

fn add_initial_stress(ipa: &str) -> String {
    let mut chars = ipa.chars().collect::<Vec<_>>();
    let Some(mut insert) = chars.iter().position(|ch| is_ipa_vowel(*ch)) else {
        return ipa.to_string();
    };
    while insert > 0 && !is_ipa_vowel(chars[insert - 1]) {
        insert -= 1;
    }
    chars.insert(insert, 'ˈ');
    chars.into_iter().collect()
}

fn is_ipa_vowel(ch: char) -> bool {
    matches!(
        ch,
        'a' | 'e' | 'ɛ' | 'i' | 'o' | 'u' | 'y' | 'ø' | 'œ' | 'ə' | 'ɐ' | 'ɔ'
    )
}

fn phoneme_inventory() -> PhonemeInventory {
    PhonemeInventory {
        phonemes: SEGMENTS
            .iter()
            .map(|symbol| {
                let phoneme = Phoneme {
                    id: PhonemeId(format!("de-DE-Standard.phoneme.{symbol}")),
                    notation: format!("/{symbol}/"),
                    features: segment_features(symbol),
                    default_phone: Some(PhoneId(format!("ipa.phone.{symbol}").into())),
                    possible_phones: vec![PhoneId(format!("ipa.phone.{symbol}").into())],
                    aliases: vec![SymbolAlias {
                        system: "ipa".into(),
                        symbol: (*symbol).into(),
                    }],
                    allophones: Vec::new(),
                    status: SegmentStatus::Core,
                };
                (phoneme.id.clone(), phoneme)
            })
            .collect(),
    }
}

fn phone_inventory() -> PhoneInventory {
    PhoneInventory {
        phones: SEGMENTS
            .iter()
            .map(|symbol| {
                let id = PhoneId(format!("ipa.phone.{symbol}").into());
                (
                    id.clone(),
                    Phone {
                        id,
                        ipa: (*symbol).into(),
                        features: segment_features(symbol),
                        aliases: vec![SymbolAlias {
                            system: "ipa".into(),
                            symbol: (*symbol).into(),
                        }],
                        status: SegmentStatus::Core,
                    },
                )
            })
            .collect::<HashMap<_, _>>(),
    }
}

fn segment_features(symbol: &str) -> FeatureBundle {
    let mut features = FeatureBundle::default();
    let is_vowel = matches!(
        symbol,
        "a" | "aː"
            | "e"
            | "ɛ"
            | "i"
            | "iː"
            | "o"
            | "oː"
            | "u"
            | "uː"
            | "y"
            | "ø"
            | "œ"
            | "ə"
            | "ɐ"
            | "aɪ̯"
            | "aʊ̯"
            | "ɔʏ̯"
    );
    features.values.insert(
        crate::ids::FeatureId("phonology.major".into()),
        Spec::Known(FeatureValue::Category(
            if is_vowel { "vowel" } else { "consonant" }.into(),
        )),
    );
    features.values.insert(
        crate::ids::FeatureId("phonology.syllabic".into()),
        Spec::Known(FeatureValue::Bool(is_vowel)),
    );
    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn german_synthesizes_common_words() {
        assert_eq!(synthesize_ipa("Sprache").as_deref(), Some("/ˈʃpraxə/"));
    }
}
