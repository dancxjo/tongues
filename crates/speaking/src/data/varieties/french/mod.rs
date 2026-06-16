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
    "a", "ɑ̃", "e", "ɛ", "ɛ̃", "i", "o", "ɔ", "ɔ̃", "u", "y", "ø", "œ", "ə", "b", "d", "f", "ɡ", "ʒ",
    "k", "l", "m", "n", "ɲ", "p", "ʁ", "s", "ʃ", "t", "v", "w", "z",
];

pub fn variety() -> LinguisticVariety {
    LinguisticVariety {
        id: VarietyId("fr-FR-Standard".into()),
        language: LanguageId("fr".into()),
        name: "Standard French".into(),
        feature_system: FeatureSystem::default(),
        phonemes: phoneme_inventory(),
        phones: phone_inventory(),
        allophone_rules: Vec::new(),
        epenthesis_rules: Vec::new(),
        weak_forms: Vec::new(),
        orthographic_unit_pronunciations: Vec::new(),
        phonotactics: None,
        orthography: Some(Orthography {
            name: "French Latin orthography".into(),
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
        let (symbol, consumed) = if rest.starts_with("eau") {
            ("o", 3)
        } else if rest.starts_with("au") {
            ("o", 2)
        } else if rest.starts_with("ou") {
            ("u", 2)
        } else if rest.starts_with("oi") {
            ("wa", 2)
        } else if rest.starts_with("eu") {
            ("ø", 2)
        } else if rest.starts_with("ch") {
            ("ʃ", 2)
        } else if rest.starts_with("gn") {
            ("ɲ", 2)
        } else if rest.starts_with("qu") {
            ("k", 2)
        } else if rest.starts_with("an") || rest.starts_with("en") {
            ("ɑ̃", 2)
        } else if rest.starts_with("on") {
            ("ɔ̃", 2)
        } else if rest.starts_with("in") || rest.starts_with("ain") || rest.starts_with("ein") {
            if rest.starts_with("ain") || rest.starts_with("ein") {
                ("ɛ̃", 3)
            } else {
                ("ɛ̃", 2)
            }
        } else {
            (single(chars[index], chars.get(index + 1).copied())?, 1)
        };
        ipa.push_str(symbol);
        index += consumed;
    }
    let ipa = trim_silent_finals(&ipa);
    let ipa = add_final_stress(&ipa);
    (!ipa.is_empty()).then_some(format!("/{ipa}/"))
}

fn single(ch: char, next: Option<char>) -> Option<&'static str> {
    Some(match ch {
        'a' => "a",
        'b' => "b",
        'c' if matches!(next, Some('e' | 'i' | 'y')) => "s",
        'c' => "k",
        'd' => "d",
        'e' => "ə",
        'f' => "f",
        'g' if matches!(next, Some('e' | 'i' | 'y')) => "ʒ",
        'g' => "ɡ",
        'h' => "",
        'i' | 'y' => "i",
        'j' => "ʒ",
        'k' => "k",
        'l' => "l",
        'm' => "m",
        'n' => "n",
        'o' => "ɔ",
        'p' => "p",
        'r' => "ʁ",
        's' => "s",
        't' => "t",
        'u' => "y",
        'v' => "v",
        'w' => "w",
        'x' => "ks",
        'z' => "z",
        '-' | '\'' | '’' => "",
        _ => return None,
    })
}

fn normalize(word: &str) -> Option<Vec<char>> {
    let normalized = word
        .trim()
        .to_lowercase()
        .replace(['à', 'â'], "a")
        .replace(['é', 'è', 'ê', 'ë'], "e")
        .replace(['î', 'ï'], "i")
        .replace(['ô'], "o")
        .replace(['ù', 'û', 'ü'], "u")
        .replace('ç', "c");
    if normalized.is_empty()
        || normalized.chars().count() > 48
        || normalized
            .chars()
            .any(|ch| !(ch.is_ascii_alphabetic() || matches!(ch, '-' | '\'' | '’')))
    {
        return None;
    }
    Some(normalized.chars().collect())
}

fn trim_silent_finals(ipa: &str) -> String {
    ipa.trim_end_matches(['ə', 's', 't', 'd']).to_string()
}

fn add_final_stress(ipa: &str) -> String {
    let mut chars = ipa.chars().collect::<Vec<_>>();
    let Some(mut insert) = chars.iter().rposition(|ch| is_ipa_vowel(*ch)) else {
        return ipa.to_string();
    };
    while insert > 0 && !is_ipa_vowel(chars[insert - 1]) {
        if is_combining_mark(chars[insert - 1]) {
            break;
        }
        insert -= 1;
    }
    chars.insert(insert, 'ˈ');
    chars.into_iter().collect()
}

fn is_combining_mark(ch: char) -> bool {
    matches!(ch, '\u{0300}'..='\u{036F}')
}

fn is_ipa_vowel(ch: char) -> bool {
    matches!(
        ch,
        'a' | 'ɑ' | 'e' | 'ɛ' | 'i' | 'o' | 'ɔ' | 'u' | 'y' | 'ø' | 'œ' | 'ə'
    )
}

fn phoneme_inventory() -> PhonemeInventory {
    PhonemeInventory {
        phonemes: SEGMENTS
            .iter()
            .map(|symbol| {
                let phoneme = Phoneme {
                    id: PhonemeId(format!("fr-FR-Standard.phoneme.{symbol}")),
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
        "a" | "ɑ̃" | "e" | "ɛ" | "ɛ̃" | "i" | "o" | "ɔ" | "ɔ̃" | "u" | "y" | "ø" | "œ" | "ə"
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
    fn french_synthesizes_common_words() {
        assert_eq!(synthesize_ipa("bonjour").as_deref(), Some("/bɔ̃ˈʒuʁ/"));
    }
}
