use std::collections::HashMap;

use crate::feature::{FeatureBundle, FeatureSystem, FeatureValue};
use crate::ids::{LanguageId, PhoneId, PhonemeId, VarietyId};
use crate::orthography::Orthography;
use crate::phonetics::{Phone, PhoneInventory};
use crate::phonology::{Phoneme, PhonemeInventory};
use crate::segment::{SegmentStatus, SymbolAlias};
use crate::spec::Spec;
use crate::syntax::HeuristicSyntaxProfile;
use crate::variety::{
    LinguisticVariety, NumberNameSet, VarietyImplementationStatus, VarietyStatus,
};

const SEGMENTS: &[&str] = &[
    "a", "aː", "i", "iː", "u", "uː", "r̩", "eː", "ai̯", "oː", "au̯", "k", "kʰ", "ɡ", "ɡʱ", "ŋ", "t͡ɕ",
    "t͡ɕʰ", "d͡ʑ", "d͡ʑʱ", "ɲ", "ʈ", "ʈʰ", "ɖ", "ɖʱ", "ɳ", "t", "tʰ", "d", "dʱ", "n", "p", "pʰ", "b",
    "bʱ", "m", "j", "r", "l", "v", "ɕ", "ʂ", "s", "ɦ", "h",
];

pub fn variety() -> LinguisticVariety {
    LinguisticVariety {
        id: VarietyId("sa-Deva-Standard".into()),
        language: LanguageId("sa".into()),
        name: "Sanskrit".into(),
        feature_system: FeatureSystem::default(),
        phonemes: phoneme_inventory(),
        phones: phone_inventory(),
        allophone_rules: Vec::new(),
        epenthesis_rules: Vec::new(),
        weak_forms: Vec::new(),
        orthographic_unit_pronunciations: Vec::new(),
        pronunciation_lexicons: Vec::new(),
        syntax_profile: Some("sanskrit".into()),
        number_names: Some(NumberNameSet {
            cardinal_0_to_20: [
                "śūnya",
                "eka",
                "dvi",
                "tri",
                "catur",
                "pañca",
                "ṣaṣ",
                "sapta",
                "aṣṭa",
                "nava",
                "daśa",
                "ekādaśa",
                "dvādaśa",
                "trayodaśa",
                "caturdaśa",
                "pañcadaśa",
                "ṣoḍaśa",
                "saptadaśa",
                "aṣṭādaśa",
                "navadaśa",
                "viṃśati",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            ordinal_suffixes: Vec::new(),
        }),
        connected_speech: Vec::new(),
        phonotactics: None,
        orthography: Some(Orthography {
            name: "Sanskrit Devanagari and transliteration".into(),
            pronunciation: Some("sanskrit".into()),
            ..Default::default()
        }),
        morphology: None,
        acoustic_profile: None,
        prosody_profile: None,
        status: VarietyStatus::Attested,
        implementation_status: VarietyImplementationStatus::Complete,
    }
}

pub fn syntax_profile() -> HeuristicSyntaxProfile {
    HeuristicSyntaxProfile {
        determiners: &[],
        pronouns: &[
            "अहम्",
            "अहं",
            "त्वम्",
            "सः",
            "सा",
            "तत्",
            "वयम्",
            "यूयम्",
            "ते",
            "असौ",
            "कः",
            "का",
            "किम्",
            "aham",
            "ahaṃ",
            "tvam",
            "saḥ",
            "sā",
            "tat",
            "vayam",
            "yūyam",
            "te",
            "kaḥ",
            "kā",
            "kim",
        ],
        object_pronouns: &[
            "माम्",
            "त्वाम्",
            "एनम्",
            "एनाम्",
            "तत्",
            "अस्मान्",
            "युष्मान्",
            "mām",
            "tvām",
            "enam",
            "enām",
            "tat",
            "asmān",
            "yuṣmān",
        ],
        auxiliaries: &["अस्मि", "असि", "अस्ति", "स्मः", "स्थ", "सन्ति", "आसीत्"],
        copulas: &[
            "अस्मि",
            "असि",
            "अस्ति",
            "स्मः",
            "स्थ",
            "सन्ति",
            "आसीत्",
            "asmi",
            "asi",
            "asti",
            "smaḥ",
            "stha",
            "santi",
            "āsīt",
        ],
        prepositions: &["प्रति", "अनु", "अधि", "उप", "परि", "वि"],
        postpositions: &[
            "मध्ये",
            "पूर्वम्",
            "परम्",
            "अर्थम्",
            "madhye",
            "pūrvam",
            "param",
            "artham",
        ],
        conjunctions: &["च", "वा", "तु", "अथ"],
        particles: &["च", "हि", "एव", "नु", "चेत्", "ca", "hi", "eva", "nu", "cet"],
        enclitic_suffixes: &[],
        complementizers: &["यत्", "यदि", "चेत्", "yat", "yadi", "cet"],
        adverbs: &["न", "मा", "सु", "एव"],
        adverb_suffixes: &[],
        adjectives: &[],
        adjective_suffixes: &[],
        verbs: &[
            "गच्छति",
            "गच्छन्ति",
            "भवति",
            "भवन्ति",
            "वदति",
            "वदन्ति",
            "खादति",
            "अस्ति",
            "सन्ति",
            "gacchati",
            "gacchanti",
            "bhavati",
            "bhavanti",
            "vadati",
            "vadanti",
            "khādati",
            "khadati",
            "asti",
            "santi",
        ],
        verb_suffixes: &[],
        subject_verb_suffixes: &["ति", "न्ति", "मि", "सि", "तः", "थ"],
        non_verbs: &[],
        ..HeuristicSyntaxProfile::empty()
    }
}

pub fn synthesize_ipa(word: &str) -> Option<String> {
    let trimmed = word.trim();
    if trimmed.is_empty() || trimmed.chars().count() > 48 {
        return None;
    }
    let ipa = if trimmed.chars().any(is_devanagari) {
        synthesize_devanagari(trimmed)?
    } else {
        synthesize_iast(trimmed)?
    };
    let ipa = add_initial_stress(&ipa);
    (!ipa.is_empty()).then_some(format!("/{ipa}/"))
}

fn synthesize_devanagari(text: &str) -> Option<String> {
    let chars = text.chars().collect::<Vec<_>>();
    let mut ipa = String::new();
    let mut index = 0usize;
    while index < chars.len() {
        let ch = chars[index];
        if matches!(ch, '-' | '\'' | '’') {
            index += 1;
            continue;
        }
        if let Some(vowel) = independent_vowel(ch) {
            ipa.push_str(vowel);
            index += 1;
            continue;
        }
        if let Some(consonant) = consonant(ch) {
            ipa.push_str(consonant);
            match chars.get(index + 1).copied() {
                Some('्') => {
                    index += 2;
                    continue;
                }
                Some(next) if vowel_mark(next).is_some() => {
                    ipa.push_str(vowel_mark(next)?);
                    index += 2;
                    continue;
                }
                _ => {
                    ipa.push('a');
                    index += 1;
                    continue;
                }
            }
        }
        match ch {
            'ं' | 'ँ' => ipa.push_str(anusvara_for_devanagari(chars.get(index + 1).copied())),
            'ः' => ipa.push('h'),
            '।' | '॥' => {}
            _ => return None,
        }
        index += 1;
    }
    Some(ipa)
}

fn synthesize_iast(text: &str) -> Option<String> {
    let normalized = text.trim().to_lowercase();
    if normalized.chars().any(|ch| {
        !(ch.is_alphabetic()
            || matches!(
                ch,
                '-' | '\''
                    | '’'
                    | 'ṃ'
                    | 'ḥ'
                    | 'ā'
                    | 'ī'
                    | 'ū'
                    | 'ṛ'
                    | 'ṅ'
                    | 'ñ'
                    | 'ṭ'
                    | 'ḍ'
                    | 'ṇ'
                    | 'ś'
                    | 'ṣ'
            ))
    }) {
        return None;
    }
    let chars = normalized.chars().collect::<Vec<_>>();
    let mut ipa = String::new();
    let mut index = 0usize;
    while index < chars.len() {
        let rest = chars[index..].iter().collect::<String>();
        let (symbol, consumed) = if rest.starts_with("ai") {
            ("ai̯", 2)
        } else if rest.starts_with("au") {
            ("au̯", 2)
        } else if rest.starts_with("kh") {
            ("kʰ", 2)
        } else if rest.starts_with("gh") {
            ("ɡʱ", 2)
        } else if rest.starts_with("ch") {
            ("t͡ɕʰ", 2)
        } else if rest.starts_with("jh") {
            ("d͡ʑʱ", 2)
        } else if rest.starts_with("ṭh") {
            ("ʈʰ", 2)
        } else if rest.starts_with("ḍh") {
            ("ɖʱ", 2)
        } else if rest.starts_with("th") {
            ("tʰ", 2)
        } else if rest.starts_with("dh") {
            ("dʱ", 2)
        } else if rest.starts_with("ph") {
            ("pʰ", 2)
        } else if rest.starts_with("bh") {
            ("bʱ", 2)
        } else if chars[index] == 'ṃ' {
            (anusvara_for_iast(chars.get(index + 1).copied()), 1)
        } else {
            (iast_single(chars[index])?, 1)
        };
        ipa.push_str(symbol);
        index += consumed;
    }
    Some(ipa)
}

fn independent_vowel(ch: char) -> Option<&'static str> {
    Some(match ch {
        'अ' => "a",
        'आ' => "aː",
        'इ' => "i",
        'ई' => "iː",
        'उ' => "u",
        'ऊ' => "uː",
        'ऋ' => "r̩",
        'ए' => "eː",
        'ऐ' => "ai̯",
        'ओ' => "oː",
        'औ' => "au̯",
        _ => return None,
    })
}

fn vowel_mark(ch: char) -> Option<&'static str> {
    Some(match ch {
        'ा' => "aː",
        'ि' => "i",
        'ी' => "iː",
        'ु' => "u",
        'ू' => "uː",
        'ृ' => "r̩",
        'े' => "eː",
        'ै' => "ai̯",
        'ो' => "oː",
        'ौ' => "au̯",
        _ => return None,
    })
}

fn consonant(ch: char) -> Option<&'static str> {
    Some(match ch {
        'क' => "k",
        'ख' => "kʰ",
        'ग' => "ɡ",
        'घ' => "ɡʱ",
        'ङ' => "ŋ",
        'च' => "t͡ɕ",
        'छ' => "t͡ɕʰ",
        'ज' => "d͡ʑ",
        'झ' => "d͡ʑʱ",
        'ञ' => "ɲ",
        'ट' => "ʈ",
        'ठ' => "ʈʰ",
        'ड' => "ɖ",
        'ढ' => "ɖʱ",
        'ण' => "ɳ",
        'त' => "t",
        'थ' => "tʰ",
        'द' => "d",
        'ध' => "dʱ",
        'न' => "n",
        'प' => "p",
        'फ' => "pʰ",
        'ब' => "b",
        'भ' => "bʱ",
        'म' => "m",
        'य' => "j",
        'र' => "r",
        'ल' => "l",
        'व' => "v",
        'श' => "ɕ",
        'ष' => "ʂ",
        'स' => "s",
        'ह' => "ɦ",
        _ => return None,
    })
}

fn iast_single(ch: char) -> Option<&'static str> {
    Some(match ch {
        'a' => "a",
        'ā' => "aː",
        'i' => "i",
        'ī' => "iː",
        'u' => "u",
        'ū' => "uː",
        'ṛ' => "r̩",
        'e' => "eː",
        'o' => "oː",
        'k' => "k",
        'g' => "ɡ",
        'ṅ' => "ŋ",
        'c' => "t͡ɕ",
        'j' => "d͡ʑ",
        'ñ' => "ɲ",
        'ṭ' => "ʈ",
        'ḍ' => "ɖ",
        'ṇ' => "ɳ",
        't' => "t",
        'd' => "d",
        'n' => "n",
        'p' => "p",
        'b' => "b",
        'm' => "m",
        'y' => "j",
        'r' => "r",
        'l' => "l",
        'v' => "v",
        'ś' => "ɕ",
        'ṣ' => "ʂ",
        's' => "s",
        'h' | 'ḥ' => "h",
        'ṃ' => "ŋ",
        '-' | '\'' | '’' => "",
        _ => return None,
    })
}

fn anusvara_for_devanagari(next: Option<char>) -> &'static str {
    match next {
        Some('क' | 'ख' | 'ग' | 'घ' | 'ङ') => "ŋ",
        Some('च' | 'छ' | 'ज' | 'झ' | 'ञ') => "ɲ",
        Some('ट' | 'ठ' | 'ड' | 'ढ' | 'ण') => "ɳ",
        Some('त' | 'थ' | 'द' | 'ध' | 'न') => "n",
        Some('प' | 'फ' | 'ब' | 'भ' | 'म') => "m",
        _ => "ŋ",
    }
}

fn anusvara_for_iast(next: Option<char>) -> &'static str {
    match next {
        Some('k' | 'g' | 'ṅ') => "ŋ",
        Some('c' | 'j' | 'ñ') => "ɲ",
        Some('ṭ' | 'ḍ' | 'ṇ') => "ɳ",
        Some('t' | 'd' | 'n') => "n",
        Some('p' | 'b' | 'm') => "m",
        _ => "ŋ",
    }
}

fn is_devanagari(ch: char) -> bool {
    ('\u{0900}'..='\u{097F}').contains(&ch)
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
    matches!(ch, 'a' | 'i' | 'u' | 'r' | 'e' | 'o')
}

fn phoneme_inventory() -> PhonemeInventory {
    PhonemeInventory {
        phonemes: SEGMENTS
            .iter()
            .map(|symbol| {
                let phoneme = Phoneme {
                    id: PhonemeId(format!("sa-Deva-Standard.phoneme.{symbol}")),
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
        "a" | "aː" | "i" | "iː" | "u" | "uː" | "r̩" | "eː" | "ai̯" | "oː" | "au̯"
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
    fn sanskrit_synthesizes_devanagari_and_iast() {
        assert_eq!(synthesize_ipa("धर्म").as_deref(), Some("/ˈdʱarma/"));
        assert_eq!(synthesize_ipa("dharma").as_deref(), Some("/ˈdʱarma/"));
    }

    #[test]
    fn sanskrit_handles_iast_diphthongs_and_contextual_anusvara() {
        assert_eq!(synthesize_ipa("aiśa").as_deref(), Some("/ˈai̯ɕa/"));
        assert_eq!(synthesize_ipa("saṃbhava").as_deref(), Some("/ˈsambʱava/"));
        assert_eq!(synthesize_ipa("संबन्ध").as_deref(), Some("/ˈsambandʱa/"));
        assert_eq!(synthesize_ipa("संकेत").as_deref(), Some("/ˈsaŋkeːta/"));
    }
}
