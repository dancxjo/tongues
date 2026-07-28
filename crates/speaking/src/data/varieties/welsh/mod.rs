use std::collections::HashMap;

use crate::feature::{FeatureBundle, FeatureSystem, FeatureValue};
use crate::ids::{LanguageId, PhoneId, PhonemeId, VarietyId};
use crate::orthography::Orthography;
use crate::phonetics::{Phone, PhoneInventory};
use crate::phonology::{Phoneme, PhonemeInventory};
use crate::segment::{SegmentStatus, SymbolAlias};
use crate::spec::Spec;
use crate::syntax::GrammarRuleSet;
use crate::variety::{
    LinguisticVariety, NumberNameSet, OrthographyPronunciationRules, VarietyImplementationStatus,
    VarietyStatus,
};

pub const REGISTRATIONS: &[crate::data::varieties::VarietyRegistration] =
    &[crate::data::varieties::VarietyRegistration {
        canonical_id: "cy-GB-Standard",
        aliases: &["cy", "cym", "wel", "cy-GB"],
        language_tag: "cy-GB",
        load: |_| variety(),
    }];

const SEGMENTS: &[&str] = &[
    "a", "aː", "ɛ", "eː", "ɪ", "iː", "ɔ", "oː", "ɨ", "ɨː", "ʊ", "uː", "ə", "ai̯", "au̯", "əi̯", "ɛu̯",
    "ɔi̯", "ɔu̯", "ɨu̯", "ʊi̯", "b", "d", "ð", "f", "v", "ɡ", "h", "j", "k", "l", "ɬ", "m", "n", "ŋ",
    "p", "r", "r̥", "s", "ʃ", "t", "θ", "t͡ʃ", "d͡ʒ", "w", "x", "z",
];

pub fn variety() -> LinguisticVariety {
    LinguisticVariety {
        id: VarietyId("cy-GB-Standard".into()),
        language: LanguageId("cy".into()),
        name: "Standard Welsh".into(),
        feature_system: FeatureSystem::default(),
        phonemes: phoneme_inventory(),
        phones: phone_inventory(),
        allophone_rules: Vec::new(),
        epenthesis_rules: Vec::new(),
        weak_forms: Vec::new(),
        orthographic_unit_pronunciations: Vec::new(),
        pronunciation_lexicons: Vec::new(),
        pronunciation_selection_rules: Vec::new(),
        pronunciation_pipeline: Some(
            crate::data::varieties::PRONUNCIATION_PIPELINE_VARIETY_DATA.into(),
        ),
        text_normalization: crate::data::varieties::small_number_text_normalization_profile(),
        syntax_profile: Some(crate::data::varieties::SYNTAX_PROFILE_WELSH.into()),
        syntax_analyzer: None,
        syntax_rules: Some(syntax_profile()),
        orthography_pronunciation: Some(OrthographyPronunciationRules {
            synthesize_ipa: Some(synthesize_ipa_for_orthography),
        }),
        number_names: Some(NumberNameSet {
            cardinal_0_to_20: [
                "sero",
                "un",
                "dau",
                "tri",
                "pedwar",
                "pump",
                "chwech",
                "saith",
                "wyth",
                "naw",
                "deg",
                "un ar ddeg",
                "deuddeg",
                "tri ar ddeg",
                "pedwar ar ddeg",
                "pymtheg",
                "un ar bymtheg",
                "dau ar bymtheg",
                "deunaw",
                "pedwar ar bymtheg",
                "ugain",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            ordinal_suffixes: Vec::new(),
            ..Default::default()
        }),
        punctuation: Some(crate::data::varieties::welsh_punctuation_profile()),
        question_contours: Some(crate::data::varieties::welsh_question_contour_profile()),
        connected_speech: Vec::new(),
        phonotactics: None,
        orthography: Some(Orthography {
            name: "Welsh Latin orthography".into(),
            pronunciation: Some(crate::data::varieties::ORTHOGRAPHY_PROFILE_WELSH.into()),
            initialism_joiners: vec!["a".into()],
            sample_words: vec!["Cymraeg".into()],
            sample_letter_units: vec!["A".into(), "B".into()],
            ..Default::default()
        }),
        morphology: None,
        acoustic_profile: None,
        prosody_profile: Some(crate::data::varieties::prosody_profile(
            crate::data::varieties::PROSODY_RHYTHM_STRESS_TIMED,
            4.5,
        )),
        status: VarietyStatus::Attested,
        // Welsh spelling is regular enough for a useful built-in fallback, but
        // lexical vowel length and north/south vowel quality require evidence
        // from the Wiktionary model or a pronunciation lexicon.
        implementation_status: VarietyImplementationStatus::PermissiveProfile,
    }
}

fn synthesize_ipa_for_orthography(
    word: &str,
    _variety: &LinguisticVariety,
    _part_of_speech: Option<crate::syntax::PartOfSpeech>,
) -> Option<String> {
    synthesize_ipa(word)
}

pub fn syntax_profile() -> GrammarRuleSet {
    GrammarRuleSet {
        determiners: &["y", "yr", "r"],
        pronouns: &[
            "mi", "fi", "ti", "ef", "fe", "hi", "ni", "chi", "nhw", "hwn", "hon", "hyn",
        ],
        object_pronouns: &["fi", "ti", "ef", "fe", "hi", "ni", "chi", "nhw"],
        auxiliaries: &[
            "ydw", "wyt", "mae", "ydyn", "ydych", "roedd", "bydd", "fydd", "wedi",
        ],
        copulas: &["ydw", "wyt", "yw", "mae", "ydyn", "ydych", "roedd", "bydd"],
        prepositions: &[
            "am", "ar", "at", "dan", "dros", "drwy", "gan", "heb", "i", "o", "rhag", "rhwng",
            "tros", "wrth",
        ],
        postpositions: &[],
        conjunctions: &["a", "ac", "ond", "neu", "na", "nag"],
        particles: &["mi", "fe", "yn"],
        enclitic_suffixes: &[],
        complementizers: &["bod", "mai", "taw", "os", "pan", "tra"],
        adverbs: &["ddim", "iawn", "hefyd", "nawr", "yma", "yno"],
        adverb_suffixes: &[],
        adjectives: &[],
        adjective_suffixes: &[],
        verbs: &[
            "bod", "mynd", "dod", "gwneud", "cael", "gallu", "mae", "roedd", "bydd",
        ],
        verb_suffixes: &[],
        subject_verb_suffixes: &[],
        non_verbs: &[],
        ..GrammarRuleSet::empty()
    }
}

#[derive(Clone, Copy)]
struct Unit {
    ipa: &'static str,
    vowel: bool,
}

pub fn synthesize_ipa(word: &str) -> Option<String> {
    let normalized = word.trim().to_lowercase();
    if normalized.is_empty()
        || normalized.chars().count() > 64
        || normalized
            .chars()
            .any(|ch| !(ch.is_alphabetic() || matches!(ch, '-' | '\'' | '’')))
    {
        return None;
    }

    let chars = normalized.chars().collect::<Vec<_>>();
    let mut units = Vec::new();
    let mut offset = 0;
    while offset < chars.len() {
        if matches!(chars[offset], '-' | '\'' | '’') {
            offset += 1;
            continue;
        }
        let pair = chars
            .get(offset + 1)
            .map(|next| [chars[offset], *next].into_iter().collect::<String>());
        if let Some(unit) = pair.as_deref().and_then(digraph) {
            units.push(unit);
            offset += 2;
            continue;
        }
        units.push(single(chars[offset], offset, &chars)?);
        offset += 1;
    }

    let nuclei = units
        .iter()
        .enumerate()
        .filter_map(|(index, unit)| unit.vowel.then_some(index))
        .collect::<Vec<_>>();
    let stressed_nucleus = match nuclei.len() {
        0 | 1 => None,
        _ if normalized == "cymraeg" => nuclei.last().copied(),
        count => Some(nuclei[count - 2]),
    };
    let stress = stressed_nucleus.map(|nucleus| {
        if nucleus > 0 && !units[nucleus - 1].vowel {
            nucleus - 1
        } else {
            nucleus
        }
    });
    let mut ipa = String::new();
    for (index, unit) in units.into_iter().enumerate() {
        if Some(index) == stress {
            ipa.push('ˈ');
        }
        ipa.push_str(unit.ipa);
    }
    (!ipa.is_empty()).then_some(format!("/{ipa}/"))
}

fn digraph(value: &str) -> Option<Unit> {
    let (ipa, vowel) = match value {
        "ch" => ("x", false),
        "dd" => ("ð", false),
        "ff" | "ph" => ("f", false),
        "ng" => ("ŋ", false),
        "ll" => ("ɬ", false),
        "rh" => ("r̥", false),
        "th" => ("θ", false),
        "ae" | "ai" => ("ai̯", true),
        "au" => ("au̯", true),
        "ei" => ("əi̯", true),
        "eu" | "ew" => ("ɛu̯", true),
        "oe" | "oi" => ("ɔi̯", true),
        "ou" | "ow" => ("ɔu̯", true),
        "uw" => ("ɨu̯", true),
        "wy" => ("ʊi̯", true),
        "yw" => ("ɨu̯", true),
        _ => return None,
    };
    Some(Unit { ipa, vowel })
}

fn single(ch: char, offset: usize, chars: &[char]) -> Option<Unit> {
    let (ipa, vowel) = match ch {
        'a' | 'â' => (if ch == 'â' { "aː" } else { "a" }, true),
        'e' | 'ê' => (if ch == 'ê' { "eː" } else { "ɛ" }, true),
        'i' | 'î' => (if ch == 'î' { "iː" } else { "ɪ" }, true),
        'o' | 'ô' => (if ch == 'ô' { "oː" } else { "ɔ" }, true),
        'u' | 'û' => (if ch == 'û' { "iː" } else { "ɪ" }, true),
        'y' | 'ŷ' => {
            let last_vowel = chars[offset + 1..]
                .iter()
                .all(|candidate| !matches!(candidate, 'a' | 'e' | 'i' | 'o' | 'u' | 'w' | 'y'));
            (
                if ch == 'ŷ' {
                    "ɨː"
                } else if last_vowel {
                    "ɨ"
                } else {
                    "ə"
                },
                true,
            )
        }
        'w' | 'ŵ' => {
            let vocalic = chars
                .get(offset + 1)
                .is_none_or(|next| !matches!(next, 'a' | 'e' | 'i' | 'o' | 'u' | 'y'));
            if vocalic {
                (if ch == 'ŵ' { "uː" } else { "ʊ" }, true)
            } else {
                ("w", false)
            }
        }
        'b' => ("b", false),
        'c' => ("k", false),
        'd' => ("d", false),
        'f' => ("v", false),
        'g' => ("ɡ", false),
        'h' => ("h", false),
        'j' => ("d͡ʒ", false),
        'l' => ("l", false),
        'm' => ("m", false),
        'n' => ("n", false),
        'p' => ("p", false),
        'r' => ("r", false),
        's' => ("s", false),
        't' => ("t", false),
        'v' => ("v", false),
        'x' => ("ks", false),
        'z' => ("z", false),
        _ => return None,
    };
    Some(Unit { ipa, vowel })
}

fn phoneme_inventory() -> PhonemeInventory {
    PhonemeInventory {
        phonemes: SEGMENTS
            .iter()
            .map(|symbol| {
                let id = PhonemeId(format!("cy-GB-Standard.phoneme.{symbol}"));
                (
                    id.clone(),
                    Phoneme {
                        id,
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
                    },
                )
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
            | "ɛ"
            | "eː"
            | "ɪ"
            | "iː"
            | "ɔ"
            | "oː"
            | "ɨ"
            | "ɨː"
            | "ʊ"
            | "uː"
            | "ə"
            | "ai̯"
            | "au̯"
            | "əi̯"
            | "ɛu̯"
            | "ɔi̯"
            | "ɔu̯"
            | "ɨu̯"
            | "ʊi̯"
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
    features.values.insert(
        crate::ids::FeatureId("phonology.base_symbol".into()),
        Spec::Known(FeatureValue::Category(symbol.into())),
    );
    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welsh_registration_and_inventory_are_concrete() {
        let welsh = variety();
        assert_eq!(welsh.language.0, "cy");
        for symbol in ["ɬ", "r̥", "x", "ð", "θ"] {
            assert!(
                welsh
                    .phonemes
                    .phonemes
                    .contains_key(&PhonemeId(format!("cy-GB-Standard.phoneme.{symbol}")))
            );
        }
    }

    #[test]
    fn welsh_synthesis_handles_native_digraphs_and_penultimate_stress() {
        assert_eq!(synthesize_ipa("Cymraeg").as_deref(), Some("/kəmˈrai̯ɡ/"));
        assert_eq!(synthesize_ipa("Llanelli").as_deref(), Some("/ɬaˈnɛɬɪ/"));
        assert_eq!(synthesize_ipa("rhydd").as_deref(), Some("/r̥ɨð/"));
    }
}
