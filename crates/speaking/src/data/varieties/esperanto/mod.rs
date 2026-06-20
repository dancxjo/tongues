use std::collections::HashMap;

use crate::feature::{FeatureBundle, FeatureSystem, FeatureValue};
use crate::ids::{LanguageId, PhoneId, PhonemeId, VarietyId};
use crate::orthography::Orthography;
use crate::phonetics::{Phone, PhoneInventory};
use crate::phonology::{Phoneme, PhonemeInventory};
use crate::rules::{PhonotacticConstraint, Phonotactics, RuleStatus, SyllableShape};
use crate::segment::{Environment, SegmentMatcher, SegmentStatus, SymbolAlias};
use crate::spec::Spec;
use crate::syntax::HeuristicSyntaxProfile;
use crate::variety::{
    LinguisticVariety, NumberNameSet, OrthographyPronunciationRules, VarietyImplementationStatus,
    VarietyStatus,
};

pub const REGISTRATIONS: &[crate::data::varieties::VarietyRegistration] =
    &[crate::data::varieties::VarietyRegistration {
        canonical_id: "eo",
        aliases: &[],
        load: |_| variety(),
    }];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EsperantoSegment {
    grapheme: &'static str,
    symbol: &'static str,
}

const PHONEMES: &[EsperantoSegment] = &[
    seg("a", "a"),
    seg("b", "b"),
    seg("c", "t͡s"),
    seg("ĉ", "t͡ʃ"),
    seg("d", "d"),
    seg("e", "e"),
    seg("f", "f"),
    seg("g", "ɡ"),
    seg("ĝ", "d͡ʒ"),
    seg("h", "h"),
    seg("ĥ", "x"),
    seg("i", "i"),
    seg("j", "j"),
    seg("ĵ", "ʒ"),
    seg("k", "k"),
    seg("l", "l"),
    seg("m", "m"),
    seg("n", "n"),
    seg("o", "o"),
    seg("p", "p"),
    seg("r", "r"),
    seg("s", "s"),
    seg("ŝ", "ʃ"),
    seg("t", "t"),
    seg("u", "u"),
    seg("ŭ", "w"),
    seg("v", "v"),
    seg("z", "z"),
];

const fn seg(grapheme: &'static str, symbol: &'static str) -> EsperantoSegment {
    EsperantoSegment { grapheme, symbol }
}

pub fn variety() -> LinguisticVariety {
    let mut phonemes = HashMap::new();
    let mut phones = HashMap::new();
    for segment in PHONEMES {
        let phone_id = PhoneId(format!("ipa.phone.{}", segment.symbol).into());
        phones.insert(
            phone_id.clone(),
            Phone {
                id: phone_id.clone(),
                ipa: segment.symbol.into(),
                features: segment_features(segment.symbol),
                aliases: vec![SymbolAlias {
                    system: "esperanto".into(),
                    symbol: segment.grapheme.into(),
                }],
                status: SegmentStatus::Core,
            },
        );
        let phoneme = Phoneme {
            id: PhonemeId(format!("eo.phoneme.{}", segment.symbol)),
            notation: format!("/{}/", segment.symbol),
            features: segment_features(segment.symbol),
            default_phone: Some(phone_id.clone()),
            possible_phones: vec![phone_id],
            aliases: aliases(segment.grapheme, segment.symbol),
            allophones: Vec::new(),
            status: SegmentStatus::Core,
        };
        phonemes.insert(phoneme.id.clone(), phoneme);
    }

    LinguisticVariety {
        id: VarietyId("eo".into()),
        language: LanguageId("eo".into()),
        name: "Esperanto".into(),
        feature_system: FeatureSystem::default(),
        phonemes: PhonemeInventory { phonemes },
        phones: PhoneInventory { phones },
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
        syntax_profile: Some(crate::data::varieties::SYNTAX_PROFILE_ESPERANTO.into()),
        syntax_analyzer: None,
        syntax_heuristics: Some(syntax_profile()),
        orthography_pronunciation: Some(OrthographyPronunciationRules {
            synthesize_ipa: Some(synthesize_ipa_for_orthography),
        }),
        number_names: Some(NumberNameSet {
            cardinal_0_to_20: [
                "nul", "unu", "du", "tri", "kvar", "kvin", "ses", "sep", "ok", "naŭ", "dek",
                "dek unu", "dek du", "dek tri", "dek kvar", "dek kvin", "dek ses", "dek sep",
                "dek ok", "dek naŭ", "dudek",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            ordinal_suffixes: Vec::new(),
            ..Default::default()
        }),
        punctuation: Some(crate::data::varieties::esperanto_punctuation_profile()),
        question_contours: Some(crate::data::varieties::esperanto_question_contour_profile()),
        connected_speech: Vec::new(),
        phonotactics: Some(Phonotactics {
            allowed_syllable_shapes: vec![
                SyllableShape {
                    pattern: "V".into(),
                },
                SyllableShape {
                    pattern: "CV".into(),
                },
                SyllableShape {
                    pattern: "CVC".into(),
                },
                SyllableShape {
                    pattern: "CCV".into(),
                },
            ],
            constraints: [
                &["p", "l"][..],
                &["p", "r"][..],
                &["b", "l"][..],
                &["b", "r"][..],
                &["t", "r"][..],
                &["d", "r"][..],
                &["k", "l"][..],
                &["k", "r"][..],
                &["ɡ", "l"][..],
                &["ɡ", "r"][..],
                &["f", "l"][..],
                &["f", "r"][..],
            ]
            .into_iter()
            .map(cluster_constraint)
            .collect(),
        }),
        orthography: Some(Orthography {
            name: "Esperanto Latin orthography".into(),
            pronunciation: Some(crate::data::varieties::ORTHOGRAPHY_PROFILE_ESPERANTO.into()),
            initialism_joiners: vec!["kaj".into()],
            sample_words: vec!["ŝipo".into()],
            sample_letter_units: vec!["A".into(), "B".into()],
            ..Default::default()
        }),
        morphology: None,
        acoustic_profile: None,
        prosody_profile: Some(crate::data::varieties::prosody_profile(
            crate::data::varieties::PROSODY_RHYTHM_SYLLABLE_TIMED,
            5.0,
        )),
        status: VarietyStatus::Attested,
        implementation_status: VarietyImplementationStatus::Complete,
    }
}

fn synthesize_ipa_for_orthography(
    word: &str,
    _variety: &LinguisticVariety,
    _part_of_speech: Option<crate::syntax::PartOfSpeech>,
) -> Option<String> {
    synthesize_ipa(word)
}

pub fn syntax_profile() -> HeuristicSyntaxProfile {
    HeuristicSyntaxProfile {
        determiners: &["la", "tiu", "tiuj", "ĉi", "ci"],
        pronouns: &[
            "mi", "vi", "li", "ŝi", "ĝi", "ni", "ili", "oni", "si", "min", "vin", "lin", "ŝin",
            "ĝin", "nin", "ilin", "kiu", "kio",
        ],
        object_pronouns: &["min", "vin", "lin", "ŝin", "ĝin", "nin", "ilin"],
        auxiliaries: &[],
        copulas: &["estas", "estis", "estos", "estus", "estu"],
        prepositions: &[
            "al", "de", "en", "kun", "sen", "por", "per", "pri", "sur", "sub", "inter", "antaŭ",
            "post", "kontraŭ",
        ],
        postpositions: &[],
        conjunctions: &["kaj", "aŭ", "sed", "nek"],
        particles: &["ĉu", "ja"],
        enclitic_suffixes: &[],
        complementizers: &["ke", "ĉu", "kiam", "se", "kiel"],
        adverbs: &["ne", "tre", "ankaŭ", "jam", "nun"],
        adverb_suffixes: &["e"],
        adjectives: &[],
        adjective_suffixes: &["a", "aj", "an"],
        verbs: &["estas", "estis", "estos", "estus", "estu"],
        verb_suffixes: &["i", "as", "is", "os", "us", "u"],
        subject_verb_suffixes: &[],
        non_verbs: &[],
        ..HeuristicSyntaxProfile::empty()
    }
}

pub fn synthesize_ipa(word: &str) -> Option<String> {
    let normalized = normalize_esperanto_word(word)?;
    let mut ipa = String::new();
    let vowel_count = normalized.chars().filter(|ch| is_vowel(*ch)).count();
    let stress_vowel = vowel_count.checked_sub(2);
    let mut vowel_index = 0usize;
    for ch in normalized.chars() {
        if ch == '-' || ch == '\'' || ch == '’' {
            continue;
        }
        if is_vowel(ch) {
            if Some(vowel_index) == stress_vowel {
                ipa.push('ˈ');
            }
            vowel_index += 1;
        }
        ipa.push_str(grapheme_symbol(ch)?);
    }
    let ipa = reposition_primary_stress(&ipa);
    (!ipa.is_empty()).then_some(format!("/{ipa}/"))
}

fn normalize_esperanto_word(word: &str) -> Option<String> {
    let normalized = word.trim().to_lowercase();
    if normalized.is_empty()
        || normalized.chars().count() > 48
        || normalized
            .chars()
            .any(|ch| !(ch.is_alphabetic() || matches!(ch, '-' | '\'' | '’')))
    {
        return None;
    }
    Some(normalized)
}

fn grapheme_symbol(ch: char) -> Option<&'static str> {
    Some(match ch {
        'a' => "a",
        'b' => "b",
        'c' => "t͡s",
        'ĉ' => "t͡ʃ",
        'd' => "d",
        'e' => "e",
        'f' => "f",
        'g' => "ɡ",
        'ĝ' => "d͡ʒ",
        'h' => "h",
        'ĥ' => "x",
        'i' => "i",
        'j' => "j",
        'ĵ' => "ʒ",
        'k' => "k",
        'l' => "l",
        'm' => "m",
        'n' => "n",
        'o' => "o",
        'p' => "p",
        'r' => "r",
        's' => "s",
        'ŝ' => "ʃ",
        't' => "t",
        'u' => "u",
        'ŭ' => "w",
        'v' => "v",
        'z' => "z",
        _ => return None,
    })
}

fn aliases(grapheme: &str, symbol: &str) -> Vec<SymbolAlias> {
    vec![
        SymbolAlias {
            system: "esperanto".into(),
            symbol: grapheme.into(),
        },
        SymbolAlias {
            system: "ipa".into(),
            symbol: symbol.into(),
        },
    ]
}

fn segment_features(symbol: &str) -> FeatureBundle {
    let mut features = FeatureBundle::default();
    let is_vowel = matches!(symbol, "a" | "e" | "i" | "o" | "u");
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

fn is_vowel(ch: char) -> bool {
    matches!(ch, 'a' | 'e' | 'i' | 'o' | 'u')
}

fn reposition_primary_stress(ipa: &str) -> String {
    let mut chars = ipa.chars().collect::<Vec<_>>();
    let Some(stress_index) = chars.iter().position(|ch| *ch == 'ˈ') else {
        return ipa.to_string();
    };
    let mut insert_index = stress_index;
    while insert_index > 0 && !is_vowel(chars[insert_index - 1]) && chars[insert_index - 1] != '|' {
        insert_index -= 1;
    }
    if insert_index == stress_index {
        return ipa.to_string();
    }
    chars.remove(stress_index);
    chars.insert(insert_index, 'ˈ');
    chars.into_iter().collect()
}

fn cluster_constraint(cluster: &[&str]) -> PhonotacticConstraint {
    let label = cluster.join("");
    PhonotacticConstraint {
        id: format!("eo.legal_onset.{label}"),
        description: format!("Legal Esperanto onset cluster {label}"),
        matcher: SegmentMatcher::Any,
        environment: Environment {
            before: cluster
                .iter()
                .map(|symbol| SegmentMatcher::Phone(PhoneId(format!("ipa.phone.{symbol}").into())))
                .collect(),
            syllable_position: Spec::Known(crate::segment::SyllablePosition::Onset),
            ..Default::default()
        },
        status: RuleStatus::Productive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn esperanto_loads_expected_phonemes() {
        let eo = variety();
        assert!(
            eo.phonemes
                .phonemes
                .contains_key(&PhonemeId("eo.phoneme.a".into()))
        );
        assert!(
            eo.phonemes
                .phonemes
                .contains_key(&PhonemeId("eo.phoneme.ʃ".into()))
        );
    }

    #[test]
    fn esperanto_synthesizes_regular_stress() {
        assert_eq!(synthesize_ipa("amiko").as_deref(), Some("/aˈmiko/"));
        assert_eq!(synthesize_ipa("ŝipo").as_deref(), Some("/ˈʃipo/"));
    }
}
