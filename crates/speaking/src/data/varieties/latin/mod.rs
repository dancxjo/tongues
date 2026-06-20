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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LatinVariety {
    Classical,
    Ecclesiastical,
}

impl LatinVariety {
    fn from_id(id: &str) -> Option<Self> {
        match id {
            "la" | "la-Classical" => Some(Self::Classical),
            "la-Ecclesiastical" | "la-Church" => Some(Self::Ecclesiastical),
            _ => None,
        }
    }

    fn id(self) -> &'static str {
        match self {
            Self::Classical => "la-Classical",
            Self::Ecclesiastical => "la-Ecclesiastical",
        }
    }
}

const CLASSICAL_SEGMENTS: &[&str] = &[
    "a", "e", "i", "o", "u", "y", "ae̯", "au̯", "oe̯", "b", "k", "kʰ", "d", "f", "ɡ", "h", "j", "l",
    "m", "n", "p", "pʰ", "r", "s", "t", "tʰ", "w", "ks", "z",
];

const ECCLESIASTICAL_SEGMENTS: &[&str] = &[
    "a", "e", "i", "o", "u", "y", "ae", "au", "oe", "b", "k", "t͡ʃ", "d", "f", "ɡ", "d͡ʒ", "h", "j",
    "l", "m", "n", "p", "r", "s", "ʃ", "t", "t͡s", "w", "v", "ks", "z",
];

pub fn variety(id: &str) -> LinguisticVariety {
    let latin = LatinVariety::from_id(id).unwrap_or(LatinVariety::Classical);
    let phonemes = phoneme_inventory(latin);
    let phones = phone_inventory(latin);

    LinguisticVariety {
        id: VarietyId(latin.id().into()),
        language: LanguageId("la".into()),
        name: match latin {
            LatinVariety::Classical => "Classical Latin".into(),
            LatinVariety::Ecclesiastical => "Ecclesiastical Latin".into(),
        },
        feature_system: FeatureSystem::default(),
        phonemes,
        phones,
        allophone_rules: Vec::new(),
        epenthesis_rules: Vec::new(),
        weak_forms: Vec::new(),
        orthographic_unit_pronunciations: Vec::new(),
        pronunciation_lexicons: Vec::new(),
        pronunciation_pipeline: Some(
            crate::data::varieties::PRONUNCIATION_PIPELINE_VARIETY_DATA.into(),
        ),
        syntax_profile: Some(crate::data::varieties::SYNTAX_PROFILE_LATIN.into()),
        syntax_analyzer: None,
        syntax_heuristics: Some(syntax_profile()),
        orthography_pronunciation: Some(OrthographyPronunciationRules {
            synthesize_ipa: Some(synthesize_ipa_for_orthography),
        }),
        number_names: Some(NumberNameSet {
            cardinal_0_to_20: [
                "nihil",
                "unus",
                "duo",
                "tres",
                "quattuor",
                "quinque",
                "sex",
                "septem",
                "octo",
                "novem",
                "decem",
                "undecim",
                "duodecim",
                "tredecim",
                "quattuordecim",
                "quindecim",
                "sedecim",
                "septendecim",
                "duodeviginti",
                "undeviginti",
                "viginti",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            ordinal_suffixes: Vec::new(),
        }),
        punctuation: Some(crate::data::varieties::latin_punctuation_profile()),
        question_contours: Some(crate::data::varieties::latin_question_contour_profile()),
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
                &["b", "r"][..],
                &["k", "l"][..],
                &["k", "r"][..],
                &["d", "r"][..],
                &["f", "l"][..],
                &["f", "r"][..],
                &["ɡ", "l"][..],
                &["ɡ", "r"][..],
                &["p", "l"][..],
                &["p", "r"][..],
                &["t", "r"][..],
            ]
            .into_iter()
            .map(|cluster| cluster_constraint(latin, cluster))
            .collect(),
        }),
        orthography: Some(Orthography {
            name: "Latin orthography".into(),
            pronunciation: Some(crate::data::varieties::ORTHOGRAPHY_PROFILE_LATIN.into()),
            initialism_joiners: vec!["et".into(), "ac".into(), "atque".into()],
            ..Default::default()
        }),
        morphology: None,
        acoustic_profile: None,
        prosody_profile: Some(crate::data::varieties::prosody_profile(
            crate::data::varieties::PROSODY_RHYTHM_MORA_TIMED,
            4.8,
        )),
        status: VarietyStatus::Attested,
        implementation_status: VarietyImplementationStatus::Complete,
    }
}

fn synthesize_ipa_for_orthography(
    word: &str,
    variety: &LinguisticVariety,
    _part_of_speech: Option<crate::syntax::PartOfSpeech>,
) -> Option<String> {
    synthesize_ipa_for_variety(word, &variety.id.0)
}

pub fn syntax_profile() -> HeuristicSyntaxProfile {
    HeuristicSyntaxProfile {
        determiners: &[
            "hic", "haec", "hoc", "ille", "illa", "illud", "iste", "ista", "istud", "is", "ea",
            "id", "meus", "mea", "tuus", "tua", "suus", "sua",
        ],
        pronouns: &[
            "ego", "tu", "nos", "vos", "me", "te", "se", "qui", "quae", "quod", "quis", "quid",
        ],
        object_pronouns: &[
            "me", "te", "se", "nos", "vos", "eum", "eam", "id", "eos", "eas",
        ],
        auxiliaries: &[
            "sum", "es", "est", "sumus", "estis", "sunt", "eram", "erat", "erant", "fui", "fuit",
            "fuerunt",
        ],
        copulas: &[
            "sum", "es", "est", "sumus", "estis", "sunt", "eram", "erat", "erant", "fui", "fuit",
            "fuerunt",
        ],
        prepositions: &[
            "ad", "in", "de", "cum", "sine", "per", "pro", "sub", "super", "ab", "ex",
        ],
        postpositions: &["causa", "gratia"],
        conjunctions: &["et", "aut", "sed", "atque", "nec", "neque", "vel"],
        particles: &["ne", "que", "ve"],
        enclitic_suffixes: &["que", "ve", "ne"],
        complementizers: &["quod", "ut", "ne", "cum", "quia", "si"],
        adverbs: &["non", "ne", "bene", "male", "iam", "nunc"],
        adverb_suffixes: &[],
        adjectives: &[],
        adjective_suffixes: &["ior", "ilis", "alis"],
        verbs: &[
            "amo", "amas", "amat", "amamus", "amatis", "amant", "video", "videt", "dico", "dicit",
            "lego", "legit", "legunt", "puto", "putat", "venio", "venit", "sum", "es", "est",
            "sunt",
        ],
        verb_suffixes: &["are", "ere", "ire"],
        subject_verb_suffixes: &["o", "m", "s", "t", "mus", "tis", "nt"],
        non_verbs: &[],
        ..HeuristicSyntaxProfile::empty()
    }
}

pub fn synthesize_ipa_for_variety(word: &str, variety_id: &str) -> Option<String> {
    let variety = LatinVariety::from_id(variety_id)?;
    synthesize_ipa(word, variety)
}

fn synthesize_ipa(word: &str, variety: LatinVariety) -> Option<String> {
    let chars = normalize_latin_word(word)?;
    let vowel_index = stress_vowel_index(&chars)?;
    let mut ipa = String::new();
    let mut vowels_seen = 0usize;
    let mut index = 0usize;
    while index < chars.len() {
        let c = chars[index];
        let next = chars.get(index + 1).copied();
        let after_next = chars.get(index + 2).copied();

        if is_vowel(c) || starts_diphthong(&chars, index, variety).is_some() {
            if vowels_seen == vowel_index {
                ipa.push('ˈ');
            }
            vowels_seen += 1;
        }

        if let Some((symbol, consumed)) = starts_diphthong(&chars, index, variety) {
            ipa.push_str(symbol);
            index += consumed;
            continue;
        }

        match c {
            'a' | 'ā' => ipa.push('a'),
            'e' | 'ē' => ipa.push('e'),
            'i' | 'ī' => {
                if is_consonantal_i(&chars, index) {
                    match variety {
                        LatinVariety::Classical => ipa.push('j'),
                        LatinVariety::Ecclesiastical => ipa.push_str("d͡ʒ"),
                    }
                } else {
                    ipa.push('i');
                }
            }
            'o' | 'ō' => ipa.push('o'),
            'u' | 'ū' => {
                if is_consonantal_u(&chars, index) {
                    match variety {
                        LatinVariety::Classical => ipa.push('w'),
                        LatinVariety::Ecclesiastical => ipa.push('v'),
                    }
                } else {
                    ipa.push('u');
                }
            }
            'y' | 'ȳ' => ipa.push('y'),
            'b' => ipa.push('b'),
            'c' if matches!(next, Some('h')) => {
                match variety {
                    LatinVariety::Classical => ipa.push_str("kʰ"),
                    LatinVariety::Ecclesiastical => ipa.push('k'),
                }
                index += 1;
            }
            'c' if matches!(variety, LatinVariety::Ecclesiastical)
                && is_ecclesiastical_front_context(&chars, index) =>
            {
                ipa.push_str("t͡ʃ");
            }
            'c' => ipa.push('k'),
            'd' => ipa.push('d'),
            'f' => ipa.push('f'),
            'g' if matches!(variety, LatinVariety::Ecclesiastical)
                && is_ecclesiastical_front_context(&chars, index) =>
            {
                ipa.push_str("d͡ʒ");
            }
            'g' => ipa.push('ɡ'),
            'h' => ipa.push('h'),
            'l' => ipa.push('l'),
            'm' => ipa.push('m'),
            'n' => ipa.push('n'),
            'p' if matches!(next, Some('h')) => {
                match variety {
                    LatinVariety::Classical => ipa.push_str("pʰ"),
                    LatinVariety::Ecclesiastical => ipa.push('f'),
                }
                index += 1;
            }
            'p' => ipa.push('p'),
            'q' if matches!(next, Some('u' | 'ū')) => {
                ipa.push('k');
                ipa.push(match variety {
                    LatinVariety::Classical => 'w',
                    LatinVariety::Ecclesiastical => 'v',
                });
                index += 1;
            }
            'q' => ipa.push('k'),
            'r' => ipa.push('r'),
            's' if matches!(variety, LatinVariety::Ecclesiastical)
                && matches!(next, Some('c'))
                && is_ecclesiastical_front_context(&chars, index + 1) =>
            {
                ipa.push('ʃ');
            }
            's' => ipa.push('s'),
            't' if matches!(next, Some('h')) => {
                match variety {
                    LatinVariety::Classical => ipa.push_str("tʰ"),
                    LatinVariety::Ecclesiastical => ipa.push('t'),
                }
                index += 1;
            }
            't' if matches!(variety, LatinVariety::Ecclesiastical)
                && matches!(next, Some('i' | 'ī'))
                && after_next.is_some_and(is_vowel) =>
            {
                ipa.push_str("t͡s");
            }
            't' => ipa.push('t'),
            'v' => match variety {
                LatinVariety::Classical => ipa.push('w'),
                LatinVariety::Ecclesiastical => ipa.push('v'),
            },
            'x' => ipa.push_str("ks"),
            'z' => ipa.push('z'),
            '-' | '\'' | '’' => {}
            _ => return None,
        }
        index += 1;
    }
    let ipa = reposition_primary_stress(&ipa);
    (!ipa.is_empty()).then_some(format!("/{ipa}/"))
}

fn phoneme_inventory(variety: LatinVariety) -> PhonemeInventory {
    let phonemes = segments(variety)
        .iter()
        .map(|symbol| {
            let phoneme = Phoneme {
                id: PhonemeId(format!("{}.phoneme.{symbol}", variety.id())),
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
        .collect();
    PhonemeInventory { phonemes }
}

fn phone_inventory(variety: LatinVariety) -> PhoneInventory {
    let phones = segments(variety)
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
        .collect();
    PhoneInventory { phones }
}

fn segments(variety: LatinVariety) -> &'static [&'static str] {
    match variety {
        LatinVariety::Classical => CLASSICAL_SEGMENTS,
        LatinVariety::Ecclesiastical => ECCLESIASTICAL_SEGMENTS,
    }
}

fn segment_features(symbol: &str) -> FeatureBundle {
    let mut features = FeatureBundle::default();
    let is_vowel = matches!(
        symbol,
        "a" | "e" | "i" | "o" | "u" | "y" | "ae̯" | "au̯" | "oe̯" | "ae" | "au" | "oe"
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

fn normalize_latin_word(word: &str) -> Option<Vec<char>> {
    let normalized = word
        .trim()
        .to_lowercase()
        .replace('æ', "ae")
        .replace('œ', "oe");
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

fn stress_vowel_index(chars: &[char]) -> Option<usize> {
    let vowels = vowel_positions(chars);
    if vowels.is_empty() {
        return None;
    }
    if vowels.len() <= 2 {
        return Some(0);
    }
    let penult = vowels[vowels.len() - 2];
    if is_heavy_syllable(chars, penult) {
        Some(vowels.len() - 2)
    } else {
        Some(vowels.len() - 3)
    }
}

fn vowel_positions(chars: &[char]) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut index = 0usize;
    while index < chars.len() {
        if starts_diphthong(chars, index, LatinVariety::Classical).is_some() {
            positions.push(index);
            index += 2;
        } else {
            if is_vowel(chars[index]) {
                positions.push(index);
            }
            index += 1;
        }
    }
    positions
}

fn is_heavy_syllable(chars: &[char], vowel_index: usize) -> bool {
    if starts_diphthong(chars, vowel_index, LatinVariety::Classical).is_some() {
        return true;
    }
    matches!(
        chars.get(vowel_index),
        Some('ā' | 'ē' | 'ī' | 'ō' | 'ū' | 'ȳ')
    ) || coda_weight_after_vowel(chars, vowel_index) >= 2
}

fn coda_weight_after_vowel(chars: &[char], vowel_index: usize) -> usize {
    let mut weight = 0usize;
    let mut index = vowel_index + 1;
    while let Some(ch) = chars.get(index).copied() {
        if is_vowel(ch) {
            break;
        }
        if matches!(ch, '-' | '\'' | '’') {
            index += 1;
            continue;
        }
        if matches!(ch, 'x' | 'z') {
            weight += 2;
        } else if matches!(ch, 'q') && matches!(chars.get(index + 1), Some('u' | 'ū')) {
            weight += 2;
            index += 1;
        } else {
            weight += 1;
        }
        index += 1;
    }
    weight
}

fn starts_diphthong(
    chars: &[char],
    index: usize,
    variety: LatinVariety,
) -> Option<(&'static str, usize)> {
    let pair = (chars.get(index).copied()?, chars.get(index + 1).copied()?);
    match (pair, variety) {
        (('a' | 'ā', 'e' | 'ē'), LatinVariety::Classical) => Some(("ae̯", 2)),
        (('a' | 'ā', 'u' | 'ū'), LatinVariety::Classical) => Some(("au̯", 2)),
        (('o' | 'ō', 'e' | 'ē'), LatinVariety::Classical) => Some(("oe̯", 2)),
        (('a' | 'ā', 'e' | 'ē'), LatinVariety::Ecclesiastical) => Some(("ae", 2)),
        (('a' | 'ā', 'u' | 'ū'), LatinVariety::Ecclesiastical) => Some(("au", 2)),
        (('o' | 'ō', 'e' | 'ē'), LatinVariety::Ecclesiastical) => Some(("oe", 2)),
        _ => None,
    }
}

fn is_ecclesiastical_front_context(chars: &[char], index: usize) -> bool {
    matches!(
        chars.get(index + 1),
        Some('e' | 'ē' | 'i' | 'ī' | 'y' | 'ȳ')
    ) || matches!(
        (chars.get(index + 1), chars.get(index + 2)),
        (Some('a' | 'ā'), Some('e' | 'ē')) | (Some('o' | 'ō'), Some('e' | 'ē'))
    )
}

fn is_consonantal_i(chars: &[char], index: usize) -> bool {
    chars.get(index).copied() == Some('i')
        && (index == 0
            || chars
                .get(index.wrapping_sub(1))
                .is_some_and(|ch| is_vowel(*ch)))
        && chars.get(index + 1).is_some_and(|ch| is_vowel(*ch))
}

fn is_consonantal_u(chars: &[char], index: usize) -> bool {
    matches!(chars.get(index).copied(), Some('u' | 'ū'))
        && index > 0
        && matches!(chars.get(index - 1), Some('q' | 'g' | 's'))
        && chars.get(index + 1).is_some_and(|ch| is_vowel(*ch))
}

fn is_vowel(ch: char) -> bool {
    matches!(
        ch,
        'a' | 'ā' | 'e' | 'ē' | 'i' | 'ī' | 'o' | 'ō' | 'u' | 'ū' | 'y' | 'ȳ'
    )
}

fn is_ipa_vowel(ch: char) -> bool {
    matches!(ch, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
}

fn reposition_primary_stress(ipa: &str) -> String {
    let mut chars = ipa.chars().collect::<Vec<_>>();
    let Some(stress_index) = chars.iter().position(|ch| *ch == 'ˈ') else {
        return ipa.to_string();
    };
    let mut insert_index = stress_index;
    while insert_index > 0
        && !is_ipa_vowel(chars[insert_index - 1])
        && chars[insert_index - 1] != '̯'
    {
        insert_index -= 1;
    }
    if insert_index == stress_index {
        return ipa.to_string();
    }
    chars.remove(stress_index);
    chars.insert(insert_index, 'ˈ');
    chars.into_iter().collect()
}

fn cluster_constraint(variety: LatinVariety, cluster: &[&str]) -> PhonotacticConstraint {
    let label = cluster.join("");
    PhonotacticConstraint {
        id: format!("{}.legal_onset.{label}", variety.id()),
        description: format!("Legal Latin onset cluster {label}"),
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
    fn latin_varieties_load() {
        assert_eq!(variety("la").id.0, "la-Classical");
        assert_eq!(variety("la-Ecclesiastical").id.0, "la-Ecclesiastical");
    }

    #[test]
    fn latin_varieties_distinguish_c_before_front_vowels() {
        assert_eq!(
            synthesize_ipa("caelum", LatinVariety::Classical).as_deref(),
            Some("/ˈkae̯lum/")
        );
        assert_eq!(
            synthesize_ipa("caelum", LatinVariety::Ecclesiastical).as_deref(),
            Some("/ˈt͡ʃaelum/")
        );
    }

    #[test]
    fn latin_handles_stress_weight_and_greek_aspirates() {
        assert_eq!(
            synthesize_ipa("dominus", LatinVariety::Classical).as_deref(),
            Some("/ˈdominus/")
        );
        assert_eq!(
            synthesize_ipa("scriptūra", LatinVariety::Classical).as_deref(),
            Some("/skriˈptura/")
        );
        assert_eq!(
            synthesize_ipa("theologia", LatinVariety::Classical).as_deref(),
            Some("/tʰeoˈloɡia/")
        );
        assert_eq!(
            synthesize_ipa("theologia", LatinVariety::Ecclesiastical).as_deref(),
            Some("/teoˈlod͡ʒia/")
        );
    }
}
