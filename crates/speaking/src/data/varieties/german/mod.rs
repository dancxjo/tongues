use std::collections::HashMap;

use crate::feature::{FeatureBundle, FeatureSystem, FeatureValue};
use crate::ids::{LanguageId, PhoneId, PhonemeId, VarietyId};
use crate::orthography::Orthography;
use crate::phonetics::{Phone, PhoneInventory};
use crate::phonology::{Phoneme, PhonemeInventory};
use crate::segment::{SegmentStatus, SymbolAlias};
use crate::spec::Spec;
use crate::syntax::LinkGrammarRuleSet;
use crate::variety::{
    LinguisticVariety, NumberNameSet, OrthographyPronunciationRules, VarietyImplementationStatus,
    VarietyStatus,
};

pub const REGISTRATIONS: &[crate::data::varieties::VarietyRegistration] =
    &[crate::data::varieties::VarietyRegistration {
        canonical_id: "de-DE-Standard",
        aliases: &["de", "deu", "de-DE"],
        load: |_| variety(),
    }];

const SEGMENTS: &[&str] = &[
    "a", "aː", "e", "eː", "ɛ", "ɛː", "i", "iː", "o", "oː", "u", "uː", "y", "yː", "ø", "øː", "œ",
    "ə", "ɐ", "aɪ̯", "aʊ̯", "ɔʏ̯", "b", "ç", "d", "f", "ɡ", "h", "j", "k", "l", "m", "n", "ŋ", "p",
    "r", "s", "ʃ", "t", "t͡s", "t͡ʃ", "v", "x", "z",
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
        pronunciation_lexicons: Vec::new(),
        pronunciation_selection_rules: Vec::new(),
        pronunciation_pipeline: Some(
            crate::data::varieties::PRONUNCIATION_PIPELINE_VARIETY_DATA.into(),
        ),
        text_normalization: crate::data::varieties::small_number_text_normalization_profile(),
        syntax_profile: Some(crate::data::varieties::SYNTAX_PROFILE_GERMAN.into()),
        syntax_analyzer: None,
        syntax_rules: Some(syntax_profile()),
        orthography_pronunciation: Some(OrthographyPronunciationRules {
            synthesize_ipa: Some(synthesize_ipa_for_orthography),
        }),
        number_names: Some(NumberNameSet {
            cardinal_0_to_20: [
                "null",
                "eins",
                "zwei",
                "drei",
                "vier",
                "fünf",
                "sechs",
                "sieben",
                "acht",
                "neun",
                "zehn",
                "elf",
                "zwölf",
                "dreizehn",
                "vierzehn",
                "fünfzehn",
                "sechzehn",
                "siebzehn",
                "achtzehn",
                "neunzehn",
                "zwanzig",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            ordinal_suffixes: Vec::new(),
            ..Default::default()
        }),
        punctuation: Some(crate::data::varieties::german_punctuation_profile()),
        question_contours: Some(crate::data::varieties::german_question_contour_profile()),
        connected_speech: Vec::new(),
        phonotactics: None,
        orthography: Some(Orthography {
            name: "German Latin orthography".into(),
            pronunciation: Some(crate::data::varieties::ORTHOGRAPHY_PROFILE_GERMAN.into()),
            initialism_joiners: vec!["und".into()],
            sample_words: vec!["Sprache".into()],
            sample_letter_units: vec!["A".into(), "B".into()],
            ..Default::default()
        }),
        morphology: None,
        acoustic_profile: None,
        prosody_profile: Some(crate::data::varieties::prosody_profile(
            crate::data::varieties::PROSODY_RHYTHM_STRESS_TIMED,
            4.6,
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

pub fn syntax_profile() -> LinkGrammarRuleSet {
    LinkGrammarRuleSet {
        determiners: &[
            "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer",
            "eines", "mein", "meine", "dein", "deine", "sein", "seine", "ihr", "ihre", "unser",
            "unsere", "dieser", "diese", "dieses",
        ],
        pronouns: &[
            "ich", "du", "er", "sie", "es", "wir", "ihr", "mich", "dich", "sich", "uns", "euch",
            "mir", "dir", "ihm", "ihnen", "wer", "was", "die",
        ],
        object_pronouns: &[
            "mich", "dich", "sich", "uns", "euch", "ihn", "mir", "dir", "ihm", "ihnen",
        ],
        auxiliaries: &[
            "bin", "bist", "ist", "sind", "seid", "war", "waren", "habe", "hast", "hat", "haben",
            "habt", "hatte", "hatten", "werde", "wirst", "wird", "werden", "wollen", "können",
            "müssen", "sollen", "dürfen", "mögen",
        ],
        copulas: &["bin", "bist", "ist", "sind", "seid", "war", "waren"],
        prepositions: &[
            "an", "auf", "aus", "bei", "durch", "für", "gegen", "in", "mit", "nach", "ohne",
            "seit", "über", "um", "unter", "von", "vor", "zu", "zwischen",
        ],
        postpositions: &[],
        conjunctions: &["und", "oder", "aber", "denn", "sondern"],
        particles: &["ja", "doch", "mal", "wohl"],
        enclitic_suffixes: &[],
        complementizers: &[
            "dass", "daß", "der", "die", "das", "ob", "wenn", "weil", "als",
        ],
        adverbs: &["nicht", "sehr", "auch", "gern", "gerne"],
        adverb_suffixes: &[],
        adjectives: &[],
        adjective_suffixes: &["ig", "lich", "isch"],
        verbs: &[
            "sein", "haben", "werden", "machen", "sagen", "gehen", "kommen", "sehen", "wissen",
            "geben", "nehmen", "sprechen", "lernen", "arbeiten", "lesen", "liest", "lese",
            "denken", "denkt", "weiß", "weiss", "kommt",
        ],
        verb_suffixes: &["en"],
        subject_verb_suffixes: &["e", "st", "t"],
        non_verbs: &[],
        object_suffixes: &["en", "em"],
        infinitival_markers: &["zu"],
        allow_noun_compounds: true,
        ..LinkGrammarRuleSet::empty()
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
        } else if rest.starts_with("tsch") {
            ("t͡ʃ", 4)
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
        } else if (rest.starts_with("sp") || rest.starts_with("st"))
            && is_word_initial(&chars, index)
        {
            ("ʃ", 1)
        } else if rest.starts_with("sp") || rest.starts_with("st") {
            ("s", 1)
        } else if rest.starts_with("pf") {
            ("pf", 2)
        } else if rest.starts_with("ck") {
            ("k", 2)
        } else if rest.starts_with("ng") {
            ("ŋ", 2)
        } else if rest.starts_with("ig") && index + 2 == chars.len() {
            ("iç", 2)
        } else if chars[index] == 'h' && previous_is_vowel(&chars, index) {
            lengthen_previous_vowel(&mut ipa);
            ("", 1)
        } else {
            (single(chars[index])?, 1)
        };
        ipa.push_str(symbol);
        index += consumed;
    }
    let ipa = add_initial_stress(&ipa, &chars);
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

fn previous_is_vowel(chars: &[char], index: usize) -> bool {
    index > 0
        && matches!(
            chars[index - 1],
            'a' | 'e' | 'i' | 'o' | 'u' | 'ä' | 'ö' | 'ü'
        )
}

fn lengthen_previous_vowel(ipa: &mut String) {
    for (short, long) in [
        ("a", "aː"),
        ("e", "eː"),
        ("ɛ", "ɛː"),
        ("i", "iː"),
        ("o", "oː"),
        ("u", "uː"),
        ("y", "yː"),
        ("ø", "øː"),
    ] {
        if ipa.ends_with(short) {
            let new_len = ipa.len() - short.len();
            ipa.truncate(new_len);
            ipa.push_str(long);
            return;
        }
    }
}

fn is_word_initial(chars: &[char], index: usize) -> bool {
    index == 0
        || chars
            .get(index.wrapping_sub(1))
            .is_some_and(|ch| matches!(ch, '-' | '\'' | '’'))
}

fn add_initial_stress(ipa: &str, word: &[char]) -> String {
    let mut chars = ipa.chars().collect::<Vec<_>>();
    let search_start = stress_search_start(ipa, word);
    let Some(mut insert) = chars
        .iter()
        .enumerate()
        .skip(search_start)
        .find_map(|(index, ch)| is_ipa_vowel(*ch).then_some(index))
    else {
        return ipa.to_string();
    };
    while insert > 0 && !is_ipa_vowel(chars[insert - 1]) {
        insert -= 1;
    }
    chars.insert(insert, 'ˈ');
    chars.into_iter().collect()
}

fn stress_search_start(ipa: &str, word: &[char]) -> usize {
    if starts_with_unstressed_ge_prefix(word) && ipa.starts_with("ɡə") {
        return 2;
    }
    0
}

fn starts_with_unstressed_ge_prefix(word: &[char]) -> bool {
    let word = word.iter().collect::<String>();
    word.starts_with("ge")
        && !matches!(
            word.as_str(),
            "geben"
                | "gebe"
                | "gebt"
                | "gegen"
                | "geh"
                | "gehe"
                | "gehen"
                | "gehst"
                | "geht"
                | "gelb"
                | "gelbe"
                | "gelben"
                | "gelber"
                | "geld"
                | "gern"
                | "gerne"
                | "gestern"
        )
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
            | "eː"
            | "ɛ"
            | "ɛː"
            | "i"
            | "iː"
            | "o"
            | "oː"
            | "u"
            | "uː"
            | "y"
            | "yː"
            | "ø"
            | "øː"
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

    #[test]
    fn german_handles_common_clusters_and_final_ig() {
        assert_eq!(synthesize_ipa("König").as_deref(), Some("/ˈkøniç/"));
        assert_eq!(synthesize_ipa("Ding").as_deref(), Some("/ˈdiŋ/"));
        assert_eq!(synthesize_ipa("backen").as_deref(), Some("/ˈbakən/"));
        assert_eq!(synthesize_ipa("Wespe").as_deref(), Some("/ˈvəspə/"));
    }

    #[test]
    fn german_unstressed_ge_prefix_does_not_take_primary_stress() {
        assert_eq!(synthesize_ipa("geneigt").as_deref(), Some("/ɡəˈnaɪ̯ɡt/"));
        assert_eq!(synthesize_ipa("gezeigt").as_deref(), Some("/ɡəˈt͡saɪ̯ɡt/"));
        assert_eq!(synthesize_ipa("Gestalten").as_deref(), Some("/ɡəˈstaltən/"));
        assert_eq!(synthesize_ipa("geben").as_deref(), Some("/ˈɡəbən/"));
        assert_eq!(synthesize_ipa("gestern").as_deref(), Some("/ˈɡəstərn/"));
    }

    #[test]
    fn german_silences_dehnungs_h_after_vowels() {
        assert_eq!(synthesize_ipa("Ihr").as_deref(), Some("/ˈiːr/"));
        assert_eq!(synthesize_ipa("naht").as_deref(), Some("/ˈnaːt/"));
        assert_eq!(synthesize_ipa("wohl").as_deref(), Some("/ˈvoːl/"));
        assert_eq!(synthesize_ipa("früh").as_deref(), Some("/ˈfryː/"));
        assert_eq!(synthesize_ipa("Fühl").as_deref(), Some("/ˈfyːl/"));
        assert_eq!(synthesize_ipa("Wahn").as_deref(), Some("/ˈvaːn/"));
        assert_eq!(synthesize_ipa("Hauch").as_deref(), Some("/ˈhaʊ̯x/"));
        assert_eq!(
            synthesize_ipa("Zauberhauch").as_deref(),
            Some("/ˈt͡saʊ̯bərhaʊ̯x/")
        );
    }
}
