pub mod chukchi;
pub mod english;
pub mod esperanto;
pub mod french;
pub mod german;
pub mod greek;
pub mod latin;
pub mod sanskrit;
pub mod spanish;

use std::collections::BTreeSet;

use crate::ids::VarietyId;
use crate::prosody::ProsodyProfile;
use crate::variety::{
    Language, LinguisticVariety, NumberName, NumberNormalizationProfile, PunctuationProfile,
    QuestionContourProfile, ScaleName, TextNormalizationProfile, TextRewrite, UnitName,
};
use isolang::Language as IsoLanguage;
use langtag::LangTag;

pub const DEFAULT_SPEAKING_VARIETY: &str = "en-US";
pub const PRONUNCIATION_PIPELINE_VARIETY_DATA: &str = "variety_data";

pub const SYNTAX_PROFILE_ENGLISH: &str = "english";
pub const SYNTAX_PROFILE_CHUKCHI: &str = "chukchi";
pub const SYNTAX_PROFILE_ESPERANTO: &str = "esperanto";
pub const SYNTAX_PROFILE_FRENCH: &str = "french";
pub const SYNTAX_PROFILE_GERMAN: &str = "german";
pub const SYNTAX_PROFILE_GREEK: &str = "greek";
pub const SYNTAX_PROFILE_LATIN: &str = "latin";
pub const SYNTAX_PROFILE_SANSKRIT: &str = "sanskrit";
pub const SYNTAX_PROFILE_SPANISH: &str = "spanish";

pub const ORTHOGRAPHY_PROFILE_ALIAS: &str = "alias";
pub const ORTHOGRAPHY_PROFILE_CHUKCHI: &str = "chukchi_cyrillic";
pub const ORTHOGRAPHY_PROFILE_ESPERANTO: &str = "esperanto";
pub const ORTHOGRAPHY_PROFILE_FRENCH: &str = "french";
pub const ORTHOGRAPHY_PROFILE_GERMAN: &str = "german";
pub const ORTHOGRAPHY_PROFILE_GREEK: &str = "greek";
pub const ORTHOGRAPHY_PROFILE_LATIN: &str = "latin";
pub const ORTHOGRAPHY_PROFILE_SANSKRIT: &str = "sanskrit";
pub const ORTHOGRAPHY_PROFILE_SPANISH: &str = "spanish";

pub const PROSODY_RHYTHM_MORA_TIMED: &str = "mora_timed";
pub const PROSODY_RHYTHM_STRESS_TIMED: &str = "stress_timed";
pub const PROSODY_RHYTHM_SYLLABLE_TIMED: &str = "syllable_timed";

pub fn prosody_profile(
    rhythm_class: &str,
    default_rate_syllables_per_second: f32,
) -> ProsodyProfile {
    ProsodyProfile {
        default_pitch_hz: None,
        default_rate_syllables_per_second: Some(default_rate_syllables_per_second),
        rhythm_class: Some(rhythm_class.into()),
    }
}

pub fn small_number_text_normalization_profile() -> TextNormalizationProfile {
    TextNormalizationProfile {
        spoken_form_rewrites: Vec::new(),
        number_normalization: NumberNormalizationProfile::SmallNumbers,
    }
}

pub fn english_text_normalization_profile() -> TextNormalizationProfile {
    let mut spoken_form_rewrites = english::normalization::SPOKEN_FORM_REWRITES
        .iter()
        .map(|rewrite| TextRewrite {
            from: rewrite.from.into(),
            to: rewrite.to.into(),
        })
        .collect::<Vec<_>>();
    spoken_form_rewrites.push(TextRewrite {
        from: "No.".into(),
        to: "Number".into(),
    });
    TextNormalizationProfile {
        spoken_form_rewrites,
        number_normalization: NumberNormalizationProfile::General,
    }
}

pub fn number_names(
    cardinal_0_to_20: &[&str],
    tens: &[(u32, &str)],
    scales: &[(u32, &str)],
    units: &[(&[&str], &str, &str)],
) -> crate::variety::NumberNameSet {
    crate::variety::NumberNameSet {
        cardinal_0_to_20: strings(cardinal_0_to_20),
        cardinal_tens: tens
            .iter()
            .map(|(value, name)| NumberName {
                value: *value,
                name: (*name).into(),
            })
            .collect(),
        scale_names: scales
            .iter()
            .map(|(power, name)| ScaleName {
                power: *power,
                name: (*name).into(),
            })
            .collect(),
        unit_names: units
            .iter()
            .map(|(aliases, singular, plural)| UnitName {
                aliases: strings(aliases),
                singular: (*singular).into(),
                plural: (*plural).into(),
            })
            .collect(),
        ..Default::default()
    }
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

pub fn english_punctuation_profile() -> PunctuationProfile {
    PunctuationProfile {
        period_abbreviations: strings(&[
            "mr", "mrs", "ms", "mme", "mlle", "m", "dr", "prof", "sen", "rep", "gen", "col",
            "capt", "sgt", "lieut", "corp", "rev", "fr", "br", "st", "ave", "av", "rd", "blvd",
            "ln", "ct", "pl", "co", "inc", "ltd", "etc", "vs", "approx", "jan", "feb", "mar",
            "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "jr", "sr",
        ]),
        title_abbreviations: strings(&[
            "mr", "mrs", "ms", "dr", "prof", "sen", "rep", "gen", "col", "capt", "sgt", "lieut",
            "corp", "rev", "fr", "br", "sr",
        ]),
        ambiguous_period_abbreviations: strings(&["st", "fr", "sr"]),
        sentence_starter_words_after_ambiguous_abbreviation: strings(&[
            "the", "he", "she", "it", "they", "we", "i", "you", "this", "that", "these", "those",
            "there", "here", "but", "and", "then", "so", "if", "when", "as", "what", "who", "how",
            "why", "my", "your", "our", "their", "his", "her", "its",
        ]),
    }
}

pub fn french_punctuation_profile() -> PunctuationProfile {
    PunctuationProfile {
        period_abbreviations: strings(&[
            "m", "mme", "mlle", "dr", "dre", "prof", "st", "ste", "av", "bd", "etc", "janv",
            "févr", "fevr", "avr", "sept", "oct", "nov", "déc", "dec",
        ]),
        title_abbreviations: strings(&["m", "mme", "mlle", "dr", "dre", "prof"]),
        ambiguous_period_abbreviations: strings(&["st", "ste"]),
        sentence_starter_words_after_ambiguous_abbreviation: strings(&[
            "le", "la", "les", "un", "une", "des", "je", "tu", "il", "elle", "nous", "vous", "ils",
            "elles", "ce", "cette", "ces", "mais", "et", "donc", "que", "qui", "quand", "pourquoi",
            "comment",
        ]),
    }
}

pub fn spanish_punctuation_profile() -> PunctuationProfile {
    PunctuationProfile {
        period_abbreviations: strings(&[
            "sr", "sra", "srta", "dr", "dra", "prof", "av", "etc", "aprox", "ene", "feb", "mar",
            "abr", "jun", "jul", "ago", "sept", "oct", "nov", "dic",
        ]),
        title_abbreviations: strings(&["sr", "sra", "srta", "dr", "dra", "prof"]),
        ambiguous_period_abbreviations: strings(&["sr"]),
        sentence_starter_words_after_ambiguous_abbreviation: strings(&[
            "el", "la", "los", "las", "un", "una", "unos", "unas", "yo", "tú", "tu", "él", "el",
            "ella", "nosotros", "ellos", "ellas", "este", "esta", "estos", "estas", "pero", "y",
            "entonces", "que", "qué", "cuando", "cuándo", "por", "quién", "quien", "cómo", "como",
        ]),
    }
}

pub fn german_punctuation_profile() -> PunctuationProfile {
    PunctuationProfile {
        period_abbreviations: strings(&[
            "hr", "fr", "dr", "prof", "bzw", "z.b", "bsp", "vgl", "ca", "u", "b.a", "d.h", "usw",
            "jan", "feb", "mär", "maerz", "apr", "jun", "jul", "aug", "sep", "okt", "nov", "dez",
        ]),
        title_abbreviations: strings(&["hr", "fr", "dr", "prof"]),
        ambiguous_period_abbreviations: strings(&["fr"]),
        sentence_starter_words_after_ambiguous_abbreviation: strings(&[
            "der", "die", "das", "ein", "eine", "ich", "du", "er", "sie", "es", "wir", "ihr",
            "aber", "und", "dann", "wenn", "was", "wer", "wie", "warum", "wo",
        ]),
    }
}

pub fn esperanto_punctuation_profile() -> PunctuationProfile {
    PunctuationProfile {
        period_abbreviations: strings(&["d-ro", "s-ro", "s-ino", "prof", "ktp", "ekz", "t.e"]),
        title_abbreviations: strings(&["d-ro", "s-ro", "s-ino", "prof"]),
        ambiguous_period_abbreviations: Vec::new(),
        sentence_starter_words_after_ambiguous_abbreviation: strings(&[
            "la", "mi", "vi", "li", "ŝi", "sxi", "ĝi", "gxi", "ni", "ili", "ĉu", "cxu", "kiu",
            "kio", "kie", "kiam", "kial", "kiel", "sed", "kaj",
        ]),
    }
}

pub fn latin_punctuation_profile() -> PunctuationProfile {
    PunctuationProfile {
        period_abbreviations: strings(&["etc", "cf", "ca"]),
        title_abbreviations: Vec::new(),
        ambiguous_period_abbreviations: Vec::new(),
        sentence_starter_words_after_ambiguous_abbreviation: strings(&[
            "hic", "haec", "hoc", "ille", "ego", "tu", "nos", "vos", "et", "sed", "aut", "quis",
            "quid", "cur", "ubi", "quando",
        ]),
    }
}

pub fn greek_punctuation_profile() -> PunctuationProfile {
    PunctuationProfile {
        period_abbreviations: strings(&["κ", "π.χ", "δηλ"]),
        title_abbreviations: Vec::new(),
        ambiguous_period_abbreviations: Vec::new(),
        sentence_starter_words_after_ambiguous_abbreviation: strings(&[
            "ο",
            "η",
            "το",
            "οι",
            "τα",
            "εγώ",
            "εγω",
            "εσύ",
            "εσυ",
            "αυτός",
            "αυτος",
            "και",
            "αλλά",
            "αλλα",
            "τι",
            "πού",
            "που",
            "πότε",
            "ποτε",
            "γιατί",
            "γιατι",
            "πώς",
            "πως",
        ]),
    }
}

pub fn sanskrit_punctuation_profile() -> PunctuationProfile {
    PunctuationProfile {
        period_abbreviations: Vec::new(),
        title_abbreviations: Vec::new(),
        ambiguous_period_abbreviations: Vec::new(),
        sentence_starter_words_after_ambiguous_abbreviation: strings(&[
            "अहम्",
            "अहं",
            "त्वम्",
            "सः",
            "सा",
            "तत्",
            "वयम्",
            "ते",
            "कः",
            "का",
            "किम्",
            "च",
            "वा",
            "aham",
            "ahaṃ",
            "tvam",
            "saḥ",
            "sā",
            "tat",
            "vayam",
            "te",
            "kaḥ",
            "kā",
            "kim",
            "ca",
            "vā",
            "va",
        ]),
    }
}

pub fn english_question_contour_profile() -> QuestionContourProfile {
    QuestionContourProfile {
        yes_no_openers: strings(&[
            "am",
            "are",
            "aren't",
            "is",
            "isn't",
            "was",
            "wasn't",
            "were",
            "weren't",
            "do",
            "don't",
            "does",
            "doesn't",
            "did",
            "didn't",
            "have",
            "haven't",
            "has",
            "hasn't",
            "had",
            "hadn't",
            "can",
            "can't",
            "could",
            "couldn't",
            "will",
            "won't",
            "would",
            "wouldn't",
            "shall",
            "shan't",
            "should",
            "shouldn't",
            "may",
            "might",
            "must",
            "ought",
            "need",
            "dare",
        ]),
        wh_openers: strings(&[
            "what", "when", "where", "why", "who", "whom", "whose", "which", "how",
        ]),
        alternative_coordinators: strings(&["or"]),
        paired_alternative_openers: strings(&["either"]),
    }
}

pub fn french_question_contour_profile() -> QuestionContourProfile {
    QuestionContourProfile {
        yes_no_openers: strings(&["est", "es", "sont", "êtes", "avez", "as", "a", "va", "vas"]),
        wh_openers: strings(&[
            "que", "quoi", "quand", "où", "ou", "pourquoi", "qui", "comment",
        ]),
        alternative_coordinators: strings(&["ou"]),
        paired_alternative_openers: strings(&["soit"]),
    }
}

pub fn spanish_question_contour_profile() -> QuestionContourProfile {
    QuestionContourProfile {
        yes_no_openers: strings(&[
            "vamos", "es", "eres", "son", "está", "esta", "están", "estan",
        ]),
        wh_openers: strings(&[
            "qué", "que", "cuándo", "cuando", "dónde", "donde", "por", "quién", "quien", "cómo",
            "como", "cuál", "cual",
        ]),
        alternative_coordinators: strings(&["o"]),
        paired_alternative_openers: strings(&["sea"]),
    }
}

pub fn german_question_contour_profile() -> QuestionContourProfile {
    QuestionContourProfile {
        yes_no_openers: strings(&["ist", "sind", "bist", "hat", "haben", "kann", "können"]),
        wh_openers: strings(&[
            "was", "wann", "wo", "warum", "wer", "wie", "welche", "welcher",
        ]),
        alternative_coordinators: strings(&["oder"]),
        paired_alternative_openers: strings(&["entweder"]),
    }
}

pub fn esperanto_question_contour_profile() -> QuestionContourProfile {
    QuestionContourProfile {
        yes_no_openers: strings(&["ĉu", "cxu"]),
        wh_openers: strings(&["kio", "kiu", "kiam", "kie", "kial", "kiel", "kiom"]),
        alternative_coordinators: strings(&["aŭ", "aux"]),
        paired_alternative_openers: strings(&["aŭ", "aux"]),
    }
}

pub fn latin_question_contour_profile() -> QuestionContourProfile {
    QuestionContourProfile {
        yes_no_openers: strings(&["num", "nonne", "utrum"]),
        wh_openers: strings(&[
            "quis", "quid", "quando", "ubi", "cur", "quomodo", "quo", "unde",
        ]),
        alternative_coordinators: strings(&["aut", "vel"]),
        paired_alternative_openers: strings(&["utrum"]),
    }
}

pub fn greek_question_contour_profile() -> QuestionContourProfile {
    QuestionContourProfile {
        yes_no_openers: strings(&["είναι", "ειναι", "ἆρα", "άρα", "ara"]),
        wh_openers: strings(&[
            "τι",
            "πού",
            "που",
            "ποιος",
            "πότε",
            "ποτε",
            "γιατί",
            "γιατι",
            "πώς",
            "πως",
            "τίς",
            "τί",
        ]),
        alternative_coordinators: strings(&["ή", "η", "ἤ"]),
        paired_alternative_openers: strings(&["είτε", "ειτε"]),
    }
}

pub fn sanskrit_question_contour_profile() -> QuestionContourProfile {
    QuestionContourProfile {
        yes_no_openers: strings(&["किम्", "किं", "kim"]),
        wh_openers: strings(&[
            "कः",
            "का",
            "किम्",
            "किं",
            "कदा",
            "कुत्र",
            "कथम्",
            "कथं",
            "कस्मात्",
            "kaḥ",
            "kā",
            "kim",
            "kadā",
            "kutra",
            "katham",
            "kasmāt",
        ]),
        alternative_coordinators: strings(&["वा", "vā", "va"]),
        paired_alternative_openers: Vec::new(),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VarietyRegistration {
    pub canonical_id: &'static str,
    pub aliases: &'static [&'static str],
    pub language_tag: &'static str,
    pub load: fn(&str) -> LinguisticVariety,
}

pub fn canonical_variety_id(code: &str) -> Option<VarietyId> {
    if let Some(registration) = find_variety_registration(code) {
        return Some(VarietyId(registration.canonical_id.into()));
    }
    if code.contains('-') {
        return None;
    }
    let code = code.to_ascii_lowercase();
    let language = IsoLanguage::from_639_1(&code).or_else(|| IsoLanguage::from_639_3(&code))?;
    let language_id = language.to_639_1().unwrap_or_else(|| language.to_639_3());
    builtin_variety_registrations()
        .find(|registration| {
            (registration.load)(registration.canonical_id).language.0 == language_id
        })
        .map(|registration| VarietyId(registration.canonical_id.into()))
}

pub fn variety_by_code(code: &str) -> Option<LinguisticVariety> {
    let canonical = canonical_variety_id(code)?;
    let registration = find_variety_registration(&canonical.0)?;
    Some((registration.load)(registration.canonical_id))
}

pub fn language_tag_for_variety(code: &str) -> Option<&'static str> {
    let canonical = canonical_variety_id(code)?;
    let registration = find_variety_registration(&canonical.0)?;
    LangTag::new(registration.language_tag).ok()?;
    Some(registration.language_tag)
}

pub fn wiktionary_language_for_variety(code: &str) -> Option<&'static str> {
    let canonical = canonical_variety_id(code)?;
    let registration = find_variety_registration(&canonical.0)?;
    let language = (registration.load)(registration.canonical_id).language;
    let parsed =
        IsoLanguage::from_639_1(&language.0).or_else(|| IsoLanguage::from_639_3(&language.0))?;
    Some(parsed.to_639_3())
}

pub fn builtin_varieties() -> Vec<LinguisticVariety> {
    builtin_variety_registrations()
        .map(|registration| (registration.load)(registration.canonical_id))
        .collect()
}

pub fn builtin_languages() -> Vec<Language> {
    let mut seen = BTreeSet::new();
    builtin_variety_registrations()
        .filter_map(|registration| {
            let language = (registration.load)(registration.canonical_id).language;
            if !seen.insert(language.0.clone()) {
                return None;
            }
            let parsed = IsoLanguage::from_639_1(&language.0)
                .or_else(|| IsoLanguage::from_639_3(&language.0))?;
            Some(Language {
                id: language,
                name: parsed.to_name().into(),
                endonym: None,
                iso_639: Some(parsed.to_639_3().into()),
            })
        })
        .collect()
}

fn find_variety_registration(code: &str) -> Option<&'static VarietyRegistration> {
    builtin_variety_registrations().find(|registration| {
        registration.canonical_id == code
            || registration.aliases.iter().any(|alias| {
                *alias == code || LangTag::new(alias).is_ok_and(|candidate| candidate == code)
            })
    })
}

fn builtin_variety_registrations() -> impl Iterator<Item = &'static VarietyRegistration> {
    [
        chukchi::REGISTRATIONS,
        english::REGISTRATIONS,
        esperanto::REGISTRATIONS,
        french::REGISTRATIONS,
        german::REGISTRATIONS,
        greek::REGISTRATIONS,
        latin::REGISTRATIONS,
        sanskrit::REGISTRATIONS,
        spanish::REGISTRATIONS,
    ]
    .into_iter()
    .flatten()
}
