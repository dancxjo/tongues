pub mod english;
pub mod esperanto;
pub mod french;
pub mod german;
pub mod greek;
pub mod latin;
pub mod sanskrit;
pub mod spanish;

use crate::ids::VarietyId;
use crate::prosody::ProsodyProfile;
use crate::variety::{LinguisticVariety, PunctuationProfile, QuestionContourProfile};

pub const DEFAULT_SPEAKING_VARIETY: &str = "en-US";
pub const PRONUNCIATION_PIPELINE_ENGLISH_CMUDICT: &str = "english_cmudict";
pub const PRONUNCIATION_PIPELINE_VARIETY_DATA: &str = "variety_data";

pub const SYNTAX_PROFILE_ENGLISH: &str = "english";
pub const SYNTAX_PROFILE_ESPERANTO: &str = "esperanto";
pub const SYNTAX_PROFILE_FRENCH: &str = "french";
pub const SYNTAX_PROFILE_GERMAN: &str = "german";
pub const SYNTAX_PROFILE_GREEK: &str = "greek";
pub const SYNTAX_PROFILE_LATIN: &str = "latin";
pub const SYNTAX_PROFILE_SANSKRIT: &str = "sanskrit";
pub const SYNTAX_PROFILE_SPANISH: &str = "spanish";

pub const ORTHOGRAPHY_PROFILE_ALIAS: &str = "alias";
pub const ORTHOGRAPHY_PROFILE_ENGLISH_CMUDICT: &str = "english_cmudict";
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

pub fn default_punctuation_profile() -> PunctuationProfile {
    PunctuationProfile {
        period_abbreviations: [
            "mr", "mrs", "ms", "mme", "mlle", "m", "dr", "prof", "sen", "rep", "gen", "col",
            "capt", "sgt", "lieut", "corp", "rev", "fr", "br", "st", "ave", "av", "rd", "blvd",
            "ln", "ct", "pl", "co", "inc", "ltd", "etc", "vs", "approx", "jan", "feb", "mar",
            "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "jr", "sr", "srta",
            "sra", "sr", "dra", "dott", "sig", "sig.ra", "hr", "frl", "fr", "u", "bzw", "z.b",
            "bsp", "vgl", "ca",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        title_abbreviations: [
            "mr", "mrs", "ms", "mme", "mlle", "m", "dr", "prof", "sen", "rep", "gen", "col",
            "capt", "sgt", "lieut", "corp", "rev", "fr", "br", "sr", "sra", "srta", "dra", "dott",
            "sig", "sig.ra", "hr", "frl",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        ambiguous_period_abbreviations: ["st", "fr", "sr"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        sentence_starter_words_after_ambiguous_abbreviation: [
            "the", "he", "she", "it", "they", "we", "i", "you", "this", "that", "these", "those",
            "there", "here", "but", "and", "then", "so", "if", "when", "as", "what", "who", "how",
            "why", "my", "your", "our", "their", "his", "her", "its", "le", "la", "les", "un",
            "une", "des", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "el", "la",
            "los", "las", "yo", "nosotros", "ellos", "der", "die", "das", "ich", "du", "wir",
            "sie", "ein", "eine",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    }
}

pub fn default_question_contour_profile() -> QuestionContourProfile {
    QuestionContourProfile {
        yes_no_openers: [
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
            "est",
            "es",
            "sont",
            "êtes",
            "avez",
            "as",
            "a",
            "va",
            "vas",
            "vamos",
            "es",
            "eres",
            "son",
            "ist",
            "sind",
            "bist",
            "hat",
            "haben",
            "kann",
            "können",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        wh_openers: [
            "what", "when", "where", "why", "who", "whom", "whose", "which", "how", "que", "quoi",
            "quand", "où", "ou", "pourquoi", "qui", "comment", "qué", "que", "cuándo", "cuando",
            "dónde", "donde", "por", "quién", "quien", "cómo", "como", "was", "wann", "wo",
            "warum", "wer", "wie",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        alternative_coordinators: ["or", "ou", "o", "oder", "aŭ", "aut", "ἤ", "वा"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        paired_alternative_openers: ["either", "soit", "sea", "entweder", "aŭ"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    }
}

struct VarietyRegistration {
    canonical_id: &'static str,
    aliases: &'static [&'static str],
    load: fn(&str) -> LinguisticVariety,
}

const BUILTIN_VARIETY_REGISTRY: &[VarietyRegistration] = &[
    VarietyRegistration {
        canonical_id: "en-US-GA",
        aliases: &["en-US"],
        load: english_variety,
    },
    VarietyRegistration {
        canonical_id: "en-US-singing",
        aliases: &[],
        load: english_variety,
    },
    VarietyRegistration {
        canonical_id: "en-GB-RP",
        aliases: &[],
        load: english_variety,
    },
    VarietyRegistration {
        canonical_id: "en-GB-ScotE",
        aliases: &[],
        load: english_variety,
    },
    VarietyRegistration {
        canonical_id: "en-US-AAE",
        aliases: &[],
        load: english_variety,
    },
    VarietyRegistration {
        canonical_id: "eo",
        aliases: &[],
        load: esperanto_variety,
    },
    VarietyRegistration {
        canonical_id: "fr-FR-Standard",
        aliases: &["fr", "fra", "fr-FR"],
        load: french_variety,
    },
    VarietyRegistration {
        canonical_id: "de-DE-Standard",
        aliases: &["de", "deu", "de-DE"],
        load: german_variety,
    },
    VarietyRegistration {
        canonical_id: "el-GR-Standard",
        aliases: &["el", "el-GR"],
        load: greek_variety,
    },
    VarietyRegistration {
        canonical_id: "grc-Attic",
        aliases: &["grc", "grc-Ancient"],
        load: greek_variety,
    },
    VarietyRegistration {
        canonical_id: "grc-Koine",
        aliases: &["el-Koine"],
        load: greek_variety,
    },
    VarietyRegistration {
        canonical_id: "la-Classical",
        aliases: &["la"],
        load: latin_variety,
    },
    VarietyRegistration {
        canonical_id: "la-Ecclesiastical",
        aliases: &["la-Church"],
        load: latin_variety,
    },
    VarietyRegistration {
        canonical_id: "sa-Deva-Standard",
        aliases: &["sa", "san", "sa-Deva"],
        load: sanskrit_variety,
    },
    VarietyRegistration {
        canonical_id: "es-ES-Castilian",
        aliases: &["es", "es-ES"],
        load: spanish_variety,
    },
    VarietyRegistration {
        canonical_id: "es-419-Standard",
        aliases: &["es-419", "es-LatAm"],
        load: spanish_variety,
    },
];

pub fn canonical_variety_id(code: &str) -> Option<VarietyId> {
    find_variety_registration(code).map(|registration| VarietyId(registration.canonical_id.into()))
}

pub fn variety_by_code(code: &str) -> Option<LinguisticVariety> {
    let canonical = canonical_variety_id(code)?;
    let registration = find_variety_registration(&canonical.0)?;
    Some((registration.load)(registration.canonical_id))
}

pub fn builtin_varieties() -> Vec<LinguisticVariety> {
    BUILTIN_VARIETY_REGISTRY
        .iter()
        .map(|registration| (registration.load)(registration.canonical_id))
        .collect()
}

fn find_variety_registration(code: &str) -> Option<&'static VarietyRegistration> {
    BUILTIN_VARIETY_REGISTRY.iter().find(|registration| {
        registration.canonical_id == code || registration.aliases.contains(&code)
    })
}

fn english_variety(id: &str) -> LinguisticVariety {
    english::variety(id)
}

fn esperanto_variety(_id: &str) -> LinguisticVariety {
    esperanto::variety()
}

fn french_variety(_id: &str) -> LinguisticVariety {
    french::variety()
}

fn german_variety(_id: &str) -> LinguisticVariety {
    german::variety()
}

fn greek_variety(id: &str) -> LinguisticVariety {
    greek::variety(id)
}

fn latin_variety(id: &str) -> LinguisticVariety {
    latin::variety(id)
}

fn sanskrit_variety(_id: &str) -> LinguisticVariety {
    sanskrit::variety()
}

fn spanish_variety(id: &str) -> LinguisticVariety {
    spanish::variety(id)
}
