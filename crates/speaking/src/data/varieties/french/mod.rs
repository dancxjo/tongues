use std::collections::HashMap;

use crate::data::lexicons::LEXIQUE383_ID;
use crate::feature::{FeatureBundle, FeatureSystem, FeatureValue};
use crate::ids::{LanguageId, PhoneId, PhonemeId, VarietyId};
use crate::orthography::Orthography;
use crate::phonetics::{Phone, PhoneInventory};
use crate::phonology::{Phoneme, PhonemeInventory};
use crate::segment::{SegmentStatus, SymbolAlias};
use crate::spec::Spec;
use crate::syntax::HeuristicSyntaxProfile;
use crate::syntax::PartOfSpeech;
use crate::variety::{
    ConnectedSpeechEntry, ConnectedSpeechRule, LinguisticVariety, NumberNameSet, OrdinalSuffixName,
    OrthographyPronunciationRules, VarietyImplementationStatus, VarietyStatus,
};

pub const REGISTRATIONS: &[crate::data::varieties::VarietyRegistration] =
    &[crate::data::varieties::VarietyRegistration {
        canonical_id: "fr-FR-Standard",
        aliases: &["fr", "fra", "fr-FR"],
        load: |_| variety(),
    }];

const SEGMENTS: &[&str] = &[
    "a", "ɑ̃", "e", "ɛ", "ɛ̃", "i", "o", "ɔ", "ɔ̃", "u", "y", "ø", "œ", "œ̃", "ə", "b", "d", "f", "ɡ",
    "ʒ", "j", "k", "l", "m", "n", "ɲ", "p", "ʁ", "s", "ʃ", "t", "v", "w", "ɥ", "z",
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
        pronunciation_lexicons: vec![LEXIQUE383_ID.into()],
        pronunciation_selection_rules: Vec::new(),
        pronunciation_pipeline: Some(
            crate::data::varieties::PRONUNCIATION_PIPELINE_VARIETY_DATA.into(),
        ),
        text_normalization: crate::data::varieties::small_number_text_normalization_profile(),
        syntax_profile: Some(crate::data::varieties::SYNTAX_PROFILE_FRENCH.into()),
        syntax_analyzer: None,
        syntax_heuristics: Some(syntax_profile()),
        orthography_pronunciation: Some(OrthographyPronunciationRules {
            synthesize_ipa: Some(synthesize_ipa_for_orthography),
        }),
        number_names: Some(NumberNameSet {
            cardinal_0_to_20: [
                "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
                "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize", "dix-sept",
                "dix-huit", "dix-neuf", "vingt",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            ordinal_suffixes: vec![OrdinalSuffixName {
                value: 1,
                suffixes: vec!["er".into(), "re".into()],
                name: "premier".into(),
            }],
            ..Default::default()
        }),
        punctuation: Some(crate::data::varieties::french_punctuation_profile()),
        question_contours: Some(crate::data::varieties::french_question_contour_profile()),
        connected_speech: vec![
            ConnectedSpeechRule::DeleteFinalPhoneBeforeConsonant { phone: "ə".into() },
            ConnectedSpeechRule::Liaison {
                entries: [
                    ("les", "z"),
                    ("des", "z"),
                    ("mes", "z"),
                    ("tes", "z"),
                    ("ses", "z"),
                    ("nos", "z"),
                    ("vos", "z"),
                    ("nous", "z"),
                    ("vous", "z"),
                    ("deux", "z"),
                    ("trois", "z"),
                    ("un", "n"),
                    ("mon", "n"),
                    ("ton", "n"),
                    ("son", "n"),
                    ("en", "n"),
                    ("on", "n"),
                ]
                .into_iter()
                .map(|(after_word, before_vowel_phone)| ConnectedSpeechEntry {
                    after_word: after_word.into(),
                    before_vowel_phone: before_vowel_phone.into(),
                })
                .collect(),
            },
        ],
        phonotactics: None,
        orthography: Some(Orthography {
            name: "French Latin orthography".into(),
            pronunciation: Some(crate::data::varieties::ORTHOGRAPHY_PROFILE_FRENCH.into()),
            initialism_joiners: vec!["et".into()],
            sample_words: vec!["bonjour".into()],
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
    part_of_speech: Option<PartOfSpeech>,
) -> Option<String> {
    synthesize_ipa_with_pos(word, part_of_speech)
}

pub fn syntax_profile() -> HeuristicSyntaxProfile {
    HeuristicSyntaxProfile {
        determiners: &[
            "le", "la", "les", "l'", "un", "une", "des", "du", "de", "mon", "ma", "mes", "ton",
            "ta", "tes", "son", "sa", "ses", "notre", "nos", "votre", "vos", "leur", "leurs", "ce",
            "cet", "cette", "ces",
        ],
        pronouns: &[
            "je", "j'", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "me", "m'", "te",
            "t'", "se", "s'", "moi", "toi", "lui", "eux", "leur", "y", "en", "qui", "que",
        ],
        object_pronouns: &[
            "me", "m'", "te", "t'", "se", "s'", "nous", "vous", "le", "la", "les", "lui", "leur",
            "y", "en",
        ],
        auxiliaries: &[
            "suis", "es", "est", "sommes", "êtes", "sont", "étais", "était", "étaient", "serai",
            "seras", "sera", "serons", "serez", "seront", "ai", "as", "a", "avons", "avez", "ont",
            "avais", "avait", "avaient", "aurai", "aura", "auront", "vais", "vas", "va", "allons",
            "allez", "vont",
        ],
        copulas: &[
            "suis", "es", "est", "sommes", "êtes", "sont", "étais", "était", "étaient", "serai",
            "sera", "seront",
        ],
        prepositions: &[
            "à", "a", "de", "d'", "dans", "en", "sur", "sous", "avec", "sans", "pour", "par",
            "chez", "avant", "après", "entre", "vers", "contre",
        ],
        postpositions: &[],
        conjunctions: &["et", "ou", "mais", "donc", "car", "ni"],
        particles: &[],
        enclitic_suffixes: &[],
        complementizers: &["que", "qu'", "qui", "si", "quand", "lorsque", "comme"],
        adverbs: &["ne", "pas", "plus", "très", "tres", "bien", "mal"],
        adverb_suffixes: &["ment"],
        adjectives: &[
            "grand",
            "grande",
            "petit",
            "petite",
            "bon",
            "bonne",
            "mauvais",
            "mauvaise",
            "intelligent",
            "intelligente",
            "important",
            "importante",
        ],
        adjective_suffixes: &["able", "ible", "ique"],
        verbs: &[
            "être",
            "etre",
            "avoir",
            "aller",
            "faire",
            "dire",
            "pouvoir",
            "vouloir",
            "savoir",
            "venir",
            "voir",
            "vois",
            "voit",
            "sais",
            "sait",
            "viens",
            "vient",
            "lit",
            "lisent",
            "pense",
            "pensent",
            "devoir",
            "prendre",
            "parler",
            "aimer",
            "donner",
            "changer",
            "manger",
            "finir",
            "choisir",
            "recueillez",
            "voulez",
            "étaient",
            "etaient",
            "parlent",
            "mangent",
            "finissent",
        ],
        verb_suffixes: &[
            "aient", "issent", "èrent", "erent", "er", "ir", "re", "ez", "ons", "ait", "ais",
        ],
        subject_verb_suffixes: &["ent"],
        non_verbs: &[
            "intelligent",
            "président",
            "president",
            "moment",
            "vent",
            "argent",
            "enfant",
            "parent",
            "client",
            "document",
            "comment",
            "souvent",
            "vraiment",
            "lentement",
            "seulement",
        ],
        ..HeuristicSyntaxProfile::empty()
    }
}

pub fn synthesize_ipa(word: &str) -> Option<String> {
    synthesize_ipa_with_pos(word, None)
}

pub fn synthesize_ipa_with_pos(word: &str, part_of_speech: Option<PartOfSpeech>) -> Option<String> {
    let chars = normalize(word)?;
    let mute_final_ent = matches!(
        part_of_speech,
        Some(PartOfSpeech::Verb | PartOfSpeech::Auxiliary)
    );
    let mut ipa = String::new();
    let mut index = 0usize;
    while index < chars.len() {
        let rest = chars[index..].iter().collect::<String>();
        let (symbol, consumed) = if rest.starts_with("eaux") {
            ("o", 4)
        } else if rest.starts_with("eau") {
            ("o", 3)
        } else if rest.starts_with("aient") {
            ("ɛ", 5)
        } else if rest.starts_with("ait") || rest.starts_with("ais") {
            ("ɛ", 3)
        } else if mute_final_ent && rest.starts_with("ent") && index + 3 == chars.len() {
            ("", 3)
        } else if rest.starts_with("ez") && index + 2 == chars.len() {
            ("e", 2)
        } else if rest.starts_with("er") && index + 2 == chars.len() {
            ("e", 2)
        } else if starts_nasal(&chars, index, &['a'], &['n', 'm'])
            || starts_nasal(&chars, index, &['e'], &['n', 'm'])
        {
            ("ɑ̃", 2)
        } else if starts_nasal(&chars, index, &['o'], &['n', 'm']) {
            ("ɔ̃", 2)
        } else if starts_nasal_sequence(&chars, index, &['a', 'i'], &['n', 'm'])
            || starts_nasal_sequence(&chars, index, &['e', 'i'], &['n', 'm'])
        {
            ("ɛ̃", 3)
        } else if starts_nasal(&chars, index, &['i', 'y'], &['n', 'm']) {
            ("ɛ̃", 2)
        } else if starts_nasal_sequence(&chars, index, &['u'], &['n', 'm']) {
            ("œ̃", 2)
        } else if rest.starts_with("au") {
            ("o", 2)
        } else if rest.starts_with("ai") || rest.starts_with("ei") {
            ("ɛ", 2)
        } else if rest.starts_with("ou") {
            ("u", 2)
        } else if rest.starts_with("oi") {
            ("wa", 2)
        } else if rest.starts_with("ui") {
            ("ɥi", 2)
        } else if rest.starts_with("eu") {
            ("ø", 2)
        } else if rest.starts_with("œu") {
            ("œ", 2)
        } else if rest.starts_with("oeu") {
            ("œ", 3)
        } else if rest.starts_with("ch") {
            ("ʃ", 2)
        } else if rest.starts_with("gn") {
            ("ɲ", 2)
        } else if rest.starts_with("qu") {
            ("k", 2)
        } else if rest.starts_with("ill") && index > 0 && is_ipa_vowel(chars[index - 1]) {
            ("j", 3)
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
        'é' => "e",
        'è' | 'ê' | 'ë' => "ɛ",
        'e' => "ə",
        'f' => "f",
        'g' if matches!(next, Some('e' | 'i' | 'y')) => "ʒ",
        'g' => "ɡ",
        'h' => "",
        'i' => "i",
        'y' => "i",
        'j' => "ʒ",
        'k' => "k",
        'l' => "l",
        'm' => "m",
        'n' => "n",
        'o' | 'ô' => "ɔ",
        'p' => "p",
        'r' => "ʁ",
        's' => "s",
        't' => "t",
        'u' | 'ù' | 'û' | 'ü' => "y",
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
        .replace(['î', 'ï'], "i")
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

fn starts_nasal(chars: &[char], index: usize, vowels: &[char], nasals: &[char]) -> bool {
    chars.get(index).is_some_and(|ch| vowels.contains(ch))
        && chars.get(index + 1).is_some_and(|ch| nasals.contains(ch))
        && chars
            .get(index + 2)
            .is_none_or(|ch| !is_plain_vowel(*ch) && !nasals.contains(ch))
}

fn starts_nasal_sequence(chars: &[char], index: usize, sequence: &[char], nasals: &[char]) -> bool {
    chars
        .get(index..index + sequence.len())
        .is_some_and(|slice| slice == sequence)
        && chars
            .get(index + sequence.len())
            .is_some_and(|ch| nasals.contains(ch))
        && chars
            .get(index + sequence.len() + 1)
            .is_none_or(|ch| !is_plain_vowel(*ch) && !nasals.contains(ch))
}

fn trim_silent_finals(ipa: &str) -> String {
    if let Some(stem) = ipa.strip_suffix('ə') {
        return stem.to_string();
    }
    ipa.trim_end_matches(['s', 't', 'd']).to_string()
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

fn is_plain_vowel(ch: char) -> bool {
    matches!(
        ch,
        'a' | 'e'
            | 'é'
            | 'è'
            | 'ê'
            | 'ë'
            | 'i'
            | 'o'
            | 'u'
            | 'y'
            | 'à'
            | 'â'
            | 'î'
            | 'ï'
            | 'ô'
            | 'ù'
            | 'û'
            | 'ü'
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
        "a" | "ɑ̃" | "e" | "ɛ" | "ɛ̃" | "i" | "o" | "ɔ" | "ɔ̃" | "u" | "y" | "ø" | "œ" | "œ̃" | "ə"
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
        assert_eq!(synthesize_ipa("pense").as_deref(), Some("/ˈpɑ̃s/"));
    }

    #[test]
    fn french_mutes_final_ent_when_syntax_marks_a_verb() {
        assert_eq!(
            synthesize_ipa_with_pos("parlent", Some(PartOfSpeech::Verb)).as_deref(),
            Some("/ˈpaʁl/")
        );
        assert_eq!(
            synthesize_ipa_with_pos("parlent", None).as_deref(),
            Some("/paˈʁlɑ̃/")
        );
    }
}
