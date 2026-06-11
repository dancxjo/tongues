use crate::data::variety_by_code;
use crate::evidence::{EvidenceProvenance, EvidenceSource};
use crate::feature::{FeatureBundle, FeatureValue};
use crate::ids::{FeatureId, PhoneId, PhonemeId};
use crate::phonology::{PhoneToken, PhonemeToken};
use crate::spec::Spec;
use crate::syllabify::{syllabify_phones, syllables_to_ipa};

#[derive(Debug, Clone, PartialEq)]
pub struct OpenEpdPronunciation {
    pub raw_ipa: String,
    pub normalized_ipa: String,
    pub phonemes: Vec<PhonemeToken>,
    pub phones: Vec<PhoneToken>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenEpdParseError {
    Empty,
    UnknownSymbol { symbol: String, byte_index: usize },
}

impl std::fmt::Display for OpenEpdParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenEpdParseError::Empty => write!(f, "empty OpenEPD IPA transcription"),
            OpenEpdParseError::UnknownSymbol { symbol, byte_index } => {
                write!(
                    f,
                    "unknown OpenEPD IPA symbol {symbol:?} at byte {byte_index}"
                )
            }
        }
    }
}

impl std::error::Error for OpenEpdParseError {}

pub fn parse_openepd_ipa(
    ipa: &str,
    variety: &str,
) -> Result<OpenEpdPronunciation, OpenEpdParseError> {
    let raw_ipa = ipa.trim();
    if raw_ipa.is_empty() {
        return Err(OpenEpdParseError::Empty);
    }

    let tokens = tokenize_openepd_ipa(raw_ipa)?;
    let phonemes = tokens_to_phonemes(&tokens, variety);
    let phones = tokens_to_phones(&tokens);
    let normalized_ipa = if let Some(variety) = variety_by_code(variety) {
        syllables_to_ipa(&syllabify_phones(&phones, &variety))
    } else {
        render_openepd_phonemes(&phonemes)
    };

    Ok(OpenEpdPronunciation {
        raw_ipa: raw_ipa.to_string(),
        normalized_ipa,
        phonemes,
        phones,
    })
}

pub fn normalize_openepd_ipa(ipa: &str) -> Result<String, OpenEpdParseError> {
    parse_openepd_ipa(ipa, "en-US-GA").map(|pronunciation| pronunciation.normalized_ipa)
}

pub fn render_openepd_phonemes(phonemes: &[PhonemeToken]) -> String {
    let mut out = String::new();
    for phoneme in phonemes {
        if feature_category(&phoneme.features, "stress") == Some("primary") {
            out.push('ˈ');
        } else if feature_category(&phoneme.features, "stress") == Some("secondary") {
            out.push('ˌ');
        }
        if feature_bool(&phoneme.features, "syllable_boundary_before") == Some(true)
            && !out.is_empty()
            && !out.ends_with(['ˈ', 'ˌ'])
        {
            out.push('.');
        }
        if let Spec::Known(id) = &phoneme.phoneme {
            out.push_str(phoneme_symbol(id));
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum OpenEpdToken {
    Segment(&'static str),
    PrimaryStress,
    SecondaryStress,
    SyllableBoundary,
}

fn tokenize_openepd_ipa(ipa: &str) -> Result<Vec<OpenEpdToken>, OpenEpdParseError> {
    let mut tokens = Vec::new();
    let mut index = 0usize;
    while index < ipa.len() {
        let rest = &ipa[index..];
        let first = rest.chars().next().unwrap();
        if first.is_whitespace() {
            index += first.len_utf8();
            continue;
        }
        if rest.starts_with('ˈ') {
            tokens.push(OpenEpdToken::PrimaryStress);
            index += 'ˈ'.len_utf8();
            continue;
        }
        if rest.starts_with('ˌ') {
            tokens.push(OpenEpdToken::SecondaryStress);
            index += 'ˌ'.len_utf8();
            continue;
        }
        if rest.starts_with('.') {
            tokens.push(OpenEpdToken::SyllableBoundary);
            index += 1;
            continue;
        }
        if rest.starts_with('ː') {
            index += 'ː'.len_utf8();
            continue;
        }

        let Some(symbol) = OPENEPD_SEGMENTS
            .iter()
            .copied()
            .find(|symbol| rest.starts_with(symbol))
        else {
            let symbol = rest
                .chars()
                .next()
                .map(|c| c.to_string())
                .unwrap_or_default();
            return Err(OpenEpdParseError::UnknownSymbol {
                symbol,
                byte_index: index,
            });
        };
        tokens.push(OpenEpdToken::Segment(symbol));
        index += symbol.len();
    }
    Ok(tokens)
}

fn tokens_to_phonemes(tokens: &[OpenEpdToken], variety: &str) -> Vec<PhonemeToken> {
    let mut phonemes = Vec::new();
    let mut pending_stress = None;
    let mut pending_syllable_boundary = false;

    for token in tokens {
        match token {
            OpenEpdToken::PrimaryStress => pending_stress = Some("primary"),
            OpenEpdToken::SecondaryStress => pending_stress = Some("secondary"),
            OpenEpdToken::SyllableBoundary => pending_syllable_boundary = true,
            OpenEpdToken::Segment(symbol) => {
                let broad = broad_symbol(symbol);
                let mut features = FeatureBundle::default();
                put(&mut features, "source_schema", "openepd");
                put(&mut features, "notation", "ipa");
                put(&mut features, "raw_symbol", symbol);
                put(&mut features, "base_symbol", broad);
                if let Some(stress) = pending_stress.take() {
                    put(&mut features, "stress", stress);
                }
                if pending_syllable_boundary {
                    put_bool(&mut features, "syllable_boundary_before", true);
                    pending_syllable_boundary = false;
                }

                phonemes.push(PhonemeToken {
                    phoneme: Spec::Known(PhonemeId(format!("{variety}.phoneme.{broad}"))),
                    span: None,
                    features,
                    realized_as: Vec::new(),
                    confidence: 1.0,
                    provenance: EvidenceProvenance {
                        source: EvidenceSource::Lexicon,
                        method: "OpenEPD IPA transcription".into(),
                        version: Some("0.1.0".into()),
                    },
                });
            }
        }
    }

    phonemes
}

fn tokens_to_phones(tokens: &[OpenEpdToken]) -> Vec<PhoneToken> {
    let mut phones = Vec::new();
    let mut pending_stress = None;

    for token in tokens {
        match token {
            OpenEpdToken::PrimaryStress => pending_stress = Some("primary"),
            OpenEpdToken::SecondaryStress => pending_stress = Some("secondary"),
            OpenEpdToken::SyllableBoundary => {}
            OpenEpdToken::Segment(symbol) => {
                let broad = broad_symbol(symbol);
                let mut features = FeatureBundle::default();
                put(&mut features, "source_schema", "openepd");
                put(&mut features, "notation", "ipa");
                put(&mut features, "raw_symbol", symbol);
                put(&mut features, "base_symbol", broad);
                put_bool(&mut features, "syllabic", is_vowelish(broad));
                if let Some(stress) = pending_stress.take() {
                    put(&mut features, "stress", stress);
                }

                phones.push(PhoneToken {
                    phone: Spec::Known(PhoneId::from(format!("ipa.phone.{broad}"))),
                    span: None,
                    features,
                    acoustic_evidence: Vec::new(),
                    confidence: 1.0,
                    provenance: EvidenceProvenance {
                        source: EvidenceSource::Lexicon,
                        method: "OpenEPD IPA transcription".into(),
                        version: Some("0.1.0".into()),
                    },
                });
            }
        }
    }

    phones
}

fn broad_symbol(symbol: &str) -> &str {
    match symbol {
        "ɾ" => "t",
        "ʔ" => "t",
        "ᵻ" => "ɪ",
        "ᵿ" => "ʊ",
        "ɫ" => "l",
        "g" => "ɡ",
        "r" => "ɹ",
        "ɒ" => "ɑ",
        other => other,
    }
}

fn is_vowelish(symbol: &str) -> bool {
    matches!(
        symbol,
        "i" | "ɪ"
            | "e"
            | "ɛ"
            | "æ"
            | "ə"
            | "ʌ"
            | "ɑ"
            | "ɔ"
            | "o"
            | "ʊ"
            | "u"
            | "ɚ"
            | "ɝ"
            | "aʊ"
            | "aɪ"
            | "eɪ"
            | "oʊ"
            | "ɔɪ"
            | "iə"
            | "eə"
            | "ʊə"
            | "ɑɹ"
            | "ɔɹ"
            | "ɛɹ"
            | "ɪɹ"
            | "ʊɹ"
            | "əɹ"
    )
}

fn phoneme_symbol(id: &PhonemeId) -> &str {
    id.0.rsplit('.').next().unwrap_or(&id.0)
}

fn put(bundle: &mut FeatureBundle, name: &str, value: &str) {
    bundle.values.insert(
        FeatureId(format!("phonology.{name}")),
        Spec::Known(FeatureValue::Category(value.into())),
    );
}

fn put_bool(bundle: &mut FeatureBundle, name: &str, value: bool) {
    bundle.values.insert(
        FeatureId(format!("phonology.{name}")),
        Spec::Known(FeatureValue::Bool(value)),
    );
}

fn feature_category<'a>(bundle: &'a FeatureBundle, name: &str) -> Option<&'a str> {
    match bundle.values.get(&FeatureId(format!("phonology.{name}")))? {
        Spec::Known(FeatureValue::Category(value)) => Some(value),
        _ => None,
    }
}

fn feature_bool(bundle: &FeatureBundle, name: &str) -> Option<bool> {
    match bundle.values.get(&FeatureId(format!("phonology.{name}")))? {
        Spec::Known(FeatureValue::Bool(value)) => Some(*value),
        _ => None,
    }
}

const OPENEPD_SEGMENTS: &[&str] = &[
    "tʃ", "dʒ", "aʊ", "aɪ", "eɪ", "oʊ", "ɔɪ", "iə", "eə", "ʊə", "ɑɹ", "ɔɹ", "ɛɹ", "ɪɹ", "ʊɹ", "əɹ",
    "ɝ", "ɚ", "p", "b", "t", "d", "k", "ɡ", "g", "m", "n", "ŋ", "f", "v", "θ", "ð", "s", "z", "ʃ",
    "ʒ", "h", "l", "ɫ", "ɹ", "r", "j", "w", "i", "ɪ", "e", "ɛ", "æ", "ə", "ʌ", "ɑ", "ɒ", "ɔ", "o",
    "ʊ", "u", "ɾ", "ʔ", "ᵻ", "ᵿ",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_openepd_ipa_without_arpabet() {
        let pronunciation = parse_openepd_ipa("stˈupəd", "en-US-GA").unwrap();

        assert_eq!(pronunciation.normalized_ipa, "ˈstu.pəd");
        assert!(
            pronunciation
                .phonemes
                .iter()
                .all(|token| matches!(&token.phoneme, Spec::Known(id) if !id.0.contains("UW")))
        );
    }

    #[test]
    fn preserves_r_colored_vowels_as_speech_framework_phones() {
        assert_eq!(normalize_openepd_ipa("hɝd").unwrap(), "hɝd");
        assert_eq!(normalize_openepd_ipa("bɚd").unwrap(), "bɚd");
    }

    #[test]
    fn moves_openepd_vowel_stress_to_syllable_onset() {
        assert_eq!(normalize_openepd_ipa("zˈɪl").unwrap(), "ˈzɪl");
        assert_eq!(normalize_openepd_ipa("zˈɪɡ").unwrap(), "ˈzɪɡ");
    }

    #[test]
    fn rejects_unknown_symbols() {
        assert!(matches!(
            normalize_openepd_ipa("a#"),
            Err(OpenEpdParseError::UnknownSymbol { .. })
        ));
    }
}
