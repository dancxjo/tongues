use std::collections::HashMap;

use crate::feature::FeatureSystem;
use crate::ids::{LanguageId, PhoneId, PhonemeId, VarietyId};
use crate::orthography::Orthography;
use crate::phonetics::{Phone, PhoneInventory};
use crate::phonology::{Phoneme, PhonemeInventory};
use crate::rules::{PhonotacticConstraint, Phonotactics, RuleStatus, SyllableShape};
use crate::segment::{Environment, SegmentMatcher, SegmentStatus, SymbolAlias};
use crate::spec::Spec;
use crate::variety::{LinguisticVariety, VarietyImplementationStatus, VarietyStatus};

const A: PhoneId = PhoneId::borrowed("ipa.phone.a");
const E: PhoneId = PhoneId::borrowed("ipa.phone.e");
const I: PhoneId = PhoneId::borrowed("ipa.phone.i");
const O: PhoneId = PhoneId::borrowed("ipa.phone.o");
const U: PhoneId = PhoneId::borrowed("ipa.phone.u");
const B: PhoneId = PhoneId::borrowed("ipa.phone.b");
const CH: PhoneId = PhoneId::borrowed("ipa.phone.tʃ");
const D: PhoneId = PhoneId::borrowed("ipa.phone.d");
const F: PhoneId = PhoneId::borrowed("ipa.phone.f");
const G: PhoneId = PhoneId::borrowed("ipa.phone.ɡ");
const X: PhoneId = PhoneId::borrowed("ipa.phone.x");
const Y: PhoneId = PhoneId::borrowed("ipa.phone.ʝ");
const K: PhoneId = PhoneId::borrowed("ipa.phone.k");
const L: PhoneId = PhoneId::borrowed("ipa.phone.l");
const LL: PhoneId = PhoneId::borrowed("ipa.phone.ʎ");
const M: PhoneId = PhoneId::borrowed("ipa.phone.m");
const N: PhoneId = PhoneId::borrowed("ipa.phone.n");
const NY: PhoneId = PhoneId::borrowed("ipa.phone.ɲ");
const P: PhoneId = PhoneId::borrowed("ipa.phone.p");
const TAP: PhoneId = PhoneId::borrowed("ipa.phone.ɾ");
const TRILL: PhoneId = PhoneId::borrowed("ipa.phone.r");
const S: PhoneId = PhoneId::borrowed("ipa.phone.s");
const T: PhoneId = PhoneId::borrowed("ipa.phone.t");
const THETA: PhoneId = PhoneId::borrowed("ipa.phone.θ");
const W: PhoneId = PhoneId::borrowed("ipa.phone.w");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanishVariety {
    Castilian,
    LatinAmericanStandard,
}

impl SpanishVariety {
    pub fn id(self) -> &'static str {
        match self {
            Self::Castilian => "es-ES-Castilian",
            Self::LatinAmericanStandard => "es-419-Standard",
        }
    }

    pub fn accent_tag(self) -> &'static str {
        match self {
            Self::Castilian => "Castilian",
            Self::LatinAmericanStandard => "LatAm",
        }
    }
}

#[derive(Debug, Clone)]
struct SpanishSegment {
    symbol: &'static str,
    phone: PhoneId,
}

const COMMON_PHONEMES: &[SpanishSegment] = &[
    SpanishSegment {
        symbol: "a",
        phone: A,
    },
    SpanishSegment {
        symbol: "e",
        phone: E,
    },
    SpanishSegment {
        symbol: "i",
        phone: I,
    },
    SpanishSegment {
        symbol: "o",
        phone: O,
    },
    SpanishSegment {
        symbol: "u",
        phone: U,
    },
    SpanishSegment {
        symbol: "b",
        phone: B,
    },
    SpanishSegment {
        symbol: "tʃ",
        phone: CH,
    },
    SpanishSegment {
        symbol: "d",
        phone: D,
    },
    SpanishSegment {
        symbol: "f",
        phone: F,
    },
    SpanishSegment {
        symbol: "ɡ",
        phone: G,
    },
    SpanishSegment {
        symbol: "x",
        phone: X,
    },
    SpanishSegment {
        symbol: "ʝ",
        phone: Y,
    },
    SpanishSegment {
        symbol: "k",
        phone: K,
    },
    SpanishSegment {
        symbol: "l",
        phone: L,
    },
    SpanishSegment {
        symbol: "m",
        phone: M,
    },
    SpanishSegment {
        symbol: "n",
        phone: N,
    },
    SpanishSegment {
        symbol: "ɲ",
        phone: NY,
    },
    SpanishSegment {
        symbol: "p",
        phone: P,
    },
    SpanishSegment {
        symbol: "ɾ",
        phone: TAP,
    },
    SpanishSegment {
        symbol: "r",
        phone: TRILL,
    },
    SpanishSegment {
        symbol: "s",
        phone: S,
    },
    SpanishSegment {
        symbol: "t",
        phone: T,
    },
    SpanishSegment {
        symbol: "w",
        phone: W,
    },
];

const CASTILIAN_EXTRA_PHONEMES: &[SpanishSegment] = &[
    SpanishSegment {
        symbol: "θ",
        phone: THETA,
    },
    SpanishSegment {
        symbol: "ʎ",
        phone: LL,
    },
];

const LATIN_AMERICAN_EXTRA_PHONEMES: &[SpanishSegment] = &[];

const LEGAL_ONSETS: &[&[PhoneId]] = &[
    &[B, L],
    &[B, TAP],
    &[K, L],
    &[K, TAP],
    &[D, TAP],
    &[F, L],
    &[F, TAP],
    &[G, L],
    &[G, TAP],
    &[P, L],
    &[P, TAP],
    &[T, TAP],
];

pub fn variety(id: &str) -> LinguisticVariety {
    let variety = match id {
        "es" | "es-ES" | "es-ES-Castilian" => SpanishVariety::Castilian,
        "es-419" | "es-419-Standard" | "es-LatAm" => SpanishVariety::LatinAmericanStandard,
        _ => SpanishVariety::LatinAmericanStandard,
    };
    let phonemes = phoneme_inventory(variety);
    let phones = phone_inventory(variety);

    LinguisticVariety {
        id: VarietyId(variety.id().into()),
        language: LanguageId("es".into()),
        name: match variety {
            SpanishVariety::Castilian => "Castilian Spanish".into(),
            SpanishVariety::LatinAmericanStandard => "Standard Latin American Spanish".into(),
        },
        feature_system: FeatureSystem::default(),
        phonemes,
        phones,
        allophone_rules: Vec::new(),
        epenthesis_rules: Vec::new(),
        weak_forms: Vec::new(),
        orthographic_unit_pronunciations: Vec::new(),
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
            constraints: LEGAL_ONSETS
                .iter()
                .map(|cluster| cluster_constraint(variety, cluster))
                .collect(),
        }),
        orthography: Some(Orthography {
            name: "Spanish Latin orthography".into(),
            ..Default::default()
        }),
        morphology: None,
        acoustic_profile: None,
        prosody_profile: None,
        status: VarietyStatus::Attested,
        implementation_status: VarietyImplementationStatus::Complete,
    }
}

pub fn synthesize_ipa(word: &str, variety: SpanishVariety) -> Option<String> {
    let normalized = normalize_spanish_word(word)?;
    let chars = normalized;
    let stress_vowel = stress_vowel_index(&chars)?;
    let mut ipa = String::new();
    let mut vowel_index = 0_usize;
    let mut index = 0_usize;
    let mut at_word_start = true;
    while index < chars.len() {
        let c = chars[index];
        let next = chars.get(index + 1).copied();
        let after_next = chars.get(index + 2).copied();

        if is_vowel(c) {
            if vowel_index == stress_vowel {
                ipa.push('ˈ');
            }
            ipa.push(base_vowel(c));
            vowel_index += 1;
            at_word_start = false;
            index += 1;
            continue;
        }

        match c {
            'b' | 'v' => ipa.push('b'),
            'c' if matches!(next, Some('h')) => {
                ipa.push_str("tʃ");
                index += 1;
            }
            'c' if matches!(next, Some('e' | 'é' | 'i' | 'í')) => match variety {
                SpanishVariety::Castilian => ipa.push('θ'),
                SpanishVariety::LatinAmericanStandard => ipa.push('s'),
            },
            'c' => ipa.push('k'),
            'd' => ipa.push('d'),
            'f' => ipa.push('f'),
            'g' if matches!(next, Some('e' | 'é' | 'i' | 'í')) => ipa.push('x'),
            'g' if matches!(next, Some('u'))
                && matches!(after_next, Some('e' | 'é' | 'i' | 'í')) =>
            {
                ipa.push('ɡ');
                index += 1;
            }
            'g' => ipa.push('ɡ'),
            'h' => {}
            'j' => ipa.push('x'),
            'k' => ipa.push('k'),
            'l' if matches!(next, Some('l')) => {
                match variety {
                    SpanishVariety::Castilian => ipa.push('ʎ'),
                    SpanishVariety::LatinAmericanStandard => ipa.push('ʝ'),
                }
                index += 1;
            }
            'l' => ipa.push('l'),
            'm' => ipa.push('m'),
            'n' if matches!(next, Some('̃')) => {
                ipa.push('ɲ');
                index += 1;
            }
            'n' => ipa.push('n'),
            'ñ' => ipa.push('ɲ'),
            'p' => ipa.push('p'),
            'q' if matches!(next, Some('u'))
                && matches!(after_next, Some('e' | 'é' | 'i' | 'í')) =>
            {
                ipa.push('k');
                index += 1;
            }
            'q' => ipa.push('k'),
            'r' if at_word_start || matches!(next, Some('r')) => {
                ipa.push('r');
                if matches!(next, Some('r')) {
                    index += 1;
                }
            }
            'r' => ipa.push('ɾ'),
            's' | 'z' => match (c, variety) {
                ('z', SpanishVariety::Castilian) => ipa.push('θ'),
                _ => ipa.push('s'),
            },
            't' => ipa.push('t'),
            'w' => ipa.push('w'),
            'x' => ipa.push_str("ks"),
            'y' => {
                if index + 1 == chars.len() {
                    ipa.push('i');
                    vowel_index += 1;
                } else {
                    ipa.push('ʝ');
                }
            }
            '-' | '\'' | '’' => {}
            _ => return None,
        }
        at_word_start = false;
        index += 1;
    }

    let ipa = reposition_primary_stress(&ipa);
    (!ipa.is_empty()).then_some(format!("/{ipa}/"))
}

pub fn synthetic_pronunciations(word: &str) -> Vec<(SpanishVariety, String)> {
    [
        SpanishVariety::Castilian,
        SpanishVariety::LatinAmericanStandard,
    ]
    .into_iter()
    .filter_map(|variety| synthesize_ipa(word, variety).map(|ipa| (variety, ipa)))
    .collect()
}

fn phoneme_inventory(variety: SpanishVariety) -> PhonemeInventory {
    let mut phonemes = HashMap::new();
    for segment in segment_rows(variety) {
        let phoneme = Phoneme {
            id: PhonemeId(format!("{}.phoneme.{}", variety.id(), segment.symbol)),
            notation: format!("/{}/", segment.symbol),
            features: Default::default(),
            default_phone: Some(segment.phone.clone()),
            possible_phones: vec![segment.phone.clone()],
            aliases: vec![SymbolAlias {
                system: "spanish".into(),
                symbol: segment.symbol.into(),
            }],
            allophones: Vec::new(),
            status: SegmentStatus::Core,
        };
        phonemes.insert(phoneme.id.clone(), phoneme);
    }
    PhonemeInventory { phonemes }
}

fn phone_inventory(variety: SpanishVariety) -> PhoneInventory {
    let mut phones = HashMap::new();
    for segment in segment_rows(variety) {
        phones
            .entry(segment.phone.clone())
            .or_insert_with(|| Phone {
                id: segment.phone.clone(),
                ipa: segment.symbol.into(),
                features: Default::default(),
                aliases: vec![SymbolAlias {
                    system: "spanish".into(),
                    symbol: segment.symbol.into(),
                }],
                status: SegmentStatus::Core,
            });
    }
    PhoneInventory { phones }
}

fn segment_rows(variety: SpanishVariety) -> Vec<SpanishSegment> {
    let mut rows = COMMON_PHONEMES.to_vec();
    match variety {
        SpanishVariety::Castilian => rows.extend_from_slice(CASTILIAN_EXTRA_PHONEMES),
        SpanishVariety::LatinAmericanStandard => {
            rows.extend_from_slice(LATIN_AMERICAN_EXTRA_PHONEMES)
        }
    }
    rows
}

fn cluster_constraint(variety: SpanishVariety, cluster: &[PhoneId]) -> PhonotacticConstraint {
    let label = cluster
        .iter()
        .map(phone_symbol)
        .collect::<Vec<_>>()
        .join("");
    PhonotacticConstraint {
        id: format!("{}.legal_onset.{label}", variety.id()),
        description: format!("Legal Spanish onset cluster {label}"),
        matcher: SegmentMatcher::Any,
        environment: Environment {
            before: cluster.iter().cloned().map(SegmentMatcher::Phone).collect(),
            syllable_position: Spec::Known(crate::segment::SyllablePosition::Onset),
            ..Default::default()
        },
        status: RuleStatus::Productive,
    }
}

fn normalize_spanish_word(word: &str) -> Option<Vec<char>> {
    let normalized = word
        .trim()
        .to_lowercase()
        .replace('á', "á")
        .replace('é', "é")
        .replace('í', "í")
        .replace('ó', "ó")
        .replace('ú', "ú");
    if normalized.is_empty()
        || normalized.chars().count() > 48
        || normalized
            .chars()
            .any(|c| !(c.is_alphabetic() || matches!(c, '-' | '\'' | '’')))
    {
        return None;
    }
    Some(normalized.chars().collect())
}

fn stress_vowel_index(chars: &[char]) -> Option<usize> {
    let mut vowels = Vec::new();
    for (index, c) in chars.iter().enumerate() {
        if is_silent_qu_gu_u(chars, index) {
            continue;
        }
        if is_vowel(*c) {
            vowels.push((index, *c));
            if matches!(*c, 'á' | 'é' | 'í' | 'ó' | 'ú') {
                return Some(vowels.len() - 1);
            }
        }
    }
    if vowels.is_empty() {
        return None;
    }
    let final_letter = chars
        .iter()
        .rev()
        .find(|c| c.is_alphabetic())
        .copied()
        .unwrap_or(' ');
    if vowels.len() == 1 {
        return Some(0);
    }
    if matches!(final_letter, 'a' | 'e' | 'i' | 'o' | 'u' | 'n' | 's') {
        Some(vowels.len().saturating_sub(2))
    } else {
        Some(vowels.len() - 1)
    }
}

fn is_vowel(c: char) -> bool {
    matches!(
        c,
        'a' | 'á' | 'e' | 'é' | 'i' | 'í' | 'o' | 'ó' | 'u' | 'ú' | 'ü'
    )
}

fn base_vowel(c: char) -> char {
    match c {
        'á' => 'a',
        'é' => 'e',
        'í' => 'i',
        'ó' => 'o',
        'ú' => 'u',
        'ü' => 'u',
        other => other,
    }
}

fn reposition_primary_stress(ipa: &str) -> String {
    let mut chars = ipa.chars().collect::<Vec<_>>();
    let Some(stress_index) = chars.iter().position(|c| *c == 'ˈ') else {
        return ipa.to_string();
    };
    if stress_index == 0
        || chars
            .get(stress_index + 1)
            .is_none_or(|c| !is_base_vowel(*c))
    {
        return ipa.to_string();
    }

    let mut insert_index = stress_index;
    while insert_index > 0 && is_stress_onset_consonant(chars[insert_index - 1]) {
        insert_index -= 1;
        if insert_index > 0 && is_base_vowel(chars[insert_index - 1]) {
            break;
        }
    }
    if insert_index == stress_index {
        return ipa.to_string();
    }
    chars.remove(stress_index);
    chars.insert(insert_index, 'ˈ');
    chars.into_iter().collect()
}

fn is_stress_onset_consonant(c: char) -> bool {
    matches!(
        c,
        'b' | 'd'
            | 'f'
            | 'ɡ'
            | 'x'
            | 'ʝ'
            | 'k'
            | 'l'
            | 'ʎ'
            | 'm'
            | 'n'
            | 'ɲ'
            | 'p'
            | 'ɾ'
            | 'r'
            | 's'
            | 't'
            | 'θ'
            | 'w'
    )
}

fn is_base_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}

fn is_silent_qu_gu_u(chars: &[char], index: usize) -> bool {
    chars.get(index).copied() == Some('u')
        && matches!(
            index.checked_sub(1).and_then(|before| chars.get(before)),
            Some('q' | 'g')
        )
        && matches!(chars.get(index + 1), Some('e' | 'é' | 'i' | 'í'))
}

fn phone_symbol(phone: &PhoneId) -> &str {
    phone
        .as_str()
        .strip_prefix("ipa.phone.")
        .unwrap_or(phone.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spanish_varieties_load() {
        let castilian = variety("es-ES-Castilian");
        let latam = variety("es-419-Standard");
        assert_eq!(castilian.language.0, "es");
        assert_eq!(latam.language.0, "es");
        assert!(
            castilian
                .phonemes
                .phonemes
                .contains_key(&PhonemeId("es-ES-Castilian.phoneme.θ".into()))
        );
    }

    #[test]
    fn synthetic_pronunciation_distinguishes_standard_varieties() {
        assert_eq!(
            synthesize_ipa("zapato", SpanishVariety::Castilian).as_deref(),
            Some("/θaˈpato/")
        );
        assert_eq!(
            synthesize_ipa("zapato", SpanishVariety::LatinAmericanStandard).as_deref(),
            Some("/saˈpato/")
        );
        assert_eq!(
            synthesize_ipa("queso", SpanishVariety::LatinAmericanStandard).as_deref(),
            Some("/ˈkeso/")
        );
        assert_eq!(
            synthesize_ipa("perro", SpanishVariety::LatinAmericanStandard).as_deref(),
            Some("/ˈpero/")
        );
    }
}
