use std::collections::HashMap;

use crate::feature::{FeatureBundle, FeatureSystem, FeatureValue};
use crate::ids::{LanguageId, PhoneId, PhonemeId, VarietyId};
use crate::orthography::Orthography;
use crate::phonetics::{Phone, PhoneInventory};
use crate::phonology::{Phoneme, PhonemeInventory};
use crate::rules::{PhonotacticConstraint, Phonotactics, RuleStatus, SyllableShape};
use crate::segment::{Environment, SegmentMatcher, SegmentStatus, SymbolAlias};
use crate::spec::Spec;
use crate::variety::{LinguisticVariety, VarietyImplementationStatus, VarietyStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GreekVariety {
    Modern,
    Ancient,
    Koine,
}

impl GreekVariety {
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "el" | "el-GR" | "el-GR-Standard" => Some(Self::Modern),
            "grc" | "grc-Attic" | "grc-Ancient" => Some(Self::Ancient),
            "grc-Koine" | "el-Koine" => Some(Self::Koine),
            _ => None,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::Modern => "el-GR-Standard",
            Self::Ancient => "grc-Attic",
            Self::Koine => "grc-Koine",
        }
    }

    fn language(self) -> &'static str {
        match self {
            Self::Modern => "el",
            Self::Ancient | Self::Koine => "grc",
        }
    }
}

const MODERN_SEGMENTS: &[&str] = &[
    "a", "e", "i", "o", "u", "v", "ɣ", "ʝ", "ð", "z", "θ", "k", "c", "l", "m", "n", "ks", "p", "r",
    "s", "t", "f", "x", "ç", "ps", "b", "d", "ɡ",
];

const ANCIENT_SEGMENTS: &[&str] = &[
    "a", "aː", "e", "eː", "ɛː", "i", "iː", "o", "oː", "u", "uː", "y", "yː", "ai̯", "au̯", "ei̯", "eu̯",
    "oi̯", "ou̯", "b", "ɡ", "d", "z", "tʰ", "k", "l", "m", "n", "ks", "p", "r", "s", "t", "pʰ", "kʰ",
    "ps", "h",
];

const KOINE_SEGMENTS: &[&str] = &[
    "a", "e", "i", "o", "u", "y", "ai̯", "au̯", "eu̯", "oi̯", "b", "β", "ɣ", "d", "ð", "z", "θ", "k",
    "l", "m", "n", "ks", "p", "r", "s", "t", "f", "x", "ps", "h",
];

pub fn variety(id: &str) -> LinguisticVariety {
    let greek = GreekVariety::from_id(id).unwrap_or(GreekVariety::Modern);
    let phonemes = phoneme_inventory(greek);
    let phones = phone_inventory(greek);

    LinguisticVariety {
        id: VarietyId(greek.id().into()),
        language: LanguageId(greek.language().into()),
        name: match greek {
            GreekVariety::Modern => "Standard Modern Greek".into(),
            GreekVariety::Ancient => "Ancient Greek (Attic)".into(),
            GreekVariety::Koine => "Koine Greek".into(),
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
            constraints: [
                &["p", "r"][..],
                &["p", "l"][..],
                &["t", "r"][..],
                &["k", "r"][..],
                &["k", "l"][..],
                &["f", "r"][..],
                &["f", "l"][..],
                &["x", "r"][..],
            ]
            .into_iter()
            .map(|cluster| cluster_constraint(greek, cluster))
            .collect(),
        }),
        orthography: Some(Orthography {
            name: "Greek orthography".into(),
            ..Default::default()
        }),
        morphology: None,
        acoustic_profile: None,
        prosody_profile: None,
        status: VarietyStatus::Attested,
        implementation_status: VarietyImplementationStatus::Complete,
    }
}

pub fn synthesize_ipa(word: &str, variety: GreekVariety) -> Option<String> {
    let normalized = normalize_greek_word(word)?;
    let stress_position = normalized
        .iter()
        .position(|letter| letter.stressed)
        .or_else(|| default_stress_position(&normalized));
    let mut ipa = String::new();
    let mut vowel_index = 0usize;
    let mut index = 0usize;
    while index < normalized.len() {
        if starts_vowel_unit(&normalized, index, variety).is_some()
            || is_vowel(normalized[index].base)
        {
            if Some(vowel_index) == stress_position {
                ipa.push('ˈ');
            }
            vowel_index += 1;
        }
        let (symbol, consumed) = match variety {
            GreekVariety::Modern => modern_symbol(&normalized, index)?,
            GreekVariety::Ancient => ancient_symbol(&normalized, index)?,
            GreekVariety::Koine => koine_symbol(&normalized, index)?,
        };
        ipa.push_str(symbol);
        index += consumed;
    }
    let ipa = reposition_primary_stress(&ipa);
    (!ipa.is_empty()).then_some(format!("/{ipa}/"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GreekLetter {
    base: char,
    stressed: bool,
}

fn normalize_greek_word(word: &str) -> Option<Vec<GreekLetter>> {
    let mut letters = Vec::new();
    for ch in word.trim().chars() {
        if matches!(ch, '-' | '\'' | '’') {
            continue;
        }
        if ch.is_whitespace() {
            return None;
        }
        let Some(letter) = normalize_greek_letter(ch) else {
            return None;
        };
        letters.push(letter);
    }
    if letters.is_empty() || letters.len() > 48 {
        return None;
    }
    Some(letters)
}

fn normalize_greek_letter(ch: char) -> Option<GreekLetter> {
    let lower = ch.to_lowercase().next().unwrap_or(ch);
    let (base, stressed) = match lower {
        'α' => ('α', false),
        'ά' | 'ᾶ' | 'ὰ' | 'ἀ' | 'ἁ' | 'ἄ' | 'ἅ' | 'ἂ' | 'ἃ' | 'ἆ' | 'ἇ' => {
            ('α', true)
        }
        'β' => ('β', false),
        'γ' => ('γ', false),
        'δ' => ('δ', false),
        'ε' => ('ε', false),
        'έ' | 'ὲ' | 'ἐ' | 'ἑ' | 'ἔ' | 'ἕ' | 'ἒ' | 'ἓ' => ('ε', true),
        'ζ' => ('ζ', false),
        'η' => ('η', false),
        'ή' | 'ῆ' | 'ὴ' | 'ἠ' | 'ἡ' | 'ἤ' | 'ἥ' | 'ἢ' | 'ἣ' | 'ἦ' | 'ἧ' => {
            ('η', true)
        }
        'θ' => ('θ', false),
        'ι' | 'ϊ' => ('ι', false),
        'ί' | 'ΐ' | 'ῖ' | 'ὶ' | 'ἰ' | 'ἱ' | 'ἴ' | 'ἵ' | 'ἲ' | 'ἳ' | 'ἶ' | 'ἷ' => {
            ('ι', true)
        }
        'κ' => ('κ', false),
        'λ' => ('λ', false),
        'μ' => ('μ', false),
        'ν' => ('ν', false),
        'ξ' => ('ξ', false),
        'ο' => ('ο', false),
        'ό' | 'ὸ' | 'ὀ' | 'ὁ' | 'ὄ' | 'ὅ' | 'ὂ' | 'ὃ' => ('ο', true),
        'π' => ('π', false),
        'ρ' | 'ῥ' => ('ρ', false),
        'σ' | 'ς' => ('σ', false),
        'τ' => ('τ', false),
        'υ' | 'ϋ' => ('υ', false),
        'ύ' | 'ΰ' | 'ῦ' | 'ὺ' | 'ὐ' | 'ὑ' | 'ὔ' | 'ὕ' | 'ὒ' | 'ὓ' | 'ὖ' | 'ὗ' => {
            ('υ', true)
        }
        'φ' => ('φ', false),
        'χ' => ('χ', false),
        'ψ' => ('ψ', false),
        'ω' => ('ω', false),
        'ώ' | 'ῶ' | 'ὼ' | 'ὠ' | 'ὡ' | 'ὤ' | 'ὥ' | 'ὢ' | 'ὣ' | 'ὦ' | 'ὧ' => {
            ('ω', true)
        }
        _ => return None,
    };
    Some(GreekLetter { base, stressed })
}

fn modern_symbol(letters: &[GreekLetter], index: usize) -> Option<(&'static str, usize)> {
    let c = letters[index].base;
    let next = letters.get(index + 1).map(|letter| letter.base);
    let after_next = letters.get(index + 2).map(|letter| letter.base);
    match (c, next) {
        ('α', Some('ι')) => Some(("e", 2)),
        ('ε', Some('ι')) | ('ο', Some('ι')) | ('υ', Some('ι')) => Some(("i", 2)),
        ('ο', Some('υ')) => Some(("u", 2)),
        ('α', Some('υ')) => Some((if is_voiceless(after_next) { "af" } else { "av" }, 2)),
        ('ε', Some('υ')) | ('η', Some('υ')) => {
            Some((if is_voiceless(after_next) { "ef" } else { "ev" }, 2))
        }
        ('μ', Some('π')) => Some(("b", 2)),
        ('ν', Some('τ')) => Some(("d", 2)),
        ('γ', Some('κ')) | ('γ', Some('γ')) => Some(("ɡ", 2)),
        _ => Some((modern_single(c, modern_front_context(letters, index))?, 1)),
    }
}

fn modern_single(c: char, front_context: bool) -> Option<&'static str> {
    Some(match c {
        'α' => "a",
        'β' => "v",
        'γ' if front_context => "ʝ",
        'γ' => "ɣ",
        'δ' => "ð",
        'ε' => "e",
        'ζ' => "z",
        'η' | 'ι' | 'υ' => "i",
        'θ' => "θ",
        'κ' if front_context => "c",
        'κ' => "k",
        'λ' => "l",
        'μ' => "m",
        'ν' => "n",
        'ξ' => "ks",
        'ο' | 'ω' => "o",
        'π' => "p",
        'ρ' => "r",
        'σ' => "s",
        'τ' => "t",
        'φ' => "f",
        'χ' if front_context => "ç",
        'χ' => "x",
        'ψ' => "ps",
        _ => return None,
    })
}

fn modern_front_context(letters: &[GreekLetter], index: usize) -> bool {
    let next = letters.get(index + 1).map(|letter| letter.base);
    let after_next = letters.get(index + 2).map(|letter| letter.base);
    next.is_some_and(is_front_vowel)
        || matches!(
            (next, after_next),
            (Some('α'), Some('ι'))
                | (Some('ε'), Some('ι'))
                | (Some('ο'), Some('ι'))
                | (Some('υ'), Some('ι'))
        )
}

fn ancient_symbol(letters: &[GreekLetter], index: usize) -> Option<(&'static str, usize)> {
    let c = letters[index].base;
    let next = letters.get(index + 1).map(|letter| letter.base);
    match (c, next) {
        ('α', Some('ι')) => Some(("ai̯", 2)),
        ('α', Some('υ')) => Some(("au̯", 2)),
        ('ε', Some('ι')) => Some(("ei̯", 2)),
        ('ε', Some('υ')) => Some(("eu̯", 2)),
        ('ο', Some('ι')) => Some(("oi̯", 2)),
        ('ο', Some('υ')) => Some(("ou̯", 2)),
        _ => Some((ancient_single(c)?, 1)),
    }
}

fn ancient_single(c: char) -> Option<&'static str> {
    Some(match c {
        'α' => "a",
        'β' => "b",
        'γ' => "ɡ",
        'δ' => "d",
        'ε' => "e",
        'ζ' => "z",
        'η' => "ɛː",
        'θ' => "tʰ",
        'ι' => "i",
        'κ' => "k",
        'λ' => "l",
        'μ' => "m",
        'ν' => "n",
        'ξ' => "ks",
        'ο' => "o",
        'π' => "p",
        'ρ' => "r",
        'σ' => "s",
        'τ' => "t",
        'υ' => "y",
        'φ' => "pʰ",
        'χ' => "kʰ",
        'ψ' => "ps",
        'ω' => "oː",
        _ => return None,
    })
}

fn koine_symbol(letters: &[GreekLetter], index: usize) -> Option<(&'static str, usize)> {
    let c = letters[index].base;
    let next = letters.get(index + 1).map(|letter| letter.base);
    match (c, next) {
        ('α', Some('ι')) => Some(("e", 2)),
        ('ε', Some('ι')) => Some(("i", 2)),
        ('ο', Some('ι')) => Some(("y", 2)),
        ('ο', Some('υ')) => Some(("u", 2)),
        ('α', Some('υ')) => Some(("au̯", 2)),
        ('ε', Some('υ')) => Some(("eu̯", 2)),
        _ => Some((koine_single(c)?, 1)),
    }
}

fn koine_single(c: char) -> Option<&'static str> {
    Some(match c {
        'α' => "a",
        'β' => "β",
        'γ' => "ɣ",
        'δ' => "ð",
        'ε' => "e",
        'ζ' => "z",
        'η' | 'ι' => "i",
        'θ' => "θ",
        'κ' => "k",
        'λ' => "l",
        'μ' => "m",
        'ν' => "n",
        'ξ' => "ks",
        'ο' | 'ω' => "o",
        'π' => "p",
        'ρ' => "r",
        'σ' => "s",
        'τ' => "t",
        'υ' => "y",
        'φ' => "f",
        'χ' => "x",
        'ψ' => "ps",
        _ => return None,
    })
}

fn phoneme_inventory(variety: GreekVariety) -> PhonemeInventory {
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

fn phone_inventory(variety: GreekVariety) -> PhoneInventory {
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

fn segments(variety: GreekVariety) -> &'static [&'static str] {
    match variety {
        GreekVariety::Modern => MODERN_SEGMENTS,
        GreekVariety::Ancient => ANCIENT_SEGMENTS,
        GreekVariety::Koine => KOINE_SEGMENTS,
    }
}

fn segment_features(symbol: &str) -> FeatureBundle {
    let mut features = FeatureBundle::default();
    let is_vowel = matches!(
        symbol,
        "a" | "aː"
            | "e"
            | "eː"
            | "ɛː"
            | "i"
            | "iː"
            | "o"
            | "oː"
            | "u"
            | "uː"
            | "y"
            | "yː"
            | "ai̯"
            | "au̯"
            | "ei̯"
            | "eu̯"
            | "oi̯"
            | "ou̯"
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

fn starts_vowel_unit(letters: &[GreekLetter], index: usize, variety: GreekVariety) -> Option<()> {
    let (symbol, _) = match variety {
        GreekVariety::Modern => modern_symbol(letters, index)?,
        GreekVariety::Ancient => ancient_symbol(letters, index)?,
        GreekVariety::Koine => koine_symbol(letters, index)?,
    };
    segment_features(symbol)
        .values
        .get(&crate::ids::FeatureId("phonology.syllabic".into()))
        .is_some_and(|value| *value == Spec::Known(FeatureValue::Bool(true)))
        .then_some(())
}

fn default_stress_position(letters: &[GreekLetter]) -> Option<usize> {
    let count = letters
        .iter()
        .filter(|letter| is_vowel(letter.base))
        .count();
    if count == 0 {
        None
    } else if count <= 2 {
        Some(0)
    } else {
        Some(count - 2)
    }
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'α' | 'ε' | 'η' | 'ι' | 'ο' | 'υ' | 'ω')
}

fn is_front_vowel(c: char) -> bool {
    matches!(c, 'ε' | 'η' | 'ι' | 'υ')
}

fn is_voiceless(c: Option<char>) -> bool {
    matches!(c, Some('θ' | 'κ' | 'ξ' | 'π' | 'σ' | 'τ' | 'φ' | 'χ' | 'ψ'))
}

fn is_ipa_vowel(ch: char) -> bool {
    matches!(ch, 'a' | 'e' | 'ɛ' | 'i' | 'o' | 'u' | 'y')
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

fn cluster_constraint(variety: GreekVariety, cluster: &[&str]) -> PhonotacticConstraint {
    let label = cluster.join("");
    PhonotacticConstraint {
        id: format!("{}.legal_onset.{label}", variety.id()),
        description: format!("Legal Greek onset cluster {label}"),
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
    fn greek_varieties_load() {
        assert_eq!(variety("el").id.0, "el-GR-Standard");
        assert_eq!(variety("grc").id.0, "grc-Attic");
        assert_eq!(variety("grc-Koine").id.0, "grc-Koine");
    }

    #[test]
    fn greek_varieties_distinguish_kai() {
        assert_eq!(
            synthesize_ipa("και", GreekVariety::Modern).as_deref(),
            Some("/ˈce/")
        );
        assert_eq!(
            synthesize_ipa("και", GreekVariety::Ancient).as_deref(),
            Some("/ˈkai̯/")
        );
        assert_eq!(
            synthesize_ipa("και", GreekVariety::Koine).as_deref(),
            Some("/ˈke/")
        );
    }
}
