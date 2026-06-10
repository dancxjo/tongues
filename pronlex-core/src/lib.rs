//! Core vocabulary types for pronlex.
//!
//! Provides `PhoneVocab` (ARPABET) and `CharVocab` (spelling characters).
//! Special tokens occupy the lowest IDs so they are stable across builds.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Special token IDs ──────────────────────────────────────────────────────

/// Padding token – used to fill sequences to a fixed length.
pub const PAD_ID: u32 = 0;
/// Mask token – replaces phones that the model must predict.
pub const MASK_ID: u32 = 1;
/// Unknown token – fallback for phones / chars not in the vocabulary.
pub const UNK_ID: u32 = 2;
/// Beginning-of-sequence marker.
pub const BOS_ID: u32 = 3;
/// End-of-sequence marker.
pub const EOS_ID: u32 = 4;
/// Number of reserved special tokens.
pub const SPECIAL_COUNT: u32 = 5;

// ── ARPABET reference inventory ────────────────────────────────────────────

/// Vowel bases whose stress variants (0/1/2) are each a distinct phone.
static ARPABET_VOWEL_BASES: &[&str] = &[
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
];

/// Consonants – no stress digit.
static ARPABET_CONSONANTS: &[&str] = &[
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH",
    "T", "TH", "V", "W", "Y", "Z", "ZH",
];

// ── PhoneVocab ─────────────────────────────────────────────────────────────

/// Bidirectional map between ARPABET phone strings and integer IDs.
///
/// IDs 0–4 are reserved for special tokens (PAD, MASK, UNK, BOS, EOS).
/// Standard CMUdict phones follow, then any extra phones discovered at
/// prepare-time are appended.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhoneVocab {
    /// Index → phone string (position == ID).
    pub phones: Vec<String>,
    /// Phone string → ID.
    pub phone_to_id: HashMap<String, u32>,
}

impl PhoneVocab {
    /// Construct the standard ARPABET vocabulary.
    pub fn build_standard() -> Self {
        let mut phones: Vec<String> = vec![
            "<PAD>".into(),
            "<MASK>".into(),
            "<UNK>".into(),
            "<BOS>".into(),
            "<EOS>".into(),
        ];
        for base in ARPABET_VOWEL_BASES {
            for stress in 0..=2u8 {
                phones.push(format!("{}{}", base, stress));
            }
        }
        for &c in ARPABET_CONSONANTS {
            phones.push(c.to_string());
        }
        let phone_to_id: HashMap<String, u32> = phones
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i as u32))
            .collect();
        PhoneVocab { phones, phone_to_id }
    }

    /// Append phones from `extra` that are not already in the vocabulary.
    /// Call this after `build_standard` to handle any non-standard CMUdict phones.
    pub fn extend_from_data(&mut self, extra: &[String]) {
        for phone in extra {
            if !self.phone_to_id.contains_key(phone) {
                let id = self.phones.len() as u32;
                self.phone_to_id.insert(phone.clone(), id);
                self.phones.push(phone.clone());
            }
        }
    }

    /// Look up the ID for a phone string, returning `UNK_ID` for unknown phones.
    pub fn get_id(&self, phone: &str) -> u32 {
        *self.phone_to_id.get(phone).unwrap_or(&UNK_ID)
    }

    /// Look up the phone string for an ID.
    pub fn get_phone(&self, id: u32) -> &str {
        self.phones
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("<UNK>")
    }

    /// Total number of tokens including specials.
    pub fn size(&self) -> usize {
        self.phones.len()
    }
}

// ── CharVocab ──────────────────────────────────────────────────────────────

/// Bidirectional map between lowercase ASCII characters and integer IDs.
///
/// ID 0 is PAD, ID 1 is UNK.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharVocab {
    /// Index → char (position == ID).
    pub chars: Vec<char>,
    /// Char → ID.
    pub char_to_id: HashMap<char, u32>,
}

impl CharVocab {
    /// Build from a collection of lowercase word strings.
    pub fn build(words: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        let mut seen: std::collections::BTreeSet<char> = std::collections::BTreeSet::new();
        for word in words {
            for c in word.as_ref().chars() {
                seen.insert(c);
            }
        }
        // ID 0 = PAD, ID 1 = UNK, then sorted chars
        let mut chars: Vec<char> = vec!['\0', '?'];
        chars.extend(seen);

        let char_to_id: HashMap<char, u32> = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();
        CharVocab { chars, char_to_id }
    }

    /// Char → ID (UNK_ID=1 for unknown chars).
    pub fn get_id(&self, c: char) -> u32 {
        *self.char_to_id.get(&c).unwrap_or(&1u32)
    }

    /// Total number of tokens including specials.
    pub fn size(&self) -> usize {
        self.chars.len()
    }
}

// ── Combined vocabulary ─────────────────────────────────────────────────────

/// Vocabulary bundle serialised to `vocab.json` at prepare time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    pub phones: PhoneVocab,
    pub chars: CharVocab,
}

impl Vocab {
    pub fn new(phones: PhoneVocab, chars: CharVocab) -> Self {
        Vocab { phones, chars }
    }
}

// ── Errors ─────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum VocabError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phone_vocab_standard_roundtrip() {
        let v = PhoneVocab::build_standard();
        // Specials
        assert_eq!(v.get_id("<PAD>"), PAD_ID);
        assert_eq!(v.get_id("<MASK>"), MASK_ID);
        assert_eq!(v.get_id("<UNK>"), UNK_ID);
        assert_eq!(v.get_phone(PAD_ID), "<PAD>");
        assert_eq!(v.get_phone(MASK_ID), "<MASK>");
        // Vowels with stress
        let ah0 = v.get_id("AH0");
        assert!(ah0 >= SPECIAL_COUNT, "AH0 should follow specials");
        assert_eq!(v.get_phone(ah0), "AH0");
        let ah1 = v.get_id("AH1");
        let ah2 = v.get_id("AH2");
        assert_ne!(ah0, ah1);
        assert_ne!(ah1, ah2);
        // Consonants have no stress digit
        let b_id = v.get_id("B");
        assert!(b_id >= SPECIAL_COUNT);
        assert_eq!(v.get_phone(b_id), "B");
        // UNK for unknown phone
        assert_eq!(v.get_id("ZZZNOPE"), UNK_ID);
    }

    #[test]
    fn phone_vocab_extend() {
        let mut v = PhoneVocab::build_standard();
        let before = v.size();
        v.extend_from_data(&["EXTRAPHONE".to_string()]);
        assert_eq!(v.size(), before + 1);
        let id = v.get_id("EXTRAPHONE");
        assert_eq!(v.get_phone(id), "EXTRAPHONE");
        // Adding again is idempotent
        v.extend_from_data(&["EXTRAPHONE".to_string()]);
        assert_eq!(v.size(), before + 1);
    }

    #[test]
    fn char_vocab_roundtrip() {
        let words = vec!["hello".to_string(), "world".to_string()];
        let v = CharVocab::build(&words);
        // Known chars
        for c in "helloworld".chars() {
            let id = v.get_id(c);
            assert!(id >= 2, "known char should be >= 2 (past PAD/UNK)");
        }
        // Unknown char
        assert_eq!(v.get_id('?'), 1, "? is UNK placeholder");
        // PAD
        assert_eq!(v.get_id('\0'), 0);
    }
}
