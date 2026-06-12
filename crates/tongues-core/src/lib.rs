//! Unified vocabulary for tongues sequence translation.
//!
//! Provides a single character-level `Vocab` that maps graphemes, broad IPA
//! phonemes, narrow IPA phones, and task/control tokens to shared IDs.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Special control token IDs ──────────────────────────────────────────────

pub const PAD_ID: u32 = 0;
pub const UNK_ID: u32 = 1;
pub const BOS_ID: u32 = 2;
pub const EOS_ID: u32 = 3;
pub const SEP_ID: u32 = 4;

// ── Task prefix token IDs ──────────────────────────────────────────────────

pub const G2P_ID: u32 = 5;
pub const P2G_ID: u32 = 6;

pub const SPECIAL_COUNT: u32 = 7;

// ── Unified Vocab ──────────────────────────────────────────────────────────

/// Bidirectional map between characters/special tokens and integer IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    /// Index → token string (position == ID).
    pub tokens: Vec<String>,
    /// Token string → ID.
    pub token_to_id: HashMap<String, u32>,
}

impl Vocab {
    /// Construct a unified vocabulary from words, phonemes, and phones.
    pub fn build(words: &[String], phonemes: &[String], phones: &[String]) -> Self {
        let mut tokens: Vec<String> = vec![
            "<PAD>".into(),
            "<UNK>".into(),
            "<BOS>".into(),
            "<EOS>".into(),
            "<SEP>".into(),
            "<G2P>".into(),
            "<P2G>".into(),
            "<task:g2p>".into(),
            "<task:p2g>".into(),
            "<task:align>".into(),
            "<task:normalize>".into(),
            "<task:guess_lang_from_spelling>".into(),
            "<task:guess_lang_from_ipa>".into(),
            "<task:guess_lang_from_spelling_and_ipa>".into(),
        ];

        let mut control_tokens = std::collections::BTreeSet::new();
        let mut seen = std::collections::BTreeSet::new();
        seed_broad_linguistic_vocab(&mut control_tokens, &mut seen);

        // Collect all unique characters
        for word in words {
            collect_angle_bracket_tokens(word, &mut control_tokens);
            for c in word.chars() {
                seen.insert(c.to_string());
            }
        }
        for pm in phonemes {
            collect_angle_bracket_tokens(pm, &mut control_tokens);
            for c in pm.chars() {
                seen.insert(c.to_string());
            }
        }
        for ph in phones {
            collect_angle_bracket_tokens(ph, &mut control_tokens);
            for c in ph.chars() {
                seen.insert(c.to_string());
            }
        }

        for token in control_tokens {
            if !tokens.contains(&token) {
                tokens.push(token);
            }
        }
        tokens.extend(seen);

        let token_to_id: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        Vocab {
            tokens,
            token_to_id,
        }
    }

    /// Look up the ID for a token string, returning `UNK_ID` for unknown tokens.
    pub fn get_id(&self, token: &str) -> u32 {
        *self.token_to_id.get(token).unwrap_or(&UNK_ID)
    }

    /// Look up the token string for an ID.
    pub fn get_token(&self, id: u32) -> &str {
        self.tokens
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("<UNK>")
    }

    /// Total number of tokens including specials.
    pub fn size(&self) -> usize {
        self.tokens.len()
    }

    /// Encode a string as IDs, preserving known `<...>` control tokens as atoms.
    pub fn encode_string(&self, s: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut index = 0;
        while index < s.len() {
            let rest = &s[index..];
            if rest.starts_with('<') {
                if let Some(end) = rest.find('>') {
                    let candidate = &rest[..=end];
                    if let Some(id) = self.token_to_id.get(candidate) {
                        ids.push(*id);
                        index += candidate.len();
                        continue;
                    }
                }
            }

            let Some(ch) = rest.chars().next() else {
                break;
            };
            ids.push(self.get_id(&ch.to_string()));
            index += ch.len_utf8();
        }
        ids
    }

    /// Decode a list of IDs back to a string (filtering out PAD/BOS/EOS/SEP).
    pub fn decode_ids(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&id| self.get_token(id))
            .filter(|&tok| tok != "<PAD>" && tok != "<BOS>" && tok != "<EOS>" && tok != "<SEP>")
            .collect::<Vec<_>>()
            .join("")
    }
}

fn collect_angle_bracket_tokens(value: &str, out: &mut std::collections::BTreeSet<String>) {
    let mut offset = 0;
    while let Some(start) = value[offset..].find('<') {
        let start = offset + start;
        let Some(end) = value[start..].find('>').map(|end| start + end) else {
            break;
        };
        if end > start + 1 {
            out.insert(value[start..=end].to_string());
        }
        offset = end + 1;
    }
}

fn seed_broad_linguistic_vocab(
    control_tokens: &mut std::collections::BTreeSet<String>,
    seen: &mut std::collections::BTreeSet<String>,
) {
    for token in [
        "<lang:eng>",
        "<lang:fra>",
        "<lang:deu>",
        "<lang:spa>",
        "<lang:lat>",
        "<lang:ell>",
        "<lang:grc>",
        "<lang:san>",
        "<N_PHONEME>",
        "<N_PHONE>",
        "<notation:phonemic>",
        "<notation:phonetic>",
        "<accent:Castilian>",
        "<accent:LatAm>",
        "<accent:GreekName>",
        "<accent:Latin>",
        "<accent:NeoLatinScientific>",
        "<accent:LegalLatin>",
    ] {
        control_tokens.insert(token.to_string());
    }

    for c in " \t\n-_'’.,;:!?/[](){}+*=~·ˈˌ.|‿͜͡".chars() {
        seen.insert(c.to_string());
    }

    for (start, end) in [
        (0x00A0, 0x024F), // Latin-1, Latin Extended-A/B, IPA-adjacent letters.
        (0x0250, 0x02AF), // IPA Extensions.
        (0x02B0, 0x02FF), // Spacing Modifier Letters.
        (0x0300, 0x036F), // Combining Diacritical Marks.
        (0x0370, 0x03FF), // Greek and Coptic.
        (0x0400, 0x04FF), // Cyrillic.
        (0x0590, 0x05FF), // Hebrew.
        (0x0900, 0x097F), // Devanagari.
        (0x1D00, 0x1D7F), // Phonetic Extensions.
        (0x1D80, 0x1DBF), // Phonetic Extensions Supplement.
        (0x1DC0, 0x1DFF), // Combining Diacritical Marks Supplement.
        (0x1E00, 0x1EFF), // Latin Extended Additional.
        (0x1F00, 0x1FFF), // Greek Extended.
        (0xA700, 0xA71F), // Modifier Tone Letters.
        (0xAB30, 0xAB6F), // Latin Extended-E.
        (0xFE20, 0xFE2F), // Combining Half Marks.
    ] {
        seed_char_range(seen, start, end);
    }
}

fn seed_char_range(seen: &mut std::collections::BTreeSet<String>, start: u32, end: u32) {
    for codepoint in start..=end {
        if let Some(c) = char::from_u32(codepoint) {
            if !c.is_control() {
                seen.insert(c.to_string());
            }
        }
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
    fn vocab_roundtrip() {
        let words = vec!["hello".to_string(), "world".to_string()];
        let phonemes = vec!["həˈloʊ".to_string()];
        let phones = vec!["hə.ˈloʊ".to_string()];
        let v = Vocab::build(&words, &phonemes, &phones);

        assert_eq!(v.get_id("<PAD>"), PAD_ID);
        assert_eq!(v.get_id("<UNK>"), UNK_ID);
        assert_eq!(v.get_id("<BOS>"), BOS_ID);

        let encoded = v.encode_string("hello");
        assert_eq!(encoded.len(), 5);
        let decoded = v.decode_ids(&encoded);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn vocab_encodes_angle_bracket_controls_as_atomic_tokens() {
        let words = vec!["<task:g2p> <lang:eng> disease".to_string()];
        let phonemes = vec!["dəˈziːz".to_string()];
        let v = Vocab::build(&words, &phonemes, &[]);

        let task_id = v.get_id("<task:g2p>");
        let lang_id = v.get_id("<lang:eng>");
        let align_id = v.get_id("<task:align>");
        assert_ne!(task_id, UNK_ID);
        assert_ne!(lang_id, UNK_ID);
        assert_ne!(align_id, UNK_ID);

        let encoded = v.encode_string("<task:g2p> <lang:eng> disease");
        assert_eq!(encoded[0], task_id);
        assert_eq!(encoded[2], lang_id);
    }

    #[test]
    fn vocab_seeds_broad_linguistic_ranges() {
        let v = Vocab::build(&[], &[], &[]);
        for token in ["<lang:lat>", "<lang:ell>", "<lang:grc>", "<lang:san>"] {
            assert_ne!(v.get_id(token), UNK_ID, "{token} should be seeded");
        }
        for c in ['θ', 'ɲ', '͡', 'ᵻ', '᷄', 'ᾱ', 'क', 'ā'] {
            assert_ne!(v.get_id(&c.to_string()), UNK_ID, "{c} should be seeded");
        }
    }
}
