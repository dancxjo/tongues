use std::collections::HashMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PronunciationStatus {
    Exact,
    Normalized,
    Guessed,
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PronunciationEntry {
    pub original: String,
    pub lookup: String,
    pub source: &'static str,
    pub candidates: Vec<Vec<CmuPhoneme>>,
    pub status: PronunciationStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CmuStress {
    Primary,
    Secondary,
    Unstressed,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CmuPhoneme {
    pub base: String,
    pub stress: Option<CmuStress>,
}

impl CmuPhoneme {
    pub fn parse(token: &str) -> Self {
        let stress = token.chars().last().and_then(|character| match character {
            '1' => Some(CmuStress::Primary),
            '2' => Some(CmuStress::Secondary),
            '0' => Some(CmuStress::Unstressed),
            _ => None,
        });
        let base = if stress.is_some() {
            token[..token.len() - 1].to_string()
        } else {
            token.to_string()
        };
        Self { base, stress }
    }

    pub fn raw_symbol(&self) -> String {
        let mut raw = self.base.clone();
        match self.stress {
            Some(CmuStress::Primary) => raw.push('1'),
            Some(CmuStress::Secondary) => raw.push('2'),
            Some(CmuStress::Unstressed) => raw.push('0'),
            None => {}
        }
        raw
    }
}

#[derive(Debug, Clone)]
pub struct CmudictLexicon {
    entries: HashMap<Box<str>, Vec<Vec<CmuPhoneme>>>,
}

impl CmudictLexicon {
    pub fn bundled() -> Self {
        let mut lexicon = Self::from_str(include_str!("cmudict.dict"));
        lexicon.extend_from_str(
            "\
mm M
mm-hm M HH M
mm-hmm M HH M
mmm M
",
        );
        lexicon
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(data: &str) -> Self {
        let mut lexicon = Self {
            entries: HashMap::new(),
        };
        lexicon.extend_from_str(data);
        lexicon
    }

    pub fn lookup_entry(&self, word: &str) -> PronunciationEntry {
        let exact_key = word.to_lowercase();
        if let Some(candidates) = self.entries.get(exact_key.as_str()) {
            return PronunciationEntry {
                original: word.into(),
                lookup: exact_key,
                source: "cmudict",
                candidates: candidates.clone(),
                status: PronunciationStatus::Exact,
            };
        }

        let normalized = normalize_for_lookup(word);
        if normalized != exact_key
            && let Some(candidates) = self.entries.get(normalized.as_str())
        {
            return PronunciationEntry {
                original: word.into(),
                lookup: normalized,
                source: "cmudict",
                candidates: candidates.clone(),
                status: PronunciationStatus::Normalized,
            };
        }

        PronunciationEntry {
            original: word.into(),
            lookup: if normalized.is_empty() {
                exact_key
            } else {
                normalized
            },
            source: "cmudict",
            candidates: Vec::new(),
            status: PronunciationStatus::Missing,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn extend_from_str(&mut self, data: &str) {
        for line in data.lines().map(str::trim) {
            if line.is_empty() || line.starts_with(";;;") {
                continue;
            }

            let mut parts = line.split_ascii_whitespace();
            let Some(raw_word) = parts.next() else {
                continue;
            };
            let word = raw_word
                .find('(')
                .map(|index| &raw_word[..index])
                .unwrap_or(raw_word);
            let phonemes = parts.map(CmuPhoneme::parse).collect::<Vec<_>>();
            if phonemes.is_empty() {
                continue;
            }
            self.entries
                .entry(word.to_lowercase().into_boxed_str())
                .or_default()
                .push(phonemes);
        }
    }
}

static BUNDLED: OnceLock<CmudictLexicon> = OnceLock::new();

pub fn bundled() -> &'static CmudictLexicon {
    BUNDLED.get_or_init(CmudictLexicon::bundled)
}

pub fn normalize_for_lookup(word: &str) -> String {
    word.trim_matches(|character: char| !character.is_alphabetic())
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_cmudict_preserves_expected_entries_and_stress() {
        let lexicon = bundled();
        assert!(lexicon.len() > 100_000);

        let okay = lexicon.lookup_entry("okay");
        assert_eq!(okay.status, PronunciationStatus::Exact);
        assert_eq!(
            okay.candidates[0]
                .iter()
                .map(CmuPhoneme::raw_symbol)
                .collect::<Vec<_>>(),
            ["OW2", "K", "EY1"]
        );

        let xylophone = lexicon.lookup_entry("xylophone");
        assert_eq!(
            xylophone.candidates[0]
                .iter()
                .map(|phoneme| phoneme.base.as_str())
                .collect::<Vec<_>>(),
            ["Z", "AY", "L", "AH", "F", "OW", "N"]
        );
    }

    #[test]
    fn lookup_entry_reports_normalized_and_missing_status() {
        assert_eq!(
            bundled().lookup_entry("\"hello!\"").status,
            PronunciationStatus::Normalized
        );
        assert_eq!(
            bundled().lookup_entry("xyzzyqux").status,
            PronunciationStatus::Missing
        );
    }
}
