use std::collections::HashMap;
use std::path::PathBuf;
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
pub struct LexiconEntry {
    pub candidates: Vec<Vec<CmuPhoneme>>,
    pub source: &'static str,
}

#[derive(Debug, Clone)]
pub struct CmudictLexicon {
    entries: HashMap<Box<str>, LexiconEntry>,
}

pub const GENERATED_OVERRIDES: &str = "\
logorrhea L AO2 G ER0 IY1 AH0
sansome S AE1 N S AH0 M
talkativeness T AO1 K AH0 T IH0 V N AH0 S
wordiness W ER1 D IY0 N AH0 S
";

impl CmudictLexicon {
    pub fn bundled() -> Self {
        if let Some(lexicon) = Self::load_from_runtime_path() {
            return lexicon;
        }

        let mut lexicon = Self {
            entries: HashMap::new(),
        };
        lexicon.extend_from_str(include_str!("cmudict.dict"), "base cmu");
        lexicon.extend_from_str(
            "\
mm M
mm-hm M HH M
mm-hmm M HH M
mmm M
",
            "extras",
        );
        lexicon.extend_from_str(GENERATED_OVERRIDES, "generated overrides");
        lexicon
    }

    fn load_from_runtime_path() -> Option<Self> {
        let home = if let Some(home_var) = std::env::var_os("MORTAR_SEA_HOME") {
            PathBuf::from(home_var)
        } else {
            dirs::data_local_dir()?.join("mortar-sea")
        };

        let mut base_path = home.join("models/speech/en-us/cmudict.dict");
        let mut vp_path = home.join("models/speech/en-us/cmudict.vp");

        if !base_path.exists() {
            base_path = home.join("models/speech/en-us/cmudict-0.7b");
            vp_path = home.join("models/speech/en-us/cmudict-0.7b.vp");
        }

        if base_path.exists() {
            if let Ok(base_data) = std::fs::read_to_string(&base_path) {
                let mut lexicon = Self {
                    entries: HashMap::new(),
                };
                lexicon.extend_from_str(&base_data, "base cmu");

                if vp_path.exists() {
                    if let Ok(vp_data) = std::fs::read_to_string(&vp_path) {
                        lexicon.extend_from_str(&vp_data, "base cmu");
                    }
                }

                lexicon.extend_from_str(
                    "\
mm M
mm-hm M HH M
mm-hmm M HH M
mmm M
",
                    "extras",
                );
                lexicon.extend_from_str(GENERATED_OVERRIDES, "generated overrides");
                return Some(lexicon);
            }
        }
        None
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(data: &str) -> Self {
        let mut lexicon = Self {
            entries: HashMap::new(),
        };
        lexicon.extend_from_str(data, "base cmu");
        lexicon
    }

    pub fn lookup_entry(&self, word: &str) -> PronunciationEntry {
        let exact_key = word.to_lowercase();
        if let Some(entry) = self.entries.get(exact_key.as_str()) {
            return PronunciationEntry {
                original: word.into(),
                lookup: exact_key,
                source: entry.source,
                candidates: entry.candidates.clone(),
                status: PronunciationStatus::Exact,
            };
        }

        let normalized = normalize_for_lookup(word);
        if normalized != exact_key
            && let Some(entry) = self.entries.get(normalized.as_str())
        {
            return PronunciationEntry {
                original: word.into(),
                lookup: normalized,
                source: entry.source,
                candidates: entry.candidates.clone(),
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
            source: "fallback",
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

    fn extend_from_str(&mut self, data: &str, source: &'static str) {
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

            let key = word.to_lowercase().into_boxed_str();
            let entry = self.entries.entry(key).or_insert_with(|| LexiconEntry {
                candidates: Vec::new(),
                source,
            });
            if !entry.candidates.contains(&phonemes) {
                entry.candidates.push(phonemes);
            }
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
        let sansome = bundled().lookup_entry("sansome");
        assert_eq!(sansome.status, PronunciationStatus::Exact);
        assert_eq!(
            sansome.candidates[0]
                .iter()
                .map(CmuPhoneme::raw_symbol)
                .collect::<Vec<_>>(),
            ["S", "AE1", "N", "S", "AH0", "M"]
        );
        assert_eq!(
            bundled().lookup_entry("xyzzyqux").status,
            PronunciationStatus::Missing
        );
    }
}
