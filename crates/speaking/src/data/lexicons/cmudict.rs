use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::data::lexicons::{self, CMUDICT_ID, LexiconAdapter, LexiconLookup, PronunciationStatus};
use crate::data::notation::PronunciationNotation;

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

pub const REGISTRATIONS: &[LexiconAdapter] = &[LexiconAdapter {
    id: CMUDICT_ID,
    notation: PronunciationNotation::Cmudict,
    lookup,
}];

struct RuntimeCmudictFiles {
    base: &'static str,
    variant_pronunciations: &'static str,
}

const RUNTIME_FILES: &[RuntimeCmudictFiles] = &[
    RuntimeCmudictFiles {
        base: "cmudict.dict",
        variant_pronunciations: "cmudict.vp",
    },
    RuntimeCmudictFiles {
        base: "cmudict-0.7b",
        variant_pronunciations: "cmudict-0.7b.vp",
    },
];

pub fn lookup(word: &str) -> LexiconLookup {
    let entry = bundled().lookup_entry(word);
    LexiconLookup {
        lookup: entry.lookup,
        source: entry.source,
        status: entry.status,
        candidates: entry
            .candidates
            .into_iter()
            .map(|candidate| {
                candidate
                    .into_iter()
                    .map(|phoneme| phoneme.raw_symbol())
                    .collect()
            })
            .collect(),
    }
}

impl CmudictLexicon {
    pub fn bundled() -> Self {
        if let Some(lexicon) = Self::load_from_runtime_path() {
            return lexicon;
        }

        let mut lexicon = Self {
            entries: HashMap::new(),
        };
        lexicon.extend_from_embedded_cmudict();
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

    fn extend_from_embedded_cmudict(&mut self) {
        let embedded = arpabet_cmudict::load_cmudict();
        let mut words = embedded.keys().map(String::as_str).collect::<Vec<_>>();
        words.sort_unstable();
        for raw_word in words {
            let Some(phonemes) = embedded.get_polyphone_str(raw_word) else {
                continue;
            };
            let word = raw_word
                .find('(')
                .map(|index| &raw_word[..index])
                .unwrap_or(raw_word);
            let candidate = phonemes.into_iter().map(CmuPhoneme::parse).collect();
            let entry = self
                .entries
                .entry(word.to_lowercase().into_boxed_str())
                .or_insert_with(|| LexiconEntry {
                    candidates: Vec::new(),
                    source: "base cmu",
                });
            if !entry.candidates.contains(&candidate) {
                entry.candidates.push(candidate);
            }
        }
    }

    fn load_from_runtime_path() -> Option<Self> {
        for runtime_files in RUNTIME_FILES {
            for base_path in lexicons::runtime_file_candidates(&[runtime_files.base]) {
                let Ok(base_data) = std::fs::read_to_string(&base_path) else {
                    continue;
                };
                let mut lexicon = Self {
                    entries: HashMap::new(),
                };
                lexicon.extend_from_str(&base_data, "base cmu");

                let vp_path = base_path.with_file_name(runtime_files.variant_pronunciations);
                if let Ok(vp_data) = std::fs::read_to_string(&vp_path) {
                    lexicon.extend_from_str(&vp_data, "base cmu");
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
static HOMOPHONES: OnceLock<HashMap<Vec<String>, Vec<String>>> = OnceLock::new();

pub fn bundled() -> &'static CmudictLexicon {
    BUNDLED.get_or_init(CmudictLexicon::bundled)
}

/// Returns every bundled CMUdict spelling that shares a stress-normalized
/// pronunciation with `word`. Results retain their own pronunciation source.
pub fn homophones(word: &str) -> Vec<PronunciationEntry> {
    let lexicon = bundled();
    let source = lexicon.lookup_entry(word);
    if source.status == PronunciationStatus::Missing {
        return Vec::new();
    }
    let reverse = HOMOPHONES.get_or_init(|| {
        let mut reverse = HashMap::<Vec<String>, Vec<String>>::new();
        for (spelling, entry) in &lexicon.entries {
            for candidate in &entry.candidates {
                let key = candidate
                    .iter()
                    .map(|phoneme| phoneme.base.clone())
                    .collect::<Vec<_>>();
                reverse.entry(key).or_default().push(spelling.to_string());
            }
        }
        for spellings in reverse.values_mut() {
            spellings.sort();
            spellings.dedup();
        }
        reverse
    });
    let mut spellings = BTreeSet::new();
    for candidate in source.candidates {
        let key = candidate
            .iter()
            .map(|phoneme| phoneme.base.clone())
            .collect::<Vec<_>>();
        if let Some(matches) = reverse.get(&key) {
            spellings.extend(matches.iter().cloned());
        }
    }
    spellings
        .into_iter()
        .map(|spelling| lexicon.lookup_entry(&spelling))
        .collect()
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
    fn homophones_find_other_spellings_without_a_corpus_vocabulary() {
        let spellings = homophones("pair")
            .into_iter()
            .map(|entry| entry.lookup)
            .collect::<BTreeSet<_>>();
        assert!(spellings.contains("pair"));
        assert!(spellings.contains("pear"));
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
