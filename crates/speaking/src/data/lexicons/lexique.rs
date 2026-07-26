use std::collections::HashMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::data::lexicons::{
    self, LEXIQUE383_ID, LexiconAdapter, LexiconLookup, PronunciationStatus,
};
use crate::data::notation::PronunciationNotation;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PronunciationEntry {
    pub original: String,
    pub lookup: String,
    pub source: &'static str,
    pub candidates: Vec<String>,
    pub status: PronunciationStatus,
}

#[derive(Debug, Clone)]
struct LexiconEntry {
    candidates: Vec<String>,
    source: &'static str,
}

#[derive(Debug, Clone)]
pub struct LexiqueLexicon {
    entries: HashMap<Box<str>, LexiconEntry>,
}

pub const REGISTRATIONS: &[LexiconAdapter] = &[LexiconAdapter {
    id: LEXIQUE383_ID,
    notation: PronunciationNotation::Ipa,
    lookup,
}];

const RUNTIME_FILES: &[&str] = &["Lexique383.tsv", "lexique383.tsv"];

pub fn lookup(word: &str) -> LexiconLookup {
    let entry = bundled().lookup_entry(word);
    LexiconLookup {
        lookup: entry.lookup,
        source: entry.source,
        status: entry.status,
        candidates: entry
            .candidates
            .into_iter()
            .map(|candidate| vec![candidate])
            .collect(),
    }
}

impl LexiqueLexicon {
    pub fn bundled() -> Self {
        if let Some(lexicon) = Self::load_from_runtime_path() {
            return lexicon;
        }

        let mut lexicon = Self {
            entries: HashMap::new(),
        };
        lexicon.extend_from_tsv(include_str!("Lexique383.tsv"), LEXIQUE383_ID);
        lexicon.extend_from_tsv(GENERATED_OVERRIDES, "generated overrides");
        lexicon
    }

    fn load_from_runtime_path() -> Option<Self> {
        for path in lexicons::runtime_file_candidates(RUNTIME_FILES) {
            if let Ok(data) = std::fs::read_to_string(&path) {
                let mut lexicon = Self {
                    entries: HashMap::new(),
                };
                lexicon.extend_from_tsv(&data, LEXIQUE383_ID);
                lexicon.extend_from_tsv(GENERATED_OVERRIDES, "generated overrides");
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
        lexicon.extend_from_tsv(data, LEXIQUE383_ID);
        lexicon
    }

    pub fn lookup_entry(&self, word: &str) -> PronunciationEntry {
        let exact_key = word.trim().to_lowercase();
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

    fn extend_from_tsv(&mut self, data: &str, source: &'static str) {
        let mut lines = data.lines();
        let Some(header) = lines.next() else {
            return;
        };
        let headers = header.split('\t').collect::<Vec<_>>();
        let ortho_idx = headers
            .iter()
            .position(|header| *header == "ortho")
            .unwrap_or(0);
        let phon_idx = headers
            .iter()
            .position(|header| *header == "phon")
            .unwrap_or(1);

        for line in lines.map(str::trim).filter(|line| !line.is_empty()) {
            let columns = line.split('\t').collect::<Vec<_>>();
            let Some(word) = columns.get(ortho_idx).map(|word| word.trim()) else {
                continue;
            };
            let Some(raw_pronunciation) = columns.get(phon_idx).map(|phon| phon.trim()) else {
                continue;
            };
            if word.is_empty() || raw_pronunciation.is_empty() {
                continue;
            }
            let Some(ipa) = lexique_phon_to_ipa(raw_pronunciation) else {
                continue;
            };
            let key = normalize_for_lookup(word).into_boxed_str();
            let entry = self.entries.entry(key).or_insert_with(|| LexiconEntry {
                candidates: Vec::new(),
                source,
            });
            if !entry.candidates.contains(&ipa) {
                entry.candidates.push(ipa);
            }
        }
    }
}

static BUNDLED: OnceLock<LexiqueLexicon> = OnceLock::new();

pub fn bundled() -> &'static LexiqueLexicon {
    BUNDLED.get_or_init(LexiqueLexicon::bundled)
}

pub fn normalize_for_lookup(word: &str) -> String {
    word.trim_matches(|character: char| {
        !(character.is_alphabetic() || matches!(character, '\'' | '’' | '-'))
    })
    .to_lowercase()
    .replace('œ', "oe")
    .replace('æ', "ae")
}

pub fn lexique_phon_to_ipa(input: &str) -> Option<String> {
    let mut out = String::new();
    let chars = input.trim().chars().collect::<Vec<_>>();
    let mut index = 0usize;
    while index < chars.len() {
        let ch = chars[index];
        match ch {
            '/' | '[' | ']' | '.' | 'ˈ' | 'ˌ' | ' ' => {}
            'a' | 'b' | 'd' | 'e' | 'f' | 'i' | 'j' | 'k' | 'l' | 'm' | 'n' | 'o' | 'p' | 's'
            | 't' | 'u' | 'v' | 'w' | 'y' | 'z' | 'ɑ' | 'ɛ' | 'ɔ' | 'ə' | 'ø' | 'œ' | 'ʁ' | 'ʃ'
            | 'ʒ' | 'ɲ' | 'ɥ' | 'ɡ' => out.push(ch),
            'g' => out.push('ɡ'),
            'R' => out.push('ʁ'),
            'S' => out.push('ʃ'),
            'Z' => out.push('ʒ'),
            'N' => out.push('ɲ'),
            'H' | '8' => out.push('ɥ'),
            'E' => out.push('ɛ'),
            'O' => out.push('ɔ'),
            'A' => out.push('ɑ'),
            '@' => out.push('ə'),
            '2' => out.push('ø'),
            '9' => out.push('œ'),
            '5' | 'ɛ' if matches!(chars.get(index + 1), Some('~') | Some('\u{0303}')) => {
                out.push_str("ɛ̃");
                index += 1;
            }
            '5' => out.push_str("ɛ̃"),
            '§' => out.push_str("ɔ̃"),
            '°' => out.push_str("ɑ̃"),
            '~' | '\u{0303}' => {
                if let Some(last) = out.pop() {
                    match last {
                        'a' | 'ɑ' => out.push_str("ɑ̃"),
                        'e' | 'ɛ' | 'i' => out.push_str("ɛ̃"),
                        'o' | 'ɔ' => out.push_str("ɔ̃"),
                        'œ' => out.push_str("œ̃"),
                        other => out.push(other),
                    }
                }
            }
            _ => return None,
        }
        index += 1;
    }
    (!out.is_empty()).then_some(format!("/{out}/"))
}

pub const GENERATED_OVERRIDES: &str = "\
ortho\tphon
des\tde
deux\tdø
dix\tdis
est\tɛ
huit\tɥit
les\tle
mes\tme
monsieur\tməsjø
myriel\tmiʁjɛl
premier\tpʁəmje
recueillez\tʁəkœje
ses\tse
tes\tte
trois\ttʁwa
très\ttʁɛ
un\tœ̃
voulez\tvule
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lexique_tsv_and_normalizes_lookup() {
        let lexicon = LexiqueLexicon::from_str(
            "ortho\tphon\tlemme\nbonjour\tb§ZuR\tbonjour\nvoulez\tvule\tvouloir\n",
        );
        assert_eq!(lexicon.lookup_entry("bonjour").candidates[0], "/bɔ̃ʒuʁ/");
        assert_eq!(lexicon.lookup_entry("\"Voulez!\"").candidates[0], "/vule/");
    }

    #[test]
    fn parses_ipa_and_common_lexique_codes() {
        assert_eq!(lexique_phon_to_ipa("b§ZuR").as_deref(), Some("/bɔ̃ʒuʁ/"));
        assert_eq!(lexique_phon_to_ipa("p2R").as_deref(), Some("/pøʁ/"));
        assert_eq!(lexique_phon_to_ipa("s9R").as_deref(), Some("/sœʁ/"));
        assert_eq!(lexique_phon_to_ipa("avɑ̃").as_deref(), Some("/avɑ̃/"));
    }
}
