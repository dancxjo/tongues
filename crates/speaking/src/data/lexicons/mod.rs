pub mod cmudict;
pub mod lexique;

use crate::data::notation::PronunciationNotation;

pub const CMUDICT_ID: &str = "cmudict";
pub const LEXIQUE383_ID: &str = "lexique383";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LexiconAdapter {
    pub id: &'static str,
    pub notation: PronunciationNotation,
    pub lookup: fn(&str) -> LexiconLookup,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexiconLookup {
    pub lookup: String,
    pub source: &'static str,
    pub status: cmudict::PronunciationStatus,
    pub candidates: Vec<Vec<String>>,
}

pub const LEXICON_IDS: &[&str] = &[CMUDICT_ID, LEXIQUE383_ID];

const LEXICON_ADAPTERS: &[LexiconAdapter] = &[
    LexiconAdapter {
        id: CMUDICT_ID,
        notation: PronunciationNotation::Arpabet,
        lookup: cmudict_lookup,
    },
    LexiconAdapter {
        id: LEXIQUE383_ID,
        notation: PronunciationNotation::Ipa,
        lookup: lexique_lookup,
    },
];

pub fn adapter_for_id(id: &str) -> Option<LexiconAdapter> {
    LEXICON_ADAPTERS
        .iter()
        .copied()
        .find(|adapter| adapter.id == id)
}

fn cmudict_lookup(word: &str) -> LexiconLookup {
    let entry = cmudict::bundled().lookup_entry(word);
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

fn lexique_lookup(word: &str) -> LexiconLookup {
    let entry = lexique::bundled().lookup_entry(word);
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
