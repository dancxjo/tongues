pub mod cmudict;
pub mod lexique;

use serde::{Deserialize, Serialize};

use crate::data::notation::PronunciationNotation;

pub const CMUDICT_ID: &str = "cmudict";
pub const LEXIQUE383_ID: &str = "lexique383";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PronunciationStatus {
    Exact,
    Normalized,
    Guessed,
    Missing,
}

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
    pub status: PronunciationStatus,
    pub candidates: Vec<Vec<String>>,
}

pub fn adapter_for_id(id: &str) -> Option<LexiconAdapter> {
    lexicon_adapters().find(|adapter| adapter.id == id)
}

pub fn lexicon_ids() -> impl Iterator<Item = &'static str> {
    lexicon_adapters().map(|adapter| adapter.id)
}

fn lexicon_adapters() -> impl Iterator<Item = LexiconAdapter> {
    [cmudict::REGISTRATIONS, lexique::REGISTRATIONS]
        .into_iter()
        .flatten()
        .copied()
}
