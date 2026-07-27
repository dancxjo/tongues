pub mod cmudict;
pub mod cmudict_rp;
pub mod lexique;

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::data::notation::PronunciationNotation;

pub const CMUDICT_ID: &str = "cmudict";
pub const CMUDICT_RP_ID: &str = "cmudict-rp";
pub const LEXIQUE383_ID: &str = "lexique383";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PronunciationStatus {
    Exact,
    Normalized,
    Guessed,
    Missing,
}

// Equality is part of this public descriptor's existing API. Its callback is
// static registry data, so preserving the derived comparison is intentional.
#[allow(unpredictable_function_pointer_comparisons)]
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
    [
        cmudict::REGISTRATIONS,
        cmudict_rp::REGISTRATIONS,
        lexique::REGISTRATIONS,
    ]
    .into_iter()
    .flatten()
    .copied()
}

pub(crate) fn runtime_file_candidates(file_names: &[&str]) -> Vec<PathBuf> {
    let Some(root) = runtime_model_root() else {
        return Vec::new();
    };
    runtime_file_candidates_from_root(&root, file_names)
}

fn runtime_model_root() -> Option<PathBuf> {
    let home = if let Some(home_var) = std::env::var_os("MORTAR_SEA_HOME") {
        PathBuf::from(home_var)
    } else {
        dirs::data_local_dir()?.join("mortar-sea")
    };
    Some(home.join("models/speaking"))
}

fn runtime_file_candidates_from_root(root: &Path, file_names: &[&str]) -> Vec<PathBuf> {
    let mut roots = vec![root.to_path_buf()];
    if let Ok(entries) = fs::read_dir(root) {
        let mut child_roots = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();
        child_roots.sort();
        roots.extend(child_roots);
    }

    roots
        .into_iter()
        .flat_map(|directory| {
            file_names
                .iter()
                .map(move |file_name| directory.join(file_name))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_file_candidates_scan_root_and_variety_directories() {
        let root = std::env::temp_dir().join(format!(
            "speaking-lexicon-candidates-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("variety-b")).expect("create temp variety-b");
        fs::create_dir_all(root.join("variety-a")).expect("create temp variety-a");

        let candidates = runtime_file_candidates_from_root(&root, &["lexicon.tsv"]);
        assert_eq!(
            candidates,
            [
                root.join("lexicon.tsv"),
                root.join("variety-a/lexicon.tsv"),
                root.join("variety-b/lexicon.tsv"),
            ]
        );

        fs::remove_dir_all(&root).expect("remove temp runtime root");
    }
}
