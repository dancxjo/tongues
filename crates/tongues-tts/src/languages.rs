use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};

/// Stable model-declared language names and their learned embedding rows.
///
/// This catalog is intentionally separate from Tongues' linguistic-variety
/// registry. A checkpoint-local language such as `en` or `pt-br` is an opaque
/// model identity, not a prefix to infer from an [`speaking::VarietyId`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanguageCatalog {
    name_to_id: BTreeMap<String, u32>,
    num_languages: u32,
}

impl LanguageCatalog {
    pub fn from_json_str(source: &str, num_languages: u32) -> Result<Self> {
        let name_to_id: BTreeMap<String, u32> =
            serde_json::from_str(source).context("failed to parse language name map")?;
        Self::new(name_to_id, num_languages)
    }

    pub fn from_file(path: impl AsRef<Path>, num_languages: u32) -> Result<Self> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read language map {}", path.display()))?;
        Self::from_json_str(&source, num_languages)
            .with_context(|| format!("invalid language map {}", path.display()))
    }

    pub fn new(name_to_id: BTreeMap<String, u32>, num_languages: u32) -> Result<Self> {
        ensure!(num_languages > 0, "language count must be positive");
        ensure!(
            !name_to_id.is_empty(),
            "language name map must not be empty"
        );
        ensure!(
            name_to_id.keys().all(|name| !name.trim().is_empty()),
            "language name map contains an empty name"
        );
        ensure!(
            name_to_id.values().all(|id| *id < num_languages),
            "language name map contains an ID outside 0..{num_languages}"
        );
        let unique_ids = name_to_id.values().copied().collect::<BTreeSet<_>>();
        ensure!(
            unique_ids.len() == name_to_id.len(),
            "language name map assigns multiple names to one embedding row"
        );
        Ok(Self {
            name_to_id,
            num_languages,
        })
    }

    pub fn num_languages(&self) -> u32 {
        self.num_languages
    }

    pub fn available_names(&self) -> Vec<&str> {
        self.entries().into_iter().map(|(name, _)| name).collect()
    }

    pub fn entries(&self) -> Vec<(&str, u32)> {
        let mut languages = self.name_to_id.iter().collect::<Vec<_>>();
        languages.sort_by_key(|(name, id)| (**id, name.as_str()));
        languages
            .into_iter()
            .map(|(name, id)| (name.as_str(), *id))
            .collect()
    }

    pub fn id_for_name(&self, name: &str) -> Option<u32> {
        self.name_to_id.get(name).copied()
    }

    pub fn resolve(&self, named: Option<&str>, direct_id: Option<u32>) -> Result<u32> {
        ensure!(
            named.is_none() || direct_id.is_none(),
            "select a model language by name or numeric ID, not both"
        );
        let id = if let Some(id) = direct_id {
            id
        } else if let Some(name) = named {
            self.id_for_name(name).with_context(|| {
                format!(
                    "unknown model language {name:?}; available languages: {}",
                    self.available_names().join(", ")
                )
            })?
        } else if self.num_languages == 1 {
            0
        } else {
            anyhow::bail!(
                "model language selection is required for this {}-language model; available languages: {}",
                self.num_languages,
                self.available_names().join(", ")
            )
        };
        ensure!(
            id < self.num_languages,
            "language ID {id} is out of range for {} languages",
            self.num_languages
        );
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MULTILINGUAL_FRAGMENT: &str = r#"{"en": 0, "fr-fr": 1, "pt-br": 2}"#;

    #[test]
    fn resolves_checkpoint_names_without_inferring_from_varieties() {
        let catalog = LanguageCatalog::from_json_str(MULTILINGUAL_FRAGMENT, 3).unwrap();

        assert_eq!(catalog.id_for_name("fr-fr"), Some(1));
        assert_eq!(
            catalog.entries(),
            vec![("en", 0), ("fr-fr", 1), ("pt-br", 2)]
        );
        assert_eq!(catalog.resolve(Some("pt-br"), None).unwrap(), 2);
        assert!(catalog.resolve(Some("pt-BR"), None).is_err());
    }

    #[test]
    fn rejects_unknown_ambiguous_and_out_of_range_selections() {
        let catalog = LanguageCatalog::from_json_str(MULTILINGUAL_FRAGMENT, 3).unwrap();

        assert!(catalog
            .resolve(None, None)
            .unwrap_err()
            .to_string()
            .contains("required"));
        assert!(catalog
            .resolve(Some("de"), None)
            .unwrap_err()
            .to_string()
            .contains("available languages"));
        assert!(catalog
            .resolve(None, Some(3))
            .unwrap_err()
            .to_string()
            .contains("out of range"));
        assert!(catalog.resolve(Some("en"), Some(0)).is_err());
    }

    #[test]
    fn a_single_language_checkpoint_may_omit_selection() {
        let catalog = LanguageCatalog::from_json_str(r#"{"es": 0}"#, 1).unwrap();

        assert_eq!(catalog.resolve(None, None).unwrap(), 0);
    }
}
