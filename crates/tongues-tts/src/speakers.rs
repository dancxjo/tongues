use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use speaking::SpeakerId;

/// Stable model-declared speaker names and their learned embedding rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerCatalog {
    name_to_id: BTreeMap<String, u32>,
    num_speakers: u32,
}

impl SpeakerCatalog {
    pub fn from_json_str(source: &str, num_speakers: u32) -> Result<Self> {
        let name_to_id: BTreeMap<String, u32> =
            serde_json::from_str(source).context("failed to parse speaker name map")?;
        Self::new(name_to_id, num_speakers)
    }

    pub fn from_file(path: impl AsRef<Path>, num_speakers: u32) -> Result<Self> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read speaker map {}", path.display()))?;
        Self::from_json_str(&source, num_speakers)
            .with_context(|| format!("invalid speaker map {}", path.display()))
    }

    pub fn new(name_to_id: BTreeMap<String, u32>, num_speakers: u32) -> Result<Self> {
        ensure!(num_speakers > 0, "speaker count must be positive");
        ensure!(!name_to_id.is_empty(), "speaker name map must not be empty");
        ensure!(
            name_to_id.keys().all(|name| !name.is_empty()),
            "speaker name map contains an empty name"
        );
        ensure!(
            name_to_id.values().all(|id| *id < num_speakers),
            "speaker name map contains an ID outside 0..{num_speakers}"
        );
        let unique_ids = name_to_id.values().copied().collect::<BTreeSet<_>>();
        ensure!(
            unique_ids.len() == name_to_id.len(),
            "speaker name map assigns multiple names to one embedding row"
        );
        Ok(Self {
            name_to_id,
            num_speakers,
        })
    }

    pub fn num_speakers(&self) -> u32 {
        self.num_speakers
    }

    pub fn available_names(&self) -> Vec<&str> {
        self.entries().into_iter().map(|(name, _)| name).collect()
    }

    pub fn entries(&self) -> Vec<(&str, u32)> {
        let mut speakers = self.name_to_id.iter().collect::<Vec<_>>();
        speakers.sort_by_key(|(name, id)| (**id, name.as_str()));
        speakers
            .into_iter()
            .map(|(name, id)| (name.as_str(), *id))
            .collect()
    }

    pub fn id_for_name(&self, name: &str) -> Option<u32> {
        self.name_to_id.get(name).copied()
    }

    pub fn resolve(&self, named: Option<&SpeakerId>, direct_id: Option<u32>) -> Result<u32> {
        ensure!(
            named.is_none() || direct_id.is_none(),
            "select a speaker by name or numeric ID, not both"
        );
        let id = if let Some(id) = direct_id {
            id
        } else if let Some(SpeakerId(name)) = named {
            self.id_for_name(name).with_context(|| {
                format!(
                    "unknown speaker {name:?}; available speakers: {}",
                    self.available_names().join(", ")
                )
            })?
        } else if self.num_speakers == 1 {
            0
        } else {
            anyhow::bail!(
                "speaker selection is required for this {}-speaker model; available speakers: {}",
                self.num_speakers,
                self.available_names().join(", ")
            )
        };
        ensure!(
            id < self.num_speakers,
            "speaker ID {id} is out of range for {} speakers",
            self.num_speakers
        );
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VCTK_FRAGMENT: &str = r#"{"ED\n": 0, "p225": 1, "p226": 2}"#;

    #[test]
    fn resolves_model_declared_names_without_parsing_their_spelling() {
        let catalog = SpeakerCatalog::from_json_str(VCTK_FRAGMENT, 3).unwrap();

        assert_eq!(catalog.id_for_name("p225"), Some(1));
        assert_eq!(
            catalog.entries(),
            vec![("ED\n", 0), ("p225", 1), ("p226", 2)]
        );
        assert_eq!(
            catalog
                .resolve(Some(&SpeakerId("p226".into())), None)
                .unwrap(),
            2
        );
        assert_eq!(catalog.available_names(), vec!["ED\n", "p225", "p226"]);
    }

    #[test]
    fn rejects_unknown_ambiguous_and_out_of_range_selections() {
        let catalog = SpeakerCatalog::from_json_str(VCTK_FRAGMENT, 3).unwrap();

        assert!(catalog
            .resolve(None, None)
            .unwrap_err()
            .to_string()
            .contains("required"));
        assert!(catalog
            .resolve(Some(&SpeakerId("p999".into())), None)
            .unwrap_err()
            .to_string()
            .contains("available speakers"));
        assert!(catalog
            .resolve(None, Some(3))
            .unwrap_err()
            .to_string()
            .contains("out of range"));
    }

    #[test]
    fn loads_the_published_vctk_speaker_map_when_available() {
        let Some(path) = std::env::var_os("TONGUES_TEST_COQUI_VITS_SPEAKERS") else {
            return;
        };
        let catalog = SpeakerCatalog::from_file(path, 109).expect("VCTK speaker catalog");

        assert_eq!(catalog.available_names().len(), 109);
        assert_eq!(catalog.id_for_name("ED\n"), Some(0));
        assert_eq!(catalog.id_for_name("p225"), Some(1));
        assert_eq!(catalog.id_for_name("p376"), Some(108));
    }
}
