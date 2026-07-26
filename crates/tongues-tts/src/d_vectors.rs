use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;

use crate::{ConditioningEmbedding, ConditioningKind, EmbeddingContract};

pub const COQUI_RESNET_SPEAKER_EMBEDDING_SPACE: &str = "coqui-resnet-speaker-encoder-0cf3265a-v1";

#[derive(Debug, Clone, Deserialize)]
struct SerializedDVector {
    name: String,
    embedding: Vec<f32>,
}

/// Checkpoint-declared speaker embeddings keyed by clip and speaker name.
///
/// The original clip identifiers remain available for exact low-level
/// selection. Speaker-name selection averages all declared enrollment clips
/// and normalizes the result before it crosses the conditioning boundary.
#[derive(Debug, Clone)]
pub struct DVectorCatalog {
    clips: BTreeMap<String, SerializedDVector>,
    speaker_clips: BTreeMap<String, Vec<String>>,
    contract: EmbeddingContract,
}

impl DVectorCatalog {
    pub fn from_json_str(
        source: &str,
        dimensions: usize,
        space: impl Into<String>,
    ) -> Result<Self> {
        ensure!(dimensions > 0, "d-vector dimensions must be positive");
        let mut clips: BTreeMap<String, SerializedDVector> =
            serde_json::from_str(source).context("failed to parse d-vector catalog")?;
        ensure!(!clips.is_empty(), "d-vector catalog must not be empty");
        let mut speaker_clips = BTreeMap::<String, Vec<String>>::new();
        for (clip, record) in &mut clips {
            ensure!(
                !clip.trim().is_empty(),
                "d-vector catalog contains an empty clip ID"
            );
            record.name = record.name.trim().to_string();
            ensure!(
                !record.name.is_empty(),
                "d-vector clip {clip:?} has an empty speaker name"
            );
            ensure!(
                record.embedding.len() == dimensions,
                "d-vector clip {clip:?} has {} values; expected {dimensions}",
                record.embedding.len()
            );
            ensure!(
                record.embedding.iter().all(|value| value.is_finite()),
                "d-vector clip {clip:?} contains non-finite values"
            );
            speaker_clips
                .entry(record.name.clone())
                .or_default()
                .push(clip.clone());
        }
        for clip_ids in speaker_clips.values_mut() {
            clip_ids.sort();
        }
        Ok(Self {
            clips,
            speaker_clips,
            contract: EmbeddingContract {
                kind: ConditioningKind::Speaker,
                space: space.into(),
                dimensions,
                l2_normalized: true,
            },
        })
    }

    pub fn from_file(
        path: impl AsRef<Path>,
        dimensions: usize,
        space: impl Into<String>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read d-vector catalog {}", path.display()))?;
        Self::from_json_str(&source, dimensions, space)
            .with_context(|| format!("invalid d-vector catalog {}", path.display()))
    }

    pub fn contract(&self) -> &EmbeddingContract {
        &self.contract
    }

    pub fn speaker_names(&self) -> Vec<&str> {
        self.speaker_clips.keys().map(String::as_str).collect()
    }

    pub fn clip_ids(&self) -> Vec<&str> {
        self.clips.keys().map(String::as_str).collect()
    }

    pub fn embedding_for_clip(&self, clip: &str) -> Result<ConditioningEmbedding> {
        let record = self.clips.get(clip).with_context(|| {
            format!(
                "unknown d-vector clip {clip:?}; available clips include: {}",
                summarize(self.clips.keys().map(String::as_str))
            )
        })?;
        self.finish(record.embedding.clone())
    }

    pub fn embedding_for_speaker(&self, speaker: &str) -> Result<ConditioningEmbedding> {
        let clips = self.speaker_clips.get(speaker).with_context(|| {
            format!(
                "unknown d-vector speaker {speaker:?}; available speakers: {}",
                summarize(self.speaker_clips.keys().map(String::as_str))
            )
        })?;
        let mut values = vec![0.0; self.contract.dimensions];
        for clip in clips {
            for (output, value) in values.iter_mut().zip(&self.clips[clip].embedding) {
                *output += *value;
            }
        }
        let divisor = clips.len() as f32;
        for value in &mut values {
            *value /= divisor;
        }
        self.finish(values)
    }

    /// Resolve an exact clip ID or stable speaker name without parsing either
    /// namespace. An identifier present in both namespaces is rejected.
    pub fn resolve(&self, selection: &str) -> Result<ConditioningEmbedding> {
        match (
            self.clips.contains_key(selection),
            self.speaker_clips.contains_key(selection),
        ) {
            (true, false) => self.embedding_for_clip(selection),
            (false, true) => self.embedding_for_speaker(selection),
            (true, true) => bail!(
                "d-vector selection {selection:?} is ambiguous between a clip and speaker name"
            ),
            (false, false) => bail!(
                "unknown d-vector selection {selection:?}; available speakers: {}",
                summarize(self.speaker_clips.keys().map(String::as_str))
            ),
        }
    }

    fn finish(&self, mut values: Vec<f32>) -> Result<ConditioningEmbedding> {
        l2_normalize(&mut values)?;
        let embedding = ConditioningEmbedding {
            contract: self.contract.clone(),
            values,
        };
        embedding.validate()?;
        Ok(embedding)
    }
}

pub fn l2_normalize(values: &mut [f32]) -> Result<()> {
    ensure!(!values.is_empty(), "cannot normalize an empty embedding");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "embedding contains non-finite values"
    );
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    ensure!(
        norm.is_finite() && norm > f32::EPSILON,
        "speaker embedding has zero or invalid L2 norm"
    );
    for value in values {
        *value /= norm;
    }
    Ok(())
}

fn summarize<'a>(values: impl Iterator<Item = &'a str>) -> String {
    let values = values.collect::<BTreeSet<_>>();
    let shown = values
        .iter()
        .take(12)
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    if values.len() > 12 {
        format!("{shown}, ... ({} total)", values.len())
    } else {
        shown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"{
        "clip-a.wav": {"name": "alice", "embedding": [3.0, 0.0, 0.0]},
        "clip-b.wav": {"name": "alice", "embedding": [0.0, 4.0, 0.0]},
        "clip-c.wav": {"name": "bob\n", "embedding": [0.0, 0.0, 2.0]}
    }"#;

    #[test]
    fn resolves_clips_and_enrolled_speakers_with_normalized_embeddings() {
        let catalog = DVectorCatalog::from_json_str(FIXTURE, 3, "fixture").unwrap();
        assert_eq!(catalog.speaker_names(), vec!["alice", "bob"]);
        assert_eq!(
            catalog.embedding_for_clip("clip-a.wav").unwrap().values,
            vec![1.0, 0.0, 0.0]
        );
        let enrolled = catalog.embedding_for_speaker("alice").unwrap();
        assert!((enrolled.values.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(catalog.resolve("bob").unwrap().values, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn rejects_shape_and_unknown_selection_errors_with_available_names() {
        assert!(DVectorCatalog::from_json_str(FIXTURE, 4, "fixture").is_err());
        let catalog = DVectorCatalog::from_json_str(FIXTURE, 3, "fixture").unwrap();
        let error = catalog.resolve("charlie").unwrap_err().to_string();
        assert!(error.contains("available speakers"));
        assert!(error.contains("alice"));
    }
}
