//! Shared neural-model artifact metadata.

use std::fs;
use std::io::{ErrorKind, Read};
use std::path::Path;

use anyhow::{Context, Result};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use serde::{Deserialize, Serialize};

pub const ARTIFACT_MANIFEST_FILE: &str = "manifest.json";
pub const ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const LEGACY_SCAFFOLD_MIGRATION_MESSAGE: &str =
    "legacy scaffold artifact is not runnable; remove it and train the model family to create a real checkpoint";

pub type FullPrecisionBinRecorder = BinFileRecorder<FullPrecisionSettings>;

pub fn make_recorder() -> FullPrecisionBinRecorder {
    FullPrecisionBinRecorder::new()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainState {
    pub current_epoch: usize,
    pub best_val_loss: f32,
    #[serde(default)]
    pub best_epoch: Option<usize>,
    #[serde(default)]
    pub best_exact_match: Option<f32>,
    #[serde(default = "default_early_stop_metric")]
    pub early_stop_metric: String,
}

fn default_early_stop_metric() -> String {
    "val_loss".to_string()
}

#[derive(Debug, Clone)]
pub struct TensorSeq2SeqBatch<B: Backend> {
    pub src_ids: Tensor<B, 2, Int>,
    pub tgt_in_ids: Tensor<B, 2, Int>,
    pub tgt_out_ids: Tensor<B, 2, Int>,
    pub src_pad_mask: Tensor<B, 2, Bool>,
    pub tgt_pad_mask: Tensor<B, 2, Bool>,
}

pub fn tensor_seq2seq_batch<B: Backend>(
    src_ids: Vec<Vec<i32>>,
    tgt_in_ids: Vec<Vec<i32>>,
    tgt_out_ids: Vec<Vec<i32>>,
    src_pad_mask: Vec<Vec<bool>>,
    tgt_pad_mask: Vec<Vec<bool>>,
    device: &B::Device,
) -> TensorSeq2SeqBatch<B> {
    let batch = src_ids.len();
    let src_len = src_ids.first().map(Vec::len).unwrap_or(0);
    let tgt_len = tgt_in_ids.first().map(Vec::len).unwrap_or(0);

    TensorSeq2SeqBatch {
        src_ids: Tensor::<B, 2, Int>::from_data(
            TensorData::new(
                src_ids.into_iter().flatten().collect::<Vec<_>>(),
                [batch, src_len],
            ),
            device,
        ),
        tgt_in_ids: Tensor::<B, 2, Int>::from_data(
            TensorData::new(
                tgt_in_ids.into_iter().flatten().collect::<Vec<_>>(),
                [batch, tgt_len],
            ),
            device,
        ),
        tgt_out_ids: Tensor::<B, 2, Int>::from_data(
            TensorData::new(
                tgt_out_ids.into_iter().flatten().collect::<Vec<_>>(),
                [batch, tgt_len],
            ),
            device,
        ),
        src_pad_mask: Tensor::<B, 2, Bool>::from_data(
            TensorData::new(
                src_pad_mask.into_iter().flatten().collect::<Vec<_>>(),
                [batch, src_len],
            ),
            device,
        ),
        tgt_pad_mask: Tensor::<B, 2, Bool>::from_data(
            TensorData::new(
                tgt_pad_mask.into_iter().flatten().collect::<Vec<_>>(),
                [batch, tgt_len],
            ),
            device,
        ),
    }
}

pub fn seq2seq_cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    pad_token_id: usize,
) -> Tensor<B, 1> {
    let [batch, seq_len, vocab] = logits.dims();
    let device = logits.device();
    let ce = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![pad_token_id]))
        .init::<B>(&device);

    ce.forward(
        logits.reshape([batch * seq_len, vocab]),
        targets.reshape([batch * seq_len]),
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArtifactManifest {
    pub schema_version: u32,
    pub family: String,
    pub architecture: String,
    pub created_by: String,
    pub data_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
    #[serde(default)]
    pub artifact_kind: ModelArtifactKind,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArtifactKind {
    #[default]
    TrainedModel,
    FamilyTemplate,
}

impl ModelArtifactManifest {
    pub fn new(
        family: impl Into<String>,
        architecture: impl Into<String>,
        data_id: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            family: family.into(),
            architecture: architecture.into(),
            created_by: "tongues".to_string(),
            data_id: data_id.into(),
            task: None,
            artifact_kind: ModelArtifactKind::TrainedModel,
        }
    }

    pub fn with_task(mut self, task: impl Into<String>) -> Self {
        self.task = Some(task.into());
        self
    }

    pub fn as_family_template(mut self) -> Self {
        self.artifact_kind = ModelArtifactKind::FamilyTemplate;
        self
    }
}

pub fn write_manifest(dir: &Path, manifest: &ModelArtifactManifest) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(ARTIFACT_MANIFEST_FILE);
    fs::write(&path, serde_json::to_string_pretty(manifest)?)
        .with_context(|| format!("writing {}", path.display()))
}

pub fn read_manifest(path: &Path) -> Result<ModelArtifactManifest> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let manifest: ModelArtifactManifest =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    anyhow::ensure!(
        manifest.artifact_kind == ModelArtifactKind::TrainedModel,
        "model artifact at {} is a non-runnable family template; implement and train the family before loading it",
        path.display()
    );
    anyhow::ensure!(
        !manifest
            .architecture
            .to_ascii_lowercase()
            .contains("scaffold"),
        "{}: {}",
        path.display(),
        LEGACY_SCAFFOLD_MIGRATION_MESSAGE
    );
    reject_legacy_scaffold_marker(path)?;
    Ok(manifest)
}

fn reject_legacy_scaffold_marker(manifest_path: &Path) -> Result<()> {
    let Some(dir) = manifest_path.parent() else {
        return Ok(());
    };
    let model_path = dir.join("model.bin");
    let mut file = match fs::File::open(&model_path) {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| format!("opening {}", model_path.display()))
        }
    };
    let mut bytes = [0_u8; 256];
    let bytes_read = file
        .read(&mut bytes)
        .with_context(|| format!("reading {}", model_path.display()))?;
    let marker = String::from_utf8_lossy(&bytes[..bytes_read]).to_ascii_lowercase();
    anyhow::ensure!(
        !marker.contains("scaffold"),
        "{}: {}",
        model_path.display(),
        LEGACY_SCAFFOLD_MIGRATION_MESSAGE
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_state_defaults_new_diagnostic_fields() {
        let state: TrainState =
            serde_json::from_str(r#"{"current_epoch":11,"best_val_loss":0.0309}"#).unwrap();

        assert_eq!(state.current_epoch, 11);
        assert_eq!(state.best_val_loss, 0.0309);
        assert_eq!(state.best_epoch, None);
        assert_eq!(state.best_exact_match, None);
        assert_eq!(state.early_stop_metric, "val_loss");
    }

    #[test]
    fn serializes_expected_contract() {
        let manifest = ModelArtifactManifest::new("g2p2g", "seq2seq-transformer", "openepd-v0")
            .with_task("both");
        let json = serde_json::to_value(&manifest).unwrap();

        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["family"], "g2p2g");
        assert_eq!(json["architecture"], "seq2seq-transformer");
        assert_eq!(json["created_by"], "tongues");
        assert_eq!(json["data_id"], "openepd-v0");
        assert_eq!(json["task"], "both");
        assert_eq!(json["artifact_kind"], "trained_model");
    }

    #[test]
    fn reads_pre_artifact_kind_manifests_as_trained_models() {
        let manifest: ModelArtifactManifest = serde_json::from_str(
            r#"{
                "schema_version": 1,
                "family": "g2p2g",
                "architecture": "seq2seq-transformer",
                "created_by": "tongues",
                "data_id": "openepd-v0"
            }"#,
        )
        .unwrap();

        assert_eq!(manifest.artifact_kind, ModelArtifactKind::TrainedModel);
    }

    #[test]
    fn rejects_legacy_scaffold_marker_bytes_with_migration_message() {
        let dir = std::env::temp_dir().join(format!(
            "tongues-neural-scaffold-rejection-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        write_manifest(
            &dir,
            &ModelArtifactManifest::new(
                "sentence-parser",
                "seq2seq-transformer",
                "sentence-parser-v0",
            ),
        )
        .unwrap();
        fs::write(dir.join("model.bin"), b"sentence-parser-scaffold\n").unwrap();

        let error = read_manifest(&dir.join(ARTIFACT_MANIFEST_FILE)).unwrap_err();
        assert!(error
            .to_string()
            .contains(LEGACY_SCAFFOLD_MIGRATION_MESSAGE));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_non_runnable_family_templates() {
        let dir = std::env::temp_dir().join(format!(
            "tongues-neural-family-template-rejection-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        write_manifest(
            &dir,
            &ModelArtifactManifest::new(
                "allophone-realizer",
                "unimplemented-family-template",
                "v0",
            )
            .as_family_template(),
        )
        .unwrap();

        let error = read_manifest(&dir.join(ARTIFACT_MANIFEST_FILE)).unwrap_err();
        assert!(error.to_string().contains("non-runnable family template"));
        let _ = fs::remove_dir_all(dir);
    }
}
