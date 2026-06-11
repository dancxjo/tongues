//! Shared neural-model artifact metadata.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use serde::{Deserialize, Serialize};

pub const ARTIFACT_MANIFEST_FILE: &str = "manifest.json";
pub const ARTIFACT_SCHEMA_VERSION: u32 = 1;

pub type FullPrecisionBinRecorder = BinFileRecorder<FullPrecisionSettings>;

pub fn make_recorder() -> FullPrecisionBinRecorder {
    FullPrecisionBinRecorder::new()
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrainState {
    pub current_epoch: usize,
    pub best_val_loss: f32,
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
        }
    }

    pub fn with_task(mut self, task: impl Into<String>) -> Self {
        self.task = Some(task.into());
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
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }
}
