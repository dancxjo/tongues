//! Shared tensor contract between native acoustic models and the model-neutral trainer.

use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// Which parameter group an acoustic training step is intended to update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcousticTrainingPhase {
    Alignment,
    Decoder,
    Acoustic,
    DurationPredictor,
    Joint,
}

/// Model-neutral text/mel batch used by native Burn acoustic training hooks.
#[derive(Debug)]
pub struct BurnAcousticTrainingBatch<B: Backend> {
    pub token_ids: Tensor<B, 2, Int>,
    pub token_lengths: Vec<usize>,
    /// Frame-major mel targets: `[batch, frames, mel_bins]`.
    pub target_mel: Tensor<B, 3>,
    pub mel_lengths: Vec<usize>,
}

/// Common outputs needed by acoustic loss and evaluation implementations.
#[derive(Debug)]
pub struct BurnAcousticTrainingOutput<B: Backend> {
    pub phase: AcousticTrainingPhase,
    pub predicted_mel: Option<Tensor<B, 3>>,
    /// Hard monotonic alignment in `[batch, frames, tokens]` layout.
    pub alignment: Tensor<B, 3>,
    pub predicted_duration_log: Option<Tensor<B, 2>>,
    pub aligned_duration_log: Tensor<B, 2>,
    /// Architecture-specific alignment likelihoods, when the model has them.
    pub alignment_log_prob: Option<Tensor<B, 3>>,
}

/// Extension point consumed by the model-neutral native training platform.
///
/// Inference adapters do not depend on this trait, so acoustic inference stays
/// independently shippable.
pub trait BurnAcousticTrainingHooks<B: Backend> {
    fn training_phase(&self, global_step: u64) -> AcousticTrainingPhase;

    fn training_forward(
        &self,
        batch: BurnAcousticTrainingBatch<B>,
        global_step: u64,
    ) -> Result<BurnAcousticTrainingOutput<B>>;
}
