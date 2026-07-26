//! Shared tensor contract between native vocoders and the model-neutral trainer.

use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Which parameter group a vocoder training step is intended to update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocoderTrainingPhase {
    Generator,
    Discriminator,
    Joint,
}

/// Model-neutral conditioning/waveform batch used by native Burn vocoder hooks.
#[derive(Debug)]
pub struct BurnVocoderTrainingBatch<B: Backend> {
    /// Frame-major mel conditioning: `[batch, frames, mel_bins]`.
    pub conditioning_mel: Tensor<B, 3>,
    /// Time-major waveform target: `[batch, channels, samples]`.
    pub target_waveform: Tensor<B, 3>,
}

/// Common outputs needed by vocoder loss and evaluation implementations.
#[derive(Debug)]
pub struct BurnVocoderTrainingOutput<B: Backend> {
    pub phase: VocoderTrainingPhase,
    pub predicted_waveform: Tensor<B, 3>,
    /// Architecture-specific discriminator outputs, when the model has them.
    pub discriminator_outputs: Option<Vec<Tensor<B, 3>>>,
}

/// Extension point consumed by the model-neutral native training platform.
///
/// Inference adapters do not depend on this trait, so vocoder inference stays
/// independently shippable.
pub trait BurnVocoderTrainingHooks<B: Backend> {
    fn training_phase(&self, global_step: u64) -> VocoderTrainingPhase;

    fn training_forward(
        &self,
        batch: BurnVocoderTrainingBatch<B>,
        global_step: u64,
    ) -> Result<BurnVocoderTrainingOutput<B>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    struct DummyVocoderTrainer;

    impl BurnVocoderTrainingHooks<NdArray> for DummyVocoderTrainer {
        fn training_phase(&self, global_step: u64) -> VocoderTrainingPhase {
            if global_step % 2 == 0 {
                VocoderTrainingPhase::Generator
            } else {
                VocoderTrainingPhase::Discriminator
            }
        }

        fn training_forward(
            &self,
            _batch: BurnVocoderTrainingBatch<NdArray>,
            _global_step: u64,
        ) -> Result<BurnVocoderTrainingOutput<NdArray>> {
            unreachable!("test only validates the shared hook contract");
        }
    }

    #[test]
    fn exposes_phase_dispatch_without_model_switch() {
        let trainer = DummyVocoderTrainer;
        assert_eq!(trainer.training_phase(0), VocoderTrainingPhase::Generator);
        assert_eq!(
            trainer.training_phase(1),
            VocoderTrainingPhase::Discriminator
        );
    }
}
