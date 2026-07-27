//! Shared tensor contract between native vocoders and the model-neutral trainer.

use anyhow::Result;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

use crate::AudioFeatureConfig;

/// Module-safe geometry for the complete differentiable mel objective.
#[derive(Module, Debug, Clone)]
pub struct VocoderMelLossConfig {
    fft_size: usize,
    win_length: usize,
    hop_length: usize,
    sample_rate: u32,
    num_mels: usize,
    mel_fmin: f32,
    mel_fmax: f32,
}

impl VocoderMelLossConfig {
    pub fn from_audio(config: &AudioFeatureConfig) -> Self {
        Self {
            fft_size: config.fft_size,
            win_length: config.win_length,
            hop_length: config.hop_length,
            sample_rate: config.sample_rate,
            num_mels: config.num_mels,
            mel_fmin: config.mel_fmin,
            mel_fmax: config.mel_fmax.unwrap_or(config.sample_rate as f32 / 2.0),
        }
    }

    pub(crate) fn audio_config(&self) -> AudioFeatureConfig {
        AudioFeatureConfig {
            fft_size: self.fft_size,
            win_length: self.win_length,
            hop_length: self.hop_length,
            sample_rate: self.sample_rate,
            preemphasis: 0.0,
            log_func: "np.log".into(),
            num_mels: self.num_mels,
            mel_fmin: self.mel_fmin,
            mel_fmax: Some(self.mel_fmax),
            spec_gain: 1.0,
            signal_norm: false,
            min_level_db: -100.0,
            ref_level_db: Some(20.0),
            symmetric_norm: true,
            max_norm: 4.0,
            clip_norm: true,
            stats_path: None,
            stats_sha256: None,
            do_amp_to_db_mel: true,
            stft_pad_mode: "reflect".into(),
            centered: true,
            stft_manual_padding: None,
        }
    }
}

/// Which parameter group a vocoder training step is intended to update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocoderTrainingPhase {
    Generator,
    Discriminator,
    Joint,
}

/// Shared adversarial update schedule for native vocoder trainers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VocoderAdversarialUpdateSchedule {
    /// Update generator and discriminator once for every batch.
    #[default]
    EveryBatch,
    /// Run a deterministic discriminator-then-generator cycle.
    ///
    /// `Cycle { discriminator_steps: 1, generator_steps: 1 }` yields
    /// discriminator, generator, discriminator, generator, ...
    ///
    /// `Cycle { discriminator_steps: 1, generator_steps: 2 }` yields
    /// discriminator, generator, generator, discriminator, ...
    Cycle {
        discriminator_steps: u64,
        generator_steps: u64,
    },
}

/// Per-step schedule metrics emitted by native vocoder trainers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VocoderTrainingProgress {
    pub global_step: u64,
    pub phase: VocoderTrainingPhase,
    pub generator_updates: u64,
    pub discriminator_updates: u64,
}

impl VocoderAdversarialUpdateSchedule {
    pub fn training_phase(&self, global_step: u64) -> VocoderTrainingPhase {
        self.progress(global_step).phase
    }

    pub fn progress(&self, global_step: u64) -> VocoderTrainingProgress {
        match self {
            Self::EveryBatch => VocoderTrainingProgress {
                global_step,
                phase: VocoderTrainingPhase::Joint,
                generator_updates: global_step + 1,
                discriminator_updates: global_step + 1,
            },
            Self::Cycle {
                discriminator_steps,
                generator_steps,
            } => {
                let discriminator_steps = (*discriminator_steps).max(1);
                let generator_steps = (*generator_steps).max(1);
                let cycle_len = discriminator_steps + generator_steps;
                let cycle_step = global_step % cycle_len;
                let completed_steps = global_step + 1;
                let completed_cycles = completed_steps / cycle_len;
                let remainder = completed_steps % cycle_len;
                let discriminator_updates =
                    completed_cycles * discriminator_steps + remainder.min(discriminator_steps);
                let generator_updates = completed_cycles * generator_steps
                    + remainder
                        .saturating_sub(discriminator_steps)
                        .min(generator_steps);

                VocoderTrainingProgress {
                    global_step,
                    phase: if cycle_step < discriminator_steps {
                        VocoderTrainingPhase::Discriminator
                    } else {
                        VocoderTrainingPhase::Generator
                    },
                    generator_updates,
                    discriminator_updates,
                }
            }
        }
    }
}

/// Model-neutral conditioning/waveform batch used by native Burn vocoder hooks.
#[derive(Debug, Clone)]
pub struct BurnVocoderTrainingBatch<B: Backend> {
    /// Frame-major mel conditioning: `[batch, frames, mel_bins]`.
    pub conditioning_mel: Tensor<B, 3>,
    /// Time-major waveform target: `[batch, channels, samples]`.
    pub target_waveform: Tensor<B, 3>,
}

/// Common outputs needed by vocoder loss and evaluation implementations.
#[derive(Debug)]
pub struct BurnVocoderTrainingOutput<B: Backend> {
    pub progress: VocoderTrainingProgress,
    pub phase: VocoderTrainingPhase,
    pub predicted_waveform: Tensor<B, 3>,
    /// Architecture-specific discriminator outputs, when the model has them.
    pub discriminator_outputs: Option<Vec<Tensor<B, 3>>>,
    /// Combined generator loss scalar for this step (present during generator phase).
    pub generator_loss: Option<Tensor<B, 1>>,
    /// Combined discriminator loss scalar for this step (present during discriminator phase).
    pub discriminator_loss: Option<Tensor<B, 1>>,
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
            VocoderAdversarialUpdateSchedule::Cycle {
                discriminator_steps: 1,
                generator_steps: 1,
            }
            .training_phase(global_step)
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
        assert_eq!(
            trainer.training_phase(0),
            VocoderTrainingPhase::Discriminator
        );
        assert_eq!(trainer.training_phase(1), VocoderTrainingPhase::Generator);
    }

    #[test]
    fn every_batch_schedule_updates_both_parameter_groups() {
        let progress = VocoderAdversarialUpdateSchedule::EveryBatch.progress(3);
        assert_eq!(progress.phase, VocoderTrainingPhase::Joint);
        assert_eq!(progress.generator_updates, 4);
        assert_eq!(progress.discriminator_updates, 4);
    }

    #[test]
    fn alternating_cycle_counts_both_parameter_groups() {
        let schedule = VocoderAdversarialUpdateSchedule::Cycle {
            discriminator_steps: 1,
            generator_steps: 1,
        };
        let expected = [
            VocoderTrainingPhase::Discriminator,
            VocoderTrainingPhase::Generator,
            VocoderTrainingPhase::Discriminator,
            VocoderTrainingPhase::Generator,
        ];
        for (step, phase) in expected.into_iter().enumerate() {
            assert_eq!(schedule.training_phase(step as u64), phase);
        }
        let progress = schedule.progress(3);
        assert_eq!(progress.generator_updates, 2);
        assert_eq!(progress.discriminator_updates, 2);
    }

    #[test]
    fn unequal_cycle_has_documented_phase_order() {
        let schedule = VocoderAdversarialUpdateSchedule::Cycle {
            discriminator_steps: 1,
            generator_steps: 2,
        };
        let expected = [
            VocoderTrainingPhase::Discriminator,
            VocoderTrainingPhase::Generator,
            VocoderTrainingPhase::Generator,
            VocoderTrainingPhase::Discriminator,
            VocoderTrainingPhase::Generator,
            VocoderTrainingPhase::Generator,
        ];
        for (step, phase) in expected.into_iter().enumerate() {
            assert_eq!(schedule.training_phase(step as u64), phase);
        }
        let progress = schedule.progress(5);
        assert_eq!(progress.generator_updates, 4);
        assert_eq!(progress.discriminator_updates, 2);
    }
}
