//! Versioned training recipe data for native vocoder training.
//!
//! A recipe is a stable, serialisable description of everything needed to
//! reproduce a training run: mel/audio contracts, generator topology,
//! discriminator configuration, loss weights, and optimiser hyperparameters.
//!
//! Recipes are *data*, not training-branch code.  The schema version allows
//! forward compatibility checks when older recipes are loaded.
//!
//! ## Stability guarantee
//!
//! Once a recipe schema version is published it will not have fields removed
//! or renamed.  New optional fields may be added in minor bumps.  A major
//! version change (e.g. `2`) indicates a breaking structural change.

use serde::{Deserialize, Serialize};

use crate::burn_hifigan::HifiganGeneratorConfig;
use crate::burn_melgan::{MelganGeneratorConfig, PqmfConfig};
use crate::burn_vocoder_losses::VocoderLossWeights;
use crate::burn_vocoder_training::VocoderAdversarialUpdateSchedule;

/// Identifies the schema version so saved recipes can be validated on load.
pub const RECIPE_SCHEMA_VERSION: u32 = 1;

/// Serialisable mel / audio boundary contract embedded in every recipe.
///
/// This mirrors the fields of [`crate::SpectrogramContract`] that are stable
/// across training runs.  The struct is deliberately flat (no nested type
/// references) so that recipes remain self-contained JSON files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecipeMelContract {
    /// Number of mel filter bins.
    pub mel_bins: usize,
    /// STFT hop size in audio samples.
    pub hop_size: usize,
    /// Audio sample rate in Hz.
    pub sample_rate_hz: u32,
    /// STFT window size in audio samples.
    pub win_length: usize,
    /// STFT FFT size.
    pub fft_size: usize,
    /// Lowest mel frequency (Hz).
    pub mel_fmin: f32,
    /// Highest mel frequency (Hz); `None` = Nyquist.
    pub mel_fmax: Option<f32>,
}

/// Shared optimiser / scheduler hyperparameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VocoderTrainingHyperparams {
    /// Initial generator learning rate (Adam default: 2e-4).
    pub learning_rate: f64,
    /// Initial discriminator learning rate.
    pub discriminator_learning_rate: f64,
    /// Adam β₁.
    #[serde(default = "default_beta1")]
    pub adam_beta1: f64,
    /// Adam β₂.
    #[serde(default = "default_beta2")]
    pub adam_beta2: f64,
    /// Training batch size.
    pub batch_size: usize,
    /// Maximum completed epochs.
    #[serde(default = "default_epochs")]
    pub epochs: u64,
    /// Maximum number of training steps (0 = unlimited).
    #[serde(default)]
    pub max_steps: u64,
    /// How many audio samples per training segment.
    pub segment_size: usize,
    /// Shared adversarial update schedule.
    ///
    /// - `EveryBatch` updates generator and discriminator once per batch.
    /// - `Cycle { discriminator_steps: 1, generator_steps: 1 }` alternates
    ///   discriminator and generator batches.
    /// - `Cycle { discriminator_steps: 1, generator_steps: 2 }` runs
    ///   discriminator, generator, generator, then repeats.
    #[serde(default)]
    pub adversarial_schedule: VocoderAdversarialUpdateSchedule,
    /// Legacy interval-based schedule retained for backward-compatible recipe
    /// loading. New recipes should leave this at `0` and use
    /// [`Self::adversarial_schedule`] instead.
    #[serde(default)]
    pub discriminator_update_interval: u64,
    /// Number of steps between checkpoint saves (0 = every epoch).
    #[serde(default)]
    pub checkpoint_interval_steps: u64,
    /// Number of steps between evaluation audio writes.
    #[serde(default = "default_eval_interval")]
    pub eval_interval_steps: u64,
    /// Multiplicative learning-rate decay after each completed epoch.
    #[serde(default = "default_scheduler_gamma")]
    pub scheduler_gamma: f64,
    /// Learning-rate floor after scheduler decay.
    #[serde(default = "default_minimum_learning_rate")]
    pub minimum_learning_rate: f64,
    /// Per-parameter gradient norm clipping. `None` disables clipping.
    #[serde(default = "default_gradient_clip_norm")]
    pub gradient_clip_norm: Option<f64>,
}

fn default_beta1() -> f64 {
    0.8
}

fn default_epochs() -> u64 {
    1_000
}

fn default_beta2() -> f64 {
    0.99
}

fn default_eval_interval() -> u64 {
    1000
}

fn default_scheduler_gamma() -> f64 {
    0.999
}

fn default_minimum_learning_rate() -> f64 {
    1.0e-6
}

fn default_gradient_clip_norm() -> Option<f64> {
    Some(1_000.0)
}

impl Default for VocoderTrainingHyperparams {
    fn default() -> Self {
        Self {
            learning_rate: 2e-4,
            discriminator_learning_rate: 2e-4,
            adam_beta1: 0.8,
            adam_beta2: 0.99,
            batch_size: 16,
            epochs: default_epochs(),
            max_steps: 0,
            segment_size: 8192,
            adversarial_schedule: VocoderAdversarialUpdateSchedule::EveryBatch,
            discriminator_update_interval: 0,
            checkpoint_interval_steps: 0,
            eval_interval_steps: 1000,
            scheduler_gamma: default_scheduler_gamma(),
            minimum_learning_rate: default_minimum_learning_rate(),
            gradient_clip_norm: default_gradient_clip_norm(),
        }
    }
}

impl VocoderTrainingHyperparams {
    /// Resolve the schedule that training code should apply.
    ///
    /// Legacy recipes may still carry `discriminator_update_interval`; a value
    /// of `1` is mapped to `EveryBatch` so the default no longer starves the
    /// generator, while larger values become a deterministic
    /// discriminator-then-generator cycle.
    pub fn resolved_adversarial_schedule(&self) -> VocoderAdversarialUpdateSchedule {
        if self.discriminator_update_interval == 0 {
            return self.adversarial_schedule;
        }

        if self.discriminator_update_interval == 1 {
            VocoderAdversarialUpdateSchedule::EveryBatch
        } else {
            VocoderAdversarialUpdateSchedule::Cycle {
                discriminator_steps: 1,
                generator_steps: self.discriminator_update_interval - 1,
            }
        }
    }
}

/// Versioned training recipe for HiFi-GAN.
///
/// A recipe file should be stored as a JSON file alongside training data so
/// that training runs are fully reproducible.
///
/// ```json
/// {
///   "schema_version": 1,
///   "mel_contract": { "mel_bins": 80, "hop_size": 256, ... },
///   "generator": { "in_channels": 80, "out_channels": 1, ... },
///   "hyperparams": { "learning_rate": 0.0002, ... },
///   "loss_weights": { "feature_matching": 10.0, ... }
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HifiganTrainingRecipe {
    /// Schema version – must equal [`RECIPE_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Exact mel / audio boundary this recipe was trained against.
    pub mel_contract: RecipeMelContract,
    /// HiFi-GAN generator configuration.
    pub generator: HifiganGeneratorConfig,
    /// Optimiser and scheduler hyperparameters.
    pub hyperparams: VocoderTrainingHyperparams,
    /// Weighted loss components.
    pub loss_weights: SerializableLossWeights,
    /// Optional human-readable note for provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl HifiganTrainingRecipe {
    /// Return a default recipe targeting the standard Coqui HiFi-GAN V1
    /// mel / audio contract (80 bins, 22 050 Hz, hop 256).
    pub fn coqui_hifigan_v1() -> Self {
        Self {
            schema_version: RECIPE_SCHEMA_VERSION,
            mel_contract: RecipeMelContract {
                mel_bins: 80,
                hop_size: 256,
                sample_rate_hz: 22_050,
                win_length: 1024,
                fft_size: 1024,
                mel_fmin: 0.0,
                mel_fmax: None,
            },
            generator: HifiganGeneratorConfig {
                in_channels: 80,
                out_channels: 1,
                resblock_type: "1".to_owned(),
                resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
                resblock_kernel_sizes: vec![3, 7, 11],
                upsample_kernel_sizes: vec![16, 16, 4, 4],
                upsample_initial_channel: 512,
                upsample_factors: vec![8, 8, 2, 2],
                inference_padding: 5,
                cond_channels: 0,
                conv_pre_weight_norm: true,
                conv_post_weight_norm: true,
                conv_post_bias: true,
            },
            hyperparams: VocoderTrainingHyperparams::default(),
            loss_weights: SerializableLossWeights::default(),
            description: Some("Coqui HiFi-GAN V1 – 80-bin / 22 050 Hz / hop-256 contract".into()),
        }
    }

    /// Load a recipe from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialise this recipe to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Validate that the schema version is supported.
    pub fn validate_schema(&self) -> Result<(), String> {
        if self.schema_version != RECIPE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported HiFi-GAN recipe schema version {}; expected {}",
                self.schema_version, RECIPE_SCHEMA_VERSION
            ));
        }
        Ok(())
    }
}

/// Versioned training recipe for MelGAN or MultiBand-MelGAN.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MelganTrainingRecipe {
    /// Schema version – must equal [`RECIPE_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Exact mel / audio boundary this recipe was trained against.
    pub mel_contract: RecipeMelContract,
    /// MelGAN generator configuration.
    pub generator: MelganGeneratorConfig,
    /// PQMF configuration for MultiBand-MelGAN. `None` = plain MelGAN.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pqmf: Option<PqmfConfig>,
    /// Optimiser and scheduler hyperparameters.
    pub hyperparams: VocoderTrainingHyperparams,
    /// Weighted loss components (default matches MelGAN paper).
    pub loss_weights: SerializableLossWeights,
    /// Optional human-readable note for provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl MelganTrainingRecipe {
    /// Whether this recipe targets a MultiBand-MelGAN generator.
    pub fn is_multiband(&self) -> bool {
        self.pqmf.is_some()
    }

    /// Return a default recipe targeting the standard Coqui MelGAN contract
    /// (80 bins, 22 050 Hz, hop 256).
    pub fn coqui_melgan() -> Self {
        Self {
            schema_version: RECIPE_SCHEMA_VERSION,
            mel_contract: RecipeMelContract {
                mel_bins: 80,
                hop_size: 256,
                sample_rate_hz: 22_050,
                win_length: 1024,
                fft_size: 1024,
                mel_fmin: 0.0,
                mel_fmax: None,
            },
            generator: MelganGeneratorConfig {
                in_channels: 80,
                out_channels: 1,
                projection_kernel_size: 7,
                base_channels: 512,
                upsample_factors: vec![8, 8, 2, 2],
                residual_kernel_size: 3,
                residual_blocks: 3,
                inference_padding: 2,
            },
            pqmf: None,
            hyperparams: VocoderTrainingHyperparams {
                learning_rate: 1e-4,
                discriminator_learning_rate: 1e-4,
                ..Default::default()
            },
            loss_weights: SerializableLossWeights::melgan(),
            description: Some("Coqui MelGAN – 80-bin / 22 050 Hz / hop-256 contract".into()),
        }
    }

    /// Return a default recipe targeting Coqui MultiBand-MelGAN (4 subbands).
    pub fn coqui_multiband_melgan() -> Self {
        Self {
            schema_version: RECIPE_SCHEMA_VERSION,
            mel_contract: RecipeMelContract {
                mel_bins: 80,
                hop_size: 256,
                sample_rate_hz: 22_050,
                win_length: 1024,
                fft_size: 1024,
                mel_fmin: 0.0,
                mel_fmax: None,
            },
            generator: MelganGeneratorConfig {
                in_channels: 80,
                out_channels: 4,
                projection_kernel_size: 7,
                base_channels: 384,
                upsample_factors: vec![8, 4, 2],
                residual_kernel_size: 3,
                residual_blocks: 4,
                inference_padding: 2,
            },
            pqmf: Some(PqmfConfig::default()),
            hyperparams: VocoderTrainingHyperparams {
                learning_rate: 1e-4,
                discriminator_learning_rate: 1e-4,
                ..Default::default()
            },
            loss_weights: SerializableLossWeights::melgan(),
            description: Some(
                "Coqui MultiBand-MelGAN 4-band – 80-bin / 22 050 Hz / hop-256 contract".into(),
            ),
        }
    }

    /// Load a recipe from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialise this recipe to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Validate that the schema version is supported.
    pub fn validate_schema(&self) -> Result<(), String> {
        if self.schema_version != RECIPE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported MelGAN recipe schema version {}; expected {}",
                self.schema_version, RECIPE_SCHEMA_VERSION
            ));
        }
        Ok(())
    }
}

/// Serialisable form of [`VocoderLossWeights`] for inclusion in recipe files.
///
/// Mirrors [`VocoderLossWeights`] exactly but derives `Serialize`/`Deserialize`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SerializableLossWeights {
    #[serde(default = "default_feature_matching")]
    pub feature_matching: f64,
    #[serde(default = "default_mel_spectrogram")]
    pub mel_spectrogram: f64,
    #[serde(default)]
    pub waveform_reconstruction: f64,
    #[serde(default = "default_adversarial_generator")]
    pub adversarial_generator: f64,
}

fn default_feature_matching() -> f64 {
    10.0
}

fn default_mel_spectrogram() -> f64 {
    45.0
}

fn default_adversarial_generator() -> f64 {
    1.0
}

impl Default for SerializableLossWeights {
    fn default() -> Self {
        Self {
            feature_matching: 10.0,
            mel_spectrogram: 45.0,
            waveform_reconstruction: 0.0,
            adversarial_generator: 1.0,
        }
    }
}

impl SerializableLossWeights {
    /// MelGAN paper weights (no mel loss, L1 reconstruction weight = 1).
    pub fn melgan() -> Self {
        Self {
            feature_matching: 10.0,
            mel_spectrogram: 0.0,
            waveform_reconstruction: 1.0,
            adversarial_generator: 1.0,
        }
    }

    /// Convert to the runtime [`VocoderLossWeights`] used by loss functions.
    pub fn to_runtime(&self) -> VocoderLossWeights {
        VocoderLossWeights {
            feature_matching: self.feature_matching,
            mel_spectrogram: self.mel_spectrogram,
            waveform_reconstruction: self.waveform_reconstruction,
            adversarial_generator: self.adversarial_generator,
        }
    }
}

/// Opaque state written to disk so that training can be resumed exactly.
///
/// The file is a small JSON sidecar next to the model checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VocoderTrainingState {
    /// Total number of optimizer steps completed across all resumptions.
    pub global_step: u64,
    /// Epoch index (0-based) at which this state was saved.
    pub epoch: u64,
    /// Best validation loss observed so far (used for best-model checkpointing).
    pub best_loss: f64,
    /// Human-readable timestamp of the last checkpoint (ISO 8601, UTC).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saved_at: Option<String>,
    /// Exact next batch within the current epoch.
    #[serde(default)]
    pub batch_in_epoch: usize,
    /// Current scheduler-controlled generator learning rate.
    #[serde(default)]
    pub generator_learning_rate: f64,
    /// Current scheduler-controlled discriminator learning rate.
    #[serde(default)]
    pub discriminator_learning_rate: f64,
}

impl VocoderTrainingState {
    /// Initial state for a fresh training run.
    pub fn initial() -> Self {
        Self {
            global_step: 0,
            epoch: 0,
            best_loss: f64::MAX,
            saved_at: None,
            batch_in_epoch: 0,
            generator_learning_rate: 0.0,
            discriminator_learning_rate: 0.0,
        }
    }

    /// Load state from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialise to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hifigan_recipe_roundtrips_through_json() {
        let recipe = HifiganTrainingRecipe::coqui_hifigan_v1();
        let json = recipe.to_json().expect("serialise");
        let reloaded = HifiganTrainingRecipe::from_json(&json).expect("deserialise");
        assert_eq!(recipe, reloaded);
    }

    #[test]
    fn melgan_recipe_roundtrips_through_json() {
        let recipe = MelganTrainingRecipe::coqui_melgan();
        let json = recipe.to_json().expect("serialise");
        let reloaded = MelganTrainingRecipe::from_json(&json).expect("deserialise");
        assert_eq!(recipe, reloaded);
    }

    #[test]
    fn multiband_melgan_recipe_roundtrips_through_json() {
        let recipe = MelganTrainingRecipe::coqui_multiband_melgan();
        assert!(recipe.is_multiband());
        let json = recipe.to_json().expect("serialise");
        let reloaded = MelganTrainingRecipe::from_json(&json).expect("deserialise");
        assert_eq!(recipe, reloaded);
    }

    #[test]
    fn recipe_schema_version_validation_rejects_future_version() {
        let mut recipe = HifiganTrainingRecipe::coqui_hifigan_v1();
        recipe.schema_version = 99;
        assert!(recipe.validate_schema().is_err());
    }

    #[test]
    fn training_state_roundtrips_through_json() {
        let state = VocoderTrainingState {
            global_step: 5000,
            epoch: 3,
            best_loss: 0.42,
            saved_at: Some("2025-01-01T00:00:00Z".into()),
            batch_in_epoch: 0,
            generator_learning_rate: 2.0e-4,
            discriminator_learning_rate: 2.0e-4,
        };
        let json = state.to_json().expect("serialise");
        let reloaded = VocoderTrainingState::from_json(&json).expect("deserialise");
        assert_eq!(state, reloaded);
    }

    #[test]
    fn initial_training_state_starts_at_zero() {
        let state = VocoderTrainingState::initial();
        assert_eq!(state.global_step, 0);
        assert_eq!(state.epoch, 0);
        assert_eq!(state.best_loss, f64::MAX);
    }

    #[test]
    fn serializable_loss_weights_convert_to_runtime() {
        let sw = SerializableLossWeights::melgan();
        let rt = sw.to_runtime();
        assert_eq!(rt.feature_matching, sw.feature_matching);
        assert_eq!(rt.mel_spectrogram, sw.mel_spectrogram);
        assert_eq!(rt.waveform_reconstruction, sw.waveform_reconstruction);
    }

    #[test]
    fn default_hyperparams_update_both_parameter_groups_per_batch() {
        let hyperparams = VocoderTrainingHyperparams::default();
        assert_eq!(
            hyperparams.resolved_adversarial_schedule(),
            VocoderAdversarialUpdateSchedule::EveryBatch
        );
    }

    #[test]
    fn legacy_discriminator_interval_maps_to_deterministic_cycle() {
        let hyperparams = VocoderTrainingHyperparams {
            discriminator_update_interval: 3,
            ..VocoderTrainingHyperparams::default()
        };
        assert_eq!(
            hyperparams.resolved_adversarial_schedule(),
            VocoderAdversarialUpdateSchedule::Cycle {
                discriminator_steps: 1,
                generator_steps: 2,
            }
        );
    }

    #[test]
    fn hifigan_recipe_mel_contract_matches_coqui_defaults() {
        let recipe = HifiganTrainingRecipe::coqui_hifigan_v1();
        assert_eq!(recipe.mel_contract.mel_bins, 80);
        assert_eq!(recipe.mel_contract.hop_size, 256);
        assert_eq!(recipe.mel_contract.sample_rate_hz, 22_050);
    }

    #[test]
    fn melgan_recipe_generator_upsample_product_matches_hop_size() {
        let recipe = MelganTrainingRecipe::coqui_melgan();
        let upsample: usize = recipe.generator.upsample_factors.iter().product();
        assert_eq!(
            upsample, recipe.mel_contract.hop_size,
            "MelGAN upsample product must equal hop_size"
        );
    }
}
