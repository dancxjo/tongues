//! Versioned, durable training contract for native VITS runs.
//!
//! The neural graph lives in [`crate::burn_vits_training`]. This module owns
//! the stable, backend-neutral files around that graph: the recipe, dataset and
//! source provenance, resumable state, checkpoint naming, progress events, and
//! model-card data. Long-running callers can therefore expose progress without
//! coupling the library to a particular CLI renderer.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

pub const VITS_TRAINING_RECIPE_SCHEMA_VERSION: u32 = 1;
pub const VITS_TRAINING_STATE_SCHEMA_VERSION: u32 = 1;
pub const VITS_TRAINING_MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VitsTrainingBackend {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VitsOptimizerConfig {
    pub generator_learning_rate: f64,
    pub discriminator_learning_rate: f64,
    pub adam_beta1: f64,
    pub adam_beta2: f64,
    pub weight_decay: f64,
    pub gradient_clip_norm: Option<f64>,
}

impl Default for VitsOptimizerConfig {
    fn default() -> Self {
        Self {
            generator_learning_rate: 2.0e-4,
            discriminator_learning_rate: 2.0e-4,
            adam_beta1: 0.8,
            adam_beta2: 0.99,
            weight_decay: 0.0,
            gradient_clip_norm: Some(1_000.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VitsSchedulerConfig {
    /// Multiplicative learning-rate decay applied after every completed epoch.
    pub gamma: f64,
    /// Lower bound applied after epoch decay.
    pub minimum_learning_rate: f64,
}

impl Default for VitsSchedulerConfig {
    fn default() -> Self {
        Self {
            gamma: 0.999_875,
            minimum_learning_rate: 1.0e-6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VitsCheckpointPolicy {
    /// Save a recoverable step checkpoint after this many optimizer steps.
    /// Zero disables intra-epoch checkpoints.
    pub every_steps: u64,
    /// Retain this many completed epoch checkpoints. Zero retains all.
    pub keep_last_epochs: usize,
    /// Evaluation samples are written every this many steps. Zero means epoch end.
    pub sample_every_steps: u64,
}

impl Default for VitsCheckpointPolicy {
    fn default() -> Self {
        Self {
            every_steps: 1_000,
            keep_last_epochs: 5,
            sample_every_steps: 1_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VitsFreezeConfig {
    #[serde(default)]
    pub text_encoder: bool,
    #[serde(default)]
    pub posterior_encoder: bool,
    #[serde(default)]
    pub duration_predictor: bool,
    #[serde(default)]
    pub flow: bool,
    #[serde(default)]
    pub waveform_decoder: bool,
    #[serde(default)]
    pub speaker_embeddings: bool,
    #[serde(default)]
    pub language_embeddings: bool,
}

impl VitsFreezeConfig {
    pub fn none() -> Self {
        Self {
            text_encoder: false,
            posterior_encoder: false,
            duration_predictor: false,
            flow: false,
            waveform_decoder: false,
            speaker_embeddings: false,
            language_embeddings: false,
        }
    }
}

impl Default for VitsFreezeConfig {
    fn default() -> Self {
        Self::none()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VitsLossWeights {
    pub adversarial: f64,
    pub feature_matching: f64,
    pub mel: f64,
    pub duration: f64,
    pub kl: f64,
}

impl Default for VitsLossWeights {
    fn default() -> Self {
        Self {
            adversarial: 1.0,
            feature_matching: 2.0,
            mel: 45.0,
            duration: 1.0,
            kl: 1.0,
        }
    }
}

/// The intentionally supported first native VITS training slice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VitsSupportedSubset {
    pub stochastic_duration_predictor: bool,
    pub maximum_path_alignment: bool,
    pub multi_speaker: bool,
    pub language_conditioning: bool,
    pub reference_d_vectors: bool,
}

impl Default for VitsSupportedSubset {
    fn default() -> Self {
        Self {
            stochastic_duration_predictor: true,
            maximum_path_alignment: true,
            multi_speaker: true,
            language_conditioning: true,
            reference_d_vectors: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VitsTrainingRecipe {
    pub schema_version: u32,
    pub seed: u64,
    pub epochs: u64,
    pub batch_size: usize,
    pub segment_frames: usize,
    pub backend: VitsTrainingBackend,
    pub optimizer: VitsOptimizerConfig,
    pub scheduler: VitsSchedulerConfig,
    pub checkpoints: VitsCheckpointPolicy,
    pub freeze: VitsFreezeConfig,
    pub loss_weights: VitsLossWeights,
    pub supported_subset: VitsSupportedSubset,
}

impl Default for VitsTrainingRecipe {
    fn default() -> Self {
        Self {
            schema_version: VITS_TRAINING_RECIPE_SCHEMA_VERSION,
            seed: 42,
            epochs: 1_000,
            batch_size: 16,
            segment_frames: 32,
            backend: VitsTrainingBackend::Cuda,
            optimizer: VitsOptimizerConfig::default(),
            scheduler: VitsSchedulerConfig::default(),
            checkpoints: VitsCheckpointPolicy::default(),
            freeze: VitsFreezeConfig::default(),
            loss_weights: VitsLossWeights::default(),
            supported_subset: VitsSupportedSubset::default(),
        }
    }
}

impl VitsTrainingRecipe {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == VITS_TRAINING_RECIPE_SCHEMA_VERSION,
            "unsupported VITS recipe schema {}; expected {}",
            self.schema_version,
            VITS_TRAINING_RECIPE_SCHEMA_VERSION
        );
        ensure!(self.epochs > 0, "VITS epochs must be positive");
        ensure!(self.batch_size > 0, "VITS batch size must be positive");
        ensure!(
            self.segment_frames > 0,
            "VITS segment length must be positive"
        );
        for (name, value) in [
            (
                "generator learning rate",
                self.optimizer.generator_learning_rate,
            ),
            (
                "discriminator learning rate",
                self.optimizer.discriminator_learning_rate,
            ),
            ("Adam beta1", self.optimizer.adam_beta1),
            ("Adam beta2", self.optimizer.adam_beta2),
            ("scheduler gamma", self.scheduler.gamma),
        ] {
            ensure!(
                value.is_finite() && value > 0.0,
                "{name} must be finite and positive"
            );
        }
        ensure!(
            self.optimizer.adam_beta1 < 1.0 && self.optimizer.adam_beta2 < 1.0,
            "Adam beta values must be below one"
        );
        ensure!(
            self.scheduler.gamma <= 1.0,
            "scheduler gamma must not increase the learning rate"
        );
        ensure!(
            self.scheduler.minimum_learning_rate.is_finite()
                && self.scheduler.minimum_learning_rate >= 0.0,
            "minimum learning rate must be finite and non-negative"
        );
        for (name, value) in [
            ("adversarial", self.loss_weights.adversarial),
            ("feature matching", self.loss_weights.feature_matching),
            ("mel", self.loss_weights.mel),
            ("duration", self.loss_weights.duration),
            ("KL", self.loss_weights.kl),
        ] {
            ensure!(
                value.is_finite() && value >= 0.0,
                "{name} loss weight must be finite and non-negative"
            );
        }
        ensure!(
            self.supported_subset.stochastic_duration_predictor
                && self.supported_subset.maximum_path_alignment,
            "the native VITS milestone requires stochastic durations and maximum-path alignment"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VitsDatasetManifest {
    pub normalized_manifest: PathBuf,
    pub train_split: PathBuf,
    pub validation_split: PathBuf,
    pub test_split: PathBuf,
    pub feature_cache: PathBuf,
    pub sample_rate_hz: u32,
    pub audio_channels: u16,
    pub train_records: usize,
    pub validation_records: usize,
    pub test_records: usize,
    pub split_seed: u64,
    pub license: String,
    pub provenance: String,
}

impl VitsDatasetManifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.sample_rate_hz > 0,
            "dataset sample rate must be positive"
        );
        ensure!(
            self.audio_channels == 1,
            "native VITS training currently requires mono audio"
        );
        ensure!(self.train_records > 0, "VITS training split is empty");
        ensure!(
            self.validation_records > 0,
            "VITS validation split is empty"
        );
        ensure!(
            !self.license.trim().is_empty(),
            "dataset license must be recorded"
        );
        ensure!(
            !self.provenance.trim().is_empty(),
            "dataset provenance must be recorded"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VitsTrainingState {
    pub schema_version: u32,
    pub epoch: u64,
    pub global_step: u64,
    pub batch_in_epoch: usize,
    pub shuffle_seed: u64,
    pub generator_learning_rate: f64,
    pub discriminator_learning_rate: f64,
    pub best_validation_loss: Option<f64>,
    pub best_epoch: Option<u64>,
    pub last_checkpoint: PathBuf,
    pub generator_optimizer_checkpoint: PathBuf,
    pub discriminator_optimizer_checkpoint: PathBuf,
}

impl VitsTrainingState {
    pub fn initial(recipe: &VitsTrainingRecipe, layout: &VitsRunLayout) -> Self {
        Self {
            schema_version: VITS_TRAINING_STATE_SCHEMA_VERSION,
            epoch: 0,
            global_step: 0,
            batch_in_epoch: 0,
            shuffle_seed: recipe.seed,
            generator_learning_rate: recipe.optimizer.generator_learning_rate,
            discriminator_learning_rate: recipe.optimizer.discriminator_learning_rate,
            best_validation_loss: None,
            best_epoch: None,
            last_checkpoint: layout.latest_checkpoint(),
            generator_optimizer_checkpoint: layout.latest_generator_optimizer(),
            discriminator_optimizer_checkpoint: layout.latest_discriminator_optimizer(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == VITS_TRAINING_STATE_SCHEMA_VERSION,
            "unsupported VITS training-state schema {}; expected {}",
            self.schema_version,
            VITS_TRAINING_STATE_SCHEMA_VERSION
        );
        ensure!(
            self.generator_learning_rate.is_finite()
                && self.generator_learning_rate >= 0.0
                && self.discriminator_learning_rate.is_finite()
                && self.discriminator_learning_rate >= 0.0,
            "training-state learning rates must be finite and non-negative"
        );
        if let Some(loss) = self.best_validation_loss {
            ensure!(
                loss.is_finite() && loss >= 0.0,
                "best validation loss must be finite and non-negative"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VitsTrainingManifest {
    pub schema_version: u32,
    pub architecture: String,
    pub source_checkpoint: Option<PathBuf>,
    pub source_checkpoint_sha256: Option<String>,
    pub source_license: String,
    pub source_provenance: String,
    pub dataset: VitsDatasetManifest,
    pub target_metric: String,
    pub baseline_metric: Option<f64>,
    pub best_metric: Option<f64>,
}

impl VitsTrainingManifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == VITS_TRAINING_MANIFEST_SCHEMA_VERSION,
            "unsupported VITS manifest schema {}; expected {}",
            self.schema_version,
            VITS_TRAINING_MANIFEST_SCHEMA_VERSION
        );
        ensure!(
            self.architecture == "vits",
            "training manifest architecture must be `vits`"
        );
        ensure!(
            !self.source_license.trim().is_empty(),
            "source-model license must be recorded"
        );
        ensure!(
            !self.source_provenance.trim().is_empty(),
            "source-model provenance must be recorded"
        );
        ensure!(
            !self.target_metric.trim().is_empty(),
            "a fine-tuning target metric must be documented"
        );
        if self.source_checkpoint.is_some() {
            ensure!(
                self.baseline_metric.is_some(),
                "fine-tuning a source checkpoint requires a baseline target metric"
            );
        }
        if let Some(baseline) = self.baseline_metric {
            ensure!(
                baseline.is_finite() && baseline >= 0.0,
                "baseline metric must be finite and non-negative"
            );
        }
        if let Some(best) = self.best_metric {
            ensure!(
                best.is_finite() && best >= 0.0,
                "best metric must be finite and non-negative"
            );
            if let Some(baseline) = self.baseline_metric {
                ensure!(
                    best < baseline,
                    "completed fine-tune metric {best} did not improve baseline {baseline}"
                );
            }
        }
        self.dataset.validate()
    }

    /// Record a lower-is-better validation metric only when it improves the
    /// prior best (or the documented fine-tuning baseline).
    pub fn record_metric(&mut self, value: f64) -> Result<bool> {
        ensure!(
            value.is_finite() && value >= 0.0,
            "validation metric must be finite and non-negative"
        );
        let comparison = self.best_metric.or(self.baseline_metric);
        if comparison.is_some_and(|current| value >= current) {
            return Ok(false);
        }
        self.best_metric = Some(value);
        Ok(true)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VitsRunLayout {
    root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VitsCheckpointStaging {
    pub model: PathBuf,
    pub generator_optimizer: PathBuf,
    pub discriminator_optimizer: PathBuf,
}

impl VitsRunLayout {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn recipe(&self) -> PathBuf {
        self.root.join("recipe.json")
    }

    pub fn manifest(&self) -> PathBuf {
        self.root.join("training-manifest.json")
    }

    pub fn model_card(&self) -> PathBuf {
        self.root.join("README.md")
    }

    pub fn train_state(&self) -> PathBuf {
        self.root.join("train_state.json")
    }

    pub fn epoch_checkpoint(&self, epoch: u64) -> PathBuf {
        self.root.join(format!("model-epoch-{epoch}.safetensors"))
    }

    pub fn latest_checkpoint(&self) -> PathBuf {
        self.root.join("model-latest.safetensors")
    }

    pub fn best_checkpoint(&self) -> PathBuf {
        self.root.join("model.safetensors")
    }

    pub fn latest_generator_optimizer(&self) -> PathBuf {
        self.root.join("optim-generator-latest.bin")
    }

    pub fn epoch_generator_optimizer(&self, epoch: u64) -> PathBuf {
        self.root.join(format!("optim-generator-epoch-{epoch}.bin"))
    }

    pub fn latest_discriminator_optimizer(&self) -> PathBuf {
        self.root.join("optim-discriminator-latest.bin")
    }

    pub fn epoch_discriminator_optimizer(&self, epoch: u64) -> PathBuf {
        self.root
            .join(format!("optim-discriminator-epoch-{epoch}.bin"))
    }

    pub fn sample_dir(&self) -> PathBuf {
        self.root.join("samples")
    }

    /// `.part` paths that model and optimizer recorders must finish before
    /// [`publish_vits_checkpoint`] advances `train_state.json`.
    pub fn checkpoint_staging(&self) -> VitsCheckpointStaging {
        VitsCheckpointStaging {
            model: part_path(&self.latest_checkpoint()),
            generator_optimizer: part_path(&self.latest_generator_optimizer()),
            discriminator_optimizer: part_path(&self.latest_discriminator_optimizer()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VitsTrainingProgress {
    Initialize {
        output: PathBuf,
    },
    Write {
        path: PathBuf,
    },
    Resume {
        epoch: u64,
        global_step: u64,
        checkpoint: PathBuf,
    },
    Epoch {
        epoch: u64,
        epochs: u64,
    },
    Batch {
        epoch: u64,
        batch: usize,
        batches: usize,
        global_step: u64,
    },
    Checkpoint {
        epoch: u64,
        global_step: u64,
        path: PathBuf,
    },
    Sample {
        global_step: u64,
        path: PathBuf,
    },
    Complete {
        best_epoch: u64,
        best_model: PathBuf,
    },
}

pub fn initialize_vits_run_with_progress(
    output: impl AsRef<Path>,
    recipe: &VitsTrainingRecipe,
    manifest: &VitsTrainingManifest,
    mut progress: impl FnMut(VitsTrainingProgress),
) -> Result<(VitsRunLayout, VitsTrainingState)> {
    recipe.validate()?;
    manifest.validate()?;
    let layout = VitsRunLayout::new(output.as_ref());
    fs::create_dir_all(layout.root())
        .with_context(|| format!("creating VITS run {}", layout.root().display()))?;
    fs::create_dir_all(layout.sample_dir())
        .with_context(|| format!("creating {}", layout.sample_dir().display()))?;
    progress(VitsTrainingProgress::Initialize {
        output: layout.root().to_path_buf(),
    });

    if layout.recipe().exists() {
        let existing: VitsTrainingRecipe = read_json(&layout.recipe())?;
        ensure!(
            existing == *recipe,
            "cannot resume {} with a different VITS recipe",
            layout.root().display()
        );
    } else {
        write_json_atomic(&layout.recipe(), recipe)?;
        progress(VitsTrainingProgress::Write {
            path: layout.recipe(),
        });
    }
    if layout.manifest().exists() {
        let existing: VitsTrainingManifest = read_json(&layout.manifest())?;
        let mut expected = manifest.clone();
        expected.best_metric = existing.best_metric;
        ensure!(
            existing == expected,
            "cannot resume {} with different VITS model or dataset provenance",
            layout.root().display()
        );
    } else {
        write_vits_training_manifest(&layout, recipe, manifest)?;
        progress(VitsTrainingProgress::Write {
            path: layout.manifest(),
        });
        progress(VitsTrainingProgress::Write {
            path: layout.model_card(),
        });
    }

    let state = if layout.train_state().exists() {
        let source = fs::read_to_string(layout.train_state())
            .with_context(|| format!("reading {}", layout.train_state().display()))?;
        let state: VitsTrainingState = serde_json::from_str(&source)
            .with_context(|| format!("parsing {}", layout.train_state().display()))?;
        state.validate()?;
        progress(VitsTrainingProgress::Resume {
            epoch: state.epoch,
            global_step: state.global_step,
            checkpoint: state.last_checkpoint.clone(),
        });
        state
    } else {
        let state = VitsTrainingState::initial(recipe, &layout);
        write_json_atomic(&layout.train_state(), &state)?;
        progress(VitsTrainingProgress::Write {
            path: layout.train_state(),
        });
        state
    };
    Ok((layout, state))
}

pub fn write_vits_training_state(layout: &VitsRunLayout, state: &VitsTrainingState) -> Result<()> {
    state.validate()?;
    write_json_atomic(&layout.train_state(), state)
}

pub fn write_vits_training_manifest(
    layout: &VitsRunLayout,
    recipe: &VitsTrainingRecipe,
    manifest: &VitsTrainingManifest,
) -> Result<()> {
    recipe.validate()?;
    manifest.validate()?;
    write_json_atomic(&layout.manifest(), manifest)?;
    write_text_atomic(&layout.model_card(), &render_model_card(recipe, manifest))
}

/// Publish a complete model+optimizer checkpoint set and only then advance the
/// resumable state.
///
/// Callers first write all three paths from
/// [`VitsRunLayout::checkpoint_staging`]. A crash before the final atomic state
/// write resumes from the previous committed cursor instead of combining
/// mismatched model and optimizer state.
pub fn publish_vits_checkpoint(
    layout: &VitsRunLayout,
    state: &mut VitsTrainingState,
    is_best: bool,
    mut progress: impl FnMut(VitsTrainingProgress),
) -> Result<()> {
    state.validate()?;
    ensure!(state.epoch > 0, "cannot publish VITS epoch zero");
    let staging = layout.checkpoint_staging();
    for path in [
        &staging.model,
        &staging.generator_optimizer,
        &staging.discriminator_optimizer,
    ] {
        ensure!(
            path.is_file(),
            "VITS checkpoint staging file is missing: {}",
            path.display()
        );
        File::open(path)
            .with_context(|| format!("opening {}", path.display()))?
            .sync_all()
            .with_context(|| format!("syncing {}", path.display()))?;
    }

    let epoch_checkpoint = layout.epoch_checkpoint(state.epoch);
    let epoch_generator_optimizer = layout.epoch_generator_optimizer(state.epoch);
    let epoch_discriminator_optimizer = layout.epoch_discriminator_optimizer(state.epoch);
    fs::rename(&staging.model, &epoch_checkpoint).with_context(|| {
        format!(
            "publishing {} -> {}",
            staging.model.display(),
            epoch_checkpoint.display()
        )
    })?;
    fs::rename(&staging.generator_optimizer, &epoch_generator_optimizer).with_context(|| {
        format!(
            "publishing {} -> {}",
            staging.generator_optimizer.display(),
            epoch_generator_optimizer.display()
        )
    })?;
    fs::rename(
        &staging.discriminator_optimizer,
        &epoch_discriminator_optimizer,
    )
    .with_context(|| {
        format!(
            "publishing {} -> {}",
            staging.discriminator_optimizer.display(),
            epoch_discriminator_optimizer.display()
        )
    })?;

    state.last_checkpoint = epoch_checkpoint.clone();
    state.generator_optimizer_checkpoint = epoch_generator_optimizer.clone();
    state.discriminator_optimizer_checkpoint = epoch_discriminator_optimizer.clone();
    write_vits_training_state(layout, state)?;

    copy_atomic(&epoch_checkpoint, &layout.latest_checkpoint())?;
    copy_atomic(
        &epoch_generator_optimizer,
        &layout.latest_generator_optimizer(),
    )?;
    copy_atomic(
        &epoch_discriminator_optimizer,
        &layout.latest_discriminator_optimizer(),
    )?;
    if is_best {
        copy_atomic(&epoch_checkpoint, &layout.best_checkpoint())?;
        state.best_epoch = Some(state.epoch);
        write_vits_training_state(layout, state)?;
    }
    progress(VitsTrainingProgress::Checkpoint {
        epoch: state.epoch,
        global_step: state.global_step,
        path: epoch_checkpoint,
    });
    Ok(())
}

pub fn render_model_card(recipe: &VitsTrainingRecipe, manifest: &VitsTrainingManifest) -> String {
    let source = manifest
        .source_checkpoint
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "random initialization".to_string());
    let requirements = match recipe.backend {
        VitsTrainingBackend::Cpu => {
            "CPU is supported for fixtures and debugging; full VITS runs are not considered time-feasible."
        }
        VitsTrainingBackend::Cuda => {
            "CUDA is recommended for fine-tuning and required for time-feasible full training. CPU remains supported for fixtures and recovery."
        }
    };
    let baseline = manifest
        .baseline_metric
        .map(|value| value.to_string())
        .unwrap_or_else(|| "not recorded".to_string());
    let best = manifest
        .best_metric
        .map(|value| value.to_string())
        .unwrap_or_else(|| "not completed".to_string());
    format!(
        "# Native VITS training run\n\n\
         - Architecture: VITS\n\
         - Source checkpoint: {source}\n\
         - Source license: {license}\n\
         - Source provenance: {provenance}\n\
         - Dataset license: {dataset_license}\n\
         - Dataset provenance: {dataset_provenance}\n\
         - Target metric: {metric}\n\
         - Baseline metric: {baseline}\n\
         - Best metric: {best}\n\
         - Seed: {seed}\n\
         - Batch size: {batch_size}\n\
         - Segment frames: {segment_frames}\n\n\
         ## Compute\n\n{requirements}\n\n\
         ## Checkpoints\n\n\
         `train_state.json` identifies the exact epoch, batch cursor, shuffle seed, \
         generator/discriminator optimizer records, and `model-latest.safetensors`. \
         Completed epochs are `model-epoch-N.safetensors`; the best inference checkpoint \
         is `model.safetensors`. Files are published from `.part` paths only after a \
         successful flush.\n",
        license = manifest.source_license,
        provenance = manifest.source_provenance,
        dataset_license = manifest.dataset.license,
        dataset_provenance = manifest.dataset.provenance,
        metric = manifest.target_metric,
        seed = recipe.seed,
        batch_size = recipe.batch_size,
        segment_frames = recipe.segment_frames,
    )
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    write_bytes_atomic(path, &bytes)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let source = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&source).with_context(|| format!("parsing {}", path.display()))
}

fn write_text_atomic(path: &Path, value: &str) -> Result<()> {
    write_bytes_atomic(path, value.as_bytes())
}

fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let part = part_path(path);
    let file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(bytes)
        .with_context(|| format!("writing {}", part.display()))?;
    writer
        .flush()
        .with_context(|| format!("flushing {}", part.display()))?;
    writer
        .get_ref()
        .sync_all()
        .with_context(|| format!("syncing {}", part.display()))?;
    fs::rename(&part, path)
        .with_context(|| format!("publishing {} -> {}", part.display(), path.display()))
}

fn copy_atomic(source: &Path, destination: &Path) -> Result<()> {
    let part = part_path(destination);
    fs::copy(source, &part).with_context(|| {
        format!(
            "copying checkpoint {} -> {}",
            source.display(),
            part.display()
        )
    })?;
    File::open(&part)
        .with_context(|| format!("opening {}", part.display()))?
        .sync_all()
        .with_context(|| format!("syncing {}", part.display()))?;
    fs::rename(&part, destination)
        .with_context(|| format!("publishing {} -> {}", part.display(), destination.display()))
}

fn part_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    path.with_file_name(format!("{file_name}.part"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(root: &Path) -> VitsTrainingManifest {
        VitsTrainingManifest {
            schema_version: VITS_TRAINING_MANIFEST_SCHEMA_VERSION,
            architecture: "vits".into(),
            source_checkpoint: Some(root.join("source.pth")),
            source_checkpoint_sha256: Some("ab".repeat(32)),
            source_license: "MPL-2.0".into(),
            source_provenance: "checksum-pinned published fixture".into(),
            dataset: VitsDatasetManifest {
                normalized_manifest: root.join("normalized.jsonl"),
                train_split: root.join("train.jsonl"),
                validation_split: root.join("valid.jsonl"),
                test_split: root.join("test.jsonl"),
                feature_cache: root.join("features"),
                sample_rate_hz: 22_050,
                audio_channels: 1,
                train_records: 8,
                validation_records: 1,
                test_records: 1,
                split_seed: 42,
                license: "CC0-1.0".into(),
                provenance: "generated test fixture".into(),
            },
            target_metric: "validation mel L1".into(),
            baseline_metric: Some(1.0),
            best_metric: None,
        }
    }

    #[test]
    fn recipe_roundtrips_and_rejects_an_unknown_schema() {
        let recipe = VitsTrainingRecipe::default();
        recipe.validate().unwrap();
        let json = serde_json::to_string(&recipe).unwrap();
        let decoded: VitsTrainingRecipe = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, recipe);

        let mut future = recipe;
        future.schema_version += 1;
        assert!(future
            .validate()
            .unwrap_err()
            .to_string()
            .contains("schema"));
    }

    #[test]
    fn fine_tune_manifest_only_records_a_real_metric_improvement() {
        let root = tempfile::tempdir().unwrap();
        let mut run = manifest(root.path());
        assert!(!run.record_metric(1.0).unwrap());
        assert!(!run.record_metric(1.25).unwrap());
        assert!(run.record_metric(0.75).unwrap());
        assert_eq!(run.best_metric, Some(0.75));
        run.validate().unwrap();

        run.best_metric = Some(1.1);
        assert!(run
            .validate()
            .unwrap_err()
            .to_string()
            .contains("did not improve"));
    }

    #[test]
    fn resume_preserves_the_recorded_best_metric() {
        let root = tempfile::tempdir().unwrap();
        let recipe = VitsTrainingRecipe::default();
        let initial = manifest(root.path());
        let (layout, _) =
            initialize_vits_run_with_progress(root.path().join("run"), &recipe, &initial, |_| {})
                .unwrap();
        let mut completed = initial.clone();
        completed.record_metric(0.75).unwrap();
        write_vits_training_manifest(&layout, &recipe, &completed).unwrap();

        initialize_vits_run_with_progress(layout.root(), &recipe, &initial, |_| {}).unwrap();
        let stored: VitsTrainingManifest = read_json(&layout.manifest()).unwrap();
        assert_eq!(stored.best_metric, Some(0.75));
        assert!(std::fs::read_to_string(layout.model_card())
            .unwrap()
            .contains("Best metric: 0.75"));
    }

    #[test]
    fn initialization_is_atomic_and_resume_preserves_batch_cursor() {
        let root = tempfile::tempdir().unwrap();
        let run = root.path().join("run");
        let recipe = VitsTrainingRecipe::default();
        let manifest = manifest(root.path());
        let mut events = Vec::new();
        let (layout, mut state) =
            initialize_vits_run_with_progress(&run, &recipe, &manifest, |event| events.push(event))
                .unwrap();

        assert!(layout.recipe().exists());
        assert!(layout.manifest().exists());
        assert!(layout.model_card().exists());
        assert!(layout.train_state().exists());
        assert!(!run.join("recipe.json.part").exists());
        assert!(events
            .iter()
            .any(|event| matches!(event, VitsTrainingProgress::Initialize { .. })));

        state.epoch = 2;
        state.global_step = 17;
        state.batch_in_epoch = 3;
        state.shuffle_seed = 44;
        write_vits_training_state(&layout, &state).unwrap();

        let mut resumed = Vec::new();
        let (_, loaded) = initialize_vits_run_with_progress(&run, &recipe, &manifest, |event| {
            resumed.push(event)
        })
        .unwrap();
        assert_eq!(loaded, state);
        assert!(resumed.iter().any(|event| matches!(
            event,
            VitsTrainingProgress::Resume {
                epoch: 2,
                global_step: 17,
                ..
            }
        )));
    }

    #[test]
    fn model_card_states_compute_and_checkpoint_boundaries() {
        let root = tempfile::tempdir().unwrap();
        let card = render_model_card(&VitsTrainingRecipe::default(), &manifest(root.path()));
        assert!(card.contains("CUDA is recommended"));
        assert!(card.contains("CPU remains supported"));
        assert!(card.contains("train_state.json"));
        assert!(card.contains("model-epoch-N.safetensors"));
        assert!(card.contains(".part"));
    }

    #[test]
    fn checkpoint_publication_commits_model_optimizers_then_resume_state() {
        let root = tempfile::tempdir().unwrap();
        let recipe = VitsTrainingRecipe::default();
        let manifest = manifest(root.path());
        let (layout, mut state) =
            initialize_vits_run_with_progress(root.path().join("run"), &recipe, &manifest, |_| {})
                .unwrap();
        state.epoch = 1;
        state.global_step = 9;
        state.batch_in_epoch = 4;
        state.best_validation_loss = Some(0.5);
        let staging = layout.checkpoint_staging();
        std::fs::write(&staging.model, b"model-epoch-one").unwrap();
        std::fs::write(&staging.generator_optimizer, b"generator-optimizer").unwrap();
        std::fs::write(&staging.discriminator_optimizer, b"discriminator-optimizer").unwrap();
        let mut events = Vec::new();
        publish_vits_checkpoint(&layout, &mut state, true, |event| events.push(event)).unwrap();

        assert_eq!(
            std::fs::read(layout.latest_checkpoint()).unwrap(),
            b"model-epoch-one"
        );
        assert_eq!(
            std::fs::read(layout.epoch_checkpoint(1)).unwrap(),
            b"model-epoch-one"
        );
        assert_eq!(
            std::fs::read(layout.best_checkpoint()).unwrap(),
            b"model-epoch-one"
        );
        assert!(!staging.model.exists());
        assert_eq!(state.best_epoch, Some(1));
        assert!(events.iter().any(|event| matches!(
            event,
            VitsTrainingProgress::Checkpoint {
                epoch: 1,
                global_step: 9,
                ..
            }
        )));

        let (_, resumed) =
            initialize_vits_run_with_progress(layout.root(), &recipe, &manifest, |_| {}).unwrap();
        assert_eq!(resumed.epoch, 1);
        assert_eq!(resumed.global_step, 9);
        assert_eq!(resumed.batch_in_epoch, 4);
        assert_eq!(
            std::fs::read(resumed.generator_optimizer_checkpoint).unwrap(),
            b"generator-optimizer"
        );
        assert_eq!(
            std::fs::read(resumed.discriminator_optimizer_checkpoint).unwrap(),
            b"discriminator-optimizer"
        );
    }
}
