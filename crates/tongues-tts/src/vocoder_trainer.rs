//! Executable native HiFi-GAN, MelGAN, and MultiBand-MelGAN runner.

use std::cmp::Reverse;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use burn::module::{AutodiffModule, Module};
use burn::optim::{
    grad_clipping::GradientClippingConfig, AdamWConfig, GradientsParams, Optimizer,
};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Tensor, TensorData};
use burn_store::{BurnToPyTorchAdapter, ModuleSnapshot, SafetensorsStore};
use serde::{Deserialize, Serialize};
use tongues_data::speech_corpus::{
    feature_cache_path, CachedSpeechFeatures, SpeechRecord,
};

use crate::{
    waveform_reconstruction_loss, AudioFeatureConfig,
    BurnVocoderTrainingBatch, BurnVocoderTrainingHooks, HifiganBundleConfig,
    HifiganTrainer, HifiganTrainingRecipe, MelganBundleConfig, MelganTrainer,
    MelganTrainingRecipe, MelganVariant, MultiPeriodDiscriminator, MultiScaleDiscriminator,
    MultibandMelganTrainer, VocoderTrainingPhase,
    VocoderTrainingState,
};

type BinaryRecorder = BinFileRecorder<FullPrecisionSettings>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeVocoderKind {
    Hifigan,
    Melgan,
    MultibandMelgan,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "architecture", content = "recipe", rename_all = "kebab-case")]
pub enum NativeVocoderRecipe {
    Hifigan(HifiganTrainingRecipe),
    Melgan(MelganTrainingRecipe),
    MultibandMelgan(MelganTrainingRecipe),
}

impl NativeVocoderRecipe {
    pub fn kind(&self) -> NativeVocoderKind {
        match self {
            Self::Hifigan(_) => NativeVocoderKind::Hifigan,
            Self::Melgan(_) => NativeVocoderKind::Melgan,
            Self::MultibandMelgan(_) => NativeVocoderKind::MultibandMelgan,
        }
    }

    pub fn mel_contract(&self) -> &crate::RecipeMelContract {
        match self {
            Self::Hifigan(recipe) => &recipe.mel_contract,
            Self::Melgan(recipe) | Self::MultibandMelgan(recipe) => &recipe.mel_contract,
        }
    }

    pub fn hyperparams(&self) -> &crate::VocoderTrainingHyperparams {
        match self {
            Self::Hifigan(recipe) => &recipe.hyperparams,
            Self::Melgan(recipe) | Self::MultibandMelgan(recipe) => &recipe.hyperparams,
        }
    }

    pub fn loss_weights(&self) -> crate::VocoderLossWeights {
        match self {
            Self::Hifigan(recipe) => recipe.loss_weights.to_runtime(),
            Self::Melgan(recipe) | Self::MultibandMelgan(recipe) => {
                recipe.loss_weights.to_runtime()
            }
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Hifigan(recipe) => {
                recipe.validate_schema().map_err(anyhow::Error::msg)?;
                recipe.generator.validate().map_err(anyhow::Error::new)?;
                ensure!(
                    recipe.generator.upsample_factor() == recipe.mel_contract.hop_size,
                    "HiFi-GAN upsample factor does not match mel hop"
                );
            }
            Self::Melgan(recipe) => {
                recipe.validate_schema().map_err(anyhow::Error::msg)?;
                ensure!(!recipe.is_multiband(), "plain MelGAN recipe declares PQMF");
                recipe.generator.validate().map_err(anyhow::Error::new)?;
                ensure!(
                    recipe.generator.upsample_factor() == recipe.mel_contract.hop_size,
                    "MelGAN upsample factor does not match mel hop"
                );
            }
            Self::MultibandMelgan(recipe) => {
                recipe.validate_schema().map_err(anyhow::Error::msg)?;
                let pqmf = recipe
                    .pqmf
                    .as_ref()
                    .context("MultiBand-MelGAN recipe requires PQMF")?;
                recipe.generator.validate().map_err(anyhow::Error::new)?;
                ensure!(
                    recipe.generator.upsample_factor() * pqmf.bands
                        == recipe.mel_contract.hop_size,
                    "MultiBand-MelGAN output factor does not match mel hop"
                );
            }
        }
        let hyper = self.hyperparams();
        ensure!(
            hyper.epochs > 0
                && hyper.batch_size > 0
                && hyper.segment_size > 0
                && hyper.learning_rate.is_finite()
                && hyper.learning_rate > 0.0
                && hyper.discriminator_learning_rate.is_finite()
                && hyper.discriminator_learning_rate > 0.0
                && hyper.scheduler_gamma.is_finite()
                && (0.0..=1.0).contains(&hyper.scheduler_gamma),
            "invalid native vocoder training hyperparameters"
        );
        Ok(())
    }

    pub fn audio_config(&self) -> AudioFeatureConfig {
        let mel = self.mel_contract();
        AudioFeatureConfig {
            fft_size: mel.fft_size,
            win_length: mel.win_length,
            hop_length: mel.hop_size,
            sample_rate: mel.sample_rate_hz,
            preemphasis: 0.0,
            log_func: "np.log".into(),
            num_mels: mel.mel_bins,
            mel_fmin: mel.mel_fmin,
            mel_fmax: mel.mel_fmax,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VocoderPreparedExample {
    pub record_id: String,
    /// Frame-major conditioning `[frames][mel_bins]`.
    pub mel: Vec<Vec<f32>>,
    pub waveform: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VocoderEvaluationReport {
    pub examples: usize,
    pub loss: f64,
    pub spectral_l1: f64,
    pub waveform_l1: f64,
    pub finite_audio: bool,
    pub generated_samples: usize,
    pub realtime_factor: f64,
    /// Deterministic parameter storage estimate (f32 bytes), not process RSS.
    pub parameter_memory_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VocoderTrainingReport {
    pub start_epoch: u64,
    pub start_batch: usize,
    pub end_epoch: u64,
    pub end_batch: usize,
    pub global_step: u64,
    pub best_loss: f64,
    pub interrupted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VocoderTrainingProgress {
    Resume {
        epoch: u64,
        batch: usize,
        global_step: u64,
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
        phase: VocoderTrainingPhase,
    },
    Checkpoint {
        global_step: u64,
        path: PathBuf,
    },
    Sample {
        global_step: u64,
        path: PathBuf,
    },
    Complete {
        best_loss_micros: u64,
        path: PathBuf,
    },
}

#[derive(Module, Debug)]
enum VocoderModel<B: Backend> {
    Hifigan(HifiganTrainer<B>),
    Melgan(MelganTrainer<B>),
    MultibandMelgan(MultibandMelganTrainer<B>),
}

impl<B: Backend> VocoderModel<B> {
    fn training_forward(
        &self,
        batch: BurnVocoderTrainingBatch<B>,
        global_step: u64,
    ) -> Result<crate::BurnVocoderTrainingOutput<B>> {
        match self {
            Self::Hifigan(model) => model.training_forward(batch, global_step),
            Self::Melgan(model) => model.training_forward(batch, global_step),
            Self::MultibandMelgan(model) => model.training_forward(batch, global_step),
        }
    }

    fn no_grad_discriminators(mut self) -> Self {
        match &mut self {
            Self::Hifigan(model) => {
                model.mpd = model.mpd.clone().no_grad();
                model.msd = model.msd.clone().no_grad();
            }
            Self::Melgan(model) => model.msd = model.msd.clone().no_grad(),
            Self::MultibandMelgan(model) => model.msd = model.msd.clone().no_grad(),
        }
        self
    }

    fn generate(&self, mel: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        match self {
            Self::Hifigan(model) => model
                .generator
                .forward(mel.swap_dims(1, 2), None)
                .map_err(anyhow::Error::new),
            Self::Melgan(model) => model
                .generator
                .forward(mel.swap_dims(1, 2))
                .map_err(anyhow::Error::new),
            Self::MultibandMelgan(model) => model
                .generator
                .inference(mel.swap_dims(1, 2))
                .map_err(anyhow::Error::new),
        }
    }
}

pub fn load_vocoder_examples(
    split_path: impl AsRef<Path>,
    feature_cache: impl AsRef<Path>,
    sample_rate_hz: u32,
) -> Result<Vec<VocoderPreparedExample>> {
    let split_path = split_path.as_ref();
    let feature_cache = feature_cache.as_ref();
    let base = split_path.parent().unwrap_or_else(|| Path::new("."));
    let reader = BufReader::new(File::open(split_path)?);
    let mut examples = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: SpeechRecord = serde_json::from_str(&line).with_context(|| {
            format!("parsing {} line {}", split_path.display(), index + 1)
        })?;
        let cache_path = feature_cache_path(feature_cache, &record.id);
        let cached: CachedSpeechFeatures =
            serde_json::from_reader(File::open(&cache_path).with_context(|| {
                format!("opening vocoder feature cache {}", cache_path.display())
            })?)?;
        let audio_path = if record.audio_path.is_absolute() {
            record.audio_path
        } else {
            base.join(record.audio_path)
        };
        let waveform = tongues_audio::read_wav(&audio_path)
            .map_err(anyhow::Error::from)?
            .convert_channels(1)
            .map_err(anyhow::Error::from)?
            .resample_linear(sample_rate_hz)
            .map_err(anyhow::Error::from)?
            .samples;
        examples.push(VocoderPreparedExample {
            record_id: record.id,
            mel: cached.acoustic_features,
            waveform,
        });
    }
    ensure!(
        !examples.is_empty(),
        "empty vocoder split {}",
        split_path.display()
    );
    Ok(examples)
}

#[allow(clippy::too_many_arguments)]
pub fn train_vocoder<B: AutodiffBackend>(
    recipe: &NativeVocoderRecipe,
    run: impl AsRef<Path>,
    mut state: VocoderTrainingState,
    train: &[VocoderPreparedExample],
    valid: &[VocoderPreparedExample],
    source_config: Option<&Path>,
    source_checkpoint: Option<&Path>,
    device: &B::Device,
    fixture: bool,
    max_new_steps: Option<u64>,
    mut progress: impl FnMut(VocoderTrainingProgress),
) -> Result<VocoderTrainingReport> {
    recipe.validate()?;
    validate_examples(recipe, train)?;
    validate_examples(recipe, valid)?;
    let run = run.as_ref();
    fs::create_dir_all(run)?;
    let state_path = run.join("train_state.json");
    let model_stem = run.join("trainer-latest");
    let gen_optim_path = run.join("optim-generator-latest.bin");
    let disc_optim_path = run.join("optim-discriminator-latest.bin");
    let recorder = BinaryRecorder::new();
    let start_epoch = state.epoch;
    let start_batch = state.batch_in_epoch;
    if state.generator_learning_rate == 0.0 {
        state.generator_learning_rate = recipe.hyperparams().learning_rate;
    }
    if state.discriminator_learning_rate == 0.0 {
        state.discriminator_learning_rate = recipe.hyperparams().discriminator_learning_rate;
    }
    let mut model = if state.global_step > 0 {
        init_model(recipe, source_config, source_checkpoint, device, fixture)?
            .load_file(&model_stem, &recorder, device)?
    } else {
        init_model(recipe, source_config, source_checkpoint, device, fixture)?
    };
    let clipping = recipe
        .hyperparams()
        .gradient_clip_norm
        .map(|value| GradientClippingConfig::Norm(value as f32));
    let mut generator_config = AdamWConfig::new()
        .with_beta_1(recipe.hyperparams().adam_beta1 as f32)
        .with_beta_2(recipe.hyperparams().adam_beta2 as f32);
    let mut discriminator_config = AdamWConfig::new()
        .with_beta_1(recipe.hyperparams().adam_beta1 as f32)
        .with_beta_2(recipe.hyperparams().adam_beta2 as f32);
    if let Some(clipping) = clipping {
        generator_config = generator_config.with_grad_clipping(Some(clipping.clone()));
        discriminator_config = discriminator_config.with_grad_clipping(Some(clipping));
    }
    let mut generator_optimizer = generator_config.init::<B, VocoderModel<B>>();
    let mut discriminator_optimizer = discriminator_config.init::<B, VocoderModel<B>>();
    if state.global_step > 0 {
        generator_optimizer = generator_optimizer.load_record(
            recorder.load(gen_optim_path.with_extension(""), device)?,
        );
        discriminator_optimizer = discriminator_optimizer.load_record(
            recorder.load(disc_optim_path.with_extension(""), device)?,
        );
        progress(VocoderTrainingProgress::Resume {
            epoch: state.epoch,
            batch: state.batch_in_epoch,
            global_step: state.global_step,
        });
    }
    let mut new_steps = 0;
    let mut interrupted = false;
    while state.epoch < recipe.hyperparams().epochs {
        let epoch = state.epoch + 1;
        let batches = batch_indices(train, recipe.hyperparams().batch_size, epoch);
        progress(VocoderTrainingProgress::Epoch {
            epoch,
            epochs: recipe.hyperparams().epochs,
        });
        for batch_index in state.batch_in_epoch..batches.len() {
            let phase = recipe
                .hyperparams()
                .resolved_adversarial_schedule()
                .training_phase(state.global_step);
            if matches!(phase, VocoderTrainingPhase::Generator | VocoderTrainingPhase::Joint) {
                let generator_view = model.clone().no_grad_discriminators();
                let output = generator_view.training_forward(
                    collate(recipe, train, &batches[batch_index], device)?,
                    state.global_step,
                )?;
                let loss = output
                    .generator_loss
                    .context("vocoder schedule omitted generator loss")?;
                let scalar: B::FloatElem = loss.clone().into_scalar();
                let value: f32 = scalar.elem();
                ensure!(value.is_finite(), "non-finite vocoder generator loss");
                let gradients = GradientsParams::from_grads(loss.backward(), &model);
                model = generator_optimizer.step(
                    state.generator_learning_rate,
                    model,
                    gradients,
                );
            }
            if matches!(phase, VocoderTrainingPhase::Discriminator | VocoderTrainingPhase::Joint) {
                let output = model.training_forward(
                    collate(recipe, train, &batches[batch_index], device)?,
                    state.global_step,
                )?;
                let loss = output
                    .discriminator_loss
                    .context("vocoder schedule omitted discriminator loss")?;
                let scalar: B::FloatElem = loss.clone().into_scalar();
                let value: f32 = scalar.elem();
                ensure!(value.is_finite(), "non-finite vocoder discriminator loss");
                let gradients = GradientsParams::from_grads(loss.backward(), &model);
                model = discriminator_optimizer.step(
                    state.discriminator_learning_rate,
                    model,
                    gradients,
                );
            }
            state.global_step += 1;
            state.batch_in_epoch = batch_index + 1;
            new_steps += 1;
            progress(VocoderTrainingProgress::Batch {
                epoch,
                batch: state.batch_in_epoch,
                batches: batches.len(),
                global_step: state.global_step,
                phase,
            });
            let checkpoint_due = recipe.hyperparams().checkpoint_interval_steps > 0
                && state
                    .global_step
                    .is_multiple_of(recipe.hyperparams().checkpoint_interval_steps);
            if checkpoint_due {
                save_checkpoint(
                    recipe,
                    run,
                    &state_path,
                    &state,
                    &model,
                    &generator_optimizer,
                    &discriminator_optimizer,
                    false,
                    &mut progress,
                )?;
            }
            if recipe.hyperparams().eval_interval_steps > 0
                && state
                    .global_step
                    .is_multiple_of(recipe.hyperparams().eval_interval_steps)
            {
                save_sample(
                    recipe,
                    run,
                    state.global_step,
                    &model.valid(),
                    &valid[0],
                    device,
                    &mut progress,
                )?;
            }
            if max_new_steps.is_some_and(|limit| new_steps >= limit) {
                if !checkpoint_due {
                    save_checkpoint(
                        recipe,
                        run,
                        &state_path,
                        &state,
                        &model,
                        &generator_optimizer,
                        &discriminator_optimizer,
                        false,
                        &mut progress,
                    )?;
                }
                interrupted = true;
                break;
            }
            if recipe.hyperparams().max_steps > 0
                && state.global_step >= recipe.hyperparams().max_steps
            {
                interrupted = true;
                break;
            }
        }
        if interrupted {
            break;
        }
        let evaluation = evaluate_model(recipe, &model.valid(), valid, device)?;
        let improved = evaluation.loss < state.best_loss;
        state.best_loss = state.best_loss.min(evaluation.loss);
        state.epoch = epoch;
        state.batch_in_epoch = 0;
        state.generator_learning_rate =
            (state.generator_learning_rate * recipe.hyperparams().scheduler_gamma)
                .max(recipe.hyperparams().minimum_learning_rate);
        state.discriminator_learning_rate =
            (state.discriminator_learning_rate * recipe.hyperparams().scheduler_gamma)
                .max(recipe.hyperparams().minimum_learning_rate);
        save_sample(
            recipe,
            run,
            state.global_step,
            &model.valid(),
            &valid[0],
            device,
            &mut progress,
        )?;
        save_checkpoint(
            recipe,
            run,
            &state_path,
            &state,
            &model,
            &generator_optimizer,
            &discriminator_optimizer,
            improved,
            &mut progress,
        )?;
    }
    if !interrupted {
        progress(VocoderTrainingProgress::Complete {
            best_loss_micros: (state.best_loss * 1_000_000.0) as u64,
            path: run.join("model.safetensors"),
        });
    }
    Ok(VocoderTrainingReport {
        start_epoch,
        start_batch,
        end_epoch: state.epoch,
        end_batch: state.batch_in_epoch,
        global_step: state.global_step,
        best_loss: state.best_loss,
        interrupted,
    })
}

pub fn evaluate_vocoder<B: Backend>(
    recipe: &NativeVocoderRecipe,
    run: impl AsRef<Path>,
    examples: &[VocoderPreparedExample],
    device: &B::Device,
    fixture: bool,
) -> Result<VocoderEvaluationReport> {
    let run = run.as_ref();
    let recorder = BinaryRecorder::new();
    let model: VocoderModel<B> =
        init_model(recipe, None, None, device, fixture)?.load_file(
            run.join("trainer-latest"),
            &recorder,
            device,
        )?;
    evaluate_model(recipe, &model, examples, device)
}

pub fn export_vocoder<B: Backend>(
    recipe: &NativeVocoderRecipe,
    run: impl AsRef<Path>,
    device: &B::Device,
    fixture: bool,
) -> Result<PathBuf> {
    let run = run.as_ref();
    let recorder = BinaryRecorder::new();
    let model: VocoderModel<B> =
        init_model(recipe, None, None, device, fixture)?.load_file(
            run.join("trainer-latest"),
            &recorder,
            device,
        )?;
    let path = run.join("model.safetensors");
    save_inference_checkpoint(&model, &path)?;
    Ok(path)
}

fn init_model<B: Backend>(
    recipe: &NativeVocoderRecipe,
    source_config: Option<&Path>,
    source_checkpoint: Option<&Path>,
    device: &B::Device,
    fixture: bool,
) -> Result<VocoderModel<B>> {
    ensure!(
        source_config.is_some() == source_checkpoint.is_some(),
        "source config and checkpoint must be supplied together"
    );
    let audio = recipe.audio_config();
    let weights = recipe.loss_weights();
    let schedule = recipe.hyperparams().resolved_adversarial_schedule();
    Ok(match recipe {
        NativeVocoderRecipe::Hifigan(recipe) => {
            let generator = if let (Some(config), Some(checkpoint)) =
                (source_config, source_checkpoint)
            {
                HifiganBundleConfig::from_file(config)?
                    .load_burn_generator(checkpoint, device)?
            } else {
                recipe.generator.init(device).map_err(anyhow::Error::new)?
            };
            let mut trainer =
                HifiganTrainer::new_complete(generator, device, weights, schedule, &audio);
            if fixture {
                trainer.mpd = MultiPeriodDiscriminator::new_fixture(device);
                trainer.msd = MultiScaleDiscriminator::new_fixture(device);
            }
            VocoderModel::Hifigan(trainer)
        }
        NativeVocoderRecipe::Melgan(recipe) => {
            let generator = if let (Some(config), Some(checkpoint)) =
                (source_config, source_checkpoint)
            {
                let config = MelganBundleConfig::from_file(config)?;
                ensure!(config.variant()? == MelganVariant::Melgan);
                config.load_burn_generator(checkpoint, device)?
            } else {
                recipe.generator.init(device).map_err(anyhow::Error::new)?
            };
            let mut trainer =
                MelganTrainer::new_complete(generator, device, weights, schedule, &audio);
            if fixture {
                trainer.msd = MultiScaleDiscriminator::new_fixture(device);
            }
            VocoderModel::Melgan(trainer)
        }
        NativeVocoderRecipe::MultibandMelgan(recipe) => {
            let generator = if let (Some(config), Some(checkpoint)) =
                (source_config, source_checkpoint)
            {
                let config = MelganBundleConfig::from_file(config)?;
                ensure!(config.variant()? == MelganVariant::Multiband);
                config.load_burn_multiband_generator(checkpoint, device)?
            } else {
                recipe
                    .generator
                    .init_multiband(recipe.pqmf.clone().context("missing PQMF")?, device)
                    .map_err(anyhow::Error::new)?
            };
            let mut trainer = MultibandMelganTrainer::new_complete(
                generator, device, weights, schedule, &audio,
            );
            if fixture {
                trainer.msd = MultiScaleDiscriminator::new_fixture(device);
            }
            VocoderModel::MultibandMelgan(trainer)
        }
    })
}

fn validate_examples(recipe: &NativeVocoderRecipe, examples: &[VocoderPreparedExample]) -> Result<()> {
    ensure!(!examples.is_empty(), "empty vocoder example set");
    let mel_bins = recipe.mel_contract().mel_bins;
    for example in examples {
        ensure!(
            !example.mel.is_empty()
                && example.mel.iter().all(|frame| frame.len() == mel_bins),
            "{} has invalid mel geometry",
            example.record_id
        );
        ensure!(
            example.waveform.len() >= example.mel.len() * recipe.mel_contract().hop_size,
            "{} waveform is shorter than mel geometry",
            example.record_id
        );
        ensure!(
            example
                .waveform
                .iter()
                .chain(example.mel.iter().flatten())
                .all(|value| value.is_finite()),
            "{} contains non-finite values",
            example.record_id
        );
    }
    Ok(())
}

fn batch_indices(examples: &[VocoderPreparedExample], size: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut indices = (0..examples.len()).collect::<Vec<_>>();
    indices.sort_by_key(|index| {
        (
            Reverse(examples[*index].waveform.len()),
            splitmix64(seed ^ *index as u64),
        )
    });
    indices.chunks(size).map(<[_]>::to_vec).collect()
}

fn collate<B: Backend>(
    recipe: &NativeVocoderRecipe,
    examples: &[VocoderPreparedExample],
    indices: &[usize],
    device: &B::Device,
) -> Result<BurnVocoderTrainingBatch<B>> {
    let batch = indices.len();
    let frames = indices
        .iter()
        .map(|index| examples[*index].mel.len())
        .min()
        .unwrap();
    let max_segment_frames = recipe.hyperparams().segment_size / recipe.mel_contract().hop_size;
    let frames = frames.min(max_segment_frames).max(1);
    let samples = frames * recipe.mel_contract().hop_size;
    let bins = recipe.mel_contract().mel_bins;
    let mut mel = vec![0.0; batch * frames * bins];
    let mut waveform = vec![0.0; batch * samples];
    for (batch_index, example_index) in indices.iter().copied().enumerate() {
        let example = &examples[example_index];
        for frame in 0..frames {
            let start = (batch_index * frames + frame) * bins;
            mel[start..start + bins].copy_from_slice(&example.mel[frame]);
        }
        waveform[batch_index * samples..(batch_index + 1) * samples]
            .copy_from_slice(&example.waveform[..samples]);
    }
    Ok(BurnVocoderTrainingBatch {
        conditioning_mel: Tensor::from_data(TensorData::new(mel, [batch, frames, bins]), device),
        target_waveform: Tensor::from_data(
            TensorData::new(waveform, [batch, 1, samples]),
            device,
        ),
    })
}

fn evaluate_model<B: Backend>(
    recipe: &NativeVocoderRecipe,
    model: &VocoderModel<B>,
    examples: &[VocoderPreparedExample],
    device: &B::Device,
) -> Result<VocoderEvaluationReport> {
    let started = Instant::now();
    let weights = recipe.loss_weights();
    let mut total_loss = 0.0;
    let mut spectral = 0.0;
    let mut waveform_l1 = 0.0;
    let mut generated_samples = 0;
    let mut finite = true;
    for example in examples {
        let batch = collate(recipe, std::slice::from_ref(example), &[0], device)?;
        let target = batch.target_waveform.clone();
        let generated = model.generate(batch.conditioning_mel)?;
        let samples = generated
            .clone()
            .into_data()
            .to_vec::<f32>()
            .context("reading vocoder audio")?;
        generated_samples += samples.len();
        finite &= !samples.is_empty() && samples.iter().all(|sample| sample.is_finite());
        // PQMF synthesis can add a few boundary samples.  Evaluation compares
        // the shared interval, matching the adversarial training objectives.
        let target_dims = target.dims();
        let generated_dims = generated.dims();
        let aligned_samples = target_dims[2].min(generated_dims[2]);
        let target =
            target.slice([0..target_dims[0], 0..1, 0..aligned_samples]);
        let generated =
            generated.slice([0..generated_dims[0], 0..1, 0..aligned_samples]);
        let generated_mel =
            crate::vits_trainer::differentiable_mel(generated.clone(), &recipe.audio_config())?;
        let target_mel =
            crate::vits_trainer::differentiable_mel(target.clone(), &recipe.audio_config())?;
        let spectral_loss = crate::mel_spectrogram_loss(target_mel, generated_mel);
        let wave_loss = waveform_reconstruction_loss(target, generated);
        let spectral_value: f32 = spectral_loss.into_scalar().elem();
        let wave_value: f32 = wave_loss.into_scalar().elem();
        spectral += spectral_value as f64;
        waveform_l1 += wave_value as f64;
        total_loss += spectral_value as f64 * weights.mel_spectrogram
            + wave_value as f64 * weights.waveform_reconstruction;
    }
    let elapsed = started.elapsed().as_secs_f64();
    let audio_seconds =
        generated_samples as f64 / recipe.mel_contract().sample_rate_hz as f64;
    Ok(VocoderEvaluationReport {
        examples: examples.len(),
        loss: total_loss / examples.len() as f64,
        spectral_l1: spectral / examples.len() as f64,
        waveform_l1: waveform_l1 / examples.len() as f64,
        finite_audio: finite,
        generated_samples,
        realtime_factor: if audio_seconds > 0.0 {
            elapsed / audio_seconds
        } else {
            f64::INFINITY
        },
        parameter_memory_bytes: model.num_params() * std::mem::size_of::<f32>(),
    })
}

#[allow(clippy::too_many_arguments)]
fn save_checkpoint<B, GO, DO>(
    recipe: &NativeVocoderRecipe,
    run: &Path,
    state_path: &Path,
    state: &VocoderTrainingState,
    model: &VocoderModel<B>,
    generator_optimizer: &GO,
    discriminator_optimizer: &DO,
    is_best: bool,
    progress: &mut impl FnMut(VocoderTrainingProgress),
) -> Result<()>
where
    B: AutodiffBackend,
    GO: Optimizer<VocoderModel<B>, B>,
    DO: Optimizer<VocoderModel<B>, B>,
{
    let recorder = BinaryRecorder::new();
    let model_part = run.join("trainer-latest.part");
    let gen_part = run.join("optim-generator-latest.part");
    let disc_part = run.join("optim-discriminator-latest.part");
    let inference_part = run.join("model-latest.safetensors.part");
    model.clone().save_file(&model_part, &recorder)?;
    recorder.record(generator_optimizer.to_record(), gen_part.clone())?;
    recorder.record(discriminator_optimizer.to_record(), disc_part.clone())?;
    save_inference_checkpoint(&model.valid(), &inference_part)?;
    let staged = [
        model_part.with_extension("bin"),
        gen_part.with_extension("bin"),
        disc_part.with_extension("bin"),
        inference_part,
    ];
    for path in &staged {
        File::open(path)?.sync_all()?;
    }
    fs::rename(&staged[0], run.join("trainer-latest.bin"))?;
    fs::rename(&staged[1], run.join("optim-generator-latest.bin"))?;
    fs::rename(&staged[2], run.join("optim-discriminator-latest.bin"))?;
    fs::rename(&staged[3], run.join("model-latest.safetensors"))?;
    if is_best {
        fs::copy(
            run.join("model-latest.safetensors"),
            run.join("model.safetensors"),
        )?;
    }
    write_json_atomic(state_path, state)?;
    write_json_atomic(&run.join("recipe.json"), recipe)?;
    progress(VocoderTrainingProgress::Checkpoint {
        global_step: state.global_step,
        path: run.join("model-latest.safetensors"),
    });
    Ok(())
}

fn save_inference_checkpoint<B: Backend>(
    model: &VocoderModel<B>,
    path: &Path,
) -> Result<()> {
    let mut store = SafetensorsStore::from_file(path)
        .overwrite(true)
        .skip_enum_variants(true)
        .with_to_adapter(BurnToPyTorchAdapter);
    match model {
        VocoderModel::Hifigan(trainer) => trainer.generator.save_into(&mut store)?,
        VocoderModel::Melgan(trainer) => trainer.generator.save_into(&mut store)?,
        VocoderModel::MultibandMelgan(trainer) => {
            trainer.generator.save_into(&mut store)?
        }
    }
    Ok(())
}

fn save_sample<B: Backend>(
    recipe: &NativeVocoderRecipe,
    run: &Path,
    global_step: u64,
    model: &VocoderModel<B>,
    example: &VocoderPreparedExample,
    device: &B::Device,
    progress: &mut impl FnMut(VocoderTrainingProgress),
) -> Result<()> {
    let generated = model.generate(
        collate(recipe, std::slice::from_ref(example), &[0], device)?.conditioning_mel,
    )?;
    let samples = generated.into_data().to_vec::<f32>()?;
    ensure!(
        !samples.is_empty() && samples.iter().all(|sample| sample.is_finite()),
        "vocoder sample is empty or non-finite"
    );
    let dir = run.join("samples");
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("validation-step-{global_step}.wav"));
    let part = path.with_extension("wav.part");
    let mut writer = hound::WavWriter::create(
        &part,
        hound::WavSpec {
            channels: 1,
            sample_rate: recipe.mel_contract().sample_rate_hz,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    for sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    fs::rename(part, &path)?;
    progress(VocoderTrainingProgress::Sample { global_step, path });
    Ok(())
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let part = path.with_extension("json.part");
    fs::write(&part, serde_json::to_vec_pretty(value)?)?;
    File::open(&part)?.sync_all()?;
    fs::rename(part, path)?;
    Ok(())
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
