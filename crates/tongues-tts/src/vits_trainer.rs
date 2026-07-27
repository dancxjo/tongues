//! Executable, library-owned native VITS training loop.
//!
//! This module owns batching, alternating optimizer updates, exact batch-cursor
//! resume, validation, sample synthesis, and inference export. The CLI is only
//! a renderer and argument adapter.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use burn::module::{AutodiffModule, Module};
use burn::optim::{grad_clipping::GradientClippingConfig, AdamWConfig, GradientsParams, Optimizer};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use tongues_data::speech_corpus::{feature_cache_path, CachedSpeechFeatures, SpeechRecord};

use crate::{
    combine_vits_generator_losses, VitsDiscriminators, VitsFreezeConfig, VitsInferenceConfig,
    VitsRunLayout, VitsTrainableGenerator, VitsTrainingBatch, VitsTrainingProgress,
    VitsTrainingRecipe, VitsTrainingState,
};

type RecorderImpl = BinFileRecorder<FullPrecisionSettings>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VitsPreparedExample {
    pub record_id: String,
    pub token_ids: Vec<u32>,
    /// Channel-major linear spectrogram `[fft_bins][frames]`.
    pub spectrogram: Vec<Vec<f32>>,
    pub waveform: Vec<f32>,
    pub speaker_id: Option<u32>,
    pub language_id: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VitsTrainOptions {
    /// Stop successfully after this many new batches. Intended for durable
    /// interruption/resume acceptance tests.
    pub max_steps: Option<u64>,
    /// Use compact discriminators for the documented CPU fixture.
    pub fixture: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VitsEvaluationReport {
    pub examples: usize,
    pub batches: usize,
    pub loss: f64,
    pub finite_audio: bool,
    pub generated_samples: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VitsTrainingReport {
    pub start_epoch: u64,
    pub start_batch: usize,
    pub end_epoch: u64,
    pub end_batch: usize,
    pub global_step: u64,
    pub best_validation_loss: Option<f64>,
    pub interrupted: bool,
}

/// Load prepared speech rows, cached token/spectrogram features, and source
/// audio into the model-neutral examples consumed by [`train_vits`].
pub fn load_vits_examples(
    split_path: impl AsRef<Path>,
    feature_cache: impl AsRef<Path>,
    sample_rate_hz: u32,
    speaker_ids: &HashMap<String, u32>,
    language_ids: &HashMap<String, u32>,
) -> Result<Vec<VitsPreparedExample>> {
    let split_path = split_path.as_ref();
    let feature_cache = feature_cache.as_ref();
    let reader = BufReader::new(
        File::open(split_path).with_context(|| format!("opening {}", split_path.display()))?,
    );
    let base = split_path.parent().unwrap_or_else(|| Path::new("."));
    let mut examples = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading {}", split_path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: SpeechRecord = serde_json::from_str(&line)
            .with_context(|| format!("parsing {} line {}", split_path.display(), index + 1))?;
        let cache_path = feature_cache_path(feature_cache, &record.id);
        let cached: CachedSpeechFeatures = serde_json::from_reader(
            File::open(&cache_path)
                .with_context(|| format!("opening VITS feature cache {}", cache_path.display()))?,
        )
        .with_context(|| format!("parsing {}", cache_path.display()))?;
        ensure!(
            cached.record_id == record.id,
            "feature cache {} belongs to {}, expected {}",
            cache_path.display(),
            cached.record_id,
            record.id
        );
        ensure!(
            !cached.text_tokens.is_empty(),
            "VITS feature cache {} has no text tokens",
            cache_path.display()
        );
        ensure!(
            !cached.acoustic_features.is_empty(),
            "VITS feature cache {} has no spectrogram frames",
            cache_path.display()
        );
        let bins = cached.acoustic_features[0].len();
        ensure!(
            bins > 0
                && cached
                    .acoustic_features
                    .iter()
                    .all(|frame| frame.len() == bins),
            "VITS feature cache {} has inconsistent spectrogram width",
            cache_path.display()
        );
        let mut spectrogram = vec![vec![0.0; cached.acoustic_features.len()]; bins];
        for (frame_index, frame) in cached.acoustic_features.iter().enumerate() {
            for (bin, value) in frame.iter().copied().enumerate() {
                spectrogram[bin][frame_index] = value;
            }
        }
        let audio_path = if record.audio_path.is_absolute() {
            record.audio_path.clone()
        } else {
            base.join(&record.audio_path)
        };
        let audio = tongues_audio::read_wav(&audio_path)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("reading {}", audio_path.display()))?
            .convert_channels(1)
            .map_err(anyhow::Error::from)?
            .resample_linear(sample_rate_hz)
            .map_err(anyhow::Error::from)?;
        let speaker_id = match record.speaker.as_deref() {
            Some(name) => Some(*speaker_ids.get(name).with_context(|| {
                format!(
                    "speaker `{name}` from {} is not in the speaker map",
                    record.id
                )
            })?),
            None => None,
        };
        let language_id = language_ids.get(&record.language).copied();
        examples.push(VitsPreparedExample {
            record_id: record.id,
            token_ids: cached.text_tokens,
            spectrogram,
            waveform: audio.samples,
            speaker_id,
            language_id,
        });
    }
    ensure!(
        !examples.is_empty(),
        "VITS split is empty: {}",
        split_path.display()
    );
    Ok(examples)
}

#[allow(clippy::too_many_arguments)]
pub fn train_vits<B: AutodiffBackend>(
    config: &VitsInferenceConfig,
    recipe: &VitsTrainingRecipe,
    layout: &VitsRunLayout,
    mut state: VitsTrainingState,
    train: &[VitsPreparedExample],
    valid: &[VitsPreparedExample],
    source_checkpoint: Option<&Path>,
    device: &B::Device,
    options: &VitsTrainOptions,
    mut progress: impl FnMut(VitsTrainingProgress),
) -> Result<VitsTrainingReport> {
    recipe.validate()?;
    validate_examples(config, recipe, train)?;
    validate_examples(config, recipe, valid)?;
    let fixture_marker = layout.root().join("fixture-mode");
    if options.fixture {
        fs::write(&fixture_marker, b"compact CPU discriminator topology\n")
            .with_context(|| format!("writing {}", fixture_marker.display()))?;
    } else {
        ensure!(
            !fixture_marker.exists(),
            "{} was initialized as a CPU fixture; resume with fixture mode enabled",
            layout.root().display()
        );
    }
    let start_epoch = state.epoch;
    let start_batch = state.batch_in_epoch;
    let recorder = RecorderImpl::new();
    let generator_stem = layout.root().join("trainer-generator-latest");
    let discriminator_stem = layout.root().join("trainer-discriminator-latest");

    B::seed(device, recipe.seed);
    let mut generator = if state.global_step > 0 {
        VitsTrainableGenerator::init_random(config, device)?
            .load_file(&generator_stem, &recorder, device)
            .with_context(|| format!("loading {}.bin", generator_stem.display()))?
    } else {
        let inference = match source_checkpoint {
            Some(path) => crate::VitsInferenceExport::load_coqui_checkpoint(config, path, device)?,
            None => crate::VitsInferenceExport::init_random(config, device)?,
        };
        VitsTrainableGenerator {
            inference,
            posterior_encoder: crate::VitsPosteriorEncoderConfig {
                input_channels: config.network.out_channels,
                latent_channels: config.network.hidden_channels,
                hidden_channels: config.network.hidden_channels,
                kernel_size: config.network.kernel_size_posterior_encoder,
                dilation_rate: config.network.dilation_rate_posterior_encoder,
                num_layers: config.network.num_layers_posterior_encoder,
                conditioning_channels: if config.network.use_speaker_embedding {
                    config.network.speaker_embedding_channels
                } else {
                    0
                },
            }
            .init(device)?,
        }
    };
    generator = apply_freeze(generator, &recipe.freeze);
    let discriminator_init = || {
        if options.fixture {
            VitsDiscriminators::new_fixture(device)
        } else {
            VitsDiscriminators::new(device)
        }
    };
    let mut discriminators = if state.global_step > 0 {
        discriminator_init()
            .load_file(&discriminator_stem, &recorder, device)
            .with_context(|| format!("loading {}.bin", discriminator_stem.display()))?
    } else {
        discriminator_init()
    };
    let mut generator_optimizer_config = AdamWConfig::new()
        .with_beta_1(recipe.optimizer.adam_beta1 as f32)
        .with_beta_2(recipe.optimizer.adam_beta2 as f32)
        .with_weight_decay(recipe.optimizer.weight_decay as f32);
    let mut discriminator_optimizer_config = AdamWConfig::new()
        .with_beta_1(recipe.optimizer.adam_beta1 as f32)
        .with_beta_2(recipe.optimizer.adam_beta2 as f32)
        .with_weight_decay(recipe.optimizer.weight_decay as f32);
    if let Some(max_norm) = recipe.optimizer.gradient_clip_norm {
        let clipping = Some(GradientClippingConfig::Norm(max_norm as f32));
        generator_optimizer_config =
            generator_optimizer_config.with_grad_clipping(clipping.clone());
        discriminator_optimizer_config =
            discriminator_optimizer_config.with_grad_clipping(clipping);
    }
    let mut generator_optimizer = generator_optimizer_config.init::<B, VitsTrainableGenerator<B>>();
    let mut discriminator_optimizer =
        discriminator_optimizer_config.init::<B, VitsDiscriminators<B>>();
    if state.global_step > 0 {
        let gen_record = recorder
            .load(stem(&state.generator_optimizer_checkpoint), device)
            .with_context(|| {
                format!("loading {}", state.generator_optimizer_checkpoint.display())
            })?;
        generator_optimizer = generator_optimizer.load_record(gen_record);
        let disc_record = recorder
            .load(stem(&state.discriminator_optimizer_checkpoint), device)
            .with_context(|| {
                format!(
                    "loading {}",
                    state.discriminator_optimizer_checkpoint.display()
                )
            })?;
        discriminator_optimizer = discriminator_optimizer.load_record(disc_record);
    }

    let mut new_steps = 0u64;
    let mut interrupted = false;
    while state.epoch < recipe.epochs {
        let epoch_number = state.epoch + 1;
        let batches = batch_indices(train, recipe.batch_size, state.shuffle_seed ^ epoch_number);
        ensure!(
            state.batch_in_epoch <= batches.len(),
            "saved VITS batch cursor {} exceeds {} batches in epoch {}",
            state.batch_in_epoch,
            batches.len(),
            epoch_number
        );
        progress(VitsTrainingProgress::Epoch {
            epoch: epoch_number,
            epochs: recipe.epochs,
        });
        for batch_index in state.batch_in_epoch..batches.len() {
            let batch = collate::<B>(
                config,
                train,
                &batches[batch_index],
                recipe.segment_frames,
                device,
            )?;
            let forward = generator.training_forward(
                batch,
                recipe.segment_frames,
                config.audio.hop_length,
                recipe.seed ^ state.global_step,
            )?;
            let (adversarial, feature_matching) = discriminators.generator_losses(
                forward.target_waveform.clone(),
                forward.generated_waveform.clone(),
            );
            let target_mel = differentiable_mel(forward.target_waveform.clone(), &config.audio)?;
            let generated_mel =
                differentiable_mel(forward.generated_waveform.clone(), &config.audio)?;
            let losses = combine_vits_generator_losses(
                adversarial,
                feature_matching,
                target_mel,
                generated_mel,
                forward.duration_loss,
                forward.kl_loss,
                &recipe.loss_weights,
            );
            let loss_value: f32 = losses.total.clone().into_scalar().elem();
            ensure!(
                loss_value.is_finite(),
                "non-finite VITS generator loss at epoch {epoch_number} batch {batch_index}"
            );
            let generator_grads = GradientsParams::from_grads(losses.total.backward(), &generator);
            generator =
                generator_optimizer.step(state.generator_learning_rate, generator, generator_grads);

            let batch = collate::<B>(
                config,
                train,
                &batches[batch_index],
                recipe.segment_frames,
                device,
            )?;
            let forward = generator.training_forward(
                batch,
                recipe.segment_frames,
                config.audio.hop_length,
                recipe.seed ^ state.global_step ^ 0xd15c,
            )?;
            let discriminator_loss = discriminators
                .discriminator_loss(forward.target_waveform, forward.generated_waveform);
            let discriminator_value: f32 = discriminator_loss.clone().into_scalar().elem();
            ensure!(
                discriminator_value.is_finite(),
                "non-finite VITS discriminator loss at epoch {epoch_number} batch {batch_index}"
            );
            let discriminator_grads =
                GradientsParams::from_grads(discriminator_loss.backward(), &discriminators);
            discriminators = discriminator_optimizer.step(
                state.discriminator_learning_rate,
                discriminators,
                discriminator_grads,
            );

            state.global_step += 1;
            state.batch_in_epoch = batch_index + 1;
            progress(VitsTrainingProgress::Batch {
                epoch: epoch_number,
                batch: state.batch_in_epoch,
                batches: batches.len(),
                global_step: state.global_step,
            });
            new_steps += 1;
            let checkpoint_due = recipe.checkpoints.every_steps > 0
                && state
                    .global_step
                    .is_multiple_of(recipe.checkpoints.every_steps);
            if checkpoint_due {
                save_training_checkpoint(
                    layout,
                    &mut state,
                    &generator,
                    &discriminators,
                    &generator_optimizer,
                    &discriminator_optimizer,
                    false,
                    &mut progress,
                )?;
            }
            if recipe.checkpoints.sample_every_steps > 0
                && state
                    .global_step
                    .is_multiple_of(recipe.checkpoints.sample_every_steps)
            {
                save_validation_sample(
                    layout,
                    state.global_step,
                    &generator.valid(),
                    config,
                    &valid[0],
                    device,
                    &mut progress,
                )?;
            }
            if options.max_steps.is_some_and(|limit| new_steps >= limit) {
                if !checkpoint_due {
                    save_training_checkpoint(
                        layout,
                        &mut state,
                        &generator,
                        &discriminators,
                        &generator_optimizer,
                        &discriminator_optimizer,
                        false,
                        &mut progress,
                    )?;
                }
                interrupted = true;
                break;
            }
        }
        if interrupted {
            break;
        }
        let evaluation = evaluate_modules(
            config,
            recipe,
            &generator.valid(),
            &discriminators.valid(),
            valid,
            device,
        )?;
        let improved = state
            .best_validation_loss
            .is_none_or(|best| evaluation.loss < best);
        if improved {
            state.best_validation_loss = Some(evaluation.loss);
        }
        state.epoch = epoch_number;
        state.batch_in_epoch = 0;
        state.generator_learning_rate = (state.generator_learning_rate * recipe.scheduler.gamma)
            .max(recipe.scheduler.minimum_learning_rate);
        state.discriminator_learning_rate = (state.discriminator_learning_rate
            * recipe.scheduler.gamma)
            .max(recipe.scheduler.minimum_learning_rate);
        save_validation_sample(
            layout,
            state.global_step,
            &generator.valid(),
            config,
            &valid[0],
            device,
            &mut progress,
        )?;
        save_training_checkpoint(
            layout,
            &mut state,
            &generator,
            &discriminators,
            &generator_optimizer,
            &discriminator_optimizer,
            improved,
            &mut progress,
        )?;
    }
    if !interrupted {
        progress(VitsTrainingProgress::Complete {
            best_epoch: state.best_epoch.unwrap_or(state.epoch),
            best_model: layout.best_checkpoint(),
        });
    }
    Ok(VitsTrainingReport {
        start_epoch,
        start_batch,
        end_epoch: state.epoch,
        end_batch: state.batch_in_epoch,
        global_step: state.global_step,
        best_validation_loss: state.best_validation_loss,
        interrupted,
    })
}

pub fn evaluate_vits<B: Backend>(
    config: &VitsInferenceConfig,
    recipe: &VitsTrainingRecipe,
    layout: &VitsRunLayout,
    examples: &[VitsPreparedExample],
    device: &B::Device,
) -> Result<VitsEvaluationReport> {
    let recorder = RecorderImpl::new();
    let generator: VitsTrainableGenerator<B> = VitsTrainableGenerator::init_random(config, device)?
        .load_file(
            layout.root().join("trainer-generator-latest"),
            &recorder,
            device,
        )?;
    let discriminators = if layout.root().join("fixture-mode").is_file() {
        VitsDiscriminators::new_fixture(device)
    } else {
        VitsDiscriminators::new(device)
    }
    .load_file(
        layout.root().join("trainer-discriminator-latest"),
        &recorder,
        device,
    )?;
    evaluate_modules(
        config,
        recipe,
        &generator,
        &discriminators,
        examples,
        device,
    )
}

pub fn export_vits<B: Backend>(
    config: &VitsInferenceConfig,
    layout: &VitsRunLayout,
    device: &B::Device,
) -> Result<PathBuf> {
    let recorder = RecorderImpl::new();
    let generator: VitsTrainableGenerator<B> = VitsTrainableGenerator::init_random(config, device)?
        .load_file(
            layout.root().join("trainer-generator-latest"),
            &recorder,
            device,
        )?;
    let output = layout.best_checkpoint();
    generator.inference.save_inference_safetensors(&output)?;
    Ok(output)
}

fn validate_examples(
    config: &VitsInferenceConfig,
    recipe: &VitsTrainingRecipe,
    examples: &[VitsPreparedExample],
) -> Result<()> {
    ensure!(!examples.is_empty(), "VITS example set is empty");
    for example in examples {
        ensure!(
            !example.token_ids.is_empty(),
            "{} has no tokens",
            example.record_id
        );
        ensure!(
            example.spectrogram.len() == config.network.out_channels,
            "{} has {} spectrogram bins; expected {}",
            example.record_id,
            example.spectrogram.len(),
            config.network.out_channels
        );
        let frames = example.spectrogram[0].len();
        ensure!(
            frames >= example.token_ids.len() && frames >= recipe.segment_frames,
            "{} needs at least max(tokens={}, segment={}) frames; found {}",
            example.record_id,
            example.token_ids.len(),
            recipe.segment_frames,
            frames
        );
        ensure!(
            example
                .spectrogram
                .iter()
                .all(|channel| channel.len() == frames),
            "{} has inconsistent spectrogram channel lengths",
            example.record_id
        );
        ensure!(
            example.waveform.len() >= frames * config.audio.hop_length,
            "{} waveform is shorter than its spectrogram geometry",
            example.record_id
        );
        ensure!(
            example
                .waveform
                .iter()
                .chain(example.spectrogram.iter().flatten())
                .all(|value| value.is_finite()),
            "{} contains non-finite audio/features",
            example.record_id
        );
    }
    Ok(())
}

fn batch_indices(
    examples: &[VitsPreparedExample],
    batch_size: usize,
    seed: u64,
) -> Vec<Vec<usize>> {
    let mut indices = (0..examples.len()).collect::<Vec<_>>();
    indices.sort_by_key(|index| {
        (
            Reverse(examples[*index].waveform.len()),
            splitmix64(seed ^ *index as u64),
        )
    });
    indices
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn collate<B: Backend>(
    config: &VitsInferenceConfig,
    examples: &[VitsPreparedExample],
    indices: &[usize],
    segment_frames: usize,
    device: &B::Device,
) -> Result<VitsTrainingBatch<B>> {
    let selected = indices
        .iter()
        .map(|index| &examples[*index])
        .collect::<Vec<_>>();
    let batch = selected.len();
    let tokens = selected
        .iter()
        .map(|row| row.token_ids.len())
        .max()
        .unwrap();
    let frames = selected
        .iter()
        .map(|row| row.spectrogram[0].len())
        .max()
        .unwrap();
    let samples = frames * config.audio.hop_length;
    let bins = config.network.out_channels;
    let mut token_values = vec![0i64; batch * tokens];
    let mut spectrogram = vec![0.0f32; batch * bins * frames];
    let mut waveform = vec![0.0f32; batch * samples];
    let mut token_lengths = Vec::with_capacity(batch);
    let mut frame_lengths = Vec::with_capacity(batch);
    let mut speakers = Vec::with_capacity(batch);
    let mut languages = Vec::with_capacity(batch);
    for (batch_index, row) in selected.iter().enumerate() {
        token_lengths.push(row.token_ids.len());
        frame_lengths.push(row.spectrogram[0].len());
        for (index, token) in row.token_ids.iter().copied().enumerate() {
            token_values[batch_index * tokens + index] = token as i64;
        }
        for bin in 0..bins {
            let start = (batch_index * bins + bin) * frames;
            spectrogram[start..start + row.spectrogram[bin].len()]
                .copy_from_slice(&row.spectrogram[bin]);
        }
        let copy_samples = row.waveform.len().min(samples);
        waveform[batch_index * samples..batch_index * samples + copy_samples]
            .copy_from_slice(&row.waveform[..copy_samples]);
        speakers.push(row.speaker_id.map(i64::from));
        languages.push(row.language_id.map(i64::from));
    }
    ensure!(
        frame_lengths.iter().all(|frames| *frames >= segment_frames),
        "a collated VITS example is shorter than the configured segment"
    );
    let speaker_ids = optional_ids(
        &speakers,
        config.network.use_speaker_embedding,
        "speaker",
        device,
    )?;
    let language_ids = optional_ids(
        &languages,
        config.network.use_language_embedding,
        "language",
        device,
    )?;
    Ok(VitsTrainingBatch {
        token_ids: Tensor::from_data(TensorData::new(token_values, [batch, tokens]), device),
        token_lengths,
        spectrogram: Tensor::from_data(TensorData::new(spectrogram, [batch, bins, frames]), device),
        frame_lengths,
        waveform: Tensor::from_data(TensorData::new(waveform, [batch, 1, samples]), device),
        speaker_ids,
        language_ids,
    })
}

fn optional_ids<B: Backend>(
    values: &[Option<i64>],
    required: bool,
    name: &str,
    device: &B::Device,
) -> Result<Option<Tensor<B, 2, Int>>> {
    if !required {
        ensure!(
            values.iter().all(Option::is_none),
            "{name} IDs were supplied to an unconditioned VITS model"
        );
        return Ok(None);
    }
    let values = values
        .iter()
        .map(|value| value.with_context(|| format!("{name} ID is missing")))
        .collect::<Result<Vec<_>>>()?;
    let count = values.len();
    Ok(Some(Tensor::from_data(
        TensorData::new(values, [count, 1]),
        device,
    )))
}

pub(crate) fn differentiable_mel<B: Backend>(
    waveform: Tensor<B, 3>,
    audio: &crate::AudioFeatureConfig,
) -> Result<Tensor<B, 3>> {
    let [batch, channels, samples] = waveform.dims();
    ensure!(channels == 1, "VITS mel loss expects mono waveform");
    ensure!(
        samples > 0 && audio.fft_size > 0 && audio.hop_length > 0,
        "invalid VITS mel geometry"
    );
    let device = waveform.device();
    let signal = waveform.squeeze_dim::<2>(1);
    let padding = audio.fft_size / 2;
    let signal = signal.pad([(0, 0), (padding, padding)], 0.0);
    let [_batch, padded] = signal.dims();
    let frames = 1 + padded.saturating_sub(audio.fft_size) / audio.hop_length;
    let window = (0..audio.fft_size)
        .map(|index| {
            0.5 - 0.5
                * (2.0 * PI * index as f32 / audio.fft_size.saturating_sub(1).max(1) as f32).cos()
        })
        .collect::<Vec<_>>();
    let window =
        Tensor::<B, 3>::from_data(TensorData::new(window, [1, 1, audio.fft_size]), &device);
    let mut framed = Vec::with_capacity(frames);
    for frame in 0..frames {
        let start = frame * audio.hop_length;
        framed.push(
            signal
                .clone()
                .slice([0..batch, start..start + audio.fft_size]),
        );
    }
    let framed: Tensor<B, 3> = Tensor::stack(framed, 1);
    let framed = framed * window;
    let frequency_bins = audio.fft_size / 2 + 1;
    let mut cosine = Vec::with_capacity(audio.fft_size * frequency_bins);
    let mut sine = Vec::with_capacity(audio.fft_size * frequency_bins);
    for sample in 0..audio.fft_size {
        for bin in 0..frequency_bins {
            let angle = 2.0 * PI * sample as f32 * bin as f32 / audio.fft_size as f32;
            cosine.push(angle.cos());
            sine.push(-angle.sin());
        }
    }
    let cosine = Tensor::<B, 3>::from_data(
        TensorData::new(cosine, [1, audio.fft_size, frequency_bins]),
        &device,
    )
    .repeat_dim(0, batch);
    let sine = Tensor::<B, 3>::from_data(
        TensorData::new(sine, [1, audio.fft_size, frequency_bins]),
        &device,
    )
    .repeat_dim(0, batch);
    let real = framed.clone().matmul(cosine);
    let imaginary = framed.matmul(sine);
    let magnitude = (real.square() + imaginary.square() + 1.0e-8).sqrt();
    let mel = Tensor::<B, 3>::from_data(
        TensorData::new(
            mel_filter_bank(
                audio.sample_rate,
                audio.fft_size,
                audio.num_mels,
                audio.mel_fmin,
                audio.mel_fmax.unwrap_or(audio.sample_rate as f32 / 2.0),
            ),
            [1, frequency_bins, audio.num_mels],
        ),
        &device,
    )
    .repeat_dim(0, batch);
    Ok(magnitude
        .matmul(mel)
        .clamp_min(1.0e-5)
        .log()
        .swap_dims(1, 2))
}

fn mel_filter_bank(
    sample_rate: u32,
    fft_size: usize,
    mel_bins: usize,
    min_hz: f32,
    max_hz: f32,
) -> Vec<f32> {
    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);
    let minimum = hz_to_mel(min_hz);
    let maximum = hz_to_mel(max_hz);
    let points = (0..mel_bins + 2)
        .map(|index| {
            mel_to_hz(minimum + (maximum - minimum) * index as f32 / (mel_bins + 1) as f32)
        })
        .collect::<Vec<_>>();
    let frequencies = (0..fft_size / 2 + 1)
        .map(|bin| bin as f32 * sample_rate as f32 / fft_size as f32)
        .collect::<Vec<_>>();
    let mut weights = vec![0.0; frequencies.len() * mel_bins];
    for (frequency_index, frequency) in frequencies.into_iter().enumerate() {
        for mel in 0..mel_bins {
            let left = points[mel];
            let center = points[mel + 1];
            let right = points[mel + 2];
            weights[frequency_index * mel_bins + mel] = if frequency < left || frequency > right {
                0.0
            } else if frequency <= center {
                (frequency - left) / (center - left).max(f32::EPSILON)
            } else {
                (right - frequency) / (right - center).max(f32::EPSILON)
            };
        }
    }
    weights
}

fn apply_freeze<B: Backend>(
    mut model: VitsTrainableGenerator<B>,
    freeze: &VitsFreezeConfig,
) -> VitsTrainableGenerator<B> {
    if freeze.text_encoder {
        model.inference.text_encoder = model.inference.text_encoder.no_grad();
    }
    if freeze.posterior_encoder {
        model.posterior_encoder = model.posterior_encoder.no_grad();
    }
    if freeze.duration_predictor {
        model.inference.duration_predictor = model.inference.duration_predictor.no_grad();
    }
    if freeze.flow {
        model.inference.flow = model.inference.flow.no_grad();
    }
    if freeze.waveform_decoder {
        model.inference.waveform_decoder = model.inference.waveform_decoder.no_grad();
    }
    if freeze.speaker_embeddings {
        model.inference.emb_g = model.inference.emb_g.map(Module::no_grad);
    }
    if freeze.language_embeddings {
        model.inference.emb_l = model.inference.emb_l.map(Module::no_grad);
    }
    model
}

#[allow(clippy::too_many_arguments)]
fn save_training_checkpoint<B, GO, DO>(
    layout: &VitsRunLayout,
    state: &mut VitsTrainingState,
    generator: &VitsTrainableGenerator<B>,
    discriminators: &VitsDiscriminators<B>,
    generator_optimizer: &GO,
    discriminator_optimizer: &DO,
    is_best: bool,
    progress: &mut impl FnMut(VitsTrainingProgress),
) -> Result<()>
where
    B: AutodiffBackend,
    GO: Optimizer<VitsTrainableGenerator<B>, B>,
    DO: Optimizer<VitsDiscriminators<B>, B>,
{
    let recorder = RecorderImpl::new();
    let generator_stem = layout.root().join("trainer-generator-latest");
    let discriminator_stem = layout.root().join("trainer-discriminator-latest");
    let generator_part = layout.root().join("trainer-generator-latest.part");
    let discriminator_part = layout.root().join("trainer-discriminator-latest.part");
    let generator_optimizer_part = layout.root().join("optim-generator-latest.part");
    let discriminator_optimizer_part = layout.root().join("optim-discriminator-latest.part");
    let inference_part = layout.root().join("model-latest.safetensors.part");
    generator
        .clone()
        .save_file(&generator_part, &recorder)
        .context("saving VITS generator training state")?;
    discriminators
        .clone()
        .save_file(&discriminator_part, &recorder)
        .context("saving VITS discriminator training state")?;
    recorder.record(
        generator_optimizer.to_record(),
        generator_optimizer_part.clone(),
    )?;
    recorder.record(
        discriminator_optimizer.to_record(),
        discriminator_optimizer_part.clone(),
    )?;
    generator
        .valid()
        .inference
        .save_inference_safetensors(&inference_part)?;
    let staged = [
        generator_part.with_extension("bin"),
        discriminator_part.with_extension("bin"),
        generator_optimizer_part.with_extension("bin"),
        discriminator_optimizer_part.with_extension("bin"),
        inference_part.clone(),
    ];
    for path in &staged {
        File::open(path)
            .with_context(|| format!("opening staged VITS checkpoint {}", path.display()))?
            .sync_all()
            .with_context(|| format!("syncing staged VITS checkpoint {}", path.display()))?;
    }
    fs::rename(&staged[0], generator_stem.with_extension("bin"))?;
    fs::rename(&staged[1], discriminator_stem.with_extension("bin"))?;
    fs::rename(&staged[2], &state.generator_optimizer_checkpoint)?;
    fs::rename(&staged[3], &state.discriminator_optimizer_checkpoint)?;
    fs::rename(&staged[4], layout.latest_checkpoint())?;
    if is_best {
        fs::copy(layout.latest_checkpoint(), layout.best_checkpoint()).with_context(|| {
            format!(
                "publishing best VITS checkpoint {}",
                layout.best_checkpoint().display()
            )
        })?;
        state.best_epoch = Some(state.epoch);
    }
    crate::write_vits_training_state(layout, state)?;
    progress(VitsTrainingProgress::Checkpoint {
        epoch: state.epoch,
        global_step: state.global_step,
        path: layout.latest_checkpoint(),
    });
    Ok(())
}

fn evaluate_modules<B: Backend>(
    config: &VitsInferenceConfig,
    recipe: &VitsTrainingRecipe,
    generator: &VitsTrainableGenerator<B>,
    discriminators: &VitsDiscriminators<B>,
    examples: &[VitsPreparedExample],
    device: &B::Device,
) -> Result<VitsEvaluationReport> {
    validate_examples(config, recipe, examples)?;
    let batches = batch_indices(examples, recipe.batch_size, recipe.seed);
    let mut loss = 0.0;
    let mut generated_samples = 0usize;
    let mut finite_audio = true;
    for (index, indices) in batches.iter().enumerate() {
        let forward = generator.training_forward(
            collate(config, examples, indices, recipe.segment_frames, device)?,
            recipe.segment_frames,
            config.audio.hop_length,
            recipe.seed ^ index as u64,
        )?;
        let (adversarial, feature_matching) = discriminators.generator_losses(
            forward.target_waveform.clone(),
            forward.generated_waveform.clone(),
        );
        let target_mel = differentiable_mel(forward.target_waveform, &config.audio)?;
        let generated_mel = differentiable_mel(forward.generated_waveform.clone(), &config.audio)?;
        let losses = combine_vits_generator_losses(
            adversarial,
            feature_matching,
            target_mel,
            generated_mel,
            forward.duration_loss,
            forward.kl_loss,
            &recipe.loss_weights,
        );
        loss += losses.total.into_scalar().elem::<f64>();
        let samples = forward
            .generated_waveform
            .into_data()
            .to_vec::<f32>()
            .context("reading generated validation audio")?;
        generated_samples += samples.len();
        finite_audio &= !samples.is_empty() && samples.iter().all(|sample| sample.is_finite());
    }
    Ok(VitsEvaluationReport {
        examples: examples.len(),
        batches: batches.len(),
        loss: loss / batches.len() as f64,
        finite_audio,
        generated_samples,
    })
}

fn save_validation_sample<B: Backend>(
    layout: &VitsRunLayout,
    global_step: u64,
    generator: &VitsTrainableGenerator<B>,
    config: &VitsInferenceConfig,
    example: &VitsPreparedExample,
    device: &B::Device,
    progress: &mut impl FnMut(VitsTrainingProgress),
) -> Result<()> {
    let frames = example.spectrogram[0].len();
    let forward = generator.training_forward(
        collate(
            config,
            std::slice::from_ref(example),
            &[0],
            frames.min(config.network.spec_segment_size).max(1),
            device,
        )?,
        frames.min(config.network.spec_segment_size).max(1),
        config.audio.hop_length,
        global_step,
    )?;
    let samples = forward
        .generated_waveform
        .into_data()
        .to_vec::<f32>()
        .context("reading VITS validation sample")?;
    ensure!(
        !samples.is_empty() && samples.iter().all(|sample| sample.is_finite()),
        "VITS validation synthesis produced empty or non-finite audio"
    );
    let path = layout
        .sample_dir()
        .join(format!("validation-step-{global_step}.wav"));
    write_wav_atomic(&path, config.audio.sample_rate, &samples)?;
    progress(VitsTrainingProgress::Sample { global_step, path });
    Ok(())
}

fn write_wav_atomic(path: &Path, sample_rate: u32, samples: &[f32]) -> Result<()> {
    let part = path.with_extension("wav.part");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&part, spec)
        .with_context(|| format!("creating {}", part.display()))?;
    for sample in samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    fs::rename(&part, path)
        .with_context(|| format!("publishing {} -> {}", part.display(), path.display()))?;
    Ok(())
}

fn stem(path: &Path) -> PathBuf {
    path.with_extension("")
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
