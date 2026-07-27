//! Emotion classification model family.
//!
//! V1 prepares labeled WAV rows into pooled log-mel feature vectors from random
//! cuts of multiple durations, then trains a small softmax classifier. The
//! feature representation is intentionally simple and durable so we can inspect
//! the dataset before committing to a heavier temporal model.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tongues_interpretation::{InterpretationConfig, DEFAULT_MEL_BINS, DEFAULT_SAMPLE_RATE_HZ};
use tongues_neural::{write_manifest, ModelArtifactManifest};

pub const FAMILY: &str = "emotions";
pub const ARCHITECTURE: &str = "pooled-log-mel-softmax";
pub const DEFAULT_DATASET_ID: &str = "emotion-cuts-v0";
pub const DEFAULT_SOURCE_MANIFEST: &str = "datasets/emotions/labels.jsonl";
pub const EMOTION_TRAIN_STATE_SCHEMA_VERSION: u32 = 1;
const EMOTION_SHUFFLE_SCHEME: &str = "sha256-seed-and-epoch-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionPrepareConfig {
    pub dataset_id: String,
    pub source_manifest: PathBuf,
    pub seed: u64,
    pub train_frac: f64,
    pub valid_frac: f64,
    pub cuts_per_wav: usize,
    pub min_cut_ms: u64,
    pub max_cut_ms: u64,
    pub include_full_cut: bool,
    pub sample_rate_hz: u32,
    pub mel_bins: usize,
}

impl Default for EmotionPrepareConfig {
    fn default() -> Self {
        Self {
            dataset_id: DEFAULT_DATASET_ID.to_string(),
            source_manifest: PathBuf::from(DEFAULT_SOURCE_MANIFEST),
            seed: 42,
            train_frac: 0.8,
            valid_frac: 0.1,
            cuts_per_wav: 8,
            min_cut_ms: 250,
            max_cut_ms: 3_500,
            include_full_cut: true,
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            mel_bins: DEFAULT_MEL_BINS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionTrainConfig {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub batch_size: usize,
    pub epochs: usize,
    pub early_stopping_patience: usize,
    pub seed: u64,
}

impl Default for EmotionTrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.03,
            weight_decay: 1e-4,
            batch_size: 64,
            epochs: 50,
            early_stopping_patience: 20,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmotionTrainMode {
    New,
    Resume,
    Restart,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrainProgress {
    Started {
        mode: EmotionTrainMode,
        start_epoch: usize,
        max_epochs: usize,
    },
    Epoch {
        epoch: usize,
        train_loss: f32,
        validation_loss: f32,
        validation_accuracy: f32,
        best_epoch: usize,
        patience: usize,
    },
    EarlyStopped {
        epoch: usize,
        patience: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionResumeConfig {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub batch_size: usize,
    pub early_stopping_patience: usize,
    pub seed: u64,
}

impl From<&EmotionTrainConfig> for EmotionResumeConfig {
    fn from(config: &EmotionTrainConfig) -> Self {
        Self {
            learning_rate: config.learning_rate,
            weight_decay: config.weight_decay,
            batch_size: config.batch_size,
            early_stopping_patience: config.early_stopping_patience,
            seed: config.seed,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionTrainState {
    pub schema_version: u32,
    pub current_epoch: usize,
    pub best_epoch: usize,
    pub best_val_loss: f32,
    pub best_val_accuracy: f32,
    pub patience: usize,
    pub shuffle_seed: u64,
    pub shuffle_scheme: String,
    pub next_shuffle_epoch: usize,
    pub effective_config: EmotionResumeConfig,
    pub requested_max_epochs: usize,
    pub data_identity: String,
    pub model_config: EmotionModelConfig,
    pub epoch_checkpoint: String,
    pub best_checkpoint: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmotionTrainReport {
    pub mode: EmotionTrainMode,
    pub completed_epoch: usize,
    pub best_epoch: usize,
    pub best_loss: f32,
    pub best_accuracy: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionSourceRow {
    pub emotion: String,
    pub path: PathBuf,
    #[serde(default)]
    pub speaker: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionExample {
    pub id: String,
    pub emotion: String,
    pub label_id: usize,
    pub source_path: String,
    pub start_ms: u64,
    pub duration_ms: u64,
    pub sample_rate_hz: u32,
    pub feature_kind: String,
    pub features: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrepareReport {
    pub source_wavs: usize,
    pub skipped_wavs: usize,
    pub labels: Vec<String>,
    pub train_examples: usize,
    pub valid_examples: usize,
    pub test_examples: usize,
    pub feature_dims: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrepareCheckpointState {
    pub status: String,
    pub dataset_id: String,
    pub source_manifest: String,
    pub report: Option<PrepareReport>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrepareProgress {
    Stage {
        message: String,
    },
    Source {
        rows: usize,
        labels: usize,
    },
    Cut {
        path: String,
        cuts_done: usize,
        cuts_total: usize,
        out_path: String,
    },
    Write {
        split: String,
        rows: usize,
        path: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionModelConfig {
    pub labels: Vec<String>,
    pub feature_dims: usize,
    pub sample_rate_hz: u32,
    pub mel_bins: usize,
    pub feature_kind: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionModel {
    pub config: EmotionModelConfig,
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    pub split: String,
    pub examples: usize,
    pub accuracy: f32,
    pub loss: f32,
    pub labels: Vec<String>,
    pub confusion: Vec<Vec<usize>>,
}

pub fn prepare_dataset(out: &Path, config: &EmotionPrepareConfig) -> Result<PrepareReport> {
    prepare_dataset_with_progress(out, config, |_| {})
}

pub fn prepare_dataset_with_progress(
    out: &Path,
    config: &EmotionPrepareConfig,
    mut progress: impl FnMut(PrepareProgress),
) -> Result<PrepareReport> {
    validate_prepare_config(config)?;
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    write_prepare_state(out, config, "running", None)?;
    progress(PrepareProgress::Stage {
        message: format!(
            "Reading emotion source manifest {}",
            config.source_manifest.display()
        ),
    });

    let sources = read_source_manifest(&config.source_manifest)?;
    let labels = labels_from_sources(&sources);
    progress(PrepareProgress::Source {
        rows: sources.len(),
        labels: labels.len(),
    });
    let label_ids = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.clone(), index))
        .collect::<BTreeMap<_, _>>();

    let examples_part = out.join("examples.jsonl.part");
    let examples_path = out.join("examples.jsonl");
    let examples_file = File::create(&examples_part)
        .with_context(|| format!("creating {}", examples_part.display()))?;
    let mut examples_writer = BufWriter::new(examples_file);

    let mut rng = StdRng::seed_from_u64(config.seed);
    let split_groups = split_source_groups(&sources, config, &mut rng);
    let mut examples = Vec::new();
    let mut train = Vec::new();
    let mut valid = Vec::new();
    let mut test = Vec::new();
    let mut skipped_wavs = 0usize;
    let mut cuts_total = 0usize;
    for source in &sources {
        cuts_total += config.cuts_per_wav + usize::from(config.include_full_cut);
        let Ok(samples) = read_wav_mono_resampled(&source.path, config.sample_rate_hz) else {
            skipped_wavs += 1;
            continue;
        };
        if samples.is_empty() {
            skipped_wavs += 1;
            continue;
        }
        let label_id = *label_ids
            .get(&source.emotion)
            .context("source label missing from label ids")?;
        let mut cuts = random_cuts(&samples, config, &mut rng);
        if config.include_full_cut {
            cuts.push((0, samples.len()));
        }
        for (cut_index, (start, end)) in cuts.into_iter().enumerate() {
            let cut_samples = &samples[start..end];
            let features = pooled_log_mel_features(cut_samples, config);
            let start_ms = samples_to_ms(start, config.sample_rate_hz);
            let duration_ms = samples_to_ms(end.saturating_sub(start), config.sample_rate_hz);
            let example = EmotionExample {
                id: format!("{}:{cut_index}:{start_ms}:{duration_ms}", examples.len()),
                emotion: source.emotion.clone(),
                label_id,
                source_path: source.path.display().to_string(),
                start_ms,
                duration_ms,
                sample_rate_hz: config.sample_rate_hz,
                feature_kind: feature_kind(config),
                features,
            };
            serde_json::to_writer(&mut examples_writer, &example)
                .context("writing emotion example")?;
            writeln!(examples_writer).context("writing emotion example newline")?;
            let group = source_group_id(source);
            if split_groups.train.contains(&group) {
                train.push(example.clone());
            } else if split_groups.valid.contains(&group) {
                valid.push(example.clone());
            } else if split_groups.test.contains(&group) {
                test.push(example.clone());
            } else {
                anyhow::bail!("emotion source group `{group}` was not assigned to a split");
            }
            examples.push(example);
            let cuts_done = examples.len();
            if cuts_done <= 8 || cuts_done % 100 == 0 {
                progress(PrepareProgress::Cut {
                    path: source.path.display().to_string(),
                    cuts_done,
                    cuts_total,
                    out_path: examples_part.display().to_string(),
                });
            }
        }
    }
    examples_writer
        .flush()
        .with_context(|| format!("flushing {}", examples_part.display()))?;
    fs::rename(&examples_part, &examples_path).with_context(|| {
        format!(
            "renaming {} to {}",
            examples_part.display(),
            examples_path.display()
        )
    })?;

    anyhow::ensure!(!examples.is_empty(), "no emotion examples were prepared");

    write_jsonl_split(out, "train", &train, &mut progress)?;
    write_jsonl_split(out, "valid", &valid, &mut progress)?;
    write_jsonl_split(out, "test", &test, &mut progress)?;
    write_json_file_atomic(&out.join("prepare_config.json"), config)?;
    write_readme(out, config, &labels)?;

    let report = PrepareReport {
        source_wavs: sources.len(),
        skipped_wavs,
        labels,
        train_examples: train.len(),
        valid_examples: valid.len(),
        test_examples: test.len(),
        feature_dims: train
            .first()
            .or_else(|| valid.first())
            .or_else(|| test.first())
            .map(|example| example.features.len())
            .unwrap_or(0),
    };
    write_prepare_state(out, config, "complete", Some(&report))?;
    Ok(report)
}

pub fn train(
    data: &Path,
    out: &Path,
    config: &EmotionTrainConfig,
    mode: EmotionTrainMode,
) -> Result<EmotionTrainReport> {
    train_with_progress(data, out, config, mode, |_| {})
}

pub fn train_with_progress(
    data: &Path,
    out: &Path,
    config: &EmotionTrainConfig,
    mode: EmotionTrainMode,
    mut progress: impl FnMut(TrainProgress),
) -> Result<EmotionTrainReport> {
    validate_train_config(config)?;
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let train_rows = read_examples(&data.join("train.jsonl"))?;
    let valid_rows = read_examples(&data.join("valid.jsonl"))?;
    anyhow::ensure!(!train_rows.is_empty(), "no training rows found");
    anyhow::ensure!(!valid_rows.is_empty(), "no validation rows found");

    let labels = labels_from_examples(&train_rows, &valid_rows);
    let feature_dims = train_rows[0].features.len();
    let (mean, std) = feature_stats(&train_rows, feature_dims);
    let model_config = EmotionModelConfig {
        labels: labels.clone(),
        feature_dims,
        sample_rate_hz: train_rows[0].sample_rate_hz,
        mel_bins: feature_dims / 2,
        feature_kind: train_rows[0].feature_kind.clone(),
    };
    let data_identity = emotion_data_identity(data)?;
    let effective_config = EmotionResumeConfig::from(config);

    let resume = match mode {
        EmotionTrainMode::New => {
            anyhow::ensure!(
                !emotion_training_artifacts_exist(out)?,
                "emotion training artifacts already exist in {}; use --resume to continue them or --restart to deliberately replace them",
                out.display()
            );
            None
        }
        EmotionTrainMode::Resume => Some(load_emotion_resume(
            out,
            &effective_config,
            &data_identity,
            &model_config,
        )?),
        EmotionTrainMode::Restart => {
            remove_emotion_training_artifacts(out)?;
            None
        }
    };

    write_json_file_atomic(&out.join("model_config.json"), &model_config)?;
    write_json_file_atomic(&out.join("train_config.json"), config)?;
    write_manifest(
        out,
        &ModelArtifactManifest::new(FAMILY, ARCHITECTURE, data.display().to_string()),
    )?;

    let (mut model, start_epoch, mut best_epoch, mut best_loss, mut best_accuracy, mut patience) =
        if let Some((state, model)) = resume {
            restore_best_model(out, &state)?;
            (
                model,
                state.current_epoch + 1,
                state.best_epoch,
                state.best_val_loss,
                state.best_val_accuracy,
                state.patience,
            )
        } else {
            (
                EmotionModel {
                    config: model_config.clone(),
                    weights: vec![vec![0.0; feature_dims]; labels.len()],
                    bias: vec![0.0; labels.len()],
                    mean,
                    std,
                },
                1,
                0,
                f32::INFINITY,
                0.0,
                0,
            )
        };

    anyhow::ensure!(
        start_epoch <= config.epochs,
        "emotion run in {} already completed epoch {}; --epochs must be greater than {} to resume",
        out.display(),
        start_epoch.saturating_sub(1),
        start_epoch.saturating_sub(1)
    );
    progress(TrainProgress::Started {
        mode,
        start_epoch,
        max_epochs: config.epochs,
    });

    let mut completed_epoch = start_epoch - 1;
    for epoch in start_epoch..=config.epochs {
        let mut epoch_rng = emotion_epoch_rng(config.seed, epoch);
        let train_loss = train_epoch(&mut model, &train_rows, config, &mut epoch_rng);
        let report = evaluate_model(&model, &valid_rows, "valid");
        anyhow::ensure!(
            train_loss.is_finite() && report.loss.is_finite() && report.accuracy.is_finite(),
            "non-finite emotion training metric at epoch {epoch}; no checkpoint was published"
        );
        let epoch_path = out.join(format!("model-epoch-{epoch}.json"));
        write_json_file_atomic(&epoch_path, &model)?;

        if report.loss < best_loss {
            best_loss = report.loss;
            best_accuracy = report.accuracy;
            best_epoch = epoch;
            patience = 0;
            copy_file_atomic(&epoch_path, &out.join("model.json"))?;
        } else {
            patience += 1;
        }

        let state = EmotionTrainState {
            schema_version: EMOTION_TRAIN_STATE_SCHEMA_VERSION,
            current_epoch: epoch,
            best_epoch,
            best_val_loss: best_loss,
            best_val_accuracy: best_accuracy,
            patience,
            shuffle_seed: config.seed,
            shuffle_scheme: EMOTION_SHUFFLE_SCHEME.to_string(),
            next_shuffle_epoch: epoch + 1,
            effective_config: effective_config.clone(),
            requested_max_epochs: config.epochs,
            data_identity: data_identity.clone(),
            model_config: model_config.clone(),
            epoch_checkpoint: epoch_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_string(),
            best_checkpoint: format!("model-epoch-{best_epoch}.json"),
        };
        write_json_file_atomic(&out.join(format!("train_state-epoch-{epoch}.json")), &state)?;
        write_json_file_atomic(&out.join("train_state.json"), &state)?;
        completed_epoch = epoch;
        progress(TrainProgress::Epoch {
            epoch,
            train_loss,
            validation_loss: report.loss,
            validation_accuracy: report.accuracy,
            best_epoch,
            patience,
        });
        if patience >= config.early_stopping_patience {
            progress(TrainProgress::EarlyStopped { epoch, patience });
            break;
        }
    }

    Ok(EmotionTrainReport {
        mode,
        completed_epoch,
        best_epoch,
        best_loss,
        best_accuracy,
    })
}

pub fn evaluate(model_dir: &Path, data: &Path, split: &str) -> Result<EvalReport> {
    let model = load_model(model_dir)?;
    let rows = read_examples(&data.join(format!("{split}.jsonl")))?;
    Ok(evaluate_model(&model, &rows, split))
}

pub fn infer(model_dir: &Path, wav: &Path) -> Result<Vec<(String, f32)>> {
    let model = load_model(model_dir)?;
    let samples = read_wav_mono_resampled(wav, model.config.sample_rate_hz)?;
    let config = EmotionPrepareConfig {
        sample_rate_hz: model.config.sample_rate_hz,
        mel_bins: model.config.mel_bins.max(1),
        ..EmotionPrepareConfig::default()
    };
    let features = pooled_log_mel_features(&samples, &config);
    Ok(predict_scores(&model, &features))
}

pub fn load_model(model_dir: &Path) -> Result<EmotionModel> {
    read_json_file(&model_dir.join("model.json"))
}

pub fn read_examples(path: &Path) -> Result<Vec<EmotionExample>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line
            .with_context(|| format!("reading line {} from {}", line_index + 1, path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        rows.push(
            serde_json::from_str(&line).with_context(|| {
                format!("parsing line {} from {}", line_index + 1, path.display())
            })?,
        );
    }
    Ok(rows)
}

fn train_epoch(
    model: &mut EmotionModel,
    rows: &[EmotionExample],
    config: &EmotionTrainConfig,
    rng: &mut StdRng,
) -> f32 {
    let mut order = (0..rows.len()).collect::<Vec<_>>();
    order.shuffle(rng);
    let mut total_loss = 0.0f32;
    let mut seen = 0usize;
    for batch in order.chunks(config.batch_size.max(1)) {
        let mut grad_w = vec![vec![0.0; model.config.feature_dims]; model.config.labels.len()];
        let mut grad_b = vec![0.0; model.config.labels.len()];
        for &index in batch {
            let row = &rows[index];
            let x = normalized_features(model, &row.features);
            let probs = softmax(&logits(model, &x));
            total_loss += -probs[row.label_id].max(1e-9).ln();
            seen += 1;
            for label in 0..model.config.labels.len() {
                let delta = probs[label] - f32::from(label == row.label_id);
                grad_b[label] += delta;
                for (dim, value) in x.iter().enumerate() {
                    grad_w[label][dim] += delta * value;
                }
            }
        }
        let scale = 1.0 / batch.len() as f32;
        for label in 0..model.config.labels.len() {
            model.bias[label] -= config.learning_rate * grad_b[label] * scale;
            for (dim, gradient) in grad_w[label]
                .iter()
                .enumerate()
                .take(model.config.feature_dims)
            {
                let regularized =
                    gradient * scale + config.weight_decay * model.weights[label][dim];
                model.weights[label][dim] -= config.learning_rate * regularized;
            }
        }
    }
    total_loss / seen.max(1) as f32
}

fn evaluate_model(model: &EmotionModel, rows: &[EmotionExample], split: &str) -> EvalReport {
    let mut correct = 0usize;
    let mut loss = 0.0f32;
    let mut confusion = vec![vec![0usize; model.config.labels.len()]; model.config.labels.len()];
    for row in rows {
        let scores = predict_scores(model, &row.features);
        let predicted = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.1.total_cmp(&b.1))
            .map(|(index, _)| index)
            .unwrap_or(0);
        let probs = scores.into_iter().map(|(_, p)| p).collect::<Vec<_>>();
        loss += -probs[row.label_id].max(1e-9).ln();
        correct += usize::from(predicted == row.label_id);
        confusion[row.label_id][predicted] += 1;
    }
    EvalReport {
        split: split.to_string(),
        examples: rows.len(),
        accuracy: correct as f32 / rows.len().max(1) as f32,
        loss: loss / rows.len().max(1) as f32,
        labels: model.config.labels.clone(),
        confusion,
    }
}

fn predict_scores(model: &EmotionModel, features: &[f32]) -> Vec<(String, f32)> {
    let x = normalized_features(model, features);
    let probs = softmax(&logits(model, &x));
    let mut scores = model
        .config
        .labels
        .iter()
        .cloned()
        .zip(probs)
        .collect::<Vec<_>>();
    scores.sort_by(|(_, a), (_, b)| b.total_cmp(a));
    scores
}

fn logits(model: &EmotionModel, x: &[f32]) -> Vec<f32> {
    model
        .weights
        .iter()
        .zip(model.bias.iter())
        .map(|(weights, bias)| {
            weights
                .iter()
                .zip(x.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
                + bias
        })
        .collect()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = logits
        .iter()
        .map(|logit| (logit - max).exp())
        .collect::<Vec<_>>();
    let sum = exp.iter().sum::<f32>().max(1e-9);
    exp.into_iter().map(|value| value / sum).collect()
}

fn normalized_features(model: &EmotionModel, features: &[f32]) -> Vec<f32> {
    (0..model.config.feature_dims)
        .map(|index| {
            let value = *features.get(index).unwrap_or(&0.0);
            (value - model.mean[index]) / model.std[index].max(1e-6)
        })
        .collect()
}

fn feature_stats(rows: &[EmotionExample], dims: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0.0; dims];
    for row in rows {
        for (index, value) in row.features.iter().enumerate().take(dims) {
            mean[index] += value;
        }
    }
    for value in &mut mean {
        *value /= rows.len().max(1) as f32;
    }
    let mut var = vec![0.0; dims];
    for row in rows {
        for (index, value) in row.features.iter().enumerate().take(dims) {
            let delta = value - mean[index];
            var[index] += delta * delta;
        }
    }
    let std = var
        .into_iter()
        .map(|value| (value / rows.len().max(1) as f32).sqrt().max(1e-4))
        .collect();
    (mean, std)
}

fn read_source_manifest(path: &Path) -> Result<Vec<EmotionSourceRow>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line
            .with_context(|| format!("reading line {} from {}", line_index + 1, path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!("parsing line {} from {}", line_index + 1, path.display()))?;
        let emotion = value
            .get("emotion")
            .and_then(|value| value.as_str())
            .context("source row missing emotion")?
            .to_string();
        let wav_path = value
            .get("path")
            .and_then(|value| value.as_str())
            .context("source row missing path")?;
        let key = (emotion.clone(), wav_path.to_string());
        if !seen.insert(key) {
            continue;
        }
        rows.push(EmotionSourceRow {
            emotion,
            path: PathBuf::from(wav_path),
            speaker: value
                .get("speaker")
                .and_then(|value| value.as_str())
                .map(str::to_string),
        });
    }
    anyhow::ensure!(!rows.is_empty(), "no rows found in {}", path.display());
    Ok(rows)
}

fn labels_from_sources(sources: &[EmotionSourceRow]) -> Vec<String> {
    sources
        .iter()
        .map(|row| row.emotion.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[derive(Debug, Clone)]
struct SplitGroups {
    train: BTreeSet<String>,
    valid: BTreeSet<String>,
    test: BTreeSet<String>,
}

fn source_group_id(source: &EmotionSourceRow) -> String {
    format!(
        "{}|{}",
        source.speaker.as_deref().unwrap_or("_"),
        source.path.display()
    )
}

fn split_source_groups(
    sources: &[EmotionSourceRow],
    config: &EmotionPrepareConfig,
    rng: &mut StdRng,
) -> SplitGroups {
    let mut groups = sources
        .iter()
        .map(source_group_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    groups.shuffle(rng);
    let n = groups.len();
    let train_end = (n as f64 * config.train_frac).round() as usize;
    let valid_end = (train_end + (n as f64 * config.valid_frac).round() as usize).min(n);
    SplitGroups {
        train: groups[..train_end.min(n)].iter().cloned().collect(),
        valid: groups[train_end.min(n)..valid_end]
            .iter()
            .cloned()
            .collect(),
        test: groups[valid_end..].iter().cloned().collect(),
    }
}

fn labels_from_examples(train: &[EmotionExample], valid: &[EmotionExample]) -> Vec<String> {
    train
        .iter()
        .chain(valid.iter())
        .map(|row| row.emotion.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn random_cuts(
    samples: &[f32],
    config: &EmotionPrepareConfig,
    rng: &mut StdRng,
) -> Vec<(usize, usize)> {
    let min_samples = ms_to_samples(config.min_cut_ms, config.sample_rate_hz).max(1);
    let max_samples = ms_to_samples(config.max_cut_ms, config.sample_rate_hz).max(min_samples);
    let mut cuts = Vec::new();
    for _ in 0..config.cuts_per_wav {
        let duration = rng.gen_range(min_samples..=max_samples).min(samples.len());
        let start = if samples.len() > duration {
            rng.gen_range(0..=(samples.len() - duration))
        } else {
            0
        };
        cuts.push((start, start + duration));
    }
    cuts
}

fn pooled_log_mel_features(samples: &[f32], config: &EmotionPrepareConfig) -> Vec<f32> {
    let interp = InterpretationConfig {
        sample_rate_hz: config.sample_rate_hz,
        mel_bins: config.mel_bins,
        compact_audio_features: false,
        ..InterpretationConfig::default()
    };
    let frames = tongues_interpretation::log_mel_features(samples, &interp);
    if frames.is_empty() {
        return vec![0.0; config.mel_bins * 2];
    }
    let bins = config.mel_bins;
    let mut mean = vec![0.0; bins];
    for frame in &frames {
        for (index, value) in frame.iter().enumerate().take(bins) {
            mean[index] += value;
        }
    }
    for value in &mut mean {
        *value /= frames.len() as f32;
    }
    let mut std = vec![0.0; bins];
    for frame in &frames {
        for (index, value) in frame.iter().enumerate().take(bins) {
            let delta = value - mean[index];
            std[index] += delta * delta;
        }
    }
    for value in &mut std {
        *value = (*value / frames.len() as f32).sqrt();
    }
    mean.extend(std);
    mean
}

fn read_wav_mono_resampled(path: &Path, target_rate: u32) -> Result<Vec<f32>> {
    let audio =
        tongues_audio::read_wav(path).with_context(|| format!("opening {}", path.display()))?;
    let mono = audio.to_mono().map_err(anyhow::Error::from)?;
    Ok(tongues_audio::AudioBuffer {
        samples: mono,
        sample_rate_hz: audio.sample_rate_hz,
        channels: 1,
    }
    .resample_linear(target_rate)
    .map_err(anyhow::Error::from)?
    .samples)
}

fn write_jsonl_split(
    out: &Path,
    split: &str,
    rows: &[EmotionExample],
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<()> {
    let path = out.join(format!("{split}.jsonl"));
    let part = path.with_extension("jsonl.part");
    let file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut writer = BufWriter::new(file);
    for row in rows {
        serde_json::to_writer(&mut writer, row)?;
        writeln!(writer)?;
    }
    writer
        .flush()
        .with_context(|| format!("flushing {}", part.display()))?;
    fs::rename(&part, &path)
        .with_context(|| format!("renaming {} to {}", part.display(), path.display()))?;
    progress(PrepareProgress::Write {
        split: split.to_string(),
        rows: rows.len(),
        path: path.display().to_string(),
    });
    Ok(())
}

fn write_prepare_state(
    out: &Path,
    config: &EmotionPrepareConfig,
    status: &str,
    report: Option<&PrepareReport>,
) -> Result<()> {
    let state = PrepareCheckpointState {
        status: status.to_string(),
        dataset_id: config.dataset_id.clone(),
        source_manifest: config.source_manifest.display().to_string(),
        report: report.cloned(),
    };
    write_json_file_atomic(&out.join("prepare_state.json"), &state)
}

fn write_readme(out: &Path, config: &EmotionPrepareConfig, labels: &[String]) -> Result<()> {
    let text = format!(
        "# Emotion cuts dataset\n\nDataset id: `{}`\nSource manifest: `{}`\nFeature kind: `{}`\nLabels: {}\n\nPrepared by `tongues emotions prepare`. Each row is a random or full-length WAV cut represented as pooled log-mel mean and standard deviation features.\n\nSplit policy: group-aware by `speaker + source WAV path`; all cuts from one WAV stay in exactly one split.\n",
        config.dataset_id,
        config.source_manifest.display(),
        feature_kind(config),
        labels.join(", ")
    );
    fs::write(out.join("README.md"), text).context("writing emotions README")
}

fn emotion_data_identity(data: &Path) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(b"tongues-emotions-data-v1\0");
    for name in ["train.jsonl", "valid.jsonl"] {
        let path = data.join(name);
        let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        digest.update(name.as_bytes());
        digest.update([0]);
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn emotion_epoch_rng(seed: u64, epoch: usize) -> StdRng {
    let mut digest = Sha256::new();
    digest.update(b"tongues-emotions-shuffle-v1\0");
    digest.update(seed.to_le_bytes());
    digest.update((epoch as u64).to_le_bytes());
    let bytes: [u8; 32] = digest.finalize().into();
    StdRng::from_seed(bytes)
}

fn emotion_training_artifacts_exist(out: &Path) -> Result<bool> {
    if !out.exists() {
        return Ok(false);
    }
    for entry in fs::read_dir(out).with_context(|| format!("reading {}", out.display()))? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if is_emotion_training_artifact(&name) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn is_emotion_training_artifact(name: &str) -> bool {
    matches!(
        name,
        "manifest.json"
            | "model.json"
            | "model.json.part"
            | "model_config.json"
            | "model_config.json.part"
            | "train_config.json"
            | "train_config.json.part"
            | "train_state.json"
            | "train_state.json.part"
    ) || (name.starts_with("model-epoch-")
        && (name.ends_with(".json") || name.ends_with(".json.part")))
        || (name.starts_with("train_state-epoch-")
            && (name.ends_with(".json") || name.ends_with(".json.part")))
}

fn remove_emotion_training_artifacts(out: &Path) -> Result<()> {
    if !out.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(out).with_context(|| format!("reading {}", out.display()))? {
        let entry = entry?;
        let name = entry.file_name();
        if is_emotion_training_artifact(&name.to_string_lossy()) {
            fs::remove_file(entry.path())
                .with_context(|| format!("removing {}", entry.path().display()))?;
        }
    }
    Ok(())
}

fn load_emotion_resume(
    out: &Path,
    effective_config: &EmotionResumeConfig,
    data_identity: &str,
    model_config: &EmotionModelConfig,
) -> Result<(EmotionTrainState, EmotionModel)> {
    let mut epochs = Vec::new();
    if out.exists() {
        for entry in fs::read_dir(out).with_context(|| format!("reading {}", out.display()))? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(epoch) = parse_epoch_state_name(&name) {
                epochs.push(epoch);
            }
        }
    }
    epochs.sort_unstable_by(|a, b| b.cmp(a));

    for epoch in epochs {
        let state_path = out.join(format!("train_state-epoch-{epoch}.json"));
        let model_path = out.join(format!("model-epoch-{epoch}.json"));
        if !model_path.is_file() {
            continue;
        }
        let Ok(state) = read_json_file::<EmotionTrainState>(&state_path) else {
            continue;
        };
        if state.current_epoch != epoch
            || state.epoch_checkpoint != format!("model-epoch-{epoch}.json")
        {
            continue;
        }
        validate_resume_compatibility(out, &state, effective_config, data_identity, model_config)?;
        let model = read_json_file::<EmotionModel>(&model_path)?;
        anyhow::ensure!(
            model.config == *model_config,
            "cannot resume emotion training in {}: epoch {} model metadata does not match the prepared data; migrate the run or use a new output directory (or --restart to replace it)",
            out.display(),
            epoch
        );
        return Ok((state, model));
    }

    if emotion_training_artifacts_exist(out)? {
        anyhow::bail!(
            "cannot resume emotion training in {}: no complete versioned epoch checkpoint was found; this may be a legacy or incomplete run. Migrate it or use a new output directory (or --restart to replace it)",
            out.display()
        );
    }
    anyhow::bail!(
        "cannot resume emotion training in {}: no prior run exists; omit --resume to start a new run",
        out.display()
    )
}

fn parse_epoch_state_name(name: &str) -> Option<usize> {
    name.strip_prefix("train_state-epoch-")?
        .strip_suffix(".json")?
        .parse()
        .ok()
}

fn validate_resume_compatibility(
    out: &Path,
    state: &EmotionTrainState,
    effective_config: &EmotionResumeConfig,
    data_identity: &str,
    model_config: &EmotionModelConfig,
) -> Result<()> {
    let compatible = state.schema_version == EMOTION_TRAIN_STATE_SCHEMA_VERSION
        && state.effective_config == *effective_config
        && state.data_identity == data_identity
        && state.model_config == *model_config
        && state.shuffle_seed == effective_config.seed
        && state.shuffle_scheme == EMOTION_SHUFFLE_SCHEME
        && state.next_shuffle_epoch == state.current_epoch + 1;
    anyhow::ensure!(
        compatible,
        "cannot resume emotion training in {}: saved state schema, data, or effective training configuration is incompatible. Migrate the run or use a new output directory (or --restart to replace it)",
        out.display()
    );
    Ok(())
}

fn restore_best_model(out: &Path, state: &EmotionTrainState) -> Result<()> {
    let expected = format!("model-epoch-{}.json", state.best_epoch);
    anyhow::ensure!(
        state.best_checkpoint == expected,
        "cannot resume emotion training in {}: best checkpoint metadata is inconsistent; migrate the run or use a new output directory",
        out.display()
    );
    let best_path = out.join(&state.best_checkpoint);
    anyhow::ensure!(
        best_path.is_file(),
        "cannot resume emotion training in {}: recorded best checkpoint {} is missing; migrate the run or use a new output directory",
        out.display(),
        best_path.display()
    );
    copy_file_atomic(&best_path, &out.join("model.json"))
}

fn copy_file_atomic(source: &Path, destination: &Path) -> Result<()> {
    let bytes = fs::read(source).with_context(|| format!("reading {}", source.display()))?;
    let part = destination.with_extension(format!(
        "{}.part",
        destination
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("json")
    ));
    let mut file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    file.write_all(&bytes)
        .with_context(|| format!("writing {}", part.display()))?;
    file.flush()
        .with_context(|| format!("flushing {}", part.display()))?;
    file.sync_all()
        .with_context(|| format!("syncing {}", part.display()))?;
    fs::rename(&part, destination)
        .with_context(|| format!("renaming {} to {}", part.display(), destination.display()))
}

fn write_json_file_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let part = path.with_extension(format!(
        "{}.part",
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or("json")
    ));
    let file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .with_context(|| format!("writing {}", part.display()))?;
    writeln!(writer)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&part, path)
        .with_context(|| format!("renaming {} to {}", part.display(), path.display()))
}

fn read_json_file<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn validate_prepare_config(config: &EmotionPrepareConfig) -> Result<()> {
    anyhow::ensure!(
        (0.0..=1.0).contains(&config.train_frac),
        "train_frac must be in 0..=1"
    );
    anyhow::ensure!(
        (0.0..=1.0).contains(&config.valid_frac),
        "valid_frac must be in 0..=1"
    );
    anyhow::ensure!(
        config.train_frac + config.valid_frac < 1.0,
        "train_frac + valid_frac must leave a test split"
    );
    anyhow::ensure!(config.cuts_per_wav > 0, "cuts_per_wav must be positive");
    anyhow::ensure!(
        config.min_cut_ms > 0 && config.max_cut_ms >= config.min_cut_ms,
        "cut duration bounds are invalid"
    );
    anyhow::ensure!(config.mel_bins > 0, "mel_bins must be positive");
    Ok(())
}

fn validate_train_config(config: &EmotionTrainConfig) -> Result<()> {
    anyhow::ensure!(config.batch_size > 0, "batch_size must be positive");
    anyhow::ensure!(config.epochs > 0, "epochs must be positive");
    anyhow::ensure!(
        config.early_stopping_patience > 0,
        "early_stopping_patience must be positive"
    );
    anyhow::ensure!(
        config.learning_rate.is_finite() && config.learning_rate > 0.0,
        "learning_rate must be positive"
    );
    Ok(())
}

fn feature_kind(config: &EmotionPrepareConfig) -> String {
    format!("pooled-log-mel-mean-std/{}bins", config.mel_bins)
}

fn ms_to_samples(ms: u64, sample_rate: u32) -> usize {
    ((ms as u128 * u128::from(sample_rate)) / 1_000) as usize
}

fn samples_to_ms(samples: usize, sample_rate: u32) -> u64 {
    ((samples as u128 * 1_000) / u128::from(sample_rate)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn softmax_sums_to_one() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let sum = probs.iter().sum::<f32>();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn random_cuts_stay_in_bounds() {
        let config = EmotionPrepareConfig {
            cuts_per_wav: 16,
            min_cut_ms: 100,
            max_cut_ms: 500,
            sample_rate_hz: 1_000,
            ..EmotionPrepareConfig::default()
        };
        let samples = vec![0.0; 1_000];
        let mut rng = StdRng::seed_from_u64(1);
        for (start, end) in random_cuts(&samples, &config, &mut rng) {
            assert!(start < end);
            assert!(end <= samples.len());
        }
    }

    #[test]
    fn split_source_groups_keeps_group_in_one_split() {
        let sources = vec![
            EmotionSourceRow {
                emotion: "happy".to_string(),
                path: PathBuf::from("a.wav"),
                speaker: Some("s1".to_string()),
            },
            EmotionSourceRow {
                emotion: "sad".to_string(),
                path: PathBuf::from("b.wav"),
                speaker: Some("s2".to_string()),
            },
            EmotionSourceRow {
                emotion: "angry".to_string(),
                path: PathBuf::from("c.wav"),
                speaker: Some("s3".to_string()),
            },
        ];
        let mut rng = StdRng::seed_from_u64(7);
        let split = split_source_groups(&sources, &EmotionPrepareConfig::default(), &mut rng);
        for source in &sources {
            let group = source_group_id(source);
            let placements = usize::from(split.train.contains(&group))
                + usize::from(split.valid.contains(&group))
                + usize::from(split.test.contains(&group));
            assert_eq!(placements, 1);
        }
    }

    #[test]
    fn interrupted_resume_matches_uninterrupted_training() {
        let root = test_dir("resume-determinism");
        let data = root.join("data");
        let resumed = root.join("resumed");
        let uninterrupted = root.join("uninterrupted");
        write_training_fixture(&data);

        let first_config = fixture_train_config(2);
        train(&data, &resumed, &first_config, EmotionTrainMode::New).unwrap();
        let interrupted_state: EmotionTrainState =
            read_json_file(&resumed.join("train_state.json")).unwrap();

        let resumed_config = fixture_train_config(4);
        let mut events = Vec::new();
        let resumed_report = train_with_progress(
            &data,
            &resumed,
            &resumed_config,
            EmotionTrainMode::Resume,
            |event| events.push(event),
        )
        .unwrap();
        let uninterrupted_report = train(
            &data,
            &uninterrupted,
            &resumed_config,
            EmotionTrainMode::New,
        )
        .unwrap();

        assert!(matches!(
            events.first(),
            Some(TrainProgress::Started {
                mode: EmotionTrainMode::Resume,
                start_epoch: 3,
                max_epochs: 4
            })
        ));
        let resumed_model: EmotionModel =
            read_json_file(&resumed.join("model-epoch-4.json")).unwrap();
        let uninterrupted_model: EmotionModel =
            read_json_file(&uninterrupted.join("model-epoch-4.json")).unwrap();
        assert_eq!(resumed_model, uninterrupted_model);
        assert_eq!(resumed_report.best_epoch, uninterrupted_report.best_epoch);
        assert_eq!(resumed_report.best_loss, uninterrupted_report.best_loss);
        assert_eq!(interrupted_state.next_shuffle_epoch, 3);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn best_epoch_identifies_the_model_copied_to_model_json() {
        let root = test_dir("best-model");
        let data = root.join("data");
        let out = root.join("model");
        write_training_fixture(&data);

        train(&data, &out, &fixture_train_config(5), EmotionTrainMode::New).unwrap();
        let state: EmotionTrainState = read_json_file(&out.join("train_state.json")).unwrap();
        let best: EmotionModel = read_json_file(&out.join("model.json")).unwrap();
        let checkpoint: EmotionModel =
            read_json_file(&out.join(format!("model-epoch-{}.json", state.best_epoch))).unwrap();

        assert_eq!(
            state.best_checkpoint,
            format!("model-epoch-{}.json", state.best_epoch)
        );
        assert_eq!(best, checkpoint);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn incompatible_resume_rejects_config_and_data_changes() {
        let root = test_dir("incompatible-resume");
        let data = root.join("data");
        let out = root.join("model");
        write_training_fixture(&data);
        let config = fixture_train_config(2);
        train(&data, &out, &config, EmotionTrainMode::New).unwrap();

        let mut changed_config = fixture_train_config(3);
        changed_config.learning_rate *= 0.5;
        let config_error = train(&data, &out, &changed_config, EmotionTrainMode::Resume)
            .unwrap_err()
            .to_string();
        assert!(config_error.contains("incompatible"));
        assert!(config_error.contains("new output directory"));

        let mut rows = read_examples(&data.join("train.jsonl")).unwrap();
        rows[0].features[0] += 1.0;
        write_examples(&data.join("train.jsonl"), &rows);
        let data_error = train(
            &data,
            &out,
            &fixture_train_config(3),
            EmotionTrainMode::Resume,
        )
        .unwrap_err()
        .to_string();
        assert!(data_error.contains("incompatible"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resume_uses_latest_complete_model_and_state_pair() {
        let root = test_dir("incomplete-recovery");
        let data = root.join("data");
        let out = root.join("model");
        write_training_fixture(&data);
        train(&data, &out, &fixture_train_config(2), EmotionTrainMode::New).unwrap();
        fs::remove_file(out.join("model-epoch-2.json")).unwrap();

        let mut events = Vec::new();
        train_with_progress(
            &data,
            &out,
            &fixture_train_config(3),
            EmotionTrainMode::Resume,
            |event| events.push(event),
        )
        .unwrap();

        assert!(matches!(
            events.first(),
            Some(TrainProgress::Started {
                mode: EmotionTrainMode::Resume,
                start_epoch: 2,
                ..
            })
        ));
        assert!(out.join("model-epoch-2.json").is_file());
        assert!(out.join("model-epoch-3.json").is_file());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn existing_runs_require_explicit_resume_or_restart() {
        let root = test_dir("explicit-mode");
        let data = root.join("data");
        let out = root.join("model");
        write_training_fixture(&data);
        train(&data, &out, &fixture_train_config(2), EmotionTrainMode::New).unwrap();

        let error = train(&data, &out, &fixture_train_config(3), EmotionTrainMode::New)
            .unwrap_err()
            .to_string();
        assert!(error.contains("--resume"));
        assert!(error.contains("--restart"));

        let report = train(
            &data,
            &out,
            &fixture_train_config(1),
            EmotionTrainMode::Restart,
        )
        .unwrap();
        assert_eq!(report.mode, EmotionTrainMode::Restart);
        assert_eq!(report.completed_epoch, 1);
        assert!(!out.join("model-epoch-2.json").exists());

        fs::remove_dir_all(root).unwrap();
    }

    fn fixture_train_config(epochs: usize) -> EmotionTrainConfig {
        EmotionTrainConfig {
            learning_rate: 0.02,
            weight_decay: 1e-4,
            batch_size: 2,
            epochs,
            early_stopping_patience: 20,
            seed: 17,
        }
    }

    fn write_training_fixture(data: &Path) {
        fs::create_dir_all(data).unwrap();
        let row = |id: &str, emotion: &str, label_id: usize, features: Vec<f32>| EmotionExample {
            id: id.to_string(),
            emotion: emotion.to_string(),
            label_id,
            source_path: format!("{id}.wav"),
            start_ms: 0,
            duration_ms: 100,
            sample_rate_hz: 16_000,
            feature_kind: "pooled-log-mel-mean-std/2bins".to_string(),
            features,
        };
        let train_rows = vec![
            row("happy-a", "happy", 0, vec![1.0, 0.8, 0.2, 0.1]),
            row("sad-a", "sad", 1, vec![-1.0, -0.8, -0.2, -0.1]),
            row("happy-b", "happy", 0, vec![0.9, 1.1, 0.1, 0.2]),
            row("sad-b", "sad", 1, vec![-0.9, -1.1, -0.1, -0.2]),
            row("happy-c", "happy", 0, vec![1.2, 0.7, 0.3, 0.0]),
            row("sad-c", "sad", 1, vec![-1.2, -0.7, -0.3, 0.0]),
        ];
        let valid_rows = vec![
            row("happy-valid", "happy", 0, vec![1.0, 0.9, 0.2, 0.1]),
            row("sad-valid", "sad", 1, vec![-1.0, -0.9, -0.2, -0.1]),
        ];
        write_examples(&data.join("train.jsonl"), &train_rows);
        write_examples(&data.join("valid.jsonl"), &valid_rows);
    }

    fn write_examples(path: &Path, rows: &[EmotionExample]) {
        let file = File::create(path).unwrap();
        let mut writer = BufWriter::new(file);
        for row in rows {
            serde_json::to_writer(&mut writer, row).unwrap();
            writeln!(writer).unwrap();
        }
        writer.flush().unwrap();
    }

    fn test_dir(slug: &str) -> PathBuf {
        let sequence = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "tongues-emotions-{slug}-{}-{sequence}",
            std::process::id()
        ))
    }
}
