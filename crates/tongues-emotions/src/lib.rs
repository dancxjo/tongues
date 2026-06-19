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
use tongues_interpretation::{InterpretationConfig, DEFAULT_MEL_BINS, DEFAULT_SAMPLE_RATE_HZ};
use tongues_neural::{write_manifest, ModelArtifactManifest, TrainState};

pub const FAMILY: &str = "emotions";
pub const ARCHITECTURE: &str = "pooled-log-mel-softmax";
pub const DEFAULT_DATASET_ID: &str = "emotion-cuts-v0";
pub const DEFAULT_SOURCE_MANIFEST: &str = "style_vectors.jsonl";

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
            early_stopping_patience: 8,
            seed: 0,
        }
    }
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
    let mut examples = Vec::new();
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
    examples.shuffle(&mut rng);
    let n = examples.len();
    let train_end = (n as f64 * config.train_frac).round() as usize;
    let valid_end = (train_end + (n as f64 * config.valid_frac).round() as usize).min(n);
    let train = examples[..train_end.min(n)].to_vec();
    let valid = examples[train_end.min(n)..valid_end].to_vec();
    let test = examples[valid_end..].to_vec();

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

pub fn train(data: &Path, out: &Path, config: &EmotionTrainConfig) -> Result<f32> {
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
    write_json_file_atomic(&out.join("model_config.json"), &model_config)?;
    write_json_file_atomic(&out.join("train_config.json"), config)?;
    write_manifest(
        out,
        &ModelArtifactManifest::new(FAMILY, ARCHITECTURE, data.display().to_string()),
    )?;

    let mut model = EmotionModel {
        config: model_config,
        weights: vec![vec![0.0; feature_dims]; labels.len()],
        bias: vec![0.0; labels.len()],
        mean,
        std,
    };
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut best_loss = f32::INFINITY;
    let mut patience = 0usize;

    for epoch in 1..=config.epochs {
        let train_loss = train_epoch(&mut model, &train_rows, config, &mut rng);
        let report = evaluate_model(&model, &valid_rows, "valid");
        let epoch_path = out.join(format!("model-epoch-{epoch}.json"));
        write_json_file_atomic(&epoch_path, &model)?;
        let state = TrainState {
            current_epoch: epoch,
            best_val_loss: best_loss.min(report.loss),
            best_epoch: None,
            best_exact_match: Some(report.accuracy),
            early_stop_metric: "val_loss".to_string(),
        };
        write_json_file_atomic(&out.join("train_state.json"), &state)?;
        println!(
            "Epoch {epoch} | train_loss={train_loss:.4} val_loss={:.4} val_acc={:.3}",
            report.loss, report.accuracy
        );
        if report.loss < best_loss {
            best_loss = report.loss;
            patience = 0;
            write_json_file_atomic(&out.join("model.json"), &model)?;
        } else {
            patience += 1;
            if patience >= config.early_stopping_patience {
                break;
            }
        }
    }
    Ok(best_loss)
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
            for dim in 0..model.config.feature_dims {
                let regularized =
                    grad_w[label][dim] * scale + config.weight_decay * model.weights[label][dim];
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
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    anyhow::ensure!(spec.channels > 0, "{} has zero channels", path.display());
    let channels = usize::from(spec.channels);
    let mut raw = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                raw.push(sample?);
            }
        }
        hound::SampleFormat::Int => {
            let denom = ((1i64 << (spec.bits_per_sample.saturating_sub(1))) - 1).max(1) as f32;
            for sample in reader.samples::<i32>() {
                raw.push(sample? as f32 / denom);
            }
        }
    }
    let mono = if channels == 1 {
        raw
    } else {
        raw.chunks_exact(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    if spec.sample_rate == target_rate {
        return Ok(mono);
    }
    Ok(resample_linear(&mono, spec.sample_rate, target_rate))
}

fn resample_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate == target_rate {
        return samples.to_vec();
    }
    let out_len = ((samples.len() as f64 * f64::from(target_rate)) / f64::from(source_rate))
        .round()
        .max(1.0) as usize;
    (0..out_len)
        .map(|index| {
            let pos = index as f64 * f64::from(source_rate) / f64::from(target_rate);
            let left = pos.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let frac = (pos - left as f64) as f32;
            samples[left] * (1.0 - frac) + samples[right] * frac
        })
        .collect()
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
        "# Emotion cuts dataset\n\nDataset id: `{}`\nSource manifest: `{}`\nFeature kind: `{}`\nLabels: {}\n\nPrepared by `tongues emotions prepare`. Each row is a random or full-length WAV cut represented as pooled log-mel mean and standard deviation features.\n",
        config.dataset_id,
        config.source_manifest.display(),
        feature_kind(config),
        labels.join(", ")
    );
    fs::write(out.join("README.md"), text).context("writing emotions README")
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
}
