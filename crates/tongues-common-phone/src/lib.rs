//! Common Phone compact-acoustic-frame model family.
//!
//! V0 prepares local Common Phone style exports into mechanical compact
//! acoustic frames and ordered phone / phonetic-feature targets. The training
//! artifact is intentionally small and CPU-friendly: it records CTC-head
//! metadata and a frequency baseline while the durable data path settles.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use tongues_core::Vocab;
use tongues_neural::{write_manifest, ModelArtifactManifest};

pub const FAMILY: &str = "common-phone";
pub const ARCHITECTURE: &str = "common-phone-compact-frame-ctc-v0";
pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 16_000;
pub const DEFAULT_FRAME_HZ: u32 = 100;
pub const DEFAULT_MEL_BINS: usize = 80;
pub const COMPACT_AUDIO_EXTRA_BINS: usize = 7;
pub const DEFAULT_COMPACT_AUDIO_FEATURE_BINS: usize =
    DEFAULT_MEL_BINS + DEFAULT_MEL_BINS + COMPACT_AUDIO_EXTRA_BINS;
pub const CTC_BLANK: &str = "<CTC_BLANK>";
pub const UNK: &str = "<UNK>";
pub const NONE: &str = "none";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommonPhoneConfig {
    pub dataset_id: String,
    pub input: String,
    pub languages: Vec<String>,
    pub max_utterances: Option<usize>,
    pub sample_rate_hz: u32,
    pub valid_ratio: f64,
    pub test_ratio: f64,
    pub seed: u64,
    pub frame_hz: u32,
    pub feature_bins: usize,
}

impl Default for CommonPhoneConfig {
    fn default() -> Self {
        Self {
            dataset_id: "common-phone-v0".to_string(),
            input: ".".to_string(),
            languages: Vec::new(),
            max_utterances: None,
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            valid_ratio: 0.05,
            test_ratio: 0.05,
            seed: 42,
            frame_hz: DEFAULT_FRAME_HZ,
            feature_bins: DEFAULT_COMPACT_AUDIO_FEATURE_BINS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommonPhoneTrainConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub seed: u64,
    pub phone_ctc_loss_weight: f32,
    pub phoneme_ctc_loss_weight: f32,
    pub feature_axis_ctc_loss_weight: f32,
}

impl Default for CommonPhoneTrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            batch_size: 8,
            epochs: 3,
            seed: 42,
            phone_ctc_loss_weight: 1.0,
            phoneme_ctc_loss_weight: 0.5,
            feature_axis_ctc_loss_weight: 0.35,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: String,
    pub input_feature_bins: usize,
    pub frame_hz: u32,
    pub phone_vocab_size: usize,
    pub phoneme_vocab_size: usize,
    pub feature_axis_vocab_sizes: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommonPhoneRow {
    pub row_source: String,
    pub utterance_id: String,
    pub lang: String,
    pub variety: Option<String>,
    pub speaker_id: Option<String>,
    pub audio_path: String,
    pub feature_path: String,
    pub sample_rate: u32,
    pub frame_hz: u32,
    pub duration_sec: f32,
    pub phones: Vec<String>,
    pub phonemes: Vec<String>,
    pub feature_targets: BTreeMap<String, Vec<String>>,
    pub raw: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrepareReport {
    pub utterances: usize,
    pub train_examples: usize,
    pub valid_examples: usize,
    pub test_examples: usize,
    pub feature_bins: usize,
    pub unknown_phone_symbols: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrepareProgress {
    Stage {
        message: String,
    },
    Parse {
        rows: usize,
        path: String,
    },
    Features {
        utterance_id: String,
        frames: usize,
        path: String,
    },
    Reuse {
        utterance_id: String,
        frames: usize,
        path: String,
    },
    Write {
        path: String,
        rows: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainReport {
    pub epochs: usize,
    pub train_examples: usize,
    pub valid_examples: usize,
    pub best_validation_phone_ter: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    pub split: String,
    pub examples: usize,
    pub phone_token_error_rate: f64,
    pub phoneme_token_error_rate: Option<f64>,
    pub feature_axis_token_error_rate: BTreeMap<String, f64>,
    pub aggregate_feature_token_error_rate: f64,
    pub unknown_phone_symbols: BTreeMap<String, usize>,
    pub language_distribution: BTreeMap<String, usize>,
    pub samples: Vec<GreedySample>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GreedySample {
    pub utterance_id: String,
    pub lang: String,
    pub phone_target: Vec<String>,
    pub phone_prediction: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShowRow {
    pub utterance_id: String,
    pub lang: String,
    pub phones: Vec<String>,
    pub feature_targets: BTreeMap<String, Vec<String>>,
    pub feature_shape: (usize, usize),
    pub first_frames: Vec<Vec<f32>>,
    pub mean: f32,
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct InputRecord {
    utterance_id: Option<String>,
    id: Option<String>,
    lang: Option<String>,
    language: Option<String>,
    variety: Option<String>,
    speaker_id: Option<String>,
    speaker: Option<String>,
    audio_path: Option<String>,
    path: Option<String>,
    wav: Option<String>,
    phones: Option<PhoneField>,
    phonemes: Option<PhoneField>,
    segments: Option<serde_json::Value>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
enum PhoneField {
    Text(String),
    Tokens(Vec<String>),
}

pub fn prepare_dataset(out: &Path, config: &CommonPhoneConfig) -> Result<PrepareReport> {
    prepare_dataset_with_progress(out, config, |_| {})
}

pub fn prepare_dataset_with_progress(
    out: &Path,
    config: &CommonPhoneConfig,
    mut progress: impl FnMut(PrepareProgress),
) -> Result<PrepareReport> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    fs::create_dir_all(out.join("features")).context("creating common-phone features directory")?;
    write_prepare_state(out, "starting", config, 0, None)?;

    let input = Path::new(&config.input);
    let input_rows = read_input_records(input, &mut progress)?;
    let allowed_langs = config
        .languages
        .iter()
        .map(|lang| lang.to_lowercase())
        .collect::<BTreeSet<_>>();
    let mut selected = Vec::new();
    for (index, record) in input_rows.into_iter().enumerate() {
        let lang = record
            .lang
            .clone()
            .or(record.language.clone())
            .unwrap_or_else(|| "und".to_string());
        if !allowed_langs.is_empty() && !allowed_langs.contains(&lang.to_lowercase()) {
            continue;
        }
        selected.push((index, record));
        if selected.len() == config.max_utterances.unwrap_or(usize::MAX) {
            break;
        }
    }
    anyhow::ensure!(
        !selected.is_empty(),
        "no Common Phone rows selected from {}",
        input.display()
    );

    let mut rows = recover_rows(&out.join("utterances.jsonl"))?;
    let mut row_by_id = rows
        .iter()
        .map(|row| (row.utterance_id.clone(), row.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(out.join("utterances.jsonl"))?,
    );

    for (index, record) in selected {
        let utterance_id = record
            .utterance_id
            .clone()
            .or(record.id.clone())
            .unwrap_or_else(|| format!("common-phone-{index:08}"));
        if let Some(existing) = row_by_id.get(&utterance_id) {
            if let Some((frames, _bins)) =
                feature_file_shape(&out.join(&existing.feature_path)).ok()
            {
                progress(PrepareProgress::Reuse {
                    utterance_id,
                    frames,
                    path: out.join(&existing.feature_path).display().to_string(),
                });
                continue;
            }
        }
        let audio_rel = record
            .audio_path
            .clone()
            .or(record.path.clone())
            .or(record.wav.clone())
            .ok_or_else(|| anyhow::anyhow!("row {utterance_id} has no audio_path/path/wav"))?;
        let audio_path = resolve_input_path(input, &audio_rel);
        let phone_tokens = record
            .phones
            .as_ref()
            .map(phone_field_tokens)
            .unwrap_or_default();
        anyhow::ensure!(
            !phone_tokens.is_empty(),
            "row {utterance_id} has no phones target"
        );
        let phoneme_tokens = record
            .phonemes
            .as_ref()
            .map(phone_field_tokens)
            .unwrap_or_else(|| phone_tokens.clone());
        let (samples, source_rate) = read_wav_mono(&audio_path)?;
        let samples = resample_linear(&samples, source_rate, config.sample_rate_hz);
        let features = compact_audio_features(&samples, config);
        let rel_feature =
            PathBuf::from("features").join(format!("{}.acf.bin", sanitize_id(&utterance_id)));
        write_feature_file(&out.join(&rel_feature), &features, config.feature_bins)?;
        progress(PrepareProgress::Features {
            utterance_id: utterance_id.clone(),
            frames: features.len(),
            path: out.join(&rel_feature).display().to_string(),
        });

        let feature_targets = feature_targets_for_phones(&phone_tokens);
        let raw = serde_json::json!({
            "common_phone_record": record,
            "segments": record.segments,
        });
        let row = CommonPhoneRow {
            row_source: FAMILY.to_string(),
            utterance_id: utterance_id.clone(),
            lang: record
                .lang
                .clone()
                .or(record.language.clone())
                .unwrap_or_else(|| "und".to_string()),
            variety: record.variety.clone(),
            speaker_id: record.speaker_id.clone().or(record.speaker.clone()),
            audio_path: audio_path.display().to_string(),
            feature_path: rel_feature.display().to_string(),
            sample_rate: config.sample_rate_hz,
            frame_hz: config.frame_hz,
            duration_sec: samples.len() as f32 / config.sample_rate_hz as f32,
            phones: phone_tokens,
            phonemes: phoneme_tokens,
            feature_targets,
            raw,
        };
        writeln!(writer, "{}", serde_json::to_string(&row)?)?;
        writer.flush()?;
        row_by_id.insert(row.utterance_id.clone(), row.clone());
        rows.push(row);
    }
    writer.flush()?;
    write_prepare_state(out, "utterances", config, rows.len(), None)?;

    let mut shuffled = rows;
    shuffled.shuffle(&mut rand::rngs::StdRng::seed_from_u64(config.seed));
    let n = shuffled.len();
    let test_len = ((n as f64) * config.test_ratio).round().min(n as f64) as usize;
    let valid_len = ((n as f64) * config.valid_ratio)
        .round()
        .min(n.saturating_sub(test_len) as f64) as usize;
    let train_len = n.saturating_sub(valid_len + test_len);
    let train = shuffled[..train_len].to_vec();
    let valid = shuffled[train_len..train_len + valid_len].to_vec();
    let test = shuffled[train_len + valid_len..].to_vec();
    write_jsonl_atomic(&out.join("train.jsonl"), &train, &mut progress)?;
    write_jsonl_atomic(&out.join("valid.jsonl"), &valid, &mut progress)?;
    write_jsonl_atomic(&out.join("test.jsonl"), &test, &mut progress)?;

    let all = [&train[..], &valid[..], &test[..]].concat();
    let phone_vocab = build_token_vocab(all.iter().flat_map(|row| row.phones.iter()));
    let phoneme_vocab = build_token_vocab(all.iter().flat_map(|row| row.phonemes.iter()));
    let axis_vocabs = build_feature_axis_vocabs(&all);
    write_text_atomic(
        &out.join("vocab.json"),
        serde_json::to_string_pretty(&phone_vocab)?,
    )?;
    write_text_atomic(
        &out.join("phone_vocab.json"),
        serde_json::to_string_pretty(&phone_vocab)?,
    )?;
    write_text_atomic(
        &out.join("phoneme_vocab.json"),
        serde_json::to_string_pretty(&phoneme_vocab)?,
    )?;
    write_text_atomic(
        &out.join("feature_axis_vocabs.json"),
        serde_json::to_string_pretty(&axis_vocabs)?,
    )?;
    write_text_atomic(
        &out.join("dataset_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    let unknown_phone_symbols = count_unknown_phones(&all);
    write_text_atomic(
        &out.join("README.md"),
        dataset_readme(config, &unknown_phone_symbols),
    )?;
    let report = PrepareReport {
        utterances: n,
        train_examples: train.len(),
        valid_examples: valid.len(),
        test_examples: test.len(),
        feature_bins: config.feature_bins,
        unknown_phone_symbols,
    };
    write_prepare_state(out, "complete", config, n, Some(&report))?;
    Ok(report)
}

pub fn train(data: &Path, out: &Path, config: &CommonPhoneTrainConfig) -> Result<TrainReport> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let train_rows = read_examples(&data.join("train.jsonl"))?;
    let valid_rows = read_examples(&data.join("valid.jsonl"))?;
    let phone_vocab: Vocab = read_json(&data.join("phone_vocab.json"))?;
    let phoneme_vocab: Vocab = read_json(&data.join("phoneme_vocab.json"))?;
    let axis_vocabs: BTreeMap<String, Vocab> = read_json(&data.join("feature_axis_vocabs.json"))?;
    let feature_bins = first_feature_bins(data, &train_rows)?;
    let model_config = ModelConfig {
        architecture: ARCHITECTURE.to_string(),
        input_feature_bins: feature_bins,
        frame_hz: DEFAULT_FRAME_HZ,
        phone_vocab_size: phone_vocab.size(),
        phoneme_vocab_size: phoneme_vocab.size(),
        feature_axis_vocab_sizes: axis_vocabs
            .iter()
            .map(|(axis, vocab)| (axis.clone(), vocab.size()))
            .collect(),
    };
    save_artifact_files(out, data, &model_config, config)?;
    let baseline = BaselineModel::fit(&train_rows);
    let mut best = f64::INFINITY;
    for epoch in 1..=config.epochs {
        let epoch_report =
            evaluate_with_model(&baseline, &valid_rows, &format!("epoch-{epoch}"), 0);
        best = best.min(epoch_report.phone_token_error_rate);
        write_text_atomic(
            &out.join(format!("model-epoch-{epoch}.bin")),
            serde_json::to_string_pretty(&baseline)?,
        )?;
        write_text_atomic(
            &out.join("train_state.json"),
            &serde_json::to_string_pretty(&serde_json::json!({
                "epoch": epoch,
                "best_validation_phone_ter": best,
                "architecture": ARCHITECTURE,
                "checkpoint": format!("model-epoch-{epoch}.bin")
            }))?,
        )?;
    }
    write_text_atomic(
        &out.join("model.bin"),
        serde_json::to_string_pretty(&baseline)?,
    )?;
    Ok(TrainReport {
        epochs: config.epochs,
        train_examples: train_rows.len(),
        valid_examples: valid_rows.len(),
        best_validation_phone_ter: if best.is_finite() { best } else { 0.0 },
    })
}

pub fn evaluate(
    model_dir: &Path,
    data: &Path,
    split: &str,
    sample_limit: usize,
) -> Result<EvalReport> {
    let rows = read_examples(&data.join(format!("{split}.jsonl")))?;
    let model: BaselineModel = read_json(&model_dir.join("model.bin"))?;
    Ok(evaluate_with_model(&model, &rows, split, sample_limit))
}

pub fn show_row(data: &Path, index: usize) -> Result<ShowRow> {
    let rows = read_examples(&data.join("train.jsonl"))?;
    let row = rows
        .get(index)
        .ok_or_else(|| anyhow::anyhow!("no train row at index {index}"))?;
    let (frames, bins, values) = read_feature_file(&data.join(&row.feature_path))?;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0;
    for value in &values {
        min = min.min(*value);
        max = max.max(*value);
        sum += *value;
    }
    let first_frames = values
        .chunks(bins)
        .take(3)
        .map(|frame| frame.iter().copied().take(12).collect())
        .collect();
    Ok(ShowRow {
        utterance_id: row.utterance_id.clone(),
        lang: row.lang.clone(),
        phones: row.phones.clone(),
        feature_targets: row.feature_targets.clone(),
        feature_shape: (frames, bins),
        first_frames,
        mean: if values.is_empty() {
            0.0
        } else {
            sum / values.len() as f32
        },
        min: if min.is_finite() { min } else { 0.0 },
        max: if max.is_finite() { max } else { 0.0 },
    })
}

pub fn read_examples(path: &Path) -> Result<Vec<CommonPhoneRow>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        rows.push(serde_json::from_str(&line)?);
    }
    Ok(rows)
}

pub fn feature_file_shape(path: &Path) -> Result<(usize, usize)> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;
    let rows = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let bins = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    Ok((rows, bins))
}

fn read_input_records(
    input: &Path,
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<Vec<InputRecord>> {
    let jsonl = input.join("metadata.jsonl");
    let csv = input.join("metadata.csv");
    let tsv = input.join("metadata.tsv");
    if jsonl.exists() {
        let file = File::open(&jsonl)?;
        let mut rows = Vec::new();
        for line in BufReader::new(file).lines() {
            let line = line?;
            if !line.trim().is_empty() {
                rows.push(serde_json::from_str(&line)?);
            }
        }
        progress(PrepareProgress::Parse {
            rows: rows.len(),
            path: jsonl.display().to_string(),
        });
        return Ok(rows);
    }
    if csv.exists() {
        return read_delimited_records(&csv, ',', progress);
    }
    if tsv.exists() {
        return read_delimited_records(&tsv, '\t', progress);
    }
    anyhow::bail!(
        "unsupported Common Phone layout at {}; expected metadata.jsonl, metadata.csv, or metadata.tsv",
        input.display()
    );
}

fn read_delimited_records(
    path: &Path,
    delimiter: char,
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<Vec<InputRecord>> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .transpose()?
        .ok_or_else(|| anyhow::anyhow!("empty metadata file {}", path.display()))?;
    let columns = split_delimited_line(&header, delimiter);
    let mut rows = Vec::new();
    for line in lines {
        let fields = split_delimited_line(&line?, delimiter);
        if fields.iter().all(|value| value.trim().is_empty()) {
            continue;
        }
        let mut object = serde_json::Map::new();
        for (name, value) in columns.iter().zip(fields.iter()) {
            object.insert(name.clone(), serde_json::Value::String(value.clone()));
        }
        rows.push(serde_json::from_value(serde_json::Value::Object(object))?);
    }
    progress(PrepareProgress::Parse {
        rows: rows.len(),
        path: path.display().to_string(),
    });
    Ok(rows)
}

fn split_delimited_line(line: &str, delimiter: char) -> Vec<String> {
    line.split(delimiter)
        .map(|value| value.trim().trim_matches('"').to_string())
        .collect()
}

fn resolve_input_path(input: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        input.join(path)
    }
}

fn phone_field_tokens(field: &PhoneField) -> Vec<String> {
    match field {
        PhoneField::Tokens(tokens) => tokens.iter().filter(|s| !s.is_empty()).cloned().collect(),
        PhoneField::Text(text) => tokenize_phone_text(text),
    }
}

fn tokenize_phone_text(text: &str) -> Vec<String> {
    let cleaned = text
        .trim()
        .trim_matches('/')
        .trim_matches('[')
        .trim_matches(']')
        .replace(['.', '|'], " ");
    if cleaned.split_whitespace().count() > 1 {
        cleaned.split_whitespace().map(str::to_string).collect()
    } else {
        cleaned
            .chars()
            .filter(|ch| !ch.is_whitespace() && *ch != 'ˈ' && *ch != 'ˌ')
            .map(|ch| ch.to_string())
            .collect()
    }
}

fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening WAV {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let mut interleaved = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                interleaved.push(sample?);
            }
        }
        hound::SampleFormat::Int => {
            let max = ((1i64 << (spec.bits_per_sample.saturating_sub(1))) - 1).max(1) as f32;
            for sample in reader.samples::<i32>() {
                interleaved.push(sample? as f32 / max);
            }
        }
    }
    let mut mono = Vec::with_capacity(interleaved.len() / channels);
    for frame in interleaved.chunks(channels) {
        mono.push(frame.iter().sum::<f32>() / frame.len() as f32);
    }
    Ok((mono, spec.sample_rate))
}

fn resample_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let target_len = ((samples.len() as f64) * target_rate as f64 / source_rate as f64)
        .round()
        .max(1.0) as usize;
    let scale = source_rate as f64 / target_rate as f64;
    (0..target_len)
        .map(|i| {
            let src = i as f64 * scale;
            let left = src.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let frac = (src - left as f64) as f32;
            samples[left] * (1.0 - frac) + samples[right] * frac
        })
        .collect()
}

fn compact_audio_features(samples: &[f32], config: &CommonPhoneConfig) -> Vec<Vec<f32>> {
    let window = (config.sample_rate_hz as f32 * 0.025).round().max(1.0) as usize;
    let hop = (config.sample_rate_hz / config.frame_hz).max(1) as usize;
    let mut rows = Vec::new();
    let mut prev_mel = vec![0.0; DEFAULT_MEL_BINS];
    let mut offset = 0usize;
    while offset < samples.len() {
        let end = (offset + window).min(samples.len());
        let slice = &samples[offset..end];
        let mel = pseudo_log_mel(slice);
        let delta = mel
            .iter()
            .zip(prev_mel.iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let energy = (slice.iter().map(|v| v * v).sum::<f32>() / slice.len().max(1) as f32)
            .max(1e-8)
            .ln();
        let vad = if energy > -8.0 { 1.0 } else { 0.0 };
        let zcr = zero_crossing_rate(slice);
        let centroid = pseudo_centroid(slice);
        let flux = mel
            .iter()
            .zip(prev_mel.iter())
            .map(|(a, b)| (a - b).max(0.0))
            .sum::<f32>()
            / DEFAULT_MEL_BINS as f32;
        let f0 = estimate_f0(slice, config.sample_rate_hz);
        let voiced_prob = if f0 > 0.0 && vad > 0.0 { 1.0 } else { 0.0 };
        let mut row = Vec::with_capacity(config.feature_bins);
        row.extend(mel.iter().copied());
        row.extend(delta);
        row.extend([energy, vad, zcr, centroid, flux, f0 / 500.0, voiced_prob]);
        row.resize(config.feature_bins, 0.0);
        prev_mel = mel;
        rows.push(row);
        offset += hop;
    }
    if rows.is_empty() {
        rows.push(vec![0.0; config.feature_bins]);
    }
    rows
}

fn pseudo_log_mel(slice: &[f32]) -> Vec<f32> {
    let mut bins = vec![0.0; DEFAULT_MEL_BINS];
    if slice.is_empty() {
        return bins;
    }
    for (i, bin) in bins.iter_mut().enumerate() {
        let start = i * slice.len() / DEFAULT_MEL_BINS;
        let end = ((i + 1) * slice.len() / DEFAULT_MEL_BINS)
            .max(start + 1)
            .min(slice.len());
        let rms =
            (slice[start..end].iter().map(|v| v * v).sum::<f32>() / (end - start) as f32).sqrt();
        *bin = (rms + 1e-5).ln();
    }
    bins
}

fn zero_crossing_rate(slice: &[f32]) -> f32 {
    if slice.len() < 2 {
        return 0.0;
    }
    let crossings = slice
        .windows(2)
        .filter(|pair| (pair[0] >= 0.0) != (pair[1] >= 0.0))
        .count();
    crossings as f32 / (slice.len() - 1) as f32
}

fn pseudo_centroid(slice: &[f32]) -> f32 {
    let denom = slice.iter().map(|v| v.abs()).sum::<f32>().max(1e-6);
    slice
        .iter()
        .enumerate()
        .map(|(i, v)| i as f32 * v.abs())
        .sum::<f32>()
        / denom
        / slice.len().max(1) as f32
}

fn estimate_f0(slice: &[f32], sample_rate: u32) -> f32 {
    if slice.len() < 32 {
        return 0.0;
    }
    let min_lag = (sample_rate / 500).max(1) as usize;
    let max_lag = (sample_rate / 70).max(min_lag as u32 + 1) as usize;
    let mut best_lag = 0usize;
    let mut best = 0.0f32;
    for lag in min_lag..max_lag.min(slice.len() / 2) {
        let score = slice[..slice.len() - lag]
            .iter()
            .zip(slice[lag..].iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
        if score > best {
            best = score;
            best_lag = lag;
        }
    }
    if best_lag == 0 {
        0.0
    } else {
        sample_rate as f32 / best_lag as f32
    }
}

fn write_feature_file(path: &Path, features: &[Vec<f32>], bins: usize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let part = path.with_extension(format!(
        "{}.part",
        path.extension().and_then(|s| s.to_str()).unwrap_or("bin")
    ));
    let mut writer = BufWriter::new(File::create(&part)?);
    writer.write_all(&(features.len() as u32).to_le_bytes())?;
    writer.write_all(&(bins as u32).to_le_bytes())?;
    for row in features {
        anyhow::ensure!(
            row.len() == bins,
            "feature row has {} bins, expected {bins}",
            row.len()
        );
        for value in row {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    writer.flush()?;
    fs::rename(&part, path)?;
    Ok(())
}

fn read_feature_file(path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;
    let rows = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let bins = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    Ok((rows, bins, values))
}

fn recover_rows(path: &Path) -> Result<Vec<CommonPhoneRow>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    read_examples(path)
}

fn write_jsonl_atomic(
    path: &Path,
    rows: &[CommonPhoneRow],
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<()> {
    let part = path.with_extension("jsonl.part");
    let mut writer = BufWriter::new(File::create(&part)?);
    for row in rows {
        writeln!(writer, "{}", serde_json::to_string(row)?)?;
    }
    writer.flush()?;
    fs::rename(&part, path)?;
    progress(PrepareProgress::Write {
        path: path.display().to_string(),
        rows: rows.len(),
    });
    Ok(())
}

fn write_text_atomic(path: &Path, text: impl AsRef<str>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let part = path.with_extension(format!(
        "{}.part",
        path.extension().and_then(|s| s.to_str()).unwrap_or("txt")
    ));
    let mut writer = BufWriter::new(File::create(&part)?);
    writer.write_all(text.as_ref().as_bytes())?;
    writer.flush()?;
    fs::rename(&part, path)?;
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(serde_json::from_str(&text)?)
}

fn build_token_vocab<'a>(tokens: impl Iterator<Item = &'a String>) -> Vocab {
    let mut vocab_tokens = vec![CTC_BLANK.to_string(), UNK.to_string()];
    let mut seen = BTreeSet::new();
    for token in tokens {
        if seen.insert(token.clone()) {
            vocab_tokens.push(token.clone());
        }
    }
    let token_to_id = vocab_tokens
        .iter()
        .enumerate()
        .map(|(i, token)| (token.clone(), i as u32))
        .collect();
    Vocab {
        tokens: vocab_tokens,
        token_to_id,
    }
}

fn build_feature_axis_vocabs(rows: &[CommonPhoneRow]) -> BTreeMap<String, Vocab> {
    feature_axes()
        .into_iter()
        .map(|axis| {
            let vocab = build_token_vocab(
                rows.iter()
                    .flat_map(|row| row.feature_targets.get(axis).into_iter().flatten()),
            );
            (axis.to_string(), vocab)
        })
        .collect()
}

fn feature_axes() -> Vec<&'static str> {
    vec![
        "manner", "place", "voicing", "syllabic", "height", "backness", "rounding",
    ]
}

fn feature_targets_for_phones(phones: &[String]) -> BTreeMap<String, Vec<String>> {
    let mut targets = feature_axes()
        .into_iter()
        .map(|axis| (axis.to_string(), Vec::new()))
        .collect::<BTreeMap<_, _>>();
    for phone in phones {
        let features = phone_features(phone);
        for (axis, value) in features {
            targets
                .entry(axis.to_string())
                .or_default()
                .push(value.to_string());
        }
    }
    targets
}

fn phone_features(phone: &str) -> BTreeMap<&'static str, &'static str> {
    let p = phone
        .trim_matches(|ch: char| ch == '/' || ch == '[' || ch == ']' || ch == 'ː' || ch == ':')
        .to_lowercase();
    let mut map = BTreeMap::new();
    let (manner, place, voicing, syllabic, height, backness, rounding) = match p.as_str() {
        "p" => ("stop", "bilabial", "voiceless", "no", NONE, NONE, NONE),
        "b" => ("stop", "bilabial", "voiced", "no", NONE, NONE, NONE),
        "t" => ("stop", "alveolar", "voiceless", "no", NONE, NONE, NONE),
        "d" => ("stop", "alveolar", "voiced", "no", NONE, NONE, NONE),
        "k" => ("stop", "velar", "voiceless", "no", NONE, NONE, NONE),
        "g" => ("stop", "velar", "voiced", "no", NONE, NONE, NONE),
        "m" => ("nasal", "bilabial", "voiced", "no", NONE, NONE, NONE),
        "n" => ("nasal", "alveolar", "voiced", "no", NONE, NONE, NONE),
        "ŋ" => ("nasal", "velar", "voiced", "no", NONE, NONE, NONE),
        "f" => (
            "fricative",
            "labiodental",
            "voiceless",
            "no",
            NONE,
            NONE,
            NONE,
        ),
        "v" => ("fricative", "labiodental", "voiced", "no", NONE, NONE, NONE),
        "s" => ("fricative", "alveolar", "voiceless", "no", NONE, NONE, NONE),
        "z" => ("fricative", "alveolar", "voiced", "no", NONE, NONE, NONE),
        "ʃ" => (
            "fricative",
            "postalveolar",
            "voiceless",
            "no",
            NONE,
            NONE,
            NONE,
        ),
        "ʒ" => (
            "fricative",
            "postalveolar",
            "voiced",
            "no",
            NONE,
            NONE,
            NONE,
        ),
        "h" => ("fricative", "glottal", "voiceless", "no", NONE, NONE, NONE),
        "l" => ("lateral", "alveolar", "voiced", "no", NONE, NONE, NONE),
        "r" | "ɹ" => ("approximant", "alveolar", "voiced", "no", NONE, NONE, NONE),
        "j" => ("approximant", "palatal", "voiced", "no", NONE, NONE, NONE),
        "w" => (
            "approximant",
            "labial-velar",
            "voiced",
            "no",
            NONE,
            NONE,
            NONE,
        ),
        "i" | "ɪ" => (
            "vowel",
            "vowel",
            "vowel",
            "yes",
            "high",
            "front",
            "unrounded",
        ),
        "e" | "ɛ" | "æ" => (
            "vowel",
            "vowel",
            "vowel",
            "yes",
            "mid",
            "front",
            "unrounded",
        ),
        "a" | "ɑ" | "ɐ" => (
            "vowel",
            "vowel",
            "vowel",
            "yes",
            "low",
            "central",
            "unrounded",
        ),
        "ə" | "ʌ" => (
            "vowel",
            "vowel",
            "vowel",
            "yes",
            "mid",
            "central",
            "unrounded",
        ),
        "o" | "ɔ" => ("vowel", "vowel", "vowel", "yes", "mid", "back", "rounded"),
        "u" | "ʊ" => ("vowel", "vowel", "vowel", "yes", "high", "back", "rounded"),
        _ => (UNK, UNK, UNK, UNK, UNK, UNK, UNK),
    };
    map.insert("manner", manner);
    map.insert("place", place);
    map.insert("voicing", voicing);
    map.insert("syllabic", syllabic);
    map.insert("height", height);
    map.insert("backness", backness);
    map.insert("rounding", rounding);
    map
}

fn count_unknown_phones(rows: &[CommonPhoneRow]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for phone in rows.iter().flat_map(|row| row.phones.iter()) {
        if phone_features(phone).values().any(|value| *value == UNK) {
            *counts.entry(phone.clone()).or_insert(0) += 1;
        }
    }
    counts
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct BaselineModel {
    default_phones: Vec<String>,
    default_phonemes: Vec<String>,
    lang_phones: BTreeMap<String, Vec<String>>,
    lang_phonemes: BTreeMap<String, Vec<String>>,
}

impl BaselineModel {
    fn fit(rows: &[CommonPhoneRow]) -> Self {
        let default_phones = most_common_sequence(rows.iter().map(|row| &row.phones));
        let default_phonemes = most_common_sequence(rows.iter().map(|row| &row.phonemes));
        let mut by_lang: BTreeMap<String, Vec<&CommonPhoneRow>> = BTreeMap::new();
        for row in rows {
            by_lang.entry(row.lang.clone()).or_default().push(row);
        }
        let lang_phones = by_lang
            .iter()
            .map(|(lang, rows)| {
                (
                    lang.clone(),
                    most_common_sequence(rows.iter().map(|row| &row.phones)),
                )
            })
            .collect();
        let lang_phonemes = by_lang
            .iter()
            .map(|(lang, rows)| {
                (
                    lang.clone(),
                    most_common_sequence(rows.iter().map(|row| &row.phonemes)),
                )
            })
            .collect();
        Self {
            default_phones,
            default_phonemes,
            lang_phones,
            lang_phonemes,
        }
    }

    fn predict_phones(&self, lang: &str) -> Vec<String> {
        self.lang_phones
            .get(lang)
            .cloned()
            .unwrap_or_else(|| self.default_phones.clone())
    }

    fn predict_phonemes(&self, lang: &str) -> Vec<String> {
        self.lang_phonemes
            .get(lang)
            .cloned()
            .unwrap_or_else(|| self.default_phonemes.clone())
    }
}

fn most_common_sequence<'a>(sequences: impl Iterator<Item = &'a Vec<String>>) -> Vec<String> {
    let mut counts: HashMap<String, (usize, Vec<String>)> = HashMap::new();
    for sequence in sequences {
        let key = sequence.join("\u{1f}");
        counts
            .entry(key)
            .and_modify(|(count, _)| *count += 1)
            .or_insert((1, sequence.clone()));
    }
    counts
        .into_values()
        .max_by_key(|(count, _)| *count)
        .map(|(_, sequence)| sequence)
        .unwrap_or_default()
}

fn evaluate_with_model(
    model: &BaselineModel,
    rows: &[CommonPhoneRow],
    split: &str,
    sample_limit: usize,
) -> EvalReport {
    let mut phone_dist = EditDistanceAccumulator::default();
    let mut phoneme_dist = EditDistanceAccumulator::default();
    let mut axis_dist = feature_axes()
        .into_iter()
        .map(|axis| (axis.to_string(), EditDistanceAccumulator::default()))
        .collect::<BTreeMap<_, _>>();
    let mut samples = Vec::new();
    let mut language_distribution = BTreeMap::new();
    for row in rows {
        *language_distribution.entry(row.lang.clone()).or_insert(0) += 1;
        let pred_phones = model.predict_phones(&row.lang);
        let pred_phonemes = model.predict_phonemes(&row.lang);
        phone_dist.add(&pred_phones, &row.phones);
        phoneme_dist.add(&pred_phonemes, &row.phonemes);
        let pred_features = feature_targets_for_phones(&pred_phones);
        for axis in feature_axes() {
            let empty = Vec::new();
            axis_dist.get_mut(axis).unwrap().add(
                pred_features.get(axis).unwrap_or(&empty),
                row.feature_targets.get(axis).unwrap_or(&empty),
            );
        }
        if samples.len() < sample_limit {
            samples.push(GreedySample {
                utterance_id: row.utterance_id.clone(),
                lang: row.lang.clone(),
                phone_target: row.phones.clone(),
                phone_prediction: pred_phones,
            });
        }
    }
    let feature_axis_token_error_rate = axis_dist
        .iter()
        .map(|(axis, dist)| (axis.clone(), dist.rate()))
        .collect::<BTreeMap<_, _>>();
    let aggregate_feature_token_error_rate = if axis_dist.is_empty() {
        0.0
    } else {
        axis_dist
            .values()
            .map(EditDistanceAccumulator::rate)
            .sum::<f64>()
            / axis_dist.len() as f64
    };
    EvalReport {
        split: split.to_string(),
        examples: rows.len(),
        phone_token_error_rate: phone_dist.rate(),
        phoneme_token_error_rate: Some(phoneme_dist.rate()),
        feature_axis_token_error_rate,
        aggregate_feature_token_error_rate,
        unknown_phone_symbols: count_unknown_phones(rows),
        language_distribution,
        samples,
    }
}

#[derive(Default)]
struct EditDistanceAccumulator {
    edits: usize,
    tokens: usize,
}

impl EditDistanceAccumulator {
    fn add(&mut self, pred: &[String], target: &[String]) {
        self.edits += edit_distance(pred, target);
        self.tokens += target.len().max(1);
    }

    fn rate(&self) -> f64 {
        if self.tokens == 0 {
            0.0
        } else {
            self.edits as f64 / self.tokens as f64
        }
    }
}

fn edit_distance(a: &[String], b: &[String]) -> usize {
    let mut prev = (0..=b.len()).collect::<Vec<_>>();
    let mut curr = vec![0; b.len() + 1];
    for (i, av) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, bv) in b.iter().enumerate() {
            let cost = usize::from(av != bv);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

fn first_feature_bins(data: &Path, rows: &[CommonPhoneRow]) -> Result<usize> {
    let first = rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("no training rows in {}", data.display()))?;
    let (_, bins) = feature_file_shape(&data.join(&first.feature_path))?;
    Ok(bins)
}

fn save_artifact_files(
    out: &Path,
    data: &Path,
    model_config: &ModelConfig,
    train_config: &CommonPhoneTrainConfig,
) -> Result<()> {
    write_text_atomic(
        &out.join("model_config.json"),
        serde_json::to_string_pretty(model_config)?,
    )?;
    write_text_atomic(
        &out.join("train_config.json"),
        serde_json::to_string_pretty(train_config)?,
    )?;
    for name in [
        "vocab.json",
        "phone_vocab.json",
        "phoneme_vocab.json",
        "feature_axis_vocabs.json",
    ] {
        fs::copy(data.join(name), out.join(name))
            .with_context(|| format!("copying {name} into {}", out.display()))?;
    }
    let manifest = ModelArtifactManifest::new(FAMILY, ARCHITECTURE, data_id_from_path(data))
        .with_task("compact-frame-phone-feature-ctc");
    write_manifest(out, &manifest)?;
    Ok(())
}

fn data_id_from_path(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn write_prepare_state(
    out: &Path,
    status: &str,
    config: &CommonPhoneConfig,
    rows: usize,
    report: Option<&PrepareReport>,
) -> Result<()> {
    write_text_atomic(
        &out.join("prepare_state.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "status": status,
            "rows": rows,
            "config": config,
            "report": report,
        }))?,
    )
}

fn dataset_readme(config: &CommonPhoneConfig, unknowns: &BTreeMap<String, usize>) -> String {
    format!(
        "# Common Phone v0 dataset\n\nInput: `{}`\n\nThis dataset uses mechanical compact acoustic frames, not learned EnCodec-style audio tokens. Each `features/*.acf.bin` file stores a little-endian header `(frames: u32, bins: u32)` followed by `{}` `f32` values per frame: pseudo log-Mel bins, deltas, energy, VAD, zero-crossing rate, spectral centroid, spectral flux, estimated f0, and voiced probability.\n\nThe first supported local layout is an input directory containing `metadata.jsonl`, `metadata.csv`, or `metadata.tsv`. Required columns/fields are `audio_path` (or `path`/`wav`) and `phones`; optional fields include `utterance_id`, `lang`, `variety`, `speaker_id`, and `phonemes`. Audio is currently decoded from WAV files.\n\nTargets are trained as ordered CTC sequences: phones, phonemes, and feature axes (`manner`, `place`, `voicing`, `syllabic`, `height`, `backness`, `rounding`). Unknown phone-feature mappings are counted instead of rejected.\n\nSample rate: {}\nFrame rate: {}\nUnknown phone symbols: {:?}\n",
        config.input, config.feature_bins, config.sample_rate_hz, config.frame_hz, unknowns
    )
}

fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}
