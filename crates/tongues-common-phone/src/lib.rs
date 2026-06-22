//! Common Phone compact-acoustic-frame model family.
//!
//! V0 prepares local Common Phone style exports into mechanical compact
//! acoustic frames and ordered phone / phonetic-feature targets. The training
//! artifact is intentionally small and CPU-friendly: it records CTC-head
//! metadata and a frequency baseline while the durable data path settles.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::{CTCLossConfig, Reduction};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use burn::tensor::{Int, Tensor, TensorData};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use tongues_core::Vocab;
use tongues_neural::{make_recorder, write_manifest, ModelArtifactManifest};

pub const FAMILY: &str = "common-phone";
pub const ARCHITECTURE: &str = "common-phone-compact-frame-ctc-v0";
pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 16_000;
pub const DEFAULT_FRAME_HZ: u32 = 100;
pub const DEFAULT_MEL_BINS: usize = 80;
pub const COMPACT_AUDIO_EXTRA_BINS: usize = 7;
pub const DEFAULT_COMPACT_AUDIO_FEATURE_BINS: usize =
    DEFAULT_MEL_BINS + DEFAULT_MEL_BINS + COMPACT_AUDIO_EXTRA_BINS;
pub const CTC_BLANK: &str = "<blank>";
pub const UNK: &str = "<unk>";
pub const NONE: &str = "none";
pub const PHONE_FEATURE_UNKNOWN: &str = "<PHONE_FEATURE_UNKNOWN>";
pub const DEFAULT_ZENODO_URL: &str =
    "https://zenodo.org/records/5846137/files/cp-1-0.tgz?download=1";
const ACF_MAGIC: &[u8; 4] = b"ACF0";
const ACF_VERSION: u32 = 1;
const DOWNLOAD_USER_AGENT: &str = "tongues-common-phone/0.1";
const LATEST_MODEL_STEM: &str = "model-latest";
const TRAIN_CHECKPOINT_BATCH_INTERVAL: usize = 100;

type CpuInferBackend = NdArray<f32>;
type CpuTrainBackend = Autodiff<CpuInferBackend>;

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
    pub window_ms: f32,
    pub hop_ms: f32,
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
            window_ms: 25.0,
            hop_ms: 10.0,
            feature_bins: DEFAULT_COMPACT_AUDIO_FEATURE_BINS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommonPhoneTrainConfig {
    pub task: CommonPhoneTask,
    pub learning_rate: f64,
    pub batch_frames: usize,
    pub epochs: usize,
    pub seed: u64,
    pub hidden_dim: usize,
    pub dropout: f64,
    pub phone_ctc_loss_weight: f32,
    pub feature_bundle_ctc_loss_weight: f32,
    pub phoneme_ctc_loss_weight: f32,
    pub feature_axis_ctc_loss_weight: f32,
}

impl Default for CommonPhoneTrainConfig {
    fn default() -> Self {
        Self {
            task: CommonPhoneTask::Frames2Phones,
            learning_rate: 3e-4,
            batch_frames: 12_000,
            epochs: 3,
            seed: 42,
            hidden_dim: 128,
            dropout: 0.1,
            phone_ctc_loss_weight: 1.0,
            feature_bundle_ctc_loss_weight: 0.5,
            phoneme_ctc_loss_weight: 0.5,
            feature_axis_ctc_loss_weight: 0.35,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CommonPhoneTask {
    Frames2Phones,
    Frames2Features,
    Frames2Phonemes,
    Multitask,
}

impl CommonPhoneTask {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "frames2phones" | "phones" => Ok(Self::Frames2Phones),
            "frames2features" | "features" | "feature-bundles" => Ok(Self::Frames2Features),
            "frames2phonemes" | "phonemes" => Ok(Self::Frames2Phonemes),
            "multitask" | "frames2phones,frames2features" => Ok(Self::Multitask),
            _ => anyhow::bail!(
                "invalid Common Phone task `{value}`; supported: frames2phones, frames2features, frames2phonemes, multitask"
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: String,
    pub input_feature_bins: usize,
    pub hidden_dim: usize,
    pub dropout: f64,
    pub frame_hz: u32,
    pub phone_vocab_size: usize,
    pub phoneme_vocab_size: usize,
    pub feature_bundle_vocab_size: usize,
    pub feature_axis_vocab_sizes: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommonPhoneRow {
    pub row_source: String,
    pub utterance_id: String,
    pub split: Option<String>,
    pub lang: String,
    pub variety: Option<String>,
    pub speaker_id: Option<String>,
    pub text: Option<String>,
    pub audio_path: String,
    pub feature_path: String,
    pub sample_rate: u32,
    pub frame_hz: u32,
    pub hop_ms: f32,
    pub window_ms: f32,
    pub frame_count: usize,
    pub frame_dim: usize,
    pub duration_sec: f32,
    pub duration_ms: u64,
    pub phones: Vec<String>,
    pub phonemes: Vec<String>,
    pub feature_bundles: Vec<String>,
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
    pub skipped_examples: usize,
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
    Download {
        url: String,
        path: String,
        bytes: u64,
    },
    Extract {
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
    Select {
        selected: usize,
        total: usize,
    },
    Split {
        train: usize,
        valid: usize,
        test: usize,
    },
    Vocab {
        name: String,
        tokens: usize,
        path: String,
    },
    State {
        status: String,
        rows: usize,
        path: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrainProgress {
    Startup {
        train_examples: usize,
        valid_examples: usize,
        epochs: usize,
        train_state_path: String,
        epoch_checkpoint_pattern: String,
        best_model_path: String,
    },
    EpochStart {
        epoch: usize,
        epochs: usize,
        train_examples: usize,
    },
    Batch {
        epoch: usize,
        examples: usize,
        total_examples: usize,
        loss: f32,
    },
    EpochComplete {
        epoch: usize,
        train_loss: f32,
        valid_error: f64,
        exact_sequence_accuracy: f64,
        blank_ratio: f64,
    },
    Checkpoint {
        epoch: usize,
        path: String,
        best: bool,
    },
    State {
        epoch: usize,
        path: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainReport {
    pub epochs: usize,
    pub train_examples: usize,
    pub valid_examples: usize,
    pub best_validation_error_rate: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    pub split: String,
    pub task: CommonPhoneTask,
    pub examples_evaluated: usize,
    pub examples_failed: usize,
    pub token_error_rate: f64,
    pub edit_distance: usize,
    pub exact_sequence_accuracy: f64,
    pub blank_ratio: f64,
    pub mean_prediction_length: f64,
    pub mean_target_length: f64,
    pub phone_token_error_rate: Option<f64>,
    pub phoneme_token_error_rate: Option<f64>,
    pub feature_bundle_error_rate: Option<f64>,
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
pub struct LiveFrameStats {
    pub rms: f32,
    pub vad: f32,
    pub frames: usize,
    pub frame_dim: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveDecode {
    pub phones: Vec<String>,
    pub feature_bundles: Vec<String>,
    pub blank_ratio: f64,
    pub prediction_length: usize,
    pub stats: LiveFrameStats,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShowRow {
    pub utterance_id: String,
    pub lang: String,
    pub phones: Vec<String>,
    pub feature_bundles: Vec<String>,
    pub feature_targets: BTreeMap<String, Vec<String>>,
    pub feature_shape: (usize, usize),
    pub first_frames: Vec<FrameSummary>,
    pub mean: f32,
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameSummary {
    pub frame: usize,
    pub energy: f32,
    pub zcr: f32,
    pub centroid: f32,
    pub f0: Option<f32>,
    pub voiced: f32,
    pub mel_head: Vec<f32>,
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
    wav_path: Option<String>,
    path: Option<String>,
    wav: Option<String>,
    split: Option<String>,
    text: Option<String>,
    duration_ms: Option<u64>,
    source_dataset: Option<String>,
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
    fs::create_dir_all(out.join("frames")).context("creating common-phone frames directory")?;
    fs::create_dir_all(out.join("vocabs")).context("creating common-phone vocabs directory")?;
    write_prepare_state(out, "starting", config, 0, None)?;
    progress(PrepareProgress::State {
        status: "starting".to_string(),
        rows: 0,
        path: out.join("prepare_state.json").display().to_string(),
    });

    let input = Path::new(&config.input);
    let input_rows = read_input_records(input, &mut progress)?;
    let total_input_rows = input_rows.len();
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
    progress(PrepareProgress::Select {
        selected: selected.len(),
        total: total_input_rows,
    });
    anyhow::ensure!(
        !selected.is_empty(),
        "no Common Phone rows selected from {}",
        input.display()
    );

    let manifest_path = out.join("manifest.jsonl");
    let manifest_part_path = manifest_path.with_extension("jsonl.part");
    if manifest_part_path.exists() {
        fs::remove_file(&manifest_part_path)
            .with_context(|| format!("removing {}", manifest_part_path.display()))?;
    }
    let mut writer = BufWriter::new(File::create(&manifest_part_path)?);
    let mut rows = Vec::new();

    for (index, record) in selected {
        let utterance_id = record
            .utterance_id
            .clone()
            .or(record.id.clone())
            .unwrap_or_else(|| format!("common-phone-{index:08}"));
        let rel_feature =
            PathBuf::from("frames").join(format!("{}.acf.bin", sanitize_id(&utterance_id)));
        let audio_rel = record
            .audio_path
            .clone()
            .or(record.wav_path.clone())
            .or(record.path.clone())
            .or(record.wav.clone())
            .ok_or_else(|| {
                anyhow::anyhow!("row {utterance_id} has no wav_path/audio_path/path/wav")
            })?;
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
        let feature_abs = out.join(&rel_feature);
        let reused_shape = feature_file_shape(&feature_abs)
            .ok()
            .filter(|(_, bins)| *bins == config.feature_bins);
        let (frame_count, duration_sec, duration_ms) = if let Some((frames, _bins)) = reused_shape {
            progress(PrepareProgress::Reuse {
                utterance_id: utterance_id.clone(),
                frames,
                path: feature_abs.display().to_string(),
            });
            let duration_ms = record
                .duration_ms
                .unwrap_or_else(|| ((frames as f64 / config.frame_hz as f64) * 1000.0) as u64);
            (frames, duration_ms as f32 / 1000.0, duration_ms)
        } else {
            let (samples, source_rate) = read_wav_mono(&audio_path)?;
            let samples = normalize_amplitude(&resample_linear(
                &samples,
                source_rate,
                config.sample_rate_hz,
            ));
            let features = compact_audio_features(&samples, config);
            write_feature_file(&feature_abs, &features, config.feature_bins)?;
            progress(PrepareProgress::Features {
                utterance_id: utterance_id.clone(),
                frames: features.len(),
                path: feature_abs.display().to_string(),
            });
            let duration_ms = record
                .duration_ms
                .unwrap_or_else(|| (samples.len() as u64 * 1000) / config.sample_rate_hz as u64);
            (
                features.len(),
                samples.len() as f32 / config.sample_rate_hz as f32,
                duration_ms,
            )
        };

        let feature_targets = feature_targets_for_phones(&phone_tokens);
        let feature_bundles = feature_bundles_for_phones(&phone_tokens);
        let raw = serde_json::json!({
            "common_phone_record": record,
            "segments": record.segments,
        });
        let row = CommonPhoneRow {
            row_source: FAMILY.to_string(),
            utterance_id: utterance_id.clone(),
            split: record.split.clone(),
            lang: record
                .lang
                .clone()
                .or(record.language.clone())
                .unwrap_or_else(|| "und".to_string()),
            variety: record.variety.clone(),
            speaker_id: record.speaker_id.clone().or(record.speaker.clone()),
            text: record.text.clone(),
            audio_path: audio_path.display().to_string(),
            feature_path: rel_feature.display().to_string(),
            sample_rate: config.sample_rate_hz,
            frame_hz: config.frame_hz,
            hop_ms: config.hop_ms,
            window_ms: config.window_ms,
            frame_count,
            frame_dim: config.feature_bins,
            duration_sec,
            duration_ms,
            phones: phone_tokens,
            phonemes: phoneme_tokens,
            feature_bundles,
            feature_targets,
            raw,
        };
        writeln!(writer, "{}", serde_json::to_string(&row)?)?;
        writer.flush()?;
        rows.push(row);
    }
    writer.flush()?;
    fs::rename(&manifest_part_path, &manifest_path)?;
    progress(PrepareProgress::Write {
        path: manifest_path.display().to_string(),
        rows: rows.len(),
    });
    write_prepare_state(out, "utterances", config, rows.len(), None)?;
    progress(PrepareProgress::State {
        status: "utterances".to_string(),
        rows: rows.len(),
        path: out.join("prepare_state.json").display().to_string(),
    });

    let (train, valid, test) = split_rows(rows, config);
    let n = train.len() + valid.len() + test.len();
    progress(PrepareProgress::Split {
        train: train.len(),
        valid: valid.len(),
        test: test.len(),
    });
    write_jsonl_atomic(&out.join("train.jsonl"), &train, &mut progress)?;
    write_jsonl_atomic(&out.join("valid.jsonl"), &valid, &mut progress)?;
    write_jsonl_atomic(&out.join("test.jsonl"), &test, &mut progress)?;

    let all = [&train[..], &valid[..], &test[..]].concat();
    let phone_vocab = build_token_vocab(all.iter().flat_map(|row| row.phones.iter()));
    let phoneme_vocab = build_token_vocab(all.iter().flat_map(|row| row.phonemes.iter()));
    let feature_bundle_vocab =
        build_token_vocab(all.iter().flat_map(|row| row.feature_bundles.iter()));
    let axis_vocabs = build_feature_axis_vocabs(&all);
    write_text_atomic(
        &out.join("vocab.json"),
        serde_json::to_string_pretty(&phone_vocab)?,
    )?;
    progress(PrepareProgress::Vocab {
        name: "legacy phone vocab".to_string(),
        tokens: phone_vocab.size(),
        path: out.join("vocab.json").display().to_string(),
    });
    write_text_atomic(
        &out.join("phone_vocab.json"),
        serde_json::to_string_pretty(&phone_vocab)?,
    )?;
    progress(PrepareProgress::Vocab {
        name: "phone vocab".to_string(),
        tokens: phone_vocab.size(),
        path: out.join("phone_vocab.json").display().to_string(),
    });
    write_text_atomic(
        &out.join("vocabs").join("phones.json"),
        serde_json::to_string_pretty(&phone_vocab)?,
    )?;
    write_text_atomic(
        &out.join("phoneme_vocab.json"),
        serde_json::to_string_pretty(&phoneme_vocab)?,
    )?;
    progress(PrepareProgress::Vocab {
        name: "phoneme vocab".to_string(),
        tokens: phoneme_vocab.size(),
        path: out.join("phoneme_vocab.json").display().to_string(),
    });
    write_text_atomic(
        &out.join("vocabs").join("phonemes.json"),
        serde_json::to_string_pretty(&phoneme_vocab)?,
    )?;
    write_text_atomic(
        &out.join("feature_bundle_vocab.json"),
        serde_json::to_string_pretty(&feature_bundle_vocab)?,
    )?;
    progress(PrepareProgress::Vocab {
        name: "feature bundle vocab".to_string(),
        tokens: feature_bundle_vocab.size(),
        path: out.join("feature_bundle_vocab.json").display().to_string(),
    });
    write_text_atomic(
        &out.join("vocabs").join("feature_bundles.json"),
        serde_json::to_string_pretty(&feature_bundle_vocab)?,
    )?;
    write_text_atomic(
        &out.join("feature_axis_vocabs.json"),
        serde_json::to_string_pretty(&axis_vocabs)?,
    )?;
    write_text_atomic(
        &out.join("dataset_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    write_text_atomic(
        &out.join("config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    let unknown_phone_symbols = count_unknown_phones(&all);
    let stats = dataset_stats(
        config,
        &train,
        &valid,
        &test,
        phone_vocab.size(),
        feature_bundle_vocab.size(),
        &unknown_phone_symbols,
        0,
    );
    write_text_atomic(
        &out.join("stats.json"),
        serde_json::to_string_pretty(&stats)?,
    )?;
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
        skipped_examples: 0,
    };
    write_prepare_state(out, "complete", config, n, Some(&report))?;
    progress(PrepareProgress::State {
        status: "complete".to_string(),
        rows: n,
        path: out.join("prepare_state.json").display().to_string(),
    });
    Ok(report)
}

pub fn download_common_phone_zenodo(
    out: &Path,
    url: &str,
    mut progress: impl FnMut(PrepareProgress),
) -> Result<()> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let archive = out.join("cp-1-0.tgz");
    if !archive.exists() {
        download_to_part(url, &archive, &mut progress)?;
    }
    let marker = out.join(".common-phone-extract-complete");
    if marker.exists() && has_common_phone_source_layout(out)? {
        return Ok(());
    }
    progress(PrepareProgress::Extract {
        path: archive.display().to_string(),
    });
    let part = out.join("extract.part");
    if part.exists() {
        fs::remove_dir_all(&part).with_context(|| format!("removing {}", part.display()))?;
    }
    fs::create_dir_all(&part)?;
    let file = File::open(&archive).with_context(|| format!("opening {}", archive.display()))?;
    let decoder = flate2::read::GzDecoder::new(file);
    let mut tar = tar::Archive::new(decoder);
    tar.unpack(&part)
        .with_context(|| format!("extracting {}", archive.display()))?;
    merge_extracted_tree(&part, out)?;
    fs::remove_dir_all(&part).with_context(|| format!("removing {}", part.display()))?;
    fs::write(marker, b"ok\n")?;
    Ok(())
}

pub fn train(data: &Path, out: &Path, config: &CommonPhoneTrainConfig) -> Result<TrainReport> {
    train_with_progress(data, out, config, |_| {})
}

pub fn train_with_progress(
    data: &Path,
    out: &Path,
    config: &CommonPhoneTrainConfig,
    mut progress: impl FnMut(TrainProgress),
) -> Result<TrainReport> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let train_rows = read_examples(&data.join("train.jsonl"))?;
    let valid_rows = read_examples(&data.join("valid.jsonl"))?;
    progress(TrainProgress::Startup {
        train_examples: train_rows.len(),
        valid_examples: valid_rows.len(),
        epochs: config.epochs,
        train_state_path: out.join("train_state.json").display().to_string(),
        epoch_checkpoint_pattern: out.join("model-epoch-N.bin").display().to_string(),
        best_model_path: out.join("model.bin").display().to_string(),
    });
    train_cpu(data, out, config, &train_rows, &valid_rows, &mut progress)
}

fn train_cpu(
    data: &Path,
    out: &Path,
    config: &CommonPhoneTrainConfig,
    train_rows: &[CommonPhoneRow],
    valid_rows: &[CommonPhoneRow],
    progress: &mut impl FnMut(TrainProgress),
) -> Result<TrainReport> {
    let phone_vocab: Vocab = read_vocab(data, "phone_vocab.json", "phones.json")?;
    let phoneme_vocab: Vocab = read_vocab(data, "phoneme_vocab.json", "phonemes.json")?;
    let feature_bundle_vocab: Vocab =
        read_vocab(data, "feature_bundle_vocab.json", "feature_bundles.json")?;
    let axis_vocabs: BTreeMap<String, Vocab> = read_json(&data.join("feature_axis_vocabs.json"))?;
    let feature_bins = first_feature_bins(data, &train_rows)?;
    let model_config = ModelConfig {
        architecture: ARCHITECTURE.to_string(),
        input_feature_bins: feature_bins,
        hidden_dim: config.hidden_dim,
        dropout: config.dropout,
        frame_hz: train_rows
            .first()
            .map(|row| row.frame_hz)
            .unwrap_or(DEFAULT_FRAME_HZ),
        phone_vocab_size: phone_vocab.size(),
        phoneme_vocab_size: phoneme_vocab.size(),
        feature_bundle_vocab_size: feature_bundle_vocab.size(),
        feature_axis_vocab_sizes: axis_vocabs
            .iter()
            .map(|(axis, vocab)| (axis.clone(), vocab.size()))
            .collect(),
    };
    save_artifact_files(out, data, &model_config, config)?;
    let device = NdArrayDevice::Cpu;
    let mut model = model_config.init::<CpuTrainBackend>(&device);
    let mut optimizer =
        AdamWConfig::new().init::<CpuTrainBackend, CommonPhoneModel<CpuTrainBackend>>();
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let model_path = out.join("model");
    model
        .valid()
        .save_file(&out.join(LATEST_MODEL_STEM), &make_recorder())?;
    write_train_state(
        out,
        0,
        f64::INFINITY,
        config,
        "initialized",
        Some(format!("{LATEST_MODEL_STEM}.bin")),
        None,
    )?;
    progress(TrainProgress::Checkpoint {
        epoch: 0,
        path: out
            .join(format!("{LATEST_MODEL_STEM}.bin"))
            .display()
            .to_string(),
        best: false,
    });
    progress(TrainProgress::State {
        epoch: 0,
        path: out.join("train_state.json").display().to_string(),
    });
    let mut best = f64::INFINITY;
    for epoch in 1..=config.epochs {
        progress(TrainProgress::EpochStart {
            epoch,
            epochs: config.epochs,
            train_examples: train_rows.len(),
        });
        let loss = train_epoch_cpu(
            &mut model,
            &mut optimizer,
            config,
            data,
            train_rows,
            &phone_vocab,
            &phoneme_vocab,
            &feature_bundle_vocab,
            &device,
            &mut rng,
            out,
            epoch,
            best,
            progress,
        )?;
        let eval_model = model.valid();
        let report = evaluate_model_cpu(
            &eval_model,
            data,
            valid_rows,
            &phone_vocab,
            &phoneme_vocab,
            &feature_bundle_vocab,
            config.task,
            0,
            "valid",
            &device,
        )?;
        best = best.min(report.token_error_rate);
        println!(
            "Epoch {} | train_loss={:.4} valid_error={:.4} exact={:.3} blank_ratio={:.3}",
            epoch,
            loss,
            report.token_error_rate,
            report.exact_sequence_accuracy,
            report.blank_ratio
        );
        eval_model
            .clone()
            .save_file(&out.join(format!("model-epoch-{epoch}")), &make_recorder())?;
        progress(TrainProgress::Checkpoint {
            epoch,
            path: out
                .join(format!("model-epoch-{epoch}.bin"))
                .display()
                .to_string(),
            best: false,
        });
        write_train_state(
            out,
            epoch,
            best,
            config,
            "epoch-complete",
            Some(format!("model-epoch-{epoch}.bin")),
            Some(loss),
        )?;
        progress(TrainProgress::State {
            epoch,
            path: out.join("train_state.json").display().to_string(),
        });
        if (report.token_error_rate - best).abs() < f64::EPSILON {
            eval_model.save_file(&model_path, &make_recorder())?;
            progress(TrainProgress::Checkpoint {
                epoch,
                path: out.join("model.bin").display().to_string(),
                best: true,
            });
        }
        progress(TrainProgress::EpochComplete {
            epoch,
            train_loss: loss,
            valid_error: report.token_error_rate,
            exact_sequence_accuracy: report.exact_sequence_accuracy,
            blank_ratio: report.blank_ratio,
        });
    }
    Ok(TrainReport {
        epochs: config.epochs,
        train_examples: train_rows.len(),
        valid_examples: valid_rows.len(),
        best_validation_error_rate: if best.is_finite() { best } else { 0.0 },
    })
}

pub fn evaluate(
    model_dir: &Path,
    data: &Path,
    split: &str,
    task: CommonPhoneTask,
    sample_limit: usize,
) -> Result<EvalReport> {
    let rows = read_examples(&data.join(format!("{split}.jsonl")))?;
    let phone_vocab: Vocab = read_vocab(model_dir, "phone_vocab.json", "phones.json")
        .or_else(|_| read_vocab(data, "phone_vocab.json", "phones.json"))?;
    let phoneme_vocab: Vocab = read_vocab(model_dir, "phoneme_vocab.json", "phonemes.json")
        .or_else(|_| read_vocab(data, "phoneme_vocab.json", "phonemes.json"))?;
    let feature_bundle_vocab: Vocab = read_vocab(
        model_dir,
        "feature_bundle_vocab.json",
        "feature_bundles.json",
    )
    .or_else(|_| read_vocab(data, "feature_bundle_vocab.json", "feature_bundles.json"))?;
    let model_config: ModelConfig = read_json(&model_dir.join("model_config.json"))?;
    let device = NdArrayDevice::Cpu;
    let model = load_model_cpu(&model_config, model_dir, &device)?;
    evaluate_model_cpu(
        &model,
        data,
        &rows,
        &phone_vocab,
        &phoneme_vocab,
        &feature_bundle_vocab,
        task,
        sample_limit,
        split,
        &device,
    )
}

pub fn live_frame_stats(
    samples: &[f32],
    source_rate: u32,
    config: &CommonPhoneConfig,
) -> LiveFrameStats {
    let prepared = normalize_amplitude(&resample_linear(
        samples,
        source_rate,
        config.sample_rate_hz,
    ));
    let features = compact_audio_features(&prepared, config);
    frame_stats_from_features(&prepared, &features, config.feature_bins)
}

pub struct CommonPhoneLiveDecoder {
    task: CommonPhoneTask,
    model_config: ModelConfig,
    model: CommonPhoneModel<CpuInferBackend>,
    phone_vocab: Vocab,
    phoneme_vocab: Vocab,
    feature_bundle_vocab: Vocab,
}

impl CommonPhoneLiveDecoder {
    pub fn load(model_dir: &Path, task: CommonPhoneTask) -> Result<Self> {
        let phone_vocab: Vocab = read_vocab(model_dir, "phone_vocab.json", "phones.json")?;
        let phoneme_vocab: Vocab = read_vocab(model_dir, "phoneme_vocab.json", "phonemes.json")?;
        let feature_bundle_vocab: Vocab = read_vocab(
            model_dir,
            "feature_bundle_vocab.json",
            "feature_bundles.json",
        )?;
        let model_config: ModelConfig = read_json(&model_dir.join("model_config.json"))?;
        let device = NdArrayDevice::Cpu;
        let model = load_model_cpu(&model_config, model_dir, &device)?;
        Ok(Self {
            task,
            model_config,
            model,
            phone_vocab,
            phoneme_vocab,
            feature_bundle_vocab,
        })
    }

    pub fn config(&self, sample_rate_hz: u32, window_ms: f32, hop_ms: f32) -> CommonPhoneConfig {
        CommonPhoneConfig {
            sample_rate_hz,
            window_ms,
            hop_ms,
            frame_hz: (1000.0 / hop_ms.max(1.0)).round() as u32,
            feature_bins: self.model_config.input_feature_bins,
            ..CommonPhoneConfig::default()
        }
    }

    pub fn decode_samples(
        &self,
        samples: &[f32],
        source_rate: u32,
        config: &CommonPhoneConfig,
    ) -> Result<LiveDecode> {
        let prepared = normalize_amplitude(&resample_linear(
            samples,
            source_rate,
            config.sample_rate_hz,
        ));
        let features = compact_audio_features(&prepared, config);
        let stats = frame_stats_from_features(&prepared, &features, config.feature_bins);
        let (tokens, blank_ratio) = self.decode_features(&features)?;
        let (phones, feature_bundles) = match self.task {
            CommonPhoneTask::Frames2Phones | CommonPhoneTask::Multitask => {
                let feature_bundles = feature_bundles_for_phones(&tokens);
                (tokens, feature_bundles)
            }
            CommonPhoneTask::Frames2Features => (Vec::new(), tokens),
            CommonPhoneTask::Frames2Phonemes => {
                let feature_bundles = feature_bundles_for_phones(&tokens);
                (tokens, feature_bundles)
            }
        };
        Ok(LiveDecode {
            prediction_length: phones.len().max(feature_bundles.len()),
            phones,
            feature_bundles,
            blank_ratio,
            stats,
        })
    }

    fn decode_features(&self, features: &[Vec<f32>]) -> Result<(Vec<String>, f64)> {
        let device = NdArrayDevice::Cpu;
        let frames = features.len().max(1);
        let bins = self.model_config.input_feature_bins.max(1);
        let mut values = Vec::with_capacity(frames * bins);
        for frame in features {
            for bin in 0..bins {
                values.push(frame.get(bin).copied().unwrap_or(0.0));
            }
        }
        if values.is_empty() {
            values.resize(bins, 0.0);
        }
        let input = Tensor::<CpuInferBackend, 3>::from_data(
            TensorData::new(values, [1, frames, bins]),
            &device,
        );
        let output = self.model.forward(input);
        let (ids, vocab) = match self.task {
            CommonPhoneTask::Frames2Phones | CommonPhoneTask::Multitask => {
                (argmax_ids(output.phone_logits, frames), &self.phone_vocab)
            }
            CommonPhoneTask::Frames2Features => (
                argmax_ids(output.feature_bundle_logits, frames),
                &self.feature_bundle_vocab,
            ),
            CommonPhoneTask::Frames2Phonemes => (
                argmax_ids(output.phoneme_logits, frames),
                &self.phoneme_vocab,
            ),
        };
        let blank_ratio = if ids.is_empty() {
            0.0
        } else {
            ids.iter().filter(|&&id| id == 0).count() as f64 / ids.len() as f64
        };
        let tokens = ctc_greedy_decode(&ids, 0)
            .into_iter()
            .map(|id| vocab.get_token(id).to_string())
            .collect();
        Ok((tokens, blank_ratio))
    }
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
        .take(8)
        .enumerate()
        .map(|(index, frame)| frame_summary(index, frame))
        .collect();
    Ok(ShowRow {
        utterance_id: row.utterance_id.clone(),
        lang: row.lang.clone(),
        phones: row.phones.clone(),
        feature_bundles: row.feature_bundles.clone(),
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
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    let (rows, bins) = if &magic == ACF_MAGIC {
        let mut header = [0u8; 12];
        file.read_exact(&mut header)?;
        let _version = u32::from_le_bytes(header[0..4].try_into().unwrap());
        (
            u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize,
            u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize,
        )
    } else {
        let mut rest = [0u8; 4];
        file.read_exact(&mut rest)?;
        (
            u32::from_le_bytes(magic) as usize,
            u32::from_le_bytes(rest) as usize,
        )
    };
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
    let common_phone = discover_common_phone_source_records(input)?;
    if !common_phone.is_empty() {
        progress(PrepareProgress::Parse {
            rows: common_phone.len(),
            path: input.display().to_string(),
        });
        return Ok(common_phone);
    }
    anyhow::bail!(
        "unsupported Common Phone layout at {}; expected metadata.jsonl/csv/tsv or extracted cp-1-0 language directories",
        input.display()
    );
}

#[derive(Module, Debug)]
pub struct CommonPhoneModel<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    dropout: Dropout,
    phone: Linear<B>,
    phoneme: Linear<B>,
    feature_bundle: Linear<B>,
}

#[derive(Debug)]
struct CommonPhoneForward<B: Backend> {
    phone_logits: Tensor<B, 3>,
    phoneme_logits: Tensor<B, 3>,
    feature_bundle_logits: Tensor<B, 3>,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CommonPhoneModel<B> {
        CommonPhoneModel {
            input: LinearConfig::new(self.input_feature_bins, self.hidden_dim).init(device),
            hidden: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            phone: LinearConfig::new(self.hidden_dim, self.phone_vocab_size).init(device),
            phoneme: LinearConfig::new(self.hidden_dim, self.phoneme_vocab_size).init(device),
            feature_bundle: LinearConfig::new(self.hidden_dim, self.feature_bundle_vocab_size)
                .init(device),
        }
    }
}

impl<B: Backend> CommonPhoneModel<B> {
    fn forward(&self, frames: Tensor<B, 3>) -> CommonPhoneForward<B> {
        let hidden = self.dropout.forward(
            self.hidden
                .forward(self.input.forward(frames).tanh())
                .tanh(),
        );
        CommonPhoneForward {
            phone_logits: self.phone.forward(hidden.clone()),
            phoneme_logits: self.phoneme.forward(hidden.clone()),
            feature_bundle_logits: self.feature_bundle.forward(hidden),
        }
    }
}

fn load_model_cpu(
    model_config: &ModelConfig,
    model_dir: &Path,
    device: &NdArrayDevice,
) -> Result<CommonPhoneModel<CpuInferBackend>> {
    model_config
        .init(device)
        .load_file(&model_dir.join("model"), &make_recorder(), device)
        .context("loading Common Phone model")
}

#[derive(Debug)]
struct CommonPhoneBatch<B: Backend> {
    frames: Tensor<B, 3>,
    input_lengths: Tensor<B, 1, Int>,
    phone_targets: Tensor<B, 2, Int>,
    phone_target_lengths: Tensor<B, 1, Int>,
    phoneme_targets: Tensor<B, 2, Int>,
    phoneme_target_lengths: Tensor<B, 1, Int>,
    feature_bundle_targets: Tensor<B, 2, Int>,
    feature_bundle_target_lengths: Tensor<B, 1, Int>,
}

fn train_epoch_cpu<R: rand::Rng>(
    model: &mut CommonPhoneModel<CpuTrainBackend>,
    optimizer: &mut impl Optimizer<CommonPhoneModel<CpuTrainBackend>, CpuTrainBackend>,
    config: &CommonPhoneTrainConfig,
    data_dir: &Path,
    rows: &[CommonPhoneRow],
    phone_vocab: &Vocab,
    phoneme_vocab: &Vocab,
    feature_bundle_vocab: &Vocab,
    device: &NdArrayDevice,
    rng: &mut R,
    out: &Path,
    epoch: usize,
    best_validation_error_rate: f64,
    progress: &mut impl FnMut(TrainProgress),
) -> Result<f32> {
    let mut indices = (0..rows.len()).collect::<Vec<_>>();
    indices.shuffle(rng);
    let batches = batch_indices_by_frames(&indices, rows, config.batch_frames);
    let pb = tongues_core::register_progress_bar(indicatif::ProgressBar::new(batches.len() as u64));
    let mut total = 0.0;
    let mut n = 0usize;
    let mut examples_seen = 0usize;
    for batch_indices in batches {
        let batch_rows = batch_indices
            .iter()
            .map(|&index| rows[index].clone())
            .collect::<Vec<_>>();
        examples_seen += batch_rows.len();
        let batch = make_common_phone_batch::<CpuTrainBackend>(
            data_dir,
            &batch_rows,
            phone_vocab,
            phoneme_vocab,
            feature_bundle_vocab,
            device,
        )?;
        let output = model.forward(batch.frames.clone());
        let loss = common_phone_loss(output, batch, config);
        let loss_val = loss.clone().into_scalar().elem::<f32>();
        let grads = GradientsParams::from_grads(loss.backward(), model);
        *model = optimizer.step(config.learning_rate, model.clone(), grads);
        total += loss_val;
        n += 1;
        if n <= 3 || n % 20 == 0 || examples_seen == rows.len() {
            progress(TrainProgress::Batch {
                epoch,
                examples: examples_seen,
                total_examples: rows.len(),
                loss: total / n as f32,
            });
        }
        if n % TRAIN_CHECKPOINT_BATCH_INTERVAL == 0 {
            model
                .valid()
                .save_file(&out.join(LATEST_MODEL_STEM), &make_recorder())?;
            write_train_state(
                out,
                epoch,
                best_validation_error_rate,
                config,
                "epoch-in-progress",
                Some(format!("{LATEST_MODEL_STEM}.bin")),
                Some(total / n as f32),
            )?;
            progress(TrainProgress::Checkpoint {
                epoch,
                path: out
                    .join(format!("{LATEST_MODEL_STEM}.bin"))
                    .display()
                    .to_string(),
                best: false,
            });
            progress(TrainProgress::State {
                epoch,
                path: out.join("train_state.json").display().to_string(),
            });
        }
        pb.set_message(format!("{:.4}", total / n as f32));
        pb.inc(1);
    }
    pb.finish_and_clear();
    Ok(if n == 0 { 0.0 } else { total / n as f32 })
}

fn common_phone_loss<B: Backend>(
    output: CommonPhoneForward<B>,
    batch: CommonPhoneBatch<B>,
    config: &CommonPhoneTrainConfig,
) -> Tensor<B, 1> {
    let phone = ctc_loss(
        output.phone_logits,
        batch.phone_targets,
        batch.input_lengths.clone(),
        batch.phone_target_lengths,
        0,
    ) * config.phone_ctc_loss_weight;
    let feature = ctc_loss(
        output.feature_bundle_logits,
        batch.feature_bundle_targets,
        batch.input_lengths.clone(),
        batch.feature_bundle_target_lengths,
        0,
    ) * config.feature_bundle_ctc_loss_weight;
    let phoneme = ctc_loss(
        output.phoneme_logits,
        batch.phoneme_targets,
        batch.input_lengths,
        batch.phoneme_target_lengths,
        0,
    ) * config.phoneme_ctc_loss_weight;
    match config.task {
        CommonPhoneTask::Frames2Phones => phone,
        CommonPhoneTask::Frames2Features => feature,
        CommonPhoneTask::Frames2Phonemes => phoneme,
        CommonPhoneTask::Multitask => phone + feature + phoneme,
    }
}

fn ctc_loss<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    input_lengths: Tensor<B, 1, Int>,
    target_lengths: Tensor<B, 1, Int>,
    blank: usize,
) -> Tensor<B, 1> {
    let log_probs = log_softmax(logits.swap_dims(0, 1), 2);
    CTCLossConfig::new()
        .with_blank(blank)
        .with_zero_infinity(true)
        .init()
        .forward_with_reduction(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            Reduction::Mean,
        )
}

fn batch_indices_by_frames(
    indices: &[usize],
    rows: &[CommonPhoneRow],
    batch_frames: usize,
) -> Vec<Vec<usize>> {
    let mut batches = Vec::new();
    let mut current = Vec::new();
    let mut frames = 0usize;
    for &index in indices {
        let row_frames = rows[index].frame_count.max(1);
        if !current.is_empty() && frames + row_frames > batch_frames.max(1) {
            batches.push(std::mem::take(&mut current));
            frames = 0;
        }
        current.push(index);
        frames += row_frames;
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

fn make_common_phone_batch<B: Backend>(
    data_dir: &Path,
    rows: &[CommonPhoneRow],
    phone_vocab: &Vocab,
    phoneme_vocab: &Vocab,
    feature_bundle_vocab: &Vocab,
    device: &B::Device,
) -> Result<CommonPhoneBatch<B>> {
    let max_frames = rows
        .iter()
        .map(|row| row.frame_count)
        .max()
        .unwrap_or(1)
        .max(1);
    let frame_dim = rows
        .iter()
        .map(|row| row.frame_dim)
        .max()
        .unwrap_or(1)
        .max(1);
    let mut frames = Vec::with_capacity(rows.len() * max_frames * frame_dim);
    let mut input_lengths = Vec::with_capacity(rows.len());
    let mut phone_sequences = Vec::new();
    let mut phoneme_sequences = Vec::new();
    let mut feature_bundle_sequences = Vec::new();
    for row in rows {
        let (frame_count, bins, values) = read_feature_file(&data_dir.join(&row.feature_path))?;
        input_lengths.push(frame_count.min(max_frames).max(1) as i32);
        for frame in 0..max_frames {
            let start = frame * bins;
            for bin in 0..frame_dim {
                frames.push(
                    values
                        .get(start + bin)
                        .copied()
                        .filter(|_| frame < frame_count && bin < bins)
                        .unwrap_or(0.0),
                );
            }
        }
        phone_sequences.push(ctc_target_within_input(
            row.phones
                .iter()
                .map(|token| phone_vocab.get_id(token).max(1) as i32)
                .collect(),
            frame_count,
        ));
        phoneme_sequences.push(ctc_target_within_input(
            row.phonemes
                .iter()
                .map(|token| phoneme_vocab.get_id(token).max(1) as i32)
                .collect(),
            frame_count,
        ));
        feature_bundle_sequences.push(ctc_target_within_input(
            row.feature_bundles
                .iter()
                .map(|token| feature_bundle_vocab.get_id(token).max(1) as i32)
                .collect(),
            frame_count,
        ));
    }
    let (phone_targets, phone_target_lengths, phone_width) =
        pad_compact_targets(phone_sequences, 1);
    let (phoneme_targets, phoneme_target_lengths, phoneme_width) =
        pad_compact_targets(phoneme_sequences, 1);
    let (feature_bundle_targets, feature_bundle_target_lengths, feature_width) =
        pad_compact_targets(feature_bundle_sequences, 1);
    Ok(CommonPhoneBatch {
        frames: Tensor::<B, 3>::from_data(
            TensorData::new(frames, [rows.len(), max_frames, frame_dim]),
            device,
        ),
        input_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(input_lengths, [rows.len()]),
            device,
        ),
        phone_targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(phone_targets, [rows.len(), phone_width]),
            device,
        ),
        phone_target_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(phone_target_lengths, [rows.len()]),
            device,
        ),
        phoneme_targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(phoneme_targets, [rows.len(), phoneme_width]),
            device,
        ),
        phoneme_target_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(phoneme_target_lengths, [rows.len()]),
            device,
        ),
        feature_bundle_targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(feature_bundle_targets, [rows.len(), feature_width]),
            device,
        ),
        feature_bundle_target_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(feature_bundle_target_lengths, [rows.len()]),
            device,
        ),
    })
}

fn pad_compact_targets(
    mut sequences: Vec<Vec<i32>>,
    fallback_id: u32,
) -> (Vec<i32>, Vec<i32>, usize) {
    let fallback = fallback_id.max(1) as i32;
    let mut lengths = Vec::with_capacity(sequences.len());
    for sequence in &mut sequences {
        sequence.retain(|id| *id != 0);
        if sequence.is_empty() {
            sequence.push(fallback);
        }
        lengths.push(sequence.len() as i32);
    }
    let width = sequences.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut padded = Vec::with_capacity(sequences.len() * width);
    for mut sequence in sequences {
        sequence.resize(width, fallback);
        padded.extend(sequence);
    }
    (padded, lengths, width)
}

fn ctc_target_within_input(mut sequence: Vec<i32>, input_len: usize) -> Vec<i32> {
    sequence.truncate(input_len.max(1));
    sequence
}

#[allow(clippy::too_many_arguments)]
fn evaluate_model_cpu(
    model: &CommonPhoneModel<CpuInferBackend>,
    data_dir: &Path,
    rows: &[CommonPhoneRow],
    phone_vocab: &Vocab,
    phoneme_vocab: &Vocab,
    feature_bundle_vocab: &Vocab,
    task: CommonPhoneTask,
    sample_limit: usize,
    split: &str,
    device: &NdArrayDevice,
) -> Result<EvalReport> {
    let mut stats = EvalStats::default();
    let mut samples = Vec::new();
    let mut language_distribution = BTreeMap::new();
    for row in rows {
        *language_distribution.entry(row.lang.clone()).or_insert(0) += 1;
        let batch = make_common_phone_batch::<CpuInferBackend>(
            data_dir,
            std::slice::from_ref(row),
            phone_vocab,
            phoneme_vocab,
            feature_bundle_vocab,
            device,
        )?;
        let output = model.forward(batch.frames);
        let (ids, vocab, target) = match task {
            CommonPhoneTask::Frames2Phones | CommonPhoneTask::Multitask => (
                argmax_ids(output.phone_logits, row.frame_count),
                phone_vocab,
                row.phones.clone(),
            ),
            CommonPhoneTask::Frames2Features => (
                argmax_ids(output.feature_bundle_logits, row.frame_count),
                feature_bundle_vocab,
                row.feature_bundles.clone(),
            ),
            CommonPhoneTask::Frames2Phonemes => (
                argmax_ids(output.phoneme_logits, row.frame_count),
                phoneme_vocab,
                row.phonemes.clone(),
            ),
        };
        stats.frame_predictions += ids.len();
        stats.blank_predictions += ids.iter().filter(|&&id| id == 0).count();
        let pred_ids = ctc_greedy_decode(&ids, 0);
        let prediction = pred_ids
            .iter()
            .map(|id| vocab.get_token(*id).to_string())
            .collect::<Vec<_>>();
        stats.add(&prediction, &target);
        if samples.len() < sample_limit {
            samples.push(GreedySample {
                utterance_id: row.utterance_id.clone(),
                lang: row.lang.clone(),
                phone_target: target,
                phone_prediction: prediction,
            });
        }
    }
    let axis_rates = BTreeMap::new();
    Ok(EvalReport {
        split: split.to_string(),
        task,
        examples_evaluated: rows.len(),
        examples_failed: 0,
        token_error_rate: stats.rate(),
        edit_distance: stats.edits,
        exact_sequence_accuracy: stats.exact_accuracy(),
        blank_ratio: stats.blank_ratio(),
        mean_prediction_length: stats.mean_prediction_length(),
        mean_target_length: stats.mean_target_length(),
        phone_token_error_rate: (task == CommonPhoneTask::Frames2Phones
            || task == CommonPhoneTask::Multitask)
            .then(|| stats.rate()),
        phoneme_token_error_rate: (task == CommonPhoneTask::Frames2Phonemes).then(|| stats.rate()),
        feature_bundle_error_rate: (task == CommonPhoneTask::Frames2Features).then(|| stats.rate()),
        feature_axis_token_error_rate: axis_rates,
        aggregate_feature_token_error_rate: 0.0,
        unknown_phone_symbols: count_unknown_phones(rows),
        language_distribution,
        samples,
    })
}

fn argmax_ids<B: Backend>(logits: Tensor<B, 3>, frames: usize) -> Vec<u32> {
    let [batch, time, classes] = logits.dims();
    debug_assert_eq!(batch, 1);
    let values: Vec<f32> = logits.into_data().to_vec().unwrap_or_default();
    let mut ids = Vec::with_capacity(time.min(frames));
    for frame in 0..time.min(frames) {
        let offset = frame * classes;
        let mut best_id = 0usize;
        let mut best = f32::NEG_INFINITY;
        for class in 0..classes {
            let value = values
                .get(offset + class)
                .copied()
                .unwrap_or(f32::NEG_INFINITY);
            if value > best {
                best = value;
                best_id = class;
            }
        }
        ids.push(best_id as u32);
    }
    ids
}

fn ctc_greedy_decode(ids: &[u32], blank: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut prev = None;
    for &id in ids {
        if Some(id) != prev && id != blank {
            out.push(id);
        }
        prev = Some(id);
    }
    out
}

#[derive(Default)]
struct EvalStats {
    edits: usize,
    target_tokens: usize,
    exact: usize,
    examples: usize,
    prediction_tokens: usize,
    frame_predictions: usize,
    blank_predictions: usize,
}

impl EvalStats {
    fn add(&mut self, prediction: &[String], target: &[String]) {
        let edits = edit_distance(prediction, target);
        self.edits += edits;
        self.target_tokens += target.len().max(1);
        self.prediction_tokens += prediction.len();
        self.exact += usize::from(edits == 0);
        self.examples += 1;
    }

    fn rate(&self) -> f64 {
        if self.target_tokens == 0 {
            0.0
        } else {
            self.edits as f64 / self.target_tokens as f64
        }
    }

    fn exact_accuracy(&self) -> f64 {
        if self.examples == 0 {
            0.0
        } else {
            self.exact as f64 / self.examples as f64
        }
    }

    fn blank_ratio(&self) -> f64 {
        if self.frame_predictions == 0 {
            0.0
        } else {
            self.blank_predictions as f64 / self.frame_predictions as f64
        }
    }

    fn mean_prediction_length(&self) -> f64 {
        if self.examples == 0 {
            0.0
        } else {
            self.prediction_tokens as f64 / self.examples as f64
        }
    }

    fn mean_target_length(&self) -> f64 {
        if self.examples == 0 {
            0.0
        } else {
            self.target_tokens as f64 / self.examples as f64
        }
    }
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
    let mut out = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            if quoted && chars.peek() == Some(&'"') {
                current.push('"');
                chars.next();
            } else {
                quoted = !quoted;
            }
        } else if ch == delimiter && !quoted {
            out.push(current.trim().to_string());
            current.clear();
        } else {
            current.push(ch);
        }
    }
    out.push(current.trim().to_string());
    out
}

fn discover_common_phone_source_records(input: &Path) -> Result<Vec<InputRecord>> {
    let mut rows = Vec::new();
    if !input.exists() {
        return Ok(rows);
    }
    for entry in fs::read_dir(input)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let lang_dir = entry.path();
        let Some(lang_name) = lang_dir.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(language) = common_phone_language_code(lang_name) else {
            continue;
        };
        for split in ["train", "dev", "valid", "test"] {
            let csv = lang_dir.join(format!("{split}.csv"));
            if !csv.exists() {
                continue;
            }
            rows.extend(read_common_phone_split_csv(
                input, &lang_dir, &language, split, &csv,
            )?);
        }
    }
    Ok(rows)
}

fn has_common_phone_source_layout(input: &Path) -> Result<bool> {
    Ok(!discover_common_phone_source_records(input)?.is_empty())
}

fn read_common_phone_split_csv(
    root: &Path,
    lang_dir: &Path,
    language: &str,
    split: &str,
    csv: &Path,
) -> Result<Vec<InputRecord>> {
    let file = File::open(csv).with_context(|| format!("opening {}", csv.display()))?;
    let mut lines = BufReader::new(file).lines();
    let Some(header) = lines.next().transpose()? else {
        return Ok(Vec::new());
    };
    let columns = split_delimited_line(&header, ',');
    let has_named_header = columns.iter().any(|col| {
        matches!(
            normalize_header(col).as_str(),
            "path" | "filename" | "file" | "client_id" | "sentence" | "text" | "speaker_id"
        )
    });
    let mut records = Vec::new();
    let data_lines = if has_named_header {
        lines.collect::<Result<Vec<_>, _>>()?
    } else {
        let mut all = vec![header];
        all.extend(lines.collect::<Result<Vec<_>, _>>()?);
        all
    };
    for (line_index, line) in data_lines.into_iter().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = split_delimited_line(&line, ',');
        let mut by_name = BTreeMap::new();
        if has_named_header {
            for (name, value) in columns.iter().zip(fields.iter()) {
                by_name.insert(normalize_header(name), value.clone());
            }
        }
        let raw_path = first_field(
            &by_name,
            &["path", "filename", "file", "wav_path", "audio_path", "clip"],
        )
        .or_else(|| fields.first().cloned())
        .unwrap_or_else(|| format!("{language}_{split}_{line_index:08}.wav"));
        let stem = Path::new(&raw_path)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or(raw_path.as_str())
            .to_string();
        let wav_path = find_common_phone_wav(root, lang_dir, &raw_path, &stem);
        let grid_path = find_common_phone_grid(lang_dir, &stem);
        let phones = grid_path
            .as_ref()
            .and_then(|path| textgrid_tier_tokens(path, "MAU").ok())
            .unwrap_or_default();
        if phones.is_empty() {
            continue;
        }
        let phonemes = grid_path
            .as_ref()
            .and_then(|path| textgrid_tier_tokens(path, "KAN-MAU").ok())
            .filter(|tokens| !tokens.is_empty())
            .unwrap_or_else(|| phones.clone());
        let speaker_id = first_field(
            &by_name,
            &[
                "speaker_id",
                "speaker",
                "client_id",
                "speakerid",
                "clientid",
            ],
        )
        .or_else(|| fields.get(1).cloned());
        let text = first_field(&by_name, &["sentence", "text", "transcript"])
            .or_else(|| fields.get(2).cloned());
        let variety = first_field(&by_name, &["accent", "variety"]);
        let split = if split == "dev" { "valid" } else { split }.to_string();
        records.push(InputRecord {
            utterance_id: Some(format!("{language}_{stem}")),
            id: None,
            lang: Some(language.to_string()),
            language: Some(language.to_string()),
            variety,
            speaker_id,
            speaker: None,
            audio_path: None,
            wav_path: Some(path_relative_to(root, &wav_path)),
            path: None,
            wav: None,
            split: Some(split),
            text,
            duration_ms: None,
            source_dataset: Some("common-phone-zenodo".to_string()),
            phones: Some(PhoneField::Tokens(phones.clone())),
            phonemes: Some(PhoneField::Tokens(phonemes)),
            segments: grid_path
                .map(|path| serde_json::json!({ "textgrid_path": path.display().to_string() })),
            extra: BTreeMap::new(),
        });
    }
    Ok(records)
}

fn normalize_header(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .to_ascii_lowercase()
        .replace(['-', ' '], "_")
}

fn first_field(map: &BTreeMap<String, String>, names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        map.get(*name)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

fn common_phone_language_code(name: &str) -> Option<String> {
    match name.to_ascii_lowercase().as_str() {
        "english" | "eng" | "en" => Some("eng".to_string()),
        "french" | "fra" | "fre" | "fr" => Some("fra".to_string()),
        "german" | "deu" | "ger" | "de" => Some("deu".to_string()),
        "italian" | "ita" | "it" => Some("ita".to_string()),
        "spanish" | "spa" | "es" => Some("spa".to_string()),
        "russian" | "rus" | "ru" => Some("rus".to_string()),
        _ => None,
    }
}

fn find_common_phone_wav(root: &Path, lang_dir: &Path, raw_path: &str, stem: &str) -> PathBuf {
    let raw = Path::new(raw_path);
    let candidates = [
        lang_dir.join(raw),
        lang_dir.join("wav").join(format!("{stem}.wav")),
        lang_dir.join("wavs").join(format!("{stem}.wav")),
        lang_dir.join("audio").join(format!("{stem}.wav")),
        root.join(raw),
    ];
    candidates
        .iter()
        .find(|path| path.exists())
        .cloned()
        .unwrap_or_else(|| lang_dir.join("wav").join(format!("{stem}.wav")))
}

fn find_common_phone_grid(lang_dir: &Path, stem: &str) -> Option<PathBuf> {
    [
        lang_dir.join("grids").join(format!("{stem}.TextGrid")),
        lang_dir.join("grids").join(format!("{stem}.textgrid")),
        lang_dir.join("grid").join(format!("{stem}.TextGrid")),
    ]
    .into_iter()
    .find(|path| path.exists())
}

fn textgrid_tier_tokens(path: &Path, tier_name: &str) -> Result<Vec<String>> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut tokens = Vec::new();
    let mut active_tier: Option<String> = None;
    let mut next_short_tier_name = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("item [") {
            active_tier = None;
            next_short_tier_name = false;
            continue;
        }
        if matches!(
            unquoted_textgrid_line(line).as_deref(),
            Some("IntervalTier" | "TextTier")
        ) {
            next_short_tier_name = true;
            continue;
        }
        if next_short_tier_name {
            if let Some(name) = unquoted_textgrid_line(line) {
                active_tier = Some(name);
                next_short_tier_name = false;
            }
            continue;
        }
        if line.starts_with("name") {
            if let Some((_, value)) = line.split_once('=') {
                active_tier = Some(value.trim().trim_matches('"').to_string());
            }
            continue;
        }
        if active_tier.as_deref() != Some(tier_name) {
            continue;
        }
        if line.starts_with("text") || line.starts_with("mark") {
            let Some((_, value)) = line.split_once('=') else {
                continue;
            };
            let symbol = value.trim().trim_matches('"').trim();
            tokens.extend(tokenize_textgrid_label(symbol));
            continue;
        }
        if let Some(symbol) = unquoted_textgrid_line(line) {
            tokens.extend(tokenize_textgrid_label(&symbol));
            continue;
        }
    }
    Ok(tokens)
}

fn unquoted_textgrid_line(line: &str) -> Option<String> {
    let line = line.trim();
    if line.len() >= 2 && line.starts_with('"') && line.ends_with('"') {
        Some(line.trim_matches('"').trim().to_string())
    } else {
        None
    }
}

fn path_relative_to(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn download_to_part(
    url: &str,
    path: &Path,
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<()> {
    let part = path.with_extension(format!(
        "{}.part",
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("download")
    ));
    if part.exists() {
        fs::remove_file(&part).with_context(|| format!("removing {}", part.display()))?;
    }
    let response = ureq::get(url)
        .header("User-Agent", DOWNLOAD_USER_AGENT)
        .call()
        .with_context(|| format!("downloading {url}"))?;
    let mut reader = response.into_body().into_reader();
    let mut writer = BufWriter::new(File::create(&part)?);
    let mut buf = [0u8; 256 * 1024];
    let mut bytes = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n])?;
        bytes += n as u64;
        if bytes < 1024 * 1024 || bytes % (64 * 1024 * 1024) < buf.len() as u64 {
            progress(PrepareProgress::Download {
                url: url.to_string(),
                path: part.display().to_string(),
                bytes,
            });
        }
    }
    writer.flush()?;
    fs::rename(&part, path)?;
    Ok(())
}

fn merge_extracted_tree(src: &Path, dst: &Path) -> Result<()> {
    let mut roots = fs::read_dir(src)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    if roots.len() == 1 && roots[0].is_dir() && has_language_children(&roots[0])? {
        move_children(&roots.remove(0), dst)
    } else {
        move_children(src, dst)
    }
}

fn has_language_children(path: &Path) -> Result<bool> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_dir()
            && entry
                .file_name()
                .to_str()
                .and_then(common_phone_language_code)
                .is_some()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn move_children(src: &Path, dst: &Path) -> Result<()> {
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let target = dst.join(entry.file_name());
        if target.exists() {
            if target.is_dir() {
                fs::remove_dir_all(&target)?;
            } else {
                fs::remove_file(&target)?;
            }
        }
        fs::rename(entry.path(), target)?;
    }
    Ok(())
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
    let tokens = match field {
        PhoneField::Tokens(tokens) => tokens.clone(),
        PhoneField::Text(text) => tokenize_phone_text(text),
    };
    sanitize_phone_tokens(tokens)
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

fn tokenize_textgrid_label(text: &str) -> Vec<String> {
    let tokens = text
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    sanitize_phone_tokens(tokens)
}

fn sanitize_phone_tokens(tokens: Vec<String>) -> Vec<String> {
    tokens
        .into_iter()
        .filter_map(|token| normalize_phone_token(&token))
        .collect()
}

fn normalize_phone_token(token: &str) -> Option<String> {
    let token = token
        .trim()
        .trim_matches('/')
        .trim_matches('[')
        .trim_matches(']')
        .trim_matches('"')
        .trim();
    if token.is_empty()
        || token == "(...)"
        || token == "."
        || token == "|"
        || token == "ˈ"
        || token == "ˌ"
        || token == CTC_BLANK
        || token == UNK
        || token.eq_ignore_ascii_case("sil")
        || token.eq_ignore_ascii_case("sp")
        || token.eq_ignore_ascii_case("silence")
        || token.eq_ignore_ascii_case("<sil>")
        || token.eq_ignore_ascii_case("<sp>")
    {
        return None;
    }
    let token = token
        .trim_start_matches(|ch| ch == 'ˈ' || ch == 'ˌ')
        .trim_end_matches(|ch| ch == 'ˈ' || ch == 'ˌ')
        .to_string();
    (!token.is_empty()).then_some(token)
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

fn normalize_amplitude(samples: &[f32]) -> Vec<f32> {
    let peak = samples
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    if peak <= 1e-6 || peak <= 0.95 {
        samples.to_vec()
    } else {
        samples.iter().map(|value| value / peak * 0.95).collect()
    }
}

fn compact_audio_features(samples: &[f32], config: &CommonPhoneConfig) -> Vec<Vec<f32>> {
    let window = (config.sample_rate_hz as f32 * config.window_ms / 1000.0)
        .round()
        .max(1.0) as usize;
    let hop = (config.sample_rate_hz as f32 * config.hop_ms / 1000.0)
        .round()
        .max(1.0) as usize;
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
    writer.write_all(ACF_MAGIC)?;
    writer.write_all(&ACF_VERSION.to_le_bytes())?;
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
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    let (rows, bins) = if &magic == ACF_MAGIC {
        let mut header = [0u8; 12];
        file.read_exact(&mut header)?;
        let _version = u32::from_le_bytes(header[0..4].try_into().unwrap());
        (
            u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize,
            u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize,
        )
    } else {
        let mut rest = [0u8; 4];
        file.read_exact(&mut rest)?;
        (
            u32::from_le_bytes(magic) as usize,
            u32::from_le_bytes(rest) as usize,
        )
    };
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    Ok((rows, bins, values))
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
    let mut seen = BTreeSet::from([CTC_BLANK.to_string(), UNK.to_string()]);
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

fn split_rows(
    mut rows: Vec<CommonPhoneRow>,
    config: &CommonPhoneConfig,
) -> (
    Vec<CommonPhoneRow>,
    Vec<CommonPhoneRow>,
    Vec<CommonPhoneRow>,
) {
    if rows.iter().any(|row| row.split.is_some()) {
        let mut train = Vec::new();
        let mut valid = Vec::new();
        let mut test = Vec::new();
        for row in rows {
            match row.split.as_deref() {
                Some("valid" | "validation" | "dev") => valid.push(row),
                Some("test") => test.push(row),
                _ => train.push(row),
            }
        }
        return (train, valid, test);
    }
    rows.shuffle(&mut rand::rngs::StdRng::seed_from_u64(config.seed));
    let n = rows.len();
    let test_len = ((n as f64) * config.test_ratio).round().min(n as f64) as usize;
    let valid_len = ((n as f64) * config.valid_ratio)
        .round()
        .min(n.saturating_sub(test_len) as f64) as usize;
    let train_len = n.saturating_sub(valid_len + test_len);
    (
        rows[..train_len].to_vec(),
        rows[train_len..train_len + valid_len].to_vec(),
        rows[train_len + valid_len..].to_vec(),
    )
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

fn feature_bundles_for_phones(phones: &[String]) -> Vec<String> {
    phones
        .iter()
        .map(|phone| phone_feature_bundle(phone).to_string())
        .collect()
}

fn phone_feature_bundle(phone: &str) -> &'static str {
    match phone_feature_key(phone).as_str() {
        "p" => "<STOP:VL:BILAB>",
        "b" => "<STOP:VOICED:BILAB>",
        "t" => "<STOP:VL:ALV>",
        "d" => "<STOP:VOICED:ALV>",
        "c" => "<STOP:VL:PALATAL>",
        "ɟ" => "<STOP:VOICED:PALATAL>",
        "k" => "<STOP:VL:VELAR>",
        "g" => "<STOP:VOICED:VELAR>",
        "q" => "<STOP:VL:UVULAR>",
        "m" => "<NASAL:VOICED:BILAB>",
        "n" => "<NASAL:VOICED:ALV>",
        "ɲ" => "<NASAL:VOICED:PALATAL>",
        "ŋ" => "<NASAL:VOICED:VELAR>",
        "f" => "<FRIC:VL:LABIODENTAL>",
        "v" => "<FRIC:VOICED:LABIODENTAL>",
        "θ" => "<FRIC:VL:DENTAL>",
        "ð" => "<FRIC:VOICED:DENTAL>",
        "s" => "<FRIC:VL:ALV>",
        "z" => "<FRIC:VOICED:ALV>",
        "ʃ" => "<FRIC:VL:POSTALV>",
        "ʒ" => "<FRIC:VOICED:POSTALV>",
        "ç" => "<FRIC:VL:PALATAL>",
        "ʝ" => "<FRIC:VOICED:PALATAL>",
        "x" => "<FRIC:VL:VELAR>",
        "ɣ" => "<FRIC:VOICED:VELAR>",
        "χ" => "<FRIC:VL:UVULAR>",
        "ʁ" => "<FRIC:VOICED:UVULAR>",
        "β" => "<FRIC:VOICED:BILAB>",
        "h" => "<FRIC:VL:GLOTTAL>",
        "ts" => "<AFFRICATE:VL:ALV>",
        "dz" => "<AFFRICATE:VOICED:ALV>",
        "tʃ" => "<AFFRICATE:VL:POSTALV>",
        "dʒ" => "<AFFRICATE:VOICED:POSTALV>",
        "l" => "<LAT:VOICED:ALV>",
        "ʎ" => "<LAT:VOICED:PALATAL>",
        "r" | "ɹ" | "ɾ" | "ʀ" => "<APPROX:VOICED:ALV>",
        "j" => "<APPROX:VOICED:PALATAL>",
        "ɥ" => "<APPROX:VOICED:LABIAL_PALATAL>",
        "w" => "<APPROX:VOICED:LABIAL_VELAR>",
        "i" | "ɪ" | "ɨ" => "<VOWEL:HIGH:FRONT:UNROUNDED>",
        "y" | "ʏ" => "<VOWEL:HIGH:FRONT:ROUNDED>",
        "e" | "ɛ" | "æ" | "eɪ" | "eə" | "ɪə" => "<VOWEL:MID:FRONT:UNROUNDED>",
        "ø" | "œ" | "ɶ" => "<VOWEL:MID:FRONT:ROUNDED>",
        "a" | "ɑ" | "ɐ" | "ɒ" | "aɪ" | "aʊ" => "<VOWEL:LOW:CENTRAL:UNROUNDED>",
        "ə" | "ʌ" => "<VOWEL:MID:CENTRAL:UNROUNDED>",
        "o" | "ɔ" | "ɔɪ" | "əʊ" => "<VOWEL:MID:BACK:ROUNDED>",
        "u" | "ʊ" | "ʊə" => "<VOWEL:HIGH:BACK:ROUNDED>",
        "ʔ" => "<STOP:VL:GLOTTAL>",
        _ => PHONE_FEATURE_UNKNOWN,
    }
}

fn phone_features(phone: &str) -> BTreeMap<&'static str, &'static str> {
    let p = phone_feature_key(phone);
    let mut map = BTreeMap::new();
    let (manner, place, voicing, syllabic, height, backness, rounding) = match p.as_str() {
        "p" => ("stop", "bilabial", "voiceless", "no", NONE, NONE, NONE),
        "b" => ("stop", "bilabial", "voiced", "no", NONE, NONE, NONE),
        "t" => ("stop", "alveolar", "voiceless", "no", NONE, NONE, NONE),
        "d" => ("stop", "alveolar", "voiced", "no", NONE, NONE, NONE),
        "c" => ("stop", "palatal", "voiceless", "no", NONE, NONE, NONE),
        "ɟ" => ("stop", "palatal", "voiced", "no", NONE, NONE, NONE),
        "k" => ("stop", "velar", "voiceless", "no", NONE, NONE, NONE),
        "g" => ("stop", "velar", "voiced", "no", NONE, NONE, NONE),
        "q" => ("stop", "uvular", "voiceless", "no", NONE, NONE, NONE),
        "m" => ("nasal", "bilabial", "voiced", "no", NONE, NONE, NONE),
        "n" => ("nasal", "alveolar", "voiced", "no", NONE, NONE, NONE),
        "ɲ" => ("nasal", "palatal", "voiced", "no", NONE, NONE, NONE),
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
        "θ" => ("fricative", "dental", "voiceless", "no", NONE, NONE, NONE),
        "ð" => ("fricative", "dental", "voiced", "no", NONE, NONE, NONE),
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
        "ç" => ("fricative", "palatal", "voiceless", "no", NONE, NONE, NONE),
        "ʝ" => ("fricative", "palatal", "voiced", "no", NONE, NONE, NONE),
        "x" => ("fricative", "velar", "voiceless", "no", NONE, NONE, NONE),
        "ɣ" => ("fricative", "velar", "voiced", "no", NONE, NONE, NONE),
        "χ" => ("fricative", "uvular", "voiceless", "no", NONE, NONE, NONE),
        "ʁ" => ("fricative", "uvular", "voiced", "no", NONE, NONE, NONE),
        "β" => ("fricative", "bilabial", "voiced", "no", NONE, NONE, NONE),
        "ts" => ("affricate", "alveolar", "voiceless", "no", NONE, NONE, NONE),
        "dz" => ("affricate", "alveolar", "voiced", "no", NONE, NONE, NONE),
        "tʃ" => (
            "affricate",
            "postalveolar",
            "voiceless",
            "no",
            NONE,
            NONE,
            NONE,
        ),
        "dʒ" => (
            "affricate",
            "postalveolar",
            "voiced",
            "no",
            NONE,
            NONE,
            NONE,
        ),
        "l" => ("lateral", "alveolar", "voiced", "no", NONE, NONE, NONE),
        "ʎ" => ("lateral", "palatal", "voiced", "no", NONE, NONE, NONE),
        "r" | "ɹ" | "ɾ" | "ʀ" => ("approximant", "alveolar", "voiced", "no", NONE, NONE, NONE),
        "j" => ("approximant", "palatal", "voiced", "no", NONE, NONE, NONE),
        "ɥ" => (
            "approximant",
            "labial-palatal",
            "voiced",
            "no",
            NONE,
            NONE,
            NONE,
        ),
        "w" => (
            "approximant",
            "labial-velar",
            "voiced",
            "no",
            NONE,
            NONE,
            NONE,
        ),
        "i" | "ɪ" | "ɨ" => (
            "vowel",
            "vowel",
            "vowel",
            "yes",
            "high",
            "front",
            "unrounded",
        ),
        "y" | "ʏ" => ("vowel", "vowel", "vowel", "yes", "high", "front", "rounded"),
        "e" | "ɛ" | "æ" | "eɪ" | "eə" | "ɪə" => (
            "vowel",
            "vowel",
            "vowel",
            "yes",
            "mid",
            "front",
            "unrounded",
        ),
        "ø" | "œ" | "ɶ" => ("vowel", "vowel", "vowel", "yes", "mid", "front", "rounded"),
        "a" | "ɑ" | "ɐ" | "ɒ" | "aɪ" | "aʊ" => (
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
        "o" | "ɔ" | "ɔɪ" | "əʊ" => {
            ("vowel", "vowel", "vowel", "yes", "mid", "back", "rounded")
        }
        "u" | "ʊ" | "ʊə" => ("vowel", "vowel", "vowel", "yes", "high", "back", "rounded"),
        "ʔ" => ("stop", "glottal", "voiceless", "no", NONE, NONE, NONE),
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

fn phone_feature_key(phone: &str) -> String {
    phone
        .trim()
        .trim_matches(|ch: char| ch == '/' || ch == '[' || ch == ']' || ch == 'ˈ' || ch == 'ˌ')
        .chars()
        .filter_map(|ch| match ch {
            'ː' | ':' | 'ʲ' => None,
            '\u{0300}'..='\u{036f}' => None,
            'ɡ' => Some('g'),
            other => Some(other.to_lowercase().next().unwrap_or(other)),
        })
        .collect()
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
        "feature_bundle_vocab.json",
        "feature_axis_vocabs.json",
    ] {
        if data.join(name).exists() {
            fs::copy(data.join(name), out.join(name))
                .with_context(|| format!("copying {name} into {}", out.display()))?;
        }
    }
    fs::create_dir_all(out.join("vocabs"))?;
    for name in ["phones.json", "phonemes.json", "feature_bundles.json"] {
        let src = data.join("vocabs").join(name);
        if src.exists() {
            fs::copy(&src, out.join("vocabs").join(name))
                .with_context(|| format!("copying {} into {}", src.display(), out.display()))?;
        }
    }
    let manifest = ModelArtifactManifest::new(FAMILY, ARCHITECTURE, data_id_from_path(data))
        .with_task("compact-frame-phone-feature-ctc");
    write_manifest(out, &manifest)?;
    Ok(())
}

fn read_vocab(data: &Path, legacy_name: &str, vocab_name: &str) -> Result<Vocab> {
    let legacy = data.join(legacy_name);
    if legacy.exists() {
        return read_json(&legacy);
    }
    read_json(&data.join("vocabs").join(vocab_name))
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

fn write_train_state(
    out: &Path,
    epoch: usize,
    best_validation_error_rate: f64,
    config: &CommonPhoneTrainConfig,
    status: &str,
    checkpoint: Option<String>,
    train_loss: Option<f32>,
) -> Result<()> {
    write_text_atomic(
        &out.join("train_state.json"),
        &serde_json::to_string_pretty(&serde_json::json!({
            "status": status,
            "epoch": epoch,
            "best_validation_error_rate": best_validation_error_rate,
            "architecture": ARCHITECTURE,
            "task": config.task,
            "checkpoint": checkpoint,
            "train_loss": train_loss,
            "latest_checkpoint": format!("{LATEST_MODEL_STEM}.bin"),
            "epoch_checkpoint_pattern": "model-epoch-N.bin",
            "best_model": "model.bin",
            "checkpoint_note": "model-latest.bin is written during epochs and at initialization; model-epoch-N.bin and model.bin are written after validation at epoch end."
        }))?,
    )
}

fn dataset_readme(config: &CommonPhoneConfig, unknowns: &BTreeMap<String, usize>) -> String {
    format!(
        "# Common Phone v0 dataset\n\nInput: `{}`\n\nThis dataset uses mechanical compact acoustic frames, not learned EnCodec-style audio tokens. Each `features/*.acf.bin` file stores a little-endian header `(frames: u32, bins: u32)` followed by `{}` `f32` values per frame: pseudo log-Mel bins, deltas, energy, VAD, zero-crossing rate, spectral centroid, spectral flux, estimated f0, and voiced probability.\n\nThe first supported local layout is an input directory containing `metadata.jsonl`, `metadata.csv`, or `metadata.tsv`. Required columns/fields are `audio_path` (or `path`/`wav`) and `phones`; optional fields include `utterance_id`, `lang`, `variety`, `speaker_id`, and `phonemes`. Audio is currently decoded from WAV files.\n\nTargets are trained as ordered CTC sequences: phones, phonemes, and feature axes (`manner`, `place`, `voicing`, `syllabic`, `height`, `backness`, `rounding`). Unknown phone-feature mappings are counted instead of rejected.\n\nSample rate: {}\nFrame rate: {}\nUnknown phone symbols: {:?}\n",
        config.input, config.feature_bins, config.sample_rate_hz, config.frame_hz, unknowns
    )
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct CommonPhoneStats {
    examples_total: usize,
    examples_train: usize,
    examples_valid: usize,
    examples_test: usize,
    languages: BTreeMap<String, usize>,
    sample_rate: u32,
    hop_ms: f32,
    window_ms: f32,
    frame_dim: usize,
    phone_vocab_size: usize,
    feature_bundle_vocab_size: usize,
    unknown_feature_phone_count: usize,
    skipped_examples: usize,
}

fn dataset_stats(
    config: &CommonPhoneConfig,
    train: &[CommonPhoneRow],
    valid: &[CommonPhoneRow],
    test: &[CommonPhoneRow],
    phone_vocab_size: usize,
    feature_bundle_vocab_size: usize,
    unknowns: &BTreeMap<String, usize>,
    skipped_examples: usize,
) -> CommonPhoneStats {
    let mut languages = BTreeMap::new();
    for row in train.iter().chain(valid).chain(test) {
        *languages.entry(row.lang.clone()).or_insert(0) += 1;
    }
    CommonPhoneStats {
        examples_total: train.len() + valid.len() + test.len(),
        examples_train: train.len(),
        examples_valid: valid.len(),
        examples_test: test.len(),
        languages,
        sample_rate: config.sample_rate_hz,
        hop_ms: config.hop_ms,
        window_ms: config.window_ms,
        frame_dim: config.feature_bins,
        phone_vocab_size,
        feature_bundle_vocab_size,
        unknown_feature_phone_count: unknowns.values().sum(),
        skipped_examples,
    }
}

fn frame_summary(index: usize, frame: &[f32]) -> FrameSummary {
    let scalar_start = DEFAULT_MEL_BINS * 2;
    let f0 = frame.get(scalar_start + 5).copied().unwrap_or(0.0) * 500.0;
    FrameSummary {
        frame: index,
        energy: frame.get(scalar_start).copied().unwrap_or(0.0),
        zcr: frame.get(scalar_start + 2).copied().unwrap_or(0.0),
        centroid: frame.get(scalar_start + 3).copied().unwrap_or(0.0),
        f0: (f0 > 0.0).then_some(f0),
        voiced: frame.get(scalar_start + 6).copied().unwrap_or(0.0),
        mel_head: frame.iter().copied().take(5).collect(),
    }
}

fn frame_stats_from_features(
    samples: &[f32],
    features: &[Vec<f32>],
    frame_dim: usize,
) -> LiveFrameStats {
    let rms = if samples.is_empty() {
        0.0
    } else {
        (samples.iter().map(|value| value * value).sum::<f32>() / samples.len() as f32).sqrt()
    };
    let scalar_start = DEFAULT_MEL_BINS * 2;
    let vad = if features.is_empty() {
        0.0
    } else {
        features
            .iter()
            .map(|frame| frame.get(scalar_start + 1).copied().unwrap_or(0.0))
            .sum::<f32>()
            / features.len() as f32
    };
    LiveFrameStats {
        rms,
        vad,
        frames: features.len(),
        frame_dim,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tongues-common-phone-{name}-{}",
            std::process::id()
        ))
    }

    fn write_test_wav(path: &Path) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: DEFAULT_SAMPLE_RATE_HZ,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).expect("create wav");
        for i in 0..DEFAULT_SAMPLE_RATE_HZ {
            let phase = i as f32 / DEFAULT_SAMPLE_RATE_HZ as f32;
            let sample = (phase * 440.0 * std::f32::consts::TAU).sin() * 0.2;
            writer
                .write_sample((sample * i16::MAX as f32) as i16)
                .expect("write wav sample");
        }
        writer.finalize().expect("finalize wav");
    }

    #[test]
    fn textgrid_tokens_are_tier_specific() {
        let path = temp_path("tier-specific.TextGrid");
        fs::write(
            &path,
            r#"File type = "ooTextFile"
Object class = "TextGrid"

item []:
    item [1]:
        class = "IntervalTier"
        name = "ORT-MAU"
        intervals: size = 1
            intervals [1]:
                text = "Line"
    item [2]:
        class = "IntervalTier"
        name = "KAN-MAU"
        intervals: size = 1
            intervals [1]:
                text = "l aɪ n"
    item [3]:
        class = "IntervalTier"
        name = "MAU"
        intervals: size = 3
            intervals [1]:
                text = "(...)"
            intervals [2]:
                text = "l"
            intervals [3]:
                text = "aɪ"
"#,
        )
        .expect("write textgrid");

        let phones = textgrid_tier_tokens(&path, "MAU").expect("phone tier");
        let phonemes = textgrid_tier_tokens(&path, "KAN-MAU").expect("phoneme tier");
        fs::remove_file(path).ok();

        assert_eq!(phones, vec!["l", "aɪ"]);
        assert_eq!(phonemes, vec!["l", "aɪ", "n"]);
    }

    #[test]
    fn textgrid_tokens_read_short_format_tiers() {
        let path = temp_path("short-format.TextGrid");
        fs::write(
            &path,
            r#""ooTextFile"
"TextGrid"

0
0.3
<exists>
2
"IntervalTier"
"KAN-MAU"
0
0.3
1
0
0.3
"l aɪ n"
"IntervalTier"
"MAU"
0
0.3
3
0
0.1
"l"
0.1
0.2
"aɪ"
0.2
0.3
"n"
"#,
        )
        .expect("write textgrid");

        let phones = textgrid_tier_tokens(&path, "MAU").expect("phone tier");
        let phonemes = textgrid_tier_tokens(&path, "KAN-MAU").expect("phoneme tier");
        fs::remove_file(path).ok();

        assert_eq!(phones, vec!["l", "aɪ", "n"]);
        assert_eq!(phonemes, vec!["l", "aɪ", "n"]);
    }

    #[test]
    fn textgrid_tokens_read_point_tier_marks() {
        let path = temp_path("point-tier.TextGrid");
        fs::write(
            &path,
            r#"File type = "ooTextFile"
Object class = "TextGrid"

item []:
    item [1]:
        class = "TextTier"
        name = "MAU"
        points: size = 2
            points [1]:
                number = 0.1
                mark = "p"
            points [2]:
                number = 0.2
                mark = "tʃ"
"#,
        )
        .expect("write textgrid");

        let phones = textgrid_tier_tokens(&path, "MAU").expect("phone tier");
        fs::remove_file(path).ok();

        assert_eq!(phones, vec!["p", "tʃ"]);
    }

    #[test]
    fn vocab_does_not_duplicate_reserved_tokens() {
        let tokens = vec![
            UNK.to_string(),
            CTC_BLANK.to_string(),
            "p".to_string(),
            UNK.to_string(),
        ];
        let vocab = build_token_vocab(tokens.iter());
        assert_eq!(vocab.tokens[0], CTC_BLANK);
        assert_eq!(vocab.tokens[1], UNK);
        assert_eq!(vocab.token_to_id[CTC_BLANK], 0);
        assert_eq!(vocab.token_to_id[UNK], 1);
        assert_eq!(vocab.tokens.iter().filter(|token| *token == UNK).count(), 1);
    }

    #[test]
    fn prepare_reuses_existing_frames_without_manifest() {
        let root = temp_path("reuse-frames");
        let input = root.join("input");
        let out = root.join("out");
        fs::remove_dir_all(&root).ok();
        fs::create_dir_all(&input).expect("create input");
        write_test_wav(&input.join("one.wav"));

        let metadata = input.join("metadata.jsonl");
        fs::write(
            &metadata,
            r#"{"utterance_id":"one","lang":"eng","audio_path":"one.wav","phones":["p"],"phonemes":["p"],"text":"one"}"#,
        )
        .expect("write metadata");
        let config = CommonPhoneConfig {
            input: input.display().to_string(),
            max_utterances: Some(1),
            ..CommonPhoneConfig::default()
        };
        prepare_dataset(&out, &config).expect("initial prepare");
        let frame_path = out.join("frames/one.acf.bin");
        let first_modified = fs::metadata(&frame_path)
            .expect("first frame metadata")
            .modified()
            .expect("first modified");

        for name in [
            "manifest.jsonl",
            "train.jsonl",
            "valid.jsonl",
            "test.jsonl",
            "phone_vocab.json",
            "phoneme_vocab.json",
            "feature_bundle_vocab.json",
            "feature_axis_vocabs.json",
            "stats.json",
            "prepare_state.json",
        ] {
            fs::remove_file(out.join(name)).ok();
        }
        fs::write(
            &metadata,
            r#"{"utterance_id":"one","lang":"eng","audio_path":"one.wav","phones":["b"],"phonemes":["b"],"text":"one"}"#,
        )
        .expect("rewrite metadata");
        let mut reused = 0usize;
        prepare_dataset_with_progress(&out, &config, |progress| {
            if matches!(progress, PrepareProgress::Reuse { .. }) {
                reused += 1;
            }
        })
        .expect("second prepare");

        let second_modified = fs::metadata(&frame_path)
            .expect("second frame metadata")
            .modified()
            .expect("second modified");
        let rows = read_examples(&out.join("train.jsonl")).expect("read regenerated train");
        fs::remove_dir_all(root).ok();

        assert_eq!(reused, 1);
        assert_eq!(first_modified, second_modified);
        assert_eq!(rows[0].phones, vec!["b"]);
    }
}
