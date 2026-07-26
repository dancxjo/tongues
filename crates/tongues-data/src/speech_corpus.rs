//! Model-neutral native speech-corpus ingestion, validation, splitting, and batching.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub const SPEECH_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const SPEECH_FEATURE_CACHE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechCorpusFormat {
    Ljspeech,
    Vctk,
    Generic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocation {
    pub path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechRecord {
    pub id: String,
    pub audio_path: PathBuf,
    pub text: String,
    pub normalized_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    pub language: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_samples: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    pub source: SourceLocation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationSeverity {
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub source: SourceLocation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record_id: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorpusStatistics {
    pub records: usize,
    pub speakers: usize,
    pub languages: usize,
    pub total_duration_seconds: f64,
    pub min_duration_seconds: Option<f64>,
    pub max_duration_seconds: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IngestedSpeechCorpus {
    pub schema_version: u32,
    pub format: SpeechCorpusFormat,
    pub root: PathBuf,
    pub records: Vec<SpeechRecord>,
    pub issues: Vec<ValidationIssue>,
    pub statistics: CorpusStatistics,
}

impl IngestedSpeechCorpus {
    pub fn has_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|issue| issue.severity == ValidationSeverity::Error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitUnit {
    Utterance,
    Speaker,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechSplitConfig {
    pub train_fraction: f64,
    pub valid_fraction: f64,
    pub seed: u64,
    pub unit: SplitUnit,
}

impl Default for SpeechSplitConfig {
    fn default() -> Self {
        Self {
            train_fraction: 0.8,
            valid_fraction: 0.1,
            seed: 42,
            unit: SplitUnit::Utterance,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechCorpusSplits {
    pub train: Vec<SpeechRecord>,
    pub valid: Vec<SpeechRecord>,
    pub test: Vec<SpeechRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechBatchConfig {
    pub max_items: usize,
    /// Zero disables the aggregate audio-sample limit.
    pub max_audio_samples: u64,
    /// Nearby lengths share a bucket before seeded ordering is applied.
    pub bucket_width_samples: u64,
    pub seed: u64,
}

impl Default for SpeechBatchConfig {
    fn default() -> Self {
        Self {
            max_items: 16,
            max_audio_samples: 0,
            bucket_width_samples: 22_050,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechBatchPlan {
    pub record_ids: Vec<String>,
    pub total_audio_samples: u64,
    pub max_audio_samples: u64,
    pub max_text_chars: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechTrainingExample {
    pub record_id: String,
    pub token_ids: Vec<u32>,
    /// Frame-major acoustic features: `[frames][bins]`.
    pub acoustic_features: Vec<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_id: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollatedSpeechBatch {
    pub record_ids: Vec<String>,
    pub token_ids: Vec<Vec<u32>>,
    pub token_padding_mask: Vec<Vec<bool>>,
    /// Flattened row-major `[batch, max_frames, acoustic_bins]`.
    pub acoustic_features: Vec<f32>,
    pub acoustic_padding_mask: Vec<Vec<bool>>,
    pub speaker_ids: Vec<Option<u32>>,
    pub batch_size: usize,
    pub max_tokens: usize,
    pub max_frames: usize,
    pub acoustic_bins: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CachedSpeechFeatures {
    pub schema_version: u32,
    pub record_id: String,
    pub config_fingerprint: String,
    pub text_tokens: Vec<u32>,
    pub phoneme_tokens: Vec<u32>,
    pub acoustic_features: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrepareSpeechCorpusConfig {
    pub format: SpeechCorpusFormat,
    pub metadata_path: Option<PathBuf>,
    pub language: String,
    pub split: SpeechSplitConfig,
    pub batch: SpeechBatchConfig,
}

impl PrepareSpeechCorpusConfig {
    pub fn for_format(format: SpeechCorpusFormat) -> Self {
        Self {
            format,
            metadata_path: None,
            language: "en-US".to_string(),
            split: SpeechSplitConfig::default(),
            batch: SpeechBatchConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrepareSpeechProgress {
    Scan {
        format: SpeechCorpusFormat,
        root: PathBuf,
    },
    Validate {
        checked: usize,
        total: usize,
    },
    Split {
        train: usize,
        valid: usize,
        test: usize,
    },
    Batch {
        split: &'static str,
        batches: usize,
    },
    Write {
        rows: usize,
        path: PathBuf,
    },
    Complete {
        output: PathBuf,
        records: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrepareSpeechCorpusReport {
    pub output: PathBuf,
    pub records: usize,
    pub train: usize,
    pub valid: usize,
    pub test: usize,
    pub train_batches: usize,
    pub valid_batches: usize,
    pub test_batches: usize,
    pub statistics: CorpusStatistics,
}

pub fn ingest_speech_corpus(
    root: &Path,
    format: SpeechCorpusFormat,
    metadata_path: Option<&Path>,
    language: &str,
) -> Result<IngestedSpeechCorpus> {
    ingest_speech_corpus_with_normalizer(root, format, metadata_path, language, |text| {
        text.trim().to_string()
    })
}

pub fn ingest_speech_corpus_with_normalizer<F>(
    root: &Path,
    format: SpeechCorpusFormat,
    metadata_path: Option<&Path>,
    language: &str,
    normalize: F,
) -> Result<IngestedSpeechCorpus>
where
    F: Fn(&str) -> String,
{
    anyhow::ensure!(
        root.is_dir(),
        "speech corpus root is not a directory: {}",
        root.display()
    );
    let (mut records, mut issues) = match format {
        SpeechCorpusFormat::Ljspeech => ingest_ljspeech(root, metadata_path, language, &normalize)?,
        SpeechCorpusFormat::Vctk => ingest_vctk(root, language, &normalize)?,
        SpeechCorpusFormat::Generic => ingest_generic(root, metadata_path, language, &normalize)?,
    };
    validate_and_probe(root, &mut records, &mut issues);
    let statistics = corpus_statistics(&records);
    Ok(IngestedSpeechCorpus {
        schema_version: SPEECH_MANIFEST_SCHEMA_VERSION,
        format,
        root: root.to_path_buf(),
        records,
        issues,
        statistics,
    })
}

pub fn split_speech_corpus(
    records: &[SpeechRecord],
    config: &SpeechSplitConfig,
) -> Result<SpeechCorpusSplits> {
    anyhow::ensure!(
        config.train_fraction.is_finite()
            && config.valid_fraction.is_finite()
            && config.train_fraction >= 0.0
            && config.valid_fraction >= 0.0
            && config.train_fraction + config.valid_fraction <= 1.0,
        "split fractions must be finite, non-negative, and sum to at most one"
    );

    let mut train = Vec::new();
    let mut valid = Vec::new();
    let mut test = Vec::new();
    for record in records {
        let key = match config.unit {
            SplitUnit::Utterance => record.id.as_str(),
            SplitUnit::Speaker => record.speaker.as_deref().unwrap_or(record.id.as_str()),
        };
        let score = stable_unit_interval(config.seed, key);
        if score < config.train_fraction {
            train.push(record.clone());
        } else if score < config.train_fraction + config.valid_fraction {
            valid.push(record.clone());
        } else {
            test.push(record.clone());
        }
    }
    for split in [&mut train, &mut valid, &mut test] {
        split.sort_by(|left, right| left.id.cmp(&right.id));
    }
    Ok(SpeechCorpusSplits { train, valid, test })
}

pub fn plan_speech_batches(
    records: &[SpeechRecord],
    config: &SpeechBatchConfig,
) -> Result<Vec<SpeechBatchPlan>> {
    anyhow::ensure!(config.max_items > 0, "batch max_items must be positive");
    let bucket_width = config.bucket_width_samples.max(1);
    let mut ordered = records.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|record| {
        let length = estimated_audio_samples(record);
        (
            length / bucket_width,
            stable_hash(config.seed, record.id.as_bytes()),
            record.id.as_str(),
        )
    });

    let mut batches = Vec::new();
    let mut current = Vec::new();
    let mut total_samples = 0u64;
    for record in ordered {
        let samples = estimated_audio_samples(record);
        let would_exceed_samples = config.max_audio_samples > 0
            && !current.is_empty()
            && total_samples.saturating_add(samples) > config.max_audio_samples;
        if current.len() == config.max_items || would_exceed_samples {
            batches.push(batch_plan(&current));
            current.clear();
            total_samples = 0;
        }
        total_samples = total_samples.saturating_add(samples);
        current.push(record);
    }
    if !current.is_empty() {
        batches.push(batch_plan(&current));
    }
    Ok(batches)
}

pub fn collate_speech_batch(
    examples: &[SpeechTrainingExample],
    token_padding_id: u32,
) -> Result<CollatedSpeechBatch> {
    anyhow::ensure!(!examples.is_empty(), "cannot collate an empty speech batch");
    let acoustic_bins = examples
        .iter()
        .find_map(|example| example.acoustic_features.first().map(Vec::len))
        .unwrap_or(0);
    anyhow::ensure!(
        acoustic_bins > 0,
        "speech batch has no acoustic feature bins"
    );
    for example in examples {
        anyhow::ensure!(
            example
                .acoustic_features
                .iter()
                .all(|frame| frame.len() == acoustic_bins),
            "record {} has inconsistent acoustic feature width",
            example.record_id
        );
    }

    let batch_size = examples.len();
    let max_tokens = examples
        .iter()
        .map(|example| example.token_ids.len())
        .max()
        .unwrap_or(0);
    let max_frames = examples
        .iter()
        .map(|example| example.acoustic_features.len())
        .max()
        .unwrap_or(0);
    let mut token_ids = vec![vec![token_padding_id; max_tokens]; batch_size];
    let mut token_padding_mask = vec![vec![true; max_tokens]; batch_size];
    let mut acoustic_features = vec![0.0; batch_size * max_frames * acoustic_bins];
    let mut acoustic_padding_mask = vec![vec![true; max_frames]; batch_size];

    for (batch_index, example) in examples.iter().enumerate() {
        for (token_index, token) in example.token_ids.iter().copied().enumerate() {
            token_ids[batch_index][token_index] = token;
            token_padding_mask[batch_index][token_index] = false;
        }
        for (frame_index, frame) in example.acoustic_features.iter().enumerate() {
            acoustic_padding_mask[batch_index][frame_index] = false;
            let start = (batch_index * max_frames + frame_index) * acoustic_bins;
            acoustic_features[start..start + acoustic_bins].copy_from_slice(frame);
        }
    }

    Ok(CollatedSpeechBatch {
        record_ids: examples
            .iter()
            .map(|example| example.record_id.clone())
            .collect(),
        token_ids,
        token_padding_mask,
        acoustic_features,
        acoustic_padding_mask,
        speaker_ids: examples.iter().map(|example| example.speaker_id).collect(),
        batch_size,
        max_tokens,
        max_frames,
        acoustic_bins,
    })
}

pub fn feature_cache_path(cache_dir: &Path, record_id: &str) -> PathBuf {
    cache_dir.join(format!(
        "{:016x}.json",
        stable_hash(0, record_id.as_bytes())
    ))
}

pub fn write_cached_speech_features(
    cache_dir: &Path,
    features: &CachedSpeechFeatures,
) -> Result<PathBuf> {
    anyhow::ensure!(
        features.schema_version == SPEECH_FEATURE_CACHE_SCHEMA_VERSION,
        "unsupported speech feature cache schema {}",
        features.schema_version
    );
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("creating feature cache {}", cache_dir.display()))?;
    let path = feature_cache_path(cache_dir, &features.record_id);
    write_json_atomic(&path, features)?;
    Ok(path)
}

pub fn read_cached_speech_features(
    cache_dir: &Path,
    record_id: &str,
    config_fingerprint: &str,
) -> Result<Option<CachedSpeechFeatures>> {
    let path = feature_cache_path(cache_dir, record_id);
    if !path.exists() {
        return Ok(None);
    }
    let source =
        fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let features: CachedSpeechFeatures =
        serde_json::from_str(&source).with_context(|| format!("parsing {}", path.display()))?;
    anyhow::ensure!(
        features.schema_version == SPEECH_FEATURE_CACHE_SCHEMA_VERSION,
        "unsupported feature cache schema in {}",
        path.display()
    );
    anyhow::ensure!(
        features.record_id == record_id,
        "feature cache {} belongs to record {}, expected {}",
        path.display(),
        features.record_id,
        record_id
    );
    if features.config_fingerprint != config_fingerprint {
        return Ok(None);
    }
    Ok(Some(features))
}

pub fn prepare_speech_corpus(
    input: &Path,
    output: &Path,
    config: &PrepareSpeechCorpusConfig,
) -> Result<PrepareSpeechCorpusReport> {
    prepare_speech_corpus_with_progress(input, output, config, |_| {})
}

pub fn prepare_speech_corpus_with_progress(
    input: &Path,
    output: &Path,
    config: &PrepareSpeechCorpusConfig,
    mut progress: impl FnMut(PrepareSpeechProgress),
) -> Result<PrepareSpeechCorpusReport> {
    progress(PrepareSpeechProgress::Scan {
        format: config.format,
        root: input.to_path_buf(),
    });
    fs::create_dir_all(output).with_context(|| format!("creating {}", output.display()))?;
    let corpus = ingest_speech_corpus(
        input,
        config.format,
        config.metadata_path.as_deref(),
        &config.language,
    )?;
    for checked in 1..=corpus.records.len() {
        if checked <= 3 || checked % 1_000 == 0 || checked == corpus.records.len() {
            progress(PrepareSpeechProgress::Validate {
                checked,
                total: corpus.records.len(),
            });
        }
    }

    let validation_path = output.join("validation.json");
    write_json_atomic(&validation_path, &corpus.issues)?;
    progress(PrepareSpeechProgress::Write {
        rows: corpus.issues.len(),
        path: validation_path,
    });
    if corpus.has_errors() {
        bail!(
            "speech corpus validation found {} error(s); inspect {}",
            corpus
                .issues
                .iter()
                .filter(|issue| issue.severity == ValidationSeverity::Error)
                .count(),
            output.join("validation.json").display()
        );
    }

    let splits = split_speech_corpus(&corpus.records, &config.split)?;
    progress(PrepareSpeechProgress::Split {
        train: splits.train.len(),
        valid: splits.valid.len(),
        test: splits.test.len(),
    });
    let train_batches = plan_speech_batches(&splits.train, &config.batch)?;
    let valid_batches = plan_speech_batches(&splits.valid, &config.batch)?;
    let test_batches = plan_speech_batches(&splits.test, &config.batch)?;
    for (split, batches) in [
        ("train", &train_batches),
        ("valid", &valid_batches),
        ("test", &test_batches),
    ] {
        progress(PrepareSpeechProgress::Batch {
            split,
            batches: batches.len(),
        });
    }

    for (name, rows) in [
        ("manifest.jsonl", corpus.records.as_slice()),
        ("train.jsonl", splits.train.as_slice()),
        ("valid.jsonl", splits.valid.as_slice()),
        ("test.jsonl", splits.test.as_slice()),
    ] {
        let path = output.join(name);
        write_jsonl_atomic(&path, rows)?;
        progress(PrepareSpeechProgress::Write {
            rows: rows.len(),
            path,
        });
    }
    let batches_path = output.join("batches.json");
    write_json_atomic(
        &batches_path,
        &BTreeMap::from([
            ("train", &train_batches),
            ("valid", &valid_batches),
            ("test", &test_batches),
        ]),
    )?;
    progress(PrepareSpeechProgress::Write {
        rows: train_batches.len() + valid_batches.len() + test_batches.len(),
        path: batches_path,
    });
    write_json_atomic(&output.join("dataset_config.json"), config)?;
    write_json_atomic(&output.join("statistics.json"), &corpus.statistics)?;
    write_text_atomic(
        &output.join("README.md"),
        &format!(
            "# Native speech corpus\n\nFormat: `{:?}`\n\nRecords: {}\nTrain/valid/test: {}/{}/{}\n\nAll JSONL outputs are normalized, model-neutral records. `batches.json` contains deterministic length-aware record-id batches.\n",
            config.format,
            corpus.records.len(),
            splits.train.len(),
            splits.valid.len(),
            splits.test.len()
        ),
    )?;
    progress(PrepareSpeechProgress::Complete {
        output: output.to_path_buf(),
        records: corpus.records.len(),
    });

    Ok(PrepareSpeechCorpusReport {
        output: output.to_path_buf(),
        records: corpus.records.len(),
        train: splits.train.len(),
        valid: splits.valid.len(),
        test: splits.test.len(),
        train_batches: train_batches.len(),
        valid_batches: valid_batches.len(),
        test_batches: test_batches.len(),
        statistics: corpus.statistics,
    })
}

fn ingest_ljspeech<F>(
    root: &Path,
    metadata_path: Option<&Path>,
    language: &str,
    normalize: &F,
) -> Result<(Vec<SpeechRecord>, Vec<ValidationIssue>)>
where
    F: Fn(&str) -> String,
{
    let metadata = resolve_metadata(root, metadata_path, "metadata.csv");
    let reader = BufReader::new(
        File::open(&metadata).with_context(|| format!("opening {}", metadata.display()))?,
    );
    let mut records = Vec::new();
    let mut issues = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = line.with_context(|| format!("reading {}:{line_number}", metadata.display()))?;
        let location = source_location(root, &metadata, Some(line_number));
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.splitn(3, '|').collect::<Vec<_>>();
        if fields.len() != 3 {
            issues.push(error_issue(
                location,
                None,
                "expected LJSpeech `id|raw text|normalized text`",
            ));
            continue;
        }
        let id = fields[0].trim();
        let raw_text = fields[1].trim();
        let published_normalized = fields[2].trim();
        if id.is_empty() || raw_text.is_empty() || published_normalized.is_empty() {
            issues.push(error_issue(
                location,
                Some(id),
                "LJSpeech id, raw text, and normalized text must be non-empty",
            ));
            continue;
        }
        records.push(SpeechRecord {
            id: id.to_string(),
            audio_path: PathBuf::from("wavs").join(format!("{id}.wav")),
            text: raw_text.to_string(),
            normalized_text: normalize(published_normalized),
            speaker: Some("ljspeech".to_string()),
            language: language.to_string(),
            emotion: None,
            style: None,
            sample_rate_hz: None,
            audio_samples: None,
            duration_seconds: None,
            source: location,
        });
    }
    Ok((records, issues))
}

fn ingest_vctk<F>(
    root: &Path,
    language: &str,
    normalize: &F,
) -> Result<(Vec<SpeechRecord>, Vec<ValidationIssue>)>
where
    F: Fn(&str) -> String,
{
    let text_root = find_named_directory(root, "txt").with_context(|| {
        format!(
            "VCTK text directory `txt` not found under {}",
            root.display()
        )
    })?;
    let text_files = walk_files(&text_root, &["txt"])?;
    let audio_files = walk_files(root, &["wav", "flac"])?;
    let mut audio_by_id = HashMap::<String, (u8, PathBuf)>::new();
    for path in audio_files {
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let (id, rank) = if let Some(id) = stem.strip_suffix("_mic1") {
            (id, 1)
        } else if let Some(id) = stem.strip_suffix("_mic2") {
            (id, 2)
        } else {
            (stem, 0)
        };
        let relative = relative_or_owned(root, &path);
        match audio_by_id.get(id) {
            Some((existing_rank, _)) if *existing_rank <= rank => {}
            _ => {
                audio_by_id.insert(id.to_string(), (rank, relative));
            }
        }
    }

    let mut records = Vec::new();
    let mut issues = Vec::new();
    for text_path in text_files {
        let Some(id) = text_path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let location = source_location(root, &text_path, Some(1));
        let text = fs::read_to_string(&text_path)
            .with_context(|| format!("reading VCTK transcript {}", text_path.display()))?;
        let text = text.trim();
        let Some((_, audio_path)) = audio_by_id.get(id) else {
            issues.push(error_issue(
                location,
                Some(id),
                "matching VCTK WAV/FLAC audio was not found",
            ));
            continue;
        };
        records.push(SpeechRecord {
            id: id.to_string(),
            audio_path: audio_path.clone(),
            text: text.to_string(),
            normalized_text: normalize(text),
            speaker: id.split('_').next().map(str::to_string),
            language: language.to_string(),
            emotion: None,
            style: None,
            sample_rate_hz: None,
            audio_samples: None,
            duration_seconds: None,
            source: location,
        });
    }
    Ok((records, issues))
}

#[derive(Debug, Deserialize)]
struct GenericMetadataRow {
    id: String,
    #[serde(alias = "wav_path", alias = "audio")]
    audio_path: PathBuf,
    text: String,
    #[serde(default)]
    normalized_text: Option<String>,
    #[serde(default)]
    speaker: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    emotion: Option<String>,
    #[serde(default)]
    style: Option<String>,
}

fn ingest_generic<F>(
    root: &Path,
    metadata_path: Option<&Path>,
    language: &str,
    normalize: &F,
) -> Result<(Vec<SpeechRecord>, Vec<ValidationIssue>)>
where
    F: Fn(&str) -> String,
{
    let metadata = metadata_path
        .map(|path| resolve_metadata(root, Some(path), "metadata.jsonl"))
        .or_else(|| {
            ["metadata.jsonl", "metadata.csv"]
                .iter()
                .map(|name| root.join(name))
                .find(|path| path.exists())
        })
        .unwrap_or_else(|| root.join("metadata.jsonl"));
    let reader = BufReader::new(
        File::open(&metadata).with_context(|| format!("opening {}", metadata.display()))?,
    );
    let jsonl = metadata.extension().and_then(|ext| ext.to_str()) == Some("jsonl");
    let mut records = Vec::new();
    let mut issues = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = line.with_context(|| format!("reading {}:{line_number}", metadata.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let location = source_location(root, &metadata, Some(line_number));
        let parsed = if jsonl {
            serde_json::from_str::<GenericMetadataRow>(&line).map_err(anyhow::Error::from)
        } else {
            parse_generic_delimited(&line, language)
        };
        let row = match parsed {
            Ok(row) => row,
            Err(error) => {
                issues.push(error_issue(
                    location,
                    None,
                    format!("invalid generic metadata row: {error}"),
                ));
                continue;
            }
        };
        let normalized_source = row.normalized_text.as_deref().unwrap_or(&row.text);
        let normalized_text = normalize(normalized_source);
        records.push(SpeechRecord {
            id: row.id,
            audio_path: row.audio_path,
            text: row.text,
            normalized_text,
            speaker: row.speaker,
            language: row.language.unwrap_or_else(|| language.to_string()),
            emotion: row.emotion,
            style: row.style,
            sample_rate_hz: None,
            audio_samples: None,
            duration_seconds: None,
            source: location,
        });
    }
    Ok((records, issues))
}

fn parse_generic_delimited(line: &str, default_language: &str) -> Result<GenericMetadataRow> {
    let delimiter = if line.contains('|') { '|' } else { '\t' };
    let fields = line.split(delimiter).map(str::trim).collect::<Vec<_>>();
    anyhow::ensure!(
        fields.len() >= 2,
        "expected `audio|text` or `id|audio|text|speaker|language`"
    );
    let (id, audio, text, offset) = if fields.len() == 2 {
        let audio = fields[0];
        let id = Path::new(audio)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .context("audio path has no UTF-8 file stem")?;
        (id, audio, fields[1], 2)
    } else {
        (fields[0], fields[1], fields[2], 3)
    };
    Ok(GenericMetadataRow {
        id: id.to_string(),
        audio_path: PathBuf::from(audio),
        text: text.to_string(),
        normalized_text: None,
        speaker: fields
            .get(offset)
            .filter(|value| !value.is_empty())
            .map(|value| value.to_string()),
        language: fields
            .get(offset + 1)
            .filter(|value| !value.is_empty())
            .map(|value| value.to_string())
            .or_else(|| Some(default_language.to_string())),
        emotion: fields
            .get(offset + 2)
            .filter(|value| !value.is_empty())
            .map(|value| value.to_string()),
        style: fields
            .get(offset + 3)
            .filter(|value| !value.is_empty())
            .map(|value| value.to_string()),
    })
}

fn validate_and_probe(
    root: &Path,
    records: &mut [SpeechRecord],
    issues: &mut Vec<ValidationIssue>,
) {
    let mut ids = HashMap::<String, SourceLocation>::new();
    let mut audio_paths = HashMap::<PathBuf, SourceLocation>::new();
    let mut texts = HashMap::<(Option<String>, String), SourceLocation>::new();
    for record in records {
        if record.id.trim().is_empty() {
            issues.push(error_issue(
                record.source.clone(),
                None,
                "record id is empty",
            ));
        }
        if record.text.trim().is_empty() || record.normalized_text.trim().is_empty() {
            issues.push(error_issue(
                record.source.clone(),
                Some(&record.id),
                "text and normalized text must be non-empty",
            ));
        }
        if record.language.trim().is_empty() {
            issues.push(error_issue(
                record.source.clone(),
                Some(&record.id),
                "language must be non-empty",
            ));
        }
        if let Some(previous) = ids.insert(record.id.clone(), record.source.clone()) {
            issues.push(error_issue(
                record.source.clone(),
                Some(&record.id),
                format!(
                    "duplicate record id; first declared at {}",
                    display_source(&previous)
                ),
            ));
        }

        let absolute_audio = if record.audio_path.is_absolute() {
            record.audio_path.clone()
        } else {
            root.join(&record.audio_path)
        };
        if let Some(previous) = audio_paths.insert(record.audio_path.clone(), record.source.clone())
        {
            issues.push(error_issue(
                record.source.clone(),
                Some(&record.id),
                format!(
                    "duplicate audio path; first declared at {}",
                    display_source(&previous)
                ),
            ));
        }
        let text_key = (record.speaker.clone(), record.normalized_text.clone());
        if let Some(previous) = texts.insert(text_key, record.source.clone()) {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                source: record.source.clone(),
                record_id: Some(record.id.clone()),
                reason: format!(
                    "duplicate normalized text for this speaker; first declared at {}",
                    display_source(&previous)
                ),
            });
        }
        match probe_audio(&absolute_audio) {
            Ok((sample_rate, samples)) => {
                record.sample_rate_hz = Some(sample_rate);
                record.audio_samples = Some(samples);
                record.duration_seconds = Some(samples as f64 / sample_rate as f64);
            }
            Err(error) => issues.push(error_issue(
                record.source.clone(),
                Some(&record.id),
                format!(
                    "invalid or missing audio {}: {error}",
                    record.audio_path.display()
                ),
            )),
        }
    }
}

fn probe_audio(path: &Path) -> Result<(u32, u64)> {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("wav") => {
            let reader = hound::WavReader::open(path)
                .with_context(|| format!("opening {}", path.display()))?;
            let spec = reader.spec();
            anyhow::ensure!(spec.sample_rate > 0, "WAV sample rate is zero");
            let samples = reader.duration() as u64;
            Ok((spec.sample_rate, samples))
        }
        Some("flac") => probe_flac_streaminfo(path),
        Some(extension) => bail!("unsupported audio extension `{extension}`"),
        None => bail!("audio path has no extension"),
    }
}

fn probe_flac_streaminfo(path: &Path) -> Result<(u32, u64)> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut marker = [0u8; 4];
    file.read_exact(&mut marker)?;
    anyhow::ensure!(&marker == b"fLaC", "missing FLAC marker");
    loop {
        let mut header = [0u8; 4];
        file.read_exact(&mut header)?;
        let last = header[0] & 0x80 != 0;
        let block_type = header[0] & 0x7f;
        let length =
            (u32::from(header[1]) << 16) | (u32::from(header[2]) << 8) | u32::from(header[3]);
        if block_type == 0 {
            anyhow::ensure!(length >= 18, "FLAC STREAMINFO block is too short");
            let mut streaminfo = vec![0u8; length as usize];
            file.read_exact(&mut streaminfo)?;
            let packed = u64::from_be_bytes(streaminfo[10..18].try_into().unwrap());
            let sample_rate = ((packed >> 44) & 0x000f_ffff) as u32;
            let total_samples = packed & 0x0000_000f_ffff_ffff;
            anyhow::ensure!(sample_rate > 0, "FLAC sample rate is zero");
            return Ok((sample_rate, total_samples));
        }
        file.seek(SeekFrom::Current(i64::from(length)))?;
        anyhow::ensure!(!last, "FLAC has no STREAMINFO block");
    }
}

fn corpus_statistics(records: &[SpeechRecord]) -> CorpusStatistics {
    let speakers = records
        .iter()
        .filter_map(|record| record.speaker.as_deref())
        .collect::<BTreeSet<_>>()
        .len();
    let languages = records
        .iter()
        .map(|record| record.language.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let durations = records
        .iter()
        .filter_map(|record| record.duration_seconds)
        .collect::<Vec<_>>();
    CorpusStatistics {
        records: records.len(),
        speakers,
        languages,
        total_duration_seconds: durations.iter().sum(),
        min_duration_seconds: durations.iter().copied().reduce(f64::min),
        max_duration_seconds: durations.iter().copied().reduce(f64::max),
    }
}

fn batch_plan(records: &[&SpeechRecord]) -> SpeechBatchPlan {
    SpeechBatchPlan {
        record_ids: records.iter().map(|record| record.id.clone()).collect(),
        total_audio_samples: records
            .iter()
            .map(|record| estimated_audio_samples(record))
            .sum(),
        max_audio_samples: records
            .iter()
            .map(|record| estimated_audio_samples(record))
            .max()
            .unwrap_or(0),
        max_text_chars: records
            .iter()
            .map(|record| record.normalized_text.chars().count())
            .max()
            .unwrap_or(0),
    }
}

fn estimated_audio_samples(record: &SpeechRecord) -> u64 {
    record.audio_samples.unwrap_or_else(|| {
        let rate = u64::from(record.sample_rate_hz.unwrap_or(22_050));
        (record.normalized_text.chars().count() as u64)
            .saturating_mul(rate)
            .saturating_div(12)
            .max(1)
    })
}

fn stable_unit_interval(seed: u64, key: &str) -> f64 {
    stable_hash(seed, key.as_bytes()) as f64 / u64::MAX as f64
}

fn stable_hash(seed: u64, bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    for byte in seed.to_le_bytes().iter().chain(bytes) {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn resolve_metadata(root: &Path, metadata_path: Option<&Path>, default_name: &str) -> PathBuf {
    match metadata_path {
        Some(path) if path.is_absolute() => path.to_path_buf(),
        Some(path) => root.join(path),
        None => root.join(default_name),
    }
}

fn find_named_directory(root: &Path, name: &str) -> Option<PathBuf> {
    if root.file_name().and_then(|part| part.to_str()) == Some(name) {
        return Some(root.to_path_buf());
    }
    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        let mut children = fs::read_dir(&directory)
            .ok()?
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();
        children.sort();
        for child in children.into_iter().rev() {
            if child.file_name().and_then(|part| part.to_str()) == Some(name) {
                return Some(child);
            }
            directories.push(child);
        }
    }
    None
}

fn walk_files(root: &Path, extensions: &[&str]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        let entries = fs::read_dir(&directory)
            .with_context(|| format!("reading directory {}", directory.display()))?;
        for entry in entries {
            let path = entry?.path();
            if path.is_dir() {
                directories.push(path);
            } else if path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    extensions
                        .iter()
                        .any(|candidate| extension.eq_ignore_ascii_case(candidate))
                })
            {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn source_location(root: &Path, path: &Path, line: Option<usize>) -> SourceLocation {
    SourceLocation {
        path: relative_or_owned(root, path),
        line,
    }
}

fn relative_or_owned(root: &Path, path: &Path) -> PathBuf {
    path.strip_prefix(root)
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| path.to_path_buf())
}

fn error_issue(
    source: SourceLocation,
    record_id: Option<&str>,
    reason: impl Into<String>,
) -> ValidationIssue {
    ValidationIssue {
        severity: ValidationSeverity::Error,
        source,
        record_id: record_id.map(str::to_string),
        reason: reason.into(),
    }
}

fn display_source(source: &SourceLocation) -> String {
    match source.line {
        Some(line) => format!("{}:{line}", source.path.display()),
        None => source.path.display().to_string(),
    }
}

fn part_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_default();
    name.push(".part");
    path.with_file_name(name)
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let part = part_path(path);
    let file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&part, path)
        .with_context(|| format!("publishing {} to {}", part.display(), path.display()))
}

fn write_jsonl_atomic(path: &Path, rows: &[SpeechRecord]) -> Result<()> {
    let part = part_path(path);
    let file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut writer = BufWriter::new(file);
    for row in rows {
        serde_json::to_writer(&mut writer, row)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&part, path)
        .with_context(|| format!("publishing {} to {}", part.display(), path.display()))
}

fn write_text_atomic(path: &Path, text: &str) -> Result<()> {
    let part = part_path(path);
    let file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(text.as_bytes())?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&part, path)
        .with_context(|| format!("publishing {} to {}", part.display(), path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "tongues-speech-corpus-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_wav(path: &Path, samples: usize) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 22_050,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for _ in 0..samples {
            writer.write_sample::<i16>(0).unwrap();
        }
        writer.finalize().unwrap();
    }

    fn write_flac_streaminfo(path: &Path, sample_rate: u32, samples: u64) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut bytes = b"fLaC".to_vec();
        bytes.extend_from_slice(&[0x80, 0, 0, 34]);
        let mut streaminfo = [0u8; 34];
        let packed =
            (u64::from(sample_rate) << 44) | (15u64 << 36) | (samples & 0x0000_000f_ffff_ffff);
        streaminfo[10..18].copy_from_slice(&packed.to_be_bytes());
        bytes.extend_from_slice(&streaminfo);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn ingests_complete_ljspeech_rows_and_reports_source_locations() {
        let root = temp_dir("ljspeech");
        fs::write(
            root.join("metadata.csv"),
            "LJ001-0001|Raw one.|Normalized one.\nLJ001-0002|Raw two.|Normalized two.\n",
        )
        .unwrap();
        write_wav(&root.join("wavs/LJ001-0001.wav"), 22_050);
        write_wav(&root.join("wavs/LJ001-0002.wav"), 11_025);

        let corpus =
            ingest_speech_corpus(&root, SpeechCorpusFormat::Ljspeech, None, "en-US").unwrap();
        assert!(!corpus.has_errors(), "{:?}", corpus.issues);
        assert_eq!(corpus.records.len(), 2);
        assert_eq!(corpus.records[0].source.line, Some(1));
        assert_eq!(corpus.records[0].normalized_text, "Normalized one.");
        assert_eq!(corpus.records[0].duration_seconds, Some(1.0));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ingests_vctk_p330_and_prefers_mic1_audio() {
        let root = temp_dir("vctk");
        fs::create_dir_all(root.join("txt/p330")).unwrap();
        fs::write(root.join("txt/p330/p330_001.txt"), "Please call Stella.").unwrap();
        write_wav(
            &root.join("wav48_silence_trimmed/p330/p330_001_mic2.wav"),
            1_000,
        );
        write_flac_streaminfo(
            &root.join("wav48_silence_trimmed/p330/p330_001_mic1.flac"),
            48_000,
            2_000,
        );

        let corpus = ingest_speech_corpus(&root, SpeechCorpusFormat::Vctk, None, "en-GB").unwrap();
        assert!(!corpus.has_errors(), "{:?}", corpus.issues);
        assert_eq!(corpus.records.len(), 1);
        assert_eq!(corpus.records[0].id, "p330_001");
        assert_eq!(corpus.records[0].speaker.as_deref(), Some("p330"));
        assert!(corpus.records[0].audio_path.ends_with("p330_001_mic1.flac"));
        assert_eq!(corpus.records[0].sample_rate_hz, Some(48_000));
        assert_eq!(corpus.records[0].audio_samples, Some(2_000));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn invalid_rows_keep_source_and_reason() {
        let root = temp_dir("invalid");
        fs::write(root.join("metadata.csv"), "broken row\n").unwrap();
        let corpus =
            ingest_speech_corpus(&root, SpeechCorpusFormat::Ljspeech, None, "en-US").unwrap();
        assert!(corpus.has_errors());
        assert_eq!(corpus.issues[0].source.line, Some(1));
        assert!(corpus.issues[0].reason.contains("expected LJSpeech"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ingests_generic_jsonl_conditioning_fields() {
        let root = temp_dir("generic");
        fs::write(
            root.join("metadata.jsonl"),
            r#"{"id":"custom-1","audio_path":"audio/custom-1.wav","text":"Raw.","normalized_text":"Normalized.","speaker":"speaker-a","language":"es-MX","emotion":"happy","style":"conversational"}"#,
        )
        .unwrap();
        write_wav(&root.join("audio/custom-1.wav"), 1_000);
        let corpus =
            ingest_speech_corpus(&root, SpeechCorpusFormat::Generic, None, "en-US").unwrap();
        assert!(!corpus.has_errors(), "{:?}", corpus.issues);
        let record = &corpus.records[0];
        assert_eq!(record.normalized_text, "Normalized.");
        assert_eq!(record.speaker.as_deref(), Some("speaker-a"));
        assert_eq!(record.language, "es-MX");
        assert_eq!(record.emotion.as_deref(), Some("happy"));
        assert_eq!(record.style.as_deref(), Some("conversational"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn splits_and_length_aware_batches_are_reproducible() {
        let records = (0..40)
            .map(|index| SpeechRecord {
                id: format!("p330_{index:03}"),
                audio_path: PathBuf::from(format!("{index}.wav")),
                text: format!("row {index}"),
                normalized_text: format!("row {index}"),
                speaker: Some("p330".to_string()),
                language: "en-GB".to_string(),
                emotion: None,
                style: None,
                sample_rate_hz: Some(22_050),
                audio_samples: Some(1_000 + index * 100),
                duration_seconds: Some((1_000 + index * 100) as f64 / 22_050.0),
                source: SourceLocation {
                    path: PathBuf::from("metadata"),
                    line: Some(index as usize + 1),
                },
            })
            .collect::<Vec<_>>();
        let split_config = SpeechSplitConfig::default();
        let first = split_speech_corpus(&records, &split_config).unwrap();
        let second = split_speech_corpus(&records, &split_config).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.train.len() + first.valid.len() + first.test.len(),
            records.len()
        );

        let batch_config = SpeechBatchConfig {
            max_items: 4,
            max_audio_samples: 12_000,
            bucket_width_samples: 1_000,
            seed: 9,
        };
        let first_batches = plan_speech_batches(&records, &batch_config).unwrap();
        let second_batches = plan_speech_batches(&records, &batch_config).unwrap();
        assert_eq!(first_batches, second_batches);
        assert!(first_batches
            .iter()
            .all(|batch| batch.record_ids.len() <= 4));
    }

    #[test]
    fn collates_tokens_acoustics_masks_and_speakers() {
        let batch = collate_speech_batch(
            &[
                SpeechTrainingExample {
                    record_id: "a".to_string(),
                    token_ids: vec![1, 2],
                    acoustic_features: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
                    speaker_id: Some(7),
                },
                SpeechTrainingExample {
                    record_id: "b".to_string(),
                    token_ids: vec![3],
                    acoustic_features: vec![vec![0.5, 0.6]],
                    speaker_id: Some(8),
                },
            ],
            0,
        )
        .unwrap();
        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.token_ids, vec![vec![1, 2], vec![3, 0]]);
        assert_eq!(
            batch.token_padding_mask,
            vec![vec![false, false], vec![false, true]]
        );
        assert_eq!(
            batch.acoustic_padding_mask,
            vec![vec![false, false], vec![false, true]]
        );
        assert_eq!(batch.acoustic_features.len(), 8);
        assert_eq!(batch.speaker_ids, vec![Some(7), Some(8)]);
    }

    #[test]
    fn feature_cache_is_atomic_and_config_sensitive() {
        let root = temp_dir("cache");
        let features = CachedSpeechFeatures {
            schema_version: SPEECH_FEATURE_CACHE_SCHEMA_VERSION,
            record_id: "p330_001".to_string(),
            config_fingerprint: "mel-v1".to_string(),
            text_tokens: vec![1, 2],
            phoneme_tokens: vec![3, 4],
            acoustic_features: vec![vec![0.25, 0.5]],
        };
        let path = write_cached_speech_features(&root, &features).unwrap();
        assert!(path.exists());
        assert!(!part_path(&path).exists());
        assert_eq!(
            read_cached_speech_features(&root, "p330_001", "mel-v1").unwrap(),
            Some(features)
        );
        assert_eq!(
            read_cached_speech_features(&root, "p330_001", "mel-v2").unwrap(),
            None
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn prepare_writes_atomic_normalized_splits_and_batches() {
        let input = temp_dir("prepare-input");
        let output = temp_dir("prepare-output");
        fs::write(
            input.join("metadata.csv"),
            "LJ001-0001|Raw one.|Normalized one.\nLJ001-0002|Raw two.|Normalized two.\n",
        )
        .unwrap();
        write_wav(&input.join("wavs/LJ001-0001.wav"), 2_000);
        write_wav(&input.join("wavs/LJ001-0002.wav"), 3_000);
        let config = PrepareSpeechCorpusConfig::for_format(SpeechCorpusFormat::Ljspeech);
        let mut events = Vec::new();
        let report = prepare_speech_corpus_with_progress(&input, &output, &config, |event| {
            events.push(event)
        })
        .unwrap();
        assert_eq!(report.records, 2);
        for name in [
            "manifest.jsonl",
            "train.jsonl",
            "valid.jsonl",
            "test.jsonl",
            "batches.json",
            "dataset_config.json",
            "statistics.json",
            "validation.json",
            "README.md",
        ] {
            assert!(output.join(name).exists(), "{name}");
            assert!(!part_path(&output.join(name)).exists(), "{name}.part");
        }
        assert!(events
            .iter()
            .any(|event| matches!(event, PrepareSpeechProgress::Complete { .. })));
        fs::remove_dir_all(input).unwrap();
        fs::remove_dir_all(output).unwrap();
    }
}
