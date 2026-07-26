//! Sentence-boundary seq2seq model-family data preparation.
//!
//! This family trains a cursor-time model to decide whether the current text
//! prefix can be emitted as a complete sentence, should continue buffering, or
//! needs to repair a previously emitted boundary.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, thread};

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use seams::SentenceDetectorDialog;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use speaking::segment::TerminalPunctuation;
use speaking::syntax::{GrammarParser, SentenceSyntaxAnalysis, VarietyGrammarParser};
use tongues_core::{Vocab, BOS_ID, EOS_ID};
use tongues_data::Seq2SeqExample;

pub const FAMILY: &str = "sentence-parser";
pub const ARCHITECTURE: &str = "seq2seq-transformer";
pub const TASK_TOKEN: &str = "<task:sentence_boundary>";
pub const PREVIOUS_TOKEN: &str = "<ctx:previous>";
pub const CURSOR_TOKEN: &str = "<ctx:cursor>";
pub const EMIT_TOKEN: &str = "<boundary:emit>";
pub const CONTINUE_TOKEN: &str = "<boundary:continue>";
pub const MISSING_HEAD_TOKEN: &str = "<boundary:missing_head>";
pub const REPAIR_TOKEN: &str = "<boundary:repair>";
const USER_AGENT: &str = "tongues-sentence-parser/0.1";
const DEFAULT_PREPARE_MAX_THREADS: usize = 8;
const DEFAULT_GUTENBERG_URLS: &[&str] = &[
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "https://www.gutenberg.org/cache/epub/98/pg98.txt",
    "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "https://www.gutenberg.org/cache/epub/345/pg345.txt",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SentenceParserConfig {
    pub dataset_id: String,
    pub lowercase: bool,
    #[serde(default)]
    pub source_paths: Vec<PathBuf>,
    #[serde(default = "default_include_default_gutenberg")]
    pub include_default_gutenberg: bool,
    #[serde(default = "default_gutenberg_urls")]
    pub gutenberg_urls: Vec<String>,
    #[serde(default = "default_include_synthetic")]
    pub include_synthetic: bool,
    #[serde(default = "default_synthetic_sentences")]
    pub synthetic_sentences: usize,
    #[serde(default = "default_train_frac")]
    pub train_frac: f64,
    #[serde(default = "default_valid_frac")]
    pub valid_frac: f64,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_min_sentence_chars")]
    pub min_sentence_chars: usize,
    #[serde(default = "default_max_sentence_chars")]
    pub max_sentence_chars: usize,
    #[serde(default = "default_max_examples_per_sentence")]
    pub max_examples_per_sentence: usize,
    #[serde(default = "default_include_naive_discrepancies")]
    pub include_naive_discrepancies: bool,
    #[serde(default = "default_max_naive_discrepancies_per_file")]
    pub max_naive_discrepancies_per_file: usize,
}

impl Default for SentenceParserConfig {
    fn default() -> Self {
        Self {
            dataset_id: "v0".to_string(),
            lowercase: false,
            source_paths: Vec::new(),
            include_default_gutenberg: default_include_default_gutenberg(),
            gutenberg_urls: default_gutenberg_urls(),
            include_synthetic: default_include_synthetic(),
            synthetic_sentences: default_synthetic_sentences(),
            train_frac: default_train_frac(),
            valid_frac: default_valid_frac(),
            seed: default_seed(),
            min_sentence_chars: default_min_sentence_chars(),
            max_sentence_chars: default_max_sentence_chars(),
            max_examples_per_sentence: default_max_examples_per_sentence(),
            include_naive_discrepancies: default_include_naive_discrepancies(),
            max_naive_discrepancies_per_file: default_max_naive_discrepancies_per_file(),
        }
    }
}

fn default_include_default_gutenberg() -> bool {
    true
}

fn default_gutenberg_urls() -> Vec<String> {
    DEFAULT_GUTENBERG_URLS
        .iter()
        .map(|url| (*url).to_string())
        .collect()
}

fn default_include_synthetic() -> bool {
    true
}

fn default_synthetic_sentences() -> usize {
    512
}

fn default_train_frac() -> f64 {
    0.8
}

fn default_valid_frac() -> f64 {
    0.1
}

fn default_seed() -> u64 {
    42
}

fn default_min_sentence_chars() -> usize {
    8
}

fn default_max_sentence_chars() -> usize {
    512
}

fn default_max_examples_per_sentence() -> usize {
    4
}

fn default_include_naive_discrepancies() -> bool {
    true
}

fn default_max_naive_discrepancies_per_file() -> usize {
    1024
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerSpec {
    pub kind: String,
    pub lowercase: bool,
}

impl Default for TokenizerSpec {
    fn default() -> Self {
        Self {
            kind: "whitespace".to_string(),
            lowercase: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelSchema {
    pub output_type: String,
    pub target_tokens: Vec<String>,
}

impl Default for LabelSchema {
    fn default() -> Self {
        Self {
            output_type: "cursor sentence-boundary action".to_string(),
            target_tokens: [EMIT_TOKEN, CONTINUE_TOKEN, MISSING_HEAD_TOKEN, REPAIR_TOKEN]
                .into_iter()
                .map(str::to_string)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BoundaryAction {
    Emit,
    Continue,
    MissingHead,
    Repair,
}

impl BoundaryAction {
    pub fn token(&self) -> &'static str {
        match self {
            Self::Emit => EMIT_TOKEN,
            Self::Continue => CONTINUE_TOKEN,
            Self::MissingHead => MISSING_HEAD_TOKEN,
            Self::Repair => REPAIR_TOKEN,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrainingRowSource {
    Seams,
    NaiveDiscrepancy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryTrainingExample {
    pub action: BoundaryAction,
    #[serde(default = "default_training_row_source")]
    pub row_source: TrainingRowSource,
    pub previous: String,
    pub cursor: String,
    pub input: String,
    pub output: String,
    pub source: String,
}

fn default_training_row_source() -> TrainingRowSource {
    TrainingRowSource::Seams
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrepareReport {
    pub source_files: usize,
    pub detected_sentences: usize,
    pub naive_discrepancy_examples: usize,
    pub train_examples: usize,
    pub valid_examples: usize,
    pub test_examples: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrepareCheckpointState {
    pub status: String,
    pub dataset_id: String,
    pub config_fingerprint: String,
    pub shards: Vec<PrepareShardManifest>,
    pub report: Option<PrepareReport>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrepareShardManifest {
    pub id: String,
    pub label: String,
    pub config_fingerprint: String,
    pub sentences_path: PathBuf,
    pub examples_path: PathBuf,
    pub discrepancies_path: Option<PathBuf>,
    pub sentences: usize,
    pub examples: usize,
    pub discrepancies: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SentenceRecord {
    sentence: String,
    source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PrepareShardData {
    sentences: Vec<SentenceRecord>,
    examples: Vec<BoundaryTrainingExample>,
    discrepancies: Vec<BoundaryTrainingExample>,
}

#[derive(Debug)]
struct PreparedSourceShard {
    index: usize,
    path: String,
    progress: Vec<PrepareProgress>,
    manifest: PrepareShardManifest,
}

#[derive(Debug)]
struct LoadedPrepareShard {
    index: usize,
    sentences: Vec<SentenceRecord>,
    examples: Vec<BoundaryTrainingExample>,
    discrepancies: Vec<BoundaryTrainingExample>,
}

pub fn prepare_dataset(out: &Path, config: &SentenceParserConfig) -> Result<PrepareReport> {
    prepare_dataset_with_progress(out, config, |_| {})
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrepareProgress {
    Stage {
        message: String,
    },
    Discover {
        files: usize,
    },
    Download {
        url: String,
        path: String,
        bytes: u64,
    },
    Synthesize {
        path: String,
        sentences: usize,
    },
    Detect {
        path: String,
        files_done: usize,
        files_total: usize,
        sentences: usize,
        naive_discrepancies: usize,
    },
    Build {
        sentences: usize,
        examples: usize,
    },
    Write {
        path: String,
        rows: usize,
    },
}

pub fn prepare_dataset_with_progress(
    out: &Path,
    config: &SentenceParserConfig,
    mut progress: impl FnMut(PrepareProgress),
) -> Result<PrepareReport> {
    progress(PrepareProgress::Stage {
        message: format!(
            "Creating sentence-parser output directory {}",
            out.display()
        ),
    });
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let files = resolve_source_files_with_progress(out, config, &mut progress)?;
    progress(PrepareProgress::Discover { files: files.len() });
    anyhow::ensure!(
        !files.is_empty(),
        "no sentence-parser source files found. Pass one or more `--input` files/directories to `sentence-parser prepare` or `sentence-parser train --prepare`, or set source_paths in the config"
    );
    let checkpoint_dir = out.join("prepare-checkpoints");
    fs::create_dir_all(&checkpoint_dir)
        .with_context(|| format!("creating {}", checkpoint_dir.display()))?;
    let prepare_threads = prepare_worker_threads();
    progress(PrepareProgress::Stage {
        message: format!(
            "Preparing with {} worker thread{}",
            prepare_threads,
            if prepare_threads == 1 { "" } else { "s" }
        ),
    });
    let prepare_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(prepare_threads)
        .build()
        .context("building sentence-parser prepare thread pool")?;
    let config_fingerprint = config_fingerprint(config)?;
    let mut shard_manifests = Vec::new();

    if config.include_synthetic {
        let manifest = build_or_load_prepare_shard(
            &checkpoint_dir,
            "000-synthetic",
            "synthetic sentences",
            &config_fingerprint,
            &mut progress,
            || {
                let text = synthesize_boundary_text(config.synthetic_sentences, config.seed);
                let sentences = detect_sentences_for_text(&text, "synthetic", config)?;
                let examples = build_boundary_examples(
                    &sentences
                        .iter()
                        .map(|record| (record.sentence.clone(), record.source.clone()))
                        .collect::<Vec<_>>(),
                    config,
                );
                Ok(PrepareShardData {
                    sentences,
                    examples,
                    discrepancies: Vec::new(),
                })
            },
        )?;
        progress(PrepareProgress::Synthesize {
            path: manifest.sentences_path.display().to_string(),
            sentences: manifest.sentences,
        });
        shard_manifests.push(manifest);
    }

    let mut prepared_source_shards = prepare_pool.install(|| {
        files
            .par_iter()
            .enumerate()
            .map(|(file_index, path)| {
                let path = path.clone();
                let shard_id = format!(
                    "{:03}-source-{}",
                    file_index + 1,
                    sanitize_checkpoint_id(&path)
                );
                let label = path.display().to_string();
                let mut shard_progress = Vec::new();
                let mut collect = |event| shard_progress.push(event);
                let manifest = build_or_load_prepare_shard(
                    &checkpoint_dir,
                    &shard_id,
                    &label,
                    &config_fingerprint,
                    &mut collect,
                    || {
                        let raw = fs::read_to_string(&path)
                            .with_context(|| format!("reading {}", path.display()))?;
                        let sentences =
                            detect_sentences_for_text(&raw, &path.display().to_string(), config)?;
                        let sentence_pairs = sentences
                            .iter()
                            .map(|record| (record.sentence.clone(), record.source.clone()))
                            .collect::<Vec<_>>();
                        let mut examples = build_boundary_examples(&sentence_pairs, config);
                        let discrepancies = if config.include_naive_discrepancies {
                            build_naive_discrepancy_examples(
                                &sentences
                                    .iter()
                                    .map(|record| record.sentence.clone())
                                    .collect::<Vec<_>>(),
                                &path.display().to_string(),
                                config,
                            )
                        } else {
                            Vec::new()
                        };
                        examples.extend(discrepancies.clone());
                        Ok(PrepareShardData {
                            sentences,
                            examples,
                            discrepancies,
                        })
                    },
                )?;
                Ok(PreparedSourceShard {
                    index: file_index,
                    path: path.display().to_string(),
                    progress: shard_progress,
                    manifest,
                })
            })
            .collect::<Result<Vec<_>>>()
    })?;
    prepared_source_shards.sort_by_key(|item| item.index);
    for prepared in prepared_source_shards {
        for event in prepared.progress {
            progress(event);
        }
        progress(PrepareProgress::Detect {
            path: prepared.path,
            files_done: prepared.index + 1,
            files_total: files.len(),
            sentences: prepared.manifest.sentences,
            naive_discrepancies: prepared.manifest.discrepancies,
        });
        shard_manifests.push(prepared.manifest);
    }

    write_prepare_state(
        out,
        "assembling",
        config,
        &config_fingerprint,
        &shard_manifests,
        None,
    )?;
    let mut sentences = Vec::new();
    let mut examples = Vec::new();
    let mut correction_examples = Vec::new();
    let mut loaded_shards = prepare_pool.install(|| {
        shard_manifests
            .par_iter()
            .enumerate()
            .map(|(index, manifest)| {
                let sentences: Vec<SentenceRecord> = read_jsonl(&manifest.sentences_path)?;
                let examples: Vec<BoundaryTrainingExample> = read_jsonl(&manifest.examples_path)?;
                let discrepancies: Vec<BoundaryTrainingExample> = manifest
                    .discrepancies_path
                    .as_ref()
                    .map(|path| read_jsonl(path))
                    .transpose()?
                    .unwrap_or_default();
                Ok(LoadedPrepareShard {
                    index,
                    sentences,
                    examples,
                    discrepancies,
                })
            })
            .collect::<Result<Vec<_>>>()
    })?;
    loaded_shards.sort_by_key(|item| item.index);
    for mut loaded in loaded_shards {
        sentences.append(&mut loaded.sentences);
        examples.append(&mut loaded.examples);
        correction_examples.append(&mut loaded.discrepancies);
    }
    let naive_discrepancy_examples = correction_examples.len();
    anyhow::ensure!(
        !sentences.is_empty(),
        "no sentence-parser sentences remained after filtering {} source files with min_sentence_chars={} and max_sentence_chars={}",
        files.len(),
        config.min_sentence_chars,
        config.max_sentence_chars
    );
    progress(PrepareProgress::Stage {
        message: format!(
            "Building boundary examples from {} detected sentences",
            sentences.len()
        ),
    });
    progress(PrepareProgress::Build {
        sentences: sentences.len(),
        examples: examples.len(),
    });
    anyhow::ensure!(
        !examples.is_empty(),
        "no sentence-parser training examples were built from {} detected sentences",
        sentences.len()
    );
    write_jsonl_with_progress(&out.join("sentences.jsonl"), &sentences, &mut progress)?;
    write_jsonl_with_progress(&out.join("examples.jsonl"), &examples, &mut progress)?;
    let (train, valid, test) =
        split_examples_by_group(examples, config.train_frac, config.valid_frac, config.seed);
    let n = train.len() + valid.len() + test.len();

    write_jsonl_with_progress(&out.join("train.jsonl"), &train, &mut progress)?;
    write_jsonl_with_progress(&out.join("valid.jsonl"), &valid, &mut progress)?;
    write_jsonl_with_progress(&out.join("test.jsonl"), &test, &mut progress)?;
    write_jsonl_with_progress(
        &out.join("naive_discrepancies.jsonl"),
        &correction_examples,
        &mut progress,
    )?;
    progress(PrepareProgress::Stage {
        message: "Building sentence-parser vocabulary".to_string(),
    });
    let vocab = build_vocab([&train[..], &valid[..], &test[..]].concat().as_slice());
    fs::write(
        out.join("vocab.json"),
        serde_json::to_string_pretty(&vocab)?,
    )?;
    progress(PrepareProgress::Write {
        path: out.join("vocab.json").display().to_string(),
        rows: train.len() + valid.len() + test.len(),
    });
    fs::write(
        out.join("dataset_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    fs::write(
        out.join("README.md"),
        dataset_readme(
            config,
            files.len(),
            sentences.len(),
            n,
            naive_discrepancy_examples,
        ),
    )?;
    let report = PrepareReport {
        source_files: files.len(),
        detected_sentences: sentences.len(),
        naive_discrepancy_examples,
        train_examples: train.len(),
        valid_examples: valid.len(),
        test_examples: test.len(),
    };
    write_prepare_state(
        out,
        "complete",
        config,
        &config_fingerprint,
        &shard_manifests,
        Some(&report),
    )?;
    Ok(report)
}

fn detect_sentences_for_text(
    text: &str,
    source: &str,
    config: &SentenceParserConfig,
) -> Result<Vec<SentenceRecord>> {
    let detector = SentenceDetectorDialog::new().context("initializing seams detector")?;
    let mut sentences = Vec::new();
    for detected in detector
        .detect_sentences_borrowed(text)
        .with_context(|| format!("detecting sentence boundaries in {source}"))?
    {
        let sentence = normalize_sentence(&detected.normalize(), config.lowercase);
        if sentence.chars().count() >= config.min_sentence_chars
            && sentence.chars().count() <= config.max_sentence_chars
        {
            sentences.push(SentenceRecord {
                sentence,
                source: source.to_string(),
            });
        }
    }
    Ok(sentences)
}

fn build_or_load_prepare_shard(
    checkpoint_dir: &Path,
    id: &str,
    label: &str,
    config_fingerprint: &str,
    progress: &mut impl FnMut(PrepareProgress),
    build: impl FnOnce() -> Result<PrepareShardData>,
) -> Result<PrepareShardManifest> {
    let manifest_path = checkpoint_dir.join(format!("{id}.manifest.json"));
    if manifest_path.exists() {
        let manifest: PrepareShardManifest = read_json_file(&manifest_path)?;
        anyhow::ensure!(
            manifest.config_fingerprint == config_fingerprint,
            "checkpoint {} was built with a different config; delete that shard or use a matching config",
            manifest_path.display()
        );
        ensure_checkpoint_file(&manifest.sentences_path)?;
        ensure_checkpoint_file(&manifest.examples_path)?;
        if let Some(path) = &manifest.discrepancies_path {
            ensure_checkpoint_file(path)?;
        }
        progress(PrepareProgress::Stage {
            message: format!(
                "Reusing checkpoint {} ({} examples)",
                manifest_path.display(),
                manifest.examples
            ),
        });
        return Ok(manifest);
    }

    let sentences_path = checkpoint_dir.join(format!("{id}.sentences.jsonl"));
    let examples_path = checkpoint_dir.join(format!("{id}.examples.jsonl"));
    let discrepancies_path = checkpoint_dir.join(format!("{id}.naive_discrepancies.jsonl"));
    archive_interrupted_part(&sentences_path)?;
    archive_interrupted_part(&examples_path)?;
    archive_interrupted_part(&discrepancies_path)?;
    archive_interrupted_part(&manifest_path)?;

    progress(PrepareProgress::Stage {
        message: format!("Building checkpoint shard {id}: {label}"),
    });
    let data = build()?;
    write_jsonl_atomic(&sentences_path, &data.sentences)?;
    write_jsonl_atomic(&examples_path, &data.examples)?;
    let discrepancies_path = if data.discrepancies.is_empty() {
        None
    } else {
        write_jsonl_atomic(&discrepancies_path, &data.discrepancies)?;
        Some(discrepancies_path)
    };
    let manifest = PrepareShardManifest {
        id: id.to_string(),
        label: label.to_string(),
        config_fingerprint: config_fingerprint.to_string(),
        sentences_path,
        examples_path,
        discrepancies_path,
        sentences: data.sentences.len(),
        examples: data.examples.len(),
        discrepancies: data.discrepancies.len(),
    };
    write_json_file_atomic(&manifest_path, &manifest)?;
    progress(PrepareProgress::Write {
        path: manifest_path.display().to_string(),
        rows: manifest.examples,
    });
    Ok(manifest)
}

fn ensure_checkpoint_file(path: &Path) -> Result<()> {
    anyhow::ensure!(
        path.exists() && path.metadata()?.len() > 0,
        "checkpoint manifest references missing or empty file {}",
        path.display()
    );
    Ok(())
}

fn write_prepare_state(
    out: &Path,
    status: &str,
    config: &SentenceParserConfig,
    config_fingerprint: &str,
    shards: &[PrepareShardManifest],
    report: Option<&PrepareReport>,
) -> Result<()> {
    let state = PrepareCheckpointState {
        status: status.to_string(),
        dataset_id: config.dataset_id.clone(),
        config_fingerprint: config_fingerprint.to_string(),
        shards: shards.to_vec(),
        report: report.cloned(),
    };
    write_json_file_atomic(&out.join("prepare_state.json"), &state)
}

fn split_examples_by_group(
    examples: Vec<BoundaryTrainingExample>,
    train_frac: f64,
    valid_frac: f64,
    seed: u64,
) -> (
    Vec<BoundaryTrainingExample>,
    Vec<BoundaryTrainingExample>,
    Vec<BoundaryTrainingExample>,
) {
    let mut grouped = BTreeMap::<String, Vec<BoundaryTrainingExample>>::new();
    for example in examples {
        grouped
            .entry(example.source.clone())
            .or_default()
            .push(example);
    }
    let mut groups = grouped.keys().cloned().collect::<Vec<_>>();
    groups.shuffle(&mut StdRng::seed_from_u64(seed));
    let n = groups.len();
    let train_end = (n as f64 * train_frac).round() as usize;
    let valid_end = (train_end + (n as f64 * valid_frac).round() as usize).min(n);

    let mut train = Vec::new();
    let mut valid = Vec::new();
    let mut test = Vec::new();
    for (index, group) in groups.iter().enumerate() {
        if let Some(rows) = grouped.remove(group) {
            if index < train_end {
                train.extend(rows);
            } else if index < valid_end {
                valid.extend(rows);
            } else {
                test.extend(rows);
            }
        }
    }
    (train, valid, test)
}

fn config_fingerprint(config: &SentenceParserConfig) -> Result<String> {
    let json = serde_json::to_string(config)?;
    Ok(format!("{:016x}", stable_hash(json.as_bytes())))
}

fn prepare_worker_threads() -> usize {
    let detected = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let configured = env::var("TONGUES_PREPARE_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(detected);
    configured.clamp(1, DEFAULT_PREPARE_MAX_THREADS)
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn sanitize_checkpoint_id(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("source")
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

fn read_json_file<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    serde_json::from_reader(file).with_context(|| format!("parsing {}", path.display()))
}

fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    raw.lines()
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(line).with_context(|| {
                format!(
                    "parsing {} line {}",
                    path.display(),
                    index.saturating_add(1)
                )
            })
        })
        .collect()
}

fn write_json_file_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let part = atomic_part_path(path);
    archive_interrupted_part(path)?;
    fs::write(&part, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("writing {}", part.display()))?;
    fs::rename(&part, path)
        .with_context(|| format!("renaming {} -> {}", part.display(), path.display()))
}

fn write_jsonl_atomic<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    let part = atomic_part_path(path);
    archive_interrupted_part(path)?;
    let mut writer = BufWriter::new(
        File::create(&part).with_context(|| format!("creating {}", part.display()))?,
    );
    for row in rows {
        writeln!(writer, "{}", serde_json::to_string(row)?)?;
    }
    writer
        .flush()
        .with_context(|| format!("flushing {}", part.display()))?;
    drop(writer);
    fs::rename(&part, path)
        .with_context(|| format!("renaming {} -> {}", part.display(), path.display()))
}

fn atomic_part_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    path.with_file_name(format!("{file_name}.writing.part"))
}

fn archive_interrupted_part(path: &Path) -> Result<()> {
    let part = atomic_part_path(path);
    if !part.exists() {
        return Ok(());
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let archive = part.with_extension(format!(
        "{}interrupted-{stamp}",
        part.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| format!("{ext}."))
            .unwrap_or_default()
    ));
    fs::rename(&part, &archive).with_context(|| {
        format!(
            "archiving interrupted partial {} -> {}",
            part.display(),
            archive.display()
        )
    })
}

pub fn build_vocab(examples: &[BoundaryTrainingExample]) -> Vocab {
    let inputs = examples
        .iter()
        .map(|example| example.input.clone())
        .collect::<Vec<_>>();
    let outputs = examples
        .iter()
        .map(|example| example.output.clone())
        .collect::<Vec<_>>();
    Vocab::build(&inputs, &outputs, &[])
}

pub fn make_seq2seq_examples(
    rows: &[BoundaryTrainingExample],
    vocab: &Vocab,
) -> Vec<Seq2SeqExample> {
    rows.iter()
        .map(|row| {
            let mut src_ids = vocab.encode_string(&row.input);
            if src_ids.first().copied() != Some(vocab.get_id(TASK_TOKEN)) {
                src_ids.insert(0, vocab.get_id(TASK_TOKEN));
            }
            let mut tgt_in_ids = vec![BOS_ID];
            tgt_in_ids.extend(vocab.encode_string(&row.output));
            let mut tgt_out_ids = vocab.encode_string(&row.output);
            tgt_out_ids.push(EOS_ID);
            Seq2SeqExample {
                src_ids,
                tgt_in_ids,
                tgt_out_ids,
            }
        })
        .collect()
}

pub fn format_boundary_input(previous: &str, cursor: &str, lowercase: bool) -> String {
    format!(
        "{}{}{}{}{}",
        TASK_TOKEN,
        PREVIOUS_TOKEN,
        normalize_sentence(previous, lowercase),
        CURSOR_TOKEN,
        normalize_sentence(cursor, lowercase)
    )
}

pub fn parse_boundary_output(output: &str) -> (&str, &str) {
    for token in [EMIT_TOKEN, CONTINUE_TOKEN, MISSING_HEAD_TOKEN, REPAIR_TOKEN] {
        if let Some(rest) = output.strip_prefix(token) {
            return (token, rest);
        }
    }
    ("", output)
}

/// Returns true if the action token represents a sentence-boundary crossing.
fn is_boundary_action(action: &str) -> bool {
    matches!(action, EMIT_TOKEN | MISSING_HEAD_TOKEN | REPAIR_TOKEN)
}

/// Per-class precision / recall / F1 scores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassMetrics {
    pub tp: usize,
    pub fp: usize,
    #[serde(rename = "fn")]
    pub fn_: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl ClassMetrics {
    fn new(tp: usize, fp: usize, fn_: usize) -> Self {
        let precision = if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = if tp + fn_ == 0 {
            0.0
        } else {
            tp as f64 / (tp + fn_) as f64
        };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        Self {
            tp,
            fp,
            fn_,
            precision,
            recall,
            f1,
        }
    }
}

/// Aggregate metrics for a sentence-parser behavioural evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMetrics {
    /// Total examples evaluated.
    pub total: usize,
    /// Examples where the predicted action token exactly matched gold.
    pub exact_action: usize,
    /// Examples where model output could not be parsed as a valid action.
    pub invalid: usize,
    /// `exact_action / total`.
    pub exact_action_accuracy: f64,
    /// `invalid / total`.
    pub invalid_rate: f64,
    /// Precision / recall / F1 treating boundary (Emit, Repair, MissingHead) as the positive class.
    pub boundary: ClassMetrics,
    /// Precision / recall / F1 treating no-boundary (Continue) as the positive class.
    pub no_boundary: ClassMetrics,
    /// Number of gold-Repair examples in the sample.
    pub repair_count: usize,
    /// Sum of character-level edit distances for Repair examples (gold text vs predicted text).
    pub repair_char_distance_sum: usize,
    /// Mean character-level edit distance for Repair examples, or 0.0 when there are none.
    pub mean_repair_char_distance: f64,
}

/// Compute character-level edit distance (Levenshtein) between two strings.
pub fn char_edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            curr[j] = if a[i - 1] == b[j - 1] {
                prev[j - 1]
            } else {
                1 + prev[j - 1].min(prev[j]).min(curr[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Evaluate sentence-parser predictions against gold outputs.
///
/// `pairs` is a slice of `(gold_output, predicted_output)` string pairs.
/// Both strings should be raw model output strings (e.g. `"<boundary:emit>Hello.\n"`).
pub fn evaluate_predictions(pairs: &[(&str, &str)]) -> EvalMetrics {
    let total = pairs.len();
    let mut exact_action = 0usize;
    let mut invalid = 0usize;

    // Binary classification: positive class = boundary-crossing action.
    let mut boundary_tp = 0usize;
    let mut boundary_fp = 0usize;
    let mut boundary_fn = 0usize;
    let mut no_boundary_tp = 0usize;
    let mut no_boundary_fp = 0usize;
    let mut no_boundary_fn = 0usize;

    let mut repair_count = 0usize;
    let mut repair_char_distance_sum = 0usize;

    for &(gold, predicted) in pairs {
        let (gold_action, gold_text) = parse_boundary_output(gold);
        let (pred_action, pred_text) = parse_boundary_output(predicted);

        if pred_action.is_empty() {
            invalid += 1;
        }

        if gold_action == pred_action {
            exact_action += 1;
        }

        let gold_is_boundary = is_boundary_action(gold_action);
        let pred_is_boundary = is_boundary_action(pred_action);

        match (gold_is_boundary, pred_is_boundary) {
            (true, true) => {
                boundary_tp += 1;
                no_boundary_fp += 0; // neither FP nor FN for no-boundary
            }
            (true, false) => {
                boundary_fn += 1;
                no_boundary_fp += 1;
            }
            (false, true) => {
                boundary_fp += 1;
                no_boundary_fn += 1;
            }
            (false, false) => {
                no_boundary_tp += 1;
            }
        }

        if gold_action == REPAIR_TOKEN {
            repair_count += 1;
            let gold_repair = gold_text.trim_end_matches('\n');
            let pred_repair = if pred_action == REPAIR_TOKEN {
                pred_text.trim_end_matches('\n')
            } else {
                predicted
            };
            repair_char_distance_sum += char_edit_distance(gold_repair, pred_repair);
        }
    }

    let exact_action_accuracy = if total == 0 {
        0.0
    } else {
        exact_action as f64 / total as f64
    };
    let invalid_rate = if total == 0 {
        0.0
    } else {
        invalid as f64 / total as f64
    };
    let mean_repair_char_distance = if repair_count == 0 {
        0.0
    } else {
        repair_char_distance_sum as f64 / repair_count as f64
    };

    EvalMetrics {
        total,
        exact_action,
        invalid,
        exact_action_accuracy,
        invalid_rate,
        boundary: ClassMetrics::new(boundary_tp, boundary_fp, boundary_fn),
        no_boundary: ClassMetrics::new(no_boundary_tp, no_boundary_fp, no_boundary_fn),
        repair_count,
        repair_char_distance_sum,
        mean_repair_char_distance,
    }
}

pub fn parse_sentence(text: &str, lowercase: bool) -> SentenceSyntaxAnalysis {
    let mut words = text
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|c: char| matches!(c, '.' | '?' | '!' | ',' | ';' | ':'))
                .to_string()
        })
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    if lowercase {
        words = words.into_iter().map(|word| word.to_lowercase()).collect();
    }
    let terminal = terminal_from_text(text);
    VarietyGrammarParser::default().parse(&words, terminal)
}

fn terminal_from_text(text: &str) -> Option<TerminalPunctuation> {
    match text.trim_end().chars().last()? {
        '?' => Some(TerminalPunctuation::Question),
        '!' => Some(TerminalPunctuation::Exclamation),
        '.' => Some(TerminalPunctuation::Period),
        _ => None,
    }
}

fn build_boundary_examples(
    sentences: &[(String, String)],
    config: &SentenceParserConfig,
) -> Vec<BoundaryTrainingExample> {
    let mut examples = Vec::new();
    for (index, (sentence, source)) in sentences.iter().enumerate() {
        let previous = index
            .checked_sub(1)
            .and_then(|prev| sentences.get(prev))
            .map(|(sentence, _)| sentence.as_str())
            .unwrap_or("");

        push_example(
            &mut examples,
            BoundaryAction::Emit,
            TrainingRowSource::Seams,
            previous,
            sentence,
            format!("{EMIT_TOKEN}{sentence}\n"),
            source,
            config.lowercase,
        );

        if config.max_examples_per_sentence > 1 {
            if let Some(prefix) = prefix_before_completion(sentence) {
                push_example(
                    &mut examples,
                    BoundaryAction::Continue,
                    TrainingRowSource::Seams,
                    previous,
                    prefix,
                    CONTINUE_TOKEN.to_string(),
                    source,
                    config.lowercase,
                );
            }
        }

        if config.max_examples_per_sentence > 2 {
            if let Some(tail) = missing_head_tail(sentence) {
                push_example(
                    &mut examples,
                    BoundaryAction::MissingHead,
                    TrainingRowSource::Seams,
                    previous,
                    tail,
                    format!("{MISSING_HEAD_TOKEN}{tail}"),
                    source,
                    config.lowercase,
                );
            }
        }

        if config.max_examples_per_sentence > 3 && index > 0 && suspicious_fragment(previous) {
            push_example(
                &mut examples,
                BoundaryAction::Repair,
                TrainingRowSource::Seams,
                previous,
                sentence,
                format!(
                    "{REPAIR_TOKEN}{} {}",
                    previous.trim_end(),
                    sentence.trim_start()
                ),
                source,
                config.lowercase,
            );
        }
    }
    examples
}

fn push_example(
    examples: &mut Vec<BoundaryTrainingExample>,
    action: BoundaryAction,
    row_source: TrainingRowSource,
    previous: &str,
    cursor: &str,
    output: String,
    source: &str,
    lowercase: bool,
) {
    let previous = normalize_sentence(previous, lowercase);
    let cursor = normalize_sentence(cursor, lowercase);
    let output = if lowercase {
        output.to_lowercase()
    } else {
        output
    };
    examples.push(BoundaryTrainingExample {
        action,
        row_source,
        input: format_boundary_input(&previous, &cursor, false),
        previous,
        cursor,
        output,
        source: source.to_string(),
    });
}

pub fn filter_examples_by_source(
    rows: Vec<BoundaryTrainingExample>,
    source: Option<TrainingRowSource>,
) -> Vec<BoundaryTrainingExample> {
    match source {
        Some(source) => rows
            .into_iter()
            .filter(|row| row.row_source == source)
            .collect(),
        None => rows,
    }
}

pub fn naive_split_sentences(text: &str, lowercase: bool) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0usize;
    for (index, ch) in text.char_indices() {
        if matches!(ch, '.' | '?' | '!') {
            let end = index + ch.len_utf8();
            let sentence = normalize_sentence(&text[start..end], lowercase);
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            start = end;
        }
    }
    let tail = normalize_sentence(&text[start..], lowercase);
    if !tail.is_empty() {
        sentences.push(tail);
    }
    sentences
}

fn build_naive_discrepancy_examples(
    seams_sentences: &[String],
    source: &str,
    config: &SentenceParserConfig,
) -> Vec<BoundaryTrainingExample> {
    let mut examples = Vec::new();
    for seams_sentence in seams_sentences {
        if examples.len() >= config.max_naive_discrepancies_per_file {
            break;
        }
        let naive = naive_split_sentences(seams_sentence, config.lowercase);
        if naive.len() <= 1 {
            continue;
        }

        let combined = normalize_sentence(&naive.join(" "), config.lowercase);
        if combined != *seams_sentence {
            continue;
        }

        let first = &naive[0];
        let cursor = naive[1..].join(" ");
        if !cursor.is_empty() {
            push_example(
                &mut examples,
                BoundaryAction::Repair,
                TrainingRowSource::NaiveDiscrepancy,
                first,
                &cursor,
                format!("{REPAIR_TOKEN}{seams_sentence}"),
                source,
                config.lowercase,
            );
        }
    }
    examples
}

fn prefix_before_completion(sentence: &str) -> Option<&str> {
    let split = sentence.char_indices().nth(sentence.chars().count() / 2)?.0;
    let prefix = sentence[..split].trim_end();
    (!prefix.is_empty()).then_some(prefix)
}

fn missing_head_tail(sentence: &str) -> Option<&str> {
    let mut word_starts = sentence.match_indices(' ').map(|(index, _)| index + 1);
    let split = word_starts.nth(1)?;
    let tail = sentence[split..].trim_start();
    (!tail.is_empty()).then_some(tail)
}

fn suspicious_fragment(previous: &str) -> bool {
    let trimmed = previous.trim_end();
    let Some(without_dot) = trimmed.strip_suffix('.') else {
        return false;
    };
    let last = without_dot.split_whitespace().last().unwrap_or("");
    last.chars().count() == 1 && last.chars().all(|c| c.is_ascii_uppercase())
}

fn normalize_sentence(text: &str, lowercase: bool) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if lowercase {
        normalized.to_lowercase()
    } else {
        normalized
    }
}

fn resolve_source_files_with_progress(
    out: &Path,
    config: &SentenceParserConfig,
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<Vec<PathBuf>> {
    let configured = discover_source_files(&config.source_paths)?;
    if !configured.is_empty() {
        return Ok(configured);
    }

    let default_dir = out.join("sources");
    fs::create_dir_all(&default_dir)
        .with_context(|| format!("creating {}", default_dir.display()))?;
    let mut generated_paths = Vec::new();

    if config.include_default_gutenberg {
        let urls = if config.gutenberg_urls.is_empty() {
            default_gutenberg_urls()
        } else {
            config.gutenberg_urls.clone()
        };
        for (index, url) in urls.iter().enumerate() {
            match download_gutenberg_source(&default_dir, index, url, progress) {
                Ok(path) => generated_paths.push(path),
                Err(error) => {
                    progress(PrepareProgress::Stage {
                        message: format!("Skipping default Gutenberg source {url}: {error}"),
                    });
                }
            }
        }
    }

    if config.include_synthetic && config.synthetic_sentences > 0 {
        let path = default_dir.join("synthetic-boundary-cases.txt");
        let text = synthesize_boundary_text(config.synthetic_sentences, config.seed);
        fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
        progress(PrepareProgress::Synthesize {
            path: path.display().to_string(),
            sentences: config.synthetic_sentences,
        });
        generated_paths.push(path);
    }

    generated_paths.sort();
    Ok(generated_paths)
}

fn download_gutenberg_source(
    dir: &Path,
    index: usize,
    url: &str,
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<PathBuf> {
    let path = dir.join(format!("{index:02}-{}", gutenberg_filename(url)));
    if path.exists() && path.metadata()?.len() > 0 {
        progress(PrepareProgress::Stage {
            message: format!("Using cached Gutenberg source {}", path.display()),
        });
        return Ok(path);
    }

    let part_path = path.with_extension("txt.part");
    progress(PrepareProgress::Stage {
        message: format!("Downloading default Gutenberg source {url}"),
    });
    let response = ureq::get(url)
        .header("User-Agent", USER_AGENT)
        .call()
        .with_context(|| format!("GET {url}"))?;
    let raw = response
        .into_body()
        .read_to_string()
        .with_context(|| format!("reading {url}"))?;
    progress(PrepareProgress::Download {
        url: url.to_string(),
        path: path.display().to_string(),
        bytes: raw.len() as u64,
    });
    let stripped = strip_gutenberg_boilerplate(&raw);
    fs::write(&part_path, stripped).with_context(|| format!("writing {}", part_path.display()))?;
    fs::rename(&part_path, &path)
        .with_context(|| format!("moving {} to {}", part_path.display(), path.display()))?;
    Ok(path)
}

fn gutenberg_filename(url: &str) -> String {
    url.rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("gutenberg.txt")
        .replace(['/', '\\', ':', '?', '&', '='], "_")
}

fn strip_gutenberg_boilerplate(raw: &str) -> String {
    let start = raw
        .find("*** START OF")
        .and_then(|index| raw[index..].find("***").map(|offset| index + offset + 3))
        .and_then(|index| raw[index..].find("***").map(|offset| index + offset + 3))
        .unwrap_or(0);
    let after_start = &raw[start..];
    let end = after_start.find("*** END OF").unwrap_or(after_start.len());
    after_start[..end].trim().to_string()
}

fn synthesize_boundary_text(sentences: usize, seed: u64) -> String {
    let first_names = ["Ada", "Mina", "Clara", "Henry", "Elias", "Nora"];
    let last_names = ["Bennet", "Weston", "Lanyon", "Murray", "Price", "Harker"];
    let places = [
        "St. Ives",
        "Washington, D.C.",
        "No. 4 station",
        "Mt. Vernon",
    ];
    let objects = [
        "the ledger",
        "a sealed note",
        "the timetable",
        "a small map",
    ];
    let verbs = [
        "examined",
        "carried",
        "misplaced",
        "copied",
        "folded",
        "delivered",
    ];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut lines = Vec::new();

    for index in 0..sentences {
        let first = first_names[rng.gen_range(0..first_names.len())];
        let last = last_names[rng.gen_range(0..last_names.len())];
        let other = last_names[rng.gen_range(0..last_names.len())];
        let place = places[rng.gen_range(0..places.len())];
        let object = objects[rng.gen_range(0..objects.len())];
        let verb = verbs[rng.gen_range(0..verbs.len())];
        let text = match index % 6 {
            0 => format!("Mr. {last} {verb} {object} before noon."),
            1 => format!("Dr. {last} met {first} at {place}, and they compared notes."),
            2 => format!("{first} J. {last} asked whether Prof. {other} had arrived."),
            3 => format!(
                "The parcel reached {place} at {hour}:15 p.m. without a label.",
                hour = 1 + index % 11
            ),
            4 => format!(
                "No. {number} was missing, but Mrs. {last} found it later.",
                number = 10 + index % 90
            ),
            _ => format!("Who told {first} F. {last} that the train had stopped?"),
        };
        lines.push(text);
        if lines.len() % 5 == 0 {
            lines.push(String::new());
        }
    }

    lines.join("\n")
}

fn discover_source_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for path in paths {
        if path.is_file() {
            files.push(path.clone());
        } else if path.is_dir() {
            discover_source_files_in_dir(path, &mut files)?;
        }
    }
    files.sort();
    Ok(files)
}

fn discover_source_files_in_dir(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            discover_source_files_in_dir(&path, files)?;
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.ends_with("-0.txt") || name.ends_with(".txt"))
            .unwrap_or(false)
        {
            files.push(path);
        }
    }
    Ok(())
}

fn write_jsonl_with_progress<T: Serialize>(
    path: &Path,
    rows: &[T],
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<()> {
    progress(PrepareProgress::Stage {
        message: format!("Writing {} rows to {}", rows.len(), path.display()),
    });
    write_jsonl_atomic(path, rows)?;
    progress(PrepareProgress::Write {
        path: path.display().to_string(),
        rows: rows.len(),
    });
    Ok(())
}

fn dataset_readme(
    config: &SentenceParserConfig,
    source_files: usize,
    sentences: usize,
    examples: usize,
    naive_discrepancy_examples: usize,
) -> String {
    format!(
        "# Sentence boundary dataset\n\nDataset: `{}`\n\nSources: {} Project Gutenberg-style text files\nDetected sentences: {}\nTraining rows: {}\nNaive-discrepancy correction rows: {}\n\nInput shape:\n\n```text\n{}{}<previous sentence>{}<cursor prefix>\n```\n\nTargets:\n\n```text\n{}<sentence>\\n\n{}\n{}<tail fragment>\n{}<repaired sentence>\n```\n\nThe source intentionally includes only the previously parsed sentence and current cursor prefix. No following sentence is provided.\n\nSplit policy: group-aware by source document path so all derived rows from one source stay together.\n",
        config.dataset_id,
        source_files,
        sentences,
        examples,
        naive_discrepancy_examples,
        TASK_TOKEN,
        PREVIOUS_TOKEN,
        CURSOR_TOKEN,
        EMIT_TOKEN,
        CONTINUE_TOKEN,
        MISSING_HEAD_TOKEN,
        REPAIR_TOKEN
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_output_matches_speech_syntax_contract() {
        let analysis = parse_sentence("The quick brown fox jumps.", false);
        let raw = serde_json::to_string(&analysis).unwrap();
        let reparsed: SentenceSyntaxAnalysis = serde_json::from_str(&raw).unwrap();

        assert_eq!(reparsed.terminal, Some(TerminalPunctuation::Period));
        assert!(!reparsed.tokens.is_empty());
    }

    #[test]
    fn repair_example_merges_bad_initial_cut() {
        let sentences = vec![
            ("Who shot John F.".to_string(), "fixture".to_string()),
            ("Kennedy?".to_string(), "fixture".to_string()),
        ];
        let config = SentenceParserConfig::default();
        let examples = build_boundary_examples(&sentences, &config);
        let repair = examples
            .iter()
            .find(|example| example.action == BoundaryAction::Repair)
            .expect("repair example");

        assert_eq!(repair.previous, "Who shot John F.");
        assert_eq!(repair.cursor, "Kennedy?");
        assert_eq!(repair.output, "<boundary:repair>Who shot John F. Kennedy?");
    }

    #[test]
    fn naive_splitter_makes_deliberate_abbreviation_mistake() {
        let naive = naive_split_sentences("Who shot John F. Kennedy?", false);

        assert_eq!(naive, vec!["Who shot John F.", "Kennedy?"]);
    }

    #[test]
    fn naive_disagreement_becomes_repair_training_row() {
        let config = SentenceParserConfig::default();
        let rows = build_naive_discrepancy_examples(
            &["Who shot John F. Kennedy?".to_string()],
            "fixture",
            &config,
        );

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].row_source, TrainingRowSource::NaiveDiscrepancy);
        assert_eq!(rows[0].action, BoundaryAction::Repair);
        assert_eq!(rows[0].previous, "Who shot John F.");
        assert_eq!(rows[0].cursor, "Kennedy?");
        assert_eq!(rows[0].output, "<boundary:repair>Who shot John F. Kennedy?");
    }

    #[test]
    fn naive_disagreement_mines_each_detected_sentence_without_raw_file_alignment() {
        let config = SentenceParserConfig::default();
        let rows = build_naive_discrepancy_examples(
            &[
                "A chapter title that would have shifted raw-file alignment.".to_string(),
                "Elizabeth met Mr. Darcy at Pemberley.".to_string(),
            ],
            "fixture",
            &config,
        );

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].previous, "Elizabeth met Mr.");
        assert_eq!(rows[0].cursor, "Darcy at Pemberley.");
        assert_eq!(
            rows[0].output,
            "<boundary:repair>Elizabeth met Mr. Darcy at Pemberley."
        );
    }

    #[test]
    fn split_examples_by_group_keeps_sources_together() {
        let mk = |source: &str, cursor: &str| BoundaryTrainingExample {
            action: BoundaryAction::Emit,
            row_source: TrainingRowSource::Seams,
            previous: String::new(),
            cursor: cursor.to_string(),
            input: cursor.to_string(),
            output: cursor.to_string(),
            source: source.to_string(),
        };
        let rows = vec![
            mk("a.txt", "a1"),
            mk("a.txt", "a2"),
            mk("b.txt", "b1"),
            mk("c.txt", "c1"),
        ];
        let (train, valid, test) = split_examples_by_group(rows, 0.5, 0.25, 9);
        for source in ["a.txt", "b.txt", "c.txt"] {
            let placements = usize::from(train.iter().any(|row| row.source == source))
                + usize::from(valid.iter().any(|row| row.source == source))
                + usize::from(test.iter().any(|row| row.source == source));
            assert_eq!(placements, 1);
        }
    }

    #[test]
    fn prepare_reuses_checkpoints_and_does_not_touch_legacy_parts() {
        let out = tempfile_path("sentence-parser-resume");
        let _ = fs::remove_dir_all(&out);
        fs::create_dir_all(&out).expect("create temp dataset dir");
        let source = out.join("source.txt");
        fs::write(
            &source,
            "Dr. Smith went home. Then he slept. Who shot John F. Kennedy?",
        )
        .expect("write source");
        let legacy_part = out.join("examples.jsonl.part");
        fs::write(&legacy_part, "legacy partial should remain untouched\n")
            .expect("write legacy partial");

        let config = SentenceParserConfig {
            dataset_id: "resume-test".to_string(),
            source_paths: vec![source],
            include_default_gutenberg: false,
            include_synthetic: false,
            include_naive_discrepancies: true,
            max_examples_per_sentence: 2,
            ..SentenceParserConfig::default()
        };

        let first = prepare_dataset(&out, &config).expect("first prepare");
        let checkpoint = out
            .join("prepare-checkpoints")
            .join("001-source-source-txt.manifest.json");
        assert!(checkpoint.exists());
        assert!(out.join("examples.jsonl").exists());
        assert_eq!(
            fs::read_to_string(&legacy_part).expect("read legacy partial"),
            "legacy partial should remain untouched\n"
        );

        for path in [
            out.join("sentences.jsonl"),
            out.join("examples.jsonl"),
            out.join("train.jsonl"),
            out.join("valid.jsonl"),
            out.join("test.jsonl"),
            out.join("naive_discrepancies.jsonl"),
            out.join("vocab.json"),
        ] {
            fs::remove_file(path).expect("remove final output");
        }

        let second = prepare_dataset(&out, &config).expect("resume prepare");
        assert_eq!(first.detected_sentences, second.detected_sentences);
        assert_eq!(
            first.naive_discrepancy_examples,
            second.naive_discrepancy_examples
        );
        assert_eq!(first.train_examples, second.train_examples);
        assert_eq!(
            fs::read_to_string(&legacy_part).expect("read legacy partial after resume"),
            "legacy partial should remain untouched\n"
        );
        fs::remove_dir_all(&out).expect("remove temp dataset dir");
    }

    fn tempfile_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("{}-{name}", std::process::id()))
    }

    #[test]
    fn evaluate_predictions_all_correct_gives_perfect_metrics() {
        let pairs = vec![
            ("<boundary:emit>Hello.\n", "<boundary:emit>Hello.\n"),
            ("<boundary:continue>", "<boundary:continue>"),
            ("<boundary:repair>A B", "<boundary:repair>A B"),
        ];
        let pairs_ref: Vec<(&str, &str)> = pairs.iter().map(|&(a, b)| (a, b)).collect();
        let m = evaluate_predictions(&pairs_ref);

        assert_eq!(m.total, 3);
        assert_eq!(m.exact_action, 3);
        assert_eq!(m.invalid, 0);
        assert!((m.exact_action_accuracy - 1.0).abs() < 1e-9);
        assert!((m.invalid_rate).abs() < 1e-9);
        assert!((m.boundary.f1 - 1.0).abs() < 1e-9);
        assert!((m.no_boundary.f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn evaluate_predictions_wrong_action_changes_accuracy() {
        // Gold says emit but model says continue
        let pairs = vec![
            ("<boundary:emit>Hello.\n", "<boundary:continue>"),
            ("<boundary:emit>World.\n", "<boundary:emit>World.\n"),
        ];
        let pairs_ref: Vec<(&str, &str)> = pairs.iter().map(|&(a, b)| (a, b)).collect();
        let m = evaluate_predictions(&pairs_ref);

        assert_eq!(m.total, 2);
        assert_eq!(m.exact_action, 1);
        assert!((m.exact_action_accuracy - 0.5).abs() < 1e-9);
        // One boundary gold missed → FN
        assert_eq!(m.boundary.fn_, 1);
        // No-boundary wrongly predicted once → no_boundary FP
        assert_eq!(m.no_boundary.fp, 1);
    }

    #[test]
    fn evaluate_predictions_invalid_output_counted() {
        let pairs = vec![
            ("<boundary:emit>Hello.\n", "garbage output"),
            ("<boundary:emit>World.\n", "<boundary:emit>World.\n"),
        ];
        let pairs_ref: Vec<(&str, &str)> = pairs.iter().map(|&(a, b)| (a, b)).collect();
        let m = evaluate_predictions(&pairs_ref);

        assert_eq!(m.invalid, 1);
        assert!((m.invalid_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn evaluate_predictions_repair_distance_computed() {
        let gold = "<boundary:repair>Who shot John F. Kennedy?";
        let pred_correct = "<boundary:repair>Who shot John F. Kennedy?";
        let pred_wrong = "<boundary:repair>Who shot John F.";

        let pairs_perfect: Vec<(&str, &str)> = vec![(gold, pred_correct)];
        let m_perfect = evaluate_predictions(&pairs_perfect);
        assert_eq!(m_perfect.repair_count, 1);
        assert_eq!(m_perfect.repair_char_distance_sum, 0);
        assert!((m_perfect.mean_repair_char_distance).abs() < 1e-9);

        let pairs_wrong: Vec<(&str, &str)> = vec![(gold, pred_wrong)];
        let m_wrong = evaluate_predictions(&pairs_wrong);
        assert_eq!(m_wrong.repair_count, 1);
        assert!(m_wrong.repair_char_distance_sum > 0);
        assert!(m_wrong.mean_repair_char_distance > 0.0);
    }

    #[test]
    fn char_edit_distance_empty_strings() {
        assert_eq!(char_edit_distance("", ""), 0);
        assert_eq!(char_edit_distance("abc", ""), 3);
        assert_eq!(char_edit_distance("", "abc"), 3);
    }

    #[test]
    fn char_edit_distance_equal_strings() {
        assert_eq!(char_edit_distance("hello", "hello"), 0);
    }

    #[test]
    fn char_edit_distance_single_substitution() {
        assert_eq!(char_edit_distance("cat", "bat"), 1);
    }
}
