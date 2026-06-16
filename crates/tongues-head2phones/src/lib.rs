//! Head-chunk-to-phones seq2seq data preparation.
//!
//! The model sees a raw rolling UTF-8 English buffer and emits either
//! `<NO_HEAD>` or phones plus the Unicode grapheme-cluster split offset for
//! the first complete TTS-speakable head chunk.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use seams::SentenceDetectorDialog;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use speaking::{
    phonemicizer_for_variety, PauseKind, PhonemicizeOutput, PhonemicizeRequest,
    SpeechBoundaryToken, TerminalPunctuation, VarietyId,
};
use tongues_core::{Vocab, BOS_ID, EOS_ID};
use tongues_data::Seq2SeqExample;
use tongues_neural::{write_manifest, ModelArtifactManifest};
use unicode_segmentation::UnicodeSegmentation;

pub const FAMILY: &str = "head2phones";
pub const ARCHITECTURE: &str = "seq2seq-transformer";
pub const TASK_TOKEN: &str = "<task:head2phones>";
pub const PHONES_OPEN: &str = "<PHONES>";
pub const PHONES_CLOSE: &str = "</PHONES>";
pub const SPLIT_AFTER: &str = "<SPLIT_AFTER>";
pub const NO_HEAD: &str = "<NO_HEAD>";
pub const ERROR_REPAIR: &str = "<ERROR_REPAIR>";
pub const ROLLBACK_GRAPHEMES: &str = "<ROLLBACK_GRAPHEMES>";
pub const CONFIDENCE: &str = "<CONFIDENCE>";
pub const CONFIDENCE_LOW: &str = "low";
pub const END_OF_TEXT: &str = "<END_OF_TEXT>";
const USER_AGENT: &str = "tongues-head2phones/0.1";
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
pub struct Head2PhonesConfig {
    pub dataset_id: String,
    #[serde(default)]
    pub source_paths: Vec<PathBuf>,
    #[serde(default = "default_include_default_gutenberg")]
    pub include_default_gutenberg: bool,
    #[serde(default = "default_gutenberg_urls")]
    pub gutenberg_urls: Vec<String>,
    #[serde(default = "default_include_synthetic")]
    pub include_synthetic: bool,
    #[serde(default = "default_synthetic_buffers")]
    pub synthetic_buffers: usize,
    #[serde(default = "default_random_cuts_per_buffer")]
    pub random_cuts_per_buffer: usize,
    #[serde(default = "default_no_head_cuts_per_head")]
    pub no_head_cuts_per_head: usize,
    #[serde(default = "default_include_exceptional")]
    pub include_exceptional: bool,
    #[serde(default = "default_include_naive_seams_discrepancies")]
    pub include_naive_seams_discrepancies: bool,
    #[serde(default = "default_max_naive_seams_discrepancies_per_file")]
    pub max_naive_seams_discrepancies_per_file: usize,
    #[serde(default = "default_train_frac")]
    pub train_frac: f64,
    #[serde(default = "default_valid_frac")]
    pub valid_frac: f64,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_min_head_graphemes")]
    pub min_head_graphemes: usize,
    #[serde(default = "default_max_head_graphemes")]
    pub max_head_graphemes: usize,
}

impl Default for Head2PhonesConfig {
    fn default() -> Self {
        Self {
            dataset_id: "v0".to_string(),
            source_paths: Vec::new(),
            include_default_gutenberg: default_include_default_gutenberg(),
            gutenberg_urls: default_gutenberg_urls(),
            include_synthetic: default_include_synthetic(),
            synthetic_buffers: default_synthetic_buffers(),
            random_cuts_per_buffer: default_random_cuts_per_buffer(),
            no_head_cuts_per_head: default_no_head_cuts_per_head(),
            include_exceptional: default_include_exceptional(),
            include_naive_seams_discrepancies: default_include_naive_seams_discrepancies(),
            max_naive_seams_discrepancies_per_file: default_max_naive_seams_discrepancies_per_file(
            ),
            train_frac: default_train_frac(),
            valid_frac: default_valid_frac(),
            seed: default_seed(),
            min_head_graphemes: default_min_head_graphemes(),
            max_head_graphemes: default_max_head_graphemes(),
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

fn default_synthetic_buffers() -> usize {
    4096
}

fn default_random_cuts_per_buffer() -> usize {
    8
}

fn default_no_head_cuts_per_head() -> usize {
    8
}

fn default_include_exceptional() -> bool {
    true
}

fn default_include_naive_seams_discrepancies() -> bool {
    true
}

fn default_max_naive_seams_discrepancies_per_file() -> usize {
    1024
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

fn default_min_head_graphemes() -> usize {
    4
}

fn default_max_head_graphemes() -> usize {
    220
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Head2PhonesTrainingExample {
    #[serde(default = "default_training_row_source")]
    pub row_source: TrainingRowSource,
    pub input: String,
    pub output: String,
    pub head: Option<String>,
    pub split_after: Option<usize>,
    pub source: String,
}

fn default_training_row_source() -> TrainingRowSource {
    TrainingRowSource::Synthetic
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrainingRowSource {
    Synthetic,
    RandomCut,
    Exceptional,
    Repair,
    SourceText,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NaiveSeamsDiscrepancy {
    pub source: String,
    pub seams_sentence: String,
    pub naive_sentences: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrepareReport {
    pub source_buffers: usize,
    pub naive_seams_discrepancies: usize,
    pub complete_examples: usize,
    pub no_head_examples: usize,
    pub repair_examples: usize,
    pub exceptional_examples: usize,
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
    pub examples_path: PathBuf,
    pub discrepancies_path: Option<PathBuf>,
    pub synthetic_buffers_path: Option<PathBuf>,
    pub source_buffers: usize,
    pub examples: usize,
    pub discrepancies: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PrepareShardData {
    examples: Vec<Head2PhonesTrainingExample>,
    discrepancies: Vec<NaiveSeamsDiscrepancy>,
    synthetic_buffers: Option<Vec<String>>,
    source_buffers: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrepareProgress {
    Stage {
        message: String,
    },
    Download {
        url: String,
        path: String,
        bytes: u64,
    },
    Read {
        path: String,
        buffers: usize,
        naive_seams_discrepancies: usize,
    },
    Synthesize {
        path: String,
        buffers: usize,
    },
    Build {
        complete: usize,
        no_head: usize,
    },
    Write {
        path: String,
        rows: usize,
    },
}

pub fn prepare_dataset(out: &Path, config: &Head2PhonesConfig) -> Result<PrepareReport> {
    prepare_dataset_with_progress(out, config, |_| {})
}

pub fn prepare_dataset_with_progress(
    out: &Path,
    config: &Head2PhonesConfig,
    mut progress: impl FnMut(PrepareProgress),
) -> Result<PrepareReport> {
    progress(PrepareProgress::Stage {
        message: format!("Creating head2phones output directory {}", out.display()),
    });
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;

    let checkpoint_dir = out.join("prepare-checkpoints");
    fs::create_dir_all(&checkpoint_dir)
        .with_context(|| format!("creating {}", checkpoint_dir.display()))?;
    let config_fingerprint = config_fingerprint(config)?;
    let mut shard_manifests = Vec::new();

    if config.include_synthetic {
        let manifest = build_or_load_prepare_shard(
            &checkpoint_dir,
            "000-synthetic",
            "synthetic buffers",
            &config_fingerprint,
            &mut progress,
            || {
                let mut rng = StdRng::seed_from_u64(unit_seed(config.seed, "synthetic"));
                let synthetic = synthetic_buffers(config.synthetic_buffers, &mut rng);
                let mut examples = Vec::new();
                let examples_part_path = checkpoint_dir.join("000-synthetic.examples.jsonl.part");
                archive_existing_path(&examples_part_path)?;
                let mut examples_part = BufWriter::new(
                    File::create(&examples_part_path)
                        .with_context(|| format!("creating {}", examples_part_path.display()))?,
                );
                for (index, buffer) in synthetic.iter().enumerate() {
                    let previous_len = examples.len();
                    add_examples_for_buffer(
                        buffer,
                        "synthetic",
                        TrainingRowSource::Synthetic,
                        config,
                        &mut rng,
                        &mut examples,
                    )?;
                    for row in &examples[previous_len..] {
                        writeln!(examples_part, "{}", serde_json::to_string(row)?)?;
                    }
                    if index < 4 || (index + 1) % 64 == 0 || index + 1 == synthetic.len() {
                        examples_part.flush().with_context(|| {
                            format!("flushing {}", examples_part_path.display())
                        })?;
                    }
                }
                examples_part
                    .flush()
                    .with_context(|| format!("flushing {}", examples_part_path.display()))?;
                Ok(PrepareShardData {
                    source_buffers: synthetic.len(),
                    examples,
                    discrepancies: Vec::new(),
                    synthetic_buffers: Some(synthetic),
                })
            },
        )?;
        progress(PrepareProgress::Synthesize {
            path: manifest
                .synthetic_buffers_path
                .as_ref()
                .unwrap_or(&manifest.examples_path)
                .display()
                .to_string(),
            buffers: manifest.source_buffers,
        });
        shard_manifests.push(manifest);
    }

    if config.include_exceptional {
        let manifest = build_or_load_prepare_shard(
            &checkpoint_dir,
            "001-exceptional",
            "exceptional buffers",
            &config_fingerprint,
            &mut progress,
            || {
                let mut rng = StdRng::seed_from_u64(unit_seed(config.seed, "exceptional"));
                let mut examples = Vec::new();
                let mut source_buffers = 0usize;
                for buffer in exceptional_buffers() {
                    source_buffers += 1;
                    add_examples_for_buffer(
                        buffer,
                        "exceptional",
                        TrainingRowSource::Exceptional,
                        config,
                        &mut rng,
                        &mut examples,
                    )?;
                }
                add_repair_examples_for_discrepancies(
                    &exceptional_repair_discrepancies(),
                    config,
                    &mut examples,
                );
                Ok(PrepareShardData {
                    source_buffers,
                    examples,
                    discrepancies: Vec::new(),
                    synthetic_buffers: None,
                })
            },
        )?;
        shard_manifests.push(manifest);
    }

    let source_files = resolve_source_files_with_progress(out, config, &mut progress)?;
    for (index, path) in source_files.iter().enumerate() {
        let shard_id = format!("{:03}-source-{}", index + 2, sanitize_checkpoint_id(path));
        let label = path.display().to_string();
        let manifest = build_or_load_prepare_shard(
            &checkpoint_dir,
            &shard_id,
            &label,
            &config_fingerprint,
            &mut progress,
            || {
                let mut rng = StdRng::seed_from_u64(unit_seed(
                    config.seed,
                    &format!("source:{}", path.display()),
                ));
                let raw = fs::read_to_string(path)
                    .with_context(|| format!("reading {}", path.display()))?;
                let seams_sentences = seams_sentences_from_text(&raw);
                let discrepancies = if config.include_naive_seams_discrepancies {
                    build_naive_seams_discrepancies(
                        &seams_sentences,
                        &path.display().to_string(),
                        config.max_naive_seams_discrepancies_per_file,
                    )
                } else {
                    Vec::new()
                };
                let buffers = source_buffers_from_sentences(&raw, &seams_sentences);
                let mut examples = Vec::new();
                for buffer in &buffers {
                    add_examples_for_buffer(
                        buffer,
                        &path.display().to_string(),
                        TrainingRowSource::SourceText,
                        config,
                        &mut rng,
                        &mut examples,
                    )?;
                }
                add_repair_examples_for_discrepancies(&discrepancies, config, &mut examples);
                Ok(PrepareShardData {
                    source_buffers: buffers.len(),
                    examples,
                    discrepancies,
                    synthetic_buffers: None,
                })
            },
        )?;
        progress(PrepareProgress::Read {
            path: path.display().to_string(),
            buffers: manifest.source_buffers,
            naive_seams_discrepancies: manifest.discrepancies,
        });
        shard_manifests.push(manifest);
    }

    write_prepare_state(
        out,
        "assembling",
        config,
        &config_fingerprint,
        &shard_manifests,
        None,
    )?;

    let mut examples = Vec::new();
    let mut naive_seams_discrepancies = Vec::new();
    let mut source_buffers = 0usize;
    for manifest in &shard_manifests {
        let mut shard_examples: Vec<Head2PhonesTrainingExample> =
            read_jsonl(&manifest.examples_path)?;
        let mut shard_discrepancies: Vec<NaiveSeamsDiscrepancy> = manifest
            .discrepancies_path
            .as_ref()
            .map(|path| read_jsonl(path))
            .transpose()?
            .unwrap_or_default();
        source_buffers += manifest.source_buffers;
        examples.append(&mut shard_examples);
        naive_seams_discrepancies.append(&mut shard_discrepancies);
    }

    let complete_examples = examples
        .iter()
        .filter(|example| example.head.is_some() && example.row_source != TrainingRowSource::Repair)
        .count();
    let no_head_examples = examples
        .iter()
        .filter(|example| example.output == NO_HEAD)
        .count();
    let repair_examples = examples
        .iter()
        .filter(|example| example.row_source == TrainingRowSource::Repair)
        .count();
    let exceptional_examples = examples
        .iter()
        .filter(|example| example.row_source == TrainingRowSource::Exceptional)
        .count();
    progress(PrepareProgress::Build {
        complete: complete_examples,
        no_head: no_head_examples,
    });
    anyhow::ensure!(
        complete_examples > 0 && no_head_examples > 0,
        "head2phones needs both complete and <NO_HEAD> examples; got complete={} no_head={}",
        complete_examples,
        no_head_examples
    );

    let mut rng = StdRng::seed_from_u64(unit_seed(config.seed, "split-shuffle"));
    examples.shuffle(&mut rng);
    let n = examples.len();
    let train_end = (n as f64 * config.train_frac).round() as usize;
    let valid_end = (train_end + (n as f64 * config.valid_frac).round() as usize).min(n);
    let train = examples[..train_end.min(n)].to_vec();
    let valid = examples[train_end.min(n)..valid_end].to_vec();
    let test = examples[valid_end..].to_vec();

    write_jsonl_with_progress(&out.join("examples.jsonl"), &examples, &mut progress)?;
    write_jsonl_with_progress(&out.join("train.jsonl"), &train, &mut progress)?;
    write_jsonl_with_progress(&out.join("valid.jsonl"), &valid, &mut progress)?;
    write_jsonl_with_progress(&out.join("test.jsonl"), &test, &mut progress)?;
    write_jsonl_with_progress(
        &out.join("naive_seams_discrepancies.jsonl"),
        &naive_seams_discrepancies,
        &mut progress,
    )?;

    progress(PrepareProgress::Stage {
        message: "Building head2phones vocabulary".to_string(),
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
            &train,
            &valid,
            &test,
            naive_seams_discrepancies.len(),
        ),
    )?;

    let report = PrepareReport {
        source_buffers,
        naive_seams_discrepancies: naive_seams_discrepancies.len(),
        complete_examples,
        no_head_examples,
        repair_examples,
        exceptional_examples,
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
        ensure_checkpoint_file(&manifest.examples_path)?;
        if let Some(path) = &manifest.discrepancies_path {
            ensure_checkpoint_file(path)?;
        }
        if let Some(path) = &manifest.synthetic_buffers_path {
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

    let examples_path = checkpoint_dir.join(format!("{id}.examples.jsonl"));
    let discrepancies_path = checkpoint_dir.join(format!("{id}.discrepancies.jsonl"));
    let synthetic_buffers_path = checkpoint_dir.join(format!("{id}.synthetic_buffers.jsonl"));
    archive_interrupted_part(&examples_path)?;
    archive_interrupted_part(&discrepancies_path)?;
    archive_interrupted_part(&synthetic_buffers_path)?;
    archive_interrupted_part(&manifest_path)?;

    progress(PrepareProgress::Stage {
        message: format!("Building checkpoint shard {id}: {label}"),
    });
    let data = build()?;
    write_jsonl_atomic(&examples_path, &data.examples)?;
    let discrepancies_path = if data.discrepancies.is_empty() {
        None
    } else {
        write_jsonl_atomic(&discrepancies_path, &data.discrepancies)?;
        Some(discrepancies_path)
    };
    let synthetic_buffers_path = if let Some(buffers) = &data.synthetic_buffers {
        write_jsonl_atomic(&synthetic_buffers_path, buffers)?;
        Some(synthetic_buffers_path)
    } else {
        None
    };
    let manifest = PrepareShardManifest {
        id: id.to_string(),
        label: label.to_string(),
        config_fingerprint: config_fingerprint.to_string(),
        examples_path,
        discrepancies_path,
        synthetic_buffers_path,
        source_buffers: data.source_buffers,
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
    config: &Head2PhonesConfig,
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

fn config_fingerprint(config: &Head2PhonesConfig) -> Result<String> {
    let json = serde_json::to_string(config)?;
    Ok(format!("{:016x}", stable_hash(json.as_bytes())))
}

fn unit_seed(base_seed: u64, label: &str) -> u64 {
    base_seed ^ stable_hash(label.as_bytes())
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
    archive_existing_path(&part)
}

fn archive_existing_path(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let archive = path.with_extension(format!(
        "{}interrupted-{stamp}",
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| format!("{ext}."))
            .unwrap_or_default()
    ));
    fs::rename(path, &archive).with_context(|| {
        format!(
            "archiving interrupted partial {} -> {}",
            path.display(),
            archive.display()
        )
    })
}

fn add_examples_for_buffer(
    buffer: &str,
    source: &str,
    row_source: TrainingRowSource,
    config: &Head2PhonesConfig,
    rng: &mut StdRng,
    examples: &mut Vec<Head2PhonesTrainingExample>,
) -> Result<()> {
    if let Some(head) = first_complete_head(buffer) {
        let head_text = buffer[..head.end_byte].trim().to_string();
        let split_after = buffer[..head.end_byte].graphemes(true).count();
        let head_len = head_text.graphemes(true).count();
        if head_len >= config.min_head_graphemes && head_len <= config.max_head_graphemes {
            if let Some(phones) = speech_symbols_for_text(&head_text) {
                let row = Head2PhonesTrainingExample {
                    row_source,
                    input: buffer.to_string(),
                    output: format!(
                        "{PHONES_OPEN} {phones} {PHONES_CLOSE}\n{SPLIT_AFTER} {split_after}"
                    ),
                    head: Some(head_text.clone()),
                    split_after: Some(split_after),
                    source: source.to_string(),
                };
                examples.push(row);
            }
        }

        for cut_buffer in random_complete_buffers(buffer, head.end_byte, config, rng) {
            if let Some(symbols) = speech_symbols_for_text(&head_text) {
                let row = Head2PhonesTrainingExample {
                    row_source: if row_source == TrainingRowSource::Exceptional {
                        TrainingRowSource::Exceptional
                    } else {
                        TrainingRowSource::RandomCut
                    },
                    input: cut_buffer,
                    output: format!(
                        "{PHONES_OPEN} {symbols} {PHONES_CLOSE}\n{SPLIT_AFTER} {split_after}"
                    ),
                    head: Some(head_text.clone()),
                    split_after: Some(split_after),
                    source: source.to_string(),
                };
                examples.push(row);
            }
        }

        for prefix in no_head_prefixes(&head_text, config, rng) {
            let row = Head2PhonesTrainingExample {
                row_source: if row_source == TrainingRowSource::Exceptional {
                    TrainingRowSource::Exceptional
                } else {
                    TrainingRowSource::RandomCut
                },
                input: prefix.clone(),
                output: NO_HEAD.to_string(),
                head: None,
                split_after: None,
                source: source.to_string(),
            };
            examples.push(row);

            if prefix_ends_at_boundary_in_head(&prefix, &head_text) {
                if let Some(flush_row) = flush_example_for_prefix(&prefix, source, row_source) {
                    examples.push(flush_row);
                }
            }
        }
    } else if !buffer.trim().is_empty() {
        let row = Head2PhonesTrainingExample {
            row_source,
            input: buffer.to_string(),
            output: NO_HEAD.to_string(),
            head: None,
            split_after: None,
            source: source.to_string(),
        };
        examples.push(row);

        if let Some(flush_row) = flush_example_for_prefix(buffer.trim(), source, row_source) {
            examples.push(flush_row);
        }
    }
    Ok(())
}

fn flush_example_for_prefix(
    prefix: &str,
    source: &str,
    row_source: TrainingRowSource,
) -> Option<Head2PhonesTrainingExample> {
    let head_text = prefix.trim();
    if prefix_ends_with_nonterminal_abbreviation(head_text) {
        return None;
    }
    if head_text.split_whitespace().count() < 2 {
        return None;
    }
    let last_word = head_text
        .split_whitespace()
        .last()
        .unwrap_or("")
        .trim_matches(|ch: char| !ch.is_alphanumeric());
    if last_word.chars().count() < 3 {
        return None;
    }
    let symbols = speech_symbols_for_text(head_text)?;
    let split_after = head_text.graphemes(true).count();
    Some(Head2PhonesTrainingExample {
        row_source,
        input: format!("{head_text}{END_OF_TEXT}"),
        output: format!("{PHONES_OPEN} {symbols} {PHONES_CLOSE}\n{SPLIT_AFTER} {split_after}"),
        head: Some(head_text.to_string()),
        split_after: Some(split_after),
        source: source.to_string(),
    })
}

fn prefix_ends_with_nonterminal_abbreviation(prefix: &str) -> bool {
    let trimmed = prefix.trim_end();
    trimmed
        .strip_suffix('.')
        .and_then(|without_dot| Some(without_dot.len()))
        .is_some_and(|dot_index| dot_is_nonterminal(trimmed, dot_index))
}

fn prefix_ends_at_boundary_in_head(prefix: &str, head: &str) -> bool {
    let Some(rest) = head.strip_prefix(prefix) else {
        return false;
    };
    rest.chars()
        .next()
        .is_none_or(|ch| ch.is_whitespace() || !ch.is_alphanumeric())
}

fn add_repair_examples_for_discrepancies(
    discrepancies: &[NaiveSeamsDiscrepancy],
    config: &Head2PhonesConfig,
    examples: &mut Vec<Head2PhonesTrainingExample>,
) {
    for discrepancy in discrepancies {
        if let Some(row) = repair_example_for_discrepancy(discrepancy, config) {
            examples.push(row);
        }
    }
}

fn repair_example_for_discrepancy(
    discrepancy: &NaiveSeamsDiscrepancy,
    config: &Head2PhonesConfig,
) -> Option<Head2PhonesTrainingExample> {
    let wrong_head = discrepancy.naive_sentences.first()?.trim();
    let repaired_head = discrepancy.seams_sentence.trim();
    if wrong_head.is_empty() || repaired_head.is_empty() || wrong_head == repaired_head {
        return None;
    }
    if !repaired_head.starts_with(wrong_head) {
        return None;
    }
    let repaired_len = repaired_head.graphemes(true).count();
    if repaired_len < config.min_head_graphemes || repaired_len > config.max_head_graphemes {
        return None;
    }
    let rollback = wrong_head.graphemes(true).count();
    let symbols = speech_symbols_for_text(repaired_head)?;
    Some(Head2PhonesTrainingExample {
        row_source: TrainingRowSource::Repair,
        input: repaired_head.to_string(),
        output: format!(
            "{CONFIDENCE} {CONFIDENCE_LOW}\n{ERROR_REPAIR}\n{ROLLBACK_GRAPHEMES} {rollback}\n{PHONES_OPEN} {symbols} {PHONES_CLOSE}\n{SPLIT_AFTER} {repaired_len}"
        ),
        head: Some(repaired_head.to_string()),
        split_after: Some(repaired_len),
        source: discrepancy.source.clone(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadChunk {
    pub end_byte: usize,
    pub hard: bool,
}

pub fn first_complete_head(input: &str) -> Option<HeadChunk> {
    let mut search_start = 0usize;
    while let Some((relative, ch)) = input[search_start..]
        .char_indices()
        .find(|(_, ch)| matches!(ch, '.' | '!' | '?' | ';' | ':' | ',' | '—' | '–' | '\n'))
    {
        let index = search_start + relative;
        let after = index + ch.len_utf8();
        if ch == '.' && dot_is_nonterminal(input, index) {
            search_start = after;
            continue;
        }
        let end = closing_punctuation_end(input, after);
        let hard = matches!(ch, '.' | '!' | '?' | ';' | ':' | '\n');
        if hard || soft_boundary_is_speakable(input, end) {
            return Some(HeadChunk {
                end_byte: end,
                hard,
            });
        }
        search_start = after;
    }
    None
}

fn dot_is_nonterminal(input: &str, dot_index: usize) -> bool {
    let after_dot = dot_index + 1;
    if input[after_dot..].chars().next() == Some('.') {
        return false;
    }
    if dot_is_inside_dotted_abbreviation(input, dot_index) {
        return true;
    }
    let prev = input[..dot_index].chars().rev().next();
    let next = input[after_dot..].chars().next();
    if prev.is_some_and(|ch| ch.is_ascii_digit()) && next.is_some_and(|ch| ch.is_ascii_digit()) {
        return true;
    }
    let prefix = input[..dot_index].trim_end();
    let token = prefix
        .split_whitespace()
        .last()
        .unwrap_or("")
        .trim_matches(|ch: char| matches!(ch, '"' | '\'' | '(' | '[' | '{' | '*' | '_' | '-'));
    let lower = token.to_ascii_lowercase();
    if lower == "no"
        && input[after_dot..]
            .trim_start()
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_digit())
    {
        return true;
    }
    matches!(
        lower.as_str(),
        "mr" | "mrs"
            | "ms"
            | "dr"
            | "rep"
            | "sen"
            | "gov"
            | "prof"
            | "sr"
            | "jr"
            | "st"
            | "mt"
            | "vs"
            | "etc"
            | "e.g"
            | "i.e"
            | "a.m"
            | "p.m"
            | "fig"
            | "dept"
            | "inc"
            | "ltd"
            | "co"
    ) || (token.chars().count() == 1 && token.chars().all(|ch| ch.is_ascii_uppercase()))
        || (token.chars().count() <= 3
            && token.chars().all(|ch| ch.is_ascii_uppercase())
            && (input[after_dot..].trim_start().starts_with(')')
                || inside_unclosed_parenthetical(input, dot_index)))
}

fn dot_is_inside_dotted_abbreviation(input: &str, dot_index: usize) -> bool {
    let bytes = input.as_bytes();
    let mut start = dot_index;
    while start > 0 {
        let ch = bytes[start - 1] as char;
        if ch.is_ascii_alphabetic() || ch == '.' {
            start -= 1;
        } else {
            break;
        }
    }
    let mut end = dot_index + 1;
    while end < bytes.len() {
        let ch = bytes[end] as char;
        if ch.is_ascii_alphabetic() || ch == '.' {
            end += 1;
        } else {
            break;
        }
    }
    let token = input[start..end].to_ascii_lowercase();
    matches!(token.as_str(), "e.g." | "i.e." | "a.m." | "p.m.")
}

fn inside_unclosed_parenthetical(input: &str, dot_index: usize) -> bool {
    let prefix = &input[..dot_index];
    prefix
        .rfind('(')
        .is_some_and(|open| prefix[open..].find(')').is_none())
}

fn closing_punctuation_end(input: &str, mut index: usize) -> usize {
    while let Some(ch) = input[index..].chars().next() {
        if matches!(ch, '"' | '\'' | ')' | ']' | '}') {
            index += ch.len_utf8();
        } else {
            break;
        }
    }
    index
}

fn soft_boundary_is_speakable(input: &str, end: usize) -> bool {
    let head = input[..end].split_whitespace().count();
    let rest = input[end..].trim_start();
    head >= 4 && !rest.is_empty()
}

pub fn grapheme_split(input: &str, split_after: usize) -> (&str, &str) {
    let byte = input
        .grapheme_indices(true)
        .nth(split_after)
        .map(|(byte, _)| byte)
        .unwrap_or(input.len());
    input.split_at(byte)
}

pub fn build_vocab(examples: &[Head2PhonesTrainingExample]) -> Vocab {
    let inputs = examples
        .iter()
        .map(|example| format!("{TASK_TOKEN}{}", example.input))
        .collect::<Vec<_>>();
    let outputs = examples
        .iter()
        .map(|example| example.output.clone())
        .collect::<Vec<_>>();
    Vocab::build(&inputs, &outputs, &[])
}

pub fn make_seq2seq_examples(
    rows: &[Head2PhonesTrainingExample],
    vocab: &Vocab,
) -> Vec<Seq2SeqExample> {
    rows.iter()
        .map(|row| {
            let mut src_ids = vocab.encode_string(&format!("{TASK_TOKEN}{}", row.input));
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

pub fn format_input(buffer: &str) -> String {
    format!("{TASK_TOKEN}{buffer}")
}

fn speech_symbols_for_text(text: &str) -> Option<String> {
    let variety = VarietyId("en-US".to_string());
    let phonemicizer = phonemicizer_for_variety(&variety).ok()?;
    let phonemicized = phonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: text.to_string(),
            variety,
            style: None,
        })
        .ok()?;
    phones_for_phonemicized(&phonemicized)
}

fn phones_for_phonemicized(phonemicized: &PhonemicizeOutput) -> Option<String> {
    let mut words: Vec<(usize, Vec<speaking::Syllable>)> = Vec::new();
    for syllable in phonemicized.syllables.iter() {
        let Some(first_phone) = syllable.phones.first() else {
            continue;
        };
        let Some(word_idx) = token_word_index(&first_phone.features) else {
            continue;
        };
        if let Some(last_word) = words.last_mut() {
            if last_word.0 == word_idx {
                last_word.1.push(syllable.clone());
                continue;
            }
        }
        words.push((word_idx, vec![syllable.clone()]));
    }
    let mut symbols = Vec::new();
    let last_index = words.len().saturating_sub(1);
    for (position, (word_index, syllables)) in words.into_iter().enumerate() {
        let word =
            syllables_to_phonemes_ipa(&syllables, &phonemicized.phonemes, &phonemicized.variety);
        if word.is_empty() {
            continue;
        }
        symbols.push(word);
        let boundary_symbols = boundary_symbols_after_word(&phonemicized.boundaries, word_index);
        if boundary_symbols.is_empty() {
            if position != last_index {
                symbols.push("|".to_string());
            }
        } else {
            symbols.extend(boundary_symbols.into_iter().map(str::to_string));
        }
    }
    (!symbols.is_empty()).then(|| symbols.join(" "))
}

fn boundary_symbols_after_word(
    boundaries: &[SpeechBoundaryToken],
    word_index: usize,
) -> Vec<&'static str> {
    let Some(boundary) = boundaries
        .iter()
        .filter(|boundary| boundary.terminal.is_some() || boundary.pause.is_some())
        .find(|boundary| boundary.after_grapheme_index == word_index)
    else {
        return Vec::new();
    };
    if let Some(terminal) = boundary.terminal {
        return match terminal {
            TerminalPunctuation::Question => vec!["↗", "?"],
            TerminalPunctuation::Period => vec!["↘", "."],
            TerminalPunctuation::Exclamation => vec!["↘", "!"],
        };
    }
    if let Some(pause) = boundary.pause {
        return match pause {
            PauseKind::Comma => vec!["→", ","],
            PauseKind::AlternativeQuestionRise => vec!["↗", ","],
        };
    }
    Vec::new()
}

fn token_word_index(features: &speaking::FeatureBundle) -> Option<usize> {
    let value = features
        .values
        .get(&speaking::FeatureId("orthography.word_index".into()))?;
    match value {
        speaking::Spec::Known(speaking::FeatureValue::Number(value))
            if value.is_finite() && *value >= 0.0 =>
        {
            Some(*value as usize)
        }
        _ => None,
    }
}

fn phone_ipa(phone: &speaking::PhoneToken) -> &str {
    match &phone.phone {
        speaking::Spec::Known(id) => id
            .as_str()
            .strip_prefix("ipa.phone.")
            .unwrap_or(id.as_str()),
        _ => "",
    }
}

fn find_phoneme_for_phone(
    phone: &speaking::PhoneToken,
    phonemes: &[speaking::PhonemeToken],
) -> Option<speaking::PhonemeId> {
    for phoneme_token in phonemes {
        for realized_phone in &phoneme_token.realized_as {
            if realized_phone.phone == phone.phone
                && realized_phone.features == phone.features
                && realized_phone.span == phone.span
            {
                if let speaking::Spec::Known(ref id) = phoneme_token.phoneme {
                    return Some(id.clone());
                }
            }
        }
    }
    None
}

fn syllables_to_phonemes_ipa(
    syllables: &[speaking::Syllable],
    phonemes: &[speaking::PhonemeToken],
    variety: &speaking::VarietyId,
) -> String {
    syllables
        .iter()
        .enumerate()
        .map(|(index, syllable)| {
            let mut text = String::new();
            let mut has_stress_mark = false;
            let stress_char = match syllable.stress {
                speaking::Spec::Known(speaking::Stress::Primary) => {
                    has_stress_mark = true;
                    Some('ˈ')
                }
                speaking::Spec::Known(speaking::Stress::Secondary) => {
                    has_stress_mark = true;
                    Some('ˌ')
                }
                _ => None,
            };
            if index > 0 && !has_stress_mark {
                text.push('.');
            }
            if let Some(c) = stress_char {
                text.push(c);
            }
            for phone in &syllable.phones {
                if let Some(phoneme_id) = find_phoneme_for_phone(phone, phonemes) {
                    let symbol =
                        speaking::phoneme_default_phone_display_symbol(&phoneme_id, variety);
                    text.push_str(&symbol);
                } else {
                    text.push_str(phone_ipa(phone));
                }
            }
            text
        })
        .collect()
}

fn synthetic_buffers(count: usize, rng: &mut StdRng) -> Vec<String> {
    let heads = [
        "Dr. Smith went home.",
        "This is the next sentence; and then the next.",
        "What happened next?",
        "Stop right there!",
        "Wait... really?",
        "\"No.\" she said.",
        "\"Are you sure?\" Mina asked.",
        "(Really?) That was the whole answer.",
        "Mr. Jones arrived after lunch.",
        "The package is ready, but the driver is late.",
        "First, open the small panel.",
        "- Bring the blue folder.",
        "I saw 3.14 written on the board.",
        "Use e.g. this example carefully.",
        "In short: the answer changed.",
        "A sudden pause — then the lamp went out.",
        "Chapter One\nThe letter arrived before breakfast.",
        "Editor's Note\nThis page was left in the archive.",
        "Appendix A\nMeasurements and notes follow.",
        "Hidden Letter",
        "Editor Notes",
        "Appendix Materials",
    ];
    let remainders = [
        " Then he slept.",
        " The next part is still streaming.",
        " and more words are coming soon.",
        "\n\nAnother paragraph starts here.",
        " \"Yes,\" she answered later.",
        "",
    ];
    (0..count)
        .map(|_| {
            let head = heads[rng.gen_range(0..heads.len())];
            let rest = remainders[rng.gen_range(0..remainders.len())];
            format!("{head}{rest}")
        })
        .collect()
}

fn exceptional_buffers() -> &'static [&'static str] {
    &[
        "Dr. Smith went home. Then he slept.",
        "Mr. Jones waited quietly for the train.",
        "Ms. Hart said e.g. this example matters.",
        "Use i.e. this case as the control.",
        "I saw 3.14 written on the board. Then I erased it.",
        "What happened next? Nobody answered.",
        "Stop right there! The guard shouted again.",
        "Wait... really? I thought that was done.",
        "\"No.\" she said. Then she closed the book.",
        "\"Are you sure?\" Mina asked. The door stayed shut.",
        "\"No,\" she said, \"not yet.\" The room went quiet.",
        "(Really?) That was the whole answer.",
        "- Bring the blue folder.\n- Leave the red folder.",
        "Chapter One\nThe letter arrived before breakfast.",
        "Editor's Note\nThis page was left in the archive.",
        "Appendix A\nMeasurements and notes follow.",
        "Chapter One",
        "Editor's Note",
        "Appendix A",
        "Hidden Letter",
        "Editor Notes",
        "Appendix Materials",
        "First line complete.\nSecond line is still arriving",
        "Prof. Adams arrived at 4:30 p.m. sharp.",
        "A. B. Carter signed the note. Then he left.",
        "The package is ready, but the driver is late.",
        "This is the next sentence; and then the next.",
        "In short: the answer changed. The stream keeps moving.",
        "A sudden pause — then the lamp went out.",
        "The signal was green; the bridge stayed closed.",
        "Pack the ledger, seal the box, and wait.",
        "No. 5 was missing from the list. Then it appeared.",
        "After the meeting, Rep. Susan Smith (D. NY.) said she didn't know what the meeting was about.",
        "After the meeting, Rep.",
        "After the meeting, Rep. Susan Smith (D.",
        "After the meeting, Rep. Susan Smith (D. NY.",
        "I think we should",
        "This final fragment should be flushed",
    ]
}

fn exceptional_repair_discrepancies() -> Vec<NaiveSeamsDiscrepancy> {
    [
        (
            "Who shot John F. Kennedy?",
            vec!["Who shot John F.", "Kennedy?"],
        ),
        (
            "Elizabeth met Mr. Darcy at Pemberley.",
            vec!["Elizabeth met Mr.", "Darcy at Pemberley."],
        ),
    ]
    .into_iter()
    .map(|(seams_sentence, naive_sentences)| NaiveSeamsDiscrepancy {
        source: "exceptional-repair".to_string(),
        seams_sentence: seams_sentence.to_string(),
        naive_sentences: naive_sentences
            .into_iter()
            .map(|sentence| sentence.to_string())
            .collect(),
    })
    .collect()
}

fn seams_sentences_from_text(raw: &str) -> Vec<String> {
    if let Ok(detector) = SentenceDetectorDialog::new() {
        if let Ok(sentences) = detector.detect_sentences_borrowed(raw) {
            return sentences
                .into_iter()
                .map(|sentence| sentence.normalize().trim().to_string())
                .filter(|sentence| !sentence.is_empty())
                .collect();
        }
    }
    Vec::new()
}

fn source_buffers_from_sentences(raw: &str, seams_sentences: &[String]) -> Vec<String> {
    let mut buffers = Vec::new();
    for (index, sentence) in seams_sentences.iter().enumerate() {
        let mut buffer = sentence.clone();
        if let Some(next) = seams_sentences.get(index + 1) {
            buffer.push(' ');
            buffer.push_str(next);
        }
        buffers.push(buffer);
    }
    buffers.extend(
        raw.split("\n\n")
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(str::to_string),
    );
    buffers.sort();
    buffers.dedup();
    buffers
}

pub fn naive_split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0usize;
    for (index, ch) in text.char_indices() {
        if matches!(ch, '.' | '?' | '!') {
            let end = index + ch.len_utf8();
            let sentence = normalize_sentence(&text[start..end]);
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            start = end;
        }
    }
    let tail = normalize_sentence(&text[start..]);
    if !tail.is_empty() {
        sentences.push(tail);
    }
    sentences
}

fn build_naive_seams_discrepancies(
    seams_sentences: &[String],
    source: &str,
    max_per_file: usize,
) -> Vec<NaiveSeamsDiscrepancy> {
    let mut discrepancies = Vec::new();
    for seams_sentence in seams_sentences {
        if discrepancies.len() >= max_per_file {
            break;
        }
        let naive_sentences = naive_split_sentences(seams_sentence);
        if naive_sentences.len() <= 1 {
            continue;
        }
        discrepancies.push(NaiveSeamsDiscrepancy {
            source: source.to_string(),
            seams_sentence: seams_sentence.clone(),
            naive_sentences,
        });
    }
    discrepancies
}

fn normalize_sentence(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn resolve_source_files_with_progress(
    out: &Path,
    config: &Head2PhonesConfig,
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

fn random_complete_buffers(
    buffer: &str,
    head_end_byte: usize,
    config: &Head2PhonesConfig,
    rng: &mut StdRng,
) -> Vec<String> {
    let rest = &buffer[head_end_byte..];
    let rest_graphemes = rest.graphemes(true).collect::<Vec<_>>();
    if rest_graphemes.is_empty() || config.random_cuts_per_buffer == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for _ in 0..config.random_cuts_per_buffer {
        let keep = rng.gen_range(0..=rest_graphemes.len());
        let suffix = rest_graphemes[..keep].concat();
        out.push(format!("{}{}", &buffer[..head_end_byte], suffix));
    }
    out
}

fn no_head_prefixes(head: &str, config: &Head2PhonesConfig, rng: &mut StdRng) -> Vec<String> {
    let graphemes = head.graphemes(true).collect::<Vec<_>>();
    if graphemes.len() < 6 {
        return Vec::new();
    }
    let mut cuts = vec![graphemes.len() / 3, graphemes.len() * 2 / 3];
    for _ in 0..config.no_head_cuts_per_head {
        cuts.push(rng.gen_range(1..graphemes.len()));
    }
    for (index, grapheme) in graphemes.iter().enumerate() {
        if grapheme.chars().any(|ch| {
            matches!(
                ch,
                '.' | '!' | '?' | ';' | ':' | ',' | '"' | '\'' | ')' | ']' | '}' | '\n'
            )
        }) {
            if index > 0 {
                cuts.push(index);
            }
        }
    }
    cuts.sort_unstable();
    cuts.dedup();
    cuts.into_iter()
        .filter(|&n| n > 0 && n < graphemes.len())
        .map(|n| graphemes[..n].concat())
        .collect()
}

fn write_jsonl_with_progress<T: Serialize>(
    path: &Path,
    rows: &[T],
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<()> {
    write_jsonl_atomic(path, rows)?;
    progress(PrepareProgress::Write {
        path: path.display().to_string(),
        rows: rows.len(),
    });
    Ok(())
}

fn dataset_readme(
    config: &Head2PhonesConfig,
    train: &[Head2PhonesTrainingExample],
    valid: &[Head2PhonesTrainingExample],
    test: &[Head2PhonesTrainingExample],
    naive_seams_discrepancies: usize,
) -> String {
    format!(
        "# head2phones {}\n\nTrain/valid/test rows: {}/{}/{}.\nNaive-vs-seams discrepancy rows: {} in `naive_seams_discrepancies.jsonl`.\n\nOutputs are exactly `{}`, `{}` broad IPA phone text `{}` plus `{}`, or `{}` repair rows. Repair rows start with `{}` confidence, then `{}`, `{}` with a Unicode grapheme-cluster rollback distance, corrected phones, and a corrected split offset. Phone text is serialized from speaking IR and may include word boundaries, punctuation, stress, and intonation markers; backend-specific downcasting happens only at synthesis time.\n",
        config.dataset_id,
        train.len(),
        valid.len(),
        test.len(),
        naive_seams_discrepancies,
        NO_HEAD,
        PHONES_OPEN,
        PHONES_CLOSE,
        SPLIT_AFTER,
        ERROR_REPAIR,
        CONFIDENCE,
        ERROR_REPAIR,
        ROLLBACK_GRAPHEMES
    )
}

pub fn write_scaffold_model(out: &Path, config: &Head2PhonesConfig) -> Result<()> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    fs::write(out.join("model.bin"), b"head2phones-scaffold\n")?;
    fs::write(
        out.join("model_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    fs::write(
        out.join("train_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    fs::write(
        out.join("train_state.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "status": "scaffold",
            "epochs": 0
        }))?,
    )?;
    write_manifest(
        out,
        &ModelArtifactManifest::new(FAMILY, ARCHITECTURE, &config.dataset_id)
            .with_task("head-chunk-to-phones"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize)]
    struct ExceptionalCaseCatalog {
        cases: Vec<ExceptionalCase>,
    }

    #[derive(Debug, Deserialize)]
    struct ExceptionalCase {
        id: String,
        kind: String,
        input: String,
        expected_head: Option<String>,
        #[serde(default)]
        output_contains: Vec<String>,
        #[serde(default)]
        output_not_contains: Vec<String>,
    }

    #[test]
    fn counts_graphemes_for_split_offsets() {
        let text = "Cafe\u{301} now. Later";
        let head = first_complete_head(text).expect("complete head");
        assert_eq!(text[..head.end_byte].graphemes(true).count(), 9);
        let (prefix, suffix) = grapheme_split(text, 9);
        assert_eq!(prefix, "Cafe\u{301} now.");
        assert_eq!(suffix, " Later");
    }

    #[test]
    fn does_not_split_common_abbreviation() {
        let text = "Dr. Smith went home. Then he slept.";
        let head = first_complete_head(text).expect("complete head");
        assert_eq!(&text[..head.end_byte], "Dr. Smith went home.");
    }

    #[test]
    fn partial_prefix_has_no_head() {
        assert!(first_complete_head("I think we should").is_none());
    }

    #[test]
    fn decimal_is_not_a_head_boundary() {
        let text = "I saw 3.14 written down.";
        let head = first_complete_head(text).expect("complete head");
        assert_eq!(&text[..head.end_byte], text);
    }

    #[test]
    fn decimal_head_phones_spell_fractional_digits() {
        let symbols = speech_symbols_for_text("I saw 3.14 written on the board.").expect("symbols");
        assert!(symbols.contains("pɔɪnt"), "{symbols}");
        assert!(symbols.contains("wən"), "{symbols}");
        assert!(symbols.contains("fɔɹ"), "{symbols}");
        assert!(!symbols.contains("fɔɹˈtiːn"), "{symbols}");
    }

    #[test]
    fn questions_exclamations_and_closing_punctuation_are_complete_heads() {
        for (text, expected) in [
            (
                "What happened next? Nobody answered.",
                "What happened next?",
            ),
            (
                "Stop right there! The guard shouted again.",
                "Stop right there!",
            ),
            ("\"Are you sure?\" Mina asked.", "\"Are you sure?\""),
            ("(Really?) That was the whole answer.", "(Really?)"),
        ] {
            let head = first_complete_head(text).expect("complete head");
            assert_eq!(&text[..head.end_byte], expected);
        }
    }

    #[test]
    fn punctuation_boundaries_cover_colons_semicolons_commas_dashes_and_titles() {
        for (text, expected) in [
            ("In short: the answer changed.", "In short:"),
            (
                "The signal was green; the bridge stayed closed.",
                "The signal was green;",
            ),
            (
                "Pack the ledger, seal the box, and wait.",
                "Pack the ledger, seal the box,",
            ),
            (
                "A sudden pause — then the lamp went out.",
                "A sudden pause —",
            ),
            (
                "Chapter One\nThe letter arrived before breakfast.",
                "Chapter One\n",
            ),
        ] {
            let head = first_complete_head(text).expect("complete head");
            assert_eq!(&text[..head.end_byte], expected);
        }
    }

    #[test]
    fn dotted_abbreviations_do_not_split_the_head() {
        let text = "Use e.g. this example carefully. Then continue.";
        let head = first_complete_head(text).expect("complete head");
        assert_eq!(&text[..head.end_byte], "Use e.g. this example carefully.");

        let text = "Use i.e. this case as the control. Then continue.";
        let head = first_complete_head(text).expect("complete head");
        assert_eq!(&text[..head.end_byte], "Use i.e. this case as the control.");
    }

    #[test]
    fn naive_splitter_demonstrates_erroneous_abbreviation_cases() {
        let text = "Dr. Smith saw No. 5 at 3.14 p.m. before leaving.";
        let seams_sentences = seams_sentences_from_text(text);
        assert_eq!(seams_sentences, vec![text.to_string()]);

        let discrepancies = build_naive_seams_discrepancies(&seams_sentences, "test", 8);
        assert_eq!(discrepancies.len(), 1);
        assert_eq!(discrepancies[0].seams_sentence, text);
        assert!(discrepancies[0]
            .naive_sentences
            .iter()
            .any(|sentence| sentence == "Dr."));
        assert!(discrepancies[0]
            .naive_sentences
            .iter()
            .any(|sentence| sentence == "14 p."));
    }

    #[test]
    fn discrepancy_mining_is_bounded_per_file() {
        let seams_sentences = vec![
            "Dr. Smith went home.".to_string(),
            "Mr. Jones waited quietly.".to_string(),
        ];
        let discrepancies = build_naive_seams_discrepancies(&seams_sentences, "test", 1);
        assert_eq!(discrepancies.len(), 1);
    }

    #[test]
    fn number_abbreviation_uses_following_context() {
        let text = "No. 5 was missing from the list. Then it appeared.";
        let head = first_complete_head(text).expect("complete head");
        assert_eq!(&text[..head.end_byte], "No. 5 was missing from the list.");
    }

    #[test]
    fn representative_party_state_cuts_stay_incomplete_until_sentence_end() {
        assert!(first_complete_head("After the meeting, Rep.").is_none());
        assert!(first_complete_head("After the meeting, Rep. Susan Smith (D.").is_none());
        assert!(first_complete_head("After the meeting, Rep. Susan Smith (D. NY.").is_none());

        let text = "After the meeting, Rep. Susan Smith (D. NY.) said she didn't know what the meeting was about.";
        let head = first_complete_head(text).expect("complete head");
        assert_eq!(&text[..head.end_byte], text);
    }

    #[test]
    fn end_of_text_flushes_incomplete_prefix() {
        let row =
            flush_example_for_prefix("I think we should", "test", TrainingRowSource::Exceptional)
                .expect("flush row");
        assert_eq!(row.input, format!("I think we should{END_OF_TEXT}"));
        assert!(row.output.starts_with(PHONES_OPEN));
        assert_eq!(
            row.split_after,
            Some("I think we should".graphemes(true).count())
        );
    }

    #[test]
    fn end_of_text_does_not_flush_nonterminal_abbreviation_prefix() {
        for prefix in [
            "After the meeting, Rep.",
            "After the meeting, Rep. Susan Smith (D.",
            "After the meeting, Rep. Susan Smith (D. NY.",
        ] {
            assert!(
                flush_example_for_prefix(prefix, "test", TrainingRowSource::Exceptional).is_none(),
                "{prefix}"
            );
        }
    }

    #[test]
    fn end_of_text_flushes_title_like_non_sentences() {
        for title in ["Hidden Letter", "Editor Notes", "Appendix Materials"] {
            let row = flush_example_for_prefix(title, "test", TrainingRowSource::Exceptional)
                .expect("flush row");
            assert_eq!(row.input, format!("{title}{END_OF_TEXT}"));
            assert_eq!(row.head.as_deref(), Some(title));
            assert_eq!(row.split_after, Some(title.graphemes(true).count()));
            assert!(row.output.starts_with(PHONES_OPEN));
        }
    }

    #[test]
    fn ipa_phones_include_boundaries_and_terminal_prosody() {
        let symbols = speech_symbols_for_text("Dr. Smith went home.").expect("symbols");
        assert!(symbols.contains("|"));
        assert!(symbols.contains("d") || symbols.contains("ɾ"));
        assert!(symbols.contains("ɑ") || symbols.contains("ɔ"));
        assert!(!symbols.split_whitespace().any(|token| token == "DH"));
        assert!(!symbols.split_whitespace().any(|token| token == "OW"));
        assert!(!symbols.contains('˭'));
        assert!(symbols.contains("."));
        assert!(symbols.contains("↘"));
    }

    #[test]
    fn broad_ipa_does_not_emit_narrow_allophone_marks() {
        let symbols = speech_symbols_for_text("Mr. Jones waited.").expect("symbols");
        assert!(symbols.contains("mɪstɚ") || symbols.contains("mɪ.stɚ"));
        assert!(!symbols.contains("t˭"));
        assert!(!symbols.contains('˭'));
    }

    #[test]
    fn broad_ipa_splits_intervocalic_r_colored_schwa_by_maximum_onset() {
        let symbols = speech_symbols_for_text("arrived").expect("symbols");
        assert!(symbols.contains("əˈɹaɪvd"), "{symbols}");
        assert!(!symbols.contains("ɚˈaɪvd"), "{symbols}");
    }

    #[test]
    fn loadstone_is_pronounced_like_lodestone() {
        let symbols =
            speech_symbols_for_text("The Loadstone Rock was drawing him.").expect("symbols");
        assert!(symbols.contains("ˈloʊdˌstoʊn"), "{symbols}");
        assert!(!symbols.contains("ˈlʌəd.stə.nɪ"), "{symbols}");
        assert!(!symbols.contains("ˈləəd.stə.nɪ"), "{symbols}");
    }

    #[test]
    fn documented_exceptional_cases_are_covered() {
        let catalog: ExceptionalCaseCatalog = serde_json::from_str(include_str!(
            "../../../docs/head2phones-exceptional-cases.json"
        ))
        .expect("exceptional case catalog should parse");
        assert!(!catalog.cases.is_empty());

        for case in catalog.cases {
            match case.kind.as_str() {
                "head" => {
                    let head = first_complete_head(&case.input)
                        .map(|head| case.input[..head.end_byte].to_string());
                    assert_eq!(head, case.expected_head, "{}", case.id);
                }
                "phones" => {
                    let symbols = speech_symbols_for_text(&case.input)
                        .unwrap_or_else(|| panic!("{} should produce phones", case.id));
                    assert_contains_all(&case.id, &symbols, &case.output_contains);
                    assert_contains_none(&case.id, &symbols, &case.output_not_contains);
                }
                "flush" => {
                    let row = flush_example_for_prefix(
                        &case.input,
                        "documented-exceptional-case",
                        TrainingRowSource::Exceptional,
                    )
                    .unwrap_or_else(|| panic!("{} should flush", case.id));
                    assert_eq!(row.head, case.expected_head, "{}", case.id);
                    assert_contains_all(&case.id, &row.output, &case.output_contains);
                    assert_contains_none(&case.id, &row.output, &case.output_not_contains);
                }
                "repair" => {
                    let discrepancy = NaiveSeamsDiscrepancy {
                        source: "documented-exceptional-case".to_string(),
                        seams_sentence: case
                            .expected_head
                            .clone()
                            .unwrap_or_else(|| case.input.clone()),
                        naive_sentences: naive_split_sentences(&case.input),
                    };
                    let row =
                        repair_example_for_discrepancy(&discrepancy, &Head2PhonesConfig::default())
                            .unwrap_or_else(|| panic!("{} should produce a repair row", case.id));
                    assert_eq!(row.head, case.expected_head, "{}", case.id);
                    assert_contains_all(&case.id, &row.output, &case.output_contains);
                    assert_contains_none(&case.id, &row.output, &case.output_not_contains);
                }
                other => panic!("{} has unsupported kind {other}", case.id),
            }
        }
    }

    #[test]
    fn repair_rows_signal_low_confidence_error_and_rollback_distance() {
        let discrepancy = NaiveSeamsDiscrepancy {
            source: "test".to_string(),
            seams_sentence: "Who shot John F. Kennedy?".to_string(),
            naive_sentences: vec!["Who shot John F.".to_string(), "Kennedy?".to_string()],
        };
        let row = repair_example_for_discrepancy(&discrepancy, &Head2PhonesConfig::default())
            .expect("repair row");

        assert_eq!(row.row_source, TrainingRowSource::Repair);
        assert_eq!(row.head.as_deref(), Some("Who shot John F. Kennedy?"));
        assert!(row
            .output
            .contains(&format!("{CONFIDENCE} {CONFIDENCE_LOW}")));
        assert!(row.output.contains(ERROR_REPAIR));
        assert!(row.output.contains(&format!(
            "{ROLLBACK_GRAPHEMES} {}",
            "Who shot John F.".graphemes(true).count()
        )));
        assert!(row.output.contains(PHONES_OPEN));
        assert!(row.output.contains(SPLIT_AFTER));
    }

    #[test]
    fn random_cuts_create_complete_and_no_head_rows() {
        let config = Head2PhonesConfig {
            synthetic_buffers: 0,
            random_cuts_per_buffer: 4,
            no_head_cuts_per_head: 4,
            ..Head2PhonesConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(3);
        let mut rows = Vec::new();
        add_examples_for_buffer(
            "This is ready. The remainder is still arriving now.",
            "test",
            TrainingRowSource::Synthetic,
            &config,
            &mut rng,
            &mut rows,
        )
        .expect("add examples");
        assert!(rows.iter().any(|row| row.head.is_some()));
        assert!(rows.iter().any(|row| row.output == NO_HEAD));
        assert!(rows
            .iter()
            .any(|row| row.row_source == TrainingRowSource::RandomCut));
    }

    #[test]
    fn prepare_reuses_checkpoints_and_does_not_touch_legacy_parts() {
        let out = tempfile_path("head2phones-resume");
        let _ = fs::remove_dir_all(&out);
        fs::create_dir_all(&out).expect("create temp dataset dir");
        let legacy_part = out.join("examples.jsonl.part");
        fs::write(&legacy_part, "legacy partial should remain untouched\n")
            .expect("write legacy partial");
        let config = Head2PhonesConfig {
            dataset_id: "resume-test".to_string(),
            include_default_gutenberg: false,
            source_paths: Vec::new(),
            include_synthetic: true,
            synthetic_buffers: 32,
            random_cuts_per_buffer: 1,
            no_head_cuts_per_head: 1,
            include_exceptional: false,
            include_naive_seams_discrepancies: false,
            ..Head2PhonesConfig::default()
        };

        let first = prepare_dataset(&out, &config).expect("first prepare");
        let checkpoint = out
            .join("prepare-checkpoints")
            .join("000-synthetic.manifest.json");
        assert!(checkpoint.exists());
        assert!(out.join("examples.jsonl").exists());
        assert_eq!(
            fs::read_to_string(&legacy_part).expect("read legacy partial"),
            "legacy partial should remain untouched\n"
        );

        for path in [
            out.join("examples.jsonl"),
            out.join("train.jsonl"),
            out.join("valid.jsonl"),
            out.join("test.jsonl"),
            out.join("vocab.json"),
        ] {
            fs::remove_file(path).expect("remove final output");
        }

        let second = prepare_dataset(&out, &config).expect("resume prepare");
        assert_eq!(first.complete_examples, second.complete_examples);
        assert_eq!(first.no_head_examples, second.no_head_examples);
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

    fn assert_contains_all(case_id: &str, haystack: &str, needles: &[String]) {
        for needle in needles {
            assert!(
                haystack.contains(needle),
                "{case_id}: expected `{haystack}` to contain `{needle}`"
            );
        }
    }

    fn assert_contains_none(case_id: &str, haystack: &str, needles: &[String]) {
        for needle in needles {
            assert!(
                !haystack.contains(needle),
                "{case_id}: expected `{haystack}` not to contain `{needle}`"
            );
        }
    }
}
