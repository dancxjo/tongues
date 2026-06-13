//! Sentence-boundary seq2seq model-family data preparation.
//!
//! This family trains a cursor-time model to decide whether the current text
//! prefix can be emitted as a complete sentence, should continue buffering, or
//! needs to repair a previously emitted boundary.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use seams::SentenceDetectorDialog;
use serde::{Deserialize, Serialize};
use speaking::segment::TerminalPunctuation;
use speaking::syntax::{HeuristicLinkGrammarParser, LinkGrammarParser, SentenceSyntaxAnalysis};
use tongues_core::{Vocab, BOS_ID, EOS_ID};
use tongues_data::Seq2SeqExample;
use tongues_neural::{write_manifest, ModelArtifactManifest};

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
    let mut sentences = Vec::new();
    let mut correction_examples = Vec::new();
    let sentences_part_path = out.join("sentences.jsonl.part");
    let discrepancies_part_path = out.join("naive_discrepancies.jsonl.part");
    let examples_part_path = out.join("examples.jsonl.part");
    let mut sentences_part = BufWriter::new(
        File::create(&sentences_part_path)
            .with_context(|| format!("creating {}", sentences_part_path.display()))?,
    );
    let mut discrepancies_part = BufWriter::new(
        File::create(&discrepancies_part_path)
            .with_context(|| format!("creating {}", discrepancies_part_path.display()))?,
    );
    let detector = SentenceDetectorDialog::new().context("initializing seams detector")?;
    for (file_index, path) in files.iter().enumerate() {
        progress(PrepareProgress::Stage {
            message: format!("Detecting sentence boundaries in {}", path.display()),
        });
        let raw =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut file_sentences = Vec::new();
        for detected in detector
            .detect_sentences_borrowed(&raw)
            .with_context(|| format!("detecting sentence boundaries in {}", path.display()))?
        {
            let sentence = normalize_sentence(&detected.normalize(), config.lowercase);
            if sentence.chars().count() >= config.min_sentence_chars
                && sentence.chars().count() <= config.max_sentence_chars
            {
                file_sentences.push(sentence);
            }
        }
        if config.include_naive_discrepancies {
            let file_corrections = build_naive_discrepancy_examples(
                &file_sentences,
                &path.display().to_string(),
                config,
            );
            for example in &file_corrections {
                writeln!(discrepancies_part, "{}", serde_json::to_string(example)?)?;
            }
            correction_examples.extend(file_corrections);
        }
        for sentence in &file_sentences {
            writeln!(
                sentences_part,
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "sentence": sentence,
                    "source": path.display().to_string()
                }))?
            )?;
        }
        sentences.extend(
            file_sentences
                .into_iter()
                .map(|sentence| (sentence, path.display().to_string())),
        );
        progress(PrepareProgress::Detect {
            path: path.display().to_string(),
            files_done: file_index + 1,
            files_total: files.len(),
            sentences: sentences.len(),
            naive_discrepancies: correction_examples.len(),
        });
    }
    sentences_part
        .flush()
        .with_context(|| format!("flushing {}", sentences_part_path.display()))?;
    discrepancies_part
        .flush()
        .with_context(|| format!("flushing {}", discrepancies_part_path.display()))?;
    drop(sentences_part);
    drop(discrepancies_part);

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
    let mut examples = build_boundary_examples(&sentences, config);
    examples.extend(correction_examples.clone());
    progress(PrepareProgress::Build {
        sentences: sentences.len(),
        examples: examples.len(),
    });
    anyhow::ensure!(
        !examples.is_empty(),
        "no sentence-parser training examples were built from {} detected sentences",
        sentences.len()
    );
    write_jsonl_with_progress(&examples_part_path, &examples, &mut progress)?;
    let mut shuffled = examples;
    shuffled.shuffle(&mut StdRng::seed_from_u64(config.seed));
    let n = shuffled.len();
    let train_end = (n as f64 * config.train_frac).round() as usize;
    let valid_end = (train_end + (n as f64 * config.valid_frac).round() as usize).min(n);
    let train = shuffled[..train_end.min(n)].to_vec();
    let valid = shuffled[train_end.min(n)..valid_end].to_vec();
    let test = shuffled[valid_end..].to_vec();

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
    fs::remove_file(&sentences_part_path)
        .with_context(|| format!("removing {}", sentences_part_path.display()))?;
    fs::remove_file(&examples_part_path)
        .with_context(|| format!("removing {}", examples_part_path.display()))?;
    Ok(PrepareReport {
        source_files: files.len(),
        detected_sentences: sentences.len(),
        naive_discrepancy_examples,
        train_examples: train.len(),
        valid_examples: valid.len(),
        test_examples: test.len(),
    })
}

pub fn write_scaffold_model(out: &Path, config: &SentenceParserConfig) -> Result<()> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let tokenizer = TokenizerSpec {
        lowercase: config.lowercase,
        ..TokenizerSpec::default()
    };
    fs::write(out.join("model.bin"), b"sentence-parser-scaffold\n")?;
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
    fs::write(
        out.join("tokenizer.json"),
        serde_json::to_string_pretty(&tokenizer)?,
    )?;
    fs::write(
        out.join("label_schema.json"),
        serde_json::to_string_pretty(&LabelSchema::default())?,
    )?;
    write_manifest(
        out,
        &ModelArtifactManifest::new(FAMILY, ARCHITECTURE, &config.dataset_id)
            .with_task("cursor-boundary"),
    )
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
    HeuristicLinkGrammarParser.parse(&words, terminal)
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
    let part_path = path.with_extension(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| format!("{extension}.part"))
            .unwrap_or_else(|| "part".to_string()),
    );
    progress(PrepareProgress::Stage {
        message: format!("Writing {} rows to {}", rows.len(), part_path.display()),
    });
    let mut file = BufWriter::new(
        File::create(&part_path).with_context(|| format!("creating {}", part_path.display()))?,
    );
    for row in rows {
        writeln!(file, "{}", serde_json::to_string(row)?)?;
    }
    file.flush()
        .with_context(|| format!("flushing {}", part_path.display()))?;
    fs::rename(&part_path, path)
        .with_context(|| format!("moving {} to {}", part_path.display(), path.display()))?;
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
        "# Sentence boundary dataset\n\nDataset: `{}`\n\nSources: {} Project Gutenberg-style text files\nDetected sentences: {}\nTraining rows: {}\nNaive-discrepancy correction rows: {}\n\nInput shape:\n\n```text\n{}{}<previous sentence>{}<cursor prefix>\n```\n\nTargets:\n\n```text\n{}<sentence>\\n\n{}\n{}<tail fragment>\n{}<repaired sentence>\n```\n\nThe source intentionally includes only the previously parsed sentence and current cursor prefix. No following sentence is provided.\n",
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
}
