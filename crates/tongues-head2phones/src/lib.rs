//! Head-chunk-to-phones seq2seq data preparation.
//!
//! The model sees a raw rolling UTF-8 English buffer and emits either
//! `<NO_HEAD>` or phones plus the Unicode grapheme-cluster split offset for
//! the first complete TTS-speakable head chunk.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use seams::SentenceDetectorDialog;
use serde::{Deserialize, Serialize};
use speaking::{
    EnglishPhonemicizer, PauseKind, PhonemicizeOutput, PhonemicizeRequest, Phonemicizer,
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
    pub exceptional_examples: usize,
    pub train_examples: usize,
    pub valid_examples: usize,
    pub test_examples: usize,
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

    let mut rng = StdRng::seed_from_u64(config.seed);
    let examples_part_path = out.join("examples.jsonl.part");
    let discrepancies_part_path = out.join("naive_seams_discrepancies.jsonl.part");
    protect_nonempty_partial(&examples_part_path)?;
    protect_nonempty_partial(&discrepancies_part_path)?;
    let mut examples_part = BufWriter::new(
        File::create(&examples_part_path)
            .with_context(|| format!("creating {}", examples_part_path.display()))?,
    );
    let mut discrepancies_part = BufWriter::new(
        File::create(&discrepancies_part_path)
            .with_context(|| format!("creating {}", discrepancies_part_path.display()))?,
    );
    let mut examples = Vec::new();
    let mut naive_seams_discrepancies = Vec::new();
    let mut source_buffers = 0usize;

    let mut temporary_parts = vec![examples_part_path.clone(), discrepancies_part_path.clone()];

    if config.include_synthetic {
        let synthetic_path = out.join("synthetic_buffers.jsonl.part");
        protect_nonempty_partial(&synthetic_path)?;
        temporary_parts.push(synthetic_path.clone());
        let synthetic = synthetic_buffers(config.synthetic_buffers, &mut rng);
        let mut writer = BufWriter::new(
            File::create(&synthetic_path)
                .with_context(|| format!("creating {}", synthetic_path.display()))?,
        );
        for buffer in &synthetic {
            source_buffers += 1;
            writeln!(writer, "{}", serde_json::to_string(buffer)?)?;
            add_examples_for_buffer(
                buffer,
                "synthetic",
                TrainingRowSource::Synthetic,
                config,
                &mut rng,
                &mut examples,
                &mut examples_part,
            )?;
        }
        writer
            .flush()
            .with_context(|| format!("flushing {}", synthetic_path.display()))?;
        progress(PrepareProgress::Synthesize {
            path: synthetic_path.display().to_string(),
            buffers: synthetic.len(),
        });
    }

    if config.include_exceptional {
        for buffer in exceptional_buffers() {
            source_buffers += 1;
            add_examples_for_buffer(
                buffer,
                "exceptional",
                TrainingRowSource::Exceptional,
                config,
                &mut rng,
                &mut examples,
                &mut examples_part,
            )?;
        }
    }

    let source_files = resolve_source_files_with_progress(out, config, &mut progress)?;
    for path in &source_files {
        let raw =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let seams_sentences = seams_sentences_from_text(&raw);
        let file_discrepancies = if config.include_naive_seams_discrepancies {
            build_naive_seams_discrepancies(
                &seams_sentences,
                &path.display().to_string(),
                config.max_naive_seams_discrepancies_per_file,
            )
        } else {
            Vec::new()
        };
        for discrepancy in &file_discrepancies {
            writeln!(
                discrepancies_part,
                "{}",
                serde_json::to_string(discrepancy)?
            )?;
        }
        naive_seams_discrepancies.extend(file_discrepancies);
        let buffers = source_buffers_from_sentences(&raw, &seams_sentences);
        for buffer in &buffers {
            source_buffers += 1;
            add_examples_for_buffer(
                buffer,
                &path.display().to_string(),
                TrainingRowSource::SourceText,
                config,
                &mut rng,
                &mut examples,
                &mut examples_part,
            )?;
        }
        progress(PrepareProgress::Read {
            path: path.display().to_string(),
            buffers: buffers.len(),
            naive_seams_discrepancies: naive_seams_discrepancies.len(),
        });
    }

    examples_part
        .flush()
        .with_context(|| format!("flushing {}", examples_part_path.display()))?;
    discrepancies_part
        .flush()
        .with_context(|| format!("flushing {}", discrepancies_part_path.display()))?;
    drop(examples_part);
    drop(discrepancies_part);

    let complete_examples = examples
        .iter()
        .filter(|example| example.head.is_some())
        .count();
    let no_head_examples = examples.len().saturating_sub(complete_examples);
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

    examples.shuffle(&mut rng);
    let n = examples.len();
    let train_end = (n as f64 * config.train_frac).round() as usize;
    let valid_end = (train_end + (n as f64 * config.valid_frac).round() as usize).min(n);
    let train = examples[..train_end.min(n)].to_vec();
    let valid = examples[train_end.min(n)..valid_end].to_vec();
    let test = examples[valid_end..].to_vec();

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
    for part in temporary_parts {
        if part.exists() {
            fs::remove_file(&part).with_context(|| format!("removing {}", part.display()))?;
        }
    }

    Ok(PrepareReport {
        source_buffers,
        naive_seams_discrepancies: naive_seams_discrepancies.len(),
        complete_examples,
        no_head_examples,
        exceptional_examples,
        train_examples: train.len(),
        valid_examples: valid.len(),
        test_examples: test.len(),
    })
}

fn protect_nonempty_partial(path: &Path) -> Result<()> {
    let Some(metadata) = fs::metadata(path).ok() else {
        return Ok(());
    };
    anyhow::ensure!(
        metadata.len() == 0,
        "refusing to overwrite nonempty partial artifact {}; move, remove, or resume it explicitly",
        path.display()
    );
    Ok(())
}

fn add_examples_for_buffer(
    buffer: &str,
    source: &str,
    row_source: TrainingRowSource,
    config: &Head2PhonesConfig,
    rng: &mut StdRng,
    examples: &mut Vec<Head2PhonesTrainingExample>,
    examples_part: &mut BufWriter<File>,
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
                writeln!(examples_part, "{}", serde_json::to_string(&row)?)?;
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
                writeln!(examples_part, "{}", serde_json::to_string(&row)?)?;
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
            writeln!(examples_part, "{}", serde_json::to_string(&row)?)?;
            examples.push(row);

            if prefix_ends_at_boundary_in_head(&prefix, &head_text) {
                if let Some(flush_row) = flush_example_for_prefix(&prefix, source, row_source) {
                    writeln!(examples_part, "{}", serde_json::to_string(&flush_row)?)?;
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
        writeln!(examples_part, "{}", serde_json::to_string(&row)?)?;
        examples.push(row);

        if let Some(flush_row) = flush_example_for_prefix(buffer.trim(), source, row_source) {
            writeln!(examples_part, "{}", serde_json::to_string(&flush_row)?)?;
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

fn prefix_ends_at_boundary_in_head(prefix: &str, head: &str) -> bool {
    let Some(rest) = head.strip_prefix(prefix) else {
        return false;
    };
    rest.chars()
        .next()
        .is_none_or(|ch| ch.is_whitespace() || !ch.is_alphanumeric())
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
    let phonemicizer = EnglishPhonemicizer;
    let phonemicized = phonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: text.to_string(),
            variety: VarietyId("en-US".to_string()),
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
    let part = path.with_extension(format!(
        "{}part",
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| format!("{ext}."))
            .unwrap_or_default()
    ));
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
    fs::rename(&part, path).with_context(|| {
        format!(
            "renaming completed JSONL {} -> {}",
            part.display(),
            path.display()
        )
    })?;
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
        "# head2phones {}\n\nTrain/valid/test rows: {}/{}/{}.\nNaive-vs-seams discrepancy rows: {} in `naive_seams_discrepancies.jsonl`.\n\nOutputs are exactly `{}` or `{}` broad IPA phone text `{}` plus `{}` and a Unicode grapheme-cluster offset. Phone text is serialized from speaking IR and may include word boundaries, punctuation, stress, and intonation markers; backend-specific downcasting happens only at synthesis time.\n",
        config.dataset_id,
        train.len(),
        valid.len(),
        test.len(),
        naive_seams_discrepancies,
        NO_HEAD,
        PHONES_OPEN,
        PHONES_CLOSE,
        SPLIT_AFTER
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
    fn random_cuts_create_complete_and_no_head_rows() {
        let config = Head2PhonesConfig {
            synthetic_buffers: 0,
            random_cuts_per_buffer: 4,
            no_head_cuts_per_head: 4,
            ..Head2PhonesConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(3);
        let mut rows = Vec::new();
        let temp = tempfile_path("head2phones-random-cuts.jsonl.part");
        let mut writer = BufWriter::new(File::create(&temp).expect("create temp jsonl"));
        add_examples_for_buffer(
            "This is ready. The remainder is still arriving now.",
            "test",
            TrainingRowSource::Synthetic,
            &config,
            &mut rng,
            &mut rows,
            &mut writer,
        )
        .expect("add examples");
        drop(writer);
        let _ = fs::remove_file(temp);
        assert!(rows.iter().any(|row| row.head.is_some()));
        assert!(rows.iter().any(|row| row.output == NO_HEAD));
        assert!(rows
            .iter()
            .any(|row| row.row_source == TrainingRowSource::RandomCut));
    }

    fn tempfile_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("{}-{name}", std::process::id()))
    }
}
