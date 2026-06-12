//! Sentence-boundary seq2seq model-family data preparation.
//!
//! This family trains a cursor-time model to decide whether the current text
//! prefix can be emitted as a complete sentence, should continue buffering, or
//! needs to repair a previously emitted boundary.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use seams::SentenceDetectorDialog;
use serde::{Deserialize, Serialize};
use speech::segment::TerminalPunctuation;
use speech::syntax::{HeuristicLinkGrammarParser, LinkGrammarParser, SentenceSyntaxAnalysis};
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SentenceParserConfig {
    pub dataset_id: String,
    pub lowercase: bool,
    #[serde(default)]
    pub source_paths: Vec<PathBuf>,
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
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let files = discover_source_files(&config.source_paths)?;
    let mut sentences = Vec::new();
    let mut correction_examples = Vec::new();
    let detector = SentenceDetectorDialog::new().context("initializing seams detector")?;
    for path in &files {
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
            correction_examples.extend(build_naive_discrepancy_examples(
                &raw,
                &file_sentences,
                &path.display().to_string(),
                config,
            ));
        }
        sentences.extend(
            file_sentences
                .into_iter()
                .map(|sentence| (sentence, path.display().to_string())),
        );
    }

    let naive_discrepancy_examples = correction_examples.len();
    let mut examples = build_boundary_examples(&sentences, config);
    examples.extend(correction_examples.clone());
    let mut shuffled = examples;
    shuffled.shuffle(&mut StdRng::seed_from_u64(config.seed));
    let n = shuffled.len();
    let train_end = (n as f64 * config.train_frac).round() as usize;
    let valid_end = (train_end + (n as f64 * config.valid_frac).round() as usize).min(n);
    let train = shuffled[..train_end.min(n)].to_vec();
    let valid = shuffled[train_end.min(n)..valid_end].to_vec();
    let test = shuffled[valid_end..].to_vec();

    write_jsonl(&out.join("train.jsonl"), &train)?;
    write_jsonl(&out.join("valid.jsonl"), &valid)?;
    write_jsonl(&out.join("test.jsonl"), &test)?;
    write_jsonl(&out.join("naive_discrepancies.jsonl"), &correction_examples)?;
    let vocab = build_vocab([&train[..], &valid[..], &test[..]].concat().as_slice());
    fs::write(
        out.join("vocab.json"),
        serde_json::to_string_pretty(&vocab)?,
    )?;
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
    raw: &str,
    seams_sentences: &[String],
    source: &str,
    config: &SentenceParserConfig,
) -> Vec<BoundaryTrainingExample> {
    let naive = naive_split_sentences(raw, config.lowercase);
    let mut examples = Vec::new();
    let mut naive_index = 0usize;
    for seams_sentence in seams_sentences {
        if examples.len() >= config.max_naive_discrepancies_per_file {
            break;
        }
        while naive_index < naive.len() && !seams_sentence.starts_with(naive[naive_index].as_str())
        {
            naive_index += 1;
        }
        if naive_index >= naive.len() {
            break;
        }

        let mut combined = naive[naive_index].clone();
        let first = naive[naive_index].clone();
        let mut consumed = 1usize;
        while normalize_sentence(&combined, config.lowercase) != *seams_sentence
            && naive_index + consumed < naive.len()
            && seams_sentence.starts_with(combined.as_str())
        {
            combined = format!("{} {}", combined.trim_end(), naive[naive_index + consumed]);
            consumed += 1;
        }

        if consumed > 1 && normalize_sentence(&combined, config.lowercase) == *seams_sentence {
            let cursor = combined
                .strip_prefix(first.as_str())
                .unwrap_or("")
                .trim_start();
            if !cursor.is_empty() {
                push_example(
                    &mut examples,
                    BoundaryAction::Repair,
                    TrainingRowSource::NaiveDiscrepancy,
                    &first,
                    cursor,
                    format!("{REPAIR_TOKEN}{seams_sentence}"),
                    source,
                    config.lowercase,
                );
            }
            naive_index += consumed;
        } else {
            naive_index += 1;
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

fn write_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    let mut file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    for row in rows {
        writeln!(file, "{}", serde_json::to_string(row)?)?;
    }
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
            "Who shot John F. Kennedy?",
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
}
