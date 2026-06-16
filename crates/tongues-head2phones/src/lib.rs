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
    EnglishPhonemicizer, EvidenceProvenance, EvidenceSource, PhonemicizeOutput, PhonemicizeRequest,
    Phonemicizer, UtteranceId, UtterancePlan, VarietyId,
};
use styletts2::{prepare_styletts2_plan, styletts2_en_us_symbol_set, StyleTts2PlanOptions};
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Head2PhonesConfig {
    pub dataset_id: String,
    #[serde(default)]
    pub source_paths: Vec<PathBuf>,
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
            include_synthetic: default_include_synthetic(),
            synthetic_buffers: default_synthetic_buffers(),
            random_cuts_per_buffer: default_random_cuts_per_buffer(),
            no_head_cuts_per_head: default_no_head_cuts_per_head(),
            include_exceptional: default_include_exceptional(),
            train_frac: default_train_frac(),
            valid_frac: default_valid_frac(),
            seed: default_seed(),
            min_head_graphemes: default_min_head_graphemes(),
            max_head_graphemes: default_max_head_graphemes(),
        }
    }
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
pub struct PrepareReport {
    pub source_buffers: usize,
    pub complete_examples: usize,
    pub no_head_examples: usize,
    pub exceptional_examples: usize,
    pub train_examples: usize,
    pub valid_examples: usize,
    pub test_examples: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrepareProgress {
    Stage { message: String },
    Read { path: String, buffers: usize },
    Synthesize { path: String, buffers: usize },
    Build { complete: usize, no_head: usize },
    Write { path: String, rows: usize },
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
    let mut examples_part = BufWriter::new(
        File::create(&examples_part_path)
            .with_context(|| format!("creating {}", examples_part_path.display()))?,
    );
    let mut examples = Vec::new();
    let mut source_buffers = 0usize;

    let mut temporary_parts = vec![examples_part_path.clone()];

    if config.include_synthetic {
        let synthetic_path = out.join("synthetic_buffers.jsonl.part");
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

    for path in &config.source_paths {
        let raw =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let buffers = source_buffers_from_text(&raw);
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
        });
    }

    examples_part
        .flush()
        .with_context(|| format!("flushing {}", examples_part_path.display()))?;
    drop(examples_part);

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
        dataset_readme(config, &train, &valid, &test),
    )?;
    for part in temporary_parts {
        if part.exists() {
            fs::remove_file(&part).with_context(|| format!("removing {}", part.display()))?;
        }
    }

    Ok(PrepareReport {
        source_buffers,
        complete_examples,
        no_head_examples,
        exceptional_examples,
        train_examples: train.len(),
        valid_examples: valid.len(),
        test_examples: test.len(),
    })
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
    let spoken_text = spoken_form_for_head(text);
    let phonemicized = phonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: spoken_text,
            variety: VarietyId("en-US".to_string()),
            style: None,
        })
        .ok()?;
    let plan = utterance_plan_from_phonemicized(&phonemicized);
    let styletts2_plan = prepare_styletts2_plan(
        &plan,
        &styletts2_en_us_symbol_set(),
        StyleTts2PlanOptions {
            max_symbols_per_chunk: 512,
            chunking_enabled: false,
        },
    )
    .ok()?;
    let symbols = styletts2_plan
        .chunks
        .iter()
        .flat_map(|chunk| chunk.symbols.iter().map(|token| token.symbol.clone()))
        .collect::<Vec<_>>();
    if !symbols.is_empty() {
        return Some(symbols.join(" "));
    }

    phones_for_phonemicized(&phonemicized)
}

fn spoken_form_for_head(text: &str) -> String {
    let mut out = text.to_string();
    for (from, to) in [
        ("Dr.", "Doctor"),
        ("Mr.", "Mister"),
        ("Mrs.", "Missus"),
        ("Ms.", "Miz"),
        ("Rep.", "Representative"),
        ("Sen.", "Senator"),
        ("Gov.", "Governor"),
        ("Prof.", "Professor"),
        ("Sr.", "Senior"),
        ("Jr.", "Junior"),
        ("St.", "Saint"),
        ("e.g.", "e g"),
        ("E.g.", "e g"),
        ("i.e.", "i e"),
        ("I.e.", "i e"),
        ("a.m.", "a m"),
        ("A.M.", "a m"),
        ("p.m.", "p m"),
        ("P.M.", "p m"),
        ("D.", "D"),
        ("R.", "R"),
        ("NY.", "New York"),
        ("N.Y.", "New York"),
    ] {
        out = out.replace(from, to);
    }
    out = replace_number_abbreviation(&out);
    out
}

fn replace_number_abbreviation(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(index) = rest.find("No.") {
        out.push_str(&rest[..index]);
        let after = &rest[index + 3..];
        if after
            .trim_start()
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_digit())
        {
            out.push_str("Number");
        } else {
            out.push_str("No.");
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

fn utterance_plan_from_phonemicized(output: &PhonemicizeOutput) -> UtterancePlan {
    UtterancePlan {
        id: UtteranceId("head2phones.training.utterance".into()),
        variety: output.variety.clone(),
        speaker: None,
        intended_text: Some(output.text.clone()),
        intended_morphemes: Vec::new(),
        intended_phonemes: output.phonemes.clone(),
        target_phones: output.phones.clone(),
        target_syllables: output.syllables.clone(),
        boundaries: output.boundaries.clone(),
        target_prosody: output.prosody.clone(),
        target_acoustics: Vec::new(),
        style: None,
        provenance: EvidenceProvenance {
            source: EvidenceSource::TtsPlan,
            method: "head2phones training StyleTTS2 lowering".into(),
            version: Some("0.1".into()),
        },
    }
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
    let ipa_words = words
        .into_iter()
        .map(|(_, syllables)| syllables_to_ipa_formatted(&syllables))
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    (!ipa_words.is_empty()).then(|| ipa_words.join(" "))
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

fn syllables_to_ipa_formatted(syllables: &[speaking::Syllable]) -> String {
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
                text.push_str(phone_ipa(phone));
            }
            text
        })
        .collect()
}

fn synthetic_buffers(count: usize, rng: &mut StdRng) -> Vec<String> {
    let heads = [
        "Dr. Smith went home.",
        "This is the next sentence; and then the next.",
        "Wait... really?",
        "\"No.\" she said.",
        "Mr. Jones arrived after lunch.",
        "The package is ready, but the driver is late.",
        "First, open the small panel.",
        "- Bring the blue folder.",
        "I saw 3.14 written on the board.",
        "Use e.g. this example carefully.",
        "In short: the answer changed.",
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
        "Wait... really? I thought that was done.",
        "\"No.\" she said. Then she closed the book.",
        "\"No,\" she said, \"not yet.\" The room went quiet.",
        "(Really?) That was the whole answer.",
        "- Bring the blue folder.\n- Leave the red folder.",
        "First line complete.\nSecond line is still arriving",
        "Prof. Adams arrived at 4:30 p.m. sharp.",
        "A. B. Carter signed the note. Then he left.",
        "The package is ready, but the driver is late.",
        "This is the next sentence; and then the next.",
        "In short: the answer changed. The stream keeps moving.",
        "No. 5 was missing from the list. Then it appeared.",
        "After the meeting, Rep. Susan Smith (D. NY.) said she didn't know what the meeting was about.",
        "After the meeting, Rep.",
        "After the meeting, Rep. Susan Smith (D.",
        "After the meeting, Rep. Susan Smith (D. NY.",
        "I think we should",
        "This final fragment should be flushed",
    ]
}

fn source_buffers_from_text(raw: &str) -> Vec<String> {
    let mut buffers = Vec::new();
    if let Ok(detector) = SentenceDetectorDialog::new() {
        if let Ok(sentences) = detector.detect_sentences_borrowed(raw) {
            let sentences = sentences
                .into_iter()
                .map(|sentence| sentence.normalize().trim().to_string())
                .filter(|sentence| !sentence.is_empty())
                .collect::<Vec<_>>();
            for (index, sentence) in sentences.iter().enumerate() {
                let mut buffer = sentence.clone();
                if let Some(next) = sentences.get(index + 1) {
                    buffer.push(' ');
                    buffer.push_str(next);
                }
                buffers.push(buffer);
            }
        }
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
) -> String {
    format!(
        "# head2phones {}\n\nTrain/valid/test rows: {}/{}/{}.\n\nOutputs are exactly `{}` or `{}` StyleTTS2-lowered speech symbols `{}` plus `{}` and a Unicode grapheme-cluster offset. The speech symbols include phones/phonemes, word boundaries, punctuation, stress, and intonation markers used by the synthesizer.\n",
        config.dataset_id,
        train.len(),
        valid.len(),
        test.len(),
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
    fn dotted_abbreviations_do_not_split_the_head() {
        let text = "Use e.g. this example carefully. Then continue.";
        let head = first_complete_head(text).expect("complete head");
        assert_eq!(&text[..head.end_byte], "Use e.g. this example carefully.");

        let text = "Use i.e. this case as the control. Then continue.";
        let head = first_complete_head(text).expect("complete head");
        assert_eq!(&text[..head.end_byte], "Use i.e. this case as the control.");
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
    fn lowered_symbols_include_boundaries_and_terminal_prosody() {
        let symbols = speech_symbols_for_text("Dr. Smith went home.").expect("symbols");
        assert!(symbols.contains("|"));
        assert!(symbols.contains("D") || symbols.contains("DH"));
        assert!(symbols.contains("."));
        assert!(symbols.contains("↘"));
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
