//! `tongues` CLI – neural lexical and speech-front-end model families.
//!
//! # Commands
//!
//! ```text
//! tongues g2p2g prepare --out datasets/g2p2g/openepd-v0
//! tongues g2p2g train --data datasets/g2p2g/openepd-v0 --out models/g2p2g/openepd-v0
//! tongues g2p2g eval --model models/g2p2g/openepd-v0 --split test
//! tongues g2p2g infer --model models/g2p2g/openepd-v0 "charlotte"
//! tongues sentence-parser parse --model models/sentence-parser/v0 "The quick fox jumps."
//! ```

mod fetch_corpora;
pub mod models;
mod speak;
mod styletts2_cmds;

use std::fs;
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::{any::Any, panic};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use burn_cuda::{Cuda, CudaDevice};

use speaking::data::notation::openepd::normalize_openepd_ipa;
use speaking::{
    AudioFrame, EvidenceProvenance, EvidenceSource, PhoneToken, Spec, SpeechRecognizer,
    UtteranceId, UtterancePlan, VarietyId, WhisperSpeechRecognizer,
};
use styletts2::{
    prepare_styletts2_plan, styletts2_en_us_symbol_set, styletts2_text_for_symbols,
    StyleTts2Backend, StyleTts2DiffusionOptions, StyleTts2OnnxBackend, StyleTts2PlanOptions,
    StyleTts2SynthesisRequest, DEFAULT_MAX_TTS_SYMBOLS,
};
use tongues_core::{Vocab, BOS_ID, EOS_ID, UNK_ID};
use tongues_data::{Lexeme, Seq2SeqExample, Task};
use tongues_g2p2g::{
    eval_report, load_model, predict, train, train_seq2seq_examples, ModelConfig, Seq2SeqModel,
    TrainConfig,
};
use tongues_interpretation::{
    InterpretationConfig, InterpretationTrainConfig, LibriSpeechSubset, TranscriptRefinement,
};
use tongues_neural::{write_manifest, ModelArtifactManifest};
use tongues_tts as speech;

// ── Backend aliases ────────────────────────────────────────────────────────

type CpuInferBackend = NdArray<f32>;
type CpuTrainBackend = Autodiff<CpuInferBackend>;

type CudaInferBackend = Cuda<f32, i32>;
type CudaTrainBackend = Autodiff<CudaInferBackend>;

const DEFAULT_WIKTIONARY_DATASET_ID: &str = "enwiktionary-2026-06-01-v0";
const DEFAULT_WIKTIONARY_CONFIG_PATH: &str = "configs/wiktionary/default.toml";
const DEFAULT_WIKTIONARY_CACHE_DIR: &str = "data/wiktionary";
const DEFAULT_WIKTIONARY_DATA_DIR: &str = "datasets/wiktionary/enwiktionary-2026-06-01-v0";
const DEFAULT_WIKTIONARY_MODEL_DIR: &str = "models/wiktionary/enwiktionary-2026-06-01-v0-phones";
const DEFAULT_G2P2G_DATA_DIR: &str = "datasets/g2p2g/openepd-v0";
const DEFAULT_G2P2G_MODEL_DIR: &str = "models/g2p2g/openepd-v0";
const DEFAULT_SENTENCE_PARSER_DATA_DIR: &str = "datasets/sentence-parser/v0";
const DEFAULT_SENTENCE_PARSER_MODEL_DIR: &str = "models/sentence-parser/v0";
const DEFAULT_HEAD2PHONES_DATA_DIR: &str = "datasets/head2phones/v0";
const DEFAULT_HEAD2PHONES_MODEL_DIR: &str = "models/head2phones/v0";
const DEFAULT_HEAD2PHONES_BATCH_SIZE: usize = 8;
const DEFAULT_INTERPRETATION_DATA_DIR: &str = "datasets/interpretation/mini-v0";
const DEFAULT_INTERPRETATION_MODEL_DIR: &str = "models/interpretation/mini-v0";
const DEFAULT_COMMON_PHONE_DATA_DIR: &str = "datasets/common-phone/v0";
const DEFAULT_COMMON_PHONE_MODEL_DIR: &str = "models/common-phone/v0";
const DEFAULT_EMOTIONS_DATA_DIR: &str = "datasets/emotions/v0";
const DEFAULT_EMOTIONS_MODEL_DIR: &str = "models/emotions/v0";
const DEFAULT_WHISPER_TRANSCRIPT_MAX_WER: f64 = 0.70;
static QUIET_OUTPUT: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
enum DeviceArg {
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, Debug)]
struct OutputMode {
    quiet: bool,
}

impl OutputMode {
    fn for_command(command: &Commands, quiet: bool, verbose: bool) -> Self {
        let quiet = if quiet {
            true
        } else if verbose {
            false
        } else {
            command_defaults_to_quiet(command)
        };
        Self { quiet }
    }

    fn verbose(self) -> bool {
        !self.quiet
    }
}

fn set_quiet_output(quiet: bool) {
    QUIET_OUTPUT.store(quiet, Ordering::Relaxed);
}

fn quiet_output() -> bool {
    QUIET_OUTPUT.load(Ordering::Relaxed)
}

// ── CLI definition ─────────────────────────────────────────────────────────

/// tongues – neural lexical and speech-front-end model families
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Use CPU instead of CUDA GPU
    #[arg(long, global = true)]
    cpu: bool,

    /// Silence status bars and diagnostic progress output
    #[arg(long, global = true, conflicts_with = "verbose")]
    quiet: bool,

    /// Show status bars and diagnostic progress output
    #[arg(long, global = true, conflicts_with = "quiet")]
    verbose: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Train and run the lexical grapheme/phoneme seq2seq model family
    G2p2g {
        #[command(subcommand)]
        command: G2p2gCommands,
    },

    /// Prepare, train, and run sentence parser models
    #[command(name = "sentence-parser")]
    SentenceParser {
        #[command(subcommand)]
        command: SentenceParserCommands,
    },

    /// Prepare, train, and run rolling head-chunk-to-phones models
    Head2phones {
        #[command(subcommand)]
        command: Head2PhonesCommands,
    },

    /// Prepare, train, evaluate, and stream LibriSpeech ASR models
    #[command(name = "interpretation")]
    Interpretation {
        #[command(subcommand)]
        command: InterpretationCommands,
    },

    /// Prepare, train, and evaluate Common Phone compact-frame CTC models
    #[command(name = "common-phone", alias = "commonphone")]
    CommonPhone {
        #[command(subcommand)]
        command: CommonPhoneCommands,
    },

    /// Prepare, train, evaluate, and run audio emotion classifiers
    Emotions {
        #[command(subcommand)]
        command: EmotionCommands,
    },

    /// Prepare English Wiktionary pronunciation data
    Wiktionary {
        #[command(subcommand)]
        command: WiktionaryCommands,
    },

    /// Download CMUdict from GitHub
    FetchCmudict {
        /// Output path for the downloaded file
        #[arg(long, default_value = "data/cmudict.dict")]
        out: PathBuf,
    },

    /// Download Lexique383 from lexique.org
    FetchLexique {
        /// Output path for the downloaded file
        #[arg(long, default_value = "data/Lexique383.tsv")]
        out: PathBuf,
    },

    /// Download and extract public emotion corpora for StyleTTS2 signatures
    FetchCorpora {
        /// Output directory for the datasets
        #[arg(long, default_value = "datasets/emotions")]
        out_dir: PathBuf,

        /// Corpus to fetch/label; repeat to choose a subset. Defaults to all known corpora.
        #[arg(long = "corpus", value_enum)]
        corpora: Vec<fetch_corpora::EmotionCorpusArg>,

        /// List available corpora and exit
        #[arg(long)]
        list: bool,
    },

    /// Compare pronunciations from lexicons, rules, and trained models
    #[command(alias = "discrepancy", alias = "discrepency", alias = "discrepencies")]
    Discrepancies {
        /// Markdown report path to write
        #[arg(long, default_value = "docs/pronunciation-discrepancies.md")]
        out: PathBuf,

        /// Limit the default OpenEPD word sample
        #[arg(long, default_value_t = 250)]
        limit: usize,

        /// Maximum OpenEPD rarity rank included in the default sample
        #[arg(long, default_value_t = 50_000.0)]
        max_rarity: f32,

        /// Add an explicit word to compare; may be passed more than once
        #[arg(long = "word")]
        words: Vec<String>,

        /// Read additional words, one per line
        #[arg(long)]
        words_file: Option<PathBuf>,

        /// Skip the G2P2G model pronouncer
        #[arg(long = "no-g2p2g")]
        no_g2p2g: bool,

        /// Skip the Wiktionary model pronouncer
        #[arg(long = "no-wiktionary")]
        no_wiktionary: bool,

        /// G2P2G model directory
        #[arg(long, default_value = "models/g2p2g/openepd-v0")]
        g2p2g_model: PathBuf,

        /// Wiktionary model directory
        #[arg(
            long,
            default_value = "models/wiktionary/enwiktionary-2026-06-01-v0-phones"
        )]
        wiktionary_model: PathBuf,

        /// Wiktionary variety tag used for the model pronouncer
        #[arg(long, default_value = "en-US.GenAm")]
        wiktionary_variety: String,
    },

    /// Parse OpenEPD, build vocabulary, and create train/valid/test splits
    Prepare {
        /// Deprecated compatibility argument; prepare now uses embedded OpenEPD.
        #[arg(long)]
        input: Option<PathBuf>,

        /// Output directory for splits and vocabulary
        #[arg(long, default_value = "runs/cmudict-v0")]
        out: PathBuf,

        /// Fraction of base words for training
        #[arg(long, default_value_t = 0.8)]
        train_frac: f64,

        /// Fraction of base words for validation
        #[arg(long, default_value_t = 0.1)]
        valid_frac: f64,

        /// Random seed for reproducible splits
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },

    /// Train the masked-phone predictor
    Train {
        /// Prepared data directory (output of `prepare`)
        #[arg(long)]
        data: PathBuf,

        /// Output directory for the model
        #[arg(long, default_value = "models/cmudict-v0")]
        out: PathBuf,

        /// Masking policy: single (always one mask) or variable (curriculum)
        #[arg(long, value_enum, default_value = "variable")]
        mask_policy: MaskPolicyArg,

        /// Max fraction of phones to mask in variable mode
        #[arg(long, default_value_t = 0.4)]
        max_mask_rate: f64,

        /// Span mask probability weight
        #[arg(long, default_value_t = 0.15)]
        span_mask_prob: f64,

        /// AdamW learning rate
        #[arg(long, default_value_t = 3e-4)]
        learning_rate: f64,

        /// AdamW weight decay
        #[arg(long, default_value_t = 1e-4)]
        weight_decay: f32,

        /// Dropout rate
        #[arg(long, default_value_t = 0.1)]
        dropout: f64,

        /// Maximum training epochs
        #[arg(long, default_value_t = 20)]
        epochs: usize,

        /// Early stopping patience (epochs with no improvement)
        #[arg(long, default_value_t = 5)]
        patience: usize,

        /// Mini-batch size
        #[arg(long, default_value_t = 64)]
        batch_size: usize,

        /// Random seed
        #[arg(long, default_value_t = 0)]
        seed: u64,

        /// Direction of translation to train: g2p, p2g, or both
        #[arg(long, default_value = "both")]
        task: String,
    },

    /// Evaluate a trained model
    Eval {
        /// Directory containing the trained model
        #[arg(long)]
        model: PathBuf,

        /// Split to evaluate on: train, valid, or test
        #[arg(long, default_value = "test")]
        split: String,

        /// Prepared data directory
        #[arg(long)]
        data: PathBuf,

        /// Direction of translation to evaluate: g2p, p2g, both, or auto (detect from train_config)
        #[arg(long, default_value = "auto")]
        task: String,
    },

    /// Fine-tune a model on validation/test discrepancies
    Refine {
        /// Directory containing the trained source model
        #[arg(long)]
        model: PathBuf,

        /// Prepared data directory
        #[arg(long)]
        data: PathBuf,

        /// Output directory for the refined model
        #[arg(long)]
        out: PathBuf,

        /// Comma-separated splits to mine for discrepancies
        #[arg(long, default_value = "valid,test")]
        splits: String,

        /// Refinement source: held-out discrepancies or the built-in sight-word list
        #[arg(long, value_enum, default_value = "discrepancies")]
        source: RefinementSourceArg,

        /// Direction to refine: g2p, p2g, or both
        #[arg(long, default_value = "g2p")]
        task: String,

        /// AdamW learning rate for refinement
        #[arg(long, default_value_t = 1e-4)]
        learning_rate: f64,

        /// AdamW weight decay
        #[arg(long, default_value_t = 1e-4)]
        weight_decay: f32,

        /// Maximum refinement epochs
        #[arg(long, default_value_t = 5)]
        epochs: usize,

        /// Early stopping patience
        #[arg(long, default_value_t = 2)]
        patience: usize,

        /// Mini-batch size
        #[arg(long, default_value_t = 32)]
        batch_size: usize,

        /// Random seed
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },

    /// Interactive REPL for sequence translation
    Repl {
        /// Direction of translation: g2p, p2g, auto
        #[arg(long, default_value = "auto")]
        task: String,

        /// Directory containing the trained model
        #[arg(long, default_value = "models/cmudict-v0")]
        model: PathBuf,

        /// Optional path to the prepared data directory containing vocab.json
        #[arg(long)]
        data: Option<PathBuf>,
    },

    /// Run translation prediction (Seq2Seq)
    #[command(alias = "infer")]
    Predict {
        /// The input sequence to translate
        input: String,

        /// Direction of translation: g2p, p2g, auto
        #[arg(long, default_value = "auto")]
        task: String,

        /// Directory containing the trained model
        #[arg(long, default_value = "models/cmudict-v0")]
        model: PathBuf,

        /// Optional path to the prepared data directory containing vocab.json
        #[arg(long)]
        data: Option<PathBuf>,
    },

    /// Speak/synthesize text into a WAV file using speech plans
    Speak(speak::SpeakCommand),

    /// Stream an Ollama story through head2phones and speech playback
    Be(BeCommand),

    /// Demonstrate the speaking library across built-in language varieties
    #[command(name = "speaking-demo", alias = "speaking")]
    SpeakingDemo {
        /// Demo mode
        #[arg(value_enum, default_value = "samples")]
        mode: SpeakingDemoMode,

        /// Restrict the demo to one variety; repeat to select multiple.
        #[arg(long = "variety")]
        varieties: Vec<String>,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: SpeakingDemoFormat,
    },

    /// Phonemize text into a broad IPA phoneme sequence
    Phonemes {
        /// The text to phonemize
        text: String,
    },

    /// Phonemize text into a narrow IPA phone sequence
    Phones {
        /// The text to phonemize
        text: String,
    },

    /// Manage local models
    Models {
        #[command(subcommand)]
        command: Option<models::ModelsCommand>,
    },

    /// Tools for the StyleTTS2 backend
    Styletts2 {
        #[command(subcommand)]
        command: Styletts2Commands,
    },
}

#[derive(Subcommand, Debug)]
enum G2p2gCommands {
    /// Archive selected default artifacts and recreate empty run directories
    Clean(CleanArgs),

    /// Parse OpenEPD, build vocabulary, and create train/valid/test splits
    Prepare {
        /// TOML config file for the G2P2G pipeline
        #[arg(long, default_value = "configs/g2p2g/default.toml")]
        config: PathBuf,

        /// Deprecated compatibility argument; prepare now uses embedded OpenEPD.
        #[arg(long)]
        input: Option<PathBuf>,

        /// Output directory for splits and vocabulary
        #[arg(long, default_value = "datasets/g2p2g/openepd-v0")]
        out: PathBuf,

        /// Fraction of base words for training
        #[arg(long)]
        train_frac: Option<f64>,

        /// Fraction of base words for validation
        #[arg(long)]
        valid_frac: Option<f64>,

        /// Random seed for reproducible splits
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Train the G2P2G seq2seq model
    Train {
        /// TOML config file for the G2P2G pipeline
        #[arg(long, default_value = "configs/g2p2g/default.toml")]
        config: PathBuf,

        /// Prepared data directory
        #[arg(long, default_value = "datasets/g2p2g/openepd-v0")]
        data: PathBuf,

        /// Output directory for the model
        #[arg(long, default_value = "models/g2p2g/openepd-v0")]
        out: PathBuf,

        /// Masking policy: single (always one mask) or variable (curriculum)
        #[arg(long, value_enum, default_value = "variable")]
        mask_policy: MaskPolicyArg,

        /// Max fraction of phones to mask in variable mode
        #[arg(long, default_value_t = 0.4)]
        max_mask_rate: f64,

        /// Span mask probability weight
        #[arg(long, default_value_t = 0.15)]
        span_mask_prob: f64,

        /// AdamW learning rate
        #[arg(long)]
        learning_rate: Option<f64>,

        /// AdamW weight decay
        #[arg(long)]
        weight_decay: Option<f32>,

        /// Dropout rate
        #[arg(long)]
        dropout: Option<f64>,

        /// Maximum training epochs
        #[arg(long)]
        epochs: Option<usize>,

        /// Early stopping patience
        #[arg(long)]
        patience: Option<usize>,

        /// Mini-batch size
        #[arg(long)]
        batch_size: Option<usize>,

        /// Random seed
        #[arg(long)]
        seed: Option<u64>,

        /// Direction of translation to train: g2p, p2g, or both
        #[arg(long)]
        task: Option<String>,

        /// Wait for an in-progress prepare in --data to finish, then start training
        #[arg(long, visible_alias = "while-preparing")]
        wait_for_prepare: bool,
    },

    /// Evaluate a trained G2P2G model
    Eval {
        /// Directory containing the trained model
        #[arg(long, default_value = "models/g2p2g/openepd-v0")]
        model: PathBuf,

        /// Split to evaluate on: train, valid, or test
        #[arg(long, default_value = "test")]
        split: String,

        /// Prepared data directory
        #[arg(long, default_value = "datasets/g2p2g/openepd-v0")]
        data: PathBuf,

        /// Direction of translation to evaluate: g2p, p2g, both, or auto
        #[arg(long, default_value = "auto")]
        task: String,
    },

    /// Fine-tune a G2P2G model on validation/test discrepancies
    Refine {
        /// Directory containing the trained source model
        #[arg(long, default_value = "models/g2p2g/openepd-v0")]
        model: PathBuf,

        /// Prepared data directory
        #[arg(long, default_value = "datasets/g2p2g/openepd-v0")]
        data: PathBuf,

        /// Output directory for the refined model
        #[arg(long)]
        out: PathBuf,

        /// Comma-separated splits to mine for discrepancies
        #[arg(long, default_value = "valid,test")]
        splits: String,

        /// Refinement source: held-out discrepancies or the built-in sight-word list
        #[arg(long, value_enum, default_value = "discrepancies")]
        source: RefinementSourceArg,

        /// Direction to refine: g2p, p2g, or both
        #[arg(long, default_value = "g2p")]
        task: String,

        /// AdamW learning rate for refinement
        #[arg(long, default_value_t = 1e-4)]
        learning_rate: f64,

        /// AdamW weight decay
        #[arg(long, default_value_t = 1e-4)]
        weight_decay: f32,

        /// Maximum refinement epochs
        #[arg(long, default_value_t = 5)]
        epochs: usize,

        /// Early stopping patience
        #[arg(long, default_value_t = 2)]
        patience: usize,

        /// Mini-batch size
        #[arg(long, default_value_t = 32)]
        batch_size: usize,

        /// Random seed
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },

    /// Interactive REPL for G2P2G sequence translation
    Repl {
        /// Direction of translation: g2p, p2g, auto
        #[arg(long, default_value = "auto")]
        task: String,

        /// Directory containing the trained model
        #[arg(long, default_value = "models/g2p2g/openepd-v0")]
        model: PathBuf,

        /// Optional path to the prepared data directory containing vocab.json
        #[arg(long)]
        data: Option<PathBuf>,
    },

    /// Run G2P2G translation inference
    #[command(alias = "predict")]
    Infer {
        /// The input sequence to translate
        input: String,

        /// Direction of translation: g2p, p2g, auto
        #[arg(long, default_value = "auto")]
        task: String,

        /// Directory containing the trained model
        #[arg(long, default_value = "models/g2p2g/openepd-v0")]
        model: PathBuf,

        /// Optional path to the prepared data directory containing vocab.json
        #[arg(long)]
        data: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
enum SentenceParserCommands {
    /// Archive selected default artifacts and recreate empty run directories
    Clean(CleanArgs),

    /// Prepare a sentence parser dataset scaffold
    Prepare {
        /// TOML config file for the sentence parser pipeline
        #[arg(long, default_value = "configs/sentence-parser/default.toml")]
        config: PathBuf,

        /// Project Gutenberg text file or directory; may be passed more than once
        #[arg(long = "input")]
        inputs: Vec<PathBuf>,

        /// Output directory for parser data
        #[arg(long, default_value = "datasets/sentence-parser/v0")]
        out: PathBuf,
    },

    /// Write a sentence parser model scaffold
    Train {
        /// TOML config file for the sentence parser pipeline
        #[arg(long, default_value = "configs/sentence-parser/default.toml")]
        config: PathBuf,

        /// Prepared data directory
        #[arg(long, default_value = "datasets/sentence-parser/v0")]
        data: PathBuf,

        /// Project Gutenberg text file or directory to use when --prepare is set; may be passed more than once
        #[arg(long = "input")]
        inputs: Vec<PathBuf>,

        /// Output directory for the model
        #[arg(long, default_value = "models/sentence-parser/v0")]
        out: PathBuf,

        /// Prepare data before training
        #[arg(long)]
        prepare: bool,

        /// Wait for an in-progress prepare in --data to finish, then start training
        #[arg(long, visible_alias = "while-preparing")]
        wait_for_prepare: bool,

        /// AdamW learning rate
        #[arg(long, default_value_t = 3e-4)]
        learning_rate: f64,

        /// AdamW weight decay
        #[arg(long, default_value_t = 1e-4)]
        weight_decay: f32,

        /// Dropout rate
        #[arg(long, default_value_t = 0.1)]
        dropout: f64,

        /// Mini-batch size
        #[arg(long, default_value_t = 64)]
        batch_size: usize,

        /// Maximum training epochs
        #[arg(long, default_value_t = 20)]
        epochs: usize,

        /// Early stopping patience
        #[arg(long, default_value_t = 5)]
        patience: usize,

        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Prepared row source to train on
        #[arg(long, value_enum, default_value = "all")]
        training_set: SentenceParserTrainingSetArg,
    },

    /// Validate a sentence parser artifact scaffold
    Eval {
        /// Directory containing the parser model
        #[arg(long, default_value = "models/sentence-parser/v0")]
        model: PathBuf,

        /// Split to evaluate on
        #[arg(long, default_value = "test")]
        split: String,
    },

    /// Parse a sentence into the speech syntax analysis shape
    Parse {
        /// Directory containing the parser model
        #[arg(long, default_value = "models/sentence-parser/v0")]
        model: PathBuf,

        /// Sentence to parse
        text: String,
    },

    /// Run cursor-time sentence-boundary seq2seq inference
    Infer {
        /// Directory containing the parser model
        #[arg(long, default_value = "models/sentence-parser/v0")]
        model: PathBuf,

        /// Previously parsed sentence to show the model
        #[arg(long, default_value = "")]
        previous: String,

        /// Current cursor prefix
        cursor: String,
    },

    /// Stream stdin through the cursor-time sentence parser
    Stream {
        /// Directory containing the parser model
        #[arg(long, default_value = "models/sentence-parser/v0")]
        model: PathBuf,

        /// ANSI control sequence emitted before a repaired sentence
        #[arg(long, default_value = "\u{1b}[1A\u{1b}[2K")]
        repair_control: String,
    },
}

#[derive(Subcommand, Debug)]
enum Head2PhonesCommands {
    /// Archive selected default artifacts and recreate empty run directories
    Clean(CleanArgs),

    /// Prepare head2phones seq2seq data
    Prepare {
        /// TOML config file for the head2phones pipeline
        #[arg(long, default_value = "configs/head2phones/default.toml")]
        config: PathBuf,

        /// Optional text file or directory; may be passed more than once
        #[arg(long = "input")]
        inputs: Vec<PathBuf>,

        /// Output directory for prepared data
        #[arg(long, default_value = "datasets/head2phones/v0")]
        out: PathBuf,

        /// Ask an Ollama model to passively scan prepared train rows
        #[arg(long)]
        verify_ollama: bool,

        /// Ollama model name for head2phones data verification
        #[arg(long)]
        ollama_model: Option<String>,

        /// Ollama server URL for head2phones data verification
        #[arg(long)]
        ollama_url: Option<String>,

        /// Maximum train rows to pass to Ollama per scan request
        #[arg(long)]
        ollama_rows: Option<usize>,

        /// Maximum JSONL characters to include in the Ollama prompt
        #[arg(long)]
        ollama_max_chars: Option<usize>,

        /// Fail prepare if Ollama reports scanned data is not sane
        #[arg(long)]
        ollama_strict: bool,
    },

    /// Passively verify an existing prepared head2phones train split with Ollama
    #[command(alias = "scan")]
    Verify {
        /// TOML config file for the head2phones pipeline
        #[arg(long, default_value = "configs/head2phones/default.toml")]
        config: PathBuf,

        /// Prepared data directory containing train.jsonl
        #[arg(long, default_value = "datasets/head2phones/v0")]
        data: PathBuf,

        /// Ollama model name for head2phones data verification
        #[arg(long)]
        ollama_model: Option<String>,

        /// Ollama server URL for head2phones data verification
        #[arg(long)]
        ollama_url: Option<String>,

        /// Maximum train rows to pass to Ollama per scan request
        #[arg(long)]
        ollama_rows: Option<usize>,

        /// Maximum JSONL characters to include in the Ollama prompt
        #[arg(long)]
        ollama_max_chars: Option<usize>,

        /// Exit non-zero if Ollama reports scanned data is not sane
        #[arg(long)]
        strict: bool,
    },

    /// Train the head2phones seq2seq model
    Train {
        /// TOML config file for the head2phones pipeline
        #[arg(long, default_value = "configs/head2phones/default.toml")]
        config: PathBuf,

        /// Prepared data directory
        #[arg(long, default_value = "datasets/head2phones/v0")]
        data: PathBuf,

        /// Optional text file or directory to use when --prepare is set; may be passed more than once
        #[arg(long = "input")]
        inputs: Vec<PathBuf>,

        /// Output directory for the model
        #[arg(long, default_value = "models/head2phones/v0")]
        out: PathBuf,

        /// Prepare data before training
        #[arg(long)]
        prepare: bool,

        /// Ask an Ollama model to passively scan train rows when preparing data
        #[arg(long)]
        verify_ollama: bool,

        /// Ollama model name for head2phones data verification
        #[arg(long)]
        ollama_model: Option<String>,

        /// Ollama server URL for head2phones data verification
        #[arg(long)]
        ollama_url: Option<String>,

        /// Maximum train rows to pass to Ollama per scan request
        #[arg(long)]
        ollama_rows: Option<usize>,

        /// Maximum JSONL characters to include in the Ollama prompt
        #[arg(long)]
        ollama_max_chars: Option<usize>,

        /// Fail prepare if Ollama reports scanned data is not sane
        #[arg(long)]
        ollama_strict: bool,

        /// Wait for an in-progress prepare in --data to finish, then start training
        #[arg(long, visible_alias = "while-preparing")]
        wait_for_prepare: bool,

        /// AdamW learning rate
        #[arg(long, default_value_t = 3e-4)]
        learning_rate: f64,

        /// AdamW weight decay
        #[arg(long, default_value_t = 1e-4)]
        weight_decay: f32,

        /// Dropout rate
        #[arg(long, default_value_t = 0.1)]
        dropout: f64,

        /// Mini-batch size
        #[arg(long, default_value_t = DEFAULT_HEAD2PHONES_BATCH_SIZE)]
        batch_size: usize,

        /// Maximum training epochs
        #[arg(long, default_value_t = 20)]
        epochs: usize,

        /// Early stopping patience
        #[arg(long, default_value_t = 5)]
        patience: usize,

        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },

    /// Run rolling-buffer head2phones inference
    #[command(alias = "predict")]
    Infer {
        /// Directory containing the head2phones model
        #[arg(long, default_value = "models/head2phones/v0")]
        model: PathBuf,

        /// Target pronunciation variety tag
        #[arg(long, default_value = "en-US")]
        variety: String,

        /// Raw rolling UTF-8 text buffer
        buffer: String,
    },

    /// Run prepared examples through the head2phones model with timings
    Eval {
        /// Directory containing the head2phones model
        #[arg(long, default_value = "models/head2phones/v0")]
        model: PathBuf,

        /// Prepared data directory containing split JSONL files
        #[arg(long, default_value = "datasets/head2phones/v0")]
        data: PathBuf,

        /// Prepared split to evaluate
        #[arg(long, default_value = "test")]
        split: String,

        /// Maximum examples to run
        #[arg(long, default_value_t = 24)]
        limit: usize,

        /// Random seed for sampling examples
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

#[derive(Debug, Args, Clone)]
struct BeCommand {
    /// Ollama server URL
    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,

    /// Ollama model to ask for text
    #[arg(long, default_value = "gpt-oss:20b")]
    ollama_model: String,

    /// Prompt sent to Ollama
    #[arg(long, default_value = "Tell me a story.")]
    prompt: String,

    /// Directory containing the resident head2phones model
    #[arg(long, default_value = DEFAULT_HEAD2PHONES_MODEL_DIR)]
    head2phones_model: PathBuf,

    /// Use seams sentence detection plus the speaking phonemicizer instead of head2phones
    #[arg(long)]
    mechanical: bool,

    /// Requested language/pronunciation variety for head2phones
    #[arg(long, default_value = "en-US")]
    variety: String,

    /// Speech backend to use for spoken output
    #[arg(long, value_enum, default_value_t = BeVoiceBackend::Onnx)]
    voice_backend: BeVoiceBackend,

    /// Maximum symbols per StyleTTS2 chunk when --voice-backend styletts2 is used
    #[arg(long, default_value_t = DEFAULT_MAX_TTS_SYMBOLS)]
    max_tts_symbols: usize,

    /// Disable StyleTTS2 text chunking when --voice-backend styletts2 is used
    #[arg(long)]
    no_tts_chunking: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BeVoiceBackend {
    Onnx,
    Styletts2,
}

#[derive(Subcommand, Debug)]
enum InterpretationCommands {
    /// Archive selected default artifacts and recreate empty run directories
    Clean(CleanArgs),

    /// Prepare sentence and pronunciation audio with Mel, sentence, and phoneme supervision
    Prepare {
        /// LibriSpeech subset: mini or train-clean-100
        #[arg(long, default_value = "mini")]
        subset: String,

        /// Output directory for prepared data
        #[arg(long, default_value = "datasets/interpretation/mini-v0")]
        out: PathBuf,

        /// Limit utterances for smoke tests
        #[arg(long)]
        max_utterances: Option<usize>,

        /// Prepared Wiktionary dataset to import single-word Commons pronunciation audio from
        #[arg(long)]
        wiktionary_audio_data: Option<PathBuf>,

        /// Do not import Wiktionary/Commons pronunciation audio rows
        #[arg(long)]
        no_wiktionary_audio: bool,

        /// Limit imported Wiktionary audio rows for smoke tests
        #[arg(long)]
        max_wiktionary_audio: Option<usize>,

        /// Do not download missing Wiktionary/Commons pronunciation audio files
        #[arg(long)]
        no_download_wiktionary_audio: bool,

        /// Whisper ggml model path for transcript recasing/punctuation.
        #[arg(long)]
        whisper_model: Option<PathBuf>,

        /// Keep original LibriSpeech transcript text instead of Whisper recasing.
        #[arg(long)]
        no_whisper_transcripts: bool,

        /// Maximum word error rate allowed between Whisper text and the original transcript.
        #[arg(long, default_value_t = DEFAULT_WHISPER_TRANSCRIPT_MAX_WER)]
        max_whisper_wer: f64,
    },

    /// Train the LibriSpeech ASR model
    Train {
        /// Prepared data directory
        #[arg(long, default_value = "datasets/interpretation/mini-v0")]
        data: PathBuf,

        /// Output directory for the model
        #[arg(long, default_value = "models/interpretation/mini-v0")]
        out: PathBuf,

        /// Wait for an in-progress prepare in --data to finish, then start training
        #[arg(long, visible_alias = "while-preparing")]
        wait_for_prepare: bool,

        /// Maximum training epochs
        #[arg(long)]
        epochs: Option<usize>,

        /// Mini-batch size
        #[arg(long)]
        batch_size: Option<usize>,

        /// Random seed
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Evaluate a LibriSpeech ASR model
    Eval {
        /// Directory containing the model
        #[arg(long, default_value = "models/interpretation/mini-v0")]
        model: PathBuf,

        /// Prepared data directory
        #[arg(long, default_value = "datasets/interpretation/mini-v0")]
        data: PathBuf,

        /// Split to evaluate: train, valid, or test
        #[arg(long, default_value = "test")]
        split: String,
    },

    /// Stream raw 16 kHz mono WAV audio from a file through the ASR model
    Stream {
        /// Directory containing the model
        #[arg(long, default_value = "models/interpretation/mini-v0")]
        model: PathBuf,

        /// WAV file to stream for v1 smoke testing
        #[arg(long)]
        wav: PathBuf,
    },
}

#[derive(Subcommand, Debug)]
enum CommonPhoneCommands {
    /// Archive selected default artifacts and recreate empty run directories
    Clean(CleanArgs),

    /// List available CPAL input devices
    #[command(name = "listen-devices")]
    ListenDevices,

    /// Stream microphone audio through a Common Phone CTC model
    Listen {
        /// Directory containing the trained Common Phone model
        #[arg(long)]
        model: Option<PathBuf>,

        /// Decode task: frames2phones, frames2features, frames2phonemes, multitask
        #[arg(long, default_value = "frames2phones")]
        task: String,

        /// Device for v0 inference; currently only cpu is supported
        #[arg(long, default_value = "cpu")]
        device: String,

        /// CPAL input device name substring
        #[arg(long)]
        input_device: Option<String>,

        /// Target model sample rate
        #[arg(long, default_value_t = tongues_common_phone::DEFAULT_SAMPLE_RATE_HZ)]
        sample_rate: u32,

        /// Audio chunk duration per live update
        #[arg(long, default_value_t = 100)]
        chunk_ms: u64,

        /// Rolling context duration for repeated CTC inference
        #[arg(long, default_value_t = 1500)]
        context_ms: u64,

        /// Show phone predictions
        #[arg(long)]
        show_phones: bool,

        /// Show feature bundles
        #[arg(long)]
        show_features: bool,

        /// Print compact frame/VAD/debug stats each tick
        #[arg(long, alias = "show-frames")]
        debug_frames: bool,

        /// Capture audio and print frame stats without loading a model
        #[arg(long)]
        dry_run: bool,

        /// Optional future phones-to-orthography model path; accepted but not used in v0
        #[arg(long)]
        phones2orth: Option<PathBuf>,
    },

    /// Create/document the expected local raw-data layout
    Fetch {
        /// Output raw-data directory
        #[arg(long, default_value = "data/common-phone/raw")]
        out: PathBuf,

        /// Acquisition source: zenodo downloads the official archive; huggingface is documented only
        #[arg(long, default_value = "zenodo")]
        source: String,

        /// Comma-separated languages to acquire externally
        #[arg(long)]
        languages: Option<String>,
    },

    /// Prepare a local Common Phone export into compact acoustic frame files
    Prepare {
        /// Local Common Phone checkout/export with metadata.jsonl/csv/tsv
        #[arg(long, default_value = "data/common-phone/raw")]
        input: PathBuf,

        /// Download and extract the official Common Phone archive before preparing
        #[arg(long)]
        download: bool,

        /// Download source: zenodo is implemented; huggingface is documented only
        #[arg(long, default_value = "zenodo")]
        source: String,

        /// Override source archive URL
        #[arg(long)]
        source_url: Option<String>,

        /// Output directory for prepared data
        #[arg(long, default_value = DEFAULT_COMMON_PHONE_DATA_DIR)]
        out: PathBuf,

        /// Comma-separated ISO-ish language filter, for example eng,fra,spa
        #[arg(long)]
        lang: Option<String>,

        /// Limit utterances for smoke tests
        #[arg(long)]
        max_utterances: Option<usize>,

        /// Target sample rate for mechanical features
        #[arg(long, default_value_t = tongues_common_phone::DEFAULT_SAMPLE_RATE_HZ)]
        sample_rate: u32,

        /// Validation split ratio
        #[arg(long, default_value_t = 0.05)]
        valid_ratio: f64,

        /// Test split ratio
        #[arg(long, default_value_t = 0.05)]
        test_ratio: f64,

        /// Random seed for split shuffling
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },

    /// Train the compact-frame phone and feature-axis CTC scaffold
    Train {
        /// Prepared data directory
        #[arg(long, default_value = DEFAULT_COMMON_PHONE_DATA_DIR)]
        data: PathBuf,

        /// Output directory for the model
        #[arg(long = "model", alias = "out", default_value = DEFAULT_COMMON_PHONE_MODEL_DIR)]
        model: PathBuf,

        /// Training task: frames2phones, frames2features, frames2phonemes, multitask
        #[arg(long, default_value = "frames2phones")]
        task: String,

        /// Maximum training epochs
        #[arg(long)]
        epochs: Option<usize>,

        /// Approximate maximum acoustic frames per batch
        #[arg(long)]
        batch_frames: Option<usize>,

        /// Learning rate
        #[arg(long)]
        lr: Option<f64>,

        /// Random seed
        #[arg(long)]
        seed: Option<u64>,

        /// Device for v0 training; currently only cpu is supported
        #[arg(long, default_value = "cpu")]
        device: String,
    },

    /// Evaluate a Common Phone model
    Eval {
        /// Directory containing the model
        #[arg(long, default_value = DEFAULT_COMMON_PHONE_MODEL_DIR)]
        model: PathBuf,

        /// Prepared data directory
        #[arg(long, default_value = DEFAULT_COMMON_PHONE_DATA_DIR)]
        data: PathBuf,

        /// Eval task: frames2phones, frames2features, frames2phonemes, multitask
        #[arg(long, default_value = "frames2phones")]
        task: String,

        /// Split to evaluate: train, valid, or test
        #[arg(long, default_value = "valid")]
        split: String,

        /// Number of greedy decode samples to include
        #[arg(long, default_value_t = 5)]
        samples: usize,
    },

    /// Print a prepared row and compact feature summary
    #[command(name = "show-row", alias = "show")]
    ShowRow {
        /// Prepared data directory
        #[arg(long, default_value = DEFAULT_COMMON_PHONE_DATA_DIR)]
        data: PathBuf,

        /// Row index in train.jsonl
        #[arg(long, default_value_t = 0)]
        index: usize,
    },
}

#[derive(Subcommand, Debug)]
enum EmotionCommands {
    /// Archive selected default artifacts and recreate empty run directories
    Clean(CleanArgs),

    /// Prepare labeled emotion WAV cuts from a style-vector/source manifest
    Prepare {
        /// TOML config file for the emotion classifier pipeline
        #[arg(long, default_value = "configs/emotions/default.toml")]
        config: PathBuf,

        /// Source JSONL with emotion and path fields; overrides config
        #[arg(long)]
        source_manifest: Option<PathBuf>,

        /// Output directory for prepared data
        #[arg(long, default_value = "datasets/emotions/v0")]
        out: PathBuf,

        /// Random cuts per WAV, not counting the optional full-length cut
        #[arg(long)]
        cuts_per_wav: Option<usize>,

        /// Minimum random cut duration
        #[arg(long)]
        min_cut_ms: Option<u64>,

        /// Maximum random cut duration
        #[arg(long)]
        max_cut_ms: Option<u64>,

        /// Skip the full-length cut for each WAV
        #[arg(long)]
        no_full_cut: bool,

        /// Log-mel bins before mean/std pooling
        #[arg(long)]
        mel_bins: Option<usize>,

        /// Random seed
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Train the emotion classifier
    Train {
        /// Prepared data directory
        #[arg(long, default_value = "datasets/emotions/v0")]
        data: PathBuf,

        /// Output directory for the model
        #[arg(long, default_value = "models/emotions/v0")]
        out: PathBuf,

        /// Maximum training epochs
        #[arg(long)]
        epochs: Option<usize>,

        /// Mini-batch size
        #[arg(long)]
        batch_size: Option<usize>,

        /// Learning rate
        #[arg(long)]
        learning_rate: Option<f32>,

        /// Early stopping patience (epochs with no validation-loss improvement)
        #[arg(long)]
        patience: Option<usize>,

        /// Random seed
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Evaluate an emotion classifier
    Eval {
        /// Directory containing the model
        #[arg(long, default_value = "models/emotions/v0")]
        model: PathBuf,

        /// Prepared data directory
        #[arg(long, default_value = "datasets/emotions/v0")]
        data: PathBuf,

        /// Split to evaluate: train, valid, or test
        #[arg(long, default_value = "test")]
        split: String,
    },

    /// Predict emotion probabilities for one WAV
    Infer {
        /// Directory containing the model
        #[arg(long, default_value = "models/emotions/v0")]
        model: PathBuf,

        /// WAV file to classify
        wav: PathBuf,
    },
}

#[derive(Subcommand, Debug)]
enum WiktionaryCommands {
    /// Archive selected default artifacts and recreate empty run directories
    Clean(CleanArgs),

    /// Download the Wiktionary dump and prepare pronunciation training JSONL
    Prepare {
        /// TOML config file for the Wiktionary pipeline
        #[arg(long, default_value = "configs/wiktionary/default.toml")]
        config: PathBuf,

        /// Existing decompressed MediaWiki XML dump to parse instead of downloading
        #[arg(long)]
        dump: Option<PathBuf>,

        /// Output directory for prepared data
        #[arg(long, default_value = "datasets/wiktionary/enwiktionary-2026-06-01-v0")]
        out: PathBuf,

        /// Cache directory for downloaded Wikimedia dumps
        #[arg(long, default_value = "data/wiktionary")]
        cache_dir: PathBuf,

        /// Override configured languages, e.g. --lang spa --lang fra or --lang spa,fra
        #[arg(long = "lang", value_delimiter = ',')]
        langs: Vec<String>,
    },

    /// Train a Wiktionary pronunciation seq2seq model
    Train {
        /// TOML config file for the Wiktionary pipeline
        #[arg(long, default_value = "configs/wiktionary/default.toml")]
        config: PathBuf,

        /// Existing decompressed MediaWiki XML dump to parse if data is missing
        #[arg(long)]
        dump: Option<PathBuf>,

        /// Prepared data directory
        #[arg(long, default_value = "datasets/wiktionary/enwiktionary-2026-06-01-v0")]
        data: PathBuf,

        /// Override configured languages, e.g. --lang spa --lang fra or --lang spa,fra
        #[arg(long = "lang", value_delimiter = ',')]
        langs: Vec<String>,

        /// Pronunciation notation to train from. Defaults to train_notations in the Wiktionary config.
        #[arg(long, value_enum)]
        notation: Option<WiktionaryNotationArg>,

        /// Wiktionary task mix: orthography-to-phonemes, orthography-to-phones, phonetic-realization, find-etymology, segment-compound, pronounce-segments, verify, normalize-phonology, lang, or all.
        /// Defaults to train_task in the Wiktionary config.
        #[arg(long)]
        task: Option<String>,

        /// Output directory for the model
        #[arg(
            long,
            default_value = "models/wiktionary/enwiktionary-2026-06-01-v0-phones"
        )]
        out: PathBuf,

        /// Cache directory for downloaded Wikimedia dumps if data is missing
        #[arg(long, default_value = "data/wiktionary")]
        cache_dir: PathBuf,

        /// Rebuild prepared split files before training
        #[arg(long)]
        prepare: bool,

        /// Add extra training copies of matching English Dolch sight-word rows.
        /// Enabled by default; pass --sight-words=false to disable.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
        sight_words: bool,

        /// Wait for an in-progress prepare in --data to finish, then start training
        #[arg(long, visible_alias = "while-preparing")]
        wait_for_prepare: bool,

        /// AdamW learning rate
        #[arg(long, default_value_t = 3e-4)]
        learning_rate: f64,

        /// AdamW weight decay
        #[arg(long, default_value_t = 1e-4)]
        weight_decay: f32,

        /// Dropout rate
        #[arg(long, default_value_t = 0.1)]
        dropout: f64,

        /// Mini-batch size
        #[arg(long, default_value_t = 64)]
        batch_size: usize,

        /// Maximum training epochs
        #[arg(long, default_value_t = 20)]
        epochs: usize,

        /// Early stopping patience
        #[arg(long, default_value_t = 5)]
        patience: usize,

        /// Random seed
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },

    /// Run inference with a trained Wiktionary seq2seq model
    Infer {
        /// Directory containing the model
        #[arg(
            long,
            default_value = "models/wiktionary/enwiktionary-2026-06-01-v0-phones"
        )]
        model: PathBuf,

        /// Wiktionary task: orthography-to-phonemes, orthography-to-phones, phonemes-to-orthography, phones-to-orthography, phonetic-realization, find-etymology, segment-compound, pronounce-segments, verify, normalize-phonology, normalize, or a language guessing task
        #[arg(long, default_value = "orthography-to-phones")]
        task: String,

        /// Wiktionary language code used for tagged tasks
        #[arg(long, default_value = "eng")]
        lang: String,

        /// Pronunciation representation used for orthography/phonology tasks
        #[arg(long, value_enum, default_value = "phones")]
        notation: WiktionaryNotationArg,

        /// Optional target pronunciation variety tag
        #[arg(long)]
        variety: Option<String>,

        /// Treat input as the exact model source string, including all control tags
        #[arg(long)]
        raw: bool,

        /// Input orthography, phoneme/phone sequence, or raw tagged source string
        input: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum Styletts2Commands {
    /// Sample random diffusion parameters and speed, and synthesize variants of the parsed text
    Discover {
        /// Text to process
        text: String,

        /// Output directory for synthesized WAVs and metadata
        #[arg(long)]
        out_dir: PathBuf,

        /// Number of samples to generate
        #[arg(long, default_value_t = 10)]
        num_samples: usize,

        /// Model path for the head2phones parser
        #[arg(long, default_value = "models/head2phones/v0")]
        head2phones_model: PathBuf,

        /// Language variety
        #[arg(long, default_value = "en-US")]
        variety: String,

        /// Seed to use for the RNG to generate configurations (for reproducibility)
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Discovery tier (1: diffusion, 2: empirical reference styles, 3: feral randomness)
        #[arg(long, default_value_t = 1)]
        tier: u8,

        /// Directory containing WAV files for tier 2 empirical randomness
        #[arg(long)]
        references_dir: Option<PathBuf>,
    },

    /// Batch-encode reference WAV files into StyleTTS2 style vectors
    EncodeStyle {
        /// Glob or directory containing reference WAV files
        refs: PathBuf,

        /// Output JSONL file for the style vectors
        #[arg(long, default_value = "style_vectors.jsonl")]
        out: PathBuf,

        /// JSONL labels mapping paths to emotions and speakers
        #[arg(long, default_value = "labels.jsonl")]
        labels: PathBuf,
    },

    /// Compute delta signatures from encoded style vectors
    EmotionSignatures {
        /// Input JSONL of encoded style vectors
        style_vectors: PathBuf,

        /// Method for computing signatures
        #[arg(long, default_value = "speaker-neutral-delta")]
        method: String,

        /// Output JSON file path
        #[arg(long, default_value = "emotion_signatures.json")]
        out: PathBuf,
    },
}

#[derive(Args, Debug, Clone)]
struct CleanArgs {
    /// Archive the default prepared dataset directory
    #[arg(long)]
    data: bool,

    /// Archive the default model directory
    #[arg(long)]
    model: bool,

    /// Archive both default dataset and model directories; this is also the default
    #[arg(long)]
    all: bool,

    /// Root directory for archived artifacts
    #[arg(long, default_value = "archive")]
    archive_dir: PathBuf,

    /// Archive run id; defaults to a unix-seconds id
    #[arg(long)]
    run_id: Option<String>,

    /// Do not recreate empty default directories after archiving
    #[arg(long)]
    no_create: bool,
}

impl CleanArgs {
    fn clean_data(&self) -> bool {
        self.all || self.data || (!self.data && !self.model)
    }

    fn clean_model(&self) -> bool {
        self.all || self.model || (!self.data && !self.model)
    }

    fn create_defaults(&self) -> bool {
        !self.no_create
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum WiktionaryNotationArg {
    /// Train from both phonemes.jsonl and phones.jsonl.
    All,
    /// Train from bracket-delimited phonetic rows in phones.jsonl.
    Phones,
    /// Train from slash-delimited phonemic rows in phonemes.jsonl.
    Phonemes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SpeakingDemoFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SpeakingDemoMode {
    Samples,
    Sentences,
    Paragraphs,
}

#[derive(Debug, Clone, ValueEnum)]
enum MaskPolicyArg {
    Single,
    Variable,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SentenceParserTrainingSetArg {
    /// Train on regular seams rows plus mined naive-discrepancy correction rows.
    All,
    /// Train only on rows whose targets come directly from seams sentence boundaries.
    Seams,
    /// Train only on correction rows mined from naive-vs-seams disagreements.
    NaiveDiscrepancy,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RefinementSourceArg {
    /// Mine held-out split examples where model predictions disagree with OpenEPD.
    Discrepancies,
    /// Fine-tune on the built-in Dolch sight-word list using OpenEPD gold pronunciations.
    SightWords,
}

fn cuda_probe_failure_reason() -> Option<String> {
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(|| {
        let device = CudaDevice::default();
        type B = Cuda<f32, i32>;
        let _tensor = burn::tensor::Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &device);
    });
    panic::set_hook(default_hook);

    match result {
        Ok(_) => None,
        Err(payload) => Some(format_panic_payload(payload.as_ref())),
    }
}

fn format_panic_payload(payload: &(dyn Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        (*msg).to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown CUDA initialization failure".to_string()
    }
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    let command = cli.command.unwrap_or_else(|| Commands::G2p2g {
        command: G2p2gCommands::Repl {
            task: "auto".to_string(),
            model: PathBuf::from("models/g2p2g/openepd-v0"),
            data: None,
        },
    });
    let output_mode = OutputMode::for_command(&command, cli.quiet, cli.verbose);
    set_quiet_output(output_mode.quiet);

    // Determine target device (CUDA with fallback to CPU, or forced CPU)
    let cuda_failure = if cli.cpu {
        None
    } else {
        cuda_probe_failure_reason()
    };
    let device_arg = if cli.cpu {
        DeviceArg::Cpu
    } else if cuda_failure.is_none() {
        DeviceArg::Cuda
    } else {
        // Only warn for commands that actually run model computations on the device
        if command_needs_device(&command) && output_mode.verbose() {
            println!(
                "Warning: CUDA is not available ({}). Falling back to CPU.",
                cuda_failure.as_deref().unwrap_or("unknown reason")
            );
        }
        DeviceArg::Cpu
    };

    match command {
        Commands::G2p2g { command } => run_g2p2g_command(command, device_arg, output_mode),
        Commands::SentenceParser { command } => run_sentence_parser_command(command, device_arg),
        Commands::Head2phones { command } => run_head2phones_command(command, device_arg),
        Commands::Styletts2 { command } => {
            styletts2_cmds::run_styletts2_command(command, device_arg)
        }
        Commands::Interpretation { command } => {
            run_interpretation_command(command, device_arg, output_mode)
        }
        Commands::CommonPhone { command } => run_common_phone_command(command),
        Commands::Emotions { command } => run_emotions_command(command),
        Commands::Wiktionary { command } => {
            run_wiktionary_command(command, device_arg, output_mode)
        }
        Commands::FetchCmudict { out } => cmd_fetch_cmudict(&out),
        Commands::FetchLexique { out } => cmd_fetch_lexique(&out),
        Commands::FetchCorpora {
            out_dir,
            corpora,
            list,
        } => fetch_corpora::cmd_fetch_corpora(&out_dir, &corpora, list),
        Commands::Discrepancies {
            out,
            limit,
            max_rarity,
            words,
            words_file,
            no_g2p2g,
            no_wiktionary,
            g2p2g_model,
            wiktionary_model,
            wiktionary_variety,
        } => cmd_discrepancies(
            &out,
            limit,
            max_rarity,
            words,
            words_file.as_deref(),
            !no_g2p2g,
            !no_wiktionary,
            &g2p2g_model,
            &wiktionary_model,
            &wiktionary_variety,
            device_arg,
            output_mode,
        ),
        Commands::Prepare {
            input,
            out,
            train_frac,
            valid_frac,
            seed,
        } => {
            warn_legacy_command("prepare", "g2p2g prepare");
            cmd_prepare(input.as_deref(), &out, train_frac, valid_frac, seed)
        }
        Commands::Train {
            data,
            out,
            mask_policy,
            max_mask_rate,
            span_mask_prob,
            learning_rate,
            weight_decay,
            dropout,
            epochs,
            patience,
            batch_size,
            seed,
            task,
        } => {
            warn_legacy_command("train", "g2p2g train");
            cmd_train(
                &data,
                &out,
                mask_policy,
                max_mask_rate,
                span_mask_prob,
                learning_rate,
                weight_decay,
                dropout,
                epochs,
                patience,
                batch_size,
                seed,
                task,
                device_arg,
            )
        }
        Commands::Eval {
            model,
            split,
            data,
            task,
        } => {
            warn_legacy_command("eval", "g2p2g eval");
            cmd_eval(&model, &split, &data, &task, device_arg)
        }
        Commands::Refine {
            model,
            data,
            out,
            splits,
            source,
            task,
            learning_rate,
            weight_decay,
            epochs,
            patience,
            batch_size,
            seed,
        } => {
            warn_legacy_command("refine", "g2p2g refine");
            cmd_refine(
                &model,
                &data,
                &out,
                &splits,
                source,
                &task,
                learning_rate,
                weight_decay,
                epochs,
                patience,
                batch_size,
                seed,
                output_mode.verbose(),
                device_arg,
            )
        }
        Commands::Predict {
            model,
            input,
            task,
            data,
        } => {
            warn_legacy_command("predict/infer", "g2p2g infer");
            cmd_predict(
                &model,
                &task,
                &input,
                device_arg,
                data.as_deref(),
                output_mode,
            )
        }
        Commands::Repl { model, task, data } => {
            warn_legacy_command("repl", "g2p2g repl");
            cmd_repl(&model, &task, device_arg, data.as_deref())
        }
        Commands::Speak(command) => speak::run_speak(command),
        Commands::Be(command) => cmd_be(command),
        Commands::SpeakingDemo {
            mode,
            varieties,
            format,
        } => cmd_speaking_demo(mode, &varieties, format),
        Commands::Phonemes { text } => cmd_phonemes(&text),
        Commands::Phones { text } => cmd_phones(&text),
        Commands::Models { command } => models::run(command),
    }
}

fn command_needs_device(command: &Commands) -> bool {
    match command {
        Commands::G2p2g { command } => matches!(
            command,
            G2p2gCommands::Train { .. }
                | G2p2gCommands::Eval { .. }
                | G2p2gCommands::Refine { .. }
                | G2p2gCommands::Infer { .. }
                | G2p2gCommands::Repl { .. }
        ),
        Commands::Interpretation { command } => matches!(
            command,
            InterpretationCommands::Train { .. }
                | InterpretationCommands::Eval { .. }
                | InterpretationCommands::Stream { .. }
        ),
        Commands::CommonPhone { .. } => false,
        Commands::SentenceParser { command } => matches!(
            command,
            SentenceParserCommands::Train { .. }
                | SentenceParserCommands::Infer { .. }
                | SentenceParserCommands::Stream { .. }
        ),
        Commands::Head2phones { command } => {
            matches!(
                command,
                Head2PhonesCommands::Train { .. } | Head2PhonesCommands::Infer { .. }
            )
        }
        Commands::Wiktionary { command } => matches!(command, WiktionaryCommands::Train { .. }),
        Commands::Be(command) => !command.mechanical,
        Commands::Train { .. }
        | Commands::Eval { .. }
        | Commands::Refine { .. }
        | Commands::Predict { .. }
        | Commands::Repl { .. }
        | Commands::Discrepancies { .. } => true,
        _ => false,
    }
}

fn command_defaults_to_quiet(command: &Commands) -> bool {
    match command {
        Commands::G2p2g {
            command: G2p2gCommands::Infer { .. },
        }
        | Commands::SentenceParser {
            command: SentenceParserCommands::Infer { .. },
        }
        | Commands::Head2phones {
            command: Head2PhonesCommands::Infer { .. },
        }
        | Commands::SentenceParser {
            command: SentenceParserCommands::Stream { .. },
        }
        | Commands::Interpretation {
            command: InterpretationCommands::Stream { .. },
        }
        | Commands::Wiktionary {
            command: WiktionaryCommands::Infer { .. },
        }
        | Commands::Emotions {
            command: EmotionCommands::Infer { .. },
        }
        | Commands::Predict { .. } => true,
        _ => false,
    }
}

fn warn_legacy_command(old: &str, new: &str) {
    if quiet_output() {
        return;
    }
    eprintln!("warning: `tongues {old}` is deprecated; use `tongues {new}` instead.");
}

fn status_spinner(message: impl Into<String>) -> indicatif::ProgressBar {
    if quiet_output() {
        return indicatif::ProgressBar::hidden();
    }
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_style(
        indicatif::ProgressStyle::with_template("{spinner:.green} {msg}")
            .expect("valid spinner template"),
    );
    pb.enable_steady_tick(Duration::from_millis(120));
    pb.set_message(message.into());
    tongues_core::register_progress_bar(pb)
}

fn finish_status(pb: indicatif::ProgressBar, message: impl AsRef<str>) {
    pb.finish_and_clear();
    if !quiet_output() {
        println!("{}", message.as_ref());
    }
}

fn format_count(value: impl std::fmt::Display) -> String {
    let value = value.to_string();
    let mut grouped = String::with_capacity(value.len() + value.len() / 3);
    let mut digits = 0usize;

    for ch in value.chars().rev() {
        if digits == 3 && ch != '-' {
            grouped.push(',');
            digits = 0;
        }
        grouped.push(ch);
        digits += 1;
    }

    grouped.chars().rev().collect()
}

fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes_f = bytes as f64;
    if bytes_f >= GIB {
        format!("{:.1} GiB", bytes_f / GIB)
    } else if bytes_f >= MIB {
        format!("{:.1} MiB", bytes_f / MIB)
    } else if bytes_f >= KIB {
        format!("{:.1} KiB", bytes_f / KIB)
    } else {
        format!("{} B", format_count(bytes))
    }
}

fn estimate_logits_bytes(batch_size: usize, seq_len: usize, vocab_size: usize) -> u64 {
    let bytes = (batch_size as u128)
        .saturating_mul(seq_len as u128)
        .saturating_mul(vocab_size as u128)
        .saturating_mul(std::mem::size_of::<f32>() as u128);
    bytes.min(u64::MAX as u128) as u64
}

fn has_model_checkpoint(out: &Path, model_path: &Path) -> bool {
    if out.join("train_state.json").exists() || model_path.with_extension("bin").exists() {
        return true;
    }
    let Some(stem) = model_path.file_stem().and_then(|stem| stem.to_str()) else {
        return false;
    };
    fs::read_dir(out)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .any(|entry| {
            let path = entry.path();
            path.extension().and_then(|ext| ext.to_str()) == Some("bin")
                && path
                    .file_stem()
                    .and_then(|file_stem| file_stem.to_str())
                    .map(|file_stem| file_stem.starts_with(&format!("{stem}-epoch-")))
                    .unwrap_or(false)
        })
}

fn counted_progress_style() -> Result<indicatif::ProgressStyle> {
    use std::fmt::Write;

    Ok(indicatif::ProgressStyle::default_bar()
        .template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {human_pos}/{human_len} ({percent}%) ETA {eta_precise} {msg}",
        )?
        .with_key(
            "human_pos",
            |state: &indicatif::ProgressState, w: &mut dyn Write| {
                write!(w, "{}", format_count(state.pos())).expect("write to progress key")
            },
        )
        .with_key(
            "human_len",
            |state: &indicatif::ProgressState, w: &mut dyn Write| {
                let len = state
                    .len()
                    .map(format_count)
                    .unwrap_or_else(|| "?".to_string());
                write!(w, "{len}").expect("write to progress key")
            },
        )
        .progress_chars("#>-"))
}

fn wiktionary_prepare_progress_message(progress: tongues_wiktionary::PrepareProgress) -> String {
    match progress {
        tongues_wiktionary::PrepareProgress::Stage { message } => message,
        tongues_wiktionary::PrepareProgress::Download { path, bytes, .. } => {
            format!("Downloading {} ({})", path, format_bytes(bytes))
        }
        tongues_wiktionary::PrepareProgress::Parse {
            pages,
            patterns,
            phonemes,
            phones,
            etymologies,
            pie_roots,
        } => format!(
            "Parsing dump: {} pages, {} patterns, {} phonemes, {} phones, {} etymologies, {} PIE roots",
            format_count(pages),
            format_count(patterns),
            format_count(phonemes),
            format_count(phones),
            format_count(etymologies),
            format_count(pie_roots)
        ),
        tongues_wiktionary::PrepareProgress::Expand {
            rows,
            examples,
            path,
        } => match path {
            Some(path) => format!(
                "Expanded {} rows into {} examples -> {path}",
                format_count(rows),
                format_count(examples)
            ),
            None => format!(
                "Expanded {} rows into {} examples",
                format_count(rows),
                format_count(examples)
            ),
        },
        tongues_wiktionary::PrepareProgress::Verify {
            model,
            url,
            rows,
            total_rows,
            path,
        } => format!(
            "Asking Ollama model {model} at {url} to scan {}/{} Wiktionary train rows into {}",
            format_count(rows),
            format_count(total_rows),
            path
        ),
        tongues_wiktionary::PrepareProgress::Write { path, rows } => {
            format!("Wrote {} rows to {path}", format_count(rows))
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct G2p2gFileConfig {
    prepare: Option<G2p2gPrepareConfig>,
    train: Option<G2p2gTrainConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct G2p2gPrepareConfig {
    train_frac: Option<f64>,
    valid_frac: Option<f64>,
    seed: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct G2p2gTrainConfig {
    learning_rate: Option<f64>,
    weight_decay: Option<f32>,
    dropout: Option<f64>,
    epochs: Option<usize>,
    patience: Option<usize>,
    batch_size: Option<usize>,
    seed: Option<u64>,
    task: Option<String>,
}

fn read_g2p2g_config(path: &Path) -> Result<G2p2gFileConfig> {
    if !path.exists() {
        return Ok(G2p2gFileConfig::default());
    }
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn read_emotion_prepare_config(path: &Path) -> Result<tongues_emotions::EmotionPrepareConfig> {
    if !path.exists() {
        return Ok(tongues_emotions::EmotionPrepareConfig::default());
    }
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

#[derive(Debug)]
struct CleanTarget {
    kind: &'static str,
    path: PathBuf,
}

fn cmd_clean_family(
    family: &str,
    args: &CleanArgs,
    data_dir: impl Into<PathBuf>,
    model_dir: impl Into<PathBuf>,
) -> Result<()> {
    let mut targets = Vec::new();
    if args.clean_data() {
        targets.push(CleanTarget {
            kind: "dataset",
            path: data_dir.into(),
        });
    }
    if args.clean_model() {
        targets.push(CleanTarget {
            kind: "model",
            path: model_dir.into(),
        });
    }

    let run_id = args.run_id.clone().unwrap_or_else(default_archive_run_id);
    let archive_root = args.archive_dir.join(&run_id);
    let mut moved = 0usize;

    for target in &targets {
        if target.path.exists() {
            let archive_path = unique_archive_path(&archive_root.join(&target.path));
            if let Some(parent) = archive_path.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            fs::rename(&target.path, &archive_path).with_context(|| {
                format!(
                    "moving {} to {}",
                    target.path.display(),
                    archive_path.display()
                )
            })?;
            println!(
                "Archived {} {}: {} -> {}",
                family,
                target.kind,
                target.path.display(),
                archive_path.display()
            );
            moved += 1;
        } else {
            println!(
                "No existing {} {} at {}",
                family,
                target.kind,
                target.path.display()
            );
        }

        if args.create_defaults() {
            fs::create_dir_all(&target.path)
                .with_context(|| format!("creating {}", target.path.display()))?;
            println!(
                "Ready {} {} directory: {}",
                family,
                target.kind,
                target.path.display()
            );
        }
    }

    if moved == 0 {
        println!("Nothing archived for {family}.");
    } else {
        println!("Archive root: {}", archive_root.display());
    }
    Ok(())
}

fn default_archive_run_id() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("run-{seconds}")
}

fn unique_archive_path(path: &Path) -> PathBuf {
    if !path.exists() {
        return path.to_path_buf();
    }

    for index in 1.. {
        let candidate = path.with_file_name(format!(
            "{}-{}",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("artifact"),
            index
        ));
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!("unbounded archive path suffix search should return")
}

fn run_g2p2g_command(
    command: G2p2gCommands,
    device_arg: DeviceArg,
    output_mode: OutputMode,
) -> Result<()> {
    match command {
        G2p2gCommands::Clean(args) => cmd_clean_family(
            "g2p2g",
            &args,
            DEFAULT_G2P2G_DATA_DIR,
            DEFAULT_G2P2G_MODEL_DIR,
        ),
        G2p2gCommands::Prepare {
            config,
            input,
            out,
            train_frac,
            valid_frac,
            seed,
        } => {
            let file_config = read_g2p2g_config(&config)?;
            let prepare = file_config.prepare.unwrap_or_default();
            cmd_prepare(
                input.as_deref(),
                &out,
                train_frac.or(prepare.train_frac).unwrap_or(0.8),
                valid_frac.or(prepare.valid_frac).unwrap_or(0.1),
                seed.or(prepare.seed).unwrap_or(42),
            )
        }
        G2p2gCommands::Train {
            config,
            data,
            out,
            mask_policy,
            max_mask_rate,
            span_mask_prob,
            learning_rate,
            weight_decay,
            dropout,
            epochs,
            patience,
            batch_size,
            seed,
            task,
            wait_for_prepare,
        } => {
            if wait_for_prepare {
                wait_for_prepared_dataset(
                    &data,
                    &["vocab.json", "train.jsonl", "valid.jsonl"],
                    "g2p2g",
                )?;
            }
            let file_config = read_g2p2g_config(&config)?;
            let train = file_config.train.unwrap_or_default();
            cmd_train(
                &data,
                &out,
                mask_policy,
                max_mask_rate,
                span_mask_prob,
                learning_rate.or(train.learning_rate).unwrap_or(3e-4),
                weight_decay.or(train.weight_decay).unwrap_or(1e-4),
                dropout.or(train.dropout).unwrap_or(0.1),
                epochs.or(train.epochs).unwrap_or(20),
                patience.or(train.patience).unwrap_or(5),
                batch_size.or(train.batch_size).unwrap_or(64),
                seed.or(train.seed).unwrap_or(0),
                task.or(train.task).unwrap_or_else(|| "both".to_string()),
                device_arg,
            )
        }
        G2p2gCommands::Eval {
            model,
            split,
            data,
            task,
        } => cmd_eval(&model, &split, &data, &task, device_arg),
        G2p2gCommands::Refine {
            model,
            data,
            out,
            splits,
            source,
            task,
            learning_rate,
            weight_decay,
            epochs,
            patience,
            batch_size,
            seed,
        } => cmd_refine(
            &model,
            &data,
            &out,
            &splits,
            source,
            &task,
            learning_rate,
            weight_decay,
            epochs,
            patience,
            batch_size,
            seed,
            output_mode.verbose(),
            device_arg,
        ),
        G2p2gCommands::Repl { model, task, data } => {
            cmd_repl(&model, &task, device_arg, data.as_deref())
        }
        G2p2gCommands::Infer {
            model,
            input,
            task,
            data,
        } => cmd_predict(
            &model,
            &task,
            &input,
            device_arg,
            data.as_deref(),
            output_mode,
        ),
    }
}

fn read_sentence_parser_config(
    path: &Path,
) -> Result<tongues_sentence_parser::SentenceParserConfig> {
    if !path.exists() {
        return Ok(tongues_sentence_parser::SentenceParserConfig::default());
    }
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn sentence_parser_prepare_progress_message(
    progress: tongues_sentence_parser::PrepareProgress,
) -> String {
    match progress {
        tongues_sentence_parser::PrepareProgress::Stage { message } => message,
        tongues_sentence_parser::PrepareProgress::Discover { files } => {
            format!(
                "Discovered {} sentence-parser source files",
                format_count(files)
            )
        }
        tongues_sentence_parser::PrepareProgress::Download { url, path, bytes } => {
            format!(
                "Downloaded {} from {} -> {}",
                format_bytes(bytes),
                url,
                path
            )
        }
        tongues_sentence_parser::PrepareProgress::Synthesize { path, sentences } => {
            format!(
                "Synthesized {} sentence-boundary cases -> {path}",
                format_count(sentences)
            )
        }
        tongues_sentence_parser::PrepareProgress::Detect {
            path,
            files_done,
            files_total,
            sentences,
            naive_discrepancies,
        } => format!(
            "Detected {} sentences and {} repairs ({}/{}: {path})",
            format_count(sentences),
            format_count(naive_discrepancies),
            format_count(files_done),
            format_count(files_total)
        ),
        tongues_sentence_parser::PrepareProgress::Build {
            sentences,
            examples,
        } => format!(
            "Built {} boundary examples from {} sentences",
            format_count(examples),
            format_count(sentences)
        ),
        tongues_sentence_parser::PrepareProgress::Write { path, rows } => {
            format!("Wrote {} rows to {path}", format_count(rows))
        }
    }
}

fn run_sentence_parser_command(
    command: SentenceParserCommands,
    device_arg: DeviceArg,
) -> Result<()> {
    match command {
        SentenceParserCommands::Clean(args) => cmd_clean_family(
            "sentence-parser",
            &args,
            DEFAULT_SENTENCE_PARSER_DATA_DIR,
            DEFAULT_SENTENCE_PARSER_MODEL_DIR,
        ),
        SentenceParserCommands::Prepare {
            config,
            inputs,
            out,
        } => {
            let mut config = read_sentence_parser_config(&config)?;
            if !inputs.is_empty() {
                config.source_paths = inputs;
            }
            let pb = status_spinner(format!(
                "Preparing sentence-parser dataset at {}",
                out.display()
            ));
            let report = tongues_sentence_parser::prepare_dataset_with_progress(&out, &config, {
                let pb = pb.clone();
                move |progress| {
                    pb.set_message(sentence_parser_prepare_progress_message(progress));
                }
            })?;
            finish_status(
                pb,
                format!(
                    "Prepared sentence-parser dataset at {}: {} train / {} valid / {} test examples from {} sentences in {} files",
                    out.display(),
                    format_count(report.train_examples),
                    format_count(report.valid_examples),
                    format_count(report.test_examples),
                    format_count(report.detected_sentences),
                    format_count(report.source_files)
                ),
            );
            if report.naive_discrepancy_examples > 0 {
                println!(
                    "  included {} naive-vs-seams correction rows",
                    format_count(report.naive_discrepancy_examples)
                );
            }
            Ok(())
        }
        SentenceParserCommands::Train {
            config,
            data,
            inputs,
            out,
            prepare,
            learning_rate,
            weight_decay,
            dropout,
            batch_size,
            epochs,
            patience,
            seed,
            training_set,
            wait_for_prepare,
        } => {
            if wait_for_prepare {
                wait_for_prepared_dataset(
                    &data,
                    &["vocab.json", "train.jsonl", "valid.jsonl"],
                    "sentence-parser",
                )?;
            }
            if prepare
                || !data.join("vocab.json").exists()
                || !data.join("train.jsonl").exists()
                || !data.join("valid.jsonl").exists()
            {
                let mut config_data = read_sentence_parser_config(&config)?;
                if !inputs.is_empty() {
                    config_data.source_paths = inputs;
                }
                let pb = status_spinner(format!(
                    "Preparing sentence-parser dataset at {}",
                    data.display()
                ));
                let report =
                    tongues_sentence_parser::prepare_dataset_with_progress(&data, &config_data, {
                        let pb = pb.clone();
                        move |progress| {
                            pb.set_message(sentence_parser_prepare_progress_message(progress));
                        }
                    })?;
                finish_status(
                    pb,
                    format!(
                        "Prepared sentence-parser dataset at {}: {} train / {} valid / {} test examples from {} sentences in {} files",
                        data.display(),
                        format_count(report.train_examples),
                        format_count(report.valid_examples),
                        format_count(report.test_examples),
                        format_count(report.detected_sentences),
                        format_count(report.source_files)
                    ),
                );
                if report.naive_discrepancy_examples > 0 {
                    println!(
                        "  included {} naive-vs-seams correction rows",
                        format_count(report.naive_discrepancy_examples)
                    );
                }
            }
            let config = read_sentence_parser_config(&config)?;
            cmd_sentence_parser_train(
                &data,
                &out,
                &config,
                learning_rate,
                weight_decay,
                dropout,
                batch_size,
                epochs,
                patience,
                seed,
                training_set,
                device_arg,
            )?;
            Ok(())
        }
        SentenceParserCommands::Eval { model, split } => {
            let manifest_path = model.join(tongues_neural::ARTIFACT_MANIFEST_FILE);
            let manifest = tongues_neural::read_manifest(&manifest_path)?;
            anyhow::ensure!(
                manifest.family == tongues_sentence_parser::FAMILY,
                "expected sentence-parser manifest, found `{}`",
                manifest.family
            );
            println!(
                "Sentence parser artifact is valid for split `{}`: {}",
                split,
                model.display()
            );
            Ok(())
        }
        SentenceParserCommands::Parse { model, text } => {
            let config_path = model.join("model_config.json");
            let lowercase = if config_path.exists() {
                let raw = fs::read_to_string(&config_path)
                    .with_context(|| format!("reading {}", config_path.display()))?;
                let config: tongues_sentence_parser::SentenceParserConfig =
                    serde_json::from_str(&raw)
                        .with_context(|| format!("parsing {}", config_path.display()))?;
                config.lowercase
            } else {
                false
            };
            let analysis = tongues_sentence_parser::parse_sentence(&text, lowercase);
            println!("{}", serde_json::to_string_pretty(&analysis)?);
            Ok(())
        }
        SentenceParserCommands::Infer {
            model,
            previous,
            cursor,
        } => cmd_sentence_parser_infer(&model, &previous, &cursor, device_arg),
        SentenceParserCommands::Stream {
            model,
            repair_control,
        } => cmd_sentence_parser_stream(&model, &repair_control, device_arg),
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_sentence_parser_train(
    data: &Path,
    out: &Path,
    config: &tongues_sentence_parser::SentenceParserConfig,
    learning_rate: f64,
    weight_decay: f32,
    dropout: f64,
    batch_size: usize,
    epochs: usize,
    patience: usize,
    seed: u64,
    training_set: SentenceParserTrainingSetArg,
    device_arg: DeviceArg,
) -> Result<()> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let vocab: Vocab = read_json_file(&data.join("vocab.json"))?;
    let train_rows: Vec<tongues_sentence_parser::BoundaryTrainingExample> =
        read_jsonl_as(&data.join("train.jsonl"))?;
    let valid_rows: Vec<tongues_sentence_parser::BoundaryTrainingExample> =
        read_jsonl_as(&data.join("valid.jsonl"))?;
    let source_filter = sentence_parser_training_source_filter(training_set);
    let train_rows = tongues_sentence_parser::filter_examples_by_source(train_rows, source_filter);
    let valid_rows = tongues_sentence_parser::filter_examples_by_source(valid_rows, source_filter);
    anyhow::ensure!(
        !train_rows.is_empty(),
        "sentence-parser train split is empty after applying training_set={}. Rebuild data with `sentence-parser train --prepare --input <file-or-dir>` or set source_paths in the config",
        sentence_parser_training_set_label(training_set)
    );
    anyhow::ensure!(
        !valid_rows.is_empty(),
        "sentence-parser valid split is empty after applying training_set={}. Rebuild data with `sentence-parser train --prepare --input <file-or-dir>` or set source_paths in the config",
        sentence_parser_training_set_label(training_set)
    );

    let train_examples = tongues_sentence_parser::make_seq2seq_examples(&train_rows, &vocab);
    let valid_examples = tongues_sentence_parser::make_seq2seq_examples(&valid_rows, &vocab);
    let model_config = if out.join("model_config.json").exists() {
        let existing: ModelConfig = read_json_file(&out.join("model_config.json"))?;
        anyhow::ensure!(
            existing.vocab_size == vocab.size(),
            "existing model_config.json vocab_size={} does not match vocab size {}; use a fresh --out directory after rebuilding sentence-parser data",
            existing.vocab_size,
            vocab.size()
        );
        existing
    } else {
        ModelConfig::new(vocab.size()).with_dropout(dropout)
    };
    let train_config = TrainConfig {
        learning_rate,
        weight_decay,
        dropout,
        batch_size,
        epochs,
        early_stopping_patience: patience,
        max_seq_len: model_config.max_seq_len,
        task: None,
        max_frequency_repeat: 1,
        frequency_rarity_cap: 0.0,
    };

    fs::write(
        out.join("model_config.json"),
        serde_json::to_string_pretty(&model_config)?,
    )?;
    fs::write(
        out.join("train_config.json"),
        serde_json::to_string_pretty(&train_config)?,
    )?;
    fs::write(
        out.join("sentence_parser_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    fs::write(
        out.join("vocab.json"),
        serde_json::to_string_pretty(&vocab)?,
    )?;
    fs::write(
        out.join("label_schema.json"),
        serde_json::to_string_pretty(&tongues_sentence_parser::LabelSchema::default())?,
    )?;
    write_manifest(
        out,
        &ModelArtifactManifest::new(
            tongues_sentence_parser::FAMILY,
            tongues_sentence_parser::ARCHITECTURE,
            data_id_from_path(data),
        )
        .with_task("cursor-boundary"),
    )?;

    let model_path = out.join("model");
    println!("Starting sentence-parser seq2seq training...");
    println!(
        "  training_set={} examples={} train / {} valid vocab={} lr={} wd={} dropout={} epochs={} patience={} batch_size={} max_seq_len={}",
        sentence_parser_training_set_label(training_set),
        format_count(train_examples.len()),
        format_count(valid_examples.len()),
        format_count(vocab.size()),
        learning_rate,
        weight_decay,
        dropout,
        format_count(epochs),
        format_count(patience),
        format_count(batch_size),
        format_count(train_config.max_seq_len)
    );
    println!("  train_state: {}", out.join("train_state.json").display());
    println!("  early_stop_metric: val_loss");
    println!(
        "  epoch checkpoints: {}",
        out.join("model-epoch-N.bin").display()
    );
    println!(
        "  best model: {}",
        model_path.with_extension("bin").display()
    );

    let mut rng = StdRng::seed_from_u64(seed);
    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            println!("  device: CPU (ndarray)");
            train_seq2seq_examples::<CpuTrainBackend, _>(
                &model_config,
                &train_config,
                &train_examples,
                &valid_examples,
                &model_path,
                &device,
                &mut rng,
            )?;
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            println!("  device: CUDA GPU");
            train_seq2seq_examples::<CudaTrainBackend, _>(
                &model_config,
                &train_config,
                &train_examples,
                &valid_examples,
                &model_path,
                &device,
                &mut rng,
            )?;
        }
    }
    Ok(())
}

fn sentence_parser_training_source_filter(
    training_set: SentenceParserTrainingSetArg,
) -> Option<tongues_sentence_parser::TrainingRowSource> {
    match training_set {
        SentenceParserTrainingSetArg::All => None,
        SentenceParserTrainingSetArg::Seams => {
            Some(tongues_sentence_parser::TrainingRowSource::Seams)
        }
        SentenceParserTrainingSetArg::NaiveDiscrepancy => {
            Some(tongues_sentence_parser::TrainingRowSource::NaiveDiscrepancy)
        }
    }
}

fn sentence_parser_training_set_label(training_set: SentenceParserTrainingSetArg) -> &'static str {
    match training_set {
        SentenceParserTrainingSetArg::All => "all",
        SentenceParserTrainingSetArg::Seams => "seams",
        SentenceParserTrainingSetArg::NaiveDiscrepancy => "naive-discrepancy",
    }
}

fn read_head2phones_config(path: &Path) -> Result<tongues_head2phones::Head2PhonesConfig> {
    if !path.exists() {
        return Ok(tongues_head2phones::Head2PhonesConfig::default());
    }
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn head2phones_prepare_progress_message(progress: tongues_head2phones::PrepareProgress) -> String {
    match progress {
        tongues_head2phones::PrepareProgress::Stage { message } => message,
        tongues_head2phones::PrepareProgress::Download { url, path, bytes } => {
            format!(
                "Downloaded {} from {} -> {}",
                format_bytes(bytes),
                url,
                path
            )
        }
        tongues_head2phones::PrepareProgress::Read {
            path,
            buffers,
            naive_seams_discrepancies,
        } => format!(
            "Read {} source buffers and {} naive/seams discrepancies from {path}",
            format_count(buffers),
            format_count(naive_seams_discrepancies)
        ),
        tongues_head2phones::PrepareProgress::Synthesize { path, buffers } => {
            format!(
                "Synthesized {} rolling buffers -> {path}",
                format_count(buffers)
            )
        }
        tongues_head2phones::PrepareProgress::Build { complete, no_head } => format!(
            "Built {} complete-head and {} no-head examples",
            format_count(complete),
            format_count(no_head)
        ),
        tongues_head2phones::PrepareProgress::Verify {
            model,
            url,
            rows,
            total_rows,
            path,
        } => format!(
            "Asking Ollama model {model} at {url} to scan {}/{} head2phones train rows into {}",
            format_count(rows),
            format_count(total_rows),
            path
        ),
        tongues_head2phones::PrepareProgress::Write { path, rows } => {
            format!("Wrote {} rows to {path}", format_count(rows))
        }
    }
}

fn apply_head2phones_ollama_overrides(
    config: &mut tongues_head2phones::Head2PhonesConfig,
    enable: bool,
    model: Option<String>,
    url: Option<String>,
    rows: Option<usize>,
    max_chars: Option<usize>,
    strict: bool,
) {
    if enable || model.is_some() || url.is_some() || rows.is_some() || max_chars.is_some() {
        config.verify_with_ollama = true;
    }
    if let Some(model) = model {
        config.ollama_model = model;
    }
    if let Some(url) = url {
        config.ollama_url = url;
    }
    if let Some(rows) = rows {
        config.ollama_verify_rows = rows;
    }
    if let Some(max_chars) = max_chars {
        config.ollama_verify_max_chars = max_chars;
    }
    if strict {
        config.ollama_verify_strict = true;
    }
}

fn print_head2phones_ollama_report(report: &tongues_head2phones::OllamaVerificationReport) {
    let path = report
        .report_path
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "not written".to_string());
    if report.sane {
        println!(
            "Ollama verification passed: model={} rows={} chunks={} report={}",
            report.model,
            format_count(report.rows),
            format_count(report.chunks),
            path
        );
    } else {
        println!(
            "Ollama verification reported an issue: model={} rows={} chunks={} report={}\n{}",
            report.model,
            format_count(report.rows),
            format_count(report.chunks),
            path,
            report
                .issue
                .as_deref()
                .unwrap_or("model reported the data is not sane without a specific issue")
        );
    }
}

fn maybe_print_head2phones_ollama_report(data: &Path) -> Result<()> {
    let path = data.join("ollama_verification.json");
    if path.exists() {
        let report: tongues_head2phones::OllamaVerificationReport = read_json_file(&path)?;
        print_head2phones_ollama_report(&report);
    }
    Ok(())
}

fn run_head2phones_command(command: Head2PhonesCommands, device_arg: DeviceArg) -> Result<()> {
    match command {
        Head2PhonesCommands::Clean(args) => cmd_clean_family(
            "head2phones",
            &args,
            DEFAULT_HEAD2PHONES_DATA_DIR,
            DEFAULT_HEAD2PHONES_MODEL_DIR,
        ),
        Head2PhonesCommands::Prepare {
            config,
            inputs,
            out,
            verify_ollama,
            ollama_model,
            ollama_url,
            ollama_rows,
            ollama_max_chars,
            ollama_strict,
        } => {
            let mut config = read_head2phones_config(&config)?;
            if !inputs.is_empty() {
                config.source_paths = inputs;
            }
            apply_head2phones_ollama_overrides(
                &mut config,
                verify_ollama,
                ollama_model,
                ollama_url,
                ollama_rows,
                ollama_max_chars,
                ollama_strict,
            );
            let pb = status_spinner(format!(
                "Preparing head2phones dataset at {}",
                out.display()
            ));
            let report = tongues_head2phones::prepare_dataset_with_progress(&out, &config, {
                let pb = pb.clone();
                move |progress| pb.set_message(head2phones_prepare_progress_message(progress))
            })?;
            finish_status(
                pb,
                format!(
                    "Prepared head2phones dataset at {}: {} train / {} valid / {} test examples ({} complete, {} no-head, {} repair, {} exceptional, {} naive/seams discrepancies)",
                    out.display(),
                    format_count(report.train_examples),
                    format_count(report.valid_examples),
                    format_count(report.test_examples),
                    format_count(report.complete_examples),
                    format_count(report.no_head_examples),
                    format_count(report.repair_examples),
                    format_count(report.exceptional_examples),
                    format_count(report.naive_seams_discrepancies)
                ),
            );
            if config.verify_with_ollama {
                maybe_print_head2phones_ollama_report(&out)?;
            }
            Ok(())
        }
        Head2PhonesCommands::Verify {
            config,
            data,
            ollama_model,
            ollama_url,
            ollama_rows,
            ollama_max_chars,
            strict,
        } => {
            let mut config = read_head2phones_config(&config)?;
            apply_head2phones_ollama_overrides(
                &mut config,
                true,
                ollama_model,
                ollama_url,
                ollama_rows,
                ollama_max_chars,
                strict,
            );
            let pb = status_spinner(format!(
                "Verifying existing head2phones train rows in {}",
                data.display()
            ));
            let train_path = data.join("train.jsonl");
            let rows: Vec<tongues_head2phones::Head2PhonesTrainingExample> =
                read_jsonl_as(&train_path)?;
            let report_path = data.join("ollama_verification.json");
            let chunks_path = data.join("ollama_verification_chunks.jsonl");
            let report = tongues_head2phones::verify_training_data_with_ollama(
                &config,
                &rows,
                &report_path,
                &chunks_path,
                {
                    let pb = pb.clone();
                    let model = config.ollama_model.clone();
                    let url = config.ollama_url.clone();
                    let total_rows = rows.len();
                    let chunks_path = chunks_path
                        .with_extension("jsonl.part")
                        .display()
                        .to_string();
                    move |scanned_rows| {
                        pb.set_message(format!(
                            "Asking Ollama model {model} at {url} to scan {}/{} head2phones train rows into {}",
                            format_count(scanned_rows),
                            format_count(total_rows),
                            chunks_path
                        ));
                    }
                },
            )
            .with_context(|| format!("verifying {}", train_path.display()))?;
            if report.sane {
                finish_status(
                    pb,
                    format!(
                        "Ollama verification passed for {} head2phones train rows in {} chunks",
                        format_count(report.rows),
                        format_count(report.chunks)
                    ),
                );
            } else {
                finish_status(
                    pb,
                    format!(
                        "Ollama verification found an issue after scanning {} head2phones train rows in {} chunks",
                        format_count(report.rows),
                        format_count(report.chunks)
                    ),
                );
            }
            print_head2phones_ollama_report(&report);
            if strict {
                anyhow::ensure!(
                    report.sane,
                    "Ollama verification failed for {} scanned head2phones training rows: {}",
                    report.rows,
                    report
                        .issue
                        .as_deref()
                        .unwrap_or("model reported the data is not sane without a specific issue")
                );
            }
            Ok(())
        }
        Head2PhonesCommands::Train {
            config,
            data,
            inputs,
            out,
            prepare,
            verify_ollama,
            ollama_model,
            ollama_url,
            ollama_rows,
            ollama_max_chars,
            ollama_strict,
            learning_rate,
            weight_decay,
            dropout,
            batch_size,
            epochs,
            patience,
            seed,
            wait_for_prepare,
        } => {
            if wait_for_prepare {
                wait_for_prepared_dataset(&data, &["train.jsonl", "valid.jsonl"], "head2phones")?;
            }
            if prepare || !data.join("train.jsonl").exists() || !data.join("valid.jsonl").exists() {
                let mut config_data = read_head2phones_config(&config)?;
                if !inputs.is_empty() {
                    config_data.source_paths = inputs;
                }
                apply_head2phones_ollama_overrides(
                    &mut config_data,
                    verify_ollama,
                    ollama_model,
                    ollama_url,
                    ollama_rows,
                    ollama_max_chars,
                    ollama_strict,
                );
                let pb = status_spinner(format!(
                    "Preparing head2phones dataset at {}",
                    data.display()
                ));
                let report =
                    tongues_head2phones::prepare_dataset_with_progress(&data, &config_data, {
                        let pb = pb.clone();
                        move |progress| {
                            pb.set_message(head2phones_prepare_progress_message(progress));
                        }
                    })?;
                finish_status(
                    pb,
                    format!(
                        "Prepared head2phones dataset at {}: {} train / {} valid / {} test examples ({} complete, {} no-head, {} repair, {} exceptional, {} naive/seams discrepancies)",
                        data.display(),
                        format_count(report.train_examples),
                        format_count(report.valid_examples),
                        format_count(report.test_examples),
                        format_count(report.complete_examples),
                        format_count(report.no_head_examples),
                        format_count(report.repair_examples),
                        format_count(report.exceptional_examples),
                        format_count(report.naive_seams_discrepancies)
                    ),
                );
                if config_data.verify_with_ollama {
                    maybe_print_head2phones_ollama_report(&data)?;
                }
            }
            let config = read_head2phones_config(&config)?;
            cmd_head2phones_train(
                &data,
                &out,
                &config,
                learning_rate,
                weight_decay,
                dropout,
                batch_size,
                epochs,
                patience,
                seed,
                device_arg,
            )
        }
        Head2PhonesCommands::Infer {
            model,
            variety,
            buffer,
        } => cmd_head2phones_infer(&model, &variety, &buffer, device_arg),
        Head2PhonesCommands::Eval {
            model,
            data,
            split,
            limit,
            seed,
        } => cmd_head2phones_eval(&model, &data, &split, limit, seed, device_arg),
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_head2phones_train(
    data: &Path,
    out: &Path,
    config: &tongues_head2phones::Head2PhonesConfig,
    learning_rate: f64,
    weight_decay: f32,
    dropout: f64,
    batch_size: usize,
    epochs: usize,
    patience: usize,
    seed: u64,
    device_arg: DeviceArg,
) -> Result<()> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;

    let pb = status_spinner(format!(
        "Loading head2phones train split from {}",
        data.join("train.jsonl").display()
    ));
    let train_rows: Vec<tongues_head2phones::Head2PhonesTrainingExample> =
        read_jsonl_as(&data.join("train.jsonl"))?;
    pb.set_message(format!(
        "Loaded {} train rows; loading valid split from {}",
        format_count(train_rows.len()),
        data.join("valid.jsonl").display()
    ));
    let valid_rows: Vec<tongues_head2phones::Head2PhonesTrainingExample> =
        read_jsonl_as(&data.join("valid.jsonl"))?;
    pb.set_message(format!(
        "Loaded {} valid rows; checking test split at {}",
        format_count(valid_rows.len()),
        data.join("test.jsonl").display()
    ));
    let test_rows: Vec<tongues_head2phones::Head2PhonesTrainingExample> =
        if data.join("test.jsonl").exists() {
            read_jsonl_as(&data.join("test.jsonl"))?
        } else {
            Vec::new()
        };
    anyhow::ensure!(!train_rows.is_empty(), "head2phones train split is empty");
    anyhow::ensure!(!valid_rows.is_empty(), "head2phones valid split is empty");
    finish_status(
        pb,
        format!(
            "Loaded head2phones rows from {}: {} train / {} valid / {} test",
            data.display(),
            format_count(train_rows.len()),
            format_count(valid_rows.len()),
            format_count(test_rows.len())
        ),
    );

    let pb = status_spinner(format!(
        "Building head2phones vocab from prepared rows in {}",
        data.display()
    ));
    let data_vocab_path = data.join("vocab.json");
    let prepared_vocab = read_json_file::<Vocab>(&data_vocab_path).ok();
    let vocab = tongues_head2phones::build_vocab_from_examples(
        train_rows
            .iter()
            .chain(valid_rows.iter())
            .chain(test_rows.iter()),
    );
    match prepared_vocab.as_ref() {
        Some(prepared_vocab) if prepared_vocab.size() != vocab.size() => {
            println!(
                "Rebuilt compact head2phones vocab from prepared rows: {} -> {} tokens",
                format_count(prepared_vocab.size()),
                format_count(vocab.size())
            );
            fs::write(&data_vocab_path, serde_json::to_string_pretty(&vocab)?)
                .with_context(|| format!("writing {}", data_vocab_path.display()))?;
        }
        None => {
            println!(
                "Built head2phones vocab from prepared rows: {} tokens",
                format_count(vocab.size())
            );
            fs::write(&data_vocab_path, serde_json::to_string_pretty(&vocab)?)
                .with_context(|| format!("writing {}", data_vocab_path.display()))?;
        }
        _ => {}
    }
    finish_status(
        pb,
        format!(
            "Ready head2phones vocab at {}: {} tokens",
            data_vocab_path.display(),
            format_count(vocab.size())
        ),
    );

    let pb = status_spinner(format!(
        "Converting {} train / {} valid head2phones rows into seq2seq examples",
        format_count(train_rows.len()),
        format_count(valid_rows.len())
    ));
    let train_examples = tongues_head2phones::make_seq2seq_examples(&train_rows, &vocab);
    pb.set_message(format!(
        "Converted {} train examples; converting {} valid rows",
        format_count(train_examples.len()),
        format_count(valid_rows.len())
    ));
    let valid_examples = tongues_head2phones::make_seq2seq_examples(&valid_rows, &vocab);
    finish_status(
        pb,
        format!(
            "Built head2phones seq2seq examples: {} train / {} valid",
            format_count(train_examples.len()),
            format_count(valid_examples.len())
        ),
    );
    let model_path = out.join("model");
    let model_config = if out.join("model_config.json").exists() {
        let existing: ModelConfig = read_json_file(&out.join("model_config.json"))?;
        if existing.vocab_size == vocab.size() {
            existing
        } else {
            anyhow::ensure!(
                !has_model_checkpoint(out, &model_path),
                "existing model_config.json vocab_size={} does not match compact vocab size {}; use a fresh --out directory or remove the existing head2phones checkpoints",
                existing.vocab_size,
                vocab.size()
            );
            println!(
                "Replacing stale head2phones model config without checkpoints: vocab_size {} -> {}",
                format_count(existing.vocab_size),
                format_count(vocab.size())
            );
            ModelConfig::new(vocab.size())
                .with_dropout(dropout)
                .with_max_seq_len(256)
        }
    } else {
        ModelConfig::new(vocab.size())
            .with_dropout(dropout)
            .with_max_seq_len(256)
    };
    let train_config = TrainConfig {
        learning_rate,
        weight_decay,
        dropout,
        batch_size,
        epochs,
        early_stopping_patience: patience,
        max_seq_len: model_config.max_seq_len,
        task: None,
        max_frequency_repeat: DEFAULT_MAX_FREQUENCY_REPEAT,
        frequency_rarity_cap: DEFAULT_FREQUENCY_RARITY_CAP,
    };

    let pb = status_spinner(format!(
        "Writing head2phones training metadata into {}",
        out.display()
    ));
    fs::write(
        out.join("model_config.json"),
        serde_json::to_string_pretty(&model_config)?,
    )?;
    fs::write(
        out.join("train_config.json"),
        serde_json::to_string_pretty(&train_config)?,
    )?;
    fs::write(
        out.join("head2phones_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    fs::write(
        out.join("vocab.json"),
        serde_json::to_string_pretty(&vocab)?,
    )?;
    write_manifest(
        out,
        &ModelArtifactManifest::new(
            tongues_head2phones::FAMILY,
            tongues_head2phones::ARCHITECTURE,
            data_id_from_path(data),
        )
        .with_task("head-chunk-to-phones"),
    )?;
    finish_status(
        pb,
        format!(
            "Wrote head2phones model metadata and vocab into {}",
            out.display()
        ),
    );

    println!("Starting head2phones seq2seq training...");
    println!(
        "  examples={} train / {} valid vocab={} lr={} wd={} dropout={} epochs={} patience={} batch_size={} max_seq_len={}",
        format_count(train_examples.len()),
        format_count(valid_examples.len()),
        format_count(vocab.size()),
        learning_rate,
        weight_decay,
        dropout,
        format_count(epochs),
        format_count(patience),
        format_count(batch_size),
        format_count(train_config.max_seq_len)
    );
    let logits_bytes = estimate_logits_bytes(
        batch_size,
        train_config.max_seq_len,
        model_config.vocab_size,
    );
    println!(
        "  estimated_logits_memory_per_batch: {}",
        format_bytes(logits_bytes)
    );
    if matches!(device_arg, DeviceArg::Cuda) && logits_bytes >= 2 * 1024 * 1024 * 1024 {
        println!(
            "  warning: large CUDA logits allocation; lower --batch-size if training hits GPU memory errors"
        );
    }
    println!("  train_state: {}", out.join("train_state.json").display());
    println!("  early_stop_metric: val_loss");
    println!(
        "  epoch checkpoints: {}",
        out.join("model-epoch-N.bin").display()
    );
    println!(
        "  best model: {}",
        model_path.with_extension("bin").display()
    );

    let mut rng = StdRng::seed_from_u64(seed);
    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            println!("  device: CPU (ndarray)");
            train_seq2seq_examples::<CpuTrainBackend, _>(
                &model_config,
                &train_config,
                &train_examples,
                &valid_examples,
                &model_path,
                &device,
                &mut rng,
            )?;
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            println!("  device: CUDA GPU");
            train_seq2seq_examples::<CudaTrainBackend, _>(
                &model_config,
                &train_config,
                &train_examples,
                &valid_examples,
                &model_path,
                &device,
                &mut rng,
            )?;
        }
    }
    Ok(())
}

fn cmd_head2phones_infer(
    model_dir: &Path,
    variety: &str,
    buffer: &str,
    device_arg: DeviceArg,
) -> Result<()> {
    let manifest =
        tongues_neural::read_manifest(&model_dir.join(tongues_neural::ARTIFACT_MANIFEST_FILE))?;
    anyhow::ensure!(
        manifest.family == tongues_head2phones::FAMILY,
        "expected head2phones manifest, found `{}`",
        manifest.family
    );
    let model_config: ModelConfig = read_json_file(&model_dir.join("model_config.json"))?;
    let vocab: Vocab = read_json_file(&model_dir.join("vocab.json"))?;
    let input = tongues_head2phones::format_input_for_variety(variety, buffer);
    let input_len = vocab.encode_string(&input).len();
    anyhow::ensure!(
        input_len <= model_config.max_seq_len,
        "head2phones input encodes to {} tokens, exceeding model max_seq_len={}",
        input_len,
        model_config.max_seq_len
    );
    let output = match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            let model =
                load_model::<CpuInferBackend>(&model_config, &model_dir.join("model"), &device)?;
            predict_sentence_boundary(&model, &input, &vocab, &device)
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            let model =
                load_model::<CudaInferBackend>(&model_config, &model_dir.join("model"), &device)?;
            predict_sentence_boundary(&model, &input, &vocab, &device)
        }
    };
    println!("{output}");
    Ok(())
}

fn cmd_be(command: BeCommand) -> Result<()> {
    let device = NdArrayDevice::Cpu;
    let head2phones = if command.mechanical {
        None
    } else {
        let manifest = tongues_neural::read_manifest(
            &command
                .head2phones_model
                .join(tongues_neural::ARTIFACT_MANIFEST_FILE),
        )?;
        anyhow::ensure!(
            manifest.family == tongues_head2phones::FAMILY,
            "expected head2phones manifest, found `{}`",
            manifest.family
        );
        let model_config: ModelConfig =
            read_json_file(&command.head2phones_model.join("model_config.json"))?;
        let vocab: Vocab = read_json_file(&command.head2phones_model.join("vocab.json"))?;
        let model = load_model::<CpuInferBackend>(
            &model_config,
            &command.head2phones_model.join("model"),
            &device,
        )?;
        Some((model, vocab, model_config))
    };
    let seams_detector = if command.mechanical {
        Some(seams::SentenceDetectorDialog::new().context("initializing seams detector")?)
    } else {
        None
    };

    eprintln!(
        "be: ollama={} model={} prompt={:?}",
        command.ollama_url, command.ollama_model, command.prompt
    );

    match command.voice_backend {
        BeVoiceBackend::Onnx => cmd_be_with_onnx(command, head2phones, seams_detector, &device),
        BeVoiceBackend::Styletts2 => {
            cmd_be_with_styletts2(command, head2phones, seams_detector, &device)
        }
    }
}

fn cmd_be_with_onnx(
    command: BeCommand,
    head2phones: Option<(Seq2SeqModel<CpuInferBackend>, Vocab, ModelConfig)>,
    seams_detector: Option<seams::SentenceDetectorDialog>,
    device: &NdArrayDevice,
) -> Result<()> {
    let voice_model = models::ensure_voice_model_available()?;
    let voice_config_path = speech::voice_config_path(&voice_model);
    let voice_config = speech::VoiceConfig::from_json_file(&voice_config_path)?;
    let mut speech_backend = speech::OnnxSpeechBackend::load(&voice_model, voice_config)?;
    let player = speak::AudioStreamPlayer::new(speech_backend.sample_rate_hz())
        .context("failed to start CPAL playback")?;
    log_be_speech_path(&command, "onnx", &voice_model);
    eprintln!("be: cpal output={}", player.description());

    let mut total_samples = 0usize;
    let mut sink = |chunk: speech::AudioChunk| -> Result<()> {
        total_samples += chunk.pcm_mono_f32.len();
        eprintln!(
            "be: queued audio chunk samples={} total={} rate={}Hz",
            chunk.pcm_mono_f32.len(),
            total_samples,
            chunk.sample_rate_hz
        );
        player.append(&chunk.pcm_mono_f32);
        Ok(())
    };

    stream_be_sentences(&command, seams_detector.as_ref(), |sentence| {
        if let Some((head2phones, vocab, model_config)) = head2phones.as_ref() {
            speak_head2phones_sentence(
                sentence,
                &command.variety,
                head2phones,
                vocab,
                model_config,
                device,
                &mut speech_backend,
                &mut sink,
            )?;
        } else {
            synthesize_mechanical_sentence(
                sentence,
                &command.variety,
                &mut speech_backend,
                &mut sink,
            )?;
        }
        Ok(())
    })?;

    eprintln!();
    drop(sink);
    eprintln!("be: waiting for CPAL playback to drain ({total_samples} queued samples)");
    player.wait_until_done(total_samples);
    Ok(())
}

fn cmd_be_with_styletts2(
    command: BeCommand,
    head2phones: Option<(Seq2SeqModel<CpuInferBackend>, Vocab, ModelConfig)>,
    seams_detector: Option<seams::SentenceDetectorDialog>,
    device: &NdArrayDevice,
) -> Result<()> {
    let primary_model = models::ensure_styletts2_model_available()?;
    let model_dir = primary_model
        .parent()
        .context("StyleTTS2 primary model path has no parent directory")?;
    let default_refs = models::ensure_styletts2_default_reference_audio_available()?;
    let diffusion_opts = StyleTts2DiffusionOptions {
        diffusion_steps: 5,
        alpha: 0.3,
        beta: 0.1,
        embedding_scale: 1.0,
        seed: 0,
    };
    let mut backend = StyleTts2OnnxBackend::from_model_dir(model_dir)
        .context("failed to load native StyleTTS2 ONNX backend")?
        .with_diffusion_options(diffusion_opts)
        .context("invalid StyleTTS2 diffusion options")?;
    let player = speak::AudioStreamPlayer::new(24_000).context("failed to start CPAL playback")?;
    log_be_speech_path(&command, "styletts2", &primary_model);
    eprintln!(
        "be: styletts2 refs voice={} style={}",
        default_refs.voice.display(),
        default_refs.style.display()
    );
    eprintln!("be: cpal output={}", player.description());

    let mut total_samples = 0usize;
    let mut cursor = String::new();
    let mut previous = String::new();
    let mut sink = |chunk: styletts2::StyleTts2AudioChunk| -> std::result::Result<
        (),
        styletts2::StyleTts2Error,
    > {
        total_samples += chunk.pcm_mono_f32.len();
        eprintln!(
            "be: queued audio chunk samples={} total={} rate={}Hz",
            chunk.pcm_mono_f32.len(),
            total_samples,
            chunk.sample_rate_hz
        );
        player.append(&chunk.pcm_mono_f32);
        Ok(())
    };

    stream_be_sentences_with_buffers(
        &command,
        seams_detector.as_ref(),
        &mut cursor,
        &mut previous,
        |sentence| {
            if let Some((head2phones, vocab, model_config)) = head2phones.as_ref() {
                speak_head2phones_sentence_styletts2(
                    sentence,
                    &command.variety,
                    head2phones,
                    vocab,
                    model_config,
                    device,
                    &mut backend,
                    &mut sink,
                    &default_refs.voice,
                    &default_refs.style,
                    command.max_tts_symbols,
                    command.no_tts_chunking,
                )?;
            } else {
                synthesize_mechanical_sentence_styletts2(
                    sentence,
                    &command.variety,
                    &mut backend,
                    &mut sink,
                    &default_refs.voice,
                    &default_refs.style,
                    command.max_tts_symbols,
                    command.no_tts_chunking,
                )?;
            }
            Ok(())
        },
    )?;

    eprintln!();
    drop(sink);
    eprintln!("be: waiting for CPAL playback to drain ({total_samples} queued samples)");
    player.wait_until_done(total_samples);
    Ok(())
}

fn log_be_speech_path(command: &BeCommand, backend_label: &str, model_path: &Path) {
    if command.mechanical {
        eprintln!(
            "be: mechanical seams+phonemicizer variety={} voice_backend={} model={}",
            command.variety,
            backend_label,
            model_path.display()
        );
    } else {
        eprintln!(
            "be: resident head2phones CPU model={} voice_backend={} model={}",
            command.head2phones_model.display(),
            backend_label,
            model_path.display()
        );
    }
}

fn stream_be_sentences(
    command: &BeCommand,
    seams_detector: Option<&seams::SentenceDetectorDialog>,
    mut on_sentence: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    let mut cursor = String::new();
    let mut previous = String::new();
    stream_be_sentences_with_buffers(
        command,
        seams_detector,
        &mut cursor,
        &mut previous,
        &mut on_sentence,
    )
}

fn stream_be_sentences_with_buffers(
    command: &BeCommand,
    seams_detector: Option<&seams::SentenceDetectorDialog>,
    cursor: &mut String,
    previous: &mut String,
    mut on_sentence: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    stream_ollama_generate(command, |piece| {
        print!("{piece}");
        std::io::stdout()
            .flush()
            .context("flushing streamed text")?;
        cursor.push_str(piece);
        let sentences = if let Some(detector) = seams_detector {
            collect_completed_seams_prefixes(cursor, previous, detector)?
        } else {
            collect_completed_sentence_parser_prefixes(cursor, previous)
        };
        for sentence in sentences {
            on_sentence(&sentence)?;
        }
        Ok(())
    })?;

    let tail = cursor.split_whitespace().collect::<Vec<_>>().join(" ");
    if !tail.is_empty() {
        on_sentence(&tail)?;
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateStreamResponse {
    #[serde(default)]
    response: String,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    error: Option<String>,
}

fn stream_ollama_generate(
    command: &BeCommand,
    mut on_piece: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    let url = format!(
        "{}/api/generate",
        command.ollama_url.trim().trim_end_matches('/')
    );
    let body = serde_json::to_string(&serde_json::json!({
        "model": command.ollama_model,
        "prompt": command.prompt,
        "stream": true,
        "think": false,
        "options": {
            "temperature": 0.8
        }
    }))?;
    let response = ureq::post(&url)
        .header("Content-Type", "application/json")
        .config()
        .http_status_as_error(false)
        .build()
        .send(body)
        .with_context(|| format!("POST {url}"))?;
    let status = response.status();
    anyhow::ensure!(status.is_success(), "POST {url} returned HTTP {status}");

    let mut body = response.into_body();
    let reader = std::io::BufReader::new(body.as_reader());
    for line in reader.lines() {
        let line = line.with_context(|| format!("reading Ollama stream from {url}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let event: OllamaGenerateStreamResponse =
            serde_json::from_str(&line).with_context(|| format!("parsing Ollama event: {line}"))?;
        if let Some(error) = event.error {
            anyhow::bail!("Ollama returned an error: {error}");
        }
        if !event.response.is_empty() {
            on_piece(&event.response)?;
        }
        if event.done {
            break;
        }
    }
    Ok(())
}

fn speak_head2phones_sentence<B: Backend>(
    sentence: &str,
    variety: &str,
    head2phones: &Seq2SeqModel<B>,
    vocab: &Vocab,
    model_config: &ModelConfig,
    device: &B::Device,
    speech_backend: &mut speech::OnnxSpeechBackend,
    sink: &mut dyn speech::AudioSink,
) -> Result<()> {
    let mut remaining = sentence.trim().to_string();
    let mut iterations = 0usize;
    while !remaining.is_empty() {
        iterations += 1;
        anyhow::ensure!(
            iterations <= 256,
            "head2phones did not consume sentence after {iterations} heads: {:?}",
            sentence
        );

        let next = speak_head2phones_head(
            &remaining,
            variety,
            head2phones,
            vocab,
            model_config,
            device,
            speech_backend,
            sink,
        )?;
        match next {
            Some(rest) => {
                let rest = rest.trim_start();
                if rest.is_empty() {
                    break;
                }
                anyhow::ensure!(
                    rest.len() < remaining.len(),
                    "head2phones split did not advance for {:?}",
                    remaining
                );
                remaining = rest.to_string();
            }
            None => break,
        }
    }
    Ok(())
}

fn speak_head2phones_head<B: Backend>(
    sentence: &str,
    variety: &str,
    head2phones: &Seq2SeqModel<B>,
    vocab: &Vocab,
    model_config: &ModelConfig,
    device: &B::Device,
    speech_backend: &mut speech::OnnxSpeechBackend,
    sink: &mut dyn speech::AudioSink,
) -> Result<Option<String>> {
    let input = tongues_head2phones::format_input_for_variety(variety, sentence);
    let input_len = vocab.encode_string(&input).len();
    if input_len > model_config.max_seq_len {
        let chunks = split_long_sentence(sentence, variety, vocab, model_config.max_seq_len)?;
        if chunks.len() > 1 || (chunks.len() == 1 && chunks[0] != sentence) {
            for chunk in chunks {
                speak_head2phones_sentence(
                    &chunk,
                    variety,
                    head2phones,
                    vocab,
                    model_config,
                    device,
                    speech_backend,
                    sink,
                )?;
            }
            return Ok(None);
        }
    }

    anyhow::ensure!(
        input_len <= model_config.max_seq_len,
        "head2phones input encodes to {} tokens, exceeding model max_seq_len={}",
        input_len,
        model_config.max_seq_len
    );
    let output = predict_sentence_boundary(head2phones, &input, vocab, device);
    let Some(prediction) = extract_head2phones_prediction(&output) else {
        let fallback = clean_be_sentence_for_fallback(sentence);
        eprintln!(
            "\nbe: head={:?}\nbe: head2phones={} ; using fallback={:?}",
            sentence,
            compact_display(&output, 160),
            fallback
        );
        if !fallback.is_empty() {
            synthesize_rule_based_fallback(&fallback, variety, speech_backend, sink)?;
        }
        return Ok(None);
    };
    let (head, rest) = head2phones_head_and_rest(sentence, prediction.split_after);
    let head = head.trim();
    let rest = rest.to_string();
    let sequence = voice_sequence_from_head2phones_phones(&prediction.phones);
    eprintln!(
        "\nbe: head={:?}\nbe: phones={}\nbe: voice={}",
        head,
        prediction.phones,
        sequence.symbols.join(" ")
    );
    if sequence.symbols.is_empty() {
        synthesize_rule_based_fallback(head, variety, speech_backend, sink)?;
        return Ok(nonempty_remainder(rest));
    }
    for chunk in speech::synthesis_chunks_from_sequence(sequence) {
        let ids = chunk
            .sequence
            .to_text_ids_compatible(speech_backend.voice_config())?;
        let mut audio = speech_backend.synthesize_ids(&ids)?.pcm_mono_f32;
        audio.extend(std::iter::repeat(0.0).take(
            (speech_backend.sample_rate_hz() as usize * chunk.pause_after_ms as usize) / 1000,
        ));
        sink.emit(speech::AudioChunk {
            chunk_index: 0,
            is_final: true,
            // The pause is already materialized in the emitted PCM.
            pause_after_ms: 0,
            sample_rate_hz: speech_backend.sample_rate_hz(),
            pcm_mono_f32: audio,
        })?;
    }
    Ok(nonempty_remainder(rest))
}

fn speak_head2phones_sentence_styletts2<B: Backend>(
    sentence: &str,
    variety: &str,
    head2phones: &Seq2SeqModel<B>,
    vocab: &Vocab,
    model_config: &ModelConfig,
    device: &B::Device,
    backend: &mut StyleTts2OnnxBackend,
    sink: &mut dyn styletts2::StyleTts2AudioSink,
    voice_ref: &Path,
    style_ref: &Path,
    max_tts_symbols: usize,
    no_tts_chunking: bool,
) -> Result<()> {
    let mut remaining = sentence.trim().to_string();
    let mut iterations = 0usize;
    while !remaining.is_empty() {
        iterations += 1;
        anyhow::ensure!(
            iterations <= 256,
            "head2phones did not consume sentence after {iterations} heads: {:?}",
            sentence
        );

        let next = speak_head2phones_head_styletts2(
            &remaining,
            variety,
            head2phones,
            vocab,
            model_config,
            device,
            backend,
            sink,
            voice_ref,
            style_ref,
            max_tts_symbols,
            no_tts_chunking,
        )?;
        match next {
            Some(rest) => {
                let rest = rest.trim_start();
                if rest.is_empty() {
                    break;
                }
                anyhow::ensure!(
                    rest.len() < remaining.len(),
                    "head2phones split did not advance for {:?}",
                    remaining
                );
                remaining = rest.to_string();
            }
            None => break,
        }
    }
    Ok(())
}

fn speak_head2phones_head_styletts2<B: Backend>(
    sentence: &str,
    variety: &str,
    head2phones: &Seq2SeqModel<B>,
    vocab: &Vocab,
    model_config: &ModelConfig,
    device: &B::Device,
    backend: &mut StyleTts2OnnxBackend,
    sink: &mut dyn styletts2::StyleTts2AudioSink,
    voice_ref: &Path,
    style_ref: &Path,
    max_tts_symbols: usize,
    no_tts_chunking: bool,
) -> Result<Option<String>> {
    let input = tongues_head2phones::format_input_for_variety(variety, sentence);
    let input_len = vocab.encode_string(&input).len();
    if input_len > model_config.max_seq_len {
        let chunks = split_long_sentence(sentence, variety, vocab, model_config.max_seq_len)?;
        if chunks.len() > 1 || (chunks.len() == 1 && chunks[0] != sentence) {
            for chunk in chunks {
                speak_head2phones_sentence_styletts2(
                    &chunk,
                    variety,
                    head2phones,
                    vocab,
                    model_config,
                    device,
                    backend,
                    sink,
                    voice_ref,
                    style_ref,
                    max_tts_symbols,
                    no_tts_chunking,
                )?;
            }
            return Ok(None);
        }
    }

    anyhow::ensure!(
        input_len <= model_config.max_seq_len,
        "head2phones input encodes to {} tokens, exceeding model max_seq_len={}",
        input_len,
        model_config.max_seq_len
    );
    let output = predict_sentence_boundary(head2phones, &input, vocab, device);
    let Some(prediction) = extract_head2phones_prediction(&output) else {
        let fallback = clean_be_sentence_for_fallback(sentence);
        eprintln!(
            "\nbe: head={:?}\nbe: head2phones={} ; using fallback={:?}",
            sentence,
            compact_display(&output, 160),
            fallback
        );
        if !fallback.is_empty() {
            synthesize_mechanical_sentence_styletts2(
                &fallback,
                variety,
                backend,
                sink,
                voice_ref,
                style_ref,
                max_tts_symbols,
                no_tts_chunking,
            )?;
        }
        return Ok(None);
    };

    let (head, rest) = head2phones_head_and_rest(sentence, prediction.split_after);
    let head = head.trim();
    let rest = rest.to_string();
    let plan = styletts2_plan_from_head2phones_prediction(variety, head, &prediction.phones)?;
    let styletts2_plan = prepare_styletts2_plan(
        &plan,
        &styletts2_en_us_symbol_set(),
        be_styletts2_options(max_tts_symbols, no_tts_chunking),
    )
    .context("failed to prepare StyleTTS2 plan from head2phones output")?;
    let backend_symbols = styletts2_plan
        .chunks
        .iter()
        .map(|chunk| styletts2_text_for_symbols(&chunk.symbols))
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("failed to format StyleTTS2 symbols")?
        .join(" || ");
    eprintln!(
        "\nbe: head={:?}\nbe: phones={}\nbe: styletts2={}",
        head, prediction.phones, backend_symbols
    );
    if styletts2_plan
        .chunks
        .iter()
        .all(|chunk| chunk.symbols.is_empty())
    {
        synthesize_mechanical_sentence_styletts2(
            head,
            variety,
            backend,
            sink,
            voice_ref,
            style_ref,
            max_tts_symbols,
            no_tts_chunking,
        )?;
        return Ok(nonempty_remainder(rest));
    }
    synthesize_styletts2_plan(backend, sink, plan, voice_ref, style_ref, styletts2_plan)?;
    Ok(nonempty_remainder(rest))
}

fn split_long_sentence(
    sentence: &str,
    variety: &str,
    vocab: &Vocab,
    max_seq_len: usize,
) -> Result<Vec<String>> {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    if words.is_empty() {
        return Ok(Vec::new());
    }

    let mut best_i = None;

    // Priorities:
    // 1: Semicolon, Colon, Dash/Em-dash (ending with ';', ':', '—', '--')
    // 2: Comma (ending with ',')
    // 3: Conjunctions ('and', 'but', 'or', 'so', 'because', 'when', 'although', 'while', 'if', 'then')
    // 4: Any space
    for priority in 1..=4 {
        for i in (1..words.len()).rev() {
            let prefix = words[..i].join(" ");
            let input = tongues_head2phones::format_input_for_variety(variety, &prefix);
            if vocab.encode_string(&input).len() <= max_seq_len {
                let matches_priority = match priority {
                    1 => {
                        let last_word = words[i - 1];
                        last_word.ends_with(';')
                            || last_word.ends_with(':')
                            || last_word.ends_with('—')
                            || last_word.ends_with("--")
                    }
                    2 => words[i - 1].ends_with(','),
                    3 => {
                        let w_prev = words[i - 1].to_lowercase();
                        let w_curr = words[i].to_lowercase();
                        let conj = |w: &str| {
                            matches!(
                                w.trim_matches(|c: char| !c.is_alphabetic()),
                                "and"
                                    | "but"
                                    | "or"
                                    | "so"
                                    | "because"
                                    | "when"
                                    | "although"
                                    | "while"
                                    | "if"
                                    | "then"
                            )
                        };
                        conj(&w_prev) || conj(&w_curr)
                    }
                    4 => true,
                    _ => unreachable!(),
                };
                if matches_priority {
                    best_i = Some(i);
                    break;
                }
            }
        }
        if best_i.is_some() {
            break;
        }
    }

    if let Some(i) = best_i {
        let first_half = words[..i].join(" ");
        let second_half = words[i..].join(" ");
        Ok(vec![first_half, second_half])
    } else {
        // Fallback: split in half by character length
        let mid = sentence.len() / 2;
        let mut split_idx = mid;
        while !sentence.is_char_boundary(split_idx) && split_idx > 0 {
            split_idx -= 1;
        }
        if split_idx == 0 {
            split_idx = mid;
            while !sentence.is_char_boundary(split_idx) && split_idx < sentence.len() {
                split_idx += 1;
            }
        }
        if split_idx > 0 && split_idx < sentence.len() {
            Ok(vec![
                sentence[..split_idx].to_string(),
                sentence[split_idx..].to_string(),
            ])
        } else {
            Ok(vec![sentence.to_string()])
        }
    }
}

fn synthesize_rule_based_fallback(
    sentence: &str,
    variety: &str,
    speech_backend: &mut speech::OnnxSpeechBackend,
    sink: &mut dyn speech::AudioSink,
) -> Result<()> {
    synthesize_mechanical_sentence(sentence, variety, speech_backend, sink)
}

fn synthesize_mechanical_sentence(
    sentence: &str,
    variety: &str,
    speech_backend: &mut speech::OnnxSpeechBackend,
    sink: &mut dyn speech::AudioSink,
) -> Result<()> {
    let variety = speaking::VarietyId(variety.to_string());
    let phonemicizer = speaking::phonemicizer_for_variety(&variety)
        .map_err(|error| anyhow::anyhow!("failed to load phonemicizer: {error}"))?;
    let output = phonemicizer.phonemicize(&speaking::PhonemicizeRequest {
        text: sentence.to_string(),
        variety,
        style: None,
    })?;
    let plan = speak::utterance_plan_from_phonemicized(&output);
    eprintln!(
        "\nbe: mechanical={:?}\nbe: phones={}",
        sentence,
        format_be_mechanical_phones(&output)
    );
    speech_backend.synthesize_plan_streaming(&plan, sink)
}

fn synthesize_mechanical_sentence_styletts2(
    sentence: &str,
    variety: &str,
    backend: &mut StyleTts2OnnxBackend,
    sink: &mut dyn styletts2::StyleTts2AudioSink,
    voice_ref: &Path,
    style_ref: &Path,
    max_tts_symbols: usize,
    no_tts_chunking: bool,
) -> Result<()> {
    let variety = VarietyId(variety.to_string());
    let phonemicizer = speaking::phonemicizer_for_variety(&variety)
        .map_err(|error| anyhow::anyhow!("failed to load phonemicizer: {error}"))?;
    let output = phonemicizer.phonemicize(&speaking::PhonemicizeRequest {
        text: sentence.to_string(),
        variety,
        style: None,
    })?;
    let plan = speak::utterance_plan_from_phonemicized(&output);
    let styletts2_plan = prepare_styletts2_plan(
        &plan,
        &styletts2_en_us_symbol_set(),
        be_styletts2_options(max_tts_symbols, no_tts_chunking),
    )
    .context("failed to prepare mechanical StyleTTS2 plan")?;
    let backend_symbols = styletts2_plan
        .chunks
        .iter()
        .map(|chunk| styletts2_text_for_symbols(&chunk.symbols))
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("failed to format StyleTTS2 mechanical symbols")?
        .join(" || ");
    eprintln!(
        "\nbe: mechanical={:?}\nbe: phones={}\nbe: styletts2={}",
        sentence,
        format_be_mechanical_phones(&output),
        backend_symbols
    );
    synthesize_styletts2_plan(backend, sink, plan, voice_ref, style_ref, styletts2_plan)
}

fn synthesize_styletts2_plan(
    backend: &mut StyleTts2OnnxBackend,
    sink: &mut dyn styletts2::StyleTts2AudioSink,
    plan: UtterancePlan,
    voice_ref: &Path,
    style_ref: &Path,
    backend_plan: styletts2::BackendSynthesisPlan,
) -> Result<()> {
    let request = StyleTts2SynthesisRequest::from_backend_plan(
        backend_plan,
        plan.speaker.clone(),
        plan.style.clone(),
        plan.target_prosody.clone(),
    )
    .with_speaker_reference_audio_uri(voice_ref.display().to_string())
    .with_style_reference_audio_uri(style_ref.display().to_string());
    backend
        .synthesize_streaming(&request, sink)
        .context("native StyleTTS2 ONNX synthesis failed")?;
    Ok(())
}

fn styletts2_plan_from_head2phones_prediction(
    variety: &str,
    text: &str,
    phones: &str,
) -> Result<UtterancePlan> {
    let target_phones = styletts2_phone_tokens_from_head2phones(phones);
    anyhow::ensure!(
        !target_phones.is_empty(),
        "head2phones output did not contain any StyleTTS2-compatible phones"
    );
    Ok(UtterancePlan {
        id: UtteranceId("be.head2phones.styletts2".into()),
        variety: VarietyId(variety.to_string()),
        speaker: None,
        intended_text: Some(text.to_string()),
        intended_morphemes: Vec::new(),
        intended_phonemes: Vec::new(),
        target_phones,
        target_syllables: Vec::new(),
        boundaries: Vec::new(),
        target_prosody: Default::default(),
        target_acoustics: Vec::new(),
        speaker_reference: None,
        style: None,
        provenance: EvidenceProvenance {
            source: EvidenceSource::TtsPlan,
            method: "tongues be head2phones StyleTTS2 plan".into(),
            version: Some("0.1".into()),
        },
    })
}

fn styletts2_phone_tokens_from_head2phones(phones: &str) -> Vec<PhoneToken> {
    let mut tokens = Vec::new();
    let mut rest = phones;
    while !rest.is_empty() {
        let Some(character) = rest.chars().next() else {
            break;
        };
        if character.is_whitespace()
            || matches!(
                character,
                'ˈ' | 'ˌ' | '.' | ',' | ';' | ':' | '?' | '!' | '↘' | '↗' | '→'
            )
        {
            rest = consume_char(rest);
            continue;
        }
        if character == '|' {
            tokens.push(styletts2_phone_token("boundary.word".to_string()));
            rest = consume_char(rest);
            continue;
        }
        if let Some((phone, remaining)) = next_styletts2_phone_from_head2phones(rest) {
            tokens.push(styletts2_phone_token(format!("ipa.phone.{phone}")));
            rest = remaining;
        } else {
            rest = consume_char(rest);
        }
    }
    tokens
}

fn styletts2_phone_token(id: String) -> PhoneToken {
    PhoneToken {
        phone: Spec::Known(speaking::ids::PhoneId(id.into())),
        span: None,
        features: Default::default(),
        acoustic_evidence: Vec::new(),
        confidence: 1.0,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Inference,
            method: "tongues be head2phones".into(),
            version: Some("0.1".into()),
        },
    }
}

fn next_styletts2_phone_from_head2phones(rest: &str) -> Option<(&'static str, &str)> {
    for (input, phone) in [
        ("t͡ʃ", "tʃ"),
        ("d͡ʒ", "dʒ"),
        ("aʊ", "aʊ"),
        ("aɪ", "aɪ"),
        ("eɪ", "eɪ"),
        ("oʊ", "oʊ"),
        ("ɔɪ", "ɔɪ"),
        ("iː", "iː"),
        ("uː", "uː"),
        ("kʰ", "kʰ"),
        ("pʰ", "pʰ"),
        ("tʰ", "tʰ"),
        ("k˭", "k˭"),
        ("p˭", "p˭"),
        ("t˭", "t˭"),
        ("tʃ", "tʃ"),
        ("dʒ", "dʒ"),
        ("ɑ", "ɑ"),
        ("æ", "æ"),
        ("ʌ", "ʌ"),
        ("ə", "ə"),
        ("ɐ", "ɐ"),
        ("ɔ", "ɔ"),
        ("b", "b"),
        ("d", "d"),
        ("ð", "ð"),
        ("ɛ", "ɛ"),
        ("ɝ", "ɝ"),
        ("ɚ", "ɚ"),
        ("f", "f"),
        ("ɡ", "ɡ"),
        ("g", "ɡ"),
        ("h", "h"),
        ("ɪ", "ɪ"),
        ("k", "k"),
        ("l", "l"),
        ("ɫ", "ɫ"),
        ("m", "m"),
        ("n", "n"),
        ("ŋ", "ŋ"),
        ("p", "p"),
        ("ɹ", "ɹ"),
        ("r", "ɹ"),
        ("s", "s"),
        ("ʃ", "ʃ"),
        ("t", "t"),
        ("θ", "θ"),
        ("ʊ", "ʊ"),
        ("v", "v"),
        ("w", "w"),
        ("j", "j"),
        ("z", "z"),
        ("ʒ", "ʒ"),
    ] {
        if let Some(remaining) = rest.strip_prefix(input) {
            return Some((phone, remaining));
        }
    }
    None
}

fn be_styletts2_options(max_tts_symbols: usize, no_tts_chunking: bool) -> StyleTts2PlanOptions {
    StyleTts2PlanOptions {
        max_symbols_per_chunk: max_tts_symbols,
        chunking_enabled: !no_tts_chunking,
    }
}

fn format_be_mechanical_phones(output: &speaking::PhonemicizeOutput) -> String {
    output
        .phones
        .iter()
        .filter_map(|phone| match &phone.phone {
            speaking::Spec::Known(id) => Some(
                id.as_str()
                    .strip_prefix("ipa.phone.")
                    .unwrap_or(id.as_str())
                    .to_string(),
            ),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn clean_be_sentence_for_fallback(sentence: &str) -> String {
    let trimmed = sentence.trim();
    let stripped = trimmed
        .trim_matches(|character: char| {
            matches!(
                character,
                '*' | '_' | '`' | '#' | '>' | '-' | '=' | '~' | '\\' | '/'
            )
        })
        .trim();
    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Head2PhonesPrediction {
    phones: String,
    split_after: Option<usize>,
}

fn extract_head2phones_prediction(output: &str) -> Option<Head2PhonesPrediction> {
    Some(Head2PhonesPrediction {
        phones: extract_head2phones_phones(output)?,
        split_after: extract_head2phones_split_after(output),
    })
}

fn extract_head2phones_phones(output: &str) -> Option<String> {
    let start =
        output.find(tongues_head2phones::PHONES_OPEN)? + tongues_head2phones::PHONES_OPEN.len();
    let end = output[start..].find(tongues_head2phones::PHONES_CLOSE)? + start;
    Some(output[start..end].trim().to_string())
}

fn extract_head2phones_split_after(output: &str) -> Option<usize> {
    let marker = tongues_head2phones::SPLIT_AFTER;
    let start = output.find(marker)? + marker.len();
    output[start..].split_whitespace().next()?.parse().ok()
}

fn head2phones_head_and_rest(sentence: &str, split_after: Option<usize>) -> (&str, &str) {
    match split_after {
        Some(split_after) if split_after > 0 => {
            tongues_head2phones::grapheme_split(sentence, split_after)
        }
        _ => (sentence, ""),
    }
}

fn nonempty_remainder(rest: String) -> Option<String> {
    if rest.trim().is_empty() {
        None
    } else {
        Some(rest)
    }
}

fn voice_sequence_from_head2phones_phones(phones: &str) -> speech::PhonemeSequence {
    let mut symbols = Vec::new();
    let mut stress = None;
    let mut rest = phones;
    while !rest.is_empty() {
        if rest
            .chars()
            .next()
            .is_some_and(|character| character.is_whitespace())
        {
            rest = consume_char(rest);
            continue;
        }
        if let Some((symbol, used_stress, remaining)) = next_voice_symbol_from_ipa(rest, stress) {
            if symbol == " " {
                if !symbols.last().is_some_and(|last| last == " ") {
                    symbols.push(symbol.to_string());
                }
            } else {
                symbols.push(symbol.to_string());
            }
            rest = remaining;
            if used_stress {
                stress = None;
            }
            continue;
        }
        if rest.starts_with('ˈ') {
            stress = Some('1');
        } else if rest.starts_with('ˌ') {
            stress = Some('2');
        }
        rest = consume_char(rest);
    }
    speech::PhonemeSequence { symbols }
}

fn next_voice_symbol_from_ipa(
    rest: &str,
    stress: Option<char>,
) -> Option<(&'static str, bool, &str)> {
    for (ipa, base, vowel) in [
        ("t͡ʃ", "CH", false),
        ("d͡ʒ", "JH", false),
        ("aʊ", "AW", true),
        ("aɪ", "AY", true),
        ("eɪ", "EY", true),
        ("oʊ", "OW", true),
        ("ɔɪ", "OY", true),
        ("iː", "IY", true),
        ("uː", "UW", true),
        ("ɑ", "AA", true),
        ("æ", "AE", true),
        ("ʌ", "AH", true),
        ("ə", "AH0", true),
        ("ɐ", "AH0", true),
        ("ɔ", "AO", true),
        ("ɛ", "EH", true),
        ("ɝ", "ER1", true),
        ("ɚ", "ER0", true),
        ("ɪ", "IH", true),
        ("i", "IY", true),
        ("ʊ", "UH", true),
        ("u", "UW", true),
        ("b", "B", false),
        ("d", "D", false),
        ("ð", "DH", false),
        ("f", "F", false),
        ("ɡ", "G", false),
        ("g", "G", false),
        ("h", "HH", false),
        ("k", "K", false),
        ("l", "L", false),
        ("ɫ", "L", false),
        ("m", "M", false),
        ("n", "N", false),
        ("ŋ", "NG", false),
        ("p", "P", false),
        ("ɹ", "R", false),
        ("r", "R", false),
        ("s", "S", false),
        ("ʃ", "SH", false),
        ("t", "T", false),
        ("θ", "TH", false),
        ("v", "V", false),
        ("w", "W", false),
        ("j", "Y", false),
        ("z", "Z", false),
        ("ʒ", "ZH", false),
        ("|", " ", false),
        (",", ",", false),
        (";", ";", false),
        (":", ":", false),
        ("!", "!", false),
        ("?", "?", false),
    ] {
        if let Some(remaining) = rest.strip_prefix(ipa) {
            if vowel && !base.ends_with(['0', '1', '2']) {
                return Some((
                    stressful_vowel_symbol(base, stress),
                    stress.is_some(),
                    remaining,
                ));
            }
            return Some((base, false, remaining));
        }
    }
    None
}

fn stressful_vowel_symbol(base: &str, stress: Option<char>) -> &'static str {
    match (base, stress.unwrap_or('0')) {
        ("AA", '1') => "AA1",
        ("AA", '2') => "AA2",
        ("AA", _) => "AA0",
        ("AE", '1') => "AE1",
        ("AE", '2') => "AE2",
        ("AE", _) => "AE0",
        ("AH", '1') => "AH1",
        ("AH", '2') => "AH2",
        ("AH", _) => "AH0",
        ("AO", '1') => "AO1",
        ("AO", '2') => "AO2",
        ("AO", _) => "AO0",
        ("AW", '1') => "AW1",
        ("AW", '2') => "AW2",
        ("AW", _) => "AW0",
        ("AY", '1') => "AY1",
        ("AY", '2') => "AY2",
        ("AY", _) => "AY0",
        ("EH", '1') => "EH1",
        ("EH", '2') => "EH2",
        ("EH", _) => "EH0",
        ("EY", '1') => "EY1",
        ("EY", '2') => "EY2",
        ("EY", _) => "EY0",
        ("IH", '1') => "IH1",
        ("IH", '2') => "IH2",
        ("IH", _) => "IH0",
        ("IY", '1') => "IY1",
        ("IY", '2') => "IY2",
        ("IY", _) => "IY0",
        ("OW", '1') => "OW1",
        ("OW", '2') => "OW2",
        ("OW", _) => "OW0",
        ("OY", '1') => "OY1",
        ("OY", '2') => "OY2",
        ("OY", _) => "OY0",
        ("UH", '1') => "UH1",
        ("UH", '2') => "UH2",
        ("UH", _) => "UH0",
        ("UW", '1') => "UW1",
        ("UW", '2') => "UW2",
        ("UW", _) => "UW0",
        _ => "AH0",
    }
}

fn consume_char(value: &str) -> &str {
    let len = value
        .chars()
        .next()
        .map(|character| character.len_utf8())
        .unwrap_or(0);
    &value[len..]
}

fn collect_completed_sentence_parser_prefixes(
    cursor: &mut String,
    previous: &mut String,
) -> Vec<String> {
    let mut sentences = Vec::new();
    loop {
        let sentence_end = completed_sentence_prefix_end(cursor);
        let paragraph_fragment = leading_paragraph_fragment_end(cursor);
        let end = match (sentence_end, paragraph_fragment) {
            (Some(sentence_end), Some((_, paragraph_end))) if paragraph_end < sentence_end => {
                paragraph_end
            }
            (Some(sentence_end), _) => sentence_end,
            (None, Some((_, paragraph_end))) => paragraph_end,
            (None, None) => break,
        };
        let sentence = cursor[..end]
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        *cursor = cursor[end..].to_string();
        if !sentence.is_empty() {
            *previous = sentence.clone();
            sentences.push(sentence);
        }
    }
    sentences
}

fn collect_completed_seams_prefixes(
    cursor: &mut String,
    previous: &mut String,
    detector: &seams::SentenceDetectorDialog,
) -> Result<Vec<String>> {
    let detections = detector
        .detect_sentences_borrowed(cursor)
        .context("detecting sentence seams")?;
    let cursor_base = cursor.as_ptr() as usize;
    let cursor_len = cursor.len();
    let mut drain_end = 0usize;
    let mut sentences = Vec::new();

    for detected in detections {
        if !seams_sentence_is_stream_complete(detected.raw_content) {
            continue;
        }

        let start = (detected.raw_content.as_ptr() as usize).saturating_sub(cursor_base);
        let end = start + detected.raw_content.len();
        if start < drain_end || end > cursor_len {
            continue;
        }

        let sentence = detected
            .normalize()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        drain_end = end;
        if !sentence.is_empty() {
            *previous = sentence.clone();
            sentences.push(sentence);
        }
    }

    if drain_end > 0 {
        *cursor = cursor[drain_end..].to_string();
    }

    Ok(sentences)
}

fn seams_sentence_is_stream_complete(sentence: &str) -> bool {
    let trimmed = sentence.trim_end();
    let without_closers = trimmed.trim_end_matches(|ch| {
        matches!(ch, '"' | '\'' | ')' | ']' | '}' | '\u{2019}' | '\u{201d}')
    });
    without_closers
        .chars()
        .last()
        .is_some_and(|ch| matches!(ch, '.' | '?' | '!'))
}

fn cmd_head2phones_eval(
    model_dir: &Path,
    data: &Path,
    split: &str,
    limit: usize,
    seed: u64,
    device_arg: DeviceArg,
) -> Result<()> {
    anyhow::ensure!(limit > 0, "--limit must be greater than zero");
    let manifest =
        tongues_neural::read_manifest(&model_dir.join(tongues_neural::ARTIFACT_MANIFEST_FILE))?;
    anyhow::ensure!(
        manifest.family == tongues_head2phones::FAMILY,
        "expected head2phones manifest, found `{}`",
        manifest.family
    );

    let split_path = data.join(format!("{split}.jsonl"));
    let rows: Vec<tongues_head2phones::Head2PhonesTrainingExample> =
        read_jsonl_as(&split_path).with_context(|| format!("loading {}", split_path.display()))?;
    anyhow::ensure!(
        !rows.is_empty(),
        "head2phones split is empty: {}",
        split_path.display()
    );

    let start_config = std::time::Instant::now();
    let model_config: ModelConfig = read_json_file(&model_dir.join("model_config.json"))?;
    let vocab: Vocab = read_json_file(&model_dir.join("vocab.json"))?;

    let mut shuffled_indexes: Vec<usize> = (0..rows.len()).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    shuffled_indexes.shuffle(&mut rng);
    let mut sample_indexes = Vec::with_capacity(limit);
    let mut skipped_long_inputs = 0usize;
    for row_index in shuffled_indexes {
        let row = &rows[row_index];
        let input_len = vocab.encode_string(&head2phones_eval_input(row)).len();
        if input_len > model_config.max_seq_len {
            skipped_long_inputs += 1;
            continue;
        }
        sample_indexes.push(row_index);
        if sample_indexes.len() >= limit {
            break;
        }
    }
    anyhow::ensure!(
        !sample_indexes.is_empty(),
        "no sampled head2phones examples fit model max_seq_len={} in {}",
        model_config.max_seq_len,
        split_path.display()
    );

    println!("Head2phones eval");
    println!("  model: {}", model_dir.display());
    println!("  data: {}", data.display());
    println!(
        "  split: {} ({} rows, running {} random examples, seed={}, skipped {} overlong while sampling)",
        split,
        format_count(rows.len()),
        format_count(sample_indexes.len()),
        seed,
        format_count(skipped_long_inputs)
    );
    println!(
        "  metadata: {} tokens, max_seq_len={} loaded in {:.1} ms",
        format_count(vocab.size()),
        format_count(model_config.max_seq_len),
        elapsed_ms(start_config.elapsed())
    );

    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            println!("  device: CPU (ndarray)");
            run_head2phones_eval::<CpuInferBackend>(
                &device,
                &model_config,
                model_dir,
                &vocab,
                &rows,
                &sample_indexes,
            )
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            println!("  device: CUDA GPU");
            run_head2phones_eval::<CudaInferBackend>(
                &device,
                &model_config,
                model_dir,
                &vocab,
                &rows,
                &sample_indexes,
            )
        }
    }
}

fn run_head2phones_eval<B: Backend>(
    device: &B::Device,
    model_config: &ModelConfig,
    model_dir: &Path,
    vocab: &Vocab,
    rows: &[tongues_head2phones::Head2PhonesTrainingExample],
    sample_indexes: &[usize],
) -> Result<()> {
    let start_load = std::time::Instant::now();
    let model = load_model::<B>(model_config, &model_dir.join("model"), device)?;
    println!(
        "  weights: loaded in {:.1} ms",
        elapsed_ms(start_load.elapsed())
    );
    println!();

    let mut total_prediction = Duration::ZERO;
    let mut exact = 0usize;
    for (sample_index, &row_index) in sample_indexes.iter().enumerate() {
        let row = &rows[row_index];
        let input = head2phones_eval_input(row);
        let start_prediction = std::time::Instant::now();
        let prediction = predict_sentence_boundary(&model, &input, vocab, device);
        let elapsed = start_prediction.elapsed();
        total_prediction += elapsed;
        let passed = prediction == row.output;
        exact += usize::from(passed);

        println!(
            "#{:02} row={} {} {:.1} ms source={:?} variety={}",
            sample_index + 1,
            format_count(row_index + 1),
            if passed { "PASS" } else { "MISS" },
            elapsed_ms(elapsed),
            row.row_source,
            row.variety
        );
        println!("  buffer: {}", compact_display(&row.input, 140));
        if let Some(head) = &row.head {
            println!("  head:   {}", compact_display(head, 140));
        }
        println!("  gold:   {}", compact_display(&row.output, 180));
        println!("  pred:   {}", compact_display(&prediction, 180));
        println!();
    }

    let mean_prediction = total_prediction.as_secs_f64() * 1000.0 / sample_indexes.len() as f64;
    println!(
        "Summary: exact={}/{} mean_prediction={:.1} ms total_prediction={:.1} ms",
        format_count(exact),
        format_count(sample_indexes.len()),
        mean_prediction,
        elapsed_ms(total_prediction)
    );
    Ok(())
}

fn head2phones_eval_input(row: &tongues_head2phones::Head2PhonesTrainingExample) -> String {
    if row.input_has_variety {
        tongues_head2phones::format_input_for_variety(&row.variety, &row.input)
    } else {
        tongues_head2phones::format_input_without_variety(&row.input)
    }
}

fn elapsed_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn compact_display(value: &str, max_chars: usize) -> String {
    let mut compact = value.replace('\n', "\\n");
    if compact.chars().count() <= max_chars {
        return compact;
    }
    compact = compact.chars().take(max_chars.saturating_sub(3)).collect();
    compact.push_str("...");
    compact
}

fn cmd_sentence_parser_infer(
    model_dir: &Path,
    previous: &str,
    cursor: &str,
    device_arg: DeviceArg,
) -> Result<()> {
    let manifest =
        tongues_neural::read_manifest(&model_dir.join(tongues_neural::ARTIFACT_MANIFEST_FILE))?;
    anyhow::ensure!(
        manifest.family == tongues_sentence_parser::FAMILY,
        "expected sentence-parser manifest, found `{}`",
        manifest.family
    );
    let model_config: ModelConfig = read_json_file(&model_dir.join("model_config.json"))?;
    let vocab: Vocab = read_json_file(&model_dir.join("vocab.json"))?;
    let lowercase = read_json_file::<tongues_sentence_parser::SentenceParserConfig>(
        &model_dir.join("sentence_parser_config.json"),
    )
    .map(|config| config.lowercase)
    .unwrap_or(false);
    let input = tongues_sentence_parser::format_boundary_input(previous, cursor, lowercase);
    let output = match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            let model =
                load_model::<CpuInferBackend>(&model_config, &model_dir.join("model"), &device)?;
            predict_sentence_boundary(&model, &input, &vocab, &device)
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            let model =
                load_model::<CudaInferBackend>(&model_config, &model_dir.join("model"), &device)?;
            predict_sentence_boundary(&model, &input, &vocab, &device)
        }
    };
    let (action, text) = tongues_sentence_parser::parse_boundary_output(&output);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "action": action,
            "text": text,
            "raw": output
        }))?
    );
    Ok(())
}

fn cmd_sentence_parser_stream(
    model_dir: &Path,
    repair_control: &str,
    device_arg: DeviceArg,
) -> Result<()> {
    let manifest =
        tongues_neural::read_manifest(&model_dir.join(tongues_neural::ARTIFACT_MANIFEST_FILE))?;
    anyhow::ensure!(
        manifest.family == tongues_sentence_parser::FAMILY,
        "expected sentence-parser manifest, found `{}`",
        manifest.family
    );
    let model_config: ModelConfig = read_json_file(&model_dir.join("model_config.json"))?;
    let vocab: Vocab = read_json_file(&model_dir.join("vocab.json"))?;
    let lowercase = read_json_file::<tongues_sentence_parser::SentenceParserConfig>(
        &model_dir.join("sentence_parser_config.json"),
    )
    .map(|config| config.lowercase)
    .unwrap_or(false);

    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            let model =
                load_model::<CpuInferBackend>(&model_config, &model_dir.join("model"), &device)?;
            run_sentence_parser_stream_with_model(
                &model,
                &vocab,
                lowercase,
                model_config.max_seq_len,
                repair_control,
                &device,
            )
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            let model =
                load_model::<CudaInferBackend>(&model_config, &model_dir.join("model"), &device)?;
            run_sentence_parser_stream_with_model(
                &model,
                &vocab,
                lowercase,
                model_config.max_seq_len,
                repair_control,
                &device,
            )
        }
    }
}

fn run_sentence_parser_stream_with_model<B: Backend>(
    _model: &Seq2SeqModel<B>,
    _vocab: &Vocab,
    _lowercase: bool,
    _max_seq_len: usize,
    _repair_control: &str,
    _device: &B::Device,
) -> Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    run_sentence_parser_stream_io(stdin.lock(), stdout.lock())
}

fn run_sentence_parser_stream_io(mut reader: impl Read, mut stdout: impl Write) -> Result<()> {
    let mut previous = String::new();
    let mut cursor = String::new();
    let mut pending_utf8 = Vec::new();
    let mut byte = [0_u8; 1];

    loop {
        let bytes = reader.read(&mut byte).context("reading stdin")?;
        if bytes == 0 {
            break;
        }
        append_utf8_chunk(&mut pending_utf8, &byte[..bytes], &mut cursor);
        drain_completed_sentence_parser_prefixes(&mut cursor, &mut previous, &mut stdout)?;
    }

    if !pending_utf8.is_empty() {
        cursor.push_str(&String::from_utf8_lossy(&pending_utf8));
    }
    drain_completed_sentence_parser_prefixes(&mut cursor, &mut previous, &mut stdout)?;

    let tail = cursor.split_whitespace().collect::<Vec<_>>().join(" ");
    if !tail.is_empty() {
        writeln!(stdout, "{tail}").context("writing final sentence-parser tail")?;
    }
    stdout.flush().context("flushing sentence-parser output")?;
    Ok(())
}

fn append_utf8_chunk(pending: &mut Vec<u8>, chunk: &[u8], output: &mut String) {
    pending.extend_from_slice(chunk);
    loop {
        match std::str::from_utf8(pending) {
            Ok(valid) => {
                output.push_str(valid);
                pending.clear();
                break;
            }
            Err(err) => {
                let valid_up_to = err.valid_up_to();
                if valid_up_to > 0 {
                    output.push_str(
                        std::str::from_utf8(&pending[..valid_up_to]).expect("valid UTF-8 prefix"),
                    );
                    pending.drain(..valid_up_to);
                }

                if let Some(error_len) = err.error_len() {
                    output.push('\u{fffd}');
                    pending.drain(..error_len);
                } else {
                    break;
                }
            }
        }
    }
}

fn drain_completed_sentence_parser_prefixes(
    cursor: &mut String,
    previous: &mut String,
    stdout: &mut impl Write,
) -> Result<usize> {
    let mut emitted = 0usize;
    loop {
        let sentence_end = completed_sentence_prefix_end(cursor);
        let paragraph_fragment = leading_paragraph_fragment_end(cursor);
        let end = match (sentence_end, paragraph_fragment) {
            (Some(sentence_end), Some((_, paragraph_end))) if paragraph_end < sentence_end => {
                paragraph_end
            }
            (Some(sentence_end), _) => sentence_end,
            (None, Some((_, paragraph_end))) => paragraph_end,
            (None, None) => break,
        };

        let sentence = cursor[..end]
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        *cursor = cursor[end..].to_string();
        if !sentence.is_empty() {
            writeln!(stdout, "{sentence}").context("writing emitted sentence")?;
            *previous = sentence;
            emitted += 1;
        }
    }
    if emitted > 0 {
        stdout.flush().context("flushing sentence-parser output")?;
    }
    Ok(emitted)
}

fn leading_paragraph_fragment_end(cursor: &str) -> Option<(usize, usize)> {
    let boundary = cursor.find("\n\n")?;
    let mut paragraph_end = boundary + 2;
    while let Some(ch) = cursor[paragraph_end..].chars().next() {
        if ch == '\n' || ch == '\r' || ch == ' ' || ch == '\t' {
            paragraph_end += ch.len_utf8();
        } else {
            break;
        }
    }
    Some((boundary, paragraph_end))
}

fn completed_sentence_prefix_end(cursor: &str) -> Option<usize> {
    let mut search_start = 0usize;
    while let Some((relative_index, terminal)) = cursor[search_start..]
        .char_indices()
        .find(|(_, ch)| matches!(ch, '.' | '?' | '!'))
    {
        let terminal_index = search_start + relative_index;
        let after_terminal = terminal_index + terminal.len_utf8();
        if terminal == '.' && sentence_parser_dot_is_abbreviation(cursor, terminal_index) {
            search_start = after_terminal;
            continue;
        }

        let end = sentence_parser_closing_punctuation_end(cursor, after_terminal);
        if cursor[end..].trim_start().is_empty() || cursor[end..].starts_with(char::is_whitespace) {
            return Some(end);
        }
        search_start = after_terminal;
    }
    None
}

fn sentence_parser_closing_punctuation_end(cursor: &str, mut index: usize) -> usize {
    while let Some(ch) = cursor[index..].chars().next() {
        if matches!(ch, '"' | '\'' | ')' | ']' | '}') {
            index += ch.len_utf8();
        } else {
            break;
        }
    }
    index
}

fn sentence_parser_dot_is_abbreviation(cursor: &str, dot_index: usize) -> bool {
    let prefix = cursor[..dot_index].trim_end();
    let token = prefix
        .split_whitespace()
        .last()
        .unwrap_or("")
        .trim_matches(|ch: char| {
            matches!(
                ch,
                '"' | '\'' | '(' | '[' | '{' | ',' | ':' | ';' | '_' | '*'
            )
        });
    if token.is_empty() {
        return false;
    }

    let lower = token.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "mr" | "mrs"
            | "ms"
            | "dr"
            | "prof"
            | "sr"
            | "jr"
            | "st"
            | "mt"
            | "vs"
            | "etc"
            | "e.g"
            | "i.e"
            | "fig"
            | "no"
            | "dept"
            | "inc"
            | "ltd"
            | "co"
    ) || (token.chars().count() == 1 && token.chars().all(|ch| ch.is_ascii_uppercase()))
}

#[cfg(test)]
fn emit_oversize_sentence_parser_prefix(
    cursor: &mut String,
    previous: &mut String,
    stdout: &mut impl Write,
) -> Result<bool> {
    let Some((end, _)) = cursor
        .char_indices()
        .find(|(_, ch)| matches!(ch, '.' | '?' | '!'))
    else {
        return Ok(false);
    };

    let sentence = cursor[..end + 1]
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let rest = cursor[end + 1..].to_string();
    if sentence.is_empty() {
        *cursor = rest;
        return Ok(true);
    }

    writeln!(stdout, "{sentence}").context("writing oversize sentence-parser sentence")?;
    *previous = sentence;
    *cursor = rest;
    Ok(true)
}

#[cfg(test)]
fn cursor_after_emitted_sentence(cursor: &str, sentence: &str) -> String {
    let cursor = cursor.trim_start();
    let sentence = sentence.trim();
    if sentence.is_empty() {
        return cursor.to_string();
    }
    if let Some(rest) = cursor.strip_prefix(sentence) {
        return rest.to_string();
    }

    let lower_cursor = cursor.to_lowercase();
    let lower_sentence = sentence.to_lowercase();
    if lower_cursor.starts_with(&lower_sentence) {
        let len = sentence.len();
        if cursor.is_char_boundary(len) {
            return cursor[len..].to_string();
        }
    }

    for (index, ch) in cursor.char_indices() {
        if matches!(ch, '.' | '?' | '!') {
            return cursor[index + ch.len_utf8()..].to_string();
        }
    }

    String::new()
}

fn effective_wiktionary_data_path(
    path: PathBuf,
    config: &tongues_wiktionary::WiktionaryConfig,
) -> PathBuf {
    if path == PathBuf::from(DEFAULT_WIKTIONARY_DATA_DIR)
        && config.dataset_id != DEFAULT_WIKTIONARY_DATASET_ID
    {
        PathBuf::from("datasets/wiktionary").join(&config.dataset_id)
    } else {
        path
    }
}

fn effective_wiktionary_model_path(
    path: PathBuf,
    config: &tongues_wiktionary::WiktionaryConfig,
) -> PathBuf {
    if path == PathBuf::from(DEFAULT_WIKTIONARY_MODEL_DIR)
        && config.dataset_id != DEFAULT_WIKTIONARY_DATASET_ID
    {
        PathBuf::from("models/wiktionary").join(&config.dataset_id)
    } else {
        path
    }
}

fn wiktionary_audio_dataset_ready(path: &Path) -> bool {
    path.join("patterns.jsonl").exists()
        && (path.join("phonemes.jsonl").exists() || path.join("phones.jsonl").exists())
}

fn ensure_wiktionary_audio_dataset_available(path: &Path) -> Result<()> {
    if wiktionary_audio_dataset_ready(path) {
        return Ok(());
    }

    let config_path = Path::new(DEFAULT_WIKTIONARY_CONFIG_PATH);
    let config = tongues_wiktionary::read_config(config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let cache_dir = PathBuf::from(DEFAULT_WIKTIONARY_CACHE_DIR);

    let pb = status_spinner(format!(
        "Wiktionary audio metadata missing; preparing dataset at {}",
        path.display()
    ));
    let report = tongues_wiktionary::prepare_dataset_with_progress(path, &cache_dir, &config, {
        let pb = pb.clone();
        move |progress| {
            pb.set_message(wiktionary_prepare_progress_message(progress));
        }
    })?;
    finish_status(
        pb,
        format!(
            "Prepared Wiktionary dataset at {} from {}",
            path.display(),
            report.dump_path
        ),
    );

    if config.verify_with_ollama {
        maybe_print_wiktionary_ollama_report(path)?;
    }

    anyhow::ensure!(
        wiktionary_audio_dataset_ready(path),
        "Wiktionary prepare completed but {} is missing required audio metadata files",
        path.display()
    );

    Ok(())
}

fn print_wiktionary_ollama_report(report: &tongues_wiktionary::OllamaVerificationReport) {
    let path = report
        .report_path
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "not written".to_string());
    if report.sane {
        println!(
            "Ollama verification passed: model={} rows={} chunks={} report={}",
            report.model,
            format_count(report.rows),
            format_count(report.chunks),
            path
        );
    } else {
        println!(
            "Ollama verification reported an issue: model={} rows={} chunks={} report={}\n{}",
            report.model,
            format_count(report.rows),
            format_count(report.chunks),
            path,
            report
                .issue
                .as_deref()
                .unwrap_or("model reported the data is not sane without a specific issue")
        );
    }
}

fn maybe_print_wiktionary_ollama_report(data: &Path) -> Result<()> {
    let path = data.join("ollama_verification.json");
    if path.exists() {
        let report: tongues_wiktionary::OllamaVerificationReport = read_json_file(&path)?;
        print_wiktionary_ollama_report(&report);
    }
    Ok(())
}

fn run_wiktionary_command(
    command: WiktionaryCommands,
    device_arg: DeviceArg,
    output_mode: OutputMode,
) -> Result<()> {
    match command {
        WiktionaryCommands::Clean(args) => cmd_clean_family(
            "wiktionary",
            &args,
            DEFAULT_WIKTIONARY_DATA_DIR,
            DEFAULT_WIKTIONARY_MODEL_DIR,
        ),
        WiktionaryCommands::Prepare {
            config,
            dump,
            out,
            cache_dir,
            langs,
        } => {
            let mut config = tongues_wiktionary::read_config(&config)?;
            if let Some(dump) = dump {
                config.dump_path = Some(dump.display().to_string());
            }
            apply_wiktionary_language_override(&mut config, langs);
            let out = effective_wiktionary_data_path(out, &config);
            let pb = status_spinner(format!("Preparing Wiktionary dataset at {}", out.display()));
            let report =
                tongues_wiktionary::prepare_dataset_with_progress(&out, &cache_dir, &config, {
                    let pb = pb.clone();
                    move |progress| {
                        pb.set_message(wiktionary_prepare_progress_message(progress));
                    }
                })?;
            finish_status(
                pb,
                format!(
                    "Prepared Wiktionary dataset at {} from {}",
                    out.display(),
                    report.dump_path
                ),
            );
            println!(
                "Wiktionary dataset written to {} from {}",
                out.display(),
                report.dump_path
            );
            println!(
                "Parsed {} phonemes, {} phones, {} etymologies, and {} PIE roots into train/valid/test examples: {}/{}/{}",
                format_count(report.parsed_phonemes),
                format_count(report.parsed_phones),
                format_count(report.parsed_etymologies),
                format_count(report.parsed_pie_roots),
                format_count(report.train_examples),
                format_count(report.valid_examples),
                format_count(report.test_examples)
            );
            if config.verify_with_ollama {
                maybe_print_wiktionary_ollama_report(&out)?;
            }
            Ok(())
        }
        WiktionaryCommands::Train {
            config,
            dump,
            data,
            langs,
            notation,
            task,
            out,
            cache_dir,
            prepare,
            sight_words,
            learning_rate,
            weight_decay,
            dropout,
            batch_size,
            epochs,
            patience,
            seed,
            wait_for_prepare,
        } => {
            let mut config = tongues_wiktionary::read_config(&config)?;
            if let Some(dump) = dump {
                config.dump_path = Some(dump.display().to_string());
            }
            apply_wiktionary_language_override(&mut config, langs);
            let data = effective_wiktionary_data_path(data, &config);
            let out = effective_wiktionary_model_path(out, &config);
            let task = task
                .as_deref()
                .unwrap_or(config.train_task.as_str())
                .to_string();
            if wait_for_prepare {
                cmd_wiktionary_train_while_preparing(
                    &data,
                    &out,
                    &config,
                    notation.as_ref(),
                    &task,
                    learning_rate,
                    weight_decay,
                    dropout,
                    batch_size,
                    epochs,
                    patience,
                    seed,
                    sight_words,
                    device_arg,
                )?;
            }
            if prepare
                || !data.join("train.jsonl").exists()
                || !data.join("valid.jsonl").exists()
                || !data.join("test.jsonl").exists()
            {
                let pb = status_spinner(format!(
                    "Training data missing; preparing Wiktionary dataset at {}",
                    data.display()
                ));
                let report = tongues_wiktionary::prepare_dataset_with_progress(
                    &data,
                    &cache_dir,
                    &config,
                    {
                        let pb = pb.clone();
                        move |progress| {
                            pb.set_message(wiktionary_prepare_progress_message(progress));
                        }
                    },
                )?;
                finish_status(
                    pb,
                    format!(
                        "Prepared {} train / {} valid / {} test examples from {}",
                        format_count(report.train_examples),
                        format_count(report.valid_examples),
                        format_count(report.test_examples),
                        report.dump_path
                    ),
                );
            }
            cmd_wiktionary_train(
                &data,
                &out,
                &config,
                notation.as_ref(),
                &task,
                learning_rate,
                weight_decay,
                dropout,
                batch_size,
                epochs,
                patience,
                seed,
                sight_words,
                device_arg,
            )
        }
        WiktionaryCommands::Infer {
            model,
            task,
            lang,
            notation,
            variety,
            raw,
            input,
        } => cmd_wiktionary_infer(
            &model,
            &task,
            &lang,
            notation,
            variety.as_deref(),
            raw,
            &input,
            device_arg,
            output_mode,
        ),
    }
}

fn apply_wiktionary_language_override(
    config: &mut tongues_wiktionary::WiktionaryConfig,
    langs: Vec<String>,
) {
    let langs = langs
        .into_iter()
        .map(|lang| lang.trim().to_string())
        .filter(|lang| !lang.is_empty())
        .collect::<Vec<_>>();
    if !langs.is_empty() {
        config.languages = langs;
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_wiktionary_train_while_preparing(
    data: &Path,
    out: &Path,
    config: &tongues_wiktionary::WiktionaryConfig,
    notation: Option<&WiktionaryNotationArg>,
    task: &str,
    learning_rate: f64,
    weight_decay: f32,
    dropout: f64,
    batch_size: usize,
    epochs: usize,
    patience: usize,
    seed: u64,
    sight_words: bool,
    device_arg: DeviceArg,
) -> Result<()> {
    if wiktionary_prepared_final_files_exist(data) {
        return Ok(());
    }
    if config.source_kind == tongues_wiktionary::WiktionarySourceKind::PieEtymology {
        wait_for_prepared_dataset(
            data,
            &["vocab.json", "train.jsonl", "valid.jsonl", "test.jsonl"],
            "wiktionary",
        )?;
        return Ok(());
    }
    if sight_words {
        println!(
            "  --sight-words will be applied to the final prepared train run; rolling while-preparing epochs use available expanded rows only"
        );
    }

    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let vocab = tongues_wiktionary::build_stable_wiktionary_vocab(&[]);
    write_cli_text_atomic(
        out.join("vocab.json").as_path(),
        serde_json::to_string_pretty(&vocab)?,
    )?;

    let notations = resolve_wiktionary_train_notations(notation, config)?;
    let min_rows = (batch_size * 4).max(256);
    let row_step = 10_000usize;
    let mut last_trained_rows = 0usize;

    println!(
        "Wiktionary while-preparing training enabled: watching {} for expanded rows",
        data.display()
    );
    println!(
        "  stable vocab: {} tokens; rolling epochs train from expanded.jsonl(.writing.part) until final splits are ready",
        format_count(vocab.size())
    );

    loop {
        if wiktionary_prepared_final_files_exist(data) {
            println!("Wiktionary prepared splits are ready; leaving while-preparing mode.");
            return Ok(());
        }

        let completed_epoch = wiktionary_completed_epoch(out)?;
        if completed_epoch >= epochs {
            println!(
                "Rolling training reached requested epochs={}; waiting for final Wiktionary prepared files",
                format_count(epochs)
            );
            wait_for_prepared_dataset(
                data,
                &["vocab.json", "train.jsonl", "valid.jsonl", "test.jsonl"],
                "wiktionary",
            )?;
            return Ok(());
        }

        let Some(expanded_path) = wiktionary_available_expanded_path(data) else {
            std::thread::sleep(Duration::from_secs(10));
            continue;
        };
        let rows_raw = read_jsonl_lossy::<tongues_wiktionary::TrainingExample>(&expanded_path)?;
        let rows = filter_wiktionary_examples(
            filter_wiktionary_examples_by_notation(rows_raw, Some(&notations)),
            task,
        )?;
        if rows.len() < min_rows {
            println!(
                "  waiting for {} rows in {}; currently {} usable rows",
                format_count(min_rows),
                expanded_path.display(),
                format_count(rows.len())
            );
            std::thread::sleep(Duration::from_secs(20));
            continue;
        }
        let enough_new_rows = rows.len().saturating_sub(last_trained_rows) >= row_step;
        if !enough_new_rows && !expanded_path.ends_with("expanded.jsonl") {
            std::thread::sleep(Duration::from_secs(20));
            continue;
        }

        let next_epoch = completed_epoch + 1;
        let (base_train_rows, valid_rows, _) =
            split_wiktionary_examples(rows, config.train_frac, config.valid_frac, config.seed);
        if base_train_rows.is_empty() || valid_rows.is_empty() {
            std::thread::sleep(Duration::from_secs(20));
            continue;
        }
        let rarity_by_word = load_openepd_rarity_by_word()?;
        let (mut train_rows, frequency_matched_rows, frequency_added_rows) =
            expand_wiktionary_frequency_weighted_training_examples(
                &base_train_rows,
                &rarity_by_word,
                DEFAULT_MAX_FREQUENCY_REPEAT,
                DEFAULT_FREQUENCY_RARITY_CAP,
            );
        if frequency_added_rows > 0 {
            println!(
                "  rolling epoch rarity expansion matched {} rows (+{} rows)",
                format_count(frequency_matched_rows),
                format_count(frequency_added_rows)
            );
        }
        if sight_words {
            let added = add_wiktionary_sight_word_training_examples(
                &mut train_rows,
                [
                    &valid_rows[..],
                    (&[] as &[tongues_wiktionary::TrainingExample]),
                ],
            );
            if added > 0 {
                println!(
                    "  rolling epoch included {} extra Wiktionary sight-word rows",
                    format_count(added)
                );
            }
        }
        let train_examples = wiktionary_seq2seq_examples(&train_rows, &vocab);
        let valid_examples = wiktionary_seq2seq_examples(&valid_rows, &vocab);
        write_wiktionary_augmented_train_rows(
            data,
            &base_train_rows,
            &valid_rows,
            &[],
            sight_words,
        )?;
        println!(
            "Rolling Wiktionary epoch {} from {} usable rows ({} train / {} valid) in {}",
            format_count(next_epoch),
            format_count(train_rows.len() + valid_rows.len()),
            format_count(train_examples.len()),
            format_count(valid_examples.len()),
            expanded_path.display()
        );

        write_and_train_wiktionary_seq2seq(
            data,
            out,
            config,
            &format!("while-preparing:{task}"),
            learning_rate,
            weight_decay,
            dropout,
            batch_size,
            next_epoch,
            patience.max(1_000_000),
            seed + next_epoch as u64,
            device_arg,
            vocab.clone(),
            train_examples,
            valid_examples,
        )?;
        last_trained_rows = train_rows.len() + valid_rows.len();
    }
}

fn wiktionary_prepared_final_files_exist(data: &Path) -> bool {
    ["vocab.json", "train.jsonl", "valid.jsonl", "test.jsonl"]
        .iter()
        .all(|file| data.join(file).exists())
}

fn wiktionary_available_expanded_path(data: &Path) -> Option<PathBuf> {
    let final_path = data.join("expanded.jsonl");
    if final_path.exists() {
        return Some(final_path);
    }
    let partial_path = data.join("expanded.jsonl.writing.part");
    partial_path.exists().then_some(partial_path)
}

fn wiktionary_completed_epoch(out: &Path) -> Result<usize> {
    let path = out.join("train_state.json");
    if !path.exists() {
        return Ok(0);
    }
    let raw = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let state: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    Ok(state
        .get("current_epoch")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize)
}

#[allow(clippy::too_many_arguments)]
fn cmd_wiktionary_train(
    data: &Path,
    out: &Path,
    config: &tongues_wiktionary::WiktionaryConfig,
    notation: Option<&WiktionaryNotationArg>,
    task: &str,
    learning_rate: f64,
    weight_decay: f32,
    dropout: f64,
    batch_size: usize,
    epochs: usize,
    patience: usize,
    seed: u64,
    sight_words: bool,
    device_arg: DeviceArg,
) -> Result<()> {
    if config.source_kind == tongues_wiktionary::WiktionarySourceKind::PieEtymology {
        let task = if matches!(task, "orthography-to-phones" | "orthography-to-phonemes") {
            "etymology-translation"
        } else {
            task
        };
        return cmd_wiktionary_train_prepared_rows(
            data,
            out,
            config,
            task,
            &format!("pie-etymology:{task}"),
            None,
            learning_rate,
            weight_decay,
            dropout,
            batch_size,
            epochs,
            patience,
            seed,
            sight_words,
            device_arg,
        );
    }

    let notations = resolve_wiktionary_train_notations(notation, config)?;
    if wiktionary_prepared_splits_exist(data) {
        let notation_label = wiktionary_notation_label(&notations);
        return cmd_wiktionary_train_prepared_rows(
            data,
            out,
            config,
            task,
            &format!("{notation_label}:{task}"),
            Some(&notations),
            learning_rate,
            weight_decay,
            dropout,
            batch_size,
            epochs,
            patience,
            seed,
            sight_words,
            device_arg,
        );
    }

    let pb = status_spinner(format!(
        "Loading Wiktionary rows for {}",
        wiktionary_notation_label(&notations)
    ));
    let mut entries = Vec::new();
    for notation in &notations {
        let source_file = wiktionary_notation_source_file(data, *notation);
        let mut rows: Vec<tongues_wiktionary::PronunciationEntry> = read_jsonl_as(&source_file)?;
        entries.append(&mut rows);
    }
    finish_status(
        pb,
        format!(
            "Loaded {} rows for {}",
            format_count(entries.len()),
            wiktionary_notation_label(&notations)
        ),
    );

    let pb = status_spinner("Expanding and filtering Wiktionary training examples");
    let expanded = tongues_wiktionary::expand_training_examples(&entries, config);
    let examples = filter_wiktionary_examples(expanded, task)?;
    finish_status(
        pb,
        format!(
            "Selected {} Wiktionary examples for task={task}",
            format_count(examples.len())
        ),
    );
    anyhow::ensure!(
        !examples.is_empty(),
        "no Wiktionary examples found for notations={} task={task}",
        wiktionary_notation_label(&notations)
    );

    let pb = status_spinner("Splitting rows, building vocabulary, and encoding examples");
    let (mut train_rows, mut valid_rows, _test_rows) =
        split_wiktionary_examples(examples, config.train_frac, config.valid_frac, config.seed);
    let rarity_by_word = load_openepd_rarity_by_word()?;
    let (frequency_expanded_rows, frequency_matched_rows, frequency_added_rows) =
        expand_wiktionary_frequency_weighted_training_examples(
            &train_rows,
            &rarity_by_word,
            DEFAULT_MAX_FREQUENCY_REPEAT,
            DEFAULT_FREQUENCY_RARITY_CAP,
        );
    train_rows = frequency_expanded_rows;
    if frequency_added_rows > 0 {
        println!(
            "  expanded {} English Wiktionary rows by OpenEPD rarity (+{} rows, max_repeat={} rarity_cap={})",
            format_count(frequency_matched_rows),
            format_count(frequency_added_rows),
            format_count(DEFAULT_MAX_FREQUENCY_REPEAT),
            DEFAULT_FREQUENCY_RARITY_CAP
        );
    }
    if sight_words {
        let added = add_wiktionary_sight_word_training_examples(
            &mut train_rows,
            [&valid_rows[..], &_test_rows[..]],
        );
        if added > 0 {
            println!(
                "  included {} extra Wiktionary sight-word training rows (repeat={})",
                format_count(added),
                format_count(SIGHT_WORD_TRAINING_REPEATS)
            );
        } else {
            println!("  no matching English Wiktionary sight-word rows found to oversample");
        }
    }
    let vocab = if out.join("vocab.json").exists() {
        println!(
            "Reusing existing vocabulary from {}",
            out.join("vocab.json").display()
        );
        let vocab: Vocab = read_json_file(&out.join("vocab.json"))?;
        let before_train = train_rows.len();
        let before_valid = valid_rows.len();
        train_rows.retain(|row| wiktionary_example_fits_vocab(row, &vocab));
        valid_rows.retain(|row| wiktionary_example_fits_vocab(row, &vocab));
        let skipped_train = before_train.saturating_sub(train_rows.len());
        let skipped_valid = before_valid.saturating_sub(valid_rows.len());
        if skipped_train > 0 || skipped_valid > 0 {
            println!(
                "Skipped {} train / {} valid Wiktionary examples containing tokens outside the existing model vocabulary. Use a new --out directory to train the full expanded language set from a rebuilt vocabulary.",
                format_count(skipped_train), format_count(skipped_valid)
            );
        }
        vocab
    } else {
        build_wiktionary_vocab(&train_rows, &valid_rows)
    };
    anyhow::ensure!(
        !train_rows.is_empty(),
        "no Wiktionary training examples remain after vocabulary filtering"
    );
    anyhow::ensure!(
        !valid_rows.is_empty(),
        "no Wiktionary validation examples remain after vocabulary filtering"
    );
    let train_examples = wiktionary_seq2seq_examples(&train_rows, &vocab);
    let valid_examples = wiktionary_seq2seq_examples(&valid_rows, &vocab);
    finish_status(
        pb,
        format!(
            "Encoded {} train / {} valid examples with vocab size {}",
            format_count(train_examples.len()),
            format_count(valid_examples.len()),
            format_count(vocab.size())
        ),
    );

    println!(
        "Loaded {} {} rows -> {} train / {} valid examples for task={}",
        format_count(entries.len()),
        wiktionary_notation_label(&notations),
        format_count(train_examples.len()),
        format_count(valid_examples.len()),
        task
    );

    write_and_train_wiktionary_seq2seq(
        data,
        out,
        config,
        &format!("{}:{task}", wiktionary_notation_label(&notations)),
        learning_rate,
        weight_decay,
        dropout,
        batch_size,
        epochs,
        patience,
        seed,
        device_arg,
        vocab,
        train_examples,
        valid_examples,
    )
}

#[allow(clippy::too_many_arguments)]
fn cmd_wiktionary_train_prepared_rows(
    data: &Path,
    out: &Path,
    config: &tongues_wiktionary::WiktionaryConfig,
    task: &str,
    task_label: &str,
    notations: Option<&[WiktionaryNotationArg]>,
    learning_rate: f64,
    weight_decay: f32,
    dropout: f64,
    batch_size: usize,
    epochs: usize,
    patience: usize,
    seed: u64,
    sight_words: bool,
    device_arg: DeviceArg,
) -> Result<()> {
    let pb = status_spinner(format!(
        "Loading prepared Wiktionary rows from {}",
        data.display()
    ));
    let train_rows_raw: Vec<tongues_wiktionary::TrainingExample> =
        read_jsonl_as(&data.join("train.jsonl"))?;
    let valid_rows_raw: Vec<tongues_wiktionary::TrainingExample> =
        read_jsonl_as(&data.join("valid.jsonl"))?;
    let test_rows_raw: Vec<tongues_wiktionary::TrainingExample> =
        if data.join("test.jsonl").exists() {
            read_jsonl_as(&data.join("test.jsonl"))?
        } else {
            Vec::new()
        };
    finish_status(
        pb,
        format!(
            "Loaded {} train / {} valid prepared rows",
            format_count(train_rows_raw.len()),
            format_count(valid_rows_raw.len())
        ),
    );

    write_wiktionary_augmented_train_rows(
        data,
        &train_rows_raw,
        &valid_rows_raw,
        &test_rows_raw,
        sight_words,
    )?;

    let pb = status_spinner(format!("Filtering prepared rows for task={task}"));
    let mut train_rows = filter_wiktionary_examples(
        filter_wiktionary_examples_by_notation(train_rows_raw, notations),
        task,
    )?;
    let valid_rows = filter_wiktionary_examples(
        filter_wiktionary_examples_by_notation(valid_rows_raw, notations),
        task,
    )?;
    let test_rows = if sight_words && !test_rows_raw.is_empty() {
        filter_wiktionary_examples(
            filter_wiktionary_examples_by_notation(test_rows_raw, notations),
            task,
        )?
    } else {
        Vec::new()
    };
    let rarity_by_word = load_openepd_rarity_by_word()?;
    let (frequency_expanded_rows, frequency_matched_rows, frequency_added_rows) =
        expand_wiktionary_frequency_weighted_training_examples(
            &train_rows,
            &rarity_by_word,
            DEFAULT_MAX_FREQUENCY_REPEAT,
            DEFAULT_FREQUENCY_RARITY_CAP,
        );
    train_rows = frequency_expanded_rows;
    if frequency_added_rows > 0 {
        println!(
            "  expanded {} English Wiktionary rows by OpenEPD rarity (+{} rows, max_repeat={} rarity_cap={})",
            format_count(frequency_matched_rows),
            format_count(frequency_added_rows),
            format_count(DEFAULT_MAX_FREQUENCY_REPEAT),
            DEFAULT_FREQUENCY_RARITY_CAP
        );
    }
    if sight_words {
        let added = add_wiktionary_sight_word_training_examples(
            &mut train_rows,
            [&valid_rows[..], &test_rows[..]],
        );
        if added > 0 {
            println!(
                "  included {} extra Wiktionary sight-word training rows (repeat={})",
                format_count(added),
                format_count(SIGHT_WORD_TRAINING_REPEATS)
            );
        } else {
            println!("  no matching English Wiktionary sight-word rows found to oversample");
        }
    }
    finish_status(
        pb,
        format!(
            "Selected {} train / {} valid rows for task={task}",
            format_count(train_rows.len()),
            format_count(valid_rows.len())
        ),
    );
    anyhow::ensure!(
        !train_rows.is_empty(),
        "no prepared Wiktionary examples found for task={task}"
    );
    anyhow::ensure!(
        !valid_rows.is_empty(),
        "no prepared Wiktionary validation examples found for task={task}"
    );

    let pb = status_spinner("Loading Wiktionary vocabulary and encoding seq2seq examples");
    let (vocab, train_rows, valid_rows) =
        load_or_build_wiktionary_vocab(data, out, train_rows, valid_rows)?;
    let train_examples = wiktionary_seq2seq_examples(&train_rows, &vocab);
    let valid_examples = wiktionary_seq2seq_examples(&valid_rows, &vocab);
    finish_status(
        pb,
        format!(
            "Encoded {} train / {} valid examples with vocab size {}",
            format_count(train_examples.len()),
            format_count(valid_examples.len()),
            format_count(vocab.size())
        ),
    );

    println!(
        "Loaded prepared rows -> {} train / {} valid examples for task={}",
        format_count(train_examples.len()),
        format_count(valid_examples.len()),
        task
    );

    write_and_train_wiktionary_seq2seq(
        data,
        out,
        config,
        task_label,
        learning_rate,
        weight_decay,
        dropout,
        batch_size,
        epochs,
        patience,
        seed,
        device_arg,
        vocab,
        train_examples,
        valid_examples,
    )
}

fn wiktionary_prepared_splits_exist(data: &Path) -> bool {
    data.join("train.jsonl").exists() && data.join("valid.jsonl").exists()
}

fn filter_wiktionary_examples_by_notation(
    examples: Vec<tongues_wiktionary::TrainingExample>,
    notations: Option<&[WiktionaryNotationArg]>,
) -> Vec<tongues_wiktionary::TrainingExample> {
    let Some(notations) = notations else {
        return examples;
    };
    let has_phonemic = notations
        .iter()
        .any(|notation| matches!(notation, WiktionaryNotationArg::Phonemes));
    let has_phonetic = notations
        .iter()
        .any(|notation| matches!(notation, WiktionaryNotationArg::Phones));
    examples
        .into_iter()
        .filter(|example| {
            let notation = example.notation.as_deref();
            if notation.is_none() {
                return true;
            }
            if notation == Some("phonetic-realization") {
                return has_phonemic && has_phonetic;
            }
            notations.iter().any(|selected| match selected {
                WiktionaryNotationArg::All => true,
                WiktionaryNotationArg::Phonemes => notation == Some("phonemic"),
                WiktionaryNotationArg::Phones => notation == Some("phonetic"),
            })
        })
        .collect()
}

fn load_or_build_wiktionary_vocab(
    data: &Path,
    out: &Path,
    mut train_rows: Vec<tongues_wiktionary::TrainingExample>,
    mut valid_rows: Vec<tongues_wiktionary::TrainingExample>,
) -> Result<(
    Vocab,
    Vec<tongues_wiktionary::TrainingExample>,
    Vec<tongues_wiktionary::TrainingExample>,
)> {
    let vocab_path = if out.join("vocab.json").exists() {
        Some(out.join("vocab.json"))
    } else if data.join("vocab.json").exists() {
        Some(data.join("vocab.json"))
    } else {
        None
    };

    let Some(vocab_path) = vocab_path else {
        return Ok((
            build_wiktionary_vocab(&train_rows, &valid_rows),
            train_rows,
            valid_rows,
        ));
    };

    println!("Reusing existing vocabulary from {}", vocab_path.display());
    let vocab: Vocab = read_json_file(&vocab_path)?;
    let before_train = train_rows.len();
    let before_valid = valid_rows.len();
    train_rows.retain(|row| wiktionary_example_fits_vocab(row, &vocab));
    valid_rows.retain(|row| wiktionary_example_fits_vocab(row, &vocab));
    let skipped_train = before_train.saturating_sub(train_rows.len());
    let skipped_valid = before_valid.saturating_sub(valid_rows.len());
    if skipped_train > 0 || skipped_valid > 0 {
        println!(
            "Skipped {} train / {} valid Wiktionary examples containing tokens outside the existing vocabulary.",
            format_count(skipped_train),
            format_count(skipped_valid)
        );
    }
    Ok((vocab, train_rows, valid_rows))
}

fn add_wiktionary_sight_word_training_examples<const N: usize>(
    train_rows: &mut Vec<tongues_wiktionary::TrainingExample>,
    extra_sources: [&[tongues_wiktionary::TrainingExample]; N],
) -> usize {
    let sight_words: std::collections::BTreeSet<&str> = SIGHT_WORDS.iter().copied().collect();
    let mut seen = std::collections::BTreeSet::new();
    let mut selected = Vec::new();

    for row in train_rows.iter().chain(extra_sources.into_iter().flatten()) {
        if wiktionary_sight_word_for_example(row, &sight_words).is_some()
            && seen.insert(wiktionary_training_example_key(row))
        {
            selected.push(row.clone());
        }
    }

    let mut added = 0usize;
    for row in selected {
        for _ in 0..SIGHT_WORD_TRAINING_REPEATS {
            train_rows.push(row.clone());
            added += 1;
        }
    }
    added
}

fn expand_wiktionary_frequency_weighted_training_examples(
    train_rows: &[tongues_wiktionary::TrainingExample],
    rarity_by_word: &std::collections::BTreeMap<String, f32>,
    max_repeat: usize,
    rarity_cap: f32,
) -> (Vec<tongues_wiktionary::TrainingExample>, usize, usize) {
    let mut expanded = Vec::new();
    let mut matched_rows = 0usize;
    let mut added_rows = 0usize;

    for row in train_rows {
        let repeat = wiktionary_frequency_repeat_count_for_example(
            row,
            rarity_by_word,
            max_repeat,
            rarity_cap,
        );
        if repeat > 1 {
            matched_rows += 1;
            added_rows += repeat - 1;
        }
        for _ in 0..repeat {
            expanded.push(row.clone());
        }
    }

    (expanded, matched_rows, added_rows)
}

fn wiktionary_frequency_repeat_count_for_example(
    row: &tongues_wiktionary::TrainingExample,
    rarity_by_word: &std::collections::BTreeMap<String, f32>,
    max_repeat: usize,
    rarity_cap: f32,
) -> usize {
    if !wiktionary_example_is_english(row) {
        return 1;
    }
    let Some(candidate) = wiktionary_training_word_candidate(row) else {
        return 1;
    };
    let Some(rarity) = rarity_by_word.get(candidate.as_str()) else {
        return 1;
    };
    frequency_repeat_count(*rarity, max_repeat, rarity_cap)
}

fn write_wiktionary_augmented_train_rows(
    data: &Path,
    train_rows: &[tongues_wiktionary::TrainingExample],
    valid_rows: &[tongues_wiktionary::TrainingExample],
    test_rows: &[tongues_wiktionary::TrainingExample],
    sight_words: bool,
) -> Result<()> {
    let rarity_by_word = load_openepd_rarity_by_word()?;
    let (mut augmented_rows, matched_rows, added_rows) =
        expand_wiktionary_frequency_weighted_training_examples(
            train_rows,
            &rarity_by_word,
            DEFAULT_MAX_FREQUENCY_REPEAT,
            DEFAULT_FREQUENCY_RARITY_CAP,
        );
    let sight_added = if sight_words {
        add_wiktionary_sight_word_training_examples(&mut augmented_rows, [valid_rows, test_rows])
    } else {
        0
    };

    let out_path = data.join("train.augmented.jsonl");
    write_training_example_jsonl_atomic(&out_path, &augmented_rows)?;
    println!(
        "Updated {} with {} rows (rarity matched={} +{} rows; sight words +{} rows)",
        out_path.display(),
        format_count(augmented_rows.len()),
        format_count(matched_rows),
        format_count(added_rows),
        format_count(sight_added)
    );

    Ok(())
}

fn write_training_example_jsonl_atomic(
    path: &Path,
    rows: &[tongues_wiktionary::TrainingExample],
) -> Result<()> {
    let part = atomic_part_path(path);
    archive_interrupted_part(path)?;
    let mut writer = std::io::BufWriter::new(
        fs::File::create(&part).with_context(|| format!("creating {}", part.display()))?,
    );
    for row in rows {
        writeln!(writer, "{}", serde_json::to_string(row)?)?;
    }
    writer
        .flush()
        .with_context(|| format!("flushing {}", part.display()))?;
    drop(writer);
    fs::rename(&part, path)
        .with_context(|| format!("moving {} to {}", part.display(), path.display()))?;
    Ok(())
}

fn wiktionary_sight_word_for_example(
    row: &tongues_wiktionary::TrainingExample,
    sight_words: &std::collections::BTreeSet<&str>,
) -> Option<String> {
    if !wiktionary_example_is_english(row) {
        return None;
    }
    let candidate = wiktionary_training_word_candidate(row)?;
    sight_words
        .contains(candidate.as_str())
        .then_some(candidate)
}

fn wiktionary_training_word_candidate(row: &tongues_wiktionary::TrainingExample) -> Option<String> {
    use tongues_wiktionary::WiktionaryTask;

    let candidate = match row.task {
        WiktionaryTask::OrthographyToPhonology | WiktionaryTask::GuessLangFromOrthography => {
            row.input.split_whitespace().last()
        }
        WiktionaryTask::PhonologyToOrthography | WiktionaryTask::NormalizeText => {
            Some(row.output.as_str())
        }
        WiktionaryTask::GuessLangFromOrthographyAndPhonology => {
            row.input.split("=>").next()?.split_whitespace().last()
        }
        _ => None,
    }?;

    let candidate = candidate.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '\'');
    if candidate.is_empty() {
        return None;
    }
    Some(candidate.to_ascii_lowercase())
}

fn wiktionary_example_is_english(row: &tongues_wiktionary::TrainingExample) -> bool {
    row.lang.as_deref() == Some("eng")
        || row.lang.is_none() && row.notation.is_some() && row.output == "eng"
}

fn wiktionary_training_example_key(row: &tongues_wiktionary::TrainingExample) -> String {
    format!(
        "{:?}\x1f{}\x1f{}\x1f{}\x1f{}",
        row.task,
        row.lang.as_deref().unwrap_or(""),
        row.notation.as_deref().unwrap_or(""),
        row.input,
        row.output
    )
}

#[allow(clippy::too_many_arguments)]
fn write_and_train_wiktionary_seq2seq(
    data: &Path,
    out: &Path,
    config: &tongues_wiktionary::WiktionaryConfig,
    task_label: &str,
    learning_rate: f64,
    weight_decay: f32,
    dropout: f64,
    batch_size: usize,
    epochs: usize,
    patience: usize,
    seed: u64,
    device_arg: DeviceArg,
    vocab: Vocab,
    train_examples: Vec<Seq2SeqExample>,
    valid_examples: Vec<Seq2SeqExample>,
) -> Result<()> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let model_config = if out.join("model_config.json").exists() {
        let existing: ModelConfig = read_json_file(&out.join("model_config.json"))?;
        anyhow::ensure!(
            existing.vocab_size == vocab.size(),
            "existing model_config.json vocab_size={} does not match vocab size {}; remove or update the model directory to train from a rebuilt vocabulary",
            existing.vocab_size,
            vocab.size()
        );
        existing
    } else {
        ModelConfig::new(vocab.size()).with_dropout(dropout)
    };
    let train_config = TrainConfig {
        learning_rate,
        weight_decay,
        dropout,
        batch_size,
        epochs,
        early_stopping_patience: patience,
        max_seq_len: model_config.max_seq_len,
        task: None,
        max_frequency_repeat: 1,
        frequency_rarity_cap: 0.0,
    };

    fs::write(
        out.join("model_config.json"),
        serde_json::to_string_pretty(&model_config)?,
    )?;
    fs::write(
        out.join("train_config.json"),
        serde_json::to_string_pretty(&train_config)?,
    )?;
    fs::write(
        out.join("wiktionary_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    fs::write(
        out.join("vocab.json"),
        serde_json::to_string_pretty(&vocab)?,
    )?;
    write_manifest(
        out,
        &ModelArtifactManifest::new("wiktionary", "seq2seq-transformer", data_id_from_path(data))
            .with_task(task_label.to_string()),
    )?;

    let model_path = out.join("model");
    println!("Starting Wiktionary training...");
    println!(
        "  examples={} train / {} valid vocab={} lr={} wd={} dropout={} epochs={} patience={} batch_size={}",
        format_count(train_examples.len()),
        format_count(valid_examples.len()),
        format_count(vocab.size()),
        learning_rate,
        weight_decay,
        dropout,
        format_count(epochs),
        format_count(patience),
        format_count(batch_size)
    );
    println!("  train_state: {}", out.join("train_state.json").display());
    println!("  early_stop_metric: val_loss");
    println!(
        "  epoch checkpoints: {}",
        out.join("model-epoch-N.bin").display()
    );
    println!(
        "  best model: {}",
        model_path.with_extension("bin").display()
    );

    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            println!("  device: CPU (ndarray)");
            run_wiktionary_train::<CpuTrainBackend>(
                &device,
                &model_config,
                &train_config,
                &train_examples,
                &valid_examples,
                &model_path,
                seed,
            )
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            println!("  device: CUDA GPU");
            run_wiktionary_train::<CudaTrainBackend>(
                &device,
                &model_config,
                &train_config,
                &train_examples,
                &valid_examples,
                &model_path,
                seed,
            )
        }
    }
}

fn filter_wiktionary_examples(
    examples: Vec<tongues_wiktionary::TrainingExample>,
    task: &str,
) -> Result<Vec<tongues_wiktionary::TrainingExample>> {
    use tongues_wiktionary::WiktionaryTask;

    let normalized = task.to_ascii_lowercase();
    let keep = |example: &tongues_wiktionary::TrainingExample| match normalized.as_str() {
        "orthography-to-phonology" => example.task == WiktionaryTask::OrthographyToPhonology,
        "orthography-to-phonemes" => {
            example.task == WiktionaryTask::OrthographyToPhonology
                && example.notation.as_deref() == Some("phonemic")
        }
        "orthography-to-phones" => {
            example.task == WiktionaryTask::OrthographyToPhonology
                && example.notation.as_deref() == Some("phonetic")
        }
        "phonology-to-orthography" => example.task == WiktionaryTask::PhonologyToOrthography,
        "phonemes-to-orthography" => {
            example.task == WiktionaryTask::PhonologyToOrthography
                && example.notation.as_deref() == Some("phonemic")
        }
        "phones-to-orthography" => {
            example.task == WiktionaryTask::PhonologyToOrthography
                && example.notation.as_deref() == Some("phonetic")
        }
        "phonetic-realization" => example.task == WiktionaryTask::PhoneticRealization,
        "segment" | "segment-compound" | "compound-segmentation" => {
            example.task == WiktionaryTask::SegmentCompound
        }
        "pronounce-segments" | "segments-to-phonology" | "segments-to-phones" => {
            example.task == WiktionaryTask::PronounceSegments
        }
        "verify" | "verify-pronunciation" | "verifier" => {
            example.task == WiktionaryTask::VerifyPronunciation
        }
        "normalize-phonology" | "normalise-phonology" | "broad-equivalence" => {
            example.task == WiktionaryTask::NormalizePhonology
        }
        "find-etymology" | "etymology-from-word" | "word-etymology" => {
            example.task == WiktionaryTask::FindEtymology
        }
        "etymology"
        | "etymology-translation"
        | "translate-etymology"
        | "pie-to-descendant"
        | "pie2daughter"
        | "pie-to-daughter"
        | "descendant-to-pie"
        | "daughter-to-pie"
        | "daughter2pie"
        | "descendant-to-descendant"
        | "daughter-to-daughter"
        | "daughter2daughter"
        | "cognate" => example.task == WiktionaryTask::EtymologyTranslation,
        "normalize" | "normalise" => example.task == WiktionaryTask::NormalizeText,
        "align" => example.task == WiktionaryTask::AlignAudioText,
        "lang" | "language" | "language-guessing" => matches!(
            example.task,
            WiktionaryTask::GuessLangFromOrthography
                | WiktionaryTask::GuessLangFromPhonology
                | WiktionaryTask::GuessLangFromOrthographyAndPhonology
        ),
        "all" => true,
        _ => false,
    };
    if !matches!(
        normalized.as_str(),
        "orthography-to-phonology"
            | "orthography-to-phonemes"
            | "orthography-to-phones"
            | "phonology-to-orthography"
            | "phonemes-to-orthography"
            | "phones-to-orthography"
            | "phonetic-realization"
            | "segment"
            | "segment-compound"
            | "compound-segmentation"
            | "pronounce-segments"
            | "segments-to-phonology"
            | "segments-to-phones"
            | "verify"
            | "verify-pronunciation"
            | "verifier"
            | "normalize-phonology"
            | "normalise-phonology"
            | "broad-equivalence"
            | "find-etymology"
            | "etymology-from-word"
            | "word-etymology"
            | "etymology"
            | "etymology-translation"
            | "translate-etymology"
            | "pie-to-descendant"
            | "pie2daughter"
            | "pie-to-daughter"
            | "descendant-to-pie"
            | "daughter-to-pie"
            | "daughter2pie"
            | "descendant-to-descendant"
            | "daughter-to-daughter"
            | "daughter2daughter"
            | "cognate"
            | "normalize"
            | "normalise"
            | "align"
            | "lang"
            | "language"
            | "language-guessing"
            | "all"
    ) {
        anyhow::bail!("Invalid Wiktionary task. Supported: orthography-to-phonemes, orthography-to-phones, phonemes-to-orthography, phones-to-orthography, phonetic-realization, find-etymology, segment-compound, pronounce-segments, verify-pronunciation, normalize-phonology, etymology-translation, normalize, align, lang, all");
    }

    Ok(examples
        .into_iter()
        .filter(|example| keep(example))
        .collect())
}

fn resolve_wiktionary_train_notations(
    notation: Option<&WiktionaryNotationArg>,
    config: &tongues_wiktionary::WiktionaryConfig,
) -> Result<Vec<WiktionaryNotationArg>> {
    let mut notations = Vec::new();
    match notation {
        Some(WiktionaryNotationArg::All) => {
            notations.push(WiktionaryNotationArg::Phonemes);
            notations.push(WiktionaryNotationArg::Phones);
        }
        Some(notation) => notations.push(*notation),
        None => {
            for notation in &config.train_notations {
                match notation.to_ascii_lowercase().as_str() {
                    "all" | "both" => {
                        notations.push(WiktionaryNotationArg::Phonemes);
                        notations.push(WiktionaryNotationArg::Phones);
                    }
                    "phonemic" | "phoneme" | "phonemes" => {
                        notations.push(WiktionaryNotationArg::Phonemes);
                    }
                    "phonetic" | "phone" | "phones" => {
                        notations.push(WiktionaryNotationArg::Phones);
                    }
                    other => anyhow::bail!(
                        "Invalid Wiktionary train_notations entry `{other}`. Supported: phonemic, phonetic, all"
                    ),
                }
            }
        }
    }

    notations.sort_by_key(|notation| match notation {
        WiktionaryNotationArg::All => 0,
        WiktionaryNotationArg::Phonemes => 1,
        WiktionaryNotationArg::Phones => 2,
    });
    notations.dedup();
    anyhow::ensure!(
        !notations.is_empty(),
        "no Wiktionary training notations configured"
    );
    Ok(notations)
}

fn wiktionary_notation_source_file(data: &Path, notation: WiktionaryNotationArg) -> PathBuf {
    match notation {
        WiktionaryNotationArg::All => unreachable!("all should be expanded before loading files"),
        WiktionaryNotationArg::Phones => data.join("phones.jsonl"),
        WiktionaryNotationArg::Phonemes => data.join("phonemes.jsonl"),
    }
}

fn wiktionary_notation_label(notations: &[WiktionaryNotationArg]) -> String {
    notations
        .iter()
        .map(|notation| match notation {
            WiktionaryNotationArg::All => "all",
            WiktionaryNotationArg::Phones => "phonetic",
            WiktionaryNotationArg::Phonemes => "phonemic",
        })
        .collect::<Vec<_>>()
        .join("+")
}

fn split_wiktionary_examples(
    mut examples: Vec<tongues_wiktionary::TrainingExample>,
    train_frac: f64,
    valid_frac: f64,
    seed: u64,
) -> (
    Vec<tongues_wiktionary::TrainingExample>,
    Vec<tongues_wiktionary::TrainingExample>,
    Vec<tongues_wiktionary::TrainingExample>,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    examples.shuffle(&mut rng);
    let train_len = ((examples.len() as f64) * train_frac).round() as usize;
    let valid_len = ((examples.len() as f64) * valid_frac).round() as usize;
    let train_end = train_len.min(examples.len());
    let valid_end = (train_end + valid_len).min(examples.len());
    let test = examples.split_off(valid_end);
    let valid = examples.split_off(train_end);
    (examples, valid, test)
}

fn build_wiktionary_vocab(
    train: &[tongues_wiktionary::TrainingExample],
    valid: &[tongues_wiktionary::TrainingExample],
) -> Vocab {
    let rows = train
        .iter()
        .chain(valid.iter())
        .cloned()
        .collect::<Vec<_>>();
    tongues_wiktionary::build_stable_wiktionary_vocab(&rows)
}

fn wiktionary_example_fits_vocab(
    example: &tongues_wiktionary::TrainingExample,
    vocab: &Vocab,
) -> bool {
    vocab
        .encode_string(&wiktionary_source_text(example))
        .into_iter()
        .all(|id| id != UNK_ID)
        && vocab
            .encode_string(&example.output)
            .into_iter()
            .all(|id| id != UNK_ID)
}

fn wiktionary_seq2seq_examples(
    rows: &[tongues_wiktionary::TrainingExample],
    vocab: &Vocab,
) -> Vec<Seq2SeqExample> {
    rows.iter()
        .map(|row| {
            let source = wiktionary_source_text(row);
            let mut tgt_in_ids = vec![BOS_ID];
            tgt_in_ids.extend(vocab.encode_string(&row.output));

            let mut tgt_out_ids = vocab.encode_string(&row.output);
            tgt_out_ids.push(EOS_ID);

            Seq2SeqExample {
                src_ids: vocab.encode_string(&source),
                tgt_in_ids,
                tgt_out_ids,
            }
        })
        .collect()
}

fn wiktionary_source_text(example: &tongues_wiktionary::TrainingExample) -> String {
    tongues_wiktionary::normalize_wiktionary_control_tokens(&example.input)
}

fn run_wiktionary_train<B: AutodiffBackend>(
    device: &B::Device,
    model_config: &ModelConfig,
    train_config: &TrainConfig,
    train_examples: &[Seq2SeqExample],
    valid_examples: &[Seq2SeqExample],
    model_path: &Path,
    seed: u64,
) -> Result<()>
where
    <Seq2SeqModel<B> as burn::module::Module<B>>::Record: Send,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let best_loss = train_seq2seq_examples::<B, _>(
        model_config,
        train_config,
        train_examples,
        valid_examples,
        model_path,
        device,
        &mut rng,
    )?;

    println!(
        "\nTraining complete. Best validation loss: {:.4}",
        best_loss
    );
    println!("Model saved to {}", model_path.display());
    Ok(())
}

fn cmd_wiktionary_infer(
    model_dir: &Path,
    task: &str,
    lang: &str,
    notation: WiktionaryNotationArg,
    variety: Option<&str>,
    raw: bool,
    input: &str,
    device_arg: DeviceArg,
    output_mode: OutputMode,
) -> Result<()> {
    let vocab: Vocab = {
        let path = model_dir.join("vocab.json");
        let s = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&s)?
    };
    let model_config: ModelConfig = {
        let path = model_dir.join("model_config.json");
        let s = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&s)?
    };
    let source = if raw {
        input.to_string()
    } else {
        wiktionary_infer_source(task, lang, notation, variety, input)?
    };

    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            run_wiktionary_infer::<CpuInferBackend>(
                &device,
                &model_config,
                model_dir,
                &vocab,
                &source,
                output_mode,
            )
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            run_wiktionary_infer::<CudaInferBackend>(
                &device,
                &model_config,
                model_dir,
                &vocab,
                &source,
                output_mode,
            )
        }
    }
}

fn wiktionary_infer_source(
    task: &str,
    lang: &str,
    notation: WiktionaryNotationArg,
    variety: Option<&str>,
    input: &str,
) -> Result<String> {
    match notation {
        WiktionaryNotationArg::All => {
            anyhow::bail!("Wiktionary inference requires one notation: phones or phonemes")
        }
        WiktionaryNotationArg::Phones | WiktionaryNotationArg::Phonemes => {}
    };
    let normalized = task.to_ascii_lowercase();
    let source = match normalized.as_str() {
        "orthography-to-phonemes" => {
            let mut controls = format!("<task:orthography_to_phonology> <lang:{lang}>");
            if let Some(variety) = variety.filter(|variety| !variety.is_empty()) {
                controls.push_str(&format!(" <variety:{variety}>"));
            }
            controls.push_str(" <repr:phonemes>");
            format!("{controls} {input}")
        }
        "orthography-to-phones" => {
            let mut controls = format!("<task:orthography_to_phonology> <lang:{lang}>");
            if let Some(variety) = variety.filter(|variety| !variety.is_empty()) {
                controls.push_str(&format!(" <variety:{variety}>"));
            }
            controls.push_str(" <repr:phones>");
            format!("{controls} {input}")
        }
        "orthography-to-phonology" => {
            let mut controls = format!("<task:orthography_to_phonology> <lang:{lang}>");
            if let Some(variety) = variety.filter(|variety| !variety.is_empty()) {
                controls.push_str(&format!(" <variety:{variety}>"));
            }
            controls.push_str(&format!(" {}", wiktionary_infer_representation_token(notation)?));
            format!("{controls} {input}")
        }
        "phonemes-to-orthography" => {
            let mut controls = format!("<task:phonology_to_orthography> <lang:{lang}>");
            if let Some(variety) = variety.filter(|variety| !variety.is_empty()) {
                controls.push_str(&format!(" <variety:{variety}>"));
            }
            controls.push_str(" <repr:phonemes>");
            format!("{controls} {input}")
        }
        "phones-to-orthography" => {
            let mut controls = format!("<task:phonology_to_orthography> <lang:{lang}>");
            if let Some(variety) = variety.filter(|variety| !variety.is_empty()) {
                controls.push_str(&format!(" <variety:{variety}>"));
            }
            controls.push_str(" <repr:phones>");
            format!("{controls} {input}")
        }
        "phonology-to-orthography" => {
            let mut controls = format!("<task:phonology_to_orthography> <lang:{lang}>");
            if let Some(variety) = variety.filter(|variety| !variety.is_empty()) {
                controls.push_str(&format!(" <variety:{variety}>"));
            }
            controls.push_str(&format!(" {}", wiktionary_infer_representation_token(notation)?));
            format!("{controls} {input}")
        }
        "phonetic-realization" => {
            let mut controls = format!("<task:phonetic_realization> <lang:{lang}>");
            if let Some(variety) = variety.filter(|variety| !variety.is_empty()) {
                controls.push_str(&format!(" <variety:{variety}>"));
            }
            controls.push_str(" <repr:phonemes>");
            format!("{controls} {input}")
        }
        "segment" | "segment-compound" | "compound-segmentation" => {
            format!("<task:segment_compound> <lang:{lang}> <SEGMENT> {input}")
        }
        "pronounce-segments" | "segments-to-phonology" | "segments-to-phones" => {
            format!(
                "<task:pronounce_segments> <lang:{lang}> <PRONOUNCE_SEGMENTS> <repr:phones> {input}"
            )
        }
        "verify" | "verify-pronunciation" | "verifier" => {
            format!("<task:verify_pronunciation> <lang:{lang}> <VERIFY> {input}")
        }
        "normalize-phonology" | "normalise-phonology" | "broad-equivalence" => {
            format!("<task:normalize_phonology> <lang:{lang}> <BROAD_EQUIV> <repr:phones> {input}")
        }
        "find-etymology" | "etymology-from-word" | "word-etymology" => {
            format!("<task:find_etymology> <lang:{lang}> {input}")
        }
        "normalize" | "normalise" => {
            format!("<task:normalize> <lang:{lang}> {input}")
        }
        "guess-lang-from-orthography" | "lang-from-orthography" => {
            let representation_token = wiktionary_infer_representation_token(notation)?;
            format!("<task:guess_lang_from_orthography> {representation_token} {input}")
        }
        "guess-lang-from-phonology" | "lang-from-phonology" => {
            let representation_token = wiktionary_infer_representation_token(notation)?;
            format!("<task:guess_lang_from_phonology> {representation_token} {input}")
        }
        "guess-lang-from-orthography-and-phonology" | "lang" | "language" | "language-guessing" => {
            let representation_token = wiktionary_infer_representation_token(notation)?;
            format!(
                "<task:guess_lang_from_orthography_and_phonology> {representation_token} {input}"
            )
        }
        _ => anyhow::bail!(
            "Invalid Wiktionary inference task. Supported: orthography-to-phonemes, orthography-to-phones, phonemes-to-orthography, phones-to-orthography, phonetic-realization, find-etymology, segment-compound, pronounce-segments, verify-pronunciation, normalize-phonology, normalize, guess-lang-from-orthography, guess-lang-from-phonology, guess-lang-from-orthography-and-phonology"
        ),
    };
    Ok(tongues_wiktionary::normalize_wiktionary_control_tokens(
        &source,
    ))
}

fn wiktionary_infer_representation_token(notation: WiktionaryNotationArg) -> Result<&'static str> {
    match notation {
        WiktionaryNotationArg::All => {
            anyhow::bail!("Wiktionary inference requires one notation: phones or phonemes")
        }
        WiktionaryNotationArg::Phones => Ok("<repr:phones>"),
        WiktionaryNotationArg::Phonemes => Ok("<repr:phonemes>"),
    }
}

fn run_wiktionary_infer<B: Backend>(
    device: &B::Device,
    model_config: &ModelConfig,
    model_dir: &Path,
    vocab: &Vocab,
    source: &str,
    output_mode: OutputMode,
) -> Result<()> {
    let model = load_model::<B>(model_config, &model_dir.join("model"), device)?;
    let src_ids = vocab.encode_string(source);
    let unknown_count = src_ids.iter().filter(|&&id| id == UNK_ID).count();
    if unknown_count > 0 && output_mode.verbose() {
        eprintln!("warning: source encoded with {unknown_count} <UNK> token(s)");
    }

    let src_len = src_ids.len();
    let src_tensor = Tensor::<B, 2, Int>::from_data(
        burn::tensor::TensorData::new(
            src_ids.iter().map(|&x| x as i32).collect::<Vec<_>>(),
            [1, src_len],
        ),
        device,
    );
    let pred_ids = model.generate(src_tensor, 128);
    let output = vocab.decode_ids(&pred_ids);

    if output_mode.verbose() {
        println!("Source:\n  {source}");
        println!("\nPrediction output:\n  {output}");
    } else {
        println!("{output}");
    }
    Ok(())
}

fn run_interpretation_command(
    command: InterpretationCommands,
    device_arg: DeviceArg,
    _output_mode: OutputMode,
) -> Result<()> {
    match command {
        InterpretationCommands::Clean(args) => cmd_clean_family(
            "interpretation",
            &args,
            DEFAULT_INTERPRETATION_DATA_DIR,
            DEFAULT_INTERPRETATION_MODEL_DIR,
        ),
        InterpretationCommands::Prepare {
            subset,
            out,
            max_utterances,
            wiktionary_audio_data,
            no_wiktionary_audio,
            max_wiktionary_audio,
            no_download_wiktionary_audio,
            whisper_model,
            no_whisper_transcripts,
            max_whisper_wer,
        } => {
            let subset = LibriSpeechSubset::parse(&subset).ok_or_else(|| {
                anyhow::anyhow!(
                    "invalid LibriSpeech subset `{subset}`; supported: mini, train-clean-100"
                )
            })?;
            anyhow::ensure!(
                (0.0..=1.0).contains(&max_whisper_wer),
                "--max-whisper-wer must be between 0.0 and 1.0"
            );
            let mut config = InterpretationConfig {
                subset,
                dataset_id: subset.dataset_id().to_string(),
                download_url: subset.archive_url().to_string(),
                ..InterpretationConfig::default()
            };
            config.max_utterances = max_utterances;
            if let Some(path) = wiktionary_audio_data {
                config.wiktionary_audio_data_dir = Some(path.display().to_string());
            }
            if no_wiktionary_audio {
                config.wiktionary_audio_data_dir = None;
            }
            if let Some(max_wiktionary_audio) = max_wiktionary_audio {
                config.max_wiktionary_audio = Some(max_wiktionary_audio);
            }
            if no_download_wiktionary_audio {
                config.download_wiktionary_audio = false;
            }
            if let Some(wiktionary_data_dir) = &config.wiktionary_audio_data_dir {
                ensure_wiktionary_audio_dataset_available(Path::new(wiktionary_data_dir))?;
            }
            let pb = status_spinner(format!(
                "Preparing interpretation dataset at {}",
                out.display()
            ));
            let progress = {
                let pb = pb.clone();
                move |progress| {
                    update_interpretation_prepare_progress(&pb, progress);
                }
            };
            let report = if no_whisper_transcripts {
                tongues_interpretation::prepare_dataset_with_progress(&out, &config, progress)?
            } else {
                let model_path = match whisper_model {
                    Some(path) => path,
                    None => models::ensure_asr_whisper_model_available()?,
                };
                pb.set_message(format!(
                    "Loading Whisper transcript model from {}",
                    model_path.display()
                ));
                let mut recognizer = WhisperSpeechRecognizer::new_quiet(&model_path)
                    .with_context(|| format!("loading Whisper model {}", model_path.display()))?;
                tongues_interpretation::prepare_dataset_with_progress_and_transcript_refiner(
                    &out,
                    &config,
                    progress,
                    move |utterance_id, audio_path, samples, original_transcript| {
                        recognizer.push_frame(&AudioFrame {
                            sample_rate_hz: tongues_interpretation::DEFAULT_SAMPLE_RATE_HZ,
                            channels: 1,
                            samples: samples.to_vec(),
                        })?;
                        let recognition = recognizer
                            .poll_timed_transcript_with_finality(true)
                            .with_context(|| {
                                format!(
                                    "Whisper transcription failed for {} ({})",
                                    utterance_id,
                                    audio_path.display()
                                )
                            })?;
                        let whisper_text = recognition.text.trim();
                        if whisper_text.is_empty() {
                            return Ok(TranscriptRefinement::Omit {
                                reason: "Whisper returned an empty transcript".to_string(),
                                source_transcript: None,
                                whisper_transcript: None,
                                wer: None,
                                max_wer: None,
                            });
                        }
                        let wer = transcript_word_error_rate(original_transcript, whisper_text);
                        if wer > max_whisper_wer {
                            return Ok(whisper_transcript_divergence_omit(
                                original_transcript,
                                whisper_text,
                                wer,
                                max_whisper_wer,
                            ));
                        }
                        Ok(TranscriptRefinement::Use(whisper_text.to_string()))
                    },
                )?
            };
            finish_status(
                pb,
                format!(
                    "Prepared interpretation dataset at {}: {} train / {} valid / {} test utterances",
                    out.display(),
                    format_count(report.train_examples),
                    format_count(report.valid_examples),
                    format_count(report.test_examples)
                ),
            );
            Ok(())
        }
        InterpretationCommands::Train {
            data,
            out,
            wait_for_prepare,
            epochs,
            batch_size,
            seed,
        } => {
            if wait_for_prepare {
                wait_for_prepared_dataset(
                    &data,
                    &[
                        "vocab.json",
                        "phoneme_vocab.json",
                        "phone_vocab.json",
                        "word_vocab.json",
                        "train.jsonl",
                        "valid.jsonl",
                    ],
                    "interpretation",
                )?;
            }
            if !data.join("vocab.json").exists()
                || !data.join("phoneme_vocab.json").exists()
                || !data.join("phone_vocab.json").exists()
                || !data.join("word_vocab.json").exists()
                || !data.join("train.jsonl").exists()
                || !data.join("valid.jsonl").exists()
            {
                let config = InterpretationConfig::default();
                if let Some(wiktionary_data_dir) = &config.wiktionary_audio_data_dir {
                    ensure_wiktionary_audio_dataset_available(Path::new(wiktionary_data_dir))?;
                }
                let pb = status_spinner(format!(
                    "Training data missing; preparing LibriSpeech ASR dataset at {}",
                    data.display()
                ));
                let progress = {
                    let pb = pb.clone();
                    move |progress| update_interpretation_prepare_progress(&pb, progress)
                };
                let model_path = models::ensure_asr_whisper_model_available()?;
                pb.set_message(format!(
                    "Loading Whisper transcript model from {}",
                    model_path.display()
                ));
                let mut recognizer = WhisperSpeechRecognizer::new_quiet(&model_path)
                    .with_context(|| format!("loading Whisper model {}", model_path.display()))?;
                tongues_interpretation::prepare_dataset_with_progress_and_transcript_refiner(
                    &data,
                    &config,
                    progress,
                    move |utterance_id, audio_path, samples, original_transcript| {
                        recognizer.push_frame(&AudioFrame {
                            sample_rate_hz: tongues_interpretation::DEFAULT_SAMPLE_RATE_HZ,
                            channels: 1,
                            samples: samples.to_vec(),
                        })?;
                        let recognition = recognizer
                            .poll_timed_transcript_with_finality(true)
                            .with_context(|| {
                                format!(
                                    "Whisper transcription failed for {} ({})",
                                    utterance_id,
                                    audio_path.display()
                                )
                            })?;
                        let whisper_text = recognition.text.trim();
                        if whisper_text.is_empty() {
                            return Ok(TranscriptRefinement::Omit {
                                reason: "Whisper returned an empty transcript".to_string(),
                                source_transcript: None,
                                whisper_transcript: None,
                                wer: None,
                                max_wer: None,
                            });
                        }
                        let wer = transcript_word_error_rate(original_transcript, whisper_text);
                        if wer > DEFAULT_WHISPER_TRANSCRIPT_MAX_WER {
                            return Ok(whisper_transcript_divergence_omit(
                                original_transcript,
                                whisper_text,
                                wer,
                                DEFAULT_WHISPER_TRANSCRIPT_MAX_WER,
                            ));
                        }
                        Ok(TranscriptRefinement::Use(whisper_text.to_string()))
                    },
                )?;
                finish_status(pb, format!("Prepared {}", data.display()));
            }
            let mut train_config = InterpretationTrainConfig::default();
            if let Some(epochs) = epochs {
                train_config.epochs = epochs;
            }
            if let Some(batch_size) = batch_size {
                train_config.batch_size = batch_size;
            }
            if let Some(seed) = seed {
                train_config.seed = seed;
            }
            let mut train_config = train_config;
            train_config.input_feature_bins = interpretation_feature_bins(&data)?;
            cmd_interpretation_train(&data, &out, &train_config, device_arg)
        }
        InterpretationCommands::Eval { model, data, split } => {
            cmd_interpretation_eval(&model, &data, &split, device_arg)
        }
        InterpretationCommands::Stream { model, wav } => {
            cmd_interpretation_stream(&model, &wav, device_arg)
        }
    }
}

fn run_common_phone_command(command: CommonPhoneCommands) -> Result<()> {
    match command {
        CommonPhoneCommands::Clean(args) => cmd_clean_family(
            "common-phone",
            &args,
            DEFAULT_COMMON_PHONE_DATA_DIR,
            DEFAULT_COMMON_PHONE_MODEL_DIR,
        ),
        CommonPhoneCommands::ListenDevices => cmd_common_phone_listen_devices(),
        CommonPhoneCommands::Listen {
            model,
            task,
            device,
            input_device,
            sample_rate,
            chunk_ms,
            context_ms,
            show_phones,
            show_features,
            debug_frames,
            dry_run,
            phones2orth,
        } => {
            anyhow::ensure!(
                device == "cpu",
                "common-phone listen v0 currently supports --device cpu only"
            );
            if phones2orth.is_some() {
                println!(
                    "phones2orth rough text is accepted for future use but is not wired in v0"
                );
            }
            let task = tongues_common_phone::CommonPhoneTask::parse(&task)?;
            cmd_common_phone_listen(CommonPhoneListenOptions {
                model,
                task,
                input_device,
                sample_rate,
                chunk_ms,
                context_ms,
                show_phones,
                show_features,
                debug_frames,
                dry_run,
            })
        }
        CommonPhoneCommands::Fetch {
            out,
            source,
            languages,
        } => {
            let source_lc = source.to_ascii_lowercase();
            if source_lc == "zenodo" {
                let pb = status_spinner(format!(
                    "Downloading Common Phone from Zenodo into {}",
                    out.display()
                ));
                let progress = {
                    let pb = pb.clone();
                    move |progress| pb.set_message(common_phone_prepare_progress_message(progress))
                };
                tongues_common_phone::download_common_phone_zenodo(
                    &out,
                    tongues_common_phone::DEFAULT_ZENODO_URL,
                    progress,
                )?;
                finish_status(
                    pb,
                    format!(
                        "Downloaded and extracted Common Phone into {}",
                        out.display()
                    ),
                );
                return Ok(());
            }
            fs::create_dir_all(out.join("audio"))
                .with_context(|| format!("creating {}", out.join("audio").display()))?;
            fs::write(
                out.join("README.md"),
                format!(
                    "# Common Phone raw data\n\nSource: `{source}`\nLanguages: `{}`\n\nPlace `metadata.jsonl` plus WAV files under `audio/` or `clips/` here. Required metadata fields: `utterance_id`, `language`, `wav_path`, `phones`.\n\nExample row:\n\n```json\n{{\"utterance_id\":\"cp_eng_000001\",\"language\":\"eng\",\"split\":\"train\",\"wav_path\":\"audio/sample_000001.wav\",\"phones\":\"t ɪ p\"}}\n```\n\nv0 does not download Hugging Face data automatically; acquire/export it externally, then run `common-phone prepare --input {}`.\n",
                    languages.unwrap_or_else(|| "all requested externally".to_string()),
                    out.display()
                ),
            )?;
            println!(
                "Created Common Phone raw-data layout at {}. Add metadata.jsonl and WAV files, then run prepare.",
                out.display()
            );
            Ok(())
        }
        CommonPhoneCommands::Prepare {
            input,
            download,
            source,
            source_url,
            out,
            lang,
            max_utterances,
            sample_rate,
            valid_ratio,
            test_ratio,
            seed,
        } => {
            anyhow::ensure!(
                (0.0..1.0).contains(&valid_ratio) && (0.0..1.0).contains(&test_ratio),
                "--valid-ratio and --test-ratio must be between 0.0 and 1.0"
            );
            anyhow::ensure!(
                valid_ratio + test_ratio < 1.0,
                "--valid-ratio plus --test-ratio must be less than 1.0"
            );
            if download {
                let source_lc = source.to_ascii_lowercase();
                anyhow::ensure!(
                    source_lc == "zenodo",
                    "common-phone prepare --download currently supports --source zenodo"
                );
                let url = source_url
                    .as_deref()
                    .unwrap_or(tongues_common_phone::DEFAULT_ZENODO_URL);
                let pb = status_spinner(format!(
                    "Downloading Common Phone source data into {}",
                    input.display()
                ));
                let progress = {
                    let pb = pb.clone();
                    move |progress| pb.set_message(common_phone_prepare_progress_message(progress))
                };
                tongues_common_phone::download_common_phone_zenodo(&input, url, progress)?;
                finish_status(
                    pb,
                    format!("Common Phone source ready at {}", input.display()),
                );
            }
            let config = tongues_common_phone::CommonPhoneConfig {
                input: input.display().to_string(),
                languages: lang
                    .unwrap_or_default()
                    .split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string)
                    .collect(),
                max_utterances,
                sample_rate_hz: sample_rate,
                valid_ratio,
                test_ratio,
                seed,
                ..tongues_common_phone::CommonPhoneConfig::default()
            };
            let pb = status_spinner(format!(
                "Preparing Common Phone compact-frame dataset at {}",
                out.display()
            ));
            let progress = {
                let pb = pb.clone();
                move |progress| pb.set_message(common_phone_prepare_progress_message(progress))
            };
            let report =
                tongues_common_phone::prepare_dataset_with_progress(&out, &config, progress)?;
            finish_status(
                pb,
                format!(
                    "Prepared Common Phone at {}: {} train / {} valid / {} test utterances, {} bins",
                    out.display(),
                    format_count(report.train_examples),
                    format_count(report.valid_examples),
                    format_count(report.test_examples),
                    report.feature_bins
                ),
            );
            if !report.unknown_phone_symbols.is_empty() {
                println!(
                    "Unknown phone mappings: {}",
                    serde_json::to_string(&report.unknown_phone_symbols)?
                );
            }
            Ok(())
        }
        CommonPhoneCommands::Train {
            data,
            model,
            task,
            epochs,
            batch_frames,
            lr,
            seed,
            device,
        } => {
            anyhow::ensure!(
                device == "cpu",
                "common-phone v0 currently supports --device cpu only"
            );
            let mut config = tongues_common_phone::CommonPhoneTrainConfig::default();
            config.task = tongues_common_phone::CommonPhoneTask::parse(&task)?;
            if let Some(epochs) = epochs {
                config.epochs = epochs;
            }
            if let Some(batch_frames) = batch_frames {
                config.batch_frames = batch_frames;
            }
            if let Some(lr) = lr {
                config.learning_rate = lr;
            }
            if let Some(seed) = seed {
                config.seed = seed;
            }
            println!("Common Phone compact-frame CTC checkpoint paths:");
            println!(
                "  train_state: {}",
                model.join("train_state.json").display()
            );
            println!(
                "  epoch checkpoints: {}",
                model.join("model-epoch-N.bin").display()
            );
            println!("  best model: {}", model.join("model.bin").display());
            println!("  CTC heads: phones, phonemes, manner, place, voicing, syllabic, height, backness, rounding");
            println!(
                "  latest model: {}",
                model.join("model-latest.bin").display()
            );
            println!(
                "  minibatch checkpoint cadence: every 2,000 minibatches -> {}",
                model.join("model-latest.bin").display()
            );
            println!("  Note: v0 writes model-latest.bin at initialization and every 2,000 minibatches during epochs; epoch checkpoints and best model are written after validation.");
            let pb = status_spinner(format!(
                "Training Common Phone compact-frame CTC scaffold from {}",
                data.display()
            ));
            let progress = {
                let pb = pb.clone();
                move |progress| pb.set_message(common_phone_train_progress_message(progress))
            };
            let report =
                tongues_common_phone::train_with_progress(&data, &model, &config, progress)?;
            finish_status(
                pb,
                format!(
                    "Common Phone training complete: {} epochs, best valid error {:.4}",
                    report.epochs, report.best_validation_error_rate
                ),
            );
            Ok(())
        }
        CommonPhoneCommands::Eval {
            model,
            data,
            task,
            split,
            samples,
        } => {
            let task = tongues_common_phone::CommonPhoneTask::parse(&task)?;
            let report = tongues_common_phone::evaluate(&model, &data, &split, task, samples)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        CommonPhoneCommands::ShowRow { data, index } => {
            let row = tongues_common_phone::show_row(&data, index)?;
            println!("{}", serde_json::to_string_pretty(&row)?);
            Ok(())
        }
    }
}

fn common_phone_prepare_progress_message(
    progress: tongues_common_phone::PrepareProgress,
) -> String {
    match progress {
        tongues_common_phone::PrepareProgress::Stage { message } => message,
        tongues_common_phone::PrepareProgress::Parse { rows, path } => {
            format!(
                "Parsed {} Common Phone rows from {path}",
                format_count(rows)
            )
        }
        tongues_common_phone::PrepareProgress::Download { url, path, bytes } => {
            format!(
                "Downloading {url} -> {path} ({} bytes)",
                format_count(bytes)
            )
        }
        tongues_common_phone::PrepareProgress::Extract { path } => {
            format!("Extracting {path}")
        }
        tongues_common_phone::PrepareProgress::Features {
            utterance_id,
            frames,
            path,
        } => format!(
            "Wrote {} compact frames for {utterance_id} -> {path}",
            format_count(frames)
        ),
        tongues_common_phone::PrepareProgress::Reuse {
            utterance_id,
            frames,
            path,
        } => format!(
            "Reusing {} compact frames for {utterance_id} from {path}",
            format_count(frames)
        ),
        tongues_common_phone::PrepareProgress::Write { path, rows } => {
            format!("Wrote {} rows to {path}", format_count(rows))
        }
        tongues_common_phone::PrepareProgress::Select { selected, total } => {
            format!(
                "Selected {} of {} Common Phone rows",
                format_count(selected),
                format_count(total)
            )
        }
        tongues_common_phone::PrepareProgress::Split { train, valid, test } => {
            format!(
                "Split rows: {} train / {} valid / {} test",
                format_count(train),
                format_count(valid),
                format_count(test)
            )
        }
        tongues_common_phone::PrepareProgress::Vocab { name, tokens, path } => {
            format!("Wrote {name}: {} tokens -> {path}", format_count(tokens))
        }
        tongues_common_phone::PrepareProgress::State { status, rows, path } => {
            format!(
                "Updated prepare_state status={status} rows={} -> {path}",
                format_count(rows)
            )
        }
    }
}

fn common_phone_train_progress_message(progress: tongues_common_phone::TrainProgress) -> String {
    match progress {
        tongues_common_phone::TrainProgress::Startup {
            train_examples,
            valid_examples,
            epochs,
            train_state_path,
            epoch_checkpoint_pattern,
            latest_checkpoint_path,
            minibatch_checkpoint_interval,
            best_model_path,
        } => format!(
            "Training {} train / {} valid examples for {} epochs; state={train_state_path}; epoch checkpoints={epoch_checkpoint_pattern}; minibatch checkpoints every {} -> {latest_checkpoint_path}; best={best_model_path}",
            format_count(train_examples),
            format_count(valid_examples),
            format_count(epochs),
            format_count(minibatch_checkpoint_interval)
        ),
        tongues_common_phone::TrainProgress::EpochStart {
            epoch,
            epochs,
            train_examples,
        } => format!(
            "Epoch {epoch}/{epochs}: training {} examples",
            format_count(train_examples)
        ),
        tongues_common_phone::TrainProgress::Resume {
            epoch,
            checkpoint_path,
            status,
        } => format!(
            "Resuming Common Phone training at epoch {epoch} from {checkpoint_path} ({status})"
        ),
        tongues_common_phone::TrainProgress::Batch {
            epoch,
            examples,
            total_examples,
            loss,
        } => format!(
            "Epoch {epoch}: trained {}/{} examples, mean loss {:.4}",
            format_count(examples),
            format_count(total_examples),
            loss
        ),
        tongues_common_phone::TrainProgress::EpochComplete {
            epoch,
            train_loss,
            valid_error,
            exact_sequence_accuracy,
            blank_ratio,
        } => format!(
            "Epoch {epoch} complete: loss {:.4}, valid error {:.4}, exact {:.3}, blank {:.3}",
            train_loss, valid_error, exact_sequence_accuracy, blank_ratio
        ),
        tongues_common_phone::TrainProgress::Checkpoint { epoch, path, best } => {
            let kind = if best { "best model" } else { "checkpoint" };
            format!("Epoch {epoch}: wrote {kind} -> {path}")
        }
        tongues_common_phone::TrainProgress::State { epoch, path } => {
            format!("Epoch {epoch}: updated train_state -> {path}")
        }
    }
}

struct CommonPhoneListenOptions {
    model: Option<PathBuf>,
    task: tongues_common_phone::CommonPhoneTask,
    input_device: Option<String>,
    sample_rate: u32,
    chunk_ms: u64,
    context_ms: u64,
    show_phones: bool,
    show_features: bool,
    debug_frames: bool,
    dry_run: bool,
}

fn cmd_common_phone_listen_devices() -> Result<()> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|device| device.name().ok());
    println!("CPAL input devices:");
    for device in host.input_devices()? {
        let name = device.name().unwrap_or_else(|_| "<unnamed>".to_string());
        let marker = if Some(name.as_str()) == default_name.as_deref() {
            " (default)"
        } else {
            ""
        };
        println!("  {name}{marker}");
    }
    Ok(())
}

fn cmd_common_phone_listen(options: CommonPhoneListenOptions) -> Result<()> {
    use cpal::traits::{DeviceTrait, StreamTrait};

    anyhow::ensure!(
        options.dry_run || options.model.is_some(),
        "common-phone listen requires --model unless --dry-run is set"
    );
    let decoder = if options.dry_run {
        None
    } else {
        Some(tongues_common_phone::CommonPhoneLiveDecoder::load(
            options.model.as_ref().expect("checked above"),
            options.task,
        )?)
    };
    let mut frame_config = if let Some(decoder) = &decoder {
        decoder.config(options.sample_rate, 25.0, 10.0)
    } else {
        tongues_common_phone::CommonPhoneConfig {
            sample_rate_hz: options.sample_rate,
            ..tongues_common_phone::CommonPhoneConfig::default()
        }
    };
    frame_config.sample_rate_hz = options.sample_rate;

    let host = cpal::default_host();
    let device = select_input_device(&host, options.input_device.as_deref())?;
    let device_name = device.name().unwrap_or_else(|_| "<unnamed>".to_string());
    let supported = device.default_input_config()?;
    let native_sample_rate = supported.sample_rate().0;
    let channels = supported.channels() as usize;
    let stream_config: cpal::StreamConfig = supported.clone().into();
    let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let err_fn = |err| eprintln!("common-phone listen stream error: {err}");
    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => {
            let callback_buffer = Arc::clone(&buffer);
            device.build_input_stream(
                &stream_config,
                move |data: &[f32], _| append_mono_samples(data, channels, &callback_buffer),
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            let callback_buffer = Arc::clone(&buffer);
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _| {
                    let converted = data
                        .iter()
                        .map(|sample| *sample as f32 / i16::MAX as f32)
                        .collect::<Vec<_>>();
                    append_mono_samples(&converted, channels, &callback_buffer);
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            let callback_buffer = Arc::clone(&buffer);
            device.build_input_stream(
                &stream_config,
                move |data: &[u16], _| {
                    let converted = data
                        .iter()
                        .map(|sample| (*sample as f32 / u16::MAX as f32) * 2.0 - 1.0)
                        .collect::<Vec<_>>();
                    append_mono_samples(&converted, channels, &callback_buffer);
                },
                err_fn,
                None,
            )?
        }
        other => anyhow::bail!("unsupported CPAL input sample format: {other:?}"),
    };

    println!("listening on {device_name}...");
    if options.debug_frames {
        println!("input device: {device_name}");
        println!("native sample rate: {native_sample_rate}");
        println!("model sample rate: {}", frame_config.sample_rate_hz);
        println!(
            "resampler: {}",
            if native_sample_rate == frame_config.sample_rate_hz {
                "disabled"
            } else {
                "enabled"
            }
        );
        println!("chunk_ms: {}", options.chunk_ms);
        println!("context_ms: {}", options.context_ms);
        println!("frame_dim: {}", frame_config.feature_bins);
    }
    stream.play()?;
    let mut elapsed_ms = 0u64;
    let mut in_speech = false;
    let mut last_line = String::new();
    let context_samples_native =
        (native_sample_rate as usize * options.context_ms as usize / 1000).max(1);
    loop {
        std::thread::sleep(Duration::from_millis(options.chunk_ms.max(10)));
        elapsed_ms += options.chunk_ms.max(10);
        let window = {
            let mut guard = buffer.lock().expect("common phone audio buffer poisoned");
            if guard.len() > context_samples_native {
                let remove = guard.len() - context_samples_native;
                guard.drain(..remove);
            }
            guard.clone()
        };
        if window.len() < (native_sample_rate as usize * options.chunk_ms as usize / 1000).max(1) {
            continue;
        }
        let stats =
            tongues_common_phone::live_frame_stats(&window, native_sample_rate, &frame_config);
        let speech = stats.vad > 0.15 || stats.rms > 0.01;
        if options.dry_run {
            print_common_phone_debug(elapsed_ms, &stats, 0.0, 0, "");
            continue;
        }
        let Some(decoder) = &decoder else {
            continue;
        };
        if !speech {
            if in_speech {
                println!("speech end");
                in_speech = false;
                last_line.clear();
            } else if options.debug_frames {
                print_common_phone_debug(elapsed_ms, &stats, 0.0, 0, "silence");
            }
            continue;
        }
        if !in_speech {
            println!("speech start");
            in_speech = true;
        }
        let decoded = decoder.decode_samples(&window, native_sample_rate, &frame_config)?;
        let phone_line = decoded.phones.join(" ");
        let feature_line = decoded.feature_bundles.join(" ");
        if options.debug_frames {
            print_common_phone_debug(
                elapsed_ms,
                &decoded.stats,
                decoded.blank_ratio,
                decoded.prediction_length,
                &phone_line,
            );
        }
        let effective_show_phones = options.show_phones || !options.show_features;
        if effective_show_phones && !phone_line.is_empty() && phone_line != last_line {
            println!(
                "[{:04.1}s] phones: {phone_line}",
                elapsed_ms as f32 / 1000.0
            );
            last_line = phone_line.clone();
        }
        if options.show_features && !feature_line.is_empty() {
            println!("features: {feature_line}");
        }
    }
}

fn select_input_device(host: &cpal::Host, name: Option<&str>) -> Result<cpal::Device> {
    use cpal::traits::{DeviceTrait, HostTrait};

    if let Some(name) = name {
        let needle = name.to_lowercase();
        for device in host.input_devices()? {
            let device_name = device.name().unwrap_or_default();
            if device_name.to_lowercase().contains(&needle) {
                return Ok(device);
            }
        }
        anyhow::bail!("no CPAL input device matching `{name}`");
    }
    host.default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no default CPAL input device available"))
}

fn append_mono_samples(samples: &[f32], channels: usize, buffer: &Arc<Mutex<Vec<f32>>>) {
    let channels = channels.max(1);
    let mut guard = buffer.lock().expect("common phone audio buffer poisoned");
    for frame in samples.chunks(channels) {
        guard.push(frame.iter().copied().sum::<f32>() / frame.len().max(1) as f32);
    }
}

fn print_common_phone_debug(
    elapsed_ms: u64,
    stats: &tongues_common_phone::LiveFrameStats,
    blank_ratio: f64,
    pred_len: usize,
    phones: &str,
) {
    println!(
        "[{:04.1}s] rms={:.3} vad={:.2} frames={} blank={:.2} pred_len={} phones=\"{}\"",
        elapsed_ms as f32 / 1000.0,
        stats.rms,
        stats.vad,
        stats.frames,
        blank_ratio,
        pred_len,
        phones
    );
}

fn run_emotions_command(command: EmotionCommands) -> Result<()> {
    match command {
        EmotionCommands::Clean(args) => cmd_clean_family(
            "emotions",
            &args,
            DEFAULT_EMOTIONS_DATA_DIR,
            DEFAULT_EMOTIONS_MODEL_DIR,
        ),
        EmotionCommands::Prepare {
            config,
            source_manifest,
            out,
            cuts_per_wav,
            min_cut_ms,
            max_cut_ms,
            no_full_cut,
            mel_bins,
            seed,
        } => {
            let mut config = read_emotion_prepare_config(&config)?;
            if let Some(source_manifest) = source_manifest {
                config.source_manifest = source_manifest;
            }
            if let Some(cuts_per_wav) = cuts_per_wav {
                config.cuts_per_wav = cuts_per_wav;
            }
            if let Some(min_cut_ms) = min_cut_ms {
                config.min_cut_ms = min_cut_ms;
            }
            if let Some(max_cut_ms) = max_cut_ms {
                config.max_cut_ms = max_cut_ms;
            }
            if no_full_cut {
                config.include_full_cut = false;
            }
            if let Some(mel_bins) = mel_bins {
                config.mel_bins = mel_bins;
            }
            if let Some(seed) = seed {
                config.seed = seed;
            }
            let pb = status_spinner(format!("Preparing emotion cuts at {}", out.display()));
            let progress = {
                let pb = pb.clone();
                move |progress| update_emotion_prepare_progress(&pb, progress)
            };
            let report = tongues_emotions::prepare_dataset_with_progress(&out, &config, progress)?;
            finish_status(
                pb,
                format!(
                    "Prepared emotion dataset at {}: {} train / {} valid / {} test cuts across {} labels",
                    out.display(),
                    format_count(report.train_examples),
                    format_count(report.valid_examples),
                    format_count(report.test_examples),
                    format_count(report.labels.len())
                ),
            );
            Ok(())
        }
        EmotionCommands::Train {
            data,
            out,
            epochs,
            batch_size,
            learning_rate,
            patience,
            seed,
        } => {
            let mut config = tongues_emotions::EmotionTrainConfig::default();
            if let Some(epochs) = epochs {
                config.epochs = epochs;
            }
            if let Some(batch_size) = batch_size {
                config.batch_size = batch_size;
            }
            if let Some(learning_rate) = learning_rate {
                config.learning_rate = learning_rate;
            }
            if let Some(patience) = patience {
                config.early_stopping_patience = patience;
            }
            if let Some(seed) = seed {
                config.seed = seed;
            }
            println!("Emotion training artifacts:");
            println!("  train_state: {}", out.join("train_state.json").display());
            println!(
                "  epoch checkpoints: {}",
                out.join("model-epoch-N.json").display()
            );
            println!("  best model: {}", out.join("model.json").display());
            println!(
                "  schedule: max_epochs={} early_stopping_patience={} batch_size={} learning_rate={}",
                format_count(config.epochs),
                format_count(config.early_stopping_patience),
                format_count(config.batch_size),
                config.learning_rate
            );
            let best_loss = tongues_emotions::train(&data, &out, &config)?;
            println!(
                "Emotion training complete. Best validation loss: {:.4}",
                best_loss
            );
            Ok(())
        }
        EmotionCommands::Eval { model, data, split } => {
            let report = tongues_emotions::evaluate(&model, &data, &split)?;
            println!(
                "{} examples={} accuracy={:.3} loss={:.4}",
                report.split,
                format_count(report.examples),
                report.accuracy,
                report.loss
            );
            println!("labels: {}", report.labels.join(", "));
            println!("confusion rows=true cols=pred");
            for (label, row) in report.labels.iter().zip(report.confusion.iter()) {
                println!(
                    "  {label}: {}",
                    row.iter()
                        .map(|value| value.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
            Ok(())
        }
        EmotionCommands::Infer { model, wav } => {
            let scores = tongues_emotions::infer(&model, &wav)?;
            for (label, probability) in scores {
                println!("{label}\t{probability:.4}");
            }
            Ok(())
        }
    }
}

fn update_emotion_prepare_progress(
    pb: &indicatif::ProgressBar,
    progress: tongues_emotions::PrepareProgress,
) {
    let message = match progress {
        tongues_emotions::PrepareProgress::Stage { message } => message,
        tongues_emotions::PrepareProgress::Source { rows, labels } => format!(
            "Loaded {} source WAV rows across {} labels",
            format_count(rows),
            format_count(labels)
        ),
        tongues_emotions::PrepareProgress::Cut {
            path,
            cuts_done,
            cuts_total,
            out_path,
        } => format!(
            "Prepared {} / {} cuts from {} -> {}",
            format_count(cuts_done),
            format_count(cuts_total),
            path,
            out_path
        ),
        tongues_emotions::PrepareProgress::Write { split, rows, path } => {
            format!("Wrote {} {} rows to {}", format_count(rows), split, path)
        }
    };
    pb.set_message(message);
}

fn interpretation_prepare_progress_message(
    progress: tongues_interpretation::PrepareProgress,
) -> String {
    match progress {
        tongues_interpretation::PrepareProgress::Stage { message } => message,
        tongues_interpretation::PrepareProgress::Download { url, path, bytes } => {
            format!(
                "Downloading {} to {} ({} bytes)",
                url,
                path,
                format_count(bytes)
            )
        }
        tongues_interpretation::PrepareProgress::Extract { path } => {
            format!("Extracting {}", path)
        }
        tongues_interpretation::PrepareProgress::Parse { transcripts } => {
            format!("Parsed {} transcript rows", format_count(transcripts))
        }
        tongues_interpretation::PrepareProgress::Features {
            utterance_id,
            rows,
            path,
        } => format!(
            "Extracted {} Mel frames for {} -> {}",
            format_count(rows),
            utterance_id,
            path
        ),
        tongues_interpretation::PrepareProgress::Reuse {
            utterance_id,
            rows,
            path,
        } => format!(
            "Reusing {} Mel frames for {} -> {}",
            format_count(rows),
            utterance_id,
            path
        ),
        tongues_interpretation::PrepareProgress::Transcribe { utterance_id, path } => {
            format!(
                "Preparing transcript supervision for {} from {}",
                utterance_id, path
            )
        }
        tongues_interpretation::PrepareProgress::ImportAudio { source, rows } => {
            format!(
                "Importing {} Wiktionary audio rows from {}",
                format_count(rows),
                source
            )
        }
        tongues_interpretation::PrepareProgress::Omit {
            utterance_id,
            reason,
            wer,
            max_wer,
            ..
        } => {
            if let (Some(wer), Some(max_wer)) = (wer, max_wer) {
                format!(
                    "Omitting {}: {} (WER {:.2} > {:.2})",
                    utterance_id, reason, wer, max_wer
                )
            } else {
                format!("Omitting {}: {}", utterance_id, reason)
            }
        }
        tongues_interpretation::PrepareProgress::Write { path, rows } => {
            format!("Wrote {} rows to {}", format_count(rows), path)
        }
    }
}

fn update_interpretation_prepare_progress(
    pb: &indicatif::ProgressBar,
    progress: tongues_interpretation::PrepareProgress,
) {
    let warning = match &progress {
        tongues_interpretation::PrepareProgress::Omit {
            utterance_id,
            reason,
            source_transcript,
            whisper_transcript,
            wer,
            max_wer,
        } => Some(interpretation_omit_warning(
            utterance_id,
            reason,
            source_transcript.as_deref(),
            whisper_transcript.as_deref(),
            *wer,
            *max_wer,
        )),
        _ => None,
    };
    pb.set_message(interpretation_prepare_progress_message(progress));
    if let Some(warning) = warning {
        if !quiet_output() {
            pb.suspend(|| eprintln!("warning: {warning}"));
        }
    }
}

fn whisper_transcript_divergence_omit(
    source_transcript: &str,
    whisper_transcript: &str,
    wer: f64,
    max_wer: f64,
) -> TranscriptRefinement {
    TranscriptRefinement::Omit {
        reason: "Whisper transcript diverged from source transcript".to_string(),
        source_transcript: Some(source_transcript.trim().to_string()),
        whisper_transcript: Some(whisper_transcript.trim().to_string()),
        wer: Some(wer),
        max_wer: Some(max_wer),
    }
}

fn interpretation_omit_warning(
    utterance_id: &str,
    reason: &str,
    source_transcript: Option<&str>,
    whisper_transcript: Option<&str>,
    wer: Option<f64>,
    max_wer: Option<f64>,
) -> String {
    let mut warning = if let (Some(wer), Some(max_wer)) = (wer, max_wer) {
        format!("omitting {utterance_id}: {reason} (WER {wer:.2} > {max_wer:.2})")
    } else {
        format!("omitting {utterance_id}: {reason}")
    };
    if let Some(source_transcript) = source_transcript {
        warning.push_str("\n  source transcript: ");
        warning.push_str(source_transcript);
    }
    if let Some(whisper_transcript) = whisper_transcript {
        warning.push_str("\n  whisper transcript: ");
        warning.push_str(whisper_transcript);
    }
    warning
}

fn transcript_word_error_rate(reference: &str, candidate: &str) -> f64 {
    let reference_words = comparable_transcript_words(reference);
    let candidate_words = comparable_transcript_words(candidate);
    if reference_words.is_empty() {
        return if candidate_words.is_empty() { 0.0 } else { 1.0 };
    }
    edit_distance_words(&reference_words, &candidate_words) as f64 / reference_words.len() as f64
}

fn comparable_transcript_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|word| {
            let normalized = word
                .chars()
                .filter_map(|ch| {
                    if ch.is_ascii_alphanumeric() {
                        Some(ch.to_ascii_uppercase())
                    } else if matches!(ch, '\'' | '\u{2019}') {
                        Some('\'')
                    } else {
                        None
                    }
                })
                .collect::<String>();
            if normalized.is_empty() {
                None
            } else if is_comparable_numeric_transcript_word(&normalized) {
                Some("<NUM>".to_string())
            } else {
                Some(normalized)
            }
        })
        .collect()
}

fn is_comparable_numeric_transcript_word(word: &str) -> bool {
    let trimmed = word.trim_matches('\'');
    let base = trimmed
        .strip_suffix("ST")
        .or_else(|| trimmed.strip_suffix("ND"))
        .or_else(|| trimmed.strip_suffix("RD"))
        .or_else(|| trimmed.strip_suffix("TH"))
        .unwrap_or(trimmed);
    if !base.is_empty() && base.chars().all(|ch| ch.is_ascii_digit()) {
        return true;
    }
    matches!(
        base,
        "ZERO"
            | "OH"
            | "O"
            | "ONE"
            | "TWO"
            | "THREE"
            | "FOUR"
            | "FIVE"
            | "SIX"
            | "SEVEN"
            | "EIGHT"
            | "NINE"
            | "TEN"
            | "ELEVEN"
            | "TWELVE"
            | "THIRTEEN"
            | "FOURTEEN"
            | "FIFTEEN"
            | "SIXTEEN"
            | "SEVENTEEN"
            | "EIGHTEEN"
            | "NINETEEN"
            | "TWENTY"
            | "THIRTY"
            | "FORTY"
            | "FOURTY"
            | "FIFTY"
            | "SIXTY"
            | "SEVENTY"
            | "EIGHTY"
            | "NINETY"
            | "HUNDRED"
            | "THOUSAND"
            | "MILLION"
            | "BILLION"
    )
}

fn edit_distance_words(reference: &[String], candidate: &[String]) -> usize {
    let mut previous = (0..=candidate.len()).collect::<Vec<_>>();
    let mut current = vec![0; candidate.len() + 1];
    for (i, reference_word) in reference.iter().enumerate() {
        current[0] = i + 1;
        for (j, candidate_word) in candidate.iter().enumerate() {
            let substitution = previous[j] + usize::from(reference_word != candidate_word);
            let insertion = current[j] + 1;
            let deletion = previous[j + 1] + 1;
            current[j + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[candidate.len()]
}

fn cmd_interpretation_train(
    data: &Path,
    out: &Path,
    train_config: &InterpretationTrainConfig,
    device_arg: DeviceArg,
) -> Result<()> {
    let pb = status_spinner(format!(
        "Loading LibriSpeech ASR data from {}",
        data.display()
    ));
    let vocab: Vocab = read_json_file(&data.join("vocab.json"))?;
    let phoneme_vocab: Vocab = read_json_file(&data.join("phoneme_vocab.json"))?;
    let phone_vocab: Vocab = read_json_file(&data.join("phone_vocab.json"))?;
    let word_vocab: Vocab = read_json_file(&data.join("word_vocab.json"))?;
    let syntax_pos_vocab: Vocab = read_json_file(&data.join("syntax_pos_vocab.json"))?;
    let syntax_link_vocab: Vocab = read_json_file(&data.join("syntax_link_vocab.json"))?;
    let syntax_head_offset_vocab: Vocab =
        read_json_file(&data.join("syntax_head_offset_vocab.json"))?;
    let train_rows = tongues_interpretation::read_examples(&data.join("train.jsonl"))?;
    let valid_rows = tongues_interpretation::read_examples(&data.join("valid.jsonl"))?;
    finish_status(
        pb,
        format!(
            "Loaded {} train / {} valid utterances, vocab={} phoneme_vocab={} phone_vocab={} word_vocab={} syntax_pos_vocab={} syntax_link_vocab={} syntax_head_offset_vocab={}",
            format_count(train_rows.len()),
            format_count(valid_rows.len()),
            format_count(vocab.size()),
            format_count(phoneme_vocab.size()),
            format_count(phone_vocab.size()),
            format_count(word_vocab.size()),
            format_count(syntax_pos_vocab.size()),
            format_count(syntax_link_vocab.size()),
            format_count(syntax_head_offset_vocab.size())
        ),
    );
    fs::create_dir_all(out).context("creating LibriSpeech ASR model directory")?;
    let feature_bins = interpretation_feature_bins(data)?;
    let model_config = tongues_interpretation::ModelConfig::new(
        feature_bins,
        vocab.size(),
        phoneme_vocab.size(),
        phone_vocab.size(),
        word_vocab.size(),
    )
    .with_syntax_pos_vocab_size(syntax_pos_vocab.size())
    .with_syntax_link_vocab_size(syntax_link_vocab.size())
    .with_syntax_head_offset_vocab_size(syntax_head_offset_vocab.size())
    .with_dropout(train_config.dropout);
    tongues_interpretation::save_artifact_files(out, data, &model_config, train_config)?;
    println!("LibriSpeech ASR checkpoint paths:");
    println!("  train_state: {}", out.join("train_state.json").display());
    println!("  early_stop_metric: val_loss");
    println!(
        "  epoch checkpoints: {}",
        out.join("model-epoch-N.bin").display()
    );
    println!(
        "  optimizer checkpoints: {}",
        out.join("optim-epoch-N.bin").display()
    );
    println!("  best model: {}", out.join("model.bin").display());
    println!(
        "  loss weights: transcript={} seq2seq={} boundary={} repair={} phoneme(frame+ctc)={} phone(frame+ctc)={} feature_ctc={} prev_word={} current_word={} next_word={} masked_word={} masked_word_phoneme={} syntax={} masked_audio={}",
        train_config.transcript_loss_weight,
        train_config.seq2seq_loss_weight,
        train_config.boundary_loss_weight,
        train_config.repair_loss_weight,
        train_config.phoneme_loss_weight,
        train_config.phone_loss_weight,
        train_config.feature_ctc_loss_weight,
        train_config.prev_word_loss_weight,
        train_config.current_word_loss_weight,
        train_config.next_word_loss_weight,
        train_config.masked_word_loss_weight,
        train_config.masked_word_phoneme_loss_weight,
        train_config.syntax_loss_weight,
        train_config.masked_audio_loss_weight
    );
    println!("  feature CTC heads: place, manner, voicing, syllabic, height, backness, rounding");
    let model_path = out.join("model");
    let best = match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            let mut rng = StdRng::seed_from_u64(train_config.seed);
            tongues_interpretation::train::<CpuTrainBackend, _>(
                &model_config,
                train_config,
                data,
                &train_rows,
                &valid_rows,
                &vocab,
                &phoneme_vocab,
                &phone_vocab,
                &word_vocab,
                &syntax_pos_vocab,
                &syntax_link_vocab,
                &syntax_head_offset_vocab,
                &model_path,
                &device,
                &mut rng,
            )?
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            let mut rng = StdRng::seed_from_u64(train_config.seed);
            tongues_interpretation::train::<CudaTrainBackend, _>(
                &model_config,
                train_config,
                data,
                &train_rows,
                &valid_rows,
                &vocab,
                &phoneme_vocab,
                &phone_vocab,
                &word_vocab,
                &syntax_pos_vocab,
                &syntax_link_vocab,
                &syntax_head_offset_vocab,
                &model_path,
                &device,
                &mut rng,
            )?
        }
    };
    println!(
        "LibriSpeech ASR training complete. Best validation loss: {:.4}",
        best
    );
    Ok(())
}

fn interpretation_feature_bins(data: &Path) -> Result<usize> {
    let rows = tongues_interpretation::read_examples(&data.join("train.jsonl"))?;
    let first = rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("no training rows in {}", data.display()))?;
    let (_, bins) = tongues_interpretation::feature_file_shape(&data.join(&first.mel_path))?;
    Ok(bins)
}

fn cmd_interpretation_eval(
    model_dir: &Path,
    data: &Path,
    split: &str,
    device_arg: DeviceArg,
) -> Result<()> {
    let vocab: Vocab = read_json_file(&model_dir.join("vocab.json"))?;
    let phoneme_vocab: Vocab = read_json_file(&model_dir.join("phoneme_vocab.json"))?;
    let phone_vocab: Vocab = read_json_file(&model_dir.join("phone_vocab.json"))?;
    let word_vocab: Vocab = read_json_file(&model_dir.join("word_vocab.json"))?;
    let syntax_pos_vocab: Vocab = read_json_file(&model_dir.join("syntax_pos_vocab.json"))?;
    let syntax_link_vocab: Vocab = read_json_file(&model_dir.join("syntax_link_vocab.json"))?;
    let syntax_head_offset_vocab: Vocab =
        read_json_file(&model_dir.join("syntax_head_offset_vocab.json"))?;
    let model_config: tongues_interpretation::ModelConfig =
        read_json_file(&model_dir.join("model_config.json"))?;
    let mut train_config: InterpretationTrainConfig =
        read_json_file(&model_dir.join("train_config.json"))?;
    train_config.input_feature_bins = model_config.mel_bins;
    let rows = tongues_interpretation::read_examples(&data.join(format!("{split}.jsonl")))?;
    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            let model = tongues_interpretation::load_model::<CpuInferBackend>(
                &model_config,
                model_dir,
                &device,
            )?;
            let report = tongues_interpretation::evaluate(
                &model,
                data,
                &rows,
                &vocab,
                &phoneme_vocab,
                &phone_vocab,
                &word_vocab,
                &syntax_pos_vocab,
                &syntax_link_vocab,
                &syntax_head_offset_vocab,
                &train_config,
                &device,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            let model = tongues_interpretation::load_model::<CudaInferBackend>(
                &model_config,
                model_dir,
                &device,
            )?;
            let report = tongues_interpretation::evaluate(
                &model,
                data,
                &rows,
                &vocab,
                &phoneme_vocab,
                &phone_vocab,
                &word_vocab,
                &syntax_pos_vocab,
                &syntax_link_vocab,
                &syntax_head_offset_vocab,
                &train_config,
                &device,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}

fn cmd_interpretation_stream(model_dir: &Path, wav: &Path, device_arg: DeviceArg) -> Result<()> {
    let vocab: Vocab = read_json_file(&model_dir.join("vocab.json"))?;
    let phoneme_vocab: Vocab = read_json_file(&model_dir.join("phoneme_vocab.json"))?;
    let word_vocab: Vocab = read_json_file(&model_dir.join("word_vocab.json"))?;
    let model_config: tongues_interpretation::ModelConfig =
        read_json_file(&model_dir.join("model_config.json"))?;
    let config = InterpretationConfig::default();
    let samples = read_wav_mono_16k(wav)?;
    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            let model = tongues_interpretation::load_model::<CpuInferBackend>(
                &model_config,
                model_dir,
                &device,
            )?;
            let event = tongues_interpretation::stream_from_samples(
                &model,
                &samples,
                &vocab,
                &word_vocab,
                &phoneme_vocab,
                &config,
                model_config.mel_bins,
                &device,
            )?;
            println!("{}", serde_json::to_string_pretty(&event)?);
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            let model = tongues_interpretation::load_model::<CudaInferBackend>(
                &model_config,
                model_dir,
                &device,
            )?;
            let event = tongues_interpretation::stream_from_samples(
                &model,
                &samples,
                &vocab,
                &word_vocab,
                &phoneme_vocab,
                &config,
                model_config.mel_bins,
                &device,
            )?;
            println!("{}", serde_json::to_string_pretty(&event)?);
        }
    }
    Ok(())
}

fn read_wav_mono_16k(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening WAV {}", path.display()))?;
    let spec = reader.spec();
    anyhow::ensure!(spec.sample_rate == 16_000, "stream WAV must be 16 kHz");
    let channels = spec.channels.max(1) as usize;
    let mut out = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Float => {
            let mut acc = 0.0f32;
            let mut ch = 0usize;
            for sample in reader.samples::<f32>() {
                acc += sample?;
                ch += 1;
                if ch == channels {
                    out.push(acc / channels as f32);
                    acc = 0.0;
                    ch = 0;
                }
            }
        }
        hound::SampleFormat::Int => {
            let denom = ((1i64 << (spec.bits_per_sample.saturating_sub(1))) - 1).max(1) as f32;
            let mut acc = 0.0f32;
            let mut ch = 0usize;
            for sample in reader.samples::<i32>() {
                acc += sample? as f32 / denom;
                ch += 1;
                if ch == channels {
                    out.push(acc / channels as f32);
                    acc = 0.0;
                    ch = 0;
                }
            }
        }
    }
    Ok(out)
}

fn read_json_file<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn predict_sentence_boundary<B: Backend>(
    model: &Seq2SeqModel<B>,
    input: &str,
    vocab: &Vocab,
    device: &B::Device,
) -> String {
    let src_ids = vocab.encode_string(input);
    let src_len = src_ids.len();
    let src_tensor = Tensor::<B, 2, Int>::from_data(
        burn::tensor::TensorData::new(
            src_ids.iter().map(|&x| x as i32).collect::<Vec<_>>(),
            [1, src_len],
        ),
        device,
    );
    let pred_ids = model.generate(src_tensor, 128);
    vocab.decode_ids(&pred_ids)
}

fn read_jsonl_as<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let f = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = std::io::BufReader::new(f);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: T = serde_json::from_str(&line)
            .with_context(|| format!("parsing JSONL line: {}", &line[..line.len().min(80)]))?;
        out.push(value);
    }
    Ok(out)
}

fn read_jsonl_lossy<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let f = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = std::io::BufReader::new(f);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line.with_context(|| format!("reading {}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str(&line) {
            Ok(value) => out.push(value),
            Err(_) => {
                // A .writing.part file can be observed between write calls; ignore the tail.
                break;
            }
        }
    }
    Ok(out)
}

fn cmd_speaking_demo(
    mode: SpeakingDemoMode,
    selected_varieties: &[String],
    format: SpeakingDemoFormat,
) -> Result<()> {
    use speaking::{PhonemicizeRequest, PhonemicizeStyle, VarietyId};

    let selected = selected_varieties
        .iter()
        .map(|code| {
            speaking::canonical_variety_id(code)
                .map(|id| id.0)
                .with_context(|| format!("Unknown speaking variety `{code}`"))
        })
        .collect::<Result<std::collections::BTreeSet<_>>>()?;

    let varieties = speaking::builtin_varieties()
        .into_iter()
        .filter(|variety| selected.is_empty() || selected.contains(&variety.id.0))
        .collect::<Vec<_>>();
    if varieties.is_empty() {
        anyhow::bail!("No speaking varieties matched the requested filters");
    }

    let mut reports = Vec::new();
    for variety in varieties {
        let phonemicizer = speaking::phonemicizer_for_variety(&variety.id).map_err(|err| {
            anyhow::anyhow!("{}: failed to load phonemicizer: {err}", variety.id.0)
        })?;
        let samples = speaking_demo_samples(&variety, mode);
        let mut case_reports = Vec::new();

        for sample in samples {
            let output = phonemicizer
                .phonemicize(&PhonemicizeRequest {
                    text: sample.text.clone(),
                    variety: VarietyId(variety.id.0.clone()),
                    style: sample.careful_style.then_some(PhonemicizeStyle {
                        careful_style: true,
                    }),
                })
                .map_err(|err| {
                    anyhow::anyhow!("{} {} sample failed: {err:?}", variety.id.0, sample.name)
                })?;

            case_reports.push(serde_json::json!({
                "name": sample.name,
                "careful_style": sample.careful_style,
                "input": sample.text,
                "source_url": sample.source_url,
                "source_path": sample.source_path.as_ref().map(|path| path.display().to_string()),
                "phonemes": speaking_demo_phoneme_words(&output),
                "phones": speaking_demo_phone_words(&output),
                "utterance_phonemes": speaking_demo_phoneme_utterance(&output),
                "utterance_phones": speaking_demo_phone_utterance(&output),
                "counts": {
                    "graphemes": output.graphemes.len(),
                    "phonemes": output.phonemes.len(),
                    "phones": output.phones.len(),
                    "syllables": output.syllables.len(),
                    "boundaries": output.boundaries.len(),
                    "prosody_labels": output.prosody.labels.len(),
                    "syntax_tokens": output.syntax.tokens.len(),
                    "warnings": output.warnings.len(),
                },
                "boundaries": output.boundaries.iter().map(|boundary| {
                    serde_json::json!({
                        "kind": format!("{:?}", boundary.kind),
                        "after_grapheme_index": boundary.after_grapheme_index,
                        "terminal": boundary.terminal.map(|terminal| format!("{terminal:?}")),
                        "pause": boundary.pause.map(|pause| format!("{pause:?}")),
                    })
                }).collect::<Vec<_>>(),
                "prosody_labels": output.prosody.labels.iter().map(|label| {
                    serde_json::json!({
                        "kind": format!("{:?}", label.kind),
                        "confidence": label.confidence,
                    })
                }).collect::<Vec<_>>(),
                "syntax": output.syntax.tokens.iter().map(|token| {
                    serde_json::json!({
                        "word_index": token.word_index,
                        "text": token.text,
                        "pos": format!("{:?}", token.pos),
                        "prosodic_role": format!("{:?}", token.prosodic_role),
                        "links": token.syntactic_links.iter().map(|link| format!("{link:?}")).collect::<Vec<_>>(),
                    })
                }).collect::<Vec<_>>(),
                "warnings": output.warnings.iter().map(|warning| {
                    serde_json::json!({
                        "token": warning.token,
                        "kind": format!("{:?}", warning.kind),
                        "message": warning.message,
                    })
                }).collect::<Vec<_>>(),
                "provenance": {
                    "source": format!("{:?}", output.provenance.source),
                    "method": output.provenance.method,
                    "version": output.provenance.version,
                },
            }));
        }

        reports.push(serde_json::json!({
            "variety": variety.id.0,
            "language": variety.language.0,
            "name": variety.name,
            "implementation_status": format!("{:?}", variety.implementation_status),
            "status": format!("{:?}", variety.status),
            "mode": format!("{:?}", mode),
            "inventories": {
                "phonemes": variety.phonemes.phonemes.len(),
                "phones": variety.phones.phones.len(),
                "allophone_rules": variety.allophone_rules.len(),
                "epenthesis_rules": variety.epenthesis_rules.len(),
                "weak_forms": variety.weak_forms.len(),
                "orthographic_unit_pronunciations": variety.orthographic_unit_pronunciations.len(),
                "pronunciation_lexicons": variety.pronunciation_lexicons.len(),
            },
            "profiles": {
                "pronunciation_pipeline": variety.pronunciation_pipeline,
                "syntax_profile": variety.syntax_profile,
                "orthography": variety.orthography.as_ref().map(|orthography| orthography.name.clone()),
                "sample_words": variety.orthography.as_ref().map(|orthography| orthography.sample_words.clone()).unwrap_or_default(),
                "number_names": variety.number_names.is_some(),
                "punctuation": variety.punctuation.is_some(),
                "question_contours": variety.question_contours.is_some(),
                "prosody": variety.prosody_profile,
                "morphology": variety.morphology.is_some(),
                "acoustics": variety.acoustic_profile.is_some(),
            },
            "cases": case_reports,
        }));
    }

    match format {
        SpeakingDemoFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&reports)?);
        }
        SpeakingDemoFormat::Text => print_speaking_demo_text(&reports),
    }
    Ok(())
}

struct SpeakingDemoSample {
    name: String,
    text: String,
    careful_style: bool,
    source_url: Option<String>,
    source_path: Option<PathBuf>,
}

fn speaking_demo_samples(
    variety: &speaking::LinguisticVariety,
    mode: SpeakingDemoMode,
) -> Vec<SpeakingDemoSample> {
    if mode == SpeakingDemoMode::Paragraphs {
        return speaking_demo_paragraph_samples(variety)
            .unwrap_or_else(|_| speaking_demo_sentence_samples(variety));
    }

    if mode == SpeakingDemoMode::Sentences {
        return speaking_demo_sentence_samples(variety);
    }

    let words = variety
        .orthography
        .as_ref()
        .map(|orthography| orthography.sample_words.clone())
        .unwrap_or_default();
    let baseline = speaking_demo_join_words(&words, 5).unwrap_or_else(|| variety.name.clone());
    let short = speaking_demo_join_words(&words, 3).unwrap_or_else(|| baseline.clone());
    let utterance = speaking_demo_utterance(variety, &words).unwrap_or_else(|| format!("{short}?"));

    vec![
        SpeakingDemoSample {
            name: "baseline".to_string(),
            text: baseline,
            careful_style: false,
            source_url: None,
            source_path: None,
        },
        SpeakingDemoSample {
            name: "question".to_string(),
            text: format!("{short}?"),
            careful_style: false,
            source_url: None,
            source_path: None,
        },
        SpeakingDemoSample {
            name: "whole-utterance".to_string(),
            text: utterance,
            careful_style: false,
            source_url: None,
            source_path: None,
        },
        SpeakingDemoSample {
            name: "careful-style".to_string(),
            text: short,
            careful_style: true,
            source_url: None,
            source_path: None,
        },
    ]
}

fn speaking_demo_sentence_samples(
    variety: &speaking::LinguisticVariety,
) -> Vec<SpeakingDemoSample> {
    let text = speaking_demo_famous_sentence(variety)
        .map(str::to_string)
        .or_else(|| {
            variety
                .orthography
                .as_ref()
                .and_then(|orthography| speaking_demo_utterance(variety, &orthography.sample_words))
        })
        .unwrap_or_else(|| variety.name.clone());
    vec![SpeakingDemoSample {
        name: "famous-line".to_string(),
        text,
        careful_style: false,
        source_url: None,
        source_path: None,
    }]
}

fn speaking_demo_paragraph_samples(
    variety: &speaking::LinguisticVariety,
) -> Result<Vec<SpeakingDemoSample>> {
    let source = speaking_demo_gutenberg_source_for_variety(&variety.id.0)?
        .with_context(|| format!("no Gutenberg source configured for {}", variety.id.0))?;
    let path = speaking_demo_cached_gutenberg_source(&source.url)?;
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let paragraph = speaking_demo_pick_paragraph(&text, &variety.id.0)
        .with_context(|| format!("no readable paragraph found in {}", path.display()))?;

    Ok(vec![SpeakingDemoSample {
        name: "gutenberg-paragraph".to_string(),
        text: paragraph,
        careful_style: false,
        source_url: Some(source.url),
        source_path: Some(path),
    }])
}

fn speaking_demo_gutenberg_source_for_variety(
    variety: &str,
) -> Result<Option<tongues_head2phones::GutenbergSourceConfig>> {
    let config = read_head2phones_config(Path::new("configs/head2phones/default.toml"))?;
    let canonical = speaking::canonical_variety_id(variety).map(|id| id.0);
    let requested_language = speaking_demo_language_prefix(variety);
    let sources = config.gutenberg_sources;
    if let Some(source) = sources
        .iter()
        .find(|source| {
            source.varieties.iter().any(|source_variety| {
                source_variety == variety
                    || speaking::canonical_variety_id(source_variety)
                        .map(|source_id| Some(source_id.0) == canonical)
                        .unwrap_or(false)
            })
        })
        .cloned()
    {
        return Ok(Some(source));
    }
    Ok(sources.into_iter().find(|source| {
        source.varieties.iter().any(|source_variety| {
            speaking_demo_language_prefix(source_variety) == requested_language
        })
    }))
}

fn speaking_demo_language_prefix(variety: &str) -> &str {
    variety.split('-').next().unwrap_or(variety)
}

fn speaking_demo_cached_gutenberg_source(url: &str) -> Result<PathBuf> {
    const USER_AGENT: &str = "tongues-speaking-paragraphs/0.1";

    let dir = PathBuf::from("data/speaking-paragraphs/gutenberg");
    fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(speaking_demo_gutenberg_filename(url));
    if path.exists() && path.metadata()?.len() > 0 {
        return Ok(path);
    }

    eprintln!("Downloading Gutenberg source {url}");
    let response = ureq::get(url)
        .header("User-Agent", USER_AGENT)
        .call()
        .with_context(|| format!("GET {url}"))?;
    let raw = response
        .into_body()
        .read_to_string()
        .with_context(|| format!("reading {url}"))?;
    let stripped = speaking_demo_strip_gutenberg_boilerplate(&raw);
    let part_path = path.with_extension("txt.part");
    fs::write(&part_path, stripped).with_context(|| format!("writing {}", part_path.display()))?;
    fs::rename(&part_path, &path)
        .with_context(|| format!("moving {} to {}", part_path.display(), path.display()))?;
    Ok(path)
}

fn speaking_demo_gutenberg_filename(url: &str) -> String {
    url.rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("gutenberg.txt")
        .replace(['/', '\\', ':', '?', '&', '='], "_")
}

fn speaking_demo_strip_gutenberg_boilerplate(raw: &str) -> String {
    let start = raw
        .find("*** START OF")
        .and_then(|index| raw[index..].find("***").map(|offset| index + offset + 3))
        .and_then(|index| raw[index..].find("***").map(|offset| index + offset + 3))
        .unwrap_or(0);
    let after_start = &raw[start..];
    let end = after_start.find("*** END OF").unwrap_or(after_start.len());
    after_start[..end].trim().to_string()
}

fn speaking_demo_pick_paragraph(text: &str, variety: &str) -> Option<String> {
    const MIN_DEEP_PARAGRAPH_INDEX: usize = 12;

    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let readable = normalized
        .split("\n\n")
        .enumerate()
        .filter_map(|(index, raw)| {
            let paragraph = speaking_demo_clean_paragraph(raw);
            speaking_demo_paragraph_is_readable(&paragraph, variety).then_some((index, paragraph))
        })
        .collect::<Vec<_>>();
    let paragraph = readable
        .iter()
        .find(|(index, _)| *index >= MIN_DEEP_PARAGRAPH_INDEX)
        .or_else(|| readable.first())?
        .1
        .clone();
    Some(speaking_demo_limit_words(&paragraph, 80))
}

fn speaking_demo_clean_paragraph(raw: &str) -> String {
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn speaking_demo_paragraph_is_readable(paragraph: &str, variety: &str) -> bool {
    let char_count = paragraph.chars().count();
    if !(80..=700).contains(&char_count) {
        return false;
    }
    if paragraph.contains("Gutenberg") || paragraph.contains("www.") || paragraph.contains("http") {
        return false;
    }
    if paragraph.contains("[Illustration") || speaking_demo_looks_like_chapter_listing(paragraph) {
        return false;
    }
    let lower = paragraph.to_lowercase();
    if lower.contains("illustration")
        || lower.contains("inscribed")
        || lower.contains("publisher")
        || lower.contains("press:")
        || lower.starts_with("note:")
        || lower.starts_with("σημείωση:")
        || lower.starts_with("σημειωση:")
        || lower.starts_with("produced by")
        || lower.contains("translation has been")
        || lower.contains("translator has")
        || lower.contains("μετάφραση")
        || lower.contains("μεταφραστ")
    {
        return false;
    }
    if paragraph.contains('_') {
        return false;
    }
    if paragraph.contains("--") {
        return false;
    }
    if !paragraph.ends_with(['.', '?', '!', '।', '॥']) {
        return false;
    }
    if paragraph.chars().filter(|c| c.is_alphabetic()).count() < 40 {
        return false;
    }
    let letters = paragraph.chars().filter(|c| c.is_alphabetic()).count();
    let uppercase = paragraph.chars().filter(|c| c.is_uppercase()).count();
    if letters > 0 && uppercase as f32 / letters as f32 > 0.6 {
        return false;
    }
    if matches!(variety, "sa-Deva-Standard") {
        return speaking_demo_script_ratio(paragraph, |c| ('\u{0900}'..='\u{097F}').contains(&c))
            >= 0.65;
    }
    if matches!(variety, "el-GR-Standard") {
        return speaking_demo_script_ratio(paragraph, |c| {
            ('\u{0370}'..='\u{03FF}').contains(&c) || ('\u{1F00}'..='\u{1FFF}').contains(&c)
        }) >= 0.65;
    }
    if matches!(variety, "grc-Attic" | "grc-Koine") {
        return speaking_demo_script_ratio(paragraph, |c| ('\u{1F00}'..='\u{1FFF}').contains(&c))
            >= 0.20;
    }
    true
}

fn speaking_demo_looks_like_chapter_listing(paragraph: &str) -> bool {
    let lower = paragraph.to_lowercase();
    if lower.contains("chapter ") {
        return true;
    }
    if speaking_demo_chapter_heading_count(&lower) >= 2 {
        return true;
    }
    let trimmed = lower.trim_start();
    ROMAN_NUMERAL_CONTEXT_WORDS
        .iter()
        .any(|word| trimmed.starts_with(&format!("{word} ")))
}

const ROMAN_NUMERAL_CONTEXT_WORDS: &[&str] = &[
    "act",
    "acte",
    "akt",
    "book",
    "buch",
    "canto",
    "capitulo",
    "capítulo",
    "chapter",
    "chapitre",
    "escena",
    "kapitulo",
    "liber",
    "libro",
    "livre",
    "part",
    "parte",
    "partie",
    "scene",
    "scène",
    "section",
    "tome",
    "volume",
];

fn speaking_demo_chapter_heading_count(lower: &str) -> usize {
    ROMAN_NUMERAL_CONTEXT_WORDS
        .iter()
        .map(|word| lower.match_indices(&format!("{word} ")).count())
        .sum()
}

fn speaking_demo_script_ratio(paragraph: &str, script: impl Fn(char) -> bool) -> f32 {
    let letters = paragraph.chars().filter(|c| c.is_alphabetic()).count();
    if letters == 0 {
        return 0.0;
    }
    let script_letters = paragraph
        .chars()
        .filter(|c| c.is_alphabetic() && script(*c))
        .count();
    script_letters as f32 / letters as f32
}

fn speaking_demo_limit_words(paragraph: &str, limit: usize) -> String {
    let mut words = paragraph.split_whitespace();
    let mut clipped = words.by_ref().take(limit).collect::<Vec<_>>().join(" ");
    if words.next().is_some() {
        clipped.push('…');
    }
    clipped
}

fn speaking_demo_famous_sentence(variety: &speaking::LinguisticVariety) -> Option<&'static str> {
    match variety.id.0.as_str() {
        "el-GR-Standard" => return Some("Σε γνωρίζω από την κόψη."),
        "grc-Attic" | "grc-Koine" => return Some("Ἄνδρα μοι ἔννεπε, Μοῦσα."),
        "la-Classical" | "la-Ecclesiastical" => return Some("Arma virumque cano."),
        "es-ES-Castilian" | "es-419-Standard" => return Some("En un lugar de la Mancha."),
        _ => {}
    }
    match variety.language.0.as_str() {
        "en" => Some("To be, or not to be?"),
        "eo" => Some("Ho, mia kor!"),
        "fr" => Some("Je pense, donc je suis."),
        "de" => Some("Am Brunnen vor dem Tore."),
        "sa" => Some("धर्मक्षेत्रे कुरुक्षेत्रे."),
        "es" => Some("En un lugar de la Mancha."),
        _ => None,
    }
}

fn speaking_demo_join_words(words: &[String], limit: usize) -> Option<String> {
    let sample = words
        .iter()
        .filter(|word| !word.trim().is_empty())
        .take(limit)
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    (!sample.is_empty()).then_some(sample)
}

fn speaking_demo_utterance(
    variety: &speaking::LinguisticVariety,
    words: &[String],
) -> Option<String> {
    if let Some(text) = speaking_demo_builtin_utterance(&variety.id.0) {
        return Some(text.to_string());
    }
    let words = words
        .iter()
        .filter(|word| !word.trim().is_empty())
        .take(4)
        .cloned()
        .collect::<Vec<_>>();
    match words.as_slice() {
        [first, second, third, fourth, ..] => Some(format!("{first} {second}, {third} {fourth}?")),
        [first, second, third] => Some(format!("{first} {second}, {third}?")),
        [first, second] => Some(format!("{first}, {second}?")),
        [first] => Some(format!("{first}?")),
        [] => None,
    }
}

fn speaking_demo_builtin_utterance(variety: &str) -> Option<&'static str> {
    match variety {
        "en-US-GA" | "en-US" => Some("Hello, world?"),
        "es-ES-Castilian" | "es-419-Standard" => Some("La casa está lista?"),
        "fr-FR-Standard" => Some("La maison est prête?"),
        "de-DE-Standard" => Some("Das Haus, ist bereit?"),
        "eo-001-Standard" => Some("La domo, estas preta?"),
        "el-GR-Standard" => Some("Το σπιτι, ειναι ετοιμο?"),
        "grc-Attic" | "grc-Koine" => Some("και λογος, και φως?"),
        "la-Classical" | "la-Ecclesiastical" => Some("Salve, amice?"),
        "sa-Deva-Standard" => Some("धर्म, कर्म?"),
        _ => None,
    }
}

fn speaking_demo_phoneme_words(output: &speaking::PhonemicizeOutput) -> String {
    speaking_demo_word_syllables(output)
        .into_iter()
        .map(|(_, syllables)| {
            syllables_to_phonemes_ipa(&syllables, &output.phonemes, &output.variety)
        })
        .filter(|ipa| !ipa.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn speaking_demo_phone_words(output: &speaking::PhonemicizeOutput) -> String {
    speaking_demo_word_syllables(output)
        .into_iter()
        .map(|(_, syllables)| syllables_to_ipa_formatted(&syllables))
        .filter(|ipa| !ipa.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn speaking_demo_phoneme_utterance(output: &speaking::PhonemicizeOutput) -> String {
    speaking_demo_utterance_parts(output, |syllables, output| {
        syllables_to_phonemes_ipa(syllables, &output.phonemes, &output.variety)
    })
}

fn speaking_demo_phone_utterance(output: &speaking::PhonemicizeOutput) -> String {
    speaking_demo_utterance_parts(output, |syllables, _| syllables_to_ipa_formatted(syllables))
}

fn speaking_demo_utterance_parts(
    output: &speaking::PhonemicizeOutput,
    format_word: impl Fn(&[speaking::Syllable], &speaking::PhonemicizeOutput) -> String,
) -> String {
    let words = speaking_demo_word_syllables(output);
    let last_index = words.len().saturating_sub(1);
    let mut parts = Vec::new();
    for (position, (word_index, syllables)) in words.into_iter().enumerate() {
        let word = format_word(&syllables, output);
        if word.is_empty() {
            continue;
        }
        parts.push(word);
        let boundary_symbols = speaking_demo_boundary_symbols_after_word(output, word_index);
        if boundary_symbols.is_empty() {
            if position != last_index {
                parts.push("|".to_string());
            }
        } else {
            parts.extend(boundary_symbols.into_iter().map(str::to_string));
        }
    }
    parts.join(" ")
}

fn speaking_demo_boundary_symbols_after_word(
    output: &speaking::PhonemicizeOutput,
    word_index: usize,
) -> Vec<&'static str> {
    let Some(boundary) = output
        .boundaries
        .iter()
        .filter(|boundary| boundary.terminal.is_some() || boundary.pause.is_some())
        .find(|boundary| boundary.after_grapheme_index == word_index)
    else {
        return Vec::new();
    };
    if let Some(terminal) = boundary.terminal {
        return match terminal {
            speaking::TerminalPunctuation::Question => vec!["↗", "?"],
            speaking::TerminalPunctuation::Period => vec!["↘", "."],
            speaking::TerminalPunctuation::Exclamation => vec!["↘", "!"],
        };
    }
    if let Some(pause) = boundary.pause {
        return match pause {
            speaking::PauseKind::Comma => vec!["→", ","],
            speaking::PauseKind::AlternativeQuestionRise => vec!["↗", ","],
        };
    }
    Vec::new()
}

fn speaking_demo_word_syllables(
    output: &speaking::PhonemicizeOutput,
) -> Vec<(usize, Vec<speaking::Syllable>)> {
    let mut words: Vec<(usize, Vec<speaking::Syllable>)> = Vec::new();
    for syllable in output.syllables.iter() {
        if let Some(first_phone) = syllable.phones.first() {
            if let Some(word_idx) = token_word_index(&first_phone.features) {
                if let Some(last_word) = words.last_mut() {
                    if last_word.0 == word_idx {
                        last_word.1.push(syllable.clone());
                        continue;
                    }
                }
                words.push((word_idx, vec![syllable.clone()]));
            }
        }
    }
    words
}

fn print_speaking_demo_text(reports: &[serde_json::Value]) {
    println!("Speaking demo: {} varieties", reports.len());
    for report in reports {
        println!(
            "\n== {} ({}) ==",
            report["variety"].as_str().unwrap_or("unknown"),
            report["name"].as_str().unwrap_or("unknown")
        );
        println!(
            "language={} status={} implementation={}",
            report["language"].as_str().unwrap_or("unknown"),
            report["status"].as_str().unwrap_or("unknown"),
            report["implementation_status"]
                .as_str()
                .unwrap_or("unknown")
        );
        let inventories = &report["inventories"];
        println!(
            "inventory: phonemes={} phones={} allophones={} epenthesis={} weak_forms={} units={} lexicons={}",
            inventories["phonemes"].as_u64().unwrap_or(0),
            inventories["phones"].as_u64().unwrap_or(0),
            inventories["allophone_rules"].as_u64().unwrap_or(0),
            inventories["epenthesis_rules"].as_u64().unwrap_or(0),
            inventories["weak_forms"].as_u64().unwrap_or(0),
            inventories["orthographic_unit_pronunciations"].as_u64().unwrap_or(0),
            inventories["pronunciation_lexicons"].as_u64().unwrap_or(0),
        );
        let profiles = &report["profiles"];
        println!(
            "profiles: pipeline={} syntax={} orthography={} numbers={} punctuation={} questions={} prosody={} morphology={} acoustics={}",
            profiles["pronunciation_pipeline"].as_str().unwrap_or("none"),
            profiles["syntax_profile"].as_str().unwrap_or("none"),
            profiles["orthography"].as_str().unwrap_or("none"),
            profiles["number_names"].as_bool().unwrap_or(false),
            profiles["punctuation"].as_bool().unwrap_or(false),
            profiles["question_contours"].as_bool().unwrap_or(false),
            !profiles["prosody"].is_null(),
            profiles["morphology"].as_bool().unwrap_or(false),
            profiles["acoustics"].as_bool().unwrap_or(false),
        );

        for case in report["cases"].as_array().into_iter().flatten() {
            let counts = &case["counts"];
            println!(
                "- {}: {:?}",
                case["name"].as_str().unwrap_or("case"),
                case["input"].as_str().unwrap_or("")
            );
            if let Some(url) = case["source_url"].as_str() {
                println!("  source: {url}");
            }
            if let Some(path) = case["source_path"].as_str() {
                println!("  cache: {path}");
            }
            println!("  /{}/", case["phonemes"].as_str().unwrap_or(""));
            println!("  [{}]", case["phones"].as_str().unwrap_or(""));
            if matches!(
                case["name"].as_str(),
                Some("whole-utterance" | "famous-line")
            ) {
                println!(
                    "  utterance /{}/",
                    case["utterance_phonemes"].as_str().unwrap_or("")
                );
                println!(
                    "  utterance [{}]",
                    case["utterance_phones"].as_str().unwrap_or("")
                );
                let boundaries = case["boundaries"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|boundary| {
                        let terminal = boundary["terminal"].as_str().unwrap_or("none");
                        let pause = boundary["pause"].as_str().unwrap_or("none");
                        format!(
                            "{}@{} terminal={} pause={}",
                            boundary["kind"].as_str().unwrap_or("Boundary"),
                            boundary["after_grapheme_index"].as_u64().unwrap_or(0),
                            terminal,
                            pause
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                if !boundaries.is_empty() {
                    println!("  boundaries: {boundaries}");
                }
            }
            println!(
                "  counts: graphemes={} phonemes={} phones={} syllables={} boundaries={} prosody_labels={} syntax_tokens={} warnings={}",
                counts["graphemes"].as_u64().unwrap_or(0),
                counts["phonemes"].as_u64().unwrap_or(0),
                counts["phones"].as_u64().unwrap_or(0),
                counts["syllables"].as_u64().unwrap_or(0),
                counts["boundaries"].as_u64().unwrap_or(0),
                counts["prosody_labels"].as_u64().unwrap_or(0),
                counts["syntax_tokens"].as_u64().unwrap_or(0),
                counts["warnings"].as_u64().unwrap_or(0),
            );
        }
    }
}

fn cmd_phonemes(text: &str) -> Result<()> {
    use speaking::{phonemicizer_for_variety, PhonemicizeRequest, VarietyId};

    let variety = VarietyId("en-US".to_string());
    let phonemicizer = phonemicizer_for_variety(&variety)
        .map_err(|e| anyhow::anyhow!("Failed to load phonemicizer: {e}"))?;
    let phonemicized = phonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: text.to_string(),
            variety,
            style: None,
        })
        .map_err(|e| anyhow::anyhow!("Failed to phonemicize: {:?}", e))?;

    let mut words: Vec<(usize, Vec<speaking::Syllable>)> = Vec::new();
    for syllable in phonemicized.syllables.iter() {
        if let Some(first_phone) = syllable.phones.first() {
            if let Some(word_idx) = token_word_index(&first_phone.features) {
                if let Some(last_word) = words.last_mut() {
                    if last_word.0 == word_idx {
                        last_word.1.push(syllable.clone());
                        continue;
                    }
                }
                words.push((word_idx, vec![syllable.clone()]));
            }
        }
    }

    let mut ipa_words = Vec::new();
    for (_, word_syllables) in words {
        let ipa = syllables_to_phonemes_ipa(
            &word_syllables,
            &phonemicized.phonemes,
            &phonemicized.variety,
        );
        if !ipa.is_empty() {
            ipa_words.push(ipa);
        }
    }

    println!("/{}/", ipa_words.join(" "));
    Ok(())
}

fn cmd_phones(text: &str) -> Result<()> {
    use speaking::{phonemicizer_for_variety, PhonemicizeRequest, VarietyId};

    let variety = VarietyId("en-US".to_string());
    let phonemicizer = phonemicizer_for_variety(&variety)
        .map_err(|e| anyhow::anyhow!("Failed to load phonemicizer: {e}"))?;
    let phonemicized = phonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: text.to_string(),
            variety,
            style: None,
        })
        .map_err(|e| anyhow::anyhow!("Failed to phonemicize: {:?}", e))?;

    let mut words: Vec<(usize, Vec<speaking::Syllable>)> = Vec::new();
    for syllable in phonemicized.syllables.iter() {
        if let Some(first_phone) = syllable.phones.first() {
            if let Some(word_idx) = token_word_index(&first_phone.features) {
                if let Some(last_word) = words.last_mut() {
                    if last_word.0 == word_idx {
                        last_word.1.push(syllable.clone());
                        continue;
                    }
                }
                words.push((word_idx, vec![syllable.clone()]));
            }
        }
    }

    let mut ipa_words = Vec::new();
    for (_, word_syllables) in words {
        let ipa = syllables_to_ipa_formatted(&word_syllables);
        if !ipa.is_empty() {
            ipa_words.push(ipa);
        }
    }

    println!("[{}]", ipa_words.join(" "));
    Ok(())
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

fn phone_ipa(phone: &speaking::PhoneToken) -> &str {
    match &phone.phone {
        speaking::Spec::Known(id) => id
            .as_str()
            .strip_prefix("ipa.phone.")
            .unwrap_or(id.as_str()),
        _ => "",
    }
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

// ── pronunciation discrepancies ───────────────────────────────────────────

fn cmd_discrepancies(
    out: &Path,
    limit: usize,
    max_rarity: f32,
    explicit_words: Vec<String>,
    words_file: Option<&Path>,
    include_g2p2g: bool,
    include_wiktionary: bool,
    g2p2g_model: &Path,
    wiktionary_model: &Path,
    wiktionary_variety: &str,
    device_arg: DeviceArg,
    output_mode: OutputMode,
) -> Result<()> {
    match device_arg {
        DeviceArg::Cpu => cmd_discrepancies_backend::<CpuInferBackend>(
            &NdArrayDevice::Cpu,
            out,
            limit,
            max_rarity,
            explicit_words,
            words_file,
            include_g2p2g,
            include_wiktionary,
            g2p2g_model,
            wiktionary_model,
            wiktionary_variety,
            output_mode,
        ),
        DeviceArg::Cuda => cmd_discrepancies_backend::<CudaInferBackend>(
            &CudaDevice::default(),
            out,
            limit,
            max_rarity,
            explicit_words,
            words_file,
            include_g2p2g,
            include_wiktionary,
            g2p2g_model,
            wiktionary_model,
            wiktionary_variety,
            output_mode,
        ),
    }
}

fn cmd_discrepancies_backend<B: Backend>(
    device: &B::Device,
    out: &Path,
    limit: usize,
    max_rarity: f32,
    explicit_words: Vec<String>,
    words_file: Option<&Path>,
    include_g2p2g: bool,
    include_wiktionary: bool,
    g2p2g_model: &Path,
    wiktionary_model: &Path,
    wiktionary_variety: &str,
    output_mode: OutputMode,
) -> Result<()>
where
    B::Device: Clone,
{
    let raw_openepd: std::collections::BTreeMap<String, OpenEpdEntry> =
        serde_json::from_str(open_english_pronouncing_dictionary::CORPUS_JSON)
            .context("parsing embedded OpenEPD JSON")?;
    let words = discrepancy_words(&raw_openepd, limit, max_rarity, explicit_words, words_file)?;

    let mut cmu = CmudictProvider {
        lexicon: speaking::data::lexicons::cmudict::bundled(),
    };
    let mut openepd = OpenEpdProvider {
        entries: &raw_openepd,
    };
    let mut rules = RuleProvider;

    let mut g2p2g = if include_g2p2g {
        match Seq2SeqPronouncer::<B>::load_g2p2g(g2p2g_model, device.clone()) {
            Ok(provider) => Some(provider),
            Err(err) => {
                println!(
                    "Warning: skipping g2p2g pronouncer from {}: {err:#}",
                    g2p2g_model.display()
                );
                None
            }
        }
    } else {
        None
    };
    let mut wiktionary = if include_wiktionary {
        match Seq2SeqPronouncer::<B>::load_wiktionary(
            wiktionary_model,
            wiktionary_variety,
            device.clone(),
        ) {
            Ok(provider) => Some(provider),
            Err(err) => {
                println!(
                    "Warning: skipping wiktionary pronouncer from {}: {err:#}",
                    wiktionary_model.display()
                );
                None
            }
        }
    } else {
        None
    };

    let mut provider_names = vec![
        "cmudict".to_string(),
        "openepd".to_string(),
        "speaking-rules".to_string(),
    ];
    if let Some(provider) = &g2p2g {
        provider_names.push(provider.name.to_string());
    }
    if let Some(provider) = &wiktionary {
        provider_names.push(provider.name.to_string());
    }

    let mut providers: Vec<&mut dyn speaking::PronunciationProvider> =
        vec![&mut cmu, &mut openepd, &mut rules];
    if let Some(provider) = g2p2g.as_mut() {
        providers.push(provider);
    }
    if let Some(provider) = wiktionary.as_mut() {
        providers.push(provider);
    }

    if output_mode.verbose() {
        println!(
            "Comparing {} words across {}...",
            format_count(words.len()),
            provider_names.join(", ")
        );
    }

    let total_checks = words.len().saturating_mul(providers.len());
    let pb = if quiet_output() {
        indicatif::ProgressBar::hidden()
    } else {
        let pb = indicatif::ProgressBar::new(total_checks as u64);
        pb.set_style(counted_progress_style()?);
        pb.set_message(format!(
            "Checking {} words across {} providers",
            format_count(words.len()),
            format_count(providers.len())
        ));
        tongues_core::register_progress_bar(pb)
    };

    let records = speaking::find_pronunciation_discrepancies_with_progress(
        &words,
        &mut providers,
        |progress| {
            pb.inc(1);
            pb.set_message(format!(
                "word {}/{} via {} ({})",
                format_count(progress.word_index),
                format_count(progress.words_total),
                progress.provider,
                progress.word
            ));
        },
    );
    pb.finish_and_clear();

    let report = speaking::render_discrepancy_markdown(&records, &provider_names, words.len());
    write_atomic_text(out, &report)?;
    println!(
        "Wrote {} discrepancies for {} checked words to {}",
        format_count(records.len()),
        format_count(words.len()),
        out.display()
    );
    Ok(())
}

fn discrepancy_words(
    raw_openepd: &std::collections::BTreeMap<String, OpenEpdEntry>,
    limit: usize,
    max_rarity: f32,
    explicit_words: Vec<String>,
    words_file: Option<&Path>,
) -> Result<Vec<String>> {
    let mut seen = std::collections::BTreeSet::new();
    let mut words = Vec::new();

    for word in ["loadstone", "lodestone"] {
        push_discrepancy_word(word, &mut seen, &mut words);
    }
    for word in explicit_words {
        push_discrepancy_word(&word, &mut seen, &mut words);
    }
    if let Some(path) = words_file {
        for line in fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?
            .lines()
        {
            push_discrepancy_word(line, &mut seen, &mut words);
        }
    }

    let mut openepd_words = raw_openepd
        .iter()
        .filter(|(_, entry)| entry.rarity <= max_rarity)
        .map(|(word, entry)| (entry.rarity, word.as_str()))
        .collect::<Vec<_>>();
    openepd_words.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.1.cmp(right.1))
    });
    for (_, word) in openepd_words.into_iter().take(limit) {
        push_discrepancy_word(word, &mut seen, &mut words);
    }

    Ok(words)
}

fn push_discrepancy_word(
    word: &str,
    seen: &mut std::collections::BTreeSet<String>,
    words: &mut Vec<String>,
) {
    let normalized = speaking::data::lexicons::cmudict::normalize_for_lookup(word);
    if normalized.is_empty() || !normalized.chars().any(|c| c.is_alphabetic()) {
        return;
    }
    if seen.insert(normalized.clone()) {
        words.push(normalized);
    }
}

struct CmudictProvider {
    lexicon: &'static speaking::data::lexicons::cmudict::CmudictLexicon,
}

impl speaking::PronunciationProvider for CmudictProvider {
    fn name(&self) -> &str {
        "cmudict"
    }

    fn pronounce(&mut self, word: &str) -> speaking::PronouncerResult {
        let entry = self.lexicon.lookup_entry(word);
        let output = entry
            .candidates
            .first()
            .map(|candidate| speaking::cmu_phonemes_to_ipa(candidate));
        speaking::PronouncerResult {
            source: self.name().to_string(),
            status: if output.is_some() {
                speaking::PronouncerStatus::Found
            } else {
                speaking::PronouncerStatus::Missing
            },
            output,
            note: Some(format!("{:?}", entry.status)),
        }
    }
}

struct OpenEpdProvider<'a> {
    entries: &'a std::collections::BTreeMap<String, OpenEpdEntry>,
}

impl speaking::PronunciationProvider for OpenEpdProvider<'_> {
    fn name(&self) -> &str {
        "openepd"
    }

    fn pronounce(&mut self, word: &str) -> speaking::PronouncerResult {
        let output = self
            .entries
            .get(word)
            .and_then(|entry| preferred_openepd_ipa(&entry.ipa))
            .and_then(|ipa| normalize_openepd_ipa(ipa).ok());
        speaking::PronouncerResult {
            source: self.name().to_string(),
            status: if output.is_some() {
                speaking::PronouncerStatus::Found
            } else {
                speaking::PronouncerStatus::Missing
            },
            output,
            note: None,
        }
    }
}

struct RuleProvider;

impl speaking::PronunciationProvider for RuleProvider {
    fn name(&self) -> &str {
        "speaking-rules"
    }

    fn pronounce(&mut self, word: &str) -> speaking::PronouncerResult {
        match phonemicized_first_word_phones(word) {
            Ok(output) if !output.is_empty() => speaking::PronouncerResult {
                source: self.name().to_string(),
                output: Some(output),
                status: speaking::PronouncerStatus::Found,
                note: None,
            },
            Ok(_) => speaking::PronouncerResult {
                source: self.name().to_string(),
                output: None,
                status: speaking::PronouncerStatus::Missing,
                note: None,
            },
            Err(err) => speaking::PronouncerResult {
                source: self.name().to_string(),
                output: None,
                status: speaking::PronouncerStatus::Error,
                note: Some(err.to_string()),
            },
        }
    }
}

enum Seq2SeqPronouncerKind {
    G2p2g,
    Wiktionary,
}

struct Seq2SeqPronouncer<B: Backend> {
    name: &'static str,
    kind: Seq2SeqPronouncerKind,
    wiktionary_variety: Option<String>,
    model: Seq2SeqModel<B>,
    vocab: Vocab,
    device: B::Device,
}

impl<B: Backend> Seq2SeqPronouncer<B> {
    fn load_g2p2g(model_dir: &Path, device: B::Device) -> Result<Self> {
        let (model, vocab) = load_seq2seq_model_and_vocab::<B>(model_dir, &device)?;
        Ok(Self {
            name: "g2p2g",
            kind: Seq2SeqPronouncerKind::G2p2g,
            wiktionary_variety: None,
            model,
            vocab,
            device,
        })
    }

    fn load_wiktionary(model_dir: &Path, variety: &str, device: B::Device) -> Result<Self> {
        let (model, vocab) = load_seq2seq_model_and_vocab::<B>(model_dir, &device)?;
        Ok(Self {
            name: "wiktionary",
            kind: Seq2SeqPronouncerKind::Wiktionary,
            wiktionary_variety: (!variety.trim().is_empty()).then(|| variety.trim().to_string()),
            model,
            vocab,
            device,
        })
    }
}

impl<B: Backend> speaking::PronunciationProvider for Seq2SeqPronouncer<B> {
    fn name(&self) -> &str {
        self.name
    }

    fn pronounce(&mut self, word: &str) -> speaking::PronouncerResult {
        let prediction = match self.kind {
            Seq2SeqPronouncerKind::G2p2g => {
                predict(&self.model, word, Task::G2P, &self.vocab, &self.device)
            }
            Seq2SeqPronouncerKind::Wiktionary => {
                match wiktionary_infer_source(
                    "orthography-to-phonemes",
                    "eng",
                    WiktionaryNotationArg::Phonemes,
                    self.wiktionary_variety.as_deref(),
                    word,
                ) {
                    Ok(source) => {
                        let src_ids = self.vocab.encode_string(&source);
                        let src_len = src_ids.len();
                        let src_tensor = Tensor::<B, 2, Int>::from_data(
                            burn::tensor::TensorData::new(
                                src_ids.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                                [1, src_len],
                            ),
                            &self.device,
                        );
                        let pred_ids = self.model.generate(src_tensor, 128);
                        self.vocab.decode_ids(&pred_ids)
                    }
                    Err(err) => {
                        return speaking::PronouncerResult {
                            source: self.name().to_string(),
                            output: None,
                            status: speaking::PronouncerStatus::Error,
                            note: Some(err.to_string()),
                        };
                    }
                }
            }
        };

        speaking::PronouncerResult {
            source: self.name().to_string(),
            output: Some(prediction),
            status: speaking::PronouncerStatus::Found,
            note: None,
        }
    }
}

fn load_seq2seq_model_and_vocab<B: Backend>(
    model_dir: &Path,
    device: &B::Device,
) -> Result<(Seq2SeqModel<B>, Vocab)> {
    let model_config: ModelConfig = serde_json::from_str(
        &fs::read_to_string(model_dir.join("model_config.json")).with_context(|| {
            format!("reading {}", model_dir.join("model_config.json").display())
        })?,
    )?;
    let vocab: Vocab = serde_json::from_str(
        &fs::read_to_string(model_dir.join("vocab.json"))
            .with_context(|| format!("reading {}", model_dir.join("vocab.json").display()))?,
    )?;
    let model = load_model::<B>(&model_config, &model_dir.join("model"), device)?;
    Ok((model, vocab))
}

fn phonemicized_first_word_phones(text: &str) -> Result<String> {
    use speaking::{phonemicizer_for_variety, PhonemicizeRequest, VarietyId};

    let variety = VarietyId("en-US".to_string());
    let phonemicizer = phonemicizer_for_variety(&variety)
        .map_err(|e| anyhow::anyhow!("Failed to load phonemicizer: {e}"))?;
    let phonemicized = phonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: text.to_string(),
            variety,
            style: None,
        })
        .map_err(|e| anyhow::anyhow!("Failed to phonemicize: {:?}", e))?;

    let mut first_word_syllables = Vec::new();
    let mut first_word_index = None;
    for syllable in phonemicized.syllables.iter() {
        let Some(first_phone) = syllable.phones.first() else {
            continue;
        };
        let Some(word_idx) = token_word_index(&first_phone.features) else {
            continue;
        };
        if first_word_index.is_none() {
            first_word_index = Some(word_idx);
        }
        if Some(word_idx) == first_word_index {
            first_word_syllables.push(syllable.clone());
        }
    }

    Ok(syllables_to_ipa_formatted(&first_word_syllables))
}

fn write_atomic_text(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let part = path.with_extension(format!(
        "{}part",
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    {
        let mut writer = std::io::BufWriter::new(
            fs::File::create(&part).with_context(|| format!("creating {}", part.display()))?,
        );
        writer.write_all(text.as_bytes())?;
        writer.flush()?;
    }
    fs::rename(&part, path)
        .with_context(|| format!("renaming {} to {}", part.display(), path.display()))?;
    Ok(())
}

// ── fetch-cmudict ──────────────────────────────────────────────────────────

fn cmd_fetch_cmudict(out: &Path) -> Result<()> {
    const URL: &str = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict";
    fetch_url(URL, out, "CMUdict", "cmudict.dict")
}

// ── fetch-lexique ──────────────────────────────────────────────────────────

fn cmd_fetch_lexique(out: &Path) -> Result<()> {
    const URL: &str = "http://www.lexique.org/databases/Lexique383/Lexique383.tsv";
    const BUNDLED: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../speaking/src/data/lexicons/Lexique383.tsv"
    );
    fetch_url_with_fallback(URL, out, "Lexique383", "Lexique383.tsv", Some(BUNDLED))
}

fn fetch_url(url: &str, out: &Path, label: &str, fallback_filename: &str) -> Result<()> {
    fetch_url_with_fallback(url, out, label, fallback_filename, None)
}

fn fetch_url_with_fallback(
    url: &str,
    out: &Path,
    label: &str,
    fallback_filename: &str,
    bundled_fallback: Option<&str>,
) -> Result<()> {
    println!("Fetching {label} from {url}");

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).context("creating output directory")?;
    }

    let part = out.with_extension(format!(
        "{}part",
        out.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let _ = fs::remove_file(&part);

    let out_arg = part.to_str().unwrap_or(fallback_filename);
    let curl_max_time = if bundled_fallback.is_some() {
        "20"
    } else {
        "120"
    };
    let status = std::process::Command::new("curl")
        .args([
            "-fsSL",
            "--connect-timeout",
            "20",
            "--max-time",
            curl_max_time,
            "-o",
            out_arg,
            url,
        ])
        .status();

    if matches!(status, Ok(status) if status.success()) {
        fs::rename(&part, out)
            .with_context(|| format!("renaming {} to {}", part.display(), out.display()))?;
        println!("Saved to {}", out.display());
        return Ok(());
    }
    let _ = fs::remove_file(&part);

    if let Some(fallback) = bundled_fallback {
        if copy_bundled_fetch_fallback(fallback, &part, out, label)? {
            return Ok(());
        }
    }

    let status = std::process::Command::new("wget")
        .args([
            "--connect-timeout=20",
            "--read-timeout=20",
            "--tries=1",
            "-qO",
            out_arg,
            url,
        ])
        .status()
        .context("neither curl nor wget succeeded")?;
    if status.success() {
        fs::rename(&part, out)
            .with_context(|| format!("renaming {} to {}", part.display(), out.display()))?;
        println!("Saved to {}", out.display());
        return Ok(());
    }
    let _ = fs::remove_file(&part);

    if let Some(fallback) = bundled_fallback {
        if copy_bundled_fetch_fallback(fallback, &part, out, label)? {
            return Ok(());
        }
    }

    anyhow::bail!(
        "Could not download {label}. Please download manually from:\n  {url}\nand save to {}",
        out.display()
    )
}

fn copy_bundled_fetch_fallback(
    fallback: &str,
    part: &Path,
    out: &Path,
    label: &str,
) -> Result<bool> {
    let fallback = Path::new(fallback);
    if !fallback.exists() {
        return Ok(false);
    }

    fs::copy(fallback, part).with_context(|| {
        format!(
            "copying bundled fallback {} to {}",
            fallback.display(),
            part.display()
        )
    })?;
    fs::rename(part, out)
        .with_context(|| format!("renaming {} to {}", part.display(), out.display()))?;
    println!(
        "Could not download {label}; copied bundled fallback to {}",
        out.display()
    );
    Ok(true)
}

// ── prepare ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
struct OpenEpdEntry {
    rarity: f32,
    ipa: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenEpdRarityEntry {
    rarity: f32,
}

const OPENEPD_SOURCE_PREFERENCE: &[&str] = &[
    "misaki_gold",
    "cmu",
    "misaki_silver",
    "phonemicchart",
    "wiktionary",
    "wikipron",
];

fn load_openepd_prepare_lexemes() -> Result<(Vec<Lexeme>, usize)> {
    let raw: std::collections::BTreeMap<String, OpenEpdEntry> =
        serde_json::from_str(open_english_pronouncing_dictionary::CORPUS_JSON)
            .context("parsing embedded OpenEPD JSON")?;

    let mut lexemes = Vec::with_capacity(raw.len());
    let mut skipped = 0usize;
    for (base_word, entry) in raw {
        match prepare_lexeme_from_openepd_entry(base_word, entry) {
            Some(lexeme) => lexemes.push(lexeme),
            None => skipped += 1,
        }
    }

    Ok((lexemes, skipped))
}

fn prepare_lexeme_from_openepd_entry(base_word: String, entry: OpenEpdEntry) -> Option<Lexeme> {
    if !is_prepare_word(&base_word) {
        return None;
    }
    let raw_ipa =
        openepd_prepare_ipa_correction(&base_word).or_else(|| preferred_openepd_ipa(&entry.ipa))?;
    let phonemes = normalize_openepd_ipa(raw_ipa).ok()?;
    Some(Lexeme {
        base_word,
        phonemes,
        rarity: entry.rarity,
    })
}

fn openepd_prepare_ipa_correction(word: &str) -> Option<&'static str> {
    match word {
        // OpenEPD 0.1.0 has only `misaki_silver: ʌnɹˈɑʔn`, which broadens to
        // `ʌnˈɹɑtn` and loses the schwa syllable in "rotten".
        "unrotten" => Some("ʌnɹˈɑtən"),
        _ => None,
    }
}

fn preferred_openepd_ipa(ipa: &std::collections::BTreeMap<String, String>) -> Option<&str> {
    for preferred_source in OPENEPD_SOURCE_PREFERENCE {
        if let Some(value) = ipa.get(*preferred_source) {
            return Some(value);
        }
        if let Some((_, value)) = ipa
            .iter()
            .find(|(source, _)| source.starts_with(preferred_source))
        {
            return Some(value);
        }
    }
    ipa.values().next().map(String::as_str)
}

fn is_prepare_word(word: &str) -> bool {
    !word.is_empty()
        && word
            .chars()
            .all(|c| c.is_alphabetic() || c == '\'' || c == '-')
}

fn cmd_prepare(
    _input: Option<&Path>,
    out: &Path,
    train_frac: f64,
    valid_frac: f64,
    _seed: u64,
) -> Result<()> {
    println!("Loading OpenEPD as prepare source ...");
    let (lexemes, skipped_openepd) = load_openepd_prepare_lexemes()?;
    let total_words = lexemes.len();
    println!(
        "  {} OpenEPD lexemes loaded ({} skipped by word/IPA filters)",
        format_count(total_words),
        format_count(skipped_openepd)
    );
    fs::create_dir_all(out).context("creating output directory")?;

    // Open output files
    let train_path = out.join("train.jsonl");
    let valid_path = out.join("valid.jsonl");
    let test_path = out.join("test.jsonl");
    write_cli_text_atomic(
        &out.join("prepare_state.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "status": "writing",
            "dataset_id": "openepd-v0",
        }))?,
    )?;

    let train_part = atomic_part_path(&train_path);
    let valid_part = atomic_part_path(&valid_path);
    let test_part = atomic_part_path(&test_path);
    archive_interrupted_part(&train_path)?;
    archive_interrupted_part(&valid_path)?;
    archive_interrupted_part(&test_path)?;

    let train_file = fs::File::create(&train_part)?;
    let valid_file = fs::File::create(&valid_part)?;
    let test_file = fs::File::create(&test_part)?;

    use indicatif::ProgressBar;
    use std::io::Write;

    let mut train_writer = std::io::BufWriter::new(train_file);
    let mut valid_writer = std::io::BufWriter::new(valid_file);
    let mut test_writer = std::io::BufWriter::new(test_file);

    // Track word lists for anti-leakage auditing
    let mut train_words = Vec::new();
    let mut valid_words = Vec::new();
    let mut test_words = Vec::new();

    // Vocab character/symbol accumulation
    let mut seen_word_chars = std::collections::BTreeSet::new();
    let mut seen_phoneme_chars = std::collections::BTreeSet::new();

    println!("Writing OpenEPD data splits ...");

    // Setup indicatif progress bar!
    let pb = tongues_core::register_progress_bar(ProgressBar::new(total_words as u64));
    pb.set_style(counted_progress_style()?);

    // Deterministic FNV-1a hash function for thread-safe split assignment
    fn fnv1a_hash(s: &str) -> u64 {
        let mut hash = 0xcbf29ce484222325;
        for byte in s.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    for lex in lexemes {
        for c in lex.base_word.chars() {
            seen_word_chars.insert(c.to_string());
        }
        for c in lex.phonemes.chars() {
            seen_phoneme_chars.insert(c.to_string());
        }

        // Split deterministically via FNV-1a hash
        let hash_val = fnv1a_hash(&lex.base_word);
        let fraction = (hash_val as f64) / (std::u64::MAX as f64);

        let line = serde_json::to_string(&lex)?;

        if fraction < train_frac {
            writeln!(train_writer, "{}", line)?;
            train_words.push(lex.base_word);
        } else if fraction < train_frac + valid_frac {
            writeln!(valid_writer, "{}", line)?;
            valid_words.push(lex.base_word);
        } else {
            writeln!(test_writer, "{}", line)?;
            test_words.push(lex.base_word);
        }

        pb.inc(1);
    }

    pb.finish_with_message("Done!");

    // Flush writers
    train_writer.flush()?;
    valid_writer.flush()?;
    test_writer.flush()?;
    drop(train_writer);
    drop(valid_writer);
    drop(test_writer);
    fs::rename(&train_part, &train_path).with_context(|| {
        format!(
            "moving {} to {}",
            train_part.display(),
            train_path.display()
        )
    })?;
    fs::rename(&valid_part, &valid_path).with_context(|| {
        format!(
            "moving {} to {}",
            valid_part.display(),
            valid_path.display()
        )
    })?;
    fs::rename(&test_part, &test_path)
        .with_context(|| format!("moving {} to {}", test_part.display(), test_path.display()))?;

    println!(
        "Data splits generated on-the-fly:\n  train={} valid={} test={}",
        format_count(train_words.len()),
        format_count(valid_words.len()),
        format_count(test_words.len())
    );

    // Save word lists
    for (name, words) in [
        ("train", &train_words),
        ("valid", &valid_words),
        ("test", &test_words),
    ] {
        let path = out.join(format!("{}_words.txt", name));
        let mut deduped = words.clone();
        deduped.sort_unstable();
        deduped.dedup();
        write_cli_text_atomic(&path, deduped.join("\n"))?;
    }

    // Build & save vocabulary
    println!("Building vocabulary from seen characters ...");
    let vocab = {
        let w_list: Vec<String> = seen_word_chars.iter().cloned().collect();
        let pm_list: Vec<String> = seen_phoneme_chars.iter().cloned().collect();
        Vocab::build(&w_list, &pm_list, &[])
    };

    println!("  Unified vocabulary size: {}", format_count(vocab.size()));
    let vocab_path = out.join("vocab.json");
    let vocab_json = serde_json::to_string_pretty(&vocab)?;
    write_cli_text_atomic(&vocab_path, &vocab_json).context("writing vocab.json")?;
    println!("  Vocab saved to {}", vocab_path.display());
    write_cli_text_atomic(
        &out.join("prepare_state.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "status": "complete",
            "dataset_id": "openepd-v0",
            "lexemes": total_words,
            "train_examples": train_words.len(),
            "valid_examples": valid_words.len(),
            "test_examples": test_words.len(),
        }))?,
    )?;

    println!("Prepare complete.");
    Ok(())
}

fn write_cli_text_atomic(path: &Path, contents: impl AsRef<str>) -> Result<()> {
    let part = atomic_part_path(path);
    archive_interrupted_part(path)?;
    fs::write(&part, contents.as_ref()).with_context(|| format!("writing {}", part.display()))?;
    fs::rename(&part, path)
        .with_context(|| format!("moving {} to {}", part.display(), path.display()))
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

fn wait_for_prepared_dataset(data: &Path, required_files: &[&str], label: &str) -> Result<()> {
    println!(
        "Waiting for {label} prepare to finish before training: {}",
        data.display()
    );
    loop {
        let files_ready = required_files_ready(data, required_files);
        let status = prepare_state_status(data)?;
        match (status.as_deref(), files_ready) {
            (Some("complete"), true) => {
                println!(
                    "{label} prepared data is ready; starting training from {}",
                    data.display()
                );
                return Ok(());
            }
            (None, true) => {
                println!(
                    "{label} prepared data is ready; starting training from {}",
                    data.display()
                );
                return Ok(());
            }
            (Some(state), _) => {
                println!(
                    "{label} prepare_state status={state}; waiting for final files in {}",
                    data.display()
                );
            }
            (None, false) => {
                println!(
                    "{label} prepared data is not ready yet; waiting for {}",
                    data.display()
                );
            }
        }
        std::thread::sleep(Duration::from_secs(30));
    }
}

fn required_files_ready(data: &Path, required_files: &[&str]) -> bool {
    required_files.iter().all(|name| {
        let path = data.join(name);
        path.exists()
            && path
                .metadata()
                .map(|metadata| metadata.len() > 0)
                .unwrap_or(false)
    })
}

fn prepare_state_status(data: &Path) -> Result<Option<String>> {
    let path = data.join("prepare_state.json");
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    Ok(value
        .get("status")
        .and_then(|status| status.as_str())
        .map(|status| status.to_string()))
}

fn read_jsonl(path: &Path) -> Result<Vec<Lexeme>> {
    let f = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = std::io::BufReader::new(f);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let lex: Lexeme = serde_json::from_str(&line)
            .with_context(|| format!("parsing JSONL line: {}", &line[..line.len().min(80)]))?;
        out.push(lex);
    }
    Ok(out)
}

fn load_openepd_rarity_by_word() -> Result<std::collections::BTreeMap<String, f32>> {
    let raw: std::collections::BTreeMap<String, OpenEpdRarityEntry> =
        serde_json::from_str(open_english_pronouncing_dictionary::CORPUS_JSON)
            .context("parsing embedded OpenEPD rarity JSON")?;
    Ok(raw
        .into_iter()
        .map(|(word, entry)| (word.to_ascii_lowercase(), entry.rarity))
        .collect())
}

const SIGHT_WORD_TRAINING_REPEATS: usize = 24;
const DEFAULT_MAX_FREQUENCY_REPEAT: usize = 8;
const DEFAULT_FREQUENCY_RARITY_CAP: f32 = 50_000.0;

fn frequency_repeat_count(rarity: f32, max_repeat: usize, rarity_cap: f32) -> usize {
    if max_repeat <= 1 || !rarity.is_finite() || !rarity_cap.is_finite() || rarity_cap <= 0.0 {
        return 1;
    }
    if rarity <= 0.0 {
        return max_repeat;
    }
    if rarity >= rarity_cap {
        return 1;
    }

    let scale = 1.0 - (rarity / rarity_cap);
    1 + ((max_repeat - 1) as f32 * scale).round() as usize
}

fn expand_frequency_weighted_training(
    lexemes: &[Lexeme],
    max_repeat: usize,
    rarity_cap: f32,
) -> Vec<Lexeme> {
    let expanded_len = lexemes
        .iter()
        .map(|lexeme| frequency_repeat_count(lexeme.rarity, max_repeat, rarity_cap))
        .sum();
    let mut expanded = Vec::with_capacity(expanded_len);

    for lexeme in lexemes {
        for _ in 0..frequency_repeat_count(lexeme.rarity, max_repeat, rarity_cap) {
            expanded.push(lexeme.clone());
        }
    }

    expanded
}

fn add_sight_word_training_examples(train_lexemes: &mut Vec<Lexeme>, data: &Path) -> Result<usize> {
    let sight_words: std::collections::BTreeSet<&str> = SIGHT_WORDS.iter().copied().collect();
    let mut selected = std::collections::BTreeMap::<String, Lexeme>::new();

    for split in ["train", "valid", "test"] {
        let path = data.join(format!("{}.jsonl", split));
        if !path.exists() {
            continue;
        }
        for lexeme in read_jsonl(&path)? {
            if sight_words.contains(lexeme.base_word.as_str()) {
                selected.entry(lexeme.base_word.clone()).or_insert(lexeme);
            }
        }
    }

    let mut added = 0usize;
    for lexeme in selected.values() {
        for _ in 0..SIGHT_WORD_TRAINING_REPEATS {
            train_lexemes.push(lexeme.clone());
            added += 1;
        }
    }

    Ok(added)
}

// ── train ──────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn cmd_train(
    data: &Path,
    out: &Path,
    _mask_policy_arg: MaskPolicyArg,
    _max_mask_rate: f64,
    _span_mask_prob: f64,
    learning_rate: f64,
    weight_decay: f32,
    dropout: f64,
    epochs: usize,
    patience: usize,
    batch_size: usize,
    seed: u64,
    task_str: String,
    device_arg: DeviceArg,
) -> Result<()> {
    if !data.join("vocab.json").exists()
        || !data.join("train.jsonl").exists()
        || !data.join("valid.jsonl").exists()
    {
        println!(
            "Data directory or required splits not found at {}. Automatically preparing...",
            data.display()
        );
        cmd_prepare(None, data, 0.8, 0.1, 42)?;
    }

    let vocab: Vocab = {
        let pb = status_spinner(format!("Loading vocabulary from {}", data.display()));
        let s = fs::read_to_string(data.join("vocab.json")).context("reading vocab.json")?;
        let vocab: Vocab = serde_json::from_str(&s)?;
        finish_status(
            pb,
            format!(
                "Loaded vocabulary with {} tokens",
                format_count(vocab.size())
            ),
        );
        vocab
    };

    let pb = status_spinner(format!(
        "Loading train/valid lexemes from {}",
        data.display()
    ));
    let base_train_lexemes = read_jsonl(&data.join("train.jsonl"))?;
    let valid_lexemes = read_jsonl(&data.join("valid.jsonl"))?;
    finish_status(
        pb,
        format!(
            "Loaded {} train / {} valid lexemes",
            format_count(base_train_lexemes.len()),
            format_count(valid_lexemes.len())
        ),
    );

    println!(
        "Loaded {} train / {} valid lexemes",
        format_count(base_train_lexemes.len()),
        format_count(valid_lexemes.len())
    );

    let model_config = ModelConfig::new(vocab.size()).with_dropout(dropout);

    let task_opt = match task_str.to_lowercase().as_str() {
        "g2p" => Some(Task::G2P),
        "p2g" => Some(Task::P2G),
        "both" => None,
        _ => anyhow::bail!("Invalid task. Supported: g2p, p2g, both"),
    };

    let train_config = TrainConfig {
        learning_rate,
        weight_decay,
        dropout,
        batch_size,
        epochs,
        early_stopping_patience: patience,
        max_seq_len: model_config.max_seq_len,
        task: task_opt,
        max_frequency_repeat: DEFAULT_MAX_FREQUENCY_REPEAT,
        frequency_rarity_cap: DEFAULT_FREQUENCY_RARITY_CAP,
    };

    let pb = status_spinner("Expanding frequency-weighted training examples");
    let mut train_lexemes = expand_frequency_weighted_training(
        &base_train_lexemes,
        train_config.max_frequency_repeat,
        train_config.frequency_rarity_cap,
    );
    finish_status(
        pb,
        format!(
            "Expanded to {} frequency-weighted train examples",
            format_count(train_lexemes.len())
        ),
    );
    println!(
        "  frequency-weighted train examples: {} (max_repeat={} rarity_cap={})",
        format_count(train_lexemes.len()),
        format_count(train_config.max_frequency_repeat),
        train_config.frequency_rarity_cap
    );

    let added_sight_word_lexemes = add_sight_word_training_examples(&mut train_lexemes, data)?;
    if added_sight_word_lexemes > 0 {
        println!(
            "  included {} extra sight-word training examples",
            format_count(added_sight_word_lexemes)
        );
    }

    fs::create_dir_all(out).context("creating model directory")?;

    // Save model config and train config for later use by eval/predict
    let model_config_path = out.join("model_config.json");
    fs::write(
        &model_config_path,
        serde_json::to_string_pretty(&model_config)?,
    )?;
    let train_config_path = out.join("train_config.json");
    fs::write(
        &train_config_path,
        serde_json::to_string_pretty(&train_config)?,
    )?;

    // Copy vocab.json to model output directory to make it self-contained
    let vocab_src = data.join("vocab.json");
    let vocab_dst = out.join("vocab.json");
    if vocab_src.exists() {
        fs::copy(&vocab_src, &vocab_dst).context("copying vocab.json to model directory")?;
    }

    write_manifest(
        out,
        &ModelArtifactManifest::new("g2p2g", "seq2seq-transformer", data_id_from_path(data))
            .with_task(task_str.to_lowercase()),
    )?;

    let model_path = out.join("model");

    println!("Starting training...");
    println!(
        "  lr={} wd={} dropout={}",
        learning_rate, weight_decay, dropout
    );
    println!(
        "  epochs={} patience={} batch_size={}",
        format_count(epochs),
        format_count(patience),
        format_count(batch_size)
    );
    println!("  train_state: {}", out.join("train_state.json").display());
    println!("  early_stop_metric: val_loss");
    println!(
        "  epoch checkpoints: {}",
        out.join("model-epoch-N.bin").display()
    );
    println!(
        "  best model: {}",
        model_path.with_extension("bin").display()
    );

    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            println!("  device: CPU (ndarray)");
            run_train::<CpuTrainBackend>(
                &device,
                &model_config,
                &train_config,
                &train_lexemes,
                &valid_lexemes,
                &vocab,
                &model_path,
                seed,
            )?;
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            println!("  device: CUDA GPU");
            run_train::<CudaTrainBackend>(
                &device,
                &model_config,
                &train_config,
                &train_lexemes,
                &valid_lexemes,
                &vocab,
                &model_path,
                seed,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_train<B: AutodiffBackend>(
    device: &B::Device,
    model_config: &ModelConfig,
    train_config: &TrainConfig,
    train_lexemes: &[Lexeme],
    valid_lexemes: &[Lexeme],
    vocab: &Vocab,
    model_path: &Path,
    seed: u64,
) -> Result<()>
where
    <Seq2SeqModel<B> as burn::module::Module<B>>::Record: Send,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let best_loss = train::<B, _>(
        model_config,
        train_config,
        train_lexemes,
        valid_lexemes,
        vocab,
        model_path,
        device,
        &mut rng,
    )?;

    println!(
        "\nTraining complete. Best validation loss: {:.4}",
        best_loss
    );
    println!("Model saved to {}", model_path.display());
    Ok(())
}

fn data_id_from_path(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown")
        .to_string()
}

// ── eval ───────────────────────────────────────────────────────────────────

fn cmd_eval(
    model_dir: &Path,
    split: &str,
    data: &Path,
    task_str: &str,
    device_arg: DeviceArg,
) -> Result<()> {
    let vocab: Vocab = {
        let s = fs::read_to_string(data.join("vocab.json")).context("reading vocab.json")?;
        serde_json::from_str(&s)?
    };
    let model_config: ModelConfig = {
        let s = fs::read_to_string(model_dir.join("model_config.json"))
            .context("reading model_config.json")?;
        serde_json::from_str(&s)?
    };

    let test_lexemes = read_jsonl(&data.join(format!("{}.jsonl", split)))?;
    let train_lexemes = read_jsonl(&data.join("train.jsonl"))?;

    let resolved_task = if task_str.to_lowercase() == "auto" {
        let config_path = model_dir.join("train_config.json");
        if config_path.exists() {
            let s = fs::read_to_string(&config_path).context("reading train_config.json")?;
            let train_config: TrainConfig = serde_json::from_str(&s)?;
            train_config.task
        } else {
            None
        }
    } else {
        match task_str.to_lowercase().as_str() {
            "g2p" => Some(Task::G2P),
            "p2g" => Some(Task::P2G),
            "both" => None,
            _ => anyhow::bail!("Invalid task. Supported: g2p, p2g, both, auto"),
        }
    };

    println!(
        "Evaluating on {} split ({} lexemes) ...",
        split,
        format_count(test_lexemes.len())
    );
    if let Some(task) = resolved_task {
        println!("  task: {:?}", task);
    } else {
        println!("  task: both");
    }

    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            println!("  device: CPU (ndarray)");
            run_eval::<CpuInferBackend>(
                &device,
                &model_config,
                model_dir,
                split,
                &vocab,
                resolved_task,
                &test_lexemes,
                &train_lexemes,
            )?;
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            println!("  device: CUDA GPU");
            run_eval::<CudaInferBackend>(
                &device,
                &model_config,
                model_dir,
                split,
                &vocab,
                resolved_task,
                &test_lexemes,
                &train_lexemes,
            )?;
        }
    }
    Ok(())
}

fn run_eval<B: Backend>(
    device: &B::Device,
    model_config: &ModelConfig,
    model_dir: &Path,
    _split: &str,
    vocab: &Vocab,
    task_filter: Option<Task>,
    test_lexemes: &[Lexeme],
    train_lexemes: &[Lexeme],
) -> Result<()> {
    let model = load_model::<B>(model_config, &model_dir.join("model"), device)?;
    let mut rng = StdRng::seed_from_u64(0);

    let report = eval_report(
        &model,
        test_lexemes,
        train_lexemes,
        vocab,
        task_filter,
        model_config.max_seq_len,
        device,
        &mut rng,
    );

    println!("\n── Evaluation Results ──");
    println!("  Loss          : {:.4}", report.val_loss);
    println!("  Exact match   : {:.3}", report.exact_match_accuracy);
    println!("  Token accuracy: {:.3}", report.token_accuracy);

    Ok(())
}

// ── refine ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct DiscrepancyRecord {
    split: String,
    task: String,
    gold_source: String,
    base_word: String,
    input: String,
    gold: String,
    prediction: String,
    gold_compare: String,
    prediction_compare: String,
    edit_distance: usize,
}

const SIGHT_WORDS: &[&str] = &[
    "a",
    "about",
    "after",
    "again",
    "all",
    "always",
    "am",
    "an",
    "and",
    "any",
    "apple",
    "are",
    "around",
    "as",
    "ask",
    "at",
    "ate",
    "away",
    "baby",
    "back",
    "ball",
    "be",
    "bear",
    "because",
    "bed",
    "been",
    "before",
    "bell",
    "best",
    "better",
    "big",
    "bird",
    "birthday",
    "black",
    "blue",
    "boat",
    "both",
    "box",
    "boy",
    "bread",
    "bring",
    "brown",
    "but",
    "buy",
    "by",
    "cake",
    "call",
    "came",
    "can",
    "car",
    "carry",
    "cat",
    "chair",
    "chicken",
    "children",
    "christmas",
    "clean",
    "coat",
    "cold",
    "come",
    "corn",
    "could",
    "cow",
    "cut",
    "day",
    "did",
    "do",
    "does",
    "dog",
    "doll",
    "done",
    "door",
    "down",
    "draw",
    "drink",
    "duck",
    "eat",
    "egg",
    "eight",
    "every",
    "eye",
    "fall",
    "far",
    "farm",
    "farmer",
    "fast",
    "father",
    "feet",
    "find",
    "fire",
    "first",
    "fish",
    "five",
    "floor",
    "flower",
    "fly",
    "for",
    "found",
    "four",
    "from",
    "full",
    "funny",
    "game",
    "garden",
    "gave",
    "get",
    "girl",
    "give",
    "go",
    "goes",
    "going",
    "good",
    "goodbye",
    "got",
    "grass",
    "green",
    "ground",
    "grow",
    "had",
    "hand",
    "has",
    "have",
    "he",
    "head",
    "help",
    "her",
    "here",
    "hill",
    "him",
    "his",
    "hold",
    "home",
    "horse",
    "hot",
    "house",
    "how",
    "hurt",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "jump",
    "just",
    "keep",
    "kind",
    "kitty",
    "know",
    "laugh",
    "leg",
    "let",
    "letter",
    "light",
    "like",
    "little",
    "live",
    "long",
    "look",
    "made",
    "make",
    "man",
    "many",
    "may",
    "me",
    "men",
    "milk",
    "money",
    "morning",
    "mother",
    "much",
    "must",
    "my",
    "myself",
    "name",
    "nest",
    "never",
    "new",
    "night",
    "no",
    "not",
    "now",
    "of",
    "off",
    "old",
    "on",
    "once",
    "one",
    "only",
    "open",
    "or",
    "our",
    "out",
    "over",
    "own",
    "paper",
    "party",
    "picture",
    "pick",
    "pig",
    "play",
    "please",
    "pretty",
    "pull",
    "put",
    "rabbit",
    "rain",
    "ran",
    "read",
    "red",
    "ride",
    "right",
    "ring",
    "robin",
    "round",
    "run",
    "said",
    "santa",
    "saw",
    "say",
    "school",
    "see",
    "seed",
    "seven",
    "shall",
    "she",
    "sheep",
    "shoe",
    "show",
    "sing",
    "sister",
    "sit",
    "six",
    "sleep",
    "small",
    "snow",
    "so",
    "some",
    "song",
    "soon",
    "squirrel",
    "start",
    "stick",
    "stop",
    "street",
    "sun",
    "table",
    "take",
    "tell",
    "ten",
    "thank",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "thing",
    "think",
    "this",
    "those",
    "three",
    "time",
    "to",
    "today",
    "together",
    "too",
    "top",
    "toy",
    "tree",
    "try",
    "two",
    "under",
    "up",
    "upon",
    "us",
    "use",
    "very",
    "walk",
    "warm",
    "was",
    "wash",
    "watch",
    "water",
    "way",
    "we",
    "well",
    "went",
    "were",
    "what",
    "when",
    "where",
    "which",
    "white",
    "who",
    "why",
    "will",
    "wind",
    "window",
    "wish",
    "with",
    "wood",
    "work",
    "would",
    "write",
    "yellow",
    "yes",
    "you",
    "your",
];

#[allow(clippy::too_many_arguments)]
fn cmd_refine(
    model_dir: &Path,
    data: &Path,
    out: &Path,
    splits: &str,
    source: RefinementSourceArg,
    task_str: &str,
    learning_rate: f64,
    weight_decay: f32,
    epochs: usize,
    patience: usize,
    batch_size: usize,
    seed: u64,
    verbose: bool,
    device_arg: DeviceArg,
) -> Result<()> {
    let vocab: Vocab = {
        let s = fs::read_to_string(data.join("vocab.json")).context("reading vocab.json")?;
        serde_json::from_str(&s)?
    };
    let model_config: ModelConfig = {
        let s = fs::read_to_string(model_dir.join("model_config.json"))
            .context("reading model_config.json")?;
        serde_json::from_str(&s)?
    };

    let task_filter = match task_str.to_lowercase().as_str() {
        "g2p" => Some(Task::G2P),
        "p2g" => Some(Task::P2G),
        "both" => None,
        _ => anyhow::bail!("Invalid task. Supported: g2p, p2g, both"),
    };

    let split_names: Vec<String> = splits
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if matches!(source, RefinementSourceArg::Discrepancies) && split_names.is_empty() {
        anyhow::bail!("At least one split is required");
    }

    if out.exists() && model_dir.exists() {
        let out_canon = out
            .canonicalize()
            .context("canonicalizing output directory")?;
        let model_canon = model_dir
            .canonicalize()
            .context("canonicalizing model directory")?;
        if out_canon == model_canon {
            anyhow::bail!(
                "Refinement output must be separate from the source model directory: {}",
                out.display()
            );
        }
    }

    let mut split_lexemes = Vec::new();
    if matches!(source, RefinementSourceArg::Discrepancies) {
        for split in &split_names {
            let path = data.join(format!("{}.jsonl", split));
            let lexemes = read_jsonl(&path)?;
            split_lexemes.push((split.clone(), lexemes));
        }
    }

    fs::create_dir_all(out).context("creating refinement output directory")?;
    if out.join("train_state.json").exists() {
        println!(
            "Existing refinement state found in {}; training will resume there",
            out.display()
        );
    } else {
        fs::copy(
            model_dir.join("model_config.json"),
            out.join("model_config.json"),
        )
        .context("copying model_config.json")?;
        fs::copy(data.join("vocab.json"), out.join("vocab.json")).context("copying vocab.json")?;
        fs::copy(model_dir.join("model.bin"), out.join("model.bin"))
            .context("copying base model")?;
    }

    println!("Mining discrepancies from {}", model_dir.display());
    println!("  gold source: OpenEPD preferred IPA");
    match source {
        RefinementSourceArg::Discrepancies => {
            println!("  source: held-out discrepancies");
            println!("  splits: {}", split_names.join(","));
            for (split, lexemes) in &split_lexemes {
                println!("  {}: {} lexemes", split, format_count(lexemes.len()));
            }
        }
        RefinementSourceArg::SightWords => {
            println!(
                "  source: built-in Dolch sight words ({} words before OpenEPD/vocab filtering)",
                format_count(SIGHT_WORDS.len())
            );
        }
    }
    if let Some(task) = task_filter {
        println!("  task: {:?}", task);
    } else {
        println!("  task: both");
    }
    println!(
        "  output: {}{}",
        out.display(),
        if verbose { " (verbose)" } else { "" }
    );

    let (records, refine_lexemes) = match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            println!("  device: CPU (ndarray)");
            match source {
                RefinementSourceArg::Discrepancies => collect_discrepancies::<CpuInferBackend>(
                    &device,
                    &model_config,
                    model_dir,
                    &vocab,
                    task_filter,
                    &split_lexemes,
                    verbose,
                )?,
                RefinementSourceArg::SightWords => {
                    collect_sight_word_refinement::<CpuInferBackend>(
                        &device,
                        &model_config,
                        model_dir,
                        &vocab,
                        task_filter,
                        verbose,
                    )?
                }
            }
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            println!("  device: CUDA GPU");
            match source {
                RefinementSourceArg::Discrepancies => collect_discrepancies::<CudaInferBackend>(
                    &device,
                    &model_config,
                    model_dir,
                    &vocab,
                    task_filter,
                    &split_lexemes,
                    verbose,
                )?,
                RefinementSourceArg::SightWords => {
                    collect_sight_word_refinement::<CudaInferBackend>(
                        &device,
                        &model_config,
                        model_dir,
                        &vocab,
                        task_filter,
                        verbose,
                    )?
                }
            }
        }
    };

    let discrepancies_path = out.join("discrepancies.jsonl");
    write_discrepancies(&discrepancies_path, &records)?;
    println!(
        "Stored {} discrepancies at {}",
        format_count(records.len()),
        discrepancies_path.display()
    );
    print_discrepancy_summary(&records);

    write_manifest(
        out,
        &ModelArtifactManifest::new("g2p2g", "seq2seq-transformer", data_id_from_path(data))
            .with_task(task_str.to_lowercase()),
    )?;

    if refine_lexemes.is_empty() {
        println!("No refinement examples found. Refinement skipped.");
        return Ok(());
    }

    let total_edit_distance: usize = records.iter().map(|r| r.edit_distance).sum();
    let mean_edit_distance = if records.is_empty() {
        0.0
    } else {
        total_edit_distance as f32 / records.len() as f32
    };
    println!(
        "Refinement set: {} lexemes, mean edit distance {:.2}",
        format_count(refine_lexemes.len()),
        mean_edit_distance
    );
    println!(
        "Refinement training: lr={} wd={} epochs={} patience={} batch_size={}",
        learning_rate,
        weight_decay,
        format_count(epochs),
        format_count(patience),
        format_count(batch_size)
    );

    let train_config = TrainConfig {
        learning_rate,
        weight_decay,
        dropout: model_config.dropout,
        batch_size,
        epochs,
        early_stopping_patience: patience,
        max_seq_len: model_config.max_seq_len,
        task: task_filter,
        max_frequency_repeat: DEFAULT_MAX_FREQUENCY_REPEAT,
        frequency_rarity_cap: DEFAULT_FREQUENCY_RARITY_CAP,
    };
    fs::write(
        out.join("train_config.json"),
        serde_json::to_string_pretty(&train_config)?,
    )?;

    let model_path = out.join("model");
    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            run_train::<CpuTrainBackend>(
                &device,
                &model_config,
                &train_config,
                &refine_lexemes,
                &refine_lexemes,
                &vocab,
                &model_path,
                seed,
            )?;
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            run_train::<CudaTrainBackend>(
                &device,
                &model_config,
                &train_config,
                &refine_lexemes,
                &refine_lexemes,
                &vocab,
                &model_path,
                seed,
            )?;
        }
    }

    Ok(())
}

fn collect_discrepancies<B: Backend>(
    device: &B::Device,
    model_config: &ModelConfig,
    model_dir: &Path,
    vocab: &Vocab,
    task_filter: Option<Task>,
    split_lexemes: &[(String, Vec<Lexeme>)],
    verbose: bool,
) -> Result<(Vec<DiscrepancyRecord>, Vec<Lexeme>)> {
    let model = load_model::<B>(model_config, &model_dir.join("model"), device)?;
    println!("Loading OpenEPD corpus...");
    let openepd = open_english_pronouncing_dictionary::load()
        .map_err(|err| anyhow::anyhow!("loading OpenEPD corpus: {}", err))?;
    println!("  OpenEPD words: {}", format_count(openepd.word_count()));

    let tasks: Vec<Task> = match task_filter {
        Some(task) => vec![task],
        None => vec![Task::G2P, Task::P2G],
    };

    let total: usize = split_lexemes
        .iter()
        .map(|(_, lexemes)| lexemes.len() * tasks.len())
        .sum();
    let pb = tongues_core::register_progress_bar(indicatif::ProgressBar::new(total as u64));
    pb.set_style(counted_progress_style()?);

    let mut records = Vec::new();
    let mut refine_lexemes = Vec::new();
    let mut refine_seen = std::collections::BTreeSet::new();
    let mut skipped_missing_openepd = 0usize;
    let mut skipped_parse_error = 0usize;
    let mut skipped_unknown_vocab = 0usize;
    for (split, lexemes) in split_lexemes {
        let mut split_checked = 0usize;
        let mut split_discrepancies = 0usize;
        let mut split_skipped_missing_openepd = 0usize;
        let mut split_skipped_parse_error = 0usize;
        let mut split_skipped_unknown_vocab = 0usize;
        for lex in lexemes {
            let base_word = lex.base_word.to_lowercase();
            let Some(raw_openepd_ipa) = openepd.preferred_ipa(&base_word) else {
                skipped_missing_openepd += tasks.len();
                split_skipped_missing_openepd += tasks.len();
                if verbose {
                    pb.println(format!(
                        "SKIP split={} word={} reason=no-openepd-entry",
                        split, base_word
                    ));
                }
                pb.inc(tasks.len() as u64);
                continue;
            };
            let openepd_ipa = match normalize_openepd_ipa(raw_openepd_ipa) {
                Ok(normalized) => normalized,
                Err(err) => {
                    skipped_parse_error += tasks.len();
                    split_skipped_parse_error += tasks.len();
                    if verbose {
                        pb.println(format!(
                            "SKIP split={} word={} reason=openepd-parse-error raw={} error={}",
                            split, base_word, raw_openepd_ipa, err
                        ));
                    }
                    pb.inc(tasks.len() as u64);
                    continue;
                }
            };
            let openepd_lexeme = Lexeme {
                base_word: base_word.clone(),
                phonemes: openepd_ipa.clone(),
                rarity: lex.rarity,
            };

            if has_unknown_vocab(vocab, &openepd_ipa) {
                skipped_unknown_vocab += tasks.len();
                split_skipped_unknown_vocab += tasks.len();
                if verbose {
                    pb.println(format!(
                        "SKIP split={} word={} reason=openepd-gold-not-in-vocab gold={}",
                        split, base_word, openepd_ipa
                    ));
                }
                pb.inc(tasks.len() as u64);
                continue;
            }

            for &task in &tasks {
                let (input, gold, task_name) = match task {
                    Task::G2P => (base_word.clone(), openepd_ipa.clone(), "g2p".to_string()),
                    Task::P2G => (openepd_ipa.clone(), base_word.clone(), "p2g".to_string()),
                };
                pb.set_message(format!("{} {}", split, base_word));
                let prediction = predict(&model, &input, task, vocab, device);
                let gold_compare = comparison_key(&gold, task);
                let prediction_compare = comparison_key(&prediction, task);
                let edit_distance = edit_distance_chars(&prediction_compare, &gold_compare);
                split_checked += 1;
                if edit_distance > 0 {
                    split_discrepancies += 1;
                    let record = DiscrepancyRecord {
                        split: split.clone(),
                        task: task_name,
                        gold_source: "openepd".to_string(),
                        base_word: base_word.clone(),
                        input,
                        gold,
                        prediction,
                        gold_compare,
                        prediction_compare,
                        edit_distance,
                    };
                    if verbose {
                        pb.println(format_discrepancy(&record));
                    }
                    records.push(record);
                    if refine_seen.insert(base_word.clone()) {
                        refine_lexemes.push(openepd_lexeme.clone());
                    }
                }
                pb.inc(1);
            }
        }
        pb.println(format!(
            "Completed split {}: checked {} examples, found {} discrepancies, skipped {} missing OpenEPD, skipped {} parse errors, skipped {} unknown-vocab golds",
            split,
            format_count(split_checked),
            format_count(split_discrepancies),
            format_count(split_skipped_missing_openepd),
            format_count(split_skipped_parse_error),
            format_count(split_skipped_unknown_vocab)
        ));
    }
    pb.finish_and_clear();
    if skipped_missing_openepd > 0 || skipped_parse_error > 0 || skipped_unknown_vocab > 0 {
        println!(
            "Skipped during OpenEPD mining: {} missing OpenEPD entries, {} parse errors, {} OpenEPD golds with chars outside vocab",
            format_count(skipped_missing_openepd),
            format_count(skipped_parse_error),
            format_count(skipped_unknown_vocab)
        );
    }

    Ok((records, refine_lexemes))
}

fn collect_sight_word_refinement<B: Backend>(
    device: &B::Device,
    model_config: &ModelConfig,
    model_dir: &Path,
    vocab: &Vocab,
    task_filter: Option<Task>,
    verbose: bool,
) -> Result<(Vec<DiscrepancyRecord>, Vec<Lexeme>)> {
    let model = load_model::<B>(model_config, &model_dir.join("model"), device)?;
    println!("Loading OpenEPD corpus...");
    let openepd = open_english_pronouncing_dictionary::load()
        .map_err(|err| anyhow::anyhow!("loading OpenEPD corpus: {}", err))?;
    println!("  OpenEPD words: {}", format_count(openepd.word_count()));

    let tasks: Vec<Task> = match task_filter {
        Some(task) => vec![task],
        None => vec![Task::G2P, Task::P2G],
    };

    let mut sight_words = std::collections::BTreeSet::new();
    for word in SIGHT_WORDS {
        sight_words.insert((*word).to_string());
    }

    let pb = tongues_core::register_progress_bar(indicatif::ProgressBar::new(
        (sight_words.len() * tasks.len()) as u64,
    ));
    pb.set_style(counted_progress_style()?);

    let mut records = Vec::new();
    let mut refine_lexemes = Vec::new();
    let mut skipped_missing_openepd = 0usize;
    let mut skipped_parse_error = 0usize;
    let mut skipped_unknown_vocab = 0usize;
    let mut checked = 0usize;

    for base_word in sight_words {
        let Some(raw_openepd_ipa) = openepd.preferred_ipa(&base_word) else {
            skipped_missing_openepd += tasks.len();
            if verbose {
                pb.println(format!(
                    "SKIP split=sight-words word={} reason=no-openepd-entry",
                    base_word
                ));
            }
            pb.inc(tasks.len() as u64);
            continue;
        };
        let openepd_ipa = match normalize_openepd_ipa(raw_openepd_ipa) {
            Ok(normalized) => normalized,
            Err(err) => {
                skipped_parse_error += tasks.len();
                if verbose {
                    pb.println(format!(
                        "SKIP split=sight-words word={} reason=openepd-parse-error raw={} error={}",
                        base_word, raw_openepd_ipa, err
                    ));
                }
                pb.inc(tasks.len() as u64);
                continue;
            }
        };

        if has_unknown_vocab(vocab, &base_word) || has_unknown_vocab(vocab, &openepd_ipa) {
            skipped_unknown_vocab += tasks.len();
            if verbose {
                pb.println(format!(
                    "SKIP split=sight-words word={} reason=gold-not-in-vocab phonemes={}",
                    base_word, openepd_ipa
                ));
            }
            pb.inc(tasks.len() as u64);
            continue;
        }

        refine_lexemes.push(Lexeme {
            base_word: base_word.clone(),
            phonemes: openepd_ipa.clone(),
            rarity: DEFAULT_FREQUENCY_RARITY_CAP,
        });

        for &task in &tasks {
            let (input, gold, task_name) = match task {
                Task::G2P => (base_word.clone(), openepd_ipa.clone(), "g2p".to_string()),
                Task::P2G => (openepd_ipa.clone(), base_word.clone(), "p2g".to_string()),
            };
            pb.set_message(format!("sight-words {}", base_word));
            let prediction = predict(&model, &input, task, vocab, device);
            let gold_compare = comparison_key(&gold, task);
            let prediction_compare = comparison_key(&prediction, task);
            let edit_distance = edit_distance_chars(&prediction_compare, &gold_compare);
            checked += 1;
            if edit_distance > 0 {
                let record = DiscrepancyRecord {
                    split: "sight-words".to_string(),
                    task: task_name,
                    gold_source: "openepd-dolch".to_string(),
                    base_word: base_word.clone(),
                    input,
                    gold,
                    prediction,
                    gold_compare,
                    prediction_compare,
                    edit_distance,
                };
                if verbose {
                    pb.println(format_discrepancy(&record));
                }
                records.push(record);
            }
            pb.inc(1);
        }
    }
    pb.println(format!(
        "Completed sight-word source: checked {} examples, found {} discrepancies, selected {} training lexemes, skipped {} missing OpenEPD, skipped {} parse errors, skipped {} unknown-vocab forms",
        format_count(checked),
        format_count(records.len()),
        format_count(refine_lexemes.len()),
        format_count(skipped_missing_openepd),
        format_count(skipped_parse_error),
        format_count(skipped_unknown_vocab)
    ));
    pb.finish_and_clear();

    Ok((records, refine_lexemes))
}

fn format_discrepancy(record: &DiscrepancyRecord) -> String {
    let mut text = format!(
        "EXCEPTION split={} task={} gold_source={} word={} edit_distance={}\n  input: {}\n  gold : {}\n  pred : {}",
        record.split,
        record.task,
        record.gold_source,
        record.base_word,
        record.edit_distance,
        record.input,
        record.gold,
        record.prediction
    );
    if record.gold_compare != record.gold || record.prediction_compare != record.prediction {
        text.push_str(&format!(
            "\n  cmp gold: {}\n  cmp pred: {}",
            record.gold_compare, record.prediction_compare
        ));
    }
    text
}

fn has_unknown_vocab(vocab: &Vocab, text: &str) -> bool {
    vocab.encode_string(text).into_iter().any(|id| id == UNK_ID)
}

fn comparison_key(value: &str, task: Task) -> String {
    match task {
        Task::G2P => pronunciation_comparison_key(value),
        Task::P2G => value.to_lowercase(),
    }
}

fn pronunciation_comparison_key(value: &str) -> String {
    let no_length = value.replace('ː', "");
    let no_syllable_marks = no_length.replace('.', "");
    no_syllable_marks
        .chars()
        .filter(|c| !matches!(c, 'ˈ' | 'ˌ'))
        .collect::<String>()
        .replace('ɝ', "ɚ")
        .replace("iə", "iɚ")
        .replace("uə", "uɚ")
        .replace("əɹ", "ɚ")
        .replace("lɹ", "lɚ")
}

fn print_discrepancy_summary(records: &[DiscrepancyRecord]) {
    if records.is_empty() {
        return;
    }

    let mut by_split_task = std::collections::BTreeMap::<(String, String), usize>::new();
    for record in records {
        *by_split_task
            .entry((record.split.clone(), record.task.clone()))
            .or_default() += 1;
    }

    println!("Discrepancy counts:");
    for ((split, task), count) in by_split_task {
        println!("  {} {}: {}", split, task, count);
    }

    let mut worst = records.to_vec();
    worst.sort_by(|a, b| {
        b.edit_distance
            .cmp(&a.edit_distance)
            .then_with(|| a.base_word.cmp(&b.base_word))
    });

    println!("Largest edit distances:");
    for record in worst.iter().take(10) {
        println!(
            "  {} {} {} edit_distance={} gold={} pred={}",
            record.split,
            record.task,
            record.base_word,
            record.edit_distance,
            record.gold,
            record.prediction
        );
    }
}

fn write_discrepancies(path: &Path, records: &[DiscrepancyRecord]) -> Result<()> {
    use std::io::Write;

    let file = fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    for record in records {
        writeln!(writer, "{}", serde_json::to_string(record)?)?;
    }
    writer.flush()?;
    Ok(())
}

fn edit_distance_chars(left: &str, right: &str) -> usize {
    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();
    let mut prev: Vec<usize> = (0..=right.len()).collect();
    let mut curr = vec![0; right.len() + 1];

    for (i, lc) in left.iter().enumerate() {
        curr[0] = i + 1;
        for (j, rc) in right.iter().enumerate() {
            let substitution = prev[j] + usize::from(lc != rc);
            let insertion = curr[j] + 1;
            let deletion = prev[j + 1] + 1;
            curr[j + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[right.len()]
}

#[cfg(test)]
mod refinement_tests {
    use super::*;

    #[test]
    fn pronunciation_comparison_ignores_length_stress_and_syllable_marks() {
        assert_eq!(
            pronunciation_comparison_key("ˈziː.ə"),
            pronunciation_comparison_key("ˈziə")
        );
        assert_eq!(
            pronunciation_comparison_key("ˈʒuː"),
            pronunciation_comparison_key("ˈʒu")
        );
    }

    #[test]
    fn pronunciation_comparison_collapses_common_r_colored_spellings() {
        assert_eq!(
            pronunciation_comparison_key("ˈziː.ɡɚ"),
            pronunciation_comparison_key("ˈziɡəɹ")
        );
        assert_eq!(
            pronunciation_comparison_key("ˈziː.ɡlɚ"),
            pronunciation_comparison_key("ˈziɡlɹ")
        );
    }
}

// ── predict ────────────────────────────────────────────────────────────────

fn cmd_predict(
    model_dir: &Path,
    task_str: &str,
    input: &str,
    device_arg: DeviceArg,
    data_arg: Option<&Path>,
    output_mode: OutputMode,
) -> Result<()> {
    let start_total = std::time::Instant::now();

    if output_mode.verbose() {
        println!("Loading vocabulary...");
    }
    let start_vocab = std::time::Instant::now();
    // Load vocab
    let vocab: Vocab = {
        let mut found = None;

        // 1. Check if data_arg was passed
        if let Some(data_path) = data_arg {
            let p = data_path.join("vocab.json");
            if p.exists() {
                found = Some(p);
            }
        }

        // 2. Check next to the model file
        if found.is_none() {
            let p = model_dir.join("vocab.json");
            if p.exists() {
                found = Some(p);
            }
        }

        // 3. Check model parent dir
        if found.is_none() {
            let p = model_dir.parent().unwrap_or(model_dir).join("vocab.json");
            if p.exists() {
                found = Some(p);
            }
        }

        // 4. Try sibling folder (substituting "models" for "runs" or next to model_dir)
        if found.is_none() {
            let p = model_dir
                .parent()
                .unwrap_or(model_dir)
                .parent()
                .unwrap_or(model_dir)
                .join("runs")
                .join(model_dir.file_name().unwrap_or_default())
                .join("vocab.json");
            if p.exists() {
                found = Some(p);
            }
        }

        let path = found.context(
            "vocab.json not found. Pass --data to specify the prepared data directory containing vocab.json, or copy vocab.json to the model directory.",
        )?;
        let s = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&s)?
    };
    if output_mode.verbose() {
        println!("  ✓ Loaded vocabulary in {:?}", start_vocab.elapsed());
    }

    let task = if task_str.to_lowercase() == "auto" {
        detect_task(input)
    } else {
        Task::from_str(task_str)
            .ok_or_else(|| anyhow::anyhow!("Invalid task. Supported: g2p, p2g, auto"))?
    };

    let model_config: ModelConfig = {
        let s = fs::read_to_string(model_dir.join("model_config.json"))
            .context("reading model_config.json")?;
        serde_json::from_str(&s)?
    };

    match device_arg {
        DeviceArg::Cpu => {
            if output_mode.verbose() {
                println!("Initializing CPU device (ndarray)...");
            }
            let start_dev = std::time::Instant::now();
            let device = NdArrayDevice::Cpu;
            if output_mode.verbose() {
                println!("  ✓ Initialized CPU device in {:?}", start_dev.elapsed());
            }
            run_predict::<CpuInferBackend>(
                &device,
                &model_config,
                model_dir,
                &vocab,
                task,
                input,
                start_total,
                output_mode,
            )?;
        }
        DeviceArg::Cuda => {
            if output_mode.verbose() {
                println!("Initializing CUDA GPU device...");
            }
            let start_dev = std::time::Instant::now();
            let device = CudaDevice::default();
            if output_mode.verbose() {
                println!(
                    "  ✓ Initialized CUDA GPU device in {:?}",
                    start_dev.elapsed()
                );
            }
            run_predict::<CudaInferBackend>(
                &device,
                &model_config,
                model_dir,
                &vocab,
                task,
                input,
                start_total,
                output_mode,
            )?;
        }
    }
    Ok(())
}

fn run_predict<B: Backend>(
    device: &B::Device,
    model_config: &ModelConfig,
    model_dir: &Path,
    vocab: &Vocab,
    task: Task,
    input: &str,
    start_total: std::time::Instant,
    output_mode: OutputMode,
) -> Result<()> {
    if output_mode.verbose() {
        println!("Loading model config & weights...");
    }
    let start_load = std::time::Instant::now();
    let model = load_model::<B>(model_config, &model_dir.join("model"), device)?;
    if output_mode.verbose() {
        println!("  ✓ Loaded model weights in {:?}", start_load.elapsed());
    }

    if output_mode.verbose() {
        println!("Translating input='{}' with task={:?}...", input, task);
    }
    let start_pred = std::time::Instant::now();
    let output = predict(&model, input, task, vocab, device);
    if output_mode.verbose() {
        println!("  ✓ Finished prediction in {:?}", start_pred.elapsed());

        println!("\nPrediction output:\n  {}", output);
        println!("Total time elapsed: {:?}", start_total.elapsed());
    } else {
        println!("{output}");
    }

    Ok(())
}

fn cmd_repl(
    model_dir: &Path,
    task_str: &str,
    device_arg: DeviceArg,
    data_arg: Option<&Path>,
) -> Result<()> {
    println!("Loading vocabulary...");
    let start_vocab = std::time::Instant::now();
    // Load vocab
    let vocab: Vocab = {
        let mut found = None;

        // 1. Check if data_arg was passed
        if let Some(data_path) = data_arg {
            let p = data_path.join("vocab.json");
            if p.exists() {
                found = Some(p);
            }
        }

        // 2. Check next to the model file
        if found.is_none() {
            let p = model_dir.join("vocab.json");
            if p.exists() {
                found = Some(p);
            }
        }

        // 3. Check model parent dir
        if found.is_none() {
            let p = model_dir.parent().unwrap_or(model_dir).join("vocab.json");
            if p.exists() {
                found = Some(p);
            }
        }

        // 4. Try sibling folder (substituting "models" for "runs" or next to model_dir)
        if found.is_none() {
            let p = model_dir
                .parent()
                .unwrap_or(model_dir)
                .parent()
                .unwrap_or(model_dir)
                .join("runs")
                .join(model_dir.file_name().unwrap_or_default())
                .join("vocab.json");
            if p.exists() {
                found = Some(p);
            }
        }

        let path = found.context(
            "vocab.json not found. Pass --data to specify the prepared data directory containing vocab.json, or copy vocab.json to the model directory.",
        )?;
        let s = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&s)?
    };
    println!("  ✓ Loaded vocabulary in {:?}", start_vocab.elapsed());

    let model_config: ModelConfig = {
        let s = fs::read_to_string(model_dir.join("model_config.json"))
            .context("reading model_config.json")?;
        serde_json::from_str(&s)?
    };

    match device_arg {
        DeviceArg::Cpu => {
            println!("Initializing CPU device (ndarray)...");
            let start_dev = std::time::Instant::now();
            let device = NdArrayDevice::Cpu;
            println!("  ✓ Initialized CPU device in {:?}", start_dev.elapsed());
            run_repl::<CpuInferBackend>(&device, &model_config, model_dir, &vocab, task_str)?;
        }
        DeviceArg::Cuda => {
            println!("Initializing CUDA GPU device...");
            let start_dev = std::time::Instant::now();
            let device = CudaDevice::default();
            println!(
                "  ✓ Initialized CUDA GPU device in {:?}",
                start_dev.elapsed()
            );
            run_repl::<CudaInferBackend>(&device, &model_config, model_dir, &vocab, task_str)?;
        }
    }
    Ok(())
}

fn run_repl<B: Backend>(
    device: &B::Device,
    model_config: &ModelConfig,
    model_dir: &Path,
    vocab: &Vocab,
    initial_task_str: &str,
) -> Result<()> {
    println!("Loading model config & weights...");
    let start_load = std::time::Instant::now();
    let model = load_model::<B>(model_config, &model_dir.join("model"), device)?;
    println!("  ✓ Loaded model weights in {:?}", start_load.elapsed());

    let mut current_task = if initial_task_str.to_lowercase() == "auto" {
        None
    } else {
        Some(
            Task::from_str(initial_task_str)
                .ok_or_else(|| anyhow::anyhow!("Invalid task. Supported: g2p, p2g, auto"))?,
        )
    };

    let mut timings_enabled = true;

    println!("\nREPL ready! Enter input, or type :help for commands.");

    use std::io::{self, Write};
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut line = String::new();

    loop {
        print!("tongues> ");
        io::stdout().flush().context("flushing stdout")?;

        line.clear();
        let bytes_read = reader.read_line(&mut line).context("reading from stdin")?;
        if bytes_read == 0 {
            // EOF (Ctrl-D)
            println!();
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.starts_with(':') {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            match parts[0] {
                ":quit" | ":q" => {
                    break;
                }
                ":task" => {
                    if parts.len() < 2 {
                        println!("Error: specify task (g2p or p2g)");
                    } else {
                        match parts[1].to_lowercase().as_str() {
                            "g2p" => {
                                current_task = Some(Task::G2P);
                                println!("Task forced to grapheme-to-phoneme (G2P)");
                            }
                            "p2g" => {
                                current_task = Some(Task::P2G);
                                println!("Task forced to phoneme-to-grapheme (P2G)");
                            }
                            _ => {
                                println!("Error: invalid task. Supported: g2p, p2g");
                            }
                        }
                    }
                }
                ":auto" => {
                    current_task = None;
                    println!("Task auto-detect enabled");
                }
                ":timings" => {
                    timings_enabled = !timings_enabled;
                    if timings_enabled {
                        println!("Timing output enabled");
                    } else {
                        println!("Timing output disabled");
                    }
                }
                ":help" => {
                    println!("Commands:");
                    println!("  :quit / :q / Ctrl-D   Exits the REPL");
                    println!("  :task g2p            Forces grapheme-to-phoneme");
                    println!("  :task p2g            Forces phoneme-to-grapheme");
                    println!("  :auto                 Returns to auto-detect task");
                    println!("  :timings              Toggles timing output");
                    println!("  :help                 Prints this help message");
                }
                _ => {
                    println!(
                        "Unknown command: {}. Type :help for list of commands",
                        parts[0]
                    );
                }
            }
            continue;
        }

        let task = match current_task {
            Some(t) => t,
            None => detect_task(trimmed),
        };

        if timings_enabled {
            println!("Translating input='{}' with task={:?}...", trimmed, task);
        }

        let start_pred = std::time::Instant::now();
        let output = predict(&model, trimmed, task, vocab, device);
        let elapsed_pred = start_pred.elapsed();

        if timings_enabled {
            println!("  ✓ Finished prediction in {:?}", elapsed_pred);
            println!("\nPrediction output:\n  {}", output);
        } else {
            println!("{}", output);
        }
        println!();
    }

    Ok(())
}

/// Auto-detect the task based on the input text.
/// If all characters are ASCII alphabetic, apostrophes, or hyphens, we assume G2P.
/// Otherwise, we assume P2G.
pub fn detect_task(input: &str) -> Task {
    let is_spelling = !input.is_empty()
        && input
            .chars()
            .all(|c| c.is_ascii_alphabetic() || c == '\'' || c == '-');
    if is_spelling {
        Task::G2P
    } else {
        Task::P2G
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_count_adds_thousands_separators() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1_000), "1,000");
        assert_eq!(format_count(12_345_678), "12,345,678");
    }

    #[test]
    fn test_split_long_sentence() {
        let vocab = Vocab::build(&[], &[], &[]);
        let sentence = "It was not a gift that was taught; it was a gift that came to her on a rainy night when she was six, when she was tracing the curve of a wooden spoon on her grandmother’s kitchen table and the spoon’s silver handle seemed to glow with a faint, pulsing light.";
        let chunks = split_long_sentence(sentence, "en-US", &vocab, 100).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "It was not a gift that was taught;");
        assert!(chunks[1].starts_with("it was a gift"));
    }

    #[test]
    fn head2phones_prediction_extracts_phones_and_split_after() {
        let prediction = extract_head2phones_prediction(
            "<HEAD_FOUND>\n<HEAD_LENGTH> 20\n<PHONES> ˈluː.nə | ˈlɪvd </PHONES>\n<SPLIT_AFTER> 26",
        )
        .expect("head2phones prediction");

        assert_eq!(
            prediction,
            Head2PhonesPrediction {
                phones: "ˈluː.nə | ˈlɪvd".to_string(),
                split_after: Some(26),
            }
        );
    }

    #[test]
    fn head2phones_head_and_rest_uses_grapheme_split_after() {
        let sentence = "Luna lived with her grandmother, Nonna Rosa.";
        let split_after = "Luna lived with her grandmother,".chars().count();
        let (head, rest) = head2phones_head_and_rest(sentence, Some(split_after));

        assert_eq!(head, "Luna lived with her grandmother,");
        assert_eq!(rest, " Nonna Rosa.");
    }

    #[test]
    fn head2phones_head_and_rest_preserves_utf8_boundaries() {
        let sentence = "Café Luna listened. Then she smiled.";
        let split_after = "Café Luna listened.".chars().count();
        let (head, rest) = head2phones_head_and_rest(sentence, Some(split_after));

        assert_eq!(head, "Café Luna listened.");
        assert_eq!(rest, " Then she smiled.");
    }

    #[test]
    fn styletts2_head2phones_scanner_splits_serialized_words_into_phones() {
        let tokens = styletts2_phone_tokens_from_head2phones("ˈwəns | ə.ˈpɑn | ə | ˈtaɪm ↘ .");
        let ids = tokens
            .iter()
            .filter_map(|token| match &token.phone {
                Spec::Known(id) => Some(id.as_str().to_string()),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(
            ids,
            vec![
                "ipa.phone.w",
                "ipa.phone.ə",
                "ipa.phone.n",
                "ipa.phone.s",
                "boundary.word",
                "ipa.phone.ə",
                "ipa.phone.p",
                "ipa.phone.ɑ",
                "ipa.phone.n",
                "boundary.word",
                "ipa.phone.ə",
                "boundary.word",
                "ipa.phone.t",
                "ipa.phone.aɪ",
                "ipa.phone.m",
            ]
        );

        let plan = styletts2_plan_from_head2phones_prediction(
            "en-US",
            "Once upon a time.",
            "ˈwəns | ə.ˈpɑn | ə | ˈtaɪm ↘ .",
        )
        .expect("head2phones StyleTTS2 plan");
        prepare_styletts2_plan(
            &plan,
            &styletts2_en_us_symbol_set(),
            be_styletts2_options(DEFAULT_MAX_TTS_SYMBOLS, false),
        )
        .expect("StyleTTS2 backend plan");
    }

    #[test]
    fn speaking_demo_paragraph_picker_skips_toc_and_prefers_deeper_text() {
        let toc = "Chapitre I Monsieur Myriel Chapitre II Monsieur Myriel devient monseigneur Bienvenu Chapitre III À bon évêque dur évêché.";
        let early = "Cette phrase lisible arrive tôt dans le fichier, mais elle ne doit pas être le premier choix du mode paragraphe.";
        let deep = "Le passage choisi vient plus loin dans le livre, avec une phrase complète qui ressemble davantage au corps du texte.";
        let mut paragraphs = vec![toc.to_string(), early.to_string()];
        paragraphs.extend((0..10).map(|_| "Trop court.".to_string()));
        paragraphs.push(deep.to_string());

        let picked = speaking_demo_pick_paragraph(&paragraphs.join("\n\n"), "fr-FR-Standard")
            .expect("readable paragraph");

        assert_eq!(picked, deep);
        assert!(!speaking_demo_paragraph_is_readable(toc, "fr-FR-Standard"));
    }

    #[test]
    fn transcript_wer_accepts_case_and_punctuation_cleanup() {
        let wer = transcript_word_error_rate(
            "THE SECRET GARDEN WAS FIRST PUBLISHED IN NINETEEN ELEVEN",
            "The Secret Garden was first published in nineteen eleven.",
        );
        assert_eq!(wer, 0.0);
    }

    #[test]
    fn transcript_wer_is_lenient_for_digit_vs_spelled_numbers() {
        let wer = transcript_word_error_rate(
            "CHAPTER TWENTY FOUR WAS RECORDED IN NINETEEN ELEVEN",
            "Chapter 24 was recorded in 1911.",
        );
        assert!(wer <= DEFAULT_WHISPER_TRANSCRIPT_MAX_WER);
    }

    #[test]
    fn transcript_wer_rejects_different_wording() {
        let wer = transcript_word_error_rate(
            "THE SECRET GARDEN WAS FIRST PUBLISHED IN NINETEEN ELEVEN",
            "This recording is from LibriVox and has nothing to do with that sentence.",
        );
        assert!(wer > DEFAULT_WHISPER_TRANSCRIPT_MAX_WER);
    }

    #[test]
    fn interpretation_omit_warning_shows_conflicting_whisper_transcripts() {
        let warning = interpretation_omit_warning(
            "19-198-0001",
            "Whisper transcript diverged from source transcript",
            Some("THE SECRET GARDEN WAS FIRST PUBLISHED IN NINETEEN ELEVEN"),
            Some("This recording is from LibriVox."),
            Some(0.82),
            Some(0.25),
        );
        assert!(warning.contains("WER 0.82 > 0.25"));
        assert!(warning.contains(
            "source transcript: THE SECRET GARDEN WAS FIRST PUBLISHED IN NINETEEN ELEVEN"
        ));
        assert!(warning.contains("whisper transcript: This recording is from LibriVox."));
    }

    #[test]
    fn test_detect_task() {
        assert_eq!(detect_task("farkle"), Task::G2P);
        assert_eq!(detect_task("farkle's"), Task::G2P);
        assert_eq!(detect_task("fark-le"), Task::G2P);
        assert_eq!(detect_task("ˈfɑɹ.kəl"), Task::P2G);
        assert_eq!(detect_task("kæt"), Task::P2G); // non-ASCII chars
        assert_eq!(detect_task(""), Task::P2G);
    }

    #[test]
    fn frequency_repeat_count_uses_bounded_linear_rarity() {
        assert_eq!(frequency_repeat_count(0.0, 8, 50_000.0), 8);
        assert_eq!(frequency_repeat_count(23.0, 8, 50_000.0), 8);
        assert_eq!(frequency_repeat_count(25_000.0, 8, 50_000.0), 5);
        assert_eq!(frequency_repeat_count(50_000.0, 8, 50_000.0), 1);
        assert_eq!(frequency_repeat_count(f32::NAN, 8, 50_000.0), 1);
    }

    #[test]
    fn frequency_weighted_training_expands_common_words() {
        let lexemes = vec![
            Lexeme {
                base_word: "the".to_string(),
                phonemes: "ðə".to_string(),
                rarity: 0.0,
            },
            Lexeme {
                base_word: "tailword".to_string(),
                phonemes: "teɪl.wɝd".to_string(),
                rarity: 50_000.0,
            },
        ];

        let expanded = expand_frequency_weighted_training(&lexemes, 8, 50_000.0);
        assert_eq!(expanded.len(), 9);
        assert_eq!(
            expanded
                .iter()
                .filter(|lexeme| lexeme.base_word == "the")
                .count(),
            8
        );
        assert_eq!(
            expanded
                .iter()
                .filter(|lexeme| lexeme.base_word == "tailword")
                .count(),
            1
        );
    }

    #[test]
    fn wiktionary_sight_word_training_adds_matching_rows() {
        let mut train_rows = vec![tongues_wiktionary::TrainingExample {
            task: tongues_wiktionary::WiktionaryTask::OrthographyToPhonology,
            lang: Some("eng".to_string()),
            notation: Some("phonetic".to_string()),
            accent: None,
            input: "<task:orthography_to_phonology> <lang:eng> <repr:phones> said".to_string(),
            output: "sɛd".to_string(),
            source: "test".to_string(),
        }];
        let valid_rows = vec![
            tongues_wiktionary::TrainingExample {
                task: tongues_wiktionary::WiktionaryTask::PhonologyToOrthography,
                lang: Some("eng".to_string()),
                notation: Some("phonetic".to_string()),
                accent: None,
                input: "<task:phonology_to_orthography> <lang:eng> <repr:phones> wʌn".to_string(),
                output: "one".to_string(),
                source: "test".to_string(),
            },
            tongues_wiktionary::TrainingExample {
                task: tongues_wiktionary::WiktionaryTask::OrthographyToPhonology,
                lang: Some("deu".to_string()),
                notation: Some("phonetic".to_string()),
                accent: None,
                input: "<task:orthography_to_phonology> <lang:deu> <repr:phones> die".to_string(),
                output: "diː".to_string(),
                source: "test".to_string(),
            },
        ];

        let added = add_wiktionary_sight_word_training_examples(&mut train_rows, [&valid_rows[..]]);

        assert_eq!(added, SIGHT_WORD_TRAINING_REPEATS * 2);
        assert_eq!(
            train_rows
                .iter()
                .filter(|row| row.input.ends_with(" said"))
                .count(),
            SIGHT_WORD_TRAINING_REPEATS + 1
        );
        assert_eq!(
            train_rows.iter().filter(|row| row.output == "one").count(),
            SIGHT_WORD_TRAINING_REPEATS
        );
        assert!(!train_rows
            .iter()
            .any(|row| row.lang.as_deref() == Some("deu")));
    }

    #[test]
    fn wiktionary_frequency_weighting_expands_english_rows_by_openepd_rarity() {
        let train_rows = vec![
            tongues_wiktionary::TrainingExample {
                task: tongues_wiktionary::WiktionaryTask::OrthographyToPhonology,
                lang: Some("eng".to_string()),
                notation: Some("phonetic".to_string()),
                accent: None,
                input: "<task:orthography_to_phonology> <lang:eng> <repr:phones> the".to_string(),
                output: "ðə".to_string(),
                source: "test".to_string(),
            },
            tongues_wiktionary::TrainingExample {
                task: tongues_wiktionary::WiktionaryTask::OrthographyToPhonology,
                lang: Some("deu".to_string()),
                notation: Some("phonetic".to_string()),
                accent: None,
                input: "<task:orthography_to_phonology> <lang:deu> <repr:phones> die".to_string(),
                output: "diː".to_string(),
                source: "test".to_string(),
            },
        ];
        let rarity_by_word = std::collections::BTreeMap::from([
            ("the".to_string(), 0.0_f32),
            ("die".to_string(), 0.0_f32),
        ]);

        let (expanded, matched_rows, added_rows) =
            expand_wiktionary_frequency_weighted_training_examples(
                &train_rows,
                &rarity_by_word,
                8,
                50_000.0,
            );

        assert_eq!(matched_rows, 1);
        assert_eq!(added_rows, 7);
        assert_eq!(expanded.len(), 9);
        assert_eq!(
            expanded
                .iter()
                .filter(|row| row.input.ends_with(" the"))
                .count(),
            8
        );
        assert_eq!(
            expanded
                .iter()
                .filter(|row| row.input.ends_with(" die"))
                .count(),
            1
        );
    }

    #[test]
    fn openepd_prepare_conversion_includes_rarity_for_have() {
        let entry = OpenEpdEntry {
            rarity: 23.0,
            ipa: std::collections::BTreeMap::from([("misaki_gold".to_string(), "hæv".to_string())]),
        };

        let have = prepare_lexeme_from_openepd_entry("have".to_string(), entry)
            .expect("have entry should prepare");

        assert_eq!(have.base_word, "have");
        assert_eq!(have.phonemes, "hæv");
        assert_eq!(have.rarity, 23.0);
    }

    #[test]
    fn openepd_prepare_corrects_unrotten_gold_transcription() {
        let entry = OpenEpdEntry {
            rarity: 271886.0,
            ipa: std::collections::BTreeMap::from([(
                "misaki_silver".to_string(),
                "ʌnɹˈɑʔn".to_string(),
            )]),
        };

        let unrotten = prepare_lexeme_from_openepd_entry("unrotten".to_string(), entry)
            .expect("unrotten entry should prepare");

        assert_eq!(unrotten.base_word, "unrotten");
        assert_eq!(unrotten.phonemes, "ʌnˈɹɑ.tən");
        assert_eq!(unrotten.rarity, 271886.0);
    }

    #[test]
    fn cli_accepts_g2p2g_family_commands() {
        let cli = Cli::try_parse_from([
            "tongues",
            "g2p2g",
            "infer",
            "--model",
            "models/g2p2g/openepd-v0",
            "farkle",
        ])
        .expect("g2p2g infer should parse");

        assert!(matches!(
            cli.command,
            Some(Commands::G2p2g {
                command: G2p2gCommands::Infer { .. }
            })
        ));
    }

    #[test]
    fn cli_accepts_sentence_parser_commands() {
        let cli = Cli::try_parse_from([
            "tongues",
            "sentence-parser",
            "parse",
            "--model",
            "models/sentence-parser/v0",
            "The quick brown fox jumps.",
        ])
        .expect("sentence parser parse should parse");

        assert!(matches!(
            cli.command,
            Some(Commands::SentenceParser {
                command: SentenceParserCommands::Parse { .. }
            })
        ));

        let cli = Cli::try_parse_from(["tongues", "sentence-parser", "stream"])
            .expect("sentence parser stream should parse");

        assert!(matches!(
            cli.command,
            Some(Commands::SentenceParser {
                command: SentenceParserCommands::Stream { .. }
            })
        ));
    }

    #[test]
    fn emitted_sentence_consumption_preserves_following_cursor() {
        assert_eq!(
            cursor_after_emitted_sentence("First sentence. Second starts", "First sentence."),
            " Second starts"
        );
        assert_eq!(
            cursor_after_emitted_sentence("First sentence. Second starts", "first sentence."),
            " Second starts"
        );
        assert_eq!(
            cursor_after_emitted_sentence("Unexpected output. Second starts", "Other output."),
            " Second starts"
        );
    }

    #[test]
    fn oversize_sentence_parser_fallback_emits_first_terminal_prefix() {
        let mut cursor = "Long sentence. Next sentence.".to_string();
        let mut previous = String::new();
        let mut output = Vec::new();

        let emitted =
            emit_oversize_sentence_parser_prefix(&mut cursor, &mut previous, &mut output).unwrap();

        assert!(emitted);
        assert_eq!(previous, "Long sentence.");
        assert_eq!(cursor, " Next sentence.");
        assert_eq!(String::from_utf8(output).unwrap(), "Long sentence.\n");
    }

    #[test]
    fn sentence_parser_stream_emits_completed_sentences_from_continuous_input() {
        let mut output = Vec::new();

        run_sentence_parser_stream_io(
            "This is a test. Testing test.\nA judge denied. A living memorial".as_bytes(),
            &mut output,
        )
        .unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "This is a test.\nTesting test.\nA judge denied.\nA living memorial\n"
        );
    }

    #[test]
    fn sentence_parser_stream_does_not_join_paragraph_fragments_to_later_sentences() {
        let mut output = Vec::new();

        run_sentence_parser_stream_io(
            "A judge denied. A living memorial\n\n\nA jduge denied.\n".as_bytes(),
            &mut output,
        )
        .unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "A judge denied.\nA living memorial\nA jduge denied.\n"
        );
    }

    #[test]
    fn sentence_parser_stream_keeps_common_abbreviations_with_sentence() {
        let mut output = Vec::new();

        run_sentence_parser_stream_io(
            "Dr. Lanyon met Henry at Mt. Vernon. Next.".as_bytes(),
            &mut output,
        )
        .unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "Dr. Lanyon met Henry at Mt. Vernon.\nNext.\n"
        );
    }

    #[test]
    fn sentence_parser_stream_preserves_utf8_across_chunks() {
        let mut pending = Vec::new();
        let mut output = String::new();
        let bytes = "café. ".as_bytes();

        append_utf8_chunk(&mut pending, &bytes[..4], &mut output);
        append_utf8_chunk(&mut pending, &bytes[4..], &mut output);

        assert_eq!(output, "café. ");
        assert!(pending.is_empty());
    }

    #[test]
    fn cli_accepts_wiktionary_family_commands() {
        let cli = Cli::try_parse_from([
            "tongues",
            "wiktionary",
            "prepare",
            "--out",
            "datasets/wiktionary/enwiktionary-2026-06-01-v0",
        ])
        .expect("wiktionary prepare should parse");

        assert!(matches!(
            cli.command,
            Some(Commands::Wiktionary {
                command: WiktionaryCommands::Prepare { .. }
            })
        ));

        let cli = Cli::try_parse_from([
            "tongues",
            "wiktionary",
            "infer",
            "--model",
            "models/wiktionary/enwiktionary-2026-06-01-v0-phones",
            "--task",
            "orthography-to-phones",
            "hello",
        ])
        .expect("wiktionary infer should parse");

        assert!(matches!(
            cli.command,
            Some(Commands::Wiktionary {
                command: WiktionaryCommands::Infer { .. }
            })
        ));

        let cli = Cli::try_parse_from(["tongues", "wiktionary", "train", "--sight-words"])
            .expect("wiktionary train --sight-words should parse");

        assert!(matches!(
            cli.command,
            Some(Commands::Wiktionary {
                command: WiktionaryCommands::Train {
                    sight_words: true,
                    ..
                }
            })
        ));
    }

    #[test]
    fn discrepancies_default_to_general_american_wiktionary_variety() {
        let cli = Cli::try_parse_from(["tongues", "discrepencies"])
            .expect("misspelled discrepancies alias should parse");

        match cli.command {
            Some(Commands::Discrepancies {
                wiktionary_variety, ..
            }) => assert_eq!(wiktionary_variety, "en-US.GenAm"),
            other => panic!("expected discrepancies command, got {other:?}"),
        }
    }

    #[test]
    fn cli_accepts_family_clean_commands() {
        let cli = Cli::try_parse_from(["tongues", "g2p2g", "clean", "--data"])
            .expect("g2p2g clean should parse");
        assert!(matches!(
            cli.command,
            Some(Commands::G2p2g {
                command: G2p2gCommands::Clean(_)
            })
        ));

        let cli = Cli::try_parse_from(["tongues", "sentence-parser", "clean", "--all"])
            .expect("sentence-parser clean should parse");
        assert!(matches!(
            cli.command,
            Some(Commands::SentenceParser {
                command: SentenceParserCommands::Clean(_)
            })
        ));

        let cli = Cli::try_parse_from(["tongues", "wiktionary", "clean"])
            .expect("wiktionary clean should parse");
        assert!(matches!(
            cli.command,
            Some(Commands::Wiktionary {
                command: WiktionaryCommands::Clean(_)
            })
        ));
    }

    #[test]
    fn cli_keeps_legacy_predict_alias() {
        let cli = Cli::try_parse_from(["tongues", "infer", "farkle"])
            .expect("legacy infer alias should parse");

        assert!(matches!(cli.command, Some(Commands::Predict { .. })));
    }
}
