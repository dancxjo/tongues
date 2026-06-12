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

pub mod models;
mod piper;
mod speak;

use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use burn_cuda::{Cuda, CudaDevice};

use speech::data::notation::openepd::normalize_openepd_ipa;
use tongues_core::{Vocab, BOS_ID, EOS_ID, UNK_ID};
use tongues_data::{Lexeme, Seq2SeqExample, Task};
use tongues_g2p2g::{
    eval_report, load_model, predict, train, train_seq2seq_examples, ModelConfig, Seq2SeqModel,
    TrainConfig,
};
use tongues_neural::{write_manifest, ModelArtifactManifest};
use tongues_speech_manifold::{
    SpeechManifoldConfig, SpeechManifoldTask, SpeechManifoldTrainConfig,
};

// ── Backend aliases ────────────────────────────────────────────────────────

type CpuInferBackend = NdArray<f32>;
type CpuTrainBackend = Autodiff<CpuInferBackend>;

type CudaInferBackend = Cuda<f32, i32>;
type CudaTrainBackend = Autodiff<CudaInferBackend>;

const DEFAULT_WIKTIONARY_DATASET_ID: &str = "enwiktionary-2026-06-01-v0";
const DEFAULT_WIKTIONARY_DATA_DIR: &str = "datasets/wiktionary/enwiktionary-2026-06-01-v0";
const DEFAULT_WIKTIONARY_MODEL_DIR: &str = "models/wiktionary/enwiktionary-2026-06-01-v0-phones";
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

    /// Prepare, train, and probe the shared multimodal speech manifold
    #[command(name = "speech-manifold")]
    SpeechManifold {
        #[command(subcommand)]
        command: SpeechManifoldCommands,
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
}

#[derive(Subcommand, Debug)]
enum G2p2gCommands {
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

        /// Output directory for the model
        #[arg(long, default_value = "models/sentence-parser/v0")]
        out: PathBuf,

        /// Prepare data before training
        #[arg(long)]
        prepare: bool,

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
}

#[derive(Subcommand, Debug)]
enum SpeechManifoldCommands {
    /// Prepare an OpenEPD-derived multimodal dataset
    Prepare {
        /// TOML config file for the speech-manifold pipeline
        #[arg(long, default_value = "configs/speech-manifold/default.toml")]
        config: PathBuf,

        /// Output directory for prepared data
        #[arg(long, default_value = "datasets/speech-manifold/openepd-synth-v0")]
        out: PathBuf,
    },

    /// Train the speech-manifold model
    Train {
        /// TOML config file for the speech-manifold pipeline
        #[arg(long, default_value = "configs/speech-manifold/default.toml")]
        config: PathBuf,

        /// Prepared data directory
        #[arg(long, default_value = "datasets/speech-manifold/openepd-synth-v0")]
        data: PathBuf,

        /// Output directory for the model
        #[arg(long, default_value = "models/speech-manifold/openepd-synth-v0")]
        out: PathBuf,

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

    /// Evaluate a speech-manifold model
    Eval {
        /// Directory containing the model
        #[arg(long, default_value = "models/speech-manifold/openepd-synth-v0")]
        model: PathBuf,

        /// Prepared data directory
        #[arg(long, default_value = "datasets/speech-manifold/openepd-synth-v0")]
        data: PathBuf,

        /// Split to evaluate on: train, valid, or test
        #[arg(long, default_value = "test")]
        split: String,

        /// Explicit task to evaluate
        #[arg(long, default_value = "spelling-to-ipa")]
        task: String,
    },

    /// Run inference for one explicit speech-manifold task
    Infer {
        /// Directory containing the model
        #[arg(long, default_value = "models/speech-manifold/openepd-synth-v0")]
        model: PathBuf,

        /// Explicit task, e.g. spelling-to-ipa
        #[arg(long)]
        task: String,

        /// Input sequence
        input: String,
    },

    /// Summarize split modalities and provenance
    Probe {
        /// Directory containing the model
        #[arg(long, default_value = "models/speech-manifold/openepd-synth-v0")]
        model: PathBuf,

        /// Prepared data directory
        #[arg(long, default_value = "datasets/speech-manifold/openepd-synth-v0")]
        data: PathBuf,

        /// Split to probe: train, valid, or test
        #[arg(long, default_value = "test")]
        split: String,
    },
}

#[derive(Subcommand, Debug)]
enum WiktionaryCommands {
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

        /// Wiktionary task mix: orthography-to-phonemes, orthography-to-phones, phonetic-realization, lang, or all.
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

        /// Wiktionary task: orthography-to-phonemes, orthography-to-phones, phonemes-to-orthography, phones-to-orthography, phonetic-realization, normalize, or a language guessing task
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum WiktionaryNotationArg {
    /// Train from both phonemes.jsonl and phones.jsonl.
    All,
    /// Train from bracket-delimited phonetic rows in phones.jsonl.
    Phones,
    /// Train from slash-delimited phonemic rows in phonemes.jsonl.
    Phonemes,
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

fn is_cuda_available() -> bool {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(|| {
        let device = CudaDevice::default();
        type B = Cuda<f32, i32>;
        let _tensor = burn::tensor::Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &device);
    });
    std::panic::set_hook(default_hook);
    result.is_ok()
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
    let device_arg = if cli.cpu {
        DeviceArg::Cpu
    } else if is_cuda_available() {
        DeviceArg::Cuda
    } else {
        // Only warn for commands that actually run model computations on the device
        if command_needs_device(&command) && output_mode.verbose() {
            println!("Warning: CUDA is not available. Falling back to CPU.");
        }
        DeviceArg::Cpu
    };

    match command {
        Commands::G2p2g { command } => run_g2p2g_command(command, device_arg, output_mode),
        Commands::SentenceParser { command } => run_sentence_parser_command(command, device_arg),
        Commands::SpeechManifold { command } => {
            run_speech_manifold_command(command, device_arg, output_mode)
        }
        Commands::Wiktionary { command } => {
            run_wiktionary_command(command, device_arg, output_mode)
        }
        Commands::FetchCmudict { out } => cmd_fetch_cmudict(&out),
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
        Commands::SpeechManifold { command } => matches!(
            command,
            SpeechManifoldCommands::Train { .. }
                | SpeechManifoldCommands::Eval { .. }
                | SpeechManifoldCommands::Infer { .. }
        ),
        Commands::SentenceParser { command } => matches!(
            command,
            SentenceParserCommands::Train { .. } | SentenceParserCommands::Infer { .. }
        ),
        Commands::Wiktionary { command } => matches!(command, WiktionaryCommands::Train { .. }),
        Commands::Train { .. }
        | Commands::Eval { .. }
        | Commands::Refine { .. }
        | Commands::Predict { .. }
        | Commands::Repl { .. } => true,
        _ => false,
    }
}

fn command_defaults_to_quiet(command: &Commands) -> bool {
    match command {
        Commands::G2p2g {
            command: G2p2gCommands::Infer { .. },
        }
        | Commands::SpeechManifold {
            command: SpeechManifoldCommands::Infer { .. },
        }
        | Commands::SentenceParser {
            command: SentenceParserCommands::Infer { .. },
        }
        | Commands::Wiktionary {
            command: WiktionaryCommands::Infer { .. },
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
    pb
}

fn finish_status(pb: indicatif::ProgressBar, message: impl AsRef<str>) {
    pb.finish_and_clear();
    if !quiet_output() {
        println!("{}", message.as_ref());
    }
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
        format!("{bytes} B")
    }
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
            pie_roots,
        } => format!(
            "Parsing dump: {pages} pages, {patterns} patterns, {phonemes} phonemes, {phones} phones, {pie_roots} PIE roots"
        ),
        tongues_wiktionary::PrepareProgress::Expand {
            rows,
            examples,
            path,
        } => match path {
            Some(path) => format!("Expanded {rows} rows into {examples} examples -> {path}"),
            None => format!("Expanded {rows} rows into {examples} examples"),
        },
        tongues_wiktionary::PrepareProgress::Write { path, rows } => {
            format!("Wrote {rows} rows to {path}")
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

#[derive(Debug, Clone, Default, Deserialize)]
struct SpeechManifoldFileConfig {
    dataset_id: Option<String>,
    train_frac: Option<f64>,
    valid_frac: Option<f64>,
    seed: Option<u64>,
    tasks: Option<Vec<SpeechManifoldTask>>,
    synthesis_backends: Option<Vec<String>>,
    allow_placeholder_acoustics: Option<bool>,
    max_examples: Option<usize>,
    max_audio_examples_per_backend: Option<usize>,
    max_espeak_examples: Option<usize>,
    max_google_translate_examples: Option<usize>,
    max_wiktionary_audio_examples: Option<usize>,
    max_styletts2_examples: Option<usize>,
    max_piper_examples: Option<usize>,
    max_anyspeak_examples: Option<usize>,
    max_mock_examples: Option<usize>,
    max_wikimedia_commons_examples: Option<usize>,
    max_wikimedia_commons_lookup_attempts: Option<usize>,
    anyspeak_dir: Option<String>,
    anyspeak_python: Option<String>,
    anyspeak_voice_tags: Option<Vec<String>>,
    include_reference_uris: Option<bool>,
    external_audio_manifests: Option<Vec<String>>,
    espeak_voices: Option<Vec<String>>,
    google_translate_speeds: Option<Vec<f32>>,
    train: Option<SpeechManifoldFileTrainConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct SpeechManifoldFileTrainConfig {
    learning_rate: Option<f64>,
    weight_decay: Option<f32>,
    dropout: Option<f64>,
    epochs: Option<usize>,
    patience: Option<usize>,
    batch_size: Option<usize>,
    seed: Option<u64>,
    tasks: Option<Vec<SpeechManifoldTask>>,
}

fn read_g2p2g_config(path: &Path) -> Result<G2p2gFileConfig> {
    if !path.exists() {
        return Ok(G2p2gFileConfig::default());
    }
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn read_speech_manifold_file_config(path: &Path) -> Result<SpeechManifoldFileConfig> {
    if !path.exists() {
        return Ok(SpeechManifoldFileConfig::default());
    }
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn speech_manifold_prepare_config(path: &Path) -> Result<SpeechManifoldConfig> {
    let file = read_speech_manifold_file_config(path)?;
    let default = SpeechManifoldConfig::default();
    Ok(SpeechManifoldConfig {
        dataset_id: file.dataset_id.unwrap_or(default.dataset_id),
        train_frac: file.train_frac.unwrap_or(default.train_frac),
        valid_frac: file.valid_frac.unwrap_or(default.valid_frac),
        seed: file.seed.unwrap_or(default.seed),
        tasks: file.tasks.unwrap_or(default.tasks),
        synthesis_backends: file
            .synthesis_backends
            .unwrap_or(default.synthesis_backends),
        allow_placeholder_acoustics: file
            .allow_placeholder_acoustics
            .unwrap_or(default.allow_placeholder_acoustics),
        max_examples: file.max_examples.or(default.max_examples),
        max_audio_examples_per_backend: file
            .max_audio_examples_per_backend
            .unwrap_or(default.max_audio_examples_per_backend),
        max_espeak_examples: file
            .max_espeak_examples
            .unwrap_or(default.max_espeak_examples),
        max_google_translate_examples: file
            .max_google_translate_examples
            .unwrap_or(default.max_google_translate_examples),
        max_wiktionary_audio_examples: file
            .max_wiktionary_audio_examples
            .unwrap_or(default.max_wiktionary_audio_examples),
        max_styletts2_examples: file
            .max_styletts2_examples
            .unwrap_or(default.max_styletts2_examples),
        max_piper_examples: file
            .max_piper_examples
            .unwrap_or(default.max_piper_examples),
        max_anyspeak_examples: file
            .max_anyspeak_examples
            .unwrap_or(default.max_anyspeak_examples),
        max_mock_examples: file.max_mock_examples.unwrap_or(default.max_mock_examples),
        max_wikimedia_commons_examples: file
            .max_wikimedia_commons_examples
            .unwrap_or(default.max_wikimedia_commons_examples),
        max_wikimedia_commons_lookup_attempts: file
            .max_wikimedia_commons_lookup_attempts
            .unwrap_or(default.max_wikimedia_commons_lookup_attempts),
        anyspeak_dir: file
            .anyspeak_dir
            .filter(|value| !value.trim().is_empty())
            .or(default.anyspeak_dir),
        anyspeak_python: file.anyspeak_python.unwrap_or(default.anyspeak_python),
        anyspeak_voice_tags: file
            .anyspeak_voice_tags
            .unwrap_or(default.anyspeak_voice_tags),
        include_reference_uris: file
            .include_reference_uris
            .unwrap_or(default.include_reference_uris),
        external_audio_manifests: file
            .external_audio_manifests
            .unwrap_or(default.external_audio_manifests),
        espeak_voices: file.espeak_voices.unwrap_or(default.espeak_voices),
        google_translate_speeds: file
            .google_translate_speeds
            .unwrap_or(default.google_translate_speeds),
    })
}

fn speech_manifold_train_config(
    path: &Path,
    epochs: Option<usize>,
    batch_size: Option<usize>,
    seed: Option<u64>,
) -> Result<SpeechManifoldTrainConfig> {
    let prepare = speech_manifold_prepare_config(path)?;
    let file = read_speech_manifold_file_config(path)?;
    let train = file.train.unwrap_or_default();
    let default = SpeechManifoldTrainConfig::default();
    Ok(SpeechManifoldTrainConfig {
        learning_rate: train.learning_rate.unwrap_or(default.learning_rate),
        weight_decay: train.weight_decay.unwrap_or(default.weight_decay),
        dropout: train.dropout.unwrap_or(default.dropout),
        batch_size: batch_size
            .or(train.batch_size)
            .unwrap_or(default.batch_size),
        epochs: epochs.or(train.epochs).unwrap_or(default.epochs),
        early_stopping_patience: train.patience.unwrap_or(default.early_stopping_patience),
        seed: seed.or(train.seed).unwrap_or(default.seed),
        tasks: train.tasks.unwrap_or(prepare.tasks),
        allow_placeholder_acoustics: prepare.allow_placeholder_acoustics,
    })
}

fn run_g2p2g_command(
    command: G2p2gCommands,
    device_arg: DeviceArg,
    output_mode: OutputMode,
) -> Result<()> {
    match command {
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
        } => {
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
            format!("Discovered {files} sentence-parser source files")
        }
        tongues_sentence_parser::PrepareProgress::Detect {
            path,
            files_done,
            files_total,
            sentences,
            naive_discrepancies,
        } => format!(
            "Detected {sentences} sentences and {naive_discrepancies} repairs ({files_done}/{files_total}: {path})"
        ),
        tongues_sentence_parser::PrepareProgress::Build {
            sentences,
            examples,
        } => format!("Built {examples} boundary examples from {sentences} sentences"),
        tongues_sentence_parser::PrepareProgress::Write { path, rows } => {
            format!("Wrote {rows} rows to {path}")
        }
    }
}

fn run_sentence_parser_command(
    command: SentenceParserCommands,
    device_arg: DeviceArg,
) -> Result<()> {
    match command {
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
            let report =
                tongues_sentence_parser::prepare_dataset_with_progress(&out, &config, {
                    let pb = pb.clone();
                    move |progress| {
                        pb.set_message(sentence_parser_prepare_progress_message(progress));
                    }
                })?;
            finish_status(
                pb,
                format!(
                    "Prepared sentence-parser dataset at {}: {} train / {} valid / {} test examples",
                    out.display(),
                    report.train_examples,
                    report.valid_examples,
                    report.test_examples
                ),
            );
            println!(
                "Prepared sentence parser dataset at {}: {} train / {} valid / {} test examples from {} sentences in {} files",
                out.display(),
                report.train_examples,
                report.valid_examples,
                report.test_examples,
                report.detected_sentences,
                report.source_files
            );
            if report.naive_discrepancy_examples > 0 {
                println!(
                    "  included {} naive-vs-seams correction rows",
                    report.naive_discrepancy_examples
                );
            }
            Ok(())
        }
        SentenceParserCommands::Train {
            config,
            data,
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
        } => {
            if prepare
                || !data.join("vocab.json").exists()
                || !data.join("train.jsonl").exists()
                || !data.join("valid.jsonl").exists()
            {
                let config_data = read_sentence_parser_config(&config)?;
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
                        "Prepared sentence-parser dataset at {}: {} train / {} valid / {} test examples",
                        data.display(),
                        report.train_examples,
                        report.valid_examples,
                        report.test_examples
                    ),
                );
                println!(
                    "Prepared sentence parser dataset at {}: {} train / {} valid / {} test examples",
                    data.display(),
                    report.train_examples,
                    report.valid_examples,
                    report.test_examples
                );
                if report.naive_discrepancy_examples > 0 {
                    println!(
                        "  included {} naive-vs-seams correction rows",
                        report.naive_discrepancy_examples
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
        "sentence-parser train split is empty"
    );
    anyhow::ensure!(
        !valid_rows.is_empty(),
        "sentence-parser valid split is empty"
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
        "  training_set={} examples={} train / {} valid vocab={} lr={} wd={} dropout={} epochs={} patience={} batch_size={}",
        sentence_parser_training_set_label(training_set),
        train_examples.len(),
        valid_examples.len(),
        vocab.size(),
        learning_rate,
        weight_decay,
        dropout,
        epochs,
        patience,
        batch_size
    );
    println!("  train_state: {}", out.join("train_state.json").display());
    println!(
        "  epoch checkpoints: {}",
        out.join("model-epoch-N.bin").display()
    );
    println!("  best model: {}", model_path.with_extension("bin").display());

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

fn run_wiktionary_command(
    command: WiktionaryCommands,
    device_arg: DeviceArg,
    output_mode: OutputMode,
) -> Result<()> {
    match command {
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
                "Parsed {} phonemes, {} phones, and {} PIE roots into train/valid/test examples: {}/{}/{}",
                report.parsed_phonemes,
                report.parsed_phones,
                report.parsed_pie_roots,
                report.train_examples,
                report.valid_examples,
                report.test_examples
            );
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
            learning_rate,
            weight_decay,
            dropout,
            batch_size,
            epochs,
            patience,
            seed,
        } => {
            let mut config = tongues_wiktionary::read_config(&config)?;
            if let Some(dump) = dump {
                config.dump_path = Some(dump.display().to_string());
            }
            apply_wiktionary_language_override(&mut config, langs);
            let data = effective_wiktionary_data_path(data, &config);
            let out = effective_wiktionary_model_path(out, &config);
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
                        report.train_examples,
                        report.valid_examples,
                        report.test_examples,
                        report.dump_path
                    ),
                );
            }
            let task = task
                .as_deref()
                .unwrap_or(config.train_task.as_str())
                .to_string();
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
            learning_rate,
            weight_decay,
            dropout,
            batch_size,
            epochs,
            patience,
            seed,
            device_arg,
        );
    }

    let notations = resolve_wiktionary_train_notations(notation, config)?;
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
            entries.len(),
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
            examples.len()
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
                skipped_train, skipped_valid
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
            train_examples.len(),
            valid_examples.len(),
            vocab.size()
        ),
    );

    println!(
        "Loaded {} {} rows -> {} train / {} valid examples for task={}",
        entries.len(),
        wiktionary_notation_label(&notations),
        train_examples.len(),
        valid_examples.len(),
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
    learning_rate: f64,
    weight_decay: f32,
    dropout: f64,
    batch_size: usize,
    epochs: usize,
    patience: usize,
    seed: u64,
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
    finish_status(
        pb,
        format!(
            "Loaded {} train / {} valid prepared rows",
            train_rows_raw.len(),
            valid_rows_raw.len()
        ),
    );

    let pb = status_spinner(format!("Filtering prepared rows for task={task}"));
    let train_rows = filter_wiktionary_examples(train_rows_raw, task)?;
    let valid_rows = filter_wiktionary_examples(valid_rows_raw, task)?;
    finish_status(
        pb,
        format!(
            "Selected {} train / {} valid rows for task={task}",
            train_rows.len(),
            valid_rows.len()
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

    let pb = status_spinner("Building Wiktionary vocabulary and encoding seq2seq examples");
    let vocab = build_wiktionary_vocab(&train_rows, &valid_rows);
    let train_examples = wiktionary_seq2seq_examples(&train_rows, &vocab);
    let valid_examples = wiktionary_seq2seq_examples(&valid_rows, &vocab);
    finish_status(
        pb,
        format!(
            "Encoded {} train / {} valid examples with vocab size {}",
            train_examples.len(),
            valid_examples.len(),
            vocab.size()
        ),
    );

    println!(
        "Loaded prepared rows -> {} train / {} valid examples for task={}",
        train_examples.len(),
        valid_examples.len(),
        task
    );

    write_and_train_wiktionary_seq2seq(
        data,
        out,
        config,
        &format!("pie-etymology:{task}"),
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
        "  lr={} wd={} dropout={} epochs={} patience={} batch_size={}",
        learning_rate, weight_decay, dropout, epochs, patience, batch_size
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
        anyhow::bail!("Invalid Wiktionary task. Supported: orthography-to-phonemes, orthography-to-phones, phonemes-to-orthography, phones-to-orthography, phonetic-realization, etymology-translation, normalize, align, lang, all");
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
    let rows = train.iter().chain(valid.iter());
    let inputs = rows
        .clone()
        .map(|example| wiktionary_source_text(example))
        .collect::<Vec<_>>();
    let outputs = rows
        .map(|example| example.output.clone())
        .collect::<Vec<_>>();
    Vocab::build(&inputs, &outputs, &[])
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
            "Invalid Wiktionary inference task. Supported: orthography-to-phonemes, orthography-to-phones, phonemes-to-orthography, phones-to-orthography, phonetic-realization, normalize, guess-lang-from-orthography, guess-lang-from-phonology, guess-lang-from-orthography-and-phonology"
        ),
    };
    Ok(source)
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

fn run_speech_manifold_command(
    command: SpeechManifoldCommands,
    device_arg: DeviceArg,
    output_mode: OutputMode,
) -> Result<()> {
    match command {
        SpeechManifoldCommands::Prepare { config, out } => {
            let config = speech_manifold_prepare_config(&config)?;
            tongues_speech_manifold::prepare_dataset(&out, &config)?;
            println!("Speech-manifold dataset written to {}", out.display());
            Ok(())
        }
        SpeechManifoldCommands::Train {
            config,
            data,
            out,
            epochs,
            batch_size,
            seed,
        } => {
            if !data.join("vocab.json").exists()
                || !data.join("train.jsonl").exists()
                || !data.join("valid.jsonl").exists()
            {
                let prepare = speech_manifold_prepare_config(&config)?;
                let pb = status_spinner(format!(
                    "Training data missing; preparing speech-manifold dataset at {}",
                    data.display()
                ));
                tongues_speech_manifold::prepare_dataset(&data, &prepare)?;
                finish_status(
                    pb,
                    format!("Prepared speech-manifold dataset at {}", data.display()),
                );
            }
            let train_config = speech_manifold_train_config(&config, epochs, batch_size, seed)?;
            cmd_speech_manifold_train(&data, &out, &train_config, device_arg)
        }
        SpeechManifoldCommands::Eval {
            model,
            data,
            split,
            task,
        } => {
            let task = parse_speech_manifold_task(&task)?;
            cmd_speech_manifold_eval(&model, &data, &split, task, device_arg)
        }
        SpeechManifoldCommands::Infer { model, task, input } => {
            let task = parse_speech_manifold_task(&task)?;
            cmd_speech_manifold_infer(&model, task, &input, device_arg, output_mode)
        }
        SpeechManifoldCommands::Probe { model, data, split } => {
            let _manifest =
                tongues_neural::read_manifest(&model.join(tongues_neural::ARTIFACT_MANIFEST_FILE))
                    .with_context(|| format!("reading manifest in {}", model.display()))?;
            let train_config: SpeechManifoldTrainConfig =
                read_json_file(&model.join("train_config.json"))?;
            let examples =
                tongues_speech_manifold::read_examples(&data.join(format!("{split}.jsonl")))?;
            let report = tongues_speech_manifold::probe(&split, &examples, &train_config.tasks);
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
    }
}

fn parse_speech_manifold_task(task: &str) -> Result<SpeechManifoldTask> {
    SpeechManifoldTask::parse(task).ok_or_else(|| {
        anyhow::anyhow!(
            "invalid speech-manifold task `{}`; supported: spelling-to-ipa, ipa-to-spelling, ipa-to-phones, stress, syllables, acoustic-to-ipa, ipa-to-acoustic",
            task
        )
    })
}

fn cmd_speech_manifold_train(
    data: &Path,
    out: &Path,
    train_config: &SpeechManifoldTrainConfig,
    device_arg: DeviceArg,
) -> Result<()> {
    let pb = status_spinner(format!(
        "Loading speech-manifold data from {}",
        data.display()
    ));
    let vocab: Vocab = read_json_file(&data.join("vocab.json"))?;
    let train_examples = tongues_speech_manifold::read_examples(&data.join("train.jsonl"))?;
    let valid_examples = tongues_speech_manifold::read_examples(&data.join("valid.jsonl"))?;
    finish_status(
        pb,
        format!(
            "Loaded {} train / {} valid examples with vocab size {}",
            train_examples.len(),
            valid_examples.len(),
            vocab.size()
        ),
    );
    fs::create_dir_all(out).context("creating speech-manifold model directory")?;
    let model_config = ModelConfig::new(vocab.size()).with_dropout(train_config.dropout);
    let pb = status_spinner(format!(
        "Writing speech-manifold artifact files to {}",
        out.display()
    ));
    tongues_speech_manifold::save_artifact_files(out, data, &model_config, train_config)?;
    finish_status(pb, format!("Wrote artifact files to {}", out.display()));
    let model_path = out.join("model");
    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            let mut rng = StdRng::seed_from_u64(train_config.seed);
            tongues_speech_manifold::train::<CpuTrainBackend, _>(
                &model_config,
                train_config,
                &train_examples,
                &valid_examples,
                &vocab,
                &model_path,
                &device,
                &mut rng,
            )?;
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            let mut rng = StdRng::seed_from_u64(train_config.seed);
            tongues_speech_manifold::train::<CudaTrainBackend, _>(
                &model_config,
                train_config,
                &train_examples,
                &valid_examples,
                &vocab,
                &model_path,
                &device,
                &mut rng,
            )?;
        }
    }
    println!("Speech-manifold model saved to {}", out.display());
    Ok(())
}

fn cmd_speech_manifold_eval(
    model_dir: &Path,
    data: &Path,
    split: &str,
    task: SpeechManifoldTask,
    device_arg: DeviceArg,
) -> Result<()> {
    let vocab: Vocab = read_json_file(&model_dir.join("vocab.json"))?;
    let model_config: ModelConfig = read_json_file(&model_dir.join("model_config.json"))?;
    let train_config: SpeechManifoldTrainConfig =
        read_json_file(&model_dir.join("train_config.json"))?;
    let examples = tongues_speech_manifold::read_examples(&data.join(format!("{split}.jsonl")))?;
    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            let model = tongues_speech_manifold::load_model::<CpuInferBackend>(
                &model_config,
                model_dir,
                &device,
            )?;
            let report = tongues_speech_manifold::evaluate(
                &model,
                &examples,
                &vocab,
                task,
                train_config.allow_placeholder_acoustics,
                &device,
            );
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            let model = tongues_speech_manifold::load_model::<CudaInferBackend>(
                &model_config,
                model_dir,
                &device,
            )?;
            let report = tongues_speech_manifold::evaluate(
                &model,
                &examples,
                &vocab,
                task,
                train_config.allow_placeholder_acoustics,
                &device,
            );
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}

fn cmd_speech_manifold_infer(
    model_dir: &Path,
    task: SpeechManifoldTask,
    input: &str,
    device_arg: DeviceArg,
    _output_mode: OutputMode,
) -> Result<()> {
    let vocab: Vocab = read_json_file(&model_dir.join("vocab.json"))?;
    let model_config: ModelConfig = read_json_file(&model_dir.join("model_config.json"))?;
    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            let model = tongues_speech_manifold::load_model::<CpuInferBackend>(
                &model_config,
                model_dir,
                &device,
            )?;
            println!(
                "{}",
                tongues_speech_manifold::predict(&model, input, task, &vocab, &device)
            );
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            let model = tongues_speech_manifold::load_model::<CudaInferBackend>(
                &model_config,
                model_dir,
                &device,
            )?;
            println!(
                "{}",
                tongues_speech_manifold::predict(&model, input, task, &vocab, &device)
            );
        }
    }
    Ok(())
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

fn cmd_phonemes(text: &str) -> Result<()> {
    use speech::{EnglishPhonemicizer, PhonemicizeRequest, Phonemicizer, VarietyId};

    let phonemicizer = EnglishPhonemicizer;
    let phonemicized = phonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: text.to_string(),
            variety: VarietyId("en-US".to_string()),
            style: None,
        })
        .map_err(|e| anyhow::anyhow!("Failed to phonemicize: {:?}", e))?;

    let mut words: Vec<(usize, Vec<speech::Syllable>)> = Vec::new();
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
    use speech::{EnglishPhonemicizer, PhonemicizeRequest, Phonemicizer, VarietyId};

    let phonemicizer = EnglishPhonemicizer;
    let phonemicized = phonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: text.to_string(),
            variety: VarietyId("en-US".to_string()),
            style: None,
        })
        .map_err(|e| anyhow::anyhow!("Failed to phonemicize: {:?}", e))?;

    let mut words: Vec<(usize, Vec<speech::Syllable>)> = Vec::new();
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
    phone: &speech::PhoneToken,
    phonemes: &[speech::PhonemeToken],
) -> Option<speech::PhonemeId> {
    for phoneme_token in phonemes {
        for realized_phone in &phoneme_token.realized_as {
            if realized_phone.phone == phone.phone
                && realized_phone.features == phone.features
                && realized_phone.span == phone.span
            {
                if let speech::Spec::Known(ref id) = phoneme_token.phoneme {
                    return Some(id.clone());
                }
            }
        }
    }
    None
}

fn phone_ipa(phone: &speech::PhoneToken) -> &str {
    match &phone.phone {
        speech::Spec::Known(id) => id
            .as_str()
            .strip_prefix("ipa.phone.")
            .unwrap_or(id.as_str()),
        _ => "",
    }
}

fn syllables_to_phonemes_ipa(
    syllables: &[speech::Syllable],
    phonemes: &[speech::PhonemeToken],
    variety: &speech::VarietyId,
) -> String {
    syllables
        .iter()
        .enumerate()
        .map(|(index, syllable)| {
            let mut text = String::new();
            let mut has_stress_mark = false;
            let stress_char = match syllable.stress {
                speech::Spec::Known(speech::Stress::Primary) => {
                    has_stress_mark = true;
                    Some('ˈ')
                }
                speech::Spec::Known(speech::Stress::Secondary) => {
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
                    let symbol = speech::phoneme_default_phone_display_symbol(&phoneme_id, variety);
                    text.push_str(&symbol);
                } else {
                    text.push_str(phone_ipa(phone));
                }
            }
            text
        })
        .collect()
}

fn syllables_to_ipa_formatted(syllables: &[speech::Syllable]) -> String {
    syllables
        .iter()
        .enumerate()
        .map(|(index, syllable)| {
            let mut text = String::new();
            let mut has_stress_mark = false;
            let stress_char = match syllable.stress {
                speech::Spec::Known(speech::Stress::Primary) => {
                    has_stress_mark = true;
                    Some('ˈ')
                }
                speech::Spec::Known(speech::Stress::Secondary) => {
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

fn token_word_index(features: &speech::FeatureBundle) -> Option<usize> {
    let value = features
        .values
        .get(&speech::FeatureId("orthography.word_index".into()))?;
    match value {
        speech::Spec::Known(speech::FeatureValue::Number(value))
            if value.is_finite() && *value >= 0.0 =>
        {
            Some(*value as usize)
        }
        _ => None,
    }
}

// ── fetch-cmudict ──────────────────────────────────────────────────────────

fn cmd_fetch_cmudict(out: &Path) -> Result<()> {
    const URL: &str = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict";
    println!("Fetching CMUdict from {}", URL);

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).context("creating output directory")?;
    }

    // Use curl if available (standard on Linux/macOS), fall back to wget
    let status = std::process::Command::new("curl")
        .args(["-fsSL", "-o", out.to_str().unwrap_or("cmudict.dict"), URL])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("Saved to {}", out.display());
            Ok(())
        }
        _ => {
            // Try wget
            let s = std::process::Command::new("wget")
                .args(["-qO", out.to_str().unwrap_or("cmudict.dict"), URL])
                .status()
                .context("neither curl nor wget succeeded")?;
            if s.success() {
                println!("Saved to {}", out.display());
                Ok(())
            } else {
                anyhow::bail!(
                    "Could not download CMUdict. \
                     Please download manually from:\n  {}\nand save to {}",
                    URL,
                    out.display()
                )
            }
        }
    }
}

// ── prepare ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
struct OpenEpdEntry {
    rarity: f32,
    ipa: std::collections::BTreeMap<String, String>,
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
        total_words, skipped_openepd
    );
    fs::create_dir_all(out).context("creating output directory")?;

    // Open output files
    let train_path = out.join("train.jsonl");
    let valid_path = out.join("valid.jsonl");
    let test_path = out.join("test.jsonl");

    let train_file = fs::File::create(&train_path)?;
    let valid_file = fs::File::create(&valid_path)?;
    let test_file = fs::File::create(&test_path)?;

    use indicatif::{ProgressBar, ProgressStyle};
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
    let pb = ProgressBar::new(total_words as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}")?
            .progress_chars("#>-")
    );

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

    println!(
        "Data splits generated on-the-fly:\n  train={} valid={} test={}",
        train_words.len(),
        valid_words.len(),
        test_words.len()
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
        fs::write(&path, deduped.join("\n"))?;
    }

    // Build & save vocabulary
    println!("Building vocabulary from seen characters ...");
    let vocab = {
        let w_list: Vec<String> = seen_word_chars.iter().cloned().collect();
        let pm_list: Vec<String> = seen_phoneme_chars.iter().cloned().collect();
        Vocab::build(&w_list, &pm_list, &[])
    };

    println!("  Unified vocabulary size: {}", vocab.size());
    let vocab_path = out.join("vocab.json");
    let vocab_json = serde_json::to_string_pretty(&vocab)?;
    fs::write(&vocab_path, &vocab_json).context("writing vocab.json")?;
    println!("  Vocab saved to {}", vocab_path.display());

    println!("Prepare complete.");
    Ok(())
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
            format!("Loaded vocabulary with {} tokens", vocab.size()),
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
            base_train_lexemes.len(),
            valid_lexemes.len()
        ),
    );

    println!(
        "Loaded {} train / {} valid lexemes",
        base_train_lexemes.len(),
        valid_lexemes.len()
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
            train_lexemes.len()
        ),
    );
    println!(
        "  frequency-weighted train examples: {} (max_repeat={} rarity_cap={})",
        train_lexemes.len(),
        train_config.max_frequency_repeat,
        train_config.frequency_rarity_cap
    );

    let added_sight_word_lexemes = add_sight_word_training_examples(&mut train_lexemes, data)?;
    if added_sight_word_lexemes > 0 {
        println!(
            "  included {} extra sight-word training examples",
            added_sight_word_lexemes
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
        epochs, patience, batch_size
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
        test_lexemes.len()
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
                println!("  {}: {} lexemes", split, lexemes.len());
            }
        }
        RefinementSourceArg::SightWords => {
            println!(
                "  source: built-in Dolch sight words ({} words before OpenEPD/vocab filtering)",
                SIGHT_WORDS.len()
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
        records.len(),
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
        refine_lexemes.len(),
        mean_edit_distance
    );
    println!(
        "Refinement training: lr={} wd={} epochs={} patience={} batch_size={}",
        learning_rate, weight_decay, epochs, patience, batch_size
    );

    let train_config = TrainConfig {
        learning_rate,
        weight_decay,
        dropout: model_config.dropout,
        batch_size,
        epochs,
        early_stopping_patience: patience,
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
    println!("  OpenEPD words: {}", openepd.word_count());

    let tasks: Vec<Task> = match task_filter {
        Some(task) => vec![task],
        None => vec![Task::G2P, Task::P2G],
    };

    let total: usize = split_lexemes
        .iter()
        .map(|(_, lexemes)| lexemes.len() * tasks.len())
        .sum();
    let pb = indicatif::ProgressBar::new(total as u64);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}")?
            .progress_chars("#>-"),
    );

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
            split_checked,
            split_discrepancies,
            split_skipped_missing_openepd,
            split_skipped_parse_error,
            split_skipped_unknown_vocab
        ));
    }
    pb.finish_and_clear();
    if skipped_missing_openepd > 0 || skipped_parse_error > 0 || skipped_unknown_vocab > 0 {
        println!(
            "Skipped during OpenEPD mining: {} missing OpenEPD entries, {} parse errors, {} OpenEPD golds with chars outside vocab",
            skipped_missing_openepd, skipped_parse_error, skipped_unknown_vocab
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
    println!("  OpenEPD words: {}", openepd.word_count());

    let tasks: Vec<Task> = match task_filter {
        Some(task) => vec![task],
        None => vec![Task::G2P, Task::P2G],
    };

    let mut sight_words = std::collections::BTreeSet::new();
    for word in SIGHT_WORDS {
        sight_words.insert((*word).to_string());
    }

    let pb = indicatif::ProgressBar::new((sight_words.len() * tasks.len()) as u64);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}")?
            .progress_chars("#>-"),
    );

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
        checked,
        records.len(),
        refine_lexemes.len(),
        skipped_missing_openepd,
        skipped_parse_error,
        skipped_unknown_vocab
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
    }

    #[test]
    fn cli_accepts_speech_manifold_commands() {
        let cli = Cli::try_parse_from([
            "tongues",
            "speech-manifold",
            "infer",
            "--model",
            "models/speech-manifold/openepd-synth-v0",
            "--task",
            "spelling-to-ipa",
            "tires",
        ])
        .expect("speech-manifold infer should parse");

        assert!(matches!(
            cli.command,
            Some(Commands::SpeechManifold {
                command: SpeechManifoldCommands::Infer { .. }
            })
        ));
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
    }

    #[test]
    fn cli_keeps_legacy_predict_alias() {
        let cli = Cli::try_parse_from(["tongues", "infer", "farkle"])
            .expect("legacy infer alias should parse");

        assert!(matches!(cli.command, Some(Commands::Predict { .. })));
    }
}
