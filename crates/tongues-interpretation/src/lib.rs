//! LibriSpeech utterance-level streaming ASR scaffold.
//!
//! V1 prepares LibriSpeech-style FLAC/transcript pairs, writes log-Mel feature
//! files durably, enriches each utterance with seams sentence splits and speech
//! phonemicizer output, and trains a small streaming frame classifier with CTC
//! style greedy collapse. Word-context and masked-word heads use Burn's native
//! CTC loss over compact target sequences.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::{CTCLossConfig, CrossEntropyLossConfig, Reduction};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::Recorder;
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use seams::SentenceDetectorDialog;
use serde::{Deserialize, Serialize};
use speaking::segment::TerminalPunctuation;
use speaking::syntax::{
    GrammarParser, PartOfSpeech, SentenceSyntaxAnalysis, SyntacticLinkKind, VarietyGrammarParser,
};
use speaking::{
    phonemicizer_for_variety, syllables_to_ipa, PhonemicizeRequest, ProsodyTrack,
    SpeechBoundaryToken, Syllable, VarietyId,
};
use tongues_core::Vocab;
use tongues_neural::{make_recorder, write_manifest, ModelArtifactManifest};

pub const FAMILY: &str = "interpretation";
pub const ARCHITECTURE: &str = "streaming-mel-native-ctc";
pub const DEFAULT_DATASET_ID: &str = "librispeech-mini-v0";
pub const DEFAULT_WIKTIONARY_AUDIO_DATA_DIR: &str =
    "datasets/wiktionary/enwiktionary-2026-06-01-v0";
pub const DEFAULT_MAX_WIKTIONARY_AUDIO: usize = 250;
pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 16_000;
pub const DEFAULT_MEL_BINS: usize = 80;
pub const COMPACT_AUDIO_EXTRA_BINS: usize = 7;
pub const DEFAULT_PREPARE_MAX_THREADS: usize = 8;
const DOWNLOAD_USER_AGENT: &str = "tongues-dataset-prep/0.1";
const DOWNLOAD_MAX_ATTEMPTS: usize = 6;
const WIKTIONARY_AUDIO_DOWNLOAD_THROTTLE: Duration = Duration::from_millis(750);
pub const DEFAULT_COMPACT_AUDIO_FEATURE_BINS: usize =
    DEFAULT_MEL_BINS + DEFAULT_MEL_BINS + COMPACT_AUDIO_EXTRA_BINS;
pub const CTC_BLANK: &str = "<CTC_BLANK>";
pub const WORD_BLANK: &str = "<WORD_BLANK>";
pub const WORD_UNK: &str = "<WORD_UNK>";
pub const WORD_NUM: &str = "<NUM>";
pub const MAX_WORD_VOCAB_TOKENS: usize = 20_000;
pub const MIN_WORD_VOCAB_COUNT: usize = 2;
pub const BOUNDARY_CONTINUE: &str = "<boundary:continue>";
pub const BOUNDARY_EMIT: &str = "<boundary:emit>";
pub const BOUNDARY_REPAIR: &str = "<boundary:repair>";

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

fn counted_progress_style(template: &str) -> indicatif::ProgressStyle {
    use std::fmt::Write;

    indicatif::ProgressStyle::default_bar()
        .template(template)
        .expect("valid template")
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
        .progress_chars("#>-")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LibriSpeechSubset {
    Mini,
    TrainClean100,
}

impl LibriSpeechSubset {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "mini" | "mini-librispeech" => Some(Self::Mini),
            "train-clean-100" => Some(Self::TrainClean100),
            _ => None,
        }
    }

    pub fn dataset_id(self) -> &'static str {
        match self {
            Self::Mini => "librispeech-mini-v0",
            Self::TrainClean100 => "librispeech-train-clean-100-v0",
        }
    }

    pub fn archive_url(self) -> &'static str {
        match self {
            Self::Mini => "https://www.openslr.org/resources/31/train-clean-5.tar.gz",
            Self::TrainClean100 => "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterpretationConfig {
    pub dataset_id: String,
    pub subset: LibriSpeechSubset,
    pub train_frac: f64,
    pub valid_frac: f64,
    pub seed: u64,
    pub sample_rate_hz: u32,
    pub window_ms: f32,
    pub hop_ms: f32,
    pub mel_bins: usize,
    #[serde(default = "default_compact_audio_features")]
    pub compact_audio_features: bool,
    pub variety: String,
    pub max_utterances: Option<usize>,
    pub download_url: String,
    #[serde(default)]
    pub wiktionary_audio_data_dir: Option<String>,
    #[serde(default)]
    pub max_wiktionary_audio: Option<usize>,
    #[serde(default)]
    pub download_wiktionary_audio: bool,
}

impl Default for InterpretationConfig {
    fn default() -> Self {
        let subset = LibriSpeechSubset::Mini;
        Self {
            dataset_id: subset.dataset_id().to_string(),
            subset,
            train_frac: 0.8,
            valid_frac: 0.1,
            seed: 42,
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            window_ms: 25.0,
            hop_ms: 10.0,
            mel_bins: DEFAULT_MEL_BINS,
            compact_audio_features: true,
            variety: "en-US".to_string(),
            max_utterances: None,
            download_url: subset.archive_url().to_string(),
            wiktionary_audio_data_dir: Some(DEFAULT_WIKTIONARY_AUDIO_DATA_DIR.to_string()),
            max_wiktionary_audio: Some(DEFAULT_MAX_WIKTIONARY_AUDIO),
            download_wiktionary_audio: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterpretationTrainConfig {
    pub learning_rate: f64,
    pub weight_decay: f32,
    pub dropout: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub early_stopping_patience: usize,
    pub seed: u64,
    pub transcript_loss_weight: f32,
    pub boundary_loss_weight: f32,
    pub phoneme_loss_weight: f32,
    #[serde(default = "default_phone_loss_weight")]
    pub phone_loss_weight: f32,
    #[serde(default = "default_feature_ctc_loss_weight")]
    pub feature_ctc_loss_weight: f32,
    #[serde(default = "default_prev_word_loss_weight")]
    pub prev_word_loss_weight: f32,
    #[serde(default = "default_current_word_loss_weight")]
    pub current_word_loss_weight: f32,
    #[serde(default = "default_next_word_loss_weight")]
    pub next_word_loss_weight: f32,
    #[serde(default = "default_masked_word_loss_weight")]
    pub masked_word_loss_weight: f32,
    #[serde(default = "default_masked_word_phoneme_loss_weight")]
    pub masked_word_phoneme_loss_weight: f32,
    #[serde(default = "default_repair_loss_weight")]
    pub repair_loss_weight: f32,
    #[serde(default = "default_masked_audio_loss_weight")]
    pub masked_audio_loss_weight: f32,
    #[serde(default = "default_syntax_loss_weight")]
    pub syntax_loss_weight: f32,
    #[serde(default = "default_seq2seq_loss_weight")]
    pub seq2seq_loss_weight: f32,
    #[serde(default = "default_word_mask_rate")]
    pub word_mask_rate: f32,
    #[serde(default = "default_mask_every_n_frames")]
    pub mask_every_n_frames: usize,
    #[serde(default = "default_mask_span_frames")]
    pub mask_span_frames: usize,
    #[serde(default = "default_resume_learning_rate_scale")]
    pub resume_learning_rate_scale: f64,
    pub max_frames: usize,
    #[serde(default = "default_max_seq2seq_tokens")]
    pub max_seq2seq_tokens: usize,
    #[serde(default = "default_input_feature_bins")]
    pub input_feature_bins: usize,
}

fn default_phone_loss_weight() -> f32 {
    0.25
}

fn default_feature_ctc_loss_weight() -> f32 {
    0.35
}

fn default_prev_word_loss_weight() -> f32 {
    0.1
}

fn default_current_word_loss_weight() -> f32 {
    0.2
}

fn default_next_word_loss_weight() -> f32 {
    0.15
}

fn default_masked_word_loss_weight() -> f32 {
    0.2
}

fn default_masked_word_phoneme_loss_weight() -> f32 {
    0.15
}

fn default_repair_loss_weight() -> f32 {
    0.15
}

fn default_masked_audio_loss_weight() -> f32 {
    0.35
}

fn default_syntax_loss_weight() -> f32 {
    0.05
}

fn default_seq2seq_loss_weight() -> f32 {
    0.35
}

fn default_word_mask_rate() -> f32 {
    1.0
}

fn default_mask_every_n_frames() -> usize {
    12
}

fn default_mask_span_frames() -> usize {
    3
}

fn default_resume_learning_rate_scale() -> f64 {
    0.25
}

fn default_max_seq2seq_tokens() -> usize {
    192
}

fn default_input_feature_bins() -> usize {
    DEFAULT_COMPACT_AUDIO_FEATURE_BINS
}

fn default_compact_audio_features() -> bool {
    true
}

impl Default for InterpretationTrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            weight_decay: 1e-4,
            dropout: 0.1,
            batch_size: 8,
            epochs: 20,
            early_stopping_patience: 5,
            seed: 0,
            transcript_loss_weight: 1.0,
            boundary_loss_weight: 0.15,
            phoneme_loss_weight: 0.25,
            phone_loss_weight: 0.25,
            feature_ctc_loss_weight: default_feature_ctc_loss_weight(),
            prev_word_loss_weight: 0.1,
            current_word_loss_weight: 0.2,
            next_word_loss_weight: 0.15,
            masked_word_loss_weight: 0.2,
            masked_word_phoneme_loss_weight: 0.15,
            repair_loss_weight: 0.15,
            masked_audio_loss_weight: 0.35,
            syntax_loss_weight: default_syntax_loss_weight(),
            seq2seq_loss_weight: default_seq2seq_loss_weight(),
            word_mask_rate: default_word_mask_rate(),
            mask_every_n_frames: default_mask_every_n_frames(),
            mask_span_frames: default_mask_span_frames(),
            resume_learning_rate_scale: default_resume_learning_rate_scale(),
            max_frames: 1600,
            max_seq2seq_tokens: default_max_seq2seq_tokens(),
            input_feature_bins: default_input_feature_bins(),
        }
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    pub mel_bins: usize,
    pub vocab_size: usize,
    pub phoneme_vocab_size: usize,
    pub phone_vocab_size: usize,
    pub word_vocab_size: usize,
    #[config(default = 8)]
    pub place_vocab_size: usize,
    #[config(default = 7)]
    pub manner_vocab_size: usize,
    #[config(default = 4)]
    pub voicing_vocab_size: usize,
    #[config(default = 4)]
    pub syllabic_vocab_size: usize,
    #[config(default = 6)]
    pub height_vocab_size: usize,
    #[config(default = 5)]
    pub backness_vocab_size: usize,
    #[config(default = 4)]
    pub rounding_vocab_size: usize,
    #[config(default = 8)]
    pub syntax_pos_vocab_size: usize,
    #[config(default = 16)]
    pub syntax_link_vocab_size: usize,
    #[config(default = 15)]
    pub syntax_head_offset_vocab_size: usize,
    #[config(default = 192)]
    pub hidden_size: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AsrModel<B> {
        AsrModel {
            input: LinearConfig::new(self.mel_bins, self.hidden_size).init(device),
            transcript: LinearConfig::new(self.hidden_size, self.vocab_size).init(device),
            seq2seq_transcript: LinearConfig::new(self.hidden_size, self.vocab_size).init(device),
            boundary: LinearConfig::new(self.hidden_size, 3).init(device),
            phoneme: LinearConfig::new(self.hidden_size, self.phoneme_vocab_size).init(device),
            phone: LinearConfig::new(self.hidden_size, self.phone_vocab_size).init(device),
            place: LinearConfig::new(self.hidden_size, self.place_vocab_size).init(device),
            manner: LinearConfig::new(self.hidden_size, self.manner_vocab_size).init(device),
            voicing: LinearConfig::new(self.hidden_size, self.voicing_vocab_size).init(device),
            syllabic: LinearConfig::new(self.hidden_size, self.syllabic_vocab_size).init(device),
            height: LinearConfig::new(self.hidden_size, self.height_vocab_size).init(device),
            backness: LinearConfig::new(self.hidden_size, self.backness_vocab_size).init(device),
            rounding: LinearConfig::new(self.hidden_size, self.rounding_vocab_size).init(device),
            prev_word: LinearConfig::new(self.hidden_size, self.word_vocab_size).init(device),
            current_word: LinearConfig::new(self.hidden_size, self.word_vocab_size).init(device),
            next_word: LinearConfig::new(self.hidden_size, self.word_vocab_size).init(device),
            masked_word: LinearConfig::new(self.hidden_size, self.word_vocab_size).init(device),
            masked_word_phoneme: LinearConfig::new(self.hidden_size, self.phoneme_vocab_size)
                .init(device),
            syntax_pos: LinearConfig::new(self.hidden_size, self.syntax_pos_vocab_size)
                .init(device),
            syntax_link: LinearConfig::new(self.hidden_size, self.syntax_link_vocab_size)
                .init(device),
            syntax_head_offset: LinearConfig::new(
                self.hidden_size,
                self.syntax_head_offset_vocab_size,
            )
            .init(device),
            parse_ok: LinearConfig::new(self.hidden_size, 2).init(device),
            phrase_boundary: LinearConfig::new(self.hidden_size, 2).init(device),
            mel_reconstruction: LinearConfig::new(self.hidden_size, self.mel_bins).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct AsrModel<B: Backend> {
    input: Linear<B>,
    transcript: Linear<B>,
    seq2seq_transcript: Linear<B>,
    boundary: Linear<B>,
    phoneme: Linear<B>,
    phone: Linear<B>,
    place: Linear<B>,
    manner: Linear<B>,
    voicing: Linear<B>,
    syllabic: Linear<B>,
    height: Linear<B>,
    backness: Linear<B>,
    rounding: Linear<B>,
    prev_word: Linear<B>,
    current_word: Linear<B>,
    next_word: Linear<B>,
    masked_word: Linear<B>,
    masked_word_phoneme: Linear<B>,
    syntax_pos: Linear<B>,
    syntax_link: Linear<B>,
    syntax_head_offset: Linear<B>,
    parse_ok: Linear<B>,
    phrase_boundary: Linear<B>,
    mel_reconstruction: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> AsrModel<B> {
    pub fn forward(&self, mel: Tensor<B, 3>) -> AsrForward<B> {
        let hidden = self.dropout.forward(self.input.forward(mel).tanh());
        AsrForward {
            transcript_logits: self.transcript.forward(hidden.clone()),
            seq2seq_transcript_logits: self.seq2seq_transcript.forward(hidden.clone()),
            boundary_logits: self.boundary.forward(hidden.clone()),
            phoneme_logits: self.phoneme.forward(hidden.clone()),
            phone_logits: self.phone.forward(hidden.clone()),
            place_logits: self.place.forward(hidden.clone()),
            manner_logits: self.manner.forward(hidden.clone()),
            voicing_logits: self.voicing.forward(hidden.clone()),
            syllabic_logits: self.syllabic.forward(hidden.clone()),
            height_logits: self.height.forward(hidden.clone()),
            backness_logits: self.backness.forward(hidden.clone()),
            rounding_logits: self.rounding.forward(hidden.clone()),
            prev_word_logits: self.prev_word.forward(hidden.clone()),
            current_word_logits: self.current_word.forward(hidden.clone()),
            next_word_logits: self.next_word.forward(hidden.clone()),
            masked_word_logits: self.masked_word.forward(hidden.clone()),
            masked_word_phoneme_logits: self.masked_word_phoneme.forward(hidden.clone()),
            syntax_pos_logits: self.syntax_pos.forward(hidden.clone()),
            syntax_link_logits: self.syntax_link.forward(hidden.clone()),
            syntax_head_offset_logits: self.syntax_head_offset.forward(hidden.clone()),
            parse_ok_logits: self.parse_ok.forward(hidden.clone()),
            phrase_boundary_logits: self.phrase_boundary.forward(hidden.clone()),
            mel_reconstruction: self.mel_reconstruction.forward(hidden),
        }
    }
}

#[derive(Debug)]
pub struct AsrForward<B: Backend> {
    pub transcript_logits: Tensor<B, 3>,
    pub seq2seq_transcript_logits: Tensor<B, 3>,
    pub boundary_logits: Tensor<B, 3>,
    pub phoneme_logits: Tensor<B, 3>,
    pub phone_logits: Tensor<B, 3>,
    pub place_logits: Tensor<B, 3>,
    pub manner_logits: Tensor<B, 3>,
    pub voicing_logits: Tensor<B, 3>,
    pub syllabic_logits: Tensor<B, 3>,
    pub height_logits: Tensor<B, 3>,
    pub backness_logits: Tensor<B, 3>,
    pub rounding_logits: Tensor<B, 3>,
    pub prev_word_logits: Tensor<B, 3>,
    pub current_word_logits: Tensor<B, 3>,
    pub next_word_logits: Tensor<B, 3>,
    pub masked_word_logits: Tensor<B, 3>,
    pub masked_word_phoneme_logits: Tensor<B, 3>,
    pub syntax_pos_logits: Tensor<B, 3>,
    pub syntax_link_logits: Tensor<B, 3>,
    pub syntax_head_offset_logits: Tensor<B, 3>,
    pub parse_ok_logits: Tensor<B, 3>,
    pub phrase_boundary_logits: Tensor<B, 3>,
    pub mel_reconstruction: Tensor<B, 3>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrepareReport {
    pub utterances: usize,
    pub train_examples: usize,
    pub valid_examples: usize,
    pub test_examples: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrepareCheckpointState {
    pub status: String,
    pub dataset_id: String,
    pub utterances: usize,
    pub report: Option<PrepareReport>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrepareProgress {
    Stage {
        message: String,
    },
    Download {
        url: String,
        path: String,
        bytes: u64,
    },
    Extract {
        path: String,
    },
    Parse {
        transcripts: usize,
    },
    Features {
        utterance_id: String,
        rows: usize,
        path: String,
    },
    Transcribe {
        utterance_id: String,
        path: String,
    },
    ImportAudio {
        source: String,
        rows: usize,
    },
    Omit {
        utterance_id: String,
        reason: String,
        source_transcript: Option<String>,
        whisper_transcript: Option<String>,
        wer: Option<f64>,
        max_wer: Option<f64>,
    },
    Reuse {
        utterance_id: String,
        rows: usize,
        path: String,
    },
    Write {
        path: String,
        rows: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptRefinement {
    Use(String),
    KeepOriginal,
    Omit {
        reason: String,
        source_transcript: Option<String>,
        whisper_transcript: Option<String>,
        wer: Option<f64>,
        max_wer: Option<f64>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LibriSpeechUtterance {
    pub utterance_id: String,
    pub speaker_id: String,
    pub chapter_id: String,
    pub audio_path: String,
    pub mel_path: String,
    pub num_frames: usize,
    pub sample_rate_hz: u32,
    pub duration_ms: u64,
    pub transcript: String,
    pub sentences: Vec<SentenceSupervision>,
    #[serde(default)]
    pub repair_examples: Vec<RepairSupervision>,
    #[serde(default)]
    pub word_supervision: Vec<WordSupervision>,
    #[serde(default)]
    pub masked_word_examples: Vec<MaskedWordExample>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SentenceSupervision {
    pub text: String,
    pub start_char: usize,
    pub end_char: usize,
    pub start_frame: usize,
    pub end_frame: usize,
    pub boundary_label: String,
    pub terminal: Option<char>,
    pub phonemes: String,
    pub phones: String,
    pub phoneme_tokens: Vec<speaking::PhonemeToken>,
    pub phone_tokens: Vec<speaking::PhoneToken>,
    pub syllables: Vec<Syllable>,
    pub boundaries: Vec<SpeechBoundaryToken>,
    pub prosody: ProsodyTrack,
    pub warnings: Vec<speaking::PronunciationWarning>,
    #[serde(default)]
    pub syntax: SyntaxSupervision,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SyntaxSupervision {
    pub words: Vec<SyntaxWordSupervision>,
    pub links: Vec<SyntaxLinkSupervision>,
    pub parse_ok: bool,
    pub parse_rank: f32,
    pub parse_cost: f32,
    pub supervision_weight: f32,
    pub analysis: SentenceSyntaxAnalysis,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntaxWordSupervision {
    pub word: String,
    pub sentence_word_index: usize,
    pub pos: String,
    pub link_labels: Vec<String>,
    pub primary_link_label: String,
    pub linked_word_index: Option<usize>,
    pub head_offset: i32,
    pub phrase_boundary: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntaxLinkSupervision {
    pub left: usize,
    pub right: usize,
    pub label: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairSupervision {
    pub misheard_text: String,
    pub corrected_text: String,
    pub start_char: usize,
    pub end_char: usize,
    pub start_frame: usize,
    pub end_frame: usize,
    pub repair_label: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordSupervision {
    pub word: String,
    pub word_index: usize,
    pub sentence_index: usize,
    pub sentence_word_index: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub start_frame: usize,
    pub end_frame: usize,
    pub phonemes: String,
    pub phones: String,
    pub previous_word: Option<String>,
    pub next_word: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaskedWordExample {
    pub left_context: String,
    pub right_context: String,
    pub masked_word: String,
    pub masked_word_phonemes: String,
    pub start_frame: usize,
    pub end_frame: usize,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    pub examples: usize,
    pub loss: f32,
    pub token_error_rate: f32,
    pub word_error_rate: f32,
    pub seq2seq_token_error_rate: f32,
    pub boundary_f1: f32,
    pub repair_f1: f32,
    pub phoneme_token_error_rate: f32,
    pub phone_token_error_rate: f32,
    pub masked_audio_mse: f32,
    pub prev_word_accuracy: f32,
    pub current_word_accuracy: f32,
    pub next_word_accuracy: f32,
    pub masked_word_accuracy: f32,
    pub masked_word_phoneme_token_error_rate: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamEvent {
    pub partial_transcript: String,
    pub seq2seq_transcript: String,
    pub final_sentences: Vec<SentenceSupervision>,
    pub repair_events: Vec<RepairSupervision>,
    pub previous_word: Option<WordPrediction>,
    pub current_word: Option<WordPrediction>,
    pub next_word: Option<WordPrediction>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordPrediction {
    pub word: Option<String>,
    pub phonemes: Option<String>,
}

pub fn prepare_dataset(out: &Path, config: &InterpretationConfig) -> Result<PrepareReport> {
    prepare_dataset_with_progress(out, config, |_| {})
}

pub fn prepare_dataset_with_progress(
    out: &Path,
    config: &InterpretationConfig,
    mut progress: impl FnMut(PrepareProgress),
) -> Result<PrepareReport> {
    prepare_dataset_inner(out, config, &mut progress, false, |_, _, _, _| {
        Ok(TranscriptRefinement::KeepOriginal)
    })
}

pub fn prepare_dataset_with_progress_and_transcript_refiner(
    out: &Path,
    config: &InterpretationConfig,
    mut progress: impl FnMut(PrepareProgress),
    mut transcript_refiner: impl FnMut(&str, &Path, &[f32], &str) -> Result<TranscriptRefinement>,
) -> Result<PrepareReport> {
    prepare_dataset_inner(
        out,
        config,
        &mut progress,
        true,
        |id, path, samples, seed| transcript_refiner(id, path, samples, seed),
    )
}

fn prepare_dataset_inner(
    out: &Path,
    config: &InterpretationConfig,
    progress: &mut impl FnMut(PrepareProgress),
    use_transcript_refiner: bool,
    mut transcript_refiner: impl FnMut(&str, &Path, &[f32], &str) -> Result<TranscriptRefinement>,
) -> Result<PrepareReport> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    fs::create_dir_all(out.join("features")).context("creating features directory")?;
    write_prepare_state(out, "starting", config, 0, None)?;
    let feature_bins = audio_feature_bins(config);
    let archive = out.join("source.tar.gz");
    if !archive.exists() {
        progress(PrepareProgress::Stage {
            message: format!("Downloading {}", config.download_url),
        });
        download_to_part(&config.download_url, &archive, progress)?;
    }
    let source_dir = out.join("source");
    let extract_marker = out.join(".extract-complete");
    if !extract_marker.exists() {
        if source_dir.exists() && !discover_transcripts(&source_dir)?.is_empty() {
            fs::write(&extract_marker, b"ok\n")?;
        } else {
            progress(PrepareProgress::Extract {
                path: archive.display().to_string(),
            });
            if source_dir.exists() {
                fs::remove_dir_all(&source_dir)
                    .with_context(|| format!("removing partial {}", source_dir.display()))?;
            }
            let source_part = out.join("source.part");
            if source_part.exists() {
                fs::remove_dir_all(&source_part)
                    .with_context(|| format!("removing partial {}", source_part.display()))?;
            }
            fs::create_dir_all(&source_part)?;
            let tar_gz = File::open(&archive)?;
            let decoder = flate2::read::GzDecoder::new(tar_gz);
            let mut archive = tar::Archive::new(decoder);
            archive.unpack(&source_part)?;
            fs::rename(&source_part, &source_dir)?;
            fs::write(&extract_marker, b"ok\n")?;
        }
    }

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
        .context("building interpretation prepare thread pool")?;
    let transcripts = discover_transcripts_with_pool(&source_dir, &prepare_pool)?;
    progress(PrepareProgress::Parse {
        transcripts: transcripts.len(),
    });
    write_prepare_state(out, "parsed", config, 0, None)?;
    anyhow::ensure!(!transcripts.is_empty(), "no LibriSpeech transcripts found");
    let selected_transcripts = transcripts
        .into_iter()
        .take(config.max_utterances.unwrap_or(usize::MAX))
        .collect::<Vec<_>>();
    let selected_ids = selected_transcripts
        .iter()
        .map(|item| item.utterance_id.clone())
        .collect::<BTreeSet<_>>();
    let detector = SentenceDetectorDialog::new().context("initializing seams detector")?;
    let utterances_path = out.join("utterances.jsonl");
    let mut rows = recover_utterance_rows(&utterances_path, out, config)?;
    rows.retain(|row| selected_ids.contains(&row.utterance_id));
    for row in &mut rows {
        progress(PrepareProgress::Stage {
            message: format!("Refreshing recovered supervision for {}", row.utterance_id),
        });
        enrich_row_supervision(row, &detector, config)?;
    }
    if utterances_path.exists() {
        write_jsonl_atomic(&utterances_path, &rows, progress)?;
    }
    let mut row_by_id = rows
        .iter()
        .map(|row| (row.utterance_id.clone(), row.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut utterance_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&utterances_path)
            .with_context(|| format!("opening {}", utterances_path.display()))?,
    );
    let mut pending_transcripts = Vec::new();
    for item in selected_transcripts {
        if let Some(existing) = row_by_id.get(&item.utterance_id) {
            progress(PrepareProgress::Reuse {
                utterance_id: item.utterance_id,
                rows: existing.num_frames,
                path: out.join(&existing.mel_path).display().to_string(),
            });
            continue;
        }
        pending_transcripts.push(item);
    }

    if use_transcript_refiner {
        for item in pending_transcripts {
            let samples = read_flac_mono(&item.audio_path)?;
            let rel_mel = PathBuf::from("features").join(format!("{}.mel.bin", item.utterance_id));
            let mel_path = out.join(&rel_mel);
            let frames = match recover_feature_frames(&mel_path, feature_bins, config)? {
                Some(frames) => {
                    progress(PrepareProgress::Reuse {
                        utterance_id: item.utterance_id.clone(),
                        rows: frames,
                        path: mel_path.display().to_string(),
                    });
                    frames
                }
                None => {
                    let features = audio_features(&samples, config);
                    write_mel_file(&mel_path, &features, feature_bins)?;
                    progress(PrepareProgress::Features {
                        utterance_id: item.utterance_id.clone(),
                        rows: features.len(),
                        path: mel_path.display().to_string(),
                    });
                    features.len()
                }
            };
            progress(PrepareProgress::Transcribe {
                utterance_id: item.utterance_id.clone(),
                path: item.audio_path.display().to_string(),
            });
            let transcript = match transcript_refiner(
                &item.utterance_id,
                &item.audio_path,
                &samples,
                &item.transcript,
            )? {
                TranscriptRefinement::Use(text) => normalize_asr_transcript(&text),
                TranscriptRefinement::KeepOriginal => normalize_librispeech_text(&item.transcript),
                TranscriptRefinement::Omit {
                    reason,
                    source_transcript,
                    whisper_transcript,
                    wer,
                    max_wer,
                } => {
                    progress(PrepareProgress::Omit {
                        utterance_id: item.utterance_id,
                        reason,
                        source_transcript,
                        whisper_transcript,
                        wer,
                        max_wer,
                    });
                    continue;
                }
            };
            let transcript = if transcript.trim().is_empty() {
                normalize_librispeech_text(&item.transcript)
            } else {
                transcript
            };
            let sentences = sentence_supervision(&detector, &transcript, frames, config)?;
            let repair_examples = repair_supervision(&sentences);
            let word_supervision = word_supervision(&sentences);
            let masked_word_examples = masked_word_examples(&word_supervision, &transcript);
            let row = LibriSpeechUtterance {
                utterance_id: item.utterance_id,
                speaker_id: item.speaker_id,
                chapter_id: item.chapter_id,
                audio_path: item.audio_path.display().to_string(),
                mel_path: rel_mel.display().to_string(),
                num_frames: frames,
                sample_rate_hz: config.sample_rate_hz,
                duration_ms: samples.len() as u64 * 1000 / config.sample_rate_hz as u64,
                transcript,
                sentences,
                repair_examples,
                word_supervision,
                masked_word_examples,
            };
            writeln!(utterance_writer, "{}", serde_json::to_string(&row)?)?;
            utterance_writer.flush()?;
            row_by_id.insert(row.utterance_id.clone(), row.clone());
            rows.push(row);
        }
    } else {
        let mut prepared_rows = prepare_pool.install(|| {
            pending_transcripts
                .par_iter()
                .enumerate()
                .map_init(
                    || SentenceDetectorDialog::new().context("initializing seams detector worker"),
                    |detector_result, (index, item)| {
                        let detector = detector_result.as_ref().map_err(|err| {
                            anyhow::anyhow!("initializing seams detector worker: {err:#}")
                        })?;
                        process_librispeech_item_without_refiner(
                            index,
                            item,
                            out,
                            config,
                            feature_bins,
                            detector,
                        )
                    },
                )
                .collect::<Result<Vec<_>>>()
        })?;
        prepared_rows.sort_by_key(|row| row.index);
        for prepared in prepared_rows {
            for event in prepared.progress {
                progress(event);
            }
            writeln!(
                utterance_writer,
                "{}",
                serde_json::to_string(&prepared.row)?
            )?;
            utterance_writer.flush()?;
            row_by_id.insert(prepared.row.utterance_id.clone(), prepared.row.clone());
            rows.push(prepared.row);
        }
    }
    utterance_writer.flush()?;
    let imported =
        import_wiktionary_audio_rows(out, config, feature_bins, &prepare_pool, progress)?;
    if !imported.is_empty() {
        for row in &imported {
            writeln!(utterance_writer, "{}", serde_json::to_string(row)?)?;
            row_by_id.insert(row.utterance_id.clone(), row.clone());
        }
        utterance_writer.flush()?;
        rows.extend(imported);
    }
    write_prepare_state(out, "utterances", config, rows.len(), None)?;

    let mut shuffled = rows;
    anyhow::ensure!(
        !shuffled.is_empty(),
        "no usable utterances remained after transcript preparation"
    );
    shuffled.shuffle(&mut rand::rngs::StdRng::seed_from_u64(config.seed));
    let n = shuffled.len();
    let train_end = ((n as f64) * config.train_frac).round().min(n as f64) as usize;
    let valid_end = (train_end + ((n as f64) * config.valid_frac).round() as usize).min(n);
    let train = shuffled[..train_end].to_vec();
    let valid = shuffled[train_end..valid_end].to_vec();
    let test = shuffled[valid_end..].to_vec();
    write_prepare_state(out, "writing", config, n, None)?;
    write_jsonl_atomic(&out.join("train.jsonl"), &train, progress)?;
    write_jsonl_atomic(&out.join("valid.jsonl"), &valid, progress)?;
    write_jsonl_atomic(&out.join("test.jsonl"), &test, progress)?;
    let vocab = build_text_vocab([&train[..], &valid[..], &test[..]].concat().as_slice());
    write_text_atomic(
        &out.join("vocab.json"),
        serde_json::to_string_pretty(&vocab)?,
    )?;
    let phoneme_vocab =
        build_phoneme_vocab([&train[..], &valid[..], &test[..]].concat().as_slice());
    write_text_atomic(
        &out.join("phoneme_vocab.json"),
        serde_json::to_string_pretty(&phoneme_vocab)?,
    )?;
    let phone_vocab = build_phone_vocab([&train[..], &valid[..], &test[..]].concat().as_slice());
    write_text_atomic(
        &out.join("phone_vocab.json"),
        serde_json::to_string_pretty(&phone_vocab)?,
    )?;
    let word_vocab = build_word_vocab([&train[..], &valid[..], &test[..]].concat().as_slice());
    write_text_atomic(
        &out.join("word_vocab.json"),
        serde_json::to_string_pretty(&word_vocab)?,
    )?;
    for (name, vocab) in feature_vocabs() {
        write_text_atomic(
            &out.join(format!("{name}_vocab.json")),
            serde_json::to_string_pretty(&vocab)?,
        )?;
    }
    let syntax_pos_vocab =
        build_syntax_pos_vocab([&train[..], &valid[..], &test[..]].concat().as_slice());
    write_text_atomic(
        &out.join("syntax_pos_vocab.json"),
        serde_json::to_string_pretty(&syntax_pos_vocab)?,
    )?;
    let syntax_link_vocab =
        build_syntax_link_vocab([&train[..], &valid[..], &test[..]].concat().as_slice());
    write_text_atomic(
        &out.join("syntax_link_vocab.json"),
        serde_json::to_string_pretty(&syntax_link_vocab)?,
    )?;
    let syntax_head_offset_vocab =
        build_syntax_head_offset_vocab([&train[..], &valid[..], &test[..]].concat().as_slice());
    write_text_atomic(
        &out.join("syntax_head_offset_vocab.json"),
        serde_json::to_string_pretty(&syntax_head_offset_vocab)?,
    )?;
    write_text_atomic(
        &out.join("dataset_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    write_text_atomic(&out.join("README.md"), dataset_readme(config))?;
    let report = PrepareReport {
        utterances: n,
        train_examples: train.len(),
        valid_examples: valid.len(),
        test_examples: test.len(),
    };
    write_prepare_state(out, "complete", config, n, Some(&report))?;
    Ok(report)
}

fn enrich_row_supervision(
    row: &mut LibriSpeechUtterance,
    detector: &SentenceDetectorDialog,
    config: &InterpretationConfig,
) -> Result<()> {
    row.sentences = sentence_supervision(detector, &row.transcript, row.num_frames, config)?;
    row.repair_examples = repair_supervision(&row.sentences);
    row.word_supervision = word_supervision(&row.sentences);
    row.masked_word_examples = masked_word_examples(&row.word_supervision, &row.transcript);
    Ok(())
}

#[derive(Debug)]
struct PreparedLibriSpeechRow {
    index: usize,
    progress: Vec<PrepareProgress>,
    row: LibriSpeechUtterance,
}

fn process_librispeech_item_without_refiner(
    index: usize,
    item: &TranscriptItem,
    out: &Path,
    config: &InterpretationConfig,
    feature_bins: usize,
    detector: &SentenceDetectorDialog,
) -> Result<PreparedLibriSpeechRow> {
    let mut progress = Vec::new();
    let samples = read_flac_mono(&item.audio_path)?;
    let rel_mel = PathBuf::from("features").join(format!("{}.mel.bin", item.utterance_id));
    let mel_path = out.join(&rel_mel);
    let frames = match recover_feature_frames(&mel_path, feature_bins, config)? {
        Some(frames) => {
            progress.push(PrepareProgress::Reuse {
                utterance_id: item.utterance_id.clone(),
                rows: frames,
                path: mel_path.display().to_string(),
            });
            frames
        }
        None => {
            let features = audio_features(&samples, config);
            write_mel_file(&mel_path, &features, feature_bins)?;
            progress.push(PrepareProgress::Features {
                utterance_id: item.utterance_id.clone(),
                rows: features.len(),
                path: mel_path.display().to_string(),
            });
            features.len()
        }
    };
    progress.push(PrepareProgress::Transcribe {
        utterance_id: item.utterance_id.clone(),
        path: item.audio_path.display().to_string(),
    });
    let transcript = normalize_librispeech_text(&item.transcript);
    let transcript = if transcript.trim().is_empty() {
        normalize_librispeech_text(&item.transcript)
    } else {
        transcript
    };
    let sentences = sentence_supervision(detector, &transcript, frames, config)?;
    let repair_examples = repair_supervision(&sentences);
    let word_supervision = word_supervision(&sentences);
    let masked_word_examples = masked_word_examples(&word_supervision, &transcript);
    Ok(PreparedLibriSpeechRow {
        index,
        progress,
        row: LibriSpeechUtterance {
            utterance_id: item.utterance_id.clone(),
            speaker_id: item.speaker_id.clone(),
            chapter_id: item.chapter_id.clone(),
            audio_path: item.audio_path.display().to_string(),
            mel_path: rel_mel.display().to_string(),
            num_frames: frames,
            sample_rate_hz: config.sample_rate_hz,
            duration_ms: samples.len() as u64 * 1000 / config.sample_rate_hz as u64,
            transcript,
            sentences,
            repair_examples,
            word_supervision,
            masked_word_examples,
        },
    })
}

fn import_wiktionary_audio_rows(
    out: &Path,
    config: &InterpretationConfig,
    feature_bins: usize,
    pool: &rayon::ThreadPool,
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<Vec<LibriSpeechUtterance>> {
    let Some(data_dir) = &config.wiktionary_audio_data_dir else {
        return Ok(Vec::new());
    };
    let data_dir = Path::new(data_dir);
    let patterns_path = data_dir.join("patterns.jsonl");
    let mut pronunciations = load_wiktionary_pronunciation_map(data_dir)?;
    let patterns = read_wiktionary_audio_patterns(&patterns_path, config.max_wiktionary_audio)?;
    progress(PrepareProgress::ImportAudio {
        source: patterns_path.display().to_string(),
        rows: patterns.len(),
    });
    let audio_dir = out.join("source").join("wiktionary-audio");
    fs::create_dir_all(&audio_dir).with_context(|| format!("creating {}", audio_dir.display()))?;
    let mut jobs = Vec::new();
    for (index, pattern) in patterns.into_iter().enumerate() {
        let key = normalize_audio_key(&pattern.spelling);
        let Some(bundle) = pronunciations.remove(&key) else {
            continue;
        };
        let Some(filename) = pattern.values.first() else {
            continue;
        };
        let Some(url) = commons_upload_url(filename) else {
            continue;
        };
        let utterance_id = format!("wiktionary-audio-{index:06}");
        let audio_path = audio_dir.join(safe_audio_filename(index, filename));
        jobs.push(WiktionaryAudioJob {
            index,
            utterance_id,
            lang: pattern.lang,
            spelling: pattern.spelling,
            audio_path,
            url,
            bundle,
        });
    }
    let download_lock = Mutex::new(());
    let mut prepared = pool.install(|| {
        jobs.par_iter()
            .map(|job| process_wiktionary_audio_job(job, out, config, feature_bins, &download_lock))
            .collect::<Result<Vec<_>>>()
    })?;
    prepared.sort_by_key(|item| item.index);
    let mut rows = Vec::with_capacity(prepared.len());
    for prepared_row in prepared {
        for event in prepared_row.progress {
            progress(event);
        }
        if let Some(row) = prepared_row.row {
            rows.push(row);
        }
    }
    Ok(rows)
}

fn process_wiktionary_audio_job(
    job: &WiktionaryAudioJob,
    out: &Path,
    config: &InterpretationConfig,
    feature_bins: usize,
    download_lock: &Mutex<()>,
) -> Result<PreparedWiktionaryAudioRow> {
    let mut progress = Vec::new();
    if (!job.audio_path.exists() || job.audio_path.metadata()?.len() == 0)
        && config.download_wiktionary_audio
    {
        let mut download_progress = |event| progress.push(event);
        let _download_guard = download_lock
            .lock()
            .map_err(|err| anyhow::anyhow!("wiktionary audio download lock was poisoned: {err}"))?;
        thread::sleep(WIKTIONARY_AUDIO_DOWNLOAD_THROTTLE);
        if let Err(err) = download_to_part(&job.url, &job.audio_path, &mut download_progress) {
            let part = atomic_part_path(&job.audio_path);
            let _ = fs::remove_file(&part);
            progress.push(PrepareProgress::Omit {
                utterance_id: job.utterance_id.clone(),
                reason: format!("could not download Wiktionary audio {}: {err:#}", job.url),
                source_transcript: None,
                whisper_transcript: None,
                wer: None,
                max_wer: None,
            });
            return Ok(PreparedWiktionaryAudioRow {
                index: job.index,
                progress,
                row: None,
            });
        }
    }
    if !job.audio_path.exists() || job.audio_path.metadata()?.len() == 0 {
        progress.push(PrepareProgress::Omit {
            utterance_id: job.utterance_id.clone(),
            reason: format!(
                "wiktionary audio not downloaded; pass --download-wiktionary-audio for {}",
                job.url
            ),
            source_transcript: None,
            whisper_transcript: None,
            wer: None,
            max_wer: None,
        });
        return Ok(PreparedWiktionaryAudioRow {
            index: job.index,
            progress,
            row: None,
        });
    }
    let samples = match read_audio_mono_16k(&job.audio_path) {
        Ok(samples) => samples,
        Err(err) => {
            progress.push(PrepareProgress::Omit {
                utterance_id: job.utterance_id.clone(),
                reason: format!("could not decode {}: {err:#}", job.audio_path.display()),
                source_transcript: None,
                whisper_transcript: None,
                wer: None,
                max_wer: None,
            });
            return Ok(PreparedWiktionaryAudioRow {
                index: job.index,
                progress,
                row: None,
            });
        }
    };
    let rel_mel = PathBuf::from("features").join(format!("{}.mel.bin", job.utterance_id));
    let mel_path = out.join(&rel_mel);
    let frames = match recover_feature_frames(&mel_path, feature_bins, config)? {
        Some(frames) => frames,
        None => {
            let features = audio_features(&samples, config);
            write_mel_file(&mel_path, &features, feature_bins)?;
            progress.push(PrepareProgress::Features {
                utterance_id: job.utterance_id.clone(),
                rows: features.len(),
                path: mel_path.display().to_string(),
            });
            features.len()
        }
    };
    let transcript = normalize_asr_transcript(&job.spelling);
    let phones = job
        .bundle
        .narrow
        .clone()
        .unwrap_or_else(|| job.bundle.broad.clone());
    let sentence = SentenceSupervision {
        text: transcript.clone(),
        start_char: 0,
        end_char: transcript.len(),
        start_frame: 0,
        end_frame: frames,
        boundary_label: BOUNDARY_EMIT.to_string(),
        terminal: None,
        phonemes: job.bundle.broad.clone(),
        phones,
        phoneme_tokens: Vec::new(),
        phone_tokens: Vec::new(),
        syllables: Vec::new(),
        boundaries: Vec::new(),
        prosody: ProsodyTrack::default(),
        warnings: Vec::new(),
        syntax: SyntaxSupervision::default(),
    };
    let sentences = vec![sentence];
    let word_supervision = word_supervision(&sentences);
    Ok(PreparedWiktionaryAudioRow {
        index: job.index,
        progress,
        row: Some(LibriSpeechUtterance {
            utterance_id: job.utterance_id.clone(),
            speaker_id: "wiktionary".to_string(),
            chapter_id: job.lang.clone(),
            audio_path: job.audio_path.display().to_string(),
            mel_path: rel_mel.display().to_string(),
            num_frames: frames,
            sample_rate_hz: config.sample_rate_hz,
            duration_ms: samples.len() as u64 * 1000 / config.sample_rate_hz as u64,
            transcript: transcript.clone(),
            repair_examples: Vec::new(),
            masked_word_examples: masked_word_examples(&word_supervision, &transcript),
            word_supervision,
            sentences,
        }),
    })
}

#[derive(Debug, Clone)]
struct WiktionaryAudioJob {
    index: usize,
    utterance_id: String,
    lang: String,
    spelling: String,
    audio_path: PathBuf,
    url: String,
    bundle: WiktionaryPronunciationBundle,
}

#[derive(Debug)]
struct PreparedWiktionaryAudioRow {
    index: usize,
    progress: Vec<PrepareProgress>,
    row: Option<LibriSpeechUtterance>,
}

fn prepare_worker_threads() -> usize {
    let detected = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let configured = env::var("TONGUES_PREPARE_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(detected);
    configured.clamp(1, DEFAULT_PREPARE_MAX_THREADS)
}

fn discover_transcripts_with_pool(
    root: &Path,
    pool: &rayon::ThreadPool,
) -> Result<Vec<TranscriptItem>> {
    let mut transcript_files = Vec::new();
    collect_files(root, "trans.txt", &mut transcript_files)?;
    let batches = pool.install(|| {
        transcript_files
            .par_iter()
            .map(|path| parse_transcript_file(path))
            .collect::<Result<Vec<_>>>()
    })?;
    let mut out = Vec::new();
    for mut batch in batches {
        out.append(&mut batch);
    }
    out.sort_by(|left, right| left.utterance_id.cmp(&right.utterance_id));
    Ok(out)
}

fn parse_transcript_file(path: &Path) -> Result<Vec<TranscriptItem>> {
    let parent = path.parent().context("transcript path has no parent")?;
    let file = File::open(path)?;
    let mut out = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        let Some((id, text)) = line.split_once(' ') else {
            continue;
        };
        let mut parts = id.split('-');
        let speaker_id = parts.next().unwrap_or("").to_string();
        let chapter_id = parts.next().unwrap_or("").to_string();
        let audio_path = parent.join(format!("{id}.flac"));
        if audio_path.exists() {
            out.push(TranscriptItem {
                utterance_id: id.to_string(),
                speaker_id,
                chapter_id,
                transcript: text.to_string(),
                audio_path,
            });
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
struct WiktionaryPronunciationBundle {
    broad: String,
    narrow: Option<String>,
}

fn load_wiktionary_pronunciation_map(
    data_dir: &Path,
) -> Result<BTreeMap<String, WiktionaryPronunciationBundle>> {
    let mut map = BTreeMap::new();
    load_wiktionary_pronunciation_file(&data_dir.join("phonemes.jsonl"), true, &mut map)?;
    load_wiktionary_pronunciation_file(&data_dir.join("phones.jsonl"), false, &mut map)?;
    Ok(map)
}

fn load_wiktionary_pronunciation_file(
    path: &Path,
    broad: bool,
    map: &mut BTreeMap<String, WiktionaryPronunciationBundle>,
) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: tongues_wiktionary::PronunciationEntry =
            serde_json::from_str(&line).with_context(|| format!("parsing {}", path.display()))?;
        let key = normalize_audio_key(&entry.spelling);
        let ipa = strip_ipa_delimiters(&entry.ipa);
        let bundle = map
            .entry(key)
            .or_insert_with(|| WiktionaryPronunciationBundle {
                broad: ipa.clone(),
                narrow: None,
            });
        if broad {
            bundle.broad = ipa;
        } else {
            bundle.narrow.get_or_insert(ipa);
        }
    }
    Ok(())
}

fn read_wiktionary_audio_patterns(
    path: &Path,
    max_rows: Option<usize>,
) -> Result<Vec<tongues_wiktionary::WiktionaryPattern>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut out = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let pattern: tongues_wiktionary::WiktionaryPattern =
            serde_json::from_str(&line).with_context(|| format!("parsing {}", path.display()))?;
        if pattern.kind == "audio" && !pattern.values.is_empty() {
            out.push(pattern);
            if out.len() >= max_rows.unwrap_or(usize::MAX) {
                break;
            }
        }
    }
    Ok(out)
}

fn read_audio_mono_16k(path: &Path) -> Result<Vec<f32>> {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("flac") => read_flac_mono(path),
        Some("wav") => read_wav_mono_16k(path),
        _ => read_audio_mono_16k_with_ffmpeg(path),
    }
}

fn read_wav_mono_16k(path: &Path) -> Result<Vec<f32>> {
    let audio =
        tongues_audio::read_wav(path).with_context(|| format!("opening {}", path.display()))?;
    anyhow::ensure!(
        audio.sample_rate_hz == DEFAULT_SAMPLE_RATE_HZ,
        "expected 16 kHz WAV"
    );
    audio.to_mono().map_err(anyhow::Error::from)
}

fn read_audio_mono_16k_with_ffmpeg(path: &Path) -> Result<Vec<f32>> {
    let output = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(path)
        .arg("-f")
        .arg("f32le")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg(DEFAULT_SAMPLE_RATE_HZ.to_string())
        .arg("pipe:1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("running ffmpeg for {}", path.display()))?;
    anyhow::ensure!(
        output.status.success(),
        "ffmpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut samples = Vec::with_capacity(output.stdout.len() / 4);
    for chunk in output.stdout.chunks_exact(4) {
        samples.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(samples)
}

fn commons_upload_url(filename_or_title: &str) -> Option<String> {
    let filename = filename_or_title
        .strip_prefix("File:")
        .unwrap_or(filename_or_title)
        .replace(' ', "_");
    if filename.is_empty() {
        return None;
    }
    let digest = format!("{:x}", md5::compute(filename.as_bytes()));
    Some(format!(
        "https://upload.wikimedia.org/wikipedia/commons/{}/{}/{}",
        &digest[0..1],
        &digest[0..2],
        percent_encode(filename.as_bytes())
    ))
}

fn safe_audio_filename(index: usize, filename_or_title: &str) -> String {
    let filename = filename_or_title
        .strip_prefix("File:")
        .unwrap_or(filename_or_title);
    let safe = filename
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '.' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_ascii_lowercase();
    format!(
        "{index:06}-{}",
        if safe.is_empty() { "audio" } else { &safe }
    )
}

fn normalize_audio_key(value: &str) -> String {
    value
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn strip_ipa_delimiters(value: &str) -> String {
    value
        .trim()
        .trim_start_matches('/')
        .trim_start_matches('[')
        .trim_end_matches('/')
        .trim_end_matches(']')
        .to_string()
}

fn percent_encode(bytes: &[u8]) -> String {
    let mut out = String::new();
    for &byte in bytes {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

#[derive(Debug)]
struct TranscriptItem {
    utterance_id: String,
    speaker_id: String,
    chapter_id: String,
    transcript: String,
    audio_path: PathBuf,
}

fn download_to_part(
    url: &str,
    path: &Path,
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<()> {
    let part = atomic_part_path(path);
    archive_interrupted_part(path)?;
    let response = download_response_with_retry(url, progress)?;
    let mut reader = response.into_body().into_reader();
    let mut writer = BufWriter::new(File::create(&part)?);
    let mut buf = [0u8; 64 * 1024];
    let mut bytes = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n])?;
        bytes += n as u64;
        if bytes < 512 * 1024 || bytes % (16 * 1024 * 1024) < 64 * 1024 {
            progress(PrepareProgress::Download {
                url: url.to_string(),
                path: part.display().to_string(),
                bytes,
            });
        }
    }
    writer.flush()?;
    drop(writer);
    fs::rename(&part, path)?;
    Ok(())
}

fn download_response_with_retry(
    url: &str,
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<ureq::http::Response<ureq::Body>> {
    for attempt in 1..=DOWNLOAD_MAX_ATTEMPTS {
        match ureq::get(url)
            .header("User-Agent", DOWNLOAD_USER_AGENT)
            .config()
            .http_status_as_error(false)
            .build()
            .call()
        {
            Ok(response) => {
                let status = response.status().as_u16();
                if status < 400 {
                    return Ok(response);
                }
                if should_retry_download_status(status) && attempt < DOWNLOAD_MAX_ATTEMPTS {
                    let delay =
                        retry_after_delay(&response).unwrap_or_else(|| retry_delay(attempt));
                    progress(PrepareProgress::Stage {
                        message: format!(
                            "Download got HTTP {status} for {url}; retrying in {}s ({attempt}/{DOWNLOAD_MAX_ATTEMPTS})",
                            delay.as_secs().max(1)
                        ),
                    });
                    thread::sleep(delay);
                } else {
                    anyhow::bail!("downloading {url}: http status: {status}");
                }
            }
            Err(err) if attempt < DOWNLOAD_MAX_ATTEMPTS => {
                let delay = retry_delay(attempt);
                progress(PrepareProgress::Stage {
                    message: format!(
                        "Download transport error for {url}: {err}; retrying in {}s ({attempt}/{DOWNLOAD_MAX_ATTEMPTS})",
                        delay.as_secs().max(1)
                    ),
                });
                thread::sleep(delay);
            }
            Err(err) => return Err(err).with_context(|| format!("downloading {url}")),
        }
    }
    unreachable!("download retry loop always returns on the final attempt")
}

fn should_retry_download_status(status: u16) -> bool {
    status == 429 || (500..600).contains(&status)
}

fn retry_after_delay(response: &ureq::http::Response<ureq::Body>) -> Option<Duration> {
    response
        .headers()
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(|seconds| Duration::from_secs(seconds.clamp(1, 120)))
}

fn retry_delay(attempt: usize) -> Duration {
    let seconds = match attempt {
        0 | 1 => 2,
        2 => 5,
        3 => 10,
        4 => 20,
        _ => 30,
    };
    Duration::from_secs(seconds)
}

fn discover_transcripts(root: &Path) -> Result<Vec<TranscriptItem>> {
    let mut transcript_files = Vec::new();
    collect_files(root, "trans.txt", &mut transcript_files)?;
    let mut out = Vec::new();
    for path in transcript_files {
        out.extend(parse_transcript_file(&path)?);
    }
    Ok(out)
}

fn collect_files(root: &Path, suffix: &str, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, suffix, out)?;
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(suffix))
        {
            out.push(path);
        }
    }
    Ok(())
}

fn read_flac_mono(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        claxon::FlacReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let info = reader.streaminfo();
    anyhow::ensure!(
        info.sample_rate == DEFAULT_SAMPLE_RATE_HZ,
        "expected 16 kHz FLAC"
    );
    let channels = info.channels as usize;
    let max = ((1i64 << (info.bits_per_sample - 1)) - 1) as f32;
    let mut samples = Vec::new();
    let mut acc = 0.0f32;
    let mut channel = 0usize;
    for sample in reader.samples() {
        acc += sample? as f32 / max;
        channel += 1;
        if channel == channels {
            samples.push(acc / channels as f32);
            acc = 0.0;
            channel = 0;
        }
    }
    Ok(samples)
}

pub fn normalize_librispeech_text(text: &str) -> String {
    text.chars()
        .map(|ch| match ch {
            'a'..='z' => ch.to_ascii_uppercase(),
            'A'..='Z' | '\'' | ' ' | '.' | '?' | '!' => ch,
            ',' | ';' | ':' => ' ',
            _ => ' ',
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn normalize_asr_transcript(text: &str) -> String {
    text.chars()
        .filter_map(|ch| match ch {
            ch if ch.is_control() => Some(' '),
            '\u{2018}' | '\u{2019}' => Some('\''),
            '\u{201c}' | '\u{201d}' => Some('"'),
            '\u{2013}' | '\u{2014}' => Some('-'),
            ch => Some(ch),
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn log_mel_features(samples: &[f32], config: &InterpretationConfig) -> Vec<Vec<f32>> {
    frame_spectral_features(samples, config)
        .into_iter()
        .map(|frame| frame.log_mel)
        .collect()
}

pub fn audio_features(samples: &[f32], config: &InterpretationConfig) -> Vec<Vec<f32>> {
    if !config.compact_audio_features {
        return log_mel_features(samples, config);
    }
    let frames = frame_spectral_features(samples, config);
    let mut previous_mel: Option<Vec<f32>> = None;
    let mut previous_power: Option<Vec<f32>> = None;
    let mut rows = Vec::with_capacity(frames.len());
    for frame in frames {
        let delta = previous_mel
            .as_ref()
            .map(|prev| {
                frame
                    .log_mel
                    .iter()
                    .zip(prev)
                    .map(|(current, previous)| current - previous)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![0.0; frame.log_mel.len()]);
        let spectral_flux = previous_power
            .as_ref()
            .map(|prev| {
                frame
                    .power
                    .iter()
                    .zip(prev)
                    .map(|(current, previous)| (current.sqrt() - previous.sqrt()).max(0.0))
                    .sum::<f32>()
                    / frame.power.len().max(1) as f32
            })
            .unwrap_or(0.0);
        let mut row = Vec::with_capacity(DEFAULT_COMPACT_AUDIO_FEATURE_BINS);
        row.extend(frame.log_mel.iter().copied());
        row.extend(delta);
        row.push(frame.rms_energy);
        row.push(frame.vad);
        row.push(frame.zcr);
        row.push(frame.spectral_centroid);
        row.push(spectral_flux);
        row.push(frame.f0);
        row.push(frame.voiced_prob);
        previous_mel = Some(frame.log_mel);
        previous_power = Some(frame.power);
        rows.push(row);
    }
    rows
}

#[derive(Debug)]
struct FrameSpectralFeatures {
    log_mel: Vec<f32>,
    power: Vec<f32>,
    rms_energy: f32,
    vad: f32,
    zcr: f32,
    spectral_centroid: f32,
    f0: f32,
    voiced_prob: f32,
}

fn frame_spectral_features(
    samples: &[f32],
    config: &InterpretationConfig,
) -> Vec<FrameSpectralFeatures> {
    let window = ((config.sample_rate_hz as f32 * config.window_ms) / 1000.0).round() as usize;
    let hop = ((config.sample_rate_hz as f32 * config.hop_ms) / 1000.0).round() as usize;
    if samples.len() < window || window == 0 || hop == 0 {
        return Vec::new();
    }
    let n_fft = window.next_power_of_two();
    let stft_config = tongues_audio::StftConfig {
        fft_size: n_fft,
        window_size: window,
        hop_size: hop,
        center: false,
        pad_mode: tongues_audio::PadMode::Constant,
        window: tongues_audio::Window::Hann,
    };
    let Ok(spectrum) = tongues_audio::stft(samples, &stft_config) else {
        return Vec::new();
    };
    let mel_config = tongues_audio::MelConfig {
        bins: config.mel_bins,
        min_frequency_hz: 0.0,
        max_frequency_hz: Some(config.sample_rate_hz as f32 / 2.0),
        scale: tongues_audio::MelScale::Slaney,
        normalization: tongues_audio::MelNormalization::Slaney,
    };
    let Ok(mel_weights) = tongues_audio::mel_filter_bank(config.sample_rate_hz, n_fft, &mel_config)
    else {
        return Vec::new();
    };
    let spectral_bins = spectrum.bins_per_frame();
    let mut rows = Vec::with_capacity(spectrum.frames);
    for frame_index in 0..spectrum.frames {
        let start = frame_index * hop;
        let complex =
            &spectrum.bins[frame_index * spectral_bins..(frame_index + 1) * spectral_bins];
        let power = complex
            .iter()
            .map(|bin| bin.re * bin.re + bin.im * bin.im)
            .collect::<Vec<_>>();
        let frame = &samples[start..start + window];
        let rms_energy = (frame.iter().map(|sample| sample * sample).sum::<f32>()
            / window.max(1) as f32)
            .sqrt()
            .ln_1p();
        let peak_energy = frame
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0f32, f32::max);
        let crossings = frame
            .windows(2)
            .filter(|pair| (pair[0] >= 0.0) != (pair[1] >= 0.0))
            .count();
        let zcr = crossings as f32 / window.max(1) as f32;
        let power_sum = power.iter().sum::<f32>().max(1e-8);
        let spectral_centroid = power
            .iter()
            .enumerate()
            .map(|(bin, value)| {
                let hz = bin as f32 * config.sample_rate_hz as f32 / n_fft as f32;
                hz * *value
            })
            .sum::<f32>()
            / power_sum
            / (config.sample_rate_hz as f32 * 0.5);
        let (f0, voiced_prob) = estimate_pitch(frame, config.sample_rate_hz);
        let log_mel = mel_weights
            .chunks_exact(spectral_bins)
            .map(|weights| {
                weights
                    .iter()
                    .zip(&power)
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>()
                    .max(1e-8)
                    .ln()
            })
            .collect();
        rows.push(FrameSpectralFeatures {
            log_mel,
            power,
            rms_energy,
            vad: if peak_energy > 0.01 || rms_energy > 0.001 {
                1.0
            } else {
                0.0
            },
            zcr,
            spectral_centroid,
            f0,
            voiced_prob,
        });
    }
    rows
}

fn estimate_pitch(frame: &[f32], sample_rate_hz: u32) -> (f32, f32) {
    if frame.is_empty() {
        return (0.0, 0.0);
    }
    let min_lag = (sample_rate_hz as f32 / 400.0).round().max(1.0) as usize;
    let max_lag = (sample_rate_hz as f32 / 60.0).round().max(min_lag as f32) as usize;
    let frame_energy = frame
        .iter()
        .map(|sample| sample * sample)
        .sum::<f32>()
        .max(1e-8);
    let mut best_lag = 0usize;
    let mut best_score = 0.0f32;
    for lag in min_lag..=max_lag.min(frame.len().saturating_sub(1)) {
        let score = frame
            .iter()
            .zip(frame.iter().skip(lag))
            .map(|(left, right)| left * right)
            .sum::<f32>()
            / frame_energy;
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }
    if best_lag == 0 || best_score < 0.25 {
        (0.0, best_score.clamp(0.0, 1.0))
    } else {
        (
            (sample_rate_hz as f32 / best_lag as f32) / 400.0,
            best_score.clamp(0.0, 1.0),
        )
    }
}

pub fn audio_feature_bins(config: &InterpretationConfig) -> usize {
    if config.compact_audio_features {
        config.mel_bins + config.mel_bins + COMPACT_AUDIO_EXTRA_BINS
    } else {
        config.mel_bins
    }
}

fn write_mel_file(path: &Path, features: &[Vec<f32>], mel_bins: usize) -> Result<()> {
    let part = atomic_part_path(path);
    archive_interrupted_part(path)?;
    let mut writer = BufWriter::new(File::create(&part)?);
    writer.write_all(b"TONGUES_MEL1")?;
    writer.write_all(&(features.len() as u32).to_le_bytes())?;
    writer.write_all(&(mel_bins as u32).to_le_bytes())?;
    for row in features {
        for value in row {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    writer.flush()?;
    drop(writer);
    fs::rename(part, path)?;
    Ok(())
}

fn valid_mel_frames(path: &Path, mel_bins: usize) -> Result<Option<usize>> {
    if !path.exists() {
        return Ok(None);
    }
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 12];
    if reader.read_exact(&mut magic).is_err() {
        return Ok(None);
    }
    if &magic != b"TONGUES_MEL1" {
        return Ok(None);
    }
    let mut buf = [0u8; 4];
    if reader.read_exact(&mut buf).is_err() {
        return Ok(None);
    }
    let frames = u32::from_le_bytes(buf) as usize;
    if reader.read_exact(&mut buf).is_err() {
        return Ok(None);
    }
    let bins = u32::from_le_bytes(buf) as usize;
    if bins != mel_bins {
        return Ok(None);
    }
    let expected_len = 12_u64 + 4 + 4 + frames as u64 * bins as u64 * 4;
    if fs::metadata(path)?.len() != expected_len {
        return Ok(None);
    }
    Ok(Some(frames))
}

fn recover_feature_frames(
    path: &Path,
    expected_bins: usize,
    config: &InterpretationConfig,
) -> Result<Option<usize>> {
    if let Some(frames) = valid_mel_frames(path, expected_bins)? {
        return Ok(Some(frames));
    }
    if !config.compact_audio_features || expected_bins == config.mel_bins {
        return Ok(None);
    }
    let Some(frames) = valid_mel_frames(path, config.mel_bins)? else {
        return Ok(None);
    };
    let mel = read_mel_file(path)?;
    let compact = compact_features_from_log_mel(&mel, config);
    write_mel_file(path, &compact, expected_bins)?;
    Ok(Some(frames))
}

fn compact_features_from_log_mel(mel: &[Vec<f32>], config: &InterpretationConfig) -> Vec<Vec<f32>> {
    let mut previous_mel: Option<&Vec<f32>> = None;
    mel.iter()
        .map(|row| {
            let delta = previous_mel
                .map(|prev| {
                    row.iter()
                        .zip(prev)
                        .map(|(current, previous)| current - previous)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec![0.0; config.mel_bins]);
            let energy = row.iter().copied().sum::<f32>() / row.len().max(1) as f32;
            let linear_energy = energy.exp().ln_1p();
            let vad = if energy > -12.0 { 1.0 } else { 0.0 };
            let spectral_centroid = mel_centroid(row);
            let spectral_flux = previous_mel
                .map(|prev| {
                    row.iter()
                        .zip(prev)
                        .map(|(current, previous)| (current.exp() - previous.exp()).max(0.0))
                        .sum::<f32>()
                        / row.len().max(1) as f32
                })
                .unwrap_or(0.0);
            let mut out = Vec::with_capacity(audio_feature_bins(config));
            out.extend(row.iter().copied());
            out.extend(delta);
            out.push(linear_energy);
            out.push(vad);
            out.push(0.0);
            out.push(spectral_centroid);
            out.push(spectral_flux);
            out.push(0.0);
            out.push(0.0);
            previous_mel = Some(row);
            out
        })
        .collect()
}

fn mel_centroid(row: &[f32]) -> f32 {
    let weights = row.iter().map(|value| value.exp()).collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>().max(1e-8);
    weights
        .iter()
        .enumerate()
        .map(|(idx, weight)| idx as f32 / row.len().max(1) as f32 * *weight)
        .sum::<f32>()
        / total
}

pub fn read_mel_file(path: &Path) -> Result<Vec<Vec<f32>>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 12];
    reader.read_exact(&mut magic)?;
    anyhow::ensure!(&magic == b"TONGUES_MEL1", "invalid Mel feature file");
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let frames = u32::from_le_bytes(buf) as usize;
    reader.read_exact(&mut buf)?;
    let bins = u32::from_le_bytes(buf) as usize;
    let mut out = vec![vec![0.0; bins]; frames];
    for row in &mut out {
        for value in row {
            let mut raw = [0u8; 4];
            reader.read_exact(&mut raw)?;
            *value = f32::from_le_bytes(raw);
        }
    }
    Ok(out)
}

pub fn feature_file_shape(path: &Path) -> Result<(usize, usize)> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 12];
    reader.read_exact(&mut magic)?;
    anyhow::ensure!(&magic == b"TONGUES_MEL1", "invalid Mel feature file");
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let frames = u32::from_le_bytes(buf) as usize;
    reader.read_exact(&mut buf)?;
    let bins = u32::from_le_bytes(buf) as usize;
    Ok((frames, bins))
}

fn sentence_supervision(
    detector: &SentenceDetectorDialog,
    transcript: &str,
    num_frames: usize,
    config: &InterpretationConfig,
) -> Result<Vec<SentenceSupervision>> {
    let detected = detector
        .detect_sentences_borrowed(transcript)
        .context("detecting transcript sentences")?;
    let variety = VarietyId(config.variety.clone());
    let phonemicizer = phonemicizer_for_variety(&variety)?;
    let syntax_parser = VarietyGrammarParser::default();
    let mut offset = 0usize;
    let mut out = Vec::new();
    for sentence in detected {
        let text = sentence.normalize();
        let start = transcript[offset..]
            .find(&text)
            .map(|idx| offset + idx)
            .unwrap_or(offset);
        let end = start + text.len();
        offset = end.min(transcript.len());
        let start_frame = char_to_frame(start, transcript.len(), num_frames);
        let end_frame = char_to_frame(end, transcript.len(), num_frames).max(start_frame + 1);
        let phonemicized = phonemicizer.phonemicize(&PhonemicizeRequest {
            text: text.clone(),
            variety: variety.clone(),
            style: None,
        })?;
        let terminal = text.chars().rev().find(|ch| matches!(ch, '.' | '?' | '!'));
        let syntax = syntax_supervision(&syntax_parser, &text, terminal);
        out.push(SentenceSupervision {
            terminal,
            text,
            start_char: start,
            end_char: end,
            start_frame,
            end_frame: end_frame.min(num_frames),
            boundary_label: BOUNDARY_EMIT.to_string(),
            phonemes: syllables_to_ipa(&phonemicized.syllables),
            phones: phones_string(&phonemicized.phones),
            phoneme_tokens: phonemicized.phonemes,
            phone_tokens: phonemicized.phones,
            syllables: phonemicized.syllables,
            boundaries: phonemicized.boundaries,
            prosody: phonemicized.prosody,
            warnings: phonemicized.warnings,
            syntax,
        });
    }
    if out.is_empty() && !transcript.trim().is_empty() {
        let phonemicized = phonemicizer.phonemicize(&PhonemicizeRequest {
            text: transcript.to_string(),
            variety,
            style: None,
        })?;
        let syntax = syntax_supervision(&syntax_parser, transcript, None);
        out.push(SentenceSupervision {
            text: transcript.to_string(),
            start_char: 0,
            end_char: transcript.len(),
            start_frame: 0,
            end_frame: num_frames,
            boundary_label: BOUNDARY_EMIT.to_string(),
            terminal: None,
            phonemes: syllables_to_ipa(&phonemicized.syllables),
            phones: phones_string(&phonemicized.phones),
            phoneme_tokens: phonemicized.phonemes,
            phone_tokens: phonemicized.phones,
            syllables: phonemicized.syllables,
            boundaries: phonemicized.boundaries,
            prosody: phonemicized.prosody,
            warnings: phonemicized.warnings,
            syntax,
        });
    }
    Ok(out)
}

fn syntax_supervision(
    parser: &impl GrammarParser,
    sentence: &str,
    terminal: Option<char>,
) -> SyntaxSupervision {
    let word_spans = word_spans(sentence);
    let words = word_spans
        .iter()
        .map(|(_, _, word)| word.clone())
        .collect::<Vec<_>>();
    if words.is_empty() {
        return SyntaxSupervision::default();
    }
    let analysis = parser.parse(&words, terminal_punctuation(terminal));
    let primary = analysis.primary_parse();
    let parse_rank = primary.map(|parse| parse.rank).unwrap_or(0.0);
    let links = primary
        .map(|parse| {
            parse
                .links
                .iter()
                .map(|link| SyntaxLinkSupervision {
                    left: link.left,
                    right: link.right,
                    label: syntax_link_label(link.kind),
                    confidence: link.confidence,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let parse_cost = links
        .iter()
        .map(|link| 1.0 - link.confidence.clamp(0.0, 1.0))
        .sum::<f32>();
    let parse_ok = !links.is_empty();
    let supervision_weight = if parse_ok {
        (parse_rank / (1.0 + parse_cost)).clamp(0.1, 1.0)
    } else {
        0.0
    };
    let syntax_words = words
        .iter()
        .enumerate()
        .map(|(index, word)| {
            let token = analysis
                .tokens
                .iter()
                .find(|token| token.word_index == index);
            let mut link_labels = token
                .map(|token| {
                    token
                        .syntactic_links
                        .iter()
                        .map(|kind| syntax_link_label(*kind))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            link_labels.sort();
            link_labels.dedup();
            let primary_link = primary.and_then(|parse| {
                parse
                    .links
                    .iter()
                    .filter(|link| link.left == index || link.right == index)
                    .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
            });
            let linked_word_index = primary_link.map(|link| {
                if link.left == index {
                    link.right
                } else {
                    link.left
                }
            });
            let head_offset = linked_word_index
                .map(|linked| linked as i32 - index as i32)
                .unwrap_or(0);
            SyntaxWordSupervision {
                word: word.clone(),
                sentence_word_index: index,
                pos: token
                    .map(|token| syntax_pos_label(token.pos))
                    .unwrap_or_else(|| "unknown".to_string()),
                primary_link_label: primary_link
                    .map(|link| syntax_link_label(link.kind))
                    .unwrap_or_else(|| "none".to_string()),
                link_labels,
                linked_word_index,
                head_offset,
                phrase_boundary: syntax_phrase_boundary(
                    index,
                    words.len(),
                    primary_link.map(|l| l.kind),
                ),
            }
        })
        .collect();
    SyntaxSupervision {
        words: syntax_words,
        links,
        parse_ok,
        parse_rank,
        parse_cost,
        supervision_weight,
        analysis,
    }
}

fn terminal_punctuation(terminal: Option<char>) -> Option<TerminalPunctuation> {
    match terminal {
        Some('.') => Some(TerminalPunctuation::Period),
        Some('?') => Some(TerminalPunctuation::Question),
        Some('!') => Some(TerminalPunctuation::Exclamation),
        _ => None,
    }
}

fn syntax_pos_label(pos: PartOfSpeech) -> String {
    format!("{pos:?}").to_ascii_lowercase()
}

fn syntax_link_label(kind: SyntacticLinkKind) -> String {
    format!("{kind:?}").to_ascii_lowercase()
}

fn syntax_phrase_boundary(
    index: usize,
    words: usize,
    link_kind: Option<SyntacticLinkKind>,
) -> bool {
    index + 1 == words
        || matches!(
            link_kind,
            Some(
                SyntacticLinkKind::Preposition
                    | SyntacticLinkKind::Coordination
                    | SyntacticLinkKind::ContrastPair
                    | SyntacticLinkKind::Apposition
                    | SyntacticLinkKind::Parenthetical
            )
        )
}

fn phones_string(phones: &[speaking::PhoneToken]) -> String {
    phones
        .iter()
        .filter_map(|token| match &token.phone {
            speaking::Spec::Known(id) => phone_training_symbol(id).map(str::to_string),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn phone_training_symbol(id: &speaking::PhoneId) -> Option<&str> {
    id.as_str()
        .strip_prefix("ipa.phone.")
        .filter(|symbol| !symbol.is_empty())
}

fn repair_supervision(sentences: &[SentenceSupervision]) -> Vec<RepairSupervision> {
    let mut repairs = Vec::new();
    for sentence in sentences {
        if let Some(misheard) = mishear_sentence(&sentence.text) {
            repairs.push(RepairSupervision {
                misheard_text: misheard,
                corrected_text: sentence.text.clone(),
                start_char: sentence.start_char,
                end_char: sentence.end_char,
                start_frame: sentence.start_frame,
                end_frame: sentence.end_frame,
                repair_label: BOUNDARY_REPAIR.to_string(),
                source: "synthetic-mishear".to_string(),
            });
        }
    }
    repairs
}

fn mishear_sentence(text: &str) -> Option<String> {
    const SUBSTITUTIONS: &[(&str, &str)] = &[
        (" TO ", " TWO "),
        (" TWO ", " TO "),
        (" FOR ", " FOUR "),
        (" FOUR ", " FOR "),
        (" THERE ", " THEIR "),
        (" THEIR ", " THERE "),
        (" YOUR ", " YOU'RE "),
        (" YOU'RE ", " YOUR "),
        (" ITS ", " IT'S "),
        (" IT'S ", " ITS "),
        (" NO ", " KNOW "),
        (" KNOW ", " NO "),
        (" RIGHT ", " WRITE "),
        (" WRITE ", " RIGHT "),
        (" HEAR ", " HERE "),
        (" HERE ", " HEAR "),
    ];
    let padded = format!(" {text} ");
    for (from, to) in SUBSTITUTIONS {
        if padded.contains(from) {
            return Some(padded.replacen(from, to, 1).trim().to_string());
        }
    }
    let words = text.split_whitespace().collect::<Vec<_>>();
    if words.len() >= 4 {
        let mut edited = words;
        let index = edited.len() / 2;
        edited.remove(index);
        return Some(edited.join(" "));
    }
    None
}

fn word_supervision(sentences: &[SentenceSupervision]) -> Vec<WordSupervision> {
    let mut words = Vec::new();
    for (sentence_index, sentence) in sentences.iter().enumerate() {
        let sentence_words = word_spans(&sentence.text);
        let phoneme_chunks = phoneme_chunks_for_words(sentence, sentence_words.len());
        let phone_chunks = phone_chunks_for_words(sentence, sentence_words.len());
        for (sentence_word_index, (start, end, word)) in sentence_words.into_iter().enumerate() {
            let global_start = sentence.start_char + start;
            let global_end = sentence.start_char + end;
            let start_frame = char_to_frame(
                global_start,
                sentence.end_char.max(1),
                sentence.end_frame.max(1),
            )
            .max(sentence.start_frame);
            let end_frame = char_to_frame(
                global_end,
                sentence.end_char.max(1),
                sentence.end_frame.max(1),
            )
            .max(start_frame + 1)
            .min(sentence.end_frame.max(start_frame + 1));
            words.push(WordSupervision {
                word,
                word_index: words.len(),
                sentence_index,
                sentence_word_index,
                start_char: global_start,
                end_char: global_end,
                start_frame,
                end_frame,
                phonemes: phoneme_chunks
                    .get(sentence_word_index)
                    .cloned()
                    .unwrap_or_default(),
                phones: phone_chunks
                    .get(sentence_word_index)
                    .cloned()
                    .unwrap_or_default(),
                previous_word: None,
                next_word: None,
            });
        }
    }
    let word_texts = words
        .iter()
        .map(|word| word.word.clone())
        .collect::<Vec<_>>();
    for (index, word) in words.iter_mut().enumerate() {
        word.previous_word = index
            .checked_sub(1)
            .and_then(|previous| word_texts.get(previous))
            .cloned();
        word.next_word = word_texts.get(index + 1).cloned();
    }
    words
}

fn word_spans(text: &str) -> Vec<(usize, usize, String)> {
    let mut spans = Vec::new();
    let mut start = None;
    for (index, ch) in text.char_indices() {
        if ch.is_alphanumeric() || ch == '\'' {
            start.get_or_insert(index);
        } else if let Some(start_index) = start.take() {
            spans.push((start_index, index, text[start_index..index].to_string()));
        }
    }
    if let Some(start_index) = start {
        spans.push((start_index, text.len(), text[start_index..].to_string()));
    }
    spans
}

fn phoneme_chunks_for_words(sentence: &SentenceSupervision, words: usize) -> Vec<String> {
    let chunks = chunks_from_phoneme_tokens(&sentence.phoneme_tokens, words);
    if chunks.iter().any(|chunk| !chunk.is_empty()) {
        chunks
    } else {
        split_tokens_for_words(&sentence.phonemes, words)
    }
}

fn phone_chunks_for_words(sentence: &SentenceSupervision, words: usize) -> Vec<String> {
    let chunks = chunks_from_phone_tokens(&sentence.phone_tokens, words);
    if chunks.iter().any(|chunk| !chunk.is_empty()) {
        chunks
    } else {
        split_tokens_for_words(&sentence.phones, words)
    }
}

fn chunks_from_phoneme_tokens(tokens: &[speaking::PhonemeToken], words: usize) -> Vec<String> {
    let mut chunks = vec![Vec::new(); words];
    for token in tokens {
        let Some(word_index) = token_word_index(&token.features) else {
            continue;
        };
        let Some(chunk) = chunks.get_mut(word_index) else {
            continue;
        };
        if let speaking::Spec::Known(id) = &token.phoneme {
            chunk.push(speaking::phoneme_display_symbol(id).to_string());
        }
    }
    chunks
        .into_iter()
        .map(|chunk| chunk.join(" "))
        .collect::<Vec<_>>()
}

fn chunks_from_phone_tokens(tokens: &[speaking::PhoneToken], words: usize) -> Vec<String> {
    let mut chunks = vec![Vec::new(); words];
    for token in tokens {
        let Some(word_index) = token_word_index(&token.features) else {
            continue;
        };
        let Some(chunk) = chunks.get_mut(word_index) else {
            continue;
        };
        if let speaking::Spec::Known(id) = &token.phone {
            if let Some(symbol) = phone_training_symbol(id) {
                chunk.push(symbol.to_string());
            }
        }
    }
    chunks
        .into_iter()
        .map(|chunk| chunk.join(" "))
        .collect::<Vec<_>>()
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

pub fn normalize_word_token(word: &str) -> Option<String> {
    let mut normalized = word
        .chars()
        .map(|ch| match ch {
            '\u{2018}' | '\u{2019}' | '\u{02bc}' | '\u{0060}' | '\u{00b4}' => '\'',
            _ => ch,
        })
        .collect::<String>()
        .trim_matches(|ch: char| {
            ch == '\''
                || ch == '"'
                || ch == '`'
                || ch == '\u{2018}'
                || ch == '\u{2019}'
                || ch == '\u{201c}'
                || ch == '\u{201d}'
        })
        .to_lowercase();
    normalized.retain(|ch| ch.is_alphanumeric() || ch == '\'' || ch == '-');
    if normalized.is_empty() {
        return None;
    }
    if is_numeric_word(&normalized) {
        return Some(WORD_NUM.to_string());
    }
    if normalized.chars().any(char::is_alphabetic) && normalized.chars().count() <= 40 {
        Some(normalized)
    } else {
        None
    }
}

fn is_numeric_word(word: &str) -> bool {
    let base = word
        .strip_suffix("st")
        .or_else(|| word.strip_suffix("nd"))
        .or_else(|| word.strip_suffix("rd"))
        .or_else(|| word.strip_suffix("th"))
        .unwrap_or(word);
    !base.is_empty() && base.chars().all(|ch| ch.is_ascii_digit())
}

fn split_tokens_for_words(tokens: &str, words: usize) -> Vec<String> {
    if words == 0 {
        return Vec::new();
    }
    let tokens = tokens.split_whitespace().collect::<Vec<_>>();
    (0..words)
        .map(|word| {
            let start = word * tokens.len() / words;
            let end = ((word + 1) * tokens.len() / words)
                .max(start + 1)
                .min(tokens.len());
            tokens[start.min(tokens.len())..end.min(tokens.len())].join(" ")
        })
        .collect()
}

fn masked_word_examples(words: &[WordSupervision], transcript: &str) -> Vec<MaskedWordExample> {
    let mut out = Vec::new();
    let mut by_sentence: BTreeMap<usize, Vec<&WordSupervision>> = BTreeMap::new();
    for word in words {
        by_sentence
            .entry(word.sentence_index)
            .or_default()
            .push(word);
    }
    for sentence_words in by_sentence.values() {
        if sentence_words.len() < 3 {
            continue;
        }
        let masked = sentence_words[sentence_words.len() / 2];
        out.push(MaskedWordExample {
            left_context: transcript[..masked.start_char.min(transcript.len())]
                .trim()
                .to_string(),
            right_context: transcript[masked.end_char.min(transcript.len())..]
                .trim()
                .to_string(),
            masked_word: masked.word.clone(),
            masked_word_phonemes: masked.phonemes.clone(),
            start_frame: masked.start_frame,
            end_frame: masked.end_frame,
            source: "deterministic-middle-word".to_string(),
        });
    }
    out
}

fn char_to_frame(char_index: usize, chars: usize, frames: usize) -> usize {
    if chars == 0 {
        0
    } else {
        ((char_index as f64 / chars as f64) * frames as f64).round() as usize
    }
}

fn write_jsonl_atomic<T: Serialize>(
    path: &Path,
    rows: &[T],
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<()> {
    let part = atomic_part_path(path);
    archive_interrupted_part(path)?;
    let mut writer = BufWriter::new(File::create(&part)?);
    for row in rows {
        writeln!(writer, "{}", serde_json::to_string(row)?)?;
    }
    writer.flush()?;
    drop(writer);
    fs::rename(&part, path)?;
    progress(PrepareProgress::Write {
        path: path.display().to_string(),
        rows: rows.len(),
    });
    Ok(())
}

fn write_text_atomic(path: &Path, contents: impl AsRef<str>) -> Result<()> {
    let part = atomic_part_path(path);
    archive_interrupted_part(path)?;
    let mut writer = BufWriter::new(
        File::create(&part).with_context(|| format!("creating {}", part.display()))?,
    );
    writer
        .write_all(contents.as_ref().as_bytes())
        .with_context(|| format!("writing {}", part.display()))?;
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

fn write_prepare_state(
    out: &Path,
    status: &str,
    config: &InterpretationConfig,
    utterances: usize,
    report: Option<&PrepareReport>,
) -> Result<()> {
    let state = PrepareCheckpointState {
        status: status.to_string(),
        dataset_id: config.dataset_id.clone(),
        utterances,
        report: report.cloned(),
    };
    write_text_atomic(
        &out.join("prepare_state.json"),
        serde_json::to_string_pretty(&state)?,
    )
}

pub fn read_examples(path: &Path) -> Result<Vec<LibriSpeechUtterance>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut rows = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if !line.trim().is_empty() {
            rows.push(serde_json::from_str(&line)?);
        }
    }
    Ok(rows)
}

fn recover_utterance_rows(
    path: &Path,
    data_dir: &Path,
    config: &InterpretationConfig,
) -> Result<Vec<LibriSpeechUtterance>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(row) = serde_json::from_str::<LibriSpeechUtterance>(&line) else {
            continue;
        };
        if !seen.insert(row.utterance_id.clone()) {
            continue;
        }
        let mel_path = data_dir.join(&row.mel_path);
        let expected_bins = audio_feature_bins(config);
        let Some(frames) = recover_feature_frames(&mel_path, expected_bins, config)? else {
            continue;
        };
        if frames == row.num_frames && row.sample_rate_hz == config.sample_rate_hz {
            rows.push(row);
        }
    }
    Ok(rows)
}

pub fn build_text_vocab(rows: &[LibriSpeechUtterance]) -> Vocab {
    let mut tokens = vec![
        CTC_BLANK.to_string(),
        BOUNDARY_CONTINUE.to_string(),
        BOUNDARY_EMIT.to_string(),
        BOUNDARY_REPAIR.to_string(),
    ];
    let mut chars = BTreeSet::new();
    for row in rows {
        for ch in row.transcript.chars() {
            chars.insert(ch.to_string());
        }
    }
    tokens.extend(chars);
    let token_to_id = tokens
        .iter()
        .enumerate()
        .map(|(idx, token)| (token.clone(), idx as u32))
        .collect();
    Vocab {
        tokens,
        token_to_id,
    }
}

pub fn build_phoneme_vocab(rows: &[LibriSpeechUtterance]) -> Vocab {
    let mut tokens = vec![CTC_BLANK.to_string()];
    let mut set = BTreeSet::new();
    for row in rows {
        for sentence in &row.sentences {
            for token in sentence_phoneme_labels(sentence) {
                set.insert(token.to_string());
            }
        }
    }
    tokens.extend(set);
    let token_to_id = tokens
        .iter()
        .enumerate()
        .map(|(idx, token)| (token.clone(), idx as u32))
        .collect();
    Vocab {
        tokens,
        token_to_id,
    }
}

pub fn build_phone_vocab(rows: &[LibriSpeechUtterance]) -> Vocab {
    let mut tokens = vec![CTC_BLANK.to_string()];
    let mut set = BTreeSet::new();
    for row in rows {
        for sentence in &row.sentences {
            for token in sentence.phones.split_whitespace() {
                set.insert(token.to_string());
            }
        }
    }
    tokens.extend(set);
    let token_to_id = tokens
        .iter()
        .enumerate()
        .map(|(idx, token)| (token.clone(), idx as u32))
        .collect();
    Vocab {
        tokens,
        token_to_id,
    }
}

pub fn build_word_vocab(rows: &[LibriSpeechUtterance]) -> Vocab {
    let mut tokens = vec![
        WORD_BLANK.to_string(),
        WORD_UNK.to_string(),
        WORD_NUM.to_string(),
    ];
    let mut counts = BTreeMap::new();
    for row in rows {
        for word in &row.word_supervision {
            if let Some(word) = normalize_word_token(&word.word) {
                *counts.entry(word).or_insert(0usize) += 1;
            }
        }
        for masked in &row.masked_word_examples {
            if let Some(word) = normalize_word_token(&masked.masked_word) {
                *counts.entry(word).or_insert(0usize) += 1;
            }
        }
    }
    let mut ranked = counts
        .into_iter()
        .filter(|(token, count)| token == WORD_NUM || *count >= MIN_WORD_VOCAB_COUNT)
        .collect::<Vec<_>>();
    ranked.sort_by(|(left_token, left_count), (right_token, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_token.cmp(right_token))
    });
    tokens.extend(
        ranked
            .into_iter()
            .filter_map(|(token, _)| (token != WORD_NUM).then_some(token))
            .take(MAX_WORD_VOCAB_TOKENS),
    );
    let token_to_id = tokens
        .iter()
        .enumerate()
        .map(|(idx, token)| (token.clone(), idx as u32))
        .collect();
    Vocab {
        tokens,
        token_to_id,
    }
}

pub fn build_syntax_pos_vocab(rows: &[LibriSpeechUtterance]) -> Vocab {
    let mut tokens = vec!["<PAD>".to_string(), "unknown".to_string()];
    let mut set = BTreeSet::new();
    for row in rows {
        for sentence in &row.sentences {
            for word in &sentence.syntax.words {
                set.insert(word.pos.clone());
            }
        }
    }
    tokens.extend(set.into_iter().filter(|token| token != "unknown"));
    let token_to_id = tokens
        .iter()
        .enumerate()
        .map(|(idx, token)| (token.clone(), idx as u32))
        .collect();
    Vocab {
        tokens,
        token_to_id,
    }
}

pub fn build_syntax_link_vocab(rows: &[LibriSpeechUtterance]) -> Vocab {
    let mut tokens = vec!["<PAD>".to_string(), "none".to_string()];
    let mut set = BTreeSet::new();
    for row in rows {
        for sentence in &row.sentences {
            for word in &sentence.syntax.words {
                set.insert(word.primary_link_label.clone());
                set.extend(word.link_labels.iter().cloned());
            }
            for link in &sentence.syntax.links {
                set.insert(link.label.clone());
            }
        }
    }
    tokens.extend(set.into_iter().filter(|token| token != "none"));
    let token_to_id = tokens
        .iter()
        .enumerate()
        .map(|(idx, token)| (token.clone(), idx as u32))
        .collect();
    Vocab {
        tokens,
        token_to_id,
    }
}

pub fn build_syntax_head_offset_vocab(rows: &[LibriSpeechUtterance]) -> Vocab {
    let mut tokens = vec!["<PAD>".to_string()];
    let mut set = BTreeSet::new();
    set.insert("0".to_string());
    for row in rows {
        for sentence in &row.sentences {
            for word in &sentence.syntax.words {
                set.insert(syntax_head_offset_label(word.head_offset));
            }
        }
    }
    tokens.extend(set);
    let token_to_id = tokens
        .iter()
        .enumerate()
        .map(|(idx, token)| (token.clone(), idx as u32))
        .collect();
    Vocab {
        tokens,
        token_to_id,
    }
}

fn fixed_vocab(tokens: &[&str]) -> Vocab {
    let tokens = tokens
        .iter()
        .map(|token| token.to_string())
        .collect::<Vec<_>>();
    let token_to_id = tokens
        .iter()
        .enumerate()
        .map(|(idx, token)| (token.clone(), idx as u32))
        .collect();
    Vocab {
        tokens,
        token_to_id,
    }
}

pub fn feature_vocabs() -> Vec<(&'static str, Vocab)> {
    vec![
        (
            "place",
            fixed_vocab(&[
                CTC_BLANK, "none", "labial", "coronal", "dorsal", "front", "back", "central",
            ]),
        ),
        (
            "manner",
            fixed_vocab(&[
                CTC_BLANK,
                "none",
                "stop",
                "fricative",
                "nasal",
                "approximant",
                "vowel",
            ]),
        ),
        (
            "voicing",
            fixed_vocab(&[CTC_BLANK, "none", "-voice", "+voice"]),
        ),
        (
            "syllabic",
            fixed_vocab(&[CTC_BLANK, "none", "-syllabic", "+syllabic"]),
        ),
        (
            "height",
            fixed_vocab(&[CTC_BLANK, "none", "high", "mid", "low", "nonvowel"]),
        ),
        (
            "backness",
            fixed_vocab(&[CTC_BLANK, "none", "front", "central", "back"]),
        ),
        (
            "rounding",
            fixed_vocab(&[CTC_BLANK, "none", "-round", "+round"]),
        ),
    ]
}

fn syntax_head_offset_label(offset: i32) -> String {
    offset.clamp(-7, 7).to_string()
}

pub fn save_artifact_files(
    out: &Path,
    data: &Path,
    model_config: &ModelConfig,
    train_config: &InterpretationTrainConfig,
) -> Result<()> {
    fs::create_dir_all(out)?;
    fs::copy(data.join("vocab.json"), out.join("vocab.json"))?;
    fs::copy(
        data.join("phoneme_vocab.json"),
        out.join("phoneme_vocab.json"),
    )?;
    fs::copy(data.join("phone_vocab.json"), out.join("phone_vocab.json"))?;
    fs::copy(data.join("word_vocab.json"), out.join("word_vocab.json"))?;
    for (name, vocab) in feature_vocabs() {
        let src = data.join(format!("{name}_vocab.json"));
        let dst = out.join(format!("{name}_vocab.json"));
        if src.exists() {
            fs::copy(src, dst)?;
        } else {
            fs::write(dst, serde_json::to_string_pretty(&vocab)?)?;
        }
    }
    fs::copy(
        data.join("syntax_pos_vocab.json"),
        out.join("syntax_pos_vocab.json"),
    )?;
    fs::copy(
        data.join("syntax_link_vocab.json"),
        out.join("syntax_link_vocab.json"),
    )?;
    fs::copy(
        data.join("syntax_head_offset_vocab.json"),
        out.join("syntax_head_offset_vocab.json"),
    )?;
    fs::write(
        out.join("model_config.json"),
        serde_json::to_string_pretty(model_config)?,
    )?;
    fs::write(
        out.join("train_config.json"),
        serde_json::to_string_pretty(train_config)?,
    )?;
    write_manifest(
        out,
        &ModelArtifactManifest::new(FAMILY, ARCHITECTURE, data.display().to_string())
            .with_task("streaming-asr-boundary-phoneme"),
    )?;
    Ok(())
}

pub fn load_model<B: Backend>(
    model_config: &ModelConfig,
    model_dir: &Path,
    device: &B::Device,
) -> Result<AsrModel<B>>
where
    <AsrModel<B> as Module<B>>::Record: Send,
{
    model_config
        .init(device)
        .load_file(&model_dir.join("model"), &make_recorder(), device)
        .context("loading LibriSpeech ASR model")
}

pub fn train<B: AutodiffBackend, R: Rng>(
    model_config: &ModelConfig,
    train_config: &InterpretationTrainConfig,
    data_dir: &Path,
    train_rows: &[LibriSpeechUtterance],
    valid_rows: &[LibriSpeechUtterance],
    vocab: &Vocab,
    phoneme_vocab: &Vocab,
    phone_vocab: &Vocab,
    word_vocab: &Vocab,
    syntax_pos_vocab: &Vocab,
    syntax_link_vocab: &Vocab,
    syntax_head_offset_vocab: &Vocab,
    model_path: &Path,
    device: &B::Device,
    rng: &mut R,
) -> Result<f32>
where
    <AsrModel<B> as Module<B>>::Record: Send,
{
    let out_dir = model_path.parent().unwrap_or(Path::new("."));
    let state_path = out_dir.join("train_state.json");
    let mut start_epoch = 1usize;
    let mut best_val_loss = f32::INFINITY;
    let mut best_epoch = None;
    let mut last_finite_epoch = None;
    let mut optimizer_resume_epoch = None;
    let mut resume_without_optimizer = false;
    let mut model = if state_path.exists() {
        let state: InterpretationTrainState =
            serde_json::from_str(&fs::read_to_string(&state_path)?)?;
        start_epoch = state.current_epoch + 1;
        best_val_loss = state.best_val_loss;
        best_epoch = state.best_epoch;
        last_finite_epoch = state.last_finite_epoch;
        let epoch_path = out_dir.join(format!("model-epoch-{}", state.current_epoch));
        if epoch_path.with_extension("bin").exists() {
            println!(
                "Resuming training from epoch {} checkpoint: {}",
                state.current_epoch,
                epoch_path.with_extension("bin").display()
            );
            let candidate =
                model_config
                    .init(device)
                    .load_file(&epoch_path, &make_recorder(), device)?;
            let resume_report = evaluate(
                &candidate.valid(),
                data_dir,
                valid_rows,
                vocab,
                phoneme_vocab,
                phone_vocab,
                word_vocab,
                syntax_pos_vocab,
                syntax_link_vocab,
                syntax_head_offset_vocab,
                train_config,
                device,
            )?;
            if report_is_finite(&resume_report) {
                optimizer_resume_epoch = Some(state.current_epoch);
                candidate
            } else if model_path.with_extension("bin").exists() {
                println!(
                    "Checkpoint {} produced non-finite validation metrics; returning to best model {}",
                    epoch_path.with_extension("bin").display(),
                    model_path.with_extension("bin").display()
                );
                let recovered_epoch = if best_epoch.is_some() {
                    best_epoch
                } else {
                    match recover_best_epoch(
                        model_config,
                        out_dir,
                        state.current_epoch,
                        best_val_loss,
                        data_dir,
                        valid_rows,
                        vocab,
                        phoneme_vocab,
                        phone_vocab,
                        word_vocab,
                        syntax_pos_vocab,
                        syntax_link_vocab,
                        syntax_head_offset_vocab,
                        train_config,
                        device,
                    ) {
                        Ok(epoch) => epoch,
                        Err(err) => {
                            println!("Could not inspect older checkpoints for recovery: {err:#}");
                            None
                        }
                    }
                };
                best_epoch = recovered_epoch;
                last_finite_epoch = recovered_epoch.or(last_finite_epoch);
                if let Some(epoch) = recovered_epoch {
                    start_epoch = epoch + 1;
                    optimizer_resume_epoch = Some(epoch);
                    write_interpretation_train_state(
                        &state_path,
                        epoch,
                        best_val_loss,
                        best_epoch,
                        last_finite_epoch,
                    )?;
                }
                model_config
                    .init(device)
                    .load_file(model_path, &make_recorder(), device)?
            } else {
                println!(
                    "Checkpoint {} produced non-finite validation metrics and no best model exists; initializing new weights",
                    epoch_path.with_extension("bin").display()
                );
                start_epoch = 1;
                best_val_loss = f32::INFINITY;
                best_epoch = None;
                last_finite_epoch = None;
                model_config.init(device)
            }
        } else {
            if model_path.with_extension("bin").exists() {
                println!(
                    "Epoch checkpoint not found. Resuming training from best model: {}",
                    model_path.with_extension("bin").display()
                );
                model_config
                    .init(device)
                    .load_file(model_path, &make_recorder(), device)?
            } else {
                model_config.init(device)
            }
        }
    } else {
        model_config.init(device)
    };
    if start_epoch > train_config.epochs {
        return Ok(best_val_loss);
    }
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(train_config.weight_decay)
        .init::<B, AsrModel<B>>();
    let mut loaded_optimizer_checkpoint = None;
    if let Some(epoch) = optimizer_resume_epoch {
        let optimizer_path = out_dir.join(format!("optim-epoch-{epoch}"));
        let optimizer_bin = optimizer_path.with_extension("bin");
        if optimizer_bin.exists() {
            let record = make_recorder()
                .load(optimizer_path.clone(), device)
                .with_context(|| {
                    format!("loading {}", optimizer_path.with_extension("bin").display())
                })?;
            optimizer = optimizer.load_record(record);
            loaded_optimizer_checkpoint = Some(optimizer_bin.clone());
            println!("Resuming optimizer state from {}", optimizer_bin.display());
        } else {
            resume_without_optimizer = true;
            println!(
                "No optimizer checkpoint found for epoch {epoch}; first resumed epoch will use lr={} ({} * {})",
                train_config.learning_rate * train_config.resume_learning_rate_scale,
                train_config.learning_rate,
                train_config.resume_learning_rate_scale
            );
        }
    }
    let mut patience = 0usize;
    for epoch in start_epoch..=train_config.epochs {
        let learning_rate = if resume_without_optimizer {
            train_config.learning_rate * train_config.resume_learning_rate_scale
        } else {
            train_config.learning_rate
        };
        resume_without_optimizer = false;
        let loss = train_epoch(
            &mut model,
            &mut optimizer,
            train_config,
            data_dir,
            train_rows,
            vocab,
            phoneme_vocab,
            phone_vocab,
            word_vocab,
            syntax_pos_vocab,
            syntax_link_vocab,
            syntax_head_offset_vocab,
            device,
            rng,
            epoch,
            learning_rate,
        )?;
        if !loss.is_finite() {
            println!(
                "Epoch {epoch} produced non-finite train_loss={loss}; not saving checkpoint or advancing train_state"
            );
            if model_path.with_extension("bin").exists() {
                println!(
                    "Preserving best model for the next run: {}",
                    model_path.with_extension("bin").display()
                );
            }
            if let Some(path) = loaded_optimizer_checkpoint.take() {
                quarantine_optimizer_checkpoint(&path)?;
            }
            if let Some(epoch) = best_epoch.or(last_finite_epoch) {
                write_interpretation_train_state(
                    &state_path,
                    epoch,
                    best_val_loss,
                    best_epoch,
                    last_finite_epoch,
                )?;
            }
            break;
        }
        let eval_model = model.valid();
        let report = evaluate(
            &eval_model,
            data_dir,
            valid_rows,
            vocab,
            phoneme_vocab,
            phone_vocab,
            word_vocab,
            syntax_pos_vocab,
            syntax_link_vocab,
            syntax_head_offset_vocab,
            train_config,
            device,
        )?;
        if !report_is_finite(&report) {
            println!(
                "Epoch {epoch} produced non-finite validation metrics; not saving checkpoint or advancing train_state"
            );
            if model_path.with_extension("bin").exists() {
                println!(
                    "Preserving best model for the next run: {}",
                    model_path.with_extension("bin").display()
                );
            }
            if let Some(epoch) = best_epoch.or(last_finite_epoch) {
                write_interpretation_train_state(
                    &state_path,
                    epoch,
                    best_val_loss,
                    best_epoch,
                    last_finite_epoch,
                )?;
            }
            break;
        }
        println!(
            "Epoch {} | train_loss={:.4} val_loss={:.4} wer={:.3} boundary_f1={:.3} repair_f1={:.3} phoneme_ter={:.3} phone_ter={:.3} audio_mse={:.4}",
            format_count(epoch),
            loss,
            report.loss,
            report.word_error_rate,
            report.boundary_f1,
            report.repair_f1,
            report.phoneme_token_error_rate,
            report.phone_token_error_rate,
            report.masked_audio_mse
        );
        eval_model.clone().save_file(
            &out_dir.join(format!("model-epoch-{epoch}")),
            &make_recorder(),
        )?;
        make_recorder()
            .record(
                optimizer.to_record(),
                out_dir.join(format!("optim-epoch-{epoch}")),
            )
            .with_context(|| {
                format!(
                    "writing {}",
                    out_dir
                        .join(format!("optim-epoch-{epoch}"))
                        .with_extension("bin")
                        .display()
                )
            })?;
        last_finite_epoch = Some(epoch);
        write_interpretation_train_state(
            &state_path,
            epoch,
            best_val_loss,
            best_epoch,
            last_finite_epoch,
        )?;
        if report.loss < best_val_loss - 1e-5 {
            best_val_loss = report.loss;
            best_epoch = Some(epoch);
            patience = 0;
            eval_model.save_file(model_path, &make_recorder())?;
            write_interpretation_train_state(
                &state_path,
                epoch,
                best_val_loss,
                best_epoch,
                last_finite_epoch,
            )?;
        } else {
            patience += 1;
            if patience >= train_config.early_stopping_patience {
                break;
            }
        }
    }
    Ok(best_val_loss)
}

fn quarantine_optimizer_checkpoint(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let mut target = path.with_extension("nan.bin");
    let mut suffix = 1usize;
    while target.exists() {
        target = path.with_extension(format!("nan-{suffix}.bin"));
        suffix += 1;
    }
    fs::rename(path, &target).with_context(|| {
        format!(
            "quarantining non-finite optimizer checkpoint {} -> {}",
            path.display(),
            target.display()
        )
    })?;
    println!(
        "Quarantined optimizer checkpoint after non-finite loss: {} -> {}",
        path.display(),
        target.display()
    );
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct InterpretationTrainState {
    current_epoch: usize,
    best_val_loss: f32,
    #[serde(default)]
    best_epoch: Option<usize>,
    #[serde(default)]
    last_finite_epoch: Option<usize>,
}

fn write_interpretation_train_state(
    path: &Path,
    current_epoch: usize,
    best_val_loss: f32,
    best_epoch: Option<usize>,
    last_finite_epoch: Option<usize>,
) -> Result<()> {
    let state = InterpretationTrainState {
        current_epoch,
        best_val_loss,
        best_epoch,
        last_finite_epoch,
    };
    fs::write(path, serde_json::to_string_pretty(&state)?)
        .with_context(|| format!("writing {}", path.display()))
}

fn report_is_finite(report: &EvalReport) -> bool {
    report.loss.is_finite()
        && report.token_error_rate.is_finite()
        && report.word_error_rate.is_finite()
        && report.seq2seq_token_error_rate.is_finite()
        && report.boundary_f1.is_finite()
        && report.repair_f1.is_finite()
        && report.phoneme_token_error_rate.is_finite()
        && report.phone_token_error_rate.is_finite()
        && report.masked_audio_mse.is_finite()
        && report.prev_word_accuracy.is_finite()
        && report.current_word_accuracy.is_finite()
        && report.next_word_accuracy.is_finite()
        && report.masked_word_accuracy.is_finite()
        && report.masked_word_phoneme_token_error_rate.is_finite()
}

#[allow(clippy::too_many_arguments)]
fn recover_best_epoch<B: AutodiffBackend>(
    model_config: &ModelConfig,
    out_dir: &Path,
    current_epoch: usize,
    best_val_loss: f32,
    data_dir: &Path,
    valid_rows: &[LibriSpeechUtterance],
    vocab: &Vocab,
    phoneme_vocab: &Vocab,
    phone_vocab: &Vocab,
    word_vocab: &Vocab,
    syntax_pos_vocab: &Vocab,
    syntax_link_vocab: &Vocab,
    syntax_head_offset_vocab: &Vocab,
    train_config: &InterpretationTrainConfig,
    device: &B::Device,
) -> Result<Option<usize>>
where
    <AsrModel<B> as Module<B>>::Record: Send,
{
    for epoch in (1..=current_epoch).rev() {
        let checkpoint = out_dir.join(format!("model-epoch-{epoch}"));
        if !checkpoint.with_extension("bin").exists() {
            continue;
        }
        let model: AsrModel<B> =
            model_config
                .init(device)
                .load_file(&checkpoint, &make_recorder(), device)?;
        let report = evaluate(
            &model.valid(),
            data_dir,
            valid_rows,
            vocab,
            phoneme_vocab,
            phone_vocab,
            word_vocab,
            syntax_pos_vocab,
            syntax_link_vocab,
            syntax_head_offset_vocab,
            train_config,
            device,
        )?;
        if report_is_finite(&report) && report.loss <= best_val_loss + 1e-3 {
            println!(
                "Recovered best finite epoch {} with val_loss={:.4}",
                epoch, report.loss
            );
            return Ok(Some(epoch));
        }
    }
    Ok(None)
}

fn train_epoch<B: AutodiffBackend, R: Rng>(
    model: &mut AsrModel<B>,
    optimizer: &mut impl Optimizer<AsrModel<B>, B>,
    config: &InterpretationTrainConfig,
    data_dir: &Path,
    rows: &[LibriSpeechUtterance],
    vocab: &Vocab,
    phoneme_vocab: &Vocab,
    phone_vocab: &Vocab,
    word_vocab: &Vocab,
    syntax_pos_vocab: &Vocab,
    syntax_link_vocab: &Vocab,
    syntax_head_offset_vocab: &Vocab,
    device: &B::Device,
    rng: &mut R,
    epoch: usize,
    learning_rate: f64,
) -> Result<f32> {
    let mut indices: Vec<_> = (0..rows.len()).collect();
    indices.shuffle(rng);
    let batches = (rows.len() + config.batch_size - 1) / config.batch_size;
    let pb = tongues_core::register_progress_bar(indicatif::ProgressBar::new(batches as u64));
    let template = format!(
        "{{spinner:.green}} LibriSpeech epoch {}/{} [{{elapsed_precise}}] [{{bar:40.cyan/blue}}] {{human_pos}}/{{human_len}} ETA {{eta_precise}} loss={{msg}}",
        format_count(epoch),
        format_count(config.epochs)
    );
    pb.set_style(counted_progress_style(&template));
    let mut total = 0.0;
    let mut n = 0usize;
    for chunk in indices.chunks(config.batch_size) {
        let batch_rows = chunk.iter().map(|&i| rows[i].clone()).collect::<Vec<_>>();
        let batch = make_batch::<B>(
            data_dir,
            &batch_rows,
            vocab,
            phoneme_vocab,
            phone_vocab,
            word_vocab,
            syntax_pos_vocab,
            syntax_link_vocab,
            syntax_head_offset_vocab,
            config,
            device,
        )?;
        let output = model.forward(batch.mel.clone());
        let loss = weighted_loss(output, batch, config);
        let loss_val = loss.clone().into_scalar().elem::<f32>();
        if !loss_val.is_finite() {
            pb.finish_and_clear();
            return Ok(loss_val);
        }
        let grads = GradientsParams::from_grads(loss.backward(), model);
        *model = optimizer.step(learning_rate, model.clone(), grads);
        total += loss_val;
        n += 1;
        pb.set_message(format!("{:.4}", total / n as f32));
        pb.inc(1);
    }
    pb.finish_and_clear();
    Ok(if n == 0 { 0.0 } else { total / n as f32 })
}

#[derive(Debug)]
struct AsrBatch<B: Backend> {
    mel: Tensor<B, 3>,
    mel_target: Tensor<B, 3>,
    transcript_labels: Tensor<B, 2, Int>,
    seq2seq_labels: Tensor<B, 2, Int>,
    boundary_labels: Tensor<B, 2, Int>,
    phoneme_labels: Tensor<B, 2, Int>,
    phone_labels: Tensor<B, 2, Int>,
    input_lengths: Tensor<B, 1, Int>,
    phoneme_targets: Tensor<B, 2, Int>,
    phoneme_target_lengths: Tensor<B, 1, Int>,
    phone_targets: Tensor<B, 2, Int>,
    phone_target_lengths: Tensor<B, 1, Int>,
    place_targets: Tensor<B, 2, Int>,
    place_target_lengths: Tensor<B, 1, Int>,
    manner_targets: Tensor<B, 2, Int>,
    manner_target_lengths: Tensor<B, 1, Int>,
    voicing_targets: Tensor<B, 2, Int>,
    voicing_target_lengths: Tensor<B, 1, Int>,
    syllabic_targets: Tensor<B, 2, Int>,
    syllabic_target_lengths: Tensor<B, 1, Int>,
    height_targets: Tensor<B, 2, Int>,
    height_target_lengths: Tensor<B, 1, Int>,
    backness_targets: Tensor<B, 2, Int>,
    backness_target_lengths: Tensor<B, 1, Int>,
    rounding_targets: Tensor<B, 2, Int>,
    rounding_target_lengths: Tensor<B, 1, Int>,
    prev_word_targets: Tensor<B, 2, Int>,
    prev_word_target_lengths: Tensor<B, 1, Int>,
    current_word_targets: Tensor<B, 2, Int>,
    current_word_target_lengths: Tensor<B, 1, Int>,
    next_word_targets: Tensor<B, 2, Int>,
    next_word_target_lengths: Tensor<B, 1, Int>,
    masked_word_targets: Tensor<B, 2, Int>,
    masked_word_target_lengths: Tensor<B, 1, Int>,
    masked_word_phoneme_targets: Tensor<B, 2, Int>,
    masked_word_phoneme_target_lengths: Tensor<B, 1, Int>,
    syntax_pos_labels: Tensor<B, 2, Int>,
    syntax_link_labels: Tensor<B, 2, Int>,
    syntax_head_offset_labels: Tensor<B, 2, Int>,
    parse_ok_labels: Tensor<B, 2, Int>,
    phrase_boundary_labels: Tensor<B, 2, Int>,
}

#[derive(Debug)]
struct AsrBatchRowParts {
    index: usize,
    input_len: i32,
    mel: Vec<f32>,
    mel_target: Vec<f32>,
    transcript_labels: Vec<i32>,
    seq2seq_labels: Vec<i32>,
    boundary_labels: Vec<i32>,
    phoneme_labels: Vec<i32>,
    phone_labels: Vec<i32>,
    syntax_pos_labels: Vec<i32>,
    syntax_link_labels: Vec<i32>,
    syntax_head_offset_labels: Vec<i32>,
    parse_ok_labels: Vec<i32>,
    phrase_boundary_labels: Vec<i32>,
    prev_word_sequence: Vec<i32>,
    current_word_sequence: Vec<i32>,
    next_word_sequence: Vec<i32>,
    masked_word_sequence: Vec<i32>,
    masked_word_phoneme_sequence: Vec<i32>,
    phoneme_sequence: Vec<i32>,
    phone_sequence: Vec<i32>,
    place_sequence: Vec<i32>,
    manner_sequence: Vec<i32>,
    voicing_sequence: Vec<i32>,
    syllabic_sequence: Vec<i32>,
    height_sequence: Vec<i32>,
    backness_sequence: Vec<i32>,
    rounding_sequence: Vec<i32>,
}

#[derive(Debug, Clone, Default)]
struct EvalSampleStats {
    token_errors: usize,
    token_total: usize,
    seq2seq_token_errors: usize,
    seq2seq_token_total: usize,
    word_errors: usize,
    word_total: usize,
    boundary_tp: usize,
    boundary_fp: usize,
    boundary_fn: usize,
    repair_tp: usize,
    repair_fp: usize,
    repair_fn: usize,
    phoneme_errors: usize,
    phoneme_total: usize,
    phone_errors: usize,
    phone_total: usize,
    prev_correct: usize,
    prev_total: usize,
    current_correct: usize,
    current_total: usize,
    next_correct: usize,
    next_total: usize,
    masked_word_correct: usize,
    masked_word_total: usize,
    masked_phoneme_errors: usize,
    masked_phoneme_total: usize,
}

impl EvalSampleStats {
    fn merge(&mut self, other: EvalSampleStats) {
        self.token_errors += other.token_errors;
        self.token_total += other.token_total;
        self.seq2seq_token_errors += other.seq2seq_token_errors;
        self.seq2seq_token_total += other.seq2seq_token_total;
        self.word_errors += other.word_errors;
        self.word_total += other.word_total;
        self.boundary_tp += other.boundary_tp;
        self.boundary_fp += other.boundary_fp;
        self.boundary_fn += other.boundary_fn;
        self.repair_tp += other.repair_tp;
        self.repair_fp += other.repair_fp;
        self.repair_fn += other.repair_fn;
        self.phoneme_errors += other.phoneme_errors;
        self.phoneme_total += other.phoneme_total;
        self.phone_errors += other.phone_errors;
        self.phone_total += other.phone_total;
        self.prev_correct += other.prev_correct;
        self.prev_total += other.prev_total;
        self.current_correct += other.current_correct;
        self.current_total += other.current_total;
        self.next_correct += other.next_correct;
        self.next_total += other.next_total;
        self.masked_word_correct += other.masked_word_correct;
        self.masked_word_total += other.masked_word_total;
        self.masked_phoneme_errors += other.masked_phoneme_errors;
        self.masked_phoneme_total += other.masked_phoneme_total;
    }
}

fn prepare_asr_batch_row(
    index: usize,
    data_dir: &Path,
    row: &LibriSpeechUtterance,
    vocab: &Vocab,
    phoneme_vocab: &Vocab,
    phone_vocab: &Vocab,
    word_vocab: &Vocab,
    syntax_pos_vocab: &Vocab,
    syntax_link_vocab: &Vocab,
    syntax_head_offset_vocab: &Vocab,
    config: &InterpretationTrainConfig,
    max_frames: usize,
    mel_bins: usize,
) -> Result<AsrBatchRowParts> {
    let input_len = row.num_frames.min(max_frames).max(1);
    let features = read_mel_file(&data_dir.join(&row.mel_path))?;
    let mut mel = Vec::with_capacity(max_frames * mel_bins);
    let mut mel_target = Vec::with_capacity(max_frames * mel_bins);
    for frame in 0..max_frames {
        let src = features.get(frame);
        let masked = frame_is_masked(frame, config) || frame_is_word_masked(row, frame, config);
        for bin in 0..mel_bins {
            let value = src.and_then(|r| r.get(bin)).copied().unwrap_or(0.0);
            mel_target.push(value);
            mel.push(if masked { 0.0 } else { value });
        }
    }
    Ok(AsrBatchRowParts {
        index,
        input_len: input_len as i32,
        mel,
        mel_target,
        transcript_labels: proportional_labels(&row.transcript, vocab, max_frames),
        seq2seq_labels: seq2seq_labels_for(
            &row.transcript,
            vocab,
            config.max_seq2seq_tokens.min(max_frames).max(1),
            max_frames,
        ),
        boundary_labels: boundary_labels_for(row, max_frames),
        phoneme_labels: proportional_phoneme_labels(row, phoneme_vocab, max_frames),
        phone_labels: proportional_phone_labels(row, phone_vocab, max_frames),
        syntax_pos_labels: syntax_pos_labels_for(row, syntax_pos_vocab, max_frames),
        syntax_link_labels: syntax_link_labels_for(row, syntax_link_vocab, max_frames),
        syntax_head_offset_labels: syntax_head_offset_labels_for(
            row,
            syntax_head_offset_vocab,
            max_frames,
        ),
        parse_ok_labels: parse_ok_labels_for(row, max_frames),
        phrase_boundary_labels: phrase_boundary_labels_for(row, max_frames),
        phoneme_sequence: ctc_target_within_input(phoneme_targets(row, phoneme_vocab), input_len),
        phone_sequence: ctc_target_within_input(phone_targets(row, phone_vocab), input_len),
        place_sequence: ctc_target_within_input(
            feature_targets(row, FeatureAxis::Place),
            input_len,
        ),
        manner_sequence: ctc_target_within_input(
            feature_targets(row, FeatureAxis::Manner),
            input_len,
        ),
        voicing_sequence: ctc_target_within_input(
            feature_targets(row, FeatureAxis::Voicing),
            input_len,
        ),
        syllabic_sequence: ctc_target_within_input(
            feature_targets(row, FeatureAxis::Syllabic),
            input_len,
        ),
        height_sequence: ctc_target_within_input(
            feature_targets(row, FeatureAxis::Height),
            input_len,
        ),
        backness_sequence: ctc_target_within_input(
            feature_targets(row, FeatureAxis::Backness),
            input_len,
        ),
        rounding_sequence: ctc_target_within_input(
            feature_targets(row, FeatureAxis::Rounding),
            input_len,
        ),
        prev_word_sequence: ctc_target_within_input(
            previous_word_targets(row, word_vocab),
            input_len,
        ),
        current_word_sequence: ctc_target_within_input(
            current_word_targets(row, word_vocab),
            input_len,
        ),
        next_word_sequence: ctc_target_within_input(next_word_targets(row, word_vocab), input_len),
        masked_word_sequence: ctc_target_within_input(
            masked_word_targets(row, word_vocab),
            input_len,
        ),
        masked_word_phoneme_sequence: ctc_target_within_input(
            masked_word_phoneme_targets(row, phoneme_vocab),
            input_len,
        ),
    })
}

fn make_batch<B: Backend>(
    data_dir: &Path,
    rows: &[LibriSpeechUtterance],
    vocab: &Vocab,
    phoneme_vocab: &Vocab,
    phone_vocab: &Vocab,
    word_vocab: &Vocab,
    syntax_pos_vocab: &Vocab,
    syntax_link_vocab: &Vocab,
    syntax_head_offset_vocab: &Vocab,
    config: &InterpretationTrainConfig,
    device: &B::Device,
) -> Result<AsrBatch<B>> {
    let max_frames = rows
        .iter()
        .map(|row| row.num_frames)
        .max()
        .unwrap_or(1)
        .min(config.max_frames)
        .max(1);
    let mel_bins = config.input_feature_bins.max(1);
    let mut mel = Vec::new();
    let mut mel_target = Vec::new();
    let mut transcript_labels = Vec::new();
    let mut seq2seq_labels = Vec::new();
    let mut boundary_labels = Vec::new();
    let mut phoneme_labels = Vec::new();
    let mut phone_labels = Vec::new();
    let mut syntax_pos_labels = Vec::new();
    let mut syntax_link_labels = Vec::new();
    let mut syntax_head_offset_labels = Vec::new();
    let mut parse_ok_labels = Vec::new();
    let mut phrase_boundary_labels = Vec::new();
    let mut input_lengths = Vec::new();
    let mut prev_word_sequences = Vec::new();
    let mut current_word_sequences = Vec::new();
    let mut next_word_sequences = Vec::new();
    let mut masked_word_sequences = Vec::new();
    let mut masked_word_phoneme_sequences = Vec::new();
    let mut phoneme_sequences = Vec::new();
    let mut phone_sequences = Vec::new();
    let mut place_sequences = Vec::new();
    let mut manner_sequences = Vec::new();
    let mut voicing_sequences = Vec::new();
    let mut syllabic_sequences = Vec::new();
    let mut height_sequences = Vec::new();
    let mut backness_sequences = Vec::new();
    let mut rounding_sequences = Vec::new();
    let mut prepared_rows = rows
        .par_iter()
        .enumerate()
        .map(|(index, row)| {
            prepare_asr_batch_row(
                index,
                data_dir,
                row,
                vocab,
                phoneme_vocab,
                phone_vocab,
                word_vocab,
                syntax_pos_vocab,
                syntax_link_vocab,
                syntax_head_offset_vocab,
                config,
                max_frames,
                mel_bins,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    prepared_rows.sort_by_key(|row| row.index);
    for prepared in prepared_rows {
        input_lengths.push(prepared.input_len);
        mel.extend(prepared.mel);
        mel_target.extend(prepared.mel_target);
        transcript_labels.extend(prepared.transcript_labels);
        seq2seq_labels.extend(prepared.seq2seq_labels);
        boundary_labels.extend(prepared.boundary_labels);
        phoneme_labels.extend(prepared.phoneme_labels);
        phone_labels.extend(prepared.phone_labels);
        syntax_pos_labels.extend(prepared.syntax_pos_labels);
        syntax_link_labels.extend(prepared.syntax_link_labels);
        syntax_head_offset_labels.extend(prepared.syntax_head_offset_labels);
        parse_ok_labels.extend(prepared.parse_ok_labels);
        phrase_boundary_labels.extend(prepared.phrase_boundary_labels);
        phoneme_sequences.push(prepared.phoneme_sequence);
        phone_sequences.push(prepared.phone_sequence);
        place_sequences.push(prepared.place_sequence);
        manner_sequences.push(prepared.manner_sequence);
        voicing_sequences.push(prepared.voicing_sequence);
        syllabic_sequences.push(prepared.syllabic_sequence);
        height_sequences.push(prepared.height_sequence);
        backness_sequences.push(prepared.backness_sequence);
        rounding_sequences.push(prepared.rounding_sequence);
        prev_word_sequences.push(prepared.prev_word_sequence);
        current_word_sequences.push(prepared.current_word_sequence);
        next_word_sequences.push(prepared.next_word_sequence);
        masked_word_sequences.push(prepared.masked_word_sequence);
        masked_word_phoneme_sequences.push(prepared.masked_word_phoneme_sequence);
    }
    let (prev_word_targets, prev_word_target_lengths, prev_word_width) =
        pad_compact_targets(prev_word_sequences, word_vocab.get_id(WORD_UNK));
    let (current_word_targets, current_word_target_lengths, current_word_width) =
        pad_compact_targets(current_word_sequences, word_vocab.get_id(WORD_UNK));
    let (next_word_targets, next_word_target_lengths, next_word_width) =
        pad_compact_targets(next_word_sequences, word_vocab.get_id(WORD_UNK));
    let (masked_word_targets, masked_word_target_lengths, masked_word_width) =
        pad_compact_targets(masked_word_sequences, word_vocab.get_id(WORD_UNK));
    let (masked_word_phoneme_targets, masked_word_phoneme_target_lengths, masked_phoneme_width) =
        pad_compact_targets(masked_word_phoneme_sequences, 1);
    let (phoneme_targets, phoneme_target_lengths, phoneme_width) =
        pad_compact_targets(phoneme_sequences, 1);
    let (phone_targets, phone_target_lengths, phone_width) =
        pad_compact_targets(phone_sequences, 1);
    let (place_targets, place_target_lengths, place_width) =
        pad_compact_targets(place_sequences, 1);
    let (manner_targets, manner_target_lengths, manner_width) =
        pad_compact_targets(manner_sequences, 1);
    let (voicing_targets, voicing_target_lengths, voicing_width) =
        pad_compact_targets(voicing_sequences, 1);
    let (syllabic_targets, syllabic_target_lengths, syllabic_width) =
        pad_compact_targets(syllabic_sequences, 1);
    let (height_targets, height_target_lengths, height_width) =
        pad_compact_targets(height_sequences, 1);
    let (backness_targets, backness_target_lengths, backness_width) =
        pad_compact_targets(backness_sequences, 1);
    let (rounding_targets, rounding_target_lengths, rounding_width) =
        pad_compact_targets(rounding_sequences, 1);
    Ok(AsrBatch {
        mel: Tensor::<B, 3>::from_data(
            TensorData::new(mel, [rows.len(), max_frames, mel_bins]),
            device,
        ),
        mel_target: Tensor::<B, 3>::from_data(
            TensorData::new(mel_target, [rows.len(), max_frames, mel_bins]),
            device,
        ),
        transcript_labels: Tensor::<B, 2, Int>::from_data(
            TensorData::new(transcript_labels, [rows.len(), max_frames]),
            device,
        ),
        seq2seq_labels: Tensor::<B, 2, Int>::from_data(
            TensorData::new(seq2seq_labels, [rows.len(), max_frames]),
            device,
        ),
        boundary_labels: Tensor::<B, 2, Int>::from_data(
            TensorData::new(boundary_labels, [rows.len(), max_frames]),
            device,
        ),
        phoneme_labels: Tensor::<B, 2, Int>::from_data(
            TensorData::new(phoneme_labels, [rows.len(), max_frames]),
            device,
        ),
        phone_labels: Tensor::<B, 2, Int>::from_data(
            TensorData::new(phone_labels, [rows.len(), max_frames]),
            device,
        ),
        input_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(input_lengths, [rows.len()]),
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
        phone_targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(phone_targets, [rows.len(), phone_width]),
            device,
        ),
        phone_target_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(phone_target_lengths, [rows.len()]),
            device,
        ),
        place_targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(place_targets, [rows.len(), place_width]),
            device,
        ),
        place_target_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(place_target_lengths, [rows.len()]),
            device,
        ),
        manner_targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(manner_targets, [rows.len(), manner_width]),
            device,
        ),
        manner_target_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(manner_target_lengths, [rows.len()]),
            device,
        ),
        voicing_targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(voicing_targets, [rows.len(), voicing_width]),
            device,
        ),
        voicing_target_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(voicing_target_lengths, [rows.len()]),
            device,
        ),
        syllabic_targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(syllabic_targets, [rows.len(), syllabic_width]),
            device,
        ),
        syllabic_target_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(syllabic_target_lengths, [rows.len()]),
            device,
        ),
        height_targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(height_targets, [rows.len(), height_width]),
            device,
        ),
        height_target_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(height_target_lengths, [rows.len()]),
            device,
        ),
        backness_targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(backness_targets, [rows.len(), backness_width]),
            device,
        ),
        backness_target_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(backness_target_lengths, [rows.len()]),
            device,
        ),
        rounding_targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(rounding_targets, [rows.len(), rounding_width]),
            device,
        ),
        rounding_target_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(rounding_target_lengths, [rows.len()]),
            device,
        ),
        prev_word_targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(prev_word_targets, [rows.len(), prev_word_width]),
            device,
        ),
        prev_word_target_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(prev_word_target_lengths, [rows.len()]),
            device,
        ),
        current_word_targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(current_word_targets, [rows.len(), current_word_width]),
            device,
        ),
        current_word_target_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(current_word_target_lengths, [rows.len()]),
            device,
        ),
        next_word_targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(next_word_targets, [rows.len(), next_word_width]),
            device,
        ),
        next_word_target_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(next_word_target_lengths, [rows.len()]),
            device,
        ),
        masked_word_targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(masked_word_targets, [rows.len(), masked_word_width]),
            device,
        ),
        masked_word_target_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(masked_word_target_lengths, [rows.len()]),
            device,
        ),
        masked_word_phoneme_targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(
                masked_word_phoneme_targets,
                [rows.len(), masked_phoneme_width],
            ),
            device,
        ),
        masked_word_phoneme_target_lengths: Tensor::<B, 1, Int>::from_data(
            TensorData::new(masked_word_phoneme_target_lengths, [rows.len()]),
            device,
        ),
        syntax_pos_labels: Tensor::<B, 2, Int>::from_data(
            TensorData::new(syntax_pos_labels, [rows.len(), max_frames]),
            device,
        ),
        syntax_link_labels: Tensor::<B, 2, Int>::from_data(
            TensorData::new(syntax_link_labels, [rows.len(), max_frames]),
            device,
        ),
        syntax_head_offset_labels: Tensor::<B, 2, Int>::from_data(
            TensorData::new(syntax_head_offset_labels, [rows.len(), max_frames]),
            device,
        ),
        parse_ok_labels: Tensor::<B, 2, Int>::from_data(
            TensorData::new(parse_ok_labels, [rows.len(), max_frames]),
            device,
        ),
        phrase_boundary_labels: Tensor::<B, 2, Int>::from_data(
            TensorData::new(phrase_boundary_labels, [rows.len(), max_frames]),
            device,
        ),
    })
}

fn weighted_loss<B: Backend>(
    output: AsrForward<B>,
    batch: AsrBatch<B>,
    config: &InterpretationTrainConfig,
) -> Tensor<B, 1> {
    let transcript_loss = ce_loss(output.transcript_logits, batch.transcript_labels, 0);
    let seq2seq_loss = ce_loss(output.seq2seq_transcript_logits, batch.seq2seq_labels, 0);
    let boundary_loss = ce_loss(output.boundary_logits, batch.boundary_labels, usize::MAX);
    let phoneme_frame_loss = ce_loss(output.phoneme_logits.clone(), batch.phoneme_labels, 0);
    let phone_frame_loss = ce_loss(output.phone_logits.clone(), batch.phone_labels, 0);
    let phoneme_ctc_loss = ctc_loss(
        output.phoneme_logits,
        batch.phoneme_targets,
        batch.input_lengths.clone(),
        batch.phoneme_target_lengths,
        0,
    );
    let phone_ctc_loss = ctc_loss(
        output.phone_logits,
        batch.phone_targets,
        batch.input_lengths.clone(),
        batch.phone_target_lengths,
        0,
    );
    let phoneme_loss = phoneme_frame_loss + phoneme_ctc_loss;
    let phone_loss = phone_frame_loss + phone_ctc_loss;
    let feature_ctc_loss = ctc_loss(
        output.place_logits,
        batch.place_targets,
        batch.input_lengths.clone(),
        batch.place_target_lengths,
        0,
    ) + ctc_loss(
        output.manner_logits,
        batch.manner_targets,
        batch.input_lengths.clone(),
        batch.manner_target_lengths,
        0,
    ) + ctc_loss(
        output.voicing_logits,
        batch.voicing_targets,
        batch.input_lengths.clone(),
        batch.voicing_target_lengths,
        0,
    ) + ctc_loss(
        output.syllabic_logits,
        batch.syllabic_targets,
        batch.input_lengths.clone(),
        batch.syllabic_target_lengths,
        0,
    ) + ctc_loss(
        output.height_logits,
        batch.height_targets,
        batch.input_lengths.clone(),
        batch.height_target_lengths,
        0,
    ) + ctc_loss(
        output.backness_logits,
        batch.backness_targets,
        batch.input_lengths.clone(),
        batch.backness_target_lengths,
        0,
    ) + ctc_loss(
        output.rounding_logits,
        batch.rounding_targets,
        batch.input_lengths.clone(),
        batch.rounding_target_lengths,
        0,
    );
    let prev_word_loss = ctc_loss(
        output.prev_word_logits,
        batch.prev_word_targets,
        batch.input_lengths.clone(),
        batch.prev_word_target_lengths,
        0,
    );
    let current_word_loss = ctc_loss(
        output.current_word_logits,
        batch.current_word_targets,
        batch.input_lengths.clone(),
        batch.current_word_target_lengths,
        0,
    );
    let next_word_loss = ctc_loss(
        output.next_word_logits,
        batch.next_word_targets,
        batch.input_lengths.clone(),
        batch.next_word_target_lengths,
        0,
    );
    let masked_word_loss = ctc_loss(
        output.masked_word_logits,
        batch.masked_word_targets,
        batch.input_lengths.clone(),
        batch.masked_word_target_lengths,
        0,
    );
    let masked_word_phoneme_loss = ctc_loss(
        output.masked_word_phoneme_logits,
        batch.masked_word_phoneme_targets,
        batch.input_lengths,
        batch.masked_word_phoneme_target_lengths,
        0,
    );
    let syntax_loss = ce_loss(output.syntax_pos_logits, batch.syntax_pos_labels, 0)
        + ce_loss(output.syntax_link_logits, batch.syntax_link_labels, 0)
        + ce_loss(
            output.syntax_head_offset_logits,
            batch.syntax_head_offset_labels,
            0,
        )
        + ce_loss(output.parse_ok_logits, batch.parse_ok_labels, 0)
        + ce_loss(
            output.phrase_boundary_logits,
            batch.phrase_boundary_labels,
            0,
        );
    let audio_loss = mse_loss(output.mel_reconstruction, batch.mel_target);
    transcript_loss * config.transcript_loss_weight
        + boundary_loss * (config.boundary_loss_weight + config.repair_loss_weight)
        + phoneme_loss * config.phoneme_loss_weight
        + phone_loss * config.phone_loss_weight
        + feature_ctc_loss * config.feature_ctc_loss_weight
        + prev_word_loss * config.prev_word_loss_weight
        + current_word_loss * config.current_word_loss_weight
        + next_word_loss * config.next_word_loss_weight
        + masked_word_loss * config.masked_word_loss_weight
        + masked_word_phoneme_loss * config.masked_word_phoneme_loss_weight
        + syntax_loss * config.syntax_loss_weight
        + seq2seq_loss * config.seq2seq_loss_weight
        + audio_loss * config.masked_audio_loss_weight
}

fn ce_loss<B: Backend>(
    logits: Tensor<B, 3>,
    labels: Tensor<B, 2, Int>,
    pad: usize,
) -> Tensor<B, 1> {
    let [batch, frames, classes] = logits.dims();
    let ce = CrossEntropyLossConfig::new()
        .with_pad_tokens((pad != usize::MAX).then_some(vec![pad]))
        .init::<B>(&logits.device());
    ce.forward(
        logits.reshape([batch * frames, classes]),
        labels.reshape([batch * frames]),
    )
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

fn mse_loss<B: Backend>(predicted: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 1> {
    let diff = predicted - target;
    (diff.clone() * diff).mean()
}

fn frame_is_masked(frame: usize, config: &InterpretationTrainConfig) -> bool {
    let every = config.mask_every_n_frames.max(1);
    let span = config.mask_span_frames.min(every).max(1);
    frame % every < span
}

fn frame_is_word_masked(
    row: &LibriSpeechUtterance,
    frame: usize,
    config: &InterpretationTrainConfig,
) -> bool {
    if config.word_mask_rate <= 0.0 {
        return false;
    }
    row.masked_word_examples
        .iter()
        .any(|example| example.start_frame <= frame && frame < example.end_frame)
}

fn proportional_labels(text: &str, vocab: &Vocab, frames: usize) -> Vec<i32> {
    let chars = text.chars().collect::<Vec<_>>();
    if chars.is_empty() {
        return vec![0; frames];
    }
    (0..frames)
        .map(|frame| {
            let idx = frame * chars.len() / frames.max(1);
            vocab.get_id(&chars[idx.min(chars.len() - 1)].to_string()) as i32
        })
        .collect()
}

fn seq2seq_labels_for(text: &str, vocab: &Vocab, seq_len: usize, frames: usize) -> Vec<i32> {
    let mut labels = vec![0; frames];
    for (idx, ch) in text.chars().take(seq_len).enumerate() {
        labels[idx] = vocab.get_id(&ch.to_string()) as i32;
    }
    labels
}

fn previous_word_targets(row: &LibriSpeechUtterance, vocab: &Vocab) -> Vec<i32> {
    row.word_supervision
        .iter()
        .filter_map(|word| word.previous_word.as_deref())
        .map(|word| word_id(vocab, word) as i32)
        .collect()
}

fn current_word_targets(row: &LibriSpeechUtterance, vocab: &Vocab) -> Vec<i32> {
    row.word_supervision
        .iter()
        .map(|word| word_id(vocab, &word.word) as i32)
        .collect()
}

fn next_word_targets(row: &LibriSpeechUtterance, vocab: &Vocab) -> Vec<i32> {
    row.word_supervision
        .iter()
        .filter_map(|word| word.next_word.as_deref())
        .map(|word| word_id(vocab, word) as i32)
        .collect()
}

fn masked_word_targets(row: &LibriSpeechUtterance, vocab: &Vocab) -> Vec<i32> {
    row.masked_word_examples
        .iter()
        .map(|masked| word_id(vocab, &masked.masked_word) as i32)
        .collect()
}

fn masked_word_phoneme_targets(row: &LibriSpeechUtterance, vocab: &Vocab) -> Vec<i32> {
    row.masked_word_examples
        .iter()
        .flat_map(|masked| masked.masked_word_phonemes.split_whitespace())
        .map(|phoneme| nonblank_id(vocab, phoneme) as i32)
        .collect()
}

fn phoneme_targets(row: &LibriSpeechUtterance, vocab: &Vocab) -> Vec<i32> {
    row.sentences
        .iter()
        .flat_map(sentence_phoneme_labels)
        .map(|phoneme| nonblank_id(vocab, &phoneme) as i32)
        .collect()
}

fn phone_targets(row: &LibriSpeechUtterance, vocab: &Vocab) -> Vec<i32> {
    row.sentences
        .iter()
        .flat_map(|sentence| sentence.phones.split_whitespace())
        .map(|phone| nonblank_id(vocab, phone) as i32)
        .collect()
}

#[derive(Clone, Copy)]
enum FeatureAxis {
    Place,
    Manner,
    Voicing,
    Syllabic,
    Height,
    Backness,
    Rounding,
}

fn feature_targets(row: &LibriSpeechUtterance, axis: FeatureAxis) -> Vec<i32> {
    row.sentences
        .iter()
        .flat_map(|sentence| sentence.phones.split_whitespace())
        .map(|phone| feature_id(axis, phone) as i32)
        .collect()
}

fn feature_id(axis: FeatureAxis, phone: &str) -> u32 {
    match axis {
        FeatureAxis::Place => match phone_place(phone) {
            "labial" => 2,
            "coronal" => 3,
            "dorsal" => 4,
            "front" => 5,
            "back" => 6,
            "central" => 7,
            _ => 1,
        },
        FeatureAxis::Manner => match phone_manner(phone) {
            "stop" => 2,
            "fricative" => 3,
            "nasal" => 4,
            "approximant" => 5,
            "vowel" => 6,
            _ => 1,
        },
        FeatureAxis::Voicing => {
            if phone_is_vowel(phone)
                || phone_is_sonorant(phone)
                || voiced_obstruents().contains(&phone_base(phone))
            {
                3
            } else if voiceless_obstruents().contains(&phone_base(phone)) {
                2
            } else {
                1
            }
        }
        FeatureAxis::Syllabic => {
            if phone_is_vowel(phone) {
                3
            } else {
                2
            }
        }
        FeatureAxis::Height => match phone_height(phone) {
            "high" => 2,
            "mid" => 3,
            "low" => 4,
            "nonvowel" => 5,
            _ => 1,
        },
        FeatureAxis::Backness => match phone_backness(phone) {
            "front" => 2,
            "central" => 3,
            "back" => 4,
            _ => 1,
        },
        FeatureAxis::Rounding => {
            if phone_is_vowel(phone) || phone_base(phone) == "w" {
                if rounded_phones().contains(&phone_base(phone)) {
                    3
                } else {
                    2
                }
            } else {
                1
            }
        }
    }
}

fn phone_base(phone: &str) -> &str {
    phone
        .trim_matches(|ch: char| matches!(ch, 'ˈ' | 'ˌ' | 'ː' | 'ʰ' | 'ʲ' | 'ʷ' | 'ˠ'))
        .split(['͡', '͜'])
        .next()
        .unwrap_or(phone)
}

fn phone_is_vowel(phone: &str) -> bool {
    "aeiouyæɑɒɔəɚɛɜɞɪiʊuʌøœɯɨɐ"
        .chars()
        .any(|ch| phone_base(phone).starts_with(ch))
}

fn phone_is_sonorant(phone: &str) -> bool {
    ["m", "n", "ŋ", "ɲ", "l", "ɫ", "r", "ɹ", "ɾ", "j", "w", "ʋ"].contains(&phone_base(phone))
}

fn voiced_obstruents() -> &'static [&'static str] {
    &["b", "d", "g", "v", "z", "ʒ", "ð", "ɣ", "ʁ"]
}

fn voiceless_obstruents() -> &'static [&'static str] {
    &["p", "t", "k", "f", "s", "ʃ", "θ", "x", "h", "q", "χ", "ʔ"]
}

fn rounded_phones() -> &'static [&'static str] {
    &["u", "ʊ", "o", "ɔ", "ø", "œ", "w"]
}

fn phone_place(phone: &str) -> &'static str {
    let base = phone_base(phone);
    if ["p", "b", "m", "f", "v", "ʋ", "w"].contains(&base) {
        "labial"
    } else if [
        "t", "d", "s", "z", "ʃ", "ʒ", "θ", "ð", "ɹ", "ɾ", "ɫ", "l", "n",
    ]
    .contains(&base)
    {
        "coronal"
    } else if ["k", "g", "x", "ɣ", "ŋ", "q", "χ", "ʁ"].contains(&base) {
        "dorsal"
    } else if phone_is_vowel(base) {
        phone_backness(base)
    } else {
        "none"
    }
}

fn phone_manner(phone: &str) -> &'static str {
    let base = phone_base(phone);
    if phone_is_vowel(base) {
        "vowel"
    } else if ["p", "b", "t", "d", "k", "g", "q", "ʔ"].contains(&base) {
        "stop"
    } else if [
        "f", "v", "s", "z", "ʃ", "ʒ", "θ", "ð", "x", "h", "ɣ", "χ", "ʁ",
    ]
    .contains(&base)
    {
        "fricative"
    } else if ["m", "n", "ŋ", "ɲ"].contains(&base) {
        "nasal"
    } else if ["l", "ɫ", "r", "ɹ", "ɾ", "j", "w", "ʋ"].contains(&base) {
        "approximant"
    } else {
        "none"
    }
}

fn phone_height(phone: &str) -> &'static str {
    let base = phone_base(phone);
    if !phone_is_vowel(base) {
        "nonvowel"
    } else if ["i", "ɪ", "u", "ʊ", "ɨ", "ɯ", "y"].contains(&base) {
        "high"
    } else if ["æ", "a", "ɑ", "ɒ"].contains(&base) {
        "low"
    } else {
        "mid"
    }
}

fn phone_backness(phone: &str) -> &'static str {
    let base = phone_base(phone);
    if !phone_is_vowel(base) {
        "none"
    } else if ["i", "ɪ", "e", "ɛ", "æ", "y", "ø", "œ"].contains(&base) {
        "front"
    } else if ["u", "ʊ", "o", "ɔ", "ɑ", "ɒ", "ʌ", "ɯ"].contains(&base) {
        "back"
    } else {
        "central"
    }
}

fn pad_compact_targets(
    mut sequences: Vec<Vec<i32>>,
    fallback_id: u32,
) -> (Vec<i32>, Vec<i32>, usize) {
    let fallback = fallback_id.max(1) as i32;
    let mut lengths = Vec::with_capacity(sequences.len());
    for sequence in &mut sequences {
        if sequence.is_empty() {
            sequence.push(fallback);
        }
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

fn word_id(vocab: &Vocab, word: &str) -> u32 {
    if let Some(word) = normalize_word_token(word) {
        if let Some(id) = vocab.token_to_id.get(&word) {
            return *id;
        }
    }
    vocab.get_id(WORD_UNK).max(1)
}

fn nonblank_id(vocab: &Vocab, token: &str) -> u32 {
    vocab.get_id(token).max(1)
}

fn proportional_phoneme_labels(
    row: &LibriSpeechUtterance,
    vocab: &Vocab,
    frames: usize,
) -> Vec<i32> {
    let phonemes = row
        .sentences
        .iter()
        .flat_map(sentence_phoneme_labels)
        .collect::<Vec<_>>();
    if phonemes.is_empty() {
        return vec![0; frames];
    }
    (0..frames)
        .map(|frame| {
            let idx = frame * phonemes.len() / frames.max(1);
            vocab.get_id(&phonemes[idx.min(phonemes.len() - 1)]) as i32
        })
        .collect()
}

fn sentence_phoneme_labels(sentence: &SentenceSupervision) -> Vec<String> {
    let labels = sentence
        .phoneme_tokens
        .iter()
        .filter_map(|token| match &token.phoneme {
            speaking::Spec::Known(id) => {
                let raw: &str = &id.0;
                Some(raw.rsplit('.').next().unwrap_or(raw).to_string())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if labels.is_empty() {
        sentence
            .phonemes
            .split_whitespace()
            .map(str::to_string)
            .collect()
    } else {
        labels
    }
}

fn proportional_phone_labels(row: &LibriSpeechUtterance, vocab: &Vocab, frames: usize) -> Vec<i32> {
    let phones = row
        .sentences
        .iter()
        .flat_map(|s| s.phones.split_whitespace().map(str::to_string))
        .collect::<Vec<_>>();
    if phones.is_empty() {
        return vec![0; frames];
    }
    (0..frames)
        .map(|frame| {
            let idx = frame * phones.len() / frames.max(1);
            vocab.get_id(&phones[idx.min(phones.len() - 1)]) as i32
        })
        .collect()
}

fn boundary_labels_for(row: &LibriSpeechUtterance, frames: usize) -> Vec<i32> {
    let mut labels = vec![0; frames];
    for sentence in &row.sentences {
        if frames > 0 {
            let idx = sentence.end_frame.min(frames - 1);
            labels[idx] = 1;
        }
    }
    for repair in &row.repair_examples {
        if frames > 0 {
            let idx = repair.end_frame.min(frames - 1);
            labels[idx] = 2;
        }
    }
    labels
}

fn syntax_pos_labels_for(row: &LibriSpeechUtterance, vocab: &Vocab, frames: usize) -> Vec<i32> {
    syntax_word_labels_for(row, frames, |word| vocab.get_id(&word.pos) as i32)
}

fn syntax_link_labels_for(row: &LibriSpeechUtterance, vocab: &Vocab, frames: usize) -> Vec<i32> {
    syntax_word_labels_for(row, frames, |word| {
        vocab.get_id(&word.primary_link_label) as i32
    })
}

fn syntax_head_offset_labels_for(
    row: &LibriSpeechUtterance,
    vocab: &Vocab,
    frames: usize,
) -> Vec<i32> {
    syntax_word_labels_for(row, frames, |word| {
        vocab.get_id(&syntax_head_offset_label(word.head_offset)) as i32
    })
}

fn phrase_boundary_labels_for(row: &LibriSpeechUtterance, frames: usize) -> Vec<i32> {
    syntax_word_labels_for(row, frames, |word| if word.phrase_boundary { 1 } else { 0 })
}

fn parse_ok_labels_for(row: &LibriSpeechUtterance, frames: usize) -> Vec<i32> {
    let mut labels = vec![0; frames];
    for sentence in &row.sentences {
        if sentence.syntax.supervision_weight <= 0.0 {
            continue;
        }
        let label = if sentence.syntax.parse_ok { 1 } else { 0 };
        for frame in sentence.start_frame.min(frames)..sentence.end_frame.min(frames) {
            labels[frame] = label;
        }
    }
    labels
}

fn syntax_word_labels_for(
    row: &LibriSpeechUtterance,
    frames: usize,
    label: impl Fn(&SyntaxWordSupervision) -> i32,
) -> Vec<i32> {
    let mut labels = vec![0; frames];
    for sentence in &row.sentences {
        if sentence.syntax.supervision_weight <= 0.0 {
            continue;
        }
        let spans = word_spans(&sentence.text);
        for word in &sentence.syntax.words {
            let Some((start, end, _)) = spans.get(word.sentence_word_index) else {
                continue;
            };
            let global_start = sentence.start_char + *start;
            let global_end = sentence.start_char + *end;
            let start_frame = char_to_frame(
                global_start,
                sentence.end_char.max(1),
                sentence.end_frame.max(1),
            )
            .max(sentence.start_frame)
            .min(frames);
            let end_frame = char_to_frame(
                global_end,
                sentence.end_char.max(1),
                sentence.end_frame.max(1),
            )
            .max(start_frame + 1)
            .min(sentence.end_frame.max(start_frame + 1))
            .min(frames);
            for frame in start_frame..end_frame {
                labels[frame] = label(word);
            }
        }
    }
    labels
}

pub fn evaluate<B: Backend>(
    model: &AsrModel<B>,
    data_dir: &Path,
    rows: &[LibriSpeechUtterance],
    vocab: &Vocab,
    phoneme_vocab: &Vocab,
    phone_vocab: &Vocab,
    word_vocab: &Vocab,
    syntax_pos_vocab: &Vocab,
    syntax_link_vocab: &Vocab,
    syntax_head_offset_vocab: &Vocab,
    config: &InterpretationTrainConfig,
    device: &B::Device,
) -> Result<EvalReport> {
    let eval_rows = rows.iter().take(100).cloned().collect::<Vec<_>>();
    if eval_rows.is_empty() {
        return Ok(EvalReport {
            examples: 0,
            loss: 0.0,
            token_error_rate: 0.0,
            word_error_rate: 0.0,
            seq2seq_token_error_rate: 0.0,
            boundary_f1: 0.0,
            repair_f1: 0.0,
            phoneme_token_error_rate: 0.0,
            phone_token_error_rate: 0.0,
            masked_audio_mse: 0.0,
            prev_word_accuracy: 0.0,
            current_word_accuracy: 0.0,
            next_word_accuracy: 0.0,
            masked_word_accuracy: 0.0,
            masked_word_phoneme_token_error_rate: 0.0,
        });
    }
    let mut total_loss = 0.0;
    let mut batches = 0usize;
    let mut stats = EvalSampleStats::default();
    let mut audio_mse_total = 0.0f32;
    for chunk in eval_rows.chunks(config.batch_size.max(1)) {
        let batch = make_batch::<B>(
            data_dir,
            chunk,
            vocab,
            phoneme_vocab,
            phone_vocab,
            word_vocab,
            syntax_pos_vocab,
            syntax_link_vocab,
            syntax_head_offset_vocab,
            config,
            device,
        )?;
        let output = model.forward(batch.mel.clone());
        let audio_mse = mse_loss(output.mel_reconstruction.clone(), batch.mel_target.clone())
            .into_scalar()
            .elem::<f32>();
        let loss = weighted_loss(
            AsrForward {
                transcript_logits: output.transcript_logits.clone(),
                seq2seq_transcript_logits: output.seq2seq_transcript_logits.clone(),
                boundary_logits: output.boundary_logits.clone(),
                phoneme_logits: output.phoneme_logits.clone(),
                phone_logits: output.phone_logits.clone(),
                place_logits: output.place_logits.clone(),
                manner_logits: output.manner_logits.clone(),
                voicing_logits: output.voicing_logits.clone(),
                syllabic_logits: output.syllabic_logits.clone(),
                height_logits: output.height_logits.clone(),
                backness_logits: output.backness_logits.clone(),
                rounding_logits: output.rounding_logits.clone(),
                prev_word_logits: output.prev_word_logits.clone(),
                current_word_logits: output.current_word_logits.clone(),
                next_word_logits: output.next_word_logits.clone(),
                masked_word_logits: output.masked_word_logits.clone(),
                masked_word_phoneme_logits: output.masked_word_phoneme_logits.clone(),
                syntax_pos_logits: output.syntax_pos_logits.clone(),
                syntax_link_logits: output.syntax_link_logits.clone(),
                syntax_head_offset_logits: output.syntax_head_offset_logits.clone(),
                parse_ok_logits: output.parse_ok_logits.clone(),
                phrase_boundary_logits: output.phrase_boundary_logits.clone(),
                mel_reconstruction: output.mel_reconstruction.clone(),
            },
            batch,
            config,
        );
        total_loss += loss.into_scalar().elem::<f32>();
        audio_mse_total += audio_mse;
        batches += 1;
        let transcript_preds = argmax_ids(output.transcript_logits);
        let seq2seq_preds = argmax_ids(output.seq2seq_transcript_logits);
        let boundary_preds = argmax_ids(output.boundary_logits);
        let phoneme_preds = argmax_ids(output.phoneme_logits);
        let phone_preds = argmax_ids(output.phone_logits);
        let prev_word_preds = argmax_ids(output.prev_word_logits);
        let current_word_preds = argmax_ids(output.current_word_logits);
        let next_word_preds = argmax_ids(output.next_word_logits);
        let masked_word_preds = argmax_ids(output.masked_word_logits);
        let masked_word_phoneme_preds = argmax_ids(output.masked_word_phoneme_logits);
        let chunk_stats = chunk
            .par_iter()
            .enumerate()
            .map(|(i, row)| {
                let mut local = EvalSampleStats::default();
                let decoded = greedy_collapse(&transcript_preds[i], vocab);
                let ref_chars = row.transcript.chars().collect::<Vec<_>>();
                let hyp_chars = decoded.chars().collect::<Vec<_>>();
                local.token_errors += edit_distance(&ref_chars, &hyp_chars);
                local.token_total += ref_chars.len();

                let seq2seq_decoded = seq2seq_decode(&seq2seq_preds[i], vocab);
                let seq2seq_hyp_chars = seq2seq_decoded.chars().collect::<Vec<_>>();
                local.seq2seq_token_errors += edit_distance(&ref_chars, &seq2seq_hyp_chars);
                local.seq2seq_token_total += ref_chars.len();

                let ref_words = row.transcript.split_whitespace().collect::<Vec<_>>();
                let hyp_words = decoded.split_whitespace().collect::<Vec<_>>();
                local.word_errors += edit_distance(&ref_words, &hyp_words);
                local.word_total += ref_words.len();

                let gold = boundary_labels_for(row, boundary_preds[i].len());
                for (pred, gold) in boundary_preds[i].iter().zip(gold) {
                    match (*pred == 1, gold == 1) {
                        (true, true) => local.boundary_tp += 1,
                        (true, false) => local.boundary_fp += 1,
                        (false, true) => local.boundary_fn += 1,
                        _ => {}
                    }
                    match (*pred == 2, gold == 2) {
                        (true, true) => local.repair_tp += 1,
                        (true, false) => local.repair_fp += 1,
                        (false, true) => local.repair_fn += 1,
                        _ => {}
                    }
                }

                let decoded_phonemes = greedy_collapse(&phoneme_preds[i], phoneme_vocab);
                let ref_phonemes = row
                    .sentences
                    .iter()
                    .flat_map(|s| s.phonemes.split_whitespace())
                    .collect::<Vec<_>>();
                let hyp_phonemes = decoded_phonemes.split_whitespace().collect::<Vec<_>>();
                local.phoneme_errors += edit_distance(&ref_phonemes, &hyp_phonemes);
                local.phoneme_total += ref_phonemes.len();

                let decoded_phones = greedy_collapse(&phone_preds[i], phone_vocab);
                let ref_phones = row
                    .sentences
                    .iter()
                    .flat_map(|s| s.phones.split_whitespace())
                    .collect::<Vec<_>>();
                let hyp_phones = decoded_phones.split_whitespace().collect::<Vec<_>>();
                local.phone_errors += edit_distance(&ref_phones, &hyp_phones);
                local.phone_total += ref_phones.len();

                let decoded_prev = ctc_greedy_decode(&prev_word_preds[i], 0);
                let target_prev = previous_word_targets(row, word_vocab);
                let (correct, total) = sequence_accuracy(&decoded_prev, &target_prev);
                local.prev_correct += correct;
                local.prev_total += total;

                let decoded_current = ctc_greedy_decode(&current_word_preds[i], 0);
                let target_current = current_word_targets(row, word_vocab);
                let (correct, total) = sequence_accuracy(&decoded_current, &target_current);
                local.current_correct += correct;
                local.current_total += total;

                let decoded_next = ctc_greedy_decode(&next_word_preds[i], 0);
                let target_next = next_word_targets(row, word_vocab);
                let (correct, total) = sequence_accuracy(&decoded_next, &target_next);
                local.next_correct += correct;
                local.next_total += total;

                let decoded_masked_word = ctc_greedy_decode(&masked_word_preds[i], 0);
                let target_masked_word = masked_word_targets(row, word_vocab);
                let (correct, total) = sequence_accuracy(&decoded_masked_word, &target_masked_word);
                local.masked_word_correct += correct;
                local.masked_word_total += total;

                let decoded_masked_phonemes = ctc_greedy_decode(&masked_word_phoneme_preds[i], 0)
                    .into_iter()
                    .map(|id| id as i32)
                    .collect::<Vec<_>>();
                let target_masked_phonemes = masked_word_phoneme_targets(row, phoneme_vocab);
                local.masked_phoneme_errors +=
                    edit_distance(&target_masked_phonemes, &decoded_masked_phonemes);
                local.masked_phoneme_total += target_masked_phonemes.len();
                local
            })
            .reduce(EvalSampleStats::default, |mut acc, item| {
                acc.merge(item);
                acc
            });
        stats.merge(chunk_stats);
    }
    let precision =
        stats.boundary_tp as f32 / (stats.boundary_tp + stats.boundary_fp).max(1) as f32;
    let recall = stats.boundary_tp as f32 / (stats.boundary_tp + stats.boundary_fn).max(1) as f32;
    let repair_precision =
        stats.repair_tp as f32 / (stats.repair_tp + stats.repair_fp).max(1) as f32;
    let repair_recall = stats.repair_tp as f32 / (stats.repair_tp + stats.repair_fn).max(1) as f32;
    Ok(EvalReport {
        examples: eval_rows.len(),
        loss: total_loss / batches.max(1) as f32,
        token_error_rate: stats.token_errors as f32 / stats.token_total.max(1) as f32,
        word_error_rate: stats.word_errors as f32 / stats.word_total.max(1) as f32,
        seq2seq_token_error_rate: stats.seq2seq_token_errors as f32
            / stats.seq2seq_token_total.max(1) as f32,
        boundary_f1: if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        },
        repair_f1: if repair_precision + repair_recall > 0.0 {
            2.0 * repair_precision * repair_recall / (repair_precision + repair_recall)
        } else {
            0.0
        },
        phoneme_token_error_rate: stats.phoneme_errors as f32 / stats.phoneme_total.max(1) as f32,
        phone_token_error_rate: stats.phone_errors as f32 / stats.phone_total.max(1) as f32,
        masked_audio_mse: audio_mse_total / batches.max(1) as f32,
        prev_word_accuracy: stats.prev_correct as f32 / stats.prev_total.max(1) as f32,
        current_word_accuracy: stats.current_correct as f32 / stats.current_total.max(1) as f32,
        next_word_accuracy: stats.next_correct as f32 / stats.next_total.max(1) as f32,
        masked_word_accuracy: stats.masked_word_correct as f32
            / stats.masked_word_total.max(1) as f32,
        masked_word_phoneme_token_error_rate: stats.masked_phoneme_errors as f32
            / stats.masked_phoneme_total.max(1) as f32,
    })
}

fn sequence_accuracy(predicted: &[u32], gold: &[i32]) -> (usize, usize) {
    if gold.is_empty() {
        return (0, 0);
    }
    let predicted = predicted.iter().map(|id| *id as i32).collect::<Vec<_>>();
    ((predicted == gold) as usize, 1)
}

fn argmax_ids<B: Backend>(logits: Tensor<B, 3>) -> Vec<Vec<u32>> {
    let [batch, frames, classes] = logits.dims();
    let values: Vec<f32> = logits.into_data().to_vec().unwrap_or_default();
    let mut out = vec![vec![0; frames]; batch];
    for b in 0..batch {
        for f in 0..frames {
            let base = (b * frames + f) * classes;
            let mut best = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for c in 0..classes {
                let score = values.get(base + c).copied().unwrap_or(f32::NEG_INFINITY);
                if score > best_score {
                    best = c;
                    best_score = score;
                }
            }
            out[b][f] = best as u32;
        }
    }
    out
}

pub fn ctc_greedy_decode(ids: &[u32], blank: u32) -> Vec<u32> {
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

pub fn greedy_collapse(ids: &[u32], vocab: &Vocab) -> String {
    let mut out = String::new();
    let mut prev = 0u32;
    for &id in ids {
        if id != 0 && id != prev {
            if let Some(token) = vocab.tokens.get(id as usize) {
                if !token.starts_with('<') {
                    out.push_str(token);
                }
            }
        }
        prev = id;
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn seq2seq_decode(ids: &[u32], vocab: &Vocab) -> String {
    let mut out = String::new();
    for &id in ids {
        if id == 0 {
            continue;
        }
        if let Some(token) = vocab.tokens.get(id as usize) {
            if !token.starts_with('<') {
                out.push_str(token);
            }
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn fit_feature_width(features: &mut [Vec<f32>], bins: usize) {
    for row in features {
        row.resize(bins, 0.0);
        row.truncate(bins);
    }
}

pub fn stream_from_samples<B: Backend>(
    model: &AsrModel<B>,
    samples: &[f32],
    vocab: &Vocab,
    word_vocab: &Vocab,
    phoneme_vocab: &Vocab,
    config: &InterpretationConfig,
    input_feature_bins: usize,
    device: &B::Device,
) -> Result<StreamEvent> {
    let mut stream_config = config.clone();
    stream_config.compact_audio_features = input_feature_bins != stream_config.mel_bins;
    let mut features = audio_features(samples, &stream_config);
    fit_feature_width(&mut features, input_feature_bins);
    let batch = Tensor::<B, 3>::from_data(
        TensorData::new(
            features.iter().flatten().copied().collect::<Vec<_>>(),
            [1, features.len(), input_feature_bins],
        ),
        device,
    );
    let output = model.forward(batch);
    let ids = argmax_ids(output.transcript_logits.clone());
    let partial = ids
        .first()
        .map(|ids| greedy_collapse(ids, vocab))
        .unwrap_or_default();
    let seq2seq_ids = argmax_ids(output.seq2seq_transcript_logits);
    let seq2seq_transcript = seq2seq_ids
        .first()
        .map(|ids| seq2seq_decode(ids, vocab))
        .unwrap_or_default();
    let prev_word_ids = argmax_ids(output.prev_word_logits);
    let current_word_ids = argmax_ids(output.current_word_logits);
    let next_word_ids = argmax_ids(output.next_word_logits);
    let phoneme_ids = argmax_ids(output.phoneme_logits);
    let phonemes = phoneme_ids
        .first()
        .map(|ids| greedy_collapse(ids, phoneme_vocab))
        .filter(|value| !value.is_empty());
    let detector = SentenceDetectorDialog::new()?;
    let sentences = sentence_supervision(&detector, &partial, features.len(), config)?;
    let repair_events = repair_supervision(&sentences);
    Ok(StreamEvent {
        partial_transcript: partial,
        seq2seq_transcript,
        final_sentences: sentences,
        repair_events,
        previous_word: word_prediction(prev_word_ids.first(), word_vocab, phonemes.clone()),
        current_word: word_prediction(current_word_ids.first(), word_vocab, phonemes.clone()),
        next_word: word_prediction(next_word_ids.first(), word_vocab, phonemes),
    })
}

fn word_prediction(
    ids: Option<&Vec<u32>>,
    vocab: &Vocab,
    phonemes: Option<String>,
) -> Option<WordPrediction> {
    let word = ids.and_then(|ids| last_decoded_word(ids, vocab));
    (word.is_some() || phonemes.is_some()).then_some(WordPrediction { word, phonemes })
}

fn last_decoded_word(ids: &[u32], vocab: &Vocab) -> Option<String> {
    ctc_greedy_decode(ids, 0).iter().rev().find_map(|id| {
        vocab
            .tokens
            .get(*id as usize)
            .filter(|token| !token.starts_with('<'))
            .cloned()
    })
}

fn edit_distance<T: Eq>(left: &[T], right: &[T]) -> usize {
    let mut dp: Vec<usize> = (0..=right.len()).collect();
    for (i, l) in left.iter().enumerate() {
        let mut prev = dp[0];
        dp[0] = i + 1;
        for (j, r) in right.iter().enumerate() {
            let old = dp[j + 1];
            dp[j + 1] = if l == r {
                prev
            } else {
                1 + prev.min(dp[j]).min(dp[j + 1])
            };
            prev = old;
        }
    }
    dp[right.len()]
}

fn dataset_readme(config: &InterpretationConfig) -> String {
    format!(
        "# LibriSpeech ASR dataset\n\nDataset id: `{}`\nSubset: `{:?}`\nFeature width: `{}`\n\nPrepared by `tongues interpretation prepare`. Rows contain FLAC provenance, durable acoustic feature paths, Whisper-refined transcript text, seams sentence labels, and speaking phonemicizer supervision. The default acoustic vector is `[log_mel_{}, delta_mel_{}, energy, vad, zcr, spectral_centroid, spectral_flux, f0, voiced_prob]`. Whisper-refined transcripts are kept only when they decompose to approximately the original LibriSpeech transcript.\n\nLibriSpeech is distributed from OpenSLR under CC BY 4.0. Preserve source attribution when redistributing derived artifacts.\n",
        config.dataset_id,
        config.subset,
        audio_feature_bins(config),
        config.mel_bins,
        config.mel_bins
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn normalizes_librispeech_text() {
        assert_eq!(normalize_librispeech_text("Hello, world?"), "HELLO WORLD?");
    }

    #[test]
    fn mel_shape_is_stable() {
        let cfg = InterpretationConfig::default();
        let samples = vec![0.0; 16_000];
        let mel = log_mel_features(&samples, &cfg);
        assert!(!mel.is_empty());
        assert_eq!(mel[0].len(), DEFAULT_MEL_BINS);
        let compact = audio_features(&samples, &cfg);
        assert_eq!(compact[0].len(), DEFAULT_COMPACT_AUDIO_FEATURE_BINS);
    }

    #[test]
    fn ctc_collapse_removes_repeats_and_blanks() {
        let vocab = Vocab {
            tokens: vec![CTC_BLANK.into(), "A".into(), "B".into()],
            token_to_id: HashMap::from([(CTC_BLANK.into(), 0), ("A".into(), 1), ("B".into(), 2)]),
        };
        assert_eq!(greedy_collapse(&[0, 1, 1, 0, 2], &vocab), "AB");
        assert_eq!(ctc_greedy_decode(&[0, 1, 1, 0, 2], 0), vec![1, 2]);
    }

    #[test]
    fn computes_commons_upload_url_for_wiktionary_audio() {
        assert_eq!(
            commons_upload_url("Acca_word.ogg").as_deref(),
            Some("https://upload.wikimedia.org/wikipedia/commons/b/be/Acca_word.ogg")
        );
    }

    #[test]
    fn phoneme_and_phone_targets_are_ctc_sequences() {
        let row = LibriSpeechUtterance {
            utterance_id: "u".into(),
            speaker_id: "s".into(),
            chapter_id: "c".into(),
            audio_path: "a.flac".into(),
            mel_path: "m.bin".into(),
            num_frames: 20,
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            duration_ms: 100,
            transcript: "CAT".into(),
            sentences: vec![SentenceSupervision {
                text: "CAT".into(),
                start_char: 0,
                end_char: 3,
                start_frame: 0,
                end_frame: 20,
                boundary_label: BOUNDARY_EMIT.into(),
                terminal: None,
                phonemes: "k æ t".into(),
                phones: "kʰ æ t".into(),
                phoneme_tokens: Vec::new(),
                phone_tokens: Vec::new(),
                syllables: Vec::new(),
                boundaries: Vec::new(),
                prosody: ProsodyTrack::default(),
                warnings: Vec::new(),
                syntax: SyntaxSupervision::default(),
            }],
            repair_examples: Vec::new(),
            word_supervision: Vec::new(),
            masked_word_examples: Vec::new(),
        };
        let phoneme_vocab = build_phoneme_vocab(std::slice::from_ref(&row));
        let phone_vocab = build_phone_vocab(std::slice::from_ref(&row));
        assert_eq!(phoneme_targets(&row, &phoneme_vocab).len(), 3);
        assert_eq!(phone_targets(&row, &phone_vocab).len(), 3);
        assert!(phoneme_targets(&row, &phoneme_vocab)
            .iter()
            .all(|id| *id > 0));
        assert!(phone_targets(&row, &phone_vocab).iter().all(|id| *id > 0));
        assert_eq!(feature_targets(&row, FeatureAxis::Place), vec![4, 5, 3]);
        assert_eq!(feature_targets(&row, FeatureAxis::Manner), vec![2, 6, 2]);
        assert_eq!(feature_targets(&row, FeatureAxis::Voicing), vec![2, 3, 2]);
        assert_eq!(feature_targets(&row, FeatureAxis::Syllabic), vec![2, 3, 2]);
        assert_eq!(feature_targets(&row, FeatureAxis::Height), vec![5, 4, 5]);
        assert_eq!(feature_targets(&row, FeatureAxis::Backness), vec![1, 2, 1]);
        assert_eq!(feature_targets(&row, FeatureAxis::Rounding), vec![1, 2, 1]);
    }

    #[test]
    fn compact_word_targets_exclude_blank() {
        let row = LibriSpeechUtterance {
            utterance_id: "u".into(),
            speaker_id: "s".into(),
            chapter_id: "c".into(),
            audio_path: "a.flac".into(),
            mel_path: "m.mel.bin".into(),
            num_frames: 10,
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            duration_ms: 100,
            transcript: "ONE TWO THREE".into(),
            sentences: Vec::new(),
            repair_examples: Vec::new(),
            word_supervision: vec![
                WordSupervision {
                    word: "ONE".into(),
                    word_index: 0,
                    sentence_index: 0,
                    sentence_word_index: 0,
                    start_char: 0,
                    end_char: 3,
                    start_frame: 0,
                    end_frame: 3,
                    phonemes: "w ʌ n".into(),
                    phones: "w ʌ n".into(),
                    previous_word: None,
                    next_word: Some("TWO".into()),
                },
                WordSupervision {
                    word: "TWO".into(),
                    word_index: 1,
                    sentence_index: 0,
                    sentence_word_index: 1,
                    start_char: 4,
                    end_char: 7,
                    start_frame: 3,
                    end_frame: 6,
                    phonemes: "t u".into(),
                    phones: "t u".into(),
                    previous_word: Some("ONE".into()),
                    next_word: Some("THREE".into()),
                },
            ],
            masked_word_examples: Vec::new(),
        };
        let vocab = build_word_vocab(&[row.clone()]);
        let current = current_word_targets(&row, &vocab);
        assert_eq!(current.len(), 2);
        assert!(current.iter().all(|id| *id != 0));
        let (padded, lengths, width) = pad_compact_targets(vec![current], vocab.get_id(WORD_UNK));
        assert_eq!(lengths, vec![2]);
        assert_eq!(width, 2);
        assert!(padded.iter().all(|id| *id != 0));
    }

    #[test]
    fn sentence_supervision_includes_phonemes() {
        let cfg = InterpretationConfig::default();
        let detector = SentenceDetectorDialog::new().unwrap();
        let rows = sentence_supervision(&detector, "HELLO WORLD.", 100, &cfg).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].boundary_label, BOUNDARY_EMIT);
        assert!(!rows[0].phonemes.is_empty());
        assert!(!rows[0].phoneme_tokens.is_empty());
        assert!(rows[0].syntax.parse_ok);
        assert!(!rows[0].syntax.words.is_empty());
        assert!(!rows[0].syntax.links.is_empty());
        assert!(rows[0].end_frame <= 100);
        assert!(!rows[0]
            .phones
            .split_whitespace()
            .any(|token| token == "word"));
        assert!(!rows[0].phones.split_whitespace().any(|token| token == "|"));
    }

    #[test]
    fn sentence_supervision_places_stress_before_syllable_onset() {
        let cfg = InterpretationConfig::default();
        let detector = SentenceDetectorDialog::new().unwrap();
        let rows = sentence_supervision(&detector, "Reflection.", 100, &cfg).unwrap();

        assert!(
            rows[0].phonemes.contains("ɹɪˈflɛk.ʃən"),
            "stress should precede the stressed syllable onset: {}",
            rows[0].phonemes
        );
        assert!(
            !rows[0].phonemes.contains("ɹɪflˈɛ"),
            "stress should not be anchored to the vowel: {}",
            rows[0].phonemes
        );
    }

    #[test]
    fn enrich_row_supervision_repairs_recovered_phoneme_extents() {
        let cfg = InterpretationConfig::default();
        let detector = SentenceDetectorDialog::new().unwrap();
        let mut row = test_utterance("u", "features/u.mel.bin", 100);
        row.transcript = "Reflection.".into();
        row.sentences = vec![SentenceSupervision {
            text: "Reflection.".into(),
            start_char: 0,
            end_char: 11,
            start_frame: 0,
            end_frame: 100,
            boundary_label: BOUNDARY_EMIT.into(),
            terminal: Some('.'),
            phonemes: "ɹɪflˈɛkʃʌn".into(),
            phones: String::new(),
            phoneme_tokens: Vec::new(),
            phone_tokens: Vec::new(),
            syllables: Vec::new(),
            boundaries: Vec::new(),
            prosody: ProsodyTrack::default(),
            warnings: Vec::new(),
            syntax: SyntaxSupervision::default(),
        }];

        enrich_row_supervision(&mut row, &detector, &cfg).unwrap();

        assert!(row.sentences[0].phonemes.contains("ɹɪˈflɛk.ʃən"));
        assert!(!row.sentences[0].phonemes.contains("ɹɪflˈɛ"));
        assert!(!row.word_supervision.is_empty());
    }

    #[test]
    fn repair_supervision_generates_mishear_correction() {
        let cfg = InterpretationConfig::default();
        let detector = SentenceDetectorDialog::new().unwrap();
        let rows = sentence_supervision(&detector, "I WENT TO TOWN.", 100, &cfg).unwrap();
        let repairs = repair_supervision(&rows);
        assert_eq!(repairs.len(), 1);
        assert_eq!(repairs[0].repair_label, BOUNDARY_REPAIR);
        assert_ne!(repairs[0].misheard_text, repairs[0].corrected_text);
        assert_eq!(repairs[0].corrected_text, "I WENT TO TOWN.");
    }

    #[test]
    fn phone_vocab_uses_sentence_phone_strings() {
        let cfg = InterpretationConfig::default();
        let detector = SentenceDetectorDialog::new().unwrap();
        let sentences = sentence_supervision(&detector, "HELLO WORLD.", 100, &cfg).unwrap();
        let word_supervision = word_supervision(&sentences);
        let row = LibriSpeechUtterance {
            utterance_id: "u".into(),
            speaker_id: "s".into(),
            chapter_id: "c".into(),
            audio_path: "a.flac".into(),
            mel_path: "m.mel.bin".into(),
            num_frames: 100,
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            duration_ms: 1000,
            transcript: "HELLO WORLD.".into(),
            repair_examples: repair_supervision(&sentences),
            masked_word_examples: masked_word_examples(&word_supervision, "HELLO WORLD."),
            word_supervision,
            sentences,
        };
        let vocab = build_phone_vocab(&[row]);
        assert!(vocab.size() > 1);
    }

    #[test]
    fn word_supervision_tracks_context_words() {
        let cfg = InterpretationConfig::default();
        let detector = SentenceDetectorDialog::new().unwrap();
        let sentences = sentence_supervision(&detector, "ONE TWO THREE FOUR.", 120, &cfg).unwrap();
        let words = word_supervision(&sentences);
        assert_eq!(words.len(), 4);
        assert_eq!(words[1].word, "TWO");
        assert_eq!(words[1].previous_word.as_deref(), Some("ONE"));
        assert_eq!(words[1].next_word.as_deref(), Some("THREE"));
        assert!(!words[1].phonemes.is_empty());
    }

    #[test]
    fn word_supervision_uses_structured_token_word_indices() {
        let cfg = InterpretationConfig::default();
        let detector = SentenceDetectorDialog::new().unwrap();
        let sentences = sentence_supervision(&detector, "HELLO WORLD.", 100, &cfg).unwrap();
        let words = word_supervision(&sentences);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "HELLO");
        assert_eq!(words[1].word, "WORLD");
        assert_eq!(words[0].phonemes, "h ʌ l oʊ");
        assert_eq!(words[1].phonemes, "w ɝ l d");
        assert_eq!(words[0].phones, "h ə l oʊ");
        assert_eq!(words[1].phones, "w ɝ ɫ d");
        assert_ne!(words[0].phonemes, sentences[0].phonemes);
        assert_ne!(words[1].phonemes, sentences[0].phonemes);
        assert!(!words.iter().any(|word| word.phones.contains("word")));
        assert!(!words.iter().any(|word| word.phones.contains('|')));
    }

    #[test]
    fn masked_word_examples_choose_non_edge_word() {
        let cfg = InterpretationConfig::default();
        let detector = SentenceDetectorDialog::new().unwrap();
        let sentences = sentence_supervision(&detector, "ONE TWO THREE FOUR.", 120, &cfg).unwrap();
        let words = word_supervision(&sentences);
        let masked = masked_word_examples(&words, "ONE TWO THREE FOUR.");
        assert_eq!(masked.len(), 1);
        assert_eq!(masked[0].masked_word, "THREE");
        assert!(masked[0].left_context.contains("ONE TWO"));
        assert!(masked[0].right_context.contains("FOUR"));
    }

    #[test]
    fn word_vocab_contains_words_and_specials() {
        let row = LibriSpeechUtterance {
            utterance_id: "u".into(),
            speaker_id: "s".into(),
            chapter_id: "c".into(),
            audio_path: "a.flac".into(),
            mel_path: "m.mel.bin".into(),
            num_frames: 10,
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            duration_ms: 100,
            transcript: "HELLO WORLD".into(),
            sentences: Vec::new(),
            repair_examples: Vec::new(),
            word_supervision: vec![WordSupervision {
                word: "HELLO".into(),
                word_index: 0,
                sentence_index: 0,
                sentence_word_index: 0,
                start_char: 0,
                end_char: 5,
                start_frame: 0,
                end_frame: 5,
                phonemes: "h ə l oʊ".into(),
                phones: "h ə l oʊ".into(),
                previous_word: None,
                next_word: Some("WORLD".into()),
            }],
            masked_word_examples: vec![MaskedWordExample {
                left_context: "".into(),
                right_context: "WORLD".into(),
                masked_word: "HELLO".into(),
                masked_word_phonemes: "h ə l oʊ".into(),
                start_frame: 0,
                end_frame: 5,
                source: "test".into(),
            }],
        };
        let vocab = build_word_vocab(&[row]);
        assert_eq!(vocab.tokens[0], WORD_BLANK);
        assert_eq!(vocab.tokens[1], WORD_UNK);
        assert_eq!(vocab.tokens[2], WORD_NUM);
        assert!(vocab.get_id("hello") > 1);
    }

    #[test]
    fn word_vocab_normalizes_surface_words() {
        assert_eq!(normalize_word_token("'Bye").as_deref(), Some("bye"));
        assert_eq!(normalize_word_token("'Don't").as_deref(), Some("don't"));
        assert_eq!(normalize_word_token("A").as_deref(), Some("a"));
        assert_eq!(normalize_word_token("1258").as_deref(), Some(WORD_NUM));
        assert_eq!(normalize_word_token("69th").as_deref(), Some(WORD_NUM));

        let mut row = test_utterance("u", "features/u.mel.bin", 100);
        row.word_supervision = vec![
            WordSupervision {
                word: "'Don't".into(),
                word_index: 0,
                sentence_index: 0,
                sentence_word_index: 0,
                start_char: 0,
                end_char: 6,
                start_frame: 0,
                end_frame: 10,
                phonemes: String::new(),
                phones: String::new(),
                previous_word: Some("A".into()),
                next_word: Some("1258".into()),
            },
            WordSupervision {
                word: "'Don't".into(),
                word_index: 1,
                sentence_index: 0,
                sentence_word_index: 1,
                start_char: 7,
                end_char: 13,
                start_frame: 10,
                end_frame: 20,
                phonemes: String::new(),
                phones: String::new(),
                previous_word: Some("'Don't".into()),
                next_word: None,
            },
        ];
        let vocab = build_word_vocab(&[row.clone()]);
        assert!(vocab.get_id("don't") > 1);
        assert!(!vocab.token_to_id.contains_key("'Don't"));
        assert_eq!(word_id(&vocab, "'Don't"), vocab.get_id("don't"));
        assert_eq!(word_id(&vocab, "1258"), vocab.get_id(WORD_NUM));
        assert_eq!(word_id(&vocab, "BUFFALOKEKILLER"), vocab.get_id(WORD_UNK));
    }

    #[test]
    fn syntax_vocabs_and_labels_include_parser_targets() {
        let cfg = InterpretationConfig::default();
        let detector = SentenceDetectorDialog::new().unwrap();
        let mut row = LibriSpeechUtterance {
            utterance_id: "u".into(),
            speaker_id: "s".into(),
            chapter_id: "c".into(),
            audio_path: "a.flac".into(),
            mel_path: "m.mel.bin".into(),
            num_frames: 100,
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            duration_ms: 1000,
            transcript: "THE QUICK FOX JUMPS.".into(),
            sentences: Vec::new(),
            repair_examples: Vec::new(),
            word_supervision: Vec::new(),
            masked_word_examples: Vec::new(),
        };
        enrich_row_supervision(&mut row, &detector, &cfg).unwrap();
        let pos_vocab = build_syntax_pos_vocab(&[row.clone()]);
        let link_vocab = build_syntax_link_vocab(&[row.clone()]);
        let offset_vocab = build_syntax_head_offset_vocab(&[row.clone()]);

        assert!(pos_vocab.size() > 2);
        assert!(link_vocab.size() > 2);
        assert!(offset_vocab.size() > 1);
        assert!(syntax_pos_labels_for(&row, &pos_vocab, 100)
            .into_iter()
            .any(|id| id != 0));
        assert!(syntax_link_labels_for(&row, &link_vocab, 100)
            .into_iter()
            .any(|id| id != 0));
        assert!(syntax_head_offset_labels_for(&row, &offset_vocab, 100)
            .into_iter()
            .any(|id| id != 0));
        assert!(parse_ok_labels_for(&row, 100).into_iter().any(|id| id != 0));
        assert!(phrase_boundary_labels_for(&row, 100)
            .into_iter()
            .any(|id| id != 0));
    }

    #[test]
    fn masking_marks_configured_frame_spans() {
        let config = InterpretationTrainConfig {
            mask_every_n_frames: 5,
            mask_span_frames: 2,
            ..InterpretationTrainConfig::default()
        };
        assert!(frame_is_masked(0, &config));
        assert!(frame_is_masked(1, &config));
        assert!(!frame_is_masked(2, &config));
        assert!(frame_is_masked(5, &config));
    }

    #[test]
    fn mel_file_round_trips() {
        let dir = Path::new("target/interpretation-tests");
        fs::create_dir_all(dir).unwrap();
        let path = dir.join("roundtrip.mel.bin");
        let rows = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        write_mel_file(&path, &rows, 3).unwrap();
        assert_eq!(valid_mel_frames(&path, 3).unwrap(), Some(2));
        assert_eq!(valid_mel_frames(&path, 4).unwrap(), None);
        assert_eq!(read_mel_file(&path).unwrap(), rows);
        assert!(!path.with_extension("mel.bin.part").exists());
    }

    #[test]
    fn recovery_keeps_only_rows_with_valid_mel_files() {
        let dir = Path::new("target/interpretation-tests/recovery");
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir.join("features")).unwrap();
        let good_mel = dir.join("features/good.mel.bin");
        write_mel_file(&good_mel, &[vec![1.0, 2.0, 3.0]], 3).unwrap();
        fs::write(dir.join("features/bad.mel.bin"), b"partial").unwrap();

        let good = test_utterance("good", "features/good.mel.bin", 1);
        let bad = test_utterance("bad", "features/bad.mel.bin", 1);
        let missing = test_utterance("missing", "features/missing.mel.bin", 1);
        let rows_path = dir.join("utterances.jsonl");
        fs::write(
            &rows_path,
            format!(
                "{}\n{}\n{}\n{{not-json\n",
                serde_json::to_string(&good).unwrap(),
                serde_json::to_string(&bad).unwrap(),
                serde_json::to_string(&missing).unwrap()
            ),
        )
        .unwrap();

        let config = InterpretationConfig {
            mel_bins: 3,
            compact_audio_features: false,
            ..InterpretationConfig::default()
        };
        let rows = recover_utterance_rows(&rows_path, dir, &config).unwrap();
        assert_eq!(rows, vec![good]);
    }

    fn test_utterance(id: &str, mel_path: &str, frames: usize) -> LibriSpeechUtterance {
        LibriSpeechUtterance {
            utterance_id: id.to_string(),
            speaker_id: "speaker".to_string(),
            chapter_id: "chapter".to_string(),
            audio_path: "audio.flac".to_string(),
            mel_path: mel_path.to_string(),
            num_frames: frames,
            sample_rate_hz: InterpretationConfig::default().sample_rate_hz,
            duration_ms: 100,
            transcript: "HELLO".to_string(),
            sentences: Vec::new(),
            repair_examples: Vec::new(),
            word_supervision: Vec::new(),
            masked_word_examples: Vec::new(),
        }
    }
}
