//! LibriSpeech utterance-level streaming ASR scaffold.
//!
//! V1 prepares LibriSpeech-style FLAC/transcript pairs, writes log-Mel feature
//! files durably, enriches each utterance with seams sentence splits and speech
//! phonemicizer output, and trains a small streaming frame classifier with CTC
//! style greedy collapse. Word-context and masked-word heads use Burn's native
//! CTC loss over compact target sequences.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::{CTCLossConfig, CrossEntropyLossConfig, Reduction};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use seams::SentenceDetectorDialog;
use serde::{Deserialize, Serialize};
use speaking::data::notation::openepd::render_openepd_phonemes;
use speaking::segment::TerminalPunctuation;
use speaking::syntax::{
    HeuristicLinkGrammarParser, LinkGrammarParser, PartOfSpeech, SentenceSyntaxAnalysis,
    SyntacticLinkKind,
};
use speaking::{
    EnglishPhonemicizer, PhonemicizeRequest, Phonemicizer, ProsodyTrack, SpeechBoundaryToken,
    Syllable, VarietyId,
};
use tongues_core::Vocab;
use tongues_neural::{make_recorder, write_manifest, ModelArtifactManifest, TrainState};

pub const FAMILY: &str = "interpretation";
pub const ARCHITECTURE: &str = "streaming-mel-native-ctc";
pub const DEFAULT_DATASET_ID: &str = "librispeech-mini-v0";
pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 16_000;
pub const DEFAULT_MEL_BINS: usize = 80;
pub const CTC_BLANK: &str = "<CTC_BLANK>";
pub const WORD_BLANK: &str = "<WORD_BLANK>";
pub const WORD_UNK: &str = "<WORD_UNK>";
pub const BOUNDARY_CONTINUE: &str = "<boundary:continue>";
pub const BOUNDARY_EMIT: &str = "<boundary:emit>";
pub const BOUNDARY_REPAIR: &str = "<boundary:repair>";

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
    pub variety: String,
    pub max_utterances: Option<usize>,
    pub download_url: String,
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
            variety: "en-US".to_string(),
            max_utterances: None,
            download_url: subset.archive_url().to_string(),
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
    #[serde(default = "default_word_mask_rate")]
    pub word_mask_rate: f32,
    #[serde(default = "default_mask_every_n_frames")]
    pub mask_every_n_frames: usize,
    #[serde(default = "default_mask_span_frames")]
    pub mask_span_frames: usize,
    pub max_frames: usize,
}

fn default_phone_loss_weight() -> f32 {
    0.25
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

fn default_word_mask_rate() -> f32 {
    1.0
}

fn default_mask_every_n_frames() -> usize {
    12
}

fn default_mask_span_frames() -> usize {
    3
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
            prev_word_loss_weight: 0.1,
            current_word_loss_weight: 0.2,
            next_word_loss_weight: 0.15,
            masked_word_loss_weight: 0.2,
            masked_word_phoneme_loss_weight: 0.15,
            repair_loss_weight: 0.15,
            masked_audio_loss_weight: 0.35,
            syntax_loss_weight: default_syntax_loss_weight(),
            word_mask_rate: default_word_mask_rate(),
            mask_every_n_frames: default_mask_every_n_frames(),
            mask_span_frames: default_mask_span_frames(),
            max_frames: 1600,
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
            boundary: LinearConfig::new(self.hidden_size, 3).init(device),
            phoneme: LinearConfig::new(self.hidden_size, self.phoneme_vocab_size).init(device),
            phone: LinearConfig::new(self.hidden_size, self.phone_vocab_size).init(device),
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
    boundary: Linear<B>,
    phoneme: Linear<B>,
    phone: Linear<B>,
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
            boundary_logits: self.boundary.forward(hidden.clone()),
            phoneme_logits: self.phoneme.forward(hidden.clone()),
            phone_logits: self.phone.forward(hidden.clone()),
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
    pub boundary_logits: Tensor<B, 3>,
    pub phoneme_logits: Tensor<B, 3>,
    pub phone_logits: Tensor<B, 3>,
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
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    fs::create_dir_all(out.join("features")).context("creating features directory")?;
    let archive = out.join("source.tar.gz");
    if !archive.exists() {
        progress(PrepareProgress::Stage {
            message: format!("Downloading {}", config.download_url),
        });
        download_to_part(&config.download_url, &archive, &mut progress)?;
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

    let transcripts = discover_transcripts(&source_dir)?;
    progress(PrepareProgress::Parse {
        transcripts: transcripts.len(),
    });
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
        if !row_has_syntax(row) {
            progress(PrepareProgress::Stage {
                message: format!("Enriching recovered syntax for {}", row.utterance_id),
            });
            enrich_row_supervision(row, &detector, config)?;
        }
    }
    if utterances_path.exists() {
        write_jsonl_atomic(&utterances_path, &rows, &mut progress)?;
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
    for item in selected_transcripts {
        if let Some(existing) = row_by_id.get(&item.utterance_id) {
            progress(PrepareProgress::Reuse {
                utterance_id: item.utterance_id,
                rows: existing.num_frames,
                path: out.join(&existing.mel_path).display().to_string(),
            });
            continue;
        }
        let samples = read_flac_mono(&item.audio_path)?;
        let rel_mel = PathBuf::from("features").join(format!("{}.mel.bin", item.utterance_id));
        let mel_path = out.join(&rel_mel);
        let frames = match valid_mel_frames(&mel_path, config.mel_bins)? {
            Some(frames) => {
                progress(PrepareProgress::Reuse {
                    utterance_id: item.utterance_id.clone(),
                    rows: frames,
                    path: mel_path.display().to_string(),
                });
                frames
            }
            None => {
                let features = log_mel_features(&samples, config);
                write_mel_file(&mel_path, &features, config.mel_bins)?;
                progress(PrepareProgress::Features {
                    utterance_id: item.utterance_id.clone(),
                    rows: features.len(),
                    path: mel_path.display().to_string(),
                });
                features.len()
            }
        };
        let transcript = normalize_librispeech_text(&item.transcript);
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
    utterance_writer.flush()?;

    let mut shuffled = rows;
    shuffled.shuffle(&mut rand::rngs::StdRng::seed_from_u64(config.seed));
    let n = shuffled.len();
    let train_end = ((n as f64) * config.train_frac).round().min(n as f64) as usize;
    let valid_end = (train_end + ((n as f64) * config.valid_frac).round() as usize).min(n);
    let train = shuffled[..train_end].to_vec();
    let valid = shuffled[train_end..valid_end].to_vec();
    let test = shuffled[valid_end..].to_vec();
    write_jsonl_atomic(&out.join("train.jsonl"), &train, &mut progress)?;
    write_jsonl_atomic(&out.join("valid.jsonl"), &valid, &mut progress)?;
    write_jsonl_atomic(&out.join("test.jsonl"), &test, &mut progress)?;
    let vocab = build_text_vocab([&train[..], &valid[..], &test[..]].concat().as_slice());
    fs::write(
        out.join("vocab.json"),
        serde_json::to_string_pretty(&vocab)?,
    )?;
    let phoneme_vocab =
        build_phoneme_vocab([&train[..], &valid[..], &test[..]].concat().as_slice());
    fs::write(
        out.join("phoneme_vocab.json"),
        serde_json::to_string_pretty(&phoneme_vocab)?,
    )?;
    let phone_vocab = build_phone_vocab([&train[..], &valid[..], &test[..]].concat().as_slice());
    fs::write(
        out.join("phone_vocab.json"),
        serde_json::to_string_pretty(&phone_vocab)?,
    )?;
    let word_vocab = build_word_vocab([&train[..], &valid[..], &test[..]].concat().as_slice());
    fs::write(
        out.join("word_vocab.json"),
        serde_json::to_string_pretty(&word_vocab)?,
    )?;
    let syntax_pos_vocab =
        build_syntax_pos_vocab([&train[..], &valid[..], &test[..]].concat().as_slice());
    fs::write(
        out.join("syntax_pos_vocab.json"),
        serde_json::to_string_pretty(&syntax_pos_vocab)?,
    )?;
    let syntax_link_vocab =
        build_syntax_link_vocab([&train[..], &valid[..], &test[..]].concat().as_slice());
    fs::write(
        out.join("syntax_link_vocab.json"),
        serde_json::to_string_pretty(&syntax_link_vocab)?,
    )?;
    let syntax_head_offset_vocab =
        build_syntax_head_offset_vocab([&train[..], &valid[..], &test[..]].concat().as_slice());
    fs::write(
        out.join("syntax_head_offset_vocab.json"),
        serde_json::to_string_pretty(&syntax_head_offset_vocab)?,
    )?;
    fs::write(
        out.join("dataset_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    fs::write(out.join("README.md"), dataset_readme(config))?;
    Ok(PrepareReport {
        utterances: n,
        train_examples: train.len(),
        valid_examples: valid.len(),
        test_examples: test.len(),
    })
}

fn row_has_syntax(row: &LibriSpeechUtterance) -> bool {
    row.sentences
        .iter()
        .any(|sentence| !sentence.syntax.words.is_empty() || !sentence.syntax.links.is_empty())
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
    let part = path.with_extension("tar.gz.part");
    let response = ureq::get(url)
        .call()
        .with_context(|| format!("downloading {url}"))?;
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

fn discover_transcripts(root: &Path) -> Result<Vec<TranscriptItem>> {
    let mut transcript_files = Vec::new();
    collect_files(root, "trans.txt", &mut transcript_files)?;
    let mut out = Vec::new();
    for path in transcript_files {
        let parent = path.parent().context("transcript path has no parent")?;
        let file = File::open(&path)?;
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

pub fn log_mel_features(samples: &[f32], config: &InterpretationConfig) -> Vec<Vec<f32>> {
    let window = ((config.sample_rate_hz as f32 * config.window_ms) / 1000.0).round() as usize;
    let hop = ((config.sample_rate_hz as f32 * config.hop_ms) / 1000.0).round() as usize;
    if samples.len() < window || window == 0 || hop == 0 {
        return Vec::new();
    }
    let n_fft = window.next_power_of_two();
    let mut rows = Vec::new();
    for start in (0..=samples.len() - window).step_by(hop) {
        let mut power = vec![0.0f32; n_fft / 2 + 1];
        for k in 0..=n_fft / 2 {
            let mut re = 0.0;
            let mut im = 0.0;
            for n in 0..window {
                let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / window as f32).cos();
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / n_fft as f32;
                let x = samples[start + n] * w;
                re += x * angle.cos();
                im += x * angle.sin();
            }
            power[k] = re * re + im * im;
        }
        rows.push(mel_project(&power, config.mel_bins));
    }
    rows
}

fn mel_project(power: &[f32], bins: usize) -> Vec<f32> {
    if bins == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0; bins];
    for (i, value) in power.iter().enumerate() {
        let bin = i * bins / power.len().max(1);
        out[bin.min(bins - 1)] += *value;
    }
    out.into_iter().map(|v| (v.max(1e-8)).ln()).collect()
}

fn write_mel_file(path: &Path, features: &[Vec<f32>], mel_bins: usize) -> Result<()> {
    let part = path.with_extension("mel.bin.part");
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

fn sentence_supervision(
    detector: &SentenceDetectorDialog,
    transcript: &str,
    num_frames: usize,
    config: &InterpretationConfig,
) -> Result<Vec<SentenceSupervision>> {
    let detected = detector
        .detect_sentences_borrowed(transcript)
        .context("detecting transcript sentences")?;
    let phonemicizer = EnglishPhonemicizer;
    let syntax_parser = HeuristicLinkGrammarParser;
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
            variety: VarietyId(config.variety.clone()),
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
            phonemes: render_openepd_phonemes(&phonemicized.phonemes),
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
            variety: VarietyId(config.variety.clone()),
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
            phonemes: render_openepd_phonemes(&phonemicized.phonemes),
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
    parser: &impl LinkGrammarParser,
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
            speaking::Spec::Known(id) => Some(
                id.as_str()
                    .rsplit('.')
                    .next()
                    .unwrap_or(id.as_str())
                    .to_string(),
            ),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
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
        let phoneme_chunks = split_tokens_for_words(&sentence.phonemes, sentence_words.len());
        let phone_chunks = split_tokens_for_words(&sentence.phones, sentence_words.len());
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
    let part = path.with_extension(
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| format!("{ext}.part"))
            .unwrap_or_else(|| "part".to_string()),
    );
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
        let Some(frames) = valid_mel_frames(&mel_path, config.mel_bins)? else {
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
            for token in sentence.phonemes.split_whitespace() {
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
    let mut tokens = vec![WORD_BLANK.to_string(), WORD_UNK.to_string()];
    let mut set = BTreeSet::new();
    for row in rows {
        for word in &row.word_supervision {
            set.insert(word.word.clone());
        }
        for masked in &row.masked_word_examples {
            set.insert(masked.masked_word.clone());
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
    let mut model = if state_path.exists() {
        let state: TrainState = serde_json::from_str(&fs::read_to_string(&state_path)?)?;
        start_epoch = state.current_epoch + 1;
        best_val_loss = state.best_val_loss;
        let epoch_path = out_dir.join(format!("model-epoch-{}", state.current_epoch));
        if epoch_path.with_extension("bin").exists() {
            println!(
                "Resuming training from epoch {} checkpoint: {}",
                state.current_epoch,
                epoch_path.with_extension("bin").display()
            );
            model_config
                .init(device)
                .load_file(&epoch_path, &make_recorder(), device)?
        } else {
            model_config.init(device)
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
    let mut patience = 0usize;
    for epoch in start_epoch..=train_config.epochs {
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
        )?;
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
        println!(
            "Epoch {:3} | train_loss={:.4} val_loss={:.4} wer={:.3} boundary_f1={:.3} repair_f1={:.3} phoneme_ter={:.3} phone_ter={:.3} audio_mse={:.4}",
            epoch,
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
        fs::write(
            &state_path,
            serde_json::to_string_pretty(&TrainState {
                current_epoch: epoch,
                best_val_loss,
            })?,
        )?;
        if report.loss < best_val_loss - 1e-5 {
            best_val_loss = report.loss;
            patience = 0;
            eval_model.save_file(model_path, &make_recorder())?;
            fs::write(
                &state_path,
                serde_json::to_string_pretty(&TrainState {
                    current_epoch: epoch,
                    best_val_loss,
                })?,
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
) -> Result<f32> {
    let mut indices: Vec<_> = (0..rows.len()).collect();
    indices.shuffle(rng);
    let batches = (rows.len() + config.batch_size - 1) / config.batch_size;
    let pb = indicatif::ProgressBar::new(batches as u64);
    pb.set_style(indicatif::ProgressStyle::default_bar().template(
        &format!("{{spinner:.green}} LibriSpeech epoch {epoch}/{} [{{bar:40.cyan/blue}}] {{pos}}/{{len}} loss={{msg}}", config.epochs)
    )?.progress_chars("#>-"));
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
        let grads = GradientsParams::from_grads(loss.backward(), model);
        *model = optimizer.step(config.learning_rate, model.clone(), grads);
        total += loss.into_scalar().elem::<f32>();
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
    boundary_labels: Tensor<B, 2, Int>,
    phoneme_labels: Tensor<B, 2, Int>,
    phone_labels: Tensor<B, 2, Int>,
    input_lengths: Tensor<B, 1, Int>,
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
    let mel_bins = DEFAULT_MEL_BINS;
    let mut mel = Vec::new();
    let mut mel_target = Vec::new();
    let mut transcript_labels = Vec::new();
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
    for row in rows {
        let input_len = row.num_frames.min(max_frames).max(1);
        input_lengths.push(input_len as i32);
        let features = read_mel_file(&data_dir.join(&row.mel_path))?;
        for frame in 0..max_frames {
            let src = features.get(frame);
            let masked = frame_is_masked(frame, config) || frame_is_word_masked(row, frame, config);
            for bin in 0..mel_bins {
                let value = src.and_then(|r| r.get(bin)).copied().unwrap_or(0.0);
                mel_target.push(value);
                mel.push(if masked { 0.0 } else { value });
            }
        }
        transcript_labels.extend(proportional_labels(&row.transcript, vocab, max_frames));
        boundary_labels.extend(boundary_labels_for(row, max_frames));
        phoneme_labels.extend(proportional_phoneme_labels(row, phoneme_vocab, max_frames));
        phone_labels.extend(proportional_phone_labels(row, phone_vocab, max_frames));
        syntax_pos_labels.extend(syntax_pos_labels_for(row, syntax_pos_vocab, max_frames));
        syntax_link_labels.extend(syntax_link_labels_for(row, syntax_link_vocab, max_frames));
        syntax_head_offset_labels.extend(syntax_head_offset_labels_for(
            row,
            syntax_head_offset_vocab,
            max_frames,
        ));
        parse_ok_labels.extend(parse_ok_labels_for(row, max_frames));
        phrase_boundary_labels.extend(phrase_boundary_labels_for(row, max_frames));
        prev_word_sequences.push(ctc_target_within_input(
            previous_word_targets(row, word_vocab),
            input_len,
        ));
        current_word_sequences.push(ctc_target_within_input(
            current_word_targets(row, word_vocab),
            input_len,
        ));
        next_word_sequences.push(ctc_target_within_input(
            next_word_targets(row, word_vocab),
            input_len,
        ));
        masked_word_sequences.push(ctc_target_within_input(
            masked_word_targets(row, word_vocab),
            input_len,
        ));
        masked_word_phoneme_sequences.push(ctc_target_within_input(
            masked_word_phoneme_targets(row, phoneme_vocab),
            input_len,
        ));
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
    let boundary_loss = ce_loss(output.boundary_logits, batch.boundary_labels, usize::MAX);
    let phoneme_loss = ce_loss(output.phoneme_logits, batch.phoneme_labels, 0);
    let phone_loss = ce_loss(output.phone_logits, batch.phone_labels, 0);
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
        + prev_word_loss * config.prev_word_loss_weight
        + current_word_loss * config.current_word_loss_weight
        + next_word_loss * config.next_word_loss_weight
        + masked_word_loss * config.masked_word_loss_weight
        + masked_word_phoneme_loss * config.masked_word_phoneme_loss_weight
        + syntax_loss * config.syntax_loss_weight
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
    let id = vocab.get_id(word);
    if id == 0 {
        vocab.get_id(WORD_UNK).max(1)
    } else {
        id
    }
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
        .flat_map(|s| s.phonemes.split_whitespace().map(str::to_string))
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
    let mut token_errors = 0usize;
    let mut token_total = 0usize;
    let mut word_errors = 0usize;
    let mut word_total = 0usize;
    let mut boundary_tp = 0usize;
    let mut boundary_fp = 0usize;
    let mut boundary_fn = 0usize;
    let mut repair_tp = 0usize;
    let mut repair_fp = 0usize;
    let mut repair_fn = 0usize;
    let mut phoneme_errors = 0usize;
    let mut phoneme_total = 0usize;
    let mut phone_errors = 0usize;
    let mut phone_total = 0usize;
    let mut audio_mse_total = 0.0f32;
    let mut prev_correct = 0usize;
    let mut prev_total = 0usize;
    let mut current_correct = 0usize;
    let mut current_total = 0usize;
    let mut next_correct = 0usize;
    let mut next_total = 0usize;
    let mut masked_word_correct = 0usize;
    let mut masked_word_total = 0usize;
    let mut masked_phoneme_errors = 0usize;
    let mut masked_phoneme_total = 0usize;
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
                boundary_logits: output.boundary_logits.clone(),
                phoneme_logits: output.phoneme_logits.clone(),
                phone_logits: output.phone_logits.clone(),
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
        let boundary_preds = argmax_ids(output.boundary_logits);
        let phoneme_preds = argmax_ids(output.phoneme_logits);
        let phone_preds = argmax_ids(output.phone_logits);
        let prev_word_preds = argmax_ids(output.prev_word_logits);
        let current_word_preds = argmax_ids(output.current_word_logits);
        let next_word_preds = argmax_ids(output.next_word_logits);
        let masked_word_preds = argmax_ids(output.masked_word_logits);
        let masked_word_phoneme_preds = argmax_ids(output.masked_word_phoneme_logits);
        for (i, row) in chunk.iter().enumerate() {
            let decoded = greedy_collapse(&transcript_preds[i], vocab);
            let ref_chars = row.transcript.chars().collect::<Vec<_>>();
            let hyp_chars = decoded.chars().collect::<Vec<_>>();
            token_errors += edit_distance(&ref_chars, &hyp_chars);
            token_total += ref_chars.len();
            let ref_words = row.transcript.split_whitespace().collect::<Vec<_>>();
            let hyp_words = decoded.split_whitespace().collect::<Vec<_>>();
            word_errors += edit_distance(&ref_words, &hyp_words);
            word_total += ref_words.len();
            let gold = boundary_labels_for(row, boundary_preds[i].len());
            for (pred, gold) in boundary_preds[i].iter().zip(gold) {
                match (*pred == 1, gold == 1) {
                    (true, true) => boundary_tp += 1,
                    (true, false) => boundary_fp += 1,
                    (false, true) => boundary_fn += 1,
                    _ => {}
                }
                match (*pred == 2, gold == 2) {
                    (true, true) => repair_tp += 1,
                    (true, false) => repair_fp += 1,
                    (false, true) => repair_fn += 1,
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
            phoneme_errors += edit_distance(&ref_phonemes, &hyp_phonemes);
            phoneme_total += ref_phonemes.len();
            let decoded_phones = greedy_collapse(&phone_preds[i], phone_vocab);
            let ref_phones = row
                .sentences
                .iter()
                .flat_map(|s| s.phones.split_whitespace())
                .collect::<Vec<_>>();
            let hyp_phones = decoded_phones.split_whitespace().collect::<Vec<_>>();
            phone_errors += edit_distance(&ref_phones, &hyp_phones);
            phone_total += ref_phones.len();
            let decoded_prev = ctc_greedy_decode(&prev_word_preds[i], 0);
            let target_prev = previous_word_targets(row, word_vocab);
            let (correct, total) = sequence_accuracy(&decoded_prev, &target_prev);
            prev_correct += correct;
            prev_total += total;
            let decoded_current = ctc_greedy_decode(&current_word_preds[i], 0);
            let target_current = current_word_targets(row, word_vocab);
            let (correct, total) = sequence_accuracy(&decoded_current, &target_current);
            current_correct += correct;
            current_total += total;
            let decoded_next = ctc_greedy_decode(&next_word_preds[i], 0);
            let target_next = next_word_targets(row, word_vocab);
            let (correct, total) = sequence_accuracy(&decoded_next, &target_next);
            next_correct += correct;
            next_total += total;
            let decoded_masked_word = ctc_greedy_decode(&masked_word_preds[i], 0);
            let target_masked_word = masked_word_targets(row, word_vocab);
            let (correct, total) = sequence_accuracy(&decoded_masked_word, &target_masked_word);
            masked_word_correct += correct;
            masked_word_total += total;
            let decoded_masked_phonemes = ctc_greedy_decode(&masked_word_phoneme_preds[i], 0);
            let decoded_masked_phonemes = decoded_masked_phonemes
                .into_iter()
                .map(|id| id as i32)
                .collect::<Vec<_>>();
            let target_masked_phonemes = masked_word_phoneme_targets(row, phoneme_vocab);
            masked_phoneme_errors +=
                edit_distance(&target_masked_phonemes, &decoded_masked_phonemes);
            masked_phoneme_total += target_masked_phonemes.len();
        }
    }
    let precision = boundary_tp as f32 / (boundary_tp + boundary_fp).max(1) as f32;
    let recall = boundary_tp as f32 / (boundary_tp + boundary_fn).max(1) as f32;
    let repair_precision = repair_tp as f32 / (repair_tp + repair_fp).max(1) as f32;
    let repair_recall = repair_tp as f32 / (repair_tp + repair_fn).max(1) as f32;
    Ok(EvalReport {
        examples: eval_rows.len(),
        loss: total_loss / batches.max(1) as f32,
        token_error_rate: token_errors as f32 / token_total.max(1) as f32,
        word_error_rate: word_errors as f32 / word_total.max(1) as f32,
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
        phoneme_token_error_rate: phoneme_errors as f32 / phoneme_total.max(1) as f32,
        phone_token_error_rate: phone_errors as f32 / phone_total.max(1) as f32,
        masked_audio_mse: audio_mse_total / batches.max(1) as f32,
        prev_word_accuracy: prev_correct as f32 / prev_total.max(1) as f32,
        current_word_accuracy: current_correct as f32 / current_total.max(1) as f32,
        next_word_accuracy: next_correct as f32 / next_total.max(1) as f32,
        masked_word_accuracy: masked_word_correct as f32 / masked_word_total.max(1) as f32,
        masked_word_phoneme_token_error_rate: masked_phoneme_errors as f32
            / masked_phoneme_total.max(1) as f32,
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

pub fn stream_from_samples<B: Backend>(
    model: &AsrModel<B>,
    samples: &[f32],
    vocab: &Vocab,
    word_vocab: &Vocab,
    phoneme_vocab: &Vocab,
    config: &InterpretationConfig,
    device: &B::Device,
) -> Result<StreamEvent> {
    let features = log_mel_features(samples, config);
    let batch = Tensor::<B, 3>::from_data(
        TensorData::new(
            features.iter().flatten().copied().collect::<Vec<_>>(),
            [1, features.len(), config.mel_bins],
        ),
        device,
    );
    let output = model.forward(batch);
    let ids = argmax_ids(output.transcript_logits.clone());
    let partial = ids
        .first()
        .map(|ids| greedy_collapse(ids, vocab))
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
        "# LibriSpeech ASR dataset\n\nDataset id: `{}`\nSubset: `{:?}`\n\nPrepared by `tongues interpretation prepare`. Rows contain FLAC provenance, durable log-Mel feature paths, seams sentence labels, and speech phonemicizer supervision.\n\nLibriSpeech is distributed from OpenSLR under CC BY 4.0. Preserve source attribution when redistributing derived artifacts.\n",
        config.dataset_id, config.subset
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
        assert!(vocab.get_id("HELLO") > 1);
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
