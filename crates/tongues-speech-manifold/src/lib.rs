//! Shared speech-manifold model family.
//!
//! V1 uses a shared seq2seq Transformer over multimodal token views. Audio
//! synthesis provenance is captured during prepare, while acoustic-frame inputs
//! are represented by deterministic vector summaries when real feature
//! extraction is not available.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::fs;
use std::io::{BufRead, Read, Write};
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use speech::data::notation::openepd::normalize_openepd_ipa;
use speech::{
    AcousticFrame, AcousticVector, EnglishPhonemicizer, Formant, PhoneToken, PhonemicizeRequest,
    Phonemicizer, Spec, Stress, Syllable, TimeSpan, VarietyId,
};
use tongues_core::{Vocab, BOS_ID, EOS_ID, PAD_ID};
use tongues_data::{collate_batch, Seq2SeqExample};
pub use tongues_g2p2g::{ModelConfig, Seq2SeqModel};
use tongues_neural::{
    make_recorder, seq2seq_cross_entropy_loss, tensor_seq2seq_batch, write_manifest,
    ModelArtifactManifest, TrainState,
};

const USER_AGENT: &str = "tongues-speech-manifold/0.1";
pub const FAMILY: &str = "speech-manifold";
pub const ARCHITECTURE: &str = "shared-multimodal-transformer";
pub const DEFAULT_DATASET_ID: &str = "openepd-synth-v0";

const DEFAULT_TASKS: &[SpeechManifoldTask] = &[
    SpeechManifoldTask::SpellingToIpa,
    SpeechManifoldTask::IpaToSpelling,
    SpeechManifoldTask::IpaToPhones,
    SpeechManifoldTask::Stress,
    SpeechManifoldTask::Syllables,
    SpeechManifoldTask::AcousticToIpa,
    SpeechManifoldTask::IpaToAcoustic,
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechManifoldConfig {
    pub dataset_id: String,
    pub train_frac: f64,
    pub valid_frac: f64,
    pub seed: u64,
    #[serde(default = "default_tasks")]
    pub tasks: Vec<SpeechManifoldTask>,
    #[serde(default = "default_synthesis_backends")]
    pub synthesis_backends: Vec<String>,
    pub allow_placeholder_acoustics: bool,
    pub max_examples: Option<usize>,
    pub max_audio_examples_per_backend: usize,
    pub max_espeak_examples: usize,
    pub max_google_translate_examples: usize,
    pub max_wiktionary_audio_examples: usize,
    pub max_styletts2_examples: usize,
    pub max_piper_examples: usize,
    pub max_anyspeak_examples: usize,
    pub max_mock_examples: usize,
    pub max_wikimedia_commons_examples: usize,
    pub max_wikimedia_commons_lookup_attempts: usize,
    #[serde(default)]
    pub anyspeak_dir: Option<String>,
    #[serde(default = "default_anyspeak_python")]
    pub anyspeak_python: String,
    #[serde(default = "default_anyspeak_voice_tags")]
    pub anyspeak_voice_tags: Vec<String>,
    pub include_reference_uris: bool,
    #[serde(default)]
    pub external_audio_manifests: Vec<String>,
    #[serde(default = "default_espeak_voices")]
    pub espeak_voices: Vec<String>,
    #[serde(default = "default_google_translate_speeds")]
    pub google_translate_speeds: Vec<f32>,
}

impl Default for SpeechManifoldConfig {
    fn default() -> Self {
        Self {
            dataset_id: DEFAULT_DATASET_ID.to_string(),
            train_frac: 0.8,
            valid_frac: 0.1,
            seed: 42,
            tasks: default_tasks(),
            synthesis_backends: default_synthesis_backends(),
            allow_placeholder_acoustics: true,
            max_examples: Some(5000),
            max_audio_examples_per_backend: 64,
            max_espeak_examples: 64,
            max_google_translate_examples: 64,
            max_wiktionary_audio_examples: 32,
            max_styletts2_examples: 16,
            max_piper_examples: 32,
            max_anyspeak_examples: 16,
            max_mock_examples: 64,
            max_wikimedia_commons_examples: 32,
            max_wikimedia_commons_lookup_attempts: 256,
            anyspeak_dir: None,
            anyspeak_python: default_anyspeak_python(),
            anyspeak_voice_tags: default_anyspeak_voice_tags(),
            include_reference_uris: true,
            external_audio_manifests: Vec::new(),
            espeak_voices: default_espeak_voices(),
            google_translate_speeds: default_google_translate_speeds(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechManifoldTrainConfig {
    pub learning_rate: f64,
    pub weight_decay: f32,
    pub dropout: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub early_stopping_patience: usize,
    pub seed: u64,
    #[serde(default = "default_tasks")]
    pub tasks: Vec<SpeechManifoldTask>,
    pub allow_placeholder_acoustics: bool,
}

impl Default for SpeechManifoldTrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            weight_decay: 1e-4,
            dropout: 0.1,
            batch_size: 64,
            epochs: 20,
            early_stopping_patience: 5,
            seed: 0,
            tasks: default_tasks(),
            allow_placeholder_acoustics: true,
        }
    }
}

fn default_tasks() -> Vec<SpeechManifoldTask> {
    DEFAULT_TASKS.to_vec()
}

fn default_synthesis_backends() -> Vec<String> {
    [
        "espeak-ng",
        "google-translate",
        "wiktionary-audio",
        "wikimedia-commons-audio",
        "anyspeak",
        "styletts2",
        "piper",
        "mock",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn default_espeak_voices() -> Vec<String> {
    ["en-us", "en-gb", "en-sc", "en-uk-north"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn default_google_translate_speeds() -> Vec<f32> {
    vec![1.0, 0.85, 0.7]
}

fn default_anyspeak_python() -> String {
    "python3".to_string()
}

fn default_anyspeak_voice_tags() -> Vec<String> {
    ["[RYAN]", "[VIVIAN]", "[AUTO]"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn join_display<T: fmt::Display>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SpeechManifoldTask {
    SpellingToIpa,
    IpaToSpelling,
    IpaToPhones,
    Stress,
    Syllables,
    AcousticToIpa,
    IpaToAcoustic,
}

impl SpeechManifoldTask {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "spelling-to-ipa" => Some(Self::SpellingToIpa),
            "ipa-to-spelling" => Some(Self::IpaToSpelling),
            "ipa-to-phones" => Some(Self::IpaToPhones),
            "stress" => Some(Self::Stress),
            "syllables" => Some(Self::Syllables),
            "acoustic-to-ipa" => Some(Self::AcousticToIpa),
            "ipa-to-acoustic" => Some(Self::IpaToAcoustic),
            _ => None,
        }
    }

    pub fn token(self) -> &'static str {
        match self {
            Self::SpellingToIpa => "<SM_SPELLING_TO_IPA>",
            Self::IpaToSpelling => "<SM_IPA_TO_SPELLING>",
            Self::IpaToPhones => "<SM_IPA_TO_PHONES>",
            Self::Stress => "<SM_STRESS>",
            Self::Syllables => "<SM_SYLLABLES>",
            Self::AcousticToIpa => "<SM_ACOUSTIC_TO_IPA>",
            Self::IpaToAcoustic => "<SM_IPA_TO_ACOUSTIC>",
        }
    }
}

impl fmt::Display for SpeechManifoldTask {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::SpellingToIpa => "spelling-to-ipa",
            Self::IpaToSpelling => "ipa-to-spelling",
            Self::IpaToPhones => "ipa-to-phones",
            Self::Stress => "stress",
            Self::Syllables => "syllables",
            Self::AcousticToIpa => "acoustic-to-ipa",
            Self::IpaToAcoustic => "ipa-to-acoustic",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechManifoldExample {
    pub id: String,
    pub text: String,
    pub spelling: String,
    pub broad_ipa: String,
    pub narrow_ipa: String,
    pub stress_pattern: String,
    pub syllables: Vec<Syllable>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub phones: Vec<PhoneToken>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub acoustic_frames: Vec<AcousticFrame>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<u32>,
    pub source_backend: String,
    pub provenance: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reference_uris: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalAudioManifestRow {
    pub word: String,
    pub audio_uri: String,
    #[serde(default)]
    pub broad_ipa: Option<String>,
    pub source: String,
    pub license: String,
    pub attribution: String,
    #[serde(default)]
    pub source_url: Option<String>,
    #[serde(default)]
    pub speaker: Option<String>,
    #[serde(default)]
    pub variety: Option<String>,
    #[serde(default)]
    pub pronunciation_assurance: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalitySchema {
    pub version: u32,
    pub tasks: Vec<String>,
    pub placeholder_acoustics_allowed: bool,
    pub output_type: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    pub examples: usize,
    pub task: SpeechManifoldTask,
    pub loss: f32,
    pub exact_match_accuracy: f32,
    pub token_accuracy: f32,
    pub placeholder_acoustic_metrics: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeReport {
    pub split: String,
    pub examples: usize,
    pub placeholder_acoustic_examples: usize,
    pub source_backends: BTreeMap<String, usize>,
    pub tasks: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenEpdEntry {
    rarity: f32,
    ipa: BTreeMap<String, String>,
}

const OPENEPD_SOURCE_PREFERENCE: &[&str] = &[
    "misaki_gold",
    "cmu",
    "misaki_silver",
    "phonemicchart",
    "wiktionary",
    "wikipron",
];

pub fn prepare_dataset(out: &Path, config: &SpeechManifoldConfig) -> Result<()> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let audio_dir = out.join("audio");
    fs::create_dir_all(&audio_dir).with_context(|| format!("creating {}", audio_dir.display()))?;

    println!("Preparing speech-manifold dataset");
    println!("  output: {}", out.display());
    println!("  dataset_id: {}", config.dataset_id);
    println!(
        "  split: train={:.2} valid={:.2} test={:.2} seed={}",
        config.train_frac,
        config.valid_frac,
        (1.0 - config.train_frac - config.valid_frac).max(0.0),
        config.seed
    );
    println!(
        "  max_examples: {}",
        config
            .max_examples
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unbounded".to_string())
    );
    println!("  tasks: {}", join_display(&config.tasks));
    println!(
        "  synthesis_backends: {}",
        config.synthesis_backends.join(", ")
    );
    if config.external_audio_manifests.is_empty() {
        println!("  external_audio_manifests: none");
    } else {
        println!(
            "  external_audio_manifests: {}",
            config.external_audio_manifests.join(", ")
        );
    }
    println!(
        "  audio caps: per_backend={} espeak={} google_translate={} wiktionary={} commons={} styletts2={} piper={} anyspeak={} mock={}",
        config.max_audio_examples_per_backend,
        config.max_espeak_examples,
        config.max_google_translate_examples,
        config.max_wiktionary_audio_examples,
        config.max_wikimedia_commons_examples,
        config.max_styletts2_examples,
        config.max_piper_examples,
        config.max_anyspeak_examples,
        config.max_mock_examples
    );
    println!("  network policy: checking robots.txt before every network audio fetch");

    let mut examples = openepd_examples(config, &audio_dir)?;
    apply_external_audio_manifests(&mut examples, config)?;
    write_splits(out, config, &examples)?;
    write_vocab(out, &examples, &config.tasks)?;
    write_sidecars(out, config)?;
    println!("Prepare complete: wrote {} examples", examples.len());
    Ok(())
}

fn openepd_examples(
    config: &SpeechManifoldConfig,
    audio_dir: &Path,
) -> Result<Vec<SpeechManifoldExample>> {
    let raw: BTreeMap<String, OpenEpdEntry> =
        serde_json::from_str(open_english_pronouncing_dictionary::CORPUS_JSON)
            .context("parsing embedded OpenEPD JSON")?;
    println!("Loaded {} OpenEPD entries", raw.len());
    let mut audio_state = AudioSynthesisState::new(config, audio_dir)?;

    let mut examples = Vec::new();
    let mut skipped_word_filter = 0usize;
    let mut skipped_ipa = 0usize;
    let mut skipped_phonemicizer = 0usize;
    for (base_word, entry) in raw {
        if config.max_examples.is_some_and(|max| examples.len() >= max) {
            break;
        }
        if !is_prepare_word(&base_word) {
            skipped_word_filter += 1;
            continue;
        }
        let Some(raw_ipa) = preferred_openepd_ipa(&entry.ipa) else {
            skipped_ipa += 1;
            continue;
        };
        let broad_ipa = match normalize_openepd_ipa(raw_ipa) {
            Ok(ipa) => ipa,
            Err(_) => {
                skipped_ipa += 1;
                continue;
            }
        };
        let source_labels = entry.ipa.keys().cloned().collect::<Vec<_>>();
        let Some(mut example) =
            example_from_word(&base_word, &broad_ipa, entry.rarity, &source_labels, config)
        else {
            skipped_phonemicizer += 1;
            continue;
        };

        audio_state.try_attach_audio(&mut example);

        examples.push(example);
        if examples.len() <= 10 || examples.len() % 250 == 0 {
            println!(
                "  prepared {:>5} examples; latest={} backend={}",
                examples.len(),
                base_word,
                examples
                    .last()
                    .map(|example| example.source_backend.as_str())
                    .unwrap_or("unknown")
            );
        }
    }
    println!(
        "OpenEPD conversion summary: examples={} skipped_word_filter={} skipped_ipa={} skipped_phonemicizer={}",
        examples.len(),
        skipped_word_filter,
        skipped_ipa,
        skipped_phonemicizer
    );
    audio_state.print_summary();
    Ok(examples)
}

#[derive(Debug)]
struct AudioSynthesisState<'a> {
    config: &'a SpeechManifoldConfig,
    audio_dir: &'a Path,
    espeak_available: bool,
    google_enabled: bool,
    wiktionary_enabled: bool,
    wikimedia_commons_enabled: bool,
    styletts2_enabled: bool,
    piper_enabled: bool,
    anyspeak_enabled: bool,
    mock_enabled: bool,
    espeak_count: usize,
    google_count: usize,
    wiktionary_count: usize,
    wikimedia_commons_count: usize,
    wikimedia_commons_lookups: usize,
    styletts2_count: usize,
    piper_count: usize,
    anyspeak_count: usize,
    mock_count: usize,
    espeak_failures: usize,
    wikimedia_commons_skips: BTreeMap<String, usize>,
    styletts2_skips: BTreeMap<String, usize>,
    piper_skips: BTreeMap<String, usize>,
    anyspeak_skips: BTreeMap<String, usize>,
    mock_failures: usize,
    google_skips: BTreeMap<String, usize>,
    wiktionary_skips: BTreeMap<String, usize>,
    next_backend: usize,
    robots: RobotsCache,
}

impl<'a> AudioSynthesisState<'a> {
    fn new(config: &'a SpeechManifoldConfig, audio_dir: &'a Path) -> Result<Self> {
        fs::create_dir_all(audio_dir.join("espeak-ng"))?;
        fs::create_dir_all(audio_dir.join("google-translate"))?;
        fs::create_dir_all(audio_dir.join("wikimedia-commons"))?;
        fs::create_dir_all(audio_dir.join("styletts2"))?;
        fs::create_dir_all(audio_dir.join("piper"))?;
        fs::create_dir_all(audio_dir.join("anyspeak"))?;
        fs::create_dir_all(audio_dir.join("mock"))?;
        Ok(Self {
            config,
            audio_dir,
            espeak_available: config.backend_enabled("espeak-ng")
                && Command::new("espeak-ng").arg("--version").output().is_ok(),
            google_enabled: config.backend_enabled("google-translate"),
            wiktionary_enabled: config.backend_enabled("wiktionary-audio"),
            wikimedia_commons_enabled: config.backend_enabled("wikimedia-commons-audio"),
            styletts2_enabled: config.backend_enabled("styletts2"),
            piper_enabled: config.backend_enabled("piper"),
            anyspeak_enabled: config.backend_enabled("anyspeak")
                && anyspeak_run_local(config).is_some(),
            mock_enabled: config.backend_enabled("mock"),
            espeak_count: 0,
            google_count: 0,
            wiktionary_count: 0,
            wikimedia_commons_count: 0,
            wikimedia_commons_lookups: 0,
            styletts2_count: 0,
            piper_count: 0,
            anyspeak_count: 0,
            mock_count: 0,
            espeak_failures: 0,
            wikimedia_commons_skips: BTreeMap::new(),
            styletts2_skips: BTreeMap::new(),
            piper_skips: BTreeMap::new(),
            anyspeak_skips: BTreeMap::new(),
            mock_failures: 0,
            google_skips: BTreeMap::new(),
            wiktionary_skips: BTreeMap::new(),
            next_backend: 0,
            robots: RobotsCache::default(),
        })
    }

    fn try_attach_audio(&mut self, example: &mut SpeechManifoldExample) {
        let backends = self.enabled_backends();
        if backends.is_empty() {
            return;
        }
        for offset in 0..backends.len() {
            let index = (self.next_backend + offset) % backends.len();
            let backend = backends[index];
            if self.try_backend(example, backend) {
                self.next_backend = (index + 1) % backends.len();
                return;
            }
        }
    }

    fn enabled_backends(&self) -> Vec<AudioBackend> {
        let mut backends = Vec::new();
        for name in &self.config.synthesis_backends {
            match name.as_str() {
                "espeak-ng" => backends.push(AudioBackend::Espeak),
                "google-translate" => backends.push(AudioBackend::GoogleTranslate),
                "wiktionary-audio" => backends.push(AudioBackend::WiktionaryAudio),
                "wikimedia-commons-audio" => backends.push(AudioBackend::WikimediaCommonsAudio),
                "styletts2" => backends.push(AudioBackend::StyleTts2),
                "piper" => backends.push(AudioBackend::Piper),
                "anyspeak" => backends.push(AudioBackend::AnySpeak),
                "mock" => backends.push(AudioBackend::Mock),
                _ => {}
            }
        }
        backends
    }

    fn try_backend(&mut self, example: &mut SpeechManifoldExample, backend: AudioBackend) -> bool {
        match backend {
            AudioBackend::Espeak => self.try_espeak(example),
            AudioBackend::GoogleTranslate => self.try_google_translate(example),
            AudioBackend::WiktionaryAudio => self.try_wiktionary_audio(example),
            AudioBackend::WikimediaCommonsAudio => self.try_wikimedia_commons_audio(example),
            AudioBackend::StyleTts2 => self.try_styletts2(example),
            AudioBackend::Piper => self.try_piper(example),
            AudioBackend::AnySpeak => self.try_anyspeak(example),
            AudioBackend::Mock => self.try_mock(example),
        }
    }

    fn try_espeak(&mut self, example: &mut SpeechManifoldExample) -> bool {
        let cap = self
            .config
            .max_espeak_examples
            .min(self.config.max_audio_examples_per_backend);
        if !self.espeak_available || self.espeak_count >= cap {
            return false;
        }
        if self.config.espeak_voices.is_empty() {
            return false;
        }
        let voice =
            self.config.espeak_voices[self.espeak_count % self.config.espeak_voices.len()].clone();
        let wav_path = self.audio_dir.join("espeak-ng").join(format!(
            "{}-{}.wav",
            safe_id(&voice),
            example.id
        ));
        if synthesize_espeak(&example.spelling, &voice, &wav_path).is_err() {
            self.espeak_failures += 1;
            return false;
        }
        self.espeak_count += 1;
        example.audio_uri = Some(path_to_uri(&wav_path));
        example.sample_rate_hz = Some(22_050);
        example.source_backend = format!("espeak-ng:{voice}+placeholder-acoustics");
        example
            .provenance
            .push_str(&format!(" + espeak-ng voice {voice} WAV"));
        true
    }

    fn try_styletts2(&mut self, example: &mut SpeechManifoldExample) -> bool {
        let cap = self
            .config
            .max_styletts2_examples
            .min(self.config.max_audio_examples_per_backend);
        if !self.styletts2_enabled || self.styletts2_count >= cap {
            return false;
        }
        let wav_path = self
            .audio_dir
            .join("styletts2")
            .join(format!("{}.wav", example.id));
        match synthesize_with_tongues_speak(&example.spelling, "styletts2", &wav_path) {
            Ok(()) => {}
            Err(error) => {
                *self.styletts2_skips.entry(error.to_string()).or_insert(0) += 1;
                if self.styletts2_skips.values().sum::<usize>() <= 3 {
                    println!("  styletts2 skipped {}: {error}", example.spelling);
                }
                self.styletts2_enabled = false;
                return false;
            }
        }
        self.styletts2_count += 1;
        example.audio_uri = Some(path_to_uri(&wav_path));
        example.sample_rate_hz = Some(24_000);
        example.source_backend = "styletts2+placeholder-acoustics".to_string();
        example
            .provenance
            .push_str(" + native StyleTTS2 WAV via `tongues speak`");
        true
    }

    fn try_piper(&mut self, example: &mut SpeechManifoldExample) -> bool {
        let cap = self
            .config
            .max_piper_examples
            .min(self.config.max_audio_examples_per_backend);
        if !self.piper_enabled || self.piper_count >= cap {
            return false;
        }
        let wav_path = self
            .audio_dir
            .join("piper")
            .join(format!("{}.wav", example.id));
        match synthesize_with_tongues_speak(&example.spelling, "piper", &wav_path) {
            Ok(()) => {}
            Err(error) => {
                *self.piper_skips.entry(error.to_string()).or_insert(0) += 1;
                if self.piper_skips.values().sum::<usize>() <= 3 {
                    println!("  piper skipped {}: {error}", example.spelling);
                }
                self.piper_enabled = false;
                return false;
            }
        }
        self.piper_count += 1;
        example.audio_uri = Some(path_to_uri(&wav_path));
        example.sample_rate_hz = Some(22_050);
        example.source_backend = "piper+placeholder-acoustics".to_string();
        example
            .provenance
            .push_str(" + native Piper WAV via `tongues speak`");
        true
    }

    fn try_anyspeak(&mut self, example: &mut SpeechManifoldExample) -> bool {
        let cap = self
            .config
            .max_anyspeak_examples
            .min(self.config.max_audio_examples_per_backend);
        if !self.anyspeak_enabled || self.anyspeak_count >= cap {
            return false;
        }
        let voice_tag = self
            .config
            .anyspeak_voice_tags
            .get(self.anyspeak_count % self.config.anyspeak_voice_tags.len().max(1))
            .map(String::as_str)
            .unwrap_or("[AUTO]");
        let mp3_path = self.audio_dir.join("anyspeak").join(format!(
            "{}-{}.mp3",
            safe_id(voice_tag),
            example.id
        ));
        match synthesize_anyspeak(&example.spelling, voice_tag, self.config, &mp3_path) {
            Ok(()) => {}
            Err(error) => {
                *self.anyspeak_skips.entry(error.to_string()).or_insert(0) += 1;
                if self.anyspeak_skips.values().sum::<usize>() <= 3 {
                    println!("  anyspeak skipped {}: {error}", example.spelling);
                }
                self.anyspeak_enabled = false;
                return false;
            }
        }
        self.anyspeak_count += 1;
        example.audio_uri = Some(path_to_uri(&mp3_path));
        example.sample_rate_hz = None;
        example.source_backend = format!("anyspeak:{voice_tag}+placeholder-acoustics");
        example
            .provenance
            .push_str(&format!(" + AnySpeak/Qwen3-TTS MP3 voice tag {voice_tag}"));
        true
    }

    fn try_mock(&mut self, example: &mut SpeechManifoldExample) -> bool {
        let cap = self
            .config
            .max_mock_examples
            .min(self.config.max_audio_examples_per_backend);
        if !self.mock_enabled || self.mock_count >= cap {
            return false;
        }
        let wav_path = self
            .audio_dir
            .join("mock")
            .join(format!("{}.wav", example.id));
        if synthesize_mock_wav(&example.spelling, &wav_path).is_err() {
            self.mock_failures += 1;
            return false;
        }
        self.mock_count += 1;
        example.audio_uri = Some(path_to_uri(&wav_path));
        example.sample_rate_hz = Some(24_000);
        example.source_backend = "mock:deterministic-tone+placeholder-acoustics".to_string();
        example.provenance.push_str(" + deterministic mock WAV");
        true
    }

    fn try_google_translate(&mut self, example: &mut SpeechManifoldExample) -> bool {
        let cap = self
            .config
            .max_google_translate_examples
            .min(self.config.max_audio_examples_per_backend);
        if !self.google_enabled || self.google_count >= cap {
            return false;
        }
        if self.config.google_translate_speeds.is_empty() {
            return false;
        }
        let speed = self.config.google_translate_speeds
            [self.google_count % self.config.google_translate_speeds.len()];
        let mp3_path = self
            .audio_dir
            .join("google-translate")
            .join(format!("speed-{:.2}-{}.mp3", speed, example.id));
        match synthesize_google_translate(&example.spelling, speed, &mp3_path, &mut self.robots) {
            Ok(()) => {}
            Err(error) => {
                *self.google_skips.entry(error.to_string()).or_insert(0) += 1;
                if self.google_skips.values().sum::<usize>() <= 3 {
                    println!("  google-translate skipped {}: {error}", example.spelling);
                }
                return false;
            }
        }
        self.google_count += 1;
        example.audio_uri = Some(path_to_uri(&mp3_path));
        example.sample_rate_hz = None;
        example.source_backend = format!("google-translate:speed-{speed:.2}+placeholder-acoustics");
        example
            .provenance
            .push_str(&format!(" + Google Translate TTS MP3 speed {speed:.2}"));
        true
    }

    fn try_wiktionary_audio(&mut self, example: &mut SpeechManifoldExample) -> bool {
        let cap = self
            .config
            .max_wiktionary_audio_examples
            .min(self.config.max_audio_examples_per_backend);
        if !self.wiktionary_enabled || self.wiktionary_count >= cap {
            return false;
        }
        let ogg_path = self
            .audio_dir
            .join("wiktionary")
            .join(format!("{}.ogg", example.id));
        if fs::create_dir_all(ogg_path.parent().expect("wiktionary audio path has parent")).is_err()
        {
            return false;
        }
        let source_url =
            match fetch_wiktionary_audio(&example.spelling, &ogg_path, &mut self.robots) {
                Ok(Some(source_url)) => source_url,
                Ok(None) => {
                    *self
                        .wiktionary_skips
                        .entry("no audio file found".to_string())
                        .or_insert(0) += 1;
                    return false;
                }
                Err(error) => {
                    *self.wiktionary_skips.entry(error.to_string()).or_insert(0) += 1;
                    if self.wiktionary_skips.values().sum::<usize>() <= 3 {
                        println!("  wiktionary-audio skipped {}: {error}", example.spelling);
                    }
                    return false;
                }
            };
        self.wiktionary_count += 1;
        example.audio_uri = Some(path_to_uri(&ogg_path));
        example.sample_rate_hz = None;
        example.source_backend = "wiktionary-audio+placeholder-acoustics".to_string();
        example
            .provenance
            .push_str(&format!(" + Wiktionary/Wikimedia audio {source_url}"));
        true
    }

    fn try_wikimedia_commons_audio(&mut self, example: &mut SpeechManifoldExample) -> bool {
        let cap = self
            .config
            .max_wikimedia_commons_examples
            .min(self.config.max_audio_examples_per_backend);
        if !self.wikimedia_commons_enabled || self.wikimedia_commons_count >= cap {
            return false;
        }
        if self.wikimedia_commons_lookups >= self.config.max_wikimedia_commons_lookup_attempts {
            return false;
        }
        self.wikimedia_commons_lookups += 1;
        let ogg_path = self
            .audio_dir
            .join("wikimedia-commons")
            .join(format!("{}.ogg", example.id));
        let source =
            match fetch_wikimedia_commons_audio(&example.spelling, &ogg_path, &mut self.robots) {
                Ok(Some(source)) => source,
                Ok(None) => {
                    *self
                        .wikimedia_commons_skips
                        .entry("no exact Commons pronunciation file found".to_string())
                        .or_insert(0) += 1;
                    return false;
                }
                Err(error) => {
                    *self
                        .wikimedia_commons_skips
                        .entry(error.to_string())
                        .or_insert(0) += 1;
                    if self.wikimedia_commons_skips.values().sum::<usize>() <= 3 {
                        println!(
                            "  wikimedia-commons-audio skipped {}: {error}",
                            example.spelling
                        );
                    }
                    return false;
                }
            };
        self.wikimedia_commons_count += 1;
        example.audio_uri = Some(path_to_uri(&ogg_path));
        example.sample_rate_hz = None;
        example.source_backend = format!(
            "wikimedia-commons-audio:{}+placeholder-acoustics",
            source.license
        );
        example.provenance.push_str(&format!(
            " + Wikimedia Commons human pronunciation `{}` license `{}` attribution `{}` media `{}`",
            source.source_url, source.license, source.attribution, source.media_url
        ));
        example.reference_uris.push(source.source_url);
        example.reference_uris.sort();
        example.reference_uris.dedup();
        true
    }

    fn print_summary(&self) {
        println!("Audio synthesis summary:");
        println!(
            "  espeak-ng: written={} failures={} voices={}",
            self.espeak_count,
            self.espeak_failures,
            self.config.espeak_voices.join(", ")
        );
        println!(
            "  google-translate: written={} skipped={}",
            self.google_count,
            self.google_skips.values().sum::<usize>()
        );
        for (reason, count) in &self.google_skips {
            println!("    skip: {count} x {reason}");
        }
        println!(
            "  wiktionary-audio: written={} skipped={}",
            self.wiktionary_count,
            self.wiktionary_skips.values().sum::<usize>()
        );
        for (reason, count) in &self.wiktionary_skips {
            println!("    skip: {count} x {reason}");
        }
        println!(
            "  wikimedia-commons-audio: written={} skipped={} lookups={}",
            self.wikimedia_commons_count,
            self.wikimedia_commons_skips.values().sum::<usize>(),
            self.wikimedia_commons_lookups
        );
        for (reason, count) in &self.wikimedia_commons_skips {
            println!("    skip: {count} x {reason}");
        }
        println!(
            "  styletts2: written={} skipped={}",
            self.styletts2_count,
            self.styletts2_skips.values().sum::<usize>()
        );
        for (reason, count) in &self.styletts2_skips {
            println!("    skip: {count} x {reason}");
        }
        println!(
            "  piper: written={} skipped={}",
            self.piper_count,
            self.piper_skips.values().sum::<usize>()
        );
        for (reason, count) in &self.piper_skips {
            println!("    skip: {count} x {reason}");
        }
        println!(
            "  anyspeak: written={} skipped={}",
            self.anyspeak_count,
            self.anyspeak_skips.values().sum::<usize>()
        );
        for (reason, count) in &self.anyspeak_skips {
            println!("    skip: {count} x {reason}");
        }
        println!(
            "  mock: written={} failures={}",
            self.mock_count, self.mock_failures
        );
        self.robots.print_summary();
    }
}

#[derive(Debug, Clone, Copy)]
enum AudioBackend {
    Espeak,
    GoogleTranslate,
    WiktionaryAudio,
    WikimediaCommonsAudio,
    StyleTts2,
    Piper,
    AnySpeak,
    Mock,
}

impl SpeechManifoldConfig {
    fn backend_enabled(&self, backend: &str) -> bool {
        self.synthesis_backends.iter().any(|name| name == backend)
    }
}

fn example_from_word(
    base_word: &str,
    broad_ipa: &str,
    rarity: f32,
    source_labels: &[String],
    config: &SpeechManifoldConfig,
) -> Option<SpeechManifoldExample> {
    let phonemicized = EnglishPhonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: base_word.to_string(),
            variety: VarietyId("en-US".to_string()),
            style: None,
        })
        .ok()?;
    if phonemicized
        .warnings
        .iter()
        .any(|warning| matches!(warning.kind, speech::PronunciationWarningKind::GuessedWord))
    {
        return None;
    }
    let narrow_ipa = syllables_to_ipa(&phonemicized.syllables);
    let stress_pattern = stress_pattern(&phonemicized.syllables);
    let acoustic_frames = placeholder_acoustic_frames(base_word, broad_ipa, rarity);

    let source_summary = if source_labels.is_empty() {
        "unknown".to_string()
    } else {
        source_labels.join(",")
    };
    Some(SpeechManifoldExample {
        id: safe_id(base_word),
        text: base_word.to_string(),
        spelling: base_word.to_lowercase(),
        broad_ipa: broad_ipa.to_string(),
        narrow_ipa,
        stress_pattern,
        syllables: phonemicized.syllables,
        phones: phonemicized.phones,
        acoustic_frames,
        audio_uri: None,
        sample_rate_hz: None,
        source_backend: "mock-placeholder-acoustics".to_string(),
        provenance: format!(
            "OpenEPD lexical source ({source_summary}) + speech phonemicizer + deterministic placeholder acoustic frames"
        ),
        reference_uris: reference_uris(base_word, source_labels, config.include_reference_uris),
    })
}

fn preferred_openepd_ipa(ipa: &BTreeMap<String, String>) -> Option<&str> {
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

fn reference_uris(word: &str, source_labels: &[String], enabled: bool) -> Vec<String> {
    if !enabled {
        return Vec::new();
    }
    let escaped = simple_url_component(word);
    let mut refs = vec![format!("https://www.dictionary.com/browse/{escaped}")];
    if source_labels
        .iter()
        .any(|label| label.starts_with("wikipron") || label.starts_with("wiktionary"))
    {
        refs.push(format!("https://en.wiktionary.org/wiki/{escaped}"));
    }
    refs
}

#[derive(Debug, Default)]
struct ExternalAudioImportSummary {
    accepted: usize,
    skipped_no_base_word: usize,
    skipped_missing_rights: usize,
    skipped_unverified_pronunciation: usize,
    skipped_parse: usize,
}

fn apply_external_audio_manifests(
    examples: &mut Vec<SpeechManifoldExample>,
    config: &SpeechManifoldConfig,
) -> Result<()> {
    if config.external_audio_manifests.is_empty() {
        return Ok(());
    }

    let base_by_word = examples
        .iter()
        .map(|example| (example.spelling.clone(), example.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut summary = ExternalAudioImportSummary::default();
    let mut imported_by_word = BTreeMap::<String, usize>::new();

    for manifest in &config.external_audio_manifests {
        println!("Importing external audio manifest: {manifest}");
        let file = match fs::File::open(manifest) {
            Ok(file) => file,
            Err(error) => {
                println!("  skipped manifest {manifest}: {error}");
                continue;
            }
        };
        let reader = std::io::BufReader::new(file);
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let row: ExternalAudioManifestRow = match serde_json::from_str(&line) {
                Ok(row) => row,
                Err(error) => {
                    summary.skipped_parse += 1;
                    if summary.skipped_parse <= 3 {
                        println!("  skipped external audio row: JSON parse error: {error}");
                    }
                    continue;
                }
            };
            let word = row.word.to_lowercase();
            let Some(base) = base_by_word.get(&word) else {
                summary.skipped_no_base_word += 1;
                continue;
            };
            if !has_rights_metadata(&row) {
                summary.skipped_missing_rights += 1;
                continue;
            }
            let Some(assurance) = pronunciation_assurance(base, &row) else {
                summary.skipped_unverified_pronunciation += 1;
                continue;
            };

            let index = imported_by_word.entry(word.clone()).or_insert(0);
            *index += 1;
            let mut imported = base.clone();
            imported.id = format!("{}-external-audio-{}", imported.id, index);
            imported.audio_uri = Some(row.audio_uri.clone());
            imported.sample_rate_hz = None;
            imported.source_backend = format!("external-audio:{}+{}", row.source, assurance);
            imported.provenance = external_audio_provenance(&row, &assurance);
            if let Some(source_url) = row.source_url.clone() {
                imported.reference_uris.push(source_url);
            }
            imported.reference_uris.sort();
            imported.reference_uris.dedup();
            examples.push(imported);
            summary.accepted += 1;
        }
    }

    println!("External audio import summary:");
    println!("  accepted={}", summary.accepted);
    println!("  skipped_no_base_word={}", summary.skipped_no_base_word);
    println!(
        "  skipped_missing_rights={}",
        summary.skipped_missing_rights
    );
    println!(
        "  skipped_unverified_pronunciation={}",
        summary.skipped_unverified_pronunciation
    );
    println!("  skipped_parse={}", summary.skipped_parse);
    Ok(())
}

fn has_rights_metadata(row: &ExternalAudioManifestRow) -> bool {
    !row.license.trim().is_empty() && !row.attribution.trim().is_empty()
}

fn pronunciation_assurance(
    base: &SpeechManifoldExample,
    row: &ExternalAudioManifestRow,
) -> Option<String> {
    if let Some(ipa) = &row.broad_ipa {
        if normalize_openepd_ipa(ipa).ok().as_deref() == Some(base.broad_ipa.as_str()) {
            return Some("openepd-ipa-match".to_string());
        }
    }
    match row.pronunciation_assurance.as_deref() {
        Some("single-word-pronunciation")
        | Some("source-pronunciation-entry")
        | Some("manually-verified") => Some(
            row.pronunciation_assurance
                .clone()
                .expect("matched Some above"),
        ),
        _ => None,
    }
}

fn external_audio_provenance(row: &ExternalAudioManifestRow, assurance: &str) -> String {
    let mut parts = vec![
        format!("external audio source `{}`", row.source),
        format!("license `{}`", row.license),
        format!("attribution `{}`", row.attribution),
        format!("pronunciation assurance `{assurance}`"),
    ];
    if let Some(speaker) = &row.speaker {
        parts.push(format!("speaker `{speaker}`"));
    }
    if let Some(variety) = &row.variety {
        parts.push(format!("variety `{variety}`"));
    }
    if let Some(source_url) = &row.source_url {
        parts.push(format!("source_url `{source_url}`"));
    }
    parts.join(" + ")
}

fn synthesize_espeak(text: &str, voice: &str, wav_path: &Path) -> Result<()> {
    let status = Command::new("espeak-ng")
        .args([
            "-v",
            voice,
            "-w",
            wav_path.to_str().context("non-UTF-8 WAV path")?,
            text,
        ])
        .status()
        .context("running espeak-ng")?;
    anyhow::ensure!(status.success(), "espeak-ng failed");
    Ok(())
}

fn synthesize_with_tongues_speak(text: &str, backend: &str, wav_path: &Path) -> Result<()> {
    if let Some(parent) = wav_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let current_exe = std::env::current_exe().context("resolving current tongues executable")?;
    let output = Command::new(current_exe)
        .args([
            "speak",
            "--backend",
            backend,
            "--quality",
            "fast",
            "--output",
            wav_path.to_str().context("non-UTF-8 WAV path")?,
            "--",
            text,
        ])
        .output()
        .with_context(|| format!("running tongues speak --backend {backend}"))?;
    anyhow::ensure!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

fn anyspeak_run_local(config: &SpeechManifoldConfig) -> Option<std::path::PathBuf> {
    config
        .anyspeak_dir
        .as_deref()
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("ANYSPEAK_DIR").map(std::path::PathBuf::from))
        .map(|dir| dir.join("run_local.py"))
        .filter(|path| path.exists())
}

fn synthesize_anyspeak(
    text: &str,
    voice_tag: &str,
    config: &SpeechManifoldConfig,
    mp3_path: &Path,
) -> Result<()> {
    if let Some(parent) = mp3_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let run_local = anyspeak_run_local(config)
        .context("AnySpeak is not configured; set anyspeak_dir or ANYSPEAK_DIR")?;
    let prompt = if voice_tag.trim().is_empty() || voice_tag == "[AUTO]" {
        text.to_string()
    } else {
        format!("{} {}", voice_tag.trim(), text)
    };
    let output = Command::new(&config.anyspeak_python)
        .arg(run_local)
        .arg(prompt)
        .arg("-o")
        .arg(mp3_path)
        .output()
        .context("running AnySpeak local CLI")?;
    anyhow::ensure!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

fn synthesize_mock_wav(text: &str, wav_path: &Path) -> Result<()> {
    if let Some(parent) = wav_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let sample_rate = 24_000u32;
    let samples_per_char = sample_rate as usize / 14;
    let silence = sample_rate as usize / 40;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(wav_path, spec)
        .with_context(|| format!("writing {}", wav_path.display()))?;
    for byte in text.bytes().filter(|byte| !byte.is_ascii_whitespace()) {
        let freq = 220.0 + f32::from(byte % 48) * 11.0;
        for index in 0..samples_per_char {
            let t = index as f32 / sample_rate as f32;
            let envelope = if index < 96 {
                index as f32 / 96.0
            } else if samples_per_char.saturating_sub(index) < 96 {
                samples_per_char.saturating_sub(index) as f32 / 96.0
            } else {
                1.0
            };
            let sample = (t * freq * std::f32::consts::TAU).sin() * 0.18 * envelope;
            writer.write_sample((sample * i16::MAX as f32) as i16)?;
        }
        for _ in 0..silence {
            writer.write_sample(0i16)?;
        }
    }
    writer.finalize()?;
    Ok(())
}

fn synthesize_google_translate(
    text: &str,
    speed: f32,
    mp3_path: &Path,
    robots: &mut RobotsCache,
) -> Result<()> {
    let url = tts_urls::google_translate::url_with_speed(text, "en", speed);
    download_to_file_checked(&url, mp3_path, robots)
}

fn fetch_wiktionary_audio(
    word: &str,
    out_path: &Path,
    robots: &mut RobotsCache,
) -> Result<Option<String>> {
    for file_name in wiktionary_audio_file_candidates(word) {
        let api_url = format!(
            "https://en.wiktionary.org/w/api.php?action=query&format=json&prop=imageinfo&iiprop=url&titles=File:{}",
            simple_url_component(&file_name)
        );
        ensure_robots_allowed(&api_url, robots)?;
        let response = ureq::get(&api_url)
            .header("User-Agent", USER_AGENT)
            .call()
            .with_context(|| format!("GET {api_url}"))?;
        let mut body = response.into_body();
        let mut reader = body.as_reader();
        let mut raw = String::new();
        reader.read_to_string(&mut raw)?;
        let value: serde_json::Value = serde_json::from_str(&raw)?;
        let Some(source_url) = value
            .pointer("/query/pages")
            .and_then(|pages| pages.as_object())
            .and_then(|pages| pages.values().next())
            .and_then(|page| page.get("imageinfo"))
            .and_then(|info| info.as_array())
            .and_then(|info| info.first())
            .and_then(|info| info.get("url"))
            .and_then(|url| url.as_str())
        else {
            continue;
        };
        if download_to_file_checked(source_url, out_path, robots).is_ok() {
            return Ok(Some(source_url.to_string()));
        }
    }
    Ok(None)
}

#[derive(Debug, Clone)]
struct CommonsAudioSource {
    media_url: String,
    source_url: String,
    license: String,
    attribution: String,
}

fn fetch_wikimedia_commons_audio(
    word: &str,
    out_path: &Path,
    robots: &mut RobotsCache,
) -> Result<Option<CommonsAudioSource>> {
    for file_name in wiktionary_audio_file_candidates(word) {
        let page_url = format!(
            "https://commons.wikimedia.org/wiki/File:{}",
            simple_url_component(&file_name)
        );
        ensure_robots_allowed(&page_url, robots)?;
        let response = match ureq::get(&page_url).header("User-Agent", USER_AGENT).call() {
            Ok(response) => response,
            Err(_) => continue,
        };
        let mut body = response.into_body();
        let mut reader = body.as_reader();
        let mut html = String::new();
        reader.read_to_string(&mut html)?;
        let Some(mut source) = parse_commons_audio_source(&html, &file_name, &page_url) else {
            continue;
        };
        if download_to_file_checked(&source.media_url, out_path, robots).is_ok() {
            source.source_url = page_url;
            return Ok(Some(source));
        }
    }
    Ok(None)
}

fn parse_commons_audio_source(
    html: &str,
    file_name: &str,
    page_url: &str,
) -> Option<CommonsAudioSource> {
    let encoded_file_name = file_name.replace(' ', "_");
    let media_url = first_between(html, "contentUrl\":\"", "\"")
        .or_else(|| first_between(html, "<source src=\"", "\""))
        .or_else(|| {
            first_between(html, "href=\"https://upload.wikimedia.org/", "\"")
                .map(|path| format!("https://upload.wikimedia.org/{path}"))
        })?;
    let media_url = media_url.replace("\\/", "/");
    if !media_url.contains(&encoded_file_name) && !media_url.contains(file_name) {
        return None;
    }
    let license = first_between(
        html,
        "licensetpl&#95;short\" style=\"display:none;\">",
        "</span>",
    )
    .or_else(|| {
        first_between(
            html,
            "licensetpl_short\" style=\"display:none;\">",
            "</span>",
        )
    })
    .or_else(|| first_between(html, "\"license\":\"", "\""))
    .map(|value| html_text(&value))
    .unwrap_or_else(|| "Wikimedia Commons file license".to_string());
    let attribution = first_between(html, "wbmi-snak-value--value'>", "<")
        .or_else(|| {
            first_between(html, "recorded by <a", "</a>")
                .and_then(|link| link.rsplit('>').next().map(str::to_string))
        })
        .or_else(|| first_between(html, "title=\"User:", "\""))
        .map(|value| html_text(&value))
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "Wikimedia Commons contributor".to_string());

    Some(CommonsAudioSource {
        media_url,
        source_url: page_url.to_string(),
        license,
        attribution,
    })
}

fn first_between(haystack: &str, start: &str, end: &str) -> Option<String> {
    let (_, rest) = haystack.split_once(start)?;
    let (value, _) = rest.split_once(end)?;
    Some(value.to_string())
}

fn html_text(value: &str) -> String {
    strip_html_tags(value)
        .replace("&amp;", "&")
        .replace("&#039;", "'")
        .replace("&quot;", "\"")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .trim()
        .to_string()
}

fn strip_html_tags(value: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for ch in value.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out
}

fn wiktionary_audio_file_candidates(word: &str) -> Vec<String> {
    let lower = word.to_lowercase();
    vec![
        format!("En-us-{lower}.ogg"),
        format!("En-us-{lower}.oga"),
        format!("En-uk-{lower}.ogg"),
        format!("En-uk-{lower}.oga"),
        format!("en-us-{lower}.ogg"),
        format!("en-uk-{lower}.ogg"),
    ]
}

fn download_to_file_checked(url: &str, path: &Path, robots: &mut RobotsCache) -> Result<()> {
    ensure_robots_allowed(url, robots)?;
    download_to_file(url, path)
}

fn download_to_file(url: &str, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let response = ureq::get(url)
        .header("User-Agent", USER_AGENT)
        .call()
        .with_context(|| format!("GET {url}"))?;
    let mut body = response.into_body();
    let mut reader = body.as_reader();
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    anyhow::ensure!(!bytes.is_empty(), "empty TTS response");
    fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

fn ensure_robots_allowed(url: &str, cache: &mut RobotsCache) -> Result<()> {
    let target = UrlParts::parse(url).with_context(|| format!("parsing URL {url}"))?;
    if cache.allows(&target)? {
        Ok(())
    } else {
        anyhow::bail!("robots.txt disallows {}{}", target.host, target.path)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UrlParts {
    scheme: String,
    host: String,
    path: String,
}

impl UrlParts {
    fn parse(url: &str) -> Result<Self> {
        let (scheme, rest) = url
            .split_once("://")
            .context("URL is missing scheme separator")?;
        let (host, path_and_query) = rest
            .split_once('/')
            .map(|(host, path)| (host, format!("/{path}")))
            .unwrap_or((rest, "/".to_string()));
        let path = path_and_query
            .split_once('?')
            .map(|(path, _)| path.to_string())
            .unwrap_or(path_and_query);
        anyhow::ensure!(
            scheme == "https" || scheme == "http",
            "unsupported URL scheme"
        );
        anyhow::ensure!(!host.is_empty(), "URL is missing host");
        Ok(Self {
            scheme: scheme.to_string(),
            host: host.to_string(),
            path,
        })
    }
}

#[derive(Debug, Default)]
struct RobotsCache {
    rules_by_host: HashMap<String, RobotsRules>,
    decisions: BTreeMap<String, bool>,
}

impl RobotsCache {
    fn allows(&mut self, target: &UrlParts) -> Result<bool> {
        if let Some(decision) = self
            .decisions
            .get(&format!("{}{}", target.host, target.path))
        {
            return Ok(*decision);
        }
        if !self.rules_by_host.contains_key(&target.host) {
            let robots_url = format!("{}://{}/robots.txt", target.scheme, target.host);
            println!("  robots.txt: checking {robots_url}");
            let rules = RobotsRules::fetch(&robots_url).unwrap_or_else(|error| {
                println!(
                    "  robots.txt: could not read {} ({}); network fetches for this host will be skipped",
                    robots_url, error
                );
                RobotsRules::deny_all(target.host.clone())
            });
            println!(
                "  robots.txt: host={} default_policy={}",
                target.host,
                if rules.default_allow {
                    "allow unless disallowed"
                } else {
                    "deny"
                }
            );
            self.rules_by_host.insert(target.host.clone(), rules);
        }
        let allowed = self
            .rules_by_host
            .get(&target.host)
            .expect("rules inserted")
            .allows(&target.path);
        self.decisions
            .insert(format!("{}{}", target.host, target.path), allowed);
        Ok(allowed)
    }

    fn print_summary(&self) {
        if self.decisions.is_empty() {
            println!("Robots policy summary: no network URL checks were needed");
            return;
        }
        println!("Robots policy summary:");
        for (target, allowed) in &self.decisions {
            println!(
                "  {} => {}",
                target,
                if *allowed { "allowed" } else { "disallowed" }
            );
        }
    }
}

#[derive(Debug, Clone)]
struct RobotsRules {
    default_allow: bool,
    records: Vec<RobotsDirective>,
}

#[derive(Debug, Clone)]
struct RobotsDirective {
    path: String,
    allow: bool,
}

impl RobotsRules {
    fn fetch(robots_url: &str) -> Result<Self> {
        let target = UrlParts::parse(robots_url)?;
        let response = ureq::get(robots_url)
            .header("User-Agent", USER_AGENT)
            .call()
            .with_context(|| format!("GET {robots_url}"))?;
        if !response.status().is_success() {
            return Ok(Self {
                default_allow: true,
                records: Vec::new(),
            });
        }
        let mut body = response.into_body();
        let mut reader = body.as_reader();
        let mut raw = String::new();
        reader.read_to_string(&mut raw)?;
        Ok(Self::parse(target.host, &raw))
    }

    fn deny_all(_host: String) -> Self {
        Self {
            default_allow: false,
            records: Vec::new(),
        }
    }

    fn parse(_host: String, raw: &str) -> Self {
        let mut records = Vec::new();
        let mut applies = false;
        let mut saw_agent_in_group = false;
        for line in raw.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                applies = false;
                saw_agent_in_group = false;
                continue;
            }
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let key = key.trim().to_ascii_lowercase();
            let value = value.trim();
            match key.as_str() {
                "user-agent" => {
                    if !saw_agent_in_group {
                        applies = false;
                    }
                    saw_agent_in_group = true;
                    let agent = value.to_ascii_lowercase();
                    if agent == "*" || USER_AGENT.to_ascii_lowercase().starts_with(&agent) {
                        applies = true;
                    }
                }
                "allow" if applies => {
                    if !value.is_empty() {
                        records.push(RobotsDirective {
                            path: value.to_string(),
                            allow: true,
                        });
                    }
                }
                "disallow" if applies => {
                    if !value.is_empty() {
                        records.push(RobotsDirective {
                            path: value.to_string(),
                            allow: false,
                        });
                    }
                }
                _ => {}
            }
        }
        Self {
            default_allow: true,
            records,
        }
    }

    fn allows(&self, path: &str) -> bool {
        if !self.default_allow {
            return false;
        }
        let mut best: Option<&RobotsDirective> = None;
        for record in &self.records {
            if robots_path_matches(&record.path, path)
                && best
                    .map(|best| record.path.len() > best.path.len())
                    .unwrap_or(true)
            {
                best = Some(record);
            }
        }
        best.map(|record| record.allow).unwrap_or(true)
    }
}

fn robots_path_matches(pattern: &str, path: &str) -> bool {
    if pattern == "/" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        path.starts_with(prefix)
    } else {
        path.starts_with(pattern)
    }
}

fn simple_url_component(value: &str) -> String {
    let mut out = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn path_to_uri(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn safe_id(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn phone_ipa(phone: &PhoneToken) -> &str {
    match &phone.phone {
        Spec::Known(id) => id
            .as_str()
            .strip_prefix("ipa.phone.")
            .unwrap_or(id.as_str()),
        _ => "",
    }
}

fn syllables_to_ipa(syllables: &[Syllable]) -> String {
    syllables
        .iter()
        .enumerate()
        .map(|(index, syllable)| {
            let mut text = String::new();
            let stress = match syllable.stress {
                Spec::Known(Stress::Primary) => Some('ˈ'),
                Spec::Known(Stress::Secondary) => Some('ˌ'),
                _ => None,
            };
            if index > 0 && stress.is_none() {
                text.push('.');
            }
            if let Some(stress) = stress {
                text.push(stress);
            }
            for phone in &syllable.phones {
                text.push_str(phone_ipa(phone));
            }
            text
        })
        .collect()
}

fn stress_pattern(syllables: &[Syllable]) -> String {
    syllables
        .iter()
        .map(|syllable| match syllable.stress {
            Spec::Known(Stress::Primary) => "1",
            Spec::Known(Stress::Secondary) => "2",
            Spec::Known(Stress::Unstressed) => "0",
            Spec::Known(Stress::Reduced) => "r",
            _ => "x",
        })
        .collect::<Vec<_>>()
        .join(".")
}

fn syllable_boundary_string(syllables: &[Syllable]) -> String {
    syllables
        .iter()
        .map(|syllable| syllable.phones.len().to_string())
        .collect::<Vec<_>>()
        .join("-")
}

fn phones_string(phones: &[PhoneToken]) -> String {
    phones.iter().map(phone_ipa).collect::<Vec<_>>().join("")
}

fn placeholder_acoustic_frames(word: &str, ipa: &str, rarity: f32) -> Vec<AcousticFrame> {
    let frame_count = ipa.chars().filter(|c| !c.is_whitespace()).count().max(1);
    let rarity_norm = if rarity.is_finite() {
        (rarity / 250_000.0).clamp(0.0, 1.0)
    } else {
        0.5
    };
    (0..frame_count)
        .map(|index| {
            let t0 = index as f64 * 0.05;
            let energy = -28.0 + ((word.len() + index) % 9) as f32;
            let f0 = 120.0 + (index % 7) as f32 * 8.0 + (1.0 - rarity_norm) * 20.0;
            AcousticFrame {
                span: TimeSpan {
                    start_s: t0,
                    end_s: t0 + 0.05,
                },
                f0_hz: Spec::Known(f0),
                energy_db: Spec::Known(energy),
                voicing_probability: Spec::Known(0.65),
                periodicity: Spec::Known(0.5),
                harmonicity: Spec::Known(0.5),
                formants: vec![
                    Formant {
                        index: 1,
                        hz: Spec::Known(450.0 + index as f32 * 3.0),
                        bandwidth_hz: Spec::Known(80.0),
                    },
                    Formant {
                        index: 2,
                        hz: Spec::Known(1500.0 + index as f32 * 5.0),
                        bandwidth_hz: Spec::Known(120.0),
                    },
                ],
                spectral_centroid_hz: Spec::Known(1800.0 + index as f32 * 11.0),
                spectral_tilt_db_per_octave: Spec::Known(-9.0),
                zero_crossing_rate: Spec::Known(0.08),
                vectors: vec![AcousticVector {
                    kind: "placeholder-summary".to_string(),
                    values: vec![f0 / 300.0, (energy + 60.0) / 60.0, rarity_norm],
                }],
            }
        })
        .collect()
}

fn acoustic_summary(frames: &[AcousticFrame]) -> String {
    if frames.is_empty() {
        return "acoustic:none".to_string();
    }
    let mut f0_sum = 0.0;
    let mut energy_sum = 0.0;
    let mut count = 0.0;
    for frame in frames {
        if let Spec::Known(f0) = frame.f0_hz {
            f0_sum += f0;
        }
        if let Spec::Known(energy) = frame.energy_db {
            energy_sum += energy;
        }
        count += 1.0;
    }
    format!(
        "acoustic:frames={} f0={:.1} energy={:.1}",
        frames.len(),
        f0_sum / count,
        energy_sum / count
    )
}

fn write_splits(
    out: &Path,
    config: &SpeechManifoldConfig,
    examples: &[SpeechManifoldExample],
) -> Result<()> {
    let mut files = BTreeMap::new();
    for split in ["train", "valid", "test"] {
        files.insert(
            split,
            std::io::BufWriter::new(fs::File::create(out.join(format!("{split}.jsonl")))?),
        );
    }
    let mut words = BTreeMap::from([
        ("train", Vec::new()),
        ("valid", Vec::new()),
        ("test", Vec::new()),
    ]);
    for example in examples {
        let fraction = hash_fraction(&format!("{}:{}", config.seed, example.spelling));
        let split = if fraction < config.train_frac {
            "train"
        } else if fraction < config.train_frac + config.valid_frac {
            "valid"
        } else {
            "test"
        };
        writeln!(
            files.get_mut(split).expect("known split"),
            "{}",
            serde_json::to_string(example)?
        )?;
        words
            .get_mut(split)
            .expect("known split")
            .push(example.spelling.clone());
    }
    for (split, mut split_words) in words {
        split_words.sort();
        split_words.dedup();
        fs::write(
            out.join(format!("{split}_words.txt")),
            split_words.join("\n"),
        )?;
    }
    Ok(())
}

fn hash_fraction(value: &str) -> f64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (hash as f64) / (u64::MAX as f64)
}

fn write_vocab(
    out: &Path,
    examples: &[SpeechManifoldExample],
    tasks: &[SpeechManifoldTask],
) -> Result<()> {
    let mut tokens = vec![
        "<PAD>".to_string(),
        "<UNK>".to_string(),
        "<BOS>".to_string(),
        "<EOS>".to_string(),
        "<SEP>".to_string(),
        "<G2P>".to_string(),
        "<P2G>".to_string(),
    ];
    for task in tasks {
        let token = task.token().to_string();
        if !tokens.contains(&token) {
            tokens.push(token);
        }
    }
    let mut chars = BTreeSet::new();
    for example in examples {
        for value in task_values(example) {
            for ch in value.chars() {
                chars.insert(ch.to_string());
            }
        }
    }
    tokens.extend(chars);
    let token_to_id = tokens
        .iter()
        .enumerate()
        .map(|(index, token)| (token.clone(), index as u32))
        .collect();
    let vocab = Vocab {
        tokens,
        token_to_id,
    };
    fs::write(
        out.join("vocab.json"),
        serde_json::to_string_pretty(&vocab)?,
    )?;
    Ok(())
}

fn task_values(example: &SpeechManifoldExample) -> Vec<String> {
    vec![
        example.spelling.clone(),
        example.broad_ipa.clone(),
        example.narrow_ipa.clone(),
        example.stress_pattern.clone(),
        syllable_boundary_string(&example.syllables),
        phones_string(&example.phones),
        acoustic_summary(&example.acoustic_frames),
    ]
}

fn write_sidecars(out: &Path, config: &SpeechManifoldConfig) -> Result<()> {
    fs::write(
        out.join("dataset_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    fs::write(
        out.join("modality_schema.json"),
        serde_json::to_string_pretty(&ModalitySchema {
            version: 1,
            tasks: config.tasks.iter().map(ToString::to_string).collect(),
            placeholder_acoustics_allowed: config.allow_placeholder_acoustics,
            output_type: "tongues_speech_manifold::SpeechManifoldExample".to_string(),
        })?,
    )?;
    fs::write(out.join("README.md"), dataset_readme(config))?;
    Ok(())
}

fn dataset_readme(config: &SpeechManifoldConfig) -> String {
    format!(
        r#"# speech-manifold dataset

Dataset id: `{dataset_id}`

This prepared dataset was generated by `tongues speech-manifold prepare`.
It contains OpenEPD-derived lexical records, locally derived speech modalities,
placeholder acoustic frames, and optional sampled audio files when a configured
backend was available and allowed by robots.txt.

## Files

| File | Purpose |
|---|---|
| `train.jsonl`, `valid.jsonl`, `test.jsonl` | Prepared `SpeechManifoldExample` rows. |
| `*_words.txt` | Split word lists. |
| `vocab.json` | Unified token vocabulary. |
| `dataset_config.json` | Exact prepare configuration. |
| `modality_schema.json` | Task/schema summary. |
| `audio/` | Optional generated or fetched audio samples. |

## Data Sources

| Source/backend | What is stored | License/terms note |
|---|---|---|
| OpenEPD (`open-english-pronouncing-dictionary`) | Primary spelling, IPA variants, rarity, and source labels. | OpenEPD is documented upstream as CC-BY-SA 4.0 because it includes WikiPron/Wiktionary-derived material. |
| WikiPron/Wiktionary-derived labels | Preserved in per-row provenance and used for optional Wiktionary reference URLs. | Share-alike provenance should be preserved when redistributing prepared data. |
| `speech` crate phonemicizer | Narrow phones, syllables, stress patterns, and placeholder acoustic frames. | Project-local generated annotations. |
| eSpeak NG | Optional local WAV samples with configured voices. | eSpeak NG is GPL-3-or-later; review eSpeak NG terms before redistributing generated audio. |
| Google Translate TTS URL support (`tts-urls`) | Optional MP3 samples, only when robots.txt allows the TTS path. | The URL helper crate is MIT; access/output from Google's service is governed by Google's terms and robots policy. This project is not affiliated with Google. |
| Wiktionary/Wikimedia audio | Optional OGG samples from public media URLs, only when robots.txt allows the requested paths. | Individual media files may have their own licenses; keep source URLs/provenance with redistributed audio. |
| Wikimedia Commons pronunciation audio | Optional real-human OGG pronunciation samples from allowed Commons file pages and direct media URLs. | Individual files carry their own licenses and attribution; source URLs, license labels, and attribution are preserved in provenance. |
| AnySpeak | Optional local MP3 samples via `python run_local.py` from an AnySpeak checkout. | AnySpeak is AGPL-3 and Qwen3-TTS-based; review AnySpeak and model/output terms before redistributing generated audio. |
| StyleTTS2 | Optional local WAV samples via `tongues speak --backend styletts2` when the local model/backend is available. | Review the selected model license before redistributing generated audio. |
| Piper | Optional local WAV samples via `tongues speak --backend piper` when the local model/backend is available. | Review the selected voice license before redistributing generated audio. |
| Mock | Optional deterministic local WAV samples for smoke tests and backend diversity. | Project-local generated test audio. |
| Dictionary.com | Reference URLs only. No Dictionary.com pages are fetched by prepare. | Respect Dictionary.com's terms if following or using those links manually. |

## External Audio Manifests

Additional permissioned audio can be imported with `external_audio_manifests`
in the prepare config. Each manifest is JSONL. Rows must include `word`,
`audio_uri`, `source`, `license`, and `attribution`. Rows are accepted only
when the word exists in OpenEPD-derived examples and either:

- `broad_ipa` normalizes to the same broad IPA as OpenEPD, or
- `pronunciation_assurance` is `single-word-pronunciation`,
  `source-pronunciation-entry`, or `manually-verified`.

Example row:

```json
{{"word":"cat","audio_uri":"/data/audio/cat-us.ogg","broad_ipa":"kæt","source":"wikimedia-commons","license":"CC BY-SA 4.0","attribution":"Example Speaker / Wikimedia Commons","source_url":"https://commons.wikimedia.org/wiki/File:En-us-cat.ogg"}}
```

Suitable manifest sources include Wikimedia Commons/Wiktionary pronunciation
audio with per-file license metadata, curated dictionary/classroom recordings
you have permission to use, public-domain or permissively licensed word-list
recordings, and locally generated TTS whose model/output terms permit your use.
Sentence corpora should only be imported after word-level segmentation and
pronunciation verification.

## Network And Robots Policy

`speech-manifold prepare` checks `robots.txt` before every network audio fetch.
If a host disallows a target path, the backend is skipped and the example falls
back to local eSpeak or mock/placeholder provenance. Failed or skipped network
attempts are reported in the prepare log.

## Redistribution Notes

The `tongues` source code is MIT licensed, but this prepared dataset may carry
additional obligations from OpenEPD/WikiPron/Wiktionary, eSpeak NG, or any
downloaded media. Treat `source_backend`, `provenance`, `reference_uris`, and
`audio_uri` as required attribution/audit metadata. Review upstream terms before
publishing generated JSONL or audio files.
"#,
        dataset_id = config.dataset_id
    )
}

pub fn read_examples(path: &Path) -> Result<Vec<SpeechManifoldExample>> {
    let f = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = std::io::BufReader::new(f);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line)?);
    }
    Ok(out)
}

pub fn make_example(
    example: &SpeechManifoldExample,
    task: SpeechManifoldTask,
    vocab: &Vocab,
    allow_placeholder_acoustics: bool,
) -> Option<Seq2SeqExample> {
    let (source, target) = match task {
        SpeechManifoldTask::SpellingToIpa => (example.spelling.clone(), example.broad_ipa.clone()),
        SpeechManifoldTask::IpaToSpelling => (example.broad_ipa.clone(), example.spelling.clone()),
        SpeechManifoldTask::IpaToPhones => (example.broad_ipa.clone(), example.narrow_ipa.clone()),
        SpeechManifoldTask::Stress => (example.spelling.clone(), example.stress_pattern.clone()),
        SpeechManifoldTask::Syllables => (
            example.broad_ipa.clone(),
            syllable_boundary_string(&example.syllables),
        ),
        SpeechManifoldTask::AcousticToIpa => {
            if example.acoustic_frames.is_empty()
                || (!allow_placeholder_acoustics && is_placeholder_acoustics(example))
            {
                return None;
            }
            (
                acoustic_summary(&example.acoustic_frames),
                example.broad_ipa.clone(),
            )
        }
        SpeechManifoldTask::IpaToAcoustic => {
            if example.acoustic_frames.is_empty()
                || (!allow_placeholder_acoustics && is_placeholder_acoustics(example))
            {
                return None;
            }
            (
                example.broad_ipa.clone(),
                acoustic_summary(&example.acoustic_frames),
            )
        }
    };
    let mut src_ids = vec![vocab.get_id(task.token())];
    src_ids.extend(vocab.encode_string(&source));

    let mut tgt_in_ids = vec![BOS_ID];
    tgt_in_ids.extend(vocab.encode_string(&target));

    let mut tgt_out_ids = vocab.encode_string(&target);
    tgt_out_ids.push(EOS_ID);

    Some(Seq2SeqExample {
        src_ids,
        tgt_in_ids,
        tgt_out_ids,
    })
}

pub fn sample_task<R: Rng>(
    example: &SpeechManifoldExample,
    tasks: &[SpeechManifoldTask],
    allow_placeholder_acoustics: bool,
    rng: &mut R,
) -> Option<SpeechManifoldTask> {
    let mut available = tasks
        .iter()
        .copied()
        .filter(|task| {
            make_example(
                example,
                *task,
                &minimal_vocab_for_sampling(),
                allow_placeholder_acoustics,
            )
            .is_some()
        })
        .collect::<Vec<_>>();
    available.shuffle(rng);
    available.first().copied()
}

fn minimal_vocab_for_sampling() -> Vocab {
    let tokens = vec![
        "<PAD>".to_string(),
        "<UNK>".to_string(),
        "<BOS>".to_string(),
        "<EOS>".to_string(),
        "<SEP>".to_string(),
        "<G2P>".to_string(),
        "<P2G>".to_string(),
    ];
    let token_to_id = tokens
        .iter()
        .enumerate()
        .map(|(index, token)| (token.clone(), index as u32))
        .collect();
    Vocab {
        tokens,
        token_to_id,
    }
}

fn is_placeholder_acoustics(example: &SpeechManifoldExample) -> bool {
    example.source_backend.contains("placeholder")
}

pub fn train<B: AutodiffBackend, R: Rng>(
    model_config: &ModelConfig,
    train_config: &SpeechManifoldTrainConfig,
    train_examples: &[SpeechManifoldExample],
    valid_examples: &[SpeechManifoldExample],
    vocab: &Vocab,
    model_path: &Path,
    device: &B::Device,
    rng: &mut R,
) -> Result<f32>
where
    <Seq2SeqModel<B> as Module<B>>::Record: Send,
{
    let out_dir = model_path.parent().unwrap_or(Path::new("."));
    let state_path = out_dir.join("train_state.json");
    let model_file = model_path.with_extension("bin");
    let mut start_epoch = 1usize;
    let mut best_val_loss = f32::INFINITY;

    let mut model = if state_path.exists() {
        let state: TrainState = serde_json::from_str(&fs::read_to_string(&state_path)?)?;
        start_epoch = state.current_epoch + 1;
        best_val_loss = state.best_val_loss;
        if model_file.exists() {
            model_config
                .init(device)
                .load_file(model_path, &make_recorder(), device)
                .context("loading speech-manifold model weights")?
        } else {
            model_config.init(device)
        }
    } else if model_file.exists() {
        model_config
            .init(device)
            .load_file(model_path, &make_recorder(), device)
            .context("loading speech-manifold model weights")?
    } else {
        model_config.init(device)
    };

    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(train_config.weight_decay)
        .init::<B, Seq2SeqModel<B>>();
    let mut patience_counter = 0usize;

    for epoch in start_epoch..=train_config.epochs {
        let train_loss = train_epoch(
            &mut model,
            &mut optimizer,
            train_examples,
            vocab,
            train_config,
            device,
            rng,
        );
        let eval_model: Seq2SeqModel<B::InnerBackend> = model.valid();
        let val_loss = evaluate_loss(
            &eval_model,
            valid_examples,
            vocab,
            train_config
                .tasks
                .first()
                .copied()
                .unwrap_or(SpeechManifoldTask::SpellingToIpa),
            train_config.allow_placeholder_acoustics,
            device,
        );
        println!("Epoch {epoch:3} | train_loss={train_loss:.4} val_loss={val_loss:.4}");

        fs::write(
            &state_path,
            serde_json::to_string_pretty(&TrainState {
                current_epoch: epoch,
                best_val_loss,
            })?,
        )?;

        if val_loss < best_val_loss - 1e-5 {
            best_val_loss = val_loss;
            patience_counter = 0;
            eval_model
                .save_file(model_path, &make_recorder())
                .context("saving speech-manifold model weights")?;
            fs::write(
                &state_path,
                serde_json::to_string_pretty(&TrainState {
                    current_epoch: epoch,
                    best_val_loss,
                })?,
            )?;
        } else {
            patience_counter += 1;
            if patience_counter >= train_config.early_stopping_patience {
                break;
            }
        }
    }
    Ok(best_val_loss)
}

fn train_epoch<B: AutodiffBackend, R: Rng>(
    model: &mut Seq2SeqModel<B>,
    optimizer: &mut impl Optimizer<Seq2SeqModel<B>, B>,
    examples: &[SpeechManifoldExample],
    vocab: &Vocab,
    config: &SpeechManifoldTrainConfig,
    device: &B::Device,
    rng: &mut R,
) -> f32 {
    let mut indices: Vec<usize> = (0..examples.len()).collect();
    indices.shuffle(rng);
    let mut total_loss = 0.0;
    let mut batches = 0usize;

    for chunk in indices.chunks(config.batch_size) {
        let seq_examples = chunk
            .iter()
            .filter_map(|&index| {
                let example = &examples[index];
                let task = sample_task(
                    example,
                    &config.tasks,
                    config.allow_placeholder_acoustics,
                    rng,
                )?;
                make_example(example, task, vocab, config.allow_placeholder_acoustics)
            })
            .collect::<Vec<_>>();
        if seq_examples.is_empty() {
            continue;
        }
        let max_src = seq_examples
            .iter()
            .map(|ex| ex.src_ids.len())
            .max()
            .unwrap_or(1);
        let max_tgt = seq_examples
            .iter()
            .map(|ex| ex.tgt_in_ids.len())
            .max()
            .unwrap_or(1);
        let batch = collate_batch(&seq_examples, max_src, max_tgt);
        let batch = tensor_seq2seq_batch(
            batch.src_ids,
            batch.tgt_in_ids,
            batch.tgt_out_ids,
            batch.src_pad_mask,
            batch.tgt_pad_mask,
            device,
        );
        let logits = model.forward(
            batch.src_ids,
            batch.tgt_in_ids,
            batch.src_pad_mask,
            batch.tgt_pad_mask,
        );
        let loss = seq2seq_cross_entropy_loss(logits, batch.tgt_out_ids, PAD_ID as usize);
        let grads = GradientsParams::from_grads(loss.backward(), model);
        *model = optimizer.step(config.learning_rate, model.clone(), grads);
        total_loss += loss.into_scalar().elem::<f32>();
        batches += 1;
    }
    if batches == 0 {
        0.0
    } else {
        total_loss / batches as f32
    }
}

fn evaluate_loss<B: Backend>(
    model: &Seq2SeqModel<B>,
    examples: &[SpeechManifoldExample],
    vocab: &Vocab,
    task: SpeechManifoldTask,
    allow_placeholder_acoustics: bool,
    device: &B::Device,
) -> f32 {
    evaluate(
        model,
        examples,
        vocab,
        task,
        allow_placeholder_acoustics,
        device,
    )
    .loss
}

pub fn evaluate<B: Backend>(
    model: &Seq2SeqModel<B>,
    examples: &[SpeechManifoldExample],
    vocab: &Vocab,
    task: SpeechManifoldTask,
    allow_placeholder_acoustics: bool,
    device: &B::Device,
) -> EvalReport {
    let seq_examples = examples
        .iter()
        .take(1000)
        .filter_map(|example| make_example(example, task, vocab, allow_placeholder_acoustics))
        .collect::<Vec<_>>();
    let placeholder_acoustic_metrics = matches!(
        task,
        SpeechManifoldTask::AcousticToIpa | SpeechManifoldTask::IpaToAcoustic
    ) && examples.iter().any(is_placeholder_acoustics);
    if seq_examples.is_empty() {
        return EvalReport {
            examples: 0,
            task,
            loss: 0.0,
            exact_match_accuracy: 0.0,
            token_accuracy: 0.0,
            placeholder_acoustic_metrics,
        };
    }

    let mut total_loss = 0.0;
    let mut batches = 0usize;
    let mut exact = 0usize;
    let mut total_tokens = 0usize;
    let mut matched_tokens = 0usize;

    for chunk in seq_examples.chunks(64) {
        let max_src = chunk.iter().map(|ex| ex.src_ids.len()).max().unwrap_or(1);
        let max_tgt = chunk
            .iter()
            .map(|ex| ex.tgt_in_ids.len())
            .max()
            .unwrap_or(1);
        let batch = collate_batch(chunk, max_src, max_tgt);
        let b = batch.size;
        let tensor_batch = tensor_seq2seq_batch(
            batch.src_ids,
            batch.tgt_in_ids,
            batch.tgt_out_ids,
            batch.src_pad_mask,
            batch.tgt_pad_mask,
            device,
        );
        let logits = model.forward(
            tensor_batch.src_ids,
            tensor_batch.tgt_in_ids,
            tensor_batch.src_pad_mask,
            tensor_batch.tgt_pad_mask,
        );
        let loss =
            seq2seq_cross_entropy_loss(logits.clone(), tensor_batch.tgt_out_ids, PAD_ID as usize);
        total_loss += loss.into_scalar().elem::<f32>();
        batches += 1;

        let [_, tgt_len, vocab_size] = logits.dims();
        for i in 0..b {
            let mut matched = true;
            for j in 0..tgt_len {
                let tgt_id = chunk[i].tgt_out_ids.get(j).copied().unwrap_or(PAD_ID);
                if tgt_id == PAD_ID {
                    break;
                }
                total_tokens += 1;
                let pos_logits = logits
                    .clone()
                    .slice([i..i + 1, j..j + 1, 0..vocab_size])
                    .reshape([vocab_size]);
                let pos_logits_vec: Vec<f32> = pos_logits.into_data().to_vec().unwrap();
                let pred = pos_logits_vec
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(PAD_ID);
                if pred == tgt_id {
                    matched_tokens += 1;
                } else {
                    matched = false;
                }
            }
            if matched {
                exact += 1;
            }
        }
    }

    EvalReport {
        examples: seq_examples.len(),
        task,
        loss: if batches == 0 {
            0.0
        } else {
            total_loss / batches as f32
        },
        exact_match_accuracy: exact as f32 / seq_examples.len() as f32,
        token_accuracy: if total_tokens == 0 {
            0.0
        } else {
            matched_tokens as f32 / total_tokens as f32
        },
        placeholder_acoustic_metrics,
    }
}

pub fn predict<B: Backend>(
    model: &Seq2SeqModel<B>,
    input: &str,
    task: SpeechManifoldTask,
    vocab: &Vocab,
    device: &B::Device,
) -> String {
    let mut src_ids = vec![vocab.get_id(task.token())];
    src_ids.extend(vocab.encode_string(input));
    let src_tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            src_ids.iter().map(|&id| id as i32).collect::<Vec<_>>(),
            [1, src_ids.len()],
        ),
        device,
    );
    let pred_ids = model.generate(src_tensor, 128);
    vocab.decode_ids(&pred_ids)
}

pub fn save_artifact_files(
    out: &Path,
    data: &Path,
    model_config: &ModelConfig,
    train_config: &SpeechManifoldTrainConfig,
) -> Result<()> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    fs::write(
        out.join("model_config.json"),
        serde_json::to_string_pretty(model_config)?,
    )?;
    fs::write(
        out.join("train_config.json"),
        serde_json::to_string_pretty(train_config)?,
    )?;
    let vocab_src = data.join("vocab.json");
    if vocab_src.exists() {
        fs::copy(vocab_src, out.join("vocab.json")).context("copying vocab.json")?;
    }
    write_manifest(
        out,
        &ModelArtifactManifest::new(FAMILY, ARCHITECTURE, data_id_from_path(data)),
    )?;
    Ok(())
}

pub fn load_model<B: Backend>(
    model_config: &ModelConfig,
    model_dir: &Path,
    device: &B::Device,
) -> Result<Seq2SeqModel<B>> {
    model_config
        .init::<B>(device)
        .load_file(&model_dir.join("model"), &make_recorder(), device)
        .with_context(|| format!("loading model from {}", model_dir.display()))
}

pub fn probe(
    split: &str,
    examples: &[SpeechManifoldExample],
    tasks: &[SpeechManifoldTask],
) -> ProbeReport {
    let mut source_backends = BTreeMap::new();
    let mut placeholder_acoustic_examples = 0usize;
    for example in examples {
        *source_backends
            .entry(example.source_backend.clone())
            .or_insert(0) += 1;
        if is_placeholder_acoustics(example) {
            placeholder_acoustic_examples += 1;
        }
    }
    ProbeReport {
        split: split.to_string(),
        examples: examples.len(),
        placeholder_acoustic_examples,
        source_backends,
        tasks: tasks.iter().map(ToString::to_string).collect(),
    }
}

fn data_id_from_path(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(DEFAULT_DATASET_ID)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_acoustic_example_roundtrips_with_provenance() {
        let example = example_from_word(
            "cat",
            "kæt",
            50_000.0,
            &[],
            &SpeechManifoldConfig::default(),
        )
        .unwrap();
        let json = serde_json::to_string(&example).unwrap();
        let parsed: SpeechManifoldExample = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.spelling, "cat");
        assert!(parsed.source_backend.contains("placeholder"));
        assert!(!parsed.acoustic_frames.is_empty());
    }

    #[test]
    fn task_sampler_skips_placeholder_acoustics_when_disabled() {
        let example =
            example_from_word("cat", "kæt", 100.0, &[], &SpeechManifoldConfig::default()).unwrap();
        let mut rng = rand::thread_rng();
        let task = sample_task(
            &example,
            &[SpeechManifoldTask::AcousticToIpa],
            false,
            &mut rng,
        );

        assert_eq!(task, None);
    }

    #[test]
    fn seq2seq_example_uses_task_prefix() {
        let example =
            example_from_word("cat", "kæt", 100.0, &[], &SpeechManifoldConfig::default()).unwrap();
        let mut tokens = vec![
            "<PAD>".to_string(),
            "<UNK>".to_string(),
            "<BOS>".to_string(),
            "<EOS>".to_string(),
            "<SEP>".to_string(),
            "<G2P>".to_string(),
            "<P2G>".to_string(),
            SpeechManifoldTask::SpellingToIpa.token().to_string(),
        ];
        tokens.extend(["c", "a", "t", "k", "æ"].into_iter().map(str::to_string));
        let token_to_id = tokens
            .iter()
            .enumerate()
            .map(|(index, token)| (token.clone(), index as u32))
            .collect();
        let vocab = Vocab {
            tokens,
            token_to_id,
        };

        let seq = make_example(&example, SpeechManifoldTask::SpellingToIpa, &vocab, true).unwrap();

        assert_eq!(
            seq.src_ids[0],
            vocab.get_id(SpeechManifoldTask::SpellingToIpa.token())
        );
    }

    #[test]
    fn generated_readme_records_sources_and_license_notes() {
        let readme = dataset_readme(&SpeechManifoldConfig::default());

        assert!(readme.contains("OpenEPD"));
        assert!(readme.contains("CC-BY-SA"));
        assert!(readme.contains("eSpeak NG"));
        assert!(readme.contains("Google Translate"));
        assert!(readme.contains("robots.txt"));
        assert!(readme.contains("Wikimedia Commons pronunciation audio"));
        assert!(readme.contains("AnySpeak"));
        assert!(readme.contains("Dictionary.com"));
        assert!(readme.contains("External Audio Manifests"));
    }

    #[test]
    fn commons_audio_parser_extracts_media_license_and_attribution() {
        let html = r#"
            <audio><source src="https://upload.wikimedia.org/wikipedia/commons/4/46/En-us-cat.ogg" type="audio/ogg"></audio>
            <span class="licensetpl_short" style="display:none;">CC BY-SA 3.0</span>
            <div class='wbmi-snak-value--value'>Dvortygirl</div>
        "#;

        let source = parse_commons_audio_source(
            html,
            "En-us-cat.ogg",
            "https://commons.wikimedia.org/wiki/File:En-us-cat.ogg",
        )
        .unwrap();

        assert_eq!(
            source.media_url,
            "https://upload.wikimedia.org/wikipedia/commons/4/46/En-us-cat.ogg"
        );
        assert_eq!(source.license, "CC BY-SA 3.0");
        assert_eq!(source.attribution, "Dvortygirl");
    }

    #[test]
    fn external_audio_manifest_requires_rights_and_pronunciation_assurance() {
        let mut examples = vec![example_from_word(
            "cat",
            "kæt",
            100.0,
            &["wikipron".to_string()],
            &SpeechManifoldConfig::default(),
        )
        .unwrap()];
        let dir = Path::new("target/test-speech-manifold");
        fs::create_dir_all(dir).unwrap();
        let manifest = dir.join("external-audio.jsonl");
        fs::write(
            &manifest,
            [
                r#"{"word":"cat","audio_uri":"/audio/cat.ogg","broad_ipa":"kæt","source":"wikimedia-commons","license":"CC BY-SA 4.0","attribution":"Example Speaker","source_url":"https://commons.wikimedia.org/wiki/File:En-us-cat.ogg"}"#,
                r#"{"word":"cat","audio_uri":"/audio/cat-bad.ogg","source":"unknown","license":"CC0","attribution":"Example Speaker"}"#,
                r#"{"word":"dog","audio_uri":"/audio/dog.ogg","broad_ipa":"dɔɡ","source":"wikimedia-commons","license":"CC BY-SA 4.0","attribution":"Example Speaker"}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        let mut config = SpeechManifoldConfig::default();
        config.external_audio_manifests = vec![manifest.display().to_string()];

        apply_external_audio_manifests(&mut examples, &config).unwrap();

        assert_eq!(examples.len(), 2);
        let imported = examples
            .iter()
            .find(|example| example.id.contains("external-audio"))
            .unwrap();
        assert_eq!(imported.audio_uri.as_deref(), Some("/audio/cat.ogg"));
        assert!(imported.source_backend.contains("openepd-ipa-match"));
        assert!(imported.provenance.contains("CC BY-SA 4.0"));
    }
}
