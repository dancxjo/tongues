//! Shared speech-manifold model family.
//!
//! V1 uses a shared seq2seq Transformer over multimodal token views. Audio
//! synthesis provenance is captured during prepare, while acoustic-frame inputs
//! are represented by deterministic vector summaries when real feature
//! extraction is not available.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{BufRead, Write};
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
    pub max_espeak_examples: usize,
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
            max_espeak_examples: 128,
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
    ["espeak-ng", "styletts2", "piper", "mock"]
        .into_iter()
        .map(str::to_string)
        .collect()
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
    let audio_dir = out.join("audio").join("espeak-ng");
    fs::create_dir_all(&audio_dir).with_context(|| format!("creating {}", audio_dir.display()))?;

    let examples = openepd_examples(config, &audio_dir)?;
    write_splits(out, config, &examples)?;
    write_vocab(out, &examples, &config.tasks)?;
    write_sidecars(out, config)?;
    Ok(())
}

fn openepd_examples(
    config: &SpeechManifoldConfig,
    audio_dir: &Path,
) -> Result<Vec<SpeechManifoldExample>> {
    let raw: BTreeMap<String, OpenEpdEntry> =
        serde_json::from_str(open_english_pronouncing_dictionary::CORPUS_JSON)
            .context("parsing embedded OpenEPD JSON")?;
    let espeak_available = config
        .synthesis_backends
        .iter()
        .any(|backend| backend == "espeak-ng")
        && Command::new("espeak-ng").arg("--version").output().is_ok();

    let mut examples = Vec::new();
    for (base_word, entry) in raw {
        if config.max_examples.is_some_and(|max| examples.len() >= max) {
            break;
        }
        if !is_prepare_word(&base_word) {
            continue;
        }
        let Some(raw_ipa) = preferred_openepd_ipa(&entry.ipa) else {
            continue;
        };
        let broad_ipa = match normalize_openepd_ipa(raw_ipa) {
            Ok(ipa) => ipa,
            Err(_) => continue,
        };
        let Some(mut example) = example_from_word(&base_word, &broad_ipa, entry.rarity) else {
            continue;
        };

        if espeak_available && examples.len() < config.max_espeak_examples {
            let wav_path = audio_dir.join(format!("{}.wav", safe_id(&base_word)));
            if synthesize_espeak(&base_word, &wav_path).is_ok() {
                example.audio_uri = Some(path_to_uri(&wav_path));
                example.sample_rate_hz = Some(22_050);
                example.source_backend = "espeak-ng+placeholder-acoustics".to_string();
            }
        }

        examples.push(example);
    }
    Ok(examples)
}

fn example_from_word(
    base_word: &str,
    broad_ipa: &str,
    rarity: f32,
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
        provenance: "OpenEPD lexical source + speech phonemicizer + deterministic placeholder acoustic frames".to_string(),
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

fn synthesize_espeak(text: &str, wav_path: &Path) -> Result<()> {
    let status = Command::new("espeak-ng")
        .args(["-w", wav_path.to_str().context("non-UTF-8 WAV path")?, text])
        .status()
        .context("running espeak-ng")?;
    anyhow::ensure!(status.success(), "espeak-ng failed");
    Ok(())
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
    fs::write(
        out.join("README.md"),
        "Speech-manifold dataset with OpenEPD lexical records, derived speech modalities, and synthetic/placeholder acoustic provenance.\n",
    )?;
    Ok(())
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
        let example = example_from_word("cat", "kæt", 50_000.0).unwrap();
        let json = serde_json::to_string(&example).unwrap();
        let parsed: SpeechManifoldExample = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.spelling, "cat");
        assert!(parsed.source_backend.contains("placeholder"));
        assert!(!parsed.acoustic_frames.is_empty());
    }

    #[test]
    fn task_sampler_skips_placeholder_acoustics_when_disabled() {
        let example = example_from_word("cat", "kæt", 100.0).unwrap();
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
        let example = example_from_word("cat", "kæt", 100.0).unwrap();
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
}
