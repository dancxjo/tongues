use std::collections::BTreeSet;

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use speaking::UtterancePlan;

use crate::{
    AudioChunk, AudioSink, SpeechModelCapabilities, SpeechSynthesisEngine, SpeechSynthesisRequest,
};

/// The role a native speech component can play in a synthesis graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeSpeechComponentKind {
    EndToEnd,
    VoiceConversion,
    Acoustic,
    Vocoder,
    Voice,
    /// Native training hooks that are not part of any synthesis path.
    Trainer,
    Test,
}

/// How far an implementation is wired into Tongues' shared speech runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeSpeechComponentReadiness {
    Runtime,
    ImportOnly,
    Experimental,
}

/// Stable, artifact-independent inventory for native speech machinery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeSpeechComponent {
    pub id: &'static str,
    pub display_name: &'static str,
    pub architecture: &'static str,
    pub kind: NativeSpeechComponentKind,
    pub readiness: NativeSpeechComponentReadiness,
    pub explanation: &'static str,
}

/// Every speech architecture implemented or imported by `tongues-tts`.
///
/// Artifact, license, compatibility, and load state are supplied by the model
/// catalog and synthesis registry at discovery time.
pub const NATIVE_SPEECH_COMPONENTS: &[NativeSpeechComponent] = &[
    NativeSpeechComponent {
        id: "speedy-speech",
        display_name: "SpeedySpeech",
        architecture: "speedy-speech",
        kind: NativeSpeechComponentKind::Acoustic,
        readiness: NativeSpeechComponentReadiness::Runtime,
        explanation: "Native acoustic inference; requires a compatible neural vocoder.",
    },
    NativeSpeechComponent {
        id: "fastpitch",
        display_name: "FastPitch",
        architecture: "fastpitch",
        kind: NativeSpeechComponentKind::Acoustic,
        readiness: NativeSpeechComponentReadiness::Runtime,
        explanation: "Native pitch-conditioned acoustic inference; requires a compatible neural vocoder.",
    },
    NativeSpeechComponent {
        id: "glow-tts",
        display_name: "Glow-TTS",
        architecture: "glow-tts",
        kind: NativeSpeechComponentKind::Acoustic,
        readiness: NativeSpeechComponentReadiness::Runtime,
        explanation: "Native acoustic inference composed through a named, package-pinned feature standardizer and MultiBand-MelGAN.",
    },
    NativeSpeechComponent {
        id: "sc-glowtts",
        display_name: "SC-GlowTTS",
        architecture: "sc-glowtts",
        kind: NativeSpeechComponentKind::Acoustic,
        readiness: NativeSpeechComponentReadiness::ImportOnly,
        explanation: "Speaker-conditioned deterministic-duration import/runtime support for user-supplied packages with affirmative artifact terms; no catalog artifact is claimed.",
    },
    NativeSpeechComponent {
        id: "tacotron2",
        display_name: "Tacotron 2",
        architecture: "tacotron2",
        kind: NativeSpeechComponentKind::Acoustic,
        readiness: NativeSpeechComponentReadiness::ImportOnly,
        explanation: "Native acoustic inference and checkpoint import support; no catalog artifact or complete synthesis path is registered.",
    },
    NativeSpeechComponent {
        id: "tacotron2-ddc",
        display_name: "Tacotron 2 DDC",
        architecture: "tacotron2-ddc",
        kind: NativeSpeechComponentKind::Acoustic,
        readiness: NativeSpeechComponentReadiness::ImportOnly,
        explanation: "Double-decoder-consistency import variant; no complete synthesis path is registered.",
    },
    NativeSpeechComponent {
        id: "capacitron",
        display_name: "Capacitron Tacotron 2",
        architecture: "capacitron",
        kind: NativeSpeechComponentKind::Acoustic,
        readiness: NativeSpeechComponentReadiness::ImportOnly,
        explanation: "Capacitron conditioning is represented by the importer, but no runnable catalog artifact is registered.",
    },
    NativeSpeechComponent {
        id: "capacitron-ddc",
        display_name: "Capacitron DDC",
        architecture: "capacitron-ddc",
        kind: NativeSpeechComponentKind::Acoustic,
        readiness: NativeSpeechComponentReadiness::ImportOnly,
        explanation: "Capacitron plus double-decoder-consistency import variant; no runnable catalog artifact is registered.",
    },
    NativeSpeechComponent {
        id: "fastspeech",
        display_name: "FastSpeech",
        architecture: "fastspeech",
        kind: NativeSpeechComponentKind::Acoustic,
        readiness: NativeSpeechComponentReadiness::ImportOnly,
        explanation: "Native variance-model inference exists, but no verified artifact and vocoder path are registered.",
    },
    NativeSpeechComponent {
        id: "fastspeech2",
        display_name: "FastSpeech 2",
        architecture: "fastspeech2",
        kind: NativeSpeechComponentKind::Acoustic,
        readiness: NativeSpeechComponentReadiness::ImportOnly,
        explanation: "Native pitch and energy conditioned inference exists, but no verified artifact and vocoder path are registered.",
    },
    NativeSpeechComponent {
        id: "delightfultts",
        display_name: "DelightfulTTS",
        architecture: "delightfultts",
        kind: NativeSpeechComponentKind::Acoustic,
        readiness: NativeSpeechComponentReadiness::ImportOnly,
        explanation: "Native conformer and variance inference exists, but no verified artifact and vocoder path are registered.",
    },
    NativeSpeechComponent {
        id: "vits",
        display_name: "VITS",
        architecture: "vits",
        kind: NativeSpeechComponentKind::EndToEnd,
        readiness: NativeSpeechComponentReadiness::Runtime,
        explanation: "Native end-to-end waveform synthesis with learned speaker and language embeddings when declared by the checkpoint.",
    },
    NativeSpeechComponent {
        id: "pocket-tts-causal",
        display_name: "Pocket TTS",
        architecture: "pocket-tts",
        kind: NativeSpeechComponentKind::EndToEnd,
        readiness: NativeSpeechComponentReadiness::Experimental,
        explanation: "Causal append-only streaming renderer under evaluation for native integration; runtime registration is intentionally deferred until artifact pinning, parity, and ledger integration are complete.",
    },
    NativeSpeechComponent {
        id: "speaker-encoder",
        display_name: "Coqui ResNet Speaker Encoder",
        architecture: "speaker-encoder",
        kind: NativeSpeechComponentKind::Voice,
        readiness: NativeSpeechComponentReadiness::Runtime,
        explanation: "Native reference-audio d-vector inference, enrollment, caching, and verification.",
    },
    NativeSpeechComponent {
        id: "yourtts",
        display_name: "YourTTS",
        architecture: "vits",
        kind: NativeSpeechComponentKind::EndToEnd,
        readiness: NativeSpeechComponentReadiness::Runtime,
        explanation: "Native multilingual voice cloning with grapheme VITS, language embeddings, d-vectors, and the Coqui ResNet speaker encoder.",
    },
    NativeSpeechComponent {
        id: "freevc",
        display_name: "FreeVC24",
        architecture: "freevc",
        kind: NativeSpeechComponentKind::VoiceConversion,
        readiness: NativeSpeechComponentReadiness::Runtime,
        explanation: "Native source-waveform voice conversion using WavLM content features and target-speaker reference audio.",
    },
    NativeSpeechComponent {
        id: "hifigan",
        display_name: "HiFi-GAN",
        architecture: "hifigan",
        kind: NativeSpeechComponentKind::Vocoder,
        readiness: NativeSpeechComponentReadiness::Runtime,
        explanation: "Native neural vocoder used by registered composable synthesis paths.",
    },
    NativeSpeechComponent {
        id: "melgan",
        display_name: "MelGAN",
        architecture: "melgan",
        kind: NativeSpeechComponentKind::Vocoder,
        readiness: NativeSpeechComponentReadiness::ImportOnly,
        explanation: "Native vocoder inference exists, but no verified catalog artifact is registered.",
    },
    NativeSpeechComponent {
        id: "multiband-melgan",
        display_name: "MultiBand-MelGAN",
        architecture: "multiband-melgan",
        kind: NativeSpeechComponentKind::Vocoder,
        readiness: NativeSpeechComponentReadiness::Experimental,
        explanation: "Native PQMF vocoder inference exists; its artifact is not registered in a compatible complete synthesis path.",
    },
    NativeSpeechComponent {
        id: "align-tts",
        display_name: "Align-TTS",
        architecture: "align-tts",
        kind: NativeSpeechComponentKind::Acoustic,
        readiness: NativeSpeechComponentReadiness::ImportOnly,
        explanation: "Native encoder, duration predictor, expansion, and mel decoding exist; no verified catalog artifact or complete synthesis path is registered.",
    },
    NativeSpeechComponent {
        id: "xtts-v2",
        display_name: "XTTS v2",
        architecture: "xtts-v2",
        kind: NativeSpeechComponentKind::EndToEnd,
        readiness: NativeSpeechComponentReadiness::Experimental,
        explanation: "Native Burn GPT-conditioned codec TTS inference exists; weights require separate Coqui license terms and no catalog artifact is claimed.",
    },
    NativeSpeechComponent {
        id: "hifigan-trainer",
        display_name: "HiFi-GAN Trainer",
        architecture: "hifigan-trainer",
        kind: NativeSpeechComponentKind::Trainer,
        readiness: NativeSpeechComponentReadiness::ImportOnly,
        explanation: "Native HiFi-GAN generator/MPD/MSD training hooks; not part of any synthesis path.",
    },
    NativeSpeechComponent {
        id: "melgan-trainer",
        display_name: "MelGAN Trainer",
        architecture: "melgan-trainer",
        kind: NativeSpeechComponentKind::Trainer,
        readiness: NativeSpeechComponentReadiness::ImportOnly,
        explanation: "Native MelGAN generator/discriminator training hooks; not part of any synthesis path.",
    },
    NativeSpeechComponent {
        id: "multiband-melgan-trainer",
        display_name: "MultiBand-MelGAN Trainer",
        architecture: "multiband-melgan-trainer",
        kind: NativeSpeechComponentKind::Trainer,
        readiness: NativeSpeechComponentReadiness::ImportOnly,
        explanation: "Native MultiBand-MelGAN generator/discriminator training hooks with PQMF bank; not part of any synthesis path.",
    },
    NativeSpeechComponent {
        id: "vits-trainer",
        display_name: "VITS Trainer",
        architecture: "vits-trainer",
        kind: NativeSpeechComponentKind::Trainer,
        readiness: NativeSpeechComponentReadiness::ImportOnly,
        explanation: "Native VITS end-to-end adversarial training hooks including posterior encoder and discriminators; not part of any synthesis path.",
    },
];

pub fn native_speech_components() -> &'static [NativeSpeechComponent] {
    NATIVE_SPEECH_COMPONENTS
}

/// Classification table for every public `burn_*` implementation module in
/// `tongues-tts`, plus the non-`burn_*` native modules that provide concrete
/// speech functionality (`freevc`, `speaker_encoder`).
///
/// Each row is `(module_name, component_ids)` where:
/// - `component_ids` is a slice of `NativeSpeechComponent::id` values from
///   [`NATIVE_SPEECH_COMPONENTS`] that the module provides; and
/// - an **empty slice** marks the module as internal infrastructure that is
///   intentionally absent from the public inventory.
///
/// Tests verify bidirectional consistency between this table and
/// [`NATIVE_SPEECH_COMPONENTS`]: every classified component ID must exist in
/// the inventory, and every inventory entry must be claimed by at least one
/// module row or appear in [`PLANNED_SPEECH_COMPONENT_IDS`].
///
/// **Developer contract**: when adding a new `burn_something.rs` module, add a
/// row here.  If the slice is non-empty the components must also be added to
/// [`NATIVE_SPEECH_COMPONENTS`]; omitting the row will cause
/// [`every_inventory_entry_has_a_module_classification`] to fail.
pub const IMPLEMENTATION_MODULE_CLASSIFICATION: &[(&str, &[&str])] = &[
    // ── shared contracts / adapters / pipeline wrappers (internal) ──────────
    ("burn_acoustic",            &[]),
    ("burn_acoustic_training",   &[]),
    ("burn_align_tts_pipeline",  &[]),
    ("burn_fast_pitch_acoustic", &[]),
    ("burn_fast_pitch_pipeline", &[]),
    ("burn_glow_tts_acoustic",   &[]),
    ("burn_glow_tts_pipeline",   &[]),
    ("burn_pipeline",            &[]),
    ("burn_tacotron_acoustic",   &[]),
    ("burn_variance_acoustic",   &[]),
    ("burn_vits_decoder",        &[]),
    ("burn_vits_duration",       &[]),
    ("burn_vits_flow",           &[]),
    ("burn_vits_text",           &[]),
    ("burn_vocoder",             &[]),
    ("burn_vocoder_discriminators", &[]),
    ("burn_vocoder_losses",      &[]),
    ("burn_vocoder_training",    &[]),
    // ── concrete inference models ────────────────────────────────────────────
    ("burn_align_tts",     &["align-tts"]),
    ("burn_delightful_tts",&["delightfultts"]),
    ("burn_fast_pitch",    &["fastpitch"]),
    ("burn_fastspeech",    &["fastspeech", "fastspeech2"]),
    ("burn_glow_tts",      &["glow-tts", "sc-glowtts"]),
    ("burn_hifigan",       &["hifigan"]),
    ("burn_melgan",        &["melgan", "multiband-melgan"]),
    ("burn_speedy_speech", &["speedy-speech"]),
    ("burn_tacotron",      &["tacotron2", "tacotron2-ddc", "capacitron", "capacitron-ddc"]),
    ("burn_vits",          &["vits", "yourtts"]),
    ("burn_xtts",          &["xtts-v2"]),
    // ── non-burn_* native modules ────────────────────────────────────────────
    ("freevc",          &["freevc"]),
    ("speaker_encoder", &["speaker-encoder"]),
    // ── training hooks ───────────────────────────────────────────────────────
    ("burn_hifigan_trainer", &["hifigan-trainer"]),
    ("burn_melgan_trainer",  &["melgan-trainer", "multiband-melgan-trainer"]),
    ("burn_vits_training",   &["vits-trainer"]),
];

/// Component IDs in [`NATIVE_SPEECH_COMPONENTS`] that correspond to planned or
/// externally-backed implementations without a native `burn_*` source module.
/// Entries here are exempt from the module-classification completeness check.
pub const PLANNED_SPEECH_COMPONENT_IDS: &[&str] = &[
    // Causal append-only streaming renderer under evaluation; no source module yet.
    "pocket-tts-causal",
];

/// Runtime used for neural inference.
///
/// Burn is the native runtime. ONNX remains explicit as a compatibility path
/// while existing voice artifacts are migrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceRuntime {
    Burn,
    OnnxCompatibility,
    PassThrough,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LinguisticIntent {
    Text,
    Morphemes,
    Phonemes,
    Phones,
    Syllables,
    Boundaries,
    Pitch,
    Energy,
    SpeakingRate,
    ProsodicBreaks,
    ProsodicLabels,
    TargetAcoustics,
    SpeakerIdentity,
    SpeakerReference,
    Style,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinguisticInputKind {
    Phones,
    Phonemes,
    Graphemes,
    TextBpe,
}

/// Declares the terminal projection performed by one model adapter.
///
/// Vocabulary IDs and token conventions are model-local. The fingerprint makes
/// accidentally pairing a projector with a different checkpoint detectable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInputContract {
    pub kind: LinguisticInputKind,
    pub vocabulary_fingerprint: String,
    /// Supported canonical varieties, registered aliases, variety prefixes, or `*`.
    pub supported_varieties: Vec<String>,
    pub consumes: BTreeSet<LinguisticIntent>,
}

impl ModelInputContract {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.vocabulary_fingerprint.trim().is_empty(),
            "model input vocabulary fingerprint must not be empty"
        );
        ensure!(
            !self.supported_varieties.is_empty(),
            "model input contract must declare supported varieties"
        );
        ensure!(
            self.supported_varieties
                .iter()
                .all(|variety| !variety.trim().is_empty()),
            "model input contract contains an empty variety"
        );
        let required = match self.kind {
            LinguisticInputKind::Phones => LinguisticIntent::Phones,
            LinguisticInputKind::Phonemes => LinguisticIntent::Phonemes,
            LinguisticInputKind::Graphemes | LinguisticInputKind::TextBpe => LinguisticIntent::Text,
        };
        ensure!(
            self.consumes.contains(&required),
            "{:?} model input contract must consume {required:?}",
            self.kind
        );
        Ok(())
    }

    pub fn ensure_supports(&self, plan: &UtterancePlan) -> Result<()> {
        self.validate()?;
        ensure!(
            self.supported_varieties
                .iter()
                .any(|variety| variety_matches(variety, &plan.variety.0)),
            "model input does not support variety `{}`; supported varieties: {}",
            plan.variety.0,
            self.supported_varieties.join(", ")
        );
        Ok(())
    }

    pub fn unconsumed_intent(&self, plan: &UtterancePlan) -> Vec<LinguisticIntent> {
        plan_intent(plan)
            .difference(&self.consumes)
            .copied()
            .collect()
    }
}

fn variety_matches(supported: &str, requested: &str) -> bool {
    supported == "*"
        || supported.eq_ignore_ascii_case(requested)
        || speaking::canonical_variety_id(supported)
            .zip(speaking::canonical_variety_id(requested))
            .is_some_and(|(supported, requested)| supported == requested)
        || requested
            .get(..supported.len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(supported))
            && matches!(requested.as_bytes().get(supported.len()), Some(b'-' | b'.'))
}

/// Terminal, model-owned lowering from Tongues' linguistic IR.
///
/// `ModelInput` may contain checkpoint-specific phone IDs, characters, or BPE
/// tokens. It must remain private to the adapter and must not replace the
/// [`UtterancePlan`] at the shared synthesis boundary.
pub trait LinguisticProjector {
    type ModelInput;

    fn contract(&self) -> &ModelInputContract;
    fn project(&self, plan: &UtterancePlan) -> Result<Self::ModelInput>;
}

fn plan_intent(plan: &UtterancePlan) -> BTreeSet<LinguisticIntent> {
    let mut intent = BTreeSet::new();
    if plan.intended_text.is_some() {
        intent.insert(LinguisticIntent::Text);
    }
    if !plan.intended_morphemes.is_empty() {
        intent.insert(LinguisticIntent::Morphemes);
    }
    if !plan.intended_phonemes.is_empty() {
        intent.insert(LinguisticIntent::Phonemes);
    }
    if !plan.target_phones.is_empty() {
        intent.insert(LinguisticIntent::Phones);
    }
    if !plan.target_syllables.is_empty() {
        intent.insert(LinguisticIntent::Syllables);
    }
    if !plan.boundaries.is_empty() {
        intent.insert(LinguisticIntent::Boundaries);
    }
    if !plan.target_prosody.pitch.points.is_empty() {
        intent.insert(LinguisticIntent::Pitch);
    }
    if !plan.target_prosody.energy.points.is_empty() {
        intent.insert(LinguisticIntent::Energy);
    }
    if !plan.target_prosody.speaking_rate.points.is_empty() {
        intent.insert(LinguisticIntent::SpeakingRate);
    }
    if !plan.target_prosody.breaks.is_empty() {
        intent.insert(LinguisticIntent::ProsodicBreaks);
    }
    if !plan.target_prosody.labels.is_empty() {
        intent.insert(LinguisticIntent::ProsodicLabels);
    }
    if !plan.target_acoustics.is_empty() {
        intent.insert(LinguisticIntent::TargetAcoustics);
    }
    if plan.speaker.is_some() {
        intent.insert(LinguisticIntent::SpeakerIdentity);
    }
    if plan.speaker_reference.is_some() {
        intent.insert(LinguisticIntent::SpeakerReference);
    }
    if plan.style.is_some() {
        intent.insert(LinguisticIntent::Style);
    }
    intent
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditioningKind {
    Speaker,
    Style,
    Language,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingContract {
    pub kind: ConditioningKind,
    pub space: String,
    pub dimensions: usize,
    pub l2_normalized: bool,
}

impl EmbeddingContract {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.space.trim().is_empty(),
            "embedding space identity must not be empty"
        );
        ensure!(self.dimensions > 0, "embedding dimensions must be positive");
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConditioningEmbedding {
    pub contract: EmbeddingContract,
    pub values: Vec<f32>,
}

impl ConditioningEmbedding {
    pub fn validate(&self) -> Result<()> {
        self.contract.validate()?;
        ensure!(
            self.values.len() == self.contract.dimensions,
            "embedding contains {} values; contract requires {}",
            self.values.len(),
            self.contract.dimensions
        );
        ensure!(
            self.values.iter().all(|value| value.is_finite()),
            "embedding contains non-finite values"
        );
        Ok(())
    }
}

/// Encodes model-specific speaker or style conditioning from the native plan.
pub trait ReferenceEncoder {
    fn runtime(&self) -> InferenceRuntime;
    fn output_contract(&self) -> &EmbeddingContract;
    fn encode(&mut self, plan: &UtterancePlan) -> Result<ConditioningEmbedding>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveformLayout {
    Interleaved,
    Planar,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveformContract {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub layout: WaveformLayout,
}

impl WaveformContract {
    pub fn mono(sample_rate_hz: u32) -> Self {
        Self {
            sample_rate_hz,
            channels: 1,
            layout: WaveformLayout::Interleaved,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.sample_rate_hz > 0,
            "waveform sample rate must be positive"
        );
        ensure!(self.channels > 0, "waveform channel count must be positive");
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Waveform {
    pub contract: WaveformContract,
    pub samples: Vec<f32>,
}

impl Waveform {
    pub fn mono(sample_rate_hz: u32, samples: Vec<f32>) -> Self {
        Self {
            contract: WaveformContract::mono(sample_rate_hz),
            samples,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.contract.validate()?;
        ensure!(
            self.samples
                .len()
                .is_multiple_of(usize::from(self.contract.channels)),
            "waveform sample count {} is not divisible by {} channels",
            self.samples.len(),
            self.contract.channels
        );
        ensure!(
            self.samples.iter().all(|sample| sample.is_finite()),
            "waveform contains non-finite samples"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectrogramKind {
    Linear,
    Mel {
        min_frequency_hz: f32,
        max_frequency_hz: Option<f32>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrogramDomain {
    Amplitude,
    Power,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectrogramScale {
    Linear,
    NaturalLog { gain: f32 },
    Log10 { gain: f32 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum SpectrogramNormalization {
    None,
    Range {
        min_db: f32,
        reference_db: f32,
        max_norm: f32,
        symmetric: bool,
        clipped: bool,
    },
    Standardized {
        mean: Vec<f32>,
        scale: Vec<f32>,
    },
    /// Mean/variance normalization whose safe, path-independent identity is
    /// the digest of an upstream opaque statistics artifact.
    ///
    /// Some legacy Coqui bundles store statistics as pickled NumPy objects.
    /// Composition needs their exact identity, but native inference does not
    /// need to execute or deserialize that object.
    OpaqueStandardized {
        sha256: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrogramLayout {
    FramesByBins,
    BinsByFrames,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpectrogramPadMode {
    Reflect,
    Constant,
    Other(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelFilterBank {
    Slaney,
    Htk,
}

/// Complete interchange contract between an acoustic model and a vocoder.
///
/// Two components may connect only when this value matches exactly. Sample
/// rate and bin count alone are not enough to establish compatibility.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectrogramContract {
    pub kind: SpectrogramKind,
    pub domain: SpectrogramDomain,
    pub scale: SpectrogramScale,
    pub normalization: SpectrogramNormalization,
    pub sample_rate_hz: u32,
    pub fft_size: usize,
    pub window_size: usize,
    pub hop_size: usize,
    pub bins: usize,
    pub centered: bool,
    /// Explicit time-domain padding applied before a non-centered STFT.
    pub frame_padding: Option<(usize, usize)>,
    pub pad_mode: SpectrogramPadMode,
    pub preemphasis: Option<f32>,
    pub mel_filter_bank: Option<MelFilterBank>,
    pub layout: SpectrogramLayout,
}

impl SpectrogramContract {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.sample_rate_hz > 0,
            "spectrogram sample rate must be positive"
        );
        ensure!(self.fft_size > 0, "spectrogram FFT size must be positive");
        ensure!(
            self.window_size > 0 && self.window_size <= self.fft_size,
            "spectrogram window size {} must be in 1..={}",
            self.window_size,
            self.fft_size
        );
        ensure!(self.hop_size > 0, "spectrogram hop size must be positive");
        ensure!(self.bins > 0, "spectrogram bin count must be positive");
        if let Some((left, right)) = self.frame_padding {
            ensure!(
                !self.centered && (left > 0 || right > 0),
                "explicit spectrogram frame padding requires a non-centered STFT and at least one padded sample"
            );
        }
        if let Some(preemphasis) = self.preemphasis {
            ensure!(
                preemphasis.is_finite() && (0.0..1.0).contains(&preemphasis),
                "spectrogram preemphasis must be finite and in 0..1"
            );
        }
        match self.scale {
            SpectrogramScale::Linear => {}
            SpectrogramScale::NaturalLog { gain } | SpectrogramScale::Log10 { gain } => ensure!(
                gain.is_finite() && gain > 0.0,
                "spectrogram logarithmic gain must be finite and positive"
            ),
        }
        match self.kind {
            SpectrogramKind::Linear => ensure!(
                {
                    ensure!(
                        self.mel_filter_bank.is_none(),
                        "linear spectrogram must not declare a mel filter bank"
                    );
                    self.bins == self.fft_size / 2 + 1
                },
                "linear spectrogram has {} bins; FFT size {} requires {}",
                self.bins,
                self.fft_size,
                self.fft_size / 2 + 1
            ),
            SpectrogramKind::Mel {
                min_frequency_hz,
                max_frequency_hz,
            } => {
                ensure!(
                    self.mel_filter_bank.is_some(),
                    "mel spectrogram must declare its filter-bank convention"
                );
                ensure!(
                    min_frequency_hz.is_finite() && min_frequency_hz >= 0.0,
                    "mel minimum frequency must be finite and non-negative"
                );
                if let Some(max_frequency_hz) = max_frequency_hz {
                    ensure!(
                        max_frequency_hz.is_finite()
                            && max_frequency_hz > min_frequency_hz
                            && max_frequency_hz <= self.sample_rate_hz as f32 / 2.0,
                        "mel maximum frequency must be above the minimum and at or below Nyquist"
                    );
                }
            }
        }
        validate_normalization(&self.normalization, self.bins)
    }

    pub fn ensure_compatible_with(&self, required: &Self) -> Result<()> {
        self.validate().context("produced spectrogram contract")?;
        required
            .validate()
            .context("required spectrogram contract")?;
        ensure!(
            self == required,
            "spectrogram contract mismatch: produced {self:?}, required {required:?}"
        );
        Ok(())
    }
}

fn validate_normalization(normalization: &SpectrogramNormalization, bins: usize) -> Result<()> {
    match normalization {
        SpectrogramNormalization::None => Ok(()),
        SpectrogramNormalization::Range {
            min_db,
            reference_db,
            max_norm,
            ..
        } => {
            ensure!(
                min_db.is_finite() && *min_db < 0.0,
                "spectrogram min_db must be finite and negative"
            );
            ensure!(
                reference_db.is_finite(),
                "spectrogram reference_db must be finite"
            );
            ensure!(
                max_norm.is_finite() && *max_norm > 0.0,
                "spectrogram max_norm must be finite and positive"
            );
            Ok(())
        }
        SpectrogramNormalization::Standardized { mean, scale } => {
            ensure!(
                mean.len() == bins && scale.len() == bins,
                "spectrogram standardization vectors must contain one value per bin"
            );
            ensure!(
                mean.iter().all(|value| value.is_finite())
                    && scale.iter().all(|value| value.is_finite() && *value != 0.0),
                "spectrogram standardization vectors contain invalid values"
            );
            Ok(())
        }
        SpectrogramNormalization::OpaqueStandardized { sha256 } => {
            ensure!(
                sha256.len() == 64
                    && sha256
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
                "opaque standardization SHA-256 must be 64 lowercase hexadecimal characters"
            );
            Ok(())
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Spectrogram {
    pub contract: SpectrogramContract,
    pub frames: usize,
    pub values: Vec<f32>,
}

impl Spectrogram {
    pub fn validate(&self) -> Result<()> {
        self.contract.validate()?;
        ensure!(
            self.frames > 0,
            "spectrogram must contain at least one frame"
        );
        ensure!(
            self.values.len() == self.frames * self.contract.bins,
            "spectrogram contains {} values; {} frames by {} bins requires {}",
            self.values.len(),
            self.frames,
            self.contract.bins,
            self.frames * self.contract.bins
        );
        ensure!(
            self.values.iter().all(|value| value.is_finite()),
            "spectrogram contains non-finite values"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecContract {
    pub codec: String,
    pub version: String,
    pub sample_rate_hz: u32,
    pub frame_rate_hz: u32,
    pub codebooks: usize,
    pub vocabulary_size: usize,
}

impl CodecContract {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.codec.trim().is_empty(),
            "codec name must not be empty"
        );
        ensure!(
            !self.version.trim().is_empty(),
            "codec version must not be empty"
        );
        ensure!(
            self.sample_rate_hz > 0,
            "codec sample rate must be positive"
        );
        ensure!(self.frame_rate_hz > 0, "codec frame rate must be positive");
        ensure!(
            self.codebooks > 0,
            "codec must contain at least one codebook"
        );
        ensure!(
            self.vocabulary_size > 0,
            "codec vocabulary size must be positive"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecTokenSequence {
    pub contract: CodecContract,
    pub frames: usize,
    /// Frame-major tokens: every frame contains one token per codebook.
    pub tokens: Vec<u32>,
}

impl CodecTokenSequence {
    pub fn validate(&self) -> Result<()> {
        self.contract.validate()?;
        ensure!(self.frames > 0, "codec sequence must contain frames");
        ensure!(
            self.tokens.len() == self.frames * self.contract.codebooks,
            "codec sequence contains {} tokens; {} frames by {} codebooks requires {}",
            self.tokens.len(),
            self.frames,
            self.contract.codebooks,
            self.frames * self.contract.codebooks
        );
        ensure!(
            self.tokens
                .iter()
                .all(|token| (*token as usize) < self.contract.vocabulary_size),
            "codec sequence contains an out-of-range token"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AcousticOutputContract {
    Waveform(WaveformContract),
    Spectrogram(SpectrogramContract),
    Codec(CodecContract),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AcousticArtifact {
    Waveform(Waveform),
    Spectrogram(Spectrogram),
    Codec(CodecTokenSequence),
}

impl AcousticArtifact {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Waveform(waveform) => waveform.validate(),
            Self::Spectrogram(spectrogram) => spectrogram.validate(),
            Self::Codec(tokens) => tokens.validate(),
        }
    }

    pub fn contract(&self) -> AcousticOutputContract {
        match self {
            Self::Waveform(waveform) => AcousticOutputContract::Waveform(waveform.contract.clone()),
            Self::Spectrogram(spectrogram) => {
                AcousticOutputContract::Spectrogram(spectrogram.contract.clone())
            }
            Self::Codec(tokens) => AcousticOutputContract::Codec(tokens.contract.clone()),
        }
    }
}

/// A model that projects Tongues' linguistic plan into an acoustic artifact.
///
/// Model-local alphabets, BPE tokens, or grapheme IDs belong inside this
/// implementation. The shared request remains an [`speaking::UtterancePlan`].
pub trait AcousticModel {
    fn runtime(&self) -> InferenceRuntime;
    fn capabilities(&self) -> SpeechModelCapabilities;
    fn input_contract(&self) -> &ModelInputContract;
    fn conditioning_contracts(&self) -> &[EmbeddingContract];
    fn output_contract(&self) -> AcousticOutputContract;
    fn synthesize(&mut self, request: &SpeechSynthesisRequest) -> Result<AcousticArtifact>;
}

pub trait NeuralVocoder {
    fn runtime(&self) -> InferenceRuntime;
    fn input_contract(&self) -> &SpectrogramContract;
    fn output_contract(&self) -> WaveformContract;
    fn synthesize(&mut self, spectrogram: &Spectrogram) -> Result<Waveform>;
}

pub trait CodecDecoder {
    fn runtime(&self) -> InferenceRuntime;
    fn input_contract(&self) -> &CodecContract;
    fn output_contract(&self) -> WaveformContract;
    fn decode(&mut self, tokens: &CodecTokenSequence) -> Result<Waveform>;
}

pub trait AudioDecoder {
    fn runtime(&self) -> InferenceRuntime;
    fn validate_input_contract(&self, input: &AcousticOutputContract) -> Result<()>;
    fn output_contract(&self) -> WaveformContract;
    fn decode(&mut self, artifact: AcousticArtifact) -> Result<Waveform>;
}

#[derive(Debug, Clone)]
pub struct IdentityAudioDecoder {
    contract: WaveformContract,
}

impl IdentityAudioDecoder {
    pub fn new(contract: WaveformContract) -> Result<Self> {
        contract.validate()?;
        Ok(Self { contract })
    }
}

impl AudioDecoder for IdentityAudioDecoder {
    fn runtime(&self) -> InferenceRuntime {
        InferenceRuntime::PassThrough
    }

    fn validate_input_contract(&self, input: &AcousticOutputContract) -> Result<()> {
        match input {
            AcousticOutputContract::Waveform(contract) if contract == &self.contract => Ok(()),
            AcousticOutputContract::Waveform(contract) => bail!(
                "waveform contract mismatch: produced {contract:?}, required {:?}",
                self.contract
            ),
            other => bail!("identity audio decoder requires waveform input, got {other:?}"),
        }
    }

    fn output_contract(&self) -> WaveformContract {
        self.contract.clone()
    }

    fn decode(&mut self, artifact: AcousticArtifact) -> Result<Waveform> {
        let AcousticArtifact::Waveform(waveform) = artifact else {
            bail!("identity audio decoder received a non-waveform artifact")
        };
        ensure!(
            waveform.contract == self.contract,
            "waveform artifact does not match the identity decoder contract"
        );
        waveform.validate()?;
        Ok(waveform)
    }
}

#[derive(Debug)]
pub struct VocoderDecoder<V> {
    vocoder: V,
}

impl<V> VocoderDecoder<V> {
    pub fn new(vocoder: V) -> Self {
        Self { vocoder }
    }

    pub fn vocoder(&self) -> &V {
        &self.vocoder
    }

    pub fn vocoder_mut(&mut self) -> &mut V {
        &mut self.vocoder
    }
}

impl<V: NeuralVocoder> AudioDecoder for VocoderDecoder<V> {
    fn runtime(&self) -> InferenceRuntime {
        self.vocoder.runtime()
    }

    fn validate_input_contract(&self, input: &AcousticOutputContract) -> Result<()> {
        let AcousticOutputContract::Spectrogram(produced) = input else {
            bail!("neural vocoder requires spectrogram input, got {input:?}")
        };
        produced.ensure_compatible_with(self.vocoder.input_contract())
    }

    fn output_contract(&self) -> WaveformContract {
        self.vocoder.output_contract()
    }

    fn decode(&mut self, artifact: AcousticArtifact) -> Result<Waveform> {
        let AcousticArtifact::Spectrogram(spectrogram) = artifact else {
            bail!("neural vocoder received a non-spectrogram artifact")
        };
        spectrogram
            .contract
            .ensure_compatible_with(self.vocoder.input_contract())?;
        self.vocoder.synthesize(&spectrogram)
    }
}

#[derive(Debug)]
pub struct CodecDecoderAdapter<D> {
    decoder: D,
}

impl<D> CodecDecoderAdapter<D> {
    pub fn new(decoder: D) -> Self {
        Self { decoder }
    }
}

impl<D: CodecDecoder> AudioDecoder for CodecDecoderAdapter<D> {
    fn runtime(&self) -> InferenceRuntime {
        self.decoder.runtime()
    }

    fn validate_input_contract(&self, input: &AcousticOutputContract) -> Result<()> {
        let AcousticOutputContract::Codec(produced) = input else {
            bail!("codec decoder requires codec-token input, got {input:?}")
        };
        ensure!(
            produced == self.decoder.input_contract(),
            "codec contract mismatch: produced {produced:?}, required {:?}",
            self.decoder.input_contract()
        );
        Ok(())
    }

    fn output_contract(&self) -> WaveformContract {
        self.decoder.output_contract()
    }

    fn decode(&mut self, artifact: AcousticArtifact) -> Result<Waveform> {
        let AcousticArtifact::Codec(tokens) = artifact else {
            bail!("codec decoder received a non-codec artifact")
        };
        ensure!(
            &tokens.contract == self.decoder.input_contract(),
            "codec artifact does not match the decoder contract"
        );
        self.decoder.decode(&tokens)
    }
}

/// Composes a Tongues-plan acoustic model with a compatible audio decoder.
pub struct SpeechPipeline<M, D> {
    acoustic_model: M,
    decoder: D,
    output_contract: WaveformContract,
}

impl<M: AcousticModel, D: AudioDecoder> SpeechPipeline<M, D> {
    pub fn new(acoustic_model: M, decoder: D) -> Result<Self> {
        acoustic_model.input_contract().validate()?;
        for contract in acoustic_model.conditioning_contracts() {
            contract.validate()?;
        }
        decoder.validate_input_contract(&acoustic_model.output_contract())?;
        let output_contract = decoder.output_contract();
        output_contract.validate()?;
        Ok(Self {
            acoustic_model,
            decoder,
            output_contract,
        })
    }

    pub fn acoustic_model(&self) -> &M {
        &self.acoustic_model
    }

    pub fn acoustic_model_mut(&mut self) -> &mut M {
        &mut self.acoustic_model
    }

    pub fn decoder(&self) -> &D {
        &self.decoder
    }

    pub fn decoder_mut(&mut self) -> &mut D {
        &mut self.decoder
    }

    pub fn runtimes(&self) -> (InferenceRuntime, InferenceRuntime) {
        (self.acoustic_model.runtime(), self.decoder.runtime())
    }

    pub fn synthesize(&mut self, request: &SpeechSynthesisRequest) -> Result<Waveform> {
        self.acoustic_model
            .input_contract()
            .ensure_supports(&request.plan)?;
        let artifact = self.acoustic_model.synthesize(request)?;
        artifact.validate()?;
        ensure!(
            artifact.contract() == self.acoustic_model.output_contract(),
            "acoustic model returned an artifact that violates its declared output contract"
        );
        let waveform = self.decoder.decode(artifact)?;
        ensure!(
            waveform.contract == self.output_contract,
            "audio decoder returned a waveform that violates its declared output contract"
        );
        waveform.validate()?;
        Ok(waveform)
    }
}

impl<M: AcousticModel, D: AudioDecoder> SpeechSynthesisEngine for SpeechPipeline<M, D> {
    fn capabilities(&self) -> SpeechModelCapabilities {
        self.acoustic_model.capabilities()
    }

    fn sample_rate_hz(&self) -> u32 {
        self.output_contract.sample_rate_hz
    }

    fn synthesize_plan_streaming(
        &mut self,
        request: &SpeechSynthesisRequest,
        sink: &mut dyn AudioSink,
    ) -> Result<()> {
        let waveform = self.synthesize(request)?;
        ensure!(
            waveform.contract.channels == 1,
            "AudioSink currently accepts mono waveform output; pipeline produced {} channels",
            waveform.contract.channels
        );
        sink.emit(AudioChunk {
            chunk_index: 0,
            is_final: true,
            pause_after_ms: 0,
            sample_rate_hz: waveform.contract.sample_rate_hz,
            pcm_mono_f32: waveform.samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use speaking::{EvidenceProvenance, EvidenceSource, UtteranceId, UtterancePlan, VarietyId};

    use super::*;
    use crate::{SpeechModelFamily, SynthesisOptions};

    fn mel_contract(hop_size: usize) -> SpectrogramContract {
        SpectrogramContract {
            kind: SpectrogramKind::Mel {
                min_frequency_hz: 0.0,
                max_frequency_hz: Some(8_000.0),
            },
            domain: SpectrogramDomain::Amplitude,
            scale: SpectrogramScale::Log10 { gain: 20.0 },
            normalization: SpectrogramNormalization::Range {
                min_db: -100.0,
                reference_db: 20.0,
                max_norm: 4.0,
                symmetric: true,
                clipped: true,
            },
            sample_rate_hz: 22_050,
            fft_size: 1_024,
            window_size: 1_024,
            hop_size,
            bins: 80,
            centered: true,
            frame_padding: None,
            pad_mode: SpectrogramPadMode::Reflect,
            preemphasis: None,
            mel_filter_bank: Some(MelFilterBank::Slaney),
            layout: SpectrogramLayout::FramesByBins,
        }
    }

    fn request() -> SpeechSynthesisRequest {
        SpeechSynthesisRequest {
            plan: UtterancePlan {
                id: UtteranceId("component-test".into()),
                variety: VarietyId("en-US".into()),
                speaker: None,
                intended_text: Some("hello".into()),
                intended_morphemes: vec![],
                intended_phonemes: vec![],
                target_phones: vec![],
                target_syllables: vec![],
                boundaries: vec![],
                target_prosody: Default::default(),
                target_acoustics: vec![],
                speaker_reference: None,
                style: None,
                provenance: EvidenceProvenance {
                    source: EvidenceSource::TtsPlan,
                    method: "component-test".into(),
                    version: None,
                },
            },
            options: SynthesisOptions::default(),
        }
    }

    fn capabilities() -> SpeechModelCapabilities {
        SpeechModelCapabilities {
            family: SpeechModelFamily::AcousticModel,
            supports_named_speakers: false,
            supports_languages: false,
            supports_reference_audio: false,
            supports_voice_conversion: false,
            integrated_vocoder: false,
        }
    }

    fn input_contract() -> ModelInputContract {
        ModelInputContract {
            kind: LinguisticInputKind::Phones,
            vocabulary_fingerprint: "test-phones-v1".into(),
            supported_varieties: vec!["en-US".into()],
            consumes: BTreeSet::from([LinguisticIntent::Phones]),
        }
    }

    #[test]
    fn base_language_contract_accepts_regional_varieties_only() {
        let mut contract = input_contract();
        contract.supported_varieties = vec!["en".into()];
        let mut regional = request().plan;
        regional.variety = VarietyId("en-GB".into());

        contract
            .ensure_supports(&regional)
            .expect("regional English");
        regional.variety = VarietyId("fr-FR".into());
        assert!(contract.ensure_supports(&regional).is_err());
    }

    #[test]
    fn variety_contract_accepts_registered_aliases_in_either_direction() {
        let mut contract = input_contract();
        let mut plan = request().plan;
        plan.variety = VarietyId("en-US-GA".into());

        contract
            .ensure_supports(&plan)
            .expect("en-US should match its canonical en-US-GA variety");

        contract.supported_varieties = vec!["en-US-GA".into()];
        plan.variety = VarietyId("en-US".into());
        contract
            .ensure_supports(&plan)
            .expect("canonical en-US-GA should match its en-US alias");
    }

    #[test]
    fn locale_contract_accepts_more_specific_varieties_but_not_sibling_locales() {
        let contract = input_contract();
        let mut plan = request().plan;
        plan.variety = VarietyId("en-US-AAE".into());

        contract
            .ensure_supports(&plan)
            .expect("en-US should support a more specific US English variety");

        plan.variety = VarietyId("en-GB-RP".into());
        assert!(contract.ensure_supports(&plan).is_err());
    }

    struct MockAcousticModel {
        contract: SpectrogramContract,
        input_contract: ModelInputContract,
        seen_plan_id: Arc<Mutex<Option<String>>>,
    }

    impl AcousticModel for MockAcousticModel {
        fn runtime(&self) -> InferenceRuntime {
            InferenceRuntime::Burn
        }

        fn capabilities(&self) -> SpeechModelCapabilities {
            capabilities()
        }

        fn input_contract(&self) -> &ModelInputContract {
            &self.input_contract
        }

        fn conditioning_contracts(&self) -> &[EmbeddingContract] {
            &[]
        }

        fn output_contract(&self) -> AcousticOutputContract {
            AcousticOutputContract::Spectrogram(self.contract.clone())
        }

        fn synthesize(&mut self, request: &SpeechSynthesisRequest) -> Result<AcousticArtifact> {
            *self.seen_plan_id.lock().expect("plan lock") = Some(request.plan.id.0.clone());
            Ok(AcousticArtifact::Spectrogram(Spectrogram {
                contract: self.contract.clone(),
                frames: 2,
                values: vec![0.0; self.contract.bins * 2],
            }))
        }
    }

    struct MockVocoder {
        contract: SpectrogramContract,
    }

    impl NeuralVocoder for MockVocoder {
        fn runtime(&self) -> InferenceRuntime {
            InferenceRuntime::Burn
        }

        fn input_contract(&self) -> &SpectrogramContract {
            &self.contract
        }

        fn output_contract(&self) -> WaveformContract {
            WaveformContract::mono(22_050)
        }

        fn synthesize(&mut self, spectrogram: &Spectrogram) -> Result<Waveform> {
            Ok(Waveform::mono(
                22_050,
                vec![0.25; spectrogram.frames * self.contract.hop_size],
            ))
        }
    }

    #[test]
    fn exact_spectrogram_contracts_compose_and_preserve_the_plan() {
        let contract = mel_contract(256);
        let seen_plan_id = Arc::new(Mutex::new(None));
        let model = MockAcousticModel {
            contract: contract.clone(),
            input_contract: input_contract(),
            seen_plan_id: seen_plan_id.clone(),
        };
        let decoder = VocoderDecoder::new(MockVocoder { contract });
        let mut pipeline = SpeechPipeline::new(model, decoder).expect("compatible pipeline");

        let waveform = pipeline.synthesize(&request()).expect("synthesis");

        assert_eq!(waveform.samples.len(), 512);
        assert_eq!(
            seen_plan_id.lock().expect("plan lock").as_deref(),
            Some("component-test")
        );
        assert_eq!(
            pipeline.runtimes(),
            (InferenceRuntime::Burn, InferenceRuntime::Burn)
        );
    }

    #[test]
    fn spectrogram_contract_mismatch_is_rejected_before_inference() {
        let produced = mel_contract(256);
        let required = mel_contract(300);
        let model = MockAcousticModel {
            contract: produced,
            input_contract: input_contract(),
            seen_plan_id: Arc::new(Mutex::new(None)),
        };
        let decoder = VocoderDecoder::new(MockVocoder { contract: required });

        let error = SpeechPipeline::new(model, decoder)
            .err()
            .expect("contract mismatch");

        assert!(error.to_string().contains("spectrogram contract mismatch"));
        assert!(error.to_string().contains("hop_size: 256"));
        assert!(error.to_string().contains("hop_size: 300"));
    }

    #[test]
    fn malformed_spectrogram_is_rejected_at_the_component_boundary() {
        let contract = mel_contract(256);
        let spectrogram = Spectrogram {
            contract,
            frames: 2,
            values: vec![0.0; 80],
        };

        let error = spectrogram.validate().expect_err("shape mismatch");

        assert!(error.to_string().contains("2 frames by 80 bins"));
    }

    #[test]
    fn model_cannot_return_a_different_declared_contract() {
        struct LyingModel {
            declared: SpectrogramContract,
            returned: SpectrogramContract,
            input_contract: ModelInputContract,
        }

        impl AcousticModel for LyingModel {
            fn runtime(&self) -> InferenceRuntime {
                InferenceRuntime::Burn
            }

            fn capabilities(&self) -> SpeechModelCapabilities {
                capabilities()
            }

            fn input_contract(&self) -> &ModelInputContract {
                &self.input_contract
            }

            fn conditioning_contracts(&self) -> &[EmbeddingContract] {
                &[]
            }

            fn output_contract(&self) -> AcousticOutputContract {
                AcousticOutputContract::Spectrogram(self.declared.clone())
            }

            fn synthesize(
                &mut self,
                _request: &SpeechSynthesisRequest,
            ) -> Result<AcousticArtifact> {
                Ok(AcousticArtifact::Spectrogram(Spectrogram {
                    contract: self.returned.clone(),
                    frames: 1,
                    values: vec![0.0; self.returned.bins],
                }))
            }
        }

        let declared = mel_contract(256);
        let returned = mel_contract(300);
        let decoder = VocoderDecoder::new(MockVocoder {
            contract: declared.clone(),
        });
        let mut pipeline = SpeechPipeline::new(
            LyingModel {
                declared,
                returned,
                input_contract: input_contract(),
            },
            decoder,
        )
        .expect("pipeline");

        let error = pipeline
            .synthesize(&request())
            .expect_err("lying component");

        assert!(error
            .to_string()
            .contains("violates its declared output contract"));
    }

    #[test]
    fn pocket_tts_component_is_present_as_experimental_only() {
        let pocket = native_speech_components()
            .iter()
            .find(|component| component.id == "pocket-tts-causal")
            .expect("Pocket TTS component must be listed for architecture tracking");
        assert_eq!(pocket.display_name, "Pocket TTS");
        assert_eq!(pocket.architecture, "pocket-tts");
        assert_eq!(pocket.kind, NativeSpeechComponentKind::EndToEnd);
        assert_eq!(
            pocket.readiness,
            NativeSpeechComponentReadiness::Experimental
        );
    }

    /// Verifies that every component ID claimed by an implementation module
    /// actually exists in [`NATIVE_SPEECH_COMPONENTS`].
    ///
    /// This test will fail when a developer adds a row to
    /// [`IMPLEMENTATION_MODULE_CLASSIFICATION`] with a component ID that has
    /// not yet been added to the inventory.
    #[test]
    fn every_classified_module_maps_to_existing_inventory_entries() {
        for (module, ids) in super::IMPLEMENTATION_MODULE_CLASSIFICATION {
            for id in *ids {
                assert!(
                    super::NATIVE_SPEECH_COMPONENTS.iter().any(|c| c.id == *id),
                    "module `{module}` is classified as providing component `{id}`, \
                     but that id is missing from NATIVE_SPEECH_COMPONENTS"
                );
            }
        }
    }

    /// Verifies that every entry in [`NATIVE_SPEECH_COMPONENTS`] is either
    /// claimed by a module row in [`IMPLEMENTATION_MODULE_CLASSIFICATION`] or
    /// explicitly listed in [`PLANNED_SPEECH_COMPONENT_IDS`].
    ///
    /// This test fails when a new component is added to the inventory without a
    /// corresponding module record, making it impossible for new implementations
    /// to slip through without classification.
    #[test]
    fn every_inventory_entry_has_a_module_classification() {
        for component in super::NATIVE_SPEECH_COMPONENTS {
            let has_module = super::IMPLEMENTATION_MODULE_CLASSIFICATION
                .iter()
                .any(|(_, ids)| ids.contains(&component.id));
            let is_planned = super::PLANNED_SPEECH_COMPONENT_IDS.contains(&component.id);
            assert!(
                has_module || is_planned,
                "component `{}` is in NATIVE_SPEECH_COMPONENTS but is not claimed by any \
                 row in IMPLEMENTATION_MODULE_CLASSIFICATION and is not in \
                 PLANNED_SPEECH_COMPONENT_IDS; add it to the appropriate list",
                component.id
            );
        }
    }
}
