use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once, OnceLock};

use anyhow::{bail, ensure, Context, Result};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Int, Tensor as BurnTensor, TensorData};
#[cfg(feature = "onnx-tts")]
use ort::ep::ExecutionProviderDispatch;
#[cfg(feature = "onnx-tts")]
use ort::session::{builder::GraphOptimizationLevel, Session};
#[cfg(feature = "onnx-tts")]
use ort::value::{DynTensorValueType, Tensor, TensorElementType};
use serde_json::Value;
#[cfg(any(feature = "onnx-tts", test))]
use speaking::SpeakerId;
use speaking::{
    phonemicizer_for_variety, wiktionary_language_for_variety, EvidenceProvenance, EvidenceSource,
    FeatureId, FeatureValue, PauseKind, PhoneToken, PhonemeToken, PhonemicizeOutput,
    PhonemicizeRequest, ProsodicLabelKind, Spec, SpeechBoundaryToken, TerminalPunctuation,
    UtteranceId, UtterancePlan, VarietyId,
};
use tongues_core::{Vocab, UNK_ID};
use tongues_g2p2g::{load_model, ModelConfig, Seq2SeqModel};
use tongues_wiktionary::{wiktionary_infer_source, WiktionaryInferNotation};

pub mod burn_acoustic;
pub mod burn_hifigan;
pub mod burn_pipeline;
pub mod burn_speedy_speech;
pub mod burn_vits;
pub mod burn_vits_decoder;
pub mod burn_vits_duration;
pub mod burn_vits_flow;
pub mod burn_vits_text;
pub mod burn_vocoder;
pub mod components;
pub mod device;
pub mod model_config;
pub mod phoneme_projector;
pub mod profiling;
pub mod speakers;
pub mod vits_config;
#[allow(dead_code)]
mod vits_projector;

pub use burn_hifigan::{HifiganError, HifiganGenerator, HifiganGeneratorConfig};
pub use burn_pipeline::BurnSpeedySpeechPipeline;
pub use burn_speedy_speech::{
    ResidualConvConfig, SpeedySpeech, SpeedySpeechConfig, SpeedySpeechError, SpeedySpeechOutput,
};
pub use burn_vits::BurnVitsSpeech;
pub use burn_vits_decoder::{
    VitsWaveformDecoder, VitsWaveformDecoderConfig, VitsWaveformDecoderError,
};
pub use burn_vits_duration::{
    StochasticDurationConfig, StochasticDurationError, StochasticDurationPredictor,
};
pub use burn_vits_flow::{
    ceil_durations, expand_prior_statistics, CeiledDurations, ExpandedPrior, ResidualCouplingFlow,
    ResidualCouplingFlowConfig, VitsFlowError,
};
pub use burn_vits_text::{
    VitsTextPriorConfig, VitsTextPriorEncoder, VitsTextPriorError, VitsTextPriorOutput,
};
pub use burn_vocoder::BurnHifiganVocoder;
pub use components::{
    AcousticArtifact, AcousticModel, AcousticOutputContract, CodecContract, CodecDecoder,
    CodecDecoderAdapter, CodecTokenSequence, ConditioningEmbedding, ConditioningKind,
    EmbeddingContract, IdentityAudioDecoder, InferenceRuntime, LinguisticInputKind,
    LinguisticIntent, LinguisticProjector, MelFilterBank, ModelInputContract, NeuralVocoder,
    ReferenceEncoder, Spectrogram, SpectrogramContract, SpectrogramDomain, SpectrogramKind,
    SpectrogramLayout, SpectrogramNormalization, SpectrogramPadMode, SpectrogramScale,
    SpeechPipeline, VocoderDecoder, Waveform, WaveformContract, WaveformLayout,
};
pub use device::{
    resolve_speech_device, ResolvedSpeechDevice, SpeechDeviceRequest, SpeechDeviceSelection,
    SpeechDeviceSelectionError, SpeechDeviceSpecError, MAX_CUDA_DEVICE_INDEX,
};
pub use model_config::{AudioFeatureConfig, HifiganBundleConfig, HifiganGeneratorParams};
pub use phoneme_projector::{
    PhonemeCharactersConfig, PhonemeTokenIds, PhonemeTokenizerConfig, PhonemeVocabularyProjector,
};
pub use profiling::{
    ModelLoadProfileEvent, ModelLoadStage, SynthesisDimension, SynthesisProfileEvent,
    SynthesisProfiler, SynthesisStage,
};
pub use speakers::SpeakerCatalog;
pub use vits_config::{VitsInferenceConfig, VitsNetworkConfig};

pub const RYAN_MEDIUM_MODEL_URL: &str = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx";
pub const RYAN_MEDIUM_CONFIG_URL: &str = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json";
pub const AMY_MEDIUM_MODEL_URL: &str = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx";
pub const AMY_MEDIUM_CONFIG_URL: &str = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json";
pub const LJSPEECH_HIGH_MODEL_URL: &str = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx";
pub const LJSPEECH_HIGH_CONFIG_URL: &str = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx.json";
pub const DEFAULT_TTS_CATALOG_MODEL: &str = "tts_models/en/ljspeech/tacotron2-DDC";
pub const DEFAULT_VOCODER_CATALOG_MODEL: &str = "vocoder_models/en/ljspeech/hifigan_v2";
pub const DEFAULT_WIKTIONARY_FALLBACK_MODEL_DIR: &str =
    "models/wiktionary/enwiktionary-2026-06-01-v0-phones";

type CpuInferBackend = NdArray<f32>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechRequest {
    pub text: String,
    pub variety: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechAudio {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub pcm_mono_f32: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceModel {
    RyanMedium,
    AmyMedium,
    LjspeechHigh,
    Path { model: PathBuf, config: PathBuf },
}

pub fn default_voice_model() -> VoiceModel {
    VoiceModel::LjspeechHigh
}

#[derive(Debug)]
pub struct OnnxSpeech {
    backend: OnnxSpeechBackend,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VoiceConfig {
    pub sample_rate_hz: u32,
    pub phoneme_id_map: HashMap<String, Vec<i64>>,
    pub num_speakers: Option<u32>,
    pub speaker_id_map: HashMap<String, u32>,
    pub length_scale: Option<f32>,
    pub noise_scale: Option<f32>,
    pub noise_w: Option<f32>,
}

impl VoiceConfig {
    pub fn available_speaker_names(&self) -> Vec<String> {
        let mut names = self.speaker_id_map.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    pub fn speaker_count(&self) -> u32 {
        self.num_speakers
            .or_else(|| u32::try_from(self.speaker_id_map.len()).ok())
            .unwrap_or(0)
    }

    pub fn is_multi_speaker(&self) -> bool {
        self.speaker_count() > 1 || self.speaker_id_map.len() > 1
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeechModelFamily {
    AcousticModel,
    NeuralVocoder,
    EndToEndSpeech,
    CrossLingualVoiceClone,
    VoiceConversion,
    Unknown(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechModelCapabilities {
    pub family: SpeechModelFamily,
    pub supports_named_speakers: bool,
    pub supports_languages: bool,
    pub supports_reference_audio: bool,
    pub supports_voice_conversion: bool,
    pub integrated_vocoder: bool,
}

impl SpeechModelCapabilities {
    pub fn onnx_voice(config: &VoiceConfig) -> Self {
        Self {
            // The imported config format does not identify the architecture.
            // Do not infer VITS merely because a compatible ONNX graph loaded.
            family: SpeechModelFamily::EndToEndSpeech,
            supports_named_speakers: !config.speaker_id_map.is_empty(),
            supports_languages: false,
            supports_reference_audio: false,
            supports_voice_conversion: false,
            integrated_vocoder: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SynthesisOptions {
    /// Low-level model speaker ID override.
    ///
    /// Named speaker identity belongs in `UtterancePlan::speaker`.
    pub speaker_id: Option<u32>,
    pub split_sentences: bool,
    pub length_scale: Option<f32>,
    pub noise_scale: Option<f32>,
    pub noise_w: Option<f32>,
    /// Backend RNG seed for repeatable stochastic inference.
    pub seed: Option<u64>,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            speaker_id: None,
            split_sentences: true,
            length_scale: None,
            noise_scale: None,
            noise_w: None,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechSynthesisRequest {
    pub plan: UtterancePlan,
    pub options: SynthesisOptions,
}

pub trait SpeechSynthesisEngine {
    fn capabilities(&self) -> SpeechModelCapabilities;
    fn sample_rate_hz(&self) -> u32;
    fn synthesize_plan_streaming(
        &mut self,
        request: &SpeechSynthesisRequest,
        sink: &mut dyn AudioSink,
    ) -> Result<()>;

    /// Synthesizes while reporting synchronized native stage timings.
    ///
    /// Backends without native stage instrumentation retain the ordinary
    /// behavior and emit no events.
    fn synthesize_plan_streaming_profiled(
        &mut self,
        request: &SpeechSynthesisRequest,
        sink: &mut dyn AudioSink,
        _profiler: &mut dyn SynthesisProfiler,
    ) -> Result<()> {
        self.synthesize_plan_streaming(request, sink)
    }
}

#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhonemeSequence {
    pub symbols: Vec<String>,
}

#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynthesisChunk {
    pub sequence: PhonemeSequence,
    pub pause_after_ms: u32,
}

#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhonemeIdSequence {
    pub ids: Vec<i64>,
}

#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct SynthesisOutput {
    pub sample_rate_hz: u32,
    pub pcm_mono_f32: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioChunk {
    pub chunk_index: usize,
    pub is_final: bool,
    pub pause_after_ms: u32,
    pub sample_rate_hz: u32,
    pub pcm_mono_f32: Vec<f32>,
}

pub trait AudioSink {
    fn emit(&mut self, chunk: AudioChunk) -> Result<()>;
}

impl<F> AudioSink for F
where
    F: FnMut(AudioChunk) -> Result<()>,
{
    fn emit(&mut self, chunk: AudioChunk) -> Result<()> {
        self(chunk)
    }
}

#[cfg(feature = "onnx-tts")]
#[derive(Debug, Clone, PartialEq)]
struct OnnxTensorSpec {
    name: String,
    tensor_type: Option<TensorElementType>,
}

#[cfg(feature = "onnx-tts")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct OnnxInferenceContract {
    id_input: String,
    id_lengths_input: String,
    scales_input: Option<String>,
    noise_scale_input: Option<String>,
    length_scale_input: Option<String>,
    noise_w_input: Option<String>,
    speaker_input: Option<String>,
    output_audio: String,
}

#[cfg(feature = "onnx-tts")]
#[derive(Debug)]
pub struct OnnxSpeechBackend {
    config: VoiceConfig,
    model_path: PathBuf,
    session: Session,
}

impl VoiceConfig {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read voice config {}", path.display()))?;
        Self::from_json_str(&json)
            .with_context(|| format!("failed to parse voice config {}", path.display()))
    }

    pub fn from_json_str(json: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(json)?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &Value) -> Result<Self> {
        let sample_rate_hz = parse_required_u32(
            value,
            &[&["audio", "sample_rate"], &["sample_rate"]],
            "audio.sample_rate",
        )?;
        let phoneme_id_map = parse_phoneme_id_map(
            find_value(value, &[&["phoneme_id_map"], &["phoneme_map"]])
                .context("missing required voice config field `phoneme_id_map`")?,
        )?;
        let speaker_id_map = find_value(value, &[&["speaker_id_map"], &["speaker_map"]])
            .map(parse_speaker_id_map)
            .transpose()?
            .unwrap_or_default();
        let num_speakers = parse_optional_u32(
            value,
            &[&["num_speakers"], &["speaker_count"]],
            "num_speakers",
        )?
        .or_else(|| {
            if speaker_id_map.is_empty() {
                None
            } else {
                u32::try_from(speaker_id_map.len()).ok()
            }
        });

        Ok(Self {
            sample_rate_hz,
            phoneme_id_map,
            num_speakers,
            speaker_id_map,
            length_scale: parse_optional_f32(
                value,
                &[&["inference", "length_scale"], &["length_scale"]],
                "inference.length_scale",
            )?,
            noise_scale: parse_optional_f32(
                value,
                &[&["inference", "noise_scale"], &["noise_scale"]],
                "inference.noise_scale",
            )?,
            noise_w: parse_optional_f32(
                value,
                &[&["inference", "noise_w"], &["noise_w"]],
                "inference.noise_w",
            )?,
        })
    }
}

pub fn voice_config_path(model_path: &Path) -> PathBuf {
    model_path.with_extension("onnx.json")
}

pub fn default_voice_model_path(voice: VoiceModel) -> PathBuf {
    match voice {
        VoiceModel::RyanMedium => default_voice_model_dir().join("en_US-ryan-medium.onnx"),
        VoiceModel::AmyMedium => default_voice_model_dir().join("en_US-amy-medium.onnx"),
        VoiceModel::LjspeechHigh => default_voice_model_dir().join("en_US-ljspeech-high.onnx"),
        VoiceModel::Path { model, .. } => model,
    }
}

pub fn default_voice_config_path(voice: VoiceModel) -> PathBuf {
    match voice {
        VoiceModel::RyanMedium => default_voice_model_dir().join("en_US-ryan-medium.onnx.json"),
        VoiceModel::AmyMedium => default_voice_model_dir().join("en_US-amy-medium.onnx.json"),
        VoiceModel::LjspeechHigh => default_voice_model_dir().join("en_US-ljspeech-high.onnx.json"),
        VoiceModel::Path { config, .. } => config,
    }
}

fn default_voice_model_dir() -> PathBuf {
    dirs::data_local_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("tongues")
        .join("models")
        .join("voices")
}

fn repo_relative_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(path)
}

fn default_wiktionary_fallback_model_dir() -> PathBuf {
    std::env::var_os("TONGUES_TTS_WIKTIONARY_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_relative_path(DEFAULT_WIKTIONARY_FALLBACK_MODEL_DIR))
}

struct WiktionaryFallbackPredictor {
    model: Seq2SeqModel<CpuInferBackend>,
    vocab: Vocab,
    device: NdArrayDevice,
}

impl WiktionaryFallbackPredictor {
    fn load(model_dir: &Path) -> Result<Self> {
        let device = NdArrayDevice::Cpu;
        let model_config: ModelConfig = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("model_config.json")).with_context(|| {
                format!("reading {}", model_dir.join("model_config.json").display())
            })?,
        )?;
        let vocab: Vocab = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("vocab.json"))
                .with_context(|| format!("reading {}", model_dir.join("vocab.json").display()))?,
        )?;
        let model =
            load_model::<CpuInferBackend>(&model_config, &model_dir.join("model"), &device)?;
        Ok(Self {
            model,
            vocab,
            device,
        })
    }

    fn predict(&self, word: &str, variety: &str) -> Result<String> {
        let language = wiktionary_language_for_variety(variety).with_context(|| {
            format!("registered variety `{variety}` has no Wiktionary fallback language")
        })?;
        let language_token = format!("<lang:{language}>");
        anyhow::ensure!(
            self.vocab.get_id(&language_token) != UNK_ID,
            "Wiktionary fallback checkpoint does not support language `{language}` for variety `{variety}`"
        );
        let source = wiktionary_fallback_source(word, variety)?;
        let src_ids = self.vocab.encode_string(&source);
        let src_len = src_ids.len();
        let src_tensor = BurnTensor::<CpuInferBackend, 2, Int>::from_data(
            TensorData::new(
                src_ids.iter().map(|&id| id as i32).collect::<Vec<_>>(),
                [1, src_len],
            ),
            &self.device,
        );
        let pred_ids = self.model.generate(src_tensor, 128);
        Ok(self.vocab.decode_ids(&pred_ids))
    }
}

fn wiktionary_fallback_source(word: &str, variety: &str) -> Result<String> {
    let language = wiktionary_language_for_variety(variety)
        .with_context(|| format!("no Wiktionary language mapping for variety `{variety}`"))?;
    wiktionary_infer_source(
        "orthography-to-phonemes",
        language,
        WiktionaryInferNotation::Phonemes,
        wiktionary_variety_for_speaking_variety(variety),
        word,
    )
}

fn wiktionary_variety_for_speaking_variety(variety: &str) -> Option<&'static str> {
    match variety {
        "en-US" | "en-US-GA" | "en-US.GenAm" => Some("en-US.GenAm"),
        _ if variety.starts_with("en-US") => Some("en-US.GenAm"),
        _ => None,
    }
}

fn install_default_unknown_pronunciation_fallback() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        speaking::set_unknown_pronunciation_fallback(Some(wiktionary_unknown_pronunciation));
    });
}

fn wiktionary_unknown_pronunciation(word: &str, variety: &str) -> Option<String> {
    static PREDICTOR: OnceLock<Mutex<Option<WiktionaryFallbackPredictor>>> = OnceLock::new();
    let predictor = PREDICTOR.get_or_init(|| {
        let loaded =
            WiktionaryFallbackPredictor::load(&default_wiktionary_fallback_model_dir()).ok();
        Mutex::new(loaded)
    });
    predictor
        .lock()
        .ok()
        .and_then(|predictor| {
            predictor
                .as_ref()
                .and_then(|predictor| predictor.predict(word, variety).ok())
        })
        .filter(|prediction| !prediction.trim().is_empty())
}

impl OnnxSpeech {
    pub fn load(voice: VoiceModel) -> Result<Self> {
        let model_path = default_voice_model_path(voice.clone());
        let config_path = default_voice_config_path(voice);
        ensure!(
            config_path.is_file(),
            "voice config file not found at {}",
            config_path.display()
        );
        let config = VoiceConfig::from_json_file(&config_path)?;
        let backend = OnnxSpeechBackend::load(&model_path, config)
            .context("failed to load ONNX speech backend")?;
        Ok(Self { backend })
    }

    pub fn synthesize(&mut self, request: SpeechRequest) -> Result<SpeechAudio> {
        self.synthesize_with_options(request, &SynthesisOptions::default())
    }

    pub fn synthesize_with_options(
        &mut self,
        request: SpeechRequest,
        options: &SynthesisOptions,
    ) -> Result<SpeechAudio> {
        let plan = utterance_plan_from_text(request)?;
        self.synthesize_plan_with_options(&plan, options)
    }

    pub fn synthesize_plan(&mut self, plan: &UtterancePlan) -> Result<SpeechAudio> {
        self.synthesize_plan_with_options(plan, &SynthesisOptions::default())
    }

    pub fn synthesize_plan_with_options(
        &mut self,
        plan: &UtterancePlan,
        options: &SynthesisOptions,
    ) -> Result<SpeechAudio> {
        let mut pcm_mono_f32 = Vec::new();
        let mut sample_rate_hz = self.backend.sample_rate_hz();
        self.synthesize_plan_streaming_with_options(plan, options, &mut |audio| {
            sample_rate_hz = audio.sample_rate_hz;
            pcm_mono_f32.extend(audio.pcm_mono_f32);
            Ok(())
        })?;
        Ok(SpeechAudio {
            sample_rate_hz,
            channels: 1,
            pcm_mono_f32,
        })
    }

    pub fn synthesize_streaming(
        &mut self,
        request: SpeechRequest,
        sink: &mut dyn FnMut(SpeechAudio) -> Result<()>,
    ) -> Result<()> {
        self.synthesize_streaming_with_options(request, &SynthesisOptions::default(), sink)
    }

    pub fn synthesize_streaming_with_options(
        &mut self,
        request: SpeechRequest,
        options: &SynthesisOptions,
        sink: &mut dyn FnMut(SpeechAudio) -> Result<()>,
    ) -> Result<()> {
        let plan = utterance_plan_from_text(request)?;
        self.synthesize_plan_streaming_with_options(&plan, options, sink)
    }

    pub fn synthesize_plan_streaming(
        &mut self,
        plan: &UtterancePlan,
        sink: &mut dyn FnMut(SpeechAudio) -> Result<()>,
    ) -> Result<()> {
        self.synthesize_plan_streaming_with_options(plan, &SynthesisOptions::default(), sink)
    }

    pub fn synthesize_plan_streaming_with_options(
        &mut self,
        plan: &UtterancePlan,
        options: &SynthesisOptions,
        sink: &mut dyn FnMut(SpeechAudio) -> Result<()>,
    ) -> Result<()> {
        self.backend.synthesize_plan_streaming_with_options(
            plan,
            options,
            &mut |chunk: AudioChunk| {
                sink(SpeechAudio {
                    sample_rate_hz: chunk.sample_rate_hz,
                    channels: 1,
                    pcm_mono_f32: chunk.pcm_mono_f32,
                })
            },
        )
    }
}

#[doc(hidden)]
pub fn phoneme_ids_from_text(
    text: &str,
    variety: &str,
    config: &VoiceConfig,
) -> Result<PhonemeIdSequence> {
    let plan = utterance_plan_from_text(SpeechRequest {
        text: text.to_string(),
        variety: variety.to_string(),
    })?;
    phoneme_sequence_from_plan(&plan)?.to_text_ids_compatible(config)
}

pub fn utterance_plan_from_text(request: SpeechRequest) -> Result<UtterancePlan> {
    install_default_unknown_pronunciation_fallback();
    let variety = VarietyId(request.variety);
    let phonemicizer = phonemicizer_for_variety(&variety)
        .map_err(|error| anyhow::anyhow!("failed to load phonemicizer: {error}"))?;
    let phonemicized = phonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: request.text,
            variety,
            style: None,
        })
        .context("failed to phonemicize text into a speech plan")?;
    Ok(utterance_plan_from_phonemicized(&phonemicized))
}

pub fn utterance_plan_from_phonemicized(output: &PhonemicizeOutput) -> UtterancePlan {
    UtterancePlan {
        id: UtteranceId("tongues-tts.speech.utterance".into()),
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
        speaker_reference: None,
        style: None,
        provenance: EvidenceProvenance {
            source: EvidenceSource::TtsPlan,
            method: "tongues-tts phonemicized ONNX speech plan".into(),
            version: Some("0.1".into()),
        },
    }
}

#[doc(hidden)]
pub fn phoneme_sequence_from_plan(plan: &UtterancePlan) -> Result<PhonemeSequence> {
    let mut symbols = Vec::new();
    if !plan.target_phones.is_empty() {
        let punctuation_after_words = typed_punctuation_after_words(&plan.boundaries)
            .or_else(|| plan.intended_text.as_deref().map(punctuation_after_words))
            .unwrap_or_default();
        let mut word_index = 0;
        let mut in_word = false;
        for (token_index, token) in plan.target_phones.iter().enumerate() {
            let Spec::Known(phone_id) = &token.phone else {
                continue;
            };
            if phone_id.0 == "boundary.word" {
                if in_word {
                    let boundary_symbol = punctuation_after_words
                        .get(word_index)
                        .and_then(|symbol| *symbol);
                    if boundary_symbol.is_some()
                        || !next_phone_is_epenthetic_linker(&plan.target_phones[token_index + 1..])
                    {
                        push_boundary_symbols(&mut symbols, boundary_symbol.unwrap_or(" "));
                    }
                    word_index += 1;
                    in_word = false;
                }
            } else if phone_id.0 == "boundary.letter" {
                continue;
            } else {
                let symbol = speech_symbol_for_phone(token).with_context(|| {
                    format!(
                        "cannot lower phone `{}` to an ONNX voice ARPAbet symbol",
                        phone_id.0
                    )
                })?;
                push_symbol(&mut symbols, &symbol);
                in_word = true;
            }
        }
        if in_word {
            if let Some(symbol) = punctuation_after_words
                .get(word_index)
                .and_then(|symbol| *symbol)
            {
                push_symbol(&mut symbols, symbol);
            }
        }
    } else {
        for token in &plan.intended_phonemes {
            let Spec::Known(phoneme_id) = &token.phoneme else {
                continue;
            };
            let symbol = speech_symbol_for_phoneme(token).with_context(|| {
                format!(
                    "cannot lower phoneme `{}` to an ONNX voice ARPAbet symbol",
                    phoneme_id.0
                )
            })?;
            push_symbol(&mut symbols, &symbol);
        }
    }
    apply_speech_prosody_terminal_hint(&mut symbols, plan);
    append_default_terminal_symbol(&mut symbols);
    Ok(PhonemeSequence { symbols })
}

#[doc(hidden)]
pub fn synthesis_chunks_from_plan(plan: &UtterancePlan) -> Result<Vec<SynthesisChunk>> {
    phoneme_sequence_from_plan(plan).map(synthesis_chunks_from_sequence)
}

#[doc(hidden)]
pub fn synthesis_chunks_from_sequence(sequence: PhonemeSequence) -> Vec<SynthesisChunk> {
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    let mut pending_pause_after_ms = None;
    let mut skip_leading_spaces = false;

    for symbol in sequence.symbols {
        if skip_leading_spaces && symbol == " " {
            continue;
        }
        skip_leading_spaces = false;

        let pause_after_ms = speech_pause_after_ms(&symbol);
        current.push(symbol);
        if let Some(pause_after_ms) = pause_after_ms {
            chunks.push(SynthesisChunk {
                sequence: PhonemeSequence {
                    symbols: std::mem::take(&mut current),
                },
                pause_after_ms: 0,
            });
            pending_pause_after_ms = Some(pause_after_ms);
            skip_leading_spaces = true;
        } else if let Some(pause_after_ms) = pending_pause_after_ms.take() {
            if let Some(previous) = chunks.last_mut() {
                previous.pause_after_ms = pause_after_ms;
            }
        }
    }

    if !current.is_empty() {
        chunks.push(SynthesisChunk {
            sequence: PhonemeSequence { symbols: current },
            pause_after_ms: 0,
        });
    }

    chunks
}

fn speech_pause_after_ms(symbol: &str) -> Option<u32> {
    match symbol {
        "," | ";" | ":" => Some(220),
        "." | "!" | "?" => Some(380),
        _ => None,
    }
}

fn speech_symbol_for_phone(token: &PhoneToken) -> Option<String> {
    let Spec::Known(phone_id) = &token.phone else {
        return None;
    };
    if let Some(symbol) = arpabet_symbol_from_features(&token.features) {
        return Some(symbol);
    }
    speech_symbol_for_phone_id(phone_id.as_str()).map(str::to_string)
}

fn speech_symbol_for_phoneme(token: &PhonemeToken) -> Option<String> {
    arpabet_symbol_from_features(&token.features)
}

fn arpabet_symbol_from_features(features: &speaking::FeatureBundle) -> Option<String> {
    let base = feature_category(features, "phonology.base_symbol")?;
    if !is_arpabet_symbol(base) {
        return None;
    }
    if is_arpabet_vowel(base) {
        if let Some(stress) = feature_category(features, "phonology.stress").and_then(stress_digit)
        {
            return Some(format!("{base}{stress}"));
        }
    }
    Some(base.to_string())
}

fn speech_symbol_for_phone_id(phone_id: &str) -> Option<&'static str> {
    Some(match phone_id {
        "ipa.phone.ɑ" => "AA",
        "ipa.phone.æ" => "AE",
        "ipa.phone.ʌ" => "AH1",
        "ipa.phone.ə" | "ipa.phone.ɐ" => "AH0",
        "ipa.phone.ɔ" => "AO",
        "ipa.phone.aʊ" => "AW",
        "ipa.phone.aɪ" => "AY",
        "ipa.phone.b" => "B",
        "ipa.phone.tʃ" => "CH",
        "ipa.phone.d" => "D",
        "ipa.phone.ð" => "DH",
        "ipa.phone.ɛ" => "EH",
        "ipa.phone.ɝ" => "ER1",
        "ipa.phone.ɚ" => "ER0",
        "ipa.phone.eɪ" => "EY",
        "ipa.phone.f" => "F",
        "ipa.phone.ɡ" => "G",
        "ipa.phone.h" => "HH",
        "ipa.phone.ɪ" => "IH",
        "ipa.phone.iː" | "ipa.phone.i" => "IY",
        "ipa.phone.dʒ" => "JH",
        "ipa.phone.k" | "ipa.phone.kʰ" | "ipa.phone.k˭" => "K",
        "ipa.phone.l" | "ipa.phone.ɫ" => "L",
        "ipa.phone.m" => "M",
        "ipa.phone.n" => "N",
        "ipa.phone.ŋ" => "NG",
        "ipa.phone.oʊ" => "OW",
        "ipa.phone.ɔɪ" => "OY",
        "ipa.phone.p" | "ipa.phone.pʰ" | "ipa.phone.p˭" => "P",
        "ipa.phone.ɹ" => "R",
        "ipa.phone.s" => "S",
        "ipa.phone.ʃ" => "SH",
        "ipa.phone.t" | "ipa.phone.tʰ" | "ipa.phone.t˭" => "T",
        "ipa.phone.θ" => "TH",
        "ipa.phone.ʊ" => "UH",
        "ipa.phone.uː" | "ipa.phone.u" => "UW",
        "ipa.phone.v" => "V",
        "ipa.phone.w" => "W",
        "ipa.phone.j" => "Y",
        "ipa.phone.z" => "Z",
        "ipa.phone.ʒ" => "ZH",
        _ => return None,
    })
}

fn append_default_terminal_symbol(symbols: &mut Vec<String>) {
    if symbols.is_empty()
        || symbols
            .last()
            .is_some_and(|symbol| is_terminal_symbol(symbol))
    {
        return;
    }
    push_symbol(symbols, ".");
}

fn apply_speech_prosody_terminal_hint(symbols: &mut Vec<String>, plan: &UtterancePlan) {
    if !plan.target_prosody.labels.iter().any(|label| {
        matches!(
            label.kind,
            ProsodicLabelKind::QuestionRise | ProsodicLabelKind::AlternativeQuestionFall
        )
    }) {
        return;
    }

    if let Some(last) = symbols.last_mut() {
        if matches!(last.as_str(), "." | "!" | "?") {
            *last = "?".to_string();
            return;
        }
    }

    push_symbol(symbols, "?");
}

fn punctuation_after_words(text: &str) -> Vec<Option<&'static str>> {
    let word_spans = word_spans(text);
    word_spans
        .iter()
        .enumerate()
        .map(|(index, (_, end))| {
            let next_start = word_spans
                .get(index + 1)
                .map(|(start, _)| *start)
                .unwrap_or(text.len());
            punctuation_symbol(&text[*end..next_start])
        })
        .collect()
}

fn typed_punctuation_after_words(
    boundaries: &[SpeechBoundaryToken],
) -> Option<Vec<Option<&'static str>>> {
    let max_word_index = boundaries
        .iter()
        .filter(|boundary| typed_punctuation_symbol(boundary).is_some())
        .map(|boundary| boundary.after_grapheme_index)
        .max()?;
    let mut punctuation = vec![None; max_word_index + 1];
    for boundary in boundaries {
        if let Some(symbol) = typed_punctuation_symbol(boundary) {
            punctuation[boundary.after_grapheme_index] = Some(symbol);
        }
    }
    Some(punctuation)
}

fn typed_punctuation_symbol(boundary: &SpeechBoundaryToken) -> Option<&'static str> {
    if let Some(terminal) = boundary.terminal {
        return Some(match terminal {
            TerminalPunctuation::Period => ".",
            TerminalPunctuation::Question => "?",
            TerminalPunctuation::Exclamation => "!",
        });
    }
    if matches!(boundary.pause, Some(PauseKind::Comma)) {
        return Some(",");
    }
    None
}

fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = None;
    for (byte_index, character) in text.char_indices() {
        if is_word_chunk_character(character) {
            start.get_or_insert(byte_index);
            continue;
        }

        if let Some(start_byte) = start.take() {
            push_word_chunk_spans(text, start_byte, byte_index, &mut spans);
        }
    }

    if let Some(start_byte) = start {
        push_word_chunk_spans(text, start_byte, text.len(), &mut spans);
    }
    spans
}

fn is_word_chunk_character(character: char) -> bool {
    character.is_alphabetic() || is_apostrophe(character) || character == '-'
}

fn is_apostrophe(character: char) -> bool {
    matches!(character, '\'' | '’' | '‘' | 'ʼ')
}

fn push_word_chunk_spans(
    text: &str,
    start_byte: usize,
    end_byte: usize,
    spans: &mut Vec<(usize, usize)>,
) {
    let mut part_start = None;
    for (offset, character) in text[start_byte..end_byte].char_indices() {
        let byte_index = start_byte + offset;
        if character == '-' {
            if let Some(part_start_byte) = part_start.take() {
                push_camelcase_word_spans(text, part_start_byte, byte_index, spans);
            }
            continue;
        }

        part_start.get_or_insert(byte_index);
    }

    if let Some(part_start_byte) = part_start {
        push_camelcase_word_spans(text, part_start_byte, end_byte, spans);
    }
}

fn push_camelcase_word_spans(
    text: &str,
    start_byte: usize,
    end_byte: usize,
    spans: &mut Vec<(usize, usize)>,
) {
    let mut part_start = start_byte;
    let mut previous = None;
    let mut iterator = text[start_byte..end_byte].char_indices().peekable();
    while let Some((offset, character)) = iterator.next() {
        let byte_index = start_byte + offset;
        if let Some(previous_character) = previous {
            if should_split_camelcase_part(previous_character, character, iterator.peek()) {
                push_word_span(text, part_start, byte_index, spans);
                part_start = byte_index;
            }
        }
        previous = Some(character);
    }

    push_word_span(text, part_start, end_byte, spans);
}

fn should_split_camelcase_part(
    previous: char,
    current: char,
    next: Option<&(usize, char)>,
) -> bool {
    previous.is_lowercase()
        && current.is_uppercase()
        && next.is_some_and(|(_, next_char)| next_char.is_uppercase())
}

fn push_word_span(text: &str, start_byte: usize, end_byte: usize, spans: &mut Vec<(usize, usize)>) {
    let surface = &text[start_byte..end_byte];
    if surface
        .trim_matches(|character: char| !character.is_alphabetic())
        .is_empty()
    {
        return;
    }
    spans.push((start_byte, end_byte));
}

fn punctuation_symbol(text: &str) -> Option<&'static str> {
    text.chars().rev().find_map(|character| match character {
        '.' | '…' => Some("."),
        '!' => Some("!"),
        '?' => Some("?"),
        ',' => Some(","),
        ';' => Some(";"),
        ':' => Some(":"),
        _ => None,
    })
}

impl PhonemeSequence {
    #[allow(dead_code)]
    #[doc(hidden)]
    pub fn to_symbols_compatible(&self, config: &VoiceConfig) -> Result<Self> {
        let text_sequence = self.with_utterance_termination(config);
        validate_plan_sequence(&text_sequence)?;
        if text_sequence
            .symbols
            .iter()
            .all(|symbol| config.phoneme_id_map.contains_key(symbol))
        {
            return Ok(text_sequence);
        }

        text_sequence.to_espeak_compatible(config)
    }

    #[doc(hidden)]
    pub fn to_text_ids_compatible(&self, config: &VoiceConfig) -> Result<PhonemeIdSequence> {
        let text_sequence = self.with_utterance_termination(config);
        validate_plan_sequence(&text_sequence)?;
        if config_has_symbol_framing(config) {
            return text_sequence.to_framed_ids(config).or_else(|_| {
                text_sequence
                    .to_espeak_compatible(config)
                    .and_then(|sequence| sequence.to_framed_ids(config))
            });
        }

        text_sequence.to_ids(config).or_else(|_| {
            text_sequence
                .to_espeak_compatible(config)
                .and_then(|sequence| sequence.to_ids(config))
        })
    }

    fn to_ids(&self, config: &VoiceConfig) -> Result<PhonemeIdSequence> {
        let mut ids = Vec::new();
        for symbol in &self.symbols {
            extend_symbol_ids(&mut ids, symbol, config)?;
        }
        Ok(PhonemeIdSequence { ids })
    }

    fn to_framed_ids(&self, config: &VoiceConfig) -> Result<PhonemeIdSequence> {
        let mut ids = Vec::new();
        extend_symbol_ids(&mut ids, "^", config)?;
        extend_symbol_ids(&mut ids, "_", config)?;
        for symbol in &self.symbols {
            extend_symbol_ids(&mut ids, symbol, config)?;
            extend_symbol_ids(&mut ids, "_", config)?;
        }
        extend_symbol_ids(&mut ids, "$", config)?;
        Ok(PhonemeIdSequence { ids })
    }

    fn with_utterance_termination(&self, config: &VoiceConfig) -> Self {
        if self.symbols.is_empty() {
            return self.clone();
        }

        let mut terminated = self.clone();
        if let Some(last) = terminated.symbols.last_mut() {
            if is_terminal_symbol(last) {
                if can_encode_symbol(last, config) {
                    return terminated;
                }
                if let Some(symbol) = compatible_terminal_symbol(Some(last), config) {
                    *last = symbol.to_string();
                }
                return terminated;
            }
        }

        if let Some(symbol) = compatible_terminal_symbol(None, config) {
            terminated.symbols.push(symbol.to_string());
        }
        terminated
    }

    fn to_espeak_compatible(&self, config: &VoiceConfig) -> Result<Self> {
        let mut symbols = Vec::new();
        for symbol in &self.symbols {
            let expanded = expand_espeak_phoneme(symbol, config)
                .with_context(|| format!("unknown voice phoneme symbol `{symbol}`"))?;
            symbols.extend(expanded);
        }
        Ok(Self { symbols })
    }
}

#[cfg(feature = "onnx-tts")]
fn load_onnx_speech_session(
    model_path: &Path,
    execution_providers: &[ExecutionProviderDispatch],
) -> Result<Session> {
    let builder = Session::builder()
        .context("failed to create ONNX speech session builder")?
        .with_intra_threads(1)
        .map_err(|error| anyhow::anyhow!("failed to configure ONNX speech threads: {error}"))?
        .with_inter_threads(1)
        .map_err(|error| anyhow::anyhow!("failed to configure ONNX speech threads: {error}"))?
        .with_intra_op_spinning(false)
        .map_err(|error| anyhow::anyhow!("failed to configure ONNX speech spinning: {error}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|error| {
            anyhow::anyhow!("failed to configure ONNX speech optimization: {error}")
        })?;
    let mut builder = if execution_providers.is_empty() {
        builder
    } else {
        builder
            .with_execution_providers(execution_providers)
            .map_err(|error| {
                anyhow::anyhow!("failed to configure ONNX speech execution providers: {error}")
            })?
    };
    builder
        .commit_from_file(model_path)
        .with_context(|| format!("failed to load ONNX speech model {}", model_path.display()))
}

#[cfg(feature = "onnx-tts")]
impl OnnxSpeechBackend {
    pub fn load(model_path: impl AsRef<Path>, config: VoiceConfig) -> Result<Self> {
        Self::load_with_cuda(model_path, config, true)
    }

    pub fn load_cpu(model_path: impl AsRef<Path>, config: VoiceConfig) -> Result<Self> {
        Self::load_with_cuda(model_path, config, false)
    }

    fn load_with_cuda(
        model_path: impl AsRef<Path>,
        config: VoiceConfig,
        use_cuda: bool,
    ) -> Result<Self> {
        validate_config(&config)?;
        let model_path = model_path.as_ref().to_path_buf();
        ensure!(
            model_path.is_file(),
            "ONNX speech model file not found at {}",
            model_path.display()
        );
        initialize_ort_runtime()?;

        let session = if use_cuda {
            load_onnx_speech_session(
                &model_path,
                &[ort::ep::CUDA::default().build().fail_silently()],
            )
            .or_else(|_| load_onnx_speech_session(&model_path, &[]))?
        } else {
            load_onnx_speech_session(&model_path, &[])?
        };

        Ok(Self {
            config,
            model_path,
            session,
        })
    }

    pub fn sample_rate_hz(&self) -> u32 {
        self.config.sample_rate_hz
    }

    pub fn voice_config(&self) -> &VoiceConfig {
        &self.config
    }

    pub fn capabilities(&self) -> SpeechModelCapabilities {
        SpeechModelCapabilities::onnx_voice(&self.config)
    }

    #[allow(dead_code)]
    pub fn synthesize_plan(&mut self, plan: &UtterancePlan) -> Result<SynthesisOutput> {
        let mut pcm_mono_f32 = Vec::new();
        self.synthesize_plan_streaming(plan, &mut |chunk: AudioChunk| {
            pcm_mono_f32.extend(chunk.pcm_mono_f32);
            Ok(())
        })?;

        Ok(SynthesisOutput {
            sample_rate_hz: self.config.sample_rate_hz,
            pcm_mono_f32,
        })
    }

    pub fn synthesize_plan_streaming(
        &mut self,
        plan: &UtterancePlan,
        sink: &mut dyn AudioSink,
    ) -> Result<()> {
        self.synthesize_plan_streaming_with_options(plan, &SynthesisOptions::default(), sink)
    }

    pub fn synthesize_plan_streaming_with_options(
        &mut self,
        plan: &UtterancePlan,
        options: &SynthesisOptions,
        sink: &mut dyn AudioSink,
    ) -> Result<()> {
        validate_onnx_options(options)?;
        let chunks = synthesis_chunks_from_plan(plan)?;
        let chunk_count = chunks.len();
        if chunk_count == 0 {
            sink.emit(AudioChunk {
                chunk_index: 0,
                is_final: true,
                pause_after_ms: 0,
                sample_rate_hz: self.config.sample_rate_hz,
                pcm_mono_f32: Vec::new(),
            })?;
            return Ok(());
        }
        for (chunk_index, chunk) in chunks.into_iter().enumerate() {
            let ids = chunk
                .sequence
                .to_text_ids_compatible(&self.config)
                .context("failed to map Mortar speech plan to voice model phoneme IDs")?;
            let mut output = self
                .synthesize_ids_with_context(&ids, options, plan.speaker.as_ref())?
                .pcm_mono_f32;
            output.extend(std::iter::repeat(0.0).take(pause_sample_count(
                self.config.sample_rate_hz,
                chunk.pause_after_ms,
            )));
            sink.emit(AudioChunk {
                chunk_index,
                is_final: chunk_index + 1 == chunk_count,
                // The pause is already materialized in pcm_mono_f32.
                pause_after_ms: 0,
                sample_rate_hz: self.config.sample_rate_hz,
                pcm_mono_f32: output,
            })?;
        }

        Ok(())
    }

    pub fn synthesize_ids(&mut self, ids: &PhonemeIdSequence) -> Result<SynthesisOutput> {
        self.synthesize_ids_with_options(ids, &SynthesisOptions::default())
    }

    pub fn synthesize_ids_with_options(
        &mut self,
        ids: &PhonemeIdSequence,
        options: &SynthesisOptions,
    ) -> Result<SynthesisOutput> {
        self.synthesize_ids_with_context(ids, options, None)
    }

    fn synthesize_ids_with_context(
        &mut self,
        ids: &PhonemeIdSequence,
        options: &SynthesisOptions,
        speaker: Option<&SpeakerId>,
    ) -> Result<SynthesisOutput> {
        validate_onnx_options(options)?;
        ensure!(
            !ids.ids.is_empty(),
            "phoneme ID sequence cannot be empty for ONNX synthesis"
        );

        let input_specs = self
            .session
            .inputs()
            .iter()
            .map(|input| OnnxTensorSpec {
                name: input.name().to_string(),
                tensor_type: input.dtype().tensor_type(),
            })
            .collect::<Vec<_>>();
        let output_specs = self
            .session
            .outputs()
            .iter()
            .map(|output| OnnxTensorSpec {
                name: output.name().to_string(),
                tensor_type: output.dtype().tensor_type(),
            })
            .collect::<Vec<_>>();
        let contract = resolve_inference_contract(
            &input_specs,
            &output_specs,
            &self.config,
            &self.model_path,
        )?;
        let ids_len = i64::try_from(ids.ids.len()).context("phoneme ID sequence is too long")?;
        let scales = inference_scales(&self.config, options);
        let mut inputs = Vec::with_capacity(6);

        inputs.push((
            contract.id_input.clone(),
            Tensor::from_array((vec![1_i64, ids_len], ids.ids.clone()))
                .context("failed to build ONNX speech ID tensor")?
                .upcast(),
        ));
        inputs.push((
            contract.id_lengths_input.clone(),
            Tensor::from_array((vec![1_i64], vec![ids_len]))
                .context("failed to build ONNX speech length tensor")?
                .upcast(),
        ));
        if let Some(name) = &contract.scales_input {
            inputs.push((
                name.clone(),
                Tensor::from_array((vec![3_i64], scales.to_vec()))
                    .with_context(|| format!("failed to build ONNX speech `{name}` tensor"))?
                    .upcast(),
            ));
        }
        if let Some(name) = &contract.noise_scale_input {
            inputs.push((
                name.clone(),
                Tensor::from_array((vec![1_i64], vec![scales[0]]))
                    .with_context(|| format!("failed to build ONNX speech `{name}` tensor"))?
                    .upcast(),
            ));
        }
        if let Some(name) = &contract.length_scale_input {
            inputs.push((
                name.clone(),
                Tensor::from_array((vec![1_i64], vec![scales[1]]))
                    .with_context(|| format!("failed to build ONNX speech `{name}` tensor"))?
                    .upcast(),
            ));
        }
        if let Some(name) = &contract.noise_w_input {
            inputs.push((
                name.clone(),
                Tensor::from_array((vec![1_i64], vec![scales[2]]))
                    .with_context(|| format!("failed to build ONNX speech `{name}` tensor"))?
                    .upcast(),
            ));
        }
        if let Some(name) = &contract.speaker_input {
            let speaker_id = resolve_speaker_id(&self.config, speaker, options.speaker_id, true)?
                .context("speaker input was present but no speaker ID was resolved")?;
            inputs.push((
                name.clone(),
                Tensor::from_array((vec![1_i64], vec![i64::from(speaker_id)]))
                    .with_context(|| format!("failed to build ONNX speech `{name}` tensor"))?
                    .upcast(),
            ));
        } else {
            resolve_speaker_id(&self.config, speaker, options.speaker_id, false)?;
        }

        let outputs = self.session.run(inputs).with_context(|| {
            format!(
                "failed to run ONNX speech inference for model {}",
                self.model_path.display()
            )
        })?;
        let output = outputs
            .get(contract.output_audio.as_str())
            .with_context(|| {
                format!(
                    "ONNX speech inference did not return `{}`",
                    contract.output_audio
                )
            })?;
        let output = output
            .downcast_ref::<DynTensorValueType>()
            .with_context(|| {
                format!(
                    "ONNX speech output `{}` is not a tensor",
                    contract.output_audio
                )
            })?;
        let (_, samples) = output.try_extract_tensor::<f32>().with_context(|| {
            format!("ONNX speech output `{}` is not f32", contract.output_audio)
        })?;
        ensure!(
            !samples.is_empty(),
            "ONNX speech inference returned an empty waveform output"
        );

        Ok(SynthesisOutput {
            sample_rate_hz: self.config.sample_rate_hz,
            pcm_mono_f32: samples.to_vec(),
        })
    }
}

#[cfg(feature = "onnx-tts")]
impl SpeechSynthesisEngine for OnnxSpeechBackend {
    fn capabilities(&self) -> SpeechModelCapabilities {
        self.capabilities()
    }

    fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz()
    }

    fn synthesize_plan_streaming(
        &mut self,
        request: &SpeechSynthesisRequest,
        sink: &mut dyn AudioSink,
    ) -> Result<()> {
        self.synthesize_plan_streaming_with_options(&request.plan, &request.options, sink)
    }
}

#[cfg(feature = "onnx-tts")]
fn validate_onnx_options(options: &SynthesisOptions) -> Result<()> {
    ensure!(
        options
            .length_scale
            .is_none_or(|value| value.is_finite() && value > 0.0),
        "length scale must be finite and positive"
    );
    ensure!(
        options
            .noise_scale
            .is_none_or(|value| value.is_finite() && value >= 0.0),
        "noise scale must be finite and non-negative"
    );
    ensure!(
        options
            .noise_w
            .is_none_or(|value| value.is_finite() && value >= 0.0),
        "duration noise scale must be finite and non-negative"
    );
    Ok(())
}

#[cfg(not(feature = "onnx-tts"))]
#[derive(Debug)]
pub struct OnnxSpeechBackend;

#[cfg(not(feature = "onnx-tts"))]
impl OnnxSpeechBackend {
    pub fn load(_model_path: impl AsRef<Path>, _config: VoiceConfig) -> Result<Self> {
        bail!("ONNX speech synthesis requires building with the `onnx-tts` feature")
    }

    pub fn load_cpu(_model_path: impl AsRef<Path>, _config: VoiceConfig) -> Result<Self> {
        bail!("ONNX speech synthesis requires building with the `onnx-tts` feature")
    }

    pub fn sample_rate_hz(&self) -> u32 {
        0
    }

    pub fn voice_config(&self) -> &VoiceConfig {
        unreachable!("ONNX speech synthesis requires building with the `onnx-tts` feature")
    }

    pub fn capabilities(&self) -> SpeechModelCapabilities {
        unreachable!("ONNX speech synthesis requires building with the `onnx-tts` feature")
    }

    #[allow(dead_code)]
    pub fn synthesize_plan(&mut self, _plan: &UtterancePlan) -> Result<SynthesisOutput> {
        bail!("ONNX speech synthesis requires building with the `onnx-tts` feature")
    }

    pub fn synthesize_plan_streaming(
        &mut self,
        _plan: &UtterancePlan,
        _sink: &mut dyn AudioSink,
    ) -> Result<()> {
        bail!("ONNX speech synthesis requires building with the `onnx-tts` feature")
    }

    pub fn synthesize_plan_streaming_with_options(
        &mut self,
        _plan: &UtterancePlan,
        _options: &SynthesisOptions,
        _sink: &mut dyn AudioSink,
    ) -> Result<()> {
        bail!("ONNX speech synthesis requires building with the `onnx-tts` feature")
    }

    pub fn synthesize_ids(&mut self, _ids: &PhonemeIdSequence) -> Result<SynthesisOutput> {
        bail!("ONNX speech synthesis requires building with the `onnx-tts` feature")
    }

    pub fn synthesize_ids_with_options(
        &mut self,
        _ids: &PhonemeIdSequence,
        _options: &SynthesisOptions,
    ) -> Result<SynthesisOutput> {
        bail!("ONNX speech synthesis requires building with the `onnx-tts` feature")
    }
}

#[cfg(not(feature = "onnx-tts"))]
impl SpeechSynthesisEngine for OnnxSpeechBackend {
    fn capabilities(&self) -> SpeechModelCapabilities {
        self.capabilities()
    }

    fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz()
    }

    fn synthesize_plan_streaming(
        &mut self,
        _request: &SpeechSynthesisRequest,
        _sink: &mut dyn AudioSink,
    ) -> Result<()> {
        bail!("ONNX speech synthesis requires building with the `onnx-tts` feature")
    }
}

fn push_symbol(symbols: &mut Vec<String>, symbol: &str) {
    if symbol == " " && symbols.last().is_some_and(|last| last == " ") {
        return;
    }
    symbols.push(symbol.to_string());
}

#[cfg(any(feature = "onnx-tts", test))]
#[cfg(feature = "onnx-tts")]
fn pause_sample_count(sample_rate_hz: u32, pause_ms: u32) -> usize {
    ((sample_rate_hz as u128 * pause_ms as u128) / 1000) as usize
}

fn push_boundary_symbols(symbols: &mut Vec<String>, symbol: &str) {
    push_symbol(symbols, symbol);
    if is_clause_pause_symbol(symbol) {
        push_symbol(symbols, " ");
    }
}

fn is_clause_pause_symbol(symbol: &str) -> bool {
    matches!(symbol, "," | ";" | ":")
}

fn next_phone_is_epenthetic_linker(tokens: &[PhoneToken]) -> bool {
    for token in tokens {
        if let Spec::Known(id) = &token.phone {
            if id.as_str().starts_with("boundary.") {
                continue;
            }
        }
        return is_epenthetic_phone(token);
    }
    false
}

fn is_epenthetic_phone(token: &PhoneToken) -> bool {
    token.provenance.method.contains("epenthesis rule")
}

fn find_value<'a>(root: &'a Value, paths: &[&[&str]]) -> Option<&'a Value> {
    paths.iter().find_map(|path| {
        let mut current = root;
        for segment in *path {
            current = current.get(*segment)?;
        }
        Some(current)
    })
}

fn parse_required_u32(root: &Value, paths: &[&[&str]], field: &'static str) -> Result<u32> {
    let value = find_value(root, paths)
        .with_context(|| format!("missing required voice config field `{field}`"))?;
    parse_u32(value, field)
}

fn parse_optional_u32(root: &Value, paths: &[&[&str]], field: &'static str) -> Result<Option<u32>> {
    find_value(root, paths)
        .map(|value| parse_u32(value, field))
        .transpose()
}

fn parse_u32(value: &Value, field: &'static str) -> Result<u32> {
    let number = value
        .as_u64()
        .with_context(|| format!("invalid voice config field `{field}`: expected integer"))?;
    u32::try_from(number)
        .with_context(|| format!("invalid voice config field `{field}`: exceeds u32"))
}

fn parse_optional_f32(root: &Value, paths: &[&[&str]], field: &'static str) -> Result<Option<f32>> {
    find_value(root, paths)
        .map(|value| parse_f32(value, field))
        .transpose()
}

fn parse_f32(value: &Value, field: &'static str) -> Result<f32> {
    let number = value
        .as_f64()
        .with_context(|| format!("invalid voice config field `{field}`: expected number"))?;
    ensure!(
        number.is_finite() && number >= f32::MIN as f64 && number <= f32::MAX as f64,
        "invalid voice config field `{field}`: value is out of f32 range"
    );
    Ok(number as f32)
}

fn parse_phoneme_id_map(value: &Value) -> Result<HashMap<String, Vec<i64>>> {
    let entries = value
        .as_object()
        .context("invalid voice config field `phoneme_id_map`: expected object")?;
    let mut map = HashMap::with_capacity(entries.len());
    for (symbol, ids) in entries {
        let ids = match ids {
            Value::Array(values) => values.iter().map(parse_i64).collect::<Result<Vec<_>>>()?,
            _ => vec![parse_i64(ids)?],
        };
        map.insert(symbol.clone(), ids);
    }
    Ok(map)
}

fn parse_speaker_id_map(value: &Value) -> Result<HashMap<String, u32>> {
    let entries = value
        .as_object()
        .context("invalid voice config field `speaker_id_map`: expected object")?;
    let mut map = HashMap::with_capacity(entries.len());
    for (speaker, id) in entries {
        map.insert(speaker.clone(), parse_u32(id, "speaker_id_map")?);
    }
    Ok(map)
}

fn parse_i64(value: &Value) -> Result<i64> {
    value
        .as_i64()
        .context("invalid voice config field `phoneme_id_map`: expected integer")
}

fn extend_symbol_ids(ids: &mut Vec<i64>, symbol: &str, config: &VoiceConfig) -> Result<()> {
    let mapped = config
        .phoneme_id_map
        .get(symbol)
        .with_context(|| format!("unknown voice phoneme symbol `{symbol}`"))?;
    ids.extend(mapped);
    Ok(())
}

fn config_has_symbol_framing(config: &VoiceConfig) -> bool {
    ["^", "_", "$"]
        .iter()
        .all(|symbol| config.phoneme_id_map.contains_key(*symbol))
}

fn is_terminal_symbol(symbol: &str) -> bool {
    matches!(symbol, "|" | "." | "!" | "?" | "$")
}

fn can_encode_symbol(symbol: &str, config: &VoiceConfig) -> bool {
    config.phoneme_id_map.contains_key(symbol) || expand_espeak_phoneme(symbol, config).is_some()
}

fn validate_plan_sequence(sequence: &PhonemeSequence) -> Result<()> {
    for symbol in &sequence.symbols {
        ensure!(
            is_plan_symbol(symbol),
            "unsupported pre-compat voice symbol `{symbol}`; expected ARPAbet, space, or punctuation"
        );
    }
    Ok(())
}

fn is_plan_symbol(symbol: &str) -> bool {
    if matches!(symbol, " " | "|" | "." | "!" | "?" | "," | ";" | ":") {
        return true;
    }
    let (base, stress) = split_arpabet_stress(symbol);
    if stress.is_some() && !is_arpabet_vowel(base) {
        return false;
    }
    is_arpabet_symbol(base)
}

fn split_arpabet_stress(symbol: &str) -> (&str, Option<char>) {
    match symbol.chars().last() {
        Some(stress @ ('0' | '1' | '2')) => (&symbol[..symbol.len() - 1], Some(stress)),
        _ => (symbol, None),
    }
}

fn compatible_terminal_symbol<'a>(
    requested: Option<&'a str>,
    config: &VoiceConfig,
) -> Option<&'a str> {
    if let Some(symbol) = requested {
        if can_encode_symbol(symbol, config) {
            return Some(symbol);
        }
    }
    if can_encode_symbol(".", config) {
        return Some(".");
    }
    if can_encode_symbol("|", config) {
        return Some("|");
    }
    None
}

fn expand_espeak_phoneme(symbol: &str, config: &VoiceConfig) -> Option<Vec<String>> {
    if symbol == " " {
        return if config.phoneme_id_map.contains_key(" ") {
            Some(vec![" ".to_string()])
        } else {
            Some(Vec::new())
        };
    }

    let stress_marker = match symbol.chars().last() {
        Some('1') => Some("ˈ"),
        Some('2') => Some("ˌ"),
        _ => None,
    };
    let base_symbol = symbol
        .strip_suffix(['0', '1', '2'])
        .filter(|base| is_arpabet_vowel(base))
        .unwrap_or(symbol);

    if base_symbol != symbol && config.phoneme_id_map.contains_key(base_symbol) {
        let mut output = Vec::new();
        if let Some(marker) = stress_marker {
            if config.phoneme_id_map.contains_key(marker) {
                output.push(marker.to_string());
            }
        }
        output.push(base_symbol.to_string());
        return Some(output);
    }

    if config.phoneme_id_map.contains_key(symbol) {
        return Some(vec![symbol.to_string()]);
    }

    let expanded = match (symbol, base_symbol) {
        ("AH0", _) => &["ə"][..],
        ("AH1" | "AH2", _) => &["ʌ"],
        (_, "AA") => &["ɑ"],
        (_, "AH") => &["ə"],
        (_, "AY") => &["a", "ɪ"],
        (_, "AE") => &["æ"],
        (_, "AO") => &["ɔ"],
        (_, "AW") => &["a", "ʊ"],
        (_, "B") => &["b"],
        (_, "CH") => &["t", "ʃ"],
        (_, "D") => &["d"],
        (_, "DH") => &["ð"],
        (_, "DX") => &["ɾ"],
        (_, "EH") => &["ɛ"],
        (_, "ER") => &["ɚ"],
        (_, "EY") => &["e", "ɪ"],
        (_, "F") => &["f"],
        (_, "G") => &["ɡ"],
        (_, "HH") => &["h"],
        (_, "IH") => &["ɪ"],
        (_, "IY") => &["i"],
        (_, "JH") => &["d", "ʒ"],
        (_, "K") => &["k"],
        (_, "L") => &["l"],
        (_, "M") => &["m"],
        (_, "N") => &["n"],
        (_, "NG") => &["ŋ"],
        (_, "OW") => &["o", "ʊ"],
        (_, "OY") => &["ɔ", "ɪ"],
        (_, "P") => &["p"],
        (_, "R") => &["ɹ"],
        (_, "S") => &["s"],
        (_, "SH") => &["ʃ"],
        (_, "T") => &["t"],
        (_, "TH") => &["θ"],
        (_, "TS") => &["t", "s"],
        (_, "UH") => &["ʊ"],
        (_, "UW") => &["u"],
        (_, "V") => &["v"],
        (_, "W") => &["w"],
        (_, "Y") => &["j"],
        (_, "Z") => &["z"],
        (_, "ZH") => &["ʒ"],
        (_, "|") => &["."],
        _ => return None,
    };

    let mut expanded = expanded
        .iter()
        .map(|sym| (*sym).to_string())
        .collect::<Vec<_>>();
    if config.phoneme_id_map.contains_key("ː") && matches!(base_symbol, "AA" | "AO" | "IY" | "UW")
    {
        expanded.push("ː".to_string());
    }
    if !expanded
        .iter()
        .all(|sym| config.phoneme_id_map.contains_key(sym))
    {
        return None;
    }

    let mut output = Vec::new();
    if let Some(marker) = stress_marker {
        if config.phoneme_id_map.contains_key(marker) {
            output.push(marker.to_string());
        }
    }
    output.extend(expanded);
    Some(output)
}

fn feature_category<'a>(
    features: &'a speaking::FeatureBundle,
    feature_id: &str,
) -> Option<&'a str> {
    let value = features.values.get(&FeatureId(feature_id.into()))?;
    match value {
        Spec::Known(FeatureValue::Category(val)) | Spec::Known(FeatureValue::Text(val)) => {
            Some(val.as_str())
        }
        _ => None,
    }
}

fn stress_digit(stress: &str) -> Option<&'static str> {
    match stress {
        "unstressed" => Some("0"),
        "primary" => Some("1"),
        "secondary" => Some("2"),
        _ => None,
    }
}

fn is_arpabet_symbol(symbol: &str) -> bool {
    is_arpabet_vowel(symbol)
        || matches!(
            symbol,
            "B" | "CH"
                | "D"
                | "DH"
                | "DX"
                | "F"
                | "G"
                | "HH"
                | "JH"
                | "K"
                | "L"
                | "M"
                | "N"
                | "NG"
                | "P"
                | "R"
                | "S"
                | "SH"
                | "T"
                | "TH"
                | "TS"
                | "V"
                | "W"
                | "Y"
                | "Z"
                | "ZH"
        )
}

fn is_arpabet_vowel(symbol: &str) -> bool {
    matches!(
        symbol,
        "AA" | "AE"
            | "AH"
            | "AO"
            | "AW"
            | "AY"
            | "EH"
            | "ER"
            | "EY"
            | "IH"
            | "IY"
            | "OW"
            | "OY"
            | "UH"
            | "UW"
    )
}

#[cfg(feature = "onnx-tts")]
fn validate_config(config: &VoiceConfig) -> Result<()> {
    ensure!(
        config.sample_rate_hz > 0,
        "missing required voice config field `audio.sample_rate`"
    );
    ensure!(
        !config.phoneme_id_map.is_empty(),
        "missing required voice config field `phoneme_id_map`"
    );
    if let Some(num_speakers) = config.num_speakers {
        ensure!(
            num_speakers > 0,
            "invalid voice config field `num_speakers`: expected a value greater than zero"
        );
    }
    Ok(())
}

#[cfg(feature = "onnx-tts")]
fn resolve_inference_contract(
    input_specs: &[OnnxTensorSpec],
    output_specs: &[OnnxTensorSpec],
    _config: &VoiceConfig,
    model_path: &Path,
) -> Result<OnnxInferenceContract> {
    ensure!(
        !input_specs.is_empty(),
        "ONNX speech model `{}` exposes no inputs",
        model_path.display()
    );
    ensure!(
        !output_specs.is_empty(),
        "ONNX speech model `{}` exposes no outputs",
        model_path.display()
    );

    let id_input = resolve_required_tensor_input(
        input_specs,
        &["input", "input_ids", "phoneme_ids", "ids"],
        TensorElementType::Int64,
        "phoneme ID input tensor",
        model_path,
    )?;
    let id_lengths_input = resolve_required_tensor_input(
        input_specs,
        &["input_lengths", "lengths", "input_lengths_tensor"],
        TensorElementType::Int64,
        "phoneme length input tensor",
        model_path,
    )?;
    let scales_input =
        resolve_optional_tensor_input(input_specs, &["scales"], TensorElementType::Float32)?;
    let noise_scale_input =
        resolve_optional_tensor_input(input_specs, &["noise_scale"], TensorElementType::Float32)?;
    let length_scale_input =
        resolve_optional_tensor_input(input_specs, &["length_scale"], TensorElementType::Float32)?;
    let noise_w_input =
        resolve_optional_tensor_input(input_specs, &["noise_w"], TensorElementType::Float32)?;
    let speaker_input = resolve_optional_tensor_input(
        input_specs,
        &["sid", "speaker_id"],
        TensorElementType::Int64,
    )?;

    let supported = [
        Some(id_input.clone()),
        Some(id_lengths_input.clone()),
        scales_input.clone(),
        noise_scale_input.clone(),
        length_scale_input.clone(),
        noise_w_input.clone(),
        speaker_input.clone(),
    ];
    for input in input_specs {
        if !supported.iter().flatten().any(|name| name == &input.name) {
            bail!(
                "unsupported ONNX speech input `{}` for model `{}`",
                input.name,
                model_path.display()
            );
        }
    }

    let output_audio = resolve_required_tensor_output(
        output_specs,
        &["output", "audio", "waveform"],
        TensorElementType::Float32,
        "audio output tensor",
        model_path,
    )?;
    if output_specs.iter().any(|spec| {
        spec.name != output_audio && spec.tensor_type == Some(TensorElementType::Float32)
    }) {
        bail!(
            "unsupported ONNX speech model `{}` contract: multiple f32 outputs detected",
            model_path.display()
        );
    }

    Ok(OnnxInferenceContract {
        id_input,
        id_lengths_input,
        scales_input,
        noise_scale_input,
        length_scale_input,
        noise_w_input,
        speaker_input,
        output_audio,
    })
}

#[cfg(feature = "onnx-tts")]
fn resolve_required_tensor_input(
    inputs: &[OnnxTensorSpec],
    aliases: &[&str],
    expected_type: TensorElementType,
    label: &str,
    model_path: &Path,
) -> Result<String> {
    let input = resolve_tensor_by_alias(inputs, aliases).with_context(|| {
        format!(
            "unsupported ONNX speech model contract for `{}`: missing {}",
            model_path.display(),
            label
        )
    })?;
    ensure!(
        input.tensor_type == Some(expected_type),
        "unsupported ONNX speech model contract for `{}`: input `{}` expected type {:?}, got {:?}",
        model_path.display(),
        input.name,
        expected_type,
        input.tensor_type
    );
    Ok(input.name.clone())
}

#[cfg(feature = "onnx-tts")]
fn resolve_optional_tensor_input(
    inputs: &[OnnxTensorSpec],
    aliases: &[&str],
    expected_type: TensorElementType,
) -> Result<Option<String>> {
    let Some(input) = resolve_tensor_by_alias(inputs, aliases) else {
        return Ok(None);
    };
    ensure!(
        input.tensor_type == Some(expected_type),
        "unsupported ONNX speech model contract: input `{}` expected type {:?}, got {:?}",
        input.name,
        expected_type,
        input.tensor_type
    );
    Ok(Some(input.name.clone()))
}

#[cfg(feature = "onnx-tts")]
fn resolve_required_tensor_output(
    outputs: &[OnnxTensorSpec],
    aliases: &[&str],
    expected_type: TensorElementType,
    label: &str,
    model_path: &Path,
) -> Result<String> {
    let output = resolve_tensor_by_alias(outputs, aliases)
        .or_else(|| {
            outputs
                .iter()
                .find(|spec| spec.tensor_type == Some(expected_type))
        })
        .with_context(|| {
            format!(
                "unsupported ONNX speech model contract for `{}`: missing {}",
                model_path.display(),
                label
            )
        })?;
    ensure!(
        output.tensor_type == Some(expected_type),
        "unsupported ONNX speech model contract for `{}`: output `{}` expected type {:?}, got {:?}",
        model_path.display(),
        output.name,
        expected_type,
        output.tensor_type
    );
    Ok(output.name.clone())
}

#[cfg(feature = "onnx-tts")]
fn resolve_tensor_by_alias<'a>(
    specs: &'a [OnnxTensorSpec],
    aliases: &[&str],
) -> Option<&'a OnnxTensorSpec> {
    aliases
        .iter()
        .find_map(|alias| specs.iter().find(|spec| spec.name == *alias))
}

#[cfg(feature = "onnx-tts")]
fn inference_scales(config: &VoiceConfig, options: &SynthesisOptions) -> [f32; 3] {
    [
        options.noise_scale.or(config.noise_scale).unwrap_or(0.667),
        options.length_scale.or(config.length_scale).unwrap_or(1.0),
        options.noise_w.or(config.noise_w).unwrap_or(0.8),
    ]
}

#[cfg(any(feature = "onnx-tts", test))]
fn resolve_speaker_id(
    config: &VoiceConfig,
    speaker: Option<&SpeakerId>,
    direct_speaker_id: Option<u32>,
    model_accepts_speaker: bool,
) -> Result<Option<u32>> {
    if !model_accepts_speaker {
        ensure!(
            speaker.is_none() && direct_speaker_id.is_none(),
            "speaker selection was provided but this ONNX speech model has no speaker input"
        );
        return Ok(None);
    }

    ensure!(
        speaker.is_none() || direct_speaker_id.is_none(),
        "speaker identity and direct speaker ID were both provided"
    );

    let speaker_id = if let Some(speaker_id) = direct_speaker_id {
        speaker_id
    } else if let Some(speaker) = speaker {
        let name = speaker.0.as_str();
        *config.speaker_id_map.get(name).with_context(|| {
            format!(
                "unknown speaker `{name}`; available speakers: {}",
                available_speakers_message(config)
            )
        })?
    } else {
        if config.is_multi_speaker() {
            bail!(
                "speaker selection is required for this multi-speaker voice model; available speakers: {}",
                available_speakers_message(config)
            );
        }
        return Ok(Some(0));
    };

    if let Some(num_speakers) = config.num_speakers {
        ensure!(
            speaker_id < num_speakers,
            "speaker id {speaker_id} is out of range for {num_speakers} speakers"
        );
    } else if !config.speaker_id_map.is_empty() {
        ensure!(
            config.speaker_id_map.values().any(|id| *id == speaker_id),
            "speaker id {speaker_id} is not declared by this voice model; available speakers: {}",
            available_speakers_message(config)
        );
    }

    Ok(Some(speaker_id))
}

#[cfg(any(feature = "onnx-tts", test))]
fn available_speakers_message(config: &VoiceConfig) -> String {
    let names = config.available_speaker_names();
    if names.is_empty() {
        return match config.num_speakers {
            Some(count) => format!("numeric IDs 0..{}", count.saturating_sub(1)),
            None => "none declared".to_string(),
        };
    }
    names.join(", ")
}

#[cfg(feature = "onnx-tts")]
fn initialize_ort_runtime() -> Result<()> {
    ort::init().commit();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    const RYAN_LIKE_CONFIG_JSON: &str = r#"
    {
      "audio": { "sample_rate": 22050 },
      "phoneme_id_map": {
        "^": [1], "_": [0], "$": [2],
        " ": [3], ".": [4], "?": [5], "!": [6], ",": [7],
        "AA": [10], "AE": [11], "AH0": [12], "AH1": [13], "AO": [14],
        "AW": [15], "AY": [16], "B": [17], "CH": [18], "D": [19],
        "DH": [20], "EH": [21], "ER0": [22], "ER1": [23], "EY": [24],
        "F": [25], "G": [26], "HH": [27], "IH": [28], "IY": [29],
        "JH": [30], "K": [31], "L": [32], "M": [33], "N": [34],
        "NG": [35], "OW": [36], "OY": [37], "P": [38], "R": [39],
        "S": [40], "SH": [41], "T": [42], "TH": [43], "UH": [44],
        "UW": [45], "V": [46], "W": [47], "Y": [48], "Z": [49],
        "ZH": [50]
      },
      "inference": { "noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8 }
    }
    "#;

    #[test]
    fn plain_text_lowers_to_non_empty_id_sequence() {
        let config = VoiceConfig::from_json_str(RYAN_LIKE_CONFIG_JSON).expect("config");
        let ids = phoneme_ids_from_text("hello world", "en-US", &config).expect("ids");
        assert!(!ids.ids.is_empty());
    }

    #[test]
    fn wiktionary_fallback_maps_supported_varieties_to_languages() {
        for (variety, expected_language) in [
            ("en-US", "eng"),
            ("en-US.GenAm", "eng"),
            ("en-GB-RP", "eng"),
            ("fr-FR-Standard", "fra"),
            ("fra", "fra"),
            ("de-DE-Standard", "deu"),
            ("es-ES-Castilian", "spa"),
            ("es-419", "spa"),
            ("el-GR-Standard", "ell"),
            ("grc-Attic", "grc"),
            ("grc-Koine", "grc"),
            ("la-Classical", "lat"),
            ("la-Ecclesiastical", "lat"),
            ("sa-Deva-Standard", "san"),
            ("eo", "epo"),
            ("epo", "epo"),
        ] {
            assert_eq!(
                wiktionary_language_for_variety(variety),
                Some(expected_language),
                "{variety}"
            );
        }
        assert_eq!(wiktionary_language_for_variety("not-a-variety"), None);
    }

    #[test]
    fn unknown_french_word_reaches_french_wiktionary_model_input() -> Result<()> {
        static RECORDED_SOURCE: OnceLock<Mutex<Option<String>>> = OnceLock::new();

        fn recording_fallback(word: &str, variety: &str) -> Option<String> {
            let source = wiktionary_fallback_source(word, variety).ok()?;
            if variety == "fr-FR-Standard" {
                if let Ok(mut recorded) = RECORDED_SOURCE.get_or_init(|| Mutex::new(None)).lock() {
                    *recorded = Some(source);
                }
            }
            Some("bɔ̃ʒuʁ".into())
        }

        struct RestoreDefaultFallback;
        impl Drop for RestoreDefaultFallback {
            fn drop(&mut self) {
                speaking::set_unknown_pronunciation_fallback(Some(
                    wiktionary_unknown_pronunciation,
                ));
            }
        }

        install_default_unknown_pronunciation_fallback();
        let _restore = RestoreDefaultFallback;
        speaking::set_unknown_pronunciation_fallback(Some(recording_fallback));

        let plan = utterance_plan_from_text(SpeechRequest {
            text: "zzézz".into(),
            variety: "fr-FR-Standard".into(),
        })?;

        let source = RECORDED_SOURCE
            .get()
            .and_then(|source| source.lock().ok().and_then(|source| source.clone()))
            .expect("French fallback should receive the unknown word");
        assert!(source.contains("<lang:fra>"), "{source}");
        assert!(source.ends_with("zzézz"), "{source}");
        assert!(
            plan.intended_phonemes
                .iter()
                .any(|token| token.provenance.source == EvidenceSource::Inference),
            "the plan should preserve inference provenance"
        );
        Ok(())
    }

    #[test]
    fn wiktionary_default_fallback_pronounces_netherwick() -> Result<()> {
        let model_dir = default_wiktionary_fallback_model_dir();
        if !model_dir.join("model.bin").is_file() {
            eprintln!(
                "skipping Wiktionary fallback smoke test: missing {}",
                model_dir.display()
            );
            return Ok(());
        }

        let plan = utterance_plan_from_text(SpeechRequest {
            text: "Netherwick".to_string(),
            variety: "en-US".to_string(),
        })?;
        assert!(
            !plan.intended_phonemes.is_empty(),
            "Netherwick should receive a fallback pronunciation"
        );
        assert!(
            plan.intended_phonemes
                .iter()
                .any(|token| token.provenance.source == EvidenceSource::Inference),
            "Netherwick should be pronounced by the Wiktionary inference fallback"
        );
        Ok(())
    }

    #[test]
    fn voice_config_loads_ryan_config_json_shape() {
        let config = VoiceConfig::from_json_str(RYAN_LIKE_CONFIG_JSON).expect("config");
        assert_eq!(config.sample_rate_hz, 22_050);
        assert!(config.phoneme_id_map.contains_key("HH"));
        assert_eq!(config.noise_scale, Some(0.667));
    }

    #[test]
    fn runnable_default_voice_tracks_coqui_ljspeech_default() {
        assert_eq!(
            DEFAULT_TTS_CATALOG_MODEL,
            "tts_models/en/ljspeech/tacotron2-DDC"
        );
        assert_eq!(
            DEFAULT_VOCODER_CATALOG_MODEL,
            "vocoder_models/en/ljspeech/hifigan_v2"
        );
        assert_eq!(
            default_voice_model_path(default_voice_model()),
            default_voice_model_dir().join("en_US-ljspeech-high.onnx")
        );
    }

    #[test]
    fn onnx_speech_path_voice_errors_clearly_when_files_are_missing() {
        let missing_dir =
            std::env::temp_dir().join(format!("tongues-tts-missing-{}", std::process::id()));
        let error = OnnxSpeech::load(VoiceModel::Path {
            model: missing_dir.join("missing.onnx"),
            config: missing_dir.join("missing.onnx.json"),
        })
        .expect_err("missing files should fail");
        let message = format!("{error:#}");
        assert!(
            message.contains("voice config file not found")
                || message.contains("ONNX speech model file not found"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn single_speaker_model_accepts_omitted_speaker() {
        let config = VoiceConfig::from_json_str(RYAN_LIKE_CONFIG_JSON).expect("config");
        assert_eq!(
            resolve_speaker_id(&config, None, None, true).unwrap(),
            Some(0)
        );
    }

    #[test]
    fn named_multi_speaker_model_resolves_vctk_style_name() {
        let config = multi_speaker_config();
        assert_eq!(
            config.available_speaker_names(),
            vec!["p225".to_string(), "p226".to_string()]
        );
        let capabilities = SpeechModelCapabilities::onnx_voice(&config);
        assert_eq!(capabilities.family, SpeechModelFamily::EndToEndSpeech);
        assert!(capabilities.supports_named_speakers);
        assert!(capabilities.integrated_vocoder);
        assert_eq!(
            resolve_speaker_id(&config, Some(&SpeakerId("p226".into())), None, true).unwrap(),
            Some(1)
        );
    }

    #[test]
    fn unknown_speaker_name_reports_available_speakers() {
        let config = multi_speaker_config();
        let error = resolve_speaker_id(&config, Some(&SpeakerId("p999".into())), None, true)
            .expect_err("unknown speaker");
        let message = format!("{error:#}");
        assert!(message.contains("unknown speaker `p999`"));
        assert!(message.contains("p225"));
        assert!(message.contains("p226"));
    }

    #[test]
    fn direct_numeric_speaker_id_is_validated() {
        let config = multi_speaker_config();
        assert_eq!(
            resolve_speaker_id(&config, None, Some(1), true).unwrap(),
            Some(1)
        );

        let error =
            resolve_speaker_id(&config, None, Some(2), true).expect_err("speaker out of range");
        assert!(format!("{error:#}").contains("speaker id 2 is out of range"));
    }

    #[test]
    fn omitted_multi_speaker_selection_is_rejected() {
        let config = multi_speaker_config();
        let error = resolve_speaker_id(&config, None, None, true).expect_err("speaker");
        let message = format!("{error:#}");
        assert!(message.contains("speaker selection is required"));
        assert!(message.contains("p225"));
    }

    #[test]
    fn speaker_selection_is_rejected_when_model_has_no_speaker_input() {
        let config = multi_speaker_config();
        let error = resolve_speaker_id(&config, Some(&SpeakerId("p225".into())), None, false)
            .expect_err("no speaker input");
        assert!(format!("{error:#}").contains("has no speaker input"));
    }

    #[cfg(feature = "onnx-tts")]
    #[test]
    fn onnx_adapter_validates_inference_scales() {
        let options = SynthesisOptions {
            length_scale: Some(0.0),
            ..SynthesisOptions::default()
        };
        let error = validate_onnx_options(&options).expect_err("invalid length scale");
        assert!(format!("{error:#}").contains("length scale"));

        let options = SynthesisOptions {
            noise_scale: Some(-1.0),
            ..SynthesisOptions::default()
        };
        let error = validate_onnx_options(&options).expect_err("invalid noise scale");
        assert!(format!("{error:#}").contains("noise scale"));
    }

    #[test]
    fn onnx_synthesis_is_env_gated() -> Result<()> {
        let Some(model) = std::env::var_os("TONGUES_TTS_ONNX_MODEL")
            .or_else(|| std::env::var_os("TONGUES_TTS_PIPER_MODEL"))
            .map(PathBuf::from)
        else {
            eprintln!(
                "skipping ONNX speech synthesis: set TONGUES_TTS_ONNX_MODEL and TONGUES_TTS_ONNX_CONFIG"
            );
            return Ok(());
        };
        let Some(config) = std::env::var_os("TONGUES_TTS_ONNX_CONFIG")
            .or_else(|| std::env::var_os("TONGUES_TTS_PIPER_CONFIG"))
            .map(PathBuf::from)
        else {
            eprintln!(
                "skipping ONNX speech synthesis: set TONGUES_TTS_ONNX_MODEL and TONGUES_TTS_ONNX_CONFIG"
            );
            return Ok(());
        };

        let mut speech = OnnxSpeech::load(VoiceModel::Path { model, config })?;
        let audio = speech.synthesize(SpeechRequest {
            text: "hello world".to_string(),
            variety: "en-US".to_string(),
        })?;
        assert_eq!(audio.channels, 1);
        assert!(audio.sample_rate_hz > 0);
        assert!(!audio.pcm_mono_f32.is_empty());
        Ok(())
    }

    fn multi_speaker_config() -> VoiceConfig {
        let mut config = VoiceConfig::from_json_str(RYAN_LIKE_CONFIG_JSON).expect("config");
        config.num_speakers = Some(2);
        config.speaker_id_map = HashMap::from([("p225".to_string(), 0), ("p226".to_string(), 1)]);
        config
    }
}
pub use burn_acoustic::BurnSpeedySpeechAcoustic;
