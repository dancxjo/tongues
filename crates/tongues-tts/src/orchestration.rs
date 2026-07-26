//! Backend-neutral speech orchestration contract.
//!
//! Model importers implement [`SynthesizerBackend`]. Callers select an
//! implementation through [`SynthesizerRegistry`], validate a single
//! [`UnifiedSynthesisRequest`] against discovered capabilities, and receive
//! normalized mono or interleaved floating-point audio chunks plus metadata.

use std::collections::BTreeMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use speaking::SpeakerId;
use thiserror::Error;

use crate::{
    utterance_plan_from_text, AudioChunk, ResolvedSpeechDevice, SpeechDeviceRequest,
    SpeechModelFamily, SpeechRequest, SpeechSynthesisEngine, SpeechSynthesisRequest,
    SynthesisOptions,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamedCapability {
    pub id: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numeric_id: Option<u32>,
}

impl NamedCapability {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            numeric_id: None,
        }
    }

    pub fn with_numeric_id(mut self, numeric_id: u32) -> Self {
        self.numeric_id = Some(numeric_id);
        self
    }
}

/// Declares whether a model rejects a feature, accepts any value, or exposes a
/// finite model-local catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "support", content = "values", rename_all = "snake_case")]
pub enum CapabilityValue {
    Unsupported,
    Any,
    Listed(Vec<NamedCapability>),
}

impl CapabilityValue {
    pub fn is_supported(&self) -> bool {
        !matches!(self, Self::Unsupported)
    }

    fn contains(&self, value: &str) -> bool {
        match self {
            Self::Unsupported => false,
            Self::Any => true,
            Self::Listed(values) => values.iter().any(|candidate| candidate.id == value),
        }
    }

    fn available(&self) -> Vec<String> {
        match self {
            Self::Listed(values) => values.iter().map(|value| value.id.clone()).collect(),
            Self::Unsupported | Self::Any => Vec::new(),
        }
    }

    fn contains_numeric(&self, value: u32) -> bool {
        match self {
            Self::Unsupported => false,
            Self::Any => true,
            Self::Listed(values) => values
                .iter()
                .any(|candidate| candidate.numeric_id == Some(value)),
        }
    }

    fn available_numeric(&self) -> Vec<String> {
        match self {
            Self::Listed(values) => values
                .iter()
                .filter_map(|value| value.numeric_id.map(|id| id.to_string()))
                .collect(),
            Self::Unsupported | Self::Any => Vec::new(),
        }
    }
}

pub fn variety_capabilities_for_language(language: &str) -> CapabilityValue {
    CapabilityValue::Listed(
        speaking::builtin_varieties()
            .into_iter()
            .filter(|variety| variety.language.0 == language)
            .map(|variety| NamedCapability::new(variety.id.0, variety.name))
            .collect(),
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeakerCapabilities {
    pub values: CapabilityValue,
    pub required: bool,
    pub numeric_ids: bool,
}

impl SpeakerCapabilities {
    pub fn unsupported() -> Self {
        Self {
            values: CapabilityValue::Unsupported,
            required: false,
            numeric_ids: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyleCapabilities {
    pub names: CapabilityValue,
    pub reference_audio: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_dimensions: Option<usize>,
}

impl StyleCapabilities {
    pub fn unsupported() -> Self {
        Self {
            names: CapabilityValue::Unsupported,
            reference_audio: false,
            embedding_dimensions: None,
        }
    }

    fn is_supported(&self) -> bool {
        self.names.is_supported() || self.reference_audio || self.embedding_dimensions.is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ReferenceAudioCapabilities {
    pub speaker: bool,
    pub style: bool,
    pub source: bool,
}

/// Discoverable token-level controls exposed by pitch-conditioned acoustic
/// models. Each flag is independent so callers can avoid sending controls a
/// selected backend cannot honor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PitchCapabilities {
    pub scale: bool,
    pub shift: bool,
    pub explicit_values: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputAudioContract {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub streaming: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// Stable implementation identifier used by API and CLI requests.
    pub backend: String,
    /// Stable model or composed-pipeline identifier.
    pub model: String,
    pub family: SpeechModelFamily,
    pub varieties: CapabilityValue,
    pub speakers: SpeakerCapabilities,
    pub styles: StyleCapabilities,
    pub reference_audio: ReferenceAudioCapabilities,
    pub speed: bool,
    #[serde(default)]
    pub pitch: PitchCapabilities,
    #[serde(default)]
    pub durations: bool,
    pub seed: bool,
    pub devices: Vec<SpeechDeviceRequest>,
    pub output: OutputAudioContract,
    #[serde(default)]
    pub provenance: Vec<String>,
}

impl BackendCapabilities {
    pub fn validate(
        &self,
        request: &UnifiedSynthesisRequest,
    ) -> Result<(), SynthesisContractError> {
        if request.text.trim().is_empty() {
            return Err(SynthesisContractError::InvalidRequest {
                field: "text",
                reason: "must not be empty".into(),
            });
        }
        if request.variety.trim().is_empty() {
            return Err(SynthesisContractError::InvalidRequest {
                field: "variety",
                reason: "must not be empty".into(),
            });
        }
        let variety_supported = self.varieties.contains(&request.variety)
            || speaking::canonical_variety_id(&request.variety)
                .is_some_and(|canonical| self.varieties.contains(&canonical.0));
        if !variety_supported {
            return Err(unsupported_value(
                &self.backend,
                "variety",
                &request.variety,
                self.varieties.available(),
            ));
        }
        if !request.speed.is_finite() || request.speed <= 0.0 {
            return Err(SynthesisContractError::InvalidRequest {
                field: "speed",
                reason: "must be finite and positive".into(),
            });
        }
        if request.speed != 1.0 && !self.speed {
            return Err(unsupported_feature(&self.backend, "speed"));
        }
        if let Some(value) = request.pitch_scale {
            if !value.is_finite() || value <= 0.0 {
                return Err(SynthesisContractError::InvalidRequest {
                    field: "pitch_scale",
                    reason: "must be finite and positive".into(),
                });
            }
            if !self.pitch.scale {
                return Err(unsupported_feature(&self.backend, "pitch_scale"));
            }
        }
        if let Some(value) = request.pitch_shift {
            if !value.is_finite() {
                return Err(SynthesisContractError::InvalidRequest {
                    field: "pitch_shift",
                    reason: "must be finite".into(),
                });
            }
            if !self.pitch.shift {
                return Err(unsupported_feature(&self.backend, "pitch_shift"));
            }
        }
        if let Some(values) = request.pitch.as_ref() {
            if values.is_empty() || !values.iter().all(|value| value.is_finite()) {
                return Err(SynthesisContractError::InvalidRequest {
                    field: "pitch",
                    reason: "must contain finite pitch-conditioning values".into(),
                });
            }
            if !self.pitch.explicit_values {
                return Err(unsupported_feature(&self.backend, "pitch"));
            }
        }
        if let Some(values) = request.durations.as_ref() {
            if values.is_empty() || values.contains(&0) {
                return Err(SynthesisContractError::InvalidRequest {
                    field: "durations",
                    reason: "must contain positive frame counts".into(),
                });
            }
            if !self.durations {
                return Err(unsupported_feature(&self.backend, "durations"));
            }
        }
        if request.seed.is_some() && !self.seed {
            return Err(unsupported_feature(&self.backend, "seed"));
        }
        for (field, value) in [
            ("noise_scale", request.noise_scale),
            ("duration_noise_scale", request.duration_noise_scale),
        ] {
            if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
                return Err(SynthesisContractError::InvalidRequest {
                    field,
                    reason: "must be finite and non-negative".into(),
                });
            }
        }
        if request.streaming && !self.output.streaming {
            return Err(unsupported_feature(&self.backend, "streaming"));
        }
        if request.max_chunk_symbols == Some(0) {
            return Err(SynthesisContractError::InvalidRequest {
                field: "max_chunk_symbols",
                reason: "must be greater than zero".into(),
            });
        }
        if !device_supported(&self.devices, request.device) {
            return Err(unsupported_value(
                &self.backend,
                "device",
                &device_label(request.device),
                self.devices.iter().copied().map(device_label).collect(),
            ));
        }

        match request.speaker.as_ref() {
            None if self.speakers.required => {
                return Err(SynthesisContractError::MissingRequiredFeature {
                    backend: self.backend.clone(),
                    feature: "speaker",
                });
            }
            Some(SpeakerSelection::Named(name)) => {
                if !self.speakers.values.contains(name) {
                    return Err(unsupported_value(
                        &self.backend,
                        "speaker",
                        name,
                        self.speakers.values.available(),
                    ));
                }
            }
            Some(SpeakerSelection::Numeric(id)) if !self.speakers.numeric_ids => {
                return Err(unsupported_value(
                    &self.backend,
                    "speaker_id",
                    &id.to_string(),
                    self.speakers.values.available(),
                ));
            }
            Some(SpeakerSelection::Numeric(id)) => {
                if !self.speakers.values.contains_numeric(*id) {
                    return Err(unsupported_value(
                        &self.backend,
                        "speaker_id",
                        &id.to_string(),
                        self.speakers.values.available_numeric(),
                    ));
                }
            }
            None => {}
        }

        if let Some(style) = request.style.as_ref() {
            if !self.styles.is_supported() {
                return Err(unsupported_feature(&self.backend, "style"));
            }
            if let Some(name) = style.name.as_deref() {
                if !self.styles.names.contains(name) {
                    return Err(unsupported_value(
                        &self.backend,
                        "style",
                        name,
                        self.styles.names.available(),
                    ));
                }
            }
            if let Some(embedding) = style.embedding.as_ref() {
                let Some(expected) = self.styles.embedding_dimensions else {
                    return Err(unsupported_feature(&self.backend, "style_embedding"));
                };
                if embedding.len() != expected || !embedding.iter().all(|value| value.is_finite()) {
                    return Err(SynthesisContractError::InvalidRequest {
                        field: "style.embedding",
                        reason: format!(
                            "must contain exactly {expected} finite values, got {}",
                            embedding.len()
                        ),
                    });
                }
            }
            if style.embedding_is_delta && style.embedding.is_none() {
                return Err(SynthesisContractError::InvalidRequest {
                    field: "style.embedding",
                    reason: "is required when embedding_is_delta is true".into(),
                });
            }
            if !style.strength.is_finite() || style.strength < 0.0 {
                return Err(SynthesisContractError::InvalidRequest {
                    field: "style.strength",
                    reason: "must be finite and non-negative".into(),
                });
            }
            for (field, value) in [
                ("style.speaker_blend", style.speaker_blend),
                ("style.style_blend", style.style_blend),
            ] {
                if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
                    return Err(SynthesisContractError::InvalidRequest {
                        field,
                        reason: "must be finite and between 0 and 1".into(),
                    });
                }
            }
            if style
                .diffusion_steps
                .is_some_and(|steps| !(1..=64).contains(&steps))
            {
                return Err(SynthesisContractError::InvalidRequest {
                    field: "style.diffusion_steps",
                    reason: "must be between 1 and 64".into(),
                });
            }
            if style
                .embedding_scale
                .is_some_and(|value| !value.is_finite() || value < 0.0)
            {
                return Err(SynthesisContractError::InvalidRequest {
                    field: "style.embedding_scale",
                    reason: "must be finite and non-negative".into(),
                });
            }
        }

        validate_reference(
            &self.backend,
            "reference_audio.speaker",
            request.reference_audio.speaker.as_deref(),
            self.reference_audio.speaker,
        )?;
        validate_reference(
            &self.backend,
            "reference_audio.style",
            request.reference_audio.style.as_deref(),
            self.reference_audio.style,
        )?;
        validate_reference(
            &self.backend,
            "reference_audio.source",
            request.reference_audio.source.as_deref(),
            self.reference_audio.source,
        )?;
        Ok(())
    }
}

fn validate_reference(
    backend: &str,
    feature: &'static str,
    value: Option<&str>,
    supported: bool,
) -> Result<(), SynthesisContractError> {
    if value.is_some_and(str::is_empty) {
        return Err(SynthesisContractError::InvalidRequest {
            field: feature,
            reason: "must not be empty".into(),
        });
    }
    if value.is_some() && !supported {
        return Err(unsupported_feature(backend, feature));
    }
    Ok(())
}

fn device_supported(supported: &[SpeechDeviceRequest], requested: SpeechDeviceRequest) -> bool {
    supported
        .iter()
        .any(|candidate| match (*candidate, requested) {
            (SpeechDeviceRequest::Auto, _) | (_, SpeechDeviceRequest::Auto) => true,
            (SpeechDeviceRequest::Cpu, SpeechDeviceRequest::Cpu) => true,
            (
                SpeechDeviceRequest::Cuda { index: available },
                SpeechDeviceRequest::Cuda { index: requested },
            ) => available == requested,
            _ => false,
        })
}

fn device_label(device: SpeechDeviceRequest) -> String {
    match device {
        SpeechDeviceRequest::Auto => "auto".into(),
        SpeechDeviceRequest::Cpu => "cpu".into(),
        SpeechDeviceRequest::Cuda { index } => format!("cuda:{index}"),
    }
}

fn unsupported_feature(backend: &str, feature: &'static str) -> SynthesisContractError {
    SynthesisContractError::UnsupportedFeature {
        backend: backend.to_string(),
        feature,
    }
}

fn unsupported_value(
    backend: &str,
    feature: &'static str,
    requested: &str,
    available: Vec<String>,
) -> SynthesisContractError {
    SynthesisContractError::UnsupportedValue {
        backend: backend.to_string(),
        feature,
        requested: requested.to_string(),
        available,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum SpeakerSelection {
    Named(String),
    Numeric(u32),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StyleSelection {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    #[serde(default)]
    pub embedding_is_delta: bool,
    #[serde(default = "default_style_strength")]
    pub strength: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_blend: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style_blend: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diffusion_steps: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_scale: Option<f64>,
}

fn default_style_strength() -> f32 {
    1.0
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceAudioRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    /// Source waveform for voice-conversion backends.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnifiedSynthesisRequest {
    pub text: String,
    pub variety: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<SpeakerSelection>,
    #[serde(default)]
    pub reference_audio: ReferenceAudioRequest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<StyleSelection>,
    #[serde(default = "default_speed")]
    pub speed: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch_scale: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch_shift: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub durations: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_scale: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_noise_scale: Option<f32>,
    #[serde(default = "default_device")]
    pub device: SpeechDeviceRequest,
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub profile: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_chunk_symbols: Option<usize>,
    #[serde(default = "default_chunking")]
    pub chunking: bool,
}

fn default_speed() -> f32 {
    1.0
}

fn default_device() -> SpeechDeviceRequest {
    SpeechDeviceRequest::Auto
}

fn default_chunking() -> bool {
    true
}

impl UnifiedSynthesisRequest {
    pub fn new(text: impl Into<String>, variety: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            variety: variety.into(),
            speaker: None,
            reference_audio: ReferenceAudioRequest::default(),
            style: None,
            speed: 1.0,
            pitch_scale: None,
            pitch_shift: None,
            pitch: None,
            durations: None,
            seed: None,
            noise_scale: None,
            duration_noise_scale: None,
            device: SpeechDeviceRequest::Auto,
            streaming: false,
            profile: false,
            max_chunk_symbols: None,
            chunking: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedAudioChunk {
    pub chunk_index: usize,
    pub is_final: bool,
    pub frame_offset: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
    /// Interleaved normalized `[-1, 1]` PCM.
    pub pcm_f32: Vec<f32>,
}

pub trait NormalizedAudioSink {
    fn emit(&mut self, chunk: NormalizedAudioChunk) -> Result<(), SynthesisContractError>;
}

impl<F> NormalizedAudioSink for F
where
    F: FnMut(NormalizedAudioChunk) -> Result<(), SynthesisContractError>,
{
    fn emit(&mut self, chunk: NormalizedAudioChunk) -> Result<(), SynthesisContractError> {
        self(chunk)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynthesisTiming {
    pub stage: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynthesisMetadata {
    pub backend: String,
    pub model: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub frames: u64,
    pub audio_seconds: f64,
    pub streaming: bool,
    #[serde(default)]
    pub timings: Vec<SynthesisTiming>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnifiedSynthesisOutput {
    pub metadata: SynthesisMetadata,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SynthesisContractError {
    #[error("backend `{backend}` does not support `{feature}`")]
    UnsupportedFeature {
        backend: String,
        feature: &'static str,
    },
    #[error(
        "backend `{backend}` does not support {feature} value `{requested}`; available: {available:?}"
    )]
    UnsupportedValue {
        backend: String,
        feature: &'static str,
        requested: String,
        available: Vec<String>,
    },
    #[error("backend `{backend}` requires `{feature}`")]
    MissingRequiredFeature {
        backend: String,
        feature: &'static str,
    },
    #[error("invalid synthesis request field `{field}`: {reason}")]
    InvalidRequest { field: &'static str, reason: String },
    #[error("speech backend failed: {message}")]
    Backend { message: String },
    #[error("audio sink failed: {message}")]
    Sink { message: String },
}

pub trait SynthesizerBackend: Send {
    fn capabilities(&self) -> BackendCapabilities;

    fn synthesize(
        &mut self,
        request: &UnifiedSynthesisRequest,
        sink: &mut dyn NormalizedAudioSink,
    ) -> Result<UnifiedSynthesisOutput, SynthesisContractError>;
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BackendRegistrationError {
    #[error("synthesizer backend id must not be empty")]
    EmptyId,
    #[error("synthesizer backend `{backend}` is already registered")]
    Duplicate { backend: String },
}

#[derive(Default)]
pub struct SynthesizerRegistry {
    backends: BTreeMap<String, Box<dyn SynthesizerBackend>>,
}

impl SynthesizerRegistry {
    pub fn register(
        &mut self,
        backend: Box<dyn SynthesizerBackend>,
    ) -> Result<(), BackendRegistrationError> {
        let id = backend.capabilities().backend;
        if id.trim().is_empty() {
            return Err(BackendRegistrationError::EmptyId);
        }
        if self.backends.contains_key(&id) {
            return Err(BackendRegistrationError::Duplicate { backend: id });
        }
        self.backends.insert(id, backend);
        Ok(())
    }

    pub fn capabilities(&self) -> Vec<BackendCapabilities> {
        self.backends
            .values()
            .map(|backend| backend.capabilities())
            .collect()
    }

    pub fn get_mut(&mut self, backend: &str) -> Option<&mut (dyn SynthesizerBackend + '_)> {
        if let Some(backend) = self.backends.get_mut(backend) {
            Some(backend.as_mut())
        } else {
            None
        }
    }

    pub fn synthesize(
        &mut self,
        backend: &str,
        request: &UnifiedSynthesisRequest,
        sink: &mut dyn NormalizedAudioSink,
    ) -> Result<UnifiedSynthesisOutput, SynthesisContractError> {
        if !self.backends.contains_key(backend) {
            return Err(SynthesisContractError::UnsupportedValue {
                backend: "registry".into(),
                feature: "backend",
                requested: backend.to_string(),
                available: self.backends.keys().cloned().collect(),
            });
        }
        let implementation = self
            .backends
            .get_mut(backend)
            .expect("backend presence checked above");
        let capabilities = implementation.capabilities();
        capabilities.validate(request)?;
        implementation.synthesize(request, sink)
    }
}

/// Adapts existing plan-oriented native engines to the public orchestration
/// contract. Imported end-to-end backends can implement [`SynthesizerBackend`]
/// directly when they require reference audio or richer style controls.
pub struct PlanEngineBackend<E> {
    capabilities: BackendCapabilities,
    resolved_device: ResolvedSpeechDevice,
    engine: E,
}

impl<E> PlanEngineBackend<E> {
    pub fn new(
        capabilities: BackendCapabilities,
        resolved_device: ResolvedSpeechDevice,
        engine: E,
    ) -> Self {
        Self {
            capabilities,
            resolved_device,
            engine,
        }
    }

    pub fn engine(&self) -> &E {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }
}

impl<E: SpeechSynthesisEngine + Send> SynthesizerBackend for PlanEngineBackend<E> {
    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities.clone()
    }

    fn synthesize(
        &mut self,
        request: &UnifiedSynthesisRequest,
        sink: &mut dyn NormalizedAudioSink,
    ) -> Result<UnifiedSynthesisOutput, SynthesisContractError> {
        self.capabilities.validate(request)?;
        if !request_device_matches_resolved(request.device, self.resolved_device) {
            return Err(unsupported_value(
                &self.capabilities.backend,
                "device",
                &device_label(request.device),
                vec![resolved_device_label(self.resolved_device)],
            ));
        }

        let mut plan = utterance_plan_from_text(SpeechRequest {
            text: request.text.clone(),
            variety: request.variety.clone(),
        })
        .map_err(backend_error)?;
        let speaker_id = match request.speaker.as_ref() {
            Some(SpeakerSelection::Named(name)) => {
                plan.speaker = Some(SpeakerId(name.clone()));
                None
            }
            Some(SpeakerSelection::Numeric(id)) => Some(*id),
            None => None,
        };
        let synthesis_request = SpeechSynthesisRequest {
            plan,
            options: SynthesisOptions {
                speaker_id,
                split_sentences: true,
                length_scale: Some(1.0 / request.speed),
                noise_scale: request.noise_scale,
                noise_w: request.duration_noise_scale,
                pitch_scale: request.pitch_scale,
                pitch_shift: request.pitch_shift,
                durations: request.durations.clone(),
                pitch: request.pitch.clone(),
                seed: request.seed,
            },
        };
        let started = Instant::now();
        let mut frames = 0_u64;
        let mut frame_offset = 0_u64;
        let mut sink_failure = None;
        let mut profile = Vec::new();
        let mut audio_sink = |chunk: AudioChunk| {
            if chunk.sample_rate_hz != self.capabilities.output.sample_rate_hz {
                return Err(anyhow::anyhow!(
                    "backend emitted {} Hz but declared {} Hz",
                    chunk.sample_rate_hz,
                    self.capabilities.output.sample_rate_hz
                ));
            }
            let chunk_frames = chunk.pcm_mono_f32.len() as u64;
            if let Err(error) = sink.emit(NormalizedAudioChunk {
                chunk_index: chunk.chunk_index,
                is_final: chunk.is_final,
                frame_offset,
                sample_rate_hz: chunk.sample_rate_hz,
                channels: 1,
                pcm_f32: chunk.pcm_mono_f32,
            }) {
                let message = error.to_string();
                sink_failure = Some(error);
                return Err(anyhow::anyhow!(message));
            }
            frames += chunk_frames;
            frame_offset += chunk_frames;
            Ok(())
        };
        let synthesis_result = if request.profile {
            self.engine.synthesize_plan_streaming_profiled(
                &synthesis_request,
                &mut audio_sink,
                &mut |event| profile.push(event),
            )
        } else {
            self.engine
                .synthesize_plan_streaming(&synthesis_request, &mut audio_sink)
        };
        if let Some(error) = sink_failure {
            return Err(error);
        }
        synthesis_result.map_err(backend_error)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let sample_rate_hz = self.capabilities.output.sample_rate_hz;
        let mut timings = profile
            .into_iter()
            .map(|event| SynthesisTiming {
                stage: event.stage.to_string(),
                elapsed_ms: event.elapsed_ms,
            })
            .collect::<Vec<_>>();
        timings.push(SynthesisTiming {
            stage: "total".into(),
            elapsed_ms,
        });
        Ok(UnifiedSynthesisOutput {
            metadata: SynthesisMetadata {
                backend: self.capabilities.backend.clone(),
                model: self.capabilities.model.clone(),
                sample_rate_hz,
                channels: self.capabilities.output.channels,
                frames,
                audio_seconds: frames as f64 / sample_rate_hz as f64,
                streaming: request.streaming,
                timings,
            },
        })
    }
}

fn backend_error(error: anyhow::Error) -> SynthesisContractError {
    SynthesisContractError::Backend {
        message: format!("{error:#}"),
    }
}

fn request_device_matches_resolved(
    requested: SpeechDeviceRequest,
    resolved: ResolvedSpeechDevice,
) -> bool {
    match (requested, resolved) {
        (SpeechDeviceRequest::Auto, _) => true,
        (SpeechDeviceRequest::Cpu, ResolvedSpeechDevice::Cpu) => true,
        (
            SpeechDeviceRequest::Cuda { index: requested },
            ResolvedSpeechDevice::Cuda { index: resolved },
        ) => requested == resolved,
        _ => false,
    }
}

fn resolved_device_label(device: ResolvedSpeechDevice) -> String {
    match device {
        ResolvedSpeechDevice::Cpu => "cpu".into(),
        ResolvedSpeechDevice::Cuda { index } => format!("cuda:{index}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioSink, SpeechModelCapabilities};

    #[derive(Debug)]
    struct FixtureEngine;

    impl SpeechSynthesisEngine for FixtureEngine {
        fn capabilities(&self) -> SpeechModelCapabilities {
            SpeechModelCapabilities {
                family: SpeechModelFamily::EndToEndSpeech,
                supports_named_speakers: true,
                supports_languages: false,
                supports_reference_audio: false,
                supports_voice_conversion: false,
                integrated_vocoder: true,
            }
        }

        fn sample_rate_hz(&self) -> u32 {
            24_000
        }

        fn synthesize_plan_streaming(
            &mut self,
            _request: &SpeechSynthesisRequest,
            sink: &mut dyn AudioSink,
        ) -> anyhow::Result<()> {
            sink.emit(AudioChunk {
                chunk_index: 0,
                is_final: true,
                pause_after_ms: 0,
                sample_rate_hz: 24_000,
                pcm_mono_f32: vec![0.0; 240],
            })
        }
    }

    fn fixture_capabilities() -> BackendCapabilities {
        BackendCapabilities {
            backend: "fixture".into(),
            model: "fixture-v1".into(),
            family: SpeechModelFamily::EndToEndSpeech,
            varieties: CapabilityValue::Listed(vec![NamedCapability::new("en-US", "English")]),
            speakers: SpeakerCapabilities {
                values: CapabilityValue::Listed(vec![
                    NamedCapability::new("p225", "p225").with_numeric_id(0)
                ]),
                required: true,
                numeric_ids: true,
            },
            styles: StyleCapabilities::unsupported(),
            reference_audio: ReferenceAudioCapabilities::default(),
            speed: true,
            pitch: PitchCapabilities::default(),
            durations: false,
            seed: true,
            devices: vec![SpeechDeviceRequest::Cpu],
            output: OutputAudioContract {
                sample_rate_hz: 24_000,
                channels: 1,
                streaming: true,
            },
            provenance: Vec::new(),
        }
    }

    #[test]
    fn capabilities_reject_unsupported_features_before_inference() {
        let capabilities = fixture_capabilities();
        let mut request = UnifiedSynthesisRequest::new("hello", "en-US");
        request.device = SpeechDeviceRequest::Cpu;
        request.speaker = Some(SpeakerSelection::Named("p225".into()));
        request.reference_audio.speaker = Some("voice.wav".into());

        assert_eq!(
            capabilities.validate(&request),
            Err(SynthesisContractError::UnsupportedFeature {
                backend: "fixture".into(),
                feature: "reference_audio.speaker",
            })
        );
    }

    #[test]
    fn fastpitch_controls_are_validated_through_discoverable_capabilities() {
        let mut capabilities = fixture_capabilities();
        capabilities.speakers = SpeakerCapabilities::unsupported();
        capabilities.pitch = PitchCapabilities {
            scale: true,
            shift: true,
            explicit_values: true,
        };
        capabilities.durations = true;

        let mut request = UnifiedSynthesisRequest::new("hello", "en-US");
        request.device = SpeechDeviceRequest::Cpu;
        request.pitch_scale = Some(1.1);
        request.pitch_shift = Some(-0.25);
        request.pitch = Some(vec![0.2, -0.1]);
        request.durations = Some(vec![3, 4]);
        capabilities
            .validate(&request)
            .expect("supported FastPitch controls");

        request.durations = Some(vec![3, 0]);
        assert!(matches!(
            capabilities.validate(&request),
            Err(SynthesisContractError::InvalidRequest {
                field: "durations",
                ..
            })
        ));
    }

    #[test]
    fn capabilities_report_available_values_for_unknown_speaker() {
        let capabilities = fixture_capabilities();
        let mut request = UnifiedSynthesisRequest::new("hello", "en-US");
        request.device = SpeechDeviceRequest::Cpu;
        request.speaker = Some(SpeakerSelection::Named("unknown".into()));

        assert_eq!(
            capabilities.validate(&request),
            Err(SynthesisContractError::UnsupportedValue {
                backend: "fixture".into(),
                feature: "speaker",
                requested: "unknown".into(),
                available: vec!["p225".into()],
            })
        );

        request.speaker = Some(SpeakerSelection::Numeric(99));
        assert_eq!(
            capabilities.validate(&request),
            Err(SynthesisContractError::UnsupportedValue {
                backend: "fixture".into(),
                feature: "speaker_id",
                requested: "99".into(),
                available: vec!["0".into()],
            })
        );
    }

    #[test]
    fn registry_routes_without_a_provider_switch_and_normalizes_metadata() {
        let mut registry = SynthesizerRegistry::default();
        registry
            .register(Box::new(PlanEngineBackend::new(
                fixture_capabilities(),
                ResolvedSpeechDevice::Cpu,
                FixtureEngine,
            )))
            .expect("register fixture backend");
        let mut request = UnifiedSynthesisRequest::new("hello", "en-US");
        request.device = SpeechDeviceRequest::Cpu;
        request.streaming = true;
        request.speaker = Some(SpeakerSelection::Named("p225".into()));
        let mut chunks = Vec::new();
        let output = registry
            .synthesize("fixture", &request, &mut |chunk| {
                chunks.push(chunk);
                Ok(())
            })
            .expect("synthesis");

        assert_eq!(output.metadata.backend, "fixture");
        assert_eq!(output.metadata.frames, 240);
        assert_eq!(output.metadata.sample_rate_hz, 24_000);
        assert_eq!(output.metadata.channels, 1);
        assert_eq!(chunks[0].frame_offset, 0);
        assert!(chunks[0].is_final);
    }

    #[test]
    fn duplicate_backend_registration_is_explicit() {
        let mut registry = SynthesizerRegistry::default();
        for expected in [
            Ok(()),
            Err(BackendRegistrationError::Duplicate {
                backend: "fixture".into(),
            }),
        ] {
            let result = registry.register(Box::new(PlanEngineBackend::new(
                fixture_capabilities(),
                ResolvedSpeechDevice::Cpu,
                FixtureEngine,
            )));
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn neutral_request_json_has_stable_defaults_and_accepts_variety_aliases() {
        let request: UnifiedSynthesisRequest =
            serde_json::from_str(r#"{"text":"hello","variety":"en-US"}"#)
                .expect("minimal public request");
        assert_eq!(request.speed, 1.0);
        assert_eq!(request.device, SpeechDeviceRequest::Auto);
        assert!(request.chunking);
        assert!(!request.streaming);

        let mut capabilities = fixture_capabilities();
        capabilities.varieties = variety_capabilities_for_language("en");
        let mut request = request;
        request.device = SpeechDeviceRequest::Cpu;
        request.speaker = Some(SpeakerSelection::Named("p225".into()));
        capabilities
            .validate(&request)
            .expect("en-US alias should match canonical en-US-GA capability");
    }
}
