use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::LanguageId;
use crate::event::StreamEvent;
use crate::transcript::{TranscriptCandidateTracker, TranscriptChunk};
use crate::word_stream::TranscriptWord;

#[derive(Debug, Clone, PartialEq)]
pub struct AudioFrame {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl Default for AudioFrame {
    fn default() -> Self {
        Self {
            sample_rate_hz: 0,
            channels: 0,
            samples: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingPartialKind {
    FinalOnly,
    Approximate,
    TokenStreaming,
}

impl StreamingPartialKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FinalOnly => "final_only",
            Self::Approximate => "approximate",
            Self::TokenStreaming => "token_streaming",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingRecognizerBackend {
    pub source: &'static str,
    pub partial_kind: StreamingPartialKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamingRecognition {
    pub text: String,
    pub words: Vec<TranscriptWord>,
    /// The central stream IR emitted directly by this provider.
    pub events: Vec<StreamEvent>,
    pub backend: StreamingRecognizerBackend,
}

pub trait SpeechRecognizer {
    fn push_frame(&mut self, frame: &AudioFrame) -> anyhow::Result<()>;

    fn poll_chunks(&mut self) -> anyhow::Result<Vec<TranscriptChunk>>;
}

pub trait StreamingSpeechRecognizer: SpeechRecognizer {
    fn poll_streaming(&mut self, is_final: bool) -> anyhow::Result<StreamingRecognition>;

    fn flush(&mut self) -> anyhow::Result<StreamingRecognition> {
        self.poll_streaming(true)
    }

    fn backend(&self) -> StreamingRecognizerBackend;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AsrSessionId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsrDecodingControl {
    BeamWidth,
    Temperature,
    Prompt,
    Timestamps,
    Punctuation,
    VocabularyBias,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrDecodingConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beam_width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamps: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub punctuation: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vocabulary_bias: Vec<String>,
}

impl Default for AsrDecodingConfig {
    fn default() -> Self {
        Self {
            beam_width: None,
            temperature: None,
            prompt: None,
            timestamps: Some(true),
            punctuation: None,
            vocabulary_bias: Vec::new(),
        }
    }
}

impl AsrDecodingConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.beam_width == Some(0) {
            anyhow::bail!("ASR beam width must be positive");
        }
        if self
            .temperature
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            anyhow::bail!("ASR temperature must be finite and non-negative");
        }
        if self
            .vocabulary_bias
            .iter()
            .any(|term| term.trim().is_empty())
        {
            anyhow::bail!("ASR vocabulary bias terms cannot be empty");
        }
        Ok(())
    }

    pub fn requested_controls(&self) -> BTreeSet<AsrDecodingControl> {
        let mut controls = BTreeSet::new();
        if self.beam_width.is_some() {
            controls.insert(AsrDecodingControl::BeamWidth);
        }
        if self.temperature.is_some() {
            controls.insert(AsrDecodingControl::Temperature);
        }
        if self.prompt.is_some() {
            controls.insert(AsrDecodingControl::Prompt);
        }
        if self.timestamps.is_some() {
            controls.insert(AsrDecodingControl::Timestamps);
        }
        if self.punctuation.is_some() {
            controls.insert(AsrDecodingControl::Punctuation);
        }
        if !self.vocabulary_bias.is_empty() {
            controls.insert(AsrDecodingControl::VocabularyBias);
        }
        controls
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum AsrStreamingCapability {
    Native,
    ChunkedSimulation { window_ms: u64, overlap_ms: u64 },
    OfflineOnly,
}

impl AsrStreamingCapability {
    pub fn validate(&self) -> anyhow::Result<()> {
        if let Self::ChunkedSimulation {
            window_ms,
            overlap_ms,
        } = self
        {
            anyhow::ensure!(*window_ms > 0, "ASR chunk window must be positive");
            anyhow::ensure!(
                overlap_ms < window_ms,
                "ASR chunk overlap must be smaller than its window"
            );
        }
        Ok(())
    }

    pub fn is_native(&self) -> bool {
        matches!(self, Self::Native)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AsrProviderCapabilities {
    pub provider_id: String,
    pub model_id: String,
    pub installed: bool,
    pub languages: Vec<LanguageId>,
    pub streaming: AsrStreamingCapability,
    pub decoding_controls: BTreeSet<AsrDecodingControl>,
    pub maximum_concurrent_sessions: usize,
    pub estimated_memory_mb_per_session: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_license: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_checksum: Option<String>,
}

impl AsrProviderCapabilities {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.provider_id.is_empty(), "ASR provider ID is empty");
        anyhow::ensure!(!self.model_id.is_empty(), "ASR model ID is empty");
        anyhow::ensure!(
            !self.languages.is_empty(),
            "ASR provider supports no languages"
        );
        anyhow::ensure!(
            self.maximum_concurrent_sessions > 0,
            "ASR provider session capacity must be positive"
        );
        self.streaming.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrSessionConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<LanguageId>,
    #[serde(default)]
    pub decoding: AsrDecodingConfig,
}

impl Default for AsrSessionConfig {
    fn default() -> Self {
        Self {
            language: None,
            decoding: AsrDecodingConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AsrResourceLimits {
    pub maximum_total_sessions: usize,
    pub maximum_estimated_memory_mb: u64,
}

impl Default for AsrResourceLimits {
    fn default() -> Self {
        Self {
            maximum_total_sessions: 4,
            maximum_estimated_memory_mb: 4_096,
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AsrRuntimeError {
    #[error("unknown ASR provider `{0}`")]
    UnknownProvider(String),
    #[error("unknown ASR session `{0}`")]
    UnknownSession(String),
    #[error("ASR provider `{0}` is not loaded")]
    ProviderNotLoaded(String),
    #[error("ASR provider `{provider_id}` does not support language `{language}`")]
    UnsupportedLanguage {
        provider_id: String,
        language: String,
    },
    #[error("ASR provider `{provider_id}` does not support decoding control `{control:?}`")]
    UnsupportedDecodingControl {
        provider_id: String,
        control: AsrDecodingControl,
    },
    #[error("cannot unload ASR provider `{0}` while it has active sessions")]
    ProviderBusy(String),
    #[error("ASR resource capacity exhausted: {0}")]
    ResourceExhausted(String),
    #[error("ASR provider `{provider_id}` failed: {message}")]
    Provider {
        provider_id: String,
        message: String,
    },
}

pub trait AsrSession: Send {
    fn push_audio(&mut self, frame: &AudioFrame) -> anyhow::Result<Vec<StreamEvent>>;
    fn finish(&mut self) -> anyhow::Result<Vec<StreamEvent>>;
    fn cancel(&mut self, reason: &str) -> anyhow::Result<Vec<StreamEvent>>;
}

pub trait AsrProvider: Send {
    fn capabilities(&self) -> AsrProviderCapabilities;
    fn is_loaded(&self) -> bool;
    fn load(&mut self) -> anyhow::Result<()>;
    fn unload(&mut self) -> anyhow::Result<()>;
    fn start_session(&mut self, config: &AsrSessionConfig) -> anyhow::Result<Box<dyn AsrSession>>;
}

struct ActiveAsrSession {
    provider_id: String,
    estimated_memory_mb: u64,
    session: Box<dyn AsrSession>,
}

pub struct AsrRuntime {
    providers: BTreeMap<String, Box<dyn AsrProvider>>,
    sessions: BTreeMap<String, ActiveAsrSession>,
    limits: AsrResourceLimits,
    next_session_id: u64,
}

impl AsrRuntime {
    pub fn new(limits: AsrResourceLimits) -> anyhow::Result<Self> {
        anyhow::ensure!(
            limits.maximum_total_sessions > 0,
            "ASR runtime session capacity must be positive"
        );
        anyhow::ensure!(
            limits.maximum_estimated_memory_mb > 0,
            "ASR runtime memory capacity must be positive"
        );
        Ok(Self {
            providers: BTreeMap::new(),
            sessions: BTreeMap::new(),
            limits,
            next_session_id: 0,
        })
    }

    pub fn register_provider(
        &mut self,
        provider: impl AsrProvider + 'static,
    ) -> anyhow::Result<()> {
        let capabilities = provider.capabilities();
        capabilities.validate()?;
        anyhow::ensure!(
            !self.providers.contains_key(&capabilities.provider_id),
            "duplicate ASR provider ID `{}`",
            capabilities.provider_id
        );
        self.providers
            .insert(capabilities.provider_id.clone(), Box::new(provider));
        Ok(())
    }

    pub fn capabilities(&self) -> Vec<AsrProviderCapabilities> {
        self.providers
            .values()
            .map(|provider| provider.capabilities())
            .collect()
    }

    pub fn load_provider(&mut self, provider_id: &str) -> Result<(), AsrRuntimeError> {
        let provider = self.provider_mut(provider_id)?;
        provider.load().map_err(|error| AsrRuntimeError::Provider {
            provider_id: provider_id.into(),
            message: error.to_string(),
        })
    }

    pub fn unload_provider(&mut self, provider_id: &str) -> Result<(), AsrRuntimeError> {
        if self
            .sessions
            .values()
            .any(|session| session.provider_id == provider_id)
        {
            return Err(AsrRuntimeError::ProviderBusy(provider_id.into()));
        }
        let provider = self.provider_mut(provider_id)?;
        provider
            .unload()
            .map_err(|error| AsrRuntimeError::Provider {
                provider_id: provider_id.into(),
                message: error.to_string(),
            })
    }

    pub fn start_session(
        &mut self,
        provider_id: &str,
        config: AsrSessionConfig,
    ) -> Result<AsrSessionId, AsrRuntimeError> {
        let capabilities = self
            .providers
            .get(provider_id)
            .ok_or_else(|| AsrRuntimeError::UnknownProvider(provider_id.into()))?
            .capabilities();
        if !self.providers[provider_id].is_loaded() {
            return Err(AsrRuntimeError::ProviderNotLoaded(provider_id.into()));
        }
        config
            .decoding
            .validate()
            .map_err(|error| AsrRuntimeError::Provider {
                provider_id: provider_id.into(),
                message: error.to_string(),
            })?;
        if let Some(language) = &config.language
            && !capabilities.languages.contains(language)
        {
            return Err(AsrRuntimeError::UnsupportedLanguage {
                provider_id: provider_id.into(),
                language: language.0.clone(),
            });
        }
        if let Some(control) = config
            .decoding
            .requested_controls()
            .difference(&capabilities.decoding_controls)
            .next()
        {
            return Err(AsrRuntimeError::UnsupportedDecodingControl {
                provider_id: provider_id.into(),
                control: *control,
            });
        }
        let provider_sessions = self
            .sessions
            .values()
            .filter(|session| session.provider_id == provider_id)
            .count();
        if provider_sessions >= capabilities.maximum_concurrent_sessions {
            return Err(AsrRuntimeError::ResourceExhausted(format!(
                "provider `{provider_id}` session limit reached"
            )));
        }
        if self.sessions.len() >= self.limits.maximum_total_sessions {
            return Err(AsrRuntimeError::ResourceExhausted(
                "runtime session limit reached".into(),
            ));
        }
        let allocated_memory = self
            .sessions
            .values()
            .map(|session| session.estimated_memory_mb)
            .sum::<u64>();
        if allocated_memory.saturating_add(capabilities.estimated_memory_mb_per_session)
            > self.limits.maximum_estimated_memory_mb
        {
            return Err(AsrRuntimeError::ResourceExhausted(
                "runtime estimated memory limit reached".into(),
            ));
        }

        let session = self.providers.get_mut(provider_id).unwrap();
        let session =
            session
                .start_session(&config)
                .map_err(|error| AsrRuntimeError::Provider {
                    provider_id: provider_id.into(),
                    message: error.to_string(),
                })?;
        self.next_session_id = self.next_session_id.checked_add(1).ok_or_else(|| {
            AsrRuntimeError::ResourceExhausted("session ID space exhausted".into())
        })?;
        let id = AsrSessionId(format!("asr-session:{}", self.next_session_id));
        self.sessions.insert(
            id.0.clone(),
            ActiveAsrSession {
                provider_id: provider_id.into(),
                estimated_memory_mb: capabilities.estimated_memory_mb_per_session,
                session,
            },
        );
        Ok(id)
    }

    pub fn push_audio(
        &mut self,
        session_id: &AsrSessionId,
        frame: &AudioFrame,
    ) -> Result<Vec<StreamEvent>, AsrRuntimeError> {
        let active = self
            .sessions
            .get_mut(&session_id.0)
            .ok_or_else(|| AsrRuntimeError::UnknownSession(session_id.0.clone()))?;
        active
            .session
            .push_audio(frame)
            .map_err(|error| AsrRuntimeError::Provider {
                provider_id: active.provider_id.clone(),
                message: error.to_string(),
            })
    }

    pub fn finish_session(
        &mut self,
        session_id: &AsrSessionId,
    ) -> Result<Vec<StreamEvent>, AsrRuntimeError> {
        let mut active = self
            .sessions
            .remove(&session_id.0)
            .ok_or_else(|| AsrRuntimeError::UnknownSession(session_id.0.clone()))?;
        active
            .session
            .finish()
            .map_err(|error| AsrRuntimeError::Provider {
                provider_id: active.provider_id,
                message: error.to_string(),
            })
    }

    pub fn cancel_session(
        &mut self,
        session_id: &AsrSessionId,
        reason: &str,
    ) -> Result<Vec<StreamEvent>, AsrRuntimeError> {
        let mut active = self
            .sessions
            .remove(&session_id.0)
            .ok_or_else(|| AsrRuntimeError::UnknownSession(session_id.0.clone()))?;
        active
            .session
            .cancel(reason)
            .map_err(|error| AsrRuntimeError::Provider {
                provider_id: active.provider_id,
                message: error.to_string(),
            })
    }

    pub fn transcribe_offline(
        &mut self,
        provider_id: &str,
        config: AsrSessionConfig,
        frames: impl IntoIterator<Item = AudioFrame>,
    ) -> Result<Vec<StreamEvent>, AsrRuntimeError> {
        let session_id = self.start_session(provider_id, config)?;
        let mut events = Vec::new();
        for frame in frames {
            match self.push_audio(&session_id, &frame) {
                Ok(next) => events.extend(next),
                Err(error) => {
                    let _ = self.cancel_session(&session_id, "offline transcription failed");
                    return Err(error);
                }
            }
        }
        events.extend(self.finish_session(&session_id)?);
        Ok(events)
    }

    fn provider_mut(
        &mut self,
        provider_id: &str,
    ) -> Result<&mut Box<dyn AsrProvider>, AsrRuntimeError> {
        self.providers
            .get_mut(provider_id)
            .ok_or_else(|| AsrRuntimeError::UnknownProvider(provider_id.into()))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FixtureAsrStep {
    pub text: String,
    pub confidence: Option<f32>,
    pub is_final: bool,
}

pub struct FixtureAsrProvider {
    capabilities: AsrProviderCapabilities,
    steps: Vec<FixtureAsrStep>,
    loaded: bool,
}

impl FixtureAsrProvider {
    pub fn new(
        capabilities: AsrProviderCapabilities,
        steps: Vec<FixtureAsrStep>,
    ) -> anyhow::Result<Self> {
        capabilities.validate()?;
        Ok(Self {
            capabilities,
            steps,
            loaded: false,
        })
    }
}

impl AsrProvider for FixtureAsrProvider {
    fn capabilities(&self) -> AsrProviderCapabilities {
        let mut capabilities = self.capabilities.clone();
        capabilities.installed = true;
        capabilities
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    fn load(&mut self) -> anyhow::Result<()> {
        self.loaded = true;
        Ok(())
    }

    fn unload(&mut self) -> anyhow::Result<()> {
        self.loaded = false;
        Ok(())
    }

    fn start_session(&mut self, _config: &AsrSessionConfig) -> anyhow::Result<Box<dyn AsrSession>> {
        anyhow::ensure!(self.loaded, "fixture ASR provider is not loaded");
        Ok(Box::new(FixtureAsrSession {
            steps: self.steps.clone().into_iter(),
            tracker: TranscriptCandidateTracker::new(),
            last_text: None,
            final_emitted: false,
        }))
    }
}

struct FixtureAsrSession {
    steps: std::vec::IntoIter<FixtureAsrStep>,
    tracker: TranscriptCandidateTracker,
    last_text: Option<(String, Option<f32>)>,
    final_emitted: bool,
}

impl AsrSession for FixtureAsrSession {
    fn push_audio(&mut self, frame: &AudioFrame) -> anyhow::Result<Vec<StreamEvent>> {
        anyhow::ensure!(
            frame.sample_rate_hz > 0 && frame.channels > 0,
            "fixture ASR received invalid audio geometry"
        );
        let Some(step) = self.steps.next() else {
            return Ok(Vec::new());
        };
        self.last_text = Some((step.text.clone(), step.confidence));
        self.final_emitted = step.is_final;
        Ok(self
            .tracker
            .ingest_candidate(step.text, step.confidence, step.is_final))
    }

    fn finish(&mut self) -> anyhow::Result<Vec<StreamEvent>> {
        if self.final_emitted {
            return Ok(vec![StreamEvent::Completed]);
        }
        let mut events = self
            .last_text
            .take()
            .map(|(text, confidence)| self.tracker.ingest_candidate(text, confidence, true))
            .unwrap_or_else(|| self.tracker.cancel_active());
        events.push(StreamEvent::Completed);
        self.final_emitted = true;
        Ok(events)
    }

    fn cancel(&mut self, reason: &str) -> anyhow::Result<Vec<StreamEvent>> {
        let mut events = self.tracker.cancel_active();
        events.push(StreamEvent::Cancelled {
            reason: reason.into(),
        });
        Ok(events)
    }
}

pub fn committed_transcript(events: &[StreamEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::CommittedSegment { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod provider_runtime_tests {
    use super::*;

    fn capability(provider_id: &str, streaming: AsrStreamingCapability) -> AsrProviderCapabilities {
        AsrProviderCapabilities {
            provider_id: provider_id.into(),
            model_id: format!("{provider_id}-model-v1"),
            installed: true,
            languages: vec![LanguageId("en".into())],
            streaming,
            decoding_controls: BTreeSet::from([
                AsrDecodingControl::BeamWidth,
                AsrDecodingControl::Timestamps,
            ]),
            maximum_concurrent_sessions: 1,
            estimated_memory_mb_per_session: 64,
            model_license: Some("fixture-only".into()),
            model_checksum: Some("sha256:fixture".into()),
        }
    }

    fn provider(provider_id: &str) -> FixtureAsrProvider {
        FixtureAsrProvider::new(
            capability(provider_id, AsrStreamingCapability::Native),
            vec![
                FixtureAsrStep {
                    text: "hello".into(),
                    confidence: Some(0.7),
                    is_final: false,
                },
                FixtureAsrStep {
                    text: "hello world".into(),
                    confidence: Some(0.9),
                    is_final: false,
                },
            ],
        )
        .unwrap()
    }

    fn frame() -> AudioFrame {
        AudioFrame {
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.1; 160],
        }
    }

    fn runtime() -> AsrRuntime {
        AsrRuntime::new(AsrResourceLimits {
            maximum_total_sessions: 2,
            maximum_estimated_memory_mb: 128,
        })
        .unwrap()
    }

    #[test]
    fn provider_lifecycle_capacity_and_unload_are_explicit() {
        let mut runtime = runtime();
        runtime.register_provider(provider("fixture")).unwrap();
        assert_eq!(
            runtime
                .start_session("fixture", AsrSessionConfig::default())
                .unwrap_err(),
            AsrRuntimeError::ProviderNotLoaded("fixture".into())
        );

        runtime.load_provider("fixture").unwrap();
        let session = runtime
            .start_session("fixture", AsrSessionConfig::default())
            .unwrap();
        assert_eq!(
            runtime
                .start_session("fixture", AsrSessionConfig::default())
                .unwrap_err(),
            AsrRuntimeError::ResourceExhausted("provider `fixture` session limit reached".into())
        );
        assert_eq!(
            runtime.unload_provider("fixture").unwrap_err(),
            AsrRuntimeError::ProviderBusy("fixture".into())
        );
        runtime.cancel_session(&session, "test complete").unwrap();
        runtime.unload_provider("fixture").unwrap();
    }

    #[test]
    fn deterministic_streaming_revisions_and_final_assembly_share_the_event_contract() {
        let mut runtime = runtime();
        runtime.register_provider(provider("fixture")).unwrap();
        runtime.load_provider("fixture").unwrap();
        let session = runtime
            .start_session("fixture", AsrSessionConfig::default())
            .unwrap();

        let first = runtime.push_audio(&session, &frame()).unwrap();
        let second = runtime.push_audio(&session, &frame()).unwrap();
        let final_events = runtime.finish_session(&session).unwrap();
        assert!(matches!(
            first.as_slice(),
            [StreamEvent::PartialHypothesis { text, .. }] if text == "hello"
        ));
        assert!(matches!(
            second.as_slice(),
            [StreamEvent::RevisedHypothesis { text, .. }] if text == " world"
        ));
        assert!(matches!(
            final_events.as_slice(),
            [
                StreamEvent::CommittedSegment { text, .. },
                StreamEvent::Completed
            ] if text == "hello world"
        ));
        assert_eq!(committed_transcript(&final_events), "hello world");
    }

    #[test]
    fn offline_transcription_uses_the_same_ordered_events() {
        let mut runtime = runtime();
        runtime.register_provider(provider("fixture")).unwrap();
        runtime.load_provider("fixture").unwrap();
        let events = runtime
            .transcribe_offline("fixture", AsrSessionConfig::default(), [frame(), frame()])
            .unwrap();
        assert!(matches!(
            events.as_slice(),
            [
                StreamEvent::PartialHypothesis { .. },
                StreamEvent::RevisedHypothesis { .. },
                StreamEvent::CommittedSegment { .. },
                StreamEvent::Completed
            ]
        ));
        assert_eq!(committed_transcript(&events), "hello world");
    }

    #[test]
    fn cancellation_releases_resources_and_is_terminal() {
        let mut runtime = runtime();
        runtime.register_provider(provider("fixture")).unwrap();
        runtime.load_provider("fixture").unwrap();
        let session = runtime
            .start_session("fixture", AsrSessionConfig::default())
            .unwrap();
        runtime.push_audio(&session, &frame()).unwrap();
        let events = runtime
            .cancel_session(&session, "operator request")
            .unwrap();
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Cancelled { reason }) if reason == "operator request"
        ));
        assert_eq!(
            runtime.push_audio(&session, &frame()).unwrap_err(),
            AsrRuntimeError::UnknownSession(session.0)
        );
        runtime.unload_provider("fixture").unwrap();
    }

    #[test]
    fn unsupported_controls_and_languages_are_provider_capabilities() {
        let mut runtime = runtime();
        runtime.register_provider(provider("fixture")).unwrap();
        runtime.load_provider("fixture").unwrap();
        let unsupported_language = AsrSessionConfig {
            language: Some(LanguageId("fr".into())),
            ..AsrSessionConfig::default()
        };
        assert!(matches!(
            runtime
                .start_session("fixture", unsupported_language)
                .unwrap_err(),
            AsrRuntimeError::UnsupportedLanguage { language, .. } if language == "fr"
        ));

        let unsupported_control = AsrSessionConfig {
            decoding: AsrDecodingConfig {
                vocabulary_bias: vec!["Tongues".into()],
                ..AsrDecodingConfig::default()
            },
            ..AsrSessionConfig::default()
        };
        assert!(matches!(
            runtime
                .start_session("fixture", unsupported_control)
                .unwrap_err(),
            AsrRuntimeError::UnsupportedDecodingControl {
                control: AsrDecodingControl::VocabularyBias,
                ..
            }
        ));
    }

    #[test]
    fn chunked_simulation_is_never_advertised_as_native_streaming() {
        let capabilities = capability(
            "windowed",
            AsrStreamingCapability::ChunkedSimulation {
                window_ms: 2_000,
                overlap_ms: 250,
            },
        );
        capabilities.validate().unwrap();
        assert!(!capabilities.streaming.is_native());
        let json = serde_json::to_string(&capabilities).unwrap();
        assert!(json.contains("\"mode\":\"chunked_simulation\""));
        assert!(json.contains("\"window_ms\":2000"));
    }

    #[test]
    fn a_second_provider_registers_without_changing_runtime_contracts() {
        let mut runtime = runtime();
        runtime.register_provider(provider("fixture-a")).unwrap();
        runtime.register_provider(provider("fixture-b")).unwrap();
        runtime.load_provider("fixture-b").unwrap();
        let events = runtime
            .transcribe_offline("fixture-b", AsrSessionConfig::default(), [frame(), frame()])
            .unwrap();
        assert_eq!(committed_transcript(&events), "hello world");
        assert_eq!(
            runtime
                .capabilities()
                .iter()
                .map(|capability| capability.provider_id.as_str())
                .collect::<Vec<_>>(),
            vec!["fixture-a", "fixture-b"]
        );
    }

    #[test]
    fn failed_offline_audio_releases_the_session() {
        let mut runtime = runtime();
        runtime.register_provider(provider("fixture")).unwrap();
        runtime.load_provider("fixture").unwrap();
        let error = runtime
            .transcribe_offline(
                "fixture",
                AsrSessionConfig::default(),
                [AudioFrame::default()],
            )
            .unwrap_err();
        assert!(matches!(error, AsrRuntimeError::Provider { .. }));
        runtime.unload_provider("fixture").unwrap();
    }
}
