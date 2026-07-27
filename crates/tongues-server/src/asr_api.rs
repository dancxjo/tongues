//! Bounded HTTP and WebSocket transports for the provider-neutral ASR runtime.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Json;
use axum::body::Bytes;
use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Query, State, WebSocketUpgrade};
use axum::http::{HeaderMap, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;
use speaking::{
    AsrDecodingControl, AsrProviderCapabilities, AsrResourceLimits, AsrRuntime, AsrRuntimeError,
    AsrSessionConfig, AsrStreamingCapability, AudioFrame, FixtureAsrProvider, FixtureAsrStep,
    LanguageId, StreamEvent, WhisperAsrProvider, committed_transcript,
};
use tokio::sync::Semaphore;
use tongues_audio::AudioBuffer;

const SCHEMA_VERSION: u16 = 1;
pub(crate) const MAX_FILE_BYTES: usize = 32 * 1024 * 1024;
const MAX_CHUNK_BYTES: usize = 256 * 1024;
const MAX_AUDIO_MS: u64 = 5 * 60 * 1_000;
const SESSION_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
const SESSION_CAPACITY: usize = 4;
const FRAME_SAMPLES: usize = 16_000;

#[derive(Clone)]
pub(crate) struct AsrApiState {
    runtime: Arc<Mutex<AsrRuntime>>,
    permits: Arc<Semaphore>,
}

impl AsrApiState {
    pub(crate) fn new() -> anyhow::Result<Self> {
        let mut runtime = AsrRuntime::new(AsrResourceLimits {
            maximum_total_sessions: SESSION_CAPACITY,
            maximum_estimated_memory_mb: 8_192,
        })?;
        runtime.register_provider(fixture_provider()?)?;
        runtime.load_provider("fixture")?;

        let model_path = tongues_cli::models::asr_whisper_model_path()
            .unwrap_or_else(|_| "models/whisper/ggml-large-v3-turbo.bin".into());
        runtime.register_provider(WhisperAsrProvider::new(
            model_path,
            "whisper-large-v3-turbo",
            supported_languages(),
        )?)?;
        Ok(Self {
            runtime: Arc::new(Mutex::new(runtime)),
            permits: Arc::new(Semaphore::new(SESSION_CAPACITY)),
        })
    }

    pub(crate) fn provider_capabilities(&self) -> anyhow::Result<Vec<AsrProviderCapabilities>> {
        self.runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("ASR runtime lock is poisoned"))
            .map(|runtime| runtime.capabilities())
    }
}

fn fixture_provider() -> anyhow::Result<FixtureAsrProvider> {
    FixtureAsrProvider::new(
        AsrProviderCapabilities {
            provider_id: "fixture".into(),
            model_id: "fixture-contract-v1".into(),
            installed: true,
            languages: supported_languages(),
            streaming: AsrStreamingCapability::Native,
            decoding_controls: BTreeSet::from([AsrDecodingControl::Timestamps]),
            maximum_concurrent_sessions: SESSION_CAPACITY,
            estimated_memory_mb_per_session: 1,
            model_license: Some("fixture-only".into()),
            model_checksum: Some("sha256:fixture".into()),
        },
        vec![
            FixtureAsrStep {
                text: "hello".into(),
                confidence: Some(0.75),
                is_final: false,
            },
            FixtureAsrStep {
                text: "hello from Tongues".into(),
                confidence: Some(0.95),
                is_final: false,
            },
        ],
    )
}

fn supported_languages() -> Vec<LanguageId> {
    ["en", "es", "fr", "de", "it", "pt", "ja", "zh"]
        .into_iter()
        .map(|value| LanguageId(value.into()))
        .collect()
}

#[derive(Serialize)]
pub(crate) struct CapabilityResponse {
    schema_version: u16,
    providers: Vec<AsrProviderCapabilities>,
    sources: tongues_audio::AudioInputCapabilities,
    limits: ApiLimits,
    retention: RetentionPolicy,
    streaming: StreamingPolicy,
}

#[derive(Serialize)]
struct ApiLimits {
    maximum_file_bytes: usize,
    maximum_chunk_bytes: usize,
    maximum_audio_ms: u64,
    maximum_sessions: usize,
    idle_timeout_ms: u64,
}

#[derive(Serialize)]
struct RetentionPolicy {
    audio_retained: bool,
    configurable: bool,
}

#[derive(Serialize)]
struct StreamingPolicy {
    web_socket: bool,
    web_rtc: bool,
    resume: &'static str,
    fallback: &'static str,
}

pub(crate) async fn capabilities(State(state): State<super::AppState>) -> Response {
    let providers = match state.asr.runtime.lock() {
        Ok(runtime) => runtime.capabilities(),
        Err(_) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "invalid_state",
                "ASR runtime lock is poisoned",
            );
        }
    };
    Json(CapabilityResponse {
        schema_version: SCHEMA_VERSION,
        providers,
        sources: tongues_audio::input_capabilities(),
        limits: ApiLimits {
            maximum_file_bytes: MAX_FILE_BYTES,
            maximum_chunk_bytes: MAX_CHUNK_BYTES,
            maximum_audio_ms: MAX_AUDIO_MS,
            maximum_sessions: SESSION_CAPACITY,
            idle_timeout_ms: SESSION_IDLE_TIMEOUT.as_millis() as u64,
        },
        retention: RetentionPolicy {
            audio_retained: false,
            configurable: false,
        },
        streaming: StreamingPolicy {
            web_socket: true,
            web_rtc: false,
            resume: "not_supported",
            fallback: "WebSocket float32 PCM is the browser and non-browser live transport",
        },
    })
    .into_response()
}

#[derive(Debug, Deserialize)]
pub(crate) struct TranscribeQuery {
    #[serde(default = "default_provider")]
    provider: String,
    language: Option<String>,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

fn default_provider() -> String {
    "whisper.cpp".into()
}

#[derive(Serialize)]
struct TranscriptionResponse {
    schema_version: u16,
    provider: String,
    transcript: String,
    events: Vec<StreamEvent>,
    audio_retained: bool,
}

pub(crate) async fn transcribe(
    State(state): State<super::AppState>,
    Query(query): Query<TranscribeQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_none_or(|value| {
            !value.eq_ignore_ascii_case("audio/wav") && !value.eq_ignore_ascii_case("audio/x-wav")
        })
    {
        return api_error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "unsupported_format",
            "Content-Type must be audio/wav",
        );
    }
    if body.is_empty() || body.len() > MAX_FILE_BYTES {
        return api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "capacity_exhausted",
            "WAV body is empty or exceeds the file-size limit",
        );
    }
    let audio = match tongues_audio::read_wav_bytes(&body)
        .and_then(|audio| audio.convert_channels(1))
        .and_then(|audio| audio.resample_linear(16_000))
    {
        Ok(audio) => audio,
        Err(error) => {
            return api_error(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_format",
                error.to_string(),
            );
        }
    };
    if duration_ms(&audio) > MAX_AUDIO_MS {
        return api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "capacity_exhausted",
            "audio exceeds the duration limit",
        );
    }
    let permit = match state.asr.permits.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return api_error(
                StatusCode::TOO_MANY_REQUESTS,
                "capacity_exhausted",
                "ASR session capacity is exhausted",
            );
        }
    };
    let runtime = Arc::clone(&state.asr.runtime);
    let provider = query.provider.clone();
    let language = query.language.map(LanguageId);
    let work = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut runtime = runtime
            .lock()
            .map_err(|_| ApiFailure::invalid_state("ASR runtime lock is poisoned"))?;
        ensure_loaded(&mut runtime, &provider)?;
        runtime
            .transcribe_offline(
                &provider,
                AsrSessionConfig {
                    language,
                    ..AsrSessionConfig::default()
                },
                audio_frames(audio),
            )
            .map_err(ApiFailure::from_runtime)
    });
    let timeout = Duration::from_millis(query.timeout_ms.unwrap_or(120_000).clamp(1, 300_000));
    match tokio::time::timeout(timeout, work).await {
        Err(_) => api_error(
            StatusCode::REQUEST_TIMEOUT,
            "timeout",
            "transcription timed out",
        ),
        Ok(Err(error)) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "invalid_state",
            format!("ASR worker failed: {error}"),
        ),
        Ok(Ok(Err(error))) => error.into_response(),
        Ok(Ok(Ok(events))) => Json(TranscriptionResponse {
            schema_version: SCHEMA_VERSION,
            provider: query.provider,
            transcript: committed_transcript(&events),
            events,
            audio_retained: false,
        })
        .into_response(),
    }
}

fn ensure_loaded(runtime: &mut AsrRuntime, provider: &str) -> Result<(), ApiFailure> {
    let capability = runtime
        .capabilities()
        .into_iter()
        .find(|candidate| candidate.provider_id == provider)
        .ok_or_else(|| {
            ApiFailure::unavailable(format!("ASR provider `{provider}` is unavailable"))
        })?;
    if !capability.installed {
        return Err(ApiFailure::unavailable(format!(
            "ASR model `{}` is not installed",
            capability.model_id
        )));
    }
    match runtime.load_provider(provider) {
        Ok(()) => Ok(()),
        Err(AsrRuntimeError::Provider { message, .. }) if message.contains("already") => Ok(()),
        Err(error) => Err(ApiFailure::from_runtime(error)),
    }
}

fn duration_ms(audio: &AudioBuffer) -> u64 {
    (audio.frames() as u64)
        .saturating_mul(1_000)
        .checked_div(u64::from(audio.sample_rate_hz))
        .unwrap_or(u64::MAX)
}

fn audio_frames(audio: AudioBuffer) -> Vec<AudioFrame> {
    audio
        .samples
        .chunks(FRAME_SAMPLES)
        .map(|samples| AudioFrame {
            sample_rate_hz: audio.sample_rate_hz,
            channels: audio.channels,
            samples: samples.to_vec(),
        })
        .collect()
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
enum ClientControl {
    Open {
        schema_version: u16,
        #[serde(default = "default_provider")]
        provider: String,
        sample_rate_hz: u32,
        channels: u16,
        language: Option<String>,
        #[serde(default)]
        resume_session_id: Option<String>,
    },
    End,
    Cancel {
        #[serde(default)]
        reason: Option<String>,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", tag = "type")]
enum ServerMessage {
    Ready {
        schema_version: u16,
        session_id: String,
        queue_capacity_chunks: usize,
        maximum_chunk_bytes: usize,
        maximum_audio_ms: u64,
        audio_retained: bool,
    },
    Recognition {
        sequence: u64,
        event: StreamEvent,
    },
    Ended {
        session_id: String,
    },
    Error {
        code: &'static str,
        message: String,
    },
}

pub(crate) async fn stream_upgrade(
    State(state): State<super::AppState>,
    headers: HeaderMap,
    uri: Uri,
    upgrade: WebSocketUpgrade,
) -> Response {
    if let Err(error) =
        super::validate_same_origin(&headers, uri.authority().map(|value| value.as_str()))
    {
        return api_error(StatusCode::FORBIDDEN, "origin_forbidden", error);
    }
    let permit = match state.asr.permits.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return api_error(
                StatusCode::TOO_MANY_REQUESTS,
                "capacity_exhausted",
                "ASR session capacity is exhausted",
            );
        }
    };
    upgrade.on_upgrade(move |socket| stream_session(socket, state.asr, permit))
}

async fn stream_session(
    mut socket: WebSocket,
    state: AsrApiState,
    _permit: tokio::sync::OwnedSemaphorePermit,
) {
    let open = match tokio::time::timeout(SESSION_IDLE_TIMEOUT, socket.recv()).await {
        Ok(Some(Ok(Message::Text(text)))) => serde_json::from_str::<ClientControl>(&text),
        Ok(_) => {
            return send_ws_error(
                &mut socket,
                "open_required",
                "the first message must open the ASR stream",
            )
            .await;
        }
        Err(_) => return send_ws_error(&mut socket, "timeout", "ASR stream open timed out").await,
    };
    let Ok(ClientControl::Open {
        schema_version,
        provider,
        sample_rate_hz,
        channels,
        language,
        resume_session_id,
    }) = open
    else {
        return send_ws_error(
            &mut socket,
            "open_required",
            "the first message must be a valid open control",
        )
        .await;
    };
    if resume_session_id.is_some() {
        return send_ws_error(
            &mut socket,
            "invalid_state",
            "reconnect/resume is not supported; open a new session",
        )
        .await;
    }
    if schema_version != SCHEMA_VERSION || sample_rate_hz == 0 || channels != 1 {
        return send_ws_error(
            &mut socket,
            "unsupported_format",
            "schema version 1 mono float32 PCM with a positive sample rate is required",
        )
        .await;
    }
    let session = match start_stream_session(&state, &provider, language) {
        Ok(session) => session,
        Err(error) => return send_ws_failure(&mut socket, error).await,
    };
    let session_name = session.0.clone();
    if send_ws(
        &mut socket,
        &ServerMessage::Ready {
            schema_version: SCHEMA_VERSION,
            session_id: session_name.clone(),
            queue_capacity_chunks: 1,
            maximum_chunk_bytes: MAX_CHUNK_BYTES,
            maximum_audio_ms: MAX_AUDIO_MS,
            audio_retained: false,
        },
    )
    .await
    .is_err()
    {
        cancel(&state, &session, "client disconnected before ready");
        return;
    }

    let mut sequence = 0_u64;
    let mut frames_received = 0_u64;
    loop {
        let message = match tokio::time::timeout(SESSION_IDLE_TIMEOUT, socket.recv()).await {
            Ok(Some(Ok(message))) => message,
            Ok(_) => {
                cancel(&state, &session, "client disconnected");
                return;
            }
            Err(_) => {
                cancel(&state, &session, "idle timeout");
                return send_ws_error(&mut socket, "timeout", "ASR stream idle timeout").await;
            }
        };
        match message {
            Message::Binary(bytes) => {
                let samples = match decode_f32(&bytes) {
                    Ok(samples) => samples,
                    Err(error) => {
                        cancel(&state, &session, "invalid audio");
                        return send_ws_error(&mut socket, "unsupported_format", error).await;
                    }
                };
                frames_received = frames_received.saturating_add(samples.len() as u64);
                if frames_received.saturating_mul(1_000) / u64::from(sample_rate_hz) > MAX_AUDIO_MS
                {
                    cancel(&state, &session, "duration limit");
                    return send_ws_error(
                        &mut socket,
                        "capacity_exhausted",
                        "audio exceeds the duration limit",
                    )
                    .await;
                }
                let events = with_runtime(&state, |runtime| {
                    runtime.push_audio(
                        &session,
                        &AudioFrame {
                            sample_rate_hz,
                            channels,
                            samples,
                        },
                    )
                });
                match events {
                    Ok(events) => {
                        for event in events {
                            if send_ws(&mut socket, &ServerMessage::Recognition { sequence, event })
                                .await
                                .is_err()
                            {
                                cancel(&state, &session, "client disconnected");
                                return;
                            }
                            sequence = sequence.saturating_add(1);
                        }
                    }
                    Err(error) => {
                        cancel(&state, &session, "recognition failed");
                        return send_ws_failure(&mut socket, ApiFailure::from_runtime(error)).await;
                    }
                }
            }
            Message::Text(text) => match serde_json::from_str::<ClientControl>(&text) {
                Ok(ClientControl::End) => {
                    let events = with_runtime(&state, |runtime| runtime.finish_session(&session));
                    match events {
                        Ok(events) => {
                            for event in events {
                                if send_ws(
                                    &mut socket,
                                    &ServerMessage::Recognition { sequence, event },
                                )
                                .await
                                .is_err()
                                {
                                    return;
                                }
                                sequence = sequence.saturating_add(1);
                            }
                        }
                        Err(error) => {
                            return send_ws_failure(&mut socket, ApiFailure::from_runtime(error))
                                .await;
                        }
                    }
                    let _ = send_ws(
                        &mut socket,
                        &ServerMessage::Ended {
                            session_id: session_name,
                        },
                    )
                    .await;
                    return;
                }
                Ok(ClientControl::Cancel { reason }) => {
                    let reason = reason.unwrap_or_else(|| "client cancelled".into());
                    let events =
                        with_runtime(&state, |runtime| runtime.cancel_session(&session, &reason));
                    if let Ok(events) = events {
                        for event in events {
                            let _ = send_ws(
                                &mut socket,
                                &ServerMessage::Recognition { sequence, event },
                            )
                            .await;
                            sequence = sequence.saturating_add(1);
                        }
                    }
                    let _ = send_ws(
                        &mut socket,
                        &ServerMessage::Ended {
                            session_id: session_name,
                        },
                    )
                    .await;
                    return;
                }
                _ => {
                    cancel(&state, &session, "invalid state");
                    return send_ws_error(&mut socket, "invalid_state", "session is already open")
                        .await;
                }
            },
            Message::Close(_) => {
                cancel(&state, &session, "client disconnected");
                return;
            }
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }
}

fn start_stream_session(
    state: &AsrApiState,
    provider: &str,
    language: Option<String>,
) -> Result<speaking::AsrSessionId, ApiFailure> {
    let mut runtime = state
        .runtime
        .lock()
        .map_err(|_| ApiFailure::invalid_state("ASR runtime lock is poisoned"))?;
    let streaming = runtime
        .capabilities()
        .into_iter()
        .find(|candidate| candidate.provider_id == provider)
        .ok_or_else(|| {
            ApiFailure::unavailable(format!("ASR provider `{provider}` is unavailable"))
        })?
        .streaming;
    if matches!(streaming, AsrStreamingCapability::OfflineOnly) {
        return Err(ApiFailure {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            code: "unsupported_configuration",
            message: format!("ASR provider `{provider}` only supports offline transcription"),
        });
    }
    ensure_loaded(&mut runtime, provider)?;
    runtime
        .start_session(
            provider,
            AsrSessionConfig {
                language: language.map(LanguageId),
                ..AsrSessionConfig::default()
            },
        )
        .map_err(ApiFailure::from_runtime)
}

fn with_runtime<T>(
    state: &AsrApiState,
    operation: impl FnOnce(&mut AsrRuntime) -> Result<T, AsrRuntimeError>,
) -> Result<T, AsrRuntimeError> {
    let mut runtime = state
        .runtime
        .lock()
        .map_err(|_| AsrRuntimeError::ResourceExhausted("ASR runtime lock is poisoned".into()))?;
    operation(&mut runtime)
}

fn cancel(state: &AsrApiState, session: &speaking::AsrSessionId, reason: &str) {
    let _ = with_runtime(state, |runtime| runtime.cancel_session(session, reason));
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.is_empty() || bytes.len() > MAX_CHUNK_BYTES || !bytes.len().is_multiple_of(4) {
        return Err("audio chunk must contain at most 256 KiB of complete float32 PCM".into());
    }
    bytes
        .chunks_exact(4)
        .map(|word| {
            let sample = f32::from_le_bytes(word.try_into().expect("four-byte chunk"));
            sample
                .is_finite()
                .then_some(sample)
                .ok_or_else(|| "audio chunk contains a non-finite sample".into())
        })
        .collect()
}

async fn send_ws(socket: &mut WebSocket, message: &ServerMessage) -> Result<(), axum::Error> {
    socket
        .send(Message::Text(
            serde_json::to_string(message)
                .expect("serializable ASR message")
                .into(),
        ))
        .await
}

async fn send_ws_error(socket: &mut WebSocket, code: &'static str, message: impl Into<String>) {
    let _ = send_ws(
        socket,
        &ServerMessage::Error {
            code,
            message: message.into(),
        },
    )
    .await;
}

async fn send_ws_failure(socket: &mut WebSocket, failure: ApiFailure) {
    let _ = send_ws(
        socket,
        &ServerMessage::Error {
            code: failure.code,
            message: failure.message,
        },
    )
    .await;
}

struct ApiFailure {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiFailure {
    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "unavailable_model",
            message: message.into(),
        }
    }

    fn invalid_state(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code: "invalid_state",
            message: message.into(),
        }
    }

    fn from_runtime(error: AsrRuntimeError) -> Self {
        match error {
            AsrRuntimeError::UnknownProvider(_) | AsrRuntimeError::ProviderNotLoaded(_) => {
                Self::unavailable(error.to_string())
            }
            AsrRuntimeError::ResourceExhausted(_) => Self {
                status: StatusCode::TOO_MANY_REQUESTS,
                code: "capacity_exhausted",
                message: error.to_string(),
            },
            AsrRuntimeError::UnknownSession(_) | AsrRuntimeError::ProviderBusy(_) => {
                Self::invalid_state(error.to_string())
            }
            AsrRuntimeError::UnsupportedLanguage { .. }
            | AsrRuntimeError::UnsupportedDecodingControl { .. } => Self {
                status: StatusCode::UNPROCESSABLE_ENTITY,
                code: "unsupported_configuration",
                message: error.to_string(),
            },
            AsrRuntimeError::Provider { .. } => Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                code: "provider_error",
                message: error.to_string(),
            },
        }
    }

    fn into_response(self) -> Response {
        api_error(self.status, self.code, self.message)
    }
}

fn api_error(status: StatusCode, code: &'static str, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({ "error": { "code": code, "message": message.into() } })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_contract_is_bounded_and_retention_free() {
        let state = AsrApiState::new().unwrap();
        let runtime = state.runtime.lock().unwrap();
        let providers = runtime.capabilities();
        assert!(
            providers
                .iter()
                .any(|provider| provider.provider_id == "fixture")
        );
        assert!(
            providers
                .iter()
                .any(|provider| provider.provider_id == "whisper.cpp")
        );
        assert_eq!(SESSION_CAPACITY, 4);
    }

    #[test]
    fn binary_contract_rejects_oversize_incomplete_and_non_finite_chunks() {
        assert!(decode_f32(&vec![0; MAX_CHUNK_BYTES + 4]).is_err());
        assert!(decode_f32(&[0, 1, 2]).is_err());
        assert!(decode_f32(&f32::NAN.to_le_bytes()).is_err());
    }

    #[test]
    fn fixture_events_are_ordered_and_terminal() {
        let state = AsrApiState::new().unwrap();
        let mut runtime = state.runtime.lock().unwrap();
        let session = runtime
            .start_session("fixture", AsrSessionConfig::default())
            .unwrap();
        let first = runtime
            .push_audio(
                &session,
                &AudioFrame {
                    sample_rate_hz: 16_000,
                    channels: 1,
                    samples: vec![0.0; 160],
                },
            )
            .unwrap();
        let second = runtime.finish_session(&session).unwrap();
        assert!(matches!(
            first.first(),
            Some(StreamEvent::PartialHypothesis { .. })
        ));
        assert!(matches!(second.last(), Some(StreamEvent::Completed)));
        assert!(matches!(
            runtime.finish_session(&session),
            Err(AsrRuntimeError::UnknownSession(_))
        ));
    }

    #[test]
    fn cancellation_releases_runtime_session_capacity() {
        let state = AsrApiState::new().unwrap();
        let mut runtime = state.runtime.lock().unwrap();
        let session = runtime
            .start_session("fixture", AsrSessionConfig::default())
            .unwrap();
        let events = runtime.cancel_session(&session, "test").unwrap();
        assert!(matches!(events.last(), Some(StreamEvent::Cancelled { .. })));
        assert!(
            runtime
                .start_session("fixture", AsrSessionConfig::default())
                .is_ok()
        );
    }
}
