use anyhow::Context as _;
use axum::{
    Json, Router,
    extract::DefaultBodyLimit,
    extract::{Path, Query, Request, State},
    http::{Method, StatusCode, header},
    middleware::{self, Next},
    response::{
        Html, IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use axum_server::tls_rustls::RustlsConfig;
use base64::Engine as _;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn_cuda::{Cuda, CudaDevice};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::any::Any;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::io::{BufRead, BufReader, Write};
use std::net::SocketAddr;
use std::panic;
use std::path::{Component, Path as FsPath, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU8, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{Semaphore, broadcast};
use tokio_stream::{StreamExt, wrappers::BroadcastStream};
use tongues_duplex::{
    DuplexFixtureSuite, DuplexSimulator, DuplexStudioProjection, FixtureCompletionProvider,
    ObservedEvidence, OracleCompletionProvider, SimulatorConfig, SimulatorJournal,
    studio_projection_from_journal,
};
use tower_http::services::ServeDir;

mod live;

const STYLE_VECTOR_DIMS: usize = 256;
const DEFAULT_DUPLEX_FIXTURES_PATH: &str = "fixtures/duplex/completion_scenarios_v1.json";
const STYLETTS2_REFERENCE_RELATIVE_DIR: &str = "models/styletts2/en-us/reference_audio";
const VITS_SPEAKER_RELATIVE_PATH: &str = "models/speech/coqui/en/vctk/vits/speaker_ids.json";
const SPEEDY_RELATIVE_DIR: &str = "models/speech/coqui/en/ljspeech/speedy-speech";
const FASTPITCH_RELATIVE_DIR: &str = "models/speech/coqui/en/ljspeech/fast-pitch";
const GLOW_RELATIVE_DIR: &str = "models/speech/coqui/en/ljspeech/glow-tts";
const HIFIGAN_RELATIVE_DIR: &str = "models/speech/coqui/en/ljspeech/hifigan-v2";
const MULTIBAND_RELATIVE_DIR: &str = "models/speech/coqui/en/ljspeech/multiband-melgan";
const VITS_RELATIVE_DIR: &str = "models/speech/coqui/en/vctk/vits";
const YOURTTS_RELATIVE_DIR: &str = "models/speech/coqui/multilingual/yourtts";
const FREEVC_RELATIVE_DIR: &str = "models/speech/coqui/multilingual/freevc24";
const VITS_SPEAKER_COUNT: u32 = 109;
const JOB_OUTPUT_LIMIT: usize = 1_000;
const DEFAULT_HTTP_PORT: u16 = 3000;
const DEFAULT_HTTPS_PORT: u16 = 8443;
const DEFAULT_HOST: &str = "127.0.0.1";
const FILE_LIST_LIMIT: usize = 500;
const DEFAULT_SPEECH_MAX_IN_FLIGHT: usize = 2;
const MAX_REQUEST_BODY_BYTES: usize = 256 * 1024;
const MAX_DOWNLOAD_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ACTIVE_JOBS: usize = 4;
const INSECURE_REMOTE_ENV: &str = "TONGUES_ALLOW_INSECURE_REMOTE";
const ALLOWED_ARTIFACT_ROOTS: &[&str] = &[
    "archive", "data", "datasets", "models", "outputs", "releases", "runs",
];
const ALLOWED_ROOT_ARTIFACT_FILES: &[&str] = &[
    "emotion_signatures.json",
    "labels.jsonl",
    "style_vectors.jsonl",
];
const DEFAULT_SPEECH_DISCOVERY_PAGE_LIMIT: usize = 32;
const MAX_SPEECH_DISCOVERY_PAGE_LIMIT: usize = 100;
const DEFAULT_STYLETTS2_VOICE_REFERENCE: &str = "1221-135767-0014.wav";
const DEFAULT_STYLETTS2_STYLE_REFERENCE: &str = "amused.wav";
const ONNX_VOICE_RELATIVE_DIR: &str = "models/voices";
const DEFAULT_ONNX_VOICE_MODEL: &str = "voice-ljspeech-high";
const ONNX_VOICE_MODELS: &[OnnxVoiceModel] = &[
    OnnxVoiceModel {
        id: "voice-ljspeech-high",
        display_name: "LJSpeech High",
        filename: "en_US-ljspeech-high.onnx",
    },
    OnnxVoiceModel {
        id: "voice-ryan-medium",
        display_name: "Ryan Medium",
        filename: "en_US-ryan-medium.onnx",
    },
    OnnxVoiceModel {
        id: "voice-amy-medium",
        display_name: "Amy Medium",
        filename: "en_US-amy-medium.onnx",
    },
];
const SPEECH_PHASE_IDLE: u8 = 0;
const SPEECH_PHASE_LOADING: u8 = 1;
const SPEECH_PHASE_SYNTHESIZING: u8 = 2;
const SPEECH_PHASE_RELOADING: u8 = 3;

struct OnnxVoiceModel {
    id: &'static str,
    display_name: &'static str,
    filename: &'static str,
}

#[derive(Clone)]
struct AppState {
    workspace_root: PathBuf,
    static_dir: PathBuf,
    jobs: JobRegistry,
    speech: ResidentSpeechRegistry,
    speech_admission: SpeechAdmission,
    speech_phase: Arc<AtomicU8>,
    speech_device: tongues_tts::ResolvedSpeechDevice,
    live_turns: Arc<Mutex<HashMap<String, Arc<std::sync::atomic::AtomicBool>>>>,
}

type JobRegistry = Arc<Mutex<HashMap<String, JobRecord>>>;
type ResidentSpeechRegistry = Arc<Mutex<ResidentSpeechService>>;
type SpeechVerification = BTreeMap<String, tongues_tts::ModelVerificationState>;

#[derive(Clone)]
struct SpeechAdmission {
    permits: Arc<Semaphore>,
    capacity: usize,
}

impl SpeechAdmission {
    fn new(capacity: usize) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(capacity)),
            capacity,
        }
    }

    fn try_acquire(
        &self,
    ) -> Result<tokio::sync::OwnedSemaphorePermit, tokio::sync::TryAcquireError> {
        Arc::clone(&self.permits).try_acquire_owned()
    }

    fn counts(&self, active: bool) -> (usize, usize) {
        let admitted = self
            .capacity
            .saturating_sub(self.permits.available_permits());
        let active = usize::from(active && admitted > 0);
        (active, admitted.saturating_sub(active))
    }
}

#[derive(Debug)]
struct StartupError {
    stage: &'static str,
    detail: String,
}

impl StartupError {
    fn new(stage: &'static str, detail: impl Into<String>) -> Self {
        Self {
            stage,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for StartupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Startup failed while {}: {}", self.stage, self.detail)
    }
}

#[tokio::main]
async fn main() {
    if let Err(error) = run_server().await {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

async fn run_server() -> Result<(), StartupError> {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let workspace_root = std::env::current_dir()
        .map_err(|error| StartupError::new("resolving the workspace root", error.to_string()))?;
    let static_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("public");
    let cert_dir = workspace_root.join(".certs");
    let speech_max_in_flight = speech_max_in_flight();
    let requested_speech_device = std::env::var("TONGUES_SPEECH_DEVICE")
        .unwrap_or_else(|_| "auto".into())
        .parse::<tongues_tts::SpeechDeviceRequest>()
        .unwrap_or_else(|error| {
            eprintln!("Invalid TONGUES_SPEECH_DEVICE: {error}");
            std::process::exit(2);
        });
    let speech_device_selection =
        tongues_tts::resolve_speech_device(requested_speech_device, |index| {
            cuda_probe_failure_reason(index).map_or(Ok(()), Err)
        })
        .unwrap_or_else(|error| {
            eprintln!("Invalid resident speech device: {error}");
            std::process::exit(2);
        });
    if let Some(reason) = speech_device_selection.fallback_reason.as_deref() {
        eprintln!(
            "Warning: CUDA device 0 is not available ({reason}). Resident speech is falling back to CPU."
        );
    }
    let speech_device = speech_device_selection.resolved;
    let state = AppState {
        workspace_root: workspace_root.clone(),
        static_dir: static_dir.clone(),
        jobs: Arc::new(Mutex::new(HashMap::new())),
        speech: Arc::new(Mutex::new(ResidentSpeechService::default())),
        speech_admission: SpeechAdmission::new(speech_max_in_flight),
        speech_phase: Arc::new(AtomicU8::new(SPEECH_PHASE_IDLE)),
        speech_device,
        live_turns: Arc::new(Mutex::new(HashMap::new())),
    };

    let app = build_app(state);

    let host = std::env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.into());
    let http_port = std::env::var("PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(DEFAULT_HTTP_PORT);
    let https_port = std::env::var("HTTPS_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(DEFAULT_HTTPS_PORT);

    let http_addr = format!("{host}:{http_port}")
        .parse::<SocketAddr>()
        .map_err(|error| StartupError::new("parsing the HTTP bind address", error.to_string()))?;
    let https_addr = format!("{host}:{https_port}")
        .parse::<SocketAddr>()
        .map_err(|error| StartupError::new("parsing the HTTPS bind address", error.to_string()))?;
    validate_bind_address(http_addr)?;
    validate_bind_address(https_addr)?;

    ensure_self_signed_cert(&cert_dir)?;
    let tls_config = RustlsConfig::from_pem_file(
        cert_dir.join("tongues-local.crt"),
        cert_dir.join("tongues-local.key"),
    )
    .await
    .map_err(|error| StartupError::new("loading the local TLS certificate", error.to_string()))?;

    println!("Web server listening on http://{http_addr}");
    println!("Web server listening on https://{https_addr}");
    println!("Self-signed certificate: {}", cert_dir.display());
    if !http_addr.ip().is_loopback() || !https_addr.ip().is_loopback() {
        println!(
            "Warning: remote bind enabled via {INSECURE_REMOTE_ENV}=1; API protections remain development-only."
        );
    }
    println!(
        "Resident speech admission: 1 active + {} queued",
        speech_max_in_flight.saturating_sub(1)
    );
    println!("Resident speech device: {}", speech_device.display_name());

    let http_app = app.clone();
    let http_listener = tokio::net::TcpListener::bind(http_addr)
        .await
        .map_err(|error| StartupError::new("binding the HTTP listener", error.to_string()))?;
    let http = async move {
        axum::serve(http_listener, http_app)
            .await
            .map_err(|error| StartupError::new("serving the HTTP listener", error.to_string()))
    };
    let https = async move {
        axum_server::bind_rustls(https_addr, tls_config)
            .serve(app.into_make_service())
            .await
            .map_err(|error| StartupError::new("serving the HTTPS listener", error.to_string()))
    };

    let (http_result, https_result) = tokio::join!(http, https);
    http_result?;
    https_result?;
    Ok(())
}

fn build_app(state: AppState) -> Router {
    let static_dir = state.static_dir.clone();
    Router::new()
        .route("/api/emotions", get(get_emotions))
        .route("/api/files", get(list_files))
        .route("/api/files/download/{*path}", get(download_file))
        .route("/api/jobs", get(list_jobs).post(start_job))
        .route("/api/jobs/{job_id}", get(get_job))
        .route("/api/jobs/{job_id}/cancel", post(cancel_job))
        .route("/api/jobs/{job_id}/events", get(job_events))
        .route(
            "/api/pronunciation-demo/models",
            get(get_pronunciation_models),
        )
        .route("/api/pronunciation-demo/infer", post(pronunciation_infer))
        .route("/api/linguistic/varieties", get(get_linguistic_varieties))
        .route("/api/styletts2-samples", get(get_styletts2_samples))
        .route("/api/models/catalog", get(get_model_catalog))
        .route("/api/speech/models", get(get_speech_models))
        .route(
            "/api/speech/models/verify/{model_id}",
            post(verify_speech_model),
        )
        .route("/api/duplex/project", post(project_duplex_request))
        .route("/api/speech/project", post(project_speech_request))
        .route("/api/speech/speakers", get(get_speech_speakers))
        .route("/api/speech/runtime", get(get_speech_runtime))
        .route("/api/speech/runtime/reload", post(reload_speech_runtime))
        .route("/api/speech/runtime/unload", post(unload_speech_runtime))
        .route("/api/live/providers", get(get_live_providers))
        .route("/api/live/turn", post(start_live_turn))
        .route("/api/live/turn/{turn_id}/cancel", post(cancel_live_turn))
        .route(
            "/api/styletts2-reference-audio/{*sample_id}",
            get(get_styletts2_reference_audio),
        )
        .route("/api/speak", post(speak))
        .route("/", get(serve_app_index))
        .route("/styletts2", get(serve_app_index))
        .route("/styletts2/", get(serve_app_index))
        .route("/speech", get(serve_app_index))
        .route("/speech/", get(serve_app_index))
        .route("/speech/{*path}", get(serve_app_index))
        .route("/jobs", get(serve_app_index))
        .route("/jobs/", get(serve_app_index))
        .route("/pronunciation-demo", get(serve_app_index))
        .route("/pronunciation-demo/", get(serve_app_index))
        .route("/g2p2g/{*path}", get(serve_app_index))
        .route("/sentence-parser/{*path}", get(serve_app_index))
        .route("/head2phones/{*path}", get(serve_app_index))
        .route("/interpretation/{*path}", get(serve_app_index))
        .route("/emotions/{*path}", get(serve_app_index))
        .route("/wiktionary/{*path}", get(serve_app_index))
        .route("/models/{*path}", get(serve_app_index))
        .route("/cli/{*path}", get(serve_app_index))
        .fallback_service(ServeDir::new(static_dir))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .layer(middleware::from_fn(enforce_request_policy))
        .with_state(state)
}

async fn get_live_providers() -> impl IntoResponse {
    Json(json!({
        "providers": live::provider_discovery().await,
    }))
}

#[derive(Deserialize)]
struct LiveTurnStartRequest {
    #[serde(flatten)]
    turn: live::ChatTurnRequest,
    synthesis: SpeakRequest,
}

async fn start_live_turn(
    State(state): State<AppState>,
    Json(mut request): Json<LiveTurnStartRequest>,
) -> Response {
    let turn_id = request.turn.turn_id.trim().to_string();
    if turn_id.is_empty()
        || turn_id.len() > 96
        || !turn_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return (
            StatusCode::BAD_REQUEST,
            "turn_id must contain only ASCII letters, digits, hyphens, or underscores",
        )
            .into_response();
    }
    if request.turn.model.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "model is required").into_response();
    }
    if request.turn.messages.is_empty()
        || !request
            .turn
            .messages
            .iter()
            .any(|message| message.role == "user" && !message.content.trim().is_empty())
    {
        return (
            StatusCode::BAD_REQUEST,
            "a non-empty user message is required",
        )
            .into_response();
    }
    if request.turn.messages.iter().any(|message| {
        !matches!(message.role.as_str(), "user" | "assistant" | "system")
            || message.content.len() > 64 * 1024
    }) {
        return (
            StatusCode::BAD_REQUEST,
            "messages must use user, assistant, or system roles and remain under 64 KiB",
        )
            .into_response();
    }
    request.synthesis.text = "Live speech validation probe.".into();
    let synthesis = match normalize_speak_request(request.synthesis) {
        Ok(synthesis) => synthesis,
        Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
    };
    if let Err(error) = validate_speak_request(&synthesis) {
        return (StatusCode::BAD_REQUEST, error).into_response();
    }

    let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let mut turns = match state.live_turns.lock() {
            Ok(turns) => turns,
            Err(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "live turn registry lock is poisoned",
                )
                    .into_response();
            }
        };
        if turns.contains_key(&turn_id) {
            return (StatusCode::CONFLICT, "turn_id is already active").into_response();
        }
        turns.insert(turn_id.clone(), Arc::clone(&cancelled));
    }

    let source = live::spawn_turn(request.turn, Arc::clone(&cancelled));
    let (stream_tx, stream_rx) = tokio::sync::mpsc::channel::<String>(64);
    let registry = Arc::clone(&state.live_turns);
    let coordinator_state = state.clone();
    let coordinator_turn_id = turn_id.clone();
    tokio::spawn(async move {
        let mut source = source;
        let (synthesis_tx, mut synthesis_rx) =
            tokio::sync::mpsc::unbounded_channel::<(usize, String)>();
        let mut synthesis_tx = Some(synthesis_tx);
        let (audio_tx, mut audio_rx) = tokio::sync::mpsc::channel::<live::TurnEvent>(16);
        let synthesis_state = coordinator_state.clone();
        let synthesis_cancelled = Arc::clone(&cancelled);
        let synthesis_turn_id = coordinator_turn_id.clone();
        let synthesis_worker = tokio::spawn(async move {
            while let Some((segment_id, text)) = synthesis_rx.recv().await {
                if synthesis_cancelled.load(Ordering::Acquire) {
                    break;
                }
                if audio_tx
                    .send(live::TurnEvent::SynthesisStarted {
                        turn_id: synthesis_turn_id.clone(),
                        segment_id,
                        text: text.clone(),
                        started_at_ms: live_event_time_ms(),
                    })
                    .await
                    .is_err()
                {
                    break;
                }
                let mut segment_request = synthesis.clone();
                segment_request.text = text.clone();
                match synthesize_live_speech(&synthesis_state, segment_request).await {
                    Ok(output) if !synthesis_cancelled.load(Ordering::Acquire) => {
                        let metadata = json!({
                            "engine": output.engine_key,
                            "device": output.device.kind(),
                            "device_index": output.device.index(),
                            "sample_rate_hz": output.sample_rate_hz,
                            "channels": output.channels,
                            "sample_count": output.sample_count,
                            "duration_seconds": output.audio_seconds,
                            "queue_ms": output.queue_ms,
                            "model_load_ms": output.load_ms,
                            "synthesis_ms": output.synthesis_ms,
                            "real_time_factor": output.real_time_factor,
                            "resident_model_reused": !output.loaded_now,
                            "pronunciation_warnings": output.pronunciation_warnings,
                        });
                        let event = live::TurnEvent::AudioSegmentReady {
                            turn_id: synthesis_turn_id.clone(),
                            segment_id,
                            text,
                            audio_base64: base64::engine::general_purpose::STANDARD
                                .encode(output.wav),
                            content_type: "audio/wav",
                            sample_rate_hz: output.sample_rate_hz,
                            duration_seconds: output.audio_seconds,
                            synthesis_ms: output.synthesis_ms,
                            speech_metadata: metadata,
                            ready_at_ms: live_event_time_ms(),
                        };
                        if audio_tx.send(event).await.is_err() {
                            break;
                        }
                    }
                    Ok(_) => break,
                    Err(message) => {
                        let _ = audio_tx
                            .send(live::TurnEvent::TurnFailed {
                                turn_id: synthesis_turn_id.clone(),
                                message,
                                failed_at_ms: live_event_time_ms(),
                            })
                            .await;
                        synthesis_cancelled.store(true, Ordering::Release);
                        break;
                    }
                }
            }
        });
        let mut generation_done: Option<(String, String)> = None;
        let mut source_open = true;
        let mut audio_open = true;
        let mut audio_segments = 0;
        while source_open || audio_open {
            tokio::select! {
                event = source.recv(), if source_open => match event {
                    Some(ref event @ live::TurnEvent::SegmentCommitted {
                        segment_id,
                        ref text,
                        ..
                    }) => {
                        if send_live_event(&stream_tx, &event).await.is_err() {
                            cancelled.store(true, Ordering::Release);
                            break;
                        }
                        if synthesis_tx
                            .as_ref()
                            .expect("synthesis queue is open while generation is open")
                            .send((segment_id, text.clone()))
                            .is_err()
                        {
                            break;
                        }
                    }
                    Some(ref event @ live::TurnEvent::GenerationCompleted {
                        ref generated_text,
                        ref committed_text,
                        ..
                    }) => {
                        generation_done = Some((generated_text.clone(), committed_text.clone()));
                        let _ = send_live_event(&stream_tx, &event).await;
                        source_open = false;
                        synthesis_tx.take();
                    }
                    Some(event) => {
                        if send_live_event(&stream_tx, &event).await.is_err() {
                            cancelled.store(true, Ordering::Release);
                            break;
                        }
                    }
                    None => {
                        source_open = false;
                        synthesis_tx.take();
                    }
                },
                event = audio_rx.recv(), if audio_open => match event {
                    Some(event) => {
                        if matches!(event, live::TurnEvent::AudioSegmentReady { .. }) {
                            audio_segments += 1;
                        }
                        if send_live_event(&stream_tx, &event).await.is_err() {
                            cancelled.store(true, Ordering::Release);
                            break;
                        }
                    }
                    None => audio_open = false,
                },
            }
            if !source_open && synthesis_worker.is_finished() && audio_rx.is_empty() {
                audio_open = false;
            }
        }
        synthesis_tx.take();
        let _ = synthesis_worker.await;
        while let Ok(event) = audio_rx.try_recv() {
            if matches!(event, live::TurnEvent::AudioSegmentReady { .. }) {
                audio_segments += 1;
            }
            let _ = send_live_event(&stream_tx, &event).await;
        }
        if !cancelled.load(Ordering::Acquire)
            && let Some((generated_text, committed_text)) = generation_done
        {
            let _ = send_live_event(
                &stream_tx,
                &live::TurnEvent::TurnCompleted {
                    turn_id: coordinator_turn_id,
                    generated_text,
                    committed_text,
                    audio_segments,
                    completed_at_ms: live_event_time_ms(),
                },
            )
            .await;
        }
        if let Ok(mut turns) = registry.lock() {
            turns.remove(&turn_id);
        }
    });
    let body = axum::body::Body::from_stream(
        tokio_stream::wrappers::ReceiverStream::new(stream_rx)
            .map(Ok::<String, std::convert::Infallible>),
    );
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .header(header::CACHE_CONTROL, "no-store")
        .header("X-Accel-Buffering", "no")
        .body(body)
        .unwrap()
}

async fn send_live_event(
    sink: &tokio::sync::mpsc::Sender<String>,
    event: &live::TurnEvent,
) -> Result<(), tokio::sync::mpsc::error::SendError<String>> {
    let line = serde_json::to_string(event).unwrap_or_else(|error| {
        serde_json::to_string(&json!({
            "type": "turn_failed",
            "message": format!("serializing live event failed: {error}"),
        }))
        .unwrap()
    });
    sink.send(format!("{line}\n")).await
}

fn live_event_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic payload".into())
}

async fn synthesize_live_speech(
    state: &AppState,
    payload: SpeakRequest,
) -> Result<ResidentSynthesisOutput, String> {
    validate_speak_request(&payload)?;
    let context = resident_synthesis_context(state, &payload)?;
    let speech_device = resident_speech_device_for(&payload, state.speech_device)
        .map_err(|error| error.to_string())?;
    let capabilities = speech_backend_capabilities(
        &resolve_mortar_home(),
        payload.backend.as_deref().unwrap_or("burn"),
        payload.model.as_deref(),
        speech_device,
        payload.sample_rate_hz.unwrap_or(24_000),
    )
    .map_err(|error| error.to_string())?;
    validate_declared_speech_controls(
        &payload,
        &speech_control_discovery(
            payload.backend.as_deref().unwrap_or("burn"),
            &capabilities,
            speech_device,
        ),
    )?;
    capabilities
        .validate(&unified_synthesis_request(
            &payload,
            &context,
            speech_device,
        ))
        .map_err(|error| error.to_string())?;
    let backend = payload.backend.as_deref().unwrap_or("burn");
    if let Some(error) =
        speech_backend_installation_error(&resolve_mortar_home(), backend, payload.model.as_deref())
    {
        return Err(format!("selected synthesis path is unavailable: {error}"));
    }
    verify_catalog_backend(&resolve_mortar_home(), backend, payload.model.as_deref())
        .map_err(|error| format!("selected synthesis path is not verified: {error:#}"))?;
    let queued_at = std::time::Instant::now();
    let permit = Arc::clone(&state.speech_admission.permits)
        .acquire_owned()
        .await
        .map_err(|_| "speech runtime admission is closed".to_string())?;
    let registry = Arc::clone(&state.speech);
    let phase = Arc::clone(&state.speech_phase);
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _phase_reset = SpeechPhaseReset(Arc::clone(&phase));
        let mut service = registry
            .lock()
            .map_err(|_| "resident speech registry lock is poisoned".to_string())?;
        let queue_ms = queued_at.elapsed().as_secs_f64() * 1_000.0;
        let mut output = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            service.synthesize(&payload, &context, &phase, speech_device)
        }))
        .map_err(|panic_payload| {
            format!(
                "resident synthesis panicked: {}",
                panic_message(panic_payload.as_ref())
            )
        })?
        .map_err(|error| format!("{error:#}"))?;
        output.queue_ms = queue_ms;
        Ok::<_, String>(output)
    })
    .await
    .map_err(|error| format!("resident synthesis task failed: {error}"))?
}

async fn cancel_live_turn(State(state): State<AppState>, Path(turn_id): Path<String>) -> Response {
    let cancelled = state
        .live_turns
        .lock()
        .ok()
        .and_then(|turns| turns.get(&turn_id).cloned());
    match cancelled {
        Some(cancelled) => {
            cancelled.store(true, Ordering::Release);
            Json(json!({ "turn_id": turn_id, "cancelled": true })).into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            format!("live turn `{turn_id}` is not active"),
        )
            .into_response(),
    }
}

fn validate_bind_address(addr: SocketAddr) -> Result<(), StartupError> {
    if bind_address_allowed(
        addr,
        std::env::var_os(INSECURE_REMOTE_ENV).as_deref() == Some(std::ffi::OsStr::new("1")),
    ) {
        return Ok(());
    }
    Err(StartupError::new(
        "validating the bind address",
        format!(
            "refusing to bind to {addr} without an explicit trust decision; set {INSECURE_REMOTE_ENV}=1 for insecure development-only remote access"
        ),
    ))
}

fn bind_address_allowed(addr: SocketAddr, insecure_remote_opt_in: bool) -> bool {
    addr.ip().is_loopback() || insecure_remote_opt_in
}

async fn enforce_request_policy(request: Request, next: Next) -> Response {
    if request.method() != Method::GET
        && request.method() != Method::HEAD
        && request.method() != Method::OPTIONS
    {
        if let Err(error) = validate_same_origin(request.headers()) {
            return (StatusCode::FORBIDDEN, error).into_response();
        }
    }
    next.run(request).await
}

fn validate_same_origin(headers: &axum::http::HeaderMap) -> Result<(), String> {
    let Some(origin) = headers.get(header::ORIGIN) else {
        return Ok(());
    };
    let origin = origin
        .to_str()
        .map_err(|_| "invalid Origin header".to_string())?;
    let host = headers
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| "missing Host header on mutating request".to_string())?;
    let allowed_http = format!("http://{host}");
    let allowed_https = format!("https://{host}");
    if origin.eq_ignore_ascii_case(&allowed_http) || origin.eq_ignore_ascii_case(&allowed_https) {
        return Ok(());
    }
    Err(format!("cross-origin request from {origin} is not allowed"))
}

fn ensure_self_signed_cert(cert_dir: &FsPath) -> Result<(), StartupError> {
    let cert = cert_dir.join("tongues-local.crt");
    let key = cert_dir.join("tongues-local.key");
    if cert.exists() && key.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(cert_dir).map_err(|error| {
        StartupError::new("creating the certificate directory", error.to_string())
    })?;
    let key_arg = key.to_str().ok_or_else(|| {
        StartupError::new("building the OpenSSL key path", "path is not valid UTF-8")
    })?;
    let cert_arg = cert.to_str().ok_or_else(|| {
        StartupError::new(
            "building the OpenSSL certificate path",
            "path is not valid UTF-8",
        )
    })?;
    let status = Command::new("openssl")
        .args([
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-days",
            "3650",
            "-nodes",
            "-keyout",
            key_arg,
            "-out",
            cert_arg,
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1",
        ])
        .status()
        .map_err(|error| {
            StartupError::new(
                "running OpenSSL to create a local certificate",
                error.to_string(),
            )
        })?;
    if !status.success() {
        return Err(StartupError::new(
            "creating the local certificate",
            format!("openssl exited with status {status}"),
        ));
    }
    Ok(())
}

async fn serve_app_index(State(state): State<AppState>) -> impl IntoResponse {
    match tokio::fs::read_to_string(state.static_dir.join("index.html")).await {
        Ok(index) => Html(index).into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to read web app index: {error}"),
        )
            .into_response(),
    }
}

#[derive(Serialize)]
struct EmotionsResponse {
    signature_path: String,
    style_vectors_path: Option<String>,
    emotions: Vec<EmotionSignature>,
    generated_from_style_vectors: bool,
    error: Option<String>,
}

#[derive(Serialize)]
struct StyleTts2SamplesResponse {
    reference_dir: Option<String>,
    samples: Vec<StyleTts2Sample>,
    defaults: StyleTts2SampleDefaults,
    error: Option<String>,
}

#[derive(Deserialize)]
struct FileListQuery {
    path: Option<String>,
}

#[derive(Serialize)]
struct FileListResponse {
    path: String,
    parent: Option<String>,
    entries: Vec<FileEntry>,
    error: Option<String>,
}

#[derive(Serialize, Clone)]
struct FileEntry {
    name: String,
    path: String,
    kind: String,
    size: Option<u64>,
    modified_ms: Option<u128>,
    download_url: Option<String>,
}

#[derive(Serialize, Clone)]
struct JobArtifact {
    label: String,
    path: String,
    kind: String,
    size: Option<u64>,
    download_url: Option<String>,
}

#[derive(Serialize, Clone)]
struct StyleTts2Sample {
    id: String,
    label: String,
    path: String,
    audio_url: String,
    duration_ms: Option<u64>,
}

#[derive(Serialize)]
struct StyleTts2SampleDefaults {
    voice: String,
    style: String,
}

#[derive(Deserialize)]
struct SpeechSpeakersQuery {
    backend: Option<String>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct SpeechSpeakerOption {
    name: String,
    label: String,
    id: u32,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct SpeechSpeakersResponse {
    backend: String,
    model: Option<String>,
    installed: bool,
    requires_selection: bool,
    speakers: Vec<SpeechSpeakerOption>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SpeechControlDiscovery {
    field: &'static str,
    label: &'static str,
    kind: &'static str,
    group: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    step: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    default: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unit: Option<&'static str>,
    help: &'static str,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    options: Vec<SpeechControlOption>,
}

#[derive(Debug, Clone, Serialize)]
struct SpeechControlOption {
    value: String,
    label: String,
}

#[derive(Debug, Clone, Serialize)]
struct SpeechCompatibility {
    component_id: String,
    compatible: bool,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct SpeechCompositionDiscovery {
    id: String,
    display_name: String,
    backend: String,
    model: String,
    pipeline: tongues_tts::SpeechPipelineSelection,
    runnable: bool,
    selected: bool,
    controls: Vec<SpeechControlDiscovery>,
    capabilities: tongues_tts::BackendCapabilities,
    statuses: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unavailable_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SpeechPresetDiscovery {
    id: String,
    display_name: String,
    composition_id: String,
    pipeline: tongues_tts::SpeechPipelineSelection,
    developer: bool,
}

#[derive(Debug, Clone, Serialize)]
struct SpeechPathDiscovery {
    #[serde(flatten)]
    capabilities: tongues_tts::BackendCapabilities,
    id: String,
    display_name: String,
    kind: &'static str,
    complete: bool,
    runnable: bool,
    selected: bool,
    installed: bool,
    verified: bool,
    verification_pending: bool,
    verification_status: tongues_tts::ModelVerificationStatus,
    load_state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    acoustic_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vocoder: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cli_vocoder: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice_model: Option<String>,
    component_ids: Vec<String>,
    compatible_vocoders: Vec<SpeechCompatibility>,
    controls: Vec<SpeechControlDiscovery>,
    catalog: Vec<tongues_tts::ModelCatalogEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    missing_catalog_ids: Vec<String>,
    statuses: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unavailable_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    install_command: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SpeechComponentDiscovery {
    id: String,
    display_name: String,
    architecture: String,
    kind: String,
    stage: tongues_tts::SpeechPipelineStage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    spans: Vec<tongues_tts::SpeechPipelineStage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    accepts: Vec<tongues_tts::SpeechPortContract>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    produces: Vec<tongues_tts::SpeechPortContract>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    control_fields: Vec<String>,
    runnable: bool,
    installed: bool,
    verified: bool,
    verification_pending: bool,
    verification_status: tongues_tts::ModelVerificationStatus,
    load_state: &'static str,
    readiness: String,
    statuses: Vec<String>,
    explanation: String,
    compatible_paths: Vec<String>,
    catalog: Vec<tongues_tts::ModelCatalogEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    install_command: Option<String>,
}

#[derive(Debug, Serialize)]
struct SpeechStudioDiscovery {
    schema_version: u32,
    page: SpeechDiscoveryPage,
    paths: Vec<SpeechPathDiscovery>,
    components: Vec<SpeechComponentDiscovery>,
    compositions: Vec<SpeechCompositionDiscovery>,
    compatibility: Vec<tongues_tts::SpeechPipelineCompatibility>,
    presets: Vec<SpeechPresetDiscovery>,
    verification_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct SpeechDiscoveryPage {
    cursor: usize,
    limit: usize,
    returned: usize,
    total: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    next_cursor: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct SpeechModelsQuery {
    #[serde(default)]
    cursor: usize,
    #[serde(default = "default_speech_discovery_page_limit")]
    limit: usize,
    search: Option<String>,
    family: Option<String>,
    license: Option<String>,
    capability: Option<String>,
    verification: Option<String>,
    device: Option<String>,
    model_ids: Option<String>,
}

fn default_speech_discovery_page_limit() -> usize {
    DEFAULT_SPEECH_DISCOVERY_PAGE_LIMIT
}

#[derive(Debug, Default)]
struct SpeechDiscoveryFilters {
    search: String,
    family: String,
    license: String,
    capability: String,
    verification: String,
    device: String,
    model_ids: std::collections::BTreeSet<String>,
}

impl SpeechModelsQuery {
    fn into_filters(self) -> SpeechDiscoveryFilters {
        SpeechDiscoveryFilters {
            search: self.search.unwrap_or_default(),
            family: self.family.unwrap_or_default(),
            license: self.license.unwrap_or_default(),
            capability: self.capability.unwrap_or_default(),
            verification: self.verification.unwrap_or_default(),
            device: self.device.unwrap_or_default(),
            model_ids: self
                .model_ids
                .unwrap_or_default()
                .split(',')
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(str::to_string)
                .collect(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct SpeechProjectionRequest {
    text: String,
    variety: String,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    pipeline: Option<tongues_tts::SpeechPipelineSelection>,
}

#[derive(Debug, Default, Deserialize)]
struct DuplexProjectionRequest {
    #[serde(default)]
    fixture: Option<String>,
    #[serde(default)]
    chunks: Vec<String>,
    #[serde(default)]
    mock_acoustics: Vec<String>,
    #[serde(default)]
    variety: Option<String>,
    #[serde(default)]
    posterior_mass: Option<f64>,
    #[serde(default)]
    journal_path: Option<String>,
}

#[derive(Debug, Serialize)]
struct SpeechProjectionResponse {
    projected_token_count: usize,
    backend_symbols: String,
    phoneme_count: usize,
    phone_count: usize,
}

#[derive(Debug, Deserialize)]
struct SpeechUnloadRequest {
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    pipeline: Option<tongues_tts::SpeechPipelineSelection>,
}

#[derive(Serialize, Clone)]
struct EmotionSignature {
    name: String,
    kind: String,
    method: String,
    dims: usize,
    vector: Vec<f32>,
    stats: EmotionStats,
    recommended_strength: RecommendedStrength,
}

#[derive(Serialize, Clone, Default)]
struct EmotionStats {
    n_speakers: usize,
    sample_count: usize,
}

#[derive(Serialize, Clone)]
struct RecommendedStrength {
    subtle: f32,
    normal: f32,
    strong: f32,
}

impl Default for RecommendedStrength {
    fn default() -> Self {
        Self {
            subtle: 0.25,
            normal: 0.65,
            strong: 1.10,
        }
    }
}

#[derive(Clone)]
struct JobRecord {
    summary: JobSummary,
    output: VecDeque<JobOutputLine>,
    events: broadcast::Sender<JobEvent>,
    child: Option<Arc<Mutex<Child>>>,
    cancel_requested: bool,
}

#[derive(Serialize, Clone)]
struct JobSummary {
    id: String,
    label: String,
    command: String,
    args: Vec<String>,
    status: JobStatus,
    created_at_ms: u128,
    updated_at_ms: u128,
    exit_code: Option<i32>,
    progress: JobProgress,
}

#[derive(Serialize, Clone)]
struct JobProgress {
    phase: String,
    current: Option<u64>,
    total: Option<u64>,
}

impl Default for JobProgress {
    fn default() -> Self {
        Self {
            phase: "Queued".into(),
            current: None,
            total: None,
        }
    }
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "kebab-case")]
enum JobStatus {
    Running,
    Succeeded,
    Failed,
    Canceled,
}

#[derive(Serialize, Clone)]
struct JobOutputLine {
    stream: String,
    line: String,
    at_ms: u128,
}

#[derive(Serialize, Clone)]
#[serde(tag = "type", rename_all = "kebab-case")]
enum JobEvent {
    Snapshot {
        summary: JobSummary,
        output: Vec<JobOutputLine>,
    },
    Output {
        stream: String,
        line: String,
        at_ms: u128,
    },
    Progress {
        progress: JobProgress,
        at_ms: u128,
    },
    Status {
        summary: JobSummary,
    },
}

#[derive(Serialize)]
struct JobDetail {
    summary: JobSummary,
    output: Vec<JobOutputLine>,
    artifacts: Vec<JobArtifact>,
}

#[derive(Deserialize)]
struct StartJobRequest {
    label: Option<String>,
    command: String,
    args: Vec<String>,
}

const ALLOWED_JOB_PREFIXES: &[&[&str]] = &[
    &["discrepancies"],
    &["emotions", "eval"],
    &["emotions", "infer"],
    &["emotions", "prepare"],
    &["emotions", "train"],
    &["eval"],
    &["fetch-cmudict"],
    &["fetch-corpora"],
    &["g2p2g", "clean"],
    &["g2p2g", "eval"],
    &["g2p2g", "infer"],
    &["g2p2g", "prepare"],
    &["g2p2g", "refine"],
    &["g2p2g", "repl"],
    &["g2p2g", "train"],
    &["head2phones", "clean"],
    &["head2phones", "infer"],
    &["head2phones", "prepare"],
    &["head2phones", "train"],
    &["head2phones", "verify"],
    &["interpretation", "clean"],
    &["interpretation", "eval"],
    &["interpretation", "prepare"],
    &["interpretation", "stream"],
    &["interpretation", "train"],
    &["models", "fetch"],
    &["models", "install"],
    &["models", "list"],
    &["models", "menu"],
    &["models", "path"],
    &["models", "status"],
    &["models", "use"],
    &["phonemes"],
    &["phones"],
    &["predict"],
    &["prepare"],
    &["refine"],
    &["repl"],
    &["sentence-parser", "clean"],
    &["sentence-parser", "eval"],
    &["sentence-parser", "infer"],
    &["sentence-parser", "parse"],
    &["sentence-parser", "prepare"],
    &["sentence-parser", "stream"],
    &["sentence-parser", "train"],
    &["speak"],
    &["speaking"],
    &["styletts2", "discover"],
    &["styletts2", "emotion-signatures"],
    &["styletts2", "encode-style"],
    &["train"],
    &["wiktionary", "clean"],
    &["wiktionary", "infer"],
    &["wiktionary", "prepare"],
    &["wiktionary", "train"],
];

const FLAG_ONLY_JOB_ARGS: &[&str] = &[
    "--all",
    "--careful-style",
    "--cpu",
    "--debug-pronunciation",
    "--fail-on-guessed-pronunciation",
    "--force",
    "--json",
    "--list",
    "--no-create",
    "--no-download-wiktionary-audio",
    "--no-full-cut",
    "--no-g2p2g",
    "--no-tts-chunking",
    "--no-whisper-transcripts",
    "--no-wiktionary",
    "--no-wiktionary-audio",
    "--ollama-strict",
    "--prepare",
    "--quiet",
    "--raw",
    "--strict",
    "--timings",
    "--verbose",
    "--verify-ollama",
    "--wait-for-prepare",
];

const VALUE_JOB_ARGS: &[&str] = &[
    "--archive-dir",
    "--backend",
    "--batch-size",
    "--cache-dir",
    "--config",
    "--corpus",
    "--cuts-per-wav",
    "--data",
    "--diffusion-steps",
    "--dropout",
    "--dump",
    "--durations",
    "--embedding-scale",
    "--emotion",
    "--emotion-signatures",
    "--emotion-strength",
    "--epochs",
    "--g2p2g-model",
    "--head2phones-model",
    "--input",
    "--labels",
    "--lang",
    "--learning-rate",
    "--limit",
    "--mask-policy",
    "--max-chars",
    "--max-cut-ms",
    "--max-mask-rate",
    "--max-rarity",
    "--max-tts-symbols",
    "--max-utterances",
    "--max-whisper-wer",
    "--max-wiktionary-audio",
    "--mel-bins",
    "--method",
    "--min-cut-ms",
    "--model",
    "--notation",
    "--num-samples",
    "--ollama-max-chars",
    "--ollama-model",
    "--ollama-rows",
    "--ollama-url",
    "--out",
    "--out-dir",
    "--output",
    "--patience",
    "--pitch",
    "--pitch-scale",
    "--pitch-shift",
    "--previous",
    "--quality",
    "--references-dir",
    "--repair-control",
    "--run-id",
    "--sample-rate-hz",
    "--seed",
    "--sight-words",
    "--source",
    "--source-manifest",
    "--span-mask-prob",
    "--speaker",
    "--speaker-reference-strength",
    "--speed",
    "--split",
    "--splits",
    "--style-alpha",
    "--style-beta",
    "--style-reference-strength",
    "--style-seed",
    "--style-wav",
    "--subset",
    "--task",
    "--tier",
    "--train-frac",
    "--training-set",
    "--valid-frac",
    "--variety",
    "--voice-wav",
    "--wav",
    "--weight-decay",
    "--whisper-model",
    "--wiktionary-audio-data",
    "--wiktionary-model",
    "--word",
    "--words-file",
];

#[derive(Serialize)]
struct StartJobResponse {
    job_id: String,
    summary: JobSummary,
}

#[derive(Serialize, Clone)]
struct PronunciationModelOption {
    id: String,
    label: String,
    family: String,
    path: String,
    available: bool,
}

#[derive(Serialize)]
struct PronunciationModelsResponse {
    models: Vec<PronunciationModelOption>,
    languages: Vec<OwnedCodeLabel>,
    varieties: Vec<OwnedCodeLabel>,
    wiktionary_tasks: Vec<CodeLabel>,
    g2p2g_tasks: Vec<CodeLabel>,
    notations: Vec<CodeLabel>,
}

#[derive(Serialize)]
struct CodeLabel {
    value: &'static str,
    label: &'static str,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct OwnedCodeLabel {
    value: String,
    label: String,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct LinguisticVarietiesResponse {
    default: String,
    varieties: Vec<LinguisticVarietyMetadata>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct LinguisticVarietyMetadata {
    value: String,
    label: String,
    language: String,
    language_tag: Option<String>,
    pronunciation_fallback: PronunciationFallbackMetadata,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "status")]
enum PronunciationFallbackMetadata {
    Mapped {
        provider: &'static str,
        language: String,
    },
    Unsupported {
        provider: &'static str,
        reason: &'static str,
    },
}

#[derive(Deserialize)]
struct PronunciationInferRequest {
    family: String,
    model: String,
    input: String,
    task: String,
    lang: Option<String>,
    variety: Option<String>,
    notation: Option<String>,
    raw: Option<bool>,
    cpu: Option<bool>,
}

#[derive(Serialize)]
struct PronunciationInferResponse {
    output: String,
    command: Vec<String>,
    source: Option<String>,
    stderr: String,
}

async fn get_emotions(State(state): State<AppState>) -> impl IntoResponse {
    match load_or_create_emotion_signatures(&state) {
        Ok(response) => Json(response).into_response(),
        Err(error) => {
            let signature_path = emotion_signatures_path(&state);
            Json(EmotionsResponse {
                signature_path: signature_path.display().to_string(),
                style_vectors_path: find_style_vectors_path(&state)
                    .map(|path| path.display().to_string()),
                emotions: Vec::new(),
                generated_from_style_vectors: false,
                error: Some(error),
            })
            .into_response()
        }
    }
}

async fn list_files(
    State(state): State<AppState>,
    Query(query): Query<FileListQuery>,
) -> impl IntoResponse {
    let requested = query.path.unwrap_or_default();
    let relative = match safe_relative_path(&requested) {
        Ok(path) => path,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(FileListResponse {
                    path: String::new(),
                    parent: None,
                    entries: Vec::new(),
                    error: Some(error),
                }),
            )
                .into_response();
        }
    };
    if relative.as_os_str().is_empty() {
        return Json(FileListResponse {
            path: String::new(),
            parent: None,
            entries: artifact_browser_root_entries(&state.workspace_root),
            error: None,
        })
        .into_response();
    }
    let target = match resolve_existing_artifact_path(&state.workspace_root, &relative) {
        Ok(path) => path,
        Err(error) => {
            return Json(FileListResponse {
                path: path_to_web(&relative),
                parent: parent_web_path(&relative),
                entries: Vec::new(),
                error: Some(error),
            })
            .into_response();
        }
    };
    let mut list_relative = relative.clone();
    let mut target_dir = target.clone();
    if target.is_file() {
        list_relative = relative
            .parent()
            .unwrap_or_else(|| FsPath::new(""))
            .to_path_buf();
        if list_relative.as_os_str().is_empty() {
            return Json(FileListResponse {
                path: String::new(),
                parent: None,
                entries: artifact_browser_root_entries(&state.workspace_root),
                error: None,
            })
            .into_response();
        }
        target_dir = match resolve_existing_artifact_path(&state.workspace_root, &list_relative) {
            Ok(path) => path,
            Err(error) => {
                return Json(FileListResponse {
                    path: path_to_web(&list_relative),
                    parent: parent_web_path(&list_relative),
                    entries: Vec::new(),
                    error: Some(error),
                })
                .into_response();
            }
        };
    }

    let mut entries = Vec::new();
    let read_dir = match std::fs::read_dir(&target_dir) {
        Ok(read_dir) => read_dir,
        Err(error) => {
            return Json(FileListResponse {
                path: path_to_web(&list_relative),
                parent: parent_web_path(&list_relative),
                entries,
                error: Some(format!("Could not read directory: {error}")),
            })
            .into_response();
        }
    };
    for entry in read_dir.flatten().take(FILE_LIST_LIMIT) {
        let name = entry.file_name().to_string_lossy().to_string();
        let entry_relative = list_relative.join(&name);
        let Ok(entry_path) = resolve_existing_artifact_path(&state.workspace_root, &entry_relative)
        else {
            continue;
        };
        let Ok(metadata) = std::fs::metadata(&entry_path) else {
            continue;
        };
        let is_dir = metadata.is_dir();
        entries.push(FileEntry {
            name,
            path: path_to_web(&entry_relative),
            kind: if is_dir { "dir" } else { "file" }.into(),
            size: if is_dir { None } else { Some(metadata.len()) },
            modified_ms: metadata.modified().ok().and_then(system_time_ms),
            download_url: if is_dir {
                None
            } else {
                Some(download_url_for(&entry_relative))
            },
        });
    }

    entries.sort_by(|left, right| {
        let left_dir = left.kind == "dir";
        let right_dir = right.kind == "dir";
        right_dir
            .cmp(&left_dir)
            .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
    });

    Json(FileListResponse {
        path: path_to_web(&list_relative),
        parent: parent_web_path(&list_relative),
        entries,
        error: None,
    })
    .into_response()
}

async fn download_file(
    State(state): State<AppState>,
    Path(path): Path<String>,
) -> impl IntoResponse {
    let relative = match safe_relative_path(&path) {
        Ok(path) => path,
        Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
    };
    let path = match resolve_existing_artifact_path(&state.workspace_root, &relative) {
        Ok(path) => path,
        Err(error) => return (StatusCode::NOT_FOUND, error).into_response(),
    };
    let Ok(metadata) = std::fs::metadata(&path) else {
        return (StatusCode::NOT_FOUND, "download path is not available").into_response();
    };
    if !metadata.is_file() {
        return (StatusCode::NOT_FOUND, "download path is not a file").into_response();
    }
    if metadata.len() > MAX_DOWNLOAD_BYTES {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("download exceeds the {MAX_DOWNLOAD_BYTES}-byte limit"),
        )
            .into_response();
    }
    match tokio::fs::read(&path).await {
        Ok(bytes) => {
            let filename = relative
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("download")
                .replace('"', "");
            Response::builder()
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .header(
                    header::CONTENT_DISPOSITION,
                    format!("attachment; filename=\"{filename}\""),
                )
                .body(axum::body::Body::from(bytes))
                .unwrap()
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to read download: {error}"),
        )
            .into_response(),
    }
}

async fn list_jobs(State(state): State<AppState>) -> impl IntoResponse {
    let mut jobs = state
        .jobs
        .lock()
        .expect("job registry lock")
        .values()
        .map(|job| job.summary.clone())
        .collect::<Vec<_>>();
    jobs.sort_by(|left, right| right.created_at_ms.cmp(&left.created_at_ms));
    Json(jobs)
}

async fn get_job(State(state): State<AppState>, Path(job_id): Path<String>) -> impl IntoResponse {
    match job_detail(&state, &job_id) {
        Some(detail) => Json(detail).into_response(),
        None => (StatusCode::NOT_FOUND, "unknown job").into_response(),
    }
}

async fn get_pronunciation_models(State(state): State<AppState>) -> impl IntoResponse {
    Json(PronunciationModelsResponse {
        models: pronunciation_model_options(&state.workspace_root),
        languages: linguistic_language_options(),
        varieties: linguistic_variety_options(true),
        wiktionary_tasks: vec![
            CodeLabel {
                value: "orthography-to-phones",
                label: "Spelling to phones",
            },
            CodeLabel {
                value: "orthography-to-phonemes",
                label: "Spelling to phonemes",
            },
            CodeLabel {
                value: "phones-to-orthography",
                label: "Phones to spelling",
            },
            CodeLabel {
                value: "phonemes-to-orthography",
                label: "Phonemes to spelling",
            },
            CodeLabel {
                value: "phonetic-realization",
                label: "Phonemes to phones",
            },
            CodeLabel {
                value: "segment-compound",
                label: "Segment compound",
            },
            CodeLabel {
                value: "pronounce-segments",
                label: "Pronounce segments",
            },
            CodeLabel {
                value: "verify-pronunciation",
                label: "Verify pronunciation",
            },
            CodeLabel {
                value: "normalize-phonology",
                label: "Normalize phonology",
            },
            CodeLabel {
                value: "find-etymology",
                label: "Find etymology",
            },
            CodeLabel {
                value: "normalize",
                label: "Normalize text",
            },
            CodeLabel {
                value: "guess-lang-from-orthography",
                label: "Guess language from spelling",
            },
            CodeLabel {
                value: "guess-lang-from-phonology",
                label: "Guess language from phonology",
            },
            CodeLabel {
                value: "guess-lang-from-orthography-and-phonology",
                label: "Guess language from both",
            },
        ],
        g2p2g_tasks: vec![
            CodeLabel {
                value: "auto",
                label: "Auto",
            },
            CodeLabel {
                value: "g2p",
                label: "Spelling to pronunciation",
            },
            CodeLabel {
                value: "p2g",
                label: "Pronunciation to spelling",
            },
        ],
        notations: vec![
            CodeLabel {
                value: "phones",
                label: "Phones",
            },
            CodeLabel {
                value: "phonemes",
                label: "Phonemes",
            },
        ],
    })
}

async fn get_linguistic_varieties() -> impl IntoResponse {
    let configured_default = speaking::data::varieties::DEFAULT_SPEAKING_VARIETY;
    let default = speaking::canonical_variety_id(configured_default)
        .map(|id| id.0)
        .unwrap_or_else(|| configured_default.into());
    Json(LinguisticVarietiesResponse {
        default,
        varieties: linguistic_variety_metadata(),
    })
}

fn linguistic_language_options() -> Vec<OwnedCodeLabel> {
    speaking::builtin_languages()
        .into_iter()
        .map(|language| OwnedCodeLabel {
            value: language.iso_639.unwrap_or(language.id.0),
            label: language.name,
        })
        .collect()
}

fn linguistic_variety_options(include_default: bool) -> Vec<OwnedCodeLabel> {
    let mut options = Vec::new();
    if include_default {
        options.push(OwnedCodeLabel {
            value: String::new(),
            label: "Default".into(),
        });
    }
    options.extend(
        speaking::builtin_varieties()
            .into_iter()
            .map(|variety| OwnedCodeLabel {
                value: variety.id.0,
                label: variety.name,
            }),
    );
    options
}

fn linguistic_variety_metadata() -> Vec<LinguisticVarietyMetadata> {
    speaking::builtin_varieties()
        .into_iter()
        .map(|variety| {
            let fallback_language =
                speaking::wiktionary_language_for_variety(&variety.id.0).map(str::to_string);
            LinguisticVarietyMetadata {
                language_tag: speaking::language_tag_for_variety(&variety.id.0).map(str::to_string),
                pronunciation_fallback: fallback_language.map_or(
                    PronunciationFallbackMetadata::Unsupported {
                        provider: "wiktionary",
                        reason: "registered language has no Wiktionary language code",
                    },
                    |language| PronunciationFallbackMetadata::Mapped {
                        provider: "wiktionary",
                        language,
                    },
                ),
                value: variety.id.0,
                label: variety.name,
                language: variety.language.0,
            }
        })
        .collect()
}

async fn pronunciation_infer(
    State(state): State<AppState>,
    Json(payload): Json<PronunciationInferRequest>,
) -> impl IntoResponse {
    let request = match build_pronunciation_command(&state, &payload) {
        Ok(request) => request,
        Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
    };

    let command_for_response = request.args.clone();
    let output = tokio::task::spawn_blocking(move || {
        Command::new("cargo")
            .args(&request.args)
            .current_dir(request.workspace_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
    })
    .await;

    let output = match output {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to run inference: {error}"),
            )
                .into_response();
        }
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("inference task failed: {error}"),
            )
                .into_response();
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !output.status.success() {
        return (
            StatusCode::BAD_REQUEST,
            format!(
                "inference failed with status {}:\n{}",
                output.status.code().unwrap_or(-1),
                if stderr.is_empty() { &stdout } else { &stderr }
            ),
        )
            .into_response();
    }

    Json(PronunciationInferResponse {
        output: stdout,
        command: command_for_response,
        source: wiktionary_demo_source(&payload),
        stderr,
    })
    .into_response()
}

async fn start_job(
    State(state): State<AppState>,
    Json(payload): Json<StartJobRequest>,
) -> impl IntoResponse {
    if let Err(error) = validate_job_request(&state.workspace_root, &payload) {
        return (StatusCode::BAD_REQUEST, error).into_response();
    }
    {
        let jobs = state.jobs.lock().expect("job registry lock");
        let running = jobs
            .values()
            .filter(|job| matches!(job.summary.status, JobStatus::Running))
            .count();
        if running >= MAX_ACTIVE_JOBS {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                format!("job queue is full; at most {MAX_ACTIVE_JOBS} jobs may run at once"),
            )
                .into_response();
        }
    }

    let id = uuid::Uuid::new_v4().to_string();
    let now = now_ms();
    let label = payload
        .label
        .filter(|label| !label.trim().is_empty())
        .unwrap_or_else(|| format!("{} {}", payload.command, payload.args.join(" ")));
    let (tx, _) = broadcast::channel(256);
    let summary = JobSummary {
        id: id.clone(),
        label,
        command: payload.command.clone(),
        args: payload.args.clone(),
        status: JobStatus::Running,
        created_at_ms: now,
        updated_at_ms: now,
        exit_code: None,
        progress: JobProgress {
            phase: "Starting".into(),
            current: None,
            total: None,
        },
    };
    {
        let mut jobs = state.jobs.lock().expect("job registry lock");
        jobs.insert(
            id.clone(),
            JobRecord {
                summary: summary.clone(),
                output: VecDeque::new(),
                events: tx.clone(),
                child: None,
                cancel_requested: false,
            },
        );
    }

    let workspace_root = state.workspace_root.clone();
    let jobs = state.jobs.clone();
    let job_id = id.clone();
    std::thread::spawn(move || run_job_process(jobs, job_id, workspace_root));

    Json(StartJobResponse {
        job_id: id,
        summary,
    })
    .into_response()
}

async fn cancel_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    let child = {
        let mut jobs = state.jobs.lock().expect("job registry lock");
        let Some(job) = jobs.get_mut(&job_id) else {
            return (StatusCode::NOT_FOUND, "unknown job").into_response();
        };
        if !matches!(job.summary.status, JobStatus::Running) {
            return (StatusCode::CONFLICT, "job is already finished").into_response();
        }
        job.cancel_requested = true;
        job.child.clone()
    };
    let Some(child) = child else {
        return (StatusCode::NOT_FOUND, "unknown or finished job").into_response();
    };
    match child.lock().expect("child lock").kill() {
        Ok(()) => {
            append_job_output(&state.jobs, &job_id, "status", "Cancel requested");
            Json(json!({ "ok": true })).into_response()
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to cancel job: {error}"),
        )
            .into_response(),
    }
}

async fn job_events(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    let (summary, output, receiver) = {
        let jobs = state.jobs.lock().expect("job registry lock");
        let Some(job) = jobs.get(&job_id) else {
            return (StatusCode::NOT_FOUND, "unknown job").into_response();
        };
        (
            job.summary.clone(),
            job.output.iter().cloned().collect::<Vec<_>>(),
            job.events.subscribe(),
        )
    };

    let stream = BroadcastStream::new(receiver).filter_map(|event| match event {
        Ok(event) => Some(Ok::<_, std::convert::Infallible>(sse_event(&event))),
        Err(_) => None,
    });
    let snapshot = JobEvent::Snapshot { summary, output };
    let stream =
        tokio_stream::once(Ok::<_, std::convert::Infallible>(sse_event(&snapshot))).chain(stream);

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn validate_job_request(workspace_root: &FsPath, payload: &StartJobRequest) -> Result<(), String> {
    if payload.command != "cargo" {
        return Err("only cargo jobs are supported".into());
    }
    if payload.args.len() < 5
        || payload.args[0] != "run"
        || payload.args[1] != "--bin"
        || payload.args[2] != "tongues"
        || payload.args[3] != "--"
    {
        return Err("job args must be `cargo run --bin tongues -- ...`".into());
    }
    if payload.args.iter().any(|arg| arg.contains('\0')) {
        return Err("job args contain invalid data".into());
    }
    if payload
        .args
        .iter()
        .any(|arg| arg.starts_with('-') && !arg.starts_with("--"))
    {
        return Err("short flags are not available through the web job API".into());
    }
    let args = &payload.args[4..];
    let mut cursor = 0;
    while matches!(
        args.get(cursor).map(String::as_str),
        Some("--cpu" | "--quiet" | "--verbose")
    ) {
        cursor += 1;
    }
    let Some(prefix_len) = ALLOWED_JOB_PREFIXES
        .iter()
        .filter(|prefix| {
            args.len() >= cursor + prefix.len()
                && prefix
                    .iter()
                    .zip(&args[cursor..cursor + prefix.len()])
                    .all(|(expected, actual)| *expected == actual)
        })
        .map(|prefix| prefix.len())
        .max()
    else {
        return Err("job args do not match an approved Tongues command".into());
    };
    cursor += prefix_len;
    let command_prefix = &args[cursor - prefix_len..cursor];
    while cursor < args.len() {
        let token = &args[cursor];
        if FLAG_ONLY_JOB_ARGS.iter().any(|flag| *flag == token) {
            cursor += 1;
            continue;
        }
        if VALUE_JOB_ARGS.iter().any(|flag| *flag == token) {
            let Some(value) = args.get(cursor + 1) else {
                return Err(format!("missing value for {token}"));
            };
            validate_job_argument_value(workspace_root, command_prefix, token, value)?;
            cursor += 2;
            continue;
        }
        if token.starts_with("--") {
            return Err(format!("job flag `{token}` is not approved"));
        }
        validate_job_positional_value(workspace_root, command_prefix, token)?;
        cursor += 1;
    }
    Ok(())
}

fn validate_job_argument_value(
    workspace_root: &FsPath,
    prefix: &[String],
    flag: &str,
    value: &str,
) -> Result<(), String> {
    if value.starts_with("--") {
        return Err(format!("missing value for {flag}"));
    }
    if value.len() > 8 * 1024 {
        return Err(format!("value for {flag} is too large"));
    }
    if is_job_config_flag(flag) {
        validate_job_config_path(workspace_root, value)?;
    } else if is_job_artifact_path_flag(flag) {
        validate_job_artifact_path(workspace_root, value)?;
    } else if job_prefix_is(prefix, &["styletts2", "discover"]) && flag == "--references-dir" {
        validate_job_artifact_path(workspace_root, value)?;
    }
    Ok(())
}

fn validate_job_positional_value(
    workspace_root: &FsPath,
    prefix: &[String],
    value: &str,
) -> Result<(), String> {
    if value.len() > 64 * 1024 {
        return Err("positional job values are too large".into());
    }
    if job_prefix_is(prefix, &["styletts2", "encode-style"])
        || job_prefix_is(prefix, &["styletts2", "emotion-signatures"])
    {
        validate_job_artifact_path(workspace_root, value)?;
    }
    Ok(())
}

fn job_prefix_is(prefix: &[String], expected: &[&str]) -> bool {
    prefix.len() == expected.len()
        && prefix
            .iter()
            .map(String::as_str)
            .zip(expected.iter().copied())
            .all(|(actual, expected)| actual == expected)
}

fn is_job_config_flag(flag: &str) -> bool {
    flag == "--config"
}

fn is_job_artifact_path_flag(flag: &str) -> bool {
    matches!(
        flag,
        "--archive-dir"
            | "--cache-dir"
            | "--data"
            | "--dump"
            | "--emotion-signatures"
            | "--g2p2g-model"
            | "--head2phones-model"
            | "--input"
            | "--labels"
            | "--model"
            | "--out"
            | "--out-dir"
            | "--output"
            | "--source-manifest"
            | "--style-wav"
            | "--voice-wav"
            | "--wav"
            | "--whisper-model"
            | "--wiktionary-audio-data"
            | "--wiktionary-model"
            | "--words-file"
    )
}

fn validate_job_config_path(workspace_root: &FsPath, value: &str) -> Result<(), String> {
    let relative = safe_relative_path(value)?;
    if artifact_root_name(&relative) != Some("configs") {
        return Err("config paths must stay inside configs/".into());
    }
    validate_existing_ancestor_within_workspace(workspace_root, &relative)
}

fn validate_job_artifact_path(workspace_root: &FsPath, value: &str) -> Result<(), String> {
    let relative = safe_relative_path(value)?;
    validate_artifact_relative_path(workspace_root, &relative)
}

struct PronunciationCommandRequest {
    workspace_root: PathBuf,
    args: Vec<String>,
}

fn build_pronunciation_command(
    state: &AppState,
    payload: &PronunciationInferRequest,
) -> Result<PronunciationCommandRequest, String> {
    let family = payload.family.trim();
    if !matches!(family, "g2p2g" | "wiktionary") {
        return Err("family must be g2p2g or wiktionary".into());
    }
    if payload.input.trim().is_empty() {
        return Err("input is required".into());
    }
    let model = safe_relative_path(&payload.model)?;
    validate_artifact_relative_path(&state.workspace_root, &model)?;
    let model_path = resolve_existing_artifact_path(&state.workspace_root, &model)?;
    if !model_path.exists() {
        return Err(format!(
            "model path does not exist: {}",
            path_to_web(&model)
        ));
    }

    let mut args = vec![
        "run".to_string(),
        "--bin".to_string(),
        "tongues".to_string(),
        "--".to_string(),
    ];
    if payload.cpu.unwrap_or(true) {
        args.push("--cpu".to_string());
    }
    args.push(family.to_string());
    args.push("infer".to_string());
    args.push("--model".to_string());
    args.push(path_to_web(&model));
    args.push("--task".to_string());
    args.push(payload.task.trim().to_string());

    if family == "wiktionary" {
        args.push("--lang".to_string());
        args.push(
            payload
                .lang
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or("eng")
                .trim()
                .to_string(),
        );
        args.push("--notation".to_string());
        args.push(
            payload
                .notation
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or("phones")
                .trim()
                .to_string(),
        );
        if let Some(variety) = payload
            .variety
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            args.push("--variety".to_string());
            args.push(variety.to_string());
        }
        if payload.raw.unwrap_or(false) {
            args.push("--raw".to_string());
        }
    }

    args.push(payload.input.trim().to_string());
    Ok(PronunciationCommandRequest {
        workspace_root: state.workspace_root.clone(),
        args,
    })
}

fn wiktionary_demo_source(payload: &PronunciationInferRequest) -> Option<String> {
    if payload.family.trim() != "wiktionary" {
        return None;
    }
    if payload.raw.unwrap_or(false) {
        return Some(payload.input.trim().to_string());
    }
    let lang = payload
        .lang
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("eng")
        .trim();
    let notation = payload
        .notation
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("phones")
        .trim();
    let repr = match notation {
        "phonemes" => "<repr:phonemes>",
        _ => "<repr:phones>",
    };
    let variety = payload
        .variety
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| format!(" <variety:{value}>"))
        .unwrap_or_default();
    let input = payload.input.trim();
    let source = match payload.task.trim() {
        "orthography-to-phonemes" => {
            format!(
                "<task:orthography_to_phonology> <lang:{lang}>{variety} <repr:phonemes> {input}"
            )
        }
        "orthography-to-phones" | "orthography-to-phonology" => {
            format!("<task:orthography_to_phonology> <lang:{lang}>{variety} {repr} {input}")
        }
        "phonemes-to-orthography" => {
            format!(
                "<task:phonology_to_orthography> <lang:{lang}>{variety} <repr:phonemes> {input}"
            )
        }
        "phones-to-orthography" | "phonology-to-orthography" => {
            format!("<task:phonology_to_orthography> <lang:{lang}>{variety} {repr} {input}")
        }
        "phonetic-realization" => {
            format!("<task:phonetic_realization> <lang:{lang}>{variety} <repr:phonemes> {input}")
        }
        "segment-compound" | "segment" | "compound-segmentation" => {
            format!("<task:segment_compound> <lang:{lang}> <SEGMENT> {input}")
        }
        "pronounce-segments" | "segments-to-phonology" | "segments-to-phones" => {
            format!(
                "<task:pronounce_segments> <lang:{lang}> <PRONOUNCE_SEGMENTS> <repr:phones> {input}"
            )
        }
        "verify-pronunciation" | "verify" | "verifier" => {
            format!("<task:verify_pronunciation> <lang:{lang}> <VERIFY> {input}")
        }
        "normalize-phonology" | "normalise-phonology" | "broad-equivalence" => {
            format!("<task:normalize_phonology> <lang:{lang}> <BROAD_EQUIV> <repr:phones> {input}")
        }
        "find-etymology" | "etymology-from-word" | "word-etymology" => {
            format!("<task:find_etymology> <lang:{lang}> {input}")
        }
        "normalize" | "normalise" => format!("<task:normalize> <lang:{lang}> {input}"),
        "guess-lang-from-orthography" | "lang-from-orthography" => {
            format!("<task:guess_lang_from_orthography> {repr} {input}")
        }
        "guess-lang-from-phonology" | "lang-from-phonology" => {
            format!("<task:guess_lang_from_phonology> {repr} {input}")
        }
        "guess-lang-from-orthography-and-phonology" | "lang" | "language" | "language-guessing" => {
            format!("<task:guess_lang_from_orthography_and_phonology> {repr} {input}")
        }
        _ => return None,
    };
    Some(source)
}

fn pronunciation_model_options(workspace_root: &FsPath) -> Vec<PronunciationModelOption> {
    let mut models = Vec::new();
    add_pronunciation_model_option(
        workspace_root,
        &mut models,
        "g2p2g:default",
        "G2P2G default",
        "g2p2g",
        "models/g2p2g/openepd-v0",
    );
    add_pronunciation_model_option(
        workspace_root,
        &mut models,
        "wiktionary:default",
        "Wiktionary default phones",
        "wiktionary",
        "models/wiktionary/enwiktionary-2026-06-01-v0-phones",
    );
    scan_pronunciation_model_dir(workspace_root, &mut models, "g2p2g", "models/g2p2g");
    scan_pronunciation_model_dir(
        workspace_root,
        &mut models,
        "wiktionary",
        "models/wiktionary",
    );
    models.sort_by(|left, right| {
        left.family
            .cmp(&right.family)
            .then_with(|| right.available.cmp(&left.available))
            .then_with(|| left.label.cmp(&right.label))
    });
    models.dedup_by(|left, right| left.family == right.family && left.path == right.path);
    models
}

fn scan_pronunciation_model_dir(
    workspace_root: &FsPath,
    models: &mut Vec<PronunciationModelOption>,
    family: &str,
    relative_dir: &str,
) {
    let dir = workspace_root.join(relative_dir);
    let Ok(read_dir) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read_dir.flatten() {
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if !metadata.is_dir() {
            continue;
        }
        let path = FsPath::new(relative_dir).join(entry.file_name());
        let path_web = path_to_web(&path);
        let label = format!(
            "{} {}",
            if family == "g2p2g" {
                "G2P2G"
            } else {
                "Wiktionary"
            },
            entry.file_name().to_string_lossy()
        );
        add_pronunciation_model_option(
            workspace_root,
            models,
            &format!("{family}:{path_web}"),
            &label,
            family,
            &path_web,
        );
    }
}

fn add_pronunciation_model_option(
    workspace_root: &FsPath,
    models: &mut Vec<PronunciationModelOption>,
    id: &str,
    label: &str,
    family: &str,
    path: &str,
) {
    let full_path = workspace_root.join(path);
    let available =
        full_path.join("model_config.json").exists() && full_path.join("vocab.json").exists();
    models.push(PronunciationModelOption {
        id: id.to_string(),
        label: label.to_string(),
        family: family.to_string(),
        path: path.to_string(),
        available,
    });
}

fn job_detail(state: &AppState, job_id: &str) -> Option<JobDetail> {
    let jobs = state.jobs.lock().expect("job registry lock");
    jobs.get(job_id).map(|job| {
        let summary = job.summary.clone();
        JobDetail {
            artifacts: artifacts_for_job(&state.workspace_root, &summary.args),
            summary,
            output: job.output.iter().cloned().collect(),
        }
    })
}

fn run_job_process(jobs: JobRegistry, job_id: String, workspace_root: PathBuf) {
    let (command, args) = {
        let jobs_guard = jobs.lock().expect("job registry lock");
        let Some(job) = jobs_guard.get(&job_id) else {
            return;
        };
        (job.summary.command.clone(), job.summary.args.clone())
    };

    update_job_progress(
        &jobs,
        &job_id,
        JobProgress {
            phase: "Launching process".into(),
            current: None,
            total: None,
        },
    );

    let mut child = match Command::new(&command)
        .args(&args)
        .current_dir(&workspace_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            append_job_output(
                &jobs,
                &job_id,
                "stderr",
                &format!("Failed to start: {error}"),
            );
            finish_job(&jobs, &job_id, JobStatus::Failed, None);
            return;
        }
    };

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let child = Arc::new(Mutex::new(child));
    {
        let mut jobs_guard = jobs.lock().expect("job registry lock");
        if let Some(job) = jobs_guard.get_mut(&job_id) {
            job.child = Some(child.clone());
        }
    }

    if let Some(stdout) = stdout {
        spawn_output_reader(jobs.clone(), job_id.clone(), "stdout", stdout);
    }
    if let Some(stderr) = stderr {
        spawn_output_reader(jobs.clone(), job_id.clone(), "stderr", stderr);
    }

    let status = loop {
        match child.lock().expect("child lock").try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(250)),
            Err(error) => {
                append_job_output(
                    &jobs,
                    &job_id,
                    "stderr",
                    &format!("Failed to wait: {error}"),
                );
                finish_job(&jobs, &job_id, JobStatus::Failed, None);
                return;
            }
        }
    };

    let exit_code = status.code();
    let canceled = jobs
        .lock()
        .expect("job registry lock")
        .get(&job_id)
        .map(|job| job.cancel_requested)
        .unwrap_or(false);
    let status = if canceled {
        JobStatus::Canceled
    } else if status.success() {
        JobStatus::Succeeded
    } else {
        JobStatus::Failed
    };
    finish_job(&jobs, &job_id, status, exit_code);
}

fn spawn_output_reader<R>(jobs: JobRegistry, job_id: String, stream: &'static str, reader: R)
where
    R: std::io::Read + Send + 'static,
{
    std::thread::spawn(move || {
        for line in BufReader::new(reader).lines().map_while(Result::ok) {
            append_job_output(&jobs, &job_id, stream, &line);
            if let Some(progress) = infer_progress(&line) {
                update_job_progress(&jobs, &job_id, progress);
            }
        }
    });
}

fn append_job_output(jobs: &JobRegistry, job_id: &str, stream: &str, line: &str) {
    let at_ms = now_ms();
    let event = JobEvent::Output {
        stream: stream.into(),
        line: line.into(),
        at_ms,
    };
    let mut jobs = jobs.lock().expect("job registry lock");
    if let Some(job) = jobs.get_mut(job_id) {
        job.summary.updated_at_ms = at_ms;
        job.output.push_back(JobOutputLine {
            stream: stream.into(),
            line: line.into(),
            at_ms,
        });
        while job.output.len() > JOB_OUTPUT_LIMIT {
            job.output.pop_front();
        }
        let _ = job.events.send(event);
    }
}

fn update_job_progress(jobs: &JobRegistry, job_id: &str, progress: JobProgress) {
    let at_ms = now_ms();
    let mut jobs = jobs.lock().expect("job registry lock");
    if let Some(job) = jobs.get_mut(job_id) {
        job.summary.updated_at_ms = at_ms;
        job.summary.progress = progress.clone();
        let _ = job.events.send(JobEvent::Progress { progress, at_ms });
    }
}

fn finish_job(jobs: &JobRegistry, job_id: &str, status: JobStatus, exit_code: Option<i32>) {
    let mut jobs = jobs.lock().expect("job registry lock");
    if let Some(job) = jobs.get_mut(job_id) {
        job.summary.updated_at_ms = now_ms();
        job.summary.status = status;
        job.summary.exit_code = exit_code;
        job.summary.progress.phase = match job.summary.status {
            JobStatus::Running => job.summary.progress.phase.clone(),
            JobStatus::Succeeded => "Complete".into(),
            JobStatus::Failed => "Failed".into(),
            JobStatus::Canceled => "Canceled".into(),
        };
        job.summary.progress.current = Some(1);
        job.summary.progress.total = Some(1);
        job.child = None;
        let _ = job.events.send(JobEvent::Status {
            summary: job.summary.clone(),
        });
    }
}

fn infer_progress(line: &str) -> Option<JobProgress> {
    let lower = line.to_ascii_lowercase();
    let phase = if lower.contains("compiling") {
        "Compiling"
    } else if lower.contains("checking") {
        "Checking"
    } else if lower.contains("downloading") {
        "Downloading"
    } else if lower.contains("training") || lower.contains("epoch") {
        "Training"
    } else if lower.contains("prepar") {
        "Preparing"
    } else if lower.contains("finished") {
        "Finishing"
    } else {
        return None;
    };
    Some(JobProgress {
        phase: phase.into(),
        current: None,
        total: None,
    })
}

fn sse_event(event: &JobEvent) -> Event {
    Event::default().json_data(event).unwrap_or_else(|_| {
        Event::default().data(
            "{\"type\":\"output\",\"stream\":\"stderr\",\"line\":\"failed to serialize event\"}",
        )
    })
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn system_time_ms(time: SystemTime) -> Option<u128> {
    time.duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis())
}

fn safe_relative_path(input: &str) -> Result<PathBuf, String> {
    let input = input.trim().trim_start_matches('/');
    let path = FsPath::new(input);
    if path.is_absolute() {
        return Err("absolute paths are not available through the web file browser".into());
    }
    let mut relative = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => relative.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err("paths must stay inside the Tongues workspace".into());
            }
        }
    }
    Ok(relative)
}

fn path_to_web(path: &FsPath) -> String {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_string_lossy().to_string()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn parent_web_path(path: &FsPath) -> Option<String> {
    let parent = path.parent()?;
    let parent = path_to_web(parent);
    if parent.is_empty() {
        None
    } else {
        Some(parent)
    }
}

fn download_url_for(path: &FsPath) -> String {
    format!(
        "/api/files/download/{}",
        url_path_escape(&path_to_web(path))
    )
}

fn artifact_browser_root_entries(workspace_root: &FsPath) -> Vec<FileEntry> {
    let mut entries = ALLOWED_ARTIFACT_ROOTS
        .iter()
        .filter_map(|root| {
            let relative = PathBuf::from(root);
            let path = resolve_existing_artifact_path(workspace_root, &relative).ok()?;
            let metadata = std::fs::metadata(path).ok()?;
            if !metadata.is_dir() {
                return None;
            }
            Some(FileEntry {
                name: (*root).into(),
                path: path_to_web(&relative),
                kind: "dir".into(),
                size: None,
                modified_ms: metadata.modified().ok().and_then(system_time_ms),
                download_url: None,
            })
        })
        .collect::<Vec<_>>();
    for file in ALLOWED_ROOT_ARTIFACT_FILES {
        let relative = PathBuf::from(file);
        let Ok(path) = resolve_existing_artifact_path(workspace_root, &relative) else {
            continue;
        };
        let Ok(metadata) = std::fs::metadata(path) else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        entries.push(FileEntry {
            name: (*file).into(),
            path: path_to_web(&relative),
            kind: "file".into(),
            size: Some(metadata.len()),
            modified_ms: metadata.modified().ok().and_then(system_time_ms),
            download_url: Some(download_url_for(&relative)),
        });
    }
    entries.sort_by(|left, right| {
        let left_dir = left.kind == "dir";
        let right_dir = right.kind == "dir";
        right_dir
            .cmp(&left_dir)
            .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
    });
    entries
}

fn resolve_existing_artifact_path(
    workspace_root: &FsPath,
    relative: &FsPath,
) -> Result<PathBuf, String> {
    validate_artifact_relative_path(workspace_root, relative)?;
    let workspace = canonical_workspace_root(workspace_root)?;
    let path = workspace_root.join(relative);
    let resolved = path
        .canonicalize()
        .map_err(|error| format!("failed to resolve {}: {error}", path.display()))?;
    if !resolved.starts_with(&workspace) {
        return Err("paths must stay inside approved artifact roots".into());
    }
    let metadata = std::fs::metadata(&resolved)
        .map_err(|error| format!("failed to read {}: {error}", resolved.display()))?;
    if !is_visible_artifact_path(relative, metadata.is_dir()) {
        return Err("requested path is not exposed by the local server".into());
    }
    if let Some(root) = artifact_root_name(relative) {
        let root_path = workspace_root.join(root);
        if root_path.exists() {
            let resolved_root = root_path
                .canonicalize()
                .map_err(|error| format!("failed to resolve {}: {error}", root_path.display()))?;
            if !resolved.starts_with(&resolved_root) {
                return Err("paths must stay inside approved artifact roots".into());
            }
        }
    }
    Ok(resolved)
}

fn validate_artifact_relative_path(
    workspace_root: &FsPath,
    relative: &FsPath,
) -> Result<(), String> {
    if relative.as_os_str().is_empty() {
        return Ok(());
    }
    if is_allowed_root_artifact_file(relative) {
        validate_existing_ancestor_within_workspace(workspace_root, relative)?;
        return Ok(());
    }
    let Some(root) = artifact_root_name(relative) else {
        return Err("paths must stay inside approved artifact roots".into());
    };
    if !ALLOWED_ARTIFACT_ROOTS
        .iter()
        .any(|allowed| *allowed == root)
    {
        return Err("paths must stay inside approved artifact roots".into());
    }
    validate_existing_ancestor_within_workspace(workspace_root, relative)
}

fn validate_existing_ancestor_within_workspace(
    workspace_root: &FsPath,
    relative: &FsPath,
) -> Result<(), String> {
    let workspace = canonical_workspace_root(workspace_root)?;
    let absolute = workspace_root.join(relative);
    let anchor = deepest_existing_path(&absolute)
        .ok_or_else(|| format!("path is not available: {}", path_to_web(relative)))?;
    let resolved_anchor = anchor
        .canonicalize()
        .map_err(|error| format!("failed to resolve {}: {error}", anchor.display()))?;
    if !resolved_anchor.starts_with(&workspace) {
        return Err("paths must stay inside the Tongues workspace".into());
    }
    Ok(())
}

fn canonical_workspace_root(workspace_root: &FsPath) -> Result<PathBuf, String> {
    workspace_root
        .canonicalize()
        .map_err(|error| format!("failed to resolve workspace root: {error}"))
}

fn deepest_existing_path(path: &FsPath) -> Option<PathBuf> {
    let mut current = path;
    loop {
        if current.exists() {
            return Some(current.to_path_buf());
        }
        current = current.parent()?;
    }
}

fn artifact_root_name(relative: &FsPath) -> Option<&str> {
    match relative.components().next()? {
        Component::Normal(component) => component.to_str(),
        _ => None,
    }
}

fn is_allowed_root_artifact_file(relative: &FsPath) -> bool {
    relative
        .parent()
        .is_none_or(|parent| parent.as_os_str().is_empty())
        && relative
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                ALLOWED_ROOT_ARTIFACT_FILES
                    .iter()
                    .any(|allowed| *allowed == name)
            })
}

fn is_visible_artifact_path(relative: &FsPath, is_dir: bool) -> bool {
    let Some(name) = relative.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    if name.starts_with('.') {
        return false;
    }
    let lower = name.to_ascii_lowercase();
    if matches!(lower.as_str(), ".certs" | ".git" | "target") {
        return false;
    }
    if is_dir {
        return true;
    }
    !matches!(
        lower.as_str(),
        ".env" | "cargo.lock" | "cargo.toml" | "config.json" | "justfile"
    ) && !lower.starts_with(".env.")
        && !lower.ends_with(".cer")
        && !lower.ends_with(".crt")
        && !lower.ends_with(".csr")
        && !lower.ends_with(".der")
        && !lower.ends_with(".key")
        && !lower.ends_with(".p12")
        && !lower.ends_with(".pem")
        && !lower.ends_with(".pfx")
        && !lower.ends_with(".toml")
        && !lower.ends_with(".yaml")
        && !lower.ends_with(".yml")
}

fn artifacts_for_job(workspace_root: &FsPath, args: &[String]) -> Vec<JobArtifact> {
    let mut artifacts = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        if output_path_flag(flag) {
            if let Some(value) = args.get(index + 1) {
                add_artifacts_for_path(workspace_root, value, flag, &mut artifacts);
            }
            index += 2;
        } else {
            index += 1;
        }
    }
    artifacts
}

fn output_path_flag(flag: &str) -> bool {
    matches!(
        flag,
        "--out" | "--out-dir" | "--output" | "--archive-dir" | "--cache-dir"
    )
}

fn add_artifacts_for_path(
    workspace_root: &FsPath,
    value: &str,
    flag: &str,
    artifacts: &mut Vec<JobArtifact>,
) {
    let Some((relative, path)) = job_path_to_relative(workspace_root, value) else {
        return;
    };
    let Ok(metadata) = std::fs::metadata(&path) else {
        return;
    };
    if metadata.is_file() {
        artifacts.push(JobArtifact {
            label: format!("{flag} {}", path_to_web(&relative)),
            path: path_to_web(&relative),
            kind: "file".into(),
            size: Some(metadata.len()),
            download_url: Some(download_url_for(&relative)),
        });
        return;
    }

    if metadata.is_dir() {
        artifacts.push(JobArtifact {
            label: format!("{flag} {}", path_to_web(&relative)),
            path: path_to_web(&relative),
            kind: "dir".into(),
            size: None,
            download_url: None,
        });
        if let Ok(read_dir) = std::fs::read_dir(path) {
            let mut files = read_dir
                .flatten()
                .filter_map(|entry| {
                    let file_relative = relative.join(entry.file_name());
                    let file_path =
                        resolve_existing_artifact_path(workspace_root, &file_relative).ok()?;
                    let metadata = std::fs::metadata(&file_path).ok()?;
                    if !metadata.is_file() || !is_visible_artifact_path(&file_relative, false) {
                        return None;
                    }
                    Some(JobArtifact {
                        label: path_to_web(&file_relative),
                        path: path_to_web(&file_relative),
                        kind: "file".into(),
                        size: Some(metadata.len()),
                        download_url: Some(download_url_for(&file_relative)),
                    })
                })
                .take(40)
                .collect::<Vec<_>>();
            files.sort_by(|left, right| left.path.cmp(&right.path));
            artifacts.extend(files);
        }
    }
}

fn job_path_to_relative(workspace_root: &FsPath, value: &str) -> Option<(PathBuf, PathBuf)> {
    let path = FsPath::new(value.trim());
    let relative = if path.is_absolute() {
        let workspace = canonical_workspace_root(workspace_root).ok()?;
        let resolved = path.canonicalize().ok()?;
        resolved.strip_prefix(&workspace).ok().map(PathBuf::from)?
    } else {
        safe_relative_path(value).ok()?
    };
    let resolved = resolve_existing_artifact_path(workspace_root, &relative).ok()?;
    Some((relative, resolved))
}

type ResidentSpeechBackend = Box<dyn tongues_tts::SynthesizerBackend>;

struct MockSynthesizer {
    capabilities: tongues_tts::BackendCapabilities,
    backend: styletts2::MockStyleTts2Backend,
}

impl tongues_tts::SynthesizerBackend for MockSynthesizer {
    fn capabilities(&self) -> tongues_tts::BackendCapabilities {
        self.capabilities.clone()
    }

    fn synthesize(
        &mut self,
        request: &tongues_tts::UnifiedSynthesisRequest,
        sink: &mut dyn tongues_tts::NormalizedAudioSink,
    ) -> Result<tongues_tts::UnifiedSynthesisOutput, tongues_tts::SynthesisContractError> {
        use styletts2::StyleTts2Backend;
        self.capabilities.validate(request)?;
        let plan = tongues_tts::utterance_plan_from_text(tongues_tts::SpeechRequest {
            text: request.text.clone(),
            variety: request.variety.clone(),
        })
        .map_err(|error| tongues_tts::SynthesisContractError::Backend {
            message: format!("{error:#}"),
        })?;
        let backend_plan = styletts2::prepare_styletts2_plan(
            &plan,
            &styletts2::styletts2_en_us_symbol_set(),
            styletts2::StyleTts2PlanOptions::default(),
        )
        .map_err(|error| tongues_tts::SynthesisContractError::Backend {
            message: error.to_string(),
        })?;
        let backend_request = styletts2::StyleTts2SynthesisRequest::from_backend_plan(
            backend_plan,
            plan.speaker,
            plan.style,
            plan.target_prosody,
        );
        let mut frames = 0_u64;
        let mut frame_offset = 0_u64;
        let mut sink_failure = None;
        let synthesis_result = self.backend.synthesize_streaming(
            &backend_request,
            &mut |chunk: styletts2::StyleTts2AudioChunk| {
                let chunk_frames = chunk.pcm_mono_f32.len() as u64;
                if let Err(error) = sink.emit(tongues_tts::NormalizedAudioChunk {
                    chunk_index: chunk.chunk_index,
                    is_final: chunk.is_final,
                    frame_offset,
                    sample_rate_hz: chunk.sample_rate_hz,
                    channels: 1,
                    pcm_f32: chunk.pcm_mono_f32,
                }) {
                    let message = error.to_string();
                    sink_failure = Some(error);
                    return Err(styletts2::StyleTts2Error::Backend { message });
                }
                frames += chunk_frames;
                frame_offset += chunk_frames;
                Ok(())
            },
        );
        if let Some(error) = sink_failure {
            return Err(error);
        }
        let output =
            synthesis_result.map_err(|error| tongues_tts::SynthesisContractError::Backend {
                message: error.to_string(),
            })?;
        Ok(tongues_tts::UnifiedSynthesisOutput {
            metadata: tongues_tts::SynthesisMetadata {
                backend: self.capabilities.backend.clone(),
                model: self.capabilities.model.clone(),
                sample_rate_hz: output.sample_rate_hz,
                channels: 1,
                frames,
                audio_seconds: frames as f64 / f64::from(output.sample_rate_hz),
                streaming: request.streaming,
                input_audio: Vec::new(),
                timings: output
                    .timings
                    .into_iter()
                    .map(|timing| tongues_tts::SynthesisTiming {
                        stage: timing.stage,
                        elapsed_ms: timing.elapsed_ms,
                    })
                    .collect(),
            },
        })
    }
}

#[derive(Default)]
struct ResidentSpeechService {
    engines: HashMap<String, ResidentSpeechBackend>,
    failures: HashMap<String, String>,
}

#[derive(Serialize)]
struct ResidentSpeechRuntimeResponse {
    state: &'static str,
    device: String,
    device_index: Option<usize>,
    concurrency: &'static str,
    busy: bool,
    capacity: usize,
    active: usize,
    queued: usize,
    loaded: Vec<String>,
    failed: BTreeMap<String, String>,
}

#[derive(Clone)]
struct ResidentSynthesisContext {
    voice_reference: Option<PathBuf>,
    style_reference: Option<PathBuf>,
    source_reference: Option<PathBuf>,
    emotion_vector: Option<Vec<f32>>,
}

struct ResidentSynthesisOutput {
    wav: Vec<u8>,
    engine_key: String,
    loaded_now: bool,
    queue_ms: f64,
    load_ms: f64,
    synthesis_ms: f64,
    sample_rate_hz: u32,
    channels: u16,
    sample_count: u64,
    audio_seconds: f64,
    real_time_factor: f64,
    profile: Vec<tongues_tts::SynthesisTiming>,
    input_audio: Vec<tongues_tts::InputAudioMetadata>,
    device: tongues_tts::ResolvedSpeechDevice,
    pronunciation_warnings: Vec<speaking::PronunciationWarning>,
    pronunciation_plan: Option<speaking::PhonemicizeOutput>,
}

impl ResidentSpeechService {
    fn snapshot(
        &self,
        phase: u8,
        admission: &SpeechAdmission,
        device: tongues_tts::ResolvedSpeechDevice,
    ) -> ResidentSpeechRuntimeResponse {
        let mut loaded = self.engines.keys().cloned().collect::<Vec<_>>();
        loaded.sort();
        let active = phase != SPEECH_PHASE_IDLE;
        let (active_count, queued) = admission.counts(active);
        ResidentSpeechRuntimeResponse {
            state: speech_runtime_state(phase, !loaded.is_empty(), !self.failures.is_empty()),
            device: device.kind().into(),
            device_index: device.index(),
            concurrency: "bounded-fifo",
            busy: active,
            capacity: admission.capacity,
            active: active_count,
            queued,
            loaded,
            failed: self
                .failures
                .iter()
                .map(|(key, error)| (key.clone(), error.clone()))
                .collect(),
        }
    }

    fn synthesize(
        &mut self,
        payload: &SpeakRequest,
        context: &ResidentSynthesisContext,
        phase: &AtomicU8,
        device: tongues_tts::ResolvedSpeechDevice,
    ) -> anyhow::Result<ResidentSynthesisOutput> {
        // Run the shared pronunciation pipeline first to obtain warnings.  This
        // uses the exact same code path that each engine backend calls internally,
        // so no separate server-side pronunciation algorithm is introduced.
        let variety = payload.variety.as_deref().unwrap_or("en-US");
        let phonemicized = tongues_tts::phonemicize_speech_text(tongues_tts::SpeechRequest {
            text: payload.text.clone(),
            variety: variety.to_string(),
        })?;

        // Enforce strict mode before loading or running the acoustic model.
        if payload.fail_on_guessed_pronunciation.unwrap_or(false) {
            let guessed: Vec<&str> = phonemicized
                .warnings
                .iter()
                .filter(|w| is_guessed_pronunciation_warning(w))
                .map(|w| w.token.as_str())
                .collect();
            if !guessed.is_empty() {
                anyhow::bail!("guessed_pronunciation: {}", guessed.join(", "));
            }
        }

        let pronunciation_warnings = phonemicized.warnings.clone();
        let pronunciation_plan = if payload.debug_pronunciation.unwrap_or(false) {
            Some(phonemicized)
        } else {
            None
        };

        let backend_name = payload.backend.as_deref().unwrap_or("burn");
        let engine_key = resident_engine_key(backend_name, device, payload)?;
        let mut loaded_now = false;
        let load_started = std::time::Instant::now();
        if !self.engines.contains_key(&engine_key) {
            phase.store(SPEECH_PHASE_LOADING, Ordering::Release);
            if let Some(error) = self.failures.get(&engine_key) {
                anyhow::bail!("resident speech engine `{engine_key}` failed to load: {error}");
            }
            match load_resident_speech_backend(backend_name, device, payload) {
                Ok(engine) => {
                    self.engines.insert(engine_key.clone(), engine);
                    loaded_now = true;
                }
                Err(error) => {
                    let error = format!("{error:#}");
                    self.failures.insert(engine_key.clone(), error.clone());
                    anyhow::bail!("failed to load resident speech engine `{engine_key}`: {error}");
                }
            }
        }
        let load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
        phase.store(SPEECH_PHASE_SYNTHESIZING, Ordering::Release);

        let request = unified_synthesis_request(payload, context, device);

        let engine = self
            .engines
            .get_mut(&engine_key)
            .expect("resident engine inserted before synthesis");
        let synthesis_started = std::time::Instant::now();
        let mut pcm = Vec::new();
        let output =
            engine.synthesize(&request, &mut |chunk: tongues_tts::NormalizedAudioChunk| {
                pcm.extend(chunk.pcm_f32);
                Ok(())
            })?;
        let sample_rate_hz = output.metadata.sample_rate_hz;
        let channels = output.metadata.channels;
        let sample_count = pcm.len() as u64;
        let profile = output.metadata.timings;
        let input_audio = output.metadata.input_audio;
        let synthesis_ms = synthesis_started.elapsed().as_secs_f64() * 1_000.0;
        let audio_seconds =
            sample_count as f64 / sample_rate_hz as f64 / f64::from(channels.max(1));
        let real_time_factor = if audio_seconds > 0.0 {
            synthesis_ms / 1_000.0 / audio_seconds
        } else {
            0.0
        };
        let wav = encode_wav_mono_f32(sample_rate_hz, &pcm)?;
        Ok(ResidentSynthesisOutput {
            wav,
            engine_key,
            loaded_now,
            queue_ms: 0.0,
            load_ms,
            synthesis_ms,
            sample_rate_hz,
            channels,
            sample_count,
            audio_seconds,
            real_time_factor,
            profile,
            input_audio,
            device,
            pronunciation_warnings,
            pronunciation_plan,
        })
    }
}

fn is_guessed_pronunciation_warning(warning: &speaking::PronunciationWarning) -> bool {
    matches!(
        warning.kind,
        speaking::PronunciationWarningKind::GuessedWord
            | speaking::PronunciationWarningKind::MixedAlphaNumeric
            | speaking::PronunciationWarningKind::UnknownPronunciation
    )
}

fn unified_synthesis_request(
    payload: &SpeakRequest,
    context: &ResidentSynthesisContext,
    device: tongues_tts::ResolvedSpeechDevice,
) -> tongues_tts::UnifiedSynthesisRequest {
    let backend = payload.backend.as_deref().unwrap_or("burn");
    let speaker = payload
        .speaker
        .clone()
        .filter(|speaker| !speaker.is_empty())
        .map(tongues_tts::SpeakerSelection::Named)
        .or_else(|| {
            payload
                .speaker_id
                .map(tongues_tts::SpeakerSelection::Numeric)
        });
    let model_language = payload
        .model_language
        .clone()
        .filter(|language| !language.is_empty())
        .map(tongues_tts::LanguageSelection::Named)
        .or_else(|| {
            payload
                .language_id
                .map(tongues_tts::LanguageSelection::Numeric)
        });
    let style = (backend == "styletts2").then(|| tongues_tts::StyleSelection {
        name: payload
            .emotion
            .clone()
            .filter(|emotion| !emotion.is_empty()),
        embedding: context.emotion_vector.clone(),
        embedding_is_delta: context.emotion_vector.is_some(),
        strength: payload.emotion_strength.unwrap_or(1.0),
        speaker_blend: Some(
            payload
                .speaker_reference_strength
                .map(|strength| 1.0 - strength)
                .or(payload.style_alpha)
                .unwrap_or(0.3),
        ),
        style_blend: Some(
            payload
                .style_reference_strength
                .map(|strength| 1.0 - strength)
                .or(payload.style_beta)
                .unwrap_or(0.1),
        ),
        diffusion_steps: Some(payload.diffusion_steps.unwrap_or_else(|| {
            if payload.quality.as_deref() == Some("fast") {
                2
            } else {
                5
            }
        })),
        embedding_scale: Some(payload.embedding_scale.unwrap_or(1.0)),
    });
    tongues_tts::UnifiedSynthesisRequest {
        text: payload.text.clone(),
        variety: payload.variety.clone().unwrap_or_else(|| "en-US".into()),
        model_language,
        speaker,
        reference_audio: tongues_tts::ReferenceAudioRequest {
            speaker: context
                .voice_reference
                .as_ref()
                .map(|path| path.display().to_string()),
            style: context
                .style_reference
                .as_ref()
                .map(|path| path.display().to_string()),
            source: context
                .source_reference
                .as_ref()
                .map(|path| path.display().to_string()),
        },
        style,
        speed: payload.speed.unwrap_or(1.0) as f32,
        pitch_scale: payload.pitch_scale,
        pitch_shift: payload.pitch_shift,
        pitch: payload.pitch.clone(),
        energy_scale: payload.energy_scale,
        energy_shift: payload.energy_shift,
        energy: payload.energy.clone(),
        durations: payload.durations.clone(),
        seed: payload
            .seed
            .or_else(|| (backend == "styletts2").then_some(payload.style_seed.unwrap_or(0))),
        noise_scale: payload.noise_scale,
        duration_noise_scale: payload.duration_noise_scale,
        device: match device {
            tongues_tts::ResolvedSpeechDevice::Cpu => tongues_tts::SpeechDeviceRequest::Cpu,
            tongues_tts::ResolvedSpeechDevice::Cuda { index } => {
                tongues_tts::SpeechDeviceRequest::Cuda { index }
            }
        },
        streaming: false,
        profile: payload.timings.unwrap_or(false),
        max_chunk_symbols: payload.max_tts_symbols,
        chunking: !payload.no_tts_chunking.unwrap_or(false),
    }
}

fn cuda_probe_failure_reason(index: usize) -> Option<String> {
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(|| {
        let device = CudaDevice::new(index);
        type B = Cuda<f32, i32>;
        let tensor = burn::tensor::Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let _ = tensor.into_data();
    });
    panic::set_hook(default_hook);

    match result {
        Ok(_) => None,
        Err(payload) => Some(format_panic_payload(payload.as_ref())),
    }
}

fn format_panic_payload(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown CUDA initialization failure".to_string()
    }
}

fn resident_speech_device_for(
    payload: &SpeakRequest,
    default_device: tongues_tts::ResolvedSpeechDevice,
) -> anyhow::Result<tongues_tts::ResolvedSpeechDevice> {
    if payload.cpu.unwrap_or(false) {
        Ok(tongues_tts::ResolvedSpeechDevice::Cpu)
    } else if let Some(index) = payload.cuda_device {
        Ok(tongues_tts::resolve_speech_device(
            tongues_tts::SpeechDeviceRequest::Cuda { index },
            |index| cuda_probe_failure_reason(index).map_or(Ok(()), Err),
        )?
        .resolved)
    } else if payload.backend.as_deref() == Some("freevc") {
        Ok(tongues_tts::ResolvedSpeechDevice::Cpu)
    } else {
        Ok(default_device)
    }
}

fn speech_max_in_flight() -> usize {
    std::env::var("TONGUES_SPEECH_MAX_IN_FLIGHT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=32).contains(value))
        .unwrap_or(DEFAULT_SPEECH_MAX_IN_FLIGHT)
}

fn speech_runtime_state(phase: u8, has_loaded: bool, has_failures: bool) -> &'static str {
    match phase {
        SPEECH_PHASE_LOADING => "loading",
        SPEECH_PHASE_SYNTHESIZING => "busy",
        SPEECH_PHASE_RELOADING => "reloading",
        _ if has_failures => "failed",
        _ if has_loaded => "ready",
        _ => "idle",
    }
}

struct SpeechPhaseReset(Arc<AtomicU8>);

impl Drop for SpeechPhaseReset {
    fn drop(&mut self) {
        self.0.store(SPEECH_PHASE_IDLE, Ordering::Release);
    }
}

fn resident_engine_key(
    backend: &str,
    device: tongues_tts::ResolvedSpeechDevice,
    payload: &SpeakRequest,
) -> anyhow::Result<String> {
    let device_key = match device {
        tongues_tts::ResolvedSpeechDevice::Cpu => "cpu".to_string(),
        tongues_tts::ResolvedSpeechDevice::Cuda { index: 0 } => "cuda".to_string(),
        tongues_tts::ResolvedSpeechDevice::Cuda { index } => format!("cuda-{index}"),
    };
    if let Some(pipeline) = payload.pipeline.as_ref() {
        return Ok(format!(
            "pipeline:{}:{device_key}",
            pipeline.canonical_id()?
        ));
    }
    Ok(match backend {
        "onnx" | "fairseq" => format!(
            "{backend}-{}-{device_key}",
            speech_model_id(&resolve_mortar_home(), backend, payload.model.as_deref(),)?
        ),
        "styletts2" => format!(
            "styletts2-{}-{device_key}",
            speech_model_id(&resolve_mortar_home(), backend, payload.model.as_deref(),)?
        ),
        "mock" => format!("mock-{}", payload.sample_rate_hz.unwrap_or(24_000)),
        _ => format!("{backend}-{device_key}"),
    })
}

fn load_resident_speech_backend(
    backend: &str,
    device: tongues_tts::ResolvedSpeechDevice,
    payload: &SpeakRequest,
) -> anyhow::Result<ResidentSpeechBackend> {
    let home = resolve_mortar_home();
    verify_catalog_backend(&home, backend, payload.model.as_deref())?;
    let provider = RESIDENT_BACKEND_PROVIDERS
        .iter()
        .find(|provider| provider.id == backend)
        .with_context(|| format!("resident speech backend `{backend}` is not registered"))?;
    let capabilities = speech_backend_capabilities(
        &home,
        backend,
        payload.model.as_deref(),
        device,
        payload.sample_rate_hz.unwrap_or(24_000),
    )?;
    (provider.load)(&home, device, payload, capabilities)
}

struct ResidentBackendProvider {
    id: &'static str,
    load: fn(
        &FsPath,
        tongues_tts::ResolvedSpeechDevice,
        &SpeakRequest,
        tongues_tts::BackendCapabilities,
    ) -> anyhow::Result<ResidentSpeechBackend>,
}

const RESIDENT_BACKEND_PROVIDERS: &[ResidentBackendProvider] = &[
    ResidentBackendProvider {
        id: "burn",
        load: load_burn_provider,
    },
    ResidentBackendProvider {
        id: "fastpitch",
        load: load_fastpitch_provider,
    },
    ResidentBackendProvider {
        id: "glow",
        load: load_glow_provider,
    },
    ResidentBackendProvider {
        id: "vits",
        load: load_vits_provider,
    },
    ResidentBackendProvider {
        id: "fairseq",
        load: load_fairseq_provider,
    },
    ResidentBackendProvider {
        id: "yourtts",
        load: load_yourtts_provider,
    },
    ResidentBackendProvider {
        id: "freevc",
        load: load_freevc_provider,
    },
    ResidentBackendProvider {
        id: "onnx",
        load: load_onnx_provider,
    },
    ResidentBackendProvider {
        id: "styletts2",
        load: load_styletts2_provider,
    },
    ResidentBackendProvider {
        id: "mock",
        load: load_mock_provider,
    },
];

fn load_burn_provider(
    home: &FsPath,
    device: tongues_tts::ResolvedSpeechDevice,
    _payload: &SpeakRequest,
    capabilities: tongues_tts::BackendCapabilities,
) -> anyhow::Result<ResidentSpeechBackend> {
    match device {
        tongues_tts::ResolvedSpeechDevice::Cpu => {
            Ok(Box::new(tongues_tts::PlanEngineBackend::new(
                capabilities,
                device,
                load_resident_burn::<NdArray<f32>>(home, NdArrayDevice::Cpu)?,
            )))
        }
        tongues_tts::ResolvedSpeechDevice::Cuda { index } => {
            Ok(Box::new(tongues_tts::PlanEngineBackend::new(
                capabilities,
                device,
                load_resident_burn::<Cuda<f32, i32>>(home, CudaDevice::new(index))?,
            )))
        }
    }
}

fn load_fastpitch_provider(
    home: &FsPath,
    device: tongues_tts::ResolvedSpeechDevice,
    _payload: &SpeakRequest,
    capabilities: tongues_tts::BackendCapabilities,
) -> anyhow::Result<ResidentSpeechBackend> {
    match device {
        tongues_tts::ResolvedSpeechDevice::Cpu => {
            Ok(Box::new(tongues_tts::PlanEngineBackend::new(
                capabilities,
                device,
                load_resident_fastpitch::<NdArray<f32>>(home, NdArrayDevice::Cpu)?,
            )))
        }
        tongues_tts::ResolvedSpeechDevice::Cuda { index } => {
            Ok(Box::new(tongues_tts::PlanEngineBackend::new(
                capabilities,
                device,
                load_resident_fastpitch::<Cuda<f32, i32>>(home, CudaDevice::new(index))?,
            )))
        }
    }
}

fn load_glow_provider(
    home: &FsPath,
    device: tongues_tts::ResolvedSpeechDevice,
    _payload: &SpeakRequest,
    capabilities: tongues_tts::BackendCapabilities,
) -> anyhow::Result<ResidentSpeechBackend> {
    match device {
        tongues_tts::ResolvedSpeechDevice::Cpu => {
            Ok(Box::new(tongues_tts::PlanEngineBackend::new(
                capabilities,
                device,
                load_resident_glow::<NdArray<f32>>(home, NdArrayDevice::Cpu)?,
            )))
        }
        tongues_tts::ResolvedSpeechDevice::Cuda { index } => {
            Ok(Box::new(tongues_tts::PlanEngineBackend::new(
                capabilities,
                device,
                load_resident_glow::<Cuda<f32, i32>>(home, CudaDevice::new(index))?,
            )))
        }
    }
}

fn load_vits_provider(
    home: &FsPath,
    device: tongues_tts::ResolvedSpeechDevice,
    _payload: &SpeakRequest,
    capabilities: tongues_tts::BackendCapabilities,
) -> anyhow::Result<ResidentSpeechBackend> {
    match device {
        tongues_tts::ResolvedSpeechDevice::Cpu => {
            Ok(Box::new(tongues_tts::PlanEngineBackend::new(
                capabilities,
                device,
                load_resident_vits::<NdArray<f32>>(home, NdArrayDevice::Cpu)?,
            )))
        }
        tongues_tts::ResolvedSpeechDevice::Cuda { index } => {
            Ok(Box::new(tongues_tts::PlanEngineBackend::new(
                capabilities,
                device,
                load_resident_vits::<Cuda<f32, i32>>(home, CudaDevice::new(index))?,
            )))
        }
    }
}

fn load_fairseq_provider(
    home: &FsPath,
    device: tongues_tts::ResolvedSpeechDevice,
    payload: &SpeakRequest,
    capabilities: tongues_tts::BackendCapabilities,
) -> anyhow::Result<ResidentSpeechBackend> {
    let model = speech_model_id(home, "fairseq", payload.model.as_deref())?;
    let catalog = tongues_tts::ModelCatalog::with_private_catalogs(
        &tongues_tts::private_catalog_paths_from_environment(),
    )?;
    let entry = catalog
        .find(&model)
        .with_context(|| format!("Fairseq MMS model `{model}` is not in the catalog"))?;
    let checkpoint = entry
        .artifacts
        .iter()
        .find(|artifact| {
            FsPath::new(&artifact.install_path)
                .file_name()
                .and_then(|name| name.to_str())
                == Some(tongues_tts::FAIRSEQ_MMS_CHECKPOINT)
        })
        .context("Fairseq MMS catalog entry has no checkpoint artifact")?;
    let model_dir = home
        .join(&checkpoint.install_path)
        .parent()
        .context("Fairseq MMS checkpoint path has no parent")?
        .to_path_buf();
    let language = entry
        .languages
        .first()
        .context("Fairseq MMS catalog entry has no language identity")?;
    match device {
        tongues_tts::ResolvedSpeechDevice::Cpu => {
            Ok(Box::new(tongues_tts::PlanEngineBackend::new(
                capabilities,
                device,
                tongues_tts::BurnVitsSpeech::<NdArray<f32>>::load_fairseq(
                    model_dir,
                    language,
                    NdArrayDevice::Cpu,
                )?,
            )))
        }
        tongues_tts::ResolvedSpeechDevice::Cuda { index } => {
            Ok(Box::new(tongues_tts::PlanEngineBackend::new(
                capabilities,
                device,
                tongues_tts::BurnVitsSpeech::<Cuda<f32, i32>>::load_fairseq(
                    model_dir,
                    language,
                    CudaDevice::new(index),
                )?,
            )))
        }
    }
}

fn load_yourtts_provider(
    home: &FsPath,
    device: tongues_tts::ResolvedSpeechDevice,
    _payload: &SpeakRequest,
    capabilities: tongues_tts::BackendCapabilities,
) -> anyhow::Result<ResidentSpeechBackend> {
    match device {
        tongues_tts::ResolvedSpeechDevice::Cpu => {
            Ok(Box::new(tongues_tts::PlanEngineBackend::new(
                capabilities,
                device,
                load_resident_yourtts::<NdArray<f32>>(home, NdArrayDevice::Cpu)?,
            )))
        }
        tongues_tts::ResolvedSpeechDevice::Cuda { index } => {
            Ok(Box::new(tongues_tts::PlanEngineBackend::new(
                capabilities,
                device,
                load_resident_yourtts::<Cuda<f32, i32>>(home, CudaDevice::new(index))?,
            )))
        }
    }
}

fn load_freevc_provider(
    home: &FsPath,
    device: tongues_tts::ResolvedSpeechDevice,
    _payload: &SpeakRequest,
    _capabilities: tongues_tts::BackendCapabilities,
) -> anyhow::Result<ResidentSpeechBackend> {
    match device {
        tongues_tts::ResolvedSpeechDevice::Cpu => {
            Ok(Box::new(tongues_tts::FreeVc::<NdArray<f32>>::load(
                home.join(FREEVC_RELATIVE_DIR),
                NdArrayDevice::Cpu,
            )?))
        }
        tongues_tts::ResolvedSpeechDevice::Cuda { .. } => {
            anyhow::bail!("the FreeVC backend currently supports CPU inference")
        }
    }
}

fn load_onnx_provider(
    home: &FsPath,
    device: tongues_tts::ResolvedSpeechDevice,
    payload: &SpeakRequest,
    capabilities: tongues_tts::BackendCapabilities,
) -> anyhow::Result<ResidentSpeechBackend> {
    let model_id = speech_model_id(home, "onnx", payload.model.as_deref())?;
    let model_path = onnx_voice_model_path(home, &model_id)?;
    let config =
        tongues_tts::VoiceConfig::from_json_file(tongues_tts::voice_config_path(&model_path))?;
    let engine = if matches!(device, tongues_tts::ResolvedSpeechDevice::Cpu) {
        tongues_tts::OnnxSpeechBackend::load_cpu(&model_path, config)?
    } else {
        tongues_tts::OnnxSpeechBackend::load(&model_path, config)?
    };
    Ok(Box::new(tongues_tts::PlanEngineBackend::new(
        capabilities,
        device,
        engine,
    )))
}

fn load_styletts2_provider(
    home: &FsPath,
    _device: tongues_tts::ResolvedSpeechDevice,
    payload: &SpeakRequest,
    capabilities: tongues_tts::BackendCapabilities,
) -> anyhow::Result<ResidentSpeechBackend> {
    let model_dir = styletts2_model_dir(home, payload.model.as_deref())?;
    Ok(Box::new(styletts2::StyleTts2Synthesizer::new(
        capabilities,
        styletts2::StyleTts2OnnxBackend::from_model_dir(model_dir)?,
    )))
}

fn load_mock_provider(
    _home: &FsPath,
    _device: tongues_tts::ResolvedSpeechDevice,
    payload: &SpeakRequest,
    capabilities: tongues_tts::BackendCapabilities,
) -> anyhow::Result<ResidentSpeechBackend> {
    Ok(Box::new(MockSynthesizer {
        capabilities,
        backend: styletts2::MockStyleTts2Backend::new(payload.sample_rate_hz.unwrap_or(24_000)),
    }))
}

fn selected_onnx_voice_model_at(home: &FsPath) -> anyhow::Result<String> {
    let selection_path = home.join("model-selection.json");
    let selected = match std::fs::read_to_string(&selection_path) {
        Ok(content) => serde_json::from_str::<serde_json::Value>(&content)
            .with_context(|| format!("failed to parse {}", selection_path.display()))?
            .get("voice_model")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(DEFAULT_ONNX_VOICE_MODEL)
            .to_string(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            DEFAULT_ONNX_VOICE_MODEL.to_string()
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to read {}", selection_path.display()));
        }
    };
    Ok(selected)
}

fn onnx_voice_model_path(home: &FsPath, model_id: &str) -> anyhow::Result<PathBuf> {
    let model = ONNX_VOICE_MODELS
        .iter()
        .find(|model| model.id == model_id)
        .with_context(|| {
            format!("selected ONNX voice model `{model_id}` is not supported by tongues-server")
        })?;
    Ok(home.join(ONNX_VOICE_RELATIVE_DIR).join(model.filename))
}

fn registered_speech_compositions_at(
    home: &FsPath,
) -> Vec<tongues_tts::RegisteredSpeechComposition> {
    let mut compositions = base_registered_speech_compositions_at(home);
    if let Ok(catalog) = tongues_tts::ModelCatalog::with_private_catalogs(
        &tongues_tts::private_catalog_paths_from_environment(),
    ) {
        extend_registered_speech_compositions(&mut compositions, &catalog);
    }
    compositions
}

fn base_registered_speech_compositions_at(
    home: &FsPath,
) -> Vec<tongues_tts::RegisteredSpeechComposition> {
    let mut compositions = tongues_tts::registered_speech_compositions();
    let selected =
        selected_onnx_voice_model_at(home).unwrap_or_else(|_| DEFAULT_ONNX_VOICE_MODEL.into());
    compositions.extend(ONNX_VOICE_MODELS.iter().map(|voice| {
        let pipeline = tongues_tts::SpeechPipelineSelection::end_to_end(
            format!("projector/{}", voice.id),
            voice.id,
            Vec::new(),
        );
        tongues_tts::RegisteredSpeechComposition {
            id: pipeline
                .canonical_id()
                .expect("registered ONNX pipeline must be valid"),
            display_name: voice.display_name.into(),
            backend: "onnx".into(),
            model: voice.id.into(),
            pipeline,
            recommended: voice.id == selected,
            capability_tier: tongues_tts::CapabilityTier::TierB,
            revision_capable: false,
        }
    }));
    compositions
}

fn extend_registered_speech_compositions(
    compositions: &mut Vec<tongues_tts::RegisteredSpeechComposition>,
    catalog: &tongues_tts::ModelCatalog,
) {
    compositions.extend(
        catalog
            .entries
            .iter()
            .filter(|entry| entry.provenance.format == "fairseq-mms-vits")
            .map(fairseq_registered_composition),
    );
    compositions.extend(
        catalog
            .entries
            .iter()
            .filter(|entry| is_styletts2_catalog_entry(entry) && entry.id != "styletts2-en-us")
            .map(styletts2_registered_composition),
    );
}

fn fairseq_registered_composition(
    entry: &tongues_tts::ModelCatalogEntry,
) -> tongues_tts::RegisteredSpeechComposition {
    let pipeline = tongues_tts::SpeechPipelineSelection::end_to_end(
        format!("projector/{}", entry.id),
        entry.id.clone(),
        Vec::new(),
    );
    tongues_tts::RegisteredSpeechComposition {
        id: pipeline
            .canonical_id()
            .expect("Fairseq MMS end-to-end pipeline must be valid"),
        display_name: entry.display_name.clone(),
        backend: "fairseq".into(),
        model: entry.id.clone(),
        pipeline,
        recommended: entry.id == "fairseq-mms-vits-eng",
        capability_tier: tongues_tts::CapabilityTier::TierB,
        revision_capable: false,
    }
}

fn styletts2_registered_composition(
    entry: &tongues_tts::ModelCatalogEntry,
) -> tongues_tts::RegisteredSpeechComposition {
    let pipeline = tongues_tts::SpeechPipelineSelection::end_to_end(
        format!("projector/{}", entry.id),
        entry.id.clone(),
        vec!["style-reference-encoder".into()],
    );
    tongues_tts::RegisteredSpeechComposition {
        id: pipeline
            .canonical_id()
            .expect("StyleTTS2-family end-to-end pipeline must be valid"),
        display_name: entry.display_name.clone(),
        backend: "styletts2".into(),
        model: entry.id.clone(),
        pipeline,
        recommended: entry.id == "styletts2-en-us",
        capability_tier: tongues_tts::CapabilityTier::TierC,
        revision_capable: false,
    }
}

fn is_styletts2_catalog_entry(entry: &tongues_tts::ModelCatalogEntry) -> bool {
    entry.architecture.eq_ignore_ascii_case("styletts2")
        || entry
            .compatible_with
            .iter()
            .any(|value| value.eq_ignore_ascii_case("styletts2-onnx"))
}

fn styletts2_catalog_entry(model: &str) -> anyhow::Result<tongues_tts::ModelCatalogEntry> {
    let catalog = tongues_tts::ModelCatalog::with_private_catalogs(
        &tongues_tts::private_catalog_paths_from_environment(),
    )?;
    let entry = catalog
        .find(model)
        .with_context(|| format!("unknown catalog model `{model}`"))?;
    anyhow::ensure!(
        is_styletts2_catalog_entry(entry),
        "catalog model `{model}` is not a StyleTTS2-family checkpoint"
    );
    Ok(entry.clone())
}

fn styletts2_model_dir(home: &FsPath, model: Option<&str>) -> anyhow::Result<PathBuf> {
    let model = speech_model_id(home, "styletts2", model)?;
    let entry = styletts2_catalog_entry(&model)?;
    entry
        .artifacts
        .iter()
        .map(|artifact| home.join(&artifact.install_path))
        .find_map(|path| path.parent().map(|parent| parent.to_path_buf()))
        .with_context(|| format!("catalog model `{model}` has no installable model directory"))
}

fn resolve_registered_pipeline(
    home: &FsPath,
    pipeline: &tongues_tts::SpeechPipelineSelection,
) -> anyhow::Result<tongues_tts::RegisteredSpeechComposition> {
    let id = pipeline.canonical_id()?;
    registered_speech_compositions_at(home)
        .into_iter()
        .find(|composition| composition.id == id)
        .with_context(|| format!("speech pipeline `{id}` is not registered"))
}

fn resolve_legacy_composition(
    home: &FsPath,
    backend: &str,
    model: Option<&str>,
) -> anyhow::Result<tongues_tts::RegisteredSpeechComposition> {
    let model = speech_model_id(home, backend, model)?;
    if backend == "fairseq" {
        let catalog = tongues_tts::ModelCatalog::with_private_catalogs(
            &tongues_tts::private_catalog_paths_from_environment(),
        )?;
        let entry = catalog
            .find(&model)
            .context("resolved Fairseq MMS model disappeared from the catalog")?;
        return Ok(fairseq_registered_composition(entry));
    }
    if backend == "styletts2" {
        let entry = styletts2_catalog_entry(&model)?;
        return Ok(styletts2_registered_composition(&entry));
    }
    registered_speech_compositions_at(home)
        .into_iter()
        .find(|composition| composition.backend == backend && composition.model == model)
        .with_context(|| {
            format!(
                "no registered component pipeline exists for backend `{backend}` model `{model}`"
            )
        })
}

fn normalize_speak_request(mut payload: SpeakRequest) -> Result<SpeakRequest, String> {
    let home = resolve_mortar_home();
    let composition = if let Some(pipeline) = payload.pipeline.as_ref() {
        if payload
            .backend
            .as_deref()
            .is_some_and(|value| !value.is_empty())
            || payload
                .model
                .as_deref()
                .is_some_and(|value| !value.is_empty())
        {
            return Err(
                "pipeline cannot be combined with legacy backend or model selection".into(),
            );
        }
        resolve_registered_pipeline(&home, pipeline).map_err(|error| error.to_string())?
    } else {
        resolve_legacy_composition(
            &home,
            payload
                .backend
                .as_deref()
                .filter(|backend| !backend.is_empty())
                .unwrap_or("burn"),
            payload.model.as_deref(),
        )
        .map_err(|error| error.to_string())?
    };
    payload.backend = Some(composition.backend);
    payload.model = Some(composition.model);
    payload.pipeline = Some(composition.pipeline);
    Ok(payload)
}

fn speech_model_id(
    home: &FsPath,
    backend: &str,
    requested_model: Option<&str>,
) -> anyhow::Result<String> {
    let requested_model = requested_model.filter(|model| !model.trim().is_empty());
    let fixed_model = match backend {
        "burn" => Some("speedyspeech-ljspeech+hifigan-v2"),
        "fastpitch" => Some("fastpitch-ljspeech+hifigan-v2"),
        "glow" => Some("glow-tts-ljspeech+standardizer+multiband-melgan"),
        "vits" => Some("vits-vctk"),
        "yourtts" => Some("yourtts-multilingual"),
        "freevc" => Some("freevc24-vctk"),
        "mock" => Some("deterministic-mock"),
        "onnx" | "fairseq" | "styletts2" => None,
        _ => anyhow::bail!("unknown speech backend `{backend}`"),
    };
    if let Some(expected) = fixed_model {
        if requested_model.is_some_and(|model| model != expected) {
            anyhow::bail!(
                "model `{}` is not available for backend `{backend}`",
                requested_model.unwrap()
            );
        }
        return Ok(expected.into());
    }

    if backend == "fairseq" {
        let requested = requested_model.unwrap_or("fairseq-mms-vits-eng");
        let catalog = tongues_tts::ModelCatalog::with_private_catalogs(
            &tongues_tts::private_catalog_paths_from_environment(),
        )?;
        let entry = catalog
            .find(requested)
            .with_context(|| format!("unknown catalog model `{requested}`"))?;
        anyhow::ensure!(
            entry.provenance.format == "fairseq-mms-vits",
            "catalog model `{requested}` is not a Fairseq MMS VITS checkpoint"
        );
        return Ok(entry.id.clone());
    }

    if backend == "styletts2" {
        let requested = requested_model.unwrap_or("styletts2-en-us");
        let entry = styletts2_catalog_entry(requested)?;
        return Ok(entry.id.clone());
    }

    let model = requested_model
        .map(str::to_string)
        .map(Ok)
        .unwrap_or_else(|| selected_onnx_voice_model_at(home))?;
    if !ONNX_VOICE_MODELS
        .iter()
        .any(|candidate| candidate.id == model)
    {
        anyhow::bail!("model `{model}` is not available for backend `onnx`");
    }
    Ok(model)
}

fn speech_backend_capabilities(
    home: &FsPath,
    backend: &str,
    model: Option<&str>,
    device: tongues_tts::ResolvedSpeechDevice,
    mock_sample_rate_hz: u32,
) -> anyhow::Result<tongues_tts::BackendCapabilities> {
    let general_american = || speech_variety_capabilities(&["en-US-GA"]);
    let received_pronunciation = || speech_variety_capabilities(&["en-GB-RP"]);
    let output = |sample_rate_hz| tongues_tts::OutputAudioContract {
        sample_rate_hz,
        channels: 1,
        streaming: true,
    };
    let devices = vec![
        tongues_tts::SpeechDeviceRequest::Cpu,
        tongues_tts::SpeechDeviceRequest::Cuda {
            index: device.index().unwrap_or(0),
        },
    ];
    let unsupported_speakers = || tongues_tts::SpeakerCapabilities {
        values: tongues_tts::CapabilityValue::Unsupported,
        required: false,
        numeric_ids: false,
    };
    let unsupported_styles = tongues_tts::StyleCapabilities::unsupported;
    Ok(match backend {
        "burn" => tongues_tts::BackendCapabilities {
            backend: "burn".into(),
            model: "speedyspeech-ljspeech+hifigan-v2".into(),
            family: tongues_tts::SpeechModelFamily::AcousticModel,
            varieties: general_american(),
            languages: tongues_tts::LanguageCapabilities::unsupported(),
            speakers: unsupported_speakers(),
            styles: unsupported_styles(),
            reference_audio: Default::default(),
            speed: true,
            pitch: Default::default(),
            energy: Default::default(),
            durations: false,
            seed: true,
            devices,
            output: output(22_050),
            provenance: vec!["Published Coqui release artifacts".into()],
            capability_tier: tongues_tts::CapabilityTier::TierA,
            revision_capable: true,
        },
        "fastpitch" => tongues_tts::BackendCapabilities {
            backend: "fastpitch".into(),
            model: "fastpitch-ljspeech+hifigan-v2".into(),
            family: tongues_tts::SpeechModelFamily::AcousticModel,
            varieties: general_american(),
            languages: tongues_tts::LanguageCapabilities::unsupported(),
            speakers: unsupported_speakers(),
            styles: unsupported_styles(),
            reference_audio: Default::default(),
            speed: true,
            pitch: tongues_tts::PitchCapabilities {
                scale: true,
                shift: true,
                explicit_values: true,
            },
            energy: Default::default(),
            durations: true,
            seed: false,
            devices,
            output: output(22_050),
            provenance: vec!["Published Coqui release artifacts".into()],
            capability_tier: tongues_tts::CapabilityTier::TierA,
            revision_capable: true,
        },
        "glow" => tongues_tts::BackendCapabilities {
            backend: "glow".into(),
            model: "glow-tts-ljspeech+standardizer+multiband-melgan".into(),
            family: tongues_tts::SpeechModelFamily::AcousticModel,
            varieties: general_american(),
            languages: tongues_tts::LanguageCapabilities::unsupported(),
            speakers: unsupported_speakers(),
            styles: unsupported_styles(),
            reference_audio: Default::default(),
            speed: true,
            pitch: Default::default(),
            energy: Default::default(),
            durations: true,
            seed: true,
            devices,
            output: output(22_050),
            provenance: vec![
                "Published Coqui Glow-TTS and MultiBand-MelGAN release artifacts".into(),
                tongues_tts::GLOW_MULTIBAND_STANDARDIZER_ID.into(),
            ],
            capability_tier: tongues_tts::CapabilityTier::TierA,
            revision_capable: true,
        },
        "vits" => {
            let catalog = tongues_tts::SpeakerCatalog::from_file(
                home.join(VITS_SPEAKER_RELATIVE_PATH),
                VITS_SPEAKER_COUNT,
            )
            .ok();
            let speaker_values = catalog
                .as_ref()
                .map(|catalog| {
                    catalog
                        .entries()
                        .into_iter()
                        .map(|(name, id)| {
                            tongues_tts::NamedCapability::new(name, name.trim()).with_numeric_id(id)
                        })
                        .collect()
                })
                .unwrap_or_default();
            tongues_tts::BackendCapabilities {
                backend: "vits".into(),
                model: "vits-vctk".into(),
                family: tongues_tts::SpeechModelFamily::EndToEndSpeech,
                varieties: received_pronunciation(),
                languages: tongues_tts::LanguageCapabilities::unsupported(),
                speakers: tongues_tts::SpeakerCapabilities {
                    values: if catalog.is_some() {
                        tongues_tts::CapabilityValue::Listed(speaker_values)
                    } else {
                        tongues_tts::CapabilityValue::Any
                    },
                    required: true,
                    numeric_ids: true,
                },
                styles: unsupported_styles(),
                reference_audio: Default::default(),
                speed: true,
                pitch: Default::default(),
                energy: Default::default(),
                durations: false,
                seed: true,
                devices,
                output: output(22_050),
                provenance: vec!["Published Coqui release artifact".into()],
                capability_tier: tongues_tts::CapabilityTier::TierB,
                revision_capable: false,
            }
        }
        "fairseq" => {
            let model = speech_model_id(home, backend, model)?;
            let catalog = tongues_tts::ModelCatalog::with_private_catalogs(
                &tongues_tts::private_catalog_paths_from_environment(),
            )?;
            let entry = catalog
                .find(&model)
                .context("resolved Fairseq MMS model disappeared from the catalog")?;
            fairseq_backend_capabilities(entry, device)
        }
        "yourtts" => {
            let dir = home.join(YOURTTS_RELATIVE_DIR);
            let config = tongues_tts::VitsInferenceConfig::from_file(dir.join("config.json")).ok();
            let catalog = config.as_ref().and_then(|config| {
                tongues_tts::DVectorCatalog::from_file(
                    dir.join("speakers.json"),
                    config.network.d_vector_dim,
                    tongues_tts::COQUI_RESNET_SPEAKER_EMBEDDING_SPACE,
                )
                .ok()
            });
            let speaker_values = catalog
                .as_ref()
                .map(|catalog| {
                    catalog
                        .speaker_names()
                        .into_iter()
                        .map(|name| tongues_tts::NamedCapability::new(name, name))
                        .collect()
                })
                .unwrap_or_default();
            let languages = config
                .as_ref()
                .and_then(|config| {
                    tongues_tts::LanguageCatalog::from_file(
                        dir.join("language_ids.json"),
                        config.network.num_languages,
                    )
                    .ok()
                })
                .map_or_else(tongues_tts::LanguageCapabilities::unsupported, |catalog| {
                    tongues_tts::LanguageCapabilities {
                        values: tongues_tts::CapabilityValue::Listed(
                            catalog
                                .entries()
                                .into_iter()
                                .map(|(name, id)| {
                                    tongues_tts::NamedCapability::new(name, name)
                                        .with_numeric_id(id)
                                })
                                .collect(),
                        ),
                        required: true,
                        numeric_ids: true,
                    }
                });
            tongues_tts::BackendCapabilities {
                backend: "yourtts".into(),
                model: "yourtts-multilingual".into(),
                family: tongues_tts::SpeechModelFamily::CrossLingualVoiceClone,
                varieties: tongues_tts::CapabilityValue::Any,
                languages,
                speakers: tongues_tts::SpeakerCapabilities {
                    values: if catalog.is_some() {
                        tongues_tts::CapabilityValue::Listed(speaker_values)
                    } else {
                        tongues_tts::CapabilityValue::Any
                    },
                    required: true,
                    numeric_ids: false,
                },
                styles: unsupported_styles(),
                reference_audio: tongues_tts::ReferenceAudioCapabilities {
                    speaker: true,
                    style: false,
                    source: false,
                    ..Default::default()
                },
                speed: true,
                pitch: Default::default(),
                energy: Default::default(),
                durations: false,
                seed: true,
                devices,
                output: output(16_000),
                provenance: vec!["Published Coqui YourTTS release artifact".into()],
                capability_tier: tongues_tts::CapabilityTier::TierB,
                revision_capable: false,
            }
        }
        "onnx" => {
            let model = speech_model_id(home, backend, model)?;
            let model_path = onnx_voice_model_path(home, &model)?;
            let config = tongues_tts::VoiceConfig::from_json_file(tongues_tts::voice_config_path(
                &model_path,
            ))
            .ok();
            let speaker_values = config
                .as_ref()
                .map(|config| {
                    config
                        .speaker_id_map
                        .iter()
                        .map(|(name, id)| {
                            tongues_tts::NamedCapability::new(name, name).with_numeric_id(*id)
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            tongues_tts::BackendCapabilities {
                backend: "onnx".into(),
                model,
                family: tongues_tts::SpeechModelFamily::EndToEndSpeech,
                varieties: general_american(),
                languages: tongues_tts::LanguageCapabilities::unsupported(),
                speakers: tongues_tts::SpeakerCapabilities {
                    values: if speaker_values.is_empty() {
                        tongues_tts::CapabilityValue::Unsupported
                    } else {
                        tongues_tts::CapabilityValue::Listed(speaker_values)
                    },
                    required: config
                        .as_ref()
                        .is_some_and(|config| config.is_multi_speaker()),
                    numeric_ids: config
                        .as_ref()
                        .is_some_and(|config| config.speaker_count() > 1),
                },
                styles: unsupported_styles(),
                reference_audio: Default::default(),
                speed: true,
                pitch: Default::default(),
                energy: Default::default(),
                durations: false,
                seed: false,
                devices,
                output: output(
                    config
                        .as_ref()
                        .map(|config| config.sample_rate_hz)
                        .unwrap_or(22_050),
                ),
                provenance: vec!["Piper-compatible ONNX import".into()],
                capability_tier: tongues_tts::CapabilityTier::TierA,
                revision_capable: false,
            }
        }
        "freevc" => tongues_tts::BackendCapabilities {
            backend: "freevc".into(),
            model: "freevc24-vctk".into(),
            family: tongues_tts::SpeechModelFamily::VoiceConversion,
            varieties: tongues_tts::CapabilityValue::Any,
            languages: tongues_tts::LanguageCapabilities::unsupported(),
            speakers: unsupported_speakers(),
            styles: unsupported_styles(),
            reference_audio: tongues_tts::ReferenceAudioCapabilities {
                speaker: true,
                source: true,
                speaker_required: true,
                source_required: true,
                ..Default::default()
            },
            speed: false,
            pitch: Default::default(),
            energy: Default::default(),
            durations: false,
            seed: true,
            devices: vec![tongues_tts::SpeechDeviceRequest::Cpu],
            output: tongues_tts::OutputAudioContract {
                sample_rate_hz: 24_000,
                channels: 1,
                streaming: false,
            },
            provenance: vec![
                "OlaWod/FreeVC".into(),
                "microsoft/unilm WavLM".into(),
                "coqui-ai/TTS FreeVC24 artifact".into(),
            ],
            capability_tier: tongues_tts::CapabilityTier::Unassigned,
            revision_capable: false,
        },
        "styletts2" => {
            let model = speech_model_id(home, backend, model)?;
            let entry = styletts2_catalog_entry(&model)?;
            styletts2_backend_capabilities(&entry, device)
        }
        "mock" => tongues_tts::BackendCapabilities {
            backend: "mock".into(),
            model: "deterministic-mock".into(),
            family: tongues_tts::SpeechModelFamily::EndToEndSpeech,
            varieties: tongues_tts::CapabilityValue::Any,
            languages: tongues_tts::LanguageCapabilities::unsupported(),
            speakers: unsupported_speakers(),
            styles: unsupported_styles(),
            reference_audio: Default::default(),
            speed: false,
            pitch: Default::default(),
            energy: Default::default(),
            durations: false,
            seed: false,
            devices,
            output: output(mock_sample_rate_hz),
            provenance: vec!["Tongues deterministic test backend".into()],
            capability_tier: tongues_tts::CapabilityTier::Unassigned,
            revision_capable: false,
        },
        _ => anyhow::bail!("unknown speech backend `{backend}`"),
    })
}

fn styletts2_backend_capabilities(
    entry: &tongues_tts::ModelCatalogEntry,
    device: tongues_tts::ResolvedSpeechDevice,
) -> tongues_tts::BackendCapabilities {
    let ids = entry
        .varieties
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let varieties = if ids.is_empty() {
        speech_variety_capabilities(&["en-US-GA"])
    } else {
        speech_variety_capabilities(&ids)
    };
    let languages = tongues_tts::LanguageCapabilities {
        values: if entry.languages.is_empty() {
            tongues_tts::CapabilityValue::Unsupported
        } else {
            tongues_tts::CapabilityValue::Listed(
                entry
                    .languages
                    .iter()
                    .map(|tag| tongues_tts::NamedCapability::new(tag, tag))
                    .collect(),
            )
        },
        required: false,
        numeric_ids: false,
    };
    let speaker_values = entry
        .speakers
        .names
        .iter()
        .map(|name| tongues_tts::NamedCapability::new(name, name))
        .collect::<Vec<_>>();
    tongues_tts::BackendCapabilities {
        backend: "styletts2".into(),
        model: entry.id.clone(),
        family: tongues_tts::SpeechModelFamily::CrossLingualVoiceClone,
        varieties,
        languages,
        speakers: tongues_tts::SpeakerCapabilities {
            values: if speaker_values.is_empty() {
                tongues_tts::CapabilityValue::Unsupported
            } else {
                tongues_tts::CapabilityValue::Listed(speaker_values)
            },
            required: false,
            numeric_ids: false,
        },
        styles: tongues_tts::StyleCapabilities {
            names: tongues_tts::CapabilityValue::Any,
            reference_audio: true,
            embedding_dimensions: Some(STYLE_VECTOR_DIMS),
        },
        reference_audio: tongues_tts::ReferenceAudioCapabilities {
            speaker: true,
            style: true,
            source: false,
            ..Default::default()
        },
        speed: true,
        pitch: Default::default(),
        energy: Default::default(),
        durations: false,
        seed: true,
        devices: vec![
            tongues_tts::SpeechDeviceRequest::Cpu,
            tongues_tts::SpeechDeviceRequest::Cuda {
                index: device.index().unwrap_or(0),
            },
        ],
        output: tongues_tts::OutputAudioContract {
            sample_rate_hz: entry.sample_rate_hz.unwrap_or(24_000),
            channels: 1,
            streaming: true,
        },
        provenance: Vec::new(),
        capability_tier: tongues_tts::CapabilityTier::TierC,
        revision_capable: false,
    }
}

fn fairseq_backend_capabilities(
    entry: &tongues_tts::ModelCatalogEntry,
    device: tongues_tts::ResolvedSpeechDevice,
) -> tongues_tts::BackendCapabilities {
    tongues_tts::BackendCapabilities {
        backend: "fairseq".into(),
        model: entry.id.clone(),
        family: tongues_tts::SpeechModelFamily::EndToEndSpeech,
        // The selected checkpoint consumes raw graphemes. Catalog `varieties`
        // remains the authority for voice-variety claims; runtime planning may
        // use any Tongues text plan.
        varieties: tongues_tts::CapabilityValue::Any,
        languages: tongues_tts::LanguageCapabilities {
            values: tongues_tts::CapabilityValue::Listed(
                entry
                    .languages
                    .iter()
                    .map(|language| tongues_tts::NamedCapability::new(language, language))
                    .collect(),
            ),
            required: false,
            numeric_ids: false,
        },
        speakers: unsupported_speaker_capabilities(),
        styles: tongues_tts::StyleCapabilities::unsupported(),
        reference_audio: Default::default(),
        speed: true,
        pitch: Default::default(),
        energy: Default::default(),
        durations: false,
        seed: true,
        devices: vec![
            tongues_tts::SpeechDeviceRequest::Cpu,
            tongues_tts::SpeechDeviceRequest::Cuda {
                index: device.index().unwrap_or(0),
            },
        ],
        output: tongues_tts::OutputAudioContract {
            sample_rate_hz: entry.sample_rate_hz.unwrap_or(16_000),
            channels: 1,
            streaming: true,
        },
        provenance: vec![entry.provenance.source.clone()],
        // Fairseq MMS VITS pipelines are Tier B: committed-phrase, offline
        // grapheme preprocessing; whole-utterance rendering only.
        capability_tier: tongues_tts::CapabilityTier::TierB,
        revision_capable: false,
    }
}

fn unsupported_speaker_capabilities() -> tongues_tts::SpeakerCapabilities {
    tongues_tts::SpeakerCapabilities {
        values: tongues_tts::CapabilityValue::Unsupported,
        required: false,
        numeric_ids: false,
    }
}

fn speech_variety_capabilities(ids: &[&str]) -> tongues_tts::CapabilityValue {
    tongues_tts::CapabilityValue::Listed(
        ids.iter()
            .filter_map(|id| speaking::variety_by_code(id))
            .map(|variety| tongues_tts::NamedCapability::new(variety.id.0, variety.name))
            .collect(),
    )
}

fn load_resident_burn<B: burn::tensor::backend::Backend>(
    home: &FsPath,
    device: B::Device,
) -> anyhow::Result<tongues_tts::BurnSpeedySpeechPipeline<B>>
where
    B::Device: Clone,
{
    let acoustic_dir = home.join(SPEEDY_RELATIVE_DIR);
    let vocoder_dir = home.join(HIFIGAN_RELATIVE_DIR);
    let acoustic = tongues_tts::BurnSpeedySpeechAcoustic::load(
        acoustic_dir.join("config.json"),
        acoustic_dir.join("model_file.pth"),
        device.clone(),
    )?;
    let vocoder = tongues_tts::BurnHifiganVocoder::load(
        vocoder_dir.join("config.json"),
        vocoder_dir.join("model_file.pth"),
        device,
    )?;
    tongues_tts::BurnSpeedySpeechPipeline::new(acoustic, vocoder)
}

fn load_resident_fastpitch<B: burn::tensor::backend::Backend>(
    home: &FsPath,
    device: B::Device,
) -> anyhow::Result<tongues_tts::BurnFastPitchPipeline<B>>
where
    B::Device: Clone,
{
    let acoustic_dir = home.join(FASTPITCH_RELATIVE_DIR);
    let vocoder_dir = home.join(HIFIGAN_RELATIVE_DIR);
    let acoustic = tongues_tts::BurnFastPitchAcoustic::load(
        acoustic_dir.join("config.json"),
        acoustic_dir.join("model_file.pth"),
        device.clone(),
    )?;
    let vocoder = tongues_tts::BurnHifiganVocoder::load(
        vocoder_dir.join("config.json"),
        vocoder_dir.join("model_file.pth"),
        device,
    )?;
    tongues_tts::BurnFastPitchPipeline::new(acoustic, vocoder)
}

fn load_resident_glow<B: burn::tensor::backend::Backend>(
    home: &FsPath,
    device: B::Device,
) -> anyhow::Result<
    tongues_tts::BurnGlowTtsPipeline<
        B,
        tongues_tts::BurnStandardizingVocoder<B, tongues_tts::BurnMultibandMelganVocoder<B>>,
    >,
>
where
    B::Device: Clone,
{
    let acoustic_dir = home.join(GLOW_RELATIVE_DIR);
    let vocoder_dir = home.join(MULTIBAND_RELATIVE_DIR);
    let acoustic = tongues_tts::BurnGlowTtsAcoustic::load(
        acoustic_dir.join("config.json"),
        acoustic_dir.join("model_file.pth.tar"),
        device.clone(),
    )?;
    let tongues_tts::AcousticOutputContract::Spectrogram(source_contract) =
        tongues_tts::AcousticModel::output_contract(&acoustic)
    else {
        anyhow::bail!("Glow-TTS did not declare a spectrogram output");
    };
    let vocoder = tongues_tts::BurnMultibandMelganVocoder::load(
        vocoder_dir.join("config.json"),
        vocoder_dir.join("model_file.pth"),
        device.clone(),
    )?;
    let converter = tongues_tts::BurnStandardizingVocoder::new(
        vocoder,
        tongues_tts::FeatureStandardizationConfig::glow_multiband()?,
        source_contract,
        device,
    )?;
    tongues_tts::BurnGlowTtsPipeline::new(acoustic, converter)
}

fn load_resident_vits<B: burn::tensor::backend::Backend>(
    home: &FsPath,
    device: B::Device,
) -> anyhow::Result<tongues_tts::BurnVitsSpeech<B>> {
    let dir = home.join(VITS_RELATIVE_DIR);
    tongues_tts::BurnVitsSpeech::load(
        dir.join("config.json"),
        dir.join("model_file.pth"),
        dir.join("speaker_ids.json"),
        device,
    )
}

fn load_resident_yourtts<B: burn::tensor::backend::Backend>(
    home: &FsPath,
    device: B::Device,
) -> anyhow::Result<tongues_tts::BurnVitsSpeech<B>> {
    let dir = home.join(YOURTTS_RELATIVE_DIR);
    tongues_tts::BurnVitsSpeech::load_your_tts(
        dir.join("config.json"),
        dir.join("model_file.pth.tar"),
        dir.join("speakers.json"),
        dir.join("language_ids.json"),
        dir.join("config_se.json"),
        dir.join("model_se.pth.tar"),
        device,
        tongues_tts::SpeakerEmbeddingCachePolicy::Memory { max_entries: 16 },
    )
}

fn encode_wav_mono_f32(sample_rate_hz: u32, samples: &[f32]) -> anyhow::Result<Vec<u8>> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: sample_rate_hz,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for sample in samples {
            writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

#[derive(Deserialize, Clone)]
struct SpeakRequest {
    text: String,
    pipeline: Option<tongues_tts::SpeechPipelineSelection>,
    cpu: Option<bool>,
    cuda_device: Option<usize>,
    quiet: Option<bool>,
    verbose: Option<bool>,
    variety: Option<String>,
    backend: Option<String>,
    model: Option<String>,
    speaker: Option<String>,
    speaker_id: Option<u32>,
    model_language: Option<String>,
    language_id: Option<u32>,
    emotion: Option<String>,
    emotion_vector: Option<Vec<f32>>,
    emotion_strength: Option<f32>,
    source_audio: Option<String>,
    target_audio: Option<String>,
    voice_sample: Option<String>,
    style_sample: Option<String>,
    quality: Option<String>,
    diffusion_steps: Option<usize>,
    speaker_reference_strength: Option<f32>,
    style_reference_strength: Option<f32>,
    style_alpha: Option<f32>,
    style_beta: Option<f32>,
    embedding_scale: Option<f64>,
    style_seed: Option<u64>,
    speed: Option<f64>,
    noise_scale: Option<f32>,
    duration_noise_scale: Option<f32>,
    pitch_scale: Option<f32>,
    pitch_shift: Option<f32>,
    pitch: Option<Vec<f32>>,
    energy_scale: Option<f32>,
    energy_shift: Option<f32>,
    energy: Option<Vec<f32>>,
    durations: Option<Vec<u32>>,
    seed: Option<u64>,
    sample_rate_hz: Option<u32>,
    max_tts_symbols: Option<usize>,
    no_tts_chunking: Option<bool>,
    debug_pronunciation: Option<bool>,
    timings: Option<bool>,
    fail_on_guessed_pronunciation: Option<bool>,
}

async fn speak(
    State(state): State<AppState>,
    Json(payload): Json<SpeakRequest>,
) -> impl IntoResponse {
    let payload = match normalize_speak_request(payload) {
        Ok(payload) => payload,
        Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
    };
    if let Err(error) = validate_speak_request(&payload) {
        return (StatusCode::BAD_REQUEST, error).into_response();
    }

    let context = match resident_synthesis_context(&state, &payload) {
        Ok(context) => context,
        Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
    };
    let speech_device = match resident_speech_device_for(&payload, state.speech_device) {
        Ok(device) => device,
        Err(error) => return (StatusCode::BAD_REQUEST, error.to_string()).into_response(),
    };
    let capabilities = match speech_backend_capabilities(
        &resolve_mortar_home(),
        payload.backend.as_deref().unwrap_or("burn"),
        payload.model.as_deref(),
        speech_device,
        payload.sample_rate_hz.unwrap_or(24_000),
    ) {
        Ok(capabilities) => capabilities,
        Err(error) => return (StatusCode::BAD_REQUEST, error.to_string()).into_response(),
    };
    if let Err(error) = validate_declared_speech_controls(
        &payload,
        &speech_control_discovery(
            payload.backend.as_deref().unwrap_or("burn"),
            &capabilities,
            speech_device,
        ),
    ) {
        return (StatusCode::BAD_REQUEST, error).into_response();
    }
    if let Err(error) = capabilities.validate(&unified_synthesis_request(
        &payload,
        &context,
        speech_device,
    )) {
        return (StatusCode::BAD_REQUEST, error.to_string()).into_response();
    }
    let backend = payload.backend.as_deref().unwrap_or("burn");
    if let Some(error) =
        speech_backend_installation_error(&resolve_mortar_home(), backend, payload.model.as_deref())
    {
        return (
            StatusCode::CONFLICT,
            format!(
                "Selected synthesis path is unavailable: {error}. Install its catalog artifacts with `tongues models install` and refresh discovery."
            ),
        )
            .into_response();
    }
    if let Err(error) =
        verify_catalog_backend(&resolve_mortar_home(), backend, payload.model.as_deref())
    {
        return (
            StatusCode::CONFLICT,
            format!(
                "Selected synthesis path is not verified: {error:#}. Reinstall the catalog model before synthesis."
            ),
        )
            .into_response();
    }
    let permit = match state.speech_admission.try_acquire() {
        Ok(permit) => permit,
        Err(_) => {
            return Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .header("Retry-After", "1")
                .header("X-Tongues-Speech-Capacity", state.speech_admission.capacity)
                .body(axum::body::Body::from(format!(
                    "Speech runtime is at its bounded capacity of {} in-flight requests",
                    state.speech_admission.capacity
                )))
                .unwrap();
        }
    };

    let registry = Arc::clone(&state.speech);
    let phase = Arc::clone(&state.speech_phase);
    let request = payload.clone();
    let queued_at = std::time::Instant::now();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _phase_reset = SpeechPhaseReset(Arc::clone(&phase));
        let mut service = registry
            .lock()
            .map_err(|_| anyhow::anyhow!("resident speech registry lock is poisoned"))?;
        let queue_ms = queued_at.elapsed().as_secs_f64() * 1_000.0;
        let mut output = service.synthesize(&request, &context, &phase, speech_device)?;
        output.queue_ms = queue_ms;
        Ok::<_, anyhow::Error>(output)
    })
    .await;
    match result {
        Ok(Ok(output)) => {
            if payload.timings.unwrap_or(false) {
                eprintln!(
                    "resident_speech_profile_json: {}",
                    json!({
                        "engine": output.engine_key,
                        "device": output.device.kind(),
                        "device_index": output.device.index(),
                        "loaded_now": output.loaded_now,
                        "queue_ms": output.queue_ms,
                        "load_ms": output.load_ms,
                        "synthesis_ms": output.synthesis_ms,
                        "audio_seconds": output.audio_seconds,
                        "real_time_factor": output.real_time_factor,
                        "stages": output.profile,
                    })
                );
            }
            let pipeline = payload
                .pipeline
                .as_ref()
                .expect("normalized speech request has a component pipeline");
            let pipeline_id = pipeline
                .canonical_id()
                .expect("normalized speech pipeline remains valid");
            let metadata = json!({
                "backend": payload.backend.as_deref().unwrap_or("burn"),
                "path": payload.model.as_deref().unwrap_or(&output.engine_key),
                "pipeline_id": pipeline_id,
                "pipeline": pipeline,
                "engine": output.engine_key,
                "projector": pipeline.projector,
                "acoustic_model": pipeline.acoustic_model,
                "conditioners": pipeline.conditioners,
                "vocoder": pipeline.vocoder,
                "voice_model": pipeline.end_to_end,
                "speaker": payload.speaker,
                "reference_voice": payload.voice_sample,
                "variety": payload.variety.as_deref().unwrap_or("en-US"),
                "device": output.device.kind(),
                "device_index": output.device.index(),
                "sample_rate_hz": output.sample_rate_hz,
                "channels": output.channels,
                "sample_count": output.sample_count,
                "duration_seconds": output.audio_seconds,
                "queue_ms": output.queue_ms,
                "model_load_ms": output.load_ms,
                "synthesis_ms": output.synthesis_ms,
                "real_time_factor": output.real_time_factor,
                "resident_model_reused": !output.loaded_now,
                "diagnostics": {
                    "stages": output.profile,
                    "pronunciation_warnings": output.pronunciation_warnings,
                    "pronunciation_plan": output.pronunciation_plan,
                },
                "input_audio": output.input_audio,
            });
            let metadata_header = serde_json::to_string(&metadata).unwrap_or_else(|_| "{}".into());
            Response::builder()
                .header("Content-Type", "audio/wav")
                .header(
                    "Content-Disposition",
                    format!(
                        "inline; filename=\"tongues-{}.wav\"",
                        payload.backend.as_deref().unwrap_or("speech")
                    ),
                )
                .header("X-Tongues-Speech-Metadata", metadata_header)
                .header("X-Tongues-Speech-Engine", output.engine_key)
                .header("X-Tongues-Speech-Device", output.device.kind())
                .header(
                    "X-Tongues-Speech-Device-Index",
                    output
                        .device
                        .index()
                        .map_or_else(String::new, |index| index.to_string()),
                )
                .header(
                    "X-Tongues-Model-Loaded",
                    if output.loaded_now { "cold" } else { "reused" },
                )
                .header(
                    "Server-Timing",
                    format!(
                        "model-load;dur={:.3}, synthesis;dur={:.3}",
                        output.load_ms, output.synthesis_ms
                    ),
                )
                .header(
                    "X-Tongues-Real-Time-Factor",
                    format!("{:.6}", output.real_time_factor),
                )
                .body(axum::body::Body::from(output.wav))
                .unwrap()
        }
        Ok(Err(error)) => {
            let message = format!("{error:#}");
            if message.starts_with("guessed_pronunciation:") {
                (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("Synthesis rejected: {message}"),
                )
                    .into_response()
            } else {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Resident synthesis failed: {message}"),
                )
                    .into_response()
            }
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Resident synthesis task failed: {error}"),
        )
            .into_response(),
    }
}

fn resident_synthesis_context(
    state: &AppState,
    payload: &SpeakRequest,
) -> Result<ResidentSynthesisContext, String> {
    if payload.backend.as_deref() == Some("freevc") {
        let source = payload
            .source_audio
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "source_audio is required for FreeVC".to_string())?;
        let target = payload
            .target_audio
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "target_audio is required for FreeVC".to_string())?;
        return Ok(ResidentSynthesisContext {
            voice_reference: Some(workspace_reference_wav(state, target)?),
            style_reference: None,
            source_reference: Some(workspace_reference_wav(state, source)?),
            emotion_vector: None,
        });
    }
    if payload.backend.as_deref() != Some("styletts2") {
        return Ok(ResidentSynthesisContext {
            voice_reference: None,
            style_reference: None,
            source_reference: None,
            emotion_vector: None,
        });
    }
    let voice_reference = styletts2_sample_path(
        state,
        payload
            .voice_sample
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or(DEFAULT_STYLETTS2_VOICE_REFERENCE),
    )?;
    let style_reference = styletts2_sample_path(
        state,
        payload
            .style_sample
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or(DEFAULT_STYLETTS2_STYLE_REFERENCE),
    )?;
    let emotion_vector = request_emotion_vector(state, payload)?;
    Ok(ResidentSynthesisContext {
        voice_reference: Some(voice_reference),
        style_reference: Some(style_reference),
        source_reference: None,
        emotion_vector,
    })
}

fn workspace_reference_wav(state: &AppState, input: &str) -> Result<PathBuf, String> {
    let relative = safe_relative_path(input)?;
    validate_artifact_relative_path(&state.workspace_root, &relative)?;
    let resolved = resolve_existing_artifact_path(&state.workspace_root, &relative)?;
    if !resolved.is_file()
        || resolved
            .extension()
            .and_then(|extension| extension.to_str())
            .is_none_or(|extension| !extension.eq_ignore_ascii_case("wav"))
    {
        return Err(format!(
            "reference audio must be a WAV file: {}",
            resolved.display()
        ));
    }
    Ok(resolved)
}

fn request_emotion_vector(
    state: &AppState,
    payload: &SpeakRequest,
) -> Result<Option<Vec<f32>>, String> {
    if let Some(vector) = payload.emotion_vector.as_ref() {
        return Ok(Some(vector.clone()));
    }
    let Some(emotion) = payload.emotion.as_deref().filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let path = emotion_signatures_path(state);
    if !path.is_file() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(&path)
        .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&content)
        .map_err(|error| format!("Failed to parse {}: {error}", path.display()))?;
    let vector = value
        .get(emotion)
        .and_then(|signature| signature.get("vector"))
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("Emotion `{emotion}` was not found in {}", path.display()))?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .ok_or_else(|| format!("Emotion `{emotion}` contains a non-numeric vector value"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_emotion_vector(emotion, &vector)?;
    Ok(Some(vector))
}

async fn get_styletts2_samples(State(state): State<AppState>) -> impl IntoResponse {
    let response = match load_styletts2_samples(&state) {
        Ok(samples) => StyleTts2SamplesResponse {
            reference_dir: Some(styletts2_reference_dir(&state).display().to_string()),
            samples,
            defaults: StyleTts2SampleDefaults {
                voice: "1221-135767-0014.wav".into(),
                style: "amused.wav".into(),
            },
            error: None,
        },
        Err(error) => StyleTts2SamplesResponse {
            reference_dir: Some(styletts2_reference_dir(&state).display().to_string()),
            samples: Vec::new(),
            defaults: StyleTts2SampleDefaults {
                voice: "1221-135767-0014.wav".into(),
                style: "amused.wav".into(),
            },
            error: Some(error),
        },
    };
    Json(response)
}

async fn get_speech_models(
    State(state): State<AppState>,
    Query(query): Query<SpeechModelsQuery>,
) -> impl IntoResponse {
    let limit = query.limit.clamp(1, MAX_SPEECH_DISCOVERY_PAGE_LIMIT);
    let cursor = query.cursor;
    Json(build_speech_discovery(&state, cursor, limit, query.into_filters()).await)
}

async fn build_speech_discovery(
    state: &AppState,
    cursor: usize,
    limit: usize,
    filters: SpeechDiscoveryFilters,
) -> SpeechStudioDiscovery {
    let home = resolve_mortar_home();
    let loaded = state
        .speech
        .try_lock()
        .ok()
        .map(|service| service.engines.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    let device = state.speech_device;
    tokio::task::spawn_blocking(move || {
        speech_studio_discovery_page(&home, device, &loaded, cursor, limit, &filters)
    })
    .await
    .unwrap_or_else(|error| SpeechStudioDiscovery {
        schema_version: 4,
        page: SpeechDiscoveryPage {
            cursor,
            limit,
            returned: 0,
            total: 0,
            next_cursor: None,
        },
        paths: Vec::new(),
        components: Vec::new(),
        compositions: Vec::new(),
        compatibility: Vec::new(),
        presets: Vec::new(),
        verification_ids: Vec::new(),
        error: Some(format!("speech model discovery task failed: {error}")),
    })
}

async fn verify_speech_model(
    State(_state): State<AppState>,
    Path(model_id): Path<String>,
) -> Response {
    let home = resolve_mortar_home();
    let requested_id = model_id.clone();
    let verification = tokio::task::spawn_blocking(move || {
        let catalog_paths = tongues_tts::private_catalog_paths_from_environment();
        let catalog = tongues_tts::ModelCatalog::with_private_catalogs(&catalog_paths)
            .map_err(|error| format!("speech model catalog discovery failed: {error:#}"))?;
        let Some(entry) = catalog.find(&requested_id) else {
            return Ok(None);
        };
        let cache = tongues_tts::default_model_cache(&home)
            .unwrap_or_else(|_| home.join("cache/model-downloads"));
        let store = tongues_tts::ModelStore::new(home, cache).with_offline(true);
        let result = store.verify(entry).map_err(|error| format!("{error:#}"));
        Ok::<_, String>(Some((entry.id.clone(), result)))
    })
    .await;
    let (_model_id, result) = match verification {
        Ok(Ok(Some(verification))) => verification,
        Ok(Ok(None)) => {
            return (
                StatusCode::NOT_FOUND,
                format!("unknown catalog speech model `{model_id}`"),
            )
                .into_response();
        }
        Ok(Err(error)) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, error).into_response();
        }
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("speech model verification task failed: {error}"),
            )
                .into_response();
        }
    };
    if let Err(error) = result {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({
                "error": error,
                "model_id": model_id,
            })),
        )
            .into_response();
    }
    Json(json!({
        "model_id": model_id,
        "verified": true,
    }))
    .into_response()
}

async fn project_speech_request(Json(request): Json<SpeechProjectionRequest>) -> impl IntoResponse {
    if request.text.trim().is_empty() || request.variety.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "text and variety are required").into_response();
    }
    if request.pipeline.is_some() && request.backend.is_some() {
        return (
            StatusCode::BAD_REQUEST,
            "pipeline cannot be combined with legacy backend selection",
        )
            .into_response();
    }
    let home = resolve_mortar_home();
    let backend = if let Some(pipeline) = request.pipeline.as_ref() {
        match resolve_registered_pipeline(&home, pipeline) {
            Ok(composition) => composition.backend,
            Err(error) => return (StatusCode::BAD_REQUEST, error.to_string()).into_response(),
        }
    } else {
        request.backend.clone().unwrap_or_else(|| "burn".into())
    };
    let relative_dir = match backend.as_str() {
        "burn" => SPEEDY_RELATIVE_DIR,
        "fastpitch" => FASTPITCH_RELATIVE_DIR,
        "glow" => GLOW_RELATIVE_DIR,
        "vits" => VITS_RELATIVE_DIR,
        "yourtts" => YOURTTS_RELATIVE_DIR,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                "the selected path does not expose token-level conditioning",
            )
                .into_response();
        }
    };
    let config_path = home.join(relative_dir).join("config.json");
    let response = (|| -> anyhow::Result<SpeechProjectionResponse> {
        let source = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let projector: Box<
            dyn tongues_tts::LinguisticProjector<ModelInput = tongues_tts::PhonemeTokenIds>,
        > = if matches!(backend.as_str(), "vits" | "yourtts") {
            Box::new(
                tongues_tts::VitsLinguisticProjector::from_json5_str(&source).with_context(
                    || {
                        format!(
                            "failed to build VITS projector from {}",
                            config_path.display()
                        )
                    },
                )?,
            )
        } else {
            Box::new(
                tongues_tts::PhonemeVocabularyProjector::from_json5_str(&source).with_context(
                    || format!("failed to build projector from {}", config_path.display()),
                )?,
            )
        };
        let plan = tongues_tts::utterance_plan_from_text(tongues_tts::SpeechRequest {
            text: request.text,
            variety: request.variety,
        })?;
        let phoneme_count = plan.intended_phonemes.len();
        let phone_count = plan.target_phones.len();
        let projected = projector.project(&plan)?;
        Ok(SpeechProjectionResponse {
            projected_token_count: projected.ids.len(),
            backend_symbols: projected.projected_symbols,
            phoneme_count,
            phone_count,
        })
    })();
    match response {
        Ok(response) => Json(response).into_response(),
        Err(error) => (
            StatusCode::CONFLICT,
            format!(
                "Unable to project tokens for expert controls: {error:#}. Install and verify the selected acoustic model first."
            ),
        )
            .into_response(),
    }
}

async fn project_duplex_request(
    State(state): State<AppState>,
    Json(request): Json<DuplexProjectionRequest>,
) -> impl IntoResponse {
    match build_duplex_projection(&state.workspace_root, request) {
        Ok(response) => Json(response).into_response(),
        Err(error) => (StatusCode::BAD_REQUEST, error).into_response(),
    }
}

fn build_duplex_projection(
    workspace_root: &FsPath,
    request: DuplexProjectionRequest,
) -> Result<DuplexStudioProjection, String> {
    let (run_id, journal) = duplex_run_and_journal(workspace_root, &request)?;
    studio_projection_from_journal(run_id, &journal).map_err(|error| error.to_string())
}

fn duplex_run_and_journal(
    workspace_root: &FsPath,
    request: &DuplexProjectionRequest,
) -> Result<(String, SimulatorJournal), String> {
    if let Some(journal_path) = request.journal_path.as_deref().map(str::trim) {
        if journal_path.is_empty() {
            return Err("journal_path cannot be empty".into());
        }
        let relative = safe_relative_path(journal_path)?;
        validate_artifact_relative_path(workspace_root, &relative)?;
        let full_path = resolve_existing_artifact_path(workspace_root, &relative)?;
        let bytes = std::fs::read(&full_path)
            .map_err(|error| format!("failed to read journal {}: {error}", full_path.display()))?;
        let journal: SimulatorJournal = serde_json::from_slice(&bytes)
            .map_err(|error| format!("failed to parse journal {}: {error}", full_path.display()))?;
        let run_id = relative
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("duplex-replay")
            .to_string();
        return Ok((run_id, journal));
    }

    if !request.chunks.is_empty() || !request.mock_acoustics.is_empty() {
        let mut config = SimulatorConfig::default();
        if let Some(posterior_mass) = request.posterior_mass {
            config.posterior_mass = posterior_mass;
        }
        let run_id = "oracle-chunks".to_string();
        let mut simulator = DuplexSimulator::new(
            speaking::UtteranceId(run_id.clone()),
            speaking::VarietyId(request.variety.clone().unwrap_or_else(|| "en-US-GA".into())),
            config,
            OracleCompletionProvider,
        )
        .map_err(|error| error.to_string())?;
        for (index, chunk) in request.chunks.iter().enumerate() {
            simulator
                .observe(ObservedEvidence::text(format!("text:{index}"), chunk))
                .map_err(|error| error.to_string())?;
        }
        for (index, transcript) in request.mock_acoustics.iter().enumerate() {
            simulator
                .observe(ObservedEvidence::acoustics(
                    format!("acoustics:{index}"),
                    transcript,
                ))
                .map_err(|error| error.to_string())?;
        }
        let (journal, _) = simulator.into_parts();
        return Ok((run_id, journal));
    }

    let suite = load_duplex_fixture_suite(workspace_root)?;
    let fixture_id = request.fixture.as_deref().unwrap_or("who-shot-john-f");
    let fixture = suite
        .fixture(fixture_id)
        .ok_or_else(|| format!("unknown duplex fixture '{fixture_id}'"))?;
    let mut config = fixture.config.clone();
    if let Some(posterior_mass) = request.posterior_mass {
        config.posterior_mass = posterior_mass;
    }
    let provider = FixtureCompletionProvider::new(fixture);
    let mut simulator = DuplexSimulator::new(
        fixture.utterance_id.clone(),
        fixture.variety.clone(),
        config,
        provider,
    )
    .map_err(|error| error.to_string())?;
    for step in &fixture.steps {
        simulator
            .observe(step.evidence.clone())
            .map_err(|error| error.to_string())?;
    }
    let (journal, _) = simulator.into_parts();
    Ok((fixture.id.clone(), journal))
}

fn load_duplex_fixture_suite(workspace_root: &FsPath) -> Result<DuplexFixtureSuite, String> {
    let path = workspace_root.join(DEFAULT_DUPLEX_FIXTURES_PATH);
    let bytes = std::fs::read(&path)
        .map_err(|error| format!("failed to read duplex fixtures {}: {error}", path.display()))?;
    let suite: DuplexFixtureSuite = serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "failed to parse duplex fixtures {}: {error}",
            path.display()
        )
    })?;
    suite.validate().map_err(|error| {
        format!(
            "failed to validate duplex fixtures {}: {error}",
            path.display()
        )
    })?;
    Ok(suite)
}

async fn get_model_catalog() -> impl IntoResponse {
    let response = tokio::task::spawn_blocking(model_catalog_response)
        .await
        .unwrap_or_else(|error| {
            json!({
                "schema_version": tongues_tts::MODEL_CATALOG_SCHEMA_VERSION,
                "entries": [],
                "error": format!("catalog verification task failed: {error}"),
            })
        });
    Json(response)
}

fn model_catalog_response() -> serde_json::Value {
    let paths = tongues_tts::private_catalog_paths_from_environment();
    let catalog = match tongues_tts::ModelCatalog::with_private_catalogs(&paths) {
        Ok(catalog) => catalog,
        Err(error) => {
            return json!({
                "schema_version": tongues_tts::MODEL_CATALOG_SCHEMA_VERSION,
                "entries": [],
                "error": format!("{error:#}"),
            });
        }
    };
    let home = resolve_mortar_home();
    let cache = tongues_tts::default_model_cache(&home)
        .unwrap_or_else(|_| home.join("cache/model-downloads"));
    let store = tongues_tts::ModelStore::new(&home, &cache).with_offline(true);
    let entries = catalog
        .entries
        .iter()
        .map(|entry| {
            let verification = store.verification_state(entry);
            json!({
                "entry": entry,
                "installed": verification.status
                    != tongues_tts::ModelVerificationStatus::Unavailable,
                "verified_files": verification.verified_files,
                "verification_status": verification.status,
                "error": verification.error,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "schema_version": catalog.schema_version,
        "catalog": catalog.id,
        "model_home": store.root(),
        "cache": store.cache(),
        "offline": store.offline(),
        "entries": entries,
        "error": null,
    })
}

fn aggregate_verification_status(
    statuses: impl IntoIterator<Item = tongues_tts::ModelVerificationStatus>,
) -> tongues_tts::ModelVerificationStatus {
    let statuses = statuses.into_iter().collect::<Vec<_>>();
    for status in [
        tongues_tts::ModelVerificationStatus::Unavailable,
        tongues_tts::ModelVerificationStatus::VerificationFailed,
        tongues_tts::ModelVerificationStatus::ChangedSinceVerification,
        tongues_tts::ModelVerificationStatus::PendingVerification,
    ] {
        if statuses.contains(&status) {
            return status;
        }
    }
    tongues_tts::ModelVerificationStatus::Verified
}

fn verification_status_label(status: tongues_tts::ModelVerificationStatus) -> &'static str {
    match status {
        tongues_tts::ModelVerificationStatus::Verified => "Verified",
        tongues_tts::ModelVerificationStatus::PendingVerification => "Pending verification",
        tongues_tts::ModelVerificationStatus::ChangedSinceVerification => {
            "Changed since verification"
        }
        tongues_tts::ModelVerificationStatus::VerificationFailed => "Verification failed",
        tongues_tts::ModelVerificationStatus::Unavailable => "Unavailable",
    }
}

fn catalog_entry_installation_error(
    home: &FsPath,
    entry: &tongues_tts::ModelCatalogEntry,
) -> Option<String> {
    let missing = entry
        .artifacts
        .iter()
        .flat_map(|artifact| {
            if artifact.members.is_empty() {
                vec![home.join(&artifact.install_path)]
            } else {
                artifact
                    .members
                    .iter()
                    .map(|member| home.join(&member.install_path))
                    .collect()
            }
        })
        .filter(|path| !path.is_file() && !path.is_dir())
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    (!missing.is_empty()).then(|| format!("missing model artifacts: {}", missing.join(", ")))
}

fn discover_speech_path(
    home: &FsPath,
    backend: &str,
    model: &str,
    display_name: &str,
    selected: bool,
    device: tongues_tts::ResolvedSpeechDevice,
    catalog: &tongues_tts::ModelCatalog,
    verification: &SpeechVerification,
    loaded: &[String],
) -> SpeechPathDiscovery {
    let capabilities = if backend == "fairseq" {
        catalog
            .find(model)
            .filter(|entry| entry.provenance.format == "fairseq-mms-vits")
            .map(|entry| fairseq_backend_capabilities(entry, device))
            .with_context(|| format!("unknown Fairseq MMS catalog model `{model}`"))
    } else if backend == "styletts2" {
        catalog
            .find(model)
            .filter(|entry| is_styletts2_catalog_entry(entry))
            .map(|entry| styletts2_backend_capabilities(entry, device))
            .with_context(|| format!("unknown StyleTTS2 catalog model `{model}`"))
    } else {
        speech_backend_capabilities(home, backend, Some(model), device, 24_000)
    }
    .unwrap_or_else(|error| tongues_tts::BackendCapabilities {
        backend: backend.into(),
        model: model.into(),
        family: tongues_tts::SpeechModelFamily::Unknown(error.to_string()),
        varieties: tongues_tts::CapabilityValue::Unsupported,
        languages: tongues_tts::LanguageCapabilities::unsupported(),
        speakers: tongues_tts::SpeakerCapabilities::unsupported(),
        styles: tongues_tts::StyleCapabilities::unsupported(),
        reference_audio: Default::default(),
        speed: false,
        pitch: Default::default(),
        energy: Default::default(),
        durations: false,
        seed: false,
        devices: Vec::new(),
        output: tongues_tts::OutputAudioContract {
            sample_rate_hz: 0,
            channels: 1,
            streaming: false,
        },
        provenance: Vec::new(),
        capability_tier: tongues_tts::CapabilityTier::Unassigned,
        revision_capable: false,
    });
    let catalog_ids = if matches!(backend, "fairseq" | "styletts2") {
        vec![model.to_string()]
    } else {
        speech_path_catalog_ids(home, backend, Some(model)).unwrap_or_default()
    };
    let catalog_entries = catalog_ids
        .iter()
        .filter_map(|id| catalog.find(id).cloned())
        .collect::<Vec<_>>();
    let missing_catalog_ids = catalog_entries
        .iter()
        .filter(|entry| catalog_entry_installation_error(home, entry).is_some())
        .map(|entry| entry.id.clone())
        .collect::<Vec<_>>();
    let installation_error = if matches!(backend, "fairseq" | "styletts2") {
        match catalog_entries.first() {
            Some(entry) => catalog_entry_installation_error(home, entry),
            None => Some(format!("unknown {backend} catalog model `{model}`")),
        }
    } else {
        speech_backend_installation_error(home, backend, Some(model))
    };
    let installed = installation_error.is_none();
    let verification_status = if backend == "mock" {
        tongues_tts::ModelVerificationStatus::Verified
    } else if catalog_entries.len() != catalog_ids.len() {
        tongues_tts::ModelVerificationStatus::Unavailable
    } else {
        aggregate_verification_status(
            catalog_entries
                .iter()
                .filter_map(|entry| verification.get(&entry.id).map(|state| state.status)),
        )
    };
    let verification_pending =
        verification_status == tongues_tts::ModelVerificationStatus::PendingVerification;
    let verification_error = catalog_entries
        .iter()
        .filter_map(|entry| verification.get(&entry.id))
        .find_map(|state| state.error.clone());
    let verified = verification_status == tongues_tts::ModelVerificationStatus::Verified;
    let runnable = installed && verified;
    let unavailable_reason = if !installed {
        installation_error
    } else if verified {
        None
    } else if let Some(error) = verification_error {
        Some(error)
    } else {
        Some(verification_status_label(verification_status).into())
    };
    let (acoustic_model, vocoder, voice_model, component_ids) =
        speech_path_components(backend, model);
    let cli_vocoder = match backend {
        "burn" | "fastpitch" => Some("hifigan".into()),
        "glow" => Some("multiband-melgan".into()),
        _ => None,
    };
    let load_state = if speech_path_is_loaded(backend, model, loaded) {
        "loaded"
    } else {
        "unloaded"
    };
    let mut statuses = Vec::new();
    if runnable {
        statuses.push("Ready".into());
    } else if !installed {
        statuses.push("Artifact missing".into());
    } else {
        statuses.push("Installed".into());
    }
    if verified {
        statuses.push("Verified".into());
    } else {
        statuses.push(verification_status_label(verification_status).into());
    }
    if load_state == "loaded" {
        statuses.push("Loaded".into());
    }
    if backend == "mock" {
        statuses.push("Test backend".into());
    }
    let install_command = (!missing_catalog_ids.is_empty()).then(|| {
        missing_catalog_ids
            .iter()
            .map(|id| format!("cargo run --bin tongues -- models install {id}"))
            .collect::<Vec<_>>()
            .join(" && ")
    });
    let controls = speech_control_discovery(backend, &capabilities, device);
    SpeechPathDiscovery {
        capabilities,
        id: model.into(),
        display_name: display_name.into(),
        kind: "synthesis_path",
        complete: true,
        runnable,
        selected,
        installed,
        verified,
        verification_pending,
        verification_status,
        load_state,
        acoustic_model,
        vocoder,
        cli_vocoder,
        voice_model,
        component_ids,
        compatible_vocoders: speech_vocoder_compatibility(backend),
        controls,
        catalog: catalog_entries,
        missing_catalog_ids,
        statuses,
        unavailable_reason,
        install_command,
    }
}

fn is_progressive_speech_catalog_entry(entry: &tongues_tts::ModelCatalogEntry) -> bool {
    entry.provenance.format == "fairseq-mms-vits"
        || (is_styletts2_catalog_entry(entry) && entry.id != "styletts2-en-us")
}

fn speech_catalog_family(entry: &tongues_tts::ModelCatalogEntry) -> &'static str {
    if entry.provenance.format == "fairseq-mms-vits" {
        "mms"
    } else if is_styletts2_catalog_entry(entry) {
        "styletts2"
    } else {
        "other"
    }
}

fn speech_catalog_entry_matches(
    store: &tongues_tts::ModelStore,
    entry: &tongues_tts::ModelCatalogEntry,
    filters: &SpeechDiscoveryFilters,
) -> bool {
    let search = filters.search.trim().to_ascii_lowercase();
    let family = filters.family.trim().to_ascii_lowercase();
    let license = filters.license.trim();
    let capability = filters.capability.trim().to_ascii_lowercase();
    let verification = filters.verification.trim().to_ascii_lowercase();
    let requested_device = filters.device.trim().to_ascii_lowercase();
    let verification_matches = if verification.is_empty() {
        true
    } else {
        let status = store.verification_state(entry).status;
        match verification.as_str() {
            "verified" => status == tongues_tts::ModelVerificationStatus::Verified,
            "pending" => matches!(
                status,
                tongues_tts::ModelVerificationStatus::PendingVerification
                    | tongues_tts::ModelVerificationStatus::ChangedSinceVerification
            ),
            "failed" => matches!(
                status,
                tongues_tts::ModelVerificationStatus::VerificationFailed
                    | tongues_tts::ModelVerificationStatus::Unavailable
            ),
            _ => false,
        }
    };
    (filters.model_ids.is_empty() || filters.model_ids.contains(&entry.id))
        && (family.is_empty() || speech_catalog_family(entry) == family)
        && (license.is_empty() || entry.license.expression.eq_ignore_ascii_case(license))
        && (capability.is_empty()
            || capability == "speech"
            || (capability == "voice_conversion" && is_styletts2_catalog_entry(entry)))
        && (requested_device.is_empty() || matches!(requested_device.as_str(), "cpu" | "cuda"))
        && verification_matches
        && (search.is_empty()
            || entry.id.to_ascii_lowercase().contains(&search)
            || entry.display_name.to_ascii_lowercase().contains(&search)
            || entry.architecture.to_ascii_lowercase().contains(&search)
            || entry
                .languages
                .iter()
                .any(|language| language.to_ascii_lowercase().contains(&search))
            || entry
                .script
                .as_ref()
                .is_some_and(|script| script.to_ascii_lowercase().contains(&search)))
}

#[cfg(test)]
fn speech_studio_discovery(
    home: &FsPath,
    device: tongues_tts::ResolvedSpeechDevice,
    loaded: &[String],
) -> SpeechStudioDiscovery {
    speech_studio_discovery_page(
        home,
        device,
        loaded,
        0,
        usize::MAX,
        &SpeechDiscoveryFilters::default(),
    )
}

fn speech_studio_discovery_page(
    home: &FsPath,
    device: tongues_tts::ResolvedSpeechDevice,
    loaded: &[String],
    cursor: usize,
    limit: usize,
    filters: &SpeechDiscoveryFilters,
) -> SpeechStudioDiscovery {
    let catalog_paths = tongues_tts::private_catalog_paths_from_environment();
    let catalog = match tongues_tts::ModelCatalog::with_private_catalogs(&catalog_paths) {
        Ok(catalog) => catalog,
        Err(error) => {
            return SpeechStudioDiscovery {
                schema_version: 4,
                page: SpeechDiscoveryPage {
                    cursor,
                    limit,
                    returned: 0,
                    total: 0,
                    next_cursor: None,
                },
                paths: Vec::new(),
                components: native_component_inventory_without_catalog(loaded),
                compositions: Vec::new(),
                compatibility: Vec::new(),
                presets: Vec::new(),
                verification_ids: Vec::new(),
                error: Some(format!("speech model catalog discovery failed: {error:#}")),
            };
        }
    };
    let cache = tongues_tts::default_model_cache(home)
        .unwrap_or_else(|_| home.join("cache/model-downloads"));
    let store = tongues_tts::ModelStore::new(home, cache).with_offline(true);
    let progressive_entries = catalog
        .entries
        .iter()
        .filter(|entry| is_progressive_speech_catalog_entry(entry))
        .filter(|entry| speech_catalog_entry_matches(&store, entry, filters))
        .collect::<Vec<_>>();
    let total = progressive_entries.len();
    let cursor = cursor.min(total);
    let page_end = cursor.saturating_add(limit).min(total);
    let page_ids = progressive_entries[cursor..page_end]
        .iter()
        .map(|entry| entry.id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let initial_page = cursor == 0;
    let scoped_catalog = tongues_tts::ModelCatalog {
        schema_version: catalog.schema_version,
        id: catalog.id.clone(),
        entries: catalog
            .entries
            .iter()
            .filter(|entry| {
                if is_progressive_speech_catalog_entry(entry) {
                    page_ids.contains(entry.id.as_str())
                } else {
                    initial_page
                }
            })
            .cloned()
            .collect(),
    };
    let verification = scoped_catalog
        .entries
        .iter()
        .map(|entry| (entry.id.clone(), store.verification_state(entry)))
        .collect::<SpeechVerification>();
    let selected_onnx =
        selected_onnx_voice_model_at(home).unwrap_or_else(|_| DEFAULT_ONNX_VOICE_MODEL.into());
    let mut paths = Vec::new();
    for provider in RESIDENT_BACKEND_PROVIDERS {
        if provider.id == "fairseq" {
            for entry in scoped_catalog
                .entries
                .iter()
                .filter(|entry| entry.provenance.format == "fairseq-mms-vits")
            {
                paths.push(discover_speech_path(
                    home,
                    provider.id,
                    &entry.id,
                    &entry.display_name,
                    entry.id == "fairseq-mms-vits-eng",
                    device,
                    &scoped_catalog,
                    &verification,
                    loaded,
                ));
            }
            continue;
        }
        if !initial_page {
            if provider.id == "styletts2" {
                for entry in scoped_catalog
                    .entries
                    .iter()
                    .filter(|entry| is_styletts2_catalog_entry(entry))
                {
                    paths.push(discover_speech_path(
                        home,
                        provider.id,
                        &entry.id,
                        &entry.display_name,
                        false,
                        device,
                        &scoped_catalog,
                        &verification,
                        loaded,
                    ));
                }
            }
            continue;
        }
        if provider.id == "onnx" {
            for model in ONNX_VOICE_MODELS {
                paths.push(discover_speech_path(
                    home,
                    provider.id,
                    model.id,
                    model.display_name,
                    model.id == selected_onnx,
                    device,
                    &scoped_catalog,
                    &verification,
                    loaded,
                ));
            }
            continue;
        }
        if provider.id == "styletts2" {
            {
                for entry in scoped_catalog
                    .entries
                    .iter()
                    .filter(|entry| is_styletts2_catalog_entry(entry))
                {
                    paths.push(discover_speech_path(
                        home,
                        provider.id,
                        &entry.id,
                        &entry.display_name,
                        entry.id == "styletts2-en-us",
                        device,
                        &scoped_catalog,
                        &verification,
                        loaded,
                    ));
                }
                continue;
            }
        }
        let model =
            speech_model_id(home, provider.id, None).unwrap_or_else(|_| "unavailable".into());
        paths.push(discover_speech_path(
            home,
            provider.id,
            &model,
            speech_model_display_name(provider.id, &model),
            provider.id != "mock",
            device,
            &scoped_catalog,
            &verification,
            loaded,
        ));
    }
    paths.sort_by_key(|path| {
        let developer = path.capabilities.backend == "mock";
        (!path.runnable, developer, path.display_name.clone())
    });
    let mut registered = base_registered_speech_compositions_at(home);
    extend_registered_speech_compositions(&mut registered, &scoped_catalog);
    registered.retain(|composition| {
        paths.iter().any(|path| {
            path.capabilities.backend == composition.backend
                && path.capabilities.model == composition.model
        })
    });
    let compositions = speech_composition_discovery(&registered, &paths);
    let presets = registered
        .iter()
        .filter(|composition| composition.recommended || composition.backend != "mock")
        .map(|composition| SpeechPresetDiscovery {
            id: format!("preset/{}", composition.model),
            display_name: composition.display_name.clone(),
            composition_id: composition.id.clone(),
            pipeline: composition.pipeline.clone(),
            developer: composition.backend == "mock",
        })
        .collect();
    let mut components = speech_component_inventory(
        &scoped_catalog,
        &store,
        &verification,
        &paths,
        loaded,
        initial_page,
    );
    add_pipeline_pseudo_components(&mut components, &registered, initial_page);
    SpeechStudioDiscovery {
        schema_version: 4,
        page: SpeechDiscoveryPage {
            cursor,
            limit,
            returned: page_end - cursor,
            total,
            next_cursor: (page_end < total).then_some(page_end),
        },
        components,
        compatibility: speech_pipeline_compatibility(&registered),
        compositions,
        presets,
        paths,
        verification_ids: scoped_catalog
            .entries
            .iter()
            .filter(|entry| {
                verification
                    .get(&entry.id)
                    .is_some_and(|state| state.status.needs_deep_verification())
            })
            .map(|entry| entry.id.clone())
            .collect(),
        error: None,
    }
}

fn speech_composition_discovery(
    registered: &[tongues_tts::RegisteredSpeechComposition],
    paths: &[SpeechPathDiscovery],
) -> Vec<SpeechCompositionDiscovery> {
    let mut compositions = registered
        .iter()
        .filter_map(|composition| {
            let path = paths.iter().find(|path| {
                path.capabilities.backend == composition.backend
                    && path.capabilities.model == composition.model
            })?;
            Some(SpeechCompositionDiscovery {
                id: composition.id.clone(),
                display_name: composition.display_name.clone(),
                backend: composition.backend.clone(),
                model: composition.model.clone(),
                pipeline: composition.pipeline.clone(),
                runnable: path.runnable,
                selected: path.selected,
                controls: path.controls.clone(),
                capabilities: path.capabilities.clone(),
                statuses: path.statuses.clone(),
                unavailable_reason: path.unavailable_reason.clone(),
            })
        })
        .collect::<Vec<_>>();
    compositions.sort_by_key(|composition| {
        (
            !composition.runnable,
            composition.backend == "mock",
            composition.display_name.clone(),
        )
    });
    compositions
}

fn linguistic_plan_port() -> tongues_tts::SpeechPortContract {
    tongues_tts::SpeechPortContract {
        kind: "linguistic_plan".into(),
        key: "tongues/utterance-plan-v1".into(),
        summary: "Backend-neutral Tongues utterance plan.".into(),
    }
}

fn waveform_port() -> tongues_tts::SpeechPortContract {
    tongues_tts::SpeechPortContract {
        kind: "waveform".into(),
        key: "audio/wav-mono-f32".into(),
        summary: "Mono waveform rendered as downloadable WAV audio.".into(),
    }
}

fn mel_port(key: &str, summary: &str) -> tongues_tts::SpeechPortContract {
    tongues_tts::SpeechPortContract {
        kind: "mel_spectrogram".into(),
        key: key.into(),
        summary: summary.into(),
    }
}

fn speech_pipeline_compatibility(
    registered: &[tongues_tts::RegisteredSpeechComposition],
) -> Vec<tongues_tts::SpeechPipelineCompatibility> {
    // Missing edges are incompatible by default in the studio. Emitting every
    // negative checkpoint-projector pairing made this response quadratic (more
    // than 1.3 million edges for the MMS catalog), so only registered positive
    // projector contracts belong on the wire.
    let mut compatibility = registered
        .iter()
        .map(|composition| {
            let generator = composition
                .pipeline
                .acoustic_model
                .as_ref()
                .or(composition.pipeline.end_to_end.as_ref())
                .expect("registered composition has a generator");
            tongues_tts::SpeechPipelineCompatibility {
                from_component_id: composition.pipeline.projector.clone(),
                to_component_id: generator.clone(),
                compatible: true,
                reason: "This checkpoint-owned projector emits the exact vocabulary and token contract required by the model.".into(),
            }
        })
        .collect::<Vec<_>>();
    let acoustic_models = [
        "speedy-speech-ljspeech",
        "fastpitch-ljspeech",
        "glow-tts-ljspeech",
    ];
    let vocoders = [
        "hifigan-v2-ljspeech",
        "glow-standardized-multiband-melgan-ljspeech",
        "multiband-melgan-ljspeech",
        "melgan",
    ];
    for acoustic in acoustic_models {
        for vocoder in vocoders {
            let registered_pair = registered.iter().any(|composition| {
                composition.pipeline.acoustic_model.as_deref() == Some(acoustic)
                    && composition.pipeline.vocoder.as_deref() == Some(vocoder)
            });
            let reason = if registered_pair {
                "Construction validates an exact match across mel layout, bins, hop size, frequency bounds, log scale, filter bank, and normalization.".into()
            } else if vocoder == "glow-standardized-multiband-melgan-ljspeech" {
                format!(
                    "This path is restricted to Glow-TTS and applies the named `{}` conversion.",
                    tongues_tts::GLOW_MULTIBAND_STANDARDIZER_ID
                )
            } else if vocoder == "multiband-melgan-ljspeech" {
                "The vocoder requires its published standardized/PQMF feature contract; this acoustic model does not emit that exact normalization identity.".into()
            } else if vocoder == "melgan" {
                "No verified MelGAN artifact with an exact matching spectrogram contract is registered.".into()
            } else {
                "No executable loader has established an exact contract match for this pair.".into()
            };
            compatibility.push(tongues_tts::SpeechPipelineCompatibility {
                from_component_id: acoustic.into(),
                to_component_id: vocoder.into(),
                compatible: registered_pair,
                reason,
            });
        }
    }
    compatibility
}

fn add_pipeline_pseudo_components(
    components: &mut Vec<SpeechComponentDiscovery>,
    registered: &[tongues_tts::RegisteredSpeechComposition],
    include_baseline: bool,
) {
    let styletts2_paths = registered
        .iter()
        .filter(|composition| composition.backend == "styletts2")
        .map(|composition| composition.model.clone())
        .collect::<Vec<_>>();
    for descriptor in tongues_tts::registered_speech_pipeline_components()
        .into_iter()
        .filter(|_| include_baseline)
    {
        if let Some(component) = components
            .iter_mut()
            .find(|component| component.id == descriptor.id)
        {
            component.stage = descriptor.stage;
            component.spans = descriptor.spans;
            component.accepts = descriptor.accepts;
            component.produces = descriptor.produces;
            component.control_fields = descriptor.controls;
            continue;
        }
        components.push(SpeechComponentDiscovery {
            id: descriptor.id,
            display_name: descriptor.display_name,
            architecture: descriptor.architecture,
            kind: match descriptor.stage {
                tongues_tts::SpeechPipelineStage::Input => "input",
                tongues_tts::SpeechPipelineStage::Projector => "projector",
                tongues_tts::SpeechPipelineStage::AcousticModel => "acoustic",
                tongues_tts::SpeechPipelineStage::Conditioner => "voice",
                tongues_tts::SpeechPipelineStage::Vocoder => "vocoder",
                tongues_tts::SpeechPipelineStage::EndToEnd => "end_to_end",
                tongues_tts::SpeechPipelineStage::Output => "output",
            }
            .into(),
            stage: descriptor.stage,
            spans: descriptor.spans,
            accepts: descriptor.accepts,
            produces: descriptor.produces,
            control_fields: descriptor.controls,
            runnable: matches!(
                descriptor.stage,
                tongues_tts::SpeechPipelineStage::Input
                    | tongues_tts::SpeechPipelineStage::Projector
                    | tongues_tts::SpeechPipelineStage::Output
            ),
            installed: true,
            verified: true,
            verification_pending: false,
            verification_status: tongues_tts::ModelVerificationStatus::Verified,
            load_state: "unloaded",
            readiness: native_component_readiness(descriptor.readiness).into(),
            statuses: vec!["Registered".into()],
            explanation: descriptor.explanation,
            compatible_paths: Vec::new(),
            catalog: Vec::new(),
            install_command: None,
        });
    }
    if include_baseline
        && !components
            .iter()
            .any(|component| component.id == "styletts2")
    {
        components.push(SpeechComponentDiscovery {
            id: "styletts2".into(),
            display_name: "StyleTTS2".into(),
            architecture: "styletts2".into(),
            kind: "end_to_end".into(),
            stage: tongues_tts::SpeechPipelineStage::EndToEnd,
            spans: vec![
                tongues_tts::SpeechPipelineStage::AcousticModel,
                tongues_tts::SpeechPipelineStage::Vocoder,
            ],
            accepts: Vec::new(),
            produces: vec![waveform_port()],
            control_fields: vec![
                "voice_sample".into(),
                "style_sample".into(),
                "emotion".into(),
            ],
            runnable: registered
                .iter()
                .any(|composition| composition.backend == "styletts2"),
            installed: false,
            verified: false,
            verification_pending: false,
            verification_status: tongues_tts::ModelVerificationStatus::Unavailable,
            load_state: "unloaded",
            readiness: "experimental".into(),
            statuses: vec!["Experimental".into()],
            explanation:
                "Reference-conditioned end-to-end compatibility engine exposed as a spanning block."
                    .into(),
            compatible_paths: if styletts2_paths.is_empty() {
                vec!["styletts2-en-us".into()]
            } else {
                styletts2_paths.clone()
            },
            catalog: Vec::new(),
            install_command: None,
        });
    }
    for composition in registered {
        if components
            .iter()
            .any(|component| component.id == composition.pipeline.projector)
        {
            continue;
        }
        components.push(SpeechComponentDiscovery {
            id: composition.pipeline.projector.clone(),
            display_name: format!("{} projector", composition.display_name),
            architecture: "checkpoint-projector".into(),
            kind: "projector".into(),
            stage: tongues_tts::SpeechPipelineStage::Projector,
            spans: Vec::new(),
            accepts: vec![linguistic_plan_port()],
            produces: vec![tongues_tts::SpeechPortContract {
                kind: "model_tokens".into(),
                key: format!("tokens/{}", composition.model),
                summary: format!(
                    "Checkpoint-local vocabulary and tokenization for {}.",
                    composition.display_name
                ),
            }],
            control_fields: Vec::new(),
            runnable: true,
            installed: true,
            verified: true,
            verification_pending: false,
            verification_status: tongues_tts::ModelVerificationStatus::Verified,
            load_state: "loaded",
            readiness: "runtime".into(),
            statuses: vec!["Checkpoint-bound".into()],
            explanation:
                "Terminal projection into a checkpoint-private vocabulary; it cannot be substituted across models."
                    .into(),
            compatible_paths: vec![composition.model.clone()],
            catalog: Vec::new(),
            install_command: None,
        });
    }
    if include_baseline
        && !components
            .iter()
            .any(|component| component.id == "style-reference-encoder")
    {
        components.push(SpeechComponentDiscovery {
            id: "style-reference-encoder".into(),
            display_name: "Style reference encoder".into(),
            architecture: "styletts2-reference-encoder".into(),
            kind: "voice".into(),
            stage: tongues_tts::SpeechPipelineStage::Conditioner,
            spans: Vec::new(),
            accepts: Vec::new(),
            produces: Vec::new(),
            control_fields: vec![
                "voice_sample".into(),
                "style_sample".into(),
                "emotion".into(),
            ],
            runnable: true,
            installed: true,
            verified: true,
            verification_pending: false,
            verification_status: tongues_tts::ModelVerificationStatus::Verified,
            load_state: "unloaded",
            readiness: "runtime".into(),
            statuses: vec!["Registered".into()],
            explanation: "StyleTTS2 speaker and style reference conditioning.".into(),
            compatible_paths: if styletts2_paths.is_empty() {
                vec!["styletts2-en-us".into()]
            } else {
                styletts2_paths
            },
            catalog: Vec::new(),
            install_command: None,
        });
    }
    for component in components.iter_mut() {
        match component.id.as_str() {
            "speedy-speech-ljspeech" | "fastpitch-ljspeech" => {
                component.accepts = vec![tongues_tts::SpeechPortContract {
                    kind: "model_tokens".into(),
                    key: format!("tokens/{}", component.id),
                    summary: "Checkpoint-private projected tokens.".into(),
                }];
                component.produces = vec![mel_port(
                    "mel/coqui-ljspeech-neutral-v1",
                    "80-bin LJSpeech mel features with the complete published analysis contract.",
                )];
            }
            "glow-tts-ljspeech" => {
                component.accepts = vec![tongues_tts::SpeechPortContract {
                    kind: "model_tokens".into(),
                    key: "tokens/glow-tts-ljspeech".into(),
                    summary: "Checkpoint-private projected tokens.".into(),
                }];
                component.produces = vec![mel_port(
                    "mel/glow-tts-ljspeech-log10-v1",
                    "Unstandardized 80-bin Glow-TTS LJSpeech mel features.",
                )];
            }
            "hifigan-v2-ljspeech" => {
                component.accepts = vec![mel_port(
                    "mel/coqui-ljspeech-neutral-v1",
                    "Exact 80-bin LJSpeech mel contract.",
                )];
                component.produces = vec![waveform_port()];
            }
            "multiband-melgan-ljspeech" => {
                component.accepts = vec![mel_port(
                    "mel/coqui-ljspeech-standardized-pqmf-v1",
                    "Standardized 80-bin mel features tied to the published statistics artifact.",
                )];
                component.produces = vec![waveform_port()];
            }
            "glow-standardized-multiband-melgan-ljspeech" => {
                component.accepts = vec![mel_port(
                    "mel/glow-tts-ljspeech-log10-v1",
                    "Glow-TTS features accepted by the pinned named standardizer.",
                )];
                component.produces = vec![waveform_port()];
            }
            _ if component.stage == tongues_tts::SpeechPipelineStage::EndToEnd => {
                component.produces = vec![waveform_port()];
            }
            _ => {}
        }
    }
    components.sort_by(|left, right| {
        format!("{:?}", left.stage)
            .cmp(&format!("{:?}", right.stage))
            .then_with(|| left.display_name.cmp(&right.display_name))
    });
}

fn speech_path_catalog_ids(
    home: &FsPath,
    backend: &str,
    model: Option<&str>,
) -> anyhow::Result<Vec<String>> {
    Ok(match backend {
        "burn" => vec![
            "speedy-speech-ljspeech".into(),
            "hifigan-v2-ljspeech".into(),
        ],
        "fastpitch" => vec!["fastpitch-ljspeech".into(), "hifigan-v2-ljspeech".into()],
        "glow" => vec![
            "glow-tts-ljspeech".into(),
            "multiband-melgan-ljspeech".into(),
        ],
        "vits" => vec!["vits-vctk".into()],
        "fairseq" => vec![speech_model_id(home, backend, model)?],
        "yourtts" => vec!["yourtts-multilingual".into()],
        "freevc" => vec!["freevc24-vctk".into()],
        "styletts2" => vec![speech_model_id(home, backend, model)?],
        "onnx" => vec![
            model
                .filter(|model| !model.trim().is_empty())
                .map(str::to_string)
                .unwrap_or(speech_model_id(home, backend, None)?),
        ],
        "mock" => Vec::new(),
        _ => anyhow::bail!("unknown speech backend `{backend}`"),
    })
}

fn speech_path_components(
    backend: &str,
    model: &str,
) -> (Option<String>, Option<String>, Option<String>, Vec<String>) {
    match backend {
        "burn" => (
            Some("speedy-speech-ljspeech".into()),
            Some("hifigan-v2-ljspeech".into()),
            None,
            vec!["speedy-speech".into(), "hifigan".into()],
        ),
        "fastpitch" => (
            Some("fastpitch-ljspeech".into()),
            Some("hifigan-v2-ljspeech".into()),
            None,
            vec!["fastpitch".into(), "hifigan".into()],
        ),
        "glow" => (
            Some("glow-tts-ljspeech".into()),
            Some("glow-standardized-multiband-melgan-ljspeech".into()),
            None,
            vec![
                "glow-tts".into(),
                tongues_tts::GLOW_MULTIBAND_STANDARDIZER_ID.into(),
                "multiband-melgan".into(),
            ],
        ),
        "vits" => (None, None, Some("vits-vctk".into()), vec!["vits".into()]),
        "fairseq" => (
            None,
            None,
            Some(model.into()),
            vec!["fairseq-mms-vits".into(), "vits".into()],
        ),
        "yourtts" => (
            None,
            None,
            Some("yourtts-multilingual".into()),
            vec!["vits".into(), "speaker-encoder".into(), "yourtts".into()],
        ),
        "freevc" => (
            None,
            None,
            Some("freevc24-vctk".into()),
            vec![
                "freevc".into(),
                "wavlm-large".into(),
                "freevc-speaker-encoder".into(),
            ],
        ),
        "styletts2" => (None, None, Some(model.into()), vec!["styletts2".into()]),
        "onnx" => (None, None, Some(model.into()), vec![model.into()]),
        "mock" => (
            None,
            None,
            Some("deterministic-mock".into()),
            vec!["deterministic-mock".into()],
        ),
        _ => (None, None, None, Vec::new()),
    }
}

fn speech_path_is_loaded(backend: &str, model: &str, loaded: &[String]) -> bool {
    loaded
        .iter()
        .any(|engine| speech_engine_matches_path(engine, backend, model))
}

fn speech_vocoder_compatibility(backend: &str) -> Vec<SpeechCompatibility> {
    if backend == "glow" {
        return vec![SpeechCompatibility {
            component_id: "glow-standardized-multiband-melgan-ljspeech".into(),
            compatible: true,
            reason: format!(
                "The named `{}` conversion applies the checksum-pinned published statistics before MultiBand-MelGAN.",
                tongues_tts::GLOW_MULTIBAND_STANDARDIZER_ID
            ),
        }];
    }
    if !matches!(backend, "burn" | "fastpitch") {
        return Vec::new();
    }
    vec![
        SpeechCompatibility {
            component_id: "hifigan-v2-ljspeech".into(),
            compatible: true,
            reason: "The acoustic and vocoder feature contracts both use the published LJSpeech 80-bin mel representation.".into(),
        },
        SpeechCompatibility {
            component_id: "multiband-melgan-ljspeech".into(),
            compatible: false,
            reason: "The published MultiBand-MelGAN normalization/PQMF feature contract is not registered as compatible with this acoustic path.".into(),
        },
        SpeechCompatibility {
            component_id: "melgan".into(),
            compatible: false,
            reason: "No verified MelGAN artifact with a matching acoustic feature contract is installed.".into(),
        },
    ]
}

fn speech_control(
    field: &'static str,
    label: &'static str,
    kind: &'static str,
    group: &'static str,
    default: Option<serde_json::Value>,
    help: &'static str,
) -> SpeechControlDiscovery {
    SpeechControlDiscovery {
        field,
        label,
        kind,
        group,
        min: None,
        max: None,
        step: None,
        default,
        unit: None,
        help,
        options: Vec::new(),
    }
}

fn speech_number_control(
    field: &'static str,
    label: &'static str,
    group: &'static str,
    range: (f64, f64, f64),
    default: serde_json::Value,
    unit: Option<&'static str>,
    help: &'static str,
) -> SpeechControlDiscovery {
    let mut control = speech_control(field, label, "number", group, Some(default), help);
    control.min = Some(range.0);
    control.max = Some(range.1);
    control.step = Some(range.2);
    control.unit = unit;
    control
}

fn speech_control_discovery(
    backend: &str,
    capabilities: &tongues_tts::BackendCapabilities,
    resolved_device: tongues_tts::ResolvedSpeechDevice,
) -> Vec<SpeechControlDiscovery> {
    let mut controls = Vec::new();
    let resolved_device_value = match resolved_device {
        tongues_tts::ResolvedSpeechDevice::Cpu => "cpu".to_string(),
        tongues_tts::ResolvedSpeechDevice::Cuda { index } => format!("cuda:{index}"),
    };
    let mut device = speech_control(
        "device",
        "Device",
        "select",
        "basic",
        Some(json!(resolved_device_value)),
        "Defaults to the resident runtime device. Choose another supported device to override it for this request.",
    );
    device.options = capabilities
        .devices
        .iter()
        .map(|request| match request {
            tongues_tts::SpeechDeviceRequest::Auto => SpeechControlOption {
                value: "auto".into(),
                label: "Automatic".into(),
            },
            tongues_tts::SpeechDeviceRequest::Cpu => SpeechControlOption {
                value: "cpu".into(),
                label: "CPU".into(),
            },
            tongues_tts::SpeechDeviceRequest::Cuda { index } => SpeechControlOption {
                value: format!("cuda:{index}"),
                label: format!("CUDA {index}"),
            },
        })
        .collect();
    controls.push(device);
    // A catalog language identifies a single-language checkpoint; it is not a
    // learned language embedding that callers may select. Only advertise a
    // request control when the checkpoint declares a real learned-language
    // selection contract.
    if (capabilities.languages.required || capabilities.languages.numeric_ids)
        && let tongues_tts::CapabilityValue::Listed(values) = &capabilities.languages.values
    {
        let mut language = speech_control(
            "model_language",
            "Model language",
            "select",
            "basic",
            values.first().map(|value| json!(value.id)),
            "Checkpoint-local learned language identity, separate from linguistic variety.",
        );
        language.options = values
            .iter()
            .map(|value| SpeechControlOption {
                value: value.id.clone(),
                label: value.label.clone(),
            })
            .collect();
        controls.push(language);
    }
    if capabilities.speed {
        controls.push(speech_number_control(
            "speed",
            "Speed",
            "advanced",
            (0.5, 1.5, 0.01),
            json!(1.0),
            Some("×"),
            "Speech speed multiplier declared by the selected path.",
        ));
    }
    if capabilities.seed {
        controls.push(speech_number_control(
            "seed",
            "Synthesis seed",
            "advanced",
            (0.0, u32::MAX as f64, 1.0),
            json!(0),
            None,
            "Seed for repeatable stochastic inference.",
        ));
    }
    if capabilities.reference_audio.speaker {
        controls.push(speech_control(
            if backend == "freevc" {
                "target_audio"
            } else {
                "voice_sample"
            },
            if backend == "freevc" {
                "Target speaker"
            } else {
                "Voice reference"
            },
            "reference_audio",
            "basic",
            None,
            "Reference WAV used for speaker timbre; includes an audio preview.",
        ));
    }
    if capabilities.reference_audio.source {
        controls.push(speech_control(
            "source_audio",
            "Source audio",
            "reference_audio",
            "basic",
            None,
            "Source WAV whose linguistic content is converted to the target speaker.",
        ));
    }
    if capabilities.reference_audio.style {
        controls.push(speech_control(
            "style_sample",
            "Style reference",
            "reference_audio",
            "basic",
            None,
            "Reference WAV used for prosody and style; includes an audio preview.",
        ));
    }
    if capabilities.styles.reference_audio || capabilities.styles.embedding_dimensions.is_some() {
        controls.push(speech_control(
            "emotion",
            "Emotion signature",
            "emotion",
            "basic",
            None,
            "Optional server-discovered StyleTTS2 emotion delta.",
        ));
        controls.push(speech_number_control(
            "emotion_strength",
            "Emotion strength",
            "basic",
            (0.0, 2.0, 0.05),
            json!(0.75),
            Some("×"),
            "Multiplier for the selected emotion delta.",
        ));
        let mut quality = speech_control(
            "quality",
            "Quality preset",
            "select",
            "advanced",
            Some(json!("balanced")),
            "Preset for diffusion-based synthesis.",
        );
        quality.options = vec![
            SpeechControlOption {
                value: "balanced".into(),
                label: "Balanced".into(),
            },
            SpeechControlOption {
                value: "fast".into(),
                label: "Fast".into(),
            },
        ];
        controls.push(quality);
        controls.push(speech_number_control(
            "diffusion_steps",
            "Diffusion steps",
            "advanced",
            (1.0, 32.0, 1.0),
            json!(5),
            Some("steps"),
            "More diffusion steps usually improve refinement and increase latency.",
        ));
        let mut blend_mode = speech_control(
            "blend_mode",
            "Reference blending",
            "select",
            "advanced",
            Some(json!("strength")),
            "Choose friendly reference strengths or raw StyleTTS2 alpha and beta.",
        );
        blend_mode.options = vec![
            SpeechControlOption {
                value: "strength".into(),
                label: "Reference strength".into(),
            },
            SpeechControlOption {
                value: "raw".into(),
                label: "Raw alpha / beta".into(),
            },
        ];
        controls.push(blend_mode);
        controls.push(speech_number_control(
            "speaker_reference_strength",
            "Speaker reference strength",
            "advanced",
            (0.0, 1.0, 0.01),
            json!(0.7),
            None,
            "Higher values retain more timbre from the voice reference.",
        ));
        controls.push(speech_number_control(
            "style_reference_strength",
            "Style reference strength",
            "advanced",
            (0.0, 1.0, 0.01),
            json!(0.9),
            None,
            "Higher values retain more prosody from the style reference.",
        ));
        controls.push(speech_number_control(
            "style_alpha",
            "Raw alpha",
            "expert",
            (0.0, 1.0, 0.01),
            json!(0.3),
            None,
            "Raw StyleTTS2 speaker/timbre blend. Sending this replaces the friendly speaker strength.",
        ));
        controls.push(speech_number_control(
            "style_beta",
            "Raw beta",
            "expert",
            (0.0, 1.0, 0.01),
            json!(0.1),
            None,
            "Raw StyleTTS2 style/prosody blend. Sending this replaces the friendly style strength.",
        ));
        controls.push(speech_number_control(
            "embedding_scale",
            "Embedding scale",
            "advanced",
            (0.0, 3.0, 0.05),
            json!(1.0),
            Some("×"),
            "Diffusion embedding guidance scale.",
        ));
    }
    if matches!(backend, "vits" | "fairseq" | "yourtts" | "onnx") {
        controls.push(speech_number_control(
            "noise_scale",
            "Noise scale",
            "advanced",
            (0.0, 2.0, 0.01),
            json!(0.667),
            None,
            "Controls acoustic variation for this VITS-compatible path.",
        ));
        controls.push(speech_number_control(
            "duration_noise_scale",
            "Duration noise scale",
            "advanced",
            (0.0, 2.0, 0.01),
            json!(0.8),
            None,
            "Controls stochastic duration variation.",
        ));
    }
    if backend == "freevc" {
        controls.push(speech_number_control(
            "noise_scale",
            "Content noise scale",
            "advanced",
            (0.0, 2.0, 0.01),
            json!(1.0),
            None,
            "Controls stochastic sampling from the FreeVC content prior.",
        ));
    }
    if capabilities.pitch.scale {
        controls.push(speech_number_control(
            "pitch_scale",
            "Pitch scale",
            "advanced",
            (0.25, 2.0, 0.01),
            json!(1.0),
            Some("×"),
            "Multiplies normalized predicted pitch.",
        ));
    }
    if capabilities.pitch.shift {
        controls.push(speech_number_control(
            "pitch_shift",
            "Pitch shift",
            "advanced",
            (-2.0, 2.0, 0.01),
            json!(0.0),
            None,
            "Adds an offset in the model's normalized pitch space.",
        ));
    }
    if capabilities.pitch.explicit_values {
        controls.push(speech_control(
            "pitch",
            "Per-token pitch",
            "number_array",
            "expert",
            None,
            "Comma-separated normalized values; the count must match projected tokens.",
        ));
    }
    if capabilities.energy.scale {
        controls.push(speech_number_control(
            "energy_scale",
            "Energy scale",
            "advanced",
            (0.25, 2.0, 0.01),
            json!(1.0),
            Some("×"),
            "Multiplies predicted token energy.",
        ));
    }
    if capabilities.energy.shift {
        controls.push(speech_number_control(
            "energy_shift",
            "Energy shift",
            "advanced",
            (-2.0, 2.0, 0.01),
            json!(0.0),
            None,
            "Adds an offset in the model's normalized energy space.",
        ));
    }
    if capabilities.energy.explicit_values {
        controls.push(speech_control(
            "energy",
            "Per-token energy",
            "number_array",
            "expert",
            None,
            "Comma-separated normalized values; the count must match projected tokens.",
        ));
    }
    if capabilities.durations {
        controls.push(speech_control(
            "durations",
            "Per-token durations",
            "positive_integer_array",
            "expert",
            None,
            "Comma-separated positive mel-frame counts; the count must match projected tokens.",
        ));
    }
    if backend == "mock" {
        let mut rate = speech_control(
            "sample_rate_hz",
            "Sample rate",
            "select",
            "developer",
            Some(json!("24000")),
            "Output sample rate for deterministic test audio.",
        );
        rate.options = [16_000, 22_050, 24_000, 48_000]
            .into_iter()
            .map(|value| SpeechControlOption {
                value: value.to_string(),
                label: format!("{value} Hz"),
            })
            .collect();
        controls.push(rate);
    }
    controls.push(speech_number_control(
        "max_tts_symbols",
        "Chunk size",
        "advanced",
        (16.0, 2048.0, 1.0),
        json!(180),
        Some("symbols"),
        "Maximum projected symbols per synthesis chunk.",
    ));
    controls.push(speech_control(
        "no_tts_chunking",
        "Disable chunking",
        "boolean",
        "advanced",
        Some(json!(false)),
        "Send the prompt as one synthesis chunk.",
    ));
    controls.push(speech_control(
        "timings",
        "Timing and diagnostics",
        "boolean",
        "advanced",
        Some(json!(false)),
        "Return model stage timing diagnostics with the audio result.",
    ));
    controls
}

fn native_component_inventory_without_catalog(loaded: &[String]) -> Vec<SpeechComponentDiscovery> {
    tongues_tts::native_speech_components()
        .iter()
        .map(|component| SpeechComponentDiscovery {
            id: component.id.into(),
            display_name: component.display_name.into(),
            architecture: component.architecture.into(),
            kind: native_component_kind(component.kind).into(),
            stage: native_component_stage(component.kind),
            spans: native_component_spans(component.kind),
            accepts: Vec::new(),
            produces: Vec::new(),
            control_fields: Vec::new(),
            runnable: false,
            installed: false,
            verified: false,
            verification_pending: false,
            verification_status: tongues_tts::ModelVerificationStatus::Unavailable,
            load_state: if loaded.iter().any(|engine| engine.contains(component.id)) {
                "loaded"
            } else {
                "unloaded"
            },
            readiness: native_component_readiness(component.readiness).into(),
            statuses: vec![native_component_status(component.readiness).into()],
            explanation: component.explanation.into(),
            compatible_paths: Vec::new(),
            catalog: Vec::new(),
            install_command: None,
        })
        .collect()
}

fn speech_component_inventory(
    catalog: &tongues_tts::ModelCatalog,
    store: &tongues_tts::ModelStore,
    verification: &SpeechVerification,
    paths: &[SpeechPathDiscovery],
    loaded: &[String],
    include_baseline: bool,
) -> Vec<SpeechComponentDiscovery> {
    let mut components = if include_baseline {
        native_component_inventory_without_catalog(loaded)
    } else {
        Vec::new()
    };
    for entry in &catalog.entries {
        let component_id = catalog_component_id(entry);
        let index = components
            .iter()
            .position(|component| component.id == component_id);
        let installed = catalog_entry_files_present(store.root(), entry);
        let verification_status = verification
            .get(&entry.id)
            .map(|state| state.status)
            .unwrap_or(tongues_tts::ModelVerificationStatus::Unavailable);
        let verified = verification_status == tongues_tts::ModelVerificationStatus::Verified;
        let verification_pending =
            verification_status == tongues_tts::ModelVerificationStatus::PendingVerification;
        let compatible_paths = paths
            .iter()
            .filter(|path| {
                path.component_ids.iter().any(|id| id == &component_id)
                    || path
                        .catalog
                        .iter()
                        .any(|candidate| candidate.id == entry.id)
            })
            .map(|path| path.id.clone())
            .collect::<Vec<_>>();
        let runnable = paths.iter().any(|path| {
            path.runnable
                && (path.component_ids.iter().any(|id| id == &component_id)
                    || path
                        .catalog
                        .iter()
                        .any(|candidate| candidate.id == entry.id))
        });
        let load_state = if paths.iter().any(|path| {
            path.load_state == "loaded"
                && (path.component_ids.iter().any(|id| id == &component_id)
                    || path
                        .catalog
                        .iter()
                        .any(|candidate| candidate.id == entry.id))
        }) {
            "loaded"
        } else {
            "unloaded"
        };
        if let Some(index) = index {
            let component = &mut components[index];
            component.catalog.push(entry.clone());
            component.installed |= installed;
            component.verified |= verified;
            component.verification_pending |= verification_pending;
            component.verification_status =
                aggregate_verification_status([component.verification_status, verification_status]);
            component.runnable |= runnable;
            component.load_state = load_state;
            component.compatible_paths.extend(compatible_paths);
            component.compatible_paths.sort();
            component.compatible_paths.dedup();
            component.statuses = component_statuses(
                component.readiness.as_str(),
                component.runnable,
                component.installed,
                component.verified,
                component.verification_pending,
                component.verification_status,
                component.load_state,
                &component.catalog,
            );
            component.install_command = (!verified && !verification_pending)
                .then(|| format!("cargo run --bin tongues -- models install {}", entry.id));
            if !runnable && component.kind == "acoustic" && installed {
                component.explanation = format!(
                    "{} The artifact still needs a registered compatible vocoder path.",
                    component.explanation
                );
            }
            continue;
        }
        let readiness = if runnable { "runtime" } else { "experimental" };
        let explanation = if runnable {
            "Catalog model exposed by a complete registered synthesis path.".into()
        } else {
            "Catalog artifact is not part of a complete registered synthesis path.".into()
        };
        let catalog_entries = vec![entry.clone()];
        components.push(SpeechComponentDiscovery {
            id: component_id,
            display_name: entry.display_name.clone(),
            architecture: entry.architecture.clone(),
            kind: catalog_component_kind(entry).into(),
            stage: catalog_component_stage(entry),
            spans: catalog_component_spans(entry),
            accepts: Vec::new(),
            produces: Vec::new(),
            control_fields: entry.capabilities.clone(),
            runnable,
            installed,
            verified,
            verification_pending,
            verification_status,
            load_state,
            readiness: readiness.into(),
            statuses: component_statuses(
                readiness,
                runnable,
                installed,
                verified,
                verification_pending,
                verification_status,
                load_state,
                &catalog_entries,
            ),
            explanation,
            compatible_paths,
            catalog: catalog_entries,
            install_command: (!verified && !verification_pending)
                .then(|| format!("cargo run --bin tongues -- models install {}", entry.id)),
        });
    }
    if include_baseline {
        components.push(SpeechComponentDiscovery {
            id: "deterministic-mock".into(),
            display_name: "Deterministic Mock".into(),
            architecture: "deterministic-test-wave".into(),
            kind: "test".into(),
            stage: tongues_tts::SpeechPipelineStage::EndToEnd,
            spans: vec![
                tongues_tts::SpeechPipelineStage::AcousticModel,
                tongues_tts::SpeechPipelineStage::Vocoder,
            ],
            accepts: Vec::new(),
            produces: vec![waveform_port()],
            control_fields: Vec::new(),
            runnable: true,
            installed: true,
            verified: true,
            verification_pending: false,
            verification_status: tongues_tts::ModelVerificationStatus::Verified,
            load_state: if loaded.iter().any(|engine| engine.starts_with("mock-")) {
                "loaded"
            } else {
                "unloaded"
            },
            readiness: "test".into(),
            statuses: vec!["Test backend".into()],
            explanation: "Deterministic waveform generator for development and contract tests; it is not a voice engine.".into(),
            compatible_paths: vec!["deterministic-mock".into()],
            catalog: Vec::new(),
            install_command: None,
        });
    }
    components.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.display_name.cmp(&right.display_name))
    });
    components
}

fn catalog_component_id(entry: &tongues_tts::ModelCatalogEntry) -> String {
    entry.id.clone()
}

fn catalog_component_kind(entry: &tongues_tts::ModelCatalogEntry) -> &'static str {
    if entry
        .capabilities
        .iter()
        .any(|value| value == "neural-vocoder")
    {
        "vocoder"
    } else if entry
        .capabilities
        .iter()
        .any(|value| value == "acoustic-model")
    {
        "acoustic"
    } else if entry.id.starts_with("voice-") {
        "voice"
    } else {
        "end_to_end"
    }
}

fn catalog_entry_files_present(root: &FsPath, entry: &tongues_tts::ModelCatalogEntry) -> bool {
    !entry.artifacts.is_empty()
        && entry.artifacts.iter().all(|artifact| {
            if artifact.members.is_empty() {
                root.join(&artifact.install_path).is_file()
            } else {
                artifact
                    .members
                    .iter()
                    .all(|member| root.join(&member.install_path).is_file())
            }
        })
}

fn component_statuses(
    readiness: &str,
    runnable: bool,
    installed: bool,
    verified: bool,
    _verification_pending: bool,
    verification_status: tongues_tts::ModelVerificationStatus,
    load_state: &str,
    catalog: &[tongues_tts::ModelCatalogEntry],
) -> Vec<String> {
    let mut statuses = Vec::new();
    if runnable {
        statuses.push("Ready".into());
    } else {
        statuses.push(
            match readiness {
                "import_only" => "Import-only",
                "experimental" => "Experimental",
                _ if !installed => "Artifact missing",
                _ => "Needs compatible vocoder",
            }
            .into(),
        );
    }
    if installed {
        statuses.push("Installed".into());
    } else if !catalog.is_empty() {
        statuses.push("Downloadable".into());
    }
    if verified {
        statuses.push("Verified".into());
    } else if !catalog.is_empty() {
        statuses.push(verification_status_label(verification_status).into());
    }
    if load_state == "loaded" {
        statuses.push("Loaded".into());
    }
    statuses.sort();
    statuses.dedup();
    statuses
}

fn native_component_kind(kind: tongues_tts::NativeSpeechComponentKind) -> &'static str {
    match kind {
        tongues_tts::NativeSpeechComponentKind::EndToEnd => "end_to_end",
        tongues_tts::NativeSpeechComponentKind::VoiceConversion => "voice_conversion",
        tongues_tts::NativeSpeechComponentKind::Acoustic => "acoustic",
        tongues_tts::NativeSpeechComponentKind::Vocoder => "vocoder",
        tongues_tts::NativeSpeechComponentKind::Voice => "voice",
        tongues_tts::NativeSpeechComponentKind::Trainer => "trainer",
        tongues_tts::NativeSpeechComponentKind::Test => "test",
    }
}

fn native_component_stage(
    kind: tongues_tts::NativeSpeechComponentKind,
) -> tongues_tts::SpeechPipelineStage {
    match kind {
        tongues_tts::NativeSpeechComponentKind::EndToEnd
        | tongues_tts::NativeSpeechComponentKind::VoiceConversion
        | tongues_tts::NativeSpeechComponentKind::Trainer
        | tongues_tts::NativeSpeechComponentKind::Test => {
            tongues_tts::SpeechPipelineStage::EndToEnd
        }
        tongues_tts::NativeSpeechComponentKind::Acoustic => {
            tongues_tts::SpeechPipelineStage::AcousticModel
        }
        tongues_tts::NativeSpeechComponentKind::Vocoder => {
            tongues_tts::SpeechPipelineStage::Vocoder
        }
        tongues_tts::NativeSpeechComponentKind::Voice => {
            tongues_tts::SpeechPipelineStage::Conditioner
        }
    }
}

fn native_component_spans(
    kind: tongues_tts::NativeSpeechComponentKind,
) -> Vec<tongues_tts::SpeechPipelineStage> {
    matches!(
        kind,
        tongues_tts::NativeSpeechComponentKind::EndToEnd
            | tongues_tts::NativeSpeechComponentKind::VoiceConversion
            | tongues_tts::NativeSpeechComponentKind::Test
    )
    .then(|| {
        vec![
            tongues_tts::SpeechPipelineStage::AcousticModel,
            tongues_tts::SpeechPipelineStage::Vocoder,
        ]
    })
    .unwrap_or_default()
}

fn catalog_component_stage(
    entry: &tongues_tts::ModelCatalogEntry,
) -> tongues_tts::SpeechPipelineStage {
    match catalog_component_kind(entry) {
        "acoustic" => tongues_tts::SpeechPipelineStage::AcousticModel,
        "vocoder" => tongues_tts::SpeechPipelineStage::Vocoder,
        "voice" => tongues_tts::SpeechPipelineStage::Conditioner,
        _ => tongues_tts::SpeechPipelineStage::EndToEnd,
    }
}

fn catalog_component_spans(
    entry: &tongues_tts::ModelCatalogEntry,
) -> Vec<tongues_tts::SpeechPipelineStage> {
    (catalog_component_stage(entry) == tongues_tts::SpeechPipelineStage::EndToEnd)
        .then(|| {
            vec![
                tongues_tts::SpeechPipelineStage::AcousticModel,
                tongues_tts::SpeechPipelineStage::Vocoder,
            ]
        })
        .unwrap_or_default()
}

fn native_component_readiness(
    readiness: tongues_tts::NativeSpeechComponentReadiness,
) -> &'static str {
    match readiness {
        tongues_tts::NativeSpeechComponentReadiness::Runtime => "runtime",
        tongues_tts::NativeSpeechComponentReadiness::ImportOnly => "import_only",
        tongues_tts::NativeSpeechComponentReadiness::Experimental => "experimental",
    }
}

fn native_component_status(readiness: tongues_tts::NativeSpeechComponentReadiness) -> &'static str {
    match readiness {
        tongues_tts::NativeSpeechComponentReadiness::Runtime => "Artifact missing",
        tongues_tts::NativeSpeechComponentReadiness::ImportOnly => "Import-only",
        tongues_tts::NativeSpeechComponentReadiness::Experimental => "Experimental",
    }
}

fn speech_model_display_name<'a>(backend: &str, model: &'a str) -> &'a str {
    match backend {
        "burn" => "SpeedySpeech + HiFi-GAN",
        "fastpitch" => "FastPitch + HiFi-GAN",
        "glow" => "Glow-TTS + standardizer + MultiBand-MelGAN",
        "vits" => "VITS VCTK",
        "fairseq" => model,
        "yourtts" => "YourTTS Multilingual",
        "freevc" => "FreeVC24 Voice Conversion",
        "styletts2" if model == "styletts2-en-us" => "StyleTTS2 en-US",
        "styletts2" => model,
        "mock" => "Deterministic Mock",
        _ => model,
    }
}

fn speech_backend_installation_error(
    home: &FsPath,
    backend: &str,
    model: Option<&str>,
) -> Option<String> {
    let required = match backend {
        "burn" => vec![
            home.join(SPEEDY_RELATIVE_DIR).join("config.json"),
            home.join(SPEEDY_RELATIVE_DIR).join("model_file.pth"),
            home.join(HIFIGAN_RELATIVE_DIR).join("config.json"),
            home.join(HIFIGAN_RELATIVE_DIR).join("model_file.pth"),
        ],
        "fastpitch" => vec![
            home.join(FASTPITCH_RELATIVE_DIR).join("config.json"),
            home.join(FASTPITCH_RELATIVE_DIR).join("model_file.pth"),
            home.join(HIFIGAN_RELATIVE_DIR).join("config.json"),
            home.join(HIFIGAN_RELATIVE_DIR).join("model_file.pth"),
        ],
        "glow" => vec![
            home.join(GLOW_RELATIVE_DIR).join("config.json"),
            home.join(GLOW_RELATIVE_DIR).join("model_file.pth.tar"),
            home.join(MULTIBAND_RELATIVE_DIR).join("config.json"),
            home.join(MULTIBAND_RELATIVE_DIR).join("model_file.pth"),
            home.join(MULTIBAND_RELATIVE_DIR).join("scale_stats.npy"),
        ],
        "vits" => vec![
            home.join(VITS_RELATIVE_DIR).join("config.json"),
            home.join(VITS_RELATIVE_DIR).join("model_file.pth"),
            home.join(VITS_RELATIVE_DIR).join("speaker_ids.json"),
        ],
        "fairseq" => {
            let model = match speech_model_id(home, backend, model) {
                Ok(model) => model,
                Err(error) => return Some(error.to_string()),
            };
            let catalog = match tongues_tts::ModelCatalog::with_private_catalogs(
                &tongues_tts::private_catalog_paths_from_environment(),
            ) {
                Ok(catalog) => catalog,
                Err(error) => return Some(error.to_string()),
            };
            let Some(entry) = catalog.find(&model) else {
                return Some(format!("unknown Fairseq MMS catalog model `{model}`"));
            };
            entry
                .artifacts
                .iter()
                .map(|artifact| home.join(&artifact.install_path))
                .collect()
        }
        "yourtts" => [
            "config.json",
            "model_file.pth.tar",
            "speakers.json",
            "language_ids.json",
            "config_se.json",
            "model_se.pth.tar",
        ]
        .into_iter()
        .map(|file| home.join(YOURTTS_RELATIVE_DIR).join(file))
        .collect(),
        "freevc" => [
            "config.json",
            "model.pth",
            "WavLM-Large.pt",
            "speaker_encoder.pt",
        ]
        .into_iter()
        .map(|file| home.join(FREEVC_RELATIVE_DIR).join(file))
        .collect(),
        "onnx" => {
            let model = match speech_model_id(home, backend, model) {
                Ok(model) => model,
                Err(error) => return Some(error.to_string()),
            };
            let model_path = match onnx_voice_model_path(home, &model) {
                Ok(path) => path,
                Err(error) => return Some(error.to_string()),
            };
            vec![
                model_path.clone(),
                tongues_tts::voice_config_path(&model_path),
            ]
        }
        "styletts2" => {
            let model = match speech_model_id(home, backend, model) {
                Ok(model) => model,
                Err(error) => return Some(error.to_string()),
            };
            let model_dir = match styletts2_model_dir(home, Some(&model)) {
                Ok(path) => path,
                Err(error) => return Some(error.to_string()),
            };
            let paths = styletts2::StyleTts2OnnxPaths::from_model_dir(model_dir);
            vec![
                paths.diffusion,
                paths.style_encoder,
                paths.text_encoder,
                paths.decoder,
            ]
        }
        "mock" => Vec::new(),
        _ => return Some(format!("unknown speech backend `{backend}`")),
    };
    let missing = required
        .into_iter()
        .filter(|path| !path.is_file() && !path.is_dir())
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Some(format!("missing model artifacts: {}", missing.join(", ")));
    }
    match backend {
        "vits" => tongues_tts::SpeakerCatalog::from_file(
            home.join(VITS_SPEAKER_RELATIVE_PATH),
            VITS_SPEAKER_COUNT,
        )
        .err()
        .map(|error| error.to_string()),
        "onnx" => {
            let model = speech_model_id(home, backend, model).ok()?;
            let model_path = onnx_voice_model_path(home, &model).ok()?;
            tongues_tts::VoiceConfig::from_json_file(tongues_tts::voice_config_path(&model_path))
                .err()
                .map(|error| error.to_string())
        }
        _ => None,
    }
}

fn verify_catalog_backend(home: &FsPath, backend: &str, model: Option<&str>) -> anyhow::Result<()> {
    let ids = speech_path_catalog_ids(home, backend, model)?;
    if ids.is_empty() {
        return Ok(());
    }
    let paths = tongues_tts::private_catalog_paths_from_environment();
    let catalog = tongues_tts::ModelCatalog::with_private_catalogs(&paths)?;
    let cache = tongues_tts::default_model_cache(home)?;
    let store = tongues_tts::ModelStore::new(home, cache).with_offline(true);
    for id in ids {
        let entry = catalog
            .find(&id)
            .with_context(|| format!("speech model `{id}` is not in the licensed catalog"))?;
        store
            .require_cached_verification(entry)
            .with_context(|| format!("speech model `{id}` is not ready"))?;
    }
    Ok(())
}

async fn get_speech_speakers(Query(query): Query<SpeechSpeakersQuery>) -> impl IntoResponse {
    let backend = query.backend.as_deref().unwrap_or("vits").trim();
    Json(speech_speakers_response(&resolve_mortar_home(), backend))
}

async fn get_speech_runtime(State(state): State<AppState>) -> impl IntoResponse {
    let phase = state.speech_phase.load(Ordering::Acquire);
    match state.speech.try_lock() {
        Ok(service) => Json(service.snapshot(phase, &state.speech_admission, state.speech_device))
            .into_response(),
        Err(std::sync::TryLockError::WouldBlock) => {
            let (active, queued) = state.speech_admission.counts(true);
            Json(ResidentSpeechRuntimeResponse {
                state: speech_runtime_state(phase, false, false),
                device: state.speech_device.kind().into(),
                device_index: state.speech_device.index(),
                concurrency: "bounded-fifo",
                busy: true,
                capacity: state.speech_admission.capacity,
                active,
                queued,
                loaded: Vec::new(),
                failed: BTreeMap::new(),
            })
            .into_response()
        }
        Err(std::sync::TryLockError::Poisoned(_)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "resident speech registry lock is poisoned",
        )
            .into_response(),
    }
}

async fn reload_speech_runtime(State(state): State<AppState>) -> impl IntoResponse {
    let permit = match state.speech_admission.try_acquire() {
        Ok(permit) => permit,
        Err(_) => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                "Speech runtime is at capacity; retry reload after synthesis completes",
            )
                .into_response();
        }
    };
    let registry = Arc::clone(&state.speech);
    let phase = Arc::clone(&state.speech_phase);
    let admission = state.speech_admission.clone();
    let speech_device = state.speech_device;
    match tokio::task::spawn_blocking(move || {
        let _phase_reset = SpeechPhaseReset(Arc::clone(&phase));
        let mut service = registry
            .lock()
            .map_err(|_| "resident speech registry lock is poisoned")?;
        phase.store(SPEECH_PHASE_RELOADING, Ordering::Release);
        service.engines.clear();
        service.failures.clear();
        phase.store(SPEECH_PHASE_IDLE, Ordering::Release);
        drop(permit);
        Ok::<_, &'static str>(service.snapshot(SPEECH_PHASE_IDLE, &admission, speech_device))
    })
    .await
    {
        Ok(Ok(snapshot)) => Json(snapshot).into_response(),
        Ok(Err(error)) => (StatusCode::INTERNAL_SERVER_ERROR, error).into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("resident speech reload task failed: {error}"),
        )
            .into_response(),
    }
}

async fn unload_speech_runtime(
    State(state): State<AppState>,
    Json(request): Json<SpeechUnloadRequest>,
) -> impl IntoResponse {
    if request.pipeline.is_some() && (request.backend.is_some() || request.model.is_some()) {
        return (
            StatusCode::BAD_REQUEST,
            "pipeline cannot be combined with legacy backend or model selection",
        )
            .into_response();
    }
    let pipeline_prefix = if let Some(pipeline) = request.pipeline.as_ref() {
        let home = resolve_mortar_home();
        let composition = match resolve_registered_pipeline(&home, pipeline) {
            Ok(composition) => composition,
            Err(error) => return (StatusCode::BAD_REQUEST, error.to_string()).into_response(),
        };
        Some(format!("pipeline:{}:", composition.id))
    } else {
        let Some(backend) = request.backend.as_deref() else {
            return (StatusCode::BAD_REQUEST, "backend or pipeline is required").into_response();
        };
        if !RESIDENT_BACKEND_PROVIDERS
            .iter()
            .any(|provider| provider.id == backend)
        {
            return (StatusCode::BAD_REQUEST, "unknown speech backend").into_response();
        }
        None
    };
    let mut service = match state.speech.lock() {
        Ok(service) => service,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "resident speech registry lock is poisoned",
            )
                .into_response();
        }
    };
    let matches = |engine: &str| {
        if let Some(prefix) = pipeline_prefix.as_deref() {
            engine.starts_with(prefix)
        } else {
            speech_engine_matches_path(
                engine,
                request.backend.as_deref().unwrap_or_default(),
                request.model.as_deref().unwrap_or_default(),
            )
        }
    };
    service.engines.retain(|engine, _| !matches(engine));
    service.failures.retain(|engine, _| !matches(engine));
    Json(service.snapshot(
        SPEECH_PHASE_IDLE,
        &state.speech_admission,
        state.speech_device,
    ))
    .into_response()
}

fn speech_engine_matches_path(engine: &str, backend: &str, model: &str) -> bool {
    if engine.starts_with("pipeline:") {
        let (acoustic, vocoder, end_to_end, _) = speech_path_components(backend, model);
        return acoustic
            .as_deref()
            .is_none_or(|component| engine.contains(component))
            && vocoder
                .as_deref()
                .is_none_or(|component| engine.contains(component))
            && end_to_end
                .as_deref()
                .is_none_or(|component| engine.contains(component));
    }
    match backend {
        "onnx" => engine.starts_with(&format!("onnx-{model}-")),
        "styletts2" => engine.starts_with(&format!("styletts2-{model}-")),
        "mock" => engine.starts_with("mock-"),
        _ => engine.starts_with(&format!("{backend}-")),
    }
}

fn speech_speakers_response(mortar_home: &FsPath, backend: &str) -> SpeechSpeakersResponse {
    if backend == "yourtts" {
        let dir = mortar_home.join(YOURTTS_RELATIVE_DIR);
        let catalog = tongues_tts::VitsInferenceConfig::from_file(dir.join("config.json"))
            .and_then(|config| {
                tongues_tts::DVectorCatalog::from_file(
                    dir.join("speakers.json"),
                    config.network.d_vector_dim,
                    tongues_tts::COQUI_RESNET_SPEAKER_EMBEDDING_SPACE,
                )
            });
        return match catalog {
            Ok(catalog) => SpeechSpeakersResponse {
                backend: backend.into(),
                model: Some("yourtts-multilingual".into()),
                installed: true,
                requires_selection: false,
                speakers: catalog
                    .speaker_names()
                    .into_iter()
                    .enumerate()
                    .map(|(id, name)| SpeechSpeakerOption {
                        name: name.into(),
                        label: name.into(),
                        id: id as u32,
                    })
                    .collect(),
                error: None,
            },
            Err(error) => SpeechSpeakersResponse {
                backend: backend.into(),
                model: Some("yourtts-multilingual".into()),
                installed: false,
                requires_selection: false,
                speakers: Vec::new(),
                error: Some(format!(
                    "{error}. Run `cargo run --bin tongues -- models fetch yourtts-multilingual`."
                )),
            },
        };
    }
    if backend != "vits" {
        return SpeechSpeakersResponse {
            backend: backend.to_string(),
            model: None,
            installed: true,
            requires_selection: false,
            speakers: Vec::new(),
            error: None,
        };
    }

    let path = mortar_home.join(VITS_SPEAKER_RELATIVE_PATH);
    let catalog = match tongues_tts::SpeakerCatalog::from_file(&path, VITS_SPEAKER_COUNT) {
        Ok(catalog) => catalog,
        Err(error) => {
            return SpeechSpeakersResponse {
                backend: backend.to_string(),
                model: Some("vits-vctk".into()),
                installed: false,
                requires_selection: true,
                speakers: Vec::new(),
                error: Some(format!(
                    "{error}. Run `cargo run --bin tongues -- models fetch vits-vctk` or synthesize with the VITS backend once."
                )),
            };
        }
    };
    let speakers = catalog
        .entries()
        .into_iter()
        .map(|(name, id)| SpeechSpeakerOption {
            name: name.to_string(),
            label: name.trim().to_string(),
            id,
        })
        .collect();
    SpeechSpeakersResponse {
        backend: backend.to_string(),
        model: Some("vits-vctk".into()),
        installed: true,
        requires_selection: true,
        speakers,
        error: None,
    }
}

async fn get_styletts2_reference_audio(
    State(state): State<AppState>,
    Path(sample_id): Path<String>,
) -> impl IntoResponse {
    let path = match styletts2_sample_path(&state, &sample_id) {
        Ok(path) => path,
        Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
    };
    match std::fs::read(&path) {
        Ok(bytes) => Response::builder()
            .header("Content-Type", "audio/wav")
            .body(axum::body::Body::from(bytes))
            .unwrap(),
        Err(error) => (
            StatusCode::NOT_FOUND,
            format!("Failed to read {}: {error}", path.display()),
        )
            .into_response(),
    }
}

fn emotion_signatures_path(state: &AppState) -> PathBuf {
    state.workspace_root.join("emotion_signatures.json")
}

fn validate_speak_request(payload: &SpeakRequest) -> Result<(), String> {
    if payload.text.trim().is_empty() && payload.backend.as_deref() != Some("freevc") {
        return Err("text is required".into());
    }
    if let Some(quality) = payload.quality.as_deref() {
        if !quality.is_empty() && quality != "balanced" && quality != "fast" {
            return Err("quality must be `balanced` or `fast`".into());
        }
    }
    if payload.quiet.unwrap_or(false) && payload.verbose.unwrap_or(false) {
        return Err("quiet and verbose cannot both be enabled".into());
    }
    if payload.cpu.unwrap_or(false) && payload.cuda_device.is_some() {
        return Err("cpu and cuda_device cannot both be selected".into());
    }
    if let Some(backend) = payload.backend.as_deref() {
        if !backend.is_empty()
            && !RESIDENT_BACKEND_PROVIDERS
                .iter()
                .any(|provider| provider.id == backend)
        {
            return Err(format!(
                "backend must be one of: {}",
                RESIDENT_BACKEND_PROVIDERS
                    .iter()
                    .map(|provider| provider.id)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        speech_model_id(
            &resolve_mortar_home(),
            if backend.is_empty() { "burn" } else { backend },
            payload.model.as_deref(),
        )
        .map_err(|error| error.to_string())?;
    }
    if payload
        .speaker
        .as_deref()
        .is_some_and(|speaker| !speaker.is_empty())
        && payload.speaker_id.is_some()
    {
        return Err("speaker and speaker_id cannot both be set".into());
    }
    if payload
        .model_language
        .as_deref()
        .is_some_and(|language| !language.is_empty())
        && payload.language_id.is_some()
    {
        return Err("model_language and language_id cannot both be set".into());
    }
    if let Some(vector) = payload.emotion_vector.as_ref() {
        let emotion = payload
            .emotion
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "emotion is required when emotion_vector is provided".to_string())?;
        validate_emotion_vector(emotion, vector)?;
    }
    if let Some(diffusion_steps) = payload.diffusion_steps {
        if !(1..=64).contains(&diffusion_steps) {
            return Err("diffusion_steps must be between 1 and 64".into());
        }
    }
    validate_f32_range(
        "speaker_reference_strength",
        payload.speaker_reference_strength,
        0.0,
        1.0,
    )?;
    validate_f32_range(
        "style_reference_strength",
        payload.style_reference_strength,
        0.0,
        1.0,
    )?;
    validate_f32_range("style_alpha", payload.style_alpha, 0.0, 1.0)?;
    validate_f32_range("style_beta", payload.style_beta, 0.0, 1.0)?;
    validate_f64_range("embedding_scale", payload.embedding_scale, 0.0, 5.0)?;
    validate_f64_range("speed", payload.speed, 0.25, 3.0)?;
    validate_f32_range("noise_scale", payload.noise_scale, 0.0, 5.0)?;
    validate_f32_range(
        "duration_noise_scale",
        payload.duration_noise_scale,
        0.0,
        5.0,
    )?;
    if let Some(sample_rate_hz) = payload.sample_rate_hz {
        if !(8_000..=48_000).contains(&sample_rate_hz) {
            return Err("sample_rate_hz must be between 8000 and 48000".into());
        }
    }
    if let Some(max_tts_symbols) = payload.max_tts_symbols {
        if !(16..=2048).contains(&max_tts_symbols) {
            return Err("max_tts_symbols must be between 16 and 2048".into());
        }
    }
    Ok(())
}

fn validate_declared_speech_controls(
    payload: &SpeakRequest,
    controls: &[SpeechControlDiscovery],
) -> Result<(), String> {
    let declared = controls
        .iter()
        .map(|control| control.field)
        .collect::<std::collections::BTreeSet<_>>();
    let fields = [
        ("speed", payload.speed.is_some()),
        (
            "seed",
            payload.seed.is_some() || payload.style_seed.is_some(),
        ),
        (
            "model_language",
            payload.model_language.is_some() || payload.language_id.is_some(),
        ),
        ("voice_sample", payload.voice_sample.is_some()),
        ("style_sample", payload.style_sample.is_some()),
        ("source_audio", payload.source_audio.is_some()),
        ("target_audio", payload.target_audio.is_some()),
        (
            "emotion",
            payload.emotion.is_some()
                || payload.emotion_vector.is_some()
                || payload.emotion_strength.is_some(),
        ),
        ("quality", payload.quality.is_some()),
        ("diffusion_steps", payload.diffusion_steps.is_some()),
        (
            "speaker_reference_strength",
            payload.speaker_reference_strength.is_some(),
        ),
        (
            "style_reference_strength",
            payload.style_reference_strength.is_some(),
        ),
        ("style_alpha", payload.style_alpha.is_some()),
        ("style_beta", payload.style_beta.is_some()),
        ("embedding_scale", payload.embedding_scale.is_some()),
        ("noise_scale", payload.noise_scale.is_some()),
        (
            "duration_noise_scale",
            payload.duration_noise_scale.is_some(),
        ),
        ("pitch_scale", payload.pitch_scale.is_some()),
        ("pitch_shift", payload.pitch_shift.is_some()),
        ("pitch", payload.pitch.is_some()),
        ("energy_scale", payload.energy_scale.is_some()),
        ("energy_shift", payload.energy_shift.is_some()),
        ("energy", payload.energy.is_some()),
        ("durations", payload.durations.is_some()),
        ("sample_rate_hz", payload.sample_rate_hz.is_some()),
        ("max_tts_symbols", payload.max_tts_symbols.is_some()),
        (
            "no_tts_chunking",
            payload.no_tts_chunking.is_some_and(|value| value),
        ),
        ("timings", payload.timings.is_some_and(|value| value)),
    ];
    if let Some((field, _)) = fields
        .into_iter()
        .find(|(field, present)| *present && !declared.contains(field))
    {
        return Err(format!(
            "the selected synthesis path does not declare support for `{field}`"
        ));
    }
    if payload.quiet.is_some_and(|value| value) || payload.verbose.is_some_and(|value| value) {
        return Err(
            "quiet and verbose are CLI presentation controls, not synthesis controls".into(),
        );
    }
    Ok(())
}

fn validate_f32_range(name: &str, value: Option<f32>, min: f32, max: f32) -> Result<(), String> {
    let Some(value) = value else {
        return Ok(());
    };
    if !value.is_finite() || value < min || value > max {
        return Err(format!("{name} must be a finite value from {min} to {max}"));
    }
    Ok(())
}

fn validate_f64_range(name: &str, value: Option<f64>, min: f64, max: f64) -> Result<(), String> {
    let Some(value) = value else {
        return Ok(());
    };
    if !value.is_finite() || value < min || value > max {
        return Err(format!("{name} must be a finite value from {min} to {max}"));
    }
    Ok(())
}

fn styletts2_reference_dir(_state: &AppState) -> PathBuf {
    resolve_mortar_home()
        .join(STYLETTS2_REFERENCE_RELATIVE_DIR)
        .canonicalize()
        .unwrap_or_else(|_| resolve_mortar_home().join(STYLETTS2_REFERENCE_RELATIVE_DIR))
}

fn resolve_mortar_home() -> PathBuf {
    if let Some(home) = std::env::var_os("MORTAR_SEA_HOME") {
        let home = PathBuf::from(home);
        if !home.as_os_str().is_empty() {
            return home;
        }
    }
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("mortar-sea")
}

fn load_styletts2_samples(state: &AppState) -> Result<Vec<StyleTts2Sample>, String> {
    let reference_dir = styletts2_reference_dir(state);
    if !reference_dir.is_dir() {
        return Err(format!(
            "StyleTTS2 reference audio is not extracted at {}. Run `cargo run --bin tongues -- models fetch styletts2` or synthesize once to download it.",
            reference_dir.display()
        ));
    }

    let mut samples = Vec::new();
    collect_wav_samples(&reference_dir, &reference_dir, &mut samples)?;
    samples.sort_by(|a, b| a.label.cmp(&b.label).then_with(|| a.id.cmp(&b.id)));
    Ok(samples)
}

fn collect_wav_samples(
    reference_dir: &FsPath,
    dir: &FsPath,
    samples: &mut Vec<StyleTts2Sample>,
) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|error| format!("Failed to read {}: {error}", dir.display()))?;
    for entry in entries {
        let entry =
            entry.map_err(|error| format!("Failed to read entry in {}: {error}", dir.display()))?;
        let path = entry.path();
        let metadata = entry
            .metadata()
            .map_err(|error| format!("Failed to read metadata for {}: {error}", path.display()))?;
        if metadata.is_dir() {
            collect_wav_samples(reference_dir, &path, samples)?;
            continue;
        }
        if !metadata.is_file() || !is_wav_path(&path) {
            continue;
        }
        let relative = path
            .strip_prefix(reference_dir)
            .map_err(|error| format!("Failed to relativize {}: {error}", path.display()))?;
        let id = relative_path_id(relative)?;
        samples.push(StyleTts2Sample {
            label: sample_label(relative),
            audio_url: format!("/api/styletts2-reference-audio/{}", url_path_escape(&id)),
            path: path.display().to_string(),
            duration_ms: wav_duration_ms(&path),
            id,
        });
    }
    Ok(())
}

fn is_wav_path(path: &FsPath) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
}

fn relative_path_id(path: &FsPath) -> Result<String, String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => {
                let part = part
                    .to_str()
                    .ok_or_else(|| "sample path contains non-UTF-8 data".to_string())?;
                parts.push(part.to_string());
            }
            _ => return Err("sample path contains invalid components".into()),
        }
    }
    Ok(parts.join("/"))
}

fn sample_label(path: &FsPath) -> String {
    path.file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("sample")
        .replace(['_', '-'], " ")
}

fn url_path_escape(path: &str) -> String {
    path.split('/')
        .map(|part| {
            part.bytes()
                .flat_map(|byte| match byte {
                    b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'.' | b'-' | b'_' | b'~' => {
                        vec![byte as char]
                    }
                    _ => format!("%{byte:02X}").chars().collect(),
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn styletts2_sample_path(state: &AppState, sample_id: &str) -> Result<PathBuf, String> {
    if sample_id.trim().is_empty() {
        return Err("sample id is required".into());
    }
    let relative = FsPath::new(sample_id);
    for component in relative.components() {
        if !matches!(component, Component::Normal(_)) {
            return Err("sample id must be a relative path under reference_audio".into());
        }
    }
    if !is_wav_path(relative) {
        return Err("sample id must point to a WAV file".into());
    }

    let reference_dir = styletts2_reference_dir(state);
    let path = reference_dir.join(relative);
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("Unknown StyleTTS2 sample `{sample_id}`: {error}"))?;
    let canonical_reference_dir = reference_dir
        .canonicalize()
        .map_err(|error| format!("StyleTTS2 reference directory is unavailable: {error}"))?;
    if !canonical.starts_with(&canonical_reference_dir) || !canonical.is_file() {
        return Err("sample id is outside the StyleTTS2 reference directory".into());
    }
    Ok(canonical)
}

fn wav_duration_ms(path: &FsPath) -> Option<u64> {
    let reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    if spec.sample_rate == 0 || spec.channels == 0 {
        return None;
    }
    let samples_per_channel = reader.duration() / u32::from(spec.channels);
    Some((u64::from(samples_per_channel) * 1_000) / u64::from(spec.sample_rate))
}

fn load_or_create_emotion_signatures(state: &AppState) -> Result<EmotionsResponse, String> {
    let signature_path = emotion_signatures_path(state);
    if signature_path.exists() {
        return load_emotion_signatures(state, false);
    }

    let Some(style_vectors_path) = find_style_vectors_path(state) else {
        return Ok(EmotionsResponse {
            signature_path: signature_path.display().to_string(),
            style_vectors_path: None,
            emotions: Vec::new(),
            generated_from_style_vectors: false,
            error: Some("No emotion_signatures.json or style_vectors.jsonl found".into()),
        });
    };

    let signatures = build_signatures_from_style_vectors(&style_vectors_path)?;
    write_emotion_signatures_file(&signature_path, &signatures)?;
    load_emotion_signatures(state, true)
}

fn load_emotion_signatures(
    state: &AppState,
    generated_from_style_vectors: bool,
) -> Result<EmotionsResponse, String> {
    let signature_path = emotion_signatures_path(state);
    let content = std::fs::read_to_string(&signature_path)
        .map_err(|error| format!("Failed to read {}: {error}", signature_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|error| format!("Failed to parse {}: {error}", signature_path.display()))?;
    let obj = json
        .as_object()
        .ok_or_else(|| "emotion_signatures.json must contain a JSON object".to_string())?;

    let sample_counts = find_style_vectors_path(state)
        .as_ref()
        .map(|path| load_emotion_sample_counts(path))
        .transpose()?
        .unwrap_or_default();

    let mut emotions = Vec::new();
    for (name, value) in obj {
        let vector = value
            .get("vector")
            .and_then(|vector| vector.as_array())
            .ok_or_else(|| format!("Emotion `{name}` is missing a vector array"))?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|value| value as f32)
                    .ok_or_else(|| format!("Emotion `{name}` contains a non-numeric vector value"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        validate_emotion_vector(name, &vector)?;

        let stats_value = value.get("stats");
        let n_speakers = stats_value
            .and_then(|stats| stats.get("n_speakers"))
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize;
        let recommended = value.get("recommended_strength");
        emotions.push(EmotionSignature {
            name: name.clone(),
            kind: value
                .get("kind")
                .and_then(|value| value.as_str())
                .unwrap_or("styletts2.emotion_signature.v1")
                .to_string(),
            method: value
                .get("method")
                .and_then(|value| value.as_str())
                .unwrap_or("speaker-neutral-delta")
                .to_string(),
            dims: value
                .get("dims")
                .and_then(|value| value.as_u64())
                .unwrap_or(vector.len() as u64) as usize,
            vector,
            stats: EmotionStats {
                n_speakers,
                sample_count: sample_counts.get(name).copied().unwrap_or(0),
            },
            recommended_strength: RecommendedStrength {
                subtle: recommended
                    .and_then(|value| value.get("subtle"))
                    .and_then(|value| value.as_f64())
                    .unwrap_or(0.25) as f32,
                normal: recommended
                    .and_then(|value| value.get("normal"))
                    .and_then(|value| value.as_f64())
                    .unwrap_or(0.65) as f32,
                strong: recommended
                    .and_then(|value| value.get("strong"))
                    .and_then(|value| value.as_f64())
                    .unwrap_or(1.10) as f32,
            },
        });
    }
    emotions.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(EmotionsResponse {
        signature_path: signature_path.display().to_string(),
        style_vectors_path: find_style_vectors_path(state).map(|path| path.display().to_string()),
        emotions,
        generated_from_style_vectors,
        error: None,
    })
}

fn find_style_vectors_path(state: &AppState) -> Option<PathBuf> {
    [
        state.workspace_root.join("style_vectors.jsonl"),
        state
            .workspace_root
            .join("datasets")
            .join("emotions")
            .join("style_vectors.jsonl"),
    ]
    .into_iter()
    .find(|path| path.exists())
}

#[derive(Deserialize)]
struct StyleVectorEntry {
    emotion: String,
    speaker: String,
    vector: Vec<f32>,
}

fn build_signatures_from_style_vectors(
    path: &PathBuf,
) -> Result<BTreeMap<String, EmotionSignature>, String> {
    let file = std::fs::File::open(path)
        .map_err(|error| format!("Failed to open {}: {error}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut speaker_map: HashMap<String, HashMap<String, Vec<Vec<f32>>>> = HashMap::new();

    for (line_index, line) in reader.lines().enumerate() {
        let line = line.map_err(|error| {
            format!(
                "Failed to read line {} from {}: {error}",
                line_index + 1,
                path.display()
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: StyleVectorEntry = serde_json::from_str(&line).map_err(|error| {
            format!(
                "Failed to parse line {} from {}: {error}",
                line_index + 1,
                path.display()
            )
        })?;
        validate_emotion_vector(&entry.emotion, &entry.vector)?;
        speaker_map
            .entry(entry.speaker)
            .or_default()
            .entry(entry.emotion)
            .or_default()
            .push(entry.vector);
    }

    let mut emotion_deltas: BTreeMap<String, Vec<Vec<f32>>> = BTreeMap::new();
    for emotions in speaker_map.values() {
        let Some(neutrals) = emotions.get("neutral") else {
            continue;
        };
        let neutral_mean = mean_vector(neutrals);
        for (emotion, vectors) in emotions {
            if emotion == "neutral" {
                continue;
            }
            let emotion_mean = mean_vector(vectors);
            let delta = emotion_mean
                .iter()
                .zip(&neutral_mean)
                .map(|(emotion, neutral)| emotion - neutral)
                .collect::<Vec<_>>();
            emotion_deltas
                .entry(emotion.clone())
                .or_default()
                .push(delta);
        }
    }

    let sample_counts = load_emotion_sample_counts(path)?;
    let mut signatures = BTreeMap::new();
    for (emotion, deltas) in emotion_deltas {
        let speakers = deltas.len();
        let vector = mean_vector(&deltas);
        signatures.insert(
            emotion.clone(),
            EmotionSignature {
                name: emotion.clone(),
                kind: "styletts2.emotion_signature.v1".into(),
                method: "speaker-neutral-delta".into(),
                dims: STYLE_VECTOR_DIMS,
                vector,
                stats: EmotionStats {
                    n_speakers: speakers,
                    sample_count: sample_counts.get(&emotion).copied().unwrap_or(0),
                },
                recommended_strength: RecommendedStrength::default(),
            },
        );
    }

    Ok(signatures)
}

fn write_emotion_signatures_file(
    path: &PathBuf,
    signatures: &BTreeMap<String, EmotionSignature>,
) -> Result<(), String> {
    let mut map = serde_json::Map::new();
    for (emotion, signature) in signatures {
        map.insert(
            emotion.clone(),
            json!({
                "kind": signature.kind,
                "emotion": signature.name,
                "method": signature.method,
                "dims": signature.dims,
                "vector": signature.vector,
                "stats": {
                    "n_speakers": signature.stats.n_speakers,
                    "sample_count": signature.stats.sample_count,
                },
                "recommended_strength": {
                    "subtle": signature.recommended_strength.subtle,
                    "normal": signature.recommended_strength.normal,
                    "strong": signature.recommended_strength.strong,
                },
            }),
        );
    }

    let part_path = path.with_extension("json.part");
    let file = std::fs::File::create(&part_path)
        .map_err(|error| format!("Failed to create {}: {error}", part_path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &serde_json::Value::Object(map))
        .map_err(|error| format!("Failed to write {}: {error}", part_path.display()))?;
    writeln!(writer)
        .map_err(|error| format!("Failed to write {}: {error}", part_path.display()))?;
    writer
        .flush()
        .map_err(|error| format!("Failed to flush {}: {error}", part_path.display()))?;
    std::fs::rename(&part_path, path).map_err(|error| {
        format!(
            "Failed to rename {} to {}: {error}",
            part_path.display(),
            path.display()
        )
    })?;
    Ok(())
}

fn load_emotion_sample_counts(path: &PathBuf) -> Result<HashMap<String, usize>, String> {
    let file = std::fs::File::open(path)
        .map_err(|error| format!("Failed to open {}: {error}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut counts = HashMap::new();
    for line in reader.lines() {
        let line = line.map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)
            .map_err(|error| format!("Failed to parse {}: {error}", path.display()))?;
        if let Some(emotion) = value.get("emotion").and_then(|value| value.as_str()) {
            *counts.entry(emotion.to_string()).or_insert(0) += 1;
        }
    }
    Ok(counts)
}

fn mean_vector(vectors: &[Vec<f32>]) -> Vec<f32> {
    let mut mean = vec![0.0; STYLE_VECTOR_DIMS];
    for vector in vectors {
        for (index, value) in vector.iter().enumerate().take(STYLE_VECTOR_DIMS) {
            mean[index] += value;
        }
    }
    for value in &mut mean {
        *value /= vectors.len() as f32;
    }
    mean
}

fn validate_emotion_vector(name: &str, vector: &[f32]) -> Result<(), String> {
    if vector.len() != STYLE_VECTOR_DIMS {
        return Err(format!(
            "Emotion `{name}` vector must have {STYLE_VECTOR_DIMS} values, got {}",
            vector.len()
        ));
    }
    if !vector.iter().all(|value| value.is_finite()) {
        return Err(format!(
            "Emotion `{name}` vector contains non-finite values"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vits_speaker_enumeration_preserves_model_names_and_embedding_ids() {
        let mortar_home =
            std::env::temp_dir().join(format!("tongues-server-speakers-{}", uuid::Uuid::new_v4()));
        let speaker_path = mortar_home.join(VITS_SPEAKER_RELATIVE_PATH);
        std::fs::create_dir_all(speaker_path.parent().expect("speaker parent"))
            .expect("create speaker directory");
        std::fs::write(&speaker_path, r#"{"ED\n":0,"p225":1,"p330":90,"p376":108}"#)
            .expect("write speaker map");

        let response = speech_speakers_response(&mortar_home, "vits");

        assert!(response.installed);
        assert!(response.requires_selection);
        assert_eq!(response.model.as_deref(), Some("vits-vctk"));
        assert_eq!(
            response
                .speakers
                .iter()
                .find(|speaker| speaker.name == "p330"),
            Some(&SpeechSpeakerOption {
                name: "p330".into(),
                label: "p330".into(),
                id: 90,
            })
        );
        assert_eq!(response.speakers[0].label, "ED");

        std::fs::remove_dir_all(&mortar_home).expect("remove speaker fixture");
    }

    #[test]
    fn single_speaker_backends_return_an_empty_optional_catalog() {
        for backend in ["burn", "fastpitch", "glow", "onnx", "styletts2", "mock"] {
            let response = speech_speakers_response(FsPath::new("."), backend);
            assert_eq!(response.backend, backend);
            assert!(response.installed);
            assert!(!response.requires_selection);
            assert!(response.speakers.is_empty());
            assert!(response.error.is_none());
        }
    }

    #[test]
    fn backend_neutral_discovery_covers_native_component_end_to_end_style_and_onnx_paths() {
        let mortar_home =
            std::env::temp_dir().join(format!("tongues-server-contract-{}", uuid::Uuid::new_v4()));
        let speaker_path = mortar_home.join(VITS_SPEAKER_RELATIVE_PATH);
        std::fs::create_dir_all(speaker_path.parent().expect("speaker parent"))
            .expect("create VITS directory");
        std::fs::write(&speaker_path, r#"{"p225":0,"p330":90}"#).expect("write speakers");
        let onnx_model = onnx_voice_model_path(&mortar_home, DEFAULT_ONNX_VOICE_MODEL)
            .expect("default ONNX model path");
        std::fs::create_dir_all(onnx_model.parent().expect("ONNX parent"))
            .expect("create ONNX directory");
        std::fs::write(
            tongues_tts::voice_config_path(&onnx_model),
            r#"{
                "audio":{"sample_rate":22050},
                "phoneme_id_map":{"_":[0],"^":[1],"$":[2]," ":[3]},
                "num_speakers":1
            }"#,
        )
        .expect("write ONNX config");

        let discovered = ["burn", "fastpitch", "glow", "vits", "styletts2", "onnx"]
            .into_iter()
            .map(|backend| {
                speech_backend_capabilities(
                    &mortar_home,
                    backend,
                    None,
                    tongues_tts::ResolvedSpeechDevice::Cpu,
                    24_000,
                )
                .expect("backend capabilities")
            })
            .collect::<Vec<_>>();

        assert_eq!(
            discovered
                .iter()
                .map(|model| model.backend.as_str())
                .collect::<Vec<_>>(),
            ["burn", "fastpitch", "glow", "vits", "styletts2", "onnx"]
        );
        assert_eq!(discovered[0].model, "speedyspeech-ljspeech+hifigan-v2");
        assert_eq!(discovered[1].model, "fastpitch-ljspeech+hifigan-v2");
        assert!(discovered[1].pitch.scale);
        assert!(discovered[1].pitch.shift);
        assert!(discovered[1].pitch.explicit_values);
        assert!(discovered[1].durations);
        assert_eq!(
            discovered[2].model,
            "glow-tts-ljspeech+standardizer+multiband-melgan"
        );
        assert!(discovered[2].durations);
        assert_eq!(discovered[2].output.sample_rate_hz, 22_050);
        assert_eq!(
            discovered[3].speakers.values,
            tongues_tts::CapabilityValue::Listed(vec![
                tongues_tts::NamedCapability::new("p225", "p225").with_numeric_id(0),
                tongues_tts::NamedCapability::new("p330", "p330").with_numeric_id(90),
            ])
        );
        assert!(discovered[4].reference_audio.speaker);
        assert!(discovered[4].reference_audio.style);
        assert_eq!(discovered[4].styles.embedding_dimensions, Some(256));
        assert_eq!(discovered[5].output.sample_rate_hz, 22_050);
        let json =
            serde_json::to_value(&discovered[3]).expect("serialize VITS backend capabilities");
        assert_eq!(json["backend"], "vits");
        assert_eq!(json["speakers"]["values"]["support"], "listed");
        assert_eq!(json["speakers"]["values"]["values"][1]["id"], "p330");
        assert_eq!(json["speakers"]["values"]["values"][1]["numeric_id"], 90);

        std::fs::remove_dir_all(&mortar_home).expect("remove capability fixture");
    }

    #[test]
    fn glow_discovery_exposes_the_named_waveform_composition() {
        let capabilities = speech_backend_capabilities(
            FsPath::new("/tmp"),
            "glow",
            None,
            tongues_tts::ResolvedSpeechDevice::Cpu,
            24_000,
        )
        .expect("Glow-TTS capabilities");
        assert_eq!(
            capabilities.model,
            "glow-tts-ljspeech+standardizer+multiband-melgan"
        );
        assert!(capabilities.durations);
        assert_eq!(capabilities.output.sample_rate_hz, 22_050);

        let composition = tongues_tts::registered_speech_compositions()
            .into_iter()
            .find(|composition| composition.backend == "glow")
            .expect("Glow-TTS composition");
        assert_eq!(
            composition.pipeline.acoustic_model.as_deref(),
            Some("glow-tts-ljspeech")
        );
        assert_eq!(
            composition.pipeline.vocoder.as_deref(),
            Some("glow-standardized-multiband-melgan-ljspeech")
        );

        let compatibility = speech_vocoder_compatibility("glow");
        assert!(compatibility.iter().any(|edge| {
            edge.component_id == "glow-standardized-multiband-melgan-ljspeech"
                && edge.compatible
                && edge.reason.contains("named")
        }));
    }

    #[test]
    fn speech_studio_discovery_separates_paths_components_and_compatibility() {
        let mortar_home =
            std::env::temp_dir().join(format!("tongues-studio-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&mortar_home).expect("create discovery home");

        let discovery = speech_studio_discovery(
            &mortar_home,
            tongues_tts::ResolvedSpeechDevice::Cpu,
            &["fastpitch-cpu".into()],
        );

        assert!(discovery.error.is_none());
        assert_eq!(discovery.schema_version, 4);
        assert!(discovery.paths.iter().all(|path| path.complete));
        assert!(discovery.compositions.iter().any(|composition| {
            composition.backend == "fastpitch"
                && composition.pipeline.acoustic_model.as_deref() == Some("fastpitch-ljspeech")
                && composition.pipeline.vocoder.as_deref() == Some("hifigan-v2-ljspeech")
        }));
        assert!(discovery.compositions.iter().any(|composition| {
            composition.backend == "vits"
                && composition.pipeline.end_to_end.as_deref() == Some("vits-vctk")
                && composition.pipeline.vocoder.is_none()
        }));
        let fairseq_compositions = discovery
            .compositions
            .iter()
            .filter(|composition| composition.backend == "fairseq")
            .collect::<Vec<_>>();
        assert_eq!(fairseq_compositions.len(), 1_143);
        assert!(
            discovery.compatibility.len() < 2_000,
            "compatibility must remain sparse instead of expanding all projector pairs"
        );
        assert!(fairseq_compositions.iter().all(|composition| {
            composition.pipeline.end_to_end.as_deref() == Some(composition.model.as_str())
                && composition.pipeline.acoustic_model.is_none()
                && composition.pipeline.vocoder.is_none()
                && composition.pipeline.projector == format!("projector/{}", composition.model)
        }));
        let scripted = discovery
            .paths
            .iter()
            .find(|path| path.id == "fairseq-mms-vits-azj-script_cyrillic")
            .expect("script-qualified Fairseq path");
        assert_eq!(scripted.catalog[0].languages, ["azj"]);
        assert_eq!(scripted.catalog[0].script.as_deref(), Some("cyrillic"));
        assert_eq!(
            scripted.catalog[0].preprocessing,
            ["lowercase-and-filter-vocab"]
        );
        assert_eq!(scripted.catalog[0].license.expression, "CC-BY-NC-4.0");
        assert!(!scripted.runnable);
        assert!(scripted.install_command.is_some());
        assert!(discovery.compatibility.iter().any(|edge| {
            edge.from_component_id == "fastpitch-ljspeech"
                && edge.to_component_id == "hifigan-v2-ljspeech"
                && edge.compatible
        }));
        assert!(discovery.compatibility.iter().any(|edge| {
            edge.from_component_id == "fastpitch-ljspeech"
                && edge.to_component_id == "multiband-melgan-ljspeech"
                && !edge.compatible
                && edge.reason.contains("standardized")
        }));
        assert!(discovery.presets.iter().any(|preset| {
            preset.pipeline.projector == "projector/fastpitch-ljspeech"
                && preset.composition_id
                    == preset.pipeline.canonical_id().expect("preset pipeline id")
        }));
        assert!(
            discovery.verification_ids.is_empty(),
            "unavailable catalog entries must not be queued for deep verification"
        );
        let fastpitch = discovery
            .paths
            .iter()
            .find(|path| path.capabilities.backend == "fastpitch")
            .expect("FastPitch path");
        assert_eq!(
            fastpitch.acoustic_model.as_deref(),
            Some("fastpitch-ljspeech")
        );
        assert_eq!(fastpitch.vocoder.as_deref(), Some("hifigan-v2-ljspeech"));
        assert_eq!(fastpitch.cli_vocoder.as_deref(), Some("hifigan"));
        assert_eq!(
            fastpitch.missing_catalog_ids,
            ["fastpitch-ljspeech", "hifigan-v2-ljspeech"]
        );
        assert_eq!(fastpitch.load_state, "loaded");
        assert!(!fastpitch.verification_pending);
        assert!(!fastpitch.verified);
        assert_eq!(
            fastpitch.verification_status,
            tongues_tts::ModelVerificationStatus::Unavailable
        );
        assert!(fastpitch.controls.iter().any(|control| {
            control.field == "pitch" && control.kind == "number_array" && control.group == "expert"
        }));
        let cuda_controls = speech_control_discovery(
            "fastpitch",
            &fastpitch.capabilities,
            tongues_tts::ResolvedSpeechDevice::Cuda { index: 0 },
        );
        assert_eq!(
            cuda_controls
                .iter()
                .find(|control| control.field == "device")
                .and_then(|control| control.default.as_ref()),
            Some(&json!("cuda:0"))
        );
        assert!(
            fastpitch
                .compatible_vocoders
                .iter()
                .any(|vocoder| vocoder.component_id == "hifigan-v2-ljspeech" && vocoder.compatible)
        );
        assert!(fastpitch.compatible_vocoders.iter().any(|vocoder| {
            vocoder.component_id == "multiband-melgan-ljspeech"
                && !vocoder.compatible
                && !vocoder.reason.is_empty()
        }));

        let first_page = speech_studio_discovery_page(
            &mortar_home,
            tongues_tts::ResolvedSpeechDevice::Cpu,
            &[],
            0,
            32,
            &SpeechDiscoveryFilters::default(),
        );
        assert_eq!(first_page.page.cursor, 0);
        assert_eq!(first_page.page.returned, 32);
        assert_eq!(first_page.page.next_cursor, Some(32));
        assert!(
            first_page.paths.len() < 50,
            "the first response must not eagerly expand the entire catalog"
        );
        let second_page = speech_studio_discovery_page(
            &mortar_home,
            tongues_tts::ResolvedSpeechDevice::Cpu,
            &[],
            32,
            32,
            &SpeechDiscoveryFilters::default(),
        );
        assert_eq!(second_page.page.cursor, 32);
        assert_eq!(second_page.page.returned, 32);
        assert!(
            second_page.paths.len() <= 32,
            "continuation pages should contain only their catalog slice"
        );
        let filtered_page = speech_studio_discovery_page(
            &mortar_home,
            tongues_tts::ResolvedSpeechDevice::Cpu,
            &[],
            0,
            32,
            &SpeechDiscoveryFilters {
                search: "azj".into(),
                family: "mms".into(),
                license: "CC-BY-NC-4.0".into(),
                capability: "speech".into(),
                device: "cuda".into(),
                ..SpeechDiscoveryFilters::default()
            },
        );
        assert!(filtered_page.page.total >= 1);
        assert!(
            filtered_page
                .paths
                .iter()
                .any(|path| { path.id == "fairseq-mms-vits-azj-script_cyrillic" })
        );
        let fairseq = filtered_page
            .paths
            .iter()
            .find(|path| path.id == "fairseq-mms-vits-azj-script_cyrillic")
            .expect("filtered MMS path");
        let saved_lookup = speech_studio_discovery_page(
            &mortar_home,
            tongues_tts::ResolvedSpeechDevice::Cpu,
            &[],
            0,
            32,
            &SpeechDiscoveryFilters {
                model_ids: ["fairseq-mms-vits-tha".into()].into_iter().collect(),
                verification: "failed".into(),
                ..SpeechDiscoveryFilters::default()
            },
        );
        assert_eq!(saved_lookup.page.total, 1);
        assert!(
            saved_lookup
                .compositions
                .iter()
                .any(|composition| composition.model == "fairseq-mms-vits-tha")
        );
        assert!(
            fairseq
                .controls
                .iter()
                .all(|control| control.field != "model_language"),
            "a single-language checkpoint identity must not become a learned-language request control"
        );
        for component in [
            "speedy-speech",
            "fastpitch",
            "glow-tts",
            "sc-glowtts",
            "tacotron2",
            "tacotron2-ddc",
            "capacitron",
            "capacitron-ddc",
            "fastspeech",
            "fastspeech2",
            "delightfultts",
            "vits",
            "styletts2",
            "hifigan",
            "melgan",
            "multiband-melgan",
            "voice-ryan-medium",
            "voice-amy-medium",
            "voice-ljspeech-high",
            "deterministic-mock",
        ] {
            assert!(
                discovery
                    .components
                    .iter()
                    .any(|candidate| candidate.id == component),
                "missing component {component}"
            );
        }
        let mock = discovery
            .components
            .iter()
            .find(|component| component.id == "deterministic-mock")
            .expect("mock component");
        assert_eq!(mock.kind, "test");
        assert!(mock.statuses.iter().any(|status| status == "Test backend"));
        let fastpitch_component = discovery
            .components
            .iter()
            .find(|component| component.id == "fastpitch-ljspeech")
            .expect("FastPitch pipeline component");
        assert_eq!(
            fastpitch_component.stage,
            tongues_tts::SpeechPipelineStage::AcousticModel
        );
        assert!(
            fastpitch_component
                .control_fields
                .iter()
                .any(|field| field == "pitch")
        );
        assert_eq!(fastpitch_component.produces[0].kind, "mel_spectrogram");

        let speedy = discovery
            .paths
            .iter()
            .find(|path| path.capabilities.backend == "burn")
            .expect("SpeedySpeech path");
        assert_eq!(
            listed_capability_ids(&speedy.capabilities.varieties),
            vec!["en-US-GA"]
        );
        let vits = discovery
            .paths
            .iter()
            .find(|path| path.capabilities.backend == "vits")
            .expect("VITS path");
        assert_eq!(
            listed_capability_ids(&vits.capabilities.varieties),
            vec!["en-GB-RP"]
        );

        std::fs::remove_dir_all(&mortar_home).expect("remove discovery home");
    }

    #[test]
    fn component_pipeline_requests_normalize_to_legacy_loaders_and_unique_resident_keys() {
        let fastpitch = tongues_tts::registered_speech_compositions()
            .into_iter()
            .find(|composition| composition.backend == "fastpitch")
            .expect("FastPitch composition");
        let payload: SpeakRequest = serde_json::from_value(json!({
            "text": "Composable.",
            "variety": "en-US-GA",
            "pipeline": fastpitch.pipeline,
        }))
        .expect("component request");
        let normalized = normalize_speak_request(payload).expect("normalized pipeline");
        assert_eq!(normalized.backend.as_deref(), Some("fastpitch"));
        assert_eq!(
            normalized.model.as_deref(),
            Some("fastpitch-ljspeech+hifigan-v2")
        );
        let key = resident_engine_key(
            "fastpitch",
            tongues_tts::ResolvedSpeechDevice::Cpu,
            &normalized,
        )
        .expect("resident key");
        assert!(key.contains("acoustic=fastpitch-ljspeech"));
        assert!(key.contains("vocoder=hifigan-v2-ljspeech"));

        let ambiguous: SpeakRequest = serde_json::from_value(json!({
            "text": "Ambiguous.",
            "backend": "fastpitch",
            "pipeline": normalized.pipeline,
        }))
        .expect("ambiguous request shape");
        assert!(
            normalize_speak_request(ambiguous)
                .err()
                .expect("ambiguous request rejected")
                .contains("cannot be combined")
        );
    }

    #[test]
    fn duplex_projection_returns_replayable_branch_and_timeline_schema() {
        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|path| path.parent())
            .expect("repository root")
            .to_path_buf();
        let projection = build_duplex_projection(
            &workspace_root,
            DuplexProjectionRequest {
                fixture: Some("who-shot-john-f".into()),
                ..DuplexProjectionRequest::default()
            },
        )
        .expect("fixture projection");

        assert!(projection.replay_verified);
        assert!(!projection.timeline.is_empty());
        assert!(
            projection
                .timeline
                .iter()
                .any(|snapshot| !snapshot.predicted.is_empty())
        );
        assert!(
            projection
                .timeline
                .iter()
                .any(|snapshot| snapshot.branches.iter().any(|branch| branch.selected))
        );
    }

    #[test]
    fn duplex_projection_can_replay_a_saved_relative_journal() {
        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|path| path.parent())
            .expect("repository root")
            .to_path_buf();
        let generated = build_duplex_projection(
            &workspace_root,
            DuplexProjectionRequest {
                chunks: vec!["Who shot".into(), "John F. Kennedy?".into()],
                variety: Some("en-US-GA".into()),
                ..DuplexProjectionRequest::default()
            },
        )
        .expect("generated duplex projection");
        let relative_path = format!("runs/duplex-tests/{}.journal.json", uuid::Uuid::new_v4());
        let full_path = workspace_root.join(&relative_path);
        std::fs::create_dir_all(full_path.parent().expect("journal parent"))
            .expect("create journal directory");
        std::fs::write(
            &full_path,
            serde_json::to_vec_pretty(&generated.journal).expect("journal json"),
        )
        .expect("write journal");

        let replayed = build_duplex_projection(
            &workspace_root,
            DuplexProjectionRequest {
                journal_path: Some(relative_path.clone()),
                ..DuplexProjectionRequest::default()
            },
        )
        .expect("replayed duplex projection");

        assert_eq!(generated.timeline, replayed.timeline);
        assert_eq!(generated.transcript_events, replayed.transcript_events);
        std::fs::remove_file(full_path).expect("remove saved journal");
    }

    #[test]
    fn server_speech_frontend_uses_the_core_conformance_corpus() {
        let corpus =
            speaking::load_pronunciation_conformance_corpus().expect("pronunciation corpus");
        for case in corpus.cases {
            if case.careful_style {
                continue;
            }
            let analysis = speaking::analyze_pronunciation(&speaking::PhonemicizeRequest {
                text: case.input_text.clone(),
                variety: speaking::VarietyId(case.variety.clone()),
                style: None,
            })
            .expect("canonical pronunciation analysis");
            let plan = tongues_tts::utterance_plan_from_text(tongues_tts::SpeechRequest {
                text: case.input_text.clone(),
                variety: case.variety.clone(),
            })
            .expect("server speech plan");
            assert_eq!(plan.variety, analysis.plan.variety, "{}", case.id);
            assert_eq!(
                plan.intended_text, analysis.plan.intended_text,
                "{}",
                case.id
            );
            assert_eq!(
                plan.intended_morphemes, analysis.plan.intended_morphemes,
                "{}",
                case.id
            );
            assert_eq!(
                plan.intended_phonemes, analysis.plan.intended_phonemes,
                "{}",
                case.id
            );
            assert_eq!(
                plan.target_phones, analysis.plan.target_phones,
                "{}",
                case.id
            );
            assert_eq!(
                plan.target_syllables, analysis.plan.target_syllables,
                "{}",
                case.id
            );
            assert_eq!(plan.boundaries, analysis.plan.boundaries, "{}", case.id);
            assert_eq!(
                plan.target_prosody, analysis.plan.target_prosody,
                "{}",
                case.id
            );
            assert_eq!(
                plan.target_acoustics, analysis.plan.target_acoustics,
                "{}",
                case.id
            );
            assert_eq!(plan.speaker, analysis.plan.speaker, "{}", case.id);
            assert_eq!(
                plan.speaker_reference, analysis.plan.speaker_reference,
                "{}",
                case.id
            );
            assert_eq!(plan.style, analysis.plan.style, "{}", case.id);
        }
    }

    fn listed_capability_ids(value: &tongues_tts::CapabilityValue) -> Vec<&str> {
        match value {
            tongues_tts::CapabilityValue::Listed(values) => {
                values.iter().map(|value| value.id.as_str()).collect()
            }
            _ => Vec::new(),
        }
    }

    #[test]
    fn unified_server_request_preserves_style_reference_seed_speed_and_device_controls() {
        let payload: SpeakRequest = serde_json::from_value(json!({
            "text": "A styled request.",
            "backend": "styletts2",
            "variety": "en-US",
            "model_language": "en",
            "emotion": "amused",
            "emotion_strength": 0.65,
            "speaker_reference_strength": 0.8,
            "style_reference_strength": 0.7,
            "diffusion_steps": 2,
            "embedding_scale": 1.2,
            "style_seed": 27,
            "speed": 1.1,
            "max_tts_symbols": 96,
            "no_tts_chunking": true
        }))
        .expect("style request");
        let context = ResidentSynthesisContext {
            voice_reference: Some(PathBuf::from("/tmp/voice.wav")),
            style_reference: Some(PathBuf::from("/tmp/style.wav")),
            source_reference: None,
            emotion_vector: Some(vec![0.1; STYLE_VECTOR_DIMS]),
        };

        let request = unified_synthesis_request(
            &payload,
            &context,
            tongues_tts::ResolvedSpeechDevice::Cuda { index: 2 },
        );

        assert_eq!(request.seed, Some(27));
        assert_eq!(request.speed, 1.1);
        assert_eq!(
            request.model_language,
            Some(tongues_tts::LanguageSelection::Named("en".into()))
        );
        assert_eq!(
            request.device,
            tongues_tts::SpeechDeviceRequest::Cuda { index: 2 }
        );
        assert_eq!(
            request.reference_audio.speaker.as_deref(),
            Some("/tmp/voice.wav")
        );
        assert_eq!(
            request.reference_audio.style.as_deref(),
            Some("/tmp/style.wav")
        );
        assert_eq!(request.max_chunk_symbols, Some(96));
        assert!(!request.chunking);
        let style = request.style.expect("style controls");
        assert_eq!(style.name.as_deref(), Some("amused"));
        assert!(style.embedding_is_delta);
        assert!((style.speaker_blend.unwrap() - 0.2).abs() < f32::EPSILON);
        assert!((style.style_blend.unwrap() - 0.3).abs() < f32::EPSILON);
        assert_eq!(style.diffusion_steps, Some(2));
    }

    #[test]
    fn freevc_uses_unified_source_and_target_references() {
        let payload: SpeakRequest = serde_json::from_value(json!({
            "text": "",
            "backend": "freevc",
            "model": "freevc24-vctk",
            "source_audio": "fixtures/source.wav",
            "target_audio": "fixtures/target.wav",
            "noise_scale": 0.8,
            "seed": 38
        }))
        .expect("FreeVC request");
        let context = ResidentSynthesisContext {
            voice_reference: Some(PathBuf::from("/workspace/fixtures/target.wav")),
            style_reference: None,
            source_reference: Some(PathBuf::from("/workspace/fixtures/source.wav")),
            emotion_vector: None,
        };

        let request =
            unified_synthesis_request(&payload, &context, tongues_tts::ResolvedSpeechDevice::Cpu);

        assert!(request.text.is_empty());
        assert_eq!(
            request.reference_audio.source.as_deref(),
            Some("/workspace/fixtures/source.wav")
        );
        assert_eq!(
            request.reference_audio.speaker.as_deref(),
            Some("/workspace/fixtures/target.wav")
        );
        assert_eq!(request.noise_scale, Some(0.8));
        assert_eq!(request.seed, Some(38));
    }

    #[test]
    fn freevc_discovery_requires_both_audio_roles() {
        let capabilities = speech_backend_capabilities(
            FsPath::new("/tmp"),
            "freevc",
            Some("freevc24-vctk"),
            tongues_tts::ResolvedSpeechDevice::Cpu,
            24_000,
        )
        .expect("FreeVC capabilities");

        assert_eq!(
            capabilities.family,
            tongues_tts::SpeechModelFamily::VoiceConversion
        );
        assert!(capabilities.reference_audio.source_required);
        assert!(capabilities.reference_audio.speaker_required);
        assert_eq!(
            capabilities.devices,
            vec![tongues_tts::SpeechDeviceRequest::Cpu]
        );
        assert!(
            registered_speech_compositions_at(FsPath::new("/tmp"))
                .iter()
                .any(|composition| {
                    composition.backend == "freevc" && composition.model == "freevc24-vctk"
                })
        );
    }

    #[test]
    fn discovered_capabilities_reject_language_style_and_speaker_mismatches_pre_inference() {
        let mortar_home =
            std::env::temp_dir().join(format!("tongues-server-preflight-{}", uuid::Uuid::new_v4()));
        let speaker_path = mortar_home.join(VITS_SPEAKER_RELATIVE_PATH);
        std::fs::create_dir_all(speaker_path.parent().expect("speaker parent"))
            .expect("create VITS directory");
        std::fs::write(&speaker_path, r#"{"p225":0,"p330":90}"#).expect("write speakers");
        let capabilities = speech_backend_capabilities(
            &mortar_home,
            "vits",
            None,
            tongues_tts::ResolvedSpeechDevice::Cpu,
            24_000,
        )
        .expect("VITS capabilities");

        let mut request = tongues_tts::UnifiedSynthesisRequest::new("Hello.", "fr-FR-Standard");
        request.device = tongues_tts::SpeechDeviceRequest::Cpu;
        request.speaker = Some(tongues_tts::SpeakerSelection::Named("p330".into()));
        assert!(matches!(
            capabilities.validate(&request),
            Err(tongues_tts::SynthesisContractError::UnsupportedValue {
                feature: "variety",
                ..
            })
        ));

        request.variety = "en-GB-RP".into();
        request.speaker = Some(tongues_tts::SpeakerSelection::Named("not-a-speaker".into()));
        assert!(matches!(
            capabilities.validate(&request),
            Err(tongues_tts::SynthesisContractError::UnsupportedValue {
                feature: "speaker",
                ..
            })
        ));

        request.speaker = Some(tongues_tts::SpeakerSelection::Named("p330".into()));
        request.style = Some(tongues_tts::StyleSelection {
            name: Some("amused".into()),
            embedding: None,
            embedding_is_delta: false,
            strength: 1.0,
            speaker_blend: None,
            style_blend: None,
            diffusion_steps: None,
            embedding_scale: None,
        });
        assert!(matches!(
            capabilities.validate(&request),
            Err(tongues_tts::SynthesisContractError::UnsupportedFeature {
                feature: "style",
                ..
            })
        ));

        std::fs::remove_dir_all(&mortar_home).expect("remove preflight fixture");
    }

    #[test]
    fn resident_runtime_reports_bounded_queueing_and_loaded_state() {
        let mut service = ResidentSpeechService::default();
        service
            .failures
            .insert("vits-cuda".into(), "fixture load failure".into());
        let admission = SpeechAdmission::new(2);

        let response = service.snapshot(
            SPEECH_PHASE_IDLE,
            &admission,
            tongues_tts::ResolvedSpeechDevice::Cuda { index: 2 },
        );

        assert_eq!(response.state, "failed");
        assert_eq!(response.device, "cuda");
        assert_eq!(response.device_index, Some(2));
        assert_eq!(response.concurrency, "bounded-fifo");
        assert_eq!(response.capacity, 2);
        assert_eq!(response.active, 0);
        assert_eq!(response.queued, 0);
        assert!(response.loaded.is_empty());
        assert_eq!(
            response.failed.get("vits-cuda").map(String::as_str),
            Some("fixture load failure")
        );
    }

    #[test]
    fn resident_engine_keys_distinguish_explicit_cuda_indices() {
        let payload: SpeakRequest =
            serde_json::from_value(json!({"text": "Indexed resident speech."}))
                .expect("minimal speech request");

        assert_eq!(
            resident_engine_key(
                "vits",
                tongues_tts::ResolvedSpeechDevice::Cuda { index: 0 },
                &payload,
            )
            .unwrap(),
            "vits-cuda"
        );
        assert_eq!(
            resident_engine_key(
                "vits",
                tongues_tts::ResolvedSpeechDevice::Cuda { index: 3 },
                &payload,
            )
            .unwrap(),
            "vits-cuda-3"
        );

        let ljspeech: SpeakRequest = serde_json::from_value(json!({
            "text": "LJSpeech resident speech.",
            "backend": "onnx",
            "model": "voice-ljspeech-high"
        }))
        .expect("LJSpeech request");
        let ryan: SpeakRequest = serde_json::from_value(json!({
            "text": "Ryan resident speech.",
            "backend": "onnx",
            "model": "voice-ryan-medium"
        }))
        .expect("Ryan request");
        assert_eq!(
            resident_engine_key("onnx", tongues_tts::ResolvedSpeechDevice::Cpu, &ljspeech,)
                .unwrap(),
            "onnx-voice-ljspeech-high-cpu"
        );
        assert_eq!(
            resident_engine_key("onnx", tongues_tts::ResolvedSpeechDevice::Cpu, &ryan).unwrap(),
            "onnx-voice-ryan-medium-cpu"
        );
        let style_alias: SpeakRequest = serde_json::from_value(json!({
            "text": "Style alias resident speech.",
            "backend": "styletts2",
            "model": "styletts2"
        }))
        .expect("Style request");
        assert_eq!(
            resident_engine_key(
                "styletts2",
                tongues_tts::ResolvedSpeechDevice::Cpu,
                &style_alias
            )
            .unwrap(),
            "styletts2-styletts2-en-us-cpu"
        );
    }

    #[test]
    fn speech_model_selection_rejects_cross_backend_and_unknown_models() {
        let home = FsPath::new(".");
        assert_eq!(
            speech_model_id(home, "onnx", Some("voice-amy-medium")).unwrap(),
            "voice-amy-medium"
        );
        assert_eq!(
            speech_model_id(home, "fairseq", Some("tts_models/eng/fairseq/vits")).unwrap(),
            "fairseq-mms-vits-eng"
        );
        assert_eq!(
            speech_model_id(home, "styletts2", Some("styletts2")).unwrap(),
            "styletts2-en-us"
        );
        assert!(speech_model_id(home, "onnx", Some("voice-unknown")).is_err());
        assert!(speech_model_id(home, "vits", Some("voice-amy-medium")).is_err());
        assert!(speech_model_id(home, "fairseq", Some("vits-vctk")).is_err());
        assert!(speech_model_id(home, "styletts2", Some("voice-amy-medium")).is_err());
    }

    #[test]
    fn resident_request_rejects_conflicting_cpu_and_cuda_selection() {
        let payload: SpeakRequest = serde_json::from_value(json!({
            "text": "Conflicting resident speech.",
            "cpu": true,
            "cuda_device": 1,
        }))
        .expect("speech request");

        assert_eq!(
            validate_speak_request(&payload),
            Err("cpu and cuda_device cannot both be selected".into())
        );
    }

    #[test]
    fn resident_admission_rejects_requests_above_capacity() {
        let admission = SpeechAdmission::new(2);
        let first = admission.try_acquire().expect("first request admitted");
        let second = admission.try_acquire().expect("second request queued");

        assert!(admission.try_acquire().is_err());
        assert_eq!(admission.counts(true), (1, 1));

        drop(second);
        assert!(admission.try_acquire().is_ok());
        drop(first);
    }

    #[test]
    fn runtime_phase_names_cover_loading_ready_busy_and_reload() {
        assert_eq!(
            speech_runtime_state(SPEECH_PHASE_LOADING, false, false),
            "loading"
        );
        assert_eq!(
            speech_runtime_state(SPEECH_PHASE_SYNTHESIZING, true, false),
            "busy"
        );
        assert_eq!(
            speech_runtime_state(SPEECH_PHASE_RELOADING, true, false),
            "reloading"
        );
        assert_eq!(
            speech_runtime_state(SPEECH_PHASE_IDLE, true, false),
            "ready"
        );
        assert_eq!(
            speech_runtime_state(SPEECH_PHASE_IDLE, true, true),
            "failed",
            "a partially warm runtime must not hide model load failures"
        );
    }

    #[test]
    fn resident_wav_encoding_is_valid_and_finite() {
        let bytes = encode_wav_mono_f32(22_050, &[0.0, -0.5, 0.5]).expect("WAV");
        let mut reader =
            hound::WavReader::new(std::io::Cursor::new(bytes)).expect("read encoded WAV");
        assert_eq!(reader.spec().sample_rate, 22_050);
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.duration(), 3);
        assert_eq!(
            reader
                .samples::<i16>()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![0, -16384, 16384]
        );
    }

    #[test]
    fn server_variety_options_are_projected_from_the_linguistic_data_registry() {
        let registered = speaking::builtin_varieties();
        let options = linguistic_variety_options(false);
        assert_eq!(options.len(), registered.len());
        for (option, variety) in options.iter().zip(registered) {
            assert_eq!(option.value, variety.id.0);
            assert_eq!(option.label, variety.name);
        }

        let registered = speaking::builtin_languages();
        let options = linguistic_language_options();
        assert_eq!(options.len(), registered.len());
        for (option, language) in options.iter().zip(registered) {
            assert_eq!(option.value, language.iso_639.unwrap_or(language.id.0));
            assert_eq!(option.label, language.name);
        }

        let metadata = linguistic_variety_metadata();
        assert_eq!(metadata.len(), speaking::builtin_varieties().len());
        let french = metadata
            .iter()
            .find(|variety| variety.value == "fr-FR-Standard")
            .expect("French metadata");
        assert_eq!(french.language, "fr");
        assert_eq!(french.language_tag.as_deref(), Some("fr-FR"));
        assert_eq!(
            french.pronunciation_fallback,
            PronunciationFallbackMetadata::Mapped {
                provider: "wiktionary",
                language: "fra".into(),
            }
        );
    }

    #[test]
    fn is_guessed_pronunciation_warning_identifies_guessed_kinds() {
        use speaking::{PronunciationWarning, PronunciationWarningKind};

        let guessed_kinds = [
            PronunciationWarningKind::GuessedWord,
            PronunciationWarningKind::MixedAlphaNumeric,
            PronunciationWarningKind::UnknownPronunciation,
        ];
        let non_guessed_kinds = [
            PronunciationWarningKind::AcronymExpanded,
            PronunciationWarningKind::WeakFormApplied,
        ];

        for kind in guessed_kinds {
            let warning = PronunciationWarning {
                token: "test".into(),
                kind,
                message: "test message".into(),
            };
            assert!(
                is_guessed_pronunciation_warning(&warning),
                "{kind:?} should be identified as a guessed pronunciation"
            );
        }
        for kind in non_guessed_kinds {
            let warning = PronunciationWarning {
                token: "test".into(),
                kind,
                message: "test message".into(),
            };
            assert!(
                !is_guessed_pronunciation_warning(&warning),
                "{kind:?} should not be identified as a guessed pronunciation"
            );
        }
    }

    #[test]
    fn pronunciation_warnings_serialize_with_stable_kind_names() {
        use speaking::{PronunciationWarning, PronunciationWarningKind};

        let warnings = vec![PronunciationWarning {
            token: "xyz123".into(),
            kind: PronunciationWarningKind::GuessedWord,
            message: "guessed pronunciation for xyz123".into(),
        }];
        let diagnostics = serde_json::json!({
            "stages": [],
            "pronunciation_warnings": &warnings,
            "pronunciation_plan": serde_json::Value::Null,
        });
        let serialized_warnings = &diagnostics["pronunciation_warnings"];
        assert_eq!(serialized_warnings.as_array().unwrap().len(), 1);
        assert_eq!(serialized_warnings[0]["token"], "xyz123");
        assert_eq!(serialized_warnings[0]["kind"], "guessed_word");
        assert_eq!(
            serialized_warnings[0]["message"],
            "guessed pronunciation for xyz123"
        );
    }

    #[test]
    fn remote_bind_requires_an_explicit_opt_in() {
        assert!(bind_address_allowed(
            "127.0.0.1:3000".parse().expect("loopback HTTP"),
            false
        ));
        assert!(bind_address_allowed(
            "[::1]:8443".parse().expect("loopback HTTPS"),
            false
        ));
        assert!(!bind_address_allowed(
            "0.0.0.0:3000".parse().expect("wildcard HTTP"),
            false
        ));
        assert!(bind_address_allowed(
            "0.0.0.0:3000".parse().expect("wildcard HTTP"),
            true
        ));
    }

    #[test]
    fn same_origin_policy_rejects_cross_origin_mutations() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(header::HOST, "localhost:3000".parse().expect("host header"));
        headers.insert(
            header::ORIGIN,
            "http://localhost:3000".parse().expect("origin header"),
        );
        assert!(validate_same_origin(&headers).is_ok());
        headers.insert(
            header::ORIGIN,
            "https://evil.example".parse().expect("evil origin"),
        );
        assert!(validate_same_origin(&headers).is_err());
    }

    #[test]
    fn artifact_paths_reject_workspace_escape_and_sensitive_files() {
        let workspace = test_workspace("artifacts");
        let runs_dir = workspace.join("runs");
        std::fs::create_dir_all(&runs_dir).expect("create runs");
        std::fs::write(runs_dir.join("visible.txt"), "ok").expect("write visible artifact");
        std::fs::write(runs_dir.join(".env"), "secret").expect("write hidden file");
        std::fs::write(runs_dir.join("secret.key"), "secret").expect("write key file");
        std::fs::write(runs_dir.join("config.json"), "{}").expect("write config file");

        assert!(is_visible_artifact_path(
            FsPath::new("runs/visible.txt"),
            false
        ));
        assert!(!is_visible_artifact_path(FsPath::new("runs/.env"), false));
        assert!(!is_visible_artifact_path(
            FsPath::new("runs/secret.key"),
            false
        ));
        assert!(!is_visible_artifact_path(
            FsPath::new("runs/config.json"),
            false
        ));
        assert!(
            validate_artifact_relative_path(&workspace, FsPath::new("runs/visible.txt")).is_ok()
        );
        assert!(validate_artifact_relative_path(&workspace, FsPath::new("README.md")).is_err());

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let outside = workspace
                .parent()
                .expect("workspace parent")
                .join(format!("outside-{}.txt", uuid::Uuid::new_v4()));
            std::fs::write(&outside, "escape").expect("write outside file");
            symlink(&outside, runs_dir.join("escape.txt")).expect("create escape symlink");
            assert!(
                resolve_existing_artifact_path(&workspace, FsPath::new("runs/escape.txt")).is_err()
            );
            std::fs::remove_file(outside).expect("remove outside file");
        }

        std::fs::remove_dir_all(workspace).expect("remove artifact workspace");
    }

    #[test]
    fn job_validation_rejects_unknown_flags_and_workspace_paths() {
        let workspace = test_workspace("jobs");
        std::fs::create_dir_all(workspace.join("configs/g2p2g")).expect("create config dir");
        std::fs::create_dir_all(workspace.join("datasets/g2p2g")).expect("create data dir");
        std::fs::create_dir_all(workspace.join("models/g2p2g/openepd-v0"))
            .expect("create model dir");
        std::fs::create_dir_all(workspace.join("runs")).expect("create runs dir");
        std::fs::write(
            workspace.join("configs/g2p2g/default.toml"),
            "mode = 'test'\n",
        )
        .expect("write config");
        std::fs::write(workspace.join("Cargo.toml"), "[package]\nname='fixture'\n")
            .expect("write cargo file");

        let valid = StartJobRequest {
            label: None,
            command: "cargo".into(),
            args: vec![
                "run".into(),
                "--bin".into(),
                "tongues".into(),
                "--".into(),
                "g2p2g".into(),
                "prepare".into(),
                "--config".into(),
                "configs/g2p2g/default.toml".into(),
                "--out".into(),
                "datasets/g2p2g/openepd-v0".into(),
            ],
        };
        assert!(validate_job_request(&workspace, &valid).is_ok());

        let unknown_flag = StartJobRequest {
            label: None,
            command: "cargo".into(),
            args: vec![
                "run".into(),
                "--bin".into(),
                "tongues".into(),
                "--".into(),
                "g2p2g".into(),
                "prepare".into(),
                "--manifest-path".into(),
                "Cargo.toml".into(),
            ],
        };
        assert!(validate_job_request(&workspace, &unknown_flag).is_err());

        let bad_config_path = StartJobRequest {
            label: None,
            command: "cargo".into(),
            args: vec![
                "run".into(),
                "--bin".into(),
                "tongues".into(),
                "--".into(),
                "g2p2g".into(),
                "prepare".into(),
                "--config".into(),
                "Cargo.toml".into(),
            ],
        };
        assert!(validate_job_request(&workspace, &bad_config_path).is_err());

        let bad_positional_path = StartJobRequest {
            label: None,
            command: "cargo".into(),
            args: vec![
                "run".into(),
                "--bin".into(),
                "tongues".into(),
                "--".into(),
                "styletts2".into(),
                "emotion-signatures".into(),
                "Cargo.toml".into(),
                "--out".into(),
                "runs/emotions.json".into(),
            ],
        };
        assert!(validate_job_request(&workspace, &bad_positional_path).is_err());

        std::fs::remove_dir_all(workspace).expect("remove job workspace");
    }

    fn test_workspace(label: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("tongues-server-{label}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).expect("create temp workspace");
        path
    }
}
