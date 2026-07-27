use axum::extract::WebSocketUpgrade;
use axum::extract::ws::{Message, WebSocket};
use axum::http::{HeaderMap, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use speaking::{
    LanguageIdentifier, LanguageRoute, LanguageRouter, LanguageSelectionMode, LanguageSwitchPolicy,
    UnsupportedLanguagePolicy, WhisperLanguageIdentifier,
};
use tongues_audio::{
    AudioSourceDescriptor, AudioSourceKind, CleanupPipeline, CleanupStageConfig, PushedAudioChunk,
    SegmentationConfig, SegmentationEvent, UtteranceSegmenter, VadBackendKind, VadPipelineEvent,
    VadSegmentationPipeline, bounded_audio_input, create_vad_backend, rms,
};

const BROWSER_AUDIO_SCHEMA_VERSION: u16 = 1;
const BROWSER_AUDIO_QUEUE_CHUNKS: usize = 8;
const BROWSER_AUDIO_HEADER_BYTES: usize = 16;
const MAX_BROWSER_AUDIO_CHUNK_BYTES: usize = 256 * 1024;

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BrowserLanguageRouting {
    mode: LanguageSelectionMode,
    #[serde(default)]
    switching: LanguageSwitchPolicy,
    unsupported: UnsupportedLanguagePolicy,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
enum ClientControl {
    Open {
        schema_version: u16,
        sample_rate_hz: u32,
        channels: u16,
        #[serde(default = "default_browser_vad")]
        vad_backend: VadBackendKind,
        #[serde(default)]
        segmentation: SegmentationConfig,
        #[serde(default)]
        cleanup: Vec<CleanupStageConfig>,
        #[serde(default)]
        language_routing: Option<BrowserLanguageRouting>,
    },
    End,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case", tag = "type")]
enum ProbeEvent {
    Ready {
        schema_version: u16,
        sample_rate_hz: u32,
        channels: u16,
        queue_capacity_chunks: usize,
        vad_backend: VadBackendKind,
        segmentation: SegmentationConfig,
        cleanup: Vec<CleanupStageConfig>,
        language_routing: Option<BrowserLanguageRouting>,
    },
    CleanupCompared {
        chunk_sequence: u64,
        raw_rms: f32,
        processed_rms: f32,
        raw_peak: f32,
        processed_peak: f32,
        algorithmic_latency_frames: usize,
        stages: Vec<tongues_audio::CleanupStageTrace>,
    },
    Level {
        chunk_sequence: u64,
        start_frame: Option<u64>,
        frame_count: usize,
        rms: f32,
        peak: f32,
        speech_probability: f32,
        is_speech: bool,
    },
    SegmentOpened {
        segment_id: String,
        pre_roll_frames: usize,
    },
    SpeechEnded {
        segment_id: String,
        endpoint_latency_ms: u64,
    },
    SegmentFinal {
        segment_id: String,
        accepted: bool,
        reason: tongues_audio::SegmentCloseReason,
        speech_duration_ms: u64,
        total_duration_ms: u64,
    },
    LanguageRouted {
        route: LanguageRoute,
    },
    Discontinuity {
        expected_chunk_sequence: u64,
        received_chunk_sequence: u64,
        reason: String,
    },
    Ended {
        metrics: tongues_audio::SegmentationMetrics,
    },
    Error {
        code: &'static str,
        message: String,
    },
}

fn default_browser_vad() -> VadBackendKind {
    VadBackendKind::WebRtc
}

pub(crate) async fn browser_audio_upgrade(
    headers: HeaderMap,
    uri: Uri,
    upgrade: WebSocketUpgrade,
) -> Response {
    if let Err(error) = super::validate_same_origin(
        &headers,
        uri.authority().map(|authority| authority.as_str()),
    ) {
        return (StatusCode::FORBIDDEN, error).into_response();
    }
    upgrade.on_upgrade(browser_audio_session)
}

async fn browser_audio_session(mut socket: WebSocket) {
    let Some(Ok(Message::Text(open))) = socket.recv().await else {
        let _ = send_error(
            &mut socket,
            "open_required",
            "the first message must open the browser audio stream",
        )
        .await;
        return;
    };
    let control = match serde_json::from_str::<ClientControl>(&open) {
        Ok(control) => control,
        Err(error) => {
            let _ = send_error(
                &mut socket,
                "invalid_open",
                format!("invalid browser audio open message: {error}"),
            )
            .await;
            return;
        }
    };
    let ClientControl::Open {
        schema_version,
        sample_rate_hz,
        channels,
        vad_backend,
        segmentation,
        cleanup,
        language_routing,
    } = control
    else {
        let _ = send_error(
            &mut socket,
            "open_required",
            "the first message must have type `open`",
        )
        .await;
        return;
    };
    if schema_version != BROWSER_AUDIO_SCHEMA_VERSION {
        let _ = send_error(
            &mut socket,
            "unsupported_schema",
            format!(
                "browser audio schema {schema_version} is unsupported; expected {BROWSER_AUDIO_SCHEMA_VERSION}"
            ),
        )
        .await;
        return;
    }
    if channels != 1 {
        let _ = send_error(
            &mut socket,
            "unsupported_channels",
            "browser capture must send mono float32 PCM",
        )
        .await;
        return;
    }
    if let Err(error) = segmentation.validate() {
        let _ = send_error(
            &mut socket,
            "invalid_segmentation_config",
            error.to_string(),
        )
        .await;
        return;
    }
    let mut cleanup_pipeline = match CleanupPipeline::new(&cleanup) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            let _ = send_error(&mut socket, "invalid_cleanup_config", error.to_string()).await;
            return;
        }
    };
    let descriptor = match AudioSourceDescriptor::live_pcm(
        "browser-microphone",
        AudioSourceKind::Browser,
        Some("getUserMedia".into()),
        sample_rate_hz,
        channels,
    ) {
        Ok(descriptor) => descriptor,
        Err(error) => {
            let _ = send_error(&mut socket, "invalid_format", error.to_string()).await;
            return;
        }
    };
    let (source_tx, source) = match bounded_audio_input(descriptor, BROWSER_AUDIO_QUEUE_CHUNKS) {
        Ok(input) => input,
        Err(error) => {
            let _ = send_error(&mut socket, "input_unavailable", error.to_string()).await;
            return;
        }
    };
    let (probe_tx, mut probe_rx) = tokio::sync::mpsc::channel(64);
    let consumer_segmentation = segmentation.clone();
    let consumer_language_routing = language_routing.clone();
    let consumer = tokio::task::spawn_blocking(move || {
        // The WebRTC implementation owns a native handle that is not `Send`.
        // Construct and consume it wholly within this blocking worker.
        let segmenter = match UtteranceSegmenter::new("browser-microphone", consumer_segmentation) {
            Ok(segmenter) => segmenter,
            Err(error) => {
                let _ = probe_tx.blocking_send(ProbeEvent::Error {
                    code: "invalid_segmentation_config",
                    message: error.to_string(),
                });
                return;
            }
        };
        let mut pipeline = match VadSegmentationPipeline::new(
            source,
            create_vad_backend(vad_backend),
            segmenter,
        ) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                let _ = probe_tx.blocking_send(ProbeEvent::Error {
                    code: "vad_pipeline_unavailable",
                    message: error.to_string(),
                });
                return;
            }
        };
        let (mut language_tx, mut language_worker) =
            spawn_language_worker(consumer_language_routing, probe_tx.clone());
        loop {
            let probe = match pipeline.next_event() {
                Ok(Some(VadPipelineEvent::VadDecision { frame, decision })) => ProbeEvent::Level {
                    chunk_sequence: frame.sequence,
                    start_frame: Some(frame.start_frame),
                    frame_count: frame.audio.frames(),
                    rms: decision.rms,
                    peak: frame
                        .audio
                        .samples
                        .iter()
                        .map(|sample| sample.abs())
                        .fold(0.0, f32::max),
                    speech_probability: decision.speech_probability,
                    is_speech: decision.is_speech,
                },
                Ok(Some(VadPipelineEvent::SourceDiscontinuity(gap))) => ProbeEvent::Discontinuity {
                    expected_chunk_sequence: gap.expected_chunk_sequence,
                    received_chunk_sequence: gap.received_chunk_sequence,
                    reason: gap.reason,
                },
                Ok(Some(VadPipelineEvent::Segmentation(event))) => match event {
                    SegmentationEvent::SegmentOpened {
                        segment_id,
                        pre_roll_frames,
                        ..
                    } => ProbeEvent::SegmentOpened {
                        segment_id: segment_id.0,
                        pre_roll_frames,
                    },
                    SegmentationEvent::SpeechEnded {
                        segment_id,
                        endpoint_latency_ms,
                        ..
                    } => ProbeEvent::SpeechEnded {
                        segment_id: segment_id.0,
                        endpoint_latency_ms,
                    },
                    SegmentationEvent::SegmentFinalized(segment) => {
                        let final_probe = ProbeEvent::SegmentFinal {
                            segment_id: segment.id.0.clone(),
                            accepted: true,
                            reason: segment.close_reason,
                            speech_duration_ms: segment.speech_duration_ms,
                            total_duration_ms: segment.total_duration_ms,
                        };
                        if probe_tx.blocking_send(final_probe).is_err() {
                            break;
                        }
                        if let Some(sender) = &language_tx
                            && let Err(error) = sender.try_send(segment)
                        {
                            let _ = probe_tx.blocking_send(ProbeEvent::Error {
                                code: "language_backpressure",
                                message: format!(
                                    "language identification queue rejected a segment: {error}"
                                ),
                            });
                        }
                        continue;
                    }
                    SegmentationEvent::SegmentDropped(segment) => ProbeEvent::SegmentFinal {
                        segment_id: segment.id.0,
                        accepted: false,
                        reason: segment.close_reason,
                        speech_duration_ms: segment.speech_duration_ms,
                        total_duration_ms: segment.total_duration_ms,
                    },
                    SegmentationEvent::SpeechStarted { .. }
                    | SegmentationEvent::SpeechResumed { .. }
                    | SegmentationEvent::SegmentUpdated { .. } => continue,
                },
                Ok(Some(VadPipelineEvent::EndOfStream { metrics })) => {
                    language_tx.take();
                    if let Some(worker) = language_worker.take() {
                        let _ = worker.join();
                    }
                    let _ = probe_tx.blocking_send(ProbeEvent::Ended { metrics });
                    break;
                }
                Ok(None) => break,
                Err(error) => {
                    let _ = probe_tx.blocking_send(ProbeEvent::Error {
                        code: "audio_input_failed",
                        message: error.to_string(),
                    });
                    break;
                }
            };
            if probe_tx.blocking_send(probe).is_err() {
                break;
            }
        }
        drop(language_tx);
        if let Some(worker) = language_worker {
            let _ = worker.join();
        }
    });
    if send_probe(
        &mut socket,
        &ProbeEvent::Ready {
            schema_version: BROWSER_AUDIO_SCHEMA_VERSION,
            sample_rate_hz,
            channels,
            queue_capacity_chunks: BROWSER_AUDIO_QUEUE_CHUNKS,
            vad_backend,
            segmentation,
            cleanup,
            language_routing,
        },
    )
    .await
    .is_err()
    {
        drop(source_tx);
        let _ = consumer.await;
        return;
    }

    let (mut socket_tx, mut socket_rx) = socket.split();
    let mut input_open = true;
    loop {
        tokio::select! {
            message = socket_rx.next(), if input_open => match message {
                Some(Ok(Message::Binary(bytes))) => {
                    let mut chunk = match decode_chunk(&bytes, sample_rate_hz, channels) {
                        Ok(chunk) => chunk,
                        Err(error) => {
                            if send_probe_sink(
                                &mut socket_tx,
                                &ProbeEvent::Error {
                                    code: "invalid_audio_chunk",
                                    message: error,
                                },
                            ).await.is_err() {
                                break;
                            }
                            continue;
                        }
                    };
                    let raw_rms = rms(&chunk.audio.samples);
                    let raw_peak = peak(&chunk.audio.samples);
                    let processed = match cleanup_pipeline.process(&chunk.audio) {
                        Ok(processed) => processed,
                        Err(error) => {
                            if send_probe_sink(
                                &mut socket_tx,
                                &ProbeEvent::Error {
                                    code: "audio_cleanup_failed",
                                    message: error.to_string(),
                                },
                            ).await.is_err() {
                                break;
                            }
                            continue;
                        }
                    };
                    let compared = ProbeEvent::CleanupCompared {
                        chunk_sequence: chunk.sequence,
                        raw_rms,
                        processed_rms: rms(&processed.audio.samples),
                        raw_peak,
                        processed_peak: peak(&processed.audio.samples),
                        algorithmic_latency_frames: processed.algorithmic_latency_frames,
                        stages: processed.stages,
                    };
                    chunk.audio = processed.audio;
                    if send_probe_sink(&mut socket_tx, &compared).await.is_err() {
                        break;
                    }
                    if let Err(error) = source_tx.try_send(chunk)
                        && send_probe_sink(
                            &mut socket_tx,
                            &ProbeEvent::Error {
                                code: "audio_backpressure",
                                message: error.to_string(),
                            },
                        ).await.is_err()
                    {
                        break;
                    }
                }
                Some(Ok(Message::Text(text))) => match serde_json::from_str::<ClientControl>(&text) {
                    Ok(ClientControl::End) => {
                        let _ = source_tx.end();
                        input_open = false;
                    }
                    Ok(ClientControl::Open { .. }) => {
                        if send_probe_sink(
                            &mut socket_tx,
                            &ProbeEvent::Error {
                                code: "already_open",
                                message: "the browser audio stream is already open".into(),
                            },
                        ).await.is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        if send_probe_sink(
                            &mut socket_tx,
                            &ProbeEvent::Error {
                                code: "invalid_control",
                                message: format!("invalid browser audio control message: {error}"),
                            },
                        ).await.is_err() {
                            break;
                        }
                    }
                },
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break,
                Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) => {}
            },
            probe = probe_rx.recv() => {
                let Some(probe) = probe else {
                    break;
                };
                let ended = matches!(probe, ProbeEvent::Ended { .. });
                if send_probe_sink(&mut socket_tx, &probe).await.is_err() || ended {
                    break;
                }
            }
        }
    }
    drop(source_tx);
    let _ = consumer.await;
}

fn decode_chunk(
    bytes: &[u8],
    sample_rate_hz: u32,
    channels: u16,
) -> Result<PushedAudioChunk, String> {
    if bytes.len() < BROWSER_AUDIO_HEADER_BYTES {
        return Err("browser audio chunk is missing its sequence/frame header".into());
    }
    if bytes.len() > MAX_BROWSER_AUDIO_CHUNK_BYTES {
        return Err(format!(
            "browser audio chunk exceeds the {MAX_BROWSER_AUDIO_CHUNK_BYTES}-byte limit"
        ));
    }
    let pcm = &bytes[BROWSER_AUDIO_HEADER_BYTES..];
    if pcm.is_empty() || !pcm.len().is_multiple_of(4 * usize::from(channels)) {
        return Err("browser audio payload is not complete interleaved float32 PCM".into());
    }
    let sequence = u64::from_le_bytes(bytes[0..8].try_into().expect("checked header length"));
    let start_frame = u64::from_le_bytes(bytes[8..16].try_into().expect("checked header length"));
    let samples = pcm
        .chunks_exact(4)
        .map(|word| {
            let sample = f32::from_le_bytes(word.try_into().expect("four-byte chunk"));
            sample
                .is_finite()
                .then_some(sample)
                .ok_or_else(|| "browser audio contains a non-finite sample".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PushedAudioChunk {
        sequence,
        start_frame: Some(start_frame),
        audio: tongues_audio::AudioBuffer {
            samples,
            sample_rate_hz,
            channels,
        },
    })
}

fn peak(samples: &[f32]) -> f32 {
    samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max)
}

fn route_audio_segment(
    router: &mut LanguageRouter,
    identifier: Option<&mut WhisperLanguageIdentifier>,
    sequence: u64,
    segment: &tongues_audio::AudioSegment,
) -> anyhow::Result<LanguageRoute> {
    let Some(first) = segment.frames.first() else {
        anyhow::bail!("cannot route an empty audio segment");
    };
    let mut audio = tongues_audio::AudioBuffer {
        samples: segment
            .frames
            .iter()
            .flat_map(|frame| frame.audio.samples.iter().copied())
            .collect(),
        sample_rate_hz: first.audio.sample_rate_hz,
        channels: first.audio.channels,
    };
    audio = audio.convert_channels(1)?.resample_linear(16_000)?;
    let detection = identifier
        .map(|identifier| {
            identifier.detect(
                &segment.id.0,
                sequence,
                &speaking::AudioFrame {
                    sample_rate_hz: audio.sample_rate_hz,
                    channels: audio.channels,
                    samples: audio.samples,
                },
            )
        })
        .transpose()?;
    router.route(sequence, detection)
}

fn spawn_language_worker(
    config: Option<BrowserLanguageRouting>,
    probe_tx: tokio::sync::mpsc::Sender<ProbeEvent>,
) -> (
    Option<std::sync::mpsc::SyncSender<tongues_audio::AudioSegment>>,
    Option<std::thread::JoinHandle<()>>,
) {
    let Some(config) = config else {
        return (None, None);
    };
    let (segment_tx, segment_rx) = std::sync::mpsc::sync_channel(2);
    let worker = std::thread::spawn(move || {
        let capabilities = tongues_cli::language_routing_cmd::capabilities();
        let mut router = match LanguageRouter::new(
            config.mode.clone(),
            config.switching,
            config.unsupported,
            capabilities.asr_providers,
        ) {
            Ok(router) => router,
            Err(error) => {
                let _ = probe_tx.blocking_send(ProbeEvent::Error {
                    code: "invalid_language_routing",
                    message: error.to_string(),
                });
                return;
            }
        };
        let mut identifier = if matches!(config.mode, LanguageSelectionMode::Detect { .. }) {
            let Some(detector) = capabilities
                .detectors
                .iter()
                .find(|detector| detector.installed)
            else {
                let _ = probe_tx.blocking_send(ProbeEvent::Error {
                    code: "language_detector_unavailable",
                    message: "no installed language detector is available".into(),
                });
                return;
            };
            match WhisperLanguageIdentifier::new_quiet(&detector.model_id) {
                Ok(identifier) => Some(identifier),
                Err(error) => {
                    let _ = probe_tx.blocking_send(ProbeEvent::Error {
                        code: "language_detector_unavailable",
                        message: error.to_string(),
                    });
                    return;
                }
            }
        } else {
            None
        };
        for (sequence, segment) in segment_rx.into_iter().enumerate() {
            match route_audio_segment(&mut router, identifier.as_mut(), sequence as u64, &segment) {
                Ok(route) => {
                    if probe_tx
                        .blocking_send(ProbeEvent::LanguageRouted { route })
                        .is_err()
                    {
                        return;
                    }
                }
                Err(error) => {
                    if probe_tx
                        .blocking_send(ProbeEvent::Error {
                            code: "language_routing_failed",
                            message: error.to_string(),
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            }
        }
    });
    (Some(segment_tx), Some(worker))
}

async fn send_error(
    socket: &mut WebSocket,
    code: &'static str,
    message: impl Into<String>,
) -> Result<(), axum::Error> {
    send_probe(
        socket,
        &ProbeEvent::Error {
            code,
            message: message.into(),
        },
    )
    .await
}

async fn send_probe(socket: &mut WebSocket, event: &ProbeEvent) -> Result<(), axum::Error> {
    let encoded = serde_json::to_string(event).expect("probe events contain only finite values");
    socket.send(Message::Text(encoded.into())).await
}

async fn send_probe_sink(
    socket: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    event: &ProbeEvent,
) -> Result<(), axum::Error> {
    let encoded = serde_json::to_string(event).expect("probe events contain only finite values");
    socket.send(Message::Text(encoded.into())).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_chunk_header_preserves_sequence_and_frame_position() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&7_u64.to_le_bytes());
        bytes.extend_from_slice(&320_u64.to_le_bytes());
        bytes.extend_from_slice(&0.25_f32.to_le_bytes());
        bytes.extend_from_slice(&(-0.5_f32).to_le_bytes());
        let chunk = decode_chunk(&bytes, 48_000, 1).unwrap();
        assert_eq!(chunk.sequence, 7);
        assert_eq!(chunk.start_frame, Some(320));
        assert_eq!(chunk.audio.samples, vec![0.25, -0.5]);
    }

    #[test]
    fn browser_chunk_rejects_non_finite_pcm() {
        let mut bytes = vec![0; BROWSER_AUDIO_HEADER_BYTES];
        bytes.extend_from_slice(&f32::NAN.to_le_bytes());
        assert!(
            decode_chunk(&bytes, 48_000, 1)
                .unwrap_err()
                .contains("non-finite")
        );
    }
}
