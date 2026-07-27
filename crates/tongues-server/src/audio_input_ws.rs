use axum::extract::WebSocketUpgrade;
use axum::extract::ws::{Message, WebSocket};
use axum::http::{HeaderMap, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use tongues_audio::{
    AudioSource, AudioSourceDescriptor, AudioSourceEvent, AudioSourceKind, PushedAudioChunk,
    bounded_audio_input,
};

const BROWSER_AUDIO_SCHEMA_VERSION: u16 = 1;
const BROWSER_AUDIO_QUEUE_CHUNKS: usize = 8;
const BROWSER_AUDIO_HEADER_BYTES: usize = 16;
const MAX_BROWSER_AUDIO_CHUNK_BYTES: usize = 256 * 1024;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
enum ClientControl {
    Open {
        schema_version: u16,
        sample_rate_hz: u32,
        channels: u16,
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
    },
    Level {
        chunk_sequence: u64,
        start_frame: Option<u64>,
        frame_count: usize,
        rms: f32,
        peak: f32,
    },
    Discontinuity {
        expected_chunk_sequence: u64,
        received_chunk_sequence: u64,
        reason: String,
    },
    Ended,
    Error {
        code: &'static str,
        message: String,
    },
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
    let (source_tx, mut source) = match bounded_audio_input(descriptor, BROWSER_AUDIO_QUEUE_CHUNKS)
    {
        Ok(input) => input,
        Err(error) => {
            let _ = send_error(&mut socket, "input_unavailable", error.to_string()).await;
            return;
        }
    };
    let (probe_tx, mut probe_rx) = tokio::sync::mpsc::channel(2);
    let consumer = tokio::task::spawn_blocking(move || {
        loop {
            let probe = match source.next_event() {
                Ok(AudioSourceEvent::Audio(chunk)) => ProbeEvent::Level {
                    chunk_sequence: chunk.sequence,
                    start_frame: chunk.start_frame,
                    frame_count: chunk.audio.frames(),
                    rms: tongues_audio::rms(&chunk.audio.samples),
                    peak: chunk
                        .audio
                        .samples
                        .iter()
                        .map(|sample| sample.abs())
                        .fold(0.0, f32::max),
                },
                Ok(AudioSourceEvent::Discontinuity(gap)) => ProbeEvent::Discontinuity {
                    expected_chunk_sequence: gap.expected_chunk_sequence,
                    received_chunk_sequence: gap.received_chunk_sequence,
                    reason: gap.reason,
                },
                Ok(AudioSourceEvent::EndOfStream) => {
                    let _ = probe_tx.blocking_send(ProbeEvent::Ended);
                    break;
                }
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
    });
    if send_probe(
        &mut socket,
        &ProbeEvent::Ready {
            schema_version: BROWSER_AUDIO_SCHEMA_VERSION,
            sample_rate_hz,
            channels,
            queue_capacity_chunks: BROWSER_AUDIO_QUEUE_CHUNKS,
        },
    )
    .await
    .is_err()
    {
        drop(source_tx);
        let _ = consumer.await;
        return;
    }

    'session: while let Some(message) = socket.recv().await {
        match message {
            Ok(Message::Binary(bytes)) => {
                let chunk = match decode_chunk(&bytes, sample_rate_hz, channels) {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        if send_error(&mut socket, "invalid_audio_chunk", error)
                            .await
                            .is_err()
                        {
                            break;
                        }
                        continue;
                    }
                };
                if let Err(error) = source_tx.try_send(chunk) {
                    if send_error(&mut socket, "audio_backpressure", error.to_string())
                        .await
                        .is_err()
                    {
                        break;
                    }
                    continue;
                }
                loop {
                    let Some(probe) = probe_rx.recv().await else {
                        break 'session;
                    };
                    let chunk_acknowledged = matches!(
                        probe,
                        ProbeEvent::Level { .. } | ProbeEvent::Error { .. } | ProbeEvent::Ended
                    );
                    if send_probe(&mut socket, &probe).await.is_err() {
                        break 'session;
                    }
                    if chunk_acknowledged {
                        break;
                    }
                }
            }
            Ok(Message::Text(text)) => match serde_json::from_str::<ClientControl>(&text) {
                Ok(ClientControl::End) => {
                    let _ = source_tx.end();
                    if let Some(probe) = probe_rx.recv().await {
                        let _ = send_probe(&mut socket, &probe).await;
                    }
                    break;
                }
                Ok(ClientControl::Open { .. }) => {
                    if send_error(
                        &mut socket,
                        "already_open",
                        "the browser audio stream is already open",
                    )
                    .await
                    .is_err()
                    {
                        break;
                    }
                }
                Err(error) => {
                    if send_error(
                        &mut socket,
                        "invalid_control",
                        format!("invalid browser audio control message: {error}"),
                    )
                    .await
                    .is_err()
                    {
                        break;
                    }
                }
            },
            Ok(Message::Close(_)) | Err(_) => break,
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => {}
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
