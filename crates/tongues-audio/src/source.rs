//! Provider-neutral audio ingestion.
//!
//! Sources decode transport-specific input into the same interleaved `f32`
//! chunks. Sequence gaps are events, never silently compressed away.

use std::collections::BTreeMap;
use std::io::{Cursor, Read};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use serde::{Deserialize, Serialize};
use speaking::{
    AudioDirection, AudioEncoding, AudioFormat, ChannelLayout, ClockOrigin, StreamEvent,
    StreamSource,
};

use crate::{invalid, read_wav, AudioBuffer, AudioError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioSourceKind {
    File,
    Stdin,
    Microphone,
    Tcp,
    Unix,
    Browser,
    Server,
    Fixture,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioSourceDescriptor {
    pub id: String,
    pub kind: AudioSourceKind,
    pub source: StreamSource,
    /// The format presented to downstream stages after transport decoding.
    pub decoded_format: AudioFormat,
    pub live: bool,
    pub seekable: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl AudioSourceDescriptor {
    pub fn stream_opened_event(&self, clock: ClockOrigin) -> StreamEvent {
        StreamEvent::StreamOpened {
            source: self.source.clone(),
            format: self.decoded_format.clone(),
            clock,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioDiscontinuity {
    pub expected_chunk_sequence: u64,
    pub received_chunk_sequence: u64,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SourceAudioChunk {
    pub sequence: u64,
    /// Absolute decoded frame position when the source can supply one.
    pub start_frame: Option<u64>,
    pub audio: AudioBuffer,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AudioSourceEvent {
    Audio(SourceAudioChunk),
    Discontinuity(AudioDiscontinuity),
    EndOfStream,
}

impl AudioSourceEvent {
    /// Project ingestion state into the shared #115 wire contract.
    ///
    /// PCM stays in the library-owned chunk; transports decide whether raw
    /// audio may be serialized according to their privacy and retention policy.
    pub fn stream_event(&self) -> StreamEvent {
        match self {
            Self::Audio(chunk) => {
                let mut metadata = BTreeMap::new();
                if let Some(start_frame) = chunk.start_frame {
                    metadata.insert("start_frame".into(), serde_json::Value::from(start_frame));
                }
                StreamEvent::AudioChunk {
                    direction: AudioDirection::Input,
                    chunk_sequence: chunk.sequence,
                    frame_count: u32::try_from(chunk.audio.frames()).unwrap_or(u32::MAX),
                    segment_id: None,
                    format: None,
                    audio_base64: None,
                    metadata,
                }
            }
            Self::Discontinuity(gap) => StreamEvent::Discontinuity {
                expected_chunk_sequence: gap.expected_chunk_sequence,
                received_chunk_sequence: gap.received_chunk_sequence,
                reason: gap.reason.clone(),
            },
            Self::EndOfStream => StreamEvent::EndOfStream,
        }
    }
}

pub trait AudioSource {
    fn descriptor(&self) -> &AudioSourceDescriptor;
    fn next_event(&mut self) -> Result<AudioSourceEvent>;
    fn cancel(&mut self);
}

#[derive(Debug)]
pub struct WavAudioSource {
    descriptor: AudioSourceDescriptor,
    audio: AudioBuffer,
    chunk_frames: usize,
    next_frame: usize,
    sequence: u64,
    ended: bool,
    cancelled: bool,
}

impl WavAudioSource {
    pub fn open(path: impl AsRef<Path>, chunk_frames: usize) -> Result<Self> {
        let path = path.as_ref();
        let audio = read_wav(path)?;
        Self::new(
            path.display().to_string(),
            StreamSource::File {
                path: path.display().to_string(),
            },
            audio,
            chunk_frames,
        )
    }

    pub fn from_bytes(id: impl Into<String>, bytes: &[u8], chunk_frames: usize) -> Result<Self> {
        let id = id.into();
        let mut reader = hound::WavReader::new(Cursor::new(bytes))?;
        let spec = reader.spec();
        let samples = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()?,
            hound::SampleFormat::Int => {
                if spec.bits_per_sample == 0 || spec.bits_per_sample > 32 {
                    return Err(invalid(format!(
                        "unsupported {}-bit integer WAV",
                        spec.bits_per_sample
                    )));
                }
                let divisor = 2_f32.powi(i32::from(spec.bits_per_sample) - 1);
                reader
                    .samples::<i32>()
                    .map(|sample| sample.map(|value| value as f32 / divisor))
                    .collect::<std::result::Result<Vec<_>, _>>()?
            }
        };
        Self::new(
            id.clone(),
            StreamSource::Replay {
                source_stream_id: speaking::StreamId(id),
            },
            AudioBuffer {
                samples,
                sample_rate_hz: spec.sample_rate,
                channels: spec.channels,
            },
            chunk_frames,
        )
    }

    fn new(
        id: String,
        source: StreamSource,
        audio: AudioBuffer,
        chunk_frames: usize,
    ) -> Result<Self> {
        audio.validate()?;
        if chunk_frames == 0 {
            return Err(invalid("audio source chunk size must be positive"));
        }
        Ok(Self {
            descriptor: AudioSourceDescriptor {
                id,
                kind: AudioSourceKind::File,
                source,
                decoded_format: decoded_format(audio.sample_rate_hz, audio.channels),
                live: false,
                seekable: true,
                metadata: BTreeMap::new(),
            },
            audio,
            chunk_frames,
            next_frame: 0,
            sequence: 0,
            ended: false,
            cancelled: false,
        })
    }
}

impl AudioSource for WavAudioSource {
    fn descriptor(&self) -> &AudioSourceDescriptor {
        &self.descriptor
    }

    fn next_event(&mut self) -> Result<AudioSourceEvent> {
        if self.cancelled {
            return Err(AudioError::Cancelled);
        }
        if self.next_frame < self.audio.frames() {
            let channels = usize::from(self.audio.channels);
            let end_frame = (self.next_frame + self.chunk_frames).min(self.audio.frames());
            let audio = AudioBuffer {
                samples: self.audio.samples[self.next_frame * channels..end_frame * channels]
                    .to_vec(),
                sample_rate_hz: self.audio.sample_rate_hz,
                channels: self.audio.channels,
            };
            let chunk = SourceAudioChunk {
                sequence: self.sequence,
                start_frame: Some(self.next_frame as u64),
                audio,
            };
            self.next_frame = end_frame;
            self.sequence = self.sequence.saturating_add(1);
            return Ok(AudioSourceEvent::Audio(chunk));
        }
        if !self.ended {
            self.ended = true;
            return Ok(AudioSourceEvent::EndOfStream);
        }
        Ok(AudioSourceEvent::EndOfStream)
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PcmEncoding {
    Signed16Le,
    Float32Le,
}

impl PcmEncoding {
    fn bytes_per_sample(self) -> usize {
        match self {
            Self::Signed16Le => 2,
            Self::Float32Le => 4,
        }
    }
}

pub struct PcmReaderSource<R> {
    descriptor: AudioSourceDescriptor,
    reader: R,
    encoding: PcmEncoding,
    channels: u16,
    read_buffer: Vec<u8>,
    pending: Vec<u8>,
    sequence: u64,
    next_frame: u64,
    ended: bool,
    cancelled: bool,
}

impl<R: Read> PcmReaderSource<R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        kind: AudioSourceKind,
        source: StreamSource,
        reader: R,
        encoding: PcmEncoding,
        sample_rate_hz: u32,
        channels: u16,
        read_bytes: usize,
    ) -> Result<Self> {
        if sample_rate_hz == 0 {
            return Err(invalid("PCM sample rate must be positive"));
        }
        if channels == 0 {
            return Err(invalid("PCM channel count must be positive"));
        }
        let bytes_per_frame = encoding.bytes_per_sample() * usize::from(channels);
        if read_bytes < bytes_per_frame {
            return Err(invalid(format!(
                "PCM read size must be at least one {bytes_per_frame}-byte frame"
            )));
        }
        Ok(Self {
            descriptor: AudioSourceDescriptor {
                id: id.into(),
                kind,
                source,
                decoded_format: decoded_format(sample_rate_hz, channels),
                live: !matches!(kind, AudioSourceKind::File | AudioSourceKind::Fixture),
                seekable: matches!(kind, AudioSourceKind::File | AudioSourceKind::Fixture),
                metadata: BTreeMap::from([
                    ("transport_encoding".into(), format!("{encoding:?}")),
                    ("read_bytes".into(), read_bytes.to_string()),
                ]),
            },
            reader,
            encoding,
            channels,
            read_buffer: vec![0; read_bytes],
            pending: Vec::with_capacity(bytes_per_frame - 1),
            sequence: 0,
            next_frame: 0,
            ended: false,
            cancelled: false,
        })
    }

    fn decode_available(&mut self, bytes_read: usize) -> Result<Option<SourceAudioChunk>> {
        self.pending
            .extend_from_slice(&self.read_buffer[..bytes_read]);
        let bytes_per_frame = self.encoding.bytes_per_sample() * usize::from(self.channels);
        let decoded_bytes = self.pending.len() / bytes_per_frame * bytes_per_frame;
        if decoded_bytes == 0 {
            return Ok(None);
        }
        let samples = decode_pcm(&self.pending[..decoded_bytes], self.encoding)?;
        self.pending.drain(..decoded_bytes);
        let audio = AudioBuffer {
            samples,
            sample_rate_hz: self.descriptor.decoded_format.sample_rate_hz,
            channels: self.channels,
        };
        audio.validate()?;
        let frames = audio.frames() as u64;
        let chunk = SourceAudioChunk {
            sequence: self.sequence,
            start_frame: Some(self.next_frame),
            audio,
        };
        self.sequence = self.sequence.saturating_add(1);
        self.next_frame = self.next_frame.saturating_add(frames);
        Ok(Some(chunk))
    }
}

impl<R: Read> AudioSource for PcmReaderSource<R> {
    fn descriptor(&self) -> &AudioSourceDescriptor {
        &self.descriptor
    }

    fn next_event(&mut self) -> Result<AudioSourceEvent> {
        if self.cancelled {
            return Err(AudioError::Cancelled);
        }
        if self.ended {
            return Ok(AudioSourceEvent::EndOfStream);
        }
        loop {
            let bytes_read = self.reader.read(&mut self.read_buffer)?;
            if bytes_read == 0 {
                if !self.pending.is_empty() {
                    return Err(invalid(format!(
                        "PCM stream ended with {} trailing byte(s), not a complete frame",
                        self.pending.len()
                    )));
                }
                self.ended = true;
                return Ok(AudioSourceEvent::EndOfStream);
            }
            if let Some(chunk) = self.decode_available(bytes_read)? {
                return Ok(AudioSourceEvent::Audio(chunk));
            }
        }
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }
}

fn decode_pcm(bytes: &[u8], encoding: PcmEncoding) -> Result<Vec<f32>> {
    match encoding {
        PcmEncoding::Signed16Le => Ok(bytes
            .chunks_exact(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32768.0)
            .collect()),
        PcmEncoding::Float32Le => bytes
            .chunks_exact(4)
            .map(|word| {
                let value = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(invalid("PCM contains a non-finite float sample"))
                }
            })
            .collect(),
    }
}

#[derive(Debug, Clone)]
pub struct PushedAudioChunk {
    pub sequence: u64,
    pub start_frame: Option<u64>,
    pub audio: AudioBuffer,
}

enum PushedMessage {
    Chunk(PushedAudioChunk),
    End,
}

#[derive(Clone)]
pub struct BoundedAudioInputSender {
    sender: mpsc::SyncSender<PushedMessage>,
    cancelled: Arc<AtomicBool>,
    capacity: usize,
}

impl BoundedAudioInputSender {
    pub fn try_send(&self, chunk: PushedAudioChunk) -> Result<()> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(AudioError::Cancelled);
        }
        chunk.audio.validate()?;
        self.sender
            .try_send(PushedMessage::Chunk(chunk))
            .map_err(|error| match error {
                mpsc::TrySendError::Full(_) => AudioError::Backpressure {
                    capacity: self.capacity,
                },
                mpsc::TrySendError::Disconnected(_) => AudioError::Cancelled,
            })
    }

    pub fn end(&self) -> Result<()> {
        self.sender
            .try_send(PushedMessage::End)
            .map_err(|error| match error {
                mpsc::TrySendError::Full(_) => AudioError::Backpressure {
                    capacity: self.capacity,
                },
                mpsc::TrySendError::Disconnected(_) => AudioError::Cancelled,
            })
    }
}

pub struct BoundedAudioInput {
    descriptor: AudioSourceDescriptor,
    receiver: mpsc::Receiver<PushedMessage>,
    cancelled: Arc<AtomicBool>,
    expected_sequence: u64,
    pending_chunk: Option<PushedAudioChunk>,
    ended: bool,
}

pub fn bounded_audio_input(
    descriptor: AudioSourceDescriptor,
    capacity: usize,
) -> Result<(BoundedAudioInputSender, BoundedAudioInput)> {
    if capacity == 0 {
        return Err(invalid("audio input queue capacity must be positive"));
    }
    let (sender, receiver) = mpsc::sync_channel(capacity);
    let cancelled = Arc::new(AtomicBool::new(false));
    Ok((
        BoundedAudioInputSender {
            sender,
            cancelled: Arc::clone(&cancelled),
            capacity,
        },
        BoundedAudioInput {
            descriptor,
            receiver,
            cancelled,
            expected_sequence: 0,
            pending_chunk: None,
            ended: false,
        },
    ))
}

impl AudioSource for BoundedAudioInput {
    fn descriptor(&self) -> &AudioSourceDescriptor {
        &self.descriptor
    }

    fn next_event(&mut self) -> Result<AudioSourceEvent> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(AudioError::Cancelled);
        }
        if let Some(chunk) = self.pending_chunk.take() {
            self.expected_sequence = chunk.sequence.saturating_add(1);
            return Ok(AudioSourceEvent::Audio(SourceAudioChunk {
                sequence: chunk.sequence,
                start_frame: chunk.start_frame,
                audio: chunk.audio,
            }));
        }
        if self.ended {
            return Ok(AudioSourceEvent::EndOfStream);
        }
        match self.receiver.recv() {
            Ok(PushedMessage::Chunk(chunk)) if chunk.sequence != self.expected_sequence => {
                let discontinuity = AudioDiscontinuity {
                    expected_chunk_sequence: self.expected_sequence,
                    received_chunk_sequence: chunk.sequence,
                    reason: if chunk.sequence < self.expected_sequence {
                        "out-of-order or duplicate audio chunk".into()
                    } else {
                        "audio chunk gap".into()
                    },
                };
                self.pending_chunk = Some(chunk);
                Ok(AudioSourceEvent::Discontinuity(discontinuity))
            }
            Ok(PushedMessage::Chunk(chunk)) => {
                self.expected_sequence = chunk.sequence.saturating_add(1);
                Ok(AudioSourceEvent::Audio(SourceAudioChunk {
                    sequence: chunk.sequence,
                    start_frame: chunk.start_frame,
                    audio: chunk.audio,
                }))
            }
            Ok(PushedMessage::End) | Err(_) => {
                self.ended = true;
                Ok(AudioSourceEvent::EndOfStream)
            }
        }
    }

    fn cancel(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

fn decoded_format(sample_rate_hz: u32, channels: u16) -> AudioFormat {
    AudioFormat {
        encoding: AudioEncoding::PcmF32Le,
        sample_rate_hz,
        channels: match channels {
            1 => ChannelLayout::Mono,
            2 => ChannelLayout::Stereo,
            count => ChannelLayout::Interleaved {
                labels: (0..count).map(|index| format!("channel_{index}")).collect(),
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use speaking::{ClockOrigin, StreamId};

    use super::*;

    struct ShortReader {
        bytes: Cursor<Vec<u8>>,
        max_read: usize,
    }

    impl Read for ShortReader {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            let limit = output.len().min(self.max_read);
            self.bytes.read(&mut output[..limit])
        }
    }

    fn descriptor() -> AudioSourceDescriptor {
        AudioSourceDescriptor {
            id: "browser-test".into(),
            kind: AudioSourceKind::Browser,
            source: StreamSource::Live {
                device: Some("browser-default".into()),
            },
            decoded_format: decoded_format(16_000, 1),
            live: true,
            seekable: false,
            metadata: BTreeMap::from([("clock".into(), format!("{:?}", ClockOrigin::StreamStart))]),
        }
    }

    fn mono_chunk(sequence: u64) -> PushedAudioChunk {
        PushedAudioChunk {
            sequence,
            start_frame: Some(sequence * 2),
            audio: AudioBuffer {
                samples: vec![0.25, -0.25],
                sample_rate_hz: 16_000,
                channels: 1,
            },
        }
    }

    #[test]
    fn pcm_reader_preserves_frames_across_arbitrary_transport_reads() {
        let values = [i16::MIN, -1, 0, i16::MAX];
        let bytes = values
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect::<Vec<_>>();
        let reader = ShortReader {
            bytes: Cursor::new(bytes),
            max_read: 3,
        };
        let mut source = PcmReaderSource::new(
            "stdin",
            AudioSourceKind::Stdin,
            StreamSource::Live {
                device: Some("stdin".into()),
            },
            reader,
            PcmEncoding::Signed16Le,
            16_000,
            2,
            5,
        )
        .unwrap();

        let mut samples = Vec::new();
        loop {
            match source.next_event().unwrap() {
                AudioSourceEvent::Audio(chunk) => samples.extend(chunk.audio.samples),
                AudioSourceEvent::EndOfStream => break,
                AudioSourceEvent::Discontinuity(_) => panic!("reader invented a gap"),
            }
        }
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0], -1.0);
        assert_eq!(samples[2], 0.0);
        assert!(samples[3] < 1.0);
    }

    #[test]
    fn pcm_reader_rejects_an_incomplete_final_frame() {
        let reader = Cursor::new(vec![0, 0, 1]);
        let mut source = PcmReaderSource::new(
            "socket",
            AudioSourceKind::Tcp,
            StreamSource::Live {
                device: Some("tcp:test".into()),
            },
            reader,
            PcmEncoding::Signed16Le,
            16_000,
            1,
            4,
        )
        .unwrap();
        assert!(matches!(
            source.next_event().unwrap(),
            AudioSourceEvent::Audio(_)
        ));
        let error = source.next_event().unwrap_err();
        assert!(error.to_string().contains("trailing byte"));
    }

    #[test]
    fn bounded_input_surfaces_gaps_before_delivering_the_chunk() {
        let (sender, mut source) = bounded_audio_input(descriptor(), 2).unwrap();
        sender.try_send(mono_chunk(0)).unwrap();
        sender.try_send(mono_chunk(2)).unwrap();

        assert!(matches!(
            source.next_event().unwrap(),
            AudioSourceEvent::Audio(SourceAudioChunk { sequence: 0, .. })
        ));
        assert_eq!(
            source.next_event().unwrap(),
            AudioSourceEvent::Discontinuity(AudioDiscontinuity {
                expected_chunk_sequence: 1,
                received_chunk_sequence: 2,
                reason: "audio chunk gap".into(),
            })
        );
        assert!(matches!(
            source.next_event().unwrap(),
            AudioSourceEvent::Audio(SourceAudioChunk { sequence: 2, .. })
        ));
    }

    #[test]
    fn bounded_input_fails_fast_instead_of_growing_without_limit() {
        let (sender, _source) = bounded_audio_input(descriptor(), 1).unwrap();
        sender.try_send(mono_chunk(0)).unwrap();
        assert!(matches!(
            sender.try_send(mono_chunk(1)),
            Err(AudioError::Backpressure { capacity: 1 })
        ));
    }

    #[test]
    fn cancellation_is_shared_between_source_and_producers() {
        let (sender, mut source) = bounded_audio_input(descriptor(), 1).unwrap();
        source.cancel();
        assert!(matches!(source.next_event(), Err(AudioError::Cancelled)));
        assert!(matches!(
            sender.try_send(mono_chunk(0)),
            Err(AudioError::Cancelled)
        ));
    }

    #[test]
    fn fixture_descriptor_can_use_replay_provenance() {
        let descriptor = AudioSourceDescriptor {
            id: "fixture".into(),
            kind: AudioSourceKind::Fixture,
            source: StreamSource::Replay {
                source_stream_id: StreamId("original".into()),
            },
            decoded_format: decoded_format(48_000, 2),
            live: false,
            seekable: true,
            metadata: BTreeMap::new(),
        };
        assert!(matches!(
            descriptor.source,
            StreamSource::Replay { source_stream_id } if source_stream_id.0 == "original"
        ));
    }

    #[test]
    fn source_events_project_into_the_shared_stream_contract() {
        let event = AudioSourceEvent::Audio(SourceAudioChunk {
            sequence: 7,
            start_frame: Some(320),
            audio: AudioBuffer {
                samples: vec![0.0; 160],
                sample_rate_hz: 16_000,
                channels: 1,
            },
        })
        .stream_event();
        assert!(matches!(
            event,
            StreamEvent::AudioChunk {
                direction: AudioDirection::Input,
                chunk_sequence: 7,
                frame_count: 160,
                metadata,
                ..
            } if metadata["start_frame"] == 320
        ));

        assert!(matches!(
            descriptor().stream_opened_event(ClockOrigin::StreamStart),
            StreamEvent::StreamOpened {
                source: StreamSource::Live { .. },
                clock: ClockOrigin::StreamStart,
                ..
            }
        ));
    }
}
