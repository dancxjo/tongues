//! Deterministic source-format normalization.

use speaking::{AudioEncoding, AudioFormat, ChannelLayout};

use crate::{
    invalid, AudioDiscontinuity, AudioSource, AudioSourceDescriptor, AudioSourceEvent, Result,
    SourceAudioChunk,
};

pub struct NormalizedAudioSource<S> {
    inner: S,
    descriptor: AudioSourceDescriptor,
    source_rate_hz: u32,
    source_channels: u16,
    target_rate_hz: u32,
    target_channels: u16,
    expected_source_frame: Option<u64>,
    pending_chunk: Option<SourceAudioChunk>,
}

impl<S: AudioSource> NormalizedAudioSource<S> {
    pub fn new(inner: S, target_rate_hz: u32, target_channels: u16) -> Result<Self> {
        if target_rate_hz == 0 {
            return Err(invalid("target sample rate must be positive"));
        }
        if target_channels == 0 {
            return Err(invalid("target channel count must be positive"));
        }
        let source_rate_hz = inner.descriptor().decoded_format.sample_rate_hz;
        let source_channels = channel_count(&inner.descriptor().decoded_format.channels)?;
        let mut descriptor = inner.descriptor().clone();
        descriptor
            .metadata
            .insert("source_sample_rate_hz".into(), source_rate_hz.to_string());
        descriptor
            .metadata
            .insert("source_channels".into(), source_channels.to_string());
        descriptor.decoded_format = AudioFormat {
            encoding: AudioEncoding::PcmF32Le,
            sample_rate_hz: target_rate_hz,
            channels: channel_layout(target_channels),
        };
        Ok(Self {
            inner,
            descriptor,
            source_rate_hz,
            source_channels,
            target_rate_hz,
            target_channels,
            expected_source_frame: None,
            pending_chunk: None,
        })
    }

    fn normalize_chunk(&mut self, chunk: SourceAudioChunk) -> Result<AudioSourceEvent> {
        if chunk.audio.sample_rate_hz != self.source_rate_hz
            || chunk.audio.channels != self.source_channels
        {
            return Err(invalid(format!(
                "source changed format without renegotiation: expected {} Hz/{} channels, received {} Hz/{} channels",
                self.source_rate_hz,
                self.source_channels,
                chunk.audio.sample_rate_hz,
                chunk.audio.channels
            )));
        }
        let source_frames = chunk.audio.frames() as u64;
        let audio = chunk
            .audio
            .convert_channels(self.target_channels)?
            .resample_linear(self.target_rate_hz)?;
        let start_frame = chunk
            .start_frame
            .map(|frame| scale_frame(frame, self.source_rate_hz, self.target_rate_hz));
        self.expected_source_frame = chunk
            .start_frame
            .map(|frame| frame.saturating_add(source_frames));
        Ok(AudioSourceEvent::Audio(SourceAudioChunk {
            sequence: chunk.sequence,
            start_frame,
            audio,
        }))
    }
}

impl<S: AudioSource> AudioSource for NormalizedAudioSource<S> {
    fn descriptor(&self) -> &AudioSourceDescriptor {
        &self.descriptor
    }

    fn next_event(&mut self) -> Result<AudioSourceEvent> {
        if let Some(chunk) = self.pending_chunk.take() {
            return self.normalize_chunk(chunk);
        }
        match self.inner.next_event()? {
            AudioSourceEvent::Audio(chunk)
                if self
                    .expected_source_frame
                    .zip(chunk.start_frame)
                    .is_some_and(|(expected, received)| expected != received) =>
            {
                let expected = self.expected_source_frame.expect("guarded above");
                let received = chunk.start_frame.expect("guarded above");
                let sequence = chunk.sequence;
                self.pending_chunk = Some(chunk);
                Ok(AudioSourceEvent::Discontinuity(AudioDiscontinuity {
                    expected_chunk_sequence: sequence,
                    received_chunk_sequence: sequence,
                    reason: format!(
                        "audio frame timeline discontinuity: expected frame {expected}, received {received}"
                    ),
                }))
            }
            AudioSourceEvent::Audio(chunk) => self.normalize_chunk(chunk),
            AudioSourceEvent::Discontinuity(gap) => {
                self.expected_source_frame = None;
                Ok(AudioSourceEvent::Discontinuity(gap))
            }
            AudioSourceEvent::EndOfStream => Ok(AudioSourceEvent::EndOfStream),
        }
    }

    fn cancel(&mut self) {
        self.inner.cancel();
    }
}

fn scale_frame(frame: u64, source_rate_hz: u32, target_rate_hz: u32) -> u64 {
    let scaled = u128::from(frame) * u128::from(target_rate_hz);
    let rounded = (scaled + u128::from(source_rate_hz) / 2) / u128::from(source_rate_hz);
    u64::try_from(rounded).unwrap_or(u64::MAX)
}

fn channel_count(layout: &ChannelLayout) -> Result<u16> {
    match layout {
        ChannelLayout::Mono => Ok(1),
        ChannelLayout::Stereo => Ok(2),
        ChannelLayout::Interleaved { labels } => u16::try_from(labels.len())
            .ok()
            .filter(|count| *count > 0)
            .ok_or_else(|| invalid("interleaved channel layout must have 1..=65535 labels")),
    }
}

fn channel_layout(channels: u16) -> ChannelLayout {
    match channels {
        1 => ChannelLayout::Mono,
        2 => ChannelLayout::Stereo,
        count => ChannelLayout::Interleaved {
            labels: (0..count).map(|index| format!("channel_{index}")).collect(),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use speaking::{StreamId, StreamSource};

    use super::*;
    use crate::{bounded_audio_input, AudioBuffer, AudioSourceKind, PushedAudioChunk};

    fn descriptor() -> AudioSourceDescriptor {
        AudioSourceDescriptor {
            id: "fixture".into(),
            kind: AudioSourceKind::Fixture,
            source: StreamSource::Replay {
                source_stream_id: StreamId("source".into()),
            },
            decoded_format: AudioFormat {
                encoding: AudioEncoding::PcmF32Le,
                sample_rate_hz: 48_000,
                channels: ChannelLayout::Stereo,
            },
            live: false,
            seekable: true,
            metadata: BTreeMap::new(),
        }
    }

    fn chunk(sequence: u64, start_frame: u64) -> PushedAudioChunk {
        PushedAudioChunk {
            sequence,
            start_frame: Some(start_frame),
            audio: AudioBuffer {
                samples: vec![0.5, -0.5, 1.0, -1.0, 0.25, -0.25],
                sample_rate_hz: 48_000,
                channels: 2,
            },
        }
    }

    #[test]
    fn format_conversion_is_deterministic_and_retains_source_metadata() {
        let (sender, source) = bounded_audio_input(descriptor(), 2).unwrap();
        sender.try_send(chunk(0, 0)).unwrap();
        let mut normalized = NormalizedAudioSource::new(source, 16_000, 1).unwrap();
        assert_eq!(
            normalized.descriptor().metadata["source_sample_rate_hz"],
            "48000"
        );
        let AudioSourceEvent::Audio(output) = normalized.next_event().unwrap() else {
            panic!("expected normalized audio");
        };
        assert_eq!(output.audio.sample_rate_hz, 16_000);
        assert_eq!(output.audio.channels, 1);
        assert_eq!(output.audio.frames(), 1);
        assert_eq!(output.start_frame, Some(0));
        assert_eq!(output.audio.samples, vec![0.0]);
    }

    #[test]
    fn frame_timeline_gaps_are_surfaced_even_with_contiguous_sequences() {
        let (sender, source) = bounded_audio_input(descriptor(), 2).unwrap();
        sender.try_send(chunk(0, 0)).unwrap();
        sender.try_send(chunk(1, 10)).unwrap();
        let mut normalized = NormalizedAudioSource::new(source, 16_000, 1).unwrap();
        assert!(matches!(
            normalized.next_event().unwrap(),
            AudioSourceEvent::Audio(_)
        ));
        assert!(matches!(
            normalized.next_event().unwrap(),
            AudioSourceEvent::Discontinuity(AudioDiscontinuity { reason, .. })
                if reason.contains("expected frame 3, received 10")
        ));
        assert!(matches!(
            normalized.next_event().unwrap(),
            AudioSourceEvent::Audio(SourceAudioChunk {
                sequence: 1,
                start_frame: Some(3),
                ..
            })
        ));
    }
}
