//! One file/live pipeline from decoded source chunks through VAD and segmentation.

use std::collections::VecDeque;

use speaking::ChannelLayout;

use crate::{
    invalid, AudioBuffer, AudioDiscontinuity, AudioSource, AudioSourceEvent, Result,
    SegmentCloseReason, SegmentationEvent, SegmentationFrame, SegmentationMetrics,
    UtteranceSegmenter, VoiceActivityDetector,
};

#[derive(Debug, Clone, PartialEq)]
pub enum VadPipelineEvent {
    Segmentation(SegmentationEvent),
    SourceDiscontinuity(AudioDiscontinuity),
    EndOfStream { metrics: SegmentationMetrics },
}

pub struct VadSegmentationPipeline<S, D> {
    source: S,
    detector: D,
    segmenter: UtteranceSegmenter,
    sample_rate_hz: u32,
    channels: u16,
    samples_per_vad_frame: usize,
    pending_samples: Vec<f32>,
    pending_start_frame: Option<u64>,
    next_frame_sequence: u64,
    expected_source_chunk_sequence: u64,
    queued: VecDeque<VadPipelineEvent>,
    ended: bool,
}

impl<S: AudioSource, D: VoiceActivityDetector> VadSegmentationPipeline<S, D> {
    pub fn new(source: S, detector: D, segmenter: UtteranceSegmenter) -> Result<Self> {
        let format = &source.descriptor().decoded_format;
        let channels = channel_count(&format.channels)?;
        let sample_rate_hz = format.sample_rate_hz;
        let frame_numerator = u128::from(sample_rate_hz) * u128::from(segmenter.frame_ms());
        if !frame_numerator.is_multiple_of(1000) {
            return Err(invalid(format!(
                "{} Hz cannot form exact {} ms VAD frames; normalize the source first",
                sample_rate_hz,
                segmenter.frame_ms()
            )));
        }
        let frames_per_vad_frame = usize::try_from(frame_numerator / 1000)
            .map_err(|_| invalid("VAD frame geometry exceeds this platform"))?;
        if frames_per_vad_frame == 0 {
            return Err(invalid("VAD frames must contain audio"));
        }
        Ok(Self {
            source,
            detector,
            segmenter,
            sample_rate_hz,
            channels,
            samples_per_vad_frame: frames_per_vad_frame * usize::from(channels),
            pending_samples: Vec::new(),
            pending_start_frame: None,
            next_frame_sequence: 0,
            expected_source_chunk_sequence: 0,
            queued: VecDeque::new(),
            ended: false,
        })
    }

    pub fn descriptor(&self) -> &crate::AudioSourceDescriptor {
        self.source.descriptor()
    }

    pub fn metrics(&self) -> &SegmentationMetrics {
        self.segmenter.metrics()
    }

    pub fn next_event(&mut self) -> Result<Option<VadPipelineEvent>> {
        loop {
            if let Some(event) = self.queued.pop_front() {
                return Ok(Some(event));
            }
            if self.ended {
                return Ok(None);
            }
            if self.pending_samples.len() >= self.samples_per_vad_frame {
                self.emit_complete_vad_frame()?;
                continue;
            }
            match self.source.next_event()? {
                AudioSourceEvent::Audio(chunk) => self.accept_source_chunk(chunk)?,
                AudioSourceEvent::Discontinuity(gap) => {
                    self.pending_samples.clear();
                    self.pending_start_frame = None;
                    for event in self.segmenter.note_source_discontinuity(
                        gap.expected_chunk_sequence,
                        gap.received_chunk_sequence,
                    ) {
                        self.queued.push_back(VadPipelineEvent::Segmentation(event));
                    }
                    self.expected_source_chunk_sequence = gap.received_chunk_sequence;
                    self.queued
                        .push_back(VadPipelineEvent::SourceDiscontinuity(gap));
                }
                AudioSourceEvent::EndOfStream => self.finish_source()?,
            }
        }
    }

    pub fn cancel(&mut self) {
        self.source.cancel();
        self.pending_samples.clear();
        self.pending_start_frame = None;
        for event in self.segmenter.force_flush(SegmentCloseReason::Cancelled) {
            self.queued.push_back(VadPipelineEvent::Segmentation(event));
        }
        self.queued.push_back(VadPipelineEvent::EndOfStream {
            metrics: self.segmenter.metrics().clone(),
        });
        self.ended = true;
    }

    fn accept_source_chunk(&mut self, chunk: crate::SourceAudioChunk) -> Result<()> {
        if chunk.audio.sample_rate_hz != self.sample_rate_hz
            || chunk.audio.channels != self.channels
        {
            return Err(invalid(format!(
                "audio source changed format without renegotiation: expected {} Hz/{} channels, received {} Hz/{} channels",
                self.sample_rate_hz,
                self.channels,
                chunk.audio.sample_rate_hz,
                chunk.audio.channels
            )));
        }
        if chunk.sequence != self.expected_source_chunk_sequence {
            let gap = AudioDiscontinuity {
                expected_chunk_sequence: self.expected_source_chunk_sequence,
                received_chunk_sequence: chunk.sequence,
                reason: "source chunk sequence gap reached the VAD pipeline".into(),
            };
            for event in self
                .segmenter
                .note_source_discontinuity(gap.expected_chunk_sequence, gap.received_chunk_sequence)
            {
                self.queued.push_back(VadPipelineEvent::Segmentation(event));
            }
            self.pending_samples.clear();
            self.pending_start_frame = None;
            self.queued
                .push_back(VadPipelineEvent::SourceDiscontinuity(gap));
        }
        self.expected_source_chunk_sequence = chunk.sequence.saturating_add(1);

        if let (Some(pending_start), Some(received_start)) =
            (self.pending_start_frame, chunk.start_frame)
        {
            let pending_frames = self.pending_samples.len() / usize::from(self.channels);
            let expected_start = pending_start.saturating_add(pending_frames as u64);
            if expected_start != received_start {
                let gap = AudioDiscontinuity {
                    expected_chunk_sequence: chunk.sequence,
                    received_chunk_sequence: chunk.sequence,
                    reason: format!(
                        "source frame timeline gap: expected frame {expected_start}, received {received_start}"
                    ),
                };
                for event in self
                    .segmenter
                    .note_source_discontinuity(chunk.sequence, chunk.sequence)
                {
                    self.queued.push_back(VadPipelineEvent::Segmentation(event));
                }
                self.pending_samples.clear();
                self.pending_start_frame = None;
                self.queued
                    .push_back(VadPipelineEvent::SourceDiscontinuity(gap));
            }
        }
        if self.pending_samples.is_empty() {
            self.pending_start_frame = chunk.start_frame;
        }
        self.pending_samples.extend(chunk.audio.samples);
        Ok(())
    }

    fn emit_complete_vad_frame(&mut self) -> Result<()> {
        let samples = self
            .pending_samples
            .drain(..self.samples_per_vad_frame)
            .collect::<Vec<_>>();
        let start_frame = self.pending_start_frame.unwrap_or_else(|| {
            self.next_frame_sequence
                .saturating_mul((self.samples_per_vad_frame / usize::from(self.channels)) as u64)
        });
        self.pending_start_frame = Some(
            start_frame
                .saturating_add((self.samples_per_vad_frame / usize::from(self.channels)) as u64),
        );
        self.emit_vad_frame(samples, start_frame)
    }

    fn emit_vad_frame(&mut self, samples: Vec<f32>, start_frame: u64) -> Result<()> {
        let audio = AudioBuffer {
            samples,
            sample_rate_hz: self.sample_rate_hz,
            channels: self.channels,
        };
        let decision = self.detector.process_frame(&audio)?;
        let frame = SegmentationFrame {
            sequence: self.next_frame_sequence,
            start_frame,
            audio,
        };
        self.next_frame_sequence = self.next_frame_sequence.saturating_add(1);
        for event in self.segmenter.process_vad(frame, decision)? {
            self.queued.push_back(VadPipelineEvent::Segmentation(event));
        }
        Ok(())
    }

    fn finish_source(&mut self) -> Result<()> {
        if !self.pending_samples.is_empty() {
            let start_frame = self.pending_start_frame.unwrap_or_else(|| {
                self.next_frame_sequence.saturating_mul(
                    (self.samples_per_vad_frame / usize::from(self.channels)) as u64,
                )
            });
            self.pending_samples.resize(self.samples_per_vad_frame, 0.0);
            let samples = std::mem::take(&mut self.pending_samples);
            self.emit_vad_frame(samples, start_frame)?;
        }
        for event in self.segmenter.force_flush(SegmentCloseReason::EndOfStream) {
            self.queued.push_back(VadPipelineEvent::Segmentation(event));
        }
        self.queued.push_back(VadPipelineEvent::EndOfStream {
            metrics: self.segmenter.metrics().clone(),
        });
        self.ended = true;
        Ok(())
    }
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::Cursor;

    use speaking::{AudioEncoding, AudioFormat, StreamId, StreamSource};

    use super::*;
    use crate::{
        bounded_audio_input, AudioSourceDescriptor, AudioSourceKind, EnergyVad, PushedAudioChunk,
        SegmentationConfig, WavAudioSource,
    };

    fn segmentation_config() -> SegmentationConfig {
        SegmentationConfig {
            speech_start_ms: 10,
            acoustic_end_silence_ms: 20,
            segment_end_silence_ms: 30,
            minimum_speech_ms: 20,
            pre_roll_ms: 20,
            maximum_segment_ms: 2_000,
            ..SegmentationConfig::default()
        }
    }

    fn fixture_samples() -> Vec<f32> {
        let mut samples = vec![0.0; 320];
        samples.extend((0..640).map(|index| if index % 2 == 0 { 0.1 } else { -0.1 }));
        samples.extend(vec![0.0; 480]);
        samples
    }

    fn wav_bytes(samples: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let cursor = Cursor::new(&mut bytes);
            let mut writer = hound::WavWriter::new(
                cursor,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 16_000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();
            for sample in samples {
                writer.write_sample(*sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        bytes
    }

    fn live_descriptor() -> AudioSourceDescriptor {
        AudioSourceDescriptor {
            id: "live-fixture".into(),
            kind: AudioSourceKind::Fixture,
            source: StreamSource::Replay {
                source_stream_id: StreamId("fixture".into()),
            },
            decoded_format: AudioFormat {
                encoding: AudioEncoding::PcmF32Le,
                sample_rate_hz: 16_000,
                channels: ChannelLayout::Mono,
            },
            live: true,
            seekable: false,
            metadata: BTreeMap::new(),
        }
    }

    fn collect_segments<S: AudioSource>(
        source: S,
    ) -> (Vec<crate::AudioSegment>, SegmentationMetrics) {
        let segmenter = UtteranceSegmenter::new("fixture", segmentation_config()).unwrap();
        let mut pipeline =
            VadSegmentationPipeline::new(source, EnergyVad::default(), segmenter).unwrap();
        let mut segments = Vec::new();
        let metrics = loop {
            match pipeline.next_event().unwrap() {
                Some(VadPipelineEvent::Segmentation(SegmentationEvent::SegmentFinalized(
                    segment,
                ))) => segments.push(segment),
                Some(VadPipelineEvent::EndOfStream { metrics }) => break metrics,
                Some(_) => {}
                None => panic!("pipeline ended without end-of-stream metrics"),
            }
        };
        (segments, metrics)
    }

    #[test]
    fn file_and_live_sources_use_the_same_segmentation_pipeline() {
        let samples = fixture_samples();
        let wav = WavAudioSource::from_bytes("wav-fixture", &wav_bytes(&samples), 317).unwrap();
        let (file_segments, file_metrics) = collect_segments(wav);

        let (sender, live) = bounded_audio_input(live_descriptor(), 4).unwrap();
        let mut start = 0usize;
        for (sequence, frames) in [173usize, 509, 758].into_iter().enumerate() {
            let end = (start + frames).min(samples.len());
            sender
                .try_send(PushedAudioChunk {
                    sequence: sequence as u64,
                    start_frame: Some(start as u64),
                    audio: AudioBuffer {
                        samples: samples[start..end].to_vec(),
                        sample_rate_hz: 16_000,
                        channels: 1,
                    },
                })
                .unwrap();
            start = end;
        }
        sender.end().unwrap();
        let (live_segments, live_metrics) = collect_segments(live);

        assert_eq!(file_segments.len(), 1);
        assert_eq!(live_segments.len(), 1);
        assert_eq!(
            file_segments[0].frames.first().unwrap().start_frame,
            live_segments[0].frames.first().unwrap().start_frame
        );
        assert_eq!(
            file_segments[0].speech_duration_ms,
            live_segments[0].speech_duration_ms
        );
        assert_eq!(file_metrics.frames_observed, live_metrics.frames_observed);
    }

    #[test]
    fn source_gaps_are_emitted_and_counted() {
        let (sender, live) = bounded_audio_input(live_descriptor(), 2).unwrap();
        sender
            .try_send(PushedAudioChunk {
                sequence: 2,
                start_frame: Some(0),
                audio: AudioBuffer {
                    samples: vec![0.0; 160],
                    sample_rate_hz: 16_000,
                    channels: 1,
                },
            })
            .unwrap();
        sender.end().unwrap();
        let segmenter = UtteranceSegmenter::new("gap", segmentation_config()).unwrap();
        let mut pipeline =
            VadSegmentationPipeline::new(live, EnergyVad::default(), segmenter).unwrap();
        assert!(matches!(
            pipeline.next_event().unwrap(),
            Some(VadPipelineEvent::SourceDiscontinuity(AudioDiscontinuity {
                expected_chunk_sequence: 0,
                received_chunk_sequence: 2,
                ..
            }))
        ));
        loop {
            if let Some(VadPipelineEvent::EndOfStream { metrics }) = pipeline.next_event().unwrap()
            {
                assert_eq!(metrics.dropped_source_chunks, 2);
                break;
            }
        }
    }
}
