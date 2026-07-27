//! Streaming utterance segmentation with bounded pre-roll and explicit endpoints.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};
use speaking::{AudioDirection, SegmentId, StreamEvent};

use crate::{invalid, AudioBuffer, Result, VadBackendKind, VadDecision};

pub const DEFAULT_VAD_FRAME_MS: u64 = 10;
pub const DEFAULT_CONVERSATIONAL_SILENCE_MS: u64 = 800;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentationConfig {
    pub frame_ms: u64,
    pub speech_start_ms: u64,
    pub acoustic_end_silence_ms: u64,
    pub segment_end_silence_ms: u64,
    pub minimum_speech_ms: u64,
    pub pre_roll_ms: u64,
    pub maximum_segment_ms: u64,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            frame_ms: DEFAULT_VAD_FRAME_MS,
            speech_start_ms: 30,
            acoustic_end_silence_ms: 300,
            segment_end_silence_ms: DEFAULT_CONVERSATIONAL_SILENCE_MS,
            minimum_speech_ms: 250,
            pre_roll_ms: 200,
            maximum_segment_ms: 30_000,
        }
    }
}

impl SegmentationConfig {
    pub fn validate(&self) -> Result<()> {
        if self.frame_ms == 0 {
            return Err(invalid("segmentation frame_ms must be positive"));
        }
        if self.speech_start_ms == 0 {
            return Err(invalid("segmentation speech_start_ms must be positive"));
        }
        if self.acoustic_end_silence_ms == 0 {
            return Err(invalid(
                "segmentation acoustic_end_silence_ms must be positive",
            ));
        }
        if self.segment_end_silence_ms < self.acoustic_end_silence_ms {
            return Err(invalid(
                "segment_end_silence_ms cannot be shorter than acoustic_end_silence_ms",
            ));
        }
        if self.maximum_segment_ms < self.frame_ms {
            return Err(invalid(
                "segmentation maximum_segment_ms must hold at least one frame",
            ));
        }
        Ok(())
    }

    fn speech_start_frames(&self) -> usize {
        duration_frames(self.speech_start_ms, self.frame_ms).max(1)
    }

    fn acoustic_end_frames(&self) -> usize {
        duration_frames(self.acoustic_end_silence_ms, self.frame_ms).max(1)
    }

    fn segment_end_frames(&self) -> usize {
        duration_frames(self.segment_end_silence_ms, self.frame_ms).max(1)
    }

    fn pre_roll_frames(&self) -> usize {
        duration_frames(self.pre_roll_ms, self.frame_ms)
    }

    fn maximum_segment_frames(&self) -> usize {
        duration_frames(self.maximum_segment_ms, self.frame_ms).max(1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryEvidenceKind {
    Vad { backend: VadBackendKind },
    Authoritative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoritativeBoundary {
    Speech,
    Silence,
    SegmentEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentCloseReason {
    Silence,
    MaximumDuration,
    AuthoritativeBoundary,
    ForcedFlush,
    EndOfStream,
    Cancelled,
    Discontinuity,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentationFrame {
    pub sequence: u64,
    pub start_frame: u64,
    pub audio: AudioBuffer,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioSegment {
    pub id: SegmentId,
    pub frames: Vec<SegmentationFrame>,
    pub close_reason: SegmentCloseReason,
    pub speech_frames: usize,
    pub pre_roll_frames: usize,
    pub post_roll_frames: usize,
    pub speech_duration_ms: u64,
    pub total_duration_ms: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SegmentationEvent {
    SegmentOpened {
        segment_id: SegmentId,
        first_frame_sequence: u64,
        pre_roll_frames: usize,
        initial_frames: Vec<SegmentationFrame>,
        evidence: BoundaryEvidenceKind,
    },
    SpeechStarted {
        segment_id: SegmentId,
        frame_sequence: u64,
    },
    SpeechResumed {
        segment_id: SegmentId,
        frame_sequence: u64,
    },
    SegmentUpdated {
        segment_id: SegmentId,
        frame: SegmentationFrame,
        speech_probability: Option<f32>,
        is_speech: bool,
        evidence: BoundaryEvidenceKind,
    },
    SpeechEnded {
        segment_id: SegmentId,
        frame_sequence: u64,
        endpoint_latency_ms: u64,
    },
    SegmentFinalized(AudioSegment),
    SegmentDropped(AudioSegment),
}

impl SegmentationEvent {
    /// Project segmentation lifecycle into the shared #115 stream contract.
    pub fn stream_event(&self) -> Option<StreamEvent> {
        match self {
            Self::SegmentOpened { segment_id, .. } => Some(StreamEvent::SpeechStarted {
                segment_id: segment_id.clone(),
            }),
            Self::SegmentUpdated {
                segment_id, frame, ..
            } => Some(StreamEvent::AudioChunk {
                direction: AudioDirection::Input,
                chunk_sequence: frame.sequence,
                frame_count: u32::try_from(frame.audio.frames()).unwrap_or(u32::MAX),
                segment_id: Some(segment_id.clone()),
                format: None,
                audio_base64: None,
                metadata: BTreeMap::from([(
                    "start_frame".into(),
                    serde_json::Value::from(frame.start_frame),
                )]),
            }),
            Self::SpeechEnded {
                segment_id,
                endpoint_latency_ms,
                ..
            } => Some(StreamEvent::SpeechEnded {
                segment_id: segment_id.clone(),
                reason: format!("acoustic_silence:{endpoint_latency_ms}ms"),
            }),
            Self::SegmentFinalized(segment) | Self::SegmentDropped(segment) => {
                Some(StreamEvent::DerivedArtifact {
                    stage: "segmentation.segment_final".into(),
                    artifact_id: segment.id.0.clone(),
                    value: serde_json::json!({
                        "close_reason": segment.close_reason,
                        "speech_frames": segment.speech_frames,
                        "pre_roll_frames": segment.pre_roll_frames,
                        "post_roll_frames": segment.post_roll_frames,
                        "speech_duration_ms": segment.speech_duration_ms,
                        "total_duration_ms": segment.total_duration_ms,
                        "accepted": matches!(self, Self::SegmentFinalized(_)),
                    }),
                })
            }
            Self::SpeechStarted { .. } | Self::SpeechResumed { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SegmentationMetrics {
    pub frames_observed: u64,
    pub speech_frames: u64,
    pub segments_opened: u64,
    pub segments_finalized: u64,
    pub segments_dropped: u64,
    pub dropped_source_chunks: u64,
    pub forced_flushes: u64,
    pub endpoint_count: u64,
    pub endpoint_latency_total_ms: u64,
    pub endpoint_latency_max_ms: u64,
}

impl SegmentationMetrics {
    pub fn speech_ratio(&self) -> f64 {
        if self.frames_observed == 0 {
            0.0
        } else {
            self.speech_frames as f64 / self.frames_observed as f64
        }
    }

    pub fn mean_endpoint_latency_ms(&self) -> f64 {
        if self.endpoint_count == 0 {
            0.0
        } else {
            self.endpoint_latency_total_ms as f64 / self.endpoint_count as f64
        }
    }
}

#[derive(Debug)]
struct ActiveSegment {
    id: SegmentId,
    frames: Vec<SegmentationFrame>,
    pre_roll_frames: usize,
    speech_frames: usize,
    last_speech_len: usize,
    consecutive_silence: usize,
    acoustic_ended: bool,
}

#[derive(Debug)]
pub struct UtteranceSegmenter {
    stream_id: String,
    config: SegmentationConfig,
    next_segment: u64,
    pre_roll: VecDeque<SegmentationFrame>,
    pending_speech: Vec<SegmentationFrame>,
    active: Option<ActiveSegment>,
    metrics: SegmentationMetrics,
}

impl UtteranceSegmenter {
    pub fn new(stream_id: impl Into<String>, config: SegmentationConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            stream_id: stream_id.into(),
            config,
            next_segment: 0,
            pre_roll: VecDeque::new(),
            pending_speech: Vec::new(),
            active: None,
            metrics: SegmentationMetrics::default(),
        })
    }

    pub fn metrics(&self) -> &SegmentationMetrics {
        &self.metrics
    }

    pub fn frame_ms(&self) -> u64 {
        self.config.frame_ms
    }

    pub fn buffered_frames(&self) -> usize {
        self.pre_roll.len()
            + self.pending_speech.len()
            + self.active.as_ref().map_or(0, |active| active.frames.len())
    }

    pub fn process_vad(
        &mut self,
        frame: SegmentationFrame,
        decision: VadDecision,
    ) -> Result<Vec<SegmentationEvent>> {
        if decision.rms.is_nan()
            || !decision.speech_probability.is_finite()
            || !(0.0..=1.0).contains(&decision.speech_probability)
        {
            return Err(invalid("VAD decision contains invalid finite/range data"));
        }
        self.process(
            frame,
            decision.is_speech,
            Some(decision.speech_probability),
            BoundaryEvidenceKind::Vad {
                backend: decision.backend,
            },
            false,
        )
    }

    pub fn process_authoritative(
        &mut self,
        frame: SegmentationFrame,
        boundary: AuthoritativeBoundary,
    ) -> Result<Vec<SegmentationEvent>> {
        if boundary == AuthoritativeBoundary::SegmentEnd {
            let mut events = Vec::new();
            if self.active.is_some() {
                events.push(self.close_active(SegmentCloseReason::AuthoritativeBoundary));
            } else if !self.pending_speech.is_empty() {
                events.push(self.drop_pending(SegmentCloseReason::AuthoritativeBoundary));
            }
            self.push_pre_roll(frame);
            return Ok(events);
        }
        self.process(
            frame,
            boundary == AuthoritativeBoundary::Speech,
            None,
            BoundaryEvidenceKind::Authoritative,
            true,
        )
    }

    pub fn note_source_discontinuity(
        &mut self,
        expected_chunk_sequence: u64,
        received_chunk_sequence: u64,
    ) -> Vec<SegmentationEvent> {
        self.metrics.dropped_source_chunks = self.metrics.dropped_source_chunks.saturating_add(
            received_chunk_sequence
                .saturating_sub(expected_chunk_sequence)
                .max(1),
        );
        self.pre_roll.clear();
        self.pending_speech.clear();
        let events = self
            .active
            .take()
            .map(|active| vec![self.finish_segment(active, SegmentCloseReason::Discontinuity)])
            .unwrap_or_default();
        self.pre_roll.clear();
        events
    }

    pub fn force_flush(&mut self, reason: SegmentCloseReason) -> Vec<SegmentationEvent> {
        if self.active.is_some() {
            self.metrics.forced_flushes = self.metrics.forced_flushes.saturating_add(1);
            vec![self.close_active(reason)]
        } else if !self.pending_speech.is_empty() {
            self.metrics.forced_flushes = self.metrics.forced_flushes.saturating_add(1);
            vec![self.drop_pending(reason)]
        } else {
            Vec::new()
        }
    }

    fn process(
        &mut self,
        frame: SegmentationFrame,
        is_speech: bool,
        speech_probability: Option<f32>,
        evidence: BoundaryEvidenceKind,
        authoritative: bool,
    ) -> Result<Vec<SegmentationEvent>> {
        frame.audio.validate()?;
        self.metrics.frames_observed = self.metrics.frames_observed.saturating_add(1);
        if is_speech {
            self.metrics.speech_frames = self.metrics.speech_frames.saturating_add(1);
        }
        let mut events = Vec::new();
        if is_speech {
            if let Some(active) = &mut self.active {
                let resumed = active.acoustic_ended;
                active.frames.push(frame);
                active.speech_frames = active.speech_frames.saturating_add(1);
                active.last_speech_len = active.frames.len();
                active.consecutive_silence = 0;
                active.acoustic_ended = false;
                if resumed {
                    events.push(SegmentationEvent::SpeechResumed {
                        segment_id: active.id.clone(),
                        frame_sequence: active.frames.last().expect("just pushed").sequence,
                    });
                }
                events.push(update_event(active, speech_probability, evidence));
            } else {
                self.pending_speech.push(frame);
                let open_frames = if authoritative {
                    1
                } else {
                    self.config.speech_start_frames()
                };
                if self.pending_speech.len() >= open_frames {
                    events.extend(self.open_active(speech_probability, evidence));
                }
            }
        } else if let Some(active) = &mut self.active {
            active.frames.push(frame);
            active.consecutive_silence = active.consecutive_silence.saturating_add(1);
            events.push(update_event(active, speech_probability, evidence));
            if !active.acoustic_ended
                && active.consecutive_silence >= self.config.acoustic_end_frames()
            {
                active.acoustic_ended = true;
                let endpoint_latency_ms = active.consecutive_silence as u64 * self.config.frame_ms;
                self.metrics.endpoint_count = self.metrics.endpoint_count.saturating_add(1);
                self.metrics.endpoint_latency_total_ms = self
                    .metrics
                    .endpoint_latency_total_ms
                    .saturating_add(endpoint_latency_ms);
                self.metrics.endpoint_latency_max_ms = self
                    .metrics
                    .endpoint_latency_max_ms
                    .max(endpoint_latency_ms);
                events.push(SegmentationEvent::SpeechEnded {
                    segment_id: active.id.clone(),
                    frame_sequence: active.frames.last().expect("just pushed").sequence,
                    endpoint_latency_ms,
                });
            }
            if active.consecutive_silence >= self.config.segment_end_frames() {
                events.push(self.close_active(SegmentCloseReason::Silence));
            }
        } else if self.pending_speech.is_empty() {
            self.push_pre_roll(frame);
        } else {
            let pending = std::mem::take(&mut self.pending_speech);
            for pending_frame in pending {
                self.push_pre_roll(pending_frame);
            }
            self.push_pre_roll(frame);
        }

        if self.active.as_ref().is_some_and(|active| {
            active.frames.len().saturating_sub(active.pre_roll_frames)
                >= self.config.maximum_segment_frames()
        }) {
            events.push(self.close_active(SegmentCloseReason::MaximumDuration));
            self.pre_roll.clear();
        }
        Ok(events)
    }

    fn open_active(
        &mut self,
        speech_probability: Option<f32>,
        evidence: BoundaryEvidenceKind,
    ) -> Vec<SegmentationEvent> {
        let id = SegmentId(format!("{}:segment:{}", self.stream_id, self.next_segment));
        self.next_segment = self.next_segment.saturating_add(1);
        let pre_roll_frames = self.pre_roll.len();
        let speech_frames = self.pending_speech.len();
        let first_frame_sequence = self
            .pre_roll
            .front()
            .or_else(|| self.pending_speech.first())
            .expect("opening requires pending speech")
            .sequence;
        let speech_start_sequence = self
            .pending_speech
            .first()
            .expect("opening requires pending speech")
            .sequence;
        let mut frames = self.pre_roll.drain(..).collect::<Vec<_>>();
        frames.append(&mut self.pending_speech);
        let mut active = ActiveSegment {
            id: id.clone(),
            last_speech_len: frames.len(),
            frames,
            pre_roll_frames,
            speech_frames,
            consecutive_silence: 0,
            acoustic_ended: false,
        };
        self.metrics.segments_opened = self.metrics.segments_opened.saturating_add(1);
        let events = vec![
            SegmentationEvent::SegmentOpened {
                segment_id: id.clone(),
                first_frame_sequence,
                pre_roll_frames,
                initial_frames: active.frames.clone(),
                evidence,
            },
            SegmentationEvent::SpeechStarted {
                segment_id: id,
                frame_sequence: speech_start_sequence,
            },
            update_event(&active, speech_probability, evidence),
        ];
        active.last_speech_len = active.frames.len();
        self.active = Some(active);
        events
    }

    fn close_active(&mut self, reason: SegmentCloseReason) -> SegmentationEvent {
        let active = self.active.take().expect("active segment exists");
        self.finish_segment(active, reason)
    }

    fn finish_segment(
        &mut self,
        active: ActiveSegment,
        reason: SegmentCloseReason,
    ) -> SegmentationEvent {
        let post_roll_frames = active.frames.len().saturating_sub(active.last_speech_len);
        let speech_duration_ms = active.speech_frames as u64 * self.config.frame_ms;
        let segment = AudioSegment {
            id: active.id,
            total_duration_ms: active.frames.len() as u64 * self.config.frame_ms,
            frames: active.frames,
            close_reason: reason,
            speech_frames: active.speech_frames,
            pre_roll_frames: active.pre_roll_frames,
            post_roll_frames,
            speech_duration_ms,
        };
        self.seed_pre_roll_from(&segment);
        if speech_duration_ms < self.config.minimum_speech_ms {
            self.metrics.segments_dropped = self.metrics.segments_dropped.saturating_add(1);
            SegmentationEvent::SegmentDropped(segment)
        } else {
            self.metrics.segments_finalized = self.metrics.segments_finalized.saturating_add(1);
            SegmentationEvent::SegmentFinalized(segment)
        }
    }

    fn drop_pending(&mut self, reason: SegmentCloseReason) -> SegmentationEvent {
        let frames = std::mem::take(&mut self.pending_speech);
        let speech_frames = frames.len();
        let id = SegmentId(format!("{}:segment:{}", self.stream_id, self.next_segment));
        self.next_segment = self.next_segment.saturating_add(1);
        self.metrics.segments_dropped = self.metrics.segments_dropped.saturating_add(1);
        SegmentationEvent::SegmentDropped(AudioSegment {
            id,
            total_duration_ms: speech_frames as u64 * self.config.frame_ms,
            frames,
            close_reason: reason,
            speech_frames,
            pre_roll_frames: 0,
            post_roll_frames: 0,
            speech_duration_ms: speech_frames as u64 * self.config.frame_ms,
        })
    }

    fn seed_pre_roll_from(&mut self, segment: &AudioSegment) {
        self.pre_roll.clear();
        for frame in segment
            .frames
            .iter()
            .skip(
                segment
                    .frames
                    .len()
                    .saturating_sub(segment.post_roll_frames),
            )
            .cloned()
        {
            self.push_pre_roll(frame);
        }
    }

    fn push_pre_roll(&mut self, frame: SegmentationFrame) {
        self.pre_roll.push_back(frame);
        while self.pre_roll.len() > self.config.pre_roll_frames() {
            self.pre_roll.pop_front();
        }
    }
}

fn update_event(
    active: &ActiveSegment,
    speech_probability: Option<f32>,
    evidence: BoundaryEvidenceKind,
) -> SegmentationEvent {
    let frame = active.frames.last().expect("active segment has frames");
    SegmentationEvent::SegmentUpdated {
        segment_id: active.id.clone(),
        frame: frame.clone(),
        speech_probability,
        is_speech: active.last_speech_len == active.frames.len(),
        evidence,
    }
}

const fn duration_frames(duration_ms: u64, frame_ms: u64) -> usize {
    if duration_ms == 0 || frame_ms == 0 {
        0
    } else {
        duration_ms.div_ceil(frame_ms) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SegmentationConfig {
        SegmentationConfig {
            minimum_speech_ms: 30,
            maximum_segment_ms: 10_000,
            ..SegmentationConfig::default()
        }
    }

    fn frame(sequence: u64, sample: f32) -> SegmentationFrame {
        SegmentationFrame {
            sequence,
            start_frame: sequence * 160,
            audio: AudioBuffer {
                samples: vec![sample; 160],
                sample_rate_hz: 16_000,
                channels: 1,
            },
        }
    }

    fn decision(speech: bool) -> VadDecision {
        VadDecision {
            backend: VadBackendKind::Energy,
            speech_probability: if speech { 0.9 } else { 0.0 },
            is_speech: speech,
            rms: if speech { 0.1 } else { 0.0 },
        }
    }

    fn feed(
        segmenter: &mut UtteranceSegmenter,
        next_sequence: &mut u64,
        speech: bool,
        count: usize,
    ) -> Vec<SegmentationEvent> {
        let mut events = Vec::new();
        for _ in 0..count {
            events.extend(
                segmenter
                    .process_vad(
                        frame(*next_sequence, if speech { 1.0 } else { 0.0 }),
                        decision(speech),
                    )
                    .unwrap(),
            );
            *next_sequence += 1;
        }
        events
    }

    fn finalized(events: &[SegmentationEvent]) -> Option<&AudioSegment> {
        events.iter().find_map(|event| match event {
            SegmentationEvent::SegmentFinalized(segment) => Some(segment),
            _ => None,
        })
    }

    #[test]
    fn listenbury_open_threshold_is_retained() {
        let mut segmenter = UtteranceSegmenter::new("test", config()).unwrap();
        let mut sequence = 0;
        assert!(!feed(&mut segmenter, &mut sequence, true, 2)
            .iter()
            .any(|event| matches!(event, SegmentationEvent::SegmentOpened { .. })));
        assert!(feed(&mut segmenter, &mut sequence, true, 1)
            .iter()
            .any(|event| matches!(event, SegmentationEvent::SegmentOpened { .. })));
    }

    #[test]
    fn listenbury_short_silence_is_bridged_before_conversational_close() {
        let mut segmenter = UtteranceSegmenter::new("test", config()).unwrap();
        let mut sequence = 0;
        feed(&mut segmenter, &mut sequence, true, 3);
        let mut events = feed(&mut segmenter, &mut sequence, false, 79);
        events.extend(feed(&mut segmenter, &mut sequence, true, 1));
        assert!(!events
            .iter()
            .any(|event| matches!(event, SegmentationEvent::SegmentFinalized(_))));
        assert!(events
            .iter()
            .any(|event| matches!(event, SegmentationEvent::SpeechResumed { .. })));
    }

    #[test]
    fn listenbury_conversational_silence_and_timeout_reasons_are_explicit() {
        let mut segmenter = UtteranceSegmenter::new("silence", config()).unwrap();
        let mut sequence = 0;
        feed(&mut segmenter, &mut sequence, true, 3);
        let events = feed(&mut segmenter, &mut sequence, false, 80);
        assert_eq!(
            finalized(&events).unwrap().close_reason,
            SegmentCloseReason::Silence
        );

        let timeout_config = SegmentationConfig {
            speech_start_ms: 10,
            minimum_speech_ms: 10,
            maximum_segment_ms: 50,
            ..config()
        };
        let mut segmenter = UtteranceSegmenter::new("timeout", timeout_config).unwrap();
        let mut sequence = 0;
        let events = feed(&mut segmenter, &mut sequence, true, 5);
        assert_eq!(
            finalized(&events).unwrap().close_reason,
            SegmentCloseReason::MaximumDuration
        );
    }

    #[test]
    fn pre_roll_and_final_speech_are_retained() {
        let mut segmenter = UtteranceSegmenter::new("roll", config()).unwrap();
        let mut sequence = 0;
        feed(&mut segmenter, &mut sequence, false, 20);
        feed(&mut segmenter, &mut sequence, true, 5);
        let events = feed(&mut segmenter, &mut sequence, false, 80);
        let segment = finalized(&events).unwrap();
        assert_eq!(segment.pre_roll_frames, 20);
        assert_eq!(segment.frames.first().unwrap().sequence, 0);
        assert_eq!(segment.frames[20].sequence, 20);
        assert_eq!(segment.frames[24].sequence, 24);
        assert_eq!(segment.post_roll_frames, 80);
    }

    #[test]
    fn silence_buffer_stays_bounded_and_emits_no_empty_segments() {
        let mut segmenter = UtteranceSegmenter::new("silence", config()).unwrap();
        let mut sequence = 0;
        let events = feed(&mut segmenter, &mut sequence, false, 10_000);
        assert!(events.is_empty());
        assert_eq!(segmenter.buffered_frames(), 20);
    }

    #[test]
    fn long_continuous_speech_is_chunked_without_losing_order() {
        let chunked = SegmentationConfig {
            speech_start_ms: 10,
            minimum_speech_ms: 10,
            pre_roll_ms: 0,
            maximum_segment_ms: 50,
            ..config()
        };
        let mut segmenter = UtteranceSegmenter::new("long", chunked).unwrap();
        let mut sequence = 0;
        let mut events = feed(&mut segmenter, &mut sequence, true, 12);
        events.extend(segmenter.force_flush(SegmentCloseReason::EndOfStream));
        let sequences = events
            .iter()
            .filter_map(|event| match event {
                SegmentationEvent::SegmentFinalized(segment) => Some(segment),
                _ => None,
            })
            .flat_map(|segment| segment.frames.iter().map(|frame| frame.sequence))
            .collect::<Vec<_>>();
        assert_eq!(sequences, (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn authoritative_boundaries_bypass_vad_thresholds() {
        let mut segmenter = UtteranceSegmenter::new("provider", config()).unwrap();
        let events = segmenter
            .process_authoritative(frame(0, 1.0), AuthoritativeBoundary::Speech)
            .unwrap();
        assert!(events
            .iter()
            .any(|event| matches!(event, SegmentationEvent::SegmentOpened { .. })));
        let events = segmenter
            .process_authoritative(frame(1, 0.0), AuthoritativeBoundary::SegmentEnd)
            .unwrap();
        assert!(events
            .iter()
            .any(|event| matches!(event, SegmentationEvent::SegmentDropped(_))));
    }

    #[test]
    fn metrics_report_speech_ratio_endpoint_latency_drops_and_flushes() {
        let mut segmenter = UtteranceSegmenter::new("metrics", config()).unwrap();
        let mut sequence = 0;
        feed(&mut segmenter, &mut sequence, true, 3);
        feed(&mut segmenter, &mut sequence, false, 30);
        segmenter.force_flush(SegmentCloseReason::ForcedFlush);
        segmenter.note_source_discontinuity(4, 7);
        assert_eq!(segmenter.metrics().frames_observed, 33);
        assert!((segmenter.metrics().speech_ratio() - 3.0 / 33.0).abs() < 1.0e-9);
        assert_eq!(segmenter.metrics().mean_endpoint_latency_ms(), 300.0);
        assert_eq!(segmenter.metrics().dropped_source_chunks, 3);
        assert_eq!(segmenter.metrics().forced_flushes, 1);
    }
}
