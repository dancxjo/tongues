//! Versioned, provider-neutral streaming audio and recognition events.
//!
//! This is the wire contract shared by recognizers, command-line JSONL output,
//! the server, and browser clients. Provider-specific values belong in
//! [`Provenance`] and [`Confidence`], never in alternate event shapes.

use std::collections::{BTreeMap, VecDeque};
use std::io::{BufRead, Write};

use serde::{Deserialize, Serialize};

pub const STREAM_EVENT_SCHEMA_V1: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StreamId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EventId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SegmentId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct UtteranceEventId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ClockOrigin {
    UnixEpoch,
    StreamStart,
    MediaTimeline { media_id: String },
    Provider { provider: String, clock: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventTime {
    pub origin: ClockOrigin,
    /// Milliseconds from `origin`. This is signed to support pre-roll.
    pub offset_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventTimes {
    /// When the represented real-world or media event occurred.
    pub occurred_at: EventTime,
    /// When Tongues observed or received the event.
    pub observed_at: EventTime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceKind {
    Direct,
    Derived,
    Recalled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventRef {
    pub stream_id: StreamId,
    pub event_id: EventId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub kind: ProvenanceKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<EventRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attributes: BTreeMap<String, serde_json::Value>,
}

impl Provenance {
    pub fn direct() -> Self {
        Self {
            kind: ProvenanceKind::Direct,
            sources: Vec::new(),
            provider: None,
            model: None,
            attributes: BTreeMap::new(),
        }
    }

    pub fn derived_from(sources: Vec<EventRef>) -> Self {
        Self {
            kind: ProvenanceKind::Derived,
            sources,
            ..Self::direct()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ConfidenceScale {
    Probability,
    LogProbability,
    ProviderNative {
        provider: String,
        metric: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        range: Option<ScoreRange>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreRange {
    pub minimum: String,
    pub maximum: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Confidence {
    pub value: f64,
    pub scale: ConfidenceScale,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioEncoding {
    PcmS16Le,
    PcmF32Le,
    Wav,
    Flac,
    Opus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ChannelLayout {
    Mono,
    Stereo,
    Interleaved { labels: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioFormat {
    pub encoding: AudioEncoding,
    pub sample_rate_hz: u32,
    pub channels: ChannelLayout,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum StreamSource {
    Live { device: Option<String> },
    File { path: String },
    Replay { source_stream_id: StreamId },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextRange {
    /// Start in Unicode scalar values, inclusive.
    pub start: u32,
    /// End in Unicode scalar values, exclusive.
    pub end: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeRange {
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimedToken {
    pub text: String,
    pub range: TimeRange,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<Confidence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageHypothesis {
    pub language: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<Confidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextRole {
    Recognition,
    Generation,
    Normalized,
    Parse,
    Interpretation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioDirection {
    Input,
    Output,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "data")]
pub enum StreamEvent {
    SessionStarted {
        purpose: String,
    },
    StreamOpened {
        source: StreamSource,
        format: AudioFormat,
        clock: ClockOrigin,
    },
    AudioChunk {
        direction: AudioDirection,
        chunk_sequence: u64,
        frame_count: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        segment_id: Option<SegmentId>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<AudioFormat>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        audio_base64: Option<String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        metadata: BTreeMap<String, serde_json::Value>,
    },
    Discontinuity {
        expected_chunk_sequence: u64,
        received_chunk_sequence: u64,
        reason: String,
    },
    EndOfStream,
    SpeechStarted {
        segment_id: SegmentId,
    },
    SpeechEnded {
        segment_id: SegmentId,
        reason: String,
    },
    PartialHypothesis {
        role: TextRole,
        segment_id: SegmentId,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        confidence: Option<Confidence>,
    },
    RevisedHypothesis {
        role: TextRole,
        segment_id: SegmentId,
        replaces: TextRange,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        confidence: Option<Confidence>,
    },
    CommittedSegment {
        role: TextRole,
        segment_id: SegmentId,
        text: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        words: Vec<TimedToken>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        language: Option<LanguageHypothesis>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        speaker_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        confidence: Option<Confidence>,
    },
    HypothesisCancelled {
        role: TextRole,
        segment_id: SegmentId,
        reason: String,
    },
    TextCompleted {
        role: TextRole,
        text: String,
    },
    WordTiming {
        segment_id: SegmentId,
        word: TimedToken,
    },
    TokenTiming {
        segment_id: SegmentId,
        token: TimedToken,
    },
    LanguageHypothesis {
        segment_id: SegmentId,
        hypothesis: LanguageHypothesis,
    },
    SpeakerAssigned {
        segment_id: SegmentId,
        speaker_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        confidence: Option<Confidence>,
    },
    Warning {
        code: String,
        message: String,
        recoverable: bool,
    },
    Error {
        code: String,
        message: String,
        recoverable: bool,
    },
    Cancelled {
        reason: String,
    },
    Completed,
    OutputRequested {
        utterance_id: UtteranceEventId,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        caused_by: Vec<EventRef>,
    },
    OutputStarted {
        utterance_id: UtteranceEventId,
    },
    OutputInterrupted {
        utterance_id: UtteranceEventId,
    },
    OutputResumed {
        utterance_id: UtteranceEventId,
    },
    OutputAborted {
        utterance_id: UtteranceEventId,
        reason: String,
    },
    OutputFinished {
        utterance_id: UtteranceEventId,
    },
    /// A parser, normalizer, or interpretation projection. The envelope's
    /// provenance must link this artifact to the event(s) it derives from.
    DerivedArtifact {
        stage: String,
        artifact_id: String,
        value: serde_json::Value,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamEventEnvelope {
    pub schema_version: u16,
    pub stream_id: StreamId,
    pub event_id: EventId,
    pub sequence: u64,
    pub times: EventTimes,
    pub provenance: Provenance,
    pub event: StreamEvent,
}

impl StreamEventEnvelope {
    pub fn event_ref(&self) -> EventRef {
        EventRef {
            stream_id: self.stream_id.clone(),
            event_id: self.event_id.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamEventSequencer {
    stream_id: StreamId,
    next_sequence: u64,
}

impl StreamEventSequencer {
    pub fn new(stream_id: impl Into<String>) -> Self {
        Self {
            stream_id: StreamId(stream_id.into()),
            next_sequence: 0,
        }
    }

    pub fn push(
        &mut self,
        event: StreamEvent,
        occurred_at: EventTime,
        provenance: Provenance,
    ) -> StreamEventEnvelope {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        StreamEventEnvelope {
            schema_version: STREAM_EVENT_SCHEMA_V1,
            stream_id: self.stream_id.clone(),
            event_id: EventId(format!("{}:{sequence}", self.stream_id.0)),
            sequence,
            times: EventTimes {
                occurred_at,
                observed_at: EventTime {
                    origin: ClockOrigin::UnixEpoch,
                    offset_ms: unix_time_ms(),
                },
            },
            provenance,
            event,
        }
    }
}

fn unix_time_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or_default()
}

pub fn write_jsonl(
    writer: &mut impl Write,
    events: impl IntoIterator<Item = StreamEventEnvelope>,
) -> anyhow::Result<()> {
    for event in events {
        serde_json::to_writer(&mut *writer, &event)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

pub fn read_jsonl(reader: impl BufRead) -> anyhow::Result<Vec<StreamEventEnvelope>> {
    reader
        .lines()
        .enumerate()
        .filter_map(|(index, line)| match line {
            Ok(line) if line.trim().is_empty() => None,
            line => Some((index, line)),
        })
        .map(|(index, line)| {
            let line = line?;
            serde_json::from_str(&line)
                .map_err(anyhow::Error::from)
                .map_err(|error| anyhow::anyhow!("invalid JSONL record {}: {error}", index + 1))
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamPolicy {
    pub max_buffered_events: usize,
    pub overflow: OverflowBehavior,
    pub malformed: InvalidEventBehavior,
    pub out_of_order: InvalidEventBehavior,
}

impl Default for StreamPolicy {
    fn default() -> Self {
        Self {
            max_buffered_events: 64,
            overflow: OverflowBehavior::Backpressure,
            malformed: InvalidEventBehavior::Reject,
            out_of_order: InvalidEventBehavior::Reject,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverflowBehavior {
    Backpressure,
    Reject,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvalidEventBehavior {
    Reject,
    EmitWarningAndDrop,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StreamContractError {
    #[error("unsupported stream event schema version {0}")]
    UnsupportedVersion(u16),
    #[error("event stream changed from {expected:?} to {received:?}")]
    StreamChanged {
        expected: StreamId,
        received: StreamId,
    },
    #[error("out-of-order event: expected sequence {expected}, received {received}")]
    OutOfOrder { expected: u64, received: u64 },
    #[error("event received after terminal event")]
    EventAfterTerminal,
    #[error("replacement range {start}..{end} is invalid for {text_len} Unicode scalars")]
    InvalidReplacementRange {
        start: u32,
        end: u32,
        text_len: usize,
    },
    #[error("segment {0:?} was committed and cannot be revised")]
    RevisedCommittedSegment(SegmentId),
    #[error("derived provenance must reference at least one source event")]
    MissingDerivation,
    #[error("buffer capacity {0} reached; producer must wait for capacity")]
    Backpressure(usize),
    #[error("buffer capacity {0} reached; event rejected")]
    BufferFull(usize),
    #[error("stream was cancelled")]
    Cancelled,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationOutcome {
    Accepted,
    DroppedWithWarning(StreamEvent),
}

#[derive(Debug, Default)]
pub struct StreamValidator {
    stream_id: Option<StreamId>,
    next_sequence: u64,
    terminal: bool,
    segment_text: BTreeMap<SegmentId, String>,
    committed: BTreeMap<SegmentId, String>,
}

impl StreamValidator {
    pub fn validate_with_policy(
        &mut self,
        envelope: &StreamEventEnvelope,
        policy: &StreamPolicy,
    ) -> Result<ValidationOutcome, StreamContractError> {
        match self.validate(envelope) {
            Ok(()) => Ok(ValidationOutcome::Accepted),
            Err(error) => {
                let behavior = if matches!(error, StreamContractError::OutOfOrder { .. }) {
                    &policy.out_of_order
                } else {
                    &policy.malformed
                };
                match behavior {
                    InvalidEventBehavior::Reject => Err(error),
                    InvalidEventBehavior::EmitWarningAndDrop => Ok(
                        ValidationOutcome::DroppedWithWarning(StreamEvent::Warning {
                            code: "stream_event_dropped".into(),
                            message: error.to_string(),
                            recoverable: true,
                        }),
                    ),
                }
            }
        }
    }

    pub fn validate(&mut self, envelope: &StreamEventEnvelope) -> Result<(), StreamContractError> {
        if envelope.schema_version != STREAM_EVENT_SCHEMA_V1 {
            return Err(StreamContractError::UnsupportedVersion(
                envelope.schema_version,
            ));
        }
        if self.terminal {
            return Err(StreamContractError::EventAfterTerminal);
        }
        if let Some(stream_id) = &self.stream_id {
            if stream_id != &envelope.stream_id {
                return Err(StreamContractError::StreamChanged {
                    expected: stream_id.clone(),
                    received: envelope.stream_id.clone(),
                });
            }
        } else {
            self.stream_id = Some(envelope.stream_id.clone());
            self.next_sequence = envelope.sequence;
        }
        if envelope.sequence != self.next_sequence {
            return Err(StreamContractError::OutOfOrder {
                expected: self.next_sequence,
                received: envelope.sequence,
            });
        }
        if envelope.provenance.kind == ProvenanceKind::Derived
            && envelope.provenance.sources.is_empty()
        {
            return Err(StreamContractError::MissingDerivation);
        }

        match &envelope.event {
            StreamEvent::PartialHypothesis {
                segment_id, text, ..
            } => {
                if self.committed.contains_key(segment_id) {
                    return Err(StreamContractError::RevisedCommittedSegment(
                        segment_id.clone(),
                    ));
                }
                self.segment_text.insert(segment_id.clone(), text.clone());
            }
            StreamEvent::RevisedHypothesis {
                segment_id,
                replaces,
                text,
                ..
            } => {
                if self.committed.contains_key(segment_id) {
                    return Err(StreamContractError::RevisedCommittedSegment(
                        segment_id.clone(),
                    ));
                }
                let current = self
                    .segment_text
                    .get(segment_id)
                    .cloned()
                    .unwrap_or_default();
                let len = current.chars().count();
                let start = replaces.start as usize;
                let end = replaces.end as usize;
                if start > end || end > len {
                    return Err(StreamContractError::InvalidReplacementRange {
                        start: replaces.start,
                        end: replaces.end,
                        text_len: len,
                    });
                }
                let prefix = current.chars().take(start).collect::<String>();
                let suffix = current.chars().skip(end).collect::<String>();
                self.segment_text
                    .insert(segment_id.clone(), format!("{prefix}{text}{suffix}"));
            }
            StreamEvent::CommittedSegment {
                segment_id, text, ..
            } => {
                self.segment_text.insert(segment_id.clone(), text.clone());
                self.committed.insert(segment_id.clone(), text.clone());
            }
            StreamEvent::Cancelled { .. }
            | StreamEvent::Completed
            | StreamEvent::Error {
                recoverable: false, ..
            } => self.terminal = true,
            _ => {}
        }
        self.next_sequence = self.next_sequence.saturating_add(1);
        Ok(())
    }

    pub fn text_for(&self, segment_id: &SegmentId) -> Option<&str> {
        self.segment_text.get(segment_id).map(String::as_str)
    }
}

#[derive(Debug)]
pub struct BoundedEventBuffer {
    policy: StreamPolicy,
    events: VecDeque<StreamEventEnvelope>,
    cancelled: bool,
}

impl BoundedEventBuffer {
    pub fn new(policy: StreamPolicy) -> Self {
        Self {
            policy,
            events: VecDeque::new(),
            cancelled: false,
        }
    }

    pub fn try_push(&mut self, event: StreamEventEnvelope) -> Result<(), StreamContractError> {
        if self.cancelled {
            return Err(StreamContractError::Cancelled);
        }
        if self.events.len() >= self.policy.max_buffered_events {
            return Err(match self.policy.overflow {
                OverflowBehavior::Backpressure => {
                    StreamContractError::Backpressure(self.policy.max_buffered_events)
                }
                OverflowBehavior::Reject => {
                    StreamContractError::BufferFull(self.policy.max_buffered_events)
                }
            });
        }
        self.events.push_back(event);
        Ok(())
    }

    pub fn pop(&mut self) -> Option<StreamEventEnvelope> {
        self.events.pop_front()
    }

    pub fn cancel(&mut self) {
        self.cancelled = true;
        self.events.clear();
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::json;
    use std::collections::BTreeSet;

    fn confidence() -> Confidence {
        Confidence {
            value: 0.72,
            scale: ConfidenceScale::ProviderNative {
                provider: "fixture".into(),
                metric: "decoder_score".into(),
                range: None,
            },
            calibration: Some("not comparable across providers".into()),
        }
    }

    fn token() -> TimedToken {
        TimedToken {
            text: "hello".into(),
            range: TimeRange {
                start_ms: 10,
                end_ms: 90,
            },
            confidence: Some(confidence()),
        }
    }

    fn every_variant() -> Vec<StreamEvent> {
        let segment = SegmentId("segment-1".into());
        let utterance = UtteranceEventId("utterance-1".into());
        let source = EventRef {
            stream_id: StreamId("stream-1".into()),
            event_id: EventId("event-1".into()),
        };
        vec![
            StreamEvent::SessionStarted {
                purpose: "recognition".into(),
            },
            StreamEvent::StreamOpened {
                source: StreamSource::Live { device: None },
                format: AudioFormat {
                    encoding: AudioEncoding::PcmF32Le,
                    sample_rate_hz: 16_000,
                    channels: ChannelLayout::Mono,
                },
                clock: ClockOrigin::StreamStart,
            },
            StreamEvent::AudioChunk {
                direction: AudioDirection::Input,
                chunk_sequence: 0,
                frame_count: 320,
                segment_id: None,
                format: None,
                audio_base64: None,
                metadata: BTreeMap::new(),
            },
            StreamEvent::Discontinuity {
                expected_chunk_sequence: 1,
                received_chunk_sequence: 3,
                reason: "loss".into(),
            },
            StreamEvent::EndOfStream,
            StreamEvent::SpeechStarted {
                segment_id: segment.clone(),
            },
            StreamEvent::SpeechEnded {
                segment_id: segment.clone(),
                reason: "vad".into(),
            },
            StreamEvent::PartialHypothesis {
                role: TextRole::Recognition,
                segment_id: segment.clone(),
                text: "hello wor".into(),
                confidence: Some(confidence()),
            },
            StreamEvent::RevisedHypothesis {
                role: TextRole::Recognition,
                segment_id: segment.clone(),
                replaces: TextRange { start: 6, end: 9 },
                text: "world".into(),
                confidence: Some(confidence()),
            },
            StreamEvent::CommittedSegment {
                role: TextRole::Recognition,
                segment_id: segment.clone(),
                text: "hello world".into(),
                words: vec![token()],
                language: Some(LanguageHypothesis {
                    language: "en".into(),
                    confidence: Some(confidence()),
                }),
                speaker_id: Some("speaker-1".into()),
                confidence: Some(confidence()),
            },
            StreamEvent::HypothesisCancelled {
                role: TextRole::Recognition,
                segment_id: segment.clone(),
                reason: "restart".into(),
            },
            StreamEvent::TextCompleted {
                role: TextRole::Generation,
                text: "hello world".into(),
            },
            StreamEvent::WordTiming {
                segment_id: segment.clone(),
                word: token(),
            },
            StreamEvent::TokenTiming {
                segment_id: segment.clone(),
                token: token(),
            },
            StreamEvent::LanguageHypothesis {
                segment_id: segment.clone(),
                hypothesis: LanguageHypothesis {
                    language: "en".into(),
                    confidence: Some(confidence()),
                },
            },
            StreamEvent::SpeakerAssigned {
                segment_id: segment,
                speaker_id: "speaker-1".into(),
                confidence: Some(confidence()),
            },
            StreamEvent::Warning {
                code: "slow".into(),
                message: "provider slow".into(),
                recoverable: true,
            },
            StreamEvent::Error {
                code: "failed".into(),
                message: "provider failed".into(),
                recoverable: false,
            },
            StreamEvent::Cancelled {
                reason: "operator".into(),
            },
            StreamEvent::Completed,
            StreamEvent::OutputRequested {
                utterance_id: utterance.clone(),
                caused_by: vec![source],
            },
            StreamEvent::OutputStarted {
                utterance_id: utterance.clone(),
            },
            StreamEvent::OutputInterrupted {
                utterance_id: utterance.clone(),
            },
            StreamEvent::OutputResumed {
                utterance_id: utterance.clone(),
            },
            StreamEvent::OutputAborted {
                utterance_id: utterance.clone(),
                reason: "barge-in".into(),
            },
            StreamEvent::OutputFinished {
                utterance_id: utterance,
            },
            StreamEvent::DerivedArtifact {
                stage: "parse".into(),
                artifact_id: "parse-1".into(),
                value: json!({"intent": "greeting"}),
            },
        ]
    }

    #[test]
    fn every_event_variant_round_trips_through_json() {
        let mut types = BTreeSet::new();
        for event in every_variant() {
            let value = serde_json::to_value(&event).unwrap();
            types.insert(value["type"].as_str().unwrap().to_owned());
            let decoded: StreamEvent = serde_json::from_value(value).unwrap();
            assert_eq!(decoded, event);
        }
        assert_eq!(types.len(), every_variant().len());
    }

    #[test]
    fn jsonl_round_trip_preserves_dual_clocks_and_provenance() {
        let mut sequencer = StreamEventSequencer::new("stream-1");
        let source = sequencer.push(
            StreamEvent::SessionStarted {
                purpose: "fixture".into(),
            },
            EventTime {
                origin: ClockOrigin::StreamStart,
                offset_ms: 12,
            },
            Provenance::direct(),
        );
        let derived = sequencer.push(
            StreamEvent::DerivedArtifact {
                stage: "interpretation".into(),
                artifact_id: "artifact-1".into(),
                value: json!({"meaning": "hello"}),
            },
            EventTime {
                origin: ClockOrigin::StreamStart,
                offset_ms: 20,
            },
            Provenance::derived_from(vec![source.event_ref()]),
        );
        let mut bytes = Vec::new();
        write_jsonl(&mut bytes, vec![source.clone(), derived.clone()]).unwrap();
        let decoded = read_jsonl(std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(decoded, vec![source, derived]);
        assert_ne!(
            decoded[1].times.occurred_at.origin,
            decoded[1].times.observed_at.origin
        );
        assert_eq!(decoded[1].provenance.sources.len(), 1);
    }

    #[test]
    fn validator_applies_exact_unicode_revisions_and_rejects_late_rewrites() {
        let mut sequencer = StreamEventSequencer::new("stream-1");
        let at = EventTime {
            origin: ClockOrigin::StreamStart,
            offset_ms: 0,
        };
        let segment = SegmentId("segment-1".into());
        let mut validator = StreamValidator::default();
        validator
            .validate(&sequencer.push(
                StreamEvent::PartialHypothesis {
                    role: TextRole::Recognition,
                    segment_id: segment.clone(),
                    text: "héllo world".into(),
                    confidence: None,
                },
                at.clone(),
                Provenance::direct(),
            ))
            .unwrap();
        validator
            .validate(&sequencer.push(
                StreamEvent::RevisedHypothesis {
                    role: TextRole::Recognition,
                    segment_id: segment.clone(),
                    replaces: TextRange { start: 6, end: 11 },
                    text: "there".into(),
                    confidence: None,
                },
                at.clone(),
                Provenance::direct(),
            ))
            .unwrap();
        assert_eq!(validator.text_for(&segment), Some("héllo there"));
        validator
            .validate(&sequencer.push(
                StreamEvent::CommittedSegment {
                    role: TextRole::Recognition,
                    segment_id: segment.clone(),
                    text: "héllo there".into(),
                    words: Vec::new(),
                    language: None,
                    speaker_id: None,
                    confidence: None,
                },
                at.clone(),
                Provenance::direct(),
            ))
            .unwrap();
        assert_eq!(
            validator.validate(&sequencer.push(
                StreamEvent::RevisedHypothesis {
                    role: TextRole::Recognition,
                    segment_id: segment.clone(),
                    replaces: TextRange { start: 0, end: 1 },
                    text: "H".into(),
                    confidence: None,
                },
                at,
                Provenance::direct(),
            )),
            Err(StreamContractError::RevisedCommittedSegment(segment))
        );
    }

    #[test]
    fn bounded_buffer_exposes_backpressure_and_cancellation() {
        let mut buffer = BoundedEventBuffer::new(StreamPolicy {
            max_buffered_events: 1,
            ..StreamPolicy::default()
        });
        let mut sequencer = StreamEventSequencer::new("stream-1");
        let mut envelope = || {
            sequencer.push(
                StreamEvent::Completed,
                EventTime {
                    origin: ClockOrigin::StreamStart,
                    offset_ms: 0,
                },
                Provenance::direct(),
            )
        };
        buffer.try_push(envelope()).unwrap();
        assert_eq!(
            buffer.try_push(envelope()),
            Err(StreamContractError::Backpressure(1))
        );
        buffer.cancel();
        assert_eq!(
            buffer.try_push(envelope()),
            Err(StreamContractError::Cancelled)
        );
    }

    #[test]
    fn configured_drop_behavior_returns_a_central_warning_without_applying_state() {
        let mut sequencer = StreamEventSequencer::new("stream-1");
        let mut validator = StreamValidator::default();
        let at = EventTime {
            origin: ClockOrigin::StreamStart,
            offset_ms: 0,
        };
        validator
            .validate(&sequencer.push(
                StreamEvent::SessionStarted {
                    purpose: "fixture".into(),
                },
                at.clone(),
                Provenance::direct(),
            ))
            .unwrap();
        let mut out_of_order = sequencer.push(StreamEvent::Completed, at, Provenance::direct());
        out_of_order.sequence = 9;
        let outcome = validator
            .validate_with_policy(
                &out_of_order,
                &StreamPolicy {
                    out_of_order: InvalidEventBehavior::EmitWarningAndDrop,
                    ..StreamPolicy::default()
                },
            )
            .unwrap();
        assert!(matches!(
            outcome,
            ValidationOutcome::DroppedWithWarning(StreamEvent::Warning {
                code,
                recoverable: true,
                ..
            }) if code == "stream_event_dropped"
        ));
        assert!(!validator.terminal);
    }

    #[derive(Deserialize)]
    struct FixtureScenario {
        id: String,
        events: Vec<StreamEvent>,
    }

    #[test]
    fn fixture_suite_covers_required_streaming_failures_and_transitions() {
        let scenarios: Vec<FixtureScenario> = serde_json::from_str(include_str!(
            "../../../fixtures/streaming/recognition_scenarios_v1.json"
        ))
        .unwrap();
        let ids = scenarios
            .iter()
            .map(|scenario| scenario.id.as_str())
            .collect::<BTreeSet<_>>();
        for required in [
            "silence",
            "overlap",
            "language_change",
            "speaker_change",
            "discontinuity",
            "cancellation",
            "provider_failure",
        ] {
            assert!(ids.contains(required), "missing fixture {required}");
        }
        for scenario in scenarios {
            assert!(!scenario.events.is_empty(), "{} has no events", scenario.id);
            for event in scenario.events {
                let json = serde_json::to_string(&event).unwrap();
                assert_eq!(serde_json::from_str::<StreamEvent>(&json).unwrap(), event);
            }
        }
    }
}
