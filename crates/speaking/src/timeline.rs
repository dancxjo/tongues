//! Versioned, non-destructive speech timeline and correction contracts.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{StreamEvent, TextRole};

pub const TIMELINE_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanModality {
    Audio,
    Transcript,
    Word,
    Phoneme,
    Morpheme,
    Playback,
    BreathGroup,
    Interruption,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentKind {
    Contains,
    AlignedTo,
    DerivedFrom,
    RevisionOf,
    Overlaps,
    PlayedAs,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineSpan {
    pub id: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub modality: SpanModality,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl TimelineSpan {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.id.is_empty(), "timeline span ID is empty");
        anyhow::ensure!(
            self.end_ms > self.start_ms,
            "timeline span `{}` has no duration",
            self.id
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineAlignment {
    pub source_span_id: String,
    pub target_span_id: String,
    pub kind: AlignmentKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechTimelineSession {
    pub schema_version: u16,
    pub session_id: String,
    /// Immutable captured/provider evidence.
    pub evidence: Vec<TimelineSpan>,
    #[serde(default)]
    pub alignments: Vec<TimelineAlignment>,
    #[serde(default)]
    pub source_events: Vec<StreamEvent>,
    #[serde(default)]
    pub operations: Vec<TimelineOperation>,
}

impl SpeechTimelineSession {
    pub fn new(
        session_id: impl Into<String>,
        evidence: Vec<TimelineSpan>,
        alignments: Vec<TimelineAlignment>,
    ) -> anyhow::Result<Self> {
        let session = Self {
            schema_version: TIMELINE_SCHEMA_VERSION,
            session_id: session_id.into(),
            evidence,
            alignments,
            source_events: Vec::new(),
            operations: Vec::new(),
        };
        session.validate()?;
        Ok(session)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema_version == TIMELINE_SCHEMA_VERSION,
            "timeline schema {} is unsupported; expected {}",
            self.schema_version,
            TIMELINE_SCHEMA_VERSION
        );
        anyhow::ensure!(!self.session_id.is_empty(), "timeline session ID is empty");
        let mut ids = BTreeSet::new();
        for span in &self.evidence {
            span.validate()?;
            anyhow::ensure!(
                ids.insert(span.id.as_str()),
                "duplicate timeline span `{}`",
                span.id
            );
        }
        for alignment in &self.alignments {
            anyhow::ensure!(
                ids.contains(alignment.source_span_id.as_str())
                    && ids.contains(alignment.target_span_id.as_str()),
                "timeline alignment references unknown evidence"
            );
        }
        let mut operation_ids = BTreeSet::new();
        for operation in &self.operations {
            operation.validate()?;
            anyhow::ensure!(
                operation_ids.insert(operation.operation_id.as_str()),
                "duplicate timeline operation `{}`",
                operation.operation_id
            );
        }
        Ok(())
    }

    pub fn append_operation(&mut self, operation: TimelineOperation) -> anyhow::Result<()> {
        operation.validate()?;
        anyhow::ensure!(
            !self
                .operations
                .iter()
                .any(|existing| existing.operation_id == operation.operation_id),
            "duplicate timeline operation `{}`",
            operation.operation_id
        );
        self.operations.push(operation);
        // Validate replay now so invalid target IDs or boundaries fail before
        // persistence.
        self.project()?;
        Ok(())
    }

    pub fn project(&self) -> anyhow::Result<TimelineProjection> {
        project_timeline(self)
    }

    pub fn from_stream_events(
        session_id: impl Into<String>,
        events: Vec<StreamEvent>,
    ) -> anyhow::Result<Self> {
        let mut evidence = Vec::new();
        let mut alignments = Vec::new();
        for event in &events {
            let StreamEvent::CommittedSegment {
                role,
                segment_id,
                text,
                words,
                language,
                speaker_id,
                ..
            } = event
            else {
                continue;
            };
            if *role != TextRole::Recognition || words.is_empty() {
                continue;
            }
            let start_ms = words.first().unwrap().range.start_ms;
            let end_ms = words.last().unwrap().range.end_ms;
            if end_ms <= start_ms {
                continue;
            }
            let transcript_id = format!("transcript:{}", segment_id.0);
            let mut metadata = BTreeMap::from([("text".into(), Value::String(text.clone()))]);
            if let Some(language) = language {
                metadata.insert("language".into(), Value::String(language.language.clone()));
            }
            if let Some(speaker_id) = speaker_id {
                metadata.insert("speaker_id".into(), Value::String(speaker_id.clone()));
            }
            evidence.push(TimelineSpan {
                id: transcript_id.clone(),
                start_ms,
                end_ms,
                modality: SpanModality::Transcript,
                metadata,
            });
            for (index, word) in words.iter().enumerate() {
                if word.range.end_ms <= word.range.start_ms {
                    continue;
                }
                let word_id = format!("word:{}:{index}", segment_id.0);
                evidence.push(TimelineSpan {
                    id: word_id.clone(),
                    start_ms: word.range.start_ms,
                    end_ms: word.range.end_ms,
                    modality: SpanModality::Word,
                    metadata: BTreeMap::from([("text".into(), Value::String(word.text.clone()))]),
                });
                alignments.push(TimelineAlignment {
                    source_span_id: word_id,
                    target_span_id: transcript_id.clone(),
                    kind: AlignmentKind::Contains,
                    confidence: word.confidence.as_ref().map(|value| value.value as f32),
                });
            }
        }
        let mut session = Self::new(session_id, evidence, alignments)?;
        session.source_events = events;
        Ok(session)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditOrigin {
    Automatic,
    Manual,
    Derived,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditProvenance {
    pub origin: EditOrigin,
    pub actor: String,
    pub at_ms: u64,
    #[serde(default)]
    pub source_span_ids: Vec<String>,
    #[serde(default)]
    pub source_event_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Boundary {
    Start,
    End,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioRegionEditKind {
    Trim,
    Split,
    Fade,
    Gain,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum TimelineOperationKind {
    TranscriptReplace {
        span_id: String,
        text: String,
    },
    AlignmentMoveBoundary {
        span_id: String,
        boundary: Boundary,
        new_time_ms: u64,
    },
    Annotate {
        span_id: String,
        key: String,
        value: Value,
    },
    SegmentSplit {
        span_id: String,
        split_at_ms: u64,
        left_span_id: String,
        right_span_id: String,
    },
    AudioRegion {
        span_id: String,
        edit: AudioRegionEditKind,
        start_ms: u64,
        end_ms: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        value: Option<f32>,
    },
    Undo {
        target_operation_id: String,
    },
    Redo {
        target_operation_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineOperation {
    pub operation_id: String,
    pub provenance: EditProvenance,
    #[serde(flatten)]
    pub operation: TimelineOperationKind,
}

impl TimelineOperation {
    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.operation_id.is_empty(),
            "timeline operation ID is empty"
        );
        anyhow::ensure!(
            !self.provenance.actor.is_empty(),
            "timeline edit actor is empty"
        );
        match &self.operation {
            TimelineOperationKind::TranscriptReplace { span_id, .. }
            | TimelineOperationKind::Annotate { span_id, .. }
            | TimelineOperationKind::AlignmentMoveBoundary { span_id, .. }
            | TimelineOperationKind::SegmentSplit { span_id, .. }
            | TimelineOperationKind::AudioRegion { span_id, .. } => {
                anyhow::ensure!(!span_id.is_empty(), "timeline edit target is empty");
            }
            TimelineOperationKind::Undo {
                target_operation_id,
            }
            | TimelineOperationKind::Redo {
                target_operation_id,
            } => anyhow::ensure!(
                !target_operation_id.is_empty(),
                "timeline control operation target is empty"
            ),
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineProjection {
    pub schema_version: u16,
    pub session_id: String,
    /// The baseline is copied unchanged for side-by-side inspection.
    pub original: Vec<TimelineSpan>,
    pub edited: Vec<TimelineSpan>,
    pub alignments: Vec<TimelineAlignment>,
    pub applied_operation_ids: Vec<String>,
    pub audio_region_edits: Vec<TimelineOperation>,
}

fn project_timeline(session: &SpeechTimelineSession) -> anyhow::Result<TimelineProjection> {
    session.validate()?;
    let mut enabled = BTreeMap::<&str, bool>::new();
    for operation in &session.operations {
        match &operation.operation {
            TimelineOperationKind::Undo {
                target_operation_id,
            } => {
                anyhow::ensure!(
                    enabled.contains_key(target_operation_id.as_str()),
                    "undo references unknown operation `{target_operation_id}`"
                );
                enabled.insert(target_operation_id, false);
            }
            TimelineOperationKind::Redo {
                target_operation_id,
            } => {
                anyhow::ensure!(
                    enabled.contains_key(target_operation_id.as_str()),
                    "redo references unknown operation `{target_operation_id}`"
                );
                enabled.insert(target_operation_id, true);
            }
            _ => {
                enabled.insert(&operation.operation_id, true);
            }
        }
    }
    let mut spans = session
        .evidence
        .iter()
        .cloned()
        .map(|span| (span.id.clone(), span))
        .collect::<BTreeMap<_, _>>();
    let mut applied = Vec::new();
    let mut audio_region_edits = Vec::new();
    for operation in &session.operations {
        if matches!(
            operation.operation,
            TimelineOperationKind::Undo { .. } | TimelineOperationKind::Redo { .. }
        ) || !enabled
            .get(operation.operation_id.as_str())
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        apply_projection_operation(&mut spans, operation, &mut audio_region_edits)?;
        applied.push(operation.operation_id.clone());
    }
    Ok(TimelineProjection {
        schema_version: session.schema_version,
        session_id: session.session_id.clone(),
        original: session.evidence.clone(),
        edited: spans.into_values().collect(),
        alignments: session.alignments.clone(),
        applied_operation_ids: applied,
        audio_region_edits,
    })
}

fn apply_projection_operation(
    spans: &mut BTreeMap<String, TimelineSpan>,
    operation: &TimelineOperation,
    audio_region_edits: &mut Vec<TimelineOperation>,
) -> anyhow::Result<()> {
    match &operation.operation {
        TimelineOperationKind::TranscriptReplace { span_id, text } => {
            let span = target_mut(spans, span_id)?;
            anyhow::ensure!(
                matches!(span.modality, SpanModality::Transcript | SpanModality::Word),
                "transcript replacement target `{span_id}` is not text"
            );
            span.metadata
                .insert("text".into(), Value::String(text.clone()));
        }
        TimelineOperationKind::AlignmentMoveBoundary {
            span_id,
            boundary,
            new_time_ms,
        } => {
            let span = target_mut(spans, span_id)?;
            match boundary {
                Boundary::Start => {
                    anyhow::ensure!(*new_time_ms < span.end_ms, "edited start must precede end");
                    span.start_ms = *new_time_ms;
                }
                Boundary::End => {
                    anyhow::ensure!(*new_time_ms > span.start_ms, "edited end must follow start");
                    span.end_ms = *new_time_ms;
                }
            }
        }
        TimelineOperationKind::Annotate {
            span_id,
            key,
            value,
        } => {
            anyhow::ensure!(!key.is_empty(), "annotation key is empty");
            target_mut(spans, span_id)?
                .metadata
                .insert(format!("annotation:{key}"), value.clone());
        }
        TimelineOperationKind::SegmentSplit {
            span_id,
            split_at_ms,
            left_span_id,
            right_span_id,
        } => {
            anyhow::ensure!(
                !spans.contains_key(left_span_id) && !spans.contains_key(right_span_id),
                "split output span ID already exists"
            );
            let source = spans
                .remove(span_id)
                .ok_or_else(|| anyhow::anyhow!("unknown timeline span `{span_id}`"))?;
            anyhow::ensure!(
                *split_at_ms > source.start_ms && *split_at_ms < source.end_ms,
                "split time is outside the source span"
            );
            let mut left = source.clone();
            left.id = left_span_id.clone();
            left.end_ms = *split_at_ms;
            let mut right = source;
            right.id = right_span_id.clone();
            right.start_ms = *split_at_ms;
            spans.insert(left.id.clone(), left);
            spans.insert(right.id.clone(), right);
        }
        TimelineOperationKind::AudioRegion {
            span_id,
            start_ms,
            end_ms,
            value,
            ..
        } => {
            let span = spans
                .get(span_id)
                .ok_or_else(|| anyhow::anyhow!("unknown timeline span `{span_id}`"))?;
            anyhow::ensure!(
                span.modality == SpanModality::Audio,
                "audio edit target `{span_id}` is not audio evidence"
            );
            anyhow::ensure!(
                end_ms > start_ms && *start_ms >= span.start_ms && *end_ms <= span.end_ms,
                "audio edit region is outside the evidence span"
            );
            anyhow::ensure!(
                value.is_none_or(f32::is_finite),
                "audio edit value is not finite"
            );
            audio_region_edits.push(operation.clone());
        }
        TimelineOperationKind::Undo { .. } | TimelineOperationKind::Redo { .. } => {}
    }
    Ok(())
}

fn target_mut<'a>(
    spans: &'a mut BTreeMap<String, TimelineSpan>,
    span_id: &str,
) -> anyhow::Result<&'a mut TimelineSpan> {
    spans
        .get_mut(span_id)
        .ok_or_else(|| anyhow::anyhow!("unknown timeline span `{span_id}`"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimelineExportKind {
    RawEvidence,
    CorrectedTranscript,
    AlignedMetadata,
    EditLog,
    Bundle,
}

pub fn export_timeline(
    session: &SpeechTimelineSession,
    kind: TimelineExportKind,
) -> anyhow::Result<Value> {
    let projection = session.project()?;
    Ok(match kind {
        TimelineExportKind::RawEvidence => serde_json::to_value(&session.evidence)?,
        TimelineExportKind::CorrectedTranscript => Value::Array(
            projection
                .edited
                .iter()
                .filter(|span| {
                    matches!(span.modality, SpanModality::Transcript | SpanModality::Word)
                })
                .filter_map(|span| {
                    span.metadata.get("text").map(|text| {
                        serde_json::json!({
                            "span_id": span.id,
                            "start_ms": span.start_ms,
                            "end_ms": span.end_ms,
                            "text": text,
                            "projection": "corrected"
                        })
                    })
                })
                .collect(),
        ),
        TimelineExportKind::AlignedMetadata => serde_json::to_value(&projection)?,
        TimelineExportKind::EditLog => serde_json::to_value(&session.operations)?,
        TimelineExportKind::Bundle => serde_json::json!({
            "session": session,
            "projection": projection,
            "evidence_authority": "observed",
            "edit_authority": "corrected_interpretation"
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Confidence, ConfidenceScale, SegmentId, TimeRange, TimedToken};

    fn provenance() -> EditProvenance {
        EditProvenance {
            origin: EditOrigin::Manual,
            actor: "operator".into(),
            at_ms: 100,
            source_span_ids: vec!["word:s:0".into()],
            source_event_ids: vec!["event:1".into()],
            reason: Some("correct recognition".into()),
        }
    }

    fn event() -> StreamEvent {
        StreamEvent::CommittedSegment {
            role: TextRole::Recognition,
            segment_id: SegmentId("s".into()),
            text: "hello world".into(),
            words: vec![
                TimedToken {
                    text: "hello".into(),
                    range: TimeRange {
                        start_ms: 0,
                        end_ms: 300,
                    },
                    confidence: Some(Confidence {
                        value: 0.8,
                        scale: ConfidenceScale::Probability,
                        calibration: None,
                    }),
                },
                TimedToken {
                    text: "world".into(),
                    range: TimeRange {
                        start_ms: 320,
                        end_ms: 700,
                    },
                    confidence: None,
                },
            ],
            language: None,
            speaker_id: Some("speaker-1".into()),
            confidence: None,
        }
    }

    #[test]
    fn stream_session_projects_words_without_losing_source_events() {
        let session = SpeechTimelineSession::from_stream_events("live:1", vec![event()]).unwrap();
        assert_eq!(session.evidence.len(), 3);
        assert_eq!(session.alignments.len(), 2);
        assert_eq!(session.source_events, vec![event()]);
    }

    #[test]
    fn corrections_never_overwrite_original_evidence_and_replay_deterministically() {
        let mut session =
            SpeechTimelineSession::from_stream_events("saved:1", vec![event()]).unwrap();
        session
            .append_operation(TimelineOperation {
                operation_id: "op:replace".into(),
                provenance: provenance(),
                operation: TimelineOperationKind::TranscriptReplace {
                    span_id: "word:s:0".into(),
                    text: "hullo".into(),
                },
            })
            .unwrap();
        session
            .append_operation(TimelineOperation {
                operation_id: "op:align".into(),
                provenance: provenance(),
                operation: TimelineOperationKind::AlignmentMoveBoundary {
                    span_id: "word:s:0".into(),
                    boundary: Boundary::End,
                    new_time_ms: 310,
                },
            })
            .unwrap();
        let first = session.project().unwrap();
        let encoded = serde_json::to_string(&session).unwrap();
        let restored: SpeechTimelineSession = serde_json::from_str(&encoded).unwrap();
        let second = restored.project().unwrap();
        assert_eq!(first, second);
        assert_eq!(
            session.evidence[1].metadata["text"],
            Value::String("hello".into())
        );
        assert_eq!(
            first.edited[1].metadata["text"],
            Value::String("hullo".into())
        );
    }

    #[test]
    fn undo_and_redo_are_replayed_control_operations() {
        let mut session =
            SpeechTimelineSession::from_stream_events("saved:undo", vec![event()]).unwrap();
        for (id, operation) in [
            (
                "replace",
                TimelineOperationKind::TranscriptReplace {
                    span_id: "word:s:0".into(),
                    text: "hullo".into(),
                },
            ),
            (
                "undo",
                TimelineOperationKind::Undo {
                    target_operation_id: "replace".into(),
                },
            ),
        ] {
            session
                .append_operation(TimelineOperation {
                    operation_id: id.into(),
                    provenance: provenance(),
                    operation,
                })
                .unwrap();
        }
        assert_eq!(
            session.project().unwrap().edited[1].metadata["text"],
            Value::String("hello".into())
        );
        session
            .append_operation(TimelineOperation {
                operation_id: "redo".into(),
                provenance: provenance(),
                operation: TimelineOperationKind::Redo {
                    target_operation_id: "replace".into(),
                },
            })
            .unwrap();
        assert_eq!(
            session.project().unwrap().edited[1].metadata["text"],
            Value::String("hullo".into())
        );
    }

    #[test]
    fn exports_label_observed_evidence_and_corrected_interpretation() {
        let session =
            SpeechTimelineSession::from_stream_events("saved:export", vec![event()]).unwrap();
        let bundle = export_timeline(&session, TimelineExportKind::Bundle).unwrap();
        assert_eq!(bundle["evidence_authority"], "observed");
        assert_eq!(bundle["edit_authority"], "corrected_interpretation");
    }

    #[test]
    fn listenbury_interruption_fixture_maps_to_explicit_timeline_spans() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../fixtures/timeline/listenbury_user_interrupts_v1.json"
        ))
        .unwrap();
        let spans: Vec<TimelineSpan> = serde_json::from_value(fixture["spans"].clone()).unwrap();
        assert!(
            spans
                .iter()
                .any(|span| span.modality == SpanModality::Interruption)
        );
        SpeechTimelineSession::new("fixture:interruption", spans, Vec::new()).unwrap();
    }
}
