use crate::text_stability::stable_prefix_len;
use crate::{Confidence, ConfidenceScale, SegmentId, StreamEvent, TextRange, TextRole};

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptChunk {
    pub text: String,
    pub is_final: bool,
}

pub type TranscriptCandidateId = SegmentId;

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptStabilityState {
    pub candidate_id: TranscriptCandidateId,
    pub text: String,
    pub stable_prefix_len: usize,
    pub stable_text: String,
    pub unstable_text: String,
    pub stable_word_prefix: Option<String>,
    pub stable_word_count: usize,
    pub confidence: Option<f32>,
}

#[derive(Debug, Default)]
pub struct TranscriptCandidateTracker {
    next_id: u64,
    active: Option<ActiveCandidate>,
}

#[derive(Debug)]
struct ActiveCandidate {
    id: TranscriptCandidateId,
    text: String,
}

impl TranscriptCandidateTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ingest_chunk(&mut self, chunk: TranscriptChunk) -> Vec<StreamEvent> {
        self.ingest_candidate(chunk.text, None, chunk.is_final)
    }

    pub fn ingest_candidate(
        &mut self,
        text: impl Into<String>,
        confidence: Option<f32>,
        is_final: bool,
    ) -> Vec<StreamEvent> {
        let text = text.into();
        if text.is_empty() {
            return if is_final {
                self.cancel_active()
            } else {
                Vec::new()
            };
        }

        let confidence = confidence.map(probability_confidence);
        let mut events = Vec::new();
        if let Some(active) = self.active.take() {
            if active.text == text {
                if is_final {
                    events.push(StreamEvent::CommittedSegment {
                        role: TextRole::Recognition,
                        segment_id: active.id,
                        text,
                        words: Vec::new(),
                        language: None,
                        speaker_id: None,
                        confidence,
                    });
                } else {
                    self.active = Some(ActiveCandidate {
                        id: active.id.clone(),
                        text: text.clone(),
                    });
                    events.push(StreamEvent::PartialHypothesis {
                        role: TextRole::Recognition,
                        segment_id: active.id,
                        text,
                        confidence,
                    });
                }
                return events;
            }

            let stable_prefix_bytes = stable_prefix_len(&active.text, &text);
            let stable_prefix_chars = active.text[..stable_prefix_bytes].chars().count();
            events.push(StreamEvent::RevisedHypothesis {
                role: TextRole::Recognition,
                segment_id: active.id.clone(),
                replaces: TextRange {
                    start: stable_prefix_chars as u32,
                    end: active.text.chars().count() as u32,
                },
                text: text.chars().skip(stable_prefix_chars).collect(),
                confidence: confidence.clone(),
            });
            if is_final {
                events.push(StreamEvent::CommittedSegment {
                    role: TextRole::Recognition,
                    segment_id: active.id,
                    text,
                    words: Vec::new(),
                    language: None,
                    speaker_id: None,
                    confidence,
                });
            } else {
                self.active = Some(ActiveCandidate {
                    id: active.id,
                    text: text.clone(),
                });
            }
            return events;
        }

        let id = self.next_id();
        if is_final {
            events.push(StreamEvent::CommittedSegment {
                role: TextRole::Recognition,
                segment_id: id,
                text,
                words: Vec::new(),
                language: None,
                speaker_id: None,
                confidence,
            });
        } else {
            self.active = Some(ActiveCandidate {
                id: id.clone(),
                text: text.clone(),
            });
            events.push(StreamEvent::PartialHypothesis {
                role: TextRole::Recognition,
                segment_id: id,
                text,
                confidence,
            });
        }
        events
    }

    pub fn cancel_active(&mut self) -> Vec<StreamEvent> {
        let Some(active) = self.active.take() else {
            return Vec::new();
        };
        vec![StreamEvent::HypothesisCancelled {
            role: TextRole::Recognition,
            segment_id: active.id,
            reason: "recognizer produced no final hypothesis".into(),
        }]
    }

    fn next_id(&mut self) -> SegmentId {
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("transcript candidate id space exhausted");
        SegmentId(format!("recognition-{}", self.next_id))
    }
}

fn probability_confidence(value: f32) -> Confidence {
    Confidence {
        value: f64::from(value),
        scale: ConfidenceScale::Probability,
        calibration: None,
    }
}

impl TranscriptStabilityState {
    pub fn from_parts(
        candidate_id: TranscriptCandidateId,
        text: &str,
        stable_prefix_len: usize,
        confidence: Option<f32>,
    ) -> Self {
        let split = stable_prefix_len.min(text.len());
        let split = if text.is_char_boundary(split) {
            split
        } else {
            text.char_indices()
                .map(|(idx, _)| idx)
                .take_while(|idx| *idx < split)
                .last()
                .unwrap_or_default()
        };
        let (stable_text, unstable_text) = text.split_at(split);
        let stable_word_split = if stable_text
            .chars()
            .next_back()
            .is_some_and(char::is_whitespace)
        {
            stable_text.trim_end().len()
        } else {
            stable_text
                .char_indices()
                .rev()
                .find_map(|(idx, ch)| ch.is_whitespace().then_some(idx + ch.len_utf8()))
                .unwrap_or_default()
        };
        let stable_word_prefix = stable_text[..stable_word_split].trim_end();
        Self {
            candidate_id,
            text: text.to_string(),
            stable_prefix_len: split,
            stable_text: stable_text.to_string(),
            unstable_text: unstable_text.to_string(),
            stable_word_prefix: (!stable_word_prefix.is_empty())
                .then(|| stable_word_prefix.to_string()),
            stable_word_count: stable_word_prefix.split_whitespace().count(),
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_finalizes_final_chunk() {
        let mut tracker = TranscriptCandidateTracker::new();
        assert_eq!(
            tracker.ingest_candidate("hello", Some(0.9), true),
            vec![StreamEvent::CommittedSegment {
                role: TextRole::Recognition,
                segment_id: SegmentId("recognition-1".into()),
                text: "hello".into(),
                words: Vec::new(),
                language: None,
                speaker_id: None,
                confidence: Some(Confidence {
                    value: f64::from(0.9_f32),
                    scale: ConfidenceScale::Probability,
                    calibration: None,
                }),
            }]
        );
    }

    #[test]
    fn revision_has_an_exact_unicode_scalar_range_and_stable_segment_id() {
        let mut tracker = TranscriptCandidateTracker::new();
        let first = tracker.ingest_candidate("héllo world", None, false);
        let revised = tracker.ingest_candidate("héllo there", None, false);
        let segment_id = match &first[0] {
            StreamEvent::PartialHypothesis { segment_id, .. } => segment_id.clone(),
            event => panic!("unexpected first event: {event:?}"),
        };
        assert_eq!(
            revised,
            vec![StreamEvent::RevisedHypothesis {
                role: TextRole::Recognition,
                segment_id,
                replaces: TextRange { start: 6, end: 11 },
                text: "there".into(),
                confidence: None,
            }]
        );
    }
}
