//! Committed transcript normalization and downstream linguistic routing.

use std::borrow::Cow;
use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{
    Confidence, EventRef, EventTimes, GrammarParser, LanguageHypothesis, Provenance, SegmentId,
    GrammarAnalysis, StreamEvent, StreamEventEnvelope, TerminalPunctuation, TextRole,
    TimedToken, VarietyGrammarParser, VarietyId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisfluencyPolicy {
    Preserve,
    RemoveFilledPauses,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonSpeechAnnotationPolicy {
    Preserve,
    RemoveBracketed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptNormalizationConfig {
    pub sentence_case_display: bool,
    pub append_terminal_punctuation: bool,
    pub inverse_text_normalization: bool,
    pub disfluency_policy: DisfluencyPolicy,
    pub non_speech_annotations: NonSpeechAnnotationPolicy,
}

impl Default for TranscriptNormalizationConfig {
    fn default() -> Self {
        Self {
            sentence_case_display: true,
            append_terminal_punctuation: true,
            inverse_text_normalization: true,
            disfluency_policy: DisfluencyPolicy::Preserve,
            non_speech_annotations: NonSpeechAnnotationPolicy::Preserve,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSourceMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_ref: Option<EventRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_times: Option<EventTimes>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedTranscriptSegment {
    pub segment_id: SegmentId,
    pub raw_text: String,
    pub display_text: String,
    pub downstream_text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<TimedToken>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<LanguageHypothesis>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<Confidence>,
    pub source: TranscriptSourceMetadata,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommittedTranscriptArtifacts {
    pub transcript: NormalizedTranscriptSegment,
    pub syntax: GrammarAnalysis,
    pub interpretation: serde_json::Value,
}

impl CommittedTranscriptArtifacts {
    pub fn derived_events(&self) -> Vec<StreamEvent> {
        let artifact_id = self.transcript.segment_id.0.clone();
        vec![
            StreamEvent::DerivedArtifact {
                stage: "transcript_normalization".into(),
                artifact_id: artifact_id.clone(),
                value: serde_json::to_value(&self.transcript)
                    .expect("normalized transcript is serializable"),
            },
            StreamEvent::DerivedArtifact {
                stage: "sentence_boundary".into(),
                artifact_id: artifact_id.clone(),
                value: serde_json::to_value(&self.syntax).expect("sentence syntax is serializable"),
            },
            StreamEvent::DerivedArtifact {
                stage: "interpretation".into(),
                artifact_id,
                value: self.interpretation.clone(),
            },
        ]
    }
}

pub trait TranscriptNormalizer {
    fn normalize(
        &self,
        segment_id: SegmentId,
        text: String,
        words: Vec<TimedToken>,
        language: Option<LanguageHypothesis>,
        speaker_id: Option<String>,
        confidence: Option<Confidence>,
        source: TranscriptSourceMetadata,
    ) -> anyhow::Result<NormalizedTranscriptSegment>;
}

#[derive(Debug, Clone)]
pub struct RuleBasedTranscriptNormalizer {
    config: TranscriptNormalizationConfig,
}

impl RuleBasedTranscriptNormalizer {
    pub fn new(config: TranscriptNormalizationConfig) -> Self {
        Self { config }
    }
}

impl Default for RuleBasedTranscriptNormalizer {
    fn default() -> Self {
        Self::new(TranscriptNormalizationConfig::default())
    }
}

impl TranscriptNormalizer for RuleBasedTranscriptNormalizer {
    fn normalize(
        &self,
        segment_id: SegmentId,
        text: String,
        words: Vec<TimedToken>,
        language: Option<LanguageHypothesis>,
        speaker_id: Option<String>,
        confidence: Option<Confidence>,
        source: TranscriptSourceMetadata,
    ) -> anyhow::Result<NormalizedTranscriptSegment> {
        anyhow::ensure!(!segment_id.0.is_empty(), "transcript segment ID is empty");
        let raw_text = text;
        let language_code = language
            .as_ref()
            .map(|hypothesis| hypothesis.language.as_str())
            .unwrap_or("und");
        let mut display_text = clean_provider_tokens(&raw_text);
        display_text = normalize_spacing(&display_text);
        if self.config.sentence_case_display {
            display_text = sentence_case(&display_text);
        }
        if self.config.append_terminal_punctuation {
            display_text = append_terminal_punctuation(display_text);
        }

        let mut downstream_text = display_text.clone();
        if self.config.non_speech_annotations == NonSpeechAnnotationPolicy::RemoveBracketed {
            downstream_text = remove_bracketed_annotations(&downstream_text);
        }
        if self.config.disfluency_policy == DisfluencyPolicy::RemoveFilledPauses {
            downstream_text = remove_filled_pauses(&downstream_text, language_code);
        }
        if self.config.inverse_text_normalization {
            downstream_text = inverse_text_normalize(&downstream_text, language_code);
        }
        downstream_text = normalize_spacing(&downstream_text);
        downstream_text = repair_terminal_spacing(downstream_text);
        if self.config.sentence_case_display {
            downstream_text = sentence_case(&downstream_text);
        }

        Ok(NormalizedTranscriptSegment {
            segment_id,
            raw_text,
            display_text,
            downstream_text,
            words,
            language,
            speaker_id,
            confidence,
            source,
        })
    }
}

pub trait CommittedTranscriptInterpreter {
    fn interpret(
        &mut self,
        transcript: &NormalizedTranscriptSegment,
        syntax: &GrammarAnalysis,
    ) -> anyhow::Result<serde_json::Value>;
}

#[derive(Debug, Default)]
pub struct StructuralTranscriptInterpreter;

impl CommittedTranscriptInterpreter for StructuralTranscriptInterpreter {
    fn interpret(
        &mut self,
        transcript: &NormalizedTranscriptSegment,
        syntax: &GrammarAnalysis,
    ) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::json!({
            "text": transcript.downstream_text,
            "token_count": syntax.tokens.len(),
            "speaker_id": transcript.speaker_id,
            "language": transcript.language.as_ref().map(|item| &item.language),
        }))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TranscriptPipelineError {
    #[error("transcript segment `{0}` was already committed")]
    DuplicateCommit(String),
    #[error("transcript normalization failed: {0}")]
    Normalization(String),
    #[error("transcript interpretation failed: {0}")]
    Interpretation(String),
}

pub struct CommittedTranscriptPipeline<N, I> {
    normalizer: N,
    interpreter: I,
    committed: HashSet<SegmentId>,
}

impl<N, I> CommittedTranscriptPipeline<N, I>
where
    N: TranscriptNormalizer,
    I: CommittedTranscriptInterpreter,
{
    pub fn new(normalizer: N, interpreter: I) -> Self {
        Self {
            normalizer,
            interpreter,
            committed: HashSet::new(),
        }
    }

    pub fn process_envelope(
        &mut self,
        envelope: &StreamEventEnvelope,
    ) -> Result<Option<CommittedTranscriptArtifacts>, TranscriptPipelineError> {
        self.process_event(
            &envelope.event,
            TranscriptSourceMetadata {
                event_ref: Some(envelope.event_ref()),
                event_times: Some(envelope.times.clone()),
                provenance: Some(envelope.provenance.clone()),
            },
        )
    }

    pub fn process_event(
        &mut self,
        event: &StreamEvent,
        source: TranscriptSourceMetadata,
    ) -> Result<Option<CommittedTranscriptArtifacts>, TranscriptPipelineError> {
        let StreamEvent::CommittedSegment {
            role: TextRole::Recognition,
            segment_id,
            text,
            words,
            language,
            speaker_id,
            confidence,
        } = event
        else {
            return Ok(None);
        };
        if !self.committed.insert(segment_id.clone()) {
            return Err(TranscriptPipelineError::DuplicateCommit(
                segment_id.0.clone(),
            ));
        }
        let transcript = self
            .normalizer
            .normalize(
                segment_id.clone(),
                text.clone(),
                words.clone(),
                language.clone(),
                speaker_id.clone(),
                confidence.clone(),
                source,
            )
            .map_err(|error| TranscriptPipelineError::Normalization(error.to_string()))?;
        let variety = variety_for_language(transcript.language.as_ref());
        let terminal = terminal_punctuation(&transcript.downstream_text);
        let words = parser_words(&transcript.downstream_text);
        let syntax = VarietyGrammarParser::new(variety).parse(&words, terminal);
        let interpretation = self
            .interpreter
            .interpret(&transcript, &syntax)
            .map_err(|error| TranscriptPipelineError::Interpretation(error.to_string()))?;
        Ok(Some(CommittedTranscriptArtifacts {
            transcript,
            syntax,
            interpretation,
        }))
    }
}

pub fn transcript_export_jsonl(
    segments: impl IntoIterator<Item = NormalizedTranscriptSegment>,
) -> anyhow::Result<String> {
    let mut output = String::new();
    for segment in segments {
        output.push_str(&serde_json::to_string(&segment)?);
        output.push('\n');
    }
    Ok(output)
}

fn clean_provider_tokens(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<|") {
        output.push_str(&rest[..start]);
        let Some(relative_end) = rest[start + 2..].find("|>") else {
            output.push_str(&rest[start..]);
            return output;
        };
        rest = &rest[start + 2 + relative_end + 2..];
    }
    output.push_str(rest);
    output
}

fn normalize_spacing(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn sentence_case(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut changed = false;
    for character in text.chars() {
        if !changed && character.is_alphabetic() {
            output.extend(character.to_uppercase());
            changed = true;
        } else {
            output.push(character);
        }
    }
    output
}

fn append_terminal_punctuation(mut text: String) -> String {
    if !text.ends_with(['.', '?', '!']) {
        text.push('.');
    }
    text
}

fn remove_bracketed_annotations(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut depth = 0_u32;
    for character in text.chars() {
        match character {
            '[' | '(' => depth = depth.saturating_add(1),
            ']' | ')' if depth > 0 => depth = depth.saturating_sub(1),
            _ if depth == 0 => output.push(character),
            _ => {}
        }
    }
    output
}

fn remove_filled_pauses(text: &str, language: &str) -> String {
    let primary = primary_language(language);
    let pauses = match primary.as_ref() {
        "fr" => &["euh", "heu"][..],
        "de" => &["äh", "ähm"][..],
        "es" => &["eh", "este"][..],
        _ => &["uh", "um", "erm", "er"][..],
    };
    text.split_whitespace()
        .filter(|word| {
            let normalized = word
                .trim_matches(|character: char| !character.is_alphanumeric())
                .to_lowercase();
            !pauses.contains(&normalized.as_str())
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn inverse_text_normalize(text: &str, language: &str) -> String {
    let primary = primary_language(language);
    let digit_words: &[(&str, char)] = match primary.as_ref() {
        "fr" => &[
            ("zéro", '0'),
            ("zero", '0'),
            ("un", '1'),
            ("deux", '2'),
            ("trois", '3'),
            ("quatre", '4'),
            ("cinq", '5'),
            ("six", '6'),
            ("sept", '7'),
            ("huit", '8'),
            ("neuf", '9'),
        ],
        "de" => &[
            ("null", '0'),
            ("eins", '1'),
            ("zwei", '2'),
            ("drei", '3'),
            ("vier", '4'),
            ("fünf", '5'),
            ("sechs", '6'),
            ("sieben", '7'),
            ("acht", '8'),
            ("neun", '9'),
        ],
        "es" => &[
            ("cero", '0'),
            ("uno", '1'),
            ("dos", '2'),
            ("tres", '3'),
            ("cuatro", '4'),
            ("cinco", '5'),
            ("seis", '6'),
            ("siete", '7'),
            ("ocho", '8'),
            ("nueve", '9'),
        ],
        _ => &[
            ("zero", '0'),
            ("oh", '0'),
            ("one", '1'),
            ("two", '2'),
            ("three", '3'),
            ("four", '4'),
            ("five", '5'),
            ("six", '6'),
            ("seven", '7'),
            ("eight", '8'),
            ("nine", '9'),
        ],
    };
    let mut output = Vec::new();
    let mut digits = String::new();
    for word in text.split_whitespace() {
        let punctuation = word
            .chars()
            .rev()
            .take_while(|character| !character.is_alphanumeric())
            .collect::<Vec<_>>();
        let normalized = word
            .trim_matches(|character: char| !character.is_alphanumeric())
            .to_lowercase();
        if let Some((_, digit)) = digit_words.iter().find(|(name, _)| *name == normalized) {
            digits.push(*digit);
            if !punctuation.is_empty() {
                for character in punctuation.iter().rev() {
                    digits.push(*character);
                }
                output.push(std::mem::take(&mut digits));
            }
        } else {
            if !digits.is_empty() {
                output.push(std::mem::take(&mut digits));
            }
            output.push(word.to_string());
        }
    }
    if !digits.is_empty() {
        output.push(digits);
    }
    output.join(" ")
}

fn repair_terminal_spacing(text: String) -> String {
    text.replace(" .", ".")
        .replace(" ,", ",")
        .replace(" ?", "?")
        .replace(" !", "!")
}

fn primary_language(language: &str) -> Cow<'_, str> {
    let primary = language.split(['-', '_']).next().unwrap_or("und");
    if primary.bytes().all(|byte| !byte.is_ascii_uppercase()) {
        Cow::Borrowed(primary)
    } else {
        Cow::Owned(primary.to_ascii_lowercase())
    }
}

fn variety_for_language(language: Option<&LanguageHypothesis>) -> VarietyId {
    let language = language
        .map(|hypothesis| primary_language(&hypothesis.language))
        .unwrap_or(Cow::Borrowed("und"));
    VarietyId(
        match language.as_ref() {
            "fr" => "fr-FR-Standard",
            "de" => "de-DE-Standard",
            "es" => "es-ES-Standard",
            "eo" => "eo",
            _ => "en-US-GA",
        }
        .into(),
    )
}

fn parser_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.trim_matches(|character: char| !character.is_alphanumeric() && character != '\'')
                .to_string()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn terminal_punctuation(text: &str) -> Option<TerminalPunctuation> {
    match text.chars().next_back()? {
        '.' => Some(TerminalPunctuation::Period),
        '?' => Some(TerminalPunctuation::Question),
        '!' => Some(TerminalPunctuation::Exclamation),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ClockOrigin, ConfidenceScale, EventId, EventTime, Provenance, ProvenanceKind, StreamId,
        TextRange, TimeRange,
    };

    fn language(code: &str) -> LanguageHypothesis {
        LanguageHypothesis {
            language: code.into(),
            confidence: Some(Confidence {
                value: 0.88,
                scale: ConfidenceScale::Probability,
                calibration: None,
            }),
        }
    }

    fn committed(text: &str, code: &str) -> StreamEvent {
        StreamEvent::CommittedSegment {
            role: TextRole::Recognition,
            segment_id: SegmentId("segment:1".into()),
            text: text.into(),
            words: vec![TimedToken {
                text: "hello".into(),
                range: TimeRange {
                    start_ms: 10,
                    end_ms: 80,
                },
                confidence: None,
            }],
            language: Some(language(code)),
            speaker_id: Some("speaker:0".into()),
            confidence: Some(Confidence {
                value: 0.91,
                scale: ConfidenceScale::Probability,
                calibration: None,
            }),
        }
    }

    fn source() -> TranscriptSourceMetadata {
        TranscriptSourceMetadata {
            event_ref: None,
            event_times: None,
            provenance: None,
        }
    }

    #[test]
    fn raw_display_and_downstream_text_remain_distinct() {
        let normalizer = RuleBasedTranscriptNormalizer::new(TranscriptNormalizationConfig {
            disfluency_policy: DisfluencyPolicy::RemoveFilledPauses,
            non_speech_annotations: NonSpeechAnnotationPolicy::RemoveBracketed,
            ..TranscriptNormalizationConfig::default()
        });
        let StreamEvent::CommittedSegment {
            segment_id,
            text,
            words,
            language,
            speaker_id,
            confidence,
            ..
        } = committed("<|en|> um call five five five [noise]", "en")
        else {
            unreachable!()
        };
        let normalized = normalizer
            .normalize(
                segment_id,
                text,
                words,
                language,
                speaker_id,
                confidence,
                source(),
            )
            .unwrap();
        assert_eq!(normalized.raw_text, "<|en|> um call five five five [noise]");
        assert_eq!(normalized.display_text, "Um call five five five [noise].");
        assert_eq!(normalized.downstream_text, "Call 555.");
        assert_eq!(normalized.speaker_id.as_deref(), Some("speaker:0"));
        assert_eq!(normalized.words[0].range.start_ms, 10);
    }

    #[test]
    fn language_rules_are_selected_from_recognition_metadata() {
        let normalizer = RuleBasedTranscriptNormalizer::new(TranscriptNormalizationConfig {
            disfluency_policy: DisfluencyPolicy::RemoveFilledPauses,
            ..TranscriptNormalizationConfig::default()
        });
        let StreamEvent::CommittedSegment {
            segment_id,
            text,
            words,
            language,
            speaker_id,
            confidence,
            ..
        } = committed("euh appelle deux trois", "fr")
        else {
            unreachable!()
        };
        let normalized = normalizer
            .normalize(
                segment_id,
                text,
                words,
                language,
                speaker_id,
                confidence,
                source(),
            )
            .unwrap();
        assert_eq!(normalized.display_text, "Euh appelle deux trois.");
        assert_eq!(normalized.downstream_text, "Appelle 23.");
    }

    #[test]
    fn partials_and_revisions_cannot_trigger_downstream_work() {
        let mut pipeline = CommittedTranscriptPipeline::new(
            RuleBasedTranscriptNormalizer::default(),
            StructuralTranscriptInterpreter,
        );
        let partial = StreamEvent::PartialHypothesis {
            role: TextRole::Recognition,
            segment_id: SegmentId("segment:1".into()),
            text: "turn on".into(),
            confidence: None,
        };
        let revised = StreamEvent::RevisedHypothesis {
            role: TextRole::Recognition,
            segment_id: SegmentId("segment:1".into()),
            replaces: TextRange { start: 5, end: 7 },
            text: "off".into(),
            confidence: None,
        };
        assert!(
            pipeline
                .process_event(&partial, source())
                .unwrap()
                .is_none()
        );
        assert!(
            pipeline
                .process_event(&revised, source())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn committed_segments_feed_parser_and_interpreter_incrementally() {
        let mut pipeline = CommittedTranscriptPipeline::new(
            RuleBasedTranscriptNormalizer::default(),
            StructuralTranscriptInterpreter,
        );
        let output = pipeline
            .process_event(&committed("hello world", "en"), source())
            .unwrap()
            .unwrap();
        assert_eq!(output.syntax.tokens.len(), 2);
        assert_eq!(output.interpretation["text"], "Hello world.");
        assert_eq!(
            output
                .derived_events()
                .iter()
                .filter(|event| matches!(event, StreamEvent::DerivedArtifact { .. }))
                .count(),
            3
        );
    }

    #[test]
    fn committed_text_cannot_be_reprocessed_after_later_context() {
        let mut pipeline = CommittedTranscriptPipeline::new(
            RuleBasedTranscriptNormalizer::default(),
            StructuralTranscriptInterpreter,
        );
        pipeline
            .process_event(&committed("hello", "en"), source())
            .unwrap();
        assert!(matches!(
            pipeline
                .process_event(&committed("hello?", "en"), source())
                .unwrap_err(),
            TranscriptPipelineError::DuplicateCommit(segment) if segment == "segment:1"
        ));
    }

    #[test]
    fn envelope_provenance_and_times_survive_export() {
        let envelope = StreamEventEnvelope {
            schema_version: 1,
            stream_id: StreamId("stream:1".into()),
            event_id: EventId("event:1".into()),
            sequence: 1,
            times: EventTimes {
                occurred_at: EventTime {
                    origin: ClockOrigin::StreamStart,
                    offset_ms: 42,
                },
                observed_at: EventTime {
                    origin: ClockOrigin::UnixEpoch,
                    offset_ms: 84,
                },
            },
            provenance: Provenance {
                kind: ProvenanceKind::Direct,
                sources: Vec::new(),
                provider: Some("fixture".into()),
                model: Some("fixture-v1".into()),
                attributes: Default::default(),
            },
            event: committed("hello", "en"),
        };
        let mut pipeline = CommittedTranscriptPipeline::new(
            RuleBasedTranscriptNormalizer::default(),
            StructuralTranscriptInterpreter,
        );
        let output = pipeline.process_envelope(&envelope).unwrap().unwrap();
        assert_eq!(
            output
                .transcript
                .source
                .event_ref
                .as_ref()
                .unwrap()
                .event_id
                .0,
            "event:1"
        );
        assert_eq!(
            output
                .transcript
                .source
                .event_times
                .as_ref()
                .unwrap()
                .occurred_at
                .offset_ms,
            42
        );
        assert_eq!(
            output
                .transcript
                .source
                .provenance
                .as_ref()
                .unwrap()
                .model
                .as_deref(),
            Some("fixture-v1")
        );
        let jsonl = transcript_export_jsonl([output.transcript]).unwrap();
        assert!(jsonl.contains("\"raw_text\":\"hello\""));
        assert!(jsonl.contains("\"display_text\":\"Hello.\""));
        assert!(jsonl.contains("\"speaker_id\":\"speaker:0\""));
    }

    #[test]
    fn language_primary_subtag_is_stable() {
        assert_eq!(primary_language("fr-CA"), "fr");
        assert_eq!(primary_language("EN_us"), "en");
    }
}
