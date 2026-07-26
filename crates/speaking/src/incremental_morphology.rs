//! Incremental, model-neutral morpheme analysis for evolving Unicode text.
//!
//! The analyzer owns occurrence identity and reversible segmentation. Lexical
//! [`MorphemeId`] values continue to come from variety data; unknown varieties
//! produce one explicitly unknown occurrence per whole word.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use serde::{Deserialize, Serialize};
use unicode_normalization::char::canonical_combining_class;
use unicode_segmentation::UnicodeSegmentation;

use crate::data::varieties::english::morphology::decompose_word;
use crate::data::variety_by_code;
use crate::duplex::{
    BeliefAction, BeliefEvent, BeliefEventJournal, EvidenceAnchor, EvidenceDelta, EvidenceFinality,
    EvidencePayload, EvidenceState, Repair, RepairTarget, Withdrawal, WithdrawalTarget,
};
use crate::evidence::{EvidenceProvenance, EvidenceSource};
use crate::feature::FeatureBundle;
use crate::ids::{MorphemeId, MorphemeOccurrenceId, UtteranceId, VarietyId};
use crate::morphology::{
    MorphemeKind, MorphemeToken, compose_morpheme_tokens, finalize_word_pronunciation,
};
use crate::phonology::PhonemeToken;
use crate::spec::Spec;
use crate::time::TextSpan;

pub const MORPHEME_DELTA_JOURNAL_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisFinality {
    Provisional,
    Final,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphemeCandidate {
    pub text: String,
    pub span: TextSpan,
    pub variety: VarietyId,
    pub finality: AnalysisFinality,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WordCandidate {
    pub text: String,
    pub span: TextSpan,
    pub variety: VarietyId,
    pub finality: AnalysisFinality,
    #[serde(default)]
    pub morpheme_occurrences: Vec<MorphemeOccurrenceId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MorphemeOccurrence {
    pub id: MorphemeOccurrenceId,
    pub morpheme: Spec<MorphemeId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<MorphemeKind>,
    /// Exact observed source text, even when the lexical form has an
    /// orthographic mutation such as `happy` -> `happi-`.
    pub surface: String,
    pub span: TextSpan,
    pub word_span: TextSpan,
    pub variety: VarietyId,
    pub finality: AnalysisFinality,
    #[serde(default)]
    pub features: FeatureBundle,
    #[serde(default)]
    pub pronunciation: Vec<PhonemeToken>,
    pub confidence: f32,
    pub provenance: EvidenceProvenance,
}

impl MorphemeOccurrence {
    pub fn as_morpheme_token(&self) -> MorphemeToken {
        MorphemeToken {
            morpheme: self.morpheme.clone(),
            surface: self.surface.clone(),
            span: Some(self.span),
            features: self.features.clone(),
            pronunciation: self.pronunciation.clone(),
            confidence: self.confidence,
        }
    }

    pub fn as_evidence_delta(&self) -> EvidenceDelta {
        EvidenceDelta {
            anchor: EvidenceAnchor::MorphemeOccurrence(self.id.clone()),
            state: EvidenceState::LinguisticInference,
            confidence: self.confidence,
            provenance: self.provenance.clone(),
            payload: EvidencePayload {
                text: Some(self.surface.clone()),
                text_span: Some(self.span),
                morphology: vec![self.as_morpheme_token()],
                pronunciation: self.pronunciation.clone(),
                finality: Some(match self.finality {
                    AnalysisFinality::Provisional => EvidenceFinality::Provisional,
                    AnalysisFinality::Final => EvidenceFinality::Final,
                }),
                ..EvidencePayload::default()
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "data")]
pub enum MorphemeAnalysisDelta {
    Append {
        occurrence: MorphemeOccurrence,
    },
    Replace {
        previous: MorphemeOccurrence,
        replacement: MorphemeOccurrence,
    },
    Split {
        withdrawn: MorphemeOccurrence,
        replacements: Vec<MorphemeOccurrence>,
    },
    Merge {
        withdrawn: Vec<MorphemeOccurrence>,
        replacement: MorphemeOccurrence,
    },
    Finalize {
        occurrence: MorphemeOccurrence,
    },
}

impl MorphemeAnalysisDelta {
    /// Project an analyzer delta into the provider-neutral duplex contract.
    ///
    /// A same-ID update is an explicit repair. A changed segmentation first
    /// withdraws every superseded occurrence and then appends its replacements.
    pub fn belief_actions(&self) -> Vec<BeliefAction> {
        let withdrawal = |occurrence: &MorphemeOccurrence, reason: &str| {
            BeliefAction::Withdraw(Withdrawal {
                target: WithdrawalTarget::MorphemeOccurrence(occurrence.id.clone()),
                reason: reason.into(),
                provenance: occurrence.provenance.clone(),
            })
        };
        match self {
            Self::Append { occurrence } => {
                vec![BeliefAction::ApplyEvidenceDelta(
                    occurrence.as_evidence_delta(),
                )]
            }
            Self::Replace {
                previous,
                replacement,
            } if previous.id == replacement.id => {
                vec![BeliefAction::Repair(Repair {
                    target: RepairTarget::MorphemeOccurrence(previous.id.clone()),
                    replacement: replacement.as_evidence_delta(),
                    reason: "incremental morpheme occurrence revised".into(),
                    provenance: replacement.provenance.clone(),
                })]
            }
            Self::Replace {
                previous,
                replacement,
            } => vec![
                withdrawal(previous, "morpheme segmentation replaced"),
                BeliefAction::ApplyEvidenceDelta(replacement.as_evidence_delta()),
            ],
            Self::Split {
                withdrawn,
                replacements,
            } => {
                let mut actions = vec![withdrawal(withdrawn, "morpheme occurrence split")];
                actions.extend(replacements.iter().map(|occurrence| {
                    BeliefAction::ApplyEvidenceDelta(occurrence.as_evidence_delta())
                }));
                actions
            }
            Self::Merge {
                withdrawn,
                replacement,
            } => {
                let mut actions = withdrawn
                    .iter()
                    .map(|occurrence| withdrawal(occurrence, "morpheme occurrences merged"))
                    .collect::<Vec<_>>();
                actions.push(BeliefAction::ApplyEvidenceDelta(
                    replacement.as_evidence_delta(),
                ));
                actions
            }
            Self::Finalize { occurrence } => {
                vec![BeliefAction::Repair(Repair {
                    target: RepairTarget::MorphemeOccurrence(occurrence.id.clone()),
                    replacement: occurrence.as_evidence_delta(),
                    reason: "morpheme occurrence finalized".into(),
                    provenance: occurrence.provenance.clone(),
                })]
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MorphemeAnalysisEvent {
    pub sequence: u64,
    pub delta: MorphemeAnalysisDelta,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MorphemeDeltaJournal {
    pub version: u32,
    pub utterance_id: UtteranceId,
    pub default_variety: VarietyId,
    #[serde(default)]
    pub events: Vec<MorphemeAnalysisEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MorphemeAnalysisState {
    pub utterance_id: UtteranceId,
    pub default_variety: VarietyId,
    pub revision: u64,
    #[serde(default)]
    pub active: BTreeMap<MorphemeOccurrenceId, MorphemeOccurrence>,
    #[serde(default)]
    pub withdrawn: Vec<MorphemeOccurrence>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MorphemeJournalError {
    UnsupportedVersion { expected: u32, found: u32 },
    OutOfOrderEvent { expected: u64, found: u64 },
    DuplicateOccurrence(MorphemeOccurrenceId),
    UnknownOccurrence(MorphemeOccurrenceId),
}

impl fmt::Display for MorphemeJournalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion { expected, found } => {
                write!(
                    formatter,
                    "unsupported morpheme journal version {found}; expected {expected}"
                )
            }
            Self::OutOfOrderEvent { expected, found } => {
                write!(
                    formatter,
                    "out-of-order morpheme event {found}; expected {expected}"
                )
            }
            Self::DuplicateOccurrence(id) => {
                write!(formatter, "duplicate morpheme occurrence '{}'", id.0)
            }
            Self::UnknownOccurrence(id) => {
                write!(formatter, "unknown morpheme occurrence '{}'", id.0)
            }
        }
    }
}

impl std::error::Error for MorphemeJournalError {}

impl MorphemeAnalysisState {
    fn apply(&mut self, delta: &MorphemeAnalysisDelta) -> Result<(), MorphemeJournalError> {
        let insert = |state: &mut Self,
                      occurrence: &MorphemeOccurrence|
         -> Result<(), MorphemeJournalError> {
            if state
                .active
                .insert(occurrence.id.clone(), occurrence.clone())
                .is_some()
            {
                return Err(MorphemeJournalError::DuplicateOccurrence(
                    occurrence.id.clone(),
                ));
            }
            Ok(())
        };
        let withdraw = |state: &mut Self,
                        occurrence: &MorphemeOccurrence|
         -> Result<(), MorphemeJournalError> {
            state
                .active
                .remove(&occurrence.id)
                .ok_or_else(|| MorphemeJournalError::UnknownOccurrence(occurrence.id.clone()))?;
            state.withdrawn.push(occurrence.clone());
            Ok(())
        };

        match delta {
            MorphemeAnalysisDelta::Append { occurrence } => insert(self, occurrence)?,
            MorphemeAnalysisDelta::Replace {
                previous,
                replacement,
            } => {
                self.active
                    .remove(&previous.id)
                    .ok_or_else(|| MorphemeJournalError::UnknownOccurrence(previous.id.clone()))?;
                if previous.id != replacement.id {
                    self.withdrawn.push(previous.clone());
                }
                insert(self, replacement)?;
            }
            MorphemeAnalysisDelta::Split {
                withdrawn,
                replacements,
            } => {
                withdraw(self, withdrawn)?;
                for occurrence in replacements {
                    insert(self, occurrence)?;
                }
            }
            MorphemeAnalysisDelta::Merge {
                withdrawn,
                replacement,
            } => {
                for occurrence in withdrawn {
                    withdraw(self, occurrence)?;
                }
                insert(self, replacement)?;
            }
            MorphemeAnalysisDelta::Finalize { occurrence } => {
                let existing = self.active.get_mut(&occurrence.id).ok_or_else(|| {
                    MorphemeJournalError::UnknownOccurrence(occurrence.id.clone())
                })?;
                *existing = occurrence.clone();
            }
        }
        self.revision += 1;
        Ok(())
    }
}

pub fn replay_morpheme_journal(
    journal: &MorphemeDeltaJournal,
) -> Result<MorphemeAnalysisState, MorphemeJournalError> {
    if journal.version != MORPHEME_DELTA_JOURNAL_VERSION {
        return Err(MorphemeJournalError::UnsupportedVersion {
            expected: MORPHEME_DELTA_JOURNAL_VERSION,
            found: journal.version,
        });
    }
    let mut state = MorphemeAnalysisState {
        utterance_id: journal.utterance_id.clone(),
        default_variety: journal.default_variety.clone(),
        revision: 0,
        active: BTreeMap::new(),
        withdrawn: Vec::new(),
    };
    for (expected, event) in journal.events.iter().enumerate() {
        if event.sequence != expected as u64 {
            return Err(MorphemeJournalError::OutOfOrderEvent {
                expected: expected as u64,
                found: event.sequence,
            });
        }
        state.apply(&event.delta)?;
    }
    Ok(state)
}

pub fn append_to_duplex_journal(
    journal: &mut BeliefEventJournal,
    deltas: &[MorphemeAnalysisDelta],
) {
    for action in deltas
        .iter()
        .flat_map(MorphemeAnalysisDelta::belief_actions)
    {
        journal.events.push(BeliefEvent {
            sequence: journal.events.len() as u64,
            action,
        });
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "data")]
pub enum IncomingTextDelta {
    Utf8 {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        variety: Option<VarietyId>,
    },
    Bytes {
        bytes: Vec<u8>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        variety: Option<VarietyId>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IncrementalMorphemeUpdate {
    #[serde(default)]
    pub graphemes: Vec<GraphemeCandidate>,
    #[serde(default)]
    pub words: Vec<WordCandidate>,
    #[serde(default)]
    pub occurrences: Vec<MorphemeOccurrence>,
    #[serde(default)]
    pub deltas: Vec<MorphemeAnalysisDelta>,
    pub pending_utf8_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IncrementalMorphemeError {
    Finished,
    InvalidUtf8 {
        valid_up_to: usize,
        error_len: usize,
    },
    IncompleteUtf8 {
        pending_bytes: usize,
    },
    VarietyChangedInsideCodepoint {
        pending_variety: VarietyId,
        requested_variety: VarietyId,
    },
}

impl fmt::Display for IncrementalMorphemeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Finished => write!(
                formatter,
                "incremental morpheme analyzer is already finished"
            ),
            Self::InvalidUtf8 {
                valid_up_to,
                error_len,
            } => write!(
                formatter,
                "invalid UTF-8 text delta at byte {valid_up_to} (invalid length {error_len})"
            ),
            Self::IncompleteUtf8 { pending_bytes } => write!(
                formatter,
                "end of stream has {pending_bytes} bytes of an incomplete UTF-8 scalar"
            ),
            Self::VarietyChangedInsideCodepoint {
                pending_variety,
                requested_variety,
            } => write!(
                formatter,
                "cannot change variety from '{}' to '{}' inside an incomplete UTF-8 scalar",
                pending_variety.0, requested_variety.0
            ),
        }
    }
}

impl std::error::Error for IncrementalMorphemeError {}

#[derive(Debug, Clone)]
struct VarietyRun {
    span: TextSpan,
    variety: VarietyId,
}

#[derive(Debug, Clone)]
struct GraphemeInternal {
    byte_start: usize,
    byte_end: usize,
    candidate: GraphemeCandidate,
}

#[derive(Debug, Clone)]
pub struct IncrementalMorphemeAnalyzer {
    utterance_id: UtteranceId,
    default_variety: VarietyId,
    text: String,
    runs: Vec<VarietyRun>,
    pending_utf8: Vec<u8>,
    pending_variety: Option<VarietyId>,
    occurrences: Vec<MorphemeOccurrence>,
    graphemes: Vec<GraphemeCandidate>,
    words: Vec<WordCandidate>,
    next_occurrence: u64,
    finished: bool,
    journal: MorphemeDeltaJournal,
}

impl IncrementalMorphemeAnalyzer {
    pub fn new(utterance_id: UtteranceId, default_variety: VarietyId) -> Self {
        Self {
            journal: MorphemeDeltaJournal {
                version: MORPHEME_DELTA_JOURNAL_VERSION,
                utterance_id: utterance_id.clone(),
                default_variety: default_variety.clone(),
                events: Vec::new(),
            },
            utterance_id,
            default_variety,
            text: String::new(),
            runs: Vec::new(),
            pending_utf8: Vec::new(),
            pending_variety: None,
            occurrences: Vec::new(),
            graphemes: Vec::new(),
            words: Vec::new(),
            next_occurrence: 0,
            finished: false,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn occurrences(&self) -> &[MorphemeOccurrence] {
        &self.occurrences
    }

    pub fn journal(&self) -> &MorphemeDeltaJournal {
        &self.journal
    }

    pub fn apply(
        &mut self,
        delta: IncomingTextDelta,
    ) -> Result<IncrementalMorphemeUpdate, IncrementalMorphemeError> {
        match delta {
            IncomingTextDelta::Utf8 { text, variety } => {
                let variety = variety.unwrap_or_else(|| self.default_variety.clone());
                self.push_str_for_variety(&text, variety)
            }
            IncomingTextDelta::Bytes { bytes, variety } => {
                let variety = variety.unwrap_or_else(|| self.default_variety.clone());
                self.push_bytes_for_variety(&bytes, variety)
            }
        }
    }

    pub fn push_str(
        &mut self,
        text: &str,
    ) -> Result<IncrementalMorphemeUpdate, IncrementalMorphemeError> {
        self.push_str_for_variety(text, self.default_variety.clone())
    }

    pub fn push_str_for_variety(
        &mut self,
        text: &str,
        variety: VarietyId,
    ) -> Result<IncrementalMorphemeUpdate, IncrementalMorphemeError> {
        self.push_bytes_for_variety(text.as_bytes(), variety)
    }

    pub fn push_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<IncrementalMorphemeUpdate, IncrementalMorphemeError> {
        self.push_bytes_for_variety(bytes, self.default_variety.clone())
    }

    pub fn push_bytes_for_variety(
        &mut self,
        bytes: &[u8],
        variety: VarietyId,
    ) -> Result<IncrementalMorphemeUpdate, IncrementalMorphemeError> {
        if self.finished {
            return Err(IncrementalMorphemeError::Finished);
        }
        if !self.pending_utf8.is_empty() && self.pending_variety.as_ref() != Some(&variety) {
            return Err(IncrementalMorphemeError::VarietyChangedInsideCodepoint {
                pending_variety: self
                    .pending_variety
                    .clone()
                    .unwrap_or_else(|| self.default_variety.clone()),
                requested_variety: variety,
            });
        }

        let mut combined = self.pending_utf8.clone();
        combined.extend_from_slice(bytes);
        match std::str::from_utf8(&combined) {
            Ok(valid) => {
                let owned = valid.to_string();
                self.pending_utf8.clear();
                self.pending_variety = None;
                self.append_valid_text(&owned, variety)
            }
            Err(error) if error.error_len().is_none() => {
                let valid_up_to = error.valid_up_to();
                let valid = std::str::from_utf8(&combined[..valid_up_to])
                    .expect("Utf8Error valid_up_to always ends at a scalar boundary")
                    .to_string();
                self.pending_utf8 = combined[valid_up_to..].to_vec();
                self.pending_variety = Some(variety.clone());
                if valid.is_empty() {
                    Ok(self.current_update(Vec::new()))
                } else {
                    self.append_valid_text(&valid, variety)
                }
            }
            Err(error) => Err(IncrementalMorphemeError::InvalidUtf8 {
                valid_up_to: error.valid_up_to(),
                error_len: error.error_len().unwrap_or(1),
            }),
        }
    }

    pub fn finish(&mut self) -> Result<IncrementalMorphemeUpdate, IncrementalMorphemeError> {
        if self.finished {
            return Err(IncrementalMorphemeError::Finished);
        }
        if !self.pending_utf8.is_empty() {
            return Err(IncrementalMorphemeError::IncompleteUtf8 {
                pending_bytes: self.pending_utf8.len(),
            });
        }
        self.finished = true;
        Ok(self.reanalyze())
    }

    fn append_valid_text(
        &mut self,
        text: &str,
        variety: VarietyId,
    ) -> Result<IncrementalMorphemeUpdate, IncrementalMorphemeError> {
        if text.is_empty() {
            return Ok(self.current_update(Vec::new()));
        }
        let start_char = self.text.chars().count();
        self.text.push_str(text);
        let end_char = self.text.chars().count();
        if let Some(last) = self.runs.last_mut()
            && last.variety == variety
            && last.span.end_char == start_char
        {
            last.span.end_char = end_char;
        } else {
            self.runs.push(VarietyRun {
                span: TextSpan {
                    start_char,
                    end_char,
                },
                variety,
            });
        }
        Ok(self.reanalyze())
    }

    fn reanalyze(&mut self) -> IncrementalMorphemeUpdate {
        let graphemes = self.segment_graphemes();
        let mut words = self.segment_words(&graphemes);
        let desired = words
            .iter()
            .flat_map(|word| self.analyze_word(word))
            .collect::<Vec<_>>();
        let (occurrences, deltas) = self.diff_occurrences(desired);

        for word in &mut words {
            word.morpheme_occurrences = occurrences
                .iter()
                .filter(|occurrence| {
                    occurrence.word_span == word.span && occurrence.variety == word.variety
                })
                .map(|occurrence| occurrence.id.clone())
                .collect();
        }

        for delta in &deltas {
            self.journal.events.push(MorphemeAnalysisEvent {
                sequence: self.journal.events.len() as u64,
                delta: delta.clone(),
            });
        }
        self.graphemes = graphemes
            .iter()
            .map(|grapheme| grapheme.candidate.clone())
            .collect();
        self.words = words;
        self.occurrences = occurrences;
        self.current_update(deltas)
    }

    fn current_update(&self, deltas: Vec<MorphemeAnalysisDelta>) -> IncrementalMorphemeUpdate {
        IncrementalMorphemeUpdate {
            graphemes: self.graphemes.clone(),
            words: self.words.clone(),
            occurrences: self.occurrences.clone(),
            deltas,
            pending_utf8_bytes: self.pending_utf8.len(),
        }
    }

    fn segment_graphemes(&self) -> Vec<GraphemeInternal> {
        // A variety run is also a hard linguistic boundary. Segment each run
        // independently so a combining mark tagged as a new variety cannot
        // silently extend a grapheme owned by the preceding variety.
        let mut raw = Vec::new();
        for run in &self.runs {
            let run_byte_start = byte_at_char(&self.text, run.span.start_char);
            let run_byte_end = byte_at_char(&self.text, run.span.end_char);
            raw.extend(
                self.text[run_byte_start..run_byte_end]
                    .grapheme_indices(true)
                    .map(|(relative_start, grapheme)| (run_byte_start + relative_start, grapheme)),
            );
        }
        raw.iter()
            .enumerate()
            .map(|(index, (byte_start, grapheme))| {
                let byte_end = raw
                    .get(index + 1)
                    .map(|(start, _)| *start)
                    .unwrap_or(self.text.len());
                let span = TextSpan {
                    start_char: self.text[..*byte_start].chars().count(),
                    end_char: self.text[..byte_end].chars().count(),
                };
                GraphemeInternal {
                    byte_start: *byte_start,
                    byte_end,
                    candidate: GraphemeCandidate {
                        text: (*grapheme).to_string(),
                        span,
                        variety: self.variety_at(span.start_char),
                        finality: if self.finished || index + 1 < raw.len() {
                            AnalysisFinality::Final
                        } else {
                            AnalysisFinality::Provisional
                        },
                    },
                }
            })
            .collect()
    }

    fn segment_words(&self, graphemes: &[GraphemeInternal]) -> Vec<WordCandidate> {
        let mut words = Vec::new();
        let mut start_index: Option<usize> = None;
        let mut active_variety: Option<VarietyId> = None;

        let flush = |words: &mut Vec<WordCandidate>,
                     start_index: &mut Option<usize>,
                     end_index: usize,
                     finality: AnalysisFinality,
                     active_variety: &mut Option<VarietyId>| {
            let Some(start) = start_index.take() else {
                return;
            };
            let first = &graphemes[start];
            let end_byte = if end_index > start {
                graphemes[end_index - 1].byte_end
            } else {
                first.byte_end
            };
            let span = TextSpan {
                start_char: first.candidate.span.start_char,
                end_char: self.text[..end_byte].chars().count(),
            };
            words.push(WordCandidate {
                text: self.text[first.byte_start..end_byte].to_string(),
                span,
                variety: active_variety
                    .take()
                    .unwrap_or_else(|| first.candidate.variety.clone()),
                finality,
                morpheme_occurrences: Vec::new(),
            });
        };

        for (index, grapheme) in graphemes.iter().enumerate() {
            if active_variety
                .as_ref()
                .is_some_and(|variety| variety != &grapheme.candidate.variety)
            {
                flush(
                    &mut words,
                    &mut start_index,
                    index,
                    AnalysisFinality::Final,
                    &mut active_variety,
                );
            }

            let core = grapheme_is_word_core(&grapheme.candidate.text);
            let connector = grapheme_is_word_connector(&grapheme.candidate.text)
                && start_index.is_some()
                && graphemes.get(index + 1).is_some_and(|next| {
                    next.candidate.variety == grapheme.candidate.variety
                        && grapheme_is_word_core(&next.candidate.text)
                });
            if core || connector {
                start_index.get_or_insert(index);
                active_variety.get_or_insert_with(|| grapheme.candidate.variety.clone());
            } else {
                flush(
                    &mut words,
                    &mut start_index,
                    index,
                    AnalysisFinality::Final,
                    &mut active_variety,
                );
            }
        }
        flush(
            &mut words,
            &mut start_index,
            graphemes.len(),
            if self.finished {
                AnalysisFinality::Final
            } else {
                AnalysisFinality::Provisional
            },
            &mut active_variety,
        );
        words
    }

    fn analyze_word(&self, word: &WordCandidate) -> Vec<MorphemeOccurrence> {
        let mut occurrences = Vec::new();
        let mut segment_start = word.span.start_char;
        for (local_index, character) in word.text.chars().enumerate() {
            if character == '-' {
                let segment_end = word.span.start_char + local_index;
                if segment_end > segment_start {
                    occurrences.extend(self.analyze_unhyphenated(
                        TextSpan {
                            start_char: segment_start,
                            end_char: segment_end,
                        },
                        word,
                    ));
                }
                segment_start = segment_end + 1;
            }
        }
        if segment_start < word.span.end_char {
            occurrences.extend(self.analyze_unhyphenated(
                TextSpan {
                    start_char: segment_start,
                    end_char: word.span.end_char,
                },
                word,
            ));
        }
        if occurrences.is_empty() {
            occurrences.push(self.fallback_occurrence(word.span, word));
        }
        occurrences
    }

    fn analyze_unhyphenated(
        &self,
        segment_span: TextSpan,
        word: &WordCandidate,
    ) -> Vec<MorphemeOccurrence> {
        let source = slice_chars(&self.text, segment_span);
        let lookup = source.replace('’', "'");
        let Some(variety) = variety_by_code(&word.variety.0) else {
            return vec![self.fallback_occurrence(segment_span, word)];
        };
        if variety.language.0 != "en" {
            return vec![self.fallback_occurrence(segment_span, word)];
        }
        let Some(mut tokens) = decompose_word(&variety, &lookup) else {
            return vec![self.fallback_occurrence(segment_span, word)];
        };

        assign_token_spans(&self.text, segment_span, &variety, &mut tokens);
        if let Some(morphology) = &variety.morphology {
            compose_morpheme_tokens(&mut tokens, &morphology.morphemes, &morphology.rules);
        }
        redistribute_final_word_pronunciation(&mut tokens);

        tokens
            .into_iter()
            .map(|token| {
                let span = token.span.unwrap_or(segment_span);
                let kind = match &token.morpheme {
                    Spec::Known(id) => variety
                        .morphology
                        .as_ref()
                        .and_then(|morphology| morphology.morphemes.get(id))
                        .map(|morpheme| morpheme.kind)
                        .or(Some(MorphemeKind::Root)),
                    _ => None,
                };
                MorphemeOccurrence {
                    id: MorphemeOccurrenceId(String::new()),
                    morpheme: token.morpheme,
                    kind,
                    surface: slice_chars(&self.text, span).to_string(),
                    span,
                    word_span: word.span,
                    variety: word.variety.clone(),
                    finality: word.finality,
                    features: token.features,
                    pronunciation: token.pronunciation,
                    confidence: token.confidence,
                    provenance: EvidenceProvenance {
                        source: EvidenceSource::Rule,
                        method: "english-incremental-morphology".into(),
                        version: Some("1".into()),
                    },
                }
            })
            .collect()
    }

    fn fallback_occurrence(&self, span: TextSpan, word: &WordCandidate) -> MorphemeOccurrence {
        MorphemeOccurrence {
            id: MorphemeOccurrenceId(String::new()),
            morpheme: Spec::Unknown,
            kind: None,
            surface: slice_chars(&self.text, span).to_string(),
            span,
            word_span: word.span,
            variety: word.variety.clone(),
            finality: word.finality,
            features: FeatureBundle::default(),
            pronunciation: Vec::new(),
            confidence: 0.0,
            provenance: EvidenceProvenance {
                source: EvidenceSource::Unknown,
                method: "language-neutral-whole-word-fallback".into(),
                version: Some("1".into()),
            },
        }
    }

    fn diff_occurrences(
        &mut self,
        mut desired: Vec<MorphemeOccurrence>,
    ) -> (Vec<MorphemeOccurrence>, Vec<MorphemeAnalysisDelta>) {
        let previous = self.occurrences.clone();
        let mut previous_by_key: HashMap<OccurrenceKey, Vec<usize>> = HashMap::new();
        for (index, occurrence) in previous.iter().enumerate() {
            previous_by_key
                .entry(OccurrenceKey::of(occurrence))
                .or_default()
                .push(index);
        }

        let mut matched_previous = vec![false; previous.len()];
        let mut matched_desired = vec![false; desired.len()];
        let mut deltas = Vec::new();
        for (desired_index, occurrence) in desired.iter_mut().enumerate() {
            let Some(indices) = previous_by_key.get_mut(&OccurrenceKey::of(occurrence)) else {
                continue;
            };
            let Some(previous_index) = indices.pop() else {
                continue;
            };
            let old = &previous[previous_index];
            occurrence.id = old.id.clone();
            matched_previous[previous_index] = true;
            matched_desired[desired_index] = true;

            let only_finalized = old.finality == AnalysisFinality::Provisional
                && occurrence.finality == AnalysisFinality::Final
                && {
                    let mut without_finality = occurrence.clone();
                    without_finality.finality = old.finality;
                    &without_finality == old
                };
            if only_finalized {
                deltas.push(MorphemeAnalysisDelta::Finalize {
                    occurrence: occurrence.clone(),
                });
            } else if occurrence != old {
                deltas.push(MorphemeAnalysisDelta::Replace {
                    previous: old.clone(),
                    replacement: occurrence.clone(),
                });
            }
        }

        let mut groups: BTreeMap<(usize, String), (Vec<usize>, Vec<usize>)> = BTreeMap::new();
        for (index, occurrence) in previous.iter().enumerate() {
            if !matched_previous[index] {
                groups
                    .entry((
                        occurrence.word_span.start_char,
                        occurrence.variety.0.clone(),
                    ))
                    .or_default()
                    .0
                    .push(index);
            }
        }
        for (index, occurrence) in desired.iter().enumerate() {
            if !matched_desired[index] {
                groups
                    .entry((
                        occurrence.word_span.start_char,
                        occurrence.variety.0.clone(),
                    ))
                    .or_default()
                    .1
                    .push(index);
            }
        }

        for (_, (old_indices, new_indices)) in groups {
            let old = old_indices
                .iter()
                .map(|index| previous[*index].clone())
                .collect::<Vec<_>>();
            let mut new = new_indices
                .iter()
                .map(|index| desired[*index].clone())
                .collect::<Vec<_>>();
            for occurrence in &mut new {
                occurrence.id = self.allocate_occurrence_id();
            }
            for (index, occurrence) in new_indices.iter().zip(&new) {
                desired[*index] = occurrence.clone();
            }

            match (old.len(), new.len()) {
                (0, _) => {
                    deltas.extend(
                        new.into_iter()
                            .map(|occurrence| MorphemeAnalysisDelta::Append { occurrence }),
                    );
                }
                (_, 0) => {
                    // Append-only text cannot erase a word. Retain the old
                    // occurrence defensively if a future tokenizer extension
                    // produces such a transition.
                    desired.extend(old);
                }
                (1, 1) => deltas.push(MorphemeAnalysisDelta::Replace {
                    previous: old[0].clone(),
                    replacement: new[0].clone(),
                }),
                (1, _) => deltas.push(MorphemeAnalysisDelta::Split {
                    withdrawn: old[0].clone(),
                    replacements: new,
                }),
                (_, 1) => deltas.push(MorphemeAnalysisDelta::Merge {
                    withdrawn: old,
                    replacement: new[0].clone(),
                }),
                (old_len, new_len) if old_len == new_len => {
                    deltas.extend(old.into_iter().zip(new).map(|(previous, replacement)| {
                        MorphemeAnalysisDelta::Replace {
                            previous,
                            replacement,
                        }
                    }));
                }
                (old_len, new_len) if old_len < new_len => {
                    let paired = old_len - 1;
                    for index in 0..paired {
                        deltas.push(MorphemeAnalysisDelta::Replace {
                            previous: old[index].clone(),
                            replacement: new[index].clone(),
                        });
                    }
                    deltas.push(MorphemeAnalysisDelta::Split {
                        withdrawn: old[paired].clone(),
                        replacements: new[paired..].to_vec(),
                    });
                }
                (_, new_len) => {
                    let paired = new_len - 1;
                    for index in 0..paired {
                        deltas.push(MorphemeAnalysisDelta::Replace {
                            previous: old[index].clone(),
                            replacement: new[index].clone(),
                        });
                    }
                    deltas.push(MorphemeAnalysisDelta::Merge {
                        withdrawn: old[paired..].to_vec(),
                        replacement: new[paired].clone(),
                    });
                }
            }
        }

        desired.sort_by(|left, right| {
            left.span
                .start_char
                .cmp(&right.span.start_char)
                .then_with(|| left.span.end_char.cmp(&right.span.end_char))
                .then_with(|| left.id.0.cmp(&right.id.0))
        });
        deltas.sort_by_key(delta_start);
        (desired, deltas)
    }

    fn allocate_occurrence_id(&mut self) -> MorphemeOccurrenceId {
        let id = MorphemeOccurrenceId(format!("{}:m{}", self.utterance_id.0, self.next_occurrence));
        self.next_occurrence += 1;
        id
    }

    fn variety_at(&self, char_index: usize) -> VarietyId {
        self.runs
            .iter()
            .find(|run| run.span.start_char <= char_index && char_index < run.span.end_char)
            .map(|run| run.variety.clone())
            .unwrap_or_else(|| self.default_variety.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OccurrenceKey {
    start_char: usize,
    variety: String,
    morpheme: Option<String>,
}

impl OccurrenceKey {
    fn of(occurrence: &MorphemeOccurrence) -> Self {
        Self {
            start_char: occurrence.span.start_char,
            variety: occurrence.variety.0.clone(),
            morpheme: match &occurrence.morpheme {
                Spec::Known(id) => Some(id.0.clone()),
                _ => None,
            },
        }
    }
}

fn delta_start(delta: &MorphemeAnalysisDelta) -> usize {
    match delta {
        MorphemeAnalysisDelta::Append { occurrence }
        | MorphemeAnalysisDelta::Finalize { occurrence } => occurrence.span.start_char,
        MorphemeAnalysisDelta::Replace { previous, .. } => previous.span.start_char,
        MorphemeAnalysisDelta::Split { withdrawn, .. } => withdrawn.span.start_char,
        MorphemeAnalysisDelta::Merge { withdrawn, .. } => withdrawn
            .first()
            .map(|occurrence| occurrence.span.start_char)
            .unwrap_or(usize::MAX),
    }
}

fn grapheme_is_word_core(grapheme: &str) -> bool {
    grapheme
        .chars()
        .any(|character| character.is_alphanumeric() || canonical_combining_class(character) != 0)
}

fn grapheme_is_word_connector(grapheme: &str) -> bool {
    matches!(grapheme, "'" | "’" | "-" | "_")
}

fn slice_chars(text: &str, span: TextSpan) -> &str {
    let start = byte_at_char(text, span.start_char);
    let end = byte_at_char(text, span.end_char);
    &text[start..end]
}

fn byte_at_char(text: &str, char_index: usize) -> usize {
    text.char_indices()
        .nth(char_index)
        .map(|(index, _)| index)
        .unwrap_or(text.len())
}

fn assign_token_spans(
    source_text: &str,
    segment_span: TextSpan,
    variety: &crate::variety::LinguisticVariety,
    tokens: &mut [MorphemeToken],
) {
    let kind = |token: &MorphemeToken| match &token.morpheme {
        Spec::Known(id) => variety
            .morphology
            .as_ref()
            .and_then(|morphology| morphology.morphemes.get(id))
            .map(|morpheme| morpheme.kind)
            .unwrap_or(MorphemeKind::Root),
        _ => MorphemeKind::Root,
    };
    let form_len = |token: &MorphemeToken| match &token.morpheme {
        Spec::Known(id) => variety
            .morphology
            .as_ref()
            .and_then(|morphology| morphology.morphemes.get(id))
            .map(|morpheme| morpheme.form.trim_matches('-').chars().count())
            .unwrap_or_else(|| token.surface.chars().count()),
        _ => token.surface.chars().count(),
    };

    let mut left = segment_span.start_char;
    let mut first_middle = 0;
    while first_middle < tokens.len() && kind(&tokens[first_middle]) == MorphemeKind::Prefix {
        let end = (left + form_len(&tokens[first_middle])).min(segment_span.end_char);
        tokens[first_middle].span = Some(TextSpan {
            start_char: left,
            end_char: end,
        });
        left = end;
        first_middle += 1;
    }

    let mut right = segment_span.end_char;
    let mut last_middle = tokens.len();
    while last_middle > first_middle
        && matches!(
            kind(&tokens[last_middle - 1]),
            MorphemeKind::Suffix | MorphemeKind::Clitic
        )
    {
        last_middle -= 1;
        let length = form_len(&tokens[last_middle]);
        let start = right.saturating_sub(length).max(left);
        tokens[last_middle].span = Some(TextSpan {
            start_char: start,
            end_char: right,
        });
        right = start;
    }

    for index in first_middle..last_middle {
        let remaining_tokens = last_middle - index;
        let end = if remaining_tokens == 1 {
            right
        } else {
            (left + form_len(&tokens[index])).min(right)
        };
        tokens[index].span = Some(TextSpan {
            start_char: left,
            end_char: end,
        });
        left = end;
    }

    for token in tokens {
        if let Some(span) = token.span {
            token.surface = slice_chars(source_text, span).to_string();
        }
    }
}

fn redistribute_final_word_pronunciation(tokens: &mut [MorphemeToken]) {
    let lengths = tokens
        .iter()
        .map(|token| token.pronunciation.len())
        .collect::<Vec<_>>();
    let mut pronunciation = tokens
        .iter()
        .flat_map(|token| token.pronunciation.clone())
        .collect::<Vec<_>>();
    finalize_word_pronunciation(&mut pronunciation);
    let mut start = 0;
    for (token, length) in tokens.iter_mut().zip(lengths) {
        let end = start + length;
        token.pronunciation = pronunciation[start..end].to_vec();
        start = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duplex::{DUPLEX_JOURNAL_VERSION, replay_journal};

    fn analyzer() -> IncrementalMorphemeAnalyzer {
        IncrementalMorphemeAnalyzer::new(
            UtteranceId("fixture".into()),
            VarietyId("en-US-GA".into()),
        )
    }

    #[test]
    fn split_utf8_and_extended_graphemes_are_safe_and_revisable() {
        let mut analyzer = analyzer();
        let encoded = "é".as_bytes();
        let first = analyzer.push_bytes(&encoded[..1]).expect("partial scalar");
        assert_eq!(first.pending_utf8_bytes, 1);
        assert!(first.occurrences.is_empty());

        let second = analyzer
            .push_bytes(&encoded[1..])
            .expect("completed scalar");
        assert_eq!(second.pending_utf8_bytes, 0);
        assert_eq!(second.graphemes[0].text, "é");
        let id = second.occurrences[0].id.clone();

        let update = analyzer.push_str("\u{301}").expect("combining mark");
        assert_eq!(update.graphemes.len(), 1);
        assert_eq!(update.occurrences[0].id, id);
        assert_eq!(update.occurrences[0].surface, "é\u{301}");

        let text_before_error = analyzer.text().to_string();
        assert!(matches!(
            analyzer.push_bytes(&[0xff]),
            Err(IncrementalMorphemeError::InvalidUtf8 { .. })
        ));
        assert_eq!(analyzer.text(), text_before_error);
    }

    #[test]
    fn stable_ids_survive_growth_in_a_later_word() {
        let mut analyzer = analyzer();
        let first = analyzer.push_str("cats happy").expect("first text");
        let cats = first
            .occurrences
            .iter()
            .find(|occurrence| occurrence.span.start_char == 0)
            .expect("cats occurrence")
            .id
            .clone();

        let next = analyzer.push_str("ness").expect("suffix");
        assert_eq!(
            next.occurrences
                .iter()
                .find(|occurrence| occurrence.span.start_char == 0)
                .expect("stable cats occurrence")
                .id,
            cats
        );
    }

    #[test]
    fn resegmentation_withdraws_a_superseded_whole_word() {
        let mut analyzer = analyzer();
        let first = analyzer.push_str("talkat").expect("provisional whole word");
        assert_eq!(first.occurrences.len(), 1);
        let superseded = first.occurrences[0].id.clone();

        let revised = analyzer.push_str("iveness").expect("derived word");
        let split = revised
            .deltas
            .iter()
            .find_map(|delta| match delta {
                MorphemeAnalysisDelta::Split {
                    withdrawn,
                    replacements,
                } => Some((withdrawn, replacements)),
                _ => None,
            })
            .expect("split delta");
        assert_eq!(split.0.id, superseded);
        assert_eq!(
            split
                .1
                .iter()
                .map(|occurrence| occurrence.surface.as_str())
                .collect::<Vec<_>>(),
            ["talk", "ative", "ness"]
        );
        for occurrence in split.1 {
            assert_eq!(
                occurrence.surface,
                slice_chars(analyzer.text(), occurrence.span)
            );
            assert_eq!(
                occurrence.word_span,
                TextSpan {
                    start_char: 0,
                    end_char: "talkativeness".chars().count(),
                }
            );
        }

        let replayed = replay_morpheme_journal(analyzer.journal()).expect("journal replay");
        assert!(replayed.withdrawn.iter().any(|item| item.id == superseded));
        assert_eq!(replayed.active.len(), 3);
    }

    #[test]
    fn deterministic_fixture_needs_no_checkpoint_or_download() {
        #[derive(Deserialize)]
        struct Fixture {
            version: u32,
            utterance_id: String,
            variety: String,
            chunks: Vec<String>,
            expected_delta_types: Vec<String>,
            expected_surfaces: Vec<String>,
        }

        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../fixtures/speaking/incremental_morphology_v1.json"
        ))
        .expect("fixture parses");
        assert_eq!(fixture.version, MORPHEME_DELTA_JOURNAL_VERSION);
        let mut analyzer = IncrementalMorphemeAnalyzer::new(
            UtteranceId(fixture.utterance_id),
            VarietyId(fixture.variety),
        );
        for chunk in fixture.chunks {
            analyzer.push_str(&chunk).expect("fixture chunk");
        }
        let delta_types = analyzer
            .journal()
            .events
            .iter()
            .map(|event| match event.delta {
                MorphemeAnalysisDelta::Append { .. } => "append",
                MorphemeAnalysisDelta::Replace { .. } => "replace",
                MorphemeAnalysisDelta::Split { .. } => "split",
                MorphemeAnalysisDelta::Merge { .. } => "merge",
                MorphemeAnalysisDelta::Finalize { .. } => "finalize",
            })
            .collect::<Vec<_>>();
        assert_eq!(delta_types, fixture.expected_delta_types);
        assert_eq!(
            analyzer
                .occurrences()
                .iter()
                .map(|occurrence| occurrence.surface.as_str())
                .collect::<Vec<_>>(),
            fixture.expected_surfaces
        );
        let json = serde_json::to_string(analyzer.journal()).expect("journal serializes");
        let reparsed: MorphemeDeltaJournal = serde_json::from_str(&json).expect("journal reparses");
        assert_eq!(reparsed, *analyzer.journal());
    }

    #[test]
    fn incomplete_end_of_stream_is_typed_and_unknown_varieties_fall_back() {
        let mut incomplete = analyzer();
        incomplete
            .push_bytes(&["é".as_bytes()[0]])
            .expect("partial scalar is buffered");
        assert!(matches!(
            incomplete.finish(),
            Err(IncrementalMorphemeError::IncompleteUtf8 { pending_bytes: 1 })
        ));

        let mut unknown = IncrementalMorphemeAnalyzer::new(
            UtteranceId("unknown".into()),
            VarietyId("qaa-Unknown".into()),
        );
        let update = unknown.push_str("xyzzy ").expect("fallback");
        assert_eq!(update.occurrences.len(), 1);
        assert_eq!(update.occurrences[0].morpheme, Spec::Unknown);
        assert_eq!(
            update.occurrences[0].provenance.method,
            "language-neutral-whole-word-fallback"
        );
    }

    #[test]
    fn source_spans_clitics_compounds_punctuation_and_code_switching_are_preserved() {
        let mut analyzer = analyzer();
        analyzer
            .push_str("unhappy dog's well-known. ")
            .expect("English");
        analyzer
            .push_str_for_variety("hola", VarietyId("es-ES-Castilian".into()))
            .expect("Spanish switch");
        let update = analyzer.finish().expect("flush");

        for occurrence in &update.occurrences {
            assert_eq!(
                occurrence.surface,
                slice_chars(analyzer.text(), occurrence.span)
            );
        }
        assert!(
            update
                .occurrences
                .iter()
                .any(|occurrence| { occurrence.morpheme == Spec::Known(MorphemeId("un-".into())) })
        );
        assert!(
            update
                .occurrences
                .iter()
                .any(|occurrence| { occurrence.morpheme == Spec::Known(MorphemeId("-'s".into())) })
        );
        assert!(
            update
                .occurrences
                .iter()
                .any(|occurrence| occurrence.surface == "well")
        );
        assert!(
            update
                .occurrences
                .iter()
                .any(|occurrence| occurrence.surface == "known")
        );
        let spanish = update
            .occurrences
            .iter()
            .find(|occurrence| occurrence.surface == "hola")
            .expect("whole-word Spanish fallback");
        assert_eq!(spanish.morpheme, Spec::Unknown);
        assert_eq!(
            spanish.provenance.method,
            "language-neutral-whole-word-fallback"
        );
        assert_eq!(spanish.finality, AnalysisFinality::Final);
    }

    #[test]
    fn stress_changing_morphology_uses_variety_pronunciation_data() {
        let mut analyzer = analyzer();
        let update = analyzer.push_str("activity ").expect("activity");
        let active = update
            .occurrences
            .iter()
            .find(|occurrence| occurrence.morpheme == Spec::Known(MorphemeId("active".into())))
            .expect("active root");
        assert!(
            active.pronunciation.iter().any(|phoneme| {
                let is_ih = phoneme.features.values.iter().any(|(id, value)| {
                    id.0 == "phonology.base_symbol"
                        && matches!(
                            value,
                            Spec::Known(crate::feature::FeatureValue::Category(symbol))
                                if symbol == "IH"
                        )
                });
                let is_primary = phoneme.features.values.iter().any(|(id, value)| {
                    id.0 == "phonology.stress"
                        && matches!(
                            value,
                                Spec::Known(crate::feature::FeatureValue::Category(stress))
                                if stress == "primary"
                        )
                });
                is_ih && is_primary
            }),
            "-ity should move primary stress to the final vowel of the root"
        );
    }

    #[test]
    fn analyzer_deltas_replay_through_the_duplex_journal() {
        let mut analyzer = analyzer();
        analyzer.push_str("happy").expect("root");
        analyzer.push_str("ness ").expect("suffix and finality");

        let mut duplex = BeliefEventJournal {
            version: DUPLEX_JOURNAL_VERSION,
            utterance_id: UtteranceId("fixture".into()),
            variety: VarietyId("en-US-GA".into()),
            events: Vec::new(),
        };
        let deltas = analyzer
            .journal()
            .events
            .iter()
            .map(|event| event.delta.clone())
            .collect::<Vec<_>>();
        append_to_duplex_journal(&mut duplex, &deltas);
        let first = replay_journal(&duplex).expect("first replay");
        let second = replay_journal(&duplex).expect("second replay");
        assert_eq!(first, second);
        assert!(
            first
                .evidence
                .values()
                .filter(|record| !record.withdrawn)
                .all(|record| record.payload.finality == Some(EvidenceFinality::Final))
        );
    }

    #[test]
    fn merge_delta_is_emitted_when_a_decomposition_becomes_unknown() {
        let mut analyzer = analyzer();
        let first = analyzer.push_str("replay").expect("prefix plus root");
        assert!(first.occurrences.len() >= 2);
        let revised = analyzer.push_str("x").expect("unknown extension");
        let merge = revised
            .deltas
            .iter()
            .find_map(|delta| match delta {
                MorphemeAnalysisDelta::Merge {
                    withdrawn,
                    replacement,
                } => Some((withdrawn, replacement)),
                _ => None,
            })
            .expect("merge delta");
        assert!(merge.0.len() >= 2);
        assert_eq!(merge.1.surface, slice_chars(analyzer.text(), merge.1.span));
        assert_eq!(merge.1.morpheme, Spec::Unknown);
    }
}
