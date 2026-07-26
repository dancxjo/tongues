use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::evidence::EvidenceProvenance;
use crate::ids::{
    CompletionHypothesisId, EmissionId, MorphemeOccurrenceId, UtteranceId, VarietyId,
};
use crate::morphology::MorphemeToken;
use crate::phonology::{PhoneToken, PhonemeToken};
use crate::prosody::ProsodyTrack;
use crate::syntax::SentenceSyntaxAnalysis;
use crate::time::{TextSpan, TimeSpan};

pub const DUPLEX_JOURNAL_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceState {
    ObservedText,
    ObservedAcoustics,
    LinguisticInference,
    PredictedCompletion,
    CommittedMaterial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceFinality {
    Provisional,
    Final,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "anchor", content = "id")]
pub enum EvidenceAnchor {
    MorphemeOccurrence(MorphemeOccurrenceId),
    Emission(EmissionId),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EvidencePayload {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_span: Option<TextSpan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_span: Option<TimeSpan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub syntax: Option<SentenceSyntaxAnalysis>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub morphology: Vec<MorphemeToken>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pronunciation: Vec<PhonemeToken>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub phones: Vec<PhoneToken>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prosody: Option<ProsodyTrack>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finality: Option<EvidenceFinality>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceDelta {
    pub anchor: EvidenceAnchor,
    pub state: EvidenceState,
    pub confidence: f32,
    pub provenance: EvidenceProvenance,
    #[serde(default)]
    pub payload: EvidencePayload,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionHypothesis {
    pub id: CompletionHypothesisId,
    pub anchor: MorphemeOccurrenceId,
    pub confidence: f32,
    pub provenance: EvidenceProvenance,
    #[serde(default)]
    pub payload: EvidencePayload,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CommitFrontier {
    #[serde(default)]
    pub morpheme_occurrences: Vec<MorphemeOccurrenceId>,
    #[serde(default)]
    pub emissions: Vec<EmissionId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "data")]
pub enum CommitFrontierUpdate {
    AdvanceMorphemeOccurrence(MorphemeOccurrenceId),
    AdvanceEmission(EmissionId),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "target", content = "id")]
pub enum WithdrawalTarget {
    MorphemeOccurrence(MorphemeOccurrenceId),
    Emission(EmissionId),
    CompletionHypothesis(CompletionHypothesisId),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Withdrawal {
    pub target: WithdrawalTarget,
    pub reason: String,
    pub provenance: EvidenceProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "target", content = "id")]
pub enum RepairTarget {
    MorphemeOccurrence(MorphemeOccurrenceId),
    Emission(EmissionId),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Repair {
    pub target: RepairTarget,
    pub replacement: EvidenceDelta,
    pub reason: String,
    pub provenance: EvidenceProvenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryState {
    Planned,
    Synthesized,
    Verified,
    Queued,
    Played,
}

impl DeliveryState {
    fn phase(self) -> u8 {
        match self {
            Self::Planned => 0,
            Self::Synthesized => 1,
            Self::Verified => 2,
            Self::Queued => 3,
            Self::Played => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeliveryStateUpdate {
    pub emission_id: EmissionId,
    pub state: DeliveryState,
    pub confidence: f32,
    pub provenance: EvidenceProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeliveryRecord {
    pub state: DeliveryState,
    pub confidence: f32,
    pub provenance: EvidenceProvenance,
    #[serde(default)]
    pub withdrawn: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    pub state: EvidenceState,
    pub confidence: f32,
    pub provenance: EvidenceProvenance,
    #[serde(default)]
    pub payload: EvidencePayload,
    #[serde(default)]
    pub withdrawn: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommittedHistoryEntry {
    pub anchor: MorphemeOccurrenceId,
    pub confidence: f32,
    pub provenance: EvidenceProvenance,
    #[serde(default)]
    pub payload: EvidencePayload,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction_of: Option<MorphemeOccurrenceId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "data")]
pub enum BeliefAction {
    ApplyEvidenceDelta(EvidenceDelta),
    AddCompletionHypothesis(CompletionHypothesis),
    CommitCompletionHypothesis {
        hypothesis_id: CompletionHypothesisId,
        committed_occurrence: MorphemeOccurrenceId,
        confidence: f32,
        provenance: EvidenceProvenance,
    },
    UpdateCommitFrontier(CommitFrontierUpdate),
    Withdraw(Withdrawal),
    Repair(Repair),
    UpdateDeliveryState(DeliveryStateUpdate),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeliefEvent {
    pub sequence: u64,
    pub action: BeliefAction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeliefEventJournal {
    pub version: u32,
    pub utterance_id: UtteranceId,
    pub variety: VarietyId,
    #[serde(default)]
    pub events: Vec<BeliefEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UtteranceBeliefState {
    pub utterance_id: UtteranceId,
    pub variety: VarietyId,
    pub revision: u64,
    #[serde(default)]
    pub evidence: BTreeMap<EvidenceAnchor, EvidenceRecord>,
    #[serde(default)]
    pub completion_hypotheses: BTreeMap<CompletionHypothesisId, CompletionHypothesis>,
    #[serde(default)]
    pub commit_frontier: CommitFrontier,
    #[serde(default)]
    pub deliveries: BTreeMap<EmissionId, DeliveryRecord>,
    #[serde(default)]
    pub committed_history: Vec<CommittedHistoryEntry>,
    #[serde(default)]
    pub withdrawals: Vec<Withdrawal>,
    #[serde(default)]
    pub repairs: Vec<Repair>,
}

impl UtteranceBeliefState {
    pub fn new(utterance_id: UtteranceId, variety: VarietyId) -> Self {
        Self {
            utterance_id,
            variety,
            revision: 0,
            evidence: BTreeMap::new(),
            completion_hypotheses: BTreeMap::new(),
            commit_frontier: CommitFrontier::default(),
            deliveries: BTreeMap::new(),
            committed_history: Vec::new(),
            withdrawals: Vec::new(),
            repairs: Vec::new(),
        }
    }

    pub fn apply_action(&mut self, action: &BeliefAction) -> Result<(), BeliefStateError> {
        match action {
            BeliefAction::ApplyEvidenceDelta(delta) => self.apply_evidence_delta(delta, false),
            BeliefAction::AddCompletionHypothesis(hypothesis) => {
                validate_non_empty_id(&hypothesis.id.0, "completion_hypothesis")?;
                validate_non_empty_id(&hypothesis.anchor.0, "morpheme_occurrence")?;
                self.completion_hypotheses
                    .insert(hypothesis.id.clone(), hypothesis.clone());
                Ok(())
            }
            BeliefAction::CommitCompletionHypothesis {
                hypothesis_id,
                committed_occurrence,
                confidence,
                provenance,
            } => self.commit_completion_hypothesis(
                hypothesis_id,
                committed_occurrence,
                *confidence,
                provenance.clone(),
            ),
            BeliefAction::UpdateCommitFrontier(update) => self.update_commit_frontier(update),
            BeliefAction::Withdraw(withdrawal) => self.withdraw(withdrawal),
            BeliefAction::Repair(repair) => self.repair(repair),
            BeliefAction::UpdateDeliveryState(update) => self.update_delivery_state(update),
        }?;

        self.revision = self.revision.saturating_add(1);
        Ok(())
    }

    fn commit_completion_hypothesis(
        &mut self,
        hypothesis_id: &CompletionHypothesisId,
        committed_occurrence: &MorphemeOccurrenceId,
        confidence: f32,
        provenance: EvidenceProvenance,
    ) -> Result<(), BeliefStateError> {
        validate_non_empty_id(&hypothesis_id.0, "completion_hypothesis")?;
        validate_non_empty_id(&committed_occurrence.0, "morpheme_occurrence")?;

        let hypothesis = self
            .completion_hypotheses
            .get(hypothesis_id)
            .cloned()
            .ok_or_else(|| BeliefStateError::UnknownCompletionHypothesis {
                id: hypothesis_id.clone(),
            })?;

        let committed_anchor = EvidenceAnchor::MorphemeOccurrence(committed_occurrence.clone());
        let delta = EvidenceDelta {
            anchor: committed_anchor,
            state: EvidenceState::CommittedMaterial,
            confidence,
            provenance: provenance.clone(),
            payload: hypothesis.payload.clone(),
        };
        self.apply_evidence_delta(&delta, true)?;
        self.completion_hypotheses.remove(hypothesis_id);
        Ok(())
    }

    fn apply_evidence_delta(
        &mut self,
        delta: &EvidenceDelta,
        explicit_promotion: bool,
    ) -> Result<(), BeliefStateError> {
        match &delta.anchor {
            EvidenceAnchor::MorphemeOccurrence(id) => {
                validate_non_empty_id(&id.0, "morpheme_occurrence")?
            }
            EvidenceAnchor::Emission(id) => validate_non_empty_id(&id.0, "emission")?,
        }

        if let Some(existing) = self.evidence.get(&delta.anchor) {
            if existing.state == EvidenceState::CommittedMaterial
                && delta.state != EvidenceState::CommittedMaterial
            {
                return Err(BeliefStateError::CommittedHistoryRequiresRepair {
                    anchor: delta.anchor.clone(),
                });
            }

            if existing.state == EvidenceState::PredictedCompletion
                && delta.state != EvidenceState::PredictedCompletion
                && !explicit_promotion
            {
                return Err(BeliefStateError::PredictedMaterialNeedsExplicitAction {
                    anchor: delta.anchor.clone(),
                });
            }
        }

        self.evidence.insert(
            delta.anchor.clone(),
            EvidenceRecord {
                state: delta.state,
                confidence: delta.confidence,
                provenance: delta.provenance.clone(),
                payload: delta.payload.clone(),
                withdrawn: false,
            },
        );

        if matches!(
            (&delta.anchor, delta.state),
            (
                EvidenceAnchor::MorphemeOccurrence(_),
                EvidenceState::CommittedMaterial
            )
        ) {
            let EvidenceAnchor::MorphemeOccurrence(anchor) = &delta.anchor else {
                unreachable!()
            };
            self.committed_history.push(CommittedHistoryEntry {
                anchor: anchor.clone(),
                confidence: delta.confidence,
                provenance: delta.provenance.clone(),
                payload: delta.payload.clone(),
                correction_of: None,
            });
        }

        Ok(())
    }

    fn update_commit_frontier(
        &mut self,
        update: &CommitFrontierUpdate,
    ) -> Result<(), BeliefStateError> {
        match update {
            CommitFrontierUpdate::AdvanceMorphemeOccurrence(id) => {
                validate_non_empty_id(&id.0, "morpheme_occurrence")?;
                if self.commit_frontier.morpheme_occurrences.contains(id) {
                    return Err(BeliefStateError::CommitFrontierRegression {
                        kind: "morpheme_occurrence".into(),
                        id: id.0.clone(),
                    });
                }
                self.commit_frontier.morpheme_occurrences.push(id.clone());
            }
            CommitFrontierUpdate::AdvanceEmission(id) => {
                validate_non_empty_id(&id.0, "emission")?;
                if self.commit_frontier.emissions.contains(id) {
                    return Err(BeliefStateError::CommitFrontierRegression {
                        kind: "emission".into(),
                        id: id.0.clone(),
                    });
                }
                self.commit_frontier.emissions.push(id.clone());
            }
        }
        Ok(())
    }

    fn withdraw(&mut self, withdrawal: &Withdrawal) -> Result<(), BeliefStateError> {
        match &withdrawal.target {
            WithdrawalTarget::MorphemeOccurrence(id) => {
                validate_non_empty_id(&id.0, "morpheme_occurrence")?;
                let anchor = EvidenceAnchor::MorphemeOccurrence(id.clone());
                let record = self.evidence.get_mut(&anchor).ok_or_else(|| {
                    BeliefStateError::UnknownMorphemeOccurrenceAnchor { id: id.clone() }
                })?;
                if record.state == EvidenceState::CommittedMaterial {
                    return Err(BeliefStateError::CommittedHistoryRequiresRepair { anchor });
                }
                record.withdrawn = true;
            }
            WithdrawalTarget::Emission(id) => {
                validate_non_empty_id(&id.0, "emission")?;
                if let Some(delivery) = self.deliveries.get(id)
                    && delivery.state == DeliveryState::Played
                {
                    return Err(BeliefStateError::PlayedEmissionCannotBeWithdrawn {
                        emission_id: id.clone(),
                    });
                }
                let anchor = EvidenceAnchor::Emission(id.clone());
                let record = self
                    .evidence
                    .get_mut(&anchor)
                    .ok_or_else(|| BeliefStateError::UnknownEmissionAnchor { id: id.clone() })?;
                record.withdrawn = true;
            }
            WithdrawalTarget::CompletionHypothesis(id) => {
                validate_non_empty_id(&id.0, "completion_hypothesis")?;
                if self.completion_hypotheses.remove(id).is_none() {
                    return Err(BeliefStateError::UnknownCompletionHypothesis { id: id.clone() });
                }
            }
        }
        self.withdrawals.push(withdrawal.clone());
        Ok(())
    }

    fn repair(&mut self, repair: &Repair) -> Result<(), BeliefStateError> {
        match &repair.target {
            RepairTarget::MorphemeOccurrence(id) => {
                validate_non_empty_id(&id.0, "morpheme_occurrence")?;
                if repair.replacement.anchor != EvidenceAnchor::MorphemeOccurrence(id.clone()) {
                    return Err(BeliefStateError::RepairAnchorMismatch {
                        target: repair.target.clone(),
                        replacement_anchor: repair.replacement.anchor.clone(),
                    });
                }
                let anchor = EvidenceAnchor::MorphemeOccurrence(id.clone());
                let old = self.evidence.get(&anchor).ok_or_else(|| {
                    BeliefStateError::UnknownMorphemeOccurrenceAnchor { id: id.clone() }
                })?;
                let corrected_committed = old.state == EvidenceState::CommittedMaterial
                    || repair.replacement.state == EvidenceState::CommittedMaterial;
                self.apply_evidence_delta(&repair.replacement, true)?;
                if corrected_committed {
                    self.committed_history.push(CommittedHistoryEntry {
                        anchor: id.clone(),
                        confidence: repair.replacement.confidence,
                        provenance: repair.provenance.clone(),
                        payload: repair.replacement.payload.clone(),
                        correction_of: Some(id.clone()),
                    });
                }
            }
            RepairTarget::Emission(id) => {
                validate_non_empty_id(&id.0, "emission")?;
                if repair.replacement.anchor != EvidenceAnchor::Emission(id.clone()) {
                    return Err(BeliefStateError::RepairAnchorMismatch {
                        target: repair.target.clone(),
                        replacement_anchor: repair.replacement.anchor.clone(),
                    });
                }
                let anchor = EvidenceAnchor::Emission(id.clone());
                if !self.evidence.contains_key(&anchor) {
                    return Err(BeliefStateError::UnknownEmissionAnchor { id: id.clone() });
                }
                self.apply_evidence_delta(&repair.replacement, true)?;
            }
        }
        self.repairs.push(repair.clone());
        Ok(())
    }

    fn update_delivery_state(
        &mut self,
        update: &DeliveryStateUpdate,
    ) -> Result<(), BeliefStateError> {
        validate_non_empty_id(&update.emission_id.0, "emission")?;

        if let Some(existing) = self.deliveries.get(&update.emission_id)
            && update.state.phase() < existing.state.phase()
        {
            return Err(BeliefStateError::DeliveryStateRegression {
                emission_id: update.emission_id.clone(),
                from: existing.state,
                to: update.state,
            });
        }

        self.deliveries.insert(
            update.emission_id.clone(),
            DeliveryRecord {
                state: update.state,
                confidence: update.confidence,
                provenance: update.provenance.clone(),
                withdrawn: false,
            },
        );
        Ok(())
    }
}

pub fn replay_journal(
    journal: &BeliefEventJournal,
) -> Result<UtteranceBeliefState, BeliefStateError> {
    if journal.version != DUPLEX_JOURNAL_VERSION {
        return Err(BeliefStateError::UnsupportedJournalVersion {
            expected: DUPLEX_JOURNAL_VERSION,
            found: journal.version,
        });
    }

    let mut state =
        UtteranceBeliefState::new(journal.utterance_id.clone(), journal.variety.clone());
    for (expected, event) in journal.events.iter().enumerate() {
        let expected = expected as u64;
        if event.sequence != expected {
            return Err(BeliefStateError::OutOfOrderEvent {
                expected,
                found: event.sequence,
            });
        }
        state.apply_action(&event.action)?;
    }
    Ok(state)
}

#[derive(Debug, Clone, PartialEq)]
pub enum BeliefStateError {
    UnsupportedJournalVersion {
        expected: u32,
        found: u32,
    },
    OutOfOrderEvent {
        expected: u64,
        found: u64,
    },
    EmptyAnchorId {
        kind: &'static str,
    },
    PredictedMaterialNeedsExplicitAction {
        anchor: EvidenceAnchor,
    },
    CommittedHistoryRequiresRepair {
        anchor: EvidenceAnchor,
    },
    CommitFrontierRegression {
        kind: String,
        id: String,
    },
    UnknownMorphemeOccurrenceAnchor {
        id: MorphemeOccurrenceId,
    },
    UnknownEmissionAnchor {
        id: EmissionId,
    },
    UnknownCompletionHypothesis {
        id: CompletionHypothesisId,
    },
    PlayedEmissionCannotBeWithdrawn {
        emission_id: EmissionId,
    },
    RepairAnchorMismatch {
        target: RepairTarget,
        replacement_anchor: EvidenceAnchor,
    },
    DeliveryStateRegression {
        emission_id: EmissionId,
        from: DeliveryState,
        to: DeliveryState,
    },
}

impl std::fmt::Display for BeliefStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedJournalVersion { expected, found } => {
                write!(
                    f,
                    "unsupported journal version {found}; expected {expected}"
                )
            }
            Self::OutOfOrderEvent { expected, found } => {
                write!(
                    f,
                    "event sequence {found} is out of order; expected {expected}"
                )
            }
            Self::EmptyAnchorId { kind } => write!(f, "{kind} id cannot be empty"),
            Self::PredictedMaterialNeedsExplicitAction { anchor } => {
                write!(
                    f,
                    "predicted material at {anchor:?} needs an explicit commit/repair action"
                )
            }
            Self::CommittedHistoryRequiresRepair { anchor } => {
                write!(
                    f,
                    "committed material at {anchor:?} is append-only; use a repair event"
                )
            }
            Self::CommitFrontierRegression { kind, id } => {
                write!(f, "commit frontier cannot move backward for {kind} '{id}'")
            }
            Self::UnknownMorphemeOccurrenceAnchor { id } => {
                write!(f, "unknown morpheme occurrence anchor '{}'", id.0)
            }
            Self::UnknownEmissionAnchor { id } => write!(f, "unknown emission anchor '{}'", id.0),
            Self::UnknownCompletionHypothesis { id } => {
                write!(f, "unknown completion hypothesis '{}'", id.0)
            }
            Self::PlayedEmissionCannotBeWithdrawn { emission_id } => {
                write!(f, "played emission '{}' cannot be withdrawn", emission_id.0)
            }
            Self::RepairAnchorMismatch {
                target,
                replacement_anchor,
            } => write!(
                f,
                "repair target {target:?} does not match replacement anchor {replacement_anchor:?}"
            ),
            Self::DeliveryStateRegression {
                emission_id,
                from,
                to,
            } => write!(
                f,
                "delivery state for emission '{}' cannot regress from {from:?} to {to:?}",
                emission_id.0
            ),
        }
    }
}

impl std::error::Error for BeliefStateError {}

fn validate_non_empty_id(id: &str, kind: &'static str) -> Result<(), BeliefStateError> {
    if id.trim().is_empty() {
        return Err(BeliefStateError::EmptyAnchorId { kind });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::EvidenceSource;

    fn provenance(method: &str) -> EvidenceProvenance {
        EvidenceProvenance {
            source: EvidenceSource::Inference,
            method: method.into(),
            version: Some("v1".into()),
        }
    }

    fn payload(text: &str) -> EvidencePayload {
        EvidencePayload {
            text: Some(text.into()),
            text_span: Some(TextSpan {
                start_char: 0,
                end_char: text.chars().count(),
            }),
            ..Default::default()
        }
    }

    #[test]
    fn fixture_journal_round_trips() {
        let fixture = include_str!("../../../fixtures/speaking/duplex_journal_v1.json");
        let journal: BeliefEventJournal = serde_json::from_str(fixture).expect("fixture parses");
        assert_eq!(journal.version, DUPLEX_JOURNAL_VERSION);

        let json = serde_json::to_string_pretty(&journal).expect("journal serializes");
        let reparsed: BeliefEventJournal = serde_json::from_str(&json).expect("journal reparses");
        assert_eq!(reparsed, journal);
    }

    #[test]
    fn replay_is_deterministic() {
        let fixture = include_str!("../../../fixtures/speaking/duplex_journal_v1.json");
        let journal: BeliefEventJournal = serde_json::from_str(fixture).expect("fixture parses");

        let first = replay_journal(&journal).expect("first replay succeeds");
        let second = replay_journal(&journal).expect("second replay succeeds");
        assert_eq!(first, second);
        assert!(
            first
                .evidence
                .values()
                .any(|record| record.state == EvidenceState::CommittedMaterial)
        );
    }

    #[test]
    fn predicted_material_needs_explicit_action_to_become_observed() {
        let mut state =
            UtteranceBeliefState::new(UtteranceId("utt-1".into()), VarietyId("en-US-GA".into()));
        let anchor = EvidenceAnchor::MorphemeOccurrence(MorphemeOccurrenceId("utt-1:m1".into()));

        state
            .apply_action(&BeliefAction::ApplyEvidenceDelta(EvidenceDelta {
                anchor: anchor.clone(),
                state: EvidenceState::PredictedCompletion,
                confidence: 0.6,
                provenance: provenance("predict"),
                payload: payload("helo"),
            }))
            .expect("initial prediction accepted");

        let error = state
            .apply_action(&BeliefAction::ApplyEvidenceDelta(EvidenceDelta {
                anchor,
                state: EvidenceState::ObservedText,
                confidence: 0.9,
                provenance: provenance("asr"),
                payload: payload("hello"),
            }))
            .expect_err("implicit promotion should fail");

        assert!(matches!(
            error,
            BeliefStateError::PredictedMaterialNeedsExplicitAction { .. }
        ));
    }

    #[test]
    fn committed_history_is_append_only_without_repair() {
        let mut state =
            UtteranceBeliefState::new(UtteranceId("utt-1".into()), VarietyId("en-US-GA".into()));
        let anchor = EvidenceAnchor::MorphemeOccurrence(MorphemeOccurrenceId("utt-1:m1".into()));

        state
            .apply_action(&BeliefAction::ApplyEvidenceDelta(EvidenceDelta {
                anchor: anchor.clone(),
                state: EvidenceState::CommittedMaterial,
                confidence: 0.8,
                provenance: provenance("commit"),
                payload: payload("hello"),
            }))
            .expect("initial commit accepted");

        let error = state
            .apply_action(&BeliefAction::ApplyEvidenceDelta(EvidenceDelta {
                anchor,
                state: EvidenceState::ObservedText,
                confidence: 0.95,
                provenance: provenance("manual"),
                payload: payload("hullo"),
            }))
            .expect_err("committed mutation should fail");

        assert!(matches!(
            error,
            BeliefStateError::CommittedHistoryRequiresRepair { .. }
        ));
    }

    #[test]
    fn played_audio_cannot_be_withdrawn() {
        let mut state =
            UtteranceBeliefState::new(UtteranceId("utt-2".into()), VarietyId("en-US-GA".into()));
        let emission_id = EmissionId("utt-2:e1".into());
        let emission_anchor = EvidenceAnchor::Emission(emission_id.clone());

        state
            .apply_action(&BeliefAction::ApplyEvidenceDelta(EvidenceDelta {
                anchor: emission_anchor,
                state: EvidenceState::ObservedAcoustics,
                confidence: 0.8,
                provenance: provenance("tts"),
                payload: EvidencePayload {
                    audio_span: Some(TimeSpan {
                        start_s: 0.0,
                        end_s: 0.4,
                    }),
                    ..Default::default()
                },
            }))
            .expect("acoustic evidence accepted");

        state
            .apply_action(&BeliefAction::UpdateDeliveryState(DeliveryStateUpdate {
                emission_id: emission_id.clone(),
                state: DeliveryState::Played,
                confidence: 1.0,
                provenance: provenance("playback"),
            }))
            .expect("played state accepted");

        let error = state
            .apply_action(&BeliefAction::Withdraw(Withdrawal {
                target: WithdrawalTarget::Emission(emission_id),
                reason: "user interruption".into(),
                provenance: provenance("ui"),
            }))
            .expect_err("played audio withdrawal must fail");

        assert!(matches!(
            error,
            BeliefStateError::PlayedEmissionCannotBeWithdrawn { .. }
        ));
    }

    #[test]
    fn repair_requires_stable_existing_anchor() {
        let mut state =
            UtteranceBeliefState::new(UtteranceId("utt-3".into()), VarietyId("en-US-GA".into()));

        let error = state
            .apply_action(&BeliefAction::Repair(Repair {
                target: RepairTarget::MorphemeOccurrence(MorphemeOccurrenceId("utt-3:m9".into())),
                replacement: EvidenceDelta {
                    anchor: EvidenceAnchor::MorphemeOccurrence(MorphemeOccurrenceId(
                        "utt-3:m9".into(),
                    )),
                    state: EvidenceState::ObservedText,
                    confidence: 0.9,
                    provenance: provenance("manual"),
                    payload: payload("replacement"),
                },
                reason: "correction".into(),
                provenance: provenance("manual"),
            }))
            .expect_err("repair target must exist");

        assert!(matches!(
            error,
            BeliefStateError::UnknownMorphemeOccurrenceAnchor { .. }
        ));
    }
}
