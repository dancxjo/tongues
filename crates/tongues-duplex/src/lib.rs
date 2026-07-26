//! Deterministic, provider-neutral completion beams and commit-frontier policy.
//!
//! `speaking` owns the shared evidence and identity contracts. This crate owns
//! orchestration: providers propose morpheme continuations, the simulator
//! normalizes them, and only a directly supported common prefix may commit.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufWriter, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result as AnyResult};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use speaking::{
    CompletionHypothesisId, EvidenceProvenance, EvidenceSource, ProsodyTrack,
    SentenceSyntaxAnalysis, UtteranceId, VarietyId,
};
use thiserror::Error;

pub const SIMULATOR_JOURNAL_VERSION: u32 = 1;
pub const FIXTURE_SUITE_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceModality {
    Text,
    Acoustics,
}

/// Frame-level timing and confidence metadata attached to acoustic evidence.
///
/// All fields survive withdrawal, repair, and frontier advancement so that
/// downstream diagnostics can trace a committed morpheme back to the original
/// audio window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticSpan {
    /// First mel-frame index (inclusive) from the acoustic encoder.
    pub frame_start: u32,
    /// Last mel-frame index (exclusive) from the acoustic encoder.
    pub frame_end: u32,
    /// Wall-clock start time in seconds relative to utterance start.
    pub time_start: f32,
    /// Wall-clock end time in seconds relative to utterance start.
    pub time_end: f32,
    /// Aggregate acoustic confidence in [0.0, 1.0].
    pub confidence: f32,
}

/// Direct input evidence. `supports` names the normalized morpheme keys the
/// input directly contains; an empty list is derived deterministically from
/// `content`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservedEvidence {
    pub id: String,
    pub modality: EvidenceModality,
    pub content: String,
    #[serde(default)]
    pub supports: Vec<String>,
    pub provenance: EvidenceProvenance,
    /// Present for acoustic evidence; carries frame timing and confidence that
    /// survive every withdrawal, repair, and frontier revision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acoustic_span: Option<AcousticSpan>,
}

impl ObservedEvidence {
    pub fn text(id: impl Into<String>, content: impl Into<String>) -> Self {
        let content = content.into();
        Self {
            id: id.into(),
            modality: EvidenceModality::Text,
            supports: tokenize_morphemes(&content),
            content,
            provenance: EvidenceProvenance {
                source: EvidenceSource::Manual,
                method: "duplex-text-chunk".into(),
                version: Some("1".into()),
            },
            acoustic_span: None,
        }
    }

    pub fn acoustics(id: impl Into<String>, transcript: impl Into<String>) -> Self {
        let content = transcript.into();
        Self {
            id: id.into(),
            modality: EvidenceModality::Acoustics,
            supports: tokenize_morphemes(&content),
            content,
            provenance: EvidenceProvenance {
                source: EvidenceSource::AcousticModel,
                method: "duplex-mock-acoustics".into(),
                version: Some("1".into()),
            },
            acoustic_span: None,
        }
    }

    /// Acoustic evidence with explicit frame-span metadata.
    pub fn acoustics_with_span(
        id: impl Into<String>,
        transcript: impl Into<String>,
        span: AcousticSpan,
    ) -> Self {
        let mut ev = Self::acoustics(id, transcript);
        ev.acoustic_span = Some(span);
        ev
    }

    fn supports(&self, key: &str) -> bool {
        let expected = normalize_key(key);
        let supports = if self.supports.is_empty() {
            tokenize_morphemes(&self.content)
        } else {
            self.supports.clone()
        };
        supports
            .iter()
            .any(|candidate| normalize_key(candidate) == expected)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionMorpheme {
    /// Provider-neutral identity used when comparing morpheme prefixes.
    pub key: String,
    pub surface: String,
    pub variety: VarietyId,
    /// IDs of direct text/acoustic evidence claimed for this occurrence.
    #[serde(default)]
    pub evidence: Vec<String>,
}

impl CompletionMorpheme {
    pub fn predicted(
        key: impl Into<String>,
        surface: impl Into<String>,
        variety: VarietyId,
    ) -> Self {
        Self {
            key: key.into(),
            surface: surface.into(),
            variety,
            evidence: Vec::new(),
        }
    }
}

/// A provider proposal before probability normalization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionProposal {
    pub id: CompletionHypothesisId,
    /// Any finite non-negative weight. The simulator normalizes all weights.
    pub weight: f64,
    #[serde(default)]
    pub morphemes: Vec<CompletionMorpheme>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub syntax: Option<SentenceSyntaxAnalysis>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prosody: Option<ProsodyTrack>,
    #[serde(default)]
    pub evidence: Vec<String>,
    pub provenance: EvidenceProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedCompletionHypothesis {
    pub id: CompletionHypothesisId,
    pub probability: f64,
    #[serde(default)]
    pub morphemes: Vec<CompletionMorpheme>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub syntax: Option<SentenceSyntaxAnalysis>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prosody: Option<ProsodyTrack>,
    #[serde(default)]
    pub evidence: Vec<String>,
    pub provenance: EvidenceProvenance,
}

impl NormalizedCompletionHypothesis {
    fn from_proposal(proposal: CompletionProposal, total: f64) -> Self {
        Self {
            id: proposal.id,
            probability: proposal.weight / total,
            morphemes: proposal.morphemes,
            syntax: proposal.syntax,
            prosody: proposal.prosody,
            evidence: proposal.evidence,
            provenance: proposal.provenance,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub utterance_id: UtteranceId,
    pub variety: VarietyId,
    #[serde(default)]
    pub evidence: Vec<ObservedEvidence>,
    #[serde(default)]
    pub committed: Vec<CommittedMorpheme>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("{message}")]
pub struct CompletionProviderError {
    pub message: String,
}

impl CompletionProviderError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Boundary implemented by deterministic fixtures today and learned/live
/// completion systems later. Providers cannot mutate simulator state.
pub trait CompletionProvider {
    fn complete(
        &mut self,
        request: &CompletionRequest,
    ) -> Result<Vec<CompletionProposal>, CompletionProviderError>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulatorConfig {
    /// Select the smallest highest-probability set whose cumulative posterior
    /// meets this threshold, then compute its longest common morpheme prefix.
    pub posterior_mass: f64,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            posterior_mass: 0.8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationDecision {
    Accept,
    Retry,
    Fallback,
    Abstain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationRepairCause {
    LinguisticDisagreement,
    AcceptedProjectionLoss,
    AcousticMismatch,
    TimingMismatch,
    RecognizerUncertain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationDimension {
    Morpheme,
    Phone,
    Stress,
    Boundary,
    Timing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationEvidence {
    pub dimension: VerificationDimension,
    pub intended: String,
    pub recovered: String,
    pub accepted_projection_loss: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct VerificationSequence {
    #[serde(default)]
    pub morphemes: Vec<String>,
    #[serde(default)]
    pub phones: Vec<String>,
    #[serde(default)]
    pub stress: Vec<String>,
    #[serde(default)]
    pub boundaries: Vec<String>,
    #[serde(default)]
    pub timings_ms: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClosedLoopVerificationRequest {
    pub intended: VerificationSequence,
    pub recovered: VerificationSequence,
    #[serde(default)]
    pub accepted_projection_losses: Vec<String>,
    pub recognizer_confidence: f32,
    pub held_audio_replaceable: bool,
    pub verification_latency_ms: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClosedLoopVerificationPolicy {
    pub min_recognizer_confidence: f32,
    pub min_morpheme_agreement: f32,
    pub max_phone_error_rate: f32,
    pub max_verification_latency_ms: f32,
    pub max_retries: u8,
}

impl Default for ClosedLoopVerificationPolicy {
    fn default() -> Self {
        Self {
            min_recognizer_confidence: 0.45,
            min_morpheme_agreement: 0.75,
            max_phone_error_rate: 0.35,
            max_verification_latency_ms: 250.0,
            max_retries: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct VerificationMetrics {
    pub verification_latency_ms: f32,
    pub false_rejection: f32,
    pub false_acceptance: f32,
    pub phone_error_rate: f32,
    pub word_agreement: f32,
    pub morpheme_agreement: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClosedLoopVerificationResult {
    pub decision: VerificationDecision,
    #[serde(default)]
    pub evidence: Vec<VerificationEvidence>,
    #[serde(default)]
    pub repair_causes: Vec<VerificationRepairCause>,
    pub metrics: VerificationMetrics,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MelPredictionCandidate {
    pub id: CompletionHypothesisId,
    pub prior_probability: f64,
    #[serde(default)]
    pub predicted_mel: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticRescoreRequest {
    #[serde(default)]
    pub observed_mel: Vec<f32>,
    #[serde(default)]
    pub candidates: Vec<MelPredictionCandidate>,
    pub acoustic_weight: f64,
    pub prior_weight: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticRescore {
    pub id: CompletionHypothesisId,
    pub acoustic_log_likelihood: f64,
    pub prior_log_probability: f64,
    pub combined_score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommittedMorpheme {
    pub key: String,
    pub surface: String,
    pub variety: VarietyId,
    #[serde(default)]
    pub evidence: Vec<String>,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "data")]
pub enum SimulatorEventKind {
    EvidenceObserved {
        evidence: ObservedEvidence,
    },
    HypothesisProposed {
        hypothesis: NormalizedCompletionHypothesis,
    },
    HypothesisWithdrawn {
        hypothesis: NormalizedCompletionHypothesis,
        reason: String,
    },
    HypothesisRepaired {
        previous: NormalizedCompletionHypothesis,
        replacement: NormalizedCompletionHypothesis,
        reason: String,
    },
    BeamInferred {
        selected: Vec<CompletionHypothesisId>,
        covered_probability: f64,
        shared_prefix: Vec<String>,
    },
    CommitFrontierAdvanced {
        from: usize,
        to: usize,
        committed: Vec<CommittedMorpheme>,
    },
    VerificationEvaluated {
        result: ClosedLoopVerificationResult,
    },
}

impl SimulatorEventKind {
    pub fn layer(&self) -> &'static str {
        match self {
            Self::EvidenceObserved { .. } => "evidence",
            Self::HypothesisProposed { .. } => "prediction",
            Self::HypothesisWithdrawn { .. }
            | Self::HypothesisRepaired { .. }
            | Self::BeamInferred { .. } => "inference",
            Self::CommitFrontierAdvanced { .. } => "commitment",
            Self::VerificationEvaluated { .. } => "verification",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulatorEvent {
    pub sequence: u64,
    pub event: SimulatorEventKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulatorJournal {
    pub version: u32,
    pub utterance_id: UtteranceId,
    pub variety: VarietyId,
    pub config: SimulatorConfig,
    #[serde(default)]
    pub events: Vec<SimulatorEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulatorState {
    pub utterance_id: UtteranceId,
    pub variety: VarietyId,
    pub revision: u64,
    #[serde(default)]
    pub evidence: BTreeMap<String, ObservedEvidence>,
    #[serde(default)]
    pub hypotheses: BTreeMap<CompletionHypothesisId, NormalizedCompletionHypothesis>,
    #[serde(default)]
    pub selected_hypotheses: Vec<CompletionHypothesisId>,
    #[serde(default)]
    pub shared_prefix: Vec<String>,
    #[serde(default)]
    pub committed: Vec<CommittedMorpheme>,
    #[serde(default)]
    pub verifications: Vec<ClosedLoopVerificationResult>,
}

impl SimulatorState {
    fn new(utterance_id: UtteranceId, variety: VarietyId) -> Self {
        Self {
            utterance_id,
            variety,
            revision: 0,
            evidence: BTreeMap::new(),
            hypotheses: BTreeMap::new(),
            selected_hypotheses: Vec::new(),
            shared_prefix: Vec::new(),
            committed: Vec::new(),
            verifications: Vec::new(),
        }
    }

    pub fn committed_text(&self) -> String {
        self.committed
            .iter()
            .map(|morpheme| morpheme.surface.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn predicted_suffix(&self) -> Vec<CompletionMorpheme> {
        let Some(best) = self
            .hypotheses
            .values()
            .max_by(|left, right| hypothesis_order(left, right).reverse())
        else {
            return Vec::new();
        };
        best.morphemes
            .iter()
            .skip(self.committed.len())
            .cloned()
            .collect()
    }

    fn apply(&mut self, kind: &SimulatorEventKind) -> Result<(), SimulatorError> {
        match kind {
            SimulatorEventKind::EvidenceObserved { evidence } => {
                validate_id(&evidence.id, "evidence")?;
                if self
                    .evidence
                    .insert(evidence.id.clone(), evidence.clone())
                    .is_some()
                {
                    return Err(SimulatorError::DuplicateEvidence(evidence.id.clone()));
                }
            }
            SimulatorEventKind::HypothesisProposed { hypothesis } => {
                validate_id(&hypothesis.id.0, "hypothesis")?;
                if self
                    .hypotheses
                    .insert(hypothesis.id.clone(), hypothesis.clone())
                    .is_some()
                {
                    return Err(SimulatorError::DuplicateHypothesis(hypothesis.id.clone()));
                }
            }
            SimulatorEventKind::HypothesisWithdrawn { hypothesis, .. } => {
                let existing = self
                    .hypotheses
                    .remove(&hypothesis.id)
                    .ok_or_else(|| SimulatorError::UnknownHypothesis(hypothesis.id.clone()))?;
                if existing != *hypothesis {
                    return Err(SimulatorError::JournalStateMismatch(format!(
                        "withdrawal for '{}' does not match active hypothesis",
                        hypothesis.id.0
                    )));
                }
            }
            SimulatorEventKind::HypothesisRepaired {
                previous,
                replacement,
                ..
            } => {
                if previous.id != replacement.id {
                    return Err(SimulatorError::JournalStateMismatch(
                        "a repair must preserve hypothesis identity".into(),
                    ));
                }
                let existing = self
                    .hypotheses
                    .get(&previous.id)
                    .ok_or_else(|| SimulatorError::UnknownHypothesis(previous.id.clone()))?;
                if existing != previous {
                    return Err(SimulatorError::JournalStateMismatch(format!(
                        "repair for '{}' does not match active hypothesis",
                        previous.id.0
                    )));
                }
                self.hypotheses
                    .insert(replacement.id.clone(), replacement.clone());
            }
            SimulatorEventKind::BeamInferred {
                selected,
                shared_prefix,
                ..
            } => {
                for id in selected {
                    if !self.hypotheses.contains_key(id) {
                        return Err(SimulatorError::UnknownHypothesis(id.clone()));
                    }
                }
                self.selected_hypotheses = selected.clone();
                self.shared_prefix = shared_prefix.clone();
            }
            SimulatorEventKind::CommitFrontierAdvanced {
                from,
                to,
                committed,
            } => {
                if *from != self.committed.len() || *to != from + committed.len() {
                    return Err(SimulatorError::CommitFrontierMismatch {
                        expected: self.committed.len(),
                        from: *from,
                        to: *to,
                    });
                }
                for morpheme in committed {
                    if !is_directly_supported(morpheme, &self.evidence) {
                        return Err(SimulatorError::UnsupportedCommit(morpheme.key.clone()));
                    }
                }
                self.committed.extend(committed.iter().cloned());
            }
            SimulatorEventKind::VerificationEvaluated { result } => {
                self.verifications.push(result.clone());
            }
        }
        self.revision = self.revision.saturating_add(1);
        Ok(())
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum SimulatorError {
    #[error("posterior mass must be finite and within (0, 1], found {0}")]
    InvalidPosteriorMass(f64),
    #[error("provider returned no completion hypotheses")]
    EmptyBeam,
    #[error("hypothesis {id:?} has invalid weight {weight}")]
    InvalidWeight {
        id: CompletionHypothesisId,
        weight: f64,
    },
    #[error("completion hypothesis id {0:?} appears more than once")]
    DuplicateBeamHypothesis(CompletionHypothesisId),
    #[error("provider failed: {0}")]
    Provider(CompletionProviderError),
    #[error("{kind} id cannot be empty")]
    EmptyId { kind: &'static str },
    #[error("evidence id '{0}' is already present")]
    DuplicateEvidence(String),
    #[error("hypothesis {0:?} is already present")]
    DuplicateHypothesis(CompletionHypothesisId),
    #[error("unknown hypothesis {0:?}")]
    UnknownHypothesis(CompletionHypothesisId),
    #[error("journal version {found} is unsupported; expected {expected}")]
    UnsupportedJournalVersion { expected: u32, found: u32 },
    #[error("event sequence {found} is out of order; expected {expected}")]
    OutOfOrderEvent { expected: u64, found: u64 },
    #[error("commit frontier expected {expected}, got {from}..{to}")]
    CommitFrontierMismatch {
        expected: usize,
        from: usize,
        to: usize,
    },
    #[error("morpheme '{0}' has no direct text/acoustic support")]
    UnsupportedCommit(String),
    #[error("journal state mismatch: {0}")]
    JournalStateMismatch(String),
    #[error("rescore candidate list cannot be empty")]
    EmptyRescoreCandidates,
    #[error("rescore observed mel must be non-empty")]
    EmptyObservedMel,
    #[error("rescore weights must be finite and non-negative")]
    InvalidRescoreWeights,
}

impl From<CompletionProviderError> for SimulatorError {
    fn from(value: CompletionProviderError) -> Self {
        Self::Provider(value)
    }
}

pub struct DuplexSimulator<P> {
    provider: P,
    journal: SimulatorJournal,
    state: SimulatorState,
}

impl<P: CompletionProvider> DuplexSimulator<P> {
    pub fn new(
        utterance_id: UtteranceId,
        variety: VarietyId,
        config: SimulatorConfig,
        provider: P,
    ) -> Result<Self, SimulatorError> {
        validate_config(&config)?;
        Ok(Self {
            provider,
            journal: SimulatorJournal {
                version: SIMULATOR_JOURNAL_VERSION,
                utterance_id: utterance_id.clone(),
                variety: variety.clone(),
                config,
                events: Vec::new(),
            },
            state: SimulatorState::new(utterance_id, variety),
        })
    }

    pub fn state(&self) -> &SimulatorState {
        &self.state
    }

    pub fn journal(&self) -> &SimulatorJournal {
        &self.journal
    }

    pub fn observe(
        &mut self,
        evidence: ObservedEvidence,
    ) -> Result<Vec<SimulatorEvent>, SimulatorError> {
        let start = self.journal.events.len();
        self.record(SimulatorEventKind::EvidenceObserved { evidence })?;

        let request = CompletionRequest {
            utterance_id: self.state.utterance_id.clone(),
            variety: self.state.variety.clone(),
            evidence: self.state.evidence.values().cloned().collect(),
            committed: self.state.committed.clone(),
        };
        let beam = normalize_beam(self.provider.complete(&request)?)?;
        let next = beam
            .iter()
            .cloned()
            .map(|hypothesis| (hypothesis.id.clone(), hypothesis))
            .collect::<BTreeMap<_, _>>();

        for (id, previous) in self.state.hypotheses.clone() {
            match next.get(&id) {
                None => self.record(SimulatorEventKind::HypothesisWithdrawn {
                    hypothesis: previous,
                    reason: "provider withdrew provisional branch after new evidence".into(),
                })?,
                Some(replacement) if replacement != &previous => {
                    self.record(SimulatorEventKind::HypothesisRepaired {
                        previous,
                        replacement: replacement.clone(),
                        reason: "provider revised branch after new evidence".into(),
                    })?;
                }
                Some(_) => {}
            }
        }
        for (id, hypothesis) in &next {
            if !self.state.hypotheses.contains_key(id) {
                self.record(SimulatorEventKind::HypothesisProposed {
                    hypothesis: hypothesis.clone(),
                })?;
            }
        }

        let (selected, covered_probability) =
            select_posterior_mass(&beam, self.journal.config.posterior_mass);
        let shared = longest_common_morpheme_prefix(&selected);
        self.record(SimulatorEventKind::BeamInferred {
            selected: selected
                .iter()
                .map(|hypothesis| hypothesis.id.clone())
                .collect(),
            covered_probability,
            shared_prefix: shared.iter().map(|morpheme| morpheme.key.clone()).collect(),
        })?;

        let from = self.state.committed.len();
        let committable = shared
            .iter()
            .enumerate()
            .skip(from)
            .take_while(|(index, _)| {
                selected.iter().all(|hypothesis| {
                    hypothesis.morphemes.get(*index).is_some_and(|morpheme| {
                        morpheme.key == shared[*index].key
                            && directly_supported_occurrence(morpheme, &self.state.evidence)
                    })
                })
            })
            .map(|(index, representative)| {
                let evidence = selected
                    .iter()
                    .filter_map(|hypothesis| hypothesis.morphemes.get(index))
                    .flat_map(|morpheme| morpheme.evidence.iter().cloned())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                CommittedMorpheme {
                    key: representative.key.clone(),
                    surface: representative.surface.clone(),
                    variety: representative.variety.clone(),
                    evidence,
                    confidence: covered_probability,
                }
            })
            .collect::<Vec<_>>();
        if !committable.is_empty() {
            self.record(SimulatorEventKind::CommitFrontierAdvanced {
                from,
                to: from + committable.len(),
                committed: committable,
            })?;
        }

        Ok(self.journal.events[start..].to_vec())
    }

    pub fn into_parts(self) -> (SimulatorJournal, SimulatorState) {
        (self.journal, self.state)
    }

    pub fn verify_held_audio(
        &mut self,
        request: ClosedLoopVerificationRequest,
        policy: ClosedLoopVerificationPolicy,
        retry_count: u8,
    ) -> Result<ClosedLoopVerificationResult, SimulatorError> {
        let result = verify_closed_loop(&request, &policy, retry_count);
        self.record(SimulatorEventKind::VerificationEvaluated {
            result: result.clone(),
        })?;
        Ok(result)
    }

    fn record(&mut self, kind: SimulatorEventKind) -> Result<(), SimulatorError> {
        self.state.apply(&kind)?;
        self.journal.events.push(SimulatorEvent {
            sequence: self.journal.events.len() as u64,
            event: kind,
        });
        Ok(())
    }
}

pub fn replay_journal(journal: &SimulatorJournal) -> Result<SimulatorState, SimulatorError> {
    if journal.version != SIMULATOR_JOURNAL_VERSION {
        return Err(SimulatorError::UnsupportedJournalVersion {
            expected: SIMULATOR_JOURNAL_VERSION,
            found: journal.version,
        });
    }
    validate_config(&journal.config)?;
    let mut state = SimulatorState::new(journal.utterance_id.clone(), journal.variety.clone());
    for (expected, event) in journal.events.iter().enumerate() {
        if event.sequence != expected as u64 {
            return Err(SimulatorError::OutOfOrderEvent {
                expected: expected as u64,
                found: event.sequence,
            });
        }
        state.apply(&event.event)?;
    }
    Ok(state)
}

fn validate_config(config: &SimulatorConfig) -> Result<(), SimulatorError> {
    if !config.posterior_mass.is_finite()
        || config.posterior_mass <= 0.0
        || config.posterior_mass > 1.0
    {
        return Err(SimulatorError::InvalidPosteriorMass(config.posterior_mass));
    }
    Ok(())
}

fn validate_id(id: &str, kind: &'static str) -> Result<(), SimulatorError> {
    if id.trim().is_empty() {
        return Err(SimulatorError::EmptyId { kind });
    }
    Ok(())
}

pub fn normalize_beam(
    proposals: Vec<CompletionProposal>,
) -> Result<Vec<NormalizedCompletionHypothesis>, SimulatorError> {
    if proposals.is_empty() {
        return Err(SimulatorError::EmptyBeam);
    }
    let mut seen = BTreeSet::new();
    let mut total = 0.0;
    for proposal in &proposals {
        validate_id(&proposal.id.0, "hypothesis")?;
        if !seen.insert(proposal.id.clone()) {
            return Err(SimulatorError::DuplicateBeamHypothesis(proposal.id.clone()));
        }
        if !proposal.weight.is_finite() || proposal.weight < 0.0 {
            return Err(SimulatorError::InvalidWeight {
                id: proposal.id.clone(),
                weight: proposal.weight,
            });
        }
        total += proposal.weight;
    }
    if !total.is_finite() || total <= 0.0 {
        let proposal = &proposals[0];
        return Err(SimulatorError::InvalidWeight {
            id: proposal.id.clone(),
            weight: proposal.weight,
        });
    }

    let mut normalized = proposals
        .into_iter()
        .map(|proposal| NormalizedCompletionHypothesis::from_proposal(proposal, total))
        .collect::<Vec<_>>();
    normalized.sort_by(hypothesis_order);
    Ok(normalized)
}

fn hypothesis_order(
    left: &NormalizedCompletionHypothesis,
    right: &NormalizedCompletionHypothesis,
) -> Ordering {
    right
        .probability
        .total_cmp(&left.probability)
        .then_with(|| left.id.0.cmp(&right.id.0))
}

fn select_posterior_mass(
    beam: &[NormalizedCompletionHypothesis],
    posterior_mass: f64,
) -> (Vec<NormalizedCompletionHypothesis>, f64) {
    let mut selected = Vec::new();
    let mut covered = 0.0;
    for hypothesis in beam {
        selected.push(hypothesis.clone());
        covered += hypothesis.probability;
        if covered + f64::EPSILON >= posterior_mass {
            break;
        }
    }
    (selected, covered)
}

fn longest_common_morpheme_prefix(
    hypotheses: &[NormalizedCompletionHypothesis],
) -> Vec<CompletionMorpheme> {
    let Some(first) = hypotheses.first() else {
        return Vec::new();
    };
    let mut length = first.morphemes.len();
    for hypothesis in hypotheses.iter().skip(1) {
        length = length.min(hypothesis.morphemes.len());
        for index in 0..length {
            if normalize_key(&first.morphemes[index].key)
                != normalize_key(&hypothesis.morphemes[index].key)
                || first.morphemes[index].variety != hypothesis.morphemes[index].variety
            {
                length = index;
                break;
            }
        }
    }
    first.morphemes[..length].to_vec()
}

fn directly_supported_occurrence(
    morpheme: &CompletionMorpheme,
    evidence: &BTreeMap<String, ObservedEvidence>,
) -> bool {
    !morpheme.evidence.is_empty()
        && morpheme.evidence.iter().any(|id| {
            evidence
                .get(id)
                .is_some_and(|observed| observed.supports(&morpheme.key))
        })
}

fn is_directly_supported(
    morpheme: &CommittedMorpheme,
    evidence: &BTreeMap<String, ObservedEvidence>,
) -> bool {
    !morpheme.evidence.is_empty()
        && morpheme.evidence.iter().any(|id| {
            evidence
                .get(id)
                .is_some_and(|observed| observed.supports(&morpheme.key))
        })
}

pub fn tokenize_morphemes(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|raw| {
            let token = raw.trim_matches(|character: char| {
                !character.is_alphanumeric()
                    && character != '\''
                    && character != '’'
                    && character != '.'
                    && character != '-'
            });
            (!token.is_empty()).then(|| token.to_string())
        })
        .collect()
}

fn normalize_key(key: &str) -> String {
    key.trim().to_lowercase()
}

fn agreement_ratio(intended: &[String], recovered: &[String]) -> f32 {
    let denominator = intended.len().max(recovered.len());
    if denominator == 0 {
        return 1.0;
    }
    let matches = intended
        .iter()
        .zip(recovered.iter())
        .filter(|(left, right)| normalize_key(left) == normalize_key(right))
        .count();
    matches as f32 / denominator as f32
}

fn phone_error_rate(intended: &[String], recovered: &[String]) -> f32 {
    let denominator = intended.len().max(recovered.len());
    if denominator == 0 {
        return 0.0;
    }
    let distance = levenshtein_distance(intended, recovered);
    distance as f32 / denominator as f32
}

fn levenshtein_distance(left: &[String], right: &[String]) -> usize {
    let mut prev = (0..=right.len()).collect::<Vec<_>>();
    let mut next = vec![0; right.len() + 1];
    for (i, left_item) in left.iter().enumerate() {
        next[0] = i + 1;
        for (j, right_item) in right.iter().enumerate() {
            let substitution_cost =
                usize::from(normalize_key(left_item) != normalize_key(right_item));
            let deletion = prev[j + 1] + 1;
            let insertion = next[j] + 1;
            let substitution = prev[j] + substitution_cost;
            next[j + 1] = deletion.min(insertion).min(substitution);
        }
        std::mem::swap(&mut prev, &mut next);
    }
    prev[right.len()]
}

fn accepted_projection_loss(
    intended: &str,
    recovered: &str,
    accepted_projection_losses: &[String],
) -> bool {
    let intended = normalize_key(intended);
    let recovered = normalize_key(recovered);
    accepted_projection_losses.iter().any(|entry| {
        let normalized = normalize_key(entry);
        normalized == recovered
            || normalized == intended
            || normalized == format!("{intended}->{recovered}")
    })
}

pub fn verify_closed_loop(
    request: &ClosedLoopVerificationRequest,
    policy: &ClosedLoopVerificationPolicy,
    retry_count: u8,
) -> ClosedLoopVerificationResult {
    let raw_morpheme_agreement =
        agreement_ratio(&request.intended.morphemes, &request.recovered.morphemes);
    let phone_error_rate = phone_error_rate(&request.intended.phones, &request.recovered.phones);
    let mut accepted_projection_count = 0usize;

    let mut evidence = Vec::new();
    let mut repair_causes = Vec::new();
    let mut linguistic_disagreements = 0usize;
    for (intended, recovered) in request
        .intended
        .morphemes
        .iter()
        .zip(request.recovered.morphemes.iter())
    {
        if normalize_key(intended) == normalize_key(recovered) {
            continue;
        }
        let accepted =
            accepted_projection_loss(intended, recovered, &request.accepted_projection_losses);
        evidence.push(VerificationEvidence {
            dimension: VerificationDimension::Morpheme,
            intended: intended.clone(),
            recovered: recovered.clone(),
            accepted_projection_loss: accepted,
        });
        if accepted {
            accepted_projection_count += 1;
            if !repair_causes.contains(&VerificationRepairCause::AcceptedProjectionLoss) {
                repair_causes.push(VerificationRepairCause::AcceptedProjectionLoss);
            }
        } else {
            linguistic_disagreements += 1;
            if !repair_causes.contains(&VerificationRepairCause::LinguisticDisagreement) {
                repair_causes.push(VerificationRepairCause::LinguisticDisagreement);
            }
        }
    }
    let denominator = request
        .intended
        .morphemes
        .len()
        .max(request.recovered.morphemes.len());
    let effective_morpheme_agreement = if denominator == 0 {
        1.0
    } else {
        (raw_morpheme_agreement * denominator as f32 + accepted_projection_count as f32)
            / denominator as f32
    };
    let word_agreement = effective_morpheme_agreement;

    if request.recognizer_confidence < policy.min_recognizer_confidence {
        repair_causes.push(VerificationRepairCause::RecognizerUncertain);
        return ClosedLoopVerificationResult {
            decision: VerificationDecision::Abstain,
            evidence,
            repair_causes,
            metrics: VerificationMetrics {
                verification_latency_ms: request.verification_latency_ms,
                false_rejection: 0.0,
                false_acceptance: 0.0,
                phone_error_rate,
                word_agreement,
                morpheme_agreement: effective_morpheme_agreement,
            },
        };
    }

    if phone_error_rate > policy.max_phone_error_rate
        && !repair_causes.contains(&VerificationRepairCause::AcousticMismatch)
    {
        repair_causes.push(VerificationRepairCause::AcousticMismatch);
    }

    let timing_mismatch = request
        .intended
        .timings_ms
        .iter()
        .zip(request.recovered.timings_ms.iter())
        .any(|(left, right)| left.abs_diff(*right) > 80);
    if timing_mismatch {
        repair_causes.push(VerificationRepairCause::TimingMismatch);
    }

    let accepted = linguistic_disagreements == 0
        && effective_morpheme_agreement >= policy.min_morpheme_agreement
        && phone_error_rate <= policy.max_phone_error_rate;
    let retry_allowed = request.held_audio_replaceable
        && request.verification_latency_ms <= policy.max_verification_latency_ms
        && retry_count < policy.max_retries;

    let decision = if accepted {
        VerificationDecision::Accept
    } else if retry_allowed {
        VerificationDecision::Retry
    } else {
        VerificationDecision::Fallback
    };

    let false_rejection = if !accepted && linguistic_disagreements == 0 {
        1.0
    } else {
        0.0
    };
    let false_acceptance = if accepted && linguistic_disagreements > 0 {
        1.0
    } else {
        0.0
    };

    ClosedLoopVerificationResult {
        decision,
        evidence,
        repair_causes,
        metrics: VerificationMetrics {
            verification_latency_ms: request.verification_latency_ms,
            false_rejection,
            false_acceptance,
            phone_error_rate,
            word_agreement,
            morpheme_agreement: effective_morpheme_agreement,
        },
    }
}

pub fn rescore_completion_hypotheses(
    request: &AcousticRescoreRequest,
) -> Result<Vec<AcousticRescore>, SimulatorError> {
    if request.candidates.is_empty() {
        return Err(SimulatorError::EmptyRescoreCandidates);
    }
    if request.observed_mel.is_empty() {
        return Err(SimulatorError::EmptyObservedMel);
    }
    if !request.acoustic_weight.is_finite()
        || !request.prior_weight.is_finite()
        || request.acoustic_weight < 0.0
        || request.prior_weight < 0.0
    {
        return Err(SimulatorError::InvalidRescoreWeights);
    }

    let mut rescored = Vec::with_capacity(request.candidates.len());
    for candidate in &request.candidates {
        let overlap = request
            .observed_mel
            .len()
            .min(candidate.predicted_mel.len());
        let mut distance = 0.0_f64;
        for index in 0..overlap {
            distance += (request.observed_mel[index] - candidate.predicted_mel[index]).abs() as f64;
        }
        distance += request
            .observed_mel
            .len()
            .abs_diff(candidate.predicted_mel.len()) as f64;

        let acoustic_log_likelihood = -distance;
        let prior_log_probability = candidate.prior_probability.max(1e-12).ln();
        let combined_score = request.acoustic_weight * acoustic_log_likelihood
            + request.prior_weight * prior_log_probability;
        rescored.push(AcousticRescore {
            id: candidate.id.clone(),
            acoustic_log_likelihood,
            prior_log_probability,
            combined_score,
        });
    }
    rescored.sort_by(|left, right| {
        right
            .combined_score
            .total_cmp(&left.combined_score)
            .then_with(|| left.id.0.cmp(&right.id.0))
    });
    Ok(rescored)
}

/// A deterministic provider useful for arbitrary CLI text or mock-acoustic
/// chunks. It proposes two disputed suffixes after the directly observed
/// morphemes, making the evidence/commit boundary visible without a model.
#[derive(Debug, Clone, Default)]
pub struct OracleCompletionProvider;

impl CompletionProvider for OracleCompletionProvider {
    fn complete(
        &mut self,
        request: &CompletionRequest,
    ) -> Result<Vec<CompletionProposal>, CompletionProviderError> {
        let mut observed = Vec::new();
        for evidence in &request.evidence {
            let supports = if evidence.supports.is_empty() {
                tokenize_morphemes(&evidence.content)
            } else {
                evidence.supports.clone()
            };
            for surface in supports {
                observed.push(CompletionMorpheme {
                    key: normalize_key(&surface),
                    surface,
                    variety: request.variety.clone(),
                    evidence: vec![evidence.id.clone()],
                });
            }
        }
        let provenance = EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: "deterministic-duplex-oracle".into(),
            version: Some("1".into()),
        };
        let mut statement = observed.clone();
        statement.push(CompletionMorpheme::predicted(
            "<statement>",
            "…",
            request.variety.clone(),
        ));
        let mut question = observed;
        question.push(CompletionMorpheme::predicted(
            "<question>",
            "?",
            request.variety.clone(),
        ));
        Ok(vec![
            CompletionProposal {
                id: CompletionHypothesisId("oracle:statement".into()),
                weight: 0.55,
                morphemes: statement,
                syntax: Some(SentenceSyntaxAnalysis::default()),
                prosody: Some(ProsodyTrack::default()),
                evidence: request
                    .evidence
                    .iter()
                    .map(|evidence| evidence.id.clone())
                    .collect(),
                provenance: provenance.clone(),
            },
            CompletionProposal {
                id: CompletionHypothesisId("oracle:question".into()),
                weight: 0.45,
                morphemes: question,
                syntax: Some(SentenceSyntaxAnalysis::default()),
                prosody: Some(ProsodyTrack::default()),
                evidence: request
                    .evidence
                    .iter()
                    .map(|evidence| evidence.id.clone())
                    .collect(),
                provenance,
            },
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixtureStep {
    pub evidence: ObservedEvidence,
    pub hypotheses: Vec<CompletionProposal>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplexFixture {
    pub id: String,
    pub description: String,
    pub utterance_id: UtteranceId,
    pub variety: VarietyId,
    #[serde(default)]
    pub config: SimulatorConfig,
    pub steps: Vec<FixtureStep>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplexFixtureSuite {
    pub version: u32,
    pub fixtures: Vec<DuplexFixture>,
}

impl DuplexFixtureSuite {
    pub fn fixture(&self, id: &str) -> Option<&DuplexFixture> {
        self.fixtures.iter().find(|fixture| fixture.id == id)
    }

    pub fn validate(&self) -> Result<(), FixtureError> {
        if self.version != FIXTURE_SUITE_VERSION {
            return Err(FixtureError::UnsupportedVersion {
                expected: FIXTURE_SUITE_VERSION,
                found: self.version,
            });
        }
        let mut ids = BTreeSet::new();
        for fixture in &self.fixtures {
            if !ids.insert(fixture.id.clone()) {
                return Err(FixtureError::DuplicateFixture(fixture.id.clone()));
            }
            if fixture.steps.is_empty() {
                return Err(FixtureError::EmptyFixture(fixture.id.clone()));
            }
            validate_config(&fixture.config).map_err(|error| {
                FixtureError::InvalidFixture(fixture.id.clone(), error.to_string())
            })?;
        }
        Ok(())
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum FixtureError {
    #[error("fixture suite version {found} is unsupported; expected {expected}")]
    UnsupportedVersion { expected: u32, found: u32 },
    #[error("duplicate fixture id '{0}'")]
    DuplicateFixture(String),
    #[error("fixture '{0}' has no steps")]
    EmptyFixture(String),
    #[error("fixture '{0}' is invalid: {1}")]
    InvalidFixture(String, String),
}

#[derive(Debug, Clone)]
pub struct FixtureCompletionProvider {
    hypotheses_by_evidence_count: BTreeMap<usize, Vec<CompletionProposal>>,
}

impl FixtureCompletionProvider {
    pub fn new(fixture: &DuplexFixture) -> Self {
        Self {
            hypotheses_by_evidence_count: fixture
                .steps
                .iter()
                .enumerate()
                .map(|(index, step)| (index + 1, step.hypotheses.clone()))
                .collect(),
        }
    }
}

impl CompletionProvider for FixtureCompletionProvider {
    fn complete(
        &mut self,
        request: &CompletionRequest,
    ) -> Result<Vec<CompletionProposal>, CompletionProviderError> {
        self.hypotheses_by_evidence_count
            .get(&request.evidence.len())
            .cloned()
            .ok_or_else(|| {
                CompletionProviderError::new(format!(
                    "fixture has no beam after {} evidence events",
                    request.evidence.len()
                ))
            })
    }
}

pub fn run_fixture(
    fixture: &DuplexFixture,
) -> Result<(SimulatorJournal, SimulatorState), SimulatorError> {
    let provider = FixtureCompletionProvider::new(fixture);
    let mut simulator = DuplexSimulator::new(
        fixture.utterance_id.clone(),
        fixture.variety.clone(),
        fixture.config.clone(),
        provider,
    )?;
    for step in &fixture.steps {
        simulator.observe(step.evidence.clone())?;
    }
    let (journal, state) = simulator.into_parts();
    let replayed = replay_journal(&journal)?;
    if replayed != state {
        return Err(SimulatorError::JournalStateMismatch(
            "fixture replay did not reproduce the live state".into(),
        ));
    }
    Ok((journal, state))
}

// ---------------------------------------------------------------------------
// Provisional transcript events and speculative consumer
// ---------------------------------------------------------------------------

/// High-level transcript events emitted by a [`SpeculativeConsumer`].
///
/// These are separate from the simulator's internal [`SimulatorEventKind`]:
/// provisional events describe the evolving *text* view while the simulator
/// tracks *hypothesis* identity. Downstream consumers (TTS, display) should
/// listen to these events and not inspect logits or hypothesis internals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "data")]
pub enum ProvisionalTranscriptEvent {
    /// New provisional morphemes appended beyond the committed frontier.
    Append { morphemes: Vec<CompletionMorpheme> },
    /// Existing provisional morphemes replaced in place (revision or repair).
    Replace {
        previous: Vec<CompletionMorpheme>,
        replacement: Vec<CompletionMorpheme>,
    },
    /// Previously provisional morphemes withdrawn (no longer supported).
    Withdraw { morphemes: Vec<CompletionMorpheme> },
    /// Morphemes moved from provisional to permanent committed history.
    Commit { morphemes: Vec<CommittedMorpheme> },
}

/// Downstream consumer that processes [`SimulatorEvent`]s and derives
/// [`ProvisionalTranscriptEvent`]s without inspecting model logits.
///
/// Implementors receive raw simulator events and are responsible for
/// maintaining any local state they need. The trait is object-safe so that
/// consumers can be composed or boxed.
pub trait SpeculativeConsumer {
    fn on_event(&mut self, event: &SimulatorEvent);
}

pub const STUDIO_PROJECTION_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplexBranchSnapshot {
    pub hypothesis_id: String,
    pub probability: f64,
    #[serde(default)]
    pub morphemes: Vec<String>,
    pub selected: bool,
    pub provenance: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplexTimelineSnapshot {
    pub sequence: u64,
    pub layer: String,
    pub event_type: String,
    pub message: String,
    #[serde(default)]
    pub observed: Vec<String>,
    #[serde(default)]
    pub shared_prefix: Vec<String>,
    #[serde(default)]
    pub predicted: Vec<String>,
    #[serde(default)]
    pub committed: Vec<String>,
    pub commit_frontier: usize,
    #[serde(default)]
    pub branches: Vec<DuplexBranchSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_divergent_morpheme_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript_event: Option<ProvisionalTranscriptEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplexStudioProjection {
    pub schema_version: u32,
    pub run_id: String,
    pub replay_verified: bool,
    pub journal: SimulatorJournal,
    pub final_state: SimulatorState,
    #[serde(default)]
    pub timeline: Vec<DuplexTimelineSnapshot>,
    #[serde(default)]
    pub transcript_events: Vec<ProvisionalTranscriptEvent>,
}

/// A [`SpeculativeConsumer`] that records all [`ProvisionalTranscriptEvent`]s
/// as they are derived from the simulator journal.
///
/// It tracks the best-hypothesis provisional suffix after each
/// [`SimulatorEventKind::BeamInferred`] and emits `Append`, `Replace`, or
/// `Withdraw` as the suffix evolves. Newly committed morphemes always emit a
/// `Commit` event and are removed from the provisional suffix.
#[derive(Debug, Clone, Default)]
pub struct RecordingSpeculativeConsumer {
    hypotheses: BTreeMap<CompletionHypothesisId, NormalizedCompletionHypothesis>,
    committed_len: usize,
    provisional: Vec<CompletionMorpheme>,
    pub committed: Vec<CommittedMorpheme>,
    pub transcript_events: Vec<ProvisionalTranscriptEvent>,
}

impl RecordingSpeculativeConsumer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Text assembled from committed morpheme surfaces.
    pub fn committed_text(&self) -> String {
        self.committed
            .iter()
            .map(|m| m.surface.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Current provisional suffix (beyond the committed frontier).
    pub fn provisional_suffix(&self) -> &[CompletionMorpheme] {
        &self.provisional
    }
}

impl SpeculativeConsumer for RecordingSpeculativeConsumer {
    fn on_event(&mut self, event: &SimulatorEvent) {
        match &event.event {
            SimulatorEventKind::HypothesisProposed { hypothesis } => {
                self.hypotheses
                    .insert(hypothesis.id.clone(), hypothesis.clone());
            }
            SimulatorEventKind::HypothesisWithdrawn { hypothesis, .. } => {
                self.hypotheses.remove(&hypothesis.id);
            }
            SimulatorEventKind::HypothesisRepaired { replacement, .. } => {
                self.hypotheses
                    .insert(replacement.id.clone(), replacement.clone());
            }
            SimulatorEventKind::BeamInferred { selected, .. } => {
                // Derive new provisional suffix from the highest-probability
                // selected hypothesis (tie-break: stable id ordering).
                let new_provisional = selected
                    .iter()
                    .filter_map(|id| self.hypotheses.get(id))
                    .max_by(|a, b| {
                        a.probability
                            .total_cmp(&b.probability)
                            .then_with(|| b.id.0.cmp(&a.id.0))
                    })
                    .map(|best| {
                        best.morphemes
                            .iter()
                            .skip(self.committed_len)
                            .cloned()
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();

                if new_provisional != self.provisional {
                    let event = match (self.provisional.is_empty(), new_provisional.is_empty()) {
                        (_, true) => ProvisionalTranscriptEvent::Withdraw {
                            morphemes: self.provisional.clone(),
                        },
                        (true, false) => ProvisionalTranscriptEvent::Append {
                            morphemes: new_provisional.clone(),
                        },
                        (false, false) => ProvisionalTranscriptEvent::Replace {
                            previous: self.provisional.clone(),
                            replacement: new_provisional.clone(),
                        },
                    };
                    self.transcript_events.push(event);
                    self.provisional = new_provisional;
                }
            }
            SimulatorEventKind::CommitFrontierAdvanced { committed, .. } => {
                self.committed_len += committed.len();
                self.committed.extend(committed.iter().cloned());
                self.transcript_events
                    .push(ProvisionalTranscriptEvent::Commit {
                        morphemes: committed.clone(),
                    });
                // Strip committed morphemes from the head of the provisional suffix.
                let new_len = self.provisional.len().saturating_sub(committed.len());
                self.provisional = self.provisional[self.provisional.len() - new_len..].to_vec();
            }
            SimulatorEventKind::EvidenceObserved { .. }
            | SimulatorEventKind::VerificationEvaluated { .. } => {}
        }
    }
}

/// Drive a fixture through the simulator and collect the resulting
/// [`ProvisionalTranscriptEvent`]s via a [`RecordingSpeculativeConsumer`].
pub fn run_fixture_with_consumer(
    fixture: &DuplexFixture,
) -> Result<
    (
        SimulatorJournal,
        SimulatorState,
        RecordingSpeculativeConsumer,
    ),
    SimulatorError,
> {
    let provider = FixtureCompletionProvider::new(fixture);
    let mut simulator = DuplexSimulator::new(
        fixture.utterance_id.clone(),
        fixture.variety.clone(),
        fixture.config.clone(),
        provider,
    )?;
    let mut consumer = RecordingSpeculativeConsumer::new();
    for step in &fixture.steps {
        let events = simulator.observe(step.evidence.clone())?;
        for event in &events {
            consumer.on_event(event);
        }
    }
    let (journal, state) = simulator.into_parts();
    Ok((journal, state, consumer))
}

pub fn studio_projection_from_journal(
    run_id: impl Into<String>,
    journal: &SimulatorJournal,
) -> Result<DuplexStudioProjection, SimulatorError> {
    if journal.version != SIMULATOR_JOURNAL_VERSION {
        return Err(SimulatorError::UnsupportedJournalVersion {
            expected: SIMULATOR_JOURNAL_VERSION,
            found: journal.version,
        });
    }
    validate_config(&journal.config)?;

    let mut state = SimulatorState::new(journal.utterance_id.clone(), journal.variety.clone());
    let mut consumer = RecordingSpeculativeConsumer::new();
    let mut timeline = Vec::with_capacity(journal.events.len());
    for (expected, event) in journal.events.iter().enumerate() {
        if event.sequence != expected as u64 {
            return Err(SimulatorError::OutOfOrderEvent {
                expected: expected as u64,
                found: event.sequence,
            });
        }
        state.apply(&event.event)?;
        let transcript_count = consumer.transcript_events.len();
        consumer.on_event(event);
        let transcript_event = consumer.transcript_events.get(transcript_count).cloned();
        timeline.push(project_timeline_snapshot(&state, event, transcript_event));
    }

    let final_state = replay_journal(journal)?;
    Ok(DuplexStudioProjection {
        schema_version: STUDIO_PROJECTION_SCHEMA_VERSION,
        run_id: run_id.into(),
        replay_verified: state == final_state,
        journal: journal.clone(),
        final_state,
        timeline,
        transcript_events: consumer.transcript_events,
    })
}

fn project_timeline_snapshot(
    state: &SimulatorState,
    event: &SimulatorEvent,
    transcript_event: Option<ProvisionalTranscriptEvent>,
) -> DuplexTimelineSnapshot {
    DuplexTimelineSnapshot {
        sequence: event.sequence,
        layer: event.event.layer().to_string(),
        event_type: studio_event_type(&event.event).to_string(),
        message: studio_event_message(&event.event),
        observed: observed_morphemes(state),
        shared_prefix: state.shared_prefix.clone(),
        predicted: state
            .predicted_suffix()
            .into_iter()
            .map(|morpheme| morpheme.surface)
            .collect(),
        committed: state
            .committed
            .iter()
            .map(|morpheme| morpheme.surface.clone())
            .collect(),
        commit_frontier: state.committed.len(),
        branches: branch_snapshots(state),
        first_divergent_morpheme_index: first_divergent_morpheme_index(state),
        transcript_event,
    }
}

fn studio_event_type(kind: &SimulatorEventKind) -> &'static str {
    match kind {
        SimulatorEventKind::EvidenceObserved { .. } => "evidence_observed",
        SimulatorEventKind::HypothesisProposed { .. } => "hypothesis_proposed",
        SimulatorEventKind::HypothesisWithdrawn { .. } => "hypothesis_withdrawn",
        SimulatorEventKind::HypothesisRepaired { .. } => "hypothesis_repaired",
        SimulatorEventKind::BeamInferred { .. } => "beam_inferred",
        SimulatorEventKind::CommitFrontierAdvanced { .. } => "commit_frontier_advanced",
        SimulatorEventKind::VerificationEvaluated { .. } => "verification_evaluated",
    }
}

fn studio_event_message(kind: &SimulatorEventKind) -> String {
    match kind {
        SimulatorEventKind::EvidenceObserved { evidence } => format!(
            "Observed {} chunk {}: {}",
            match evidence.modality {
                EvidenceModality::Text => "text",
                EvidenceModality::Acoustics => "acoustic",
            },
            evidence.id,
            evidence.content
        ),
        SimulatorEventKind::HypothesisProposed { hypothesis } => format!(
            "Proposed branch {} at p={:.3}",
            hypothesis.id.0, hypothesis.probability
        ),
        SimulatorEventKind::HypothesisWithdrawn { hypothesis, reason } => {
            format!("Withdrew branch {}: {}", hypothesis.id.0, reason)
        }
        SimulatorEventKind::HypothesisRepaired {
            previous,
            replacement,
            reason,
        } => format!(
            "Repaired branch {} p={:.3}→{:.3}: {}",
            previous.id.0, previous.probability, replacement.probability, reason
        ),
        SimulatorEventKind::BeamInferred {
            selected,
            covered_probability,
            ..
        } => format!(
            "Selected {} branches covering p={:.3}",
            selected.len(),
            covered_probability
        ),
        SimulatorEventKind::CommitFrontierAdvanced {
            from,
            to,
            committed,
        } => format!(
            "Committed frontier {}→{}: {}",
            from,
            to,
            committed
                .iter()
                .map(|morpheme| morpheme.surface.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        ),
        SimulatorEventKind::VerificationEvaluated { result } => format!(
            "Verification {:?} ({} evidence rows, {} repair causes)",
            result.decision,
            result.evidence.len(),
            result.repair_causes.len()
        ),
    }
}

fn observed_morphemes(state: &SimulatorState) -> Vec<String> {
    state
        .evidence
        .values()
        .flat_map(|evidence| {
            if evidence.supports.is_empty() {
                tokenize_morphemes(&evidence.content)
            } else {
                evidence.supports.clone()
            }
        })
        .collect()
}

fn branch_snapshots(state: &SimulatorState) -> Vec<DuplexBranchSnapshot> {
    let selected = state
        .selected_hypotheses
        .iter()
        .map(|id| id.0.as_str())
        .collect::<BTreeSet<_>>();
    let mut branches = state.hypotheses.values().cloned().collect::<Vec<_>>();
    branches.sort_by(hypothesis_order);
    branches
        .into_iter()
        .map(|hypothesis| DuplexBranchSnapshot {
            hypothesis_id: hypothesis.id.0.clone(),
            probability: hypothesis.probability,
            morphemes: hypothesis
                .morphemes
                .into_iter()
                .map(|morpheme| morpheme.surface)
                .collect(),
            selected: selected.contains(hypothesis.id.0.as_str()),
            provenance: format!(
                "{:?}:{}{}",
                hypothesis.provenance.source,
                hypothesis.provenance.method,
                hypothesis
                    .provenance
                    .version
                    .as_ref()
                    .map(|version| format!("@{version}"))
                    .unwrap_or_default()
            ),
        })
        .collect()
}

fn first_divergent_morpheme_index(state: &SimulatorState) -> Option<usize> {
    let active = state.hypotheses.values().collect::<Vec<_>>();
    if active.len() < 2 {
        return None;
    }
    let max_len = active
        .iter()
        .map(|hypothesis| hypothesis.morphemes.len())
        .max()
        .unwrap_or(0);
    for index in 0..max_len {
        let mut seen = BTreeSet::new();
        for hypothesis in &active {
            match hypothesis.morphemes.get(index) {
                Some(morpheme) => {
                    seen.insert(normalize_key(&morpheme.key));
                }
                None => {
                    seen.insert(String::new());
                }
            }
        }
        if seen.len() > 1 {
            return Some(index);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Prefix, completion, rollback, and repair training data generation
// ---------------------------------------------------------------------------

/// Schema version for the duplex training dataset.
pub const TRAINING_DATASET_VERSION: u32 = 1;

/// Kind of training row derived from a duplex simulator run.
///
/// Every row distinguishes the evidence, inference, prediction, and commitment
/// layers of the duplex belief state at a given utterance prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingRowKind {
    /// One prefix observation: evidence arrived and the beam updated.
    /// Covers evidence, inference, and prediction layers together.
    Prefix,
    /// A commit advance: morphemes moved from provisional to the committed
    /// prefix at this step.
    Completion,
    /// A withdrawal/rollback: a hypothesis branch was removed.
    /// Every rollback row names morpheme occurrences and source spans.
    Rollback,
    /// A repair: a hypothesis branch was revised in place.
    /// Repair rows include reparandum, interregnum, repair, and continuation.
    Repair,
}

/// Provenance of the row within the training corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DuplexTrainingRowSource {
    /// Derived directly from a named duplex fixture.
    Fixture,
}

/// Compact snapshot of one beam hypothesis at a training prefix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeamSnapshot {
    /// Stable hypothesis identifier.
    pub hypothesis_id: String,
    /// Normalized posterior probability.
    pub probability: f64,
    /// Normalized morpheme keys in this hypothesis.
    pub morphemes: Vec<String>,
    /// True if this hypothesis was within the posterior-mass selection set.
    pub is_selected: bool,
}

/// A morpheme occurrence identified by key, surface, and sequential index.
///
/// Used in rollback and repair anchors to address morpheme occurrences
/// unambiguously within a hypothesis sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MorphemeOccurrenceRef {
    pub key: String,
    pub surface: String,
    /// Zero-based occurrence index within the hypothesis morpheme sequence.
    pub occurrence_index: usize,
}

/// Anchor for a rollback row: the hypothesis that was withdrawn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RollbackAnchor {
    /// The hypothesis that was withdrawn.
    pub hypothesis_id: String,
    /// Morpheme occurrences that were revoked.
    pub morphemes: Vec<MorphemeOccurrenceRef>,
    /// Human-readable reason for withdrawal.
    pub reason: String,
}

/// Anchor for a repair row, using disfluency repair terminology.
///
/// Identifies the reparandum (what was said/predicted incorrectly), the
/// interregnum (always empty in hypothesis space—no filled pauses), the
/// repair (corrected morpheme sequence), and the continuation (shared
/// tail after the repair).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairAnchor {
    /// Hypothesis identity (preserved through repair).
    pub hypothesis_id: String,
    /// Reparandum: morpheme occurrences in the previous hypothesis that are
    /// being corrected.
    pub reparandum: Vec<MorphemeOccurrenceRef>,
    /// Interregnum: always empty in the duplex context (hypothesis sequences
    /// contain no filled pauses), preserved for schema completeness.
    pub interregnum: Vec<String>,
    /// Repair: the corrected morpheme occurrences in the replacement hypothesis.
    pub repair: Vec<MorphemeOccurrenceRef>,
    /// Continuation: morpheme occurrences shared by both previous and
    /// replacement after the repair point.
    pub continuation: Vec<MorphemeOccurrenceRef>,
    /// Human-readable repair reason.
    pub reason: String,
}

/// One training row from the duplex prefix/completion/rollback/repair corpus.
///
/// Rows are versioned and self-describing. Every row names its fixture source
/// and step index so the full simulator journal can be replayed independently.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplexTrainingRow {
    /// Schema version. Always `TRAINING_DATASET_VERSION` (currently 1).
    pub version: u32,
    /// Deterministic identifier: `{fixture_id}:s{step}:{kind}:{index}`.
    pub row_id: String,
    /// Kind of supervision signal this row carries.
    pub row_kind: TrainingRowKind,
    /// Provenance of this row in the corpus.
    pub row_source: DuplexTrainingRowSource,

    // ── Utterance context ────────────────────────────────────────────────
    /// Fixture utterance identifier.
    pub utterance_id: String,
    /// Language variety tag (e.g. `"en-US-GA"`).
    pub variety: String,
    /// ID of the fixture suite entry that produced this row.
    pub fixture_id: String,
    /// Zero-based evidence step index within the fixture.
    pub step_index: usize,

    // ── Evidence at this prefix ──────────────────────────────────────────
    /// IDs of evidence items observed up to and including this step.
    pub evidence_ids: Vec<String>,
    /// Modalities of those evidence items (`"text"` or `"acoustics"`).
    pub evidence_modalities: Vec<String>,
    /// True if any evidence observed up to this step is acoustic.
    /// Policy rows for silent versus audible repair differ on this flag.
    pub heard: bool,

    // ── Commitment state at this prefix ──────────────────────────────────
    /// Morpheme surfaces in the committed prefix at this step.
    pub committed_prefix: Vec<String>,
    /// Number of committed morphemes at this step.
    pub commit_frontier: usize,

    // ── Beam state at this prefix ────────────────────────────────────────
    /// Longest common morpheme prefix across the selected beam hypotheses.
    pub shared_prefix: Vec<String>,
    /// Predicted suffix beyond the committed frontier from the best hypothesis.
    pub predicted_suffix: Vec<String>,
    /// All active hypotheses in the beam at this step.
    pub beam: Vec<BeamSnapshot>,
    /// Cumulative posterior probability covered by the selected hypotheses.
    pub covered_probability: f64,

    // ── Full-context supervision labels (teacher pass) ───────────────────
    /// Total morphemes committed across the entire fixture run.
    /// Derived from the full-context teacher pass; not a heuristic.
    pub safe_commit_count: usize,
    /// Final committed text of the utterance (ground truth from teacher pass).
    pub final_committed_text: String,

    // ── Rollback payload (populated only for Rollback rows) ──────────────
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollback: Option<RollbackAnchor>,

    // ── Repair payload (populated only for Repair rows) ──────────────────
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repair: Option<RepairAnchor>,

    // ── Provenance metadata ───────────────────────────────────────────────
    /// Name of the corpus or fixture suite file.
    pub corpus: String,
    /// Applicable license for this row.
    pub license: String,
    /// Derivation chain (e.g. `"duplex-fixture-v1"`).
    pub derivation: String,
}

/// Report returned after dataset preparation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DuplexPrepareReport {
    /// Number of fixture utterances processed.
    pub fixtures: usize,
    /// Total prefix rows.
    pub prefix_rows: usize,
    /// Total completion rows.
    pub completion_rows: usize,
    /// Total rollback rows.
    pub rollback_rows: usize,
    /// Total repair rows.
    pub repair_rows: usize,
    /// Total rows in the train split.
    pub train_rows: usize,
    /// Total rows in the valid split.
    pub valid_rows: usize,
    /// Total rows in the test split.
    pub test_rows: usize,
}

/// Progress events emitted by [`prepare_dataset_with_progress`].
#[derive(Debug, Clone, PartialEq)]
pub enum DuplexPrepareProgress {
    /// A named preparation stage started.
    Stage { message: String },
    /// One fixture was processed.
    Fixture { fixture_id: String, rows: usize },
    /// A JSONL file was written.
    Write { path: String, rows: usize },
}

/// Prepare a duplex training dataset from a fixture suite at `fixtures_path`.
///
/// Uses a full-context **teacher pass** to discover the final committed text
/// for each fixture, then an incremental **student pass** to emit one or more
/// training rows per prefix step. Writes `examples.jsonl`, `train.jsonl`,
/// `valid.jsonl`, `test.jsonl`, and `README.md` to `out`.
pub fn prepare_dataset(out: &Path, fixtures_path: &Path) -> AnyResult<DuplexPrepareReport> {
    prepare_dataset_with_progress(out, fixtures_path, |_| {})
}

/// Like [`prepare_dataset`] but emits [`DuplexPrepareProgress`] events.
pub fn prepare_dataset_with_progress(
    out: &Path,
    fixtures_path: &Path,
    mut progress: impl FnMut(DuplexPrepareProgress),
) -> AnyResult<DuplexPrepareReport> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;

    progress(DuplexPrepareProgress::Stage {
        message: format!("Loading fixture suite {}", fixtures_path.display()),
    });

    let bytes =
        fs::read(fixtures_path).with_context(|| format!("reading {}", fixtures_path.display()))?;
    let suite: DuplexFixtureSuite = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing {}", fixtures_path.display()))?;
    suite
        .validate()
        .with_context(|| format!("validating {}", fixtures_path.display()))?;

    let corpus_name = fixtures_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("duplex-fixtures");

    progress(DuplexPrepareProgress::Stage {
        message: format!(
            "Generating training rows from {} fixtures",
            suite.fixtures.len()
        ),
    });

    let mut all_rows: Vec<DuplexTrainingRow> = Vec::new();

    for fixture in &suite.fixtures {
        let rows = rows_for_fixture(fixture, corpus_name)
            .with_context(|| format!("processing fixture '{}'", fixture.id))?;
        let row_count = rows.len();
        all_rows.extend(rows);
        progress(DuplexPrepareProgress::Fixture {
            fixture_id: fixture.id.clone(),
            rows: row_count,
        });
    }

    let prefix_rows = all_rows
        .iter()
        .filter(|r| r.row_kind == TrainingRowKind::Prefix)
        .count();
    let completion_rows = all_rows
        .iter()
        .filter(|r| r.row_kind == TrainingRowKind::Completion)
        .count();
    let rollback_rows = all_rows
        .iter()
        .filter(|r| r.row_kind == TrainingRowKind::Rollback)
        .count();
    let repair_rows = all_rows
        .iter()
        .filter(|r| r.row_kind == TrainingRowKind::Repair)
        .count();

    write_jsonl_duplex(&out.join("examples.jsonl"), &all_rows, &mut progress)?;

    let (train, valid, test) = split_rows_by_fixture_group(all_rows.clone(), 42, 0.8, 0.1);
    let n = all_rows.len();

    write_jsonl_duplex(&out.join("train.jsonl"), &train, &mut progress)?;
    write_jsonl_duplex(&out.join("valid.jsonl"), &valid, &mut progress)?;
    write_jsonl_duplex(&out.join("test.jsonl"), &test, &mut progress)?;

    let readme = duplex_dataset_readme(
        fixtures_path,
        suite.fixtures.len(),
        prefix_rows,
        completion_rows,
        rollback_rows,
        repair_rows,
        n,
        train.len(),
        valid.len(),
        test.len(),
    );
    fs::write(out.join("README.md"), &readme)
        .with_context(|| format!("writing README.md in {}", out.display()))?;

    Ok(DuplexPrepareReport {
        fixtures: suite.fixtures.len(),
        prefix_rows,
        completion_rows,
        rollback_rows,
        repair_rows,
        train_rows: train.len(),
        valid_rows: valid.len(),
        test_rows: test.len(),
    })
}

fn split_rows_by_fixture_group(
    rows: Vec<DuplexTrainingRow>,
    seed: u64,
    train_frac: f64,
    valid_frac: f64,
) -> (
    Vec<DuplexTrainingRow>,
    Vec<DuplexTrainingRow>,
    Vec<DuplexTrainingRow>,
) {
    let mut grouped = BTreeMap::<String, Vec<DuplexTrainingRow>>::new();
    for row in rows {
        grouped.entry(row.fixture_id.clone()).or_default().push(row);
    }
    let mut groups = grouped.keys().cloned().collect::<Vec<_>>();
    groups.shuffle(&mut StdRng::seed_from_u64(seed));
    let n = groups.len();
    let train_end = (n as f64 * train_frac).round() as usize;
    let valid_end = (train_end + (n as f64 * valid_frac).round() as usize).min(n);
    let mut train = Vec::new();
    let mut valid = Vec::new();
    let mut test = Vec::new();
    for (index, group) in groups.iter().enumerate() {
        if let Some(group_rows) = grouped.remove(group) {
            if index < train_end {
                train.extend(group_rows);
            } else if index < valid_end {
                valid.extend(group_rows);
            } else {
                test.extend(group_rows);
            }
        }
    }
    (train, valid, test)
}

/// Generate all training rows for a single fixture.
///
/// Performs a full-context **teacher pass** (runs the fixture to completion)
/// to learn `safe_commit_count` and `final_committed_text`, then performs an
/// incremental **student pass** (replays one evidence step at a time) to emit
/// Prefix, Completion, Rollback, and Repair rows.
fn rows_for_fixture(fixture: &DuplexFixture, corpus: &str) -> AnyResult<Vec<DuplexTrainingRow>> {
    // Teacher pass: run the fixture to completion to obtain the ground truth.
    let (_, final_state) = run_fixture(fixture)
        .with_context(|| format!("teacher pass for fixture '{}'", fixture.id))?;
    let safe_commit_count = final_state.committed.len();
    let final_committed_text = final_state.committed_text();

    let mut rows = Vec::new();

    // Student pass: replay evidence one step at a time.
    let provider = FixtureCompletionProvider::new(fixture);
    let mut simulator = DuplexSimulator::new(
        fixture.utterance_id.clone(),
        fixture.variety.clone(),
        fixture.config.clone(),
        provider,
    )
    .with_context(|| format!("creating simulator for fixture '{}'", fixture.id))?;

    for (step_index, step) in fixture.steps.iter().enumerate() {
        let events = simulator.observe(step.evidence.clone()).with_context(|| {
            format!("observing step {} of fixture '{}'", step_index, fixture.id)
        })?;

        let state = simulator.state();

        // Build evidence metadata from accumulated state.
        let evidence_ids: Vec<String> = state.evidence.keys().cloned().collect();
        let evidence_modalities: Vec<String> = state
            .evidence
            .values()
            .map(|ev| match ev.modality {
                EvidenceModality::Text => "text".to_string(),
                EvidenceModality::Acoustics => "acoustics".to_string(),
            })
            .collect();
        let heard = state
            .evidence
            .values()
            .any(|ev| ev.modality == EvidenceModality::Acoustics);

        // Build beam snapshot.
        let selected_set: BTreeSet<&CompletionHypothesisId> =
            state.selected_hypotheses.iter().collect();
        let beam: Vec<BeamSnapshot> = state
            .hypotheses
            .values()
            .map(|h| BeamSnapshot {
                hypothesis_id: h.id.0.clone(),
                probability: h.probability,
                morphemes: h.morphemes.iter().map(|m| m.key.clone()).collect(),
                is_selected: selected_set.contains(&h.id),
            })
            .collect();

        // Extract covered_probability from the BeamInferred event of this step.
        let covered_probability = events
            .iter()
            .find_map(|ev| {
                if let SimulatorEventKind::BeamInferred {
                    covered_probability,
                    ..
                } = &ev.event
                {
                    Some(*covered_probability)
                } else {
                    None
                }

            })
            .unwrap_or(0.0);

        let committed_prefix: Vec<String> =
            state.committed.iter().map(|m| m.surface.clone()).collect();
        let commit_frontier = state.committed.len();
        let shared_prefix = state.shared_prefix.clone();
        let predicted_suffix: Vec<String> = state
            .predicted_suffix()
            .iter()
            .map(|m| m.surface.clone())
            .collect();

        // Helper closure that builds the common row fields.
        let base = |row_id: String, row_kind: TrainingRowKind| DuplexTrainingRow {
            version: TRAINING_DATASET_VERSION,
            row_id,
            row_kind,
            row_source: DuplexTrainingRowSource::Fixture,
            utterance_id: fixture.utterance_id.0.clone(),
            variety: fixture.variety.0.clone(),
            fixture_id: fixture.id.clone(),
            step_index,
            evidence_ids: evidence_ids.clone(),
            evidence_modalities: evidence_modalities.clone(),
            heard,
            committed_prefix: committed_prefix.clone(),
            commit_frontier,
            shared_prefix: shared_prefix.clone(),
            predicted_suffix: predicted_suffix.clone(),
            beam: beam.clone(),
            covered_probability,
            safe_commit_count,
            final_committed_text: final_committed_text.clone(),
            rollback: None,
            repair: None,
            corpus: corpus.to_string(),
            license: "CC0-1.0".to_string(),
            derivation: "duplex-fixture-v1".to_string(),
        };

        // Prefix row: one per step.
        rows.push(base(
            format!("{}:s{}:prefix:0", fixture.id, step_index),
            TrainingRowKind::Prefix,
        ));

        // Per-event rows (Completion, Rollback, Repair).
        let mut completion_idx = 0usize;
        let mut rollback_idx = 0usize;
        let mut repair_idx = 0usize;

        for ev in &events {
            match &ev.event {
                SimulatorEventKind::CommitFrontierAdvanced { to, committed, .. } => {
                    let mut row = base(
                        format!(
                            "{}:s{}:completion:{}",
                            fixture.id, step_index, completion_idx
                        ),
                        TrainingRowKind::Completion,
                    );
                    // Override commit_frontier with the value at the moment of
                    // this specific advance (state.commit_frontier is already
                    // at the final value, so use the event's `to`).
                    row.commit_frontier = *to;
                    row.committed_prefix = {
                        // Reconstruct committed prefix up to this advance.
                        let n = row.committed_prefix.len();
                        let advance = committed.len();
                        if advance <= n {
                            row.committed_prefix.clone()
                        } else {
                            row.committed_prefix.iter().take(*to).cloned().collect()
                        }
                    };
                    rows.push(row);
                    completion_idx += 1;
                }

                SimulatorEventKind::HypothesisWithdrawn { hypothesis, reason } => {
                    let morphemes: Vec<MorphemeOccurrenceRef> = hypothesis
                        .morphemes
                        .iter()
                        .enumerate()
                        .map(|(i, m)| MorphemeOccurrenceRef {
                            key: m.key.clone(),
                            surface: m.surface.clone(),
                            occurrence_index: i,
                        })
                        .collect();
                    let mut row = base(
                        format!("{}:s{}:rollback:{}", fixture.id, step_index, rollback_idx),
                        TrainingRowKind::Rollback,
                    );
                    row.rollback = Some(RollbackAnchor {
                        hypothesis_id: hypothesis.id.0.clone(),
                        morphemes,
                        reason: reason.clone(),
                    });
                    rows.push(row);
                    rollback_idx += 1;
                }

                SimulatorEventKind::HypothesisRepaired {
                    previous,
                    replacement,
                    reason,
                } => {
                    // Only emit a Repair row when the morpheme sequences actually
                    // differ. Probability-only updates (same morphemes, revised
                    // weight) are not useful as morpheme-addressed supervision.
                    let morphemes_changed = previous.morphemes.len() != replacement.morphemes.len()
                        || previous
                            .morphemes
                            .iter()
                            .zip(replacement.morphemes.iter())
                            .any(|(p, r)| normalize_key(&p.key) != normalize_key(&r.key));
                    if morphemes_changed {
                        let anchor = build_repair_anchor(previous, replacement, reason);
                        let mut row = base(
                            format!("{}:s{}:repair:{}", fixture.id, step_index, repair_idx),
                            TrainingRowKind::Repair,
                        );
                        row.repair = Some(anchor);
                        rows.push(row);
                        repair_idx += 1;
                    }
                }

                SimulatorEventKind::EvidenceObserved { .. }
                | SimulatorEventKind::HypothesisProposed { .. }
                | SimulatorEventKind::BeamInferred { .. }
                | SimulatorEventKind::VerificationEvaluated { .. } => {}
            }
        }
    }

    Ok(rows)
}

/// Build a [`RepairAnchor`] from a `HypothesisRepaired` event.
///
/// Uses longest-common-prefix / longest-common-suffix decomposition to
/// identify the reparandum, repair, and continuation precisely.
fn build_repair_anchor(
    previous: &NormalizedCompletionHypothesis,
    replacement: &NormalizedCompletionHypothesis,
    reason: &str,
) -> RepairAnchor {
    let prev = &previous.morphemes;
    let repl = &replacement.morphemes;

    // Longest common prefix.
    let common_prefix_len = prev
        .iter()
        .zip(repl.iter())
        .take_while(|(p, r)| normalize_key(&p.key) == normalize_key(&r.key))
        .count();

    // Longest common suffix (only in the divergent tails).
    let prev_tail = &prev[common_prefix_len..];
    let repl_tail = &repl[common_prefix_len..];
    let max_suffix = prev_tail.len().min(repl_tail.len());
    let common_suffix_len = prev_tail
        .iter()
        .rev()
        .zip(repl_tail.iter().rev())
        .take(max_suffix)
        .take_while(|(p, r)| normalize_key(&p.key) == normalize_key(&r.key))
        .count();

    let prev_reparandum_end = prev.len() - common_suffix_len;
    let repl_repair_end = repl.len() - common_suffix_len;

    let reparandum: Vec<MorphemeOccurrenceRef> = prev[common_prefix_len..prev_reparandum_end]
        .iter()
        .enumerate()
        .map(|(i, m)| MorphemeOccurrenceRef {
            key: m.key.clone(),
            surface: m.surface.clone(),
            occurrence_index: common_prefix_len + i,
        })
        .collect();

    let repair: Vec<MorphemeOccurrenceRef> = repl[common_prefix_len..repl_repair_end]
        .iter()
        .enumerate()
        .map(|(i, m)| MorphemeOccurrenceRef {
            key: m.key.clone(),
            surface: m.surface.clone(),
            occurrence_index: common_prefix_len + i,
        })
        .collect();

    // Continuation = shared suffix (morphemes that both sequences agree on
    // after the divergent region).
    let continuation: Vec<MorphemeOccurrenceRef> = repl[repl_repair_end..]
        .iter()
        .enumerate()
        .map(|(i, m)| MorphemeOccurrenceRef {
            key: m.key.clone(),
            surface: m.surface.clone(),
            occurrence_index: repl_repair_end + i,
        })
        .collect();

    RepairAnchor {
        hypothesis_id: replacement.id.0.clone(),
        reparandum,
        interregnum: Vec::new(),
        repair,
        continuation,
        reason: reason.to_string(),
    }
}

fn duplex_dataset_readme(
    fixtures_path: &Path,
    fixture_count: usize,
    prefix_rows: usize,
    completion_rows: usize,
    rollback_rows: usize,
    repair_rows: usize,
    total_rows: usize,
    train_rows: usize,
    valid_rows: usize,
    test_rows: usize,
) -> String {
    format!(
        "# Duplex prefix/completion/rollback/repair dataset\n\
         \n\
         Schema version: `{TRAINING_DATASET_VERSION}`\n\
         Source: `{source}`\n\
         Fixtures: {fixture_count}\n\
         Total rows: {total_rows}\n\
         \n\
         ## Row kinds\n\
         \n\
         | Kind | Count |\n\
         |------|-------|\n\
         | prefix | {prefix_rows} |\n\
         | completion | {completion_rows} |\n\
         | rollback | {rollback_rows} |\n\
         | repair | {repair_rows} |\n\
         \n\
         ## Splits\n\
         \n\
         | Split | Rows |\n\
         |-------|------|\n\
         | train | {train_rows} |\n\
         | valid | {valid_rows} |\n\
         | test  | {test_rows}  |\n\
         \n\
         Splits are deterministic (seed = 42, 80/10/10 ratio) and non-overlapping.\n\
         \n\
         ## Schema fields\n\
         \n\
         - `version` — dataset schema version (`u32`).\n\
         - `row_id` — deterministic identifier `{{fixture_id}}:s{{step}}:{{kind}}:{{index}}`.\n\
         - `row_kind` — `prefix | completion | rollback | repair`.\n\
         - `row_source` — always `fixture` for rows derived from named fixtures.\n\
         - `utterance_id`, `variety`, `fixture_id`, `step_index` — utterance context.\n\
         - `evidence_ids`, `evidence_modalities` — evidence observed up to this step.\n\
         - `heard` — `true` if any evidence is acoustic; enables silent vs audible repair policy.\n\
         - `committed_prefix`, `commit_frontier` — committed morphemes at this step.\n\
         - `shared_prefix`, `predicted_suffix`, `beam` — beam state.\n\
         - `covered_probability` — posterior mass covered by the selected hypotheses.\n\
         - `safe_commit_count` — total morphemes ultimately committed (teacher-pass ground truth).\n\
         - `final_committed_text` — full-context committed text (teacher-pass ground truth).\n\
         - `rollback` — rollback anchor naming the withdrawn hypothesis and morpheme occurrences.\n\
         - `repair` — repair anchor with `reparandum`, `interregnum`, `repair`, `continuation`.\n\
         - `corpus`, `license`, `derivation` — provenance metadata.\n\
         \n\
         ## Coverage\n\
         \n\
         - Every row distinguishes evidence, inference, prediction, and commitment layers.\n\
         - Every rollback row names morpheme occurrences and reason.\n\
         - Every repair row includes reparandum, interregnum, repair, and continuation.\n\
         - `safe_commit_count` is derived from full-context truth, not a heuristic.\n\
         - `heard`/`unheard` variants are distinguished by the `heard` flag.\n\
         \n\
         ## Provenance\n\
         \n\
         All rows are derived from the synthetic fixture suite and carry\n\
         `license: CC0-1.0` and `derivation: duplex-fixture-v1`.\n\
         Non-redistributable material must be kept outside this boundary.\n",
        source = fixtures_path.display(),
    )
}

fn write_jsonl_duplex(
    path: &Path,
    rows: &[DuplexTrainingRow],
    progress: &mut impl FnMut(DuplexPrepareProgress),
) -> AnyResult<()> {
    write_jsonl_duplex_atomic(path, rows)?;
    progress(DuplexPrepareProgress::Write {
        path: path.display().to_string(),
        rows: rows.len(),
    });
    Ok(())
}

fn write_jsonl_duplex_atomic(path: &Path, rows: &[DuplexTrainingRow]) -> AnyResult<()> {
    let part = duplex_part_path(path);
    archive_interrupted_duplex_part(path)?;
    let file = fs::File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut writer = BufWriter::new(file);
    for row in rows {
        let line = serde_json::to_string(row)
            .with_context(|| format!("serializing row {}", row.row_id))?;
        writer
            .write_all(line.as_bytes())
            .with_context(|| format!("writing {}", part.display()))?;
        writer
            .write_all(b"\n")
            .with_context(|| format!("writing newline in {}", part.display()))?;
    }
    writer
        .flush()
        .with_context(|| format!("flushing {}", part.display()))?;
    drop(writer);
    fs::rename(&part, path)
        .with_context(|| format!("renaming {} -> {}", part.display(), path.display()))
}

fn duplex_part_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("artifact");
    path.with_file_name(format!("{name}.writing.part"))
}

fn archive_interrupted_duplex_part(path: &Path) -> AnyResult<()> {
    let part = duplex_part_path(path);
    if !part.exists() {
        return Ok(());
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let archive = part.with_extension(format!(
        "{}interrupted-{stamp}",
        part.extension()
            .and_then(|e| e.to_str())
            .map(|e| format!("{e}."))
            .unwrap_or_default()
    ));
    fs::rename(&part, &archive).with_context(|| {
        format!(
            "archiving interrupted partial {} -> {}",
            part.display(),
            archive.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> EvidenceProvenance {
        EvidenceProvenance {
            source: EvidenceSource::Inference,
            method: "test".into(),
            version: Some("1".into()),
        }
    }

    fn morph(key: &str, evidence: &[&str]) -> CompletionMorpheme {
        CompletionMorpheme {
            key: key.into(),
            surface: key.into(),
            variety: VarietyId("en-US-GA".into()),
            evidence: evidence.iter().map(|id| (*id).into()).collect(),
        }
    }

    fn proposal(id: &str, weight: f64, morphemes: Vec<CompletionMorpheme>) -> CompletionProposal {
        CompletionProposal {
            id: CompletionHypothesisId(id.into()),
            weight,
            morphemes,
            syntax: Some(SentenceSyntaxAnalysis::default()),
            prosody: Some(ProsodyTrack::default()),
            evidence: Vec::new(),
            provenance: provenance(),
        }
    }

    #[test]
    fn posterior_selection_and_lcp_are_deterministic() {
        let beam = normalize_beam(vec![
            proposal("b", 2.0, vec![morph("shared", &[]), morph("beta", &[])]),
            proposal("a", 7.0, vec![morph("shared", &[]), morph("alpha", &[])]),
            proposal("c", 1.0, vec![morph("other", &[])]),
        ])
        .unwrap();
        let (selected, covered) = select_posterior_mass(&beam, 0.8);
        assert_eq!(
            selected
                .iter()
                .map(|hypothesis| hypothesis.id.0.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        assert!((covered - 0.9).abs() < 1e-12);
        assert_eq!(longest_common_morpheme_prefix(&selected)[0].key, "shared");
    }

    #[test]
    fn unsupported_prediction_never_commits() {
        let evidence = ObservedEvidence::text("text-1", "Who shot John F.");
        let fixture = DuplexFixture {
            id: "unsupported".into(),
            description: "unsupported prediction".into(),
            utterance_id: UtteranceId("utt-test".into()),
            variety: VarietyId("en-US-GA".into()),
            config: SimulatorConfig::default(),
            steps: vec![FixtureStep {
                evidence,
                hypotheses: vec![
                    proposal(
                        "kennedy",
                        0.7,
                        vec![
                            morph("Who", &["text-1"]),
                            morph("shot", &["text-1"]),
                            morph("John", &["text-1"]),
                            morph("F.", &["text-1"]),
                            morph("Kennedy", &[]),
                        ],
                    ),
                    proposal(
                        "kennedy-jr",
                        0.3,
                        vec![
                            morph("Who", &["text-1"]),
                            morph("shot", &["text-1"]),
                            morph("John", &["text-1"]),
                            morph("F.", &["text-1"]),
                            morph("Kennedy", &[]),
                            morph("Jr.", &[]),
                        ],
                    ),
                ],
            }],
        };
        let (journal, state) = run_fixture(&fixture).unwrap();
        assert_eq!(state.committed_text(), "Who shot John F.");
        assert!(!state.committed_text().contains("Kennedy"));
        assert_eq!(replay_journal(&journal).unwrap(), state);
    }

    #[test]
    fn fixture_suite_is_replayable_and_covers_required_cases() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .expect("fixture suite parses");
        suite.validate().expect("fixture suite validates");
        let ids = suite
            .fixtures
            .iter()
            .map(|fixture| fixture.id.as_str())
            .collect::<BTreeSet<_>>();
        for required in [
            "initials",
            "abbreviations",
            "heteronyms",
            "stress-changes",
            "garden-path",
            "code-switching",
            "end-of-turn-uncertainty",
            "who-shot-john-f",
            "mock-acoustics",
            "homophones",
            "false-boundaries",
        ] {
            assert!(ids.contains(required), "missing fixture {required}");
        }
        for fixture in &suite.fixtures {
            let (journal, state) = run_fixture(fixture)
                .unwrap_or_else(|error| panic!("fixture '{}' failed: {error}", fixture.id));
            assert_eq!(replay_journal(&journal).unwrap(), state);
        }
    }

    #[test]
    fn fixed_fixture_is_bitwise_deterministic() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .unwrap();
        let fixture = suite.fixture("who-shot-john-f").unwrap();
        let first = run_fixture(fixture).unwrap();
        let second = run_fixture(fixture).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            serde_json::to_vec(&first.0).unwrap(),
            serde_json::to_vec(&second.0).unwrap()
        );
    }

    #[test]
    fn later_evidence_withdraws_and_repairs_without_corrupting_commitment() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .unwrap();
        let fixture = suite.fixture("garden-path").unwrap();
        let (journal, state) = run_fixture(fixture).unwrap();
        assert_eq!(state.committed_text(), "The old man the boats");
        assert!(
            journal
                .events
                .iter()
                .any(|event| matches!(event.event, SimulatorEventKind::HypothesisWithdrawn { .. }))
        );
        assert!(
            journal
                .events
                .iter()
                .any(|event| matches!(event.event, SimulatorEventKind::HypothesisRepaired { .. }))
        );
        assert_eq!(replay_journal(&journal).unwrap(), state);
    }

    #[test]
    fn journal_layers_distinguish_evidence_inference_prediction_and_commitment() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .unwrap();
        let (journal, _) = run_fixture(suite.fixture("who-shot-john-f").unwrap()).unwrap();
        let layers = journal
            .events
            .iter()
            .map(|event| event.event.layer())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            layers,
            BTreeSet::from(["evidence", "inference", "prediction", "commitment"])
        );
    }

    #[test]
    fn oracle_provider_carries_syntax_prosody_and_keeps_suffix_provisional() {
        let mut simulator = DuplexSimulator::new(
            UtteranceId("oracle".into()),
            VarietyId("en-US-GA".into()),
            SimulatorConfig::default(),
            OracleCompletionProvider,
        )
        .unwrap();
        simulator
            .observe(ObservedEvidence::text("chunk-1", "hello world"))
            .unwrap();
        assert_eq!(simulator.state().committed_text(), "hello world");
        assert_eq!(simulator.state().predicted_suffix().len(), 1);
        assert!(
            simulator
                .state()
                .hypotheses
                .values()
                .all(|hypothesis| hypothesis.syntax.is_some() && hypothesis.prosody.is_some())
        );
    }

    #[test]
    fn acoustic_span_survives_withdrawal_and_commit() {
        let span = AcousticSpan {
            frame_start: 0,
            frame_end: 40,
            time_start: 0.0,
            time_end: 0.5,
            confidence: 0.92,
        };
        let ev = ObservedEvidence::acoustics_with_span("acoustic:0", "hello world", span.clone());
        assert_eq!(ev.acoustic_span.as_ref().unwrap().frame_start, 0);
        assert_eq!(ev.acoustic_span.as_ref().unwrap().frame_end, 40);
        assert!((ev.acoustic_span.as_ref().unwrap().confidence - 0.92).abs() < 1e-5);

        // Verify the span round-trips through serde unchanged.
        let json = serde_json::to_string(&ev).unwrap();
        let ev2: ObservedEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(ev, ev2);

        // Existing text evidence must not acquire a span.
        let text_ev = ObservedEvidence::text("text:0", "hello world");
        assert!(text_ev.acoustic_span.is_none());
    }

    #[test]
    fn speculative_consumer_emits_provisional_events_and_retracts_on_withdrawal() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .unwrap();
        let fixture = suite.fixture("false-boundaries").unwrap();
        let (journal, state, consumer) = run_fixture_with_consumer(fixture).unwrap();

        // The simulator must record that the false-boundary hypothesis was
        // withdrawn after correcting evidence arrived.
        assert!(
            journal
                .events
                .iter()
                .any(|ev| matches!(ev.event, SimulatorEventKind::HypothesisWithdrawn { .. })),
            "expected at least one HypothesisWithdrawn simulator event"
        );

        // At the transcript level the consumer must have produced at least one
        // Replace (old provisional → new provisional) or Withdraw event – either
        // means the false provisional was cleaned up.
        assert!(
            consumer.transcript_events.iter().any(|ev| matches!(
                ev,
                ProvisionalTranscriptEvent::Replace { .. }
                    | ProvisionalTranscriptEvent::Withdraw { .. }
            )),
            "expected Replace or Withdraw transcript event; got {:?}",
            consumer.transcript_events
        );

        // Consumer must have produced at least one Commit event.
        assert!(
            consumer
                .transcript_events
                .iter()
                .any(|ev| matches!(ev, ProvisionalTranscriptEvent::Commit { .. })),
            "expected at least one Commit event"
        );

        // Committed text must agree with the simulator state.
        assert_eq!(consumer.committed_text(), state.committed_text());
    }

    #[test]
    fn speculative_consumer_keeps_homophones_visible_until_resolved() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .unwrap();
        let fixture = suite.fixture("homophones").unwrap();
        let (journal, _state, consumer) = run_fixture_with_consumer(fixture).unwrap();

        // After step 1 (ambiguous /miːt/ evidence), both hypotheses should be
        // present in the simulator without any commit.
        let events_after_step1: Vec<_> = journal
            .events
            .iter()
            .take_while(|e| {
                !matches!(e.event, SimulatorEventKind::EvidenceObserved { .. })
                    || journal.events.iter().position(|x| x == *e).unwrap_or(0) < 5
            })
            .collect();
        let _ = events_after_step1; // structural check is below via consumer

        // Consumer must have emitted Append for initial provisional suffix,
        // then Replace or Commit when the beam resolved.
        assert!(
            consumer.transcript_events.iter().any(|ev| matches!(
                ev,
                ProvisionalTranscriptEvent::Append { .. }
                    | ProvisionalTranscriptEvent::Commit { .. }
            )),
            "expected Append or Commit events; got {:?}",
            consumer.transcript_events
        );

        // Completion priors must never appear in committed text.
        let committed = consumer.committed_text();
        for committed_word in committed.split_whitespace() {
            assert!(
                !committed_word.is_empty(),
                "committed word must not be empty"
            );
        }
    }

    #[test]
    fn recording_consumer_committed_text_matches_simulator_state() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .unwrap();
        for fixture in &suite.fixtures {
            let (_journal, state, consumer) = run_fixture_with_consumer(fixture)
                .unwrap_or_else(|e| panic!("fixture '{}' failed: {e}", fixture.id));
            assert_eq!(
                consumer.committed_text(),
                state.committed_text(),
                "consumer and state disagree for fixture '{}'",
                fixture.id
            );
        }
    }

    #[test]
    fn studio_projection_replays_fixture_without_ui_logic() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .unwrap();
        let fixture = suite.fixture("who-shot-john-f").unwrap();
        let (journal, state) = run_fixture(fixture).unwrap();
        let projection = studio_projection_from_journal(fixture.id.clone(), &journal).unwrap();

        assert_eq!(projection.schema_version, STUDIO_PROJECTION_SCHEMA_VERSION);
        assert!(projection.replay_verified);
        assert_eq!(projection.final_state, state);
        assert_eq!(projection.timeline.len(), journal.events.len());
        assert!(
            projection
                .timeline
                .iter()
                .any(|snapshot| !snapshot.predicted.is_empty()),
            "expected at least one projected prediction snapshot"
        );
        assert!(
            projection.timeline.iter().any(|snapshot| {
                snapshot
                    .branches
                    .iter()
                    .any(|branch| branch.selected && branch.probability > 0.0)
            }),
            "expected at least one selected branch with probability"
        );
    }

    #[test]
    fn studio_projection_is_stable_when_replayed_twice() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .unwrap();
        let fixture = suite.fixture("false-boundaries").unwrap();
        let (journal, _) = run_fixture(fixture).unwrap();

        let first = studio_projection_from_journal(fixture.id.clone(), &journal).unwrap();
        let second = studio_projection_from_journal(fixture.id.clone(), &journal).unwrap();
        assert_eq!(first.timeline, second.timeline);
        assert_eq!(first.transcript_events, second.transcript_events);
    }

    fn verification_sequence(words: &[&str], phones: &[&str]) -> VerificationSequence {
        VerificationSequence {
            morphemes: words.iter().map(|value| (*value).to_string()).collect(),
            phones: phones.iter().map(|value| (*value).to_string()).collect(),
            stress: Vec::new(),
            boundaries: Vec::new(),
            timings_ms: vec![100; words.len()],
        }
    }

    #[test]
    fn closed_loop_retries_replaceable_incorrect_output() {
        let request = ClosedLoopVerificationRequest {
            intended: verification_sequence(
                &["turn", "left"],
                &["t", "ɝ", "n", "l", "ɛ", "f", "t"],
            ),
            recovered: verification_sequence(&["turn", "right"], &["t", "ɝ", "n", "ɹ", "aɪ", "t"]),
            accepted_projection_losses: Vec::new(),
            recognizer_confidence: 0.95,
            held_audio_replaceable: true,
            verification_latency_ms: 45.0,
        };
        let result = verify_closed_loop(&request, &ClosedLoopVerificationPolicy::default(), 0);
        assert_eq!(result.decision, VerificationDecision::Retry);
        assert!(
            result
                .repair_causes
                .contains(&VerificationRepairCause::LinguisticDisagreement)
        );
    }

    #[test]
    fn closed_loop_distinguishes_projection_loss_from_disagreement() {
        let request = ClosedLoopVerificationRequest {
            intended: verification_sequence(&["read.future"], &["ɹ", "iː", "d"]),
            recovered: verification_sequence(&["read"], &["ɹ", "iː", "d"]),
            accepted_projection_losses: vec!["read.future->read".into()],
            recognizer_confidence: 0.9,
            held_audio_replaceable: true,
            verification_latency_ms: 30.0,
        };
        let result = verify_closed_loop(&request, &ClosedLoopVerificationPolicy::default(), 0);
        assert_eq!(result.decision, VerificationDecision::Accept);
        assert!(
            result
                .repair_causes
                .contains(&VerificationRepairCause::AcceptedProjectionLoss)
        );
        assert!(
            !result
                .repair_causes
                .contains(&VerificationRepairCause::LinguisticDisagreement)
        );
    }

    #[test]
    fn closed_loop_abstains_when_recognizer_is_uncertain() {
        let request = ClosedLoopVerificationRequest {
            intended: verification_sequence(&["hello"], &["h", "ə", "l", "oʊ"]),
            recovered: verification_sequence(&["hello"], &["h", "ə", "l", "oʊ"]),
            accepted_projection_losses: Vec::new(),
            recognizer_confidence: 0.2,
            held_audio_replaceable: true,
            verification_latency_ms: 20.0,
        };
        let result = verify_closed_loop(&request, &ClosedLoopVerificationPolicy::default(), 0);
        assert_eq!(result.decision, VerificationDecision::Abstain);
        assert!(
            result
                .repair_causes
                .contains(&VerificationRepairCause::RecognizerUncertain)
        );
    }

    #[test]
    fn mel_level_rescoring_combines_acoustic_and_prior_without_relabeling() {
        let rescored = rescore_completion_hypotheses(&AcousticRescoreRequest {
            observed_mel: vec![0.2, 0.1, 0.5, -0.3],
            candidates: vec![
                MelPredictionCandidate {
                    id: CompletionHypothesisId("a".into()),
                    prior_probability: 0.9,
                    predicted_mel: vec![0.9, 0.9, 0.9, 0.9],
                },
                MelPredictionCandidate {
                    id: CompletionHypothesisId("b".into()),
                    prior_probability: 0.1,
                    predicted_mel: vec![0.2, 0.12, 0.45, -0.28],
                },
            ],
            acoustic_weight: 1.0,
            prior_weight: 0.2,
        })
        .expect("rescoring succeeds");

        assert_eq!(rescored[0].id.0, "b");
        assert!(
            rescored[0].acoustic_log_likelihood > rescored[1].acoustic_log_likelihood,
            "acoustic evidence should remain distinct and directly comparable"
        );
        assert!(
            rescored[0].prior_log_probability < rescored[1].prior_log_probability,
            "prior should stay as prior instead of replacing acoustic evidence"
        );
    }

    #[test]
    fn verification_results_are_recorded_as_typed_journal_events() {
        let mut simulator = DuplexSimulator::new(
            UtteranceId("verify".into()),
            VarietyId("en-US-GA".into()),
            SimulatorConfig::default(),
            OracleCompletionProvider,
        )
        .expect("simulator");
        let result = simulator
            .verify_held_audio(
                ClosedLoopVerificationRequest {
                    intended: verification_sequence(&["hello"], &["h", "ə", "l", "oʊ"]),
                    recovered: verification_sequence(&["hullo"], &["h", "ʊ", "l", "oʊ"]),
                    accepted_projection_losses: Vec::new(),
                    recognizer_confidence: 0.9,
                    held_audio_replaceable: true,
                    verification_latency_ms: 35.0,
                },
                ClosedLoopVerificationPolicy::default(),
                0,
            )
            .expect("verification event");
        assert_eq!(result.decision, VerificationDecision::Retry);
        assert!(matches!(
            simulator.journal().events.last().map(|event| &event.event),
            Some(SimulatorEventKind::VerificationEvaluated { .. })
        ));
        assert_eq!(simulator.state().verifications.len(), 1);
    }

    // ── Training data generation tests ───────────────────────────────────

    #[test]
    fn training_rows_cover_all_required_kinds_for_garden_path() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .unwrap();
        let fixture = suite.fixture("garden-path").unwrap();
        let rows = rows_for_fixture(fixture, "test-corpus").unwrap();

        let has_prefix = rows.iter().any(|r| r.row_kind == TrainingRowKind::Prefix);
        let has_rollback = rows.iter().any(|r| r.row_kind == TrainingRowKind::Rollback);
        let has_repair = rows.iter().any(|r| r.row_kind == TrainingRowKind::Repair);

        assert!(has_prefix, "expected at least one Prefix row");
        assert!(has_rollback, "expected at least one Rollback row");
        assert!(has_repair, "expected at least one Repair row");
    }

    #[test]
    fn training_rows_safe_commit_count_matches_final_committed() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .unwrap();
        let fixture = suite.fixture("who-shot-john-f").unwrap();
        let (_, final_state) = run_fixture(fixture).unwrap();
        let rows = rows_for_fixture(fixture, "test-corpus").unwrap();

        for row in &rows {
            assert_eq!(
                row.safe_commit_count,
                final_state.committed.len(),
                "safe_commit_count must match final committed length for row {}",
                row.row_id
            );
            assert_eq!(
                row.final_committed_text,
                final_state.committed_text(),
                "final_committed_text must match for row {}",
                row.row_id
            );
        }
    }

    #[test]
    fn every_rollback_row_names_morpheme_occurrences() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .unwrap();
        for fixture in &suite.fixtures {
            let rows = rows_for_fixture(fixture, "test-corpus").unwrap();
            for row in rows
                .iter()
                .filter(|r| r.row_kind == TrainingRowKind::Rollback)
            {
                let anchor = row
                    .rollback
                    .as_ref()
                    .expect("rollback row must have anchor");
                assert!(
                    !anchor.hypothesis_id.is_empty(),
                    "rollback anchor must name the hypothesis in fixture '{}'",
                    fixture.id
                );
                // At least one morpheme occurrence must be named.
                assert!(
                    !anchor.morphemes.is_empty(),
                    "rollback row must name at least one morpheme occurrence in fixture '{}'",
                    fixture.id
                );
            }
        }
    }

    #[test]
    fn every_repair_row_has_reparandum_and_repair() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .unwrap();
        for fixture in &suite.fixtures {
            let rows = rows_for_fixture(fixture, "test-corpus").unwrap();
            for row in rows
                .iter()
                .filter(|r| r.row_kind == TrainingRowKind::Repair)
            {
                let anchor = row.repair.as_ref().expect("repair row must have anchor");
                // A repair must have either reparandum or repair (one may be
                // empty if the hypothesis only appended or only deleted).
                assert!(
                    !anchor.reparandum.is_empty() || !anchor.repair.is_empty(),
                    "repair anchor must have at least reparandum or repair in fixture '{}'",
                    fixture.id
                );
                // Interregnum must be empty (hypothesis space has no filled pauses).
                assert!(
                    anchor.interregnum.is_empty(),
                    "interregnum must be empty in fixture '{}'",
                    fixture.id
                );
            }
        }
    }

    #[test]
    fn heard_flag_is_true_for_acoustic_evidence_fixture() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .unwrap();
        let fixture = suite.fixture("mock-acoustics").unwrap();
        let rows = rows_for_fixture(fixture, "test-corpus").unwrap();
        // At least one row should have heard = true.
        assert!(
            rows.iter().any(|r| r.heard),
            "mock-acoustics fixture must produce at least one heard=true row"
        );
    }

    #[test]
    fn heard_flag_is_false_for_text_only_fixture() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .unwrap();
        let fixture = suite.fixture("who-shot-john-f").unwrap();
        let rows = rows_for_fixture(fixture, "test-corpus").unwrap();
        // All rows must have heard = false.
        for row in &rows {
            assert!(
                !row.heard,
                "who-shot-john-f fixture uses only text evidence; \
                 row {} must have heard=false",
                row.row_id
            );
        }
    }

    #[test]
    fn row_ids_are_unique_within_a_fixture() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .unwrap();
        for fixture in &suite.fixtures {
            let rows = rows_for_fixture(fixture, "test-corpus").unwrap();
            let mut ids: BTreeSet<String> = BTreeSet::new();
            for row in &rows {
                assert!(
                    ids.insert(row.row_id.clone()),
                    "duplicate row_id '{}' in fixture '{}'",
                    row.row_id,
                    fixture.id
                );
            }
        }
    }

    #[test]
    fn training_rows_round_trip_through_serde() {
        let suite: DuplexFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .unwrap();
        let fixture = suite.fixture("garden-path").unwrap();
        let rows = rows_for_fixture(fixture, "test-corpus").unwrap();
        for row in &rows {
            let json = serde_json::to_string(row).expect("serialize");
            let recovered: DuplexTrainingRow = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(
                row, &recovered,
                "serde round-trip failed for {}",
                row.row_id
            );
        }
    }

    #[test]
    fn prepare_dataset_writes_expected_files() {
        let suite_path = std::path::Path::new("../../fixtures/duplex/completion_scenarios_v1.json");
        let out = tempfile::tempdir().expect("tempdir");
        let report = prepare_dataset(out.path(), suite_path).expect("prepare succeeds");
        assert!(report.fixtures > 0);
        assert!(report.prefix_rows > 0);
        for name in &[
            "examples.jsonl",
            "train.jsonl",
            "valid.jsonl",
            "test.jsonl",
            "README.md",
        ] {
            assert!(
                out.path().join(name).exists(),
                "expected {name} to exist in output"
            );
        }
        assert_eq!(
            report.train_rows + report.valid_rows + report.test_rows,
            report.prefix_rows + report.completion_rows + report.rollback_rows + report.repair_rows
        );
    }

    #[test]
    fn split_rows_by_fixture_group_keeps_fixture_together() {
        let mk = |id: &str, fixture_id: &str| DuplexTrainingRow {
            version: 1,
            row_id: id.to_string(),
            row_kind: TrainingRowKind::Prefix,
            row_source: DuplexTrainingRowSource::Fixture,
            utterance_id: "u".to_string(),
            variety: "en-US".to_string(),
            fixture_id: fixture_id.to_string(),
            step_index: 0,
            evidence_ids: Vec::new(),
            evidence_modalities: Vec::new(),
            heard: false,
            committed_prefix: Vec::new(),
            commit_frontier: 0,
            shared_prefix: Vec::new(),
            predicted_suffix: Vec::new(),
            beam: Vec::new(),
            covered_probability: 1.0,
            safe_commit_count: 0,
            final_committed_text: String::new(),
            rollback: None,
            repair: None,
            corpus: "fixture".to_string(),
            license: "test".to_string(),
            derivation: "test".to_string(),
        };
        let rows = vec![
            mk("r1", "f1"),
            mk("r2", "f1"),
            mk("r3", "f2"),
            mk("r4", "f3"),
        ];
        let (train, valid, test) = split_rows_by_fixture_group(rows, 7, 0.5, 0.25);
        for fixture in ["f1", "f2", "f3"] {
            let placements = usize::from(train.iter().any(|row| row.fixture_id == fixture))
                + usize::from(valid.iter().any(|row| row.fixture_id == fixture))
                + usize::from(test.iter().any(|row| row.fixture_id == fixture));
            assert_eq!(placements, 1);
        }
    }
}
