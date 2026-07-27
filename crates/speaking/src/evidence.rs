//! Shared provenance, lifecycle-aware linguistic claims, and deterministic
//! conflict resolution.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::event::{StreamEvent, TextRange};
use crate::ids::{MorphemeId, PhoneId, PhonemeId, UtteranceId};
use crate::prosody::Stress;
use crate::segment::BoundaryKind;
use crate::syntax::{PartOfSpeech, ProsodicRole, SyntacticLinkKind};

pub const LINGUISTIC_EVIDENCE_SCHEMA_V1: u16 = 1;
pub const LINGUISTIC_RESOLUTION_POLICY_V1: &str = "priority-confidence-support-lifecycle-id-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceProvenance {
    pub source: EvidenceSource,
    pub method: String,
    pub version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSource {
    Manual,
    ManualOverride,
    UserMarkup,
    CommittedAcoustics,
    Lexicon,
    Grammar,
    Morphology,
    Prosody,
    Punctuation,
    ImportedData,
    LearnedPrediction,
    Rule,
    AcousticModel,
    ForcedAlignment,
    G2p,
    Asr,
    TtsPlan,
    Memory,
    Inference,
    Unknown,
}

/// Default source priority for linguistic conflict resolution.
///
/// Confidence is compared only after source priority, so uncalibrated scores
/// from unrelated source categories are never treated as interchangeable.
pub const fn source_default_priority(source: &EvidenceSource) -> i32 {
    match source {
        EvidenceSource::ManualOverride => 1_000,
        EvidenceSource::Manual => 950,
        EvidenceSource::UserMarkup => 900,
        EvidenceSource::CommittedAcoustics => 800,
        EvidenceSource::AcousticModel => 750,
        EvidenceSource::ForcedAlignment => 740,
        EvidenceSource::Asr => 720,
        EvidenceSource::Lexicon => 700,
        EvidenceSource::Grammar => 650,
        EvidenceSource::Morphology => 600,
        EvidenceSource::Prosody => 550,
        EvidenceSource::Punctuation => 500,
        EvidenceSource::Rule => 450,
        EvidenceSource::ImportedData => 400,
        EvidenceSource::G2p => 350,
        EvidenceSource::LearnedPrediction | EvidenceSource::Inference => 300,
        EvidenceSource::Memory => 250,
        EvidenceSource::TtsPlan => 200,
        EvidenceSource::Unknown => 0,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PronunciationSource {
    Lexicon,
    MorphologicalComposition,
    LearnedSuffix,
    GraphemeToPhoneme,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LinguisticClaimId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ClaimResolutionId(pub String);

/// Stable identity of the linguistic object a claim describes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "scope")]
pub enum LinguisticTargetScope {
    Utterance,
    TextRange,
    Token { id: String },
    Word { id: String },
    Morpheme { id: String },
    Phoneme { id: String },
    Phone { id: String },
    Boundary { id: String },
    SyntaxLink { id: String },
    Pronunciation { id: String },
    Parse { id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LinguisticTarget {
    pub utterance_id: UtteranceId,
    pub scope: LinguisticTargetScope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_range: Option<TextRange>,
}

impl LinguisticTarget {
    pub fn new(
        utterance_id: UtteranceId,
        scope: LinguisticTargetScope,
        text_range: Option<TextRange>,
    ) -> Self {
        Self {
            utterance_id,
            scope,
            text_range,
        }
    }

    pub fn text_range(utterance_id: UtteranceId, range: TextRange) -> Self {
        Self::new(utterance_id, LinguisticTargetScope::TextRange, Some(range))
    }

    pub fn word(utterance_id: UtteranceId, id: impl Into<String>, range: TextRange) -> Self {
        Self::new(
            utterance_id,
            LinguisticTargetScope::Word { id: id.into() },
            Some(range),
        )
    }

    pub fn parse(
        utterance_id: UtteranceId,
        id: impl Into<String>,
        range: Option<TextRange>,
    ) -> Self {
        Self::new(
            utterance_id,
            LinguisticTargetScope::Parse { id: id.into() },
            range,
        )
    }

    fn validate(&self) -> Result<(), LinguisticClaimError> {
        if self.utterance_id.0.trim().is_empty() {
            return Err(LinguisticClaimError::EmptyUtteranceId);
        }
        if matches!(self.scope, LinguisticTargetScope::TextRange) && self.text_range.is_none() {
            return Err(LinguisticClaimError::TextRangeTargetMissingRange);
        }
        let scoped_id = match &self.scope {
            LinguisticTargetScope::Utterance | LinguisticTargetScope::TextRange => None,
            LinguisticTargetScope::Token { id }
            | LinguisticTargetScope::Word { id }
            | LinguisticTargetScope::Morpheme { id }
            | LinguisticTargetScope::Phoneme { id }
            | LinguisticTargetScope::Phone { id }
            | LinguisticTargetScope::Boundary { id }
            | LinguisticTargetScope::SyntaxLink { id }
            | LinguisticTargetScope::Pronunciation { id }
            | LinguisticTargetScope::Parse { id } => Some(id),
        };
        if scoped_id.is_some_and(|id| id.trim().is_empty()) {
            return Err(LinguisticClaimError::EmptyTargetId);
        }
        if let Some(range) = &self.text_range {
            validate_text_range(range)?;
        }
        Ok(())
    }

    /// A transcript revision invalidates claims at or after the revised
    /// position. Claims ending inside the stable prefix retain identity.
    fn affected_by_revision(&self, revised: &TextRange) -> bool {
        if matches!(self.scope, LinguisticTargetScope::Utterance) {
            return true;
        }
        self.text_range
            .as_ref()
            .map(|range| range.end > revised.start || range.start == revised.start)
            .unwrap_or(true)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinguisticClaimKind {
    PartOfSpeech,
    DependencyLink,
    LexicalIdentity,
    Pronunciation,
    MorphologicalForm,
    PhonemeRealization,
    PhoneRealization,
    Reduction,
    ProsodicRole,
    Stress,
    Boundary,
    Parse,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "value")]
pub enum LinguisticClaimValue {
    PartOfSpeech(PartOfSpeech),
    DependencyLink {
        left: usize,
        right: usize,
        kind: SyntacticLinkKind,
    },
    LexicalIdentity {
        lexeme_id: String,
    },
    Pronunciation {
        phonemes: Vec<PhonemeId>,
    },
    MorphologicalForm {
        surface: String,
        morphemes: Vec<MorphemeId>,
    },
    PhonemeRealization {
        phoneme: PhonemeId,
    },
    PhoneRealization {
        phone: PhoneId,
    },
    Reduction {
        reduced: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        form: Option<String>,
    },
    ProsodicRole(ProsodicRole),
    Stress(Stress),
    Boundary(BoundaryKind),
    Parse {
        parse_id: String,
    },
}

impl LinguisticClaimValue {
    pub const fn kind(&self) -> LinguisticClaimKind {
        match self {
            Self::PartOfSpeech(_) => LinguisticClaimKind::PartOfSpeech,
            Self::DependencyLink { .. } => LinguisticClaimKind::DependencyLink,
            Self::LexicalIdentity { .. } => LinguisticClaimKind::LexicalIdentity,
            Self::Pronunciation { .. } => LinguisticClaimKind::Pronunciation,
            Self::MorphologicalForm { .. } => LinguisticClaimKind::MorphologicalForm,
            Self::PhonemeRealization { .. } => LinguisticClaimKind::PhonemeRealization,
            Self::PhoneRealization { .. } => LinguisticClaimKind::PhoneRealization,
            Self::Reduction { .. } => LinguisticClaimKind::Reduction,
            Self::ProsodicRole(_) => LinguisticClaimKind::ProsodicRole,
            Self::Stress(_) => LinguisticClaimKind::Stress,
            Self::Boundary(_) => LinguisticClaimKind::Boundary,
            Self::Parse { .. } => LinguisticClaimKind::Parse,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimConfidence {
    /// Normalized probability in the inclusive range [0, 1].
    pub probability: f64,
    /// Calibration set or policy. `None` means the producer normalized the
    /// score but has not supplied a measured calibration record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<String>,
}

impl ClaimConfidence {
    pub fn new(
        probability: f64,
        calibration: Option<String>,
    ) -> Result<Self, LinguisticClaimError> {
        let confidence = Self {
            probability,
            calibration,
        };
        confidence.validate()?;
        Ok(confidence)
    }

    fn validate(&self) -> Result<(), LinguisticClaimError> {
        if !self.probability.is_finite() || !(0.0..=1.0).contains(&self.probability) {
            return Err(LinguisticClaimError::InvalidConfidence(self.probability));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimRationale {
    /// Stable machine-readable reason code.
    pub code: String,
    /// Concise human-readable explanation.
    pub summary: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attributes: BTreeMap<String, String>,
}

impl ClaimRationale {
    pub fn new(code: impl Into<String>, summary: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            summary: summary.into(),
            attributes: BTreeMap::new(),
        }
    }

    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimLifecycle {
    Hypothesis,
    Stable,
    Revised,
    Invalidated,
    Committed,
}

impl ClaimLifecycle {
    pub const fn is_resolution_eligible(self) -> bool {
        matches!(self, Self::Hypothesis | Self::Stable | Self::Committed)
    }

    const fn stability_rank(self) -> u8 {
        match self {
            Self::Committed => 3,
            Self::Stable => 2,
            Self::Hypothesis => 1,
            Self::Revised | Self::Invalidated => 0,
        }
    }

    const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (
                Self::Hypothesis,
                Self::Stable | Self::Revised | Self::Invalidated | Self::Committed
            ) | (
                Self::Stable,
                Self::Revised | Self::Invalidated | Self::Committed
            )
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinguisticClaim {
    pub id: LinguisticClaimId,
    pub target: LinguisticTarget,
    pub kind: LinguisticClaimKind,
    pub value: LinguisticClaimValue,
    pub provenance: EvidenceProvenance,
    /// Explicit priority. Builders initialize this from
    /// [`source_default_priority`].
    pub source_priority: i32,
    pub confidence: ClaimConfidence,
    pub lifecycle: ClaimLifecycle,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supports: Vec<LinguisticClaimId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conflicts_with: Vec<LinguisticClaimId>,
    pub rationale: ClaimRationale,
}

impl LinguisticClaim {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: LinguisticClaimId,
        target: LinguisticTarget,
        kind: LinguisticClaimKind,
        value: LinguisticClaimValue,
        provenance: EvidenceProvenance,
        confidence: ClaimConfidence,
        rationale: ClaimRationale,
    ) -> Result<Self, LinguisticClaimError> {
        let claim = Self {
            id,
            target,
            kind,
            value,
            provenance: provenance.clone(),
            source_priority: source_default_priority(&provenance.source),
            confidence,
            lifecycle: ClaimLifecycle::Hypothesis,
            supports: Vec::new(),
            conflicts_with: Vec::new(),
            rationale,
        };
        claim.validate()?;
        Ok(claim)
    }

    fn from_producer(
        id: LinguisticClaimId,
        target: LinguisticTarget,
        value: LinguisticClaimValue,
        source: EvidenceSource,
        method: impl Into<String>,
        probability: f64,
        rationale: ClaimRationale,
    ) -> Result<Self, LinguisticClaimError> {
        Self::new(
            id,
            target,
            value.kind(),
            value,
            EvidenceProvenance {
                source,
                method: method.into(),
                version: Some("1".into()),
            },
            ClaimConfidence::new(probability, None)?,
            rationale,
        )
    }

    pub fn grammar(
        id: LinguisticClaimId,
        target: LinguisticTarget,
        value: LinguisticClaimValue,
        probability: f64,
        rationale: ClaimRationale,
    ) -> Result<Self, LinguisticClaimError> {
        Self::from_producer(
            id,
            target,
            value,
            EvidenceSource::Grammar,
            "grammar",
            probability,
            rationale,
        )
    }

    pub fn lexicon(
        id: LinguisticClaimId,
        target: LinguisticTarget,
        value: LinguisticClaimValue,
        probability: f64,
        rationale: ClaimRationale,
    ) -> Result<Self, LinguisticClaimError> {
        Self::from_producer(
            id,
            target,
            value,
            EvidenceSource::Lexicon,
            "lexicon",
            probability,
            rationale,
        )
    }

    pub fn acoustics(
        id: LinguisticClaimId,
        target: LinguisticTarget,
        value: LinguisticClaimValue,
        committed: bool,
        probability: f64,
        rationale: ClaimRationale,
    ) -> Result<Self, LinguisticClaimError> {
        Self::from_producer(
            id,
            target,
            value,
            if committed {
                EvidenceSource::CommittedAcoustics
            } else {
                EvidenceSource::AcousticModel
            },
            if committed {
                "committed-acoustics"
            } else {
                "acoustic-model"
            },
            probability,
            rationale,
        )
    }

    pub fn morphology(
        id: LinguisticClaimId,
        target: LinguisticTarget,
        value: LinguisticClaimValue,
        probability: f64,
        rationale: ClaimRationale,
    ) -> Result<Self, LinguisticClaimError> {
        Self::from_producer(
            id,
            target,
            value,
            EvidenceSource::Morphology,
            "morphology",
            probability,
            rationale,
        )
    }

    pub fn user_markup(
        id: LinguisticClaimId,
        target: LinguisticTarget,
        value: LinguisticClaimValue,
        probability: f64,
        rationale: ClaimRationale,
    ) -> Result<Self, LinguisticClaimError> {
        Self::from_producer(
            id,
            target,
            value,
            EvidenceSource::UserMarkup,
            "user-markup",
            probability,
            rationale,
        )
    }

    pub fn manual_override(
        id: LinguisticClaimId,
        target: LinguisticTarget,
        value: LinguisticClaimValue,
        rationale: ClaimRationale,
    ) -> Result<Self, LinguisticClaimError> {
        Self::from_producer(
            id,
            target,
            value,
            EvidenceSource::ManualOverride,
            "manual-override",
            1.0,
            rationale,
        )
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.source_priority = priority;
        self
    }

    pub fn with_support(mut self, claim_id: LinguisticClaimId) -> Self {
        self.supports.push(claim_id);
        self
    }

    pub fn with_conflict(mut self, claim_id: LinguisticClaimId) -> Self {
        self.conflicts_with.push(claim_id);
        self
    }

    pub fn validate(&self) -> Result<(), LinguisticClaimError> {
        if self.id.0.trim().is_empty() {
            return Err(LinguisticClaimError::EmptyClaimId);
        }
        self.target.validate()?;
        self.confidence.validate()?;
        if self.kind != self.value.kind() {
            return Err(LinguisticClaimError::KindValueMismatch {
                kind: self.kind,
                value_kind: self.value.kind(),
            });
        }
        if self.provenance.method.trim().is_empty()
            || self.rationale.code.trim().is_empty()
            || self.rationale.summary.trim().is_empty()
        {
            return Err(LinguisticClaimError::MissingExplanation(self.id.clone()));
        }
        if self.supports.contains(&self.id) || self.conflicts_with.contains(&self.id) {
            return Err(LinguisticClaimError::SelfEdge(self.id.clone()));
        }
        let supports = self.supports.iter().collect::<BTreeSet<_>>();
        if supports.len() != self.supports.len() {
            return Err(LinguisticClaimError::DuplicateClaimEdge(self.id.clone()));
        }
        let conflicts = self.conflicts_with.iter().collect::<BTreeSet<_>>();
        if conflicts.len() != self.conflicts_with.len() {
            return Err(LinguisticClaimError::DuplicateClaimEdge(self.id.clone()));
        }
        if supports.iter().any(|id| conflicts.contains(id)) {
            return Err(LinguisticClaimError::AmbiguousClaimEdge(self.id.clone()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimLifecycleTransition {
    pub sequence: u64,
    pub claim_id: LinguisticClaimId,
    pub from: ClaimLifecycle,
    pub to: ClaimLifecycle,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<LinguisticClaimId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "reason")]
pub enum ClaimResolutionReason {
    Selected,
    Revised,
    Invalidated,
    CommittedWinner,
    LowerPriority {
        winner_priority: i32,
        candidate_priority: i32,
    },
    LowerConfidence {
        winner_probability: f64,
        candidate_probability: f64,
    },
    FewerSupports {
        winner_supports: usize,
        candidate_supports: usize,
    },
    LessStableLifecycle {
        winner: ClaimLifecycle,
        candidate: ClaimLifecycle,
    },
    DeterministicIdTieBreak {
        winner_id: LinguisticClaimId,
    },
}

impl ClaimResolutionReason {
    pub fn explanation(&self) -> String {
        match self {
            Self::Selected => "selected by the resolution policy".into(),
            Self::Revised => "claim was superseded and remains diagnostic history".into(),
            Self::Invalidated => "claim was invalidated by changed evidence".into(),
            Self::CommittedWinner => "a committed claim is locked against later analysis".into(),
            Self::LowerPriority {
                winner_priority,
                candidate_priority,
            } => format!(
                "source priority {candidate_priority} is lower than winner priority {winner_priority}"
            ),
            Self::LowerConfidence {
                winner_probability,
                candidate_probability,
            } => format!(
                "equal-priority normalized confidence {candidate_probability:.6} is lower than {winner_probability:.6}"
            ),
            Self::FewerSupports {
                winner_supports,
                candidate_supports,
            } => format!(
                "candidate has {candidate_supports} active supports; winner has {winner_supports}"
            ),
            Self::LessStableLifecycle { winner, candidate } => {
                format!("{candidate:?} is less stable than {winner:?}")
            }
            Self::DeterministicIdTieBreak { winner_id } => {
                format!("all policy scores tied; stable claim ID {winner_id:?} sorts first")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimResolutionCandidate {
    pub claim_id: LinguisticClaimId,
    pub value: LinguisticClaimValue,
    pub lifecycle: ClaimLifecycle,
    pub selected: bool,
    pub conflicts_with_winner: bool,
    pub reason: ClaimResolutionReason,
    pub explanation: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinguisticClaimResolution {
    pub id: ClaimResolutionId,
    pub target: LinguisticTarget,
    pub kind: LinguisticClaimKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner: Option<LinguisticClaimId>,
    pub policy: String,
    pub candidates: Vec<ClaimResolutionCandidate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinguisticEvidenceArtifact {
    pub schema_version: u16,
    pub utterance_id: UtteranceId,
    #[serde(default)]
    pub claims: Vec<LinguisticClaim>,
    #[serde(default)]
    pub lifecycle: Vec<ClaimLifecycleTransition>,
    #[serde(default)]
    pub resolutions: Vec<LinguisticClaimResolution>,
}

impl LinguisticEvidenceArtifact {
    pub fn new(utterance_id: UtteranceId) -> Self {
        Self {
            schema_version: LINGUISTIC_EVIDENCE_SCHEMA_V1,
            utterance_id,
            claims: Vec::new(),
            lifecycle: Vec::new(),
            resolutions: Vec::new(),
        }
    }

    pub fn insert_claim(&mut self, claim: LinguisticClaim) -> Result<(), LinguisticClaimError> {
        claim.validate()?;
        if claim.lifecycle != ClaimLifecycle::Hypothesis {
            return Err(LinguisticClaimError::InitialLifecycleMustBeHypothesis {
                id: claim.id,
                found: claim.lifecycle,
            });
        }
        if claim.target.utterance_id != self.utterance_id {
            return Err(LinguisticClaimError::WrongUtterance {
                expected: self.utterance_id.clone(),
                found: claim.target.utterance_id,
            });
        }
        if self.claim(&claim.id).is_some() {
            return Err(LinguisticClaimError::DuplicateClaimId(claim.id));
        }
        self.claims.push(claim);
        Ok(())
    }

    pub fn claim(&self, id: &LinguisticClaimId) -> Option<&LinguisticClaim> {
        self.claims.iter().find(|claim| &claim.id == id)
    }

    pub fn transition_claim(
        &mut self,
        id: &LinguisticClaimId,
        next: ClaimLifecycle,
        reason: impl Into<String>,
        superseded_by: Option<LinguisticClaimId>,
    ) -> Result<(), LinguisticClaimError> {
        let reason = reason.into();
        if reason.trim().is_empty() {
            return Err(LinguisticClaimError::MissingTransitionReason(id.clone()));
        }
        let sequence = self.lifecycle.len() as u64;
        let position = self
            .claims
            .iter()
            .position(|claim| &claim.id == id)
            .ok_or_else(|| LinguisticClaimError::UnknownClaimId(id.clone()))?;
        let claim = &self.claims[position];
        let from = claim.lifecycle;
        if !from.can_transition_to(next) {
            return Err(LinguisticClaimError::InvalidLifecycleTransition {
                id: id.clone(),
                from,
                to: next,
            });
        }
        if let Some(replacement_id) = &superseded_by {
            if next != ClaimLifecycle::Revised || replacement_id == id {
                return Err(LinguisticClaimError::InvalidSupersession {
                    id: id.clone(),
                    replacement: replacement_id.clone(),
                });
            }
            let replacement = self
                .claims
                .iter()
                .find(|claim| &claim.id == replacement_id)
                .ok_or_else(|| LinguisticClaimError::UnknownClaimId(replacement_id.clone()))?;
            if replacement.target != claim.target || replacement.kind != claim.kind {
                return Err(LinguisticClaimError::RevisionTargetMismatch {
                    old: id.clone(),
                    replacement: replacement_id.clone(),
                });
            }
        }
        self.claims[position].lifecycle = next;
        self.lifecycle.push(ClaimLifecycleTransition {
            sequence,
            claim_id: id.clone(),
            from,
            to: next,
            reason,
            superseded_by,
        });
        Ok(())
    }

    pub fn stabilize_claim(
        &mut self,
        id: &LinguisticClaimId,
        reason: impl Into<String>,
    ) -> Result<(), LinguisticClaimError> {
        self.transition_claim(id, ClaimLifecycle::Stable, reason, None)
    }

    pub fn commit_claim(
        &mut self,
        id: &LinguisticClaimId,
        reason: impl Into<String>,
    ) -> Result<(), LinguisticClaimError> {
        self.transition_claim(id, ClaimLifecycle::Committed, reason, None)
    }

    pub fn revise_claim(
        &mut self,
        old_id: &LinguisticClaimId,
        replacement: LinguisticClaim,
        reason: impl Into<String>,
    ) -> Result<(), LinguisticClaimError> {
        let reason = reason.into();
        let old = self
            .claim(old_id)
            .ok_or_else(|| LinguisticClaimError::UnknownClaimId(old_id.clone()))?;
        if old.lifecycle == ClaimLifecycle::Committed {
            return Err(LinguisticClaimError::CommittedClaimCannotChange(
                old_id.clone(),
            ));
        }
        if !old.lifecycle.can_transition_to(ClaimLifecycle::Revised) {
            return Err(LinguisticClaimError::InvalidLifecycleTransition {
                id: old_id.clone(),
                from: old.lifecycle,
                to: ClaimLifecycle::Revised,
            });
        }
        if reason.trim().is_empty() {
            return Err(LinguisticClaimError::MissingTransitionReason(
                old_id.clone(),
            ));
        }
        if old.target != replacement.target || old.kind != replacement.kind {
            return Err(LinguisticClaimError::RevisionTargetMismatch {
                old: old_id.clone(),
                replacement: replacement.id,
            });
        }
        replacement.validate()?;
        if replacement.target.utterance_id != self.utterance_id {
            return Err(LinguisticClaimError::WrongUtterance {
                expected: self.utterance_id.clone(),
                found: replacement.target.utterance_id,
            });
        }
        if self.claim(&replacement.id).is_some() {
            return Err(LinguisticClaimError::DuplicateClaimId(replacement.id));
        }
        let replacement_id = replacement.id.clone();
        self.insert_claim(replacement)?;
        self.transition_claim(
            old_id,
            ClaimLifecycle::Revised,
            reason,
            Some(replacement_id),
        )
    }

    /// Invalidates only claims whose ranges are at or after the repaired
    /// prefix. A revision touching committed evidence fails before mutation.
    pub fn invalidate_text_revision(
        &mut self,
        revised: TextRange,
        reason: impl Into<String>,
    ) -> Result<Vec<LinguisticClaimId>, LinguisticClaimError> {
        validate_text_range(&revised)?;
        let mut affected = self
            .claims
            .iter()
            .filter(|claim| {
                claim.lifecycle.is_resolution_eligible()
                    && claim.target.affected_by_revision(&revised)
            })
            .map(|claim| claim.id.clone())
            .collect::<Vec<_>>();
        affected.sort();
        if let Some(committed) = affected.iter().find(|id| {
            self.claim(id)
                .is_some_and(|claim| claim.lifecycle == ClaimLifecycle::Committed)
        }) {
            return Err(LinguisticClaimError::CommittedClaimCannotChange(
                committed.clone(),
            ));
        }
        let reason = reason.into();
        if reason.trim().is_empty()
            && let Some(id) = affected.first()
        {
            return Err(LinguisticClaimError::MissingTransitionReason(id.clone()));
        }
        for id in &affected {
            self.transition_claim(id, ClaimLifecycle::Invalidated, reason.clone(), None)?;
        }
        Ok(affected)
    }

    pub fn resolve(
        &mut self,
        id: ClaimResolutionId,
        target: &LinguisticTarget,
        kind: LinguisticClaimKind,
    ) -> Result<LinguisticClaimResolution, LinguisticClaimError> {
        self.validate()?;
        if id.0.trim().is_empty() {
            return Err(LinguisticClaimError::EmptyResolutionId);
        }
        if self
            .resolutions
            .iter()
            .any(|resolution| resolution.id == id)
        {
            return Err(LinguisticClaimError::DuplicateResolutionId(id));
        }
        let candidates = self
            .claims
            .iter()
            .filter(|claim| claim.target == *target && claim.kind == kind)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Err(LinguisticClaimError::NoClaimsForTarget);
        }

        let active_ids = self
            .claims
            .iter()
            .filter(|claim| claim.lifecycle.is_resolution_eligible())
            .map(|claim| claim.id.clone())
            .collect::<BTreeSet<_>>();
        let support_count = |claim: &LinguisticClaim| {
            claim
                .supports
                .iter()
                .filter(|support| active_ids.contains(*support))
                .collect::<BTreeSet<_>>()
                .len()
        };
        let committed = candidates
            .iter()
            .copied()
            .filter(|claim| claim.lifecycle == ClaimLifecycle::Committed)
            .collect::<Vec<_>>();
        if committed.len() > 1
            && committed
                .windows(2)
                .any(|pair| pair[0].value != pair[1].value)
        {
            return Err(LinguisticClaimError::ConflictingCommittedClaims);
        }

        let mut eligible = if committed.is_empty() {
            candidates
                .iter()
                .copied()
                .filter(|claim| claim.lifecycle.is_resolution_eligible())
                .collect::<Vec<_>>()
        } else {
            committed
        };
        eligible.sort_by(|left, right| compare_claims(left, right, &support_count));
        let winner = eligible.first().copied();

        let mut resolved_candidates = candidates
            .iter()
            .map(|candidate| {
                let reason = resolution_reason(candidate, winner, &support_count);
                let conflicts_with_winner = winner.is_some_and(|winner| {
                    candidate.id != winner.id
                        && (candidate.value != winner.value
                            || candidate.conflicts_with.contains(&winner.id)
                            || winner.conflicts_with.contains(&candidate.id))
                });
                ClaimResolutionCandidate {
                    claim_id: candidate.id.clone(),
                    value: candidate.value.clone(),
                    lifecycle: candidate.lifecycle,
                    selected: winner.is_some_and(|winner| winner.id == candidate.id),
                    conflicts_with_winner,
                    explanation: reason.explanation(),
                    reason,
                }
            })
            .collect::<Vec<_>>();
        resolved_candidates.sort_by(|left, right| left.claim_id.cmp(&right.claim_id));

        let resolution = LinguisticClaimResolution {
            id,
            target: target.clone(),
            kind,
            winner: winner.map(|winner| winner.id.clone()),
            policy: LINGUISTIC_RESOLUTION_POLICY_V1.into(),
            candidates: resolved_candidates,
        };
        self.resolutions.push(resolution.clone());
        Ok(resolution)
    }

    pub fn validate(&self) -> Result<(), LinguisticClaimError> {
        if self.schema_version != LINGUISTIC_EVIDENCE_SCHEMA_V1 {
            return Err(LinguisticClaimError::UnsupportedSchema {
                found: u64::from(self.schema_version),
                expected: LINGUISTIC_EVIDENCE_SCHEMA_V1,
            });
        }
        if self.utterance_id.0.trim().is_empty() {
            return Err(LinguisticClaimError::EmptyUtteranceId);
        }
        let mut ids = BTreeSet::new();
        for claim in &self.claims {
            claim.validate()?;
            if claim.target.utterance_id != self.utterance_id {
                return Err(LinguisticClaimError::WrongUtterance {
                    expected: self.utterance_id.clone(),
                    found: claim.target.utterance_id.clone(),
                });
            }
            if !ids.insert(claim.id.clone()) {
                return Err(LinguisticClaimError::DuplicateClaimId(claim.id.clone()));
            }
        }
        for claim in &self.claims {
            for edge in claim.supports.iter().chain(&claim.conflicts_with) {
                if !ids.contains(edge) {
                    return Err(LinguisticClaimError::UnknownClaimEdge {
                        from: claim.id.clone(),
                        to: edge.clone(),
                    });
                }
            }
        }
        let mut replayed = ids
            .iter()
            .cloned()
            .map(|id| (id, ClaimLifecycle::Hypothesis))
            .collect::<BTreeMap<_, _>>();
        for (index, transition) in self.lifecycle.iter().enumerate() {
            if transition.sequence != index as u64 {
                return Err(LinguisticClaimError::InvalidLifecycleSequence {
                    found: transition.sequence,
                    expected: index as u64,
                });
            }
            if transition.reason.trim().is_empty() {
                return Err(LinguisticClaimError::MissingTransitionReason(
                    transition.claim_id.clone(),
                ));
            }
            let state = replayed
                .get_mut(&transition.claim_id)
                .ok_or_else(|| LinguisticClaimError::UnknownClaimId(transition.claim_id.clone()))?;
            if *state != transition.from || !transition.from.can_transition_to(transition.to) {
                return Err(LinguisticClaimError::InvalidLifecycleHistory {
                    id: transition.claim_id.clone(),
                    expected: *state,
                    found: transition.from,
                    to: transition.to,
                });
            }
            if let Some(replacement) = &transition.superseded_by
                && !ids.contains(replacement)
            {
                return Err(LinguisticClaimError::UnknownClaimEdge {
                    from: transition.claim_id.clone(),
                    to: replacement.clone(),
                });
            }
            if let Some(replacement_id) = &transition.superseded_by {
                if transition.to != ClaimLifecycle::Revised
                    || replacement_id == &transition.claim_id
                {
                    return Err(LinguisticClaimError::InvalidSupersession {
                        id: transition.claim_id.clone(),
                        replacement: replacement_id.clone(),
                    });
                }
                let original = self
                    .claim(&transition.claim_id)
                    .expect("lifecycle claim ID was validated above");
                let replacement = self
                    .claim(replacement_id)
                    .expect("replacement claim ID was validated above");
                if original.target != replacement.target || original.kind != replacement.kind {
                    return Err(LinguisticClaimError::RevisionTargetMismatch {
                        old: original.id.clone(),
                        replacement: replacement.id.clone(),
                    });
                }
            }
            *state = transition.to;
        }
        for claim in &self.claims {
            let replayed_state = replayed
                .get(&claim.id)
                .copied()
                .unwrap_or(ClaimLifecycle::Hypothesis);
            if replayed_state != claim.lifecycle {
                return Err(LinguisticClaimError::LifecycleStateMismatch {
                    id: claim.id.clone(),
                    replayed: replayed_state,
                    stored: claim.lifecycle,
                });
            }
        }
        let mut resolution_ids = BTreeSet::new();
        for resolution in &self.resolutions {
            if resolution.id.0.trim().is_empty() {
                return Err(LinguisticClaimError::EmptyResolutionId);
            }
            if !resolution_ids.insert(resolution.id.clone()) {
                return Err(LinguisticClaimError::DuplicateResolutionId(
                    resolution.id.clone(),
                ));
            }
            if resolution.policy != LINGUISTIC_RESOLUTION_POLICY_V1 {
                return Err(LinguisticClaimError::UnsupportedResolutionPolicy(
                    resolution.policy.clone(),
                ));
            }
            resolution.target.validate()?;
            if resolution.target.utterance_id != self.utterance_id {
                return Err(LinguisticClaimError::WrongUtterance {
                    expected: self.utterance_id.clone(),
                    found: resolution.target.utterance_id.clone(),
                });
            }
            let selected = resolution
                .candidates
                .iter()
                .filter(|candidate| candidate.selected)
                .map(|candidate| candidate.claim_id.clone())
                .collect::<Vec<_>>();
            if selected.as_slice() != resolution.winner.as_slice() {
                return Err(LinguisticClaimError::InvalidResolutionSelection(
                    resolution.id.clone(),
                ));
            }
            if resolution.candidates.is_empty()
                || resolution.candidates.iter().any(|candidate| {
                    candidate.selected && !candidate.lifecycle.is_resolution_eligible()
                })
            {
                return Err(LinguisticClaimError::InvalidResolutionSelection(
                    resolution.id.clone(),
                ));
            }
            let mut candidate_ids = BTreeSet::new();
            for candidate in &resolution.candidates {
                if !candidate_ids.insert(candidate.claim_id.clone()) {
                    return Err(LinguisticClaimError::InvalidResolutionCandidate {
                        resolution: resolution.id.clone(),
                        claim: candidate.claim_id.clone(),
                    });
                }
                let claim = self.claim(&candidate.claim_id).ok_or_else(|| {
                    LinguisticClaimError::UnknownClaimId(candidate.claim_id.clone())
                })?;
                if claim.target != resolution.target
                    || claim.kind != resolution.kind
                    || claim.value != candidate.value
                {
                    return Err(LinguisticClaimError::InvalidResolutionCandidate {
                        resolution: resolution.id.clone(),
                        claim: candidate.claim_id.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn to_json_pretty(&self) -> Result<String, LinguisticClaimError> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| LinguisticClaimError::Serialization(error.to_string()))
    }

    pub fn from_json_str(json: &str) -> Result<Self, LinguisticClaimError> {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|error| LinguisticClaimError::Serialization(error.to_string()))?;
        Self::from_json_value(value)
    }

    pub fn from_json_value(value: serde_json::Value) -> Result<Self, LinguisticClaimError> {
        let Some(version) = value
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
        else {
            return Err(LinguisticClaimError::MissingSchemaVersion);
        };
        if version != u64::from(LINGUISTIC_EVIDENCE_SCHEMA_V1) {
            return Err(LinguisticClaimError::UnsupportedSchema {
                found: version,
                expected: LINGUISTIC_EVIDENCE_SCHEMA_V1,
            });
        }
        let artifact: Self = serde_json::from_value(value)
            .map_err(|error| LinguisticClaimError::Serialization(error.to_string()))?;
        artifact.validate()?;
        Ok(artifact)
    }

    /// Projects the contract into the existing provider-neutral derived
    /// artifact event used by CLI JSONL, server APIs, and browser views.
    pub fn as_derived_artifact(
        &self,
        artifact_id: impl Into<String>,
    ) -> Result<StreamEvent, LinguisticClaimError> {
        self.validate()?;
        let artifact_id = artifact_id.into();
        if artifact_id.trim().is_empty() {
            return Err(LinguisticClaimError::EmptyArtifactId);
        }
        let value = serde_json::to_value(self)
            .map_err(|error| LinguisticClaimError::Serialization(error.to_string()))?;
        Ok(StreamEvent::DerivedArtifact {
            stage: "linguistic_claims".into(),
            artifact_id,
            value,
        })
    }

    pub fn from_derived_artifact(event: &StreamEvent) -> Result<Self, LinguisticClaimError> {
        match event {
            StreamEvent::DerivedArtifact { stage, value, .. } if stage == "linguistic_claims" => {
                Self::from_json_value(value.clone())
            }
            StreamEvent::DerivedArtifact { stage, .. } => {
                Err(LinguisticClaimError::UnexpectedArtifactStage(stage.clone()))
            }
            _ => Err(LinguisticClaimError::ExpectedDerivedArtifact),
        }
    }
}

fn compare_claims(
    left: &LinguisticClaim,
    right: &LinguisticClaim,
    support_count: &impl Fn(&LinguisticClaim) -> usize,
) -> Ordering {
    right
        .source_priority
        .cmp(&left.source_priority)
        .then_with(|| {
            right
                .confidence
                .probability
                .total_cmp(&left.confidence.probability)
        })
        .then_with(|| support_count(right).cmp(&support_count(left)))
        .then_with(|| {
            right
                .lifecycle
                .stability_rank()
                .cmp(&left.lifecycle.stability_rank())
        })
        .then_with(|| left.id.cmp(&right.id))
}

fn resolution_reason(
    candidate: &LinguisticClaim,
    winner: Option<&LinguisticClaim>,
    support_count: &impl Fn(&LinguisticClaim) -> usize,
) -> ClaimResolutionReason {
    let Some(winner) = winner else {
        return match candidate.lifecycle {
            ClaimLifecycle::Revised => ClaimResolutionReason::Revised,
            ClaimLifecycle::Invalidated => ClaimResolutionReason::Invalidated,
            _ => ClaimResolutionReason::DeterministicIdTieBreak {
                winner_id: candidate.id.clone(),
            },
        };
    };
    if candidate.id == winner.id {
        return ClaimResolutionReason::Selected;
    }
    match candidate.lifecycle {
        ClaimLifecycle::Revised => return ClaimResolutionReason::Revised,
        ClaimLifecycle::Invalidated => return ClaimResolutionReason::Invalidated,
        _ => {}
    }
    if winner.lifecycle == ClaimLifecycle::Committed
        && candidate.lifecycle != ClaimLifecycle::Committed
    {
        return ClaimResolutionReason::CommittedWinner;
    }
    if candidate.source_priority != winner.source_priority {
        return ClaimResolutionReason::LowerPriority {
            winner_priority: winner.source_priority,
            candidate_priority: candidate.source_priority,
        };
    }
    if candidate.confidence.probability != winner.confidence.probability {
        return ClaimResolutionReason::LowerConfidence {
            winner_probability: winner.confidence.probability,
            candidate_probability: candidate.confidence.probability,
        };
    }
    let winner_supports = support_count(winner);
    let candidate_supports = support_count(candidate);
    if candidate_supports != winner_supports {
        return ClaimResolutionReason::FewerSupports {
            winner_supports,
            candidate_supports,
        };
    }
    if candidate.lifecycle.stability_rank() != winner.lifecycle.stability_rank() {
        return ClaimResolutionReason::LessStableLifecycle {
            winner: winner.lifecycle,
            candidate: candidate.lifecycle,
        };
    }
    ClaimResolutionReason::DeterministicIdTieBreak {
        winner_id: winner.id.clone(),
    }
}

fn validate_text_range(range: &TextRange) -> Result<(), LinguisticClaimError> {
    if range.start > range.end {
        return Err(LinguisticClaimError::InvalidTextRange {
            start: range.start,
            end: range.end,
        });
    }
    Ok(())
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum LinguisticClaimError {
    #[error("utterance IDs must not be empty")]
    EmptyUtteranceId,
    #[error("claim IDs must not be empty")]
    EmptyClaimId,
    #[error("target IDs must not be empty")]
    EmptyTargetId,
    #[error("resolution IDs must not be empty")]
    EmptyResolutionId,
    #[error("artifact IDs must not be empty")]
    EmptyArtifactId,
    #[error("claim confidence must be finite and in [0, 1], found {0}")]
    InvalidConfidence(f64),
    #[error("claim kind {kind:?} does not match value kind {value_kind:?}")]
    KindValueMismatch {
        kind: LinguisticClaimKind,
        value_kind: LinguisticClaimKind,
    },
    #[error("text-range targets must include a text_range")]
    TextRangeTargetMissingRange,
    #[error("invalid text range {start}..{end}: start must not exceed end")]
    InvalidTextRange { start: u32, end: u32 },
    #[error("claim {0:?} must include provenance method and machine/human rationale")]
    MissingExplanation(LinguisticClaimId),
    #[error("claim {0:?} cannot support or conflict with itself")]
    SelfEdge(LinguisticClaimId),
    #[error("claim {0:?} repeats a support or conflict edge")]
    DuplicateClaimEdge(LinguisticClaimId),
    #[error("claim {0:?} marks the same edge as both support and conflict")]
    AmbiguousClaimEdge(LinguisticClaimId),
    #[error("duplicate claim ID {0:?}")]
    DuplicateClaimId(LinguisticClaimId),
    #[error("unknown claim ID {0:?}")]
    UnknownClaimId(LinguisticClaimId),
    #[error("claim {from:?} references unknown claim edge {to:?}")]
    UnknownClaimEdge {
        from: LinguisticClaimId,
        to: LinguisticClaimId,
    },
    #[error("claim targets utterance {found:?}; artifact owns {expected:?}")]
    WrongUtterance {
        expected: UtteranceId,
        found: UtteranceId,
    },
    #[error("invalid lifecycle transition for {id:?}: {from:?} -> {to:?}")]
    InvalidLifecycleTransition {
        id: LinguisticClaimId,
        from: ClaimLifecycle,
        to: ClaimLifecycle,
    },
    #[error("new claim {id:?} must begin as hypothesis, found {found:?}")]
    InitialLifecycleMustBeHypothesis {
        id: LinguisticClaimId,
        found: ClaimLifecycle,
    },
    #[error("lifecycle sequence {found} is not the expected append-only sequence {expected}")]
    InvalidLifecycleSequence { found: u64, expected: u64 },
    #[error("lifecycle history for {id:?} expected {expected:?}, found {found:?} -> {to:?}")]
    InvalidLifecycleHistory {
        id: LinguisticClaimId,
        expected: ClaimLifecycle,
        found: ClaimLifecycle,
        to: ClaimLifecycle,
    },
    #[error("claim {id:?} stores {stored:?}, but lifecycle history replays to {replayed:?}")]
    LifecycleStateMismatch {
        id: LinguisticClaimId,
        replayed: ClaimLifecycle,
        stored: ClaimLifecycle,
    },
    #[error("lifecycle transition for {0:?} requires a reason")]
    MissingTransitionReason(LinguisticClaimId),
    #[error("committed claim {0:?} cannot be revised or invalidated")]
    CommittedClaimCannotChange(LinguisticClaimId),
    #[error("replacement {replacement:?} does not share target and kind with {old:?}")]
    RevisionTargetMismatch {
        old: LinguisticClaimId,
        replacement: LinguisticClaimId,
    },
    #[error("claim {id:?} can name replacement {replacement:?} only when revised")]
    InvalidSupersession {
        id: LinguisticClaimId,
        replacement: LinguisticClaimId,
    },
    #[error("duplicate resolution ID {0:?}")]
    DuplicateResolutionId(ClaimResolutionId),
    #[error("unsupported linguistic resolution policy {0:?}")]
    UnsupportedResolutionPolicy(String),
    #[error("resolution {0:?} winner does not match its selected candidate")]
    InvalidResolutionSelection(ClaimResolutionId),
    #[error("resolution {resolution:?} candidate {claim:?} does not match its claim")]
    InvalidResolutionCandidate {
        resolution: ClaimResolutionId,
        claim: LinguisticClaimId,
    },
    #[error("no claims exist for the requested target and kind")]
    NoClaimsForTarget,
    #[error("conflicting committed claims violate stable commitment")]
    ConflictingCommittedClaims,
    #[error("linguistic evidence artifact is missing schema_version")]
    MissingSchemaVersion,
    #[error("unsupported linguistic evidence schema_version={found}; expected {expected}")]
    UnsupportedSchema { found: u64, expected: u16 },
    #[error("expected a derived_artifact stream event")]
    ExpectedDerivedArtifact,
    #[error("expected linguistic_claims artifact stage, found {0:?}")]
    UnexpectedArtifactStage(String),
    #[error("linguistic evidence serialization failed: {0}")]
    Serialization(String),
}
