//! Typed linguistic scoring, identity agreement, and commit diagnostics for
//! the deterministic duplex simulator.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use speaking::{
    ClaimResolutionId, CompletionHypothesisId, EvidenceSource, LinguisticClaimId,
    MorphemeOccurrenceId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinguisticScoreComponent {
    AcousticLikelihood,
    ProviderPrior,
    LexicalEvidence,
    GrammarParseRank,
    ProsodyCompatibility,
    UserMarkup,
    DirectObservation,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LinguisticScoreHints {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acoustic_likelihood: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lexical_evidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar_parse_rank: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prosody_compatibility: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_markup: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinguisticScoreWeights {
    pub acoustic_likelihood: f64,
    pub provider_prior: f64,
    pub lexical_evidence: f64,
    pub grammar_parse_rank: f64,
    pub prosody_compatibility: f64,
    pub user_markup: f64,
    pub direct_observation: f64,
}

impl Default for LinguisticScoreWeights {
    fn default() -> Self {
        Self {
            acoustic_likelihood: 2.0,
            provider_prior: 1.0,
            lexical_evidence: 0.8,
            grammar_parse_rank: 0.7,
            prosody_compatibility: 0.4,
            user_markup: 2.5,
            direct_observation: 2.0,
        }
    }
}

impl LinguisticScoreWeights {
    pub(crate) fn validate(&self) -> bool {
        let values = [
            self.acoustic_likelihood,
            self.provider_prior,
            self.lexical_evidence,
            self.grammar_parse_rank,
            self.prosody_compatibility,
            self.user_markup,
            self.direct_observation,
        ];
        values
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            && values.iter().any(|value| *value > 0.0)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NormalizedLinguisticScore {
    pub acoustic_likelihood: f64,
    pub provider_prior: f64,
    pub lexical_evidence: f64,
    pub grammar_parse_rank: f64,
    pub prosody_compatibility: f64,
    pub user_markup: f64,
    pub direct_observation: f64,
    pub combined: f64,
    #[serde(default)]
    pub available_components: BTreeSet<LinguisticScoreComponent>,
    #[serde(default)]
    pub claim_attribution: BTreeMap<LinguisticScoreComponent, Vec<LinguisticClaimId>>,
}

impl NormalizedLinguisticScore {
    pub(crate) fn combine(&mut self, weights: &LinguisticScoreWeights) {
        let components = [
            (
                LinguisticScoreComponent::AcousticLikelihood,
                self.acoustic_likelihood,
                weights.acoustic_likelihood,
            ),
            (
                LinguisticScoreComponent::ProviderPrior,
                self.provider_prior,
                weights.provider_prior,
            ),
            (
                LinguisticScoreComponent::LexicalEvidence,
                self.lexical_evidence,
                weights.lexical_evidence,
            ),
            (
                LinguisticScoreComponent::GrammarParseRank,
                self.grammar_parse_rank,
                weights.grammar_parse_rank,
            ),
            (
                LinguisticScoreComponent::ProsodyCompatibility,
                self.prosody_compatibility,
                weights.prosody_compatibility,
            ),
            (
                LinguisticScoreComponent::UserMarkup,
                self.user_markup,
                weights.user_markup,
            ),
            (
                LinguisticScoreComponent::DirectObservation,
                self.direct_observation,
                weights.direct_observation,
            ),
        ];
        let weighted = components
            .iter()
            .filter(|(component, _, _)| self.available_components.contains(component))
            .map(|(_, value, weight)| value * weight)
            .sum::<f64>();
        let denominator = components
            .iter()
            .filter(|(component, _, _)| self.available_components.contains(component))
            .map(|(_, _, weight)| *weight)
            .sum::<f64>();
        self.combined = (weighted / denominator).clamp(0.0, 1.0);

        // A tidy parse must not erase overwhelming contradictory acoustics.
        if self
            .available_components
            .contains(&LinguisticScoreComponent::AcousticLikelihood)
            && self.acoustic_likelihood < 0.15
            && self.grammar_parse_rank > 0.8
        {
            self.combined = self.combined.min(0.25);
        }
    }

    pub(crate) fn ranking_mass(&self, weights: &LinguisticScoreWeights) -> f64 {
        let evidence = [
            (
                LinguisticScoreComponent::AcousticLikelihood,
                self.acoustic_likelihood,
                weights.acoustic_likelihood,
            ),
            (
                LinguisticScoreComponent::LexicalEvidence,
                self.lexical_evidence,
                weights.lexical_evidence,
            ),
            (
                LinguisticScoreComponent::GrammarParseRank,
                self.grammar_parse_rank,
                weights.grammar_parse_rank,
            ),
            (
                LinguisticScoreComponent::ProsodyCompatibility,
                self.prosody_compatibility,
                weights.prosody_compatibility,
            ),
            (
                LinguisticScoreComponent::UserMarkup,
                self.user_markup,
                weights.user_markup,
            ),
            (
                LinguisticScoreComponent::DirectObservation,
                self.direct_observation,
                weights.direct_observation,
            ),
        ];
        let evidence_weighted = evidence
            .iter()
            .map(|(component, value, weight)| {
                if self.available_components.contains(component) {
                    value * weight
                } else {
                    0.5 * weight
                }
            })
            .sum::<f64>();
        let evidence_denominator = evidence.iter().map(|(_, _, weight)| *weight).sum::<f64>();
        let evidence_factor = if evidence_denominator > 0.0 {
            0.5 + evidence_weighted / evidence_denominator
        } else {
            1.0
        };
        let mut mass = self.provider_prior * evidence_factor;
        if self
            .available_components
            .contains(&LinguisticScoreComponent::AcousticLikelihood)
            && self.acoustic_likelihood < 0.15
            && self.grammar_parse_rank > 0.8
        {
            mass = mass.min(self.provider_prior * 0.25);
        }
        mass.max(f64::EPSILON)
    }

    pub(crate) fn mark_available(&mut self, component: LinguisticScoreComponent) {
        self.available_components.insert(component);
    }

    pub fn has_component(&self, component: LinguisticScoreComponent) -> bool {
        self.available_components.contains(&component)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MorphemeIdentityEvidence {
    pub morpheme_index: usize,
    #[serde(default)]
    pub morpheme_claim_ids: Vec<LinguisticClaimId>,
    #[serde(default)]
    pub word_claim_ids: Vec<LinguisticClaimId>,
    #[serde(default)]
    pub pronunciation_claim_ids: Vec<LinguisticClaimId>,
}

impl MorphemeIdentityEvidence {
    pub(crate) fn all_layers(&self) -> [(&'static str, &[LinguisticClaimId]); 3] {
        [
            ("morpheme", &self.morpheme_claim_ids),
            ("word", &self.word_claim_ids),
            ("pronunciation", &self.pronunciation_claim_ids),
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisDecisionStatus {
    Candidate,
    Selected,
    Committed,
    Verified,
    Revised,
    Invalidated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommitBlockReason {
    LowScoreMargin,
    DirectEvidenceMissing {
        morpheme_index: usize,
        key: String,
    },
    IdentityLayerUnresolved {
        morpheme_index: usize,
        layer: String,
    },
    ClaimMissing {
        claim_id: LinguisticClaimId,
    },
    ResolutionMissing {
        resolution_id: ClaimResolutionId,
    },
    ProviderDisagreement,
    AcousticContradiction,
    NoSharedPrefix,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HypothesisRanking {
    pub id: CompletionHypothesisId,
    pub probability: f64,
    pub score: NormalizedLinguisticScore,
    pub status: HypothesisDecisionStatus,
    #[serde(default)]
    pub block_reasons: Vec<CommitBlockReason>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommitDecisionDiagnostic {
    pub frontier_from: usize,
    pub frontier_to: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leading_hypothesis_id: Option<CompletionHypothesisId>,
    pub leading_probability: f64,
    pub score_margin: f64,
    pub committed: bool,
    #[serde(default)]
    pub reasons: Vec<CommitBlockReason>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HypothesisAuditEntry {
    pub sequence: u64,
    pub status: HypothesisDecisionStatus,
    pub probability: f64,
    pub score: NormalizedLinguisticScore,
    #[serde(default)]
    pub claim_ids: Vec<LinguisticClaimId>,
    #[serde(default)]
    pub resolution_ids: Vec<ClaimResolutionId>,
    #[serde(default)]
    pub reasons: Vec<CommitBlockReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynthesisDeliveryState {
    Planned,
    Prepared,
    Held,
    Played,
    Verified,
    Invalidated,
}

impl SynthesisDeliveryState {
    pub(crate) const fn phase(self) -> u8 {
        match self {
            Self::Planned => 0,
            Self::Prepared => 1,
            Self::Held => 2,
            Self::Played => 3,
            Self::Verified => 4,
            Self::Invalidated => 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynthesisDeliveryRecord {
    pub emission_id: String,
    pub hypothesis_id: CompletionHypothesisId,
    pub state: SynthesisDeliveryState,
    pub text: String,
    #[serde(default)]
    pub claim_ids: Vec<LinguisticClaimId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepairDeliveryPolicy {
    ReplaceHeldAudio,
    DeliverPostPlaybackCorrection,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairDeliveryDecision {
    pub emission_id: String,
    pub policy: RepairDeliveryPolicy,
    pub reason: String,
}

pub(crate) fn component_for_source(source: &EvidenceSource) -> Option<LinguisticScoreComponent> {
    match source {
        EvidenceSource::CommittedAcoustics
        | EvidenceSource::AcousticModel
        | EvidenceSource::ForcedAlignment
        | EvidenceSource::Asr => Some(LinguisticScoreComponent::AcousticLikelihood),
        EvidenceSource::Lexicon | EvidenceSource::Morphology | EvidenceSource::G2p => {
            Some(LinguisticScoreComponent::LexicalEvidence)
        }
        EvidenceSource::Grammar => Some(LinguisticScoreComponent::GrammarParseRank),
        EvidenceSource::Prosody | EvidenceSource::Punctuation => {
            Some(LinguisticScoreComponent::ProsodyCompatibility)
        }
        EvidenceSource::Manual | EvidenceSource::ManualOverride | EvidenceSource::UserMarkup => {
            Some(LinguisticScoreComponent::UserMarkup)
        }
        EvidenceSource::ImportedData
        | EvidenceSource::LearnedPrediction
        | EvidenceSource::Rule
        | EvidenceSource::TtsPlan
        | EvidenceSource::Memory
        | EvidenceSource::Inference
        | EvidenceSource::Unknown => None,
    }
}

pub(crate) fn validate_unit_score(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

pub(crate) fn stable_prefix_len(left: &[String], right: &[String]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(left, right)| left.trim().eq_ignore_ascii_case(right.trim()))
        .count()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptEvidenceRevision {
    pub evidence_id: String,
    pub replacement_content: String,
    pub revised_range: speaking::TextRange,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StableIdentityRetention {
    pub stable_morpheme_count: usize,
    #[serde(default)]
    pub retained_occurrence_ids: Vec<MorphemeOccurrenceId>,
    #[serde(default)]
    pub invalidated_claim_ids: Vec<LinguisticClaimId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinguisticClaimUpdateKind {
    Created,
    Revised,
    Invalidated,
}
