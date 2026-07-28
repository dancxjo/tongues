//! Versioned, bounded interpretation-evidence views shared by CLI, server, and UI.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use speaking::{
    ClaimLifecycle, ClaimLifecycleTransition, ClaimResolutionId, EvidenceProvenance,
    EvidenceSource, GrammarAnalysisStatus, GrammarBackendReport, LinguisticClaim,
    LinguisticClaimId, LinguisticClaimKind, LinguisticClaimResolution, LinguisticClaimValue,
    LinguisticTarget, RankedGrammarParse, TextRange, UtteranceId,
};

use crate::{
    AcousticSpan, CommitBlockReason, HypothesisDecisionStatus, NormalizedLinguisticScore,
    SimulatorState, SynthesisDeliveryRecord, VerificationEvidence,
};

pub const INTERPRETATION_INSPECTION_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_INTERPRETATION_PAGE_LIMIT: usize = 20;
pub const MAX_INTERPRETATION_PAGE_LIMIT: usize = 100;
const MAX_OPTIONS_PER_TARGET: usize = 32;
const MAX_LINKED_CLAIMS_PER_TARGET: usize = 100;
const MAX_BACKEND_REPORTS: usize = 32;
const MAX_LIFECYCLE_EVENTS_PER_PAGE: usize = 200;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterpretationInspectionQuery {
    #[serde(default)]
    pub cursor: usize,
    #[serde(default = "default_interpretation_page_limit")]
    pub limit: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_id: Option<String>,
}

const fn default_interpretation_page_limit() -> usize {
    DEFAULT_INTERPRETATION_PAGE_LIMIT
}

impl Default for InterpretationInspectionQuery {
    fn default() -> Self {
        Self {
            cursor: 0,
            limit: DEFAULT_INTERPRETATION_PAGE_LIMIT,
            target_id: None,
        }
    }
}

impl InterpretationInspectionQuery {
    fn normalized(&self) -> Self {
        Self {
            cursor: self.cursor,
            limit: self.limit.clamp(1, MAX_INTERPRETATION_PAGE_LIMIT),
            target_id: self
                .target_id
                .as_deref()
                .map(str::trim)
                .filter(|target| !target.is_empty())
                .map(str::to_string),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InspectionEvidenceStatus {
    Available,
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpretationTargetStatus {
    Resolved,
    Unresolved,
    Historical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceAuthorityClass {
    AcousticEvidence,
    DirectObservation,
    ContextualInference,
    GrammarInference,
    LexicalEvidence,
    ProsodyInference,
    ManualOverride,
    Fallback,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterpretationClaimView {
    pub claim_id: LinguisticClaimId,
    pub target: LinguisticTarget,
    pub kind: LinguisticClaimKind,
    pub value: LinguisticClaimValue,
    pub authority: EvidenceAuthorityClass,
    pub provenance: EvidenceProvenance,
    pub confidence: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<String>,
    pub lifecycle: ClaimLifecycle,
    pub selected: bool,
    pub conflicts_with_winner: bool,
    #[serde(default)]
    pub supports: Vec<LinguisticClaimId>,
    #[serde(default)]
    pub conflicts_with: Vec<LinguisticClaimId>,
    pub rationale_code: String,
    pub rationale: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution_explanation: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticEvidenceLink {
    pub evidence_id: String,
    pub transcript: String,
    pub span: AcousticSpan,
    /// Current evidence spans are utterance/chunk aligned. This field prevents
    /// clients from implying token-exact timing where none was measured.
    pub alignment: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterpretationConsequenceView {
    pub hypothesis_id: String,
    pub selected: bool,
    #[serde(default)]
    pub statuses: Vec<HypothesisDecisionStatus>,
    pub output_text: String,
    pub score: NormalizedLinguisticScore,
    #[serde(default)]
    pub block_reasons: Vec<CommitBlockReason>,
    #[serde(default)]
    pub deliveries: Vec<SynthesisDeliveryRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterpretationTargetView {
    pub target_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution_id: Option<ClaimResolutionId>,
    pub target: LinguisticTarget,
    pub kind: LinguisticClaimKind,
    pub status: InterpretationTargetStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner: Option<InterpretationClaimView>,
    #[serde(default)]
    pub alternatives: Vec<InterpretationClaimView>,
    pub option_total: usize,
    pub options_truncated: bool,
    #[serde(default)]
    pub linked_claim_ids: Vec<LinguisticClaimId>,
    pub linked_claims_truncated: bool,
    #[serde(default)]
    pub acoustic_links: Vec<AcousticEvidenceLink>,
    #[serde(default)]
    pub consequences: Vec<InterpretationConsequenceView>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrammarBackendInspection {
    pub hypothesis_id: String,
    pub status: GrammarAnalysisStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<String>,
    pub report: GrammarBackendReport,
    #[serde(default)]
    pub parse_alternatives: Vec<RankedGrammarParse>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectionLossInspection {
    pub verification_index: usize,
    pub evidence: VerificationEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InspectionWarning {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterpretationInspectionPage {
    pub schema_version: u32,
    pub utterance_id: UtteranceId,
    pub evidence_status: InspectionEvidenceStatus,
    pub cursor: usize,
    pub limit: usize,
    pub returned: usize,
    pub total: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_target_id: Option<String>,
    #[serde(default)]
    pub targets: Vec<InterpretationTargetView>,
    #[serde(default)]
    pub backend_reports: Vec<GrammarBackendInspection>,
    pub backend_reports_truncated: bool,
    #[serde(default)]
    pub lifecycle: Vec<ClaimLifecycleTransition>,
    pub lifecycle_truncated: bool,
    #[serde(default)]
    pub projection_losses: Vec<ProjectionLossInspection>,
    #[serde(default)]
    pub warnings: Vec<InspectionWarning>,
}

pub fn interpretation_inspection_from_state(
    state: &SimulatorState,
    query: &InterpretationInspectionQuery,
) -> InterpretationInspectionPage {
    let query = query.normalized();
    let Some(artifact) = state.linguistic_evidence.as_ref() else {
        let (backend_reports, backend_reports_truncated) = backend_reports(state);
        return InterpretationInspectionPage {
            schema_version: INTERPRETATION_INSPECTION_SCHEMA_VERSION,
            utterance_id: state.utterance_id.clone(),
            evidence_status: InspectionEvidenceStatus::Missing,
            cursor: 0,
            limit: query.limit,
            returned: 0,
            total: 0,
            next_cursor: None,
            selected_target_id: query.target_id,
            targets: Vec::new(),
            backend_reports,
            backend_reports_truncated,
            lifecycle: Vec::new(),
            lifecycle_truncated: false,
            projection_losses: projection_losses(state),
            warnings: vec![InspectionWarning {
                code: "linguistic_evidence_missing".into(),
                message:
                    "No linguistic claim artifact is attached; confidence and alternatives are unknown."
                        .into(),
            }],
        };
    };

    let claims = artifact
        .claims
        .iter()
        .map(|claim| (claim.id.clone(), claim))
        .collect::<BTreeMap<_, _>>();
    let mut represented = BTreeSet::new();
    let mut targets = artifact
        .resolutions
        .iter()
        .map(|resolution| {
            represented.extend(
                resolution
                    .candidates
                    .iter()
                    .map(|candidate| candidate.claim_id.clone()),
            );
            target_from_resolution(state, artifact, &claims, resolution)
        })
        .collect::<Vec<_>>();
    for claim in artifact
        .claims
        .iter()
        .filter(|claim| !represented.contains(&claim.id))
    {
        targets.push(target_from_claim(state, artifact, claim));
    }
    targets.sort_by(|left, right| left.target_id.cmp(&right.target_id));

    let total = targets.len();
    let filtered = if let Some(target_id) = query.target_id.as_deref() {
        targets
            .into_iter()
            .filter(|target| target.target_id == target_id)
            .collect::<Vec<_>>()
    } else {
        targets
    };
    let cursor = query.cursor.min(filtered.len());
    let end = cursor.saturating_add(query.limit).min(filtered.len());
    let page_targets = filtered[cursor..end].to_vec();
    let next_cursor = (end < filtered.len()).then_some(end);
    let page_claim_ids = page_targets
        .iter()
        .flat_map(|target| {
            target
                .alternatives
                .iter()
                .map(|claim| claim.claim_id.clone())
                .chain(target.winner.iter().map(|claim| claim.claim_id.clone()))
        })
        .collect::<BTreeSet<_>>();
    let lifecycle_all = artifact
        .lifecycle
        .iter()
        .filter(|transition| page_claim_ids.contains(&transition.claim_id))
        .cloned()
        .collect::<Vec<_>>();
    let lifecycle_truncated = lifecycle_all.len() > MAX_LIFECYCLE_EVENTS_PER_PAGE;
    let lifecycle = lifecycle_all
        .into_iter()
        .take(MAX_LIFECYCLE_EVENTS_PER_PAGE)
        .collect();
    let (backend_reports, backend_reports_truncated) = backend_reports(state);
    let mut warnings = Vec::new();
    if query.target_id.is_some() && page_targets.is_empty() {
        warnings.push(InspectionWarning {
            code: "target_not_found".into(),
            message: "The requested evidence target is not present in this immutable run.".into(),
        });
    }
    if lifecycle_truncated {
        warnings.push(InspectionWarning {
            code: "lifecycle_truncated".into(),
            message: format!(
                "Lifecycle history is limited to {MAX_LIFECYCLE_EVENTS_PER_PAGE} events per page."
            ),
        });
    }
    if backend_reports_truncated {
        warnings.push(InspectionWarning {
            code: "backend_reports_truncated".into(),
            message: format!("Backend diagnostics are limited to {MAX_BACKEND_REPORTS} branches."),
        });
    }

    InterpretationInspectionPage {
        schema_version: INTERPRETATION_INSPECTION_SCHEMA_VERSION,
        utterance_id: state.utterance_id.clone(),
        evidence_status: InspectionEvidenceStatus::Available,
        cursor,
        limit: query.limit,
        returned: page_targets.len(),
        total,
        next_cursor,
        selected_target_id: query.target_id,
        targets: page_targets,
        backend_reports,
        backend_reports_truncated,
        lifecycle,
        lifecycle_truncated,
        projection_losses: projection_losses(state),
        warnings,
    }
}

fn target_from_resolution(
    state: &SimulatorState,
    artifact: &speaking::LinguisticEvidenceArtifact,
    claims: &BTreeMap<LinguisticClaimId, &LinguisticClaim>,
    resolution: &LinguisticClaimResolution,
) -> InterpretationTargetView {
    let mut options = resolution
        .candidates
        .iter()
        .filter_map(|candidate| {
            claims.get(&candidate.claim_id).map(|claim| {
                claim_view(
                    claim,
                    candidate.selected,
                    candidate.conflicts_with_winner,
                    Some(candidate.explanation.clone()),
                )
            })
        })
        .collect::<Vec<_>>();
    options.sort_by(|left, right| {
        right
            .selected
            .cmp(&left.selected)
            .then_with(|| left.claim_id.cmp(&right.claim_id))
    });
    let option_total = options.len();
    let options_truncated = option_total > MAX_OPTIONS_PER_TARGET;
    options.truncate(MAX_OPTIONS_PER_TARGET);
    let winner = resolution
        .winner
        .as_ref()
        .and_then(|winner| options.iter().find(|claim| &claim.claim_id == winner))
        .cloned();
    let alternatives = options
        .into_iter()
        .filter(|claim| !claim.selected)
        .collect::<Vec<_>>();
    let status = if winner.is_some() {
        InterpretationTargetStatus::Resolved
    } else {
        InterpretationTargetStatus::Unresolved
    };
    build_target(
        state,
        artifact,
        format!("resolution:{}", resolution.id.0),
        Some(resolution.id.clone()),
        resolution.target.clone(),
        resolution.kind,
        status,
        winner,
        alternatives,
        option_total,
        options_truncated,
        resolution
            .candidates
            .iter()
            .map(|candidate| candidate.claim_id.clone())
            .collect(),
        Some(&resolution.id),
    )
}

fn target_from_claim(
    state: &SimulatorState,
    artifact: &speaking::LinguisticEvidenceArtifact,
    claim: &LinguisticClaim,
) -> InterpretationTargetView {
    let status = if claim.lifecycle.is_resolution_eligible() {
        InterpretationTargetStatus::Unresolved
    } else {
        InterpretationTargetStatus::Historical
    };
    build_target(
        state,
        artifact,
        format!("claim:{}", claim.id.0),
        None,
        claim.target.clone(),
        claim.kind,
        status,
        None,
        vec![claim_view(claim, false, false, None)],
        1,
        false,
        vec![claim.id.clone()],
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_target(
    state: &SimulatorState,
    artifact: &speaking::LinguisticEvidenceArtifact,
    target_id: String,
    resolution_id: Option<ClaimResolutionId>,
    target: LinguisticTarget,
    kind: LinguisticClaimKind,
    status: InterpretationTargetStatus,
    winner: Option<InterpretationClaimView>,
    alternatives: Vec<InterpretationClaimView>,
    option_total: usize,
    options_truncated: bool,
    option_ids: Vec<LinguisticClaimId>,
    resolution: Option<&ClaimResolutionId>,
) -> InterpretationTargetView {
    let mut linked_claim_ids = artifact
        .claims
        .iter()
        .filter(|claim| targets_overlap(&target, &claim.target))
        .map(|claim| claim.id.clone())
        .collect::<Vec<_>>();
    linked_claim_ids.sort();
    linked_claim_ids.dedup();
    let linked_claims_truncated = linked_claim_ids.len() > MAX_LINKED_CLAIMS_PER_TARGET;
    linked_claim_ids.truncate(MAX_LINKED_CLAIMS_PER_TARGET);

    InterpretationTargetView {
        target_id,
        resolution_id,
        acoustic_links: acoustic_links(state),
        consequences: consequences(state, &option_ids, resolution),
        target,
        kind,
        status,
        winner,
        alternatives,
        option_total,
        options_truncated,
        linked_claim_ids,
        linked_claims_truncated,
    }
}

fn claim_view(
    claim: &LinguisticClaim,
    selected: bool,
    conflicts_with_winner: bool,
    resolution_explanation: Option<String>,
) -> InterpretationClaimView {
    InterpretationClaimView {
        claim_id: claim.id.clone(),
        target: claim.target.clone(),
        kind: claim.kind,
        value: claim.value.clone(),
        authority: authority_class(&claim.provenance.source),
        provenance: claim.provenance.clone(),
        confidence: claim.confidence.probability,
        calibration: claim.confidence.calibration.clone(),
        lifecycle: claim.lifecycle,
        selected,
        conflicts_with_winner,
        supports: claim.supports.clone(),
        conflicts_with: claim.conflicts_with.clone(),
        rationale_code: claim.rationale.code.clone(),
        rationale: claim.rationale.summary.clone(),
        resolution_explanation,
    }
}

fn authority_class(source: &EvidenceSource) -> EvidenceAuthorityClass {
    match source {
        EvidenceSource::CommittedAcoustics
        | EvidenceSource::AcousticModel
        | EvidenceSource::ForcedAlignment
        | EvidenceSource::Asr => EvidenceAuthorityClass::AcousticEvidence,
        EvidenceSource::Manual => EvidenceAuthorityClass::DirectObservation,
        EvidenceSource::ManualOverride | EvidenceSource::UserMarkup => {
            EvidenceAuthorityClass::ManualOverride
        }
        EvidenceSource::Grammar => EvidenceAuthorityClass::GrammarInference,
        EvidenceSource::Lexicon | EvidenceSource::Morphology => {
            EvidenceAuthorityClass::LexicalEvidence
        }
        EvidenceSource::Prosody | EvidenceSource::Punctuation => {
            EvidenceAuthorityClass::ProsodyInference
        }
        EvidenceSource::G2p
        | EvidenceSource::Rule
        | EvidenceSource::ImportedData
        | EvidenceSource::TtsPlan => EvidenceAuthorityClass::Fallback,
        EvidenceSource::LearnedPrediction | EvidenceSource::Inference | EvidenceSource::Memory => {
            EvidenceAuthorityClass::ContextualInference
        }
        EvidenceSource::Unknown => EvidenceAuthorityClass::Unknown,
    }
}

fn targets_overlap(left: &LinguisticTarget, right: &LinguisticTarget) -> bool {
    if left.utterance_id != right.utterance_id {
        return false;
    }
    match (&left.text_range, &right.text_range) {
        (Some(left), Some(right)) => ranges_overlap(left, right),
        _ => left.scope == right.scope,
    }
}

fn ranges_overlap(left: &TextRange, right: &TextRange) -> bool {
    left.start < right.end && right.start < left.end
}

fn acoustic_links(state: &SimulatorState) -> Vec<AcousticEvidenceLink> {
    state
        .evidence
        .values()
        .filter_map(|evidence| {
            evidence
                .acoustic_span
                .as_ref()
                .map(|span| AcousticEvidenceLink {
                    evidence_id: evidence.id.clone(),
                    transcript: evidence.content.clone(),
                    span: span.clone(),
                    alignment: "utterance_or_chunk_span".into(),
                })
        })
        .collect()
}

fn consequences(
    state: &SimulatorState,
    claim_ids: &[LinguisticClaimId],
    resolution_id: Option<&ClaimResolutionId>,
) -> Vec<InterpretationConsequenceView> {
    let claim_ids = claim_ids.iter().collect::<BTreeSet<_>>();
    state
        .hypotheses
        .values()
        .filter(|hypothesis| {
            hypothesis
                .claim_ids
                .iter()
                .any(|claim_id| claim_ids.contains(claim_id))
                || resolution_id
                    .is_some_and(|resolution_id| hypothesis.resolution_ids.contains(resolution_id))
        })
        .map(|hypothesis| {
            let audit = state
                .hypothesis_audit
                .get(&hypothesis.id)
                .cloned()
                .unwrap_or_default();
            let mut statuses = audit.iter().map(|entry| entry.status).collect::<Vec<_>>();
            statuses.dedup();
            let block_reasons = state
                .rankings
                .iter()
                .find(|ranking| ranking.id == hypothesis.id)
                .map(|ranking| ranking.block_reasons.clone())
                .unwrap_or_default();
            let deliveries = state
                .deliveries
                .values()
                .filter(|delivery| delivery.hypothesis_id == hypothesis.id)
                .cloned()
                .collect();
            InterpretationConsequenceView {
                hypothesis_id: hypothesis.id.0.clone(),
                selected: state.selected_hypotheses.contains(&hypothesis.id),
                statuses,
                output_text: hypothesis
                    .morphemes
                    .iter()
                    .map(|morpheme| morpheme.surface.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
                score: hypothesis.score.clone(),
                block_reasons,
                deliveries,
            }
        })
        .collect()
}

fn backend_reports(state: &SimulatorState) -> (Vec<GrammarBackendInspection>, bool) {
    let mut reports = state
        .hypotheses
        .values()
        .filter_map(|hypothesis| {
            hypothesis
                .syntax
                .as_ref()
                .map(|syntax| GrammarBackendInspection {
                    hypothesis_id: hypothesis.id.0.clone(),
                    status: syntax.status,
                    diagnostic: syntax.diagnostic.clone(),
                    report: syntax.backend_report.clone(),
                    parse_alternatives: syntax.ranked_parses.clone(),
                })
        })
        .collect::<Vec<_>>();
    reports.sort_by(|left, right| left.hypothesis_id.cmp(&right.hypothesis_id));
    let truncated = reports.len() > MAX_BACKEND_REPORTS;
    reports.truncate(MAX_BACKEND_REPORTS);
    (reports, truncated)
}

fn projection_losses(state: &SimulatorState) -> Vec<ProjectionLossInspection> {
    state
        .verifications
        .iter()
        .enumerate()
        .flat_map(|(verification_index, verification)| {
            verification
                .evidence
                .iter()
                .filter(|evidence| evidence.accepted_projection_loss)
                .cloned()
                .map(move |evidence| ProjectionLossInspection {
                    verification_index,
                    evidence,
                })
        })
        .collect()
}
