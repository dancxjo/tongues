//! Evidence-preserving phone alignment.
//!
//! Schema v2 extends the v1 phonetic-segmentation artifact rather than
//! replacing its timebase or timeline attachment. The engine consumes acoustic
//! frame posteriors through a backend-neutral CTC contract, retains competing
//! pronunciation paths, and represents boundary uncertainty explicitly.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    atomic::{AtomicBool, Ordering as AtomicOrdering},
    Arc,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use speaking::timeline::{
    AlignmentKind, SpanModality, SpeechTimelineSession, TimelineAlignment, TimelineAttachment,
    TimelineAttachmentKind, TimelineSpan,
};

use crate::{
    invalid, AlignmentSourceIdentity, AudioBuffer, FrameInterval, PhoneticBoundaryOrigin,
    PhoneticSegmentArtifact, PhoneticSegmentationContext, Result, SegmentKind,
};

pub const PHONE_ALIGNMENT_SCHEMA_VERSION: u32 = 2;
pub const PHONE_ALIGNMENT_ALGORITHM_VERSION: &str = "tongues.phone-alignment.ctc-lattice-v2";
const NEG_INFINITY: f64 = f64::NEG_INFINITY;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentMode {
    Unconstrained,
    TranscriptConstrained,
    PronunciationConstrained,
    SynthesisKnown,
    Imported,
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimingAuthority {
    ForcedAlignment,
    RecognitionDerived,
    SynthesisKnown,
    ImportedAnnotation,
    ManualCorrection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentLifecycle {
    Proposed,
    Provisional,
    Stable,
    Revised,
    Invalidated,
    Committed,
    Corrected,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentUnitRelation {
    Match,
    Insertion,
    Deletion,
    Substitution,
    Silence,
    NonSpeech,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionKind {
    Realizes,
    Contains,
    AlignedTo,
    Overlaps,
    Supports,
    ConflictsWith,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionLoss {
    Lossless,
    ManyToOne,
    OneToMany,
    Approximate,
    Unaligned,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioAlignmentInput {
    pub artifact_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_sha256: Option<String>,
    pub channel: u16,
    #[serde(default)]
    pub selected_regions: Vec<FrameInterval>,
    #[serde(default)]
    pub preprocessing_artifacts: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptToken {
    pub id: String,
    pub text: String,
    pub language_tag: String,
    #[serde(default)]
    pub normalized_from: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptLattice {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supplied_text: Option<String>,
    pub paths: Vec<Vec<TranscriptToken>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignmentUnitSpec {
    pub id: String,
    pub symbol: String,
    pub kind: SegmentKind,
    pub language_tag: String,
    pub inventory_id: String,
    #[serde(default)]
    pub utterance_ids: Vec<String>,
    #[serde(default)]
    pub transcript_token_ids: Vec<String>,
    #[serde(default)]
    pub word_ids: Vec<String>,
    #[serde(default)]
    pub morpheme_ids: Vec<String>,
    #[serde(default)]
    pub syllable_ids: Vec<String>,
    #[serde(default)]
    pub phoneme_ids: Vec<String>,
    #[serde(default)]
    pub speaker_span_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PronunciationPath {
    pub id: String,
    pub lexical_source: String,
    pub language_tag: String,
    pub inventory_id: String,
    /// A within-source prior. It is never reported as acoustic evidence.
    pub prior_probability: f64,
    pub units: Vec<AlignmentUnitSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignmentProjection {
    pub from_ids: Vec<String>,
    pub to_ids: Vec<String>,
    pub kind: ProjectionKind,
    pub loss: ProjectionLoss,
    pub provenance: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryHint {
    pub unit_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<BoundaryEstimate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<BoundaryEstimate>,
    pub authority: TimingAuthority,
    pub source: AlignmentSourceIdentity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DurationPrior {
    pub unit_id: String,
    pub mean_frames: f64,
    pub standard_deviation_frames: f64,
    #[serde(default = "default_prior_weight")]
    pub weight: f64,
    pub source: String,
}

fn default_prior_weight() -> f64 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentCorrection {
    pub id: String,
    pub actor: String,
    pub reason: String,
    pub unit_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replacement_symbol: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<BoundaryEstimate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<BoundaryEstimate>,
    #[serde(default)]
    pub supports: Vec<String>,
    #[serde(default)]
    pub conflicts_with: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AlignmentLimits {
    pub top_k: usize,
    pub max_pronunciation_paths: usize,
    pub max_posterior_frames: usize,
    pub max_symbols: usize,
    pub max_lattice_states: usize,
    pub max_lattice_cells: usize,
    pub minimum_path_posterior: f64,
    pub minimum_selection_margin: f64,
    pub insertion_probability: f64,
}

impl Default for AlignmentLimits {
    fn default() -> Self {
        Self {
            top_k: 5,
            max_pronunciation_paths: 64,
            max_posterior_frames: 120_000,
            max_symbols: 4_096,
            max_lattice_states: 8_193,
            max_lattice_cells: 2_000_000,
            minimum_path_posterior: 0.50,
            minimum_selection_margin: 0.05,
            insertion_probability: 0.70,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhoneAlignmentRequest {
    pub schema_version: u32,
    pub mode: AlignmentMode,
    pub audio: AudioAlignmentInput,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript: Option<TranscriptLattice>,
    pub pronunciations: Vec<PronunciationPath>,
    #[serde(default)]
    pub timing_hints: Vec<BoundaryHint>,
    #[serde(default)]
    pub duration_priors: Vec<DurationPrior>,
    #[serde(default)]
    pub corrections: Vec<AlignmentCorrection>,
    #[serde(default)]
    pub projections: Vec<AlignmentProjection>,
    #[serde(default)]
    pub limits: AlignmentLimits,
    pub context: PhoneticSegmentationContext,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CtcPosteriorMatrix {
    pub schema_version: u32,
    pub source: AlignmentSourceIdentity,
    pub language_tags: Vec<String>,
    pub inventory_id: String,
    pub sample_rate_hz: u32,
    pub frame_start: u64,
    pub frame_stride: u64,
    pub frame_width: u64,
    pub blank_index: usize,
    pub symbols: Vec<String>,
    /// Rows are probabilities in `symbols` order. They are normalized by the
    /// adapter after finite/non-negative validation.
    pub probabilities: Vec<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_checksum: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryEstimate {
    pub estimate_frame: u64,
    pub lower_frame: u64,
    pub upper_frame: u64,
    pub coverage_probability: f64,
    pub method: String,
}

impl BoundaryEstimate {
    fn point(frame: u64, method: impl Into<String>) -> Self {
        Self {
            estimate_frame: frame,
            lower_frame: frame,
            upper_frame: frame,
            coverage_probability: 1.0,
            method: method.into(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AlignmentScoreBreakdown {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acoustic_log_likelihood: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pronunciation_log_prior: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_log_prior: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub insertion_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deletion_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction_contribution: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_score: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignedUnit {
    pub id: String,
    pub input_unit_ids: Vec<String>,
    pub symbol: String,
    pub kind: SegmentKind,
    pub relation: AlignmentUnitRelation,
    pub lifecycle: AlignmentLifecycle,
    pub timing_authority: TimingAuthority,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<FrameInterval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_boundary: Option<BoundaryEstimate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_boundary: Option<BoundaryEstimate>,
    pub scores: AlignmentScoreBreakdown,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_probability: Option<f64>,
    /// Calibration record for `presence_probability`. `None` means the backend
    /// supplied normalized support but no measured calibration set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_calibration: Option<String>,
    #[serde(default)]
    pub supports: Vec<String>,
    #[serde(default)]
    pub conflicts_with: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentHypothesis {
    pub id: String,
    pub pronunciation_path_id: String,
    pub rank: usize,
    pub lifecycle: AlignmentLifecycle,
    pub units: Vec<AlignedUnit>,
    pub scores: AlignmentScoreBreakdown,
    pub normalized_path_posterior: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pruning_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignmentBackendCapabilities {
    pub forced_alignment: bool,
    pub recognition_alignment: bool,
    pub synthesis_timing: bool,
    pub imported_timing: bool,
    pub corrections: bool,
    pub streaming: bool,
    pub boundary_uncertainty: bool,
    pub word_output: bool,
    pub syllable_output: bool,
    pub requires_transcript: bool,
    pub expected_sample_rate_hz: u32,
    pub supported_languages: Vec<String>,
    pub supported_inventories: Vec<String>,
    pub resource_cost: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentReadiness {
    Ready,
    Partial,
    Abstained,
    Unsupported,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignmentDiagnostic {
    pub code: String,
    pub detail: String,
    #[serde(default)]
    pub related_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhoneAlignmentArtifact {
    pub schema_version: u32,
    pub algorithm_version: String,
    pub readiness: AlignmentReadiness,
    pub mode: AlignmentMode,
    pub timebase: String,
    pub audio: AudioAlignmentInput,
    pub audio_sha256: String,
    pub request_sha256: String,
    pub audio_frames: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub backend: AlignmentSourceIdentity,
    pub backend_capabilities: AlignmentBackendCapabilities,
    pub hypotheses: Vec<AlignmentHypothesis>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_hypothesis_id: Option<String>,
    #[serde(default)]
    pub projections: Vec<AlignmentProjection>,
    #[serde(default)]
    pub unaligned_audio: Vec<FrameInterval>,
    #[serde(default)]
    pub unaligned_linguistic_ids: Vec<String>,
    #[serde(default)]
    pub diagnostics: Vec<AlignmentDiagnostic>,
    pub context: PhoneticSegmentationContext,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub migrated_from_schema: Option<u32>,
}

impl PhoneAlignmentArtifact {
    pub fn selected_hypothesis(&self) -> Option<&AlignmentHypothesis> {
        let selected = self.selected_hypothesis_id.as_deref()?;
        self.hypotheses
            .iter()
            .find(|hypothesis| hypothesis.id == selected)
    }

    /// Stores the complete lattice as one immutable v2 attachment and projects
    /// only the selected path into timeline spans. Alternatives remain in the
    /// attachment and never masquerade as simultaneous observations.
    pub fn attach_to_timeline(&self, session: &mut SpeechTimelineSession) -> Result<String> {
        if let Some(expected) = self.context.session_id.as_deref() {
            if expected != session.session_id {
                return Err(invalid(format!(
                    "alignment expects timeline session `{expected}`, got `{}`",
                    session.session_id
                )));
            }
        }
        let artifact_id = format!(
            "phone-alignment:{}",
            self.request_sha256
                .strip_prefix("sha256:")
                .unwrap_or(&self.request_sha256)
        );
        if session
            .attachments
            .iter()
            .any(|attachment| attachment.artifact_id == artifact_id)
        {
            return Err(invalid(format!(
                "timeline already contains alignment attachment `{artifact_id}`"
            )));
        }
        let mut next = session.clone();
        next.attachments.push(TimelineAttachment {
            artifact_id: artifact_id.clone(),
            kind: TimelineAttachmentKind::PhoneticSegmentation,
            schema_version: self.schema_version,
            payload: serde_json::to_value(self)
                .map_err(|error| invalid(format!("serializing alignment attachment: {error}")))?,
        });

        let existing = next
            .evidence
            .iter()
            .map(|span| span.id.clone())
            .collect::<BTreeSet<_>>();
        if let Some(hypothesis) = self.selected_hypothesis() {
            for unit in &hypothesis.units {
                let Some(interval) = &unit.interval else {
                    continue;
                };
                let span_id = format!("{artifact_id}:{}", unit.id);
                if existing.contains(&span_id) {
                    return Err(invalid(format!(
                        "alignment span `{span_id}` collides with existing evidence"
                    )));
                }
                let start_ms = frames_to_ms(interval.start_frame, interval.sample_rate_hz);
                let end_ms = frames_to_ms(interval.end_frame, interval.sample_rate_hz)
                    .max(start_ms.saturating_add(1));
                let metadata = BTreeMap::from([
                    ("symbol".into(), serde_json::json!(unit.symbol)),
                    (
                        "segment_kind".into(),
                        serde_json::to_value(unit.kind).unwrap_or_default(),
                    ),
                    (
                        "relation".into(),
                        serde_json::to_value(unit.relation).unwrap_or_default(),
                    ),
                    (
                        "lifecycle".into(),
                        serde_json::to_value(unit.lifecycle).unwrap_or_default(),
                    ),
                    (
                        "timing_authority".into(),
                        serde_json::to_value(unit.timing_authority).unwrap_or_default(),
                    ),
                    (
                        "start_boundary".into(),
                        serde_json::to_value(&unit.start_boundary).unwrap_or_default(),
                    ),
                    (
                        "end_boundary".into(),
                        serde_json::to_value(&unit.end_boundary).unwrap_or_default(),
                    ),
                    (
                        "score_breakdown".into(),
                        serde_json::to_value(&unit.scores).unwrap_or_default(),
                    ),
                    (
                        "path_posterior".into(),
                        serde_json::json!(hypothesis.normalized_path_posterior),
                    ),
                    ("hypothesis_id".into(), serde_json::json!(hypothesis.id)),
                    (
                        "alignment_provider".into(),
                        serde_json::json!(self.backend.provider),
                    ),
                    (
                        "alignment_model".into(),
                        serde_json::json!(self.backend.model),
                    ),
                    (
                        "alignment_version".into(),
                        serde_json::json!(self.backend.version),
                    ),
                    (
                        "algorithm_version".into(),
                        serde_json::json!(self.algorithm_version),
                    ),
                    ("artifact_id".into(), serde_json::json!(artifact_id)),
                    (
                        "audio_artifact_id".into(),
                        serde_json::json!(self.audio.artifact_id),
                    ),
                    (
                        "request_sha256".into(),
                        serde_json::json!(self.request_sha256),
                    ),
                    ("graph_id".into(), serde_json::json!(self.context.graph_id)),
                    (
                        "graph_revision".into(),
                        serde_json::json!(self.context.graph_revision),
                    ),
                    (
                        "recipe_id".into(),
                        serde_json::json!(self.context.recipe_id),
                    ),
                    (
                        "execution_record_id".into(),
                        serde_json::json!(self.context.execution_record_id),
                    ),
                    (
                        "evidence_authority".into(),
                        serde_json::json!("selected_alignment_hypothesis"),
                    ),
                ]);
                next.evidence.push(TimelineSpan {
                    id: span_id.clone(),
                    start_ms,
                    end_ms,
                    modality: match unit.kind {
                        SegmentKind::Phone | SegmentKind::Silence | SegmentKind::Pause => {
                            SpanModality::Phone
                        }
                        _ => SpanModality::Phoneme,
                    },
                    metadata,
                });

                for projection in self.projections.iter().filter(|projection| {
                    projection.from_ids.iter().any(|id| {
                        id == &unit.id || unit.input_unit_ids.iter().any(|input| input == id)
                    })
                }) {
                    let kind = match projection.kind {
                        ProjectionKind::Contains | ProjectionKind::Realizes => {
                            AlignmentKind::Contains
                        }
                        _ => AlignmentKind::AlignedTo,
                    };
                    for target in &projection.to_ids {
                        if !existing.contains(target) {
                            return Err(invalid(format!(
                                "alignment projection target `{target}` is absent from timeline evidence"
                            )));
                        }
                        next.alignments.push(TimelineAlignment {
                            source_span_id: span_id.clone(),
                            target_span_id: target.clone(),
                            kind,
                            confidence: unit.presence_probability.map(|value| value as f32),
                        });
                    }
                }
                if let Some(audio_span) = self.context.audio_span_id.as_deref() {
                    if !existing.contains(audio_span) {
                        return Err(invalid(format!(
                            "alignment audio span `{audio_span}` is absent from timeline evidence"
                        )));
                    }
                    next.alignments.push(TimelineAlignment {
                        source_span_id: span_id,
                        target_span_id: audio_span.into(),
                        kind: AlignmentKind::AlignedTo,
                        confidence: unit.presence_probability.map(|value| value as f32),
                    });
                }
            }
        }
        next.validate()
            .map_err(|error| invalid(format!("attached alignment is invalid: {error}")))?;
        *session = next;
        Ok(artifact_id)
    }

    /// Explicitly migrates a v1 flattened segmentation. Point boundaries remain
    /// marked as legacy, not falsely described as posterior uncertainty.
    pub fn migrate_v1(value: &PhoneticSegmentArtifact) -> Self {
        let hypothesis_id = stable_id("migrated", &value.recipe_sha256);
        let units = value
            .segments
            .iter()
            .map(|segment| {
                let authority = match segment.boundary_origin {
                    Some(PhoneticBoundaryOrigin::Corrected) => TimingAuthority::ManualCorrection,
                    Some(PhoneticBoundaryOrigin::SourceProvided) => {
                        TimingAuthority::ImportedAnnotation
                    }
                    _ => TimingAuthority::ForcedAlignment,
                };
                AlignedUnit {
                    id: format!("legacy:{}", segment.expected_index),
                    input_unit_ids: vec![format!("expected:{}", segment.expected_index)],
                    symbol: segment.symbol.clone(),
                    kind: segment.kind,
                    relation: if segment.interval.is_some() {
                        AlignmentUnitRelation::Match
                    } else {
                        AlignmentUnitRelation::Deletion
                    },
                    lifecycle: AlignmentLifecycle::Stable,
                    timing_authority: authority,
                    interval: segment.interval.clone(),
                    start_boundary: segment.interval.as_ref().map(|interval| {
                        BoundaryEstimate::point(
                            interval.start_frame,
                            "legacy_v1_point_boundary_no_uncertainty",
                        )
                    }),
                    end_boundary: segment.interval.as_ref().map(|interval| {
                        BoundaryEstimate::point(
                            interval.end_frame,
                            "legacy_v1_point_boundary_no_uncertainty",
                        )
                    }),
                    scores: AlignmentScoreBreakdown {
                        backend_score: segment.confidence.map(f64::from),
                        ..Default::default()
                    },
                    presence_probability: segment.confidence.map(f64::from),
                    presence_calibration: None,
                    supports: Vec::new(),
                    conflicts_with: Vec::new(),
                }
            })
            .collect::<Vec<_>>();
        let selected_hypothesis_id = units
            .iter()
            .any(|unit| unit.interval.is_some())
            .then(|| hypothesis_id.clone());
        Self {
            schema_version: PHONE_ALIGNMENT_SCHEMA_VERSION,
            algorithm_version: PHONE_ALIGNMENT_ALGORITHM_VERSION.into(),
            readiness: match value.readiness {
                crate::PhoneticSegmentationReadiness::Ready => AlignmentReadiness::Ready,
                crate::PhoneticSegmentationReadiness::Partial => AlignmentReadiness::Partial,
                crate::PhoneticSegmentationReadiness::Unsupported => {
                    AlignmentReadiness::Unsupported
                }
            },
            mode: AlignmentMode::Hybrid,
            timebase: value.timebase.clone(),
            audio: AudioAlignmentInput {
                artifact_id: value.audio_artifact_id.clone(),
                expected_sha256: Some(value.audio_sha256.clone()),
                channel: 0,
                selected_regions: Vec::new(),
                preprocessing_artifacts: Vec::new(),
            },
            audio_sha256: value.audio_sha256.clone(),
            request_sha256: value.recipe_sha256.clone(),
            audio_frames: value.audio_frames,
            sample_rate_hz: value.sample_rate_hz,
            channels: value.channels,
            backend: AlignmentSourceIdentity {
                provider: "tongues-v1-migration".into(),
                model: value.algorithm_version.clone(),
                version: "1".into(),
                artifact_id: None,
            },
            backend_capabilities: AlignmentBackendCapabilities {
                forced_alignment: true,
                recognition_alignment: false,
                synthesis_timing: false,
                imported_timing: true,
                corrections: true,
                streaming: false,
                boundary_uncertainty: false,
                word_output: false,
                syllable_output: false,
                requires_transcript: false,
                expected_sample_rate_hz: value.sample_rate_hz,
                supported_languages: Vec::new(),
                supported_inventories: Vec::new(),
                resource_cost: "migration_only".into(),
            },
            hypotheses: vec![AlignmentHypothesis {
                id: hypothesis_id,
                pronunciation_path_id: "legacy-v1-expected-sequence".into(),
                rank: 0,
                lifecycle: AlignmentLifecycle::Stable,
                units,
                scores: AlignmentScoreBreakdown::default(),
                normalized_path_posterior: 1.0,
                selection_reason: Some("migrated_selected_v1_path".into()),
                pruning_reason: None,
            }],
            selected_hypothesis_id,
            projections: Vec::new(),
            unaligned_audio: value
                .unaligned_regions
                .iter()
                .map(|region| region.interval.clone())
                .collect(),
            unaligned_linguistic_ids: value
                .segments
                .iter()
                .filter(|segment| segment.interval.is_none())
                .map(|segment| format!("expected:{}", segment.expected_index))
                .collect(),
            diagnostics: vec![AlignmentDiagnostic {
                code: "migration.v1_point_boundaries".into(),
                detail: "v1 point intervals were preserved; no posterior uncertainty was invented"
                    .into(),
                related_ids: Vec::new(),
            }],
            context: value.graph.clone(),
            migrated_from_schema: Some(value.schema_version),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PhoneAlignmentEngine;

#[derive(Debug, Clone, Default)]
pub struct AlignmentCancellation(Arc<AtomicBool>);

impl AlignmentCancellation {
    pub fn cancel(&self) {
        self.0.store(true, AtomicOrdering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(AtomicOrdering::Acquire)
    }
}

pub trait PhoneAlignmentBackend: Send + Sync {
    fn identity(&self) -> &AlignmentSourceIdentity;
    fn capabilities(&self) -> AlignmentBackendCapabilities;
    fn align(
        &self,
        audio: &AudioBuffer,
        request: &PhoneAlignmentRequest,
    ) -> Result<PhoneAlignmentArtifact>;
}

#[derive(Debug, Clone)]
pub struct CtcPosteriorBackend {
    pub posteriors: CtcPosteriorMatrix,
}

impl PhoneAlignmentBackend for CtcPosteriorBackend {
    fn identity(&self) -> &AlignmentSourceIdentity {
        &self.posteriors.source
    }

    fn capabilities(&self) -> AlignmentBackendCapabilities {
        ctc_capabilities(&self.posteriors, None)
    }

    fn align(
        &self,
        audio: &AudioBuffer,
        request: &PhoneAlignmentRequest,
    ) -> Result<PhoneAlignmentArtifact> {
        PhoneAlignmentEngine.align_ctc(audio, request, &self.posteriors)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignmentBackendConformanceReport {
    pub backend: AlignmentSourceIdentity,
    pub passed: bool,
    pub diagnostics: Vec<AlignmentDiagnostic>,
}

pub fn check_alignment_conformance(
    artifact: &PhoneAlignmentArtifact,
) -> AlignmentBackendConformanceReport {
    let mut diagnostics = Vec::new();
    if artifact.schema_version != PHONE_ALIGNMENT_SCHEMA_VERSION {
        diagnostics.push(AlignmentDiagnostic {
            code: "conformance.schema".into(),
            detail: "artifact does not use the canonical schema version".into(),
            related_ids: Vec::new(),
        });
    }
    if artifact.backend.provider.trim().is_empty()
        || artifact.backend.model.trim().is_empty()
        || artifact.backend.version.trim().is_empty()
    {
        diagnostics.push(AlignmentDiagnostic {
            code: "conformance.backend_identity".into(),
            detail: "backend provider, model, and version must be explicit".into(),
            related_ids: Vec::new(),
        });
    }
    let mut hypothesis_ids = BTreeSet::new();
    let mut unit_ids = BTreeSet::new();
    for hypothesis in &artifact.hypotheses {
        if !hypothesis_ids.insert(hypothesis.id.clone()) {
            diagnostics.push(AlignmentDiagnostic {
                code: "conformance.duplicate_hypothesis".into(),
                detail: "hypothesis IDs must be unique".into(),
                related_ids: vec![hypothesis.id.clone()],
            });
        }
        for unit in &hypothesis.units {
            if !unit_ids.insert((hypothesis.id.clone(), unit.id.clone())) {
                diagnostics.push(AlignmentDiagnostic {
                    code: "conformance.duplicate_unit".into(),
                    detail: "unit IDs must be unique within a hypothesis".into(),
                    related_ids: vec![hypothesis.id.clone(), unit.id.clone()],
                });
            }
            for boundary in [&unit.start_boundary, &unit.end_boundary]
                .into_iter()
                .flatten()
            {
                if boundary.lower_frame > boundary.estimate_frame
                    || boundary.estimate_frame > boundary.upper_frame
                    || !boundary.coverage_probability.is_finite()
                    || !(0.0..=1.0).contains(&boundary.coverage_probability)
                {
                    diagnostics.push(AlignmentDiagnostic {
                        code: "conformance.boundary_range".into(),
                        detail: "boundary support must contain its estimate and valid coverage"
                            .into(),
                        related_ids: vec![hypothesis.id.clone(), unit.id.clone()],
                    });
                }
            }
        }
    }
    if artifact
        .selected_hypothesis_id
        .as_ref()
        .is_some_and(|selected| !hypothesis_ids.contains(selected))
    {
        diagnostics.push(AlignmentDiagnostic {
            code: "conformance.selected_path".into(),
            detail: "selected hypothesis is absent from retained hypotheses".into(),
            related_ids: artifact.selected_hypothesis_id.iter().cloned().collect(),
        });
    }
    AlignmentBackendConformanceReport {
        backend: artifact.backend.clone(),
        passed: diagnostics.is_empty(),
        diagnostics,
    }
}

impl PhoneAlignmentEngine {
    pub fn align_ctc(
        &self,
        audio: &AudioBuffer,
        request: &PhoneAlignmentRequest,
        posterior: &CtcPosteriorMatrix,
    ) -> Result<PhoneAlignmentArtifact> {
        self.align_ctc_with_cancellation(audio, request, posterior, None)
    }

    pub fn align_ctc_with_cancellation(
        &self,
        audio: &AudioBuffer,
        request: &PhoneAlignmentRequest,
        posterior: &CtcPosteriorMatrix,
        cancellation: Option<&AlignmentCancellation>,
    ) -> Result<PhoneAlignmentArtifact> {
        validate_request(audio, request, posterior)?;
        let audio_hash = crate::audio_sha256(audio);
        if request
            .audio
            .expected_sha256
            .as_deref()
            .is_some_and(|expected| expected != audio_hash)
        {
            return Err(invalid(format!(
                "audio checksum mismatch for `{}`",
                request.audio.artifact_id
            )));
        }
        let normalized = normalize_posteriors(posterior)?;
        let mut hypotheses = Vec::new();
        let mut diagnostics = Vec::new();
        let paths = request
            .pronunciations
            .iter()
            .take(request.limits.max_pronunciation_paths);
        for path in paths {
            if cancellation.is_some_and(AlignmentCancellation::is_cancelled) {
                return Err(crate::AudioError::Cancelled);
            }
            match align_pronunciation(
                path,
                posterior,
                &normalized,
                request,
                audio.frames() as u64,
                cancellation,
            ) {
                Ok(hypothesis) => hypotheses.push(hypothesis),
                Err(error) => diagnostics.push(AlignmentDiagnostic {
                    code: "path.abstained".into(),
                    detail: error.to_string(),
                    related_ids: vec![path.id.clone()],
                }),
            }
        }
        if request.pronunciations.len() > request.limits.max_pronunciation_paths {
            diagnostics.push(AlignmentDiagnostic {
                code: "limits.pronunciation_paths".into(),
                detail: format!(
                    "{} paths were omitted by the configured limit of {}",
                    request.pronunciations.len() - request.limits.max_pronunciation_paths,
                    request.limits.max_pronunciation_paths
                ),
                related_ids: Vec::new(),
            });
        }
        hypotheses.sort_by(|left, right| {
            right
                .scores
                .backend_score
                .partial_cmp(&left.scores.backend_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.id.cmp(&right.id))
        });
        normalize_hypothesis_posteriors(&mut hypotheses);
        for (rank, hypothesis) in hypotheses.iter_mut().enumerate() {
            hypothesis.rank = rank;
        }

        let selected = select_hypothesis(&hypotheses, &request.limits);
        for hypothesis in &mut hypotheses {
            if Some(hypothesis.id.as_str()) == selected.as_deref() {
                hypothesis.lifecycle = AlignmentLifecycle::Stable;
                hypothesis.selection_reason = Some("best_supported_path".into());
            } else {
                hypothesis.pruning_reason = Some("retained_top_k_not_selected".into());
            }
        }
        if hypotheses.len() > request.limits.top_k {
            let pruned = hypotheses.len() - request.limits.top_k;
            hypotheses.truncate(request.limits.top_k);
            diagnostics.push(AlignmentDiagnostic {
                code: "limits.top_k".into(),
                detail: format!("{pruned} lower-ranked paths were pruned deterministically"),
                related_ids: Vec::new(),
            });
        }
        let selected = selected.filter(|id| hypotheses.iter().any(|path| &path.id == id));
        if selected.is_none() && !hypotheses.is_empty() {
            diagnostics.push(AlignmentDiagnostic {
                code: "selection.abstained".into(),
                detail: "no path met the configured posterior and winner-margin selection policy"
                    .into(),
                related_ids: hypotheses.iter().map(|path| path.id.clone()).collect(),
            });
        }

        let selected_units = selected
            .as_deref()
            .and_then(|id| hypotheses.iter().find(|path| path.id == id))
            .map(|path| path.units.as_slice())
            .unwrap_or(&[]);
        let unaligned_audio =
            unaligned_intervals(selected_units, audio.frames() as u64, audio.sample_rate_hz);
        let aligned_inputs = selected_units
            .iter()
            .filter(|unit| unit.interval.is_some())
            .flat_map(|unit| unit.input_unit_ids.iter().cloned())
            .collect::<BTreeSet<_>>();
        let selected_path = selected
            .as_deref()
            .and_then(|id| hypotheses.iter().find(|path| path.id == id))
            .and_then(|hypothesis| {
                request
                    .pronunciations
                    .iter()
                    .find(|path| path.id == hypothesis.pronunciation_path_id)
            });
        let mut linguistic_ids = request
            .transcript
            .iter()
            .flat_map(|lattice| lattice.paths.iter())
            .flatten()
            .map(|token| token.id.clone())
            .collect::<BTreeSet<_>>();
        if let Some(path) = selected_path {
            for unit in &path.units {
                linguistic_ids.insert(unit.id.clone());
                for id in unit
                    .utterance_ids
                    .iter()
                    .chain(&unit.transcript_token_ids)
                    .chain(&unit.word_ids)
                    .chain(&unit.morpheme_ids)
                    .chain(&unit.syllable_ids)
                    .chain(&unit.phoneme_ids)
                {
                    linguistic_ids.insert(id.clone());
                }
                if aligned_inputs.contains(&unit.id) {
                    linguistic_ids.remove(&unit.id);
                    for id in unit
                        .utterance_ids
                        .iter()
                        .chain(&unit.transcript_token_ids)
                        .chain(&unit.word_ids)
                        .chain(&unit.morpheme_ids)
                        .chain(&unit.syllable_ids)
                        .chain(&unit.phoneme_ids)
                    {
                        linguistic_ids.remove(id);
                    }
                } else {
                    linguistic_ids.extend(unit.transcript_token_ids.iter().cloned());
                }
            }
        }
        let unaligned_linguistic_ids = linguistic_ids.into_iter().collect();
        let request_sha256 = sha256_json(request)?;
        let readiness = if selected.is_none() {
            if hypotheses.is_empty() {
                AlignmentReadiness::Unsupported
            } else {
                AlignmentReadiness::Abstained
            }
        } else if selected_units
            .iter()
            .any(|unit| unit.relation != AlignmentUnitRelation::Match || unit.interval.is_none())
            || !diagnostics.is_empty()
        {
            AlignmentReadiness::Partial
        } else {
            AlignmentReadiness::Ready
        };
        Ok(PhoneAlignmentArtifact {
            schema_version: PHONE_ALIGNMENT_SCHEMA_VERSION,
            algorithm_version: PHONE_ALIGNMENT_ALGORITHM_VERSION.into(),
            readiness,
            mode: request.mode,
            timebase:
                "half-open original-audio frames; boundary ranges are inclusive credible supports"
                    .into(),
            audio: request.audio.clone(),
            audio_sha256: audio_hash,
            request_sha256,
            audio_frames: audio.frames() as u64,
            sample_rate_hz: audio.sample_rate_hz,
            channels: audio.channels,
            backend: posterior.source.clone(),
            backend_capabilities: ctc_capabilities(posterior, Some(request)),
            hypotheses,
            selected_hypothesis_id: selected,
            projections: effective_projections(request),
            unaligned_audio,
            unaligned_linguistic_ids,
            diagnostics,
            context: request.context.clone(),
            migrated_from_schema: None,
        })
    }
}

fn ctc_capabilities(
    posterior: &CtcPosteriorMatrix,
    request: Option<&PhoneAlignmentRequest>,
) -> AlignmentBackendCapabilities {
    AlignmentBackendCapabilities {
        forced_alignment: true,
        recognition_alignment: true,
        synthesis_timing: false,
        imported_timing: false,
        corrections: true,
        streaming: true,
        boundary_uncertainty: true,
        word_output: request.is_some_and(|request| {
            request
                .pronunciations
                .iter()
                .any(|path| path.units.iter().any(|unit| !unit.word_ids.is_empty()))
        }),
        syllable_output: request.is_some_and(|request| {
            request
                .pronunciations
                .iter()
                .any(|path| path.units.iter().any(|unit| !unit.syllable_ids.is_empty()))
        }),
        requires_transcript: false,
        expected_sample_rate_hz: posterior.sample_rate_hz,
        supported_languages: posterior.language_tags.clone(),
        supported_inventories: vec![posterior.inventory_id.clone()],
        resource_cost: "bounded_ctc_trellis".into(),
    }
}

fn validate_request(
    audio: &AudioBuffer,
    request: &PhoneAlignmentRequest,
    posterior: &CtcPosteriorMatrix,
) -> Result<()> {
    audio.validate()?;
    if audio.frames() == 0 {
        return Err(invalid("phone alignment requires non-empty audio"));
    }
    if request.schema_version != PHONE_ALIGNMENT_SCHEMA_VERSION {
        return Err(invalid(format!(
            "phone alignment request schema {} is unsupported; expected {}",
            request.schema_version, PHONE_ALIGNMENT_SCHEMA_VERSION
        )));
    }
    if posterior.schema_version != 1 {
        return Err(invalid(format!(
            "CTC posterior schema {} is unsupported; expected 1",
            posterior.schema_version
        )));
    }
    if request.audio.channel >= audio.channels {
        return Err(invalid(format!(
            "requested channel {} is outside {}-channel audio",
            request.audio.channel, audio.channels
        )));
    }
    if request.audio.artifact_id.trim().is_empty()
        || request.context.graph_id.trim().is_empty()
        || request.context.recipe_id.trim().is_empty()
        || request.context.execution_record_id.trim().is_empty()
        || request.context.runtime.trim().is_empty()
        || request.context.runtime_version.trim().is_empty()
    {
        return Err(invalid(
            "audio, graph, recipe, execution, and runtime identities must be explicit",
        ));
    }
    for region in &request.audio.selected_regions {
        if region.sample_rate_hz != audio.sample_rate_hz
            || region.start_frame >= region.end_frame
            || region.end_frame > audio.frames() as u64
        {
            return Err(invalid(
                "selected audio regions must be valid intervals in the original timebase",
            ));
        }
    }
    if posterior.sample_rate_hz != audio.sample_rate_hz {
        return Err(invalid(format!(
            "posterior sample rate {} does not match audio sample rate {}",
            posterior.sample_rate_hz, audio.sample_rate_hz
        )));
    }
    if posterior.frame_stride == 0 || posterior.frame_width == 0 {
        return Err(invalid("posterior frame stride and width must be positive"));
    }
    if posterior.source.provider.trim().is_empty()
        || posterior.source.model.trim().is_empty()
        || posterior.source.version.trim().is_empty()
        || posterior.inventory_id.trim().is_empty()
    {
        return Err(invalid(
            "posterior backend and inventory identities must be explicit",
        ));
    }
    if posterior.blank_index >= posterior.symbols.len() {
        return Err(invalid(
            "posterior blank index is outside the symbol vocabulary",
        ));
    }
    if posterior.probabilities.is_empty() {
        return Err(invalid("CTC posterior matrix has no frames"));
    }
    let last_evidence_start = posterior.frame_start.saturating_add(
        (posterior.probabilities.len().saturating_sub(1) as u64)
            .saturating_mul(posterior.frame_stride),
    );
    if last_evidence_start >= audio.frames() as u64 {
        return Err(invalid(format!(
            "posterior evidence starts at frame {last_evidence_start}, beyond {} audio frames",
            audio.frames()
        )));
    }
    if !request.audio.selected_regions.is_empty()
        && !request.audio.selected_regions.iter().any(|region| {
            posterior.frame_start >= region.start_frame && last_evidence_start < region.end_frame
        })
    {
        return Err(invalid(
            "posterior evidence must be contained in one selected audio region",
        ));
    }
    let unique_symbols = posterior.symbols.iter().collect::<BTreeSet<_>>();
    if unique_symbols.len() != posterior.symbols.len()
        || posterior
            .symbols
            .iter()
            .any(|symbol| symbol.trim().is_empty())
    {
        return Err(invalid(
            "posterior vocabulary symbols must be non-empty and unique",
        ));
    }
    if posterior.probabilities.len() > request.limits.max_posterior_frames {
        return Err(invalid(format!(
            "CTC posterior matrix has {} frames, above the configured limit {}",
            posterior.probabilities.len(),
            request.limits.max_posterior_frames
        )));
    }
    if posterior.symbols.len() > request.limits.max_symbols {
        return Err(invalid(
            "CTC vocabulary exceeds the configured symbol limit",
        ));
    }
    let max_units = request
        .pronunciations
        .iter()
        .map(|path| path.units.len())
        .max()
        .unwrap_or(0);
    if max_units.saturating_mul(2).saturating_add(1) > request.limits.max_lattice_states {
        return Err(invalid("CTC lattice exceeds the configured state limit"));
    }
    let lattice_states = max_units.saturating_mul(2).saturating_add(1);
    if posterior
        .probabilities
        .len()
        .checked_mul(lattice_states)
        .is_none_or(|cells| cells > request.limits.max_lattice_cells)
    {
        return Err(invalid(format!(
            "CTC lattice exceeds the configured {}-cell memory limit",
            request.limits.max_lattice_cells
        )));
    }
    if request.pronunciations.is_empty() {
        return Err(invalid(
            "phone alignment requires at least one pronunciation path",
        ));
    }
    if request.limits.top_k == 0 || request.limits.max_pronunciation_paths == 0 {
        return Err(invalid("alignment path limits must be positive"));
    }
    for path in &request.pronunciations {
        if path.id.trim().is_empty()
            || path.lexical_source.trim().is_empty()
            || path.inventory_id.trim().is_empty()
            || !path.prior_probability.is_finite()
            || !(0.0..=1.0).contains(&path.prior_probability)
        {
            return Err(invalid(format!(
                "pronunciation path `{}` has invalid identity, inventory, or prior",
                path.id
            )));
        }
        let mut ids = BTreeSet::new();
        for unit in &path.units {
            if unit.id.trim().is_empty()
                || unit.symbol.trim().is_empty()
                || !ids.insert(unit.id.clone())
            {
                return Err(invalid(format!(
                    "pronunciation path `{}` has empty or duplicate unit identity",
                    path.id
                )));
            }
        }
    }
    for prior in &request.duration_priors {
        if prior.unit_id.trim().is_empty()
            || prior.source.trim().is_empty()
            || !prior.mean_frames.is_finite()
            || prior.mean_frames < 0.0
            || !prior.standard_deviation_frames.is_finite()
            || prior.standard_deviation_frames <= 0.0
            || !prior.weight.is_finite()
            || prior.weight < 0.0
        {
            return Err(invalid(format!(
                "duration prior for `{}` is invalid",
                prior.unit_id
            )));
        }
    }
    Ok(())
}

fn normalize_posteriors(posterior: &CtcPosteriorMatrix) -> Result<Vec<Vec<f64>>> {
    posterior
        .probabilities
        .iter()
        .enumerate()
        .map(|(frame, row)| {
            if row.len() != posterior.symbols.len() {
                return Err(invalid(format!(
                    "posterior frame {frame} has {} classes; expected {}",
                    row.len(),
                    posterior.symbols.len()
                )));
            }
            if row.iter().any(|value| !value.is_finite() || *value < 0.0) {
                return Err(invalid(format!(
                    "posterior frame {frame} contains a negative or non-finite probability"
                )));
            }
            let sum = row.iter().sum::<f64>();
            if sum <= 0.0 {
                return Err(invalid(format!(
                    "posterior frame {frame} has zero probability mass"
                )));
            }
            Ok(row.iter().map(|value| value / sum).collect())
        })
        .collect()
}

fn align_pronunciation(
    path: &PronunciationPath,
    posterior: &CtcPosteriorMatrix,
    probabilities: &[Vec<f64>],
    request: &PhoneAlignmentRequest,
    audio_frames: u64,
    cancellation: Option<&AlignmentCancellation>,
) -> Result<AlignmentHypothesis> {
    if path.units.is_empty() {
        return Err(invalid(format!(
            "pronunciation path `{}` has no units",
            path.id
        )));
    }
    if path.inventory_id != posterior.inventory_id {
        return Err(invalid(format!(
            "pronunciation path `{}` uses inventory `{}` but backend supplies `{}`",
            path.id, path.inventory_id, posterior.inventory_id
        )));
    }
    if !posterior.language_tags.is_empty()
        && path.language_tag != "mul"
        && !posterior
            .language_tags
            .iter()
            .any(|language| language.eq_ignore_ascii_case(&path.language_tag))
    {
        return Err(invalid(format!(
            "pronunciation path `{}` language `{}` is unsupported by the backend",
            path.id, path.language_tag
        )));
    }
    let symbol_ids = posterior
        .symbols
        .iter()
        .enumerate()
        .map(|(index, symbol)| (symbol.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let target = path
        .units
        .iter()
        .map(|unit| {
            symbol_ids
                .get(unit.symbol.as_str())
                .copied()
                .ok_or_else(|| {
                    invalid(format!(
                        "path `{}` symbol `{}` is absent from backend inventory",
                        path.id, unit.symbol
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let trellis = ctc_trellis(probabilities, &target, posterior.blank_index, cancellation)?;
    let total_frames = probabilities.len().max(1) as f64;
    let acoustic = trellis.best_log_likelihood / total_frames;
    let pronunciation_prior = path.prior_probability.max(1.0e-12).ln();
    let mut duration_score = 0.0;
    let mut units = Vec::new();
    for (index, spec) in path.units.iter().enumerate() {
        let assigned = trellis
            .state_path
            .iter()
            .enumerate()
            .filter_map(|(frame, state)| (*state == index * 2 + 1).then_some(frame))
            .collect::<Vec<_>>();
        let occupancy = trellis
            .occupancy
            .iter()
            .map(|row| row[index * 2 + 1])
            .collect::<Vec<_>>();
        let (interval, start_boundary, end_boundary, relation) =
            if let (Some(first), Some(last)) = (assigned.first(), assigned.last()) {
                let start = posterior
                    .frame_start
                    .saturating_add(*first as u64 * posterior.frame_stride);
                let end = posterior
                    .frame_start
                    .saturating_add(*last as u64 * posterior.frame_stride)
                    .saturating_add(posterior.frame_width)
                    .min(audio_frames);
                let (lower, median, upper) = occupancy_quantiles(&occupancy);
                let lower_frame = posterior
                    .frame_start
                    .saturating_add(lower as u64 * posterior.frame_stride);
                let median_frame = posterior
                    .frame_start
                    .saturating_add(median as u64 * posterior.frame_stride);
                let upper_frame = posterior
                    .frame_start
                    .saturating_add(upper as u64 * posterior.frame_stride)
                    .saturating_add(posterior.frame_width)
                    .min(audio_frames);
                (
                    Some(FrameInterval {
                        start_frame: start,
                        end_frame: end,
                        sample_rate_hz: posterior.sample_rate_hz,
                    }),
                    Some(BoundaryEstimate {
                        estimate_frame: start,
                        lower_frame: lower_frame.min(start),
                        upper_frame: median_frame.max(start),
                        coverage_probability: 0.90,
                        method: "ctc_forward_backward_credible_support".into(),
                    }),
                    Some(BoundaryEstimate {
                        estimate_frame: end,
                        lower_frame: median_frame.min(end),
                        upper_frame: upper_frame.max(end),
                        coverage_probability: 0.90,
                        method: "ctc_forward_backward_credible_support".into(),
                    }),
                    if spec.kind == SegmentKind::Silence || spec.kind == SegmentKind::Pause {
                        AlignmentUnitRelation::Silence
                    } else {
                        AlignmentUnitRelation::Match
                    },
                )
            } else {
                (None, None, None, AlignmentUnitRelation::Deletion)
            };
        let presence = occupancy.iter().copied().fold(0.0_f64, f64::max);
        let mut unit = AlignedUnit {
            id: stable_id(&path.id, &spec.id),
            input_unit_ids: vec![spec.id.clone()],
            symbol: spec.symbol.clone(),
            kind: spec.kind,
            relation,
            lifecycle: AlignmentLifecycle::Proposed,
            timing_authority: match request.mode {
                AlignmentMode::SynthesisKnown => TimingAuthority::SynthesisKnown,
                AlignmentMode::Imported => TimingAuthority::ImportedAnnotation,
                AlignmentMode::Unconstrained => TimingAuthority::RecognitionDerived,
                _ => TimingAuthority::ForcedAlignment,
            },
            interval,
            start_boundary,
            end_boundary,
            scores: AlignmentScoreBreakdown {
                acoustic_log_likelihood: Some(acoustic),
                pronunciation_log_prior: Some(pronunciation_prior),
                deletion_penalty: (relation == AlignmentUnitRelation::Deletion).then_some(-8.0),
                ..Default::default()
            },
            presence_probability: Some(presence.clamp(0.0, 1.0)),
            presence_calibration: None,
            supports: Vec::new(),
            conflicts_with: Vec::new(),
        };
        if let (Some(interval), Some(prior)) = (
            unit.interval.as_ref(),
            request
                .duration_priors
                .iter()
                .find(|prior| prior.unit_id == spec.id),
        ) {
            let observed = interval.end_frame.saturating_sub(interval.start_frame) as f64;
            let z = (observed - prior.mean_frames) / prior.standard_deviation_frames;
            let score = -0.5 * z * z * prior.weight;
            unit.scores.duration_log_prior = Some(score);
            duration_score += score;
            unit.supports.push(prior.source.clone());
        }
        apply_hints_and_corrections(&mut unit, spec, request, posterior.sample_rate_hz);
        units.push(unit);
    }
    let insertions = detect_insertions(
        path,
        posterior,
        probabilities,
        &trellis.state_path,
        &target,
        request.limits.insertion_probability,
        audio_frames,
    );
    let insertion_score = -4.0 * insertions.len() as f64;
    units.extend(insertions);
    let correction_score = units
        .iter()
        .filter_map(|unit| unit.scores.correction_contribution)
        .sum::<f64>();
    let backend_score = acoustic
        + (pronunciation_prior + duration_score + insertion_score + correction_score)
            / total_frames;
    for unit in &mut units {
        unit.scores.backend_score = Some(backend_score);
    }
    Ok(AlignmentHypothesis {
        id: stable_id("hypothesis", &path.id),
        pronunciation_path_id: path.id.clone(),
        rank: 0,
        lifecycle: AlignmentLifecycle::Proposed,
        units,
        scores: AlignmentScoreBreakdown {
            acoustic_log_likelihood: Some(acoustic),
            pronunciation_log_prior: Some(pronunciation_prior),
            duration_log_prior: Some(duration_score),
            insertion_penalty: (insertion_score != 0.0).then_some(insertion_score),
            correction_contribution: (correction_score != 0.0).then_some(correction_score),
            backend_score: Some(backend_score),
            ..Default::default()
        },
        normalized_path_posterior: 0.0,
        selection_reason: None,
        pruning_reason: None,
    })
}

fn apply_hints_and_corrections(
    unit: &mut AlignedUnit,
    spec: &AlignmentUnitSpec,
    request: &PhoneAlignmentRequest,
    sample_rate_hz: u32,
) {
    for hint in request
        .timing_hints
        .iter()
        .filter(|hint| hint.unit_id == spec.id)
    {
        unit.supports
            .push(hint.source.artifact_id.clone().unwrap_or_else(|| {
                format!(
                    "{}:{}:{}",
                    hint.source.provider, hint.source.model, hint.source.version
                )
            }));
        if matches!(
            request.mode,
            AlignmentMode::SynthesisKnown | AlignmentMode::Imported
        ) {
            unit.start_boundary = hint.start.clone().or(unit.start_boundary.clone());
            unit.end_boundary = hint.end.clone().or(unit.end_boundary.clone());
            unit.timing_authority = hint.authority;
            if let (Some(start), Some(end)) = (&unit.start_boundary, &unit.end_boundary) {
                if start.estimate_frame < end.estimate_frame {
                    unit.interval = Some(FrameInterval {
                        start_frame: start.estimate_frame,
                        end_frame: end.estimate_frame,
                        sample_rate_hz: unit
                            .interval
                            .as_ref()
                            .map(|interval| interval.sample_rate_hz)
                            .unwrap_or(sample_rate_hz),
                    });
                }
            }
        }
    }
    for correction in request
        .corrections
        .iter()
        .filter(|correction| correction.unit_id == spec.id)
    {
        if let Some(symbol) = &correction.replacement_symbol {
            unit.symbol = symbol.clone();
        }
        unit.start_boundary = correction.start.clone().or(unit.start_boundary.clone());
        unit.end_boundary = correction.end.clone().or(unit.end_boundary.clone());
        unit.lifecycle = AlignmentLifecycle::Corrected;
        unit.timing_authority = TimingAuthority::ManualCorrection;
        unit.supports.push(correction.id.clone());
        unit.supports.extend(correction.supports.clone());
        unit.conflicts_with
            .extend(correction.conflicts_with.clone());
        unit.scores.correction_contribution = Some(1.0);
        if let (Some(start), Some(end)) = (&unit.start_boundary, &unit.end_boundary) {
            if start.estimate_frame < end.estimate_frame {
                unit.interval = Some(FrameInterval {
                    start_frame: start.estimate_frame,
                    end_frame: end.estimate_frame,
                    sample_rate_hz,
                });
            }
        }
    }
}

fn detect_insertions(
    path: &PronunciationPath,
    posterior: &CtcPosteriorMatrix,
    probabilities: &[Vec<f64>],
    state_path: &[usize],
    target: &[usize],
    threshold: f64,
    audio_frames: u64,
) -> Vec<AlignedUnit> {
    let target = target.iter().copied().collect::<BTreeSet<_>>();
    let mut groups = Vec::<(usize, usize, usize, f64)>::new();
    for (frame, row) in probabilities.iter().enumerate() {
        let blank_state = state_path.get(frame).is_some_and(|state| state % 2 == 0);
        if !blank_state {
            continue;
        }
        let Some((symbol, probability)) = row
            .iter()
            .copied()
            .enumerate()
            .filter(|(symbol, _)| *symbol != posterior.blank_index && !target.contains(symbol))
            .max_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal))
        else {
            continue;
        };
        if probability < threshold {
            continue;
        }
        if let Some(last) = groups.last_mut() {
            if last.2 == symbol && last.1 + 1 == frame {
                last.1 = frame;
                last.3 = last.3.max(probability);
                continue;
            }
        }
        groups.push((frame, frame, symbol, probability));
    }
    groups
        .into_iter()
        .enumerate()
        .map(|(index, (first, last, symbol_index, probability))| {
            let start = posterior
                .frame_start
                .saturating_add((first as u64).saturating_mul(posterior.frame_stride));
            let end = posterior
                .frame_start
                .saturating_add((last as u64).saturating_mul(posterior.frame_stride))
                .saturating_add(posterior.frame_width);
            let end = end.min(audio_frames);
            AlignedUnit {
                id: stable_id(&path.id, &format!("insertion:{index}:{symbol_index}")),
                input_unit_ids: Vec::new(),
                symbol: posterior.symbols[symbol_index].clone(),
                kind: SegmentKind::Phone,
                relation: AlignmentUnitRelation::Insertion,
                lifecycle: AlignmentLifecycle::Proposed,
                timing_authority: TimingAuthority::RecognitionDerived,
                interval: Some(FrameInterval {
                    start_frame: start,
                    end_frame: end,
                    sample_rate_hz: posterior.sample_rate_hz,
                }),
                start_boundary: Some(BoundaryEstimate {
                    estimate_frame: start,
                    lower_frame: start.saturating_sub(posterior.frame_stride),
                    upper_frame: start.saturating_add(posterior.frame_stride),
                    coverage_probability: probability,
                    method: "ctc_competing_non_target_support".into(),
                }),
                end_boundary: Some(BoundaryEstimate {
                    estimate_frame: end,
                    lower_frame: end.saturating_sub(posterior.frame_stride),
                    upper_frame: end.saturating_add(posterior.frame_stride),
                    coverage_probability: probability,
                    method: "ctc_competing_non_target_support".into(),
                }),
                scores: AlignmentScoreBreakdown {
                    acoustic_log_likelihood: Some(probability.max(1.0e-12).ln()),
                    insertion_penalty: Some(-4.0),
                    backend_score: Some(probability),
                    ..Default::default()
                },
                presence_probability: Some(probability),
                presence_calibration: None,
                supports: vec![format!(
                    "{}:{}:{}",
                    posterior.source.provider, posterior.source.model, posterior.source.version
                )],
                conflicts_with: Vec::new(),
            }
        })
        .collect()
}

struct CtcTrellis {
    best_log_likelihood: f64,
    state_path: Vec<usize>,
    occupancy: Vec<Vec<f64>>,
}

fn ctc_trellis(
    probabilities: &[Vec<f64>],
    target: &[usize],
    blank: usize,
    cancellation: Option<&AlignmentCancellation>,
) -> Result<CtcTrellis> {
    let frames = probabilities.len();
    let states = target.len() * 2 + 1;
    if frames < target.len() {
        return Err(invalid(format!(
            "CTC evidence has {frames} frames for {} required symbols",
            target.len()
        )));
    }
    let labels = (0..states)
        .map(|state| {
            if state % 2 == 0 {
                blank
            } else {
                target[state / 2]
            }
        })
        .collect::<Vec<_>>();
    let logp = probabilities
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| value.max(1.0e-12).ln())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut viterbi = vec![vec![NEG_INFINITY; states]; frames];
    let mut back = vec![vec![0usize; states]; frames];
    viterbi[0][0] = logp[0][blank];
    if states > 1 {
        viterbi[0][1] = logp[0][labels[1]];
    }
    for time in 1..frames {
        if time % 256 == 0 && cancellation.is_some_and(AlignmentCancellation::is_cancelled) {
            return Err(crate::AudioError::Cancelled);
        }
        for state in 0..states {
            let mut options = vec![(viterbi[time - 1][state], state)];
            if state > 0 {
                options.push((viterbi[time - 1][state - 1], state - 1));
            }
            if state > 1
                && labels[state] != blank
                && labels[state] != labels[state.saturating_sub(2)]
            {
                options.push((viterbi[time - 1][state - 2], state - 2));
            }
            let (best, predecessor) = options
                .into_iter()
                .max_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal))
                .unwrap();
            viterbi[time][state] = best + logp[time][labels[state]];
            back[time][state] = predecessor;
        }
    }
    let final_states = if states == 1 {
        vec![0]
    } else {
        vec![states - 1, states - 2]
    };
    let (best, mut state) = final_states
        .into_iter()
        .map(|state| (viterbi[frames - 1][state], state))
        .max_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal))
        .unwrap();
    if !best.is_finite() {
        return Err(invalid(
            "no valid CTC path reaches the supplied pronunciation",
        ));
    }
    let mut state_path = vec![0usize; frames];
    for time in (0..frames).rev() {
        state_path[time] = state;
        if time > 0 {
            state = back[time][state];
        }
    }

    let mut forward = vec![vec![NEG_INFINITY; states]; frames];
    forward[0][0] = logp[0][blank];
    if states > 1 {
        forward[0][1] = logp[0][labels[1]];
    }
    for time in 1..frames {
        if time % 256 == 0 && cancellation.is_some_and(AlignmentCancellation::is_cancelled) {
            return Err(crate::AudioError::Cancelled);
        }
        for state in 0..states {
            let mut incoming = vec![forward[time - 1][state]];
            if state > 0 {
                incoming.push(forward[time - 1][state - 1]);
            }
            if state > 1
                && labels[state] != blank
                && labels[state] != labels[state.saturating_sub(2)]
            {
                incoming.push(forward[time - 1][state - 2]);
            }
            forward[time][state] = log_sum_exp(&incoming) + logp[time][labels[state]];
        }
    }
    let log_z = if states == 1 {
        forward[frames - 1][0]
    } else {
        log_sum_exp(&[
            forward[frames - 1][states - 1],
            forward[frames - 1][states - 2],
        ])
    };
    let mut backward = vec![vec![NEG_INFINITY; states]; frames];
    backward[frames - 1][states - 1] = 0.0;
    if states > 1 {
        backward[frames - 1][states - 2] = 0.0;
    }
    for time in (0..frames - 1).rev() {
        if time % 256 == 0 && cancellation.is_some_and(AlignmentCancellation::is_cancelled) {
            return Err(crate::AudioError::Cancelled);
        }
        for state in 0..states {
            let mut outgoing = vec![backward[time + 1][state] + logp[time + 1][labels[state]]];
            if state + 1 < states {
                outgoing.push(backward[time + 1][state + 1] + logp[time + 1][labels[state + 1]]);
            }
            if state + 2 < states
                && labels[state] != labels[state + 2]
                && labels[state + 2] != blank
            {
                outgoing.push(backward[time + 1][state + 2] + logp[time + 1][labels[state + 2]]);
            }
            backward[time][state] = log_sum_exp(&outgoing);
        }
    }
    let occupancy = (0..frames)
        .map(|time| {
            (0..states)
                .map(|state| (forward[time][state] + backward[time][state] - log_z).exp())
                .collect()
        })
        .collect();
    Ok(CtcTrellis {
        best_log_likelihood: best,
        state_path,
        occupancy,
    })
}

fn log_sum_exp(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return NEG_INFINITY;
    }
    maximum
        + values
            .iter()
            .map(|value| (*value - maximum).exp())
            .sum::<f64>()
            .ln()
}

fn occupancy_quantiles(values: &[f64]) -> (usize, usize, usize) {
    let total = values.iter().sum::<f64>();
    if total <= 0.0 {
        return (0, 0, values.len().saturating_sub(1));
    }
    let find = |quantile: f64| {
        let target = total * quantile;
        let mut sum = 0.0;
        for (index, value) in values.iter().enumerate() {
            sum += value;
            if sum >= target {
                return index;
            }
        }
        values.len().saturating_sub(1)
    };
    (find(0.05), find(0.50), find(0.95))
}

fn normalize_hypothesis_posteriors(hypotheses: &mut [AlignmentHypothesis]) {
    let maximum = hypotheses
        .iter()
        .filter_map(|path| path.scores.backend_score)
        .fold(NEG_INFINITY, f64::max);
    let denominator = hypotheses
        .iter()
        .filter_map(|path| path.scores.backend_score)
        .map(|score| (score - maximum).exp())
        .sum::<f64>();
    for hypothesis in hypotheses {
        hypothesis.normalized_path_posterior = hypothesis
            .scores
            .backend_score
            .map(|score| (score - maximum).exp() / denominator.max(1.0e-12))
            .unwrap_or(0.0);
    }
}

fn select_hypothesis(
    hypotheses: &[AlignmentHypothesis],
    limits: &AlignmentLimits,
) -> Option<String> {
    let best = hypotheses.first()?;
    if best.normalized_path_posterior < limits.minimum_path_posterior {
        return None;
    }
    let runner_up = hypotheses
        .get(1)
        .map(|path| path.normalized_path_posterior)
        .unwrap_or(0.0);
    if best.normalized_path_posterior - runner_up < limits.minimum_selection_margin {
        return None;
    }
    Some(best.id.clone())
}

fn effective_projections(request: &PhoneAlignmentRequest) -> Vec<AlignmentProjection> {
    let mut projections = request.projections.clone();
    for path in &request.pronunciations {
        for unit in &path.units {
            for (targets, kind, loss, provenance) in [
                (
                    &unit.utterance_ids,
                    ProjectionKind::Contains,
                    ProjectionLoss::ManyToOne,
                    "pronunciation_phone_to_utterance",
                ),
                (
                    &unit.phoneme_ids,
                    ProjectionKind::Realizes,
                    ProjectionLoss::ManyToOne,
                    "pronunciation_phone_to_phoneme",
                ),
                (
                    &unit.syllable_ids,
                    ProjectionKind::Contains,
                    ProjectionLoss::ManyToOne,
                    "pronunciation_phone_to_syllable",
                ),
                (
                    &unit.morpheme_ids,
                    ProjectionKind::Contains,
                    ProjectionLoss::ManyToOne,
                    "pronunciation_phone_to_morpheme",
                ),
                (
                    &unit.word_ids,
                    ProjectionKind::Contains,
                    ProjectionLoss::ManyToOne,
                    "pronunciation_phone_to_word",
                ),
                (
                    &unit.transcript_token_ids,
                    ProjectionKind::AlignedTo,
                    ProjectionLoss::ManyToOne,
                    "pronunciation_phone_to_transcript_token",
                ),
                (
                    &unit.speaker_span_ids,
                    ProjectionKind::Overlaps,
                    ProjectionLoss::Approximate,
                    "acoustic_phone_to_speaker_region",
                ),
            ] {
                if !targets.is_empty() {
                    projections.push(AlignmentProjection {
                        from_ids: vec![unit.id.clone()],
                        to_ids: targets.clone(),
                        kind,
                        loss,
                        provenance: provenance.into(),
                    });
                }
            }
        }
    }
    projections.sort_by(|left, right| {
        left.from_ids
            .cmp(&right.from_ids)
            .then_with(|| left.to_ids.cmp(&right.to_ids))
            .then_with(|| format!("{:?}", left.kind).cmp(&format!("{:?}", right.kind)))
    });
    projections.dedup();
    projections
}

fn unaligned_intervals(
    units: &[AlignedUnit],
    audio_frames: u64,
    sample_rate_hz: u32,
) -> Vec<FrameInterval> {
    let mut intervals = units
        .iter()
        .filter_map(|unit| unit.interval.clone())
        .collect::<Vec<_>>();
    intervals.sort_by_key(|interval| (interval.start_frame, interval.end_frame));
    let mut cursor = 0;
    let mut gaps = Vec::new();
    for interval in intervals {
        if cursor < interval.start_frame {
            gaps.push(FrameInterval {
                start_frame: cursor,
                end_frame: interval.start_frame,
                sample_rate_hz,
            });
        }
        cursor = cursor.max(interval.end_frame);
    }
    if cursor < audio_frames {
        gaps.push(FrameInterval {
            start_frame: cursor,
            end_frame: audio_frames,
            sample_rate_hz,
        });
    }
    gaps
}

fn frames_to_ms(frame: u64, sample_rate_hz: u32) -> u64 {
    frame.saturating_mul(1_000) / u64::from(sample_rate_hz)
}

fn stable_id(namespace: &str, value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(namespace.as_bytes());
    hasher.update([0]);
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    let short = digest[..12]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{namespace}:{short}")
}

fn sha256_json(value: &impl Serialize) -> Result<String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| invalid(format!("serializing alignment request: {error}")))?;
    let digest = Sha256::digest(bytes);
    Ok(format!(
        "sha256:{}",
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentDeltaKind {
    Append,
    Replace,
    Withdraw,
    Commit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentDelta {
    pub kind: AlignmentDeltaKind,
    pub unit_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<AlignedUnit>,
    pub revision: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamingAlignmentUpdate {
    pub commit_frontier_frame: u64,
    pub deltas: Vec<AlignmentDelta>,
    pub artifact: PhoneAlignmentArtifact,
    pub metrics: StreamingAlignmentMetrics,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamingAlignmentMetrics {
    pub evidence_end_frame: u64,
    pub revision_count: u64,
    pub maximum_revision_depth_frames: u64,
    pub provisional_unit_count: usize,
    pub committed_unit_count: usize,
    pub churn_ratio: f64,
    pub mean_time_to_stability_frames: f64,
}

#[derive(Debug, Clone)]
pub struct StreamingPhoneAligner {
    engine: PhoneAlignmentEngine,
    future_context_frames: u64,
    revision: u64,
    previous_units: BTreeMap<String, AlignedUnit>,
    committed: BTreeSet<String>,
}

impl StreamingPhoneAligner {
    pub fn new(future_context_frames: u64) -> Self {
        Self {
            engine: PhoneAlignmentEngine,
            future_context_frames,
            revision: 0,
            previous_units: BTreeMap::new(),
            committed: BTreeSet::new(),
        }
    }

    pub fn update(
        &mut self,
        audio: &AudioBuffer,
        request: &PhoneAlignmentRequest,
        posterior: &CtcPosteriorMatrix,
    ) -> Result<StreamingAlignmentUpdate> {
        self.revision += 1;
        let mut artifact = self.engine.align_ctc(audio, request, posterior)?;
        let evidence_end = posterior
            .frame_start
            .saturating_add(
                (posterior.probabilities.len() as u64).saturating_mul(posterior.frame_stride),
            )
            .min(audio.frames() as u64);
        let frontier = evidence_end.saturating_sub(self.future_context_frames);
        let mut current = BTreeMap::new();
        if let Some(selected) = artifact
            .selected_hypothesis_id
            .as_deref()
            .and_then(|id| artifact.hypotheses.iter_mut().find(|path| path.id == id))
        {
            for unit in &mut selected.units {
                if self.committed.contains(&unit.id) {
                    if let Some(previous) = self.previous_units.get(&unit.id) {
                        // Re-running the bounded lattice may move old estimates,
                        // but committed material is immutable without an explicit
                        // correction/repair request.
                        *unit = previous.clone();
                    }
                    current.insert(unit.id.clone(), unit.clone());
                    continue;
                }
                let is_committed = unit
                    .interval
                    .as_ref()
                    .is_some_and(|interval| interval.end_frame <= frontier);
                unit.lifecycle = if is_committed {
                    self.committed.insert(unit.id.clone());
                    AlignmentLifecycle::Committed
                } else {
                    AlignmentLifecycle::Provisional
                };
                current.insert(unit.id.clone(), unit.clone());
            }
        }
        let mut deltas = Vec::new();
        let mut maximum_revision_depth_frames = 0;
        for (id, previous) in &self.previous_units {
            match current.get(id) {
                None if !self.committed.contains(id) => deltas.push(AlignmentDelta {
                    kind: AlignmentDeltaKind::Withdraw,
                    unit_id: id.clone(),
                    unit: None,
                    revision: self.revision,
                }),
                Some(updated) if updated != previous => {
                    let previous_start = previous
                        .interval
                        .as_ref()
                        .map(|interval| interval.start_frame)
                        .unwrap_or(evidence_end);
                    maximum_revision_depth_frames = maximum_revision_depth_frames
                        .max(evidence_end.saturating_sub(previous_start));
                    deltas.push(AlignmentDelta {
                        kind: AlignmentDeltaKind::Replace,
                        unit_id: id.clone(),
                        unit: Some(updated.clone()),
                        revision: self.revision,
                    });
                }
                _ => {}
            }
        }
        for (id, unit) in &current {
            if !self.previous_units.contains_key(id) {
                deltas.push(AlignmentDelta {
                    kind: AlignmentDeltaKind::Append,
                    unit_id: id.clone(),
                    unit: Some(unit.clone()),
                    revision: self.revision,
                });
            } else if unit.lifecycle == AlignmentLifecycle::Committed
                && self.previous_units[id].lifecycle != AlignmentLifecycle::Committed
            {
                deltas.push(AlignmentDelta {
                    kind: AlignmentDeltaKind::Commit,
                    unit_id: id.clone(),
                    unit: Some(unit.clone()),
                    revision: self.revision,
                });
            }
        }
        deltas.sort_by(|left, right| {
            left.unit_id
                .cmp(&right.unit_id)
                .then_with(|| format!("{:?}", left.kind).cmp(&format!("{:?}", right.kind)))
        });
        let revision_count = deltas
            .iter()
            .filter(|delta| {
                matches!(
                    delta.kind,
                    AlignmentDeltaKind::Replace | AlignmentDeltaKind::Withdraw
                )
            })
            .count() as u64;
        let stability_delays = deltas
            .iter()
            .filter(|delta| delta.kind == AlignmentDeltaKind::Commit)
            .filter_map(|delta| delta.unit.as_ref()?.interval.as_ref())
            .map(|interval| evidence_end.saturating_sub(interval.end_frame) as f64)
            .collect::<Vec<_>>();
        let committed_unit_count = current
            .values()
            .filter(|unit| unit.lifecycle == AlignmentLifecycle::Committed)
            .count();
        let provisional_unit_count = current.len().saturating_sub(committed_unit_count);
        let churn_ratio = if current.is_empty() {
            0.0
        } else {
            revision_count as f64 / current.len() as f64
        };
        self.previous_units = current;
        Ok(StreamingAlignmentUpdate {
            commit_frontier_frame: frontier,
            metrics: StreamingAlignmentMetrics {
                evidence_end_frame: evidence_end,
                revision_count,
                maximum_revision_depth_frames,
                provisional_unit_count,
                committed_unit_count,
                churn_ratio,
                mean_time_to_stability_frames: if stability_delays.is_empty() {
                    0.0
                } else {
                    stability_delays.iter().sum::<f64>() / stability_delays.len() as f64
                },
            },
            deltas,
            artifact,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceAlignmentUnit {
    pub symbol: String,
    pub interval: FrameInterval,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceAlignmentWord {
    pub id: String,
    pub interval: FrameInterval,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentEvaluationReference {
    pub id: String,
    pub language_tag: String,
    pub variety: String,
    pub units: Vec<ReferenceAlignmentUnit>,
    #[serde(default)]
    pub words: Vec<ReferenceAlignmentWord>,
    #[serde(default)]
    pub annotator_tolerance_frames: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentEvaluationReport {
    pub reference_id: String,
    pub selected_path_error_rate: f64,
    pub oracle_top_k_path_recall: f64,
    pub boundary_mean_absolute_error_frames: f64,
    pub boundary_tolerance_accuracy: BTreeMap<u64, f64>,
    pub boundary_interval_coverage: f64,
    pub phone_presence_brier_score: f64,
    pub path_selection_brier_score: f64,
    pub word_boundary_mean_absolute_error_frames: f64,
    pub word_tolerance_accuracy: f64,
    pub insertions: usize,
    pub deletions: usize,
    pub substitutions: usize,
    pub unaligned_reference_units: usize,
    pub language_tag: String,
    pub variety: String,
}

pub fn evaluate_alignment(
    artifact: &PhoneAlignmentArtifact,
    reference: &AlignmentEvaluationReference,
    tolerances: &[u64],
) -> AlignmentEvaluationReport {
    let selected = artifact
        .selected_hypothesis()
        .map(|path| {
            path.units
                .iter()
                .filter(|unit| {
                    matches!(
                        unit.relation,
                        AlignmentUnitRelation::Match | AlignmentUnitRelation::Substitution
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let predicted_symbols = selected
        .iter()
        .map(|unit| unit.symbol.as_str())
        .collect::<Vec<_>>();
    let reference_symbols = reference
        .units
        .iter()
        .map(|unit| unit.symbol.as_str())
        .collect::<Vec<_>>();
    let (edits, insertions, deletions, substitutions) =
        edit_counts(&reference_symbols, &predicted_symbols);
    let aligned = selected.len().min(reference.units.len());
    let mut errors = Vec::new();
    let mut covered = 0usize;
    for (predicted, expected) in selected.iter().zip(&reference.units) {
        let Some(interval) = &predicted.interval else {
            continue;
        };
        let start_error = interval.start_frame.abs_diff(expected.interval.start_frame);
        let end_error = interval.end_frame.abs_diff(expected.interval.end_frame);
        errors.extend([start_error as f64, end_error as f64]);
        if predicted.start_boundary.as_ref().is_some_and(|range| {
            (range.lower_frame..=range.upper_frame).contains(&expected.interval.start_frame)
        }) && predicted.end_boundary.as_ref().is_some_and(|range| {
            (range.lower_frame..=range.upper_frame).contains(&expected.interval.end_frame)
        }) {
            covered += 1;
        }
    }
    let tolerance_accuracy = tolerances
        .iter()
        .copied()
        .map(|tolerance| {
            let count = errors
                .iter()
                .filter(|error| **error <= tolerance as f64)
                .count();
            (
                tolerance,
                if errors.is_empty() {
                    0.0
                } else {
                    count as f64 / errors.len() as f64
                },
            )
        })
        .collect();
    let oracle = artifact.hypotheses.iter().any(|path| {
        path.units
            .iter()
            .filter(|unit| unit.relation == AlignmentUnitRelation::Match)
            .map(|unit| unit.symbol.as_str())
            .eq(reference_symbols.iter().copied())
    });
    let presence_scores = artifact
        .selected_hypothesis()
        .into_iter()
        .flat_map(|path| path.units.iter())
        .map(|unit| {
            let observed = if unit.relation == AlignmentUnitRelation::Insertion
                || unit.input_unit_ids.is_empty()
            {
                0.0
            } else {
                1.0
            };
            let probability = unit.presence_probability.unwrap_or(0.0);
            (probability - observed) * (probability - observed)
        })
        .collect::<Vec<_>>();
    let selected_path_correct = artifact.selected_hypothesis().is_some_and(|path| {
        path.units
            .iter()
            .filter(|unit| unit.relation == AlignmentUnitRelation::Match)
            .map(|unit| unit.symbol.as_str())
            .eq(reference_symbols.iter().copied())
    });
    let selected_probability = artifact
        .selected_hypothesis()
        .map(|path| path.normalized_path_posterior)
        .unwrap_or(0.0);
    let path_outcome = if selected_path_correct { 1.0 } else { 0.0 };
    let mut word_errors = Vec::new();
    for word in &reference.words {
        let linked_input_ids = artifact
            .projections
            .iter()
            .filter(|projection| projection.to_ids.iter().any(|id| id == &word.id))
            .flat_map(|projection| projection.from_ids.iter().cloned())
            .collect::<BTreeSet<_>>();
        let intervals = artifact
            .selected_hypothesis()
            .into_iter()
            .flat_map(|path| path.units.iter())
            .filter(|unit| {
                unit.input_unit_ids
                    .iter()
                    .any(|id| linked_input_ids.contains(id))
            })
            .filter_map(|unit| unit.interval.as_ref())
            .collect::<Vec<_>>();
        if let (Some(start), Some(end)) = (
            intervals.iter().map(|interval| interval.start_frame).min(),
            intervals.iter().map(|interval| interval.end_frame).max(),
        ) {
            word_errors.extend([
                start.abs_diff(word.interval.start_frame) as f64,
                end.abs_diff(word.interval.end_frame) as f64,
            ]);
        }
    }
    let word_tolerance = reference.annotator_tolerance_frames as f64;
    AlignmentEvaluationReport {
        reference_id: reference.id.clone(),
        selected_path_error_rate: if reference.units.is_empty() {
            0.0
        } else {
            edits as f64 / reference.units.len() as f64
        },
        oracle_top_k_path_recall: if oracle { 1.0 } else { 0.0 },
        boundary_mean_absolute_error_frames: if errors.is_empty() {
            0.0
        } else {
            errors.iter().sum::<f64>() / errors.len() as f64
        },
        boundary_tolerance_accuracy: tolerance_accuracy,
        boundary_interval_coverage: if aligned == 0 {
            0.0
        } else {
            covered as f64 / aligned as f64
        },
        phone_presence_brier_score: if presence_scores.is_empty() {
            0.0
        } else {
            presence_scores.iter().sum::<f64>() / presence_scores.len() as f64
        },
        path_selection_brier_score: (selected_probability - path_outcome)
            * (selected_probability - path_outcome),
        word_boundary_mean_absolute_error_frames: if word_errors.is_empty() {
            0.0
        } else {
            word_errors.iter().sum::<f64>() / word_errors.len() as f64
        },
        word_tolerance_accuracy: if word_errors.is_empty() {
            0.0
        } else {
            word_errors
                .iter()
                .filter(|error| **error <= word_tolerance)
                .count() as f64
                / word_errors.len() as f64
        },
        insertions,
        deletions,
        substitutions,
        unaligned_reference_units: reference.units.len().saturating_sub(aligned),
        language_tag: reference.language_tag.clone(),
        variety: reference.variety.clone(),
    }
}

fn edit_counts<T: Eq>(reference: &[T], prediction: &[T]) -> (usize, usize, usize, usize) {
    let mut table =
        vec![vec![(0usize, 0usize, 0usize, 0usize); prediction.len() + 1]; reference.len() + 1];
    for (index, row) in table.iter_mut().enumerate().skip(1) {
        row[0] = (index, 0, index, 0);
    }
    for (index, cell) in table[0].iter_mut().enumerate().skip(1) {
        *cell = (index, index, 0, 0);
    }
    for left in 1..=reference.len() {
        for right in 1..=prediction.len() {
            if reference[left - 1] == prediction[right - 1] {
                table[left][right] = table[left - 1][right - 1];
                continue;
            }
            let insertion = table[left][right - 1];
            let deletion = table[left - 1][right];
            let substitution = table[left - 1][right - 1];
            table[left][right] = [
                (insertion.0 + 1, insertion.1 + 1, insertion.2, insertion.3),
                (deletion.0 + 1, deletion.1, deletion.2 + 1, deletion.3),
                (
                    substitution.0 + 1,
                    substitution.1,
                    substitution.2,
                    substitution.3 + 1,
                ),
            ]
            .into_iter()
            .min_by_key(|value| (value.0, value.3, value.2, value.1))
            .unwrap();
        }
    }
    table[reference.len()][prediction.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio(frames: usize) -> AudioBuffer {
        AudioBuffer {
            samples: vec![0.05; frames],
            sample_rate_hz: 1_000,
            channels: 1,
        }
    }

    fn unit(id: &str, symbol: &str) -> AlignmentUnitSpec {
        AlignmentUnitSpec {
            id: id.into(),
            symbol: symbol.into(),
            kind: SegmentKind::Phone,
            language_tag: "en".into(),
            inventory_id: "fixture-ipa".into(),
            utterance_ids: vec!["utterance:1".into()],
            transcript_token_ids: vec!["token:1".into()],
            word_ids: vec!["word:1".into()],
            morpheme_ids: Vec::new(),
            syllable_ids: vec!["syllable:1".into()],
            phoneme_ids: vec![format!("phoneme:{id}")],
            speaker_span_ids: vec!["speaker:1".into()],
        }
    }

    fn request(paths: Vec<PronunciationPath>) -> PhoneAlignmentRequest {
        PhoneAlignmentRequest {
            schema_version: PHONE_ALIGNMENT_SCHEMA_VERSION,
            mode: AlignmentMode::PronunciationConstrained,
            audio: AudioAlignmentInput {
                artifact_id: "fixture.wav".into(),
                expected_sha256: None,
                channel: 0,
                selected_regions: Vec::new(),
                preprocessing_artifacts: vec!["mono-v1".into()],
            },
            transcript: Some(TranscriptLattice {
                id: "transcript:1".into(),
                supplied_text: Some("cat".into()),
                paths: vec![vec![TranscriptToken {
                    id: "token:1".into(),
                    text: "cat".into(),
                    language_tag: "en".into(),
                    normalized_from: Vec::new(),
                }]],
            }),
            pronunciations: paths,
            timing_hints: Vec::new(),
            duration_priors: Vec::new(),
            corrections: Vec::new(),
            projections: Vec::new(),
            limits: AlignmentLimits {
                minimum_path_posterior: 0.0,
                minimum_selection_margin: 0.0,
                insertion_probability: 0.8,
                ..Default::default()
            },
            context: PhoneticSegmentationContext {
                graph_id: "graph:1".into(),
                graph_revision: 1,
                recipe_id: "recipe:1".into(),
                execution_record_id: "run:1".into(),
                session_id: None,
                audio_span_id: None,
                runtime: "test".into(),
                runtime_version: "1".into(),
            },
        }
    }

    fn posterior(rows: Vec<Vec<f64>>) -> CtcPosteriorMatrix {
        CtcPosteriorMatrix {
            schema_version: 1,
            source: AlignmentSourceIdentity {
                provider: "common-phone".into(),
                model: "fixture-ctc".into(),
                version: "1".into(),
                artifact_id: Some("fixture-posteriors".into()),
            },
            language_tags: vec!["en".into()],
            inventory_id: "fixture-ipa".into(),
            sample_rate_hz: 1_000,
            frame_start: 0,
            frame_stride: 10,
            frame_width: 10,
            blank_index: 0,
            symbols: vec![
                "<blank>".into(),
                "k".into(),
                "æ".into(),
                "t".into(),
                "x".into(),
            ],
            probabilities: rows,
            model_checksum: Some("sha256:fixture".into()),
        }
    }

    fn rows(middle: usize) -> Vec<Vec<f64>> {
        let mut rows = vec![vec![0.96, 0.01, 0.01, 0.01, 0.01]; 12];
        for (range, symbol) in [(1..4, 1), (4..middle, 2), (middle..10, 3)] {
            for frame in range {
                rows[frame] = vec![0.02; 5];
                rows[frame][symbol] = 0.92;
            }
        }
        rows
    }

    #[test]
    fn ctc_alignment_retains_paths_scores_ranges_and_many_to_many_projections() {
        let paths = vec![
            PronunciationPath {
                id: "cat-primary".into(),
                lexical_source: "fixture-lexicon".into(),
                language_tag: "en".into(),
                inventory_id: "fixture-ipa".into(),
                prior_probability: 0.8,
                units: vec![unit("k", "k"), unit("ae", "æ"), unit("t", "t")],
            },
            PronunciationPath {
                id: "cat-alternative".into(),
                lexical_source: "fixture-lexicon".into(),
                language_tag: "en".into(),
                inventory_id: "fixture-ipa".into(),
                prior_probability: 0.2,
                units: vec![unit("k2", "k"), unit("ae2", "æ")],
            },
        ];
        let mut request = request(paths);
        request.duration_priors.push(DurationPrior {
            unit_id: "ae".into(),
            mean_frames: 30.0,
            standard_deviation_frames: 10.0,
            weight: 0.5,
            source: "fixture-duration-model".into(),
        });
        let artifact = PhoneAlignmentEngine
            .align_ctc(&audio(120), &request, &posterior(rows(7)))
            .unwrap();
        assert_eq!(artifact.schema_version, 2);
        assert_eq!(artifact.hypotheses.len(), 2);
        let selected = artifact.selected_hypothesis().unwrap();
        assert_eq!(selected.pronunciation_path_id, "cat-primary");
        assert_eq!(selected.units.len(), 3);
        assert!(selected.units.iter().all(|unit| {
            unit.start_boundary.as_ref().is_some_and(|range| {
                range.lower_frame <= range.estimate_frame
                    && range.estimate_frame <= range.upper_frame
            })
        }));
        assert!(artifact
            .projections
            .iter()
            .any(|projection| { projection.from_ids == ["k"] && projection.to_ids == ["word:1"] }));
        assert!(selected.scores.acoustic_log_likelihood.is_some());
        assert!(selected.scores.pronunciation_log_prior.is_some());
        assert!(selected.scores.duration_log_prior.is_some());
        assert!(selected.units[1].scores.duration_log_prior.is_some());
        assert!(check_alignment_conformance(&artifact).passed);
    }

    #[test]
    fn timing_authorities_remain_distinct_in_serialized_artifacts() {
        let path = PronunciationPath {
            id: "cat".into(),
            lexical_source: "fixture".into(),
            language_tag: "en".into(),
            inventory_id: "fixture-ipa".into(),
            prior_probability: 1.0,
            units: vec![unit("k", "k"), unit("ae", "æ"), unit("t", "t")],
        };
        let mut forced = request(vec![path.clone()]);
        forced.mode = AlignmentMode::PronunciationConstrained;
        let artifact = PhoneAlignmentEngine
            .align_ctc(&audio(120), &forced, &posterior(rows(7)))
            .unwrap();
        assert_eq!(
            artifact.selected_hypothesis().unwrap().units[0].timing_authority,
            TimingAuthority::ForcedAlignment
        );

        let mut recognition = request(vec![path.clone()]);
        recognition.mode = AlignmentMode::Unconstrained;
        let artifact = PhoneAlignmentEngine
            .align_ctc(&audio(120), &recognition, &posterior(rows(7)))
            .unwrap();
        assert_eq!(
            artifact.selected_hypothesis().unwrap().units[0].timing_authority,
            TimingAuthority::RecognitionDerived
        );

        let mut imported = request(vec![path.clone()]);
        imported.mode = AlignmentMode::Imported;
        imported.timing_hints.push(BoundaryHint {
            unit_id: "k".into(),
            start: Some(BoundaryEstimate::point(12, "fixture-import")),
            end: Some(BoundaryEstimate::point(28, "fixture-import")),
            authority: TimingAuthority::ImportedAnnotation,
            source: posterior(rows(7)).source,
        });
        let artifact = PhoneAlignmentEngine
            .align_ctc(&audio(120), &imported, &posterior(rows(7)))
            .unwrap();
        let unit = &artifact.selected_hypothesis().unwrap().units[0];
        assert_eq!(unit.timing_authority, TimingAuthority::ImportedAnnotation);
        assert_eq!(unit.interval.as_ref().unwrap().start_frame, 12);

        let mut synthesis = request(vec![path.clone()]);
        synthesis.mode = AlignmentMode::SynthesisKnown;
        synthesis.timing_hints.push(BoundaryHint {
            unit_id: "k".into(),
            start: Some(BoundaryEstimate::point(14, "fixture-tts-plan")),
            end: Some(BoundaryEstimate::point(26, "fixture-tts-plan")),
            authority: TimingAuthority::SynthesisKnown,
            source: posterior(rows(7)).source,
        });
        let artifact = PhoneAlignmentEngine
            .align_ctc(&audio(120), &synthesis, &posterior(rows(7)))
            .unwrap();
        assert_eq!(
            artifact.selected_hypothesis().unwrap().units[0].timing_authority,
            TimingAuthority::SynthesisKnown
        );

        let mut corrected = request(vec![path]);
        corrected.corrections.push(AlignmentCorrection {
            id: "correction:1".into(),
            actor: "fixture-reviewer".into(),
            reason: "spectrogram review".into(),
            unit_id: "k".into(),
            replacement_symbol: None,
            start: Some(BoundaryEstimate::point(11, "manual-range")),
            end: Some(BoundaryEstimate::point(29, "manual-range")),
            supports: vec!["evidence:spectrogram".into()],
            conflicts_with: vec!["boundary:original".into()],
        });
        let artifact = PhoneAlignmentEngine
            .align_ctc(&audio(120), &corrected, &posterior(rows(7)))
            .unwrap();
        let unit = &artifact.selected_hypothesis().unwrap().units[0];
        assert_eq!(unit.timing_authority, TimingAuthority::ManualCorrection);
        assert_eq!(unit.lifecycle, AlignmentLifecycle::Corrected);
        assert!(unit.supports.contains(&"correction:1".into()));
        assert!(serde_json::to_string(&artifact)
            .unwrap()
            .contains("\"manual_correction\""));
    }

    #[test]
    fn ambiguous_paths_abstain_instead_of_fabricating_selection() {
        let path = PronunciationPath {
            id: "same".into(),
            lexical_source: "fixture".into(),
            language_tag: "en".into(),
            inventory_id: "fixture-ipa".into(),
            prior_probability: 0.5,
            units: vec![unit("k", "k"), unit("ae", "æ"), unit("t", "t")],
        };
        let mut other = path.clone();
        other.id = "same-too".into();
        let mut request = request(vec![path, other]);
        request.limits.minimum_selection_margin = 0.1;
        let artifact = PhoneAlignmentEngine
            .align_ctc(&audio(120), &request, &posterior(rows(7)))
            .unwrap();
        assert_eq!(artifact.readiness, AlignmentReadiness::Abstained);
        assert!(artifact.selected_hypothesis_id.is_none());
        assert_eq!(artifact.hypotheses.len(), 2);
    }

    #[test]
    fn streaming_revises_only_the_uncommitted_tail_and_advances_frontier() {
        let path = PronunciationPath {
            id: "cat".into(),
            lexical_source: "fixture".into(),
            language_tag: "en".into(),
            inventory_id: "fixture-ipa".into(),
            prior_probability: 1.0,
            units: vec![unit("k", "k"), unit("ae", "æ"), unit("t", "t")],
        };
        let request = request(vec![path]);
        let mut stream = StreamingPhoneAligner::new(40);
        let first = stream
            .update(&audio(120), &request, &posterior(rows(7)))
            .unwrap();
        assert_eq!(first.commit_frontier_frame, 80);
        assert!(first
            .deltas
            .iter()
            .any(|delta| delta.kind == AlignmentDeltaKind::Append));
        let second = stream
            .update(&audio(120), &request, &posterior(rows(8)))
            .unwrap();
        assert!(second.deltas.iter().any(|delta| {
            matches!(
                delta.kind,
                AlignmentDeltaKind::Replace | AlignmentDeltaKind::Commit
            )
        }));
        assert!(second
            .artifact
            .selected_hypothesis()
            .unwrap()
            .units
            .iter()
            .any(|unit| unit.lifecycle == AlignmentLifecycle::Committed));
        assert!(second.metrics.committed_unit_count > 0);
        assert!(second.metrics.churn_ratio >= 0.0);
    }

    #[test]
    fn evaluation_reports_path_boundary_coverage_and_error_components() {
        let path = PronunciationPath {
            id: "cat".into(),
            lexical_source: "fixture".into(),
            language_tag: "en".into(),
            inventory_id: "fixture-ipa".into(),
            prior_probability: 1.0,
            units: vec![unit("k", "k"), unit("ae", "æ"), unit("t", "t")],
        };
        let artifact = PhoneAlignmentEngine
            .align_ctc(&audio(120), &request(vec![path]), &posterior(rows(7)))
            .unwrap();
        let reference = AlignmentEvaluationReference {
            id: "fixture-reference".into(),
            language_tag: "en".into(),
            variety: "en-US".into(),
            units: artifact
                .selected_hypothesis()
                .unwrap()
                .units
                .iter()
                .map(|unit| ReferenceAlignmentUnit {
                    symbol: unit.symbol.clone(),
                    interval: unit.interval.clone().unwrap(),
                })
                .collect(),
            words: Vec::new(),
            annotator_tolerance_frames: 10,
        };
        let report = evaluate_alignment(&artifact, &reference, &[5, 10, 20]);
        assert_eq!(report.selected_path_error_rate, 0.0);
        assert_eq!(report.oracle_top_k_path_recall, 1.0);
        assert_eq!(report.boundary_mean_absolute_error_frames, 0.0);
        assert_eq!(report.boundary_tolerance_accuracy[&5], 1.0);
        assert_eq!(report.boundary_interval_coverage, 1.0);
    }

    #[test]
    fn redistributable_multilingual_suite_aligns_without_whitespace_assumptions() {
        #[derive(Deserialize)]
        struct Suite {
            license: String,
            cases: Vec<Case>,
        }
        #[derive(Deserialize)]
        struct Case {
            id: String,
            audio_frames: usize,
            request: PhoneAlignmentRequest,
            posteriors: CtcPosteriorMatrix,
            reference: AlignmentEvaluationReference,
        }
        let suite: Suite = serde_json::from_str(include_str!(
            "../../../fixtures/phone-alignment/multilingual-synthetic-v1.json"
        ))
        .unwrap();
        assert_eq!(suite.license, "CC0-1.0");
        assert_eq!(suite.cases.len(), 3);
        let mut languages = BTreeSet::new();
        for case in suite.cases {
            let artifact = PhoneAlignmentEngine
                .align_ctc(&audio(case.audio_frames), &case.request, &case.posteriors)
                .unwrap_or_else(|error| panic!("{}: {error}", case.id));
            assert!(
                artifact.selected_hypothesis_id.is_some(),
                "{} abstained unexpectedly",
                case.id
            );
            let tolerance = case.reference.annotator_tolerance_frames;
            let report =
                evaluate_alignment(&artifact, &case.reference, &[tolerance, tolerance * 2]);
            assert_eq!(report.selected_path_error_rate, 0.0, "{}", case.id);
            assert_eq!(report.oracle_top_k_path_recall, 1.0, "{}", case.id);
            languages.insert(report.language_tag);
        }
        assert_eq!(
            languages,
            BTreeSet::from(["en".into(), "ja".into(), "mul".into()])
        );
    }

    #[test]
    fn incompatible_language_inventory_and_symbols_abstain_per_path() {
        let path = PronunciationPath {
            id: "unsupported".into(),
            lexical_source: "fixture".into(),
            language_tag: "fr".into(),
            inventory_id: "other-inventory".into(),
            prior_probability: 1.0,
            units: vec![unit("unknown", "ɲ")],
        };
        let artifact = PhoneAlignmentEngine
            .align_ctc(&audio(120), &request(vec![path]), &posterior(rows(7)))
            .unwrap();
        assert_eq!(artifact.readiness, AlignmentReadiness::Unsupported);
        assert!(artifact.selected_hypothesis_id.is_none());
        assert_eq!(artifact.diagnostics[0].code, "path.abstained");
        assert!(artifact.diagnostics[0].detail.contains("inventory"));
    }

    #[test]
    fn cancellation_stops_before_extending_more_lattice_paths() {
        let path = PronunciationPath {
            id: "cancelled".into(),
            lexical_source: "fixture".into(),
            language_tag: "en".into(),
            inventory_id: "fixture-ipa".into(),
            prior_probability: 1.0,
            units: vec![unit("k", "k"), unit("ae", "æ"), unit("t", "t")],
        };
        let cancellation = AlignmentCancellation::default();
        cancellation.cancel();
        let error = PhoneAlignmentEngine
            .align_ctc_with_cancellation(
                &audio(120),
                &request(vec![path]),
                &posterior(rows(7)),
                Some(&cancellation),
            )
            .unwrap_err();
        assert!(matches!(error, crate::AudioError::Cancelled));
    }

    #[test]
    fn v1_migration_preserves_points_without_inventing_uncertainty() {
        let legacy = crate::PhoneticSegmentationEngine::default()
            .segment_recipe(
                &audio(100),
                &crate::AlignmentRecipe {
                    schema_version: crate::ALIGNMENT_RECIPE_SCHEMA_VERSION,
                    audio_artifact_id: "legacy.wav".into(),
                    expected_audio_sha256: None,
                    transcript: None,
                    expected: vec![crate::ExpectedSegment {
                        symbol: "t".into(),
                        kind: SegmentKind::Phone,
                        inventory_membership: crate::InventoryMembership::Known,
                        language_tag: "en".into(),
                        inventory_id: "fixture-ipa".into(),
                        pronunciation_source: "fixture".into(),
                        evidence_links: Default::default(),
                    }],
                    candidates: vec![crate::AlignmentCandidate {
                        expected_index: 0,
                        start_frame: 10,
                        end_frame: 30,
                        confidence: 0.9,
                        boundary_origin: PhoneticBoundaryOrigin::Inferred,
                        source: posterior(rows(7)).source,
                        evidence: Default::default(),
                    }],
                    context: request(Vec::new()).context,
                },
            )
            .unwrap();
        let migrated = PhoneAlignmentArtifact::migrate_v1(&legacy);
        assert_eq!(migrated.migrated_from_schema, Some(1));
        let unit = &migrated.hypotheses[0].units[0];
        assert_eq!(unit.interval, legacy.segments[0].interval);
        assert_eq!(
            unit.start_boundary.as_ref().unwrap().method,
            "legacy_v1_point_boundary_no_uncertainty"
        );
        assert!(unit.presence_calibration.is_none());
        assert!(!migrated.backend_capabilities.boundary_uncertainty);
    }
}
