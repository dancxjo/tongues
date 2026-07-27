//! Evidence-bound phonetic segmentation.
//!
//! This module adapts Listenbury's speech-hypothesis lattice into a stricter
//! Tongues artifact contract. Competing adapter hypotheses are fused
//! deterministically, but an interval is emitted only when a source supplied
//! that interval with sufficient confidence. The engine never divides audio
//! evenly across an expected pronunciation.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use speaking::timeline::{
    AlignmentKind, SpanModality, SpeechTimelineSession, TimelineAlignment, TimelineAttachment,
    TimelineAttachmentKind, TimelineSpan,
};

use crate::{invalid, AudioBuffer, Result};

pub const ALIGNMENT_RECIPE_SCHEMA_VERSION: u32 = 1;
pub const PHONETIC_SEGMENTATION_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const PHONETIC_SEGMENTATION_ALGORITHM_VERSION: &str =
    "tongues.phonetic-segmentation.listenbury-lattice-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentKind {
    Phone,
    Phoneme,
    Silence,
    Pause,
    WordBoundary,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InventoryMembership {
    Known,
    Unknown,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhoneticEvidenceLinks {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub word_span_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript_span_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_span_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectedSegment {
    pub symbol: String,
    pub kind: SegmentKind,
    pub inventory_membership: InventoryMembership,
    pub language_tag: String,
    pub inventory_id: String,
    pub pronunciation_source: String,
    #[serde(default)]
    pub evidence_links: PhoneticEvidenceLinks,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignmentSourceIdentity {
    pub provider: String,
    pub model: String,
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_id: Option<String>,
}

impl AlignmentSourceIdentity {
    fn stable_id(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.provider,
            self.model,
            self.version,
            self.artifact_id.as_deref().unwrap_or("")
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AlignmentEvidence {
    pub asr_confidence: Option<f32>,
    pub energy_alignment_quality: Option<f32>,
    pub phone_segmentation_agreement: Option<f32>,
    pub pronunciation_fit: Option<f32>,
    pub spectral_evidence: Option<f32>,
    pub timing_coherence: Option<f32>,
    pub mechanical_recognizer_score: Option<f32>,
}

impl AlignmentEvidence {
    fn weighted_confidence(&self) -> Option<f32> {
        let signals = [
            (self.asr_confidence, 3.0_f32),
            (self.energy_alignment_quality, 1.5),
            (self.phone_segmentation_agreement, 1.0),
            (self.pronunciation_fit, 1.0),
            (self.spectral_evidence, 0.75),
            (self.timing_coherence, 1.25),
            (self.mechanical_recognizer_score, 1.0),
        ];
        let (sum, weight) = signals.into_iter().fold(
            (0.0_f32, 0.0_f32),
            |(sum, weight), (value, signal_weight)| match value {
                Some(value) if value.is_finite() => (
                    sum + value.clamp(0.0, 1.0) * signal_weight,
                    weight + signal_weight,
                ),
                _ => (sum, weight),
            },
        );
        (weight > 0.0).then(|| (sum / weight).clamp(0.0, 1.0))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentCandidate {
    /// Index into `AlignmentRecipe::expected`.
    pub expected_index: usize,
    /// Half-open frame interval in the original audio timebase.
    pub start_frame: u64,
    pub end_frame: u64,
    pub confidence: f32,
    #[serde(default)]
    pub boundary_origin: PhoneticBoundaryOrigin,
    pub source: AlignmentSourceIdentity,
    #[serde(default)]
    pub evidence: AlignmentEvidence,
}

impl AlignmentCandidate {
    fn fused_confidence(&self) -> f32 {
        let own = self.confidence.clamp(0.0, 1.0);
        self.evidence
            .weighted_confidence()
            .map(|external| (own + external * 3.0) / 4.0)
            .unwrap_or(own)
            .clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhoneticBoundaryOrigin {
    SourceProvided,
    #[default]
    Inferred,
    Corrected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhoneticSegmentationContext {
    pub graph_id: String,
    pub graph_revision: u64,
    pub recipe_id: String,
    pub execution_record_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_span_id: Option<String>,
    pub runtime: String,
    pub runtime_version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentRecipe {
    pub schema_version: u32,
    pub audio_artifact_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_audio_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
    pub expected: Vec<ExpectedSegment>,
    #[serde(default)]
    pub candidates: Vec<AlignmentCandidate>,
    pub context: PhoneticSegmentationContext,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhoneticSegmentStatus {
    Aligned,
    Clipped,
    LowConfidence,
    MissingEvidence,
    UnknownSymbol,
    InconsistentEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameInterval {
    /// Inclusive start frame in the original waveform.
    pub start_frame: u64,
    /// Exclusive end frame in the original waveform.
    pub end_frame: u64,
    pub sample_rate_hz: u32,
}

impl FrameInterval {
    pub fn start_seconds(&self) -> f64 {
        self.start_frame as f64 / f64::from(self.sample_rate_hz)
    }

    pub fn end_seconds(&self) -> f64 {
        self.end_frame as f64 / f64::from(self.sample_rate_hz)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhoneticSegment {
    pub expected_index: usize,
    pub symbol: String,
    pub kind: SegmentKind,
    pub status: PhoneticSegmentStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<FrameInterval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alignment_source: Option<AlignmentSourceIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boundary_origin: Option<PhoneticBoundaryOrigin>,
    pub pronunciation_source: String,
    pub language_tag: String,
    pub inventory_id: String,
    #[serde(default)]
    pub evidence_links: PhoneticEvidenceLinks,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnalignedRegion {
    pub interval: FrameInterval,
    /// An unaligned gap is not claimed to be silence or non-speech.
    pub state: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhoneticSegmentationReadiness {
    Ready,
    Partial,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhoneticSegmentationIssue {
    pub code: String,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhoneticSegmentArtifact {
    pub schema_version: u32,
    pub algorithm_version: String,
    pub readiness: PhoneticSegmentationReadiness,
    pub timebase: String,
    pub audio_artifact_id: String,
    pub audio_sha256: String,
    pub recipe_sha256: String,
    pub audio_frames: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
    pub expected: Vec<ExpectedSegment>,
    pub segments: Vec<PhoneticSegment>,
    pub unaligned_regions: Vec<UnalignedRegion>,
    pub issues: Vec<PhoneticSegmentationIssue>,
    pub graph: PhoneticSegmentationContext,
    pub source_artifacts: Vec<AlignmentSourceIdentity>,
}

impl PhoneticSegmentArtifact {
    /// Attach this immutable artifact and its timed spans to a timeline session.
    ///
    /// Untimed/unsupported rows remain inspectable in the attachment payload but
    /// do not become fabricated timeline intervals.
    pub fn attach_to_timeline(&self, session: &mut SpeechTimelineSession) -> Result<String> {
        if let Some(expected_session_id) = self.graph.session_id.as_deref() {
            if expected_session_id != session.session_id {
                return Err(invalid(format!(
                    "segmentation expects timeline session `{expected_session_id}`, got `{}`",
                    session.session_id
                )));
            }
        }
        let artifact_id = format!(
            "phonetic-segmentation:{}",
            self.recipe_sha256
                .strip_prefix("sha256:")
                .unwrap_or(&self.recipe_sha256)
        );
        if session
            .attachments
            .iter()
            .any(|attachment| attachment.artifact_id == artifact_id)
        {
            return Err(invalid(format!(
                "timeline already contains segmentation attachment `{artifact_id}`"
            )));
        }

        let mut next = session.clone();
        let payload = serde_json::to_value(self)
            .map_err(|error| invalid(format!("serializing segmentation attachment: {error}")))?;
        next.attachments.push(TimelineAttachment {
            artifact_id: artifact_id.clone(),
            kind: TimelineAttachmentKind::PhoneticSegmentation,
            schema_version: self.schema_version,
            payload,
        });

        let known_ids = next
            .evidence
            .iter()
            .map(|span| span.id.clone())
            .collect::<std::collections::BTreeSet<_>>();
        for segment in &self.segments {
            let Some(interval) = &segment.interval else {
                continue;
            };
            let span_id = format!("{artifact_id}:{}", segment.expected_index);
            if known_ids.contains(&span_id) {
                return Err(invalid(format!(
                    "segmentation span `{span_id}` collides with existing evidence"
                )));
            }
            let start_ms = frames_to_ms(interval.start_frame, interval.sample_rate_hz);
            let mut end_ms = frames_to_ms(interval.end_frame, interval.sample_rate_hz);
            if end_ms <= start_ms {
                end_ms = start_ms.saturating_add(1);
            }
            let mut metadata = BTreeMap::from([
                ("symbol".into(), serde_json::json!(segment.symbol)),
                (
                    "segment_kind".into(),
                    serde_json::to_value(segment.kind).unwrap_or_default(),
                ),
                (
                    "status".into(),
                    serde_json::to_value(segment.status).unwrap_or_default(),
                ),
                (
                    "expected_index".into(),
                    serde_json::json!(segment.expected_index),
                ),
                (
                    "language_tag".into(),
                    serde_json::json!(segment.language_tag),
                ),
                (
                    "inventory_id".into(),
                    serde_json::json!(segment.inventory_id),
                ),
                (
                    "pronunciation_source".into(),
                    serde_json::json!(segment.pronunciation_source),
                ),
                (
                    "algorithm_version".into(),
                    serde_json::json!(self.algorithm_version),
                ),
                ("artifact_id".into(), serde_json::json!(artifact_id)),
                (
                    "audio_artifact_id".into(),
                    serde_json::json!(self.audio_artifact_id),
                ),
                ("audio_sha256".into(), serde_json::json!(self.audio_sha256)),
                (
                    "recipe_sha256".into(),
                    serde_json::json!(self.recipe_sha256),
                ),
                ("graph_id".into(), serde_json::json!(self.graph.graph_id)),
                (
                    "graph_revision".into(),
                    serde_json::json!(self.graph.graph_revision),
                ),
                ("recipe_id".into(), serde_json::json!(self.graph.recipe_id)),
                (
                    "execution_record_id".into(),
                    serde_json::json!(self.graph.execution_record_id),
                ),
                ("runtime".into(), serde_json::json!(self.graph.runtime)),
                (
                    "runtime_version".into(),
                    serde_json::json!(self.graph.runtime_version),
                ),
                (
                    "evidence_authority".into(),
                    serde_json::json!("observed_alignment_evidence"),
                ),
            ]);
            if let Some(confidence) = segment.confidence {
                metadata.insert("confidence".into(), serde_json::json!(confidence));
            }
            if let Some(origin) = segment.boundary_origin {
                metadata.insert(
                    "boundary_origin".into(),
                    serde_json::to_value(origin).unwrap_or_default(),
                );
            }
            if let Some(source) = &segment.alignment_source {
                metadata.insert(
                    "alignment_provider".into(),
                    serde_json::json!(source.provider),
                );
                metadata.insert("alignment_model".into(), serde_json::json!(source.model));
                metadata.insert(
                    "alignment_version".into(),
                    serde_json::json!(source.version),
                );
                if let Some(source_artifact_id) = &source.artifact_id {
                    metadata.insert(
                        "alignment_artifact_id".into(),
                        serde_json::json!(source_artifact_id),
                    );
                }
            }
            next.evidence.push(TimelineSpan {
                id: span_id.clone(),
                start_ms,
                end_ms,
                modality: match segment.kind {
                    SegmentKind::Phone | SegmentKind::Silence | SegmentKind::Pause => {
                        SpanModality::Phone
                    }
                    SegmentKind::Phoneme | SegmentKind::WordBoundary | SegmentKind::Unknown => {
                        SpanModality::Phoneme
                    }
                },
                metadata,
            });

            for (target, kind) in [
                (
                    segment.evidence_links.word_span_id.as_deref(),
                    AlignmentKind::Contains,
                ),
                (
                    segment.evidence_links.transcript_span_id.as_deref(),
                    AlignmentKind::Contains,
                ),
                (
                    segment.evidence_links.speaker_span_id.as_deref(),
                    AlignmentKind::AlignedTo,
                ),
                (
                    self.graph.audio_span_id.as_deref(),
                    AlignmentKind::AlignedTo,
                ),
            ] {
                let Some(target) = target else {
                    continue;
                };
                if !known_ids.contains(target) {
                    return Err(invalid(format!(
                        "segmentation link target `{target}` is absent from timeline evidence"
                    )));
                }
                next.alignments.push(TimelineAlignment {
                    source_span_id: span_id.clone(),
                    target_span_id: target.into(),
                    kind,
                    confidence: segment.confidence,
                });
            }
        }
        next.validate()
            .map_err(|error| invalid(format!("attached segmentation is invalid: {error}")))?;
        *session = next;
        Ok(artifact_id)
    }
}

fn frames_to_ms(frame: u64, sample_rate_hz: u32) -> u64 {
    frame.saturating_mul(1_000) / u64::from(sample_rate_hz)
}

/// Model/runtime-specific aligners implement this boundary. Their candidates
/// remain inspectable in the final artifact through `AlignmentSourceIdentity`.
pub trait PhoneticAlignmentSource: Send + Sync {
    fn identity(&self) -> &AlignmentSourceIdentity;
    fn collect(
        &self,
        audio: &AudioBuffer,
        expected: &[ExpectedSegment],
        transcript: Option<&str>,
    ) -> Result<Vec<AlignmentCandidate>>;
}

#[derive(Debug, Clone)]
pub struct HintAlignmentSource {
    pub identity: AlignmentSourceIdentity,
    pub candidates: Vec<AlignmentCandidate>,
}

impl PhoneticAlignmentSource for HintAlignmentSource {
    fn identity(&self) -> &AlignmentSourceIdentity {
        &self.identity
    }

    fn collect(
        &self,
        _audio: &AudioBuffer,
        _expected: &[ExpectedSegment],
        _transcript: Option<&str>,
    ) -> Result<Vec<AlignmentCandidate>> {
        Ok(self.candidates.clone())
    }
}

#[derive(Debug, Clone)]
pub struct PhoneticSegmentationEngine {
    pub minimum_confidence: f32,
}

impl Default for PhoneticSegmentationEngine {
    fn default() -> Self {
        Self {
            minimum_confidence: 0.75,
        }
    }
}

impl PhoneticSegmentationEngine {
    pub fn segment_recipe(
        &self,
        audio: &AudioBuffer,
        recipe: &AlignmentRecipe,
    ) -> Result<PhoneticSegmentArtifact> {
        if recipe.schema_version != ALIGNMENT_RECIPE_SCHEMA_VERSION {
            return Err(invalid(format!(
                "alignment recipe schema {} is unsupported; expected {}",
                recipe.schema_version, ALIGNMENT_RECIPE_SCHEMA_VERSION
            )));
        }
        let mut identities = recipe
            .candidates
            .iter()
            .map(|candidate| candidate.source.clone())
            .collect::<Vec<_>>();
        deduplicate_sources(&mut identities);
        self.segment_candidates(audio, recipe, recipe.candidates.clone(), identities)
    }

    pub fn segment_with_sources(
        &self,
        audio: &AudioBuffer,
        recipe: &AlignmentRecipe,
        sources: &[&dyn PhoneticAlignmentSource],
    ) -> Result<PhoneticSegmentArtifact> {
        if recipe.schema_version != ALIGNMENT_RECIPE_SCHEMA_VERSION {
            return Err(invalid(format!(
                "alignment recipe schema {} is unsupported; expected {}",
                recipe.schema_version, ALIGNMENT_RECIPE_SCHEMA_VERSION
            )));
        }
        let mut candidates = recipe.candidates.clone();
        let mut identities = Vec::new();
        for source in sources {
            identities.push(source.identity().clone());
            candidates.extend(source.collect(
                audio,
                &recipe.expected,
                recipe.transcript.as_deref(),
            )?);
        }
        identities.extend(candidates.iter().map(|candidate| candidate.source.clone()));
        deduplicate_sources(&mut identities);
        self.segment_candidates(audio, recipe, candidates, identities)
    }

    fn segment_candidates(
        &self,
        audio: &AudioBuffer,
        recipe: &AlignmentRecipe,
        candidates: Vec<AlignmentCandidate>,
        source_artifacts: Vec<AlignmentSourceIdentity>,
    ) -> Result<PhoneticSegmentArtifact> {
        audio.validate()?;
        if audio.frames() == 0 {
            return Err(invalid("phonetic segmentation requires non-empty audio"));
        }
        if !self.minimum_confidence.is_finite() || !(0.0..=1.0).contains(&self.minimum_confidence) {
            return Err(invalid(
                "phonetic segmentation minimum confidence must be in [0, 1]",
            ));
        }
        require_non_empty("audio_artifact_id", &recipe.audio_artifact_id)?;
        require_non_empty("context.graph_id", &recipe.context.graph_id)?;
        require_non_empty("context.recipe_id", &recipe.context.recipe_id)?;
        require_non_empty(
            "context.execution_record_id",
            &recipe.context.execution_record_id,
        )?;
        require_non_empty("context.runtime", &recipe.context.runtime)?;
        require_non_empty("context.runtime_version", &recipe.context.runtime_version)?;
        for (index, expected) in recipe.expected.iter().enumerate() {
            require_non_empty(
                &format!("expected[{index}].language_tag"),
                &expected.language_tag,
            )?;
            require_non_empty(
                &format!("expected[{index}].inventory_id"),
                &expected.inventory_id,
            )?;
            require_non_empty(
                &format!("expected[{index}].pronunciation_source"),
                &expected.pronunciation_source,
            )?;
        }

        let audio_hash = audio_sha256(audio);
        let recipe_hash = recipe_sha256(recipe)?;
        if let Some(expected) = recipe.expected_audio_sha256.as_deref() {
            if expected != audio_hash {
                return Err(invalid(format!(
                    "audio checksum mismatch: recipe expects {expected}, loaded {audio_hash}"
                )));
            }
        }

        let audio_frames = audio.frames() as u64;
        let mut issues = Vec::new();
        let mut by_expected = vec![Vec::<AlignmentCandidate>::new(); recipe.expected.len()];
        for candidate in candidates {
            if candidate.expected_index >= recipe.expected.len() {
                issues.push(PhoneticSegmentationIssue {
                    code: "candidate.expected_index_out_of_range".into(),
                    detail: format!(
                        "candidate index {} has no expected segment",
                        candidate.expected_index
                    ),
                    expected_index: None,
                });
                continue;
            }
            if !candidate.confidence.is_finite()
                || !(0.0..=1.0).contains(&candidate.confidence)
                || candidate.start_frame >= candidate.end_frame
                || candidate.source.provider.trim().is_empty()
                || candidate.source.model.trim().is_empty()
                || candidate.source.version.trim().is_empty()
            {
                issues.push(PhoneticSegmentationIssue {
                    code: "candidate.invalid".into(),
                    detail: "candidate source identity, confidence in [0,1], and half-open frame interval must be valid".into(),
                    expected_index: Some(candidate.expected_index),
                });
                continue;
            }
            by_expected[candidate.expected_index].push(candidate);
        }

        for candidates in &mut by_expected {
            candidates.sort_by(compare_candidates);
        }

        let mut previous_end = 0_u64;
        let mut segments = Vec::with_capacity(recipe.expected.len());
        for (expected_index, expected) in recipe.expected.iter().enumerate() {
            let selected = by_expected[expected_index].first();
            let mut segment = PhoneticSegment {
                expected_index,
                symbol: expected.symbol.clone(),
                kind: expected.kind,
                status: PhoneticSegmentStatus::MissingEvidence,
                interval: None,
                confidence: None,
                alignment_source: None,
                boundary_origin: None,
                pronunciation_source: expected.pronunciation_source.clone(),
                language_tag: expected.language_tag.clone(),
                inventory_id: expected.inventory_id.clone(),
                evidence_links: expected.evidence_links.clone(),
            };

            if expected.symbol.trim().is_empty()
                || expected.inventory_membership == InventoryMembership::Unknown
                || expected.kind == SegmentKind::Unknown
            {
                segment.status = PhoneticSegmentStatus::UnknownSymbol;
                issues.push(issue(
                    "expected.unknown_symbol",
                    "unknown symbols remain untimed",
                    expected_index,
                ));
            } else if let Some(candidate) = selected {
                let confidence = candidate.fused_confidence();
                segment.confidence = Some(confidence);
                segment.alignment_source = Some(candidate.source.clone());
                segment.boundary_origin = Some(candidate.boundary_origin);
                if confidence < self.minimum_confidence {
                    segment.status = PhoneticSegmentStatus::LowConfidence;
                    issues.push(issue(
                        "alignment.low_confidence",
                        "candidate timing was withheld because confidence is below threshold",
                        expected_index,
                    ));
                } else {
                    let start = candidate.start_frame;
                    let end = candidate.end_frame.min(audio_frames);
                    if start >= audio_frames || start >= end {
                        segment.status = PhoneticSegmentStatus::Clipped;
                        issues.push(issue(
                            "alignment.outside_audio",
                            "candidate lies outside the loaded audio and remains untimed",
                            expected_index,
                        ));
                    } else if start < previous_end {
                        segment.status = PhoneticSegmentStatus::InconsistentEvidence;
                        issues.push(issue(
                            "alignment.non_monotonic",
                            "candidate overlaps a preceding accepted interval and remains untimed",
                            expected_index,
                        ));
                    } else {
                        segment.status = if candidate.end_frame > audio_frames {
                            issues.push(issue(
                                "alignment.clipped",
                                "candidate end was clipped to the loaded audio boundary",
                                expected_index,
                            ));
                            PhoneticSegmentStatus::Clipped
                        } else {
                            PhoneticSegmentStatus::Aligned
                        };
                        segment.interval = Some(FrameInterval {
                            start_frame: start,
                            end_frame: end,
                            sample_rate_hz: audio.sample_rate_hz,
                        });
                        previous_end = end;
                    }
                }
            } else {
                issues.push(issue(
                    "alignment.missing_evidence",
                    "no timing candidate was supplied; no interval was fabricated",
                    expected_index,
                ));
            }
            segments.push(segment);
        }

        let accepted = segments
            .iter()
            .filter(|segment| segment.interval.is_some())
            .count();
        let readiness = if accepted == recipe.expected.len() && issues.is_empty() {
            PhoneticSegmentationReadiness::Ready
        } else if accepted > 0 {
            PhoneticSegmentationReadiness::Partial
        } else {
            PhoneticSegmentationReadiness::Unsupported
        };
        let unaligned_regions = unaligned_regions(&segments, audio_frames, audio.sample_rate_hz);

        Ok(PhoneticSegmentArtifact {
            schema_version: PHONETIC_SEGMENTATION_ARTIFACT_SCHEMA_VERSION,
            algorithm_version: PHONETIC_SEGMENTATION_ALGORITHM_VERSION.into(),
            readiness,
            timebase: "half-open original-audio frames [start_frame,end_frame); seconds = frame / sample_rate_hz".into(),
            audio_artifact_id: recipe.audio_artifact_id.clone(),
            audio_sha256: audio_hash,
            recipe_sha256: recipe_hash,
            audio_frames,
            sample_rate_hz: audio.sample_rate_hz,
            channels: audio.channels,
            transcript: recipe.transcript.clone(),
            expected: recipe.expected.clone(),
            segments,
            unaligned_regions,
            issues,
            graph: recipe.context.clone(),
            source_artifacts,
        })
    }
}

pub fn audio_sha256(audio: &AudioBuffer) -> String {
    let mut hasher = Sha256::new();
    hasher.update(audio.sample_rate_hz.to_le_bytes());
    hasher.update(audio.channels.to_le_bytes());
    for sample in &audio.samples {
        hasher.update(sample.to_bits().to_le_bytes());
    }
    let hex = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("sha256:{hex}")
}

fn recipe_sha256(recipe: &AlignmentRecipe) -> Result<String> {
    let bytes = serde_json::to_vec(recipe)
        .map_err(|error| invalid(format!("failed to serialize alignment recipe: {error}")))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let hex = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(format!("sha256:{hex}"))
}

fn require_non_empty(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        Err(invalid(format!("{field} must not be empty")))
    } else {
        Ok(())
    }
}

fn compare_candidates(left: &AlignmentCandidate, right: &AlignmentCandidate) -> Ordering {
    right
        .fused_confidence()
        .partial_cmp(&left.fused_confidence())
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.start_frame.cmp(&right.start_frame))
        .then_with(|| left.end_frame.cmp(&right.end_frame))
        .then_with(|| left.source.stable_id().cmp(&right.source.stable_id()))
}

fn issue(code: &str, detail: &str, expected_index: usize) -> PhoneticSegmentationIssue {
    PhoneticSegmentationIssue {
        code: code.into(),
        detail: detail.into(),
        expected_index: Some(expected_index),
    }
}

fn deduplicate_sources(sources: &mut Vec<AlignmentSourceIdentity>) {
    sources.sort_by_key(AlignmentSourceIdentity::stable_id);
    sources.dedup_by(|left, right| left.stable_id() == right.stable_id());
}

fn unaligned_regions(
    segments: &[PhoneticSegment],
    audio_frames: u64,
    sample_rate_hz: u32,
) -> Vec<UnalignedRegion> {
    let mut cursor = 0_u64;
    let mut regions = Vec::new();
    for interval in segments
        .iter()
        .filter_map(|segment| segment.interval.as_ref())
    {
        if cursor < interval.start_frame {
            regions.push(UnalignedRegion {
                interval: FrameInterval {
                    start_frame: cursor,
                    end_frame: interval.start_frame,
                    sample_rate_hz,
                },
                state: "unaligned_not_assumed_silence".into(),
            });
        }
        cursor = interval.end_frame;
    }
    if cursor < audio_frames {
        regions.push(UnalignedRegion {
            interval: FrameInterval {
                start_frame: cursor,
                end_frame: audio_frames,
                sample_rate_hz,
            },
            state: "unaligned_not_assumed_silence".into(),
        });
    }
    regions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio(frames: usize) -> AudioBuffer {
        AudioBuffer {
            samples: vec![0.1; frames],
            sample_rate_hz: 1_000,
            channels: 1,
        }
    }

    fn expected(symbol: &str) -> ExpectedSegment {
        ExpectedSegment {
            symbol: symbol.into(),
            kind: SegmentKind::Phone,
            inventory_membership: InventoryMembership::Known,
            language_tag: "mul".into(),
            inventory_id: "fixture-ipa".into(),
            pronunciation_source: "fixture-pronunciation-v1".into(),
            evidence_links: PhoneticEvidenceLinks::default(),
        }
    }

    fn source(model: &str) -> AlignmentSourceIdentity {
        AlignmentSourceIdentity {
            provider: "listenbury-reference".into(),
            model: model.into(),
            version: "1".into(),
            artifact_id: None,
        }
    }

    fn candidate(index: usize, start: u64, end: u64, confidence: f32) -> AlignmentCandidate {
        AlignmentCandidate {
            expected_index: index,
            start_frame: start,
            end_frame: end,
            confidence,
            boundary_origin: PhoneticBoundaryOrigin::Inferred,
            source: source("fixture"),
            evidence: AlignmentEvidence::default(),
        }
    }

    fn recipe(
        expected: Vec<ExpectedSegment>,
        candidates: Vec<AlignmentCandidate>,
    ) -> AlignmentRecipe {
        AlignmentRecipe {
            schema_version: ALIGNMENT_RECIPE_SCHEMA_VERSION,
            audio_artifact_id: "fixture.wav".into(),
            expected_audio_sha256: None,
            transcript: Some("fixture".into()),
            expected,
            candidates,
            context: PhoneticSegmentationContext {
                graph_id: "fixture-graph".into(),
                graph_revision: 7,
                recipe_id: "fixture-recipe".into(),
                execution_record_id: "fixture-run".into(),
                session_id: None,
                audio_span_id: None,
                runtime: "tongues-test".into(),
                runtime_version: env!("CARGO_PKG_VERSION").into(),
            },
        }
    }

    #[test]
    fn reference_fixture_matches_listenbury_weighted_winner() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../fixtures/phonetic-segmentation/listenbury-fusion-v1.json"
        ))
        .unwrap();
        let candidates: Vec<AlignmentCandidate> =
            serde_json::from_value(fixture["candidates"].clone()).unwrap();
        let artifact = PhoneticSegmentationEngine::default()
            .segment_recipe(&audio(500), &recipe(vec![expected("t")], candidates))
            .unwrap();
        let segment = &artifact.segments[0];
        assert_eq!(segment.interval.as_ref().unwrap().start_frame, 120);
        assert!(
            (segment.confidence.unwrap()
                - fixture["expected_fused_confidence"].as_f64().unwrap() as f32)
                .abs()
                <= fixture["confidence_tolerance"].as_f64().unwrap() as f32
        );
    }

    #[test]
    fn emits_monotonic_non_overlapping_intervals_for_repeated_phones() {
        let artifact = PhoneticSegmentationEngine::default()
            .segment_recipe(
                &audio(500),
                &recipe(
                    vec![expected("t"), expected("t")],
                    vec![candidate(0, 100, 180, 0.9), candidate(1, 180, 260, 0.9)],
                ),
            )
            .unwrap();
        assert_eq!(artifact.readiness, PhoneticSegmentationReadiness::Ready);
        assert_eq!(artifact.segments[0].symbol, artifact.segments[1].symbol);
        assert_eq!(
            artifact.segments[0].interval.as_ref().unwrap().end_frame,
            artifact.segments[1].interval.as_ref().unwrap().start_frame
        );
    }

    #[test]
    fn missing_and_low_confidence_evidence_never_fabricates_boundaries() {
        let artifact = PhoneticSegmentationEngine::default()
            .segment_recipe(
                &audio(500),
                &recipe(
                    vec![expected("a"), expected("b")],
                    vec![candidate(0, 10, 50, 0.4)],
                ),
            )
            .unwrap();
        assert_eq!(
            artifact.readiness,
            PhoneticSegmentationReadiness::Unsupported
        );
        assert_eq!(
            artifact.segments[0].status,
            PhoneticSegmentStatus::LowConfidence
        );
        assert_eq!(
            artifact.segments[1].status,
            PhoneticSegmentStatus::MissingEvidence
        );
        assert!(artifact
            .segments
            .iter()
            .all(|segment| segment.interval.is_none()));
    }

    #[test]
    fn unknown_overlap_and_clipped_audio_are_explicit() {
        let mut unknown = expected("?");
        unknown.inventory_membership = InventoryMembership::Unknown;
        let artifact = PhoneticSegmentationEngine::default()
            .segment_recipe(
                &audio(100),
                &recipe(
                    vec![expected("a"), expected("b"), expected("c"), unknown],
                    vec![
                        candidate(0, 10, 60, 0.9),
                        candidate(1, 50, 80, 0.9),
                        candidate(2, 80, 120, 0.9),
                        candidate(3, 90, 100, 0.9),
                    ],
                ),
            )
            .unwrap();
        assert_eq!(
            artifact.segments[1].status,
            PhoneticSegmentStatus::InconsistentEvidence
        );
        assert_eq!(artifact.segments[2].status, PhoneticSegmentStatus::Clipped);
        assert_eq!(
            artifact.segments[3].status,
            PhoneticSegmentStatus::UnknownSymbol
        );
        assert_eq!(
            artifact.segments[2].interval.as_ref().unwrap().end_frame,
            100
        );
    }

    #[test]
    fn missing_audio_and_mismatched_audio_fail_closed() {
        let engine = PhoneticSegmentationEngine::default();
        let request = recipe(vec![expected("a")], vec![candidate(0, 0, 1, 0.9)]);
        assert!(engine.segment_recipe(&audio(0), &request).is_err());

        let mut mismatch = request;
        mismatch.expected_audio_sha256 = Some("sha256:not-the-audio".into());
        assert!(engine.segment_recipe(&audio(10), &mismatch).is_err());
    }

    #[test]
    fn silence_without_expected_symbols_is_ready_without_inventing_phones() {
        let artifact = PhoneticSegmentationEngine::default()
            .segment_recipe(&audio(100), &recipe(Vec::new(), Vec::new()))
            .unwrap();
        assert_eq!(artifact.readiness, PhoneticSegmentationReadiness::Ready);
        assert!(artifact.segments.is_empty());
        assert_eq!(artifact.unaligned_regions.len(), 1);
        assert_eq!(
            artifact.unaligned_regions[0].state,
            "unaligned_not_assumed_silence"
        );
    }

    #[test]
    fn pause_and_word_boundary_segments_use_supplied_spans() {
        let mut pause = expected("<pause>");
        pause.kind = SegmentKind::Pause;
        let mut boundary = expected("#");
        boundary.kind = SegmentKind::WordBoundary;
        let artifact = PhoneticSegmentationEngine::default()
            .segment_recipe(
                &audio(500),
                &recipe(
                    vec![pause, boundary],
                    vec![candidate(0, 100, 150, 0.9), candidate(1, 150, 151, 0.9)],
                ),
            )
            .unwrap();
        assert_eq!(artifact.readiness, PhoneticSegmentationReadiness::Ready);
        assert_eq!(artifact.segments[0].kind, SegmentKind::Pause);
        assert_eq!(artifact.segments[1].kind, SegmentKind::WordBoundary);
    }

    #[test]
    fn typed_artifact_attaches_timed_spans_without_fabricating_unsupported_rows() {
        let mut timed = expected("t");
        timed.evidence_links = PhoneticEvidenceLinks {
            word_span_id: Some("word:1".into()),
            transcript_span_id: Some("transcript:1".into()),
            speaker_span_id: Some("speaker:1".into()),
        };
        let mut unknown = expected("�");
        unknown.kind = SegmentKind::Unknown;
        unknown.inventory_membership = InventoryMembership::Unknown;
        let mut recipe = recipe(
            vec![timed, unknown],
            vec![AlignmentCandidate {
                boundary_origin: PhoneticBoundaryOrigin::SourceProvided,
                ..candidate(0, 100, 200, 0.95)
            }],
        );
        recipe.context.session_id = Some("session:phonetic".into());
        recipe.context.audio_span_id = Some("audio:1".into());
        let artifact = PhoneticSegmentationEngine::default()
            .segment_recipe(&audio(500), &recipe)
            .unwrap();
        assert_eq!(artifact.readiness, PhoneticSegmentationReadiness::Partial);

        let spans = [
            ("audio:1", SpanModality::Audio),
            ("transcript:1", SpanModality::Transcript),
            ("word:1", SpanModality::Word),
            ("speaker:1", SpanModality::Speaker),
        ]
        .into_iter()
        .map(|(id, modality)| TimelineSpan {
            id: id.into(),
            start_ms: 1,
            end_ms: 400,
            modality,
            metadata: BTreeMap::new(),
        })
        .collect();
        let mut session =
            SpeechTimelineSession::new("session:phonetic", spans, Vec::new()).unwrap();
        let artifact_id = artifact.attach_to_timeline(&mut session).unwrap();

        assert_eq!(session.attachments.len(), 1);
        assert_eq!(session.attachments[0].artifact_id, artifact_id);
        let phones = session
            .evidence
            .iter()
            .filter(|span| span.modality == SpanModality::Phone)
            .collect::<Vec<_>>();
        assert_eq!(phones.len(), 1);
        assert_eq!(phones[0].metadata["symbol"], "t");
        assert_eq!(phones[0].metadata["boundary_origin"], "source_provided");
        assert_eq!(session.alignments.len(), 4);
        assert!(!session
            .evidence
            .iter()
            .any(|span| span.metadata.get("symbol") == Some(&serde_json::json!("�"))));
        assert_eq!(
            session.attachments[0].payload["segments"][1]["status"],
            "unknown_symbol"
        );
    }
}
