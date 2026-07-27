//! Evidence-bound phonetic segmentation.
//!
//! This module adapts Listenbury's speech-hypothesis lattice into a stricter
//! Tongues artifact contract. Competing adapter hypotheses are fused
//! deterministically, but an interval is emitted only when a source supplied
//! that interval with sufficient confidence. The engine never divides audio
//! evenly across an expected pronunciation.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectedSegment {
    pub symbol: String,
    pub kind: SegmentKind,
    pub inventory_membership: InventoryMembership,
    pub language_tag: String,
    pub inventory_id: String,
    pub pronunciation_source: String,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhoneticSegmentationContext {
    pub graph_id: String,
    pub graph_revision: u64,
    pub recipe_id: String,
    pub execution_record_id: String,
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
    pub pronunciation_source: String,
    pub language_tag: String,
    pub inventory_id: String,
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
                pronunciation_source: expected.pronunciation_source.clone(),
                language_tag: expected.language_tag.clone(),
                inventory_id: expected.inventory_id.clone(),
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
}
