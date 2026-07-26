//! TTS planning delta and delivery ledger types.
//!
//! [`UtterancePlanDelta`] describes incremental changes to a TTS synthesis
//! schedule. Each change is anchored to morpheme occurrence IDs (from the
//! belief state) and emission IDs (the synthesis chunks they produce).
//!
//! These types are provider-neutral: no renderer policy or PCM data lives
//! here. The delivery ledger in `tongues-tts` builds on top of these types to
//! track actual PCM buffers and enforce immutability of played audio.

use serde::{Deserialize, Serialize};

use crate::evidence::EvidenceProvenance;
use crate::ids::{EmissionId, MorphemeOccurrenceId, UtteranceId};

/// An incremental change set applied to a TTS utterance plan.
///
/// Deltas may describe predicted continuations that have not yet been observed.
/// Predicted emissions are associated with morpheme occurrences the belief
/// state has not yet committed, and the ledger guarantees they can never be
/// played until committed.
///
/// # Revision safety
///
/// The `revision` field is monotonically increasing. Applying a delta whose
/// `revision` is less than or equal to the current revision is an error.
/// Applying deltas out of order violates the immutability guarantee for played
/// PCM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UtterancePlanDelta {
    /// Identity of the utterance being revised.
    pub utterance_id: UtteranceId,
    /// Monotonically increasing revision counter for this utterance.
    pub revision: u64,
    /// Ordered list of changes to apply atomically.
    pub changes: Vec<PlanChange>,
    /// Source and method provenance for the delta.
    pub provenance: EvidenceProvenance,
}

/// One atomic change to a planned utterance's synthesis schedule.
///
/// Changes are applied in order within a single [`UtterancePlanDelta`].
/// A `ReplaceSuffix` takes effect before any `Schedule` changes that follow
/// it in the same delta so that the replacement and the new suffix are
/// described in one round-trip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum PlanChange {
    /// Schedule a new emission for a morpheme occurrence.
    ///
    /// When `committed` is `false` the emission is *predicted*: it may be
    /// synthesized and queued but must never be played. The ledger enforces
    /// this invariant without knowing which renderer produced the PCM.
    ///
    /// When `committed` is `true` the emission may advance through the full
    /// delivery pipeline including `Played`.
    Schedule {
        /// Which morpheme occurrence this emission represents.
        morpheme_occurrence_id: MorphemeOccurrenceId,
        /// Unique identity of the synthesis chunk to be produced.
        emission_id: EmissionId,
        /// Whether the underlying morpheme has been observed and committed.
        /// Only committed emissions may advance to `Played`.
        committed: bool,
    },

    /// Promote a predicted emission to committed status.
    ///
    /// Once committed the emission may advance through the delivery pipeline
    /// up to and including `Played`. Committing an already-committed emission
    /// is a no-op.
    Commit {
        /// The emission to promote.
        emission_id: EmissionId,
        /// The morpheme occurrence that has now been confirmed.
        morpheme_occurrence_id: MorphemeOccurrenceId,
    },

    /// Cancel and silence a planned, synthesized, or queued emission.
    ///
    /// The emission must not have been played. Cancelling a played emission
    /// is an error that surfaces in the ledger as a
    /// [`PlanDeltaError::CannotCancelPlayedEmission`]. The associated PCM
    /// is discarded without being emitted to the audio device.
    Cancel {
        /// The emission to cancel.
        emission_id: EmissionId,
    },

    /// Silently replace an unplayed suffix of the plan.
    ///
    /// All emissions at or after `from_emission_id` that have not yet been
    /// played are discarded and their PCM is never sent to the audio device.
    /// The delivery ledger crossfades the regenerated suffix into the
    /// already-played prefix so that no committed samples are dropped or
    /// duplicated.
    ReplaceSuffix {
        /// The first emission in the suffix to be replaced. Must not be in
        /// `Played` state.
        from_emission_id: EmissionId,
    },
}

/// Errors that can occur when applying a [`UtterancePlanDelta`] to a ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanDeltaError {
    /// The delta's `revision` is not strictly greater than the current ledger
    /// revision, violating monotonicity.
    OutOfOrderRevision {
        /// The current ledger revision.
        current: u64,
        /// The revision in the rejected delta.
        received: u64,
    },
    /// The delta targets an utterance different from the ledger's utterance.
    UtteranceMismatch {
        expected: UtteranceId,
        received: UtteranceId,
    },
    /// A `Cancel` change targeted an emission that has already been played.
    CannotCancelPlayedEmission { emission_id: EmissionId },
    /// A `ReplaceSuffix` change targeted an emission that has already been
    /// played.
    CannotReplacePlayedEmission { emission_id: EmissionId },
    /// A `Schedule` or `Commit` change used an emission ID that is already
    /// present in the ledger.
    DuplicateEmissionId { emission_id: EmissionId },
    /// A change referenced an emission ID that does not exist in the ledger.
    UnknownEmissionId { emission_id: EmissionId },
}

impl std::fmt::Display for PlanDeltaError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfOrderRevision { current, received } => write!(
                formatter,
                "out-of-order plan delta: current revision {current}, received {received}"
            ),
            Self::UtteranceMismatch { expected, received } => write!(
                formatter,
                "plan delta utterance mismatch: expected {}, received {}",
                expected.0, received.0
            ),
            Self::CannotCancelPlayedEmission { emission_id } => {
                write!(formatter, "cannot cancel played emission {}", emission_id.0)
            }
            Self::CannotReplacePlayedEmission { emission_id } => write!(
                formatter,
                "cannot replace played emission {}",
                emission_id.0
            ),
            Self::DuplicateEmissionId { emission_id } => {
                write!(formatter, "duplicate emission id {}", emission_id.0)
            }
            Self::UnknownEmissionId { emission_id } => {
                write!(formatter, "unknown emission id {}", emission_id.0)
            }
        }
    }
}

impl std::error::Error for PlanDeltaError {}

/// Per-stage latency and throughput metrics for a TTS playback session.
///
/// Collected by the delivery ledger in `tongues-tts` and reported by the
/// vertical slice benchmarks. All latency values are in milliseconds.
///
/// Fields are `Option` so that partial sessions (e.g. cancelled before first
/// audio) can still report meaningful partial metrics.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LedgerMetrics {
    /// Wall-clock latency from plan start to first PCM delivered to the device.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_audio_latency_ms: Option<f64>,

    /// Number of suffix replacements that occurred during this session.
    #[serde(default)]
    pub suffix_regeneration_count: u64,

    /// Per-regeneration latency from `ReplaceSuffix` to first new audio.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suffix_regeneration_latency_ms: Vec<f64>,

    /// Number of explicit cancellations during this session.
    #[serde(default)]
    pub cancellation_count: u64,

    /// Per-cancellation latency from `Cancel` delta to silence confirmed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cancellation_latency_ms: Vec<f64>,

    /// Number of plan revisions (deltas) applied during this session.
    #[serde(default)]
    pub revision_count: u64,

    /// Maximum number of entries simultaneously present in the playback queue.
    #[serde(default)]
    pub queue_depth_high_water: usize,

    /// Crossfade and withheld-tail overhead, recorded per revision event.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub crossfade_cost_ms: Vec<f64>,

    /// Real-time factor: ratio of synthesis wall time to audio duration.
    /// Values below 1.0 indicate faster-than-real-time synthesis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steady_state_rtf: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::{EvidenceProvenance, EvidenceSource};

    fn provenance() -> EvidenceProvenance {
        EvidenceProvenance {
            source: EvidenceSource::Manual,
            method: "test".into(),
            version: None,
        }
    }

    #[test]
    fn plan_delta_serializes_round_trip() {
        let delta = UtterancePlanDelta {
            utterance_id: UtteranceId("u1".into()),
            revision: 1,
            changes: vec![
                PlanChange::Schedule {
                    morpheme_occurrence_id: MorphemeOccurrenceId("m1".into()),
                    emission_id: EmissionId("e1".into()),
                    committed: true,
                },
                PlanChange::ReplaceSuffix {
                    from_emission_id: EmissionId("e1".into()),
                },
                PlanChange::Cancel {
                    emission_id: EmissionId("e2".into()),
                },
                PlanChange::Commit {
                    emission_id: EmissionId("e3".into()),
                    morpheme_occurrence_id: MorphemeOccurrenceId("m3".into()),
                },
            ],
            provenance: provenance(),
        };

        let json = serde_json::to_string(&delta).expect("serialization failed");
        let round_tripped: UtterancePlanDelta =
            serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(delta, round_tripped);
    }

    #[test]
    fn ledger_metrics_default_is_empty() {
        let metrics = LedgerMetrics::default();
        assert!(metrics.first_audio_latency_ms.is_none());
        assert_eq!(metrics.suffix_regeneration_count, 0);
        assert_eq!(metrics.revision_count, 0);
        assert_eq!(metrics.queue_depth_high_water, 0);
    }

    #[test]
    fn plan_delta_error_display() {
        let error = PlanDeltaError::OutOfOrderRevision {
            current: 5,
            received: 3,
        };
        let message = error.to_string();
        assert!(message.contains("5"));
        assert!(message.contains("3"));
    }
}
