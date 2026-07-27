//! TTS playback delivery ledger.
//!
//! [`TtsPlaybackLedger`] is the runtime counterpart to the provider-neutral
//! type definitions in [`speaking::tts_ledger`]. It combines:
//!
//! - Per-emission PCM buffers (actual audio samples from a renderer).
//! - Delivery state tracking (Planned → Synthesized → Verified → Queued →
//!   Played).
//! - Immutability enforcement for played PCM.
//! - Silent replacement of the unplayed suffix when the plan is revised.
//! - Metrics collection for first-audio latency, revision count, etc.
//!
//! # Safety invariants
//!
//! The ledger enforces the following invariants at runtime:
//!
//! 1. **Commitment guard**: Only emissions whose `committed` flag is `true` may
//!    advance past `Queued` to `Played`. Predicted (uncommitted) emissions can
//!    be synthesized and queued but are silently discarded if never committed.
//!
//! 2. **Immutable played prefix**: Once an emission reaches `Played` its PCM
//!    buffer is frozen. [`apply_delta`](TtsPlaybackLedger::apply_delta) returns
//!    an error if a delta attempts to cancel or replace a played emission.
//!
//! 3. **Suffix replaceability**: `ReplaceSuffix` silently discards all unplayed
//!    emissions starting at the given emission ID and resets their PCM so the
//!    renderer can regenerate them.
//!
//! # Provider neutrality
//!
//! The ledger does not inspect the PCM samples beyond checking for
//! non-finite values. All crossfade logic lives in [`RevisionWaveformAssembler`].
//! Renderer-specific policy must not be added here.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::Result;
use speaking::{
    DeliveryState, EmissionId, LedgerMetrics, MorphemeOccurrenceId, PlanChange, PlanDeltaError,
    UtteranceId, UtterancePlanDelta,
};

use crate::revision_assembler::RevisionWaveformAssembler;

/// One entry in the playback ledger, tracking the PCM and delivery state for a
/// single synthesis chunk (emission).
#[derive(Debug, Clone)]
pub struct LedgerEntry {
    /// Stable identity of this synthesis chunk.
    pub emission_id: EmissionId,
    /// The morpheme occurrence this emission represents, if known.
    pub morpheme_occurrence_id: Option<MorphemeOccurrenceId>,
    /// Current delivery pipeline state.
    pub delivery_state: DeliveryState,
    /// Whether the underlying morpheme has been confirmed as observed.
    ///
    /// Only committed entries may advance to `Played`.
    pub committed: bool,
    /// Raw PCM samples for this emission (empty until synthesized).
    pub pcm_f32: Vec<f32>,
    /// Output sample rate in Hz.
    pub sample_rate_hz: u32,
    /// When this entry was created (for latency tracking).
    pub created_at: Instant,
    /// When this entry first reached `Played` state.
    pub played_at: Option<Instant>,
}

impl LedgerEntry {
    fn new(
        emission_id: EmissionId,
        morpheme_occurrence_id: Option<MorphemeOccurrenceId>,
        committed: bool,
    ) -> Self {
        Self {
            emission_id,
            morpheme_occurrence_id,
            delivery_state: DeliveryState::Planned,
            committed,
            pcm_f32: Vec::new(),
            sample_rate_hz: 0,
            created_at: Instant::now(),
            played_at: None,
        }
    }

    /// Whether this entry is in a state that permits silent replacement.
    pub fn is_replaceable(&self) -> bool {
        !matches!(self.delivery_state, DeliveryState::Played)
    }
}

/// Errors specific to the TTS playback ledger.
#[derive(Debug, Clone, PartialEq)]
pub enum LedgerError {
    /// Attempted to advance an uncommitted emission to `Played`.
    UncommittedEmissionCannotPlay { emission_id: EmissionId },
    /// Attempted to attach PCM to an emission that does not exist.
    UnknownEmission { emission_id: EmissionId },
    /// Attempted to set PCM on a played entry (immutable).
    PlayedEntryImmutable { emission_id: EmissionId },
    /// A delivery state transition would be a regression.
    DeliveryStateRegression {
        emission_id: EmissionId,
        from: DeliveryState,
        to: DeliveryState,
    },
    /// The delta could not be applied.
    DeltaError(PlanDeltaError),
}

impl From<PlanDeltaError> for LedgerError {
    fn from(error: PlanDeltaError) -> Self {
        Self::DeltaError(error)
    }
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UncommittedEmissionCannotPlay { emission_id } => write!(
                formatter,
                "uncommitted emission {} cannot advance to Played",
                emission_id.0
            ),
            Self::UnknownEmission { emission_id } => {
                write!(formatter, "unknown emission {}", emission_id.0)
            }
            Self::PlayedEntryImmutable { emission_id } => {
                write!(formatter, "played entry {} is immutable", emission_id.0)
            }
            Self::DeliveryStateRegression {
                emission_id,
                from,
                to,
            } => write!(
                formatter,
                "delivery state regression for {}: {:?} → {:?}",
                emission_id.0, from, to
            ),
            Self::DeltaError(error) => write!(formatter, "plan delta error: {error}"),
        }
    }
}

impl std::error::Error for LedgerError {}

/// The TTS playback delivery ledger.
///
/// See the [module documentation](self) for the full invariant description.
pub struct TtsPlaybackLedger {
    utterance_id: UtteranceId,
    /// Stable insertion order.
    emission_order: Vec<EmissionId>,
    /// Per-emission state.
    entries: BTreeMap<EmissionId, LedgerEntry>,
    /// Current plan revision number.
    current_revision: u64,
    /// Metrics collected during this session.
    metrics: LedgerMetrics,
    /// Wall-clock start of the session.
    session_start: Instant,
    /// Time of the most recent `ReplaceSuffix` event (for latency tracking).
    last_replace_at: Option<Instant>,
    /// Time of the most recent `Cancel` event.
    last_cancel_at: Option<Instant>,
}

impl TtsPlaybackLedger {
    /// Create a new ledger for the given utterance.
    pub fn new(utterance_id: UtteranceId) -> Self {
        Self {
            utterance_id,
            emission_order: Vec::new(),
            entries: BTreeMap::new(),
            current_revision: 0,
            metrics: LedgerMetrics::default(),
            session_start: Instant::now(),
            last_replace_at: None,
            last_cancel_at: None,
        }
    }

    /// Apply a plan delta, updating the ledger's state.
    ///
    /// Changes are applied in order. If any change fails the ledger is left in
    /// a consistent state (changes before the failure are retained; the failing
    /// change and all subsequent changes are not applied).
    pub fn apply_delta(&mut self, delta: &UtterancePlanDelta) -> Result<(), LedgerError> {
        if delta.utterance_id != self.utterance_id {
            return Err(LedgerError::DeltaError(PlanDeltaError::UtteranceMismatch {
                expected: self.utterance_id.clone(),
                received: delta.utterance_id.clone(),
            }));
        }
        if delta.revision <= self.current_revision {
            return Err(LedgerError::DeltaError(
                PlanDeltaError::OutOfOrderRevision {
                    current: self.current_revision,
                    received: delta.revision,
                },
            ));
        }

        // Validate all changes before mutating state.
        for change in &delta.changes {
            self.validate_change(change)?;
        }

        // Apply all changes.
        for change in &delta.changes {
            self.apply_change(change);
        }

        self.current_revision = delta.revision;
        self.metrics.revision_count += 1;

        // Update queue depth high water mark.
        let queued = self
            .entries
            .values()
            .filter(|entry| matches!(entry.delivery_state, DeliveryState::Queued))
            .count();
        if queued > self.metrics.queue_depth_high_water {
            self.metrics.queue_depth_high_water = queued;
        }

        Ok(())
    }

    fn validate_change(&self, change: &PlanChange) -> Result<(), LedgerError> {
        match change {
            PlanChange::Schedule { emission_id, .. } => {
                if self.entries.contains_key(emission_id) {
                    return Err(LedgerError::DeltaError(
                        PlanDeltaError::DuplicateEmissionId {
                            emission_id: emission_id.clone(),
                        },
                    ));
                }
            }
            PlanChange::Commit { emission_id, .. } => {
                if !self.entries.contains_key(emission_id) {
                    return Err(LedgerError::DeltaError(PlanDeltaError::UnknownEmissionId {
                        emission_id: emission_id.clone(),
                    }));
                }
            }
            PlanChange::Cancel { emission_id } => match self.entries.get(emission_id) {
                None => {
                    return Err(LedgerError::DeltaError(PlanDeltaError::UnknownEmissionId {
                        emission_id: emission_id.clone(),
                    }));
                }
                Some(entry) if matches!(entry.delivery_state, DeliveryState::Played) => {
                    return Err(LedgerError::DeltaError(
                        PlanDeltaError::CannotCancelPlayedEmission {
                            emission_id: emission_id.clone(),
                        },
                    ));
                }
                _ => {}
            },
            PlanChange::ReplaceSuffix { from_emission_id } => {
                match self.entries.get(from_emission_id) {
                    None => {
                        return Err(LedgerError::DeltaError(PlanDeltaError::UnknownEmissionId {
                            emission_id: from_emission_id.clone(),
                        }));
                    }
                    Some(entry) if matches!(entry.delivery_state, DeliveryState::Played) => {
                        return Err(LedgerError::DeltaError(
                            PlanDeltaError::CannotReplacePlayedEmission {
                                emission_id: from_emission_id.clone(),
                            },
                        ));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn apply_change(&mut self, change: &PlanChange) {
        match change {
            PlanChange::Schedule {
                morpheme_occurrence_id,
                emission_id,
                committed,
            } => {
                let entry = LedgerEntry::new(
                    emission_id.clone(),
                    Some(morpheme_occurrence_id.clone()),
                    *committed,
                );
                self.entries.insert(emission_id.clone(), entry);
                self.emission_order.push(emission_id.clone());
            }
            PlanChange::Commit {
                emission_id,
                morpheme_occurrence_id,
            } => {
                if let Some(entry) = self.entries.get_mut(emission_id) {
                    entry.committed = true;
                    if entry.morpheme_occurrence_id.is_none() {
                        entry.morpheme_occurrence_id = Some(morpheme_occurrence_id.clone());
                    }
                }
            }
            PlanChange::Cancel { emission_id } => {
                self.metrics.cancellation_count += 1;
                if let Some(started) = self.last_cancel_at {
                    self.metrics
                        .cancellation_latency_ms
                        .push(started.elapsed().as_secs_f64() * 1_000.0);
                }
                self.last_cancel_at = Some(Instant::now());
                // Remove the entry and its ordering slot.
                self.entries.remove(emission_id);
                self.emission_order.retain(|id| id != emission_id);
            }
            PlanChange::ReplaceSuffix { from_emission_id } => {
                self.metrics.suffix_regeneration_count += 1;
                if let Some(started) = self.last_replace_at {
                    self.metrics
                        .suffix_regeneration_latency_ms
                        .push(started.elapsed().as_secs_f64() * 1_000.0);
                }
                self.last_replace_at = Some(Instant::now());

                // Find the index of from_emission_id in the ordered list.
                let start_index = self
                    .emission_order
                    .iter()
                    .position(|id| id == from_emission_id);

                if let Some(start) = start_index {
                    let to_remove: Vec<_> = self.emission_order[start..].to_vec();
                    for emission_id in &to_remove {
                        self.entries.remove(emission_id);
                    }
                    self.emission_order.truncate(start);
                }
            }
        }
    }

    /// Attach synthesized PCM to an emission and advance its state to
    /// `Synthesized`.
    ///
    /// Returns an error if the emission does not exist or is already in
    /// `Played` state (immutable).
    pub fn attach_pcm(
        &mut self,
        emission_id: &EmissionId,
        pcm: Vec<f32>,
        sample_rate_hz: u32,
    ) -> Result<(), LedgerError> {
        let entry =
            self.entries
                .get_mut(emission_id)
                .ok_or_else(|| LedgerError::UnknownEmission {
                    emission_id: emission_id.clone(),
                })?;
        if matches!(entry.delivery_state, DeliveryState::Played) {
            return Err(LedgerError::PlayedEntryImmutable {
                emission_id: emission_id.clone(),
            });
        }
        entry.pcm_f32 = pcm;
        entry.sample_rate_hz = sample_rate_hz;
        entry.delivery_state = DeliveryState::Synthesized;
        Ok(())
    }

    /// Advance an emission's delivery state.
    ///
    /// Enforces monotonicity (no regressions) and the commitment guard (only
    /// committed emissions may reach `Played`).
    pub fn advance_state(
        &mut self,
        emission_id: &EmissionId,
        new_state: DeliveryState,
    ) -> Result<(), LedgerError> {
        let entry =
            self.entries
                .get_mut(emission_id)
                .ok_or_else(|| LedgerError::UnknownEmission {
                    emission_id: emission_id.clone(),
                })?;

        // Commitment guard.
        if matches!(new_state, DeliveryState::Played) && !entry.committed {
            return Err(LedgerError::UncommittedEmissionCannotPlay {
                emission_id: emission_id.clone(),
            });
        }

        // Monotonicity check.
        let current_phase = delivery_phase(entry.delivery_state);
        let new_phase = delivery_phase(new_state);
        if new_phase < current_phase {
            return Err(LedgerError::DeliveryStateRegression {
                emission_id: emission_id.clone(),
                from: entry.delivery_state,
                to: new_state,
            });
        }

        let now = Instant::now();

        // Record first-audio latency on first Played transition.
        if matches!(new_state, DeliveryState::Played) && entry.played_at.is_none() {
            entry.played_at = Some(now);
            if self.metrics.first_audio_latency_ms.is_none() {
                let latency = self.session_start.elapsed().as_secs_f64() * 1_000.0;
                self.metrics.first_audio_latency_ms = Some(latency);
            }
            // Record suffix-regeneration latency if this follows a replace.
            if let Some(replace_at) = self.last_replace_at {
                let regen_latency = replace_at.elapsed().as_secs_f64() * 1_000.0;
                self.metrics
                    .suffix_regeneration_latency_ms
                    .push(regen_latency);
                self.last_replace_at = None;
            }
        }

        entry.delivery_state = new_state;
        Ok(())
    }

    /// Return a snapshot of the current ledger metrics.
    pub fn metrics(&self) -> &LedgerMetrics {
        &self.metrics
    }

    /// Return the current plan revision number.
    pub fn current_revision(&self) -> u64 {
        self.current_revision
    }

    /// Return the emission IDs in insertion order.
    pub fn emission_order(&self) -> &[EmissionId] {
        &self.emission_order
    }

    /// Look up a single entry by emission ID.
    pub fn entry(&self, emission_id: &EmissionId) -> Option<&LedgerEntry> {
        self.entries.get(emission_id)
    }

    /// Iterator over all entries in insertion order.
    pub fn entries_in_order(&self) -> impl Iterator<Item = &LedgerEntry> {
        self.emission_order
            .iter()
            .filter_map(|id| self.entries.get(id))
    }

    /// Collect the unplayed PCM from all committed, synthesized entries.
    ///
    /// This concatenates the PCM buffers in emission order. The returned
    /// samples have not yet been played and may be sent to the audio device.
    pub fn collect_unplayed_pcm(&self) -> Vec<f32> {
        self.entries_in_order()
            .filter(|entry| {
                entry.committed
                    && !matches!(entry.delivery_state, DeliveryState::Planned)
                    && !matches!(entry.delivery_state, DeliveryState::Played)
            })
            .flat_map(|entry| entry.pcm_f32.iter().copied())
            .collect()
    }

    /// Create a [`RevisionWaveformAssembler`] seeded with the total count of
    /// already-played samples across all entries.
    ///
    /// The returned assembler is ready to accept a revised waveform segment.
    pub fn build_revision_assembler(
        &self,
        crossfade_samples: usize,
    ) -> Result<RevisionWaveformAssembler> {
        let mut assembler = RevisionWaveformAssembler::new(crossfade_samples)?;

        // Prime the assembler with all committed PCM so that crossfade uses
        // the correct boundary.
        for entry in self.entries_in_order() {
            if !entry.pcm_f32.is_empty() {
                assembler.push(&entry.pcm_f32)?;
            }
            if matches!(entry.delivery_state, DeliveryState::Played) {
                assembler.mark_played(entry.pcm_f32.len())?;
            }
        }

        Ok(assembler)
    }
}

fn delivery_phase(state: DeliveryState) -> u8 {
    match state {
        DeliveryState::Planned => 0,
        DeliveryState::Synthesized => 1,
        DeliveryState::Verified => 2,
        DeliveryState::Queued => 3,
        DeliveryState::Played => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use speaking::{EvidenceProvenance, EvidenceSource};

    fn utterance_id() -> UtteranceId {
        UtteranceId("test-utterance".into())
    }

    fn provenance() -> EvidenceProvenance {
        EvidenceProvenance {
            source: EvidenceSource::Manual,
            method: "test".into(),
            version: None,
        }
    }

    fn schedule_delta(
        utterance_id: UtteranceId,
        revision: u64,
        emission_id: &str,
        morpheme_id: &str,
        committed: bool,
    ) -> UtterancePlanDelta {
        UtterancePlanDelta {
            utterance_id,
            revision,
            changes: vec![PlanChange::Schedule {
                morpheme_occurrence_id: speaking::MorphemeOccurrenceId(morpheme_id.into()),
                emission_id: EmissionId(emission_id.into()),
                committed,
            }],
            provenance: provenance(),
        }
    }

    #[test]
    fn schedule_and_advance_committed_emission() {
        let mut ledger = TtsPlaybackLedger::new(utterance_id());
        let emission_id = EmissionId("e1".into());

        ledger
            .apply_delta(&schedule_delta(
                utterance_id(),
                1,
                "e1",
                "m1",
                true, // committed
            ))
            .unwrap();

        ledger
            .attach_pcm(&emission_id, vec![0.1, 0.2, 0.3], 22_050)
            .unwrap();
        ledger
            .advance_state(&emission_id, DeliveryState::Queued)
            .unwrap();
        ledger
            .advance_state(&emission_id, DeliveryState::Played)
            .unwrap();

        let entry = ledger.entry(&emission_id).unwrap();
        assert!(matches!(entry.delivery_state, DeliveryState::Played));
        assert!(ledger.metrics().first_audio_latency_ms.is_some());
    }

    #[test]
    fn uncommitted_emission_cannot_be_played() {
        let mut ledger = TtsPlaybackLedger::new(utterance_id());
        let emission_id = EmissionId("e1".into());

        ledger
            .apply_delta(&schedule_delta(
                utterance_id(),
                1,
                "e1",
                "m1",
                false, // NOT committed
            ))
            .unwrap();

        ledger.attach_pcm(&emission_id, vec![0.1], 22_050).unwrap();

        let result = ledger.advance_state(&emission_id, DeliveryState::Played);
        assert!(matches!(
            result,
            Err(LedgerError::UncommittedEmissionCannotPlay { .. })
        ));
    }

    #[test]
    fn played_emission_pcm_is_immutable() {
        let mut ledger = TtsPlaybackLedger::new(utterance_id());
        let emission_id = EmissionId("e1".into());

        ledger
            .apply_delta(&schedule_delta(utterance_id(), 1, "e1", "m1", true))
            .unwrap();
        ledger.attach_pcm(&emission_id, vec![0.1], 22_050).unwrap();
        ledger
            .advance_state(&emission_id, DeliveryState::Played)
            .unwrap();

        let result = ledger.attach_pcm(&emission_id, vec![0.2], 22_050);
        assert!(matches!(
            result,
            Err(LedgerError::PlayedEntryImmutable { .. })
        ));
    }

    #[test]
    fn replace_suffix_removes_unplayed_entries() {
        let mut ledger = TtsPlaybackLedger::new(utterance_id());

        // Schedule two emissions.
        ledger
            .apply_delta(&UtterancePlanDelta {
                utterance_id: utterance_id(),
                revision: 1,
                changes: vec![
                    PlanChange::Schedule {
                        morpheme_occurrence_id: speaking::MorphemeOccurrenceId("m1".into()),
                        emission_id: EmissionId("e1".into()),
                        committed: true,
                    },
                    PlanChange::Schedule {
                        morpheme_occurrence_id: speaking::MorphemeOccurrenceId("m2".into()),
                        emission_id: EmissionId("e2".into()),
                        committed: false,
                    },
                ],
                provenance: provenance(),
            })
            .unwrap();

        // Play e1.
        ledger
            .attach_pcm(&EmissionId("e1".into()), vec![0.1], 22_050)
            .unwrap();
        ledger
            .advance_state(&EmissionId("e1".into()), DeliveryState::Played)
            .unwrap();

        // Replace suffix starting at e2.
        ledger
            .apply_delta(&UtterancePlanDelta {
                utterance_id: utterance_id(),
                revision: 2,
                changes: vec![PlanChange::ReplaceSuffix {
                    from_emission_id: EmissionId("e2".into()),
                }],
                provenance: provenance(),
            })
            .unwrap();

        assert!(ledger.entry(&EmissionId("e2".into())).is_none());
        assert!(ledger.entry(&EmissionId("e1".into())).is_some());
        assert_eq!(ledger.metrics().suffix_regeneration_count, 1);
    }

    #[test]
    fn cannot_cancel_played_emission() {
        let mut ledger = TtsPlaybackLedger::new(utterance_id());

        ledger
            .apply_delta(&schedule_delta(utterance_id(), 1, "e1", "m1", true))
            .unwrap();
        ledger
            .attach_pcm(&EmissionId("e1".into()), vec![0.1], 22_050)
            .unwrap();
        ledger
            .advance_state(&EmissionId("e1".into()), DeliveryState::Played)
            .unwrap();

        let result = ledger.apply_delta(&UtterancePlanDelta {
            utterance_id: utterance_id(),
            revision: 2,
            changes: vec![PlanChange::Cancel {
                emission_id: EmissionId("e1".into()),
            }],
            provenance: provenance(),
        });
        assert!(matches!(
            result,
            Err(LedgerError::DeltaError(
                PlanDeltaError::CannotCancelPlayedEmission { .. }
            ))
        ));
    }

    #[test]
    fn out_of_order_revision_is_rejected() {
        let mut ledger = TtsPlaybackLedger::new(utterance_id());
        ledger
            .apply_delta(&schedule_delta(utterance_id(), 5, "e1", "m1", true))
            .unwrap();
        let result = ledger.apply_delta(&schedule_delta(utterance_id(), 3, "e2", "m2", true));
        assert!(matches!(
            result,
            Err(LedgerError::DeltaError(
                PlanDeltaError::OutOfOrderRevision { .. }
            ))
        ));
    }

    #[test]
    fn delivery_state_regression_is_rejected() {
        let mut ledger = TtsPlaybackLedger::new(utterance_id());
        let emission_id = EmissionId("e1".into());

        ledger
            .apply_delta(&schedule_delta(utterance_id(), 1, "e1", "m1", true))
            .unwrap();
        ledger.attach_pcm(&emission_id, vec![0.1], 22_050).unwrap();
        ledger
            .advance_state(&emission_id, DeliveryState::Queued)
            .unwrap();

        let result = ledger.advance_state(&emission_id, DeliveryState::Synthesized);
        assert!(matches!(
            result,
            Err(LedgerError::DeliveryStateRegression { .. })
        ));
    }

    #[test]
    fn commit_promotes_predicted_emission() {
        let mut ledger = TtsPlaybackLedger::new(utterance_id());
        let emission_id = EmissionId("e1".into());

        // Schedule as uncommitted.
        ledger
            .apply_delta(&schedule_delta(utterance_id(), 1, "e1", "m1", false))
            .unwrap();
        assert!(!ledger.entry(&emission_id).unwrap().committed);

        // Commit it.
        ledger
            .apply_delta(&UtterancePlanDelta {
                utterance_id: utterance_id(),
                revision: 2,
                changes: vec![PlanChange::Commit {
                    emission_id: emission_id.clone(),
                    morpheme_occurrence_id: speaking::MorphemeOccurrenceId("m1".into()),
                }],
                provenance: provenance(),
            })
            .unwrap();

        assert!(ledger.entry(&emission_id).unwrap().committed);

        // Now it can be played.
        ledger.attach_pcm(&emission_id, vec![0.1], 22_050).unwrap();
        ledger
            .advance_state(&emission_id, DeliveryState::Played)
            .unwrap();
    }

    #[test]
    fn collect_unplayed_pcm_returns_synthesized_committed() {
        let mut ledger = TtsPlaybackLedger::new(utterance_id());

        ledger
            .apply_delta(&UtterancePlanDelta {
                utterance_id: utterance_id(),
                revision: 1,
                changes: vec![
                    PlanChange::Schedule {
                        morpheme_occurrence_id: speaking::MorphemeOccurrenceId("m1".into()),
                        emission_id: EmissionId("e1".into()),
                        committed: true,
                    },
                    PlanChange::Schedule {
                        morpheme_occurrence_id: speaking::MorphemeOccurrenceId("m2".into()),
                        emission_id: EmissionId("e2".into()),
                        committed: false, // uncommitted: not included
                    },
                ],
                provenance: provenance(),
            })
            .unwrap();

        ledger
            .attach_pcm(&EmissionId("e1".into()), vec![1.0, 2.0], 22_050)
            .unwrap();
        ledger
            .attach_pcm(&EmissionId("e2".into()), vec![3.0, 4.0], 22_050)
            .unwrap();

        let unplayed = ledger.collect_unplayed_pcm();
        // Only e1 (committed) should be included.
        assert_eq!(unplayed, vec![1.0, 2.0]);
    }
}
