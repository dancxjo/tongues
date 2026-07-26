//! Backend-neutral revision waveform assembler.
//!
//! [`RevisionWaveformAssembler`] generalizes the XTTS cumulative-overlap
//! semantics into a provider-independent utility. Any renderer that can
//! produce a new version of a waveform segment (because the plan changed)
//! can use this type to splice the revision into the already-delivered stream
//! without dropping or duplicating played samples.
//!
//! # Design
//!
//! The assembler maintains:
//! - A count of samples that have been reported as played (immutable).
//! - A pending buffer of samples that have been synthesized but not yet played.
//!
//! When a revision arrives:
//! 1. The assembler discards the pending (unplayed) samples.
//! 2. It crossfades the tail of the already-committed output with the head of
//!    the new waveform so that the transition is smooth.
//! 3. It returns only the non-duplicated portion of the new waveform.
//!
//! The crossfade is applied as a linear fade: the outgoing signal fades from
//! 1.0 to 0.0 while the incoming signal fades from 0.0 to 1.0 over
//! `crossfade_samples` samples. This matches the existing XTTS behavior.
//!
//! # Fixture determinism
//!
//! Given identical inputs the assembler always produces identical outputs, so
//! crossfade behavior can be verified in unit tests without an audio device.

use anyhow::{Result, ensure};

/// Backend-neutral assembler for streaming revision of TTS waveforms.
///
/// See the [module documentation](self) for a full description of the
/// invariants this type enforces.
#[derive(Debug, Clone)]
pub struct RevisionWaveformAssembler {
    /// Samples reported as played. The assembler never allows this to shrink.
    played_count: usize,
    /// Samples to overlap/crossfade at each revision boundary.
    crossfade_samples: usize,
    /// All samples delivered so far (played + pending).
    buffer: Vec<f32>,
}

impl RevisionWaveformAssembler {
    /// Create a new assembler.
    ///
    /// `crossfade_samples` must be greater than zero.
    pub fn new(crossfade_samples: usize) -> Result<Self> {
        ensure!(
            crossfade_samples > 0,
            "crossfade_samples must be positive (got 0)"
        );
        Ok(Self {
            played_count: 0,
            crossfade_samples,
            buffer: Vec::new(),
        })
    }

    /// Append newly synthesized samples to the pending (unplayed) buffer.
    ///
    /// Returns an error if any sample is non-finite.
    pub fn push(&mut self, samples: &[f32]) -> Result<()> {
        ensure!(
            samples.iter().all(|s| s.is_finite()),
            "revision assembler received non-finite PCM samples"
        );
        self.buffer.extend_from_slice(samples);
        Ok(())
    }

    /// Report that `count` additional samples have been played.
    ///
    /// The played count must not advance beyond the total number of buffered
    /// samples, and must not shrink.
    pub fn mark_played(&mut self, count: usize) -> Result<()> {
        let new_played = self.played_count + count;
        ensure!(
            new_played <= self.buffer.len(),
            "cannot mark {count} samples played: only {} samples buffered, {} already played",
            self.buffer.len(),
            self.played_count,
        );
        self.played_count = new_played;
        Ok(())
    }

    /// Replace the unplayed suffix with a new waveform.
    ///
    /// The assembler:
    /// 1. Discards all samples after `played_count`.
    /// 2. Applies a linear crossfade over the last `crossfade_samples` of the
    ///    played region and the first `crossfade_samples` of `new_pcm`.
    /// 3. Returns the combined crossfade region plus the remainder of
    ///    `new_pcm`, which the caller should send to the playback queue.
    ///
    /// Returns an error if `new_pcm` contains non-finite samples.
    pub fn replace_suffix(&mut self, new_pcm: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            new_pcm.iter().all(|s| s.is_finite()),
            "revision assembler: replacement waveform contains non-finite samples"
        );

        // Truncate to the played prefix.
        self.buffer.truncate(self.played_count);

        let crossfade_len = self.crossfade_samples.min(self.played_count).min(new_pcm.len());

        // Build the output: crossfade region + remainder.
        let mut output = Vec::with_capacity(new_pcm.len());

        if crossfade_len == 0 {
            output.extend_from_slice(new_pcm);
        } else {
            let fade_start = self.played_count - crossfade_len;
            let outgoing = &self.buffer[fade_start..self.played_count];

            // Crossfade prefix.
            let denominator = if crossfade_len == 1 {
                1.0_f32
            } else {
                (crossfade_len - 1) as f32
            };
            for index in 0..crossfade_len {
                let incoming_weight = index as f32 / denominator;
                let outgoing_weight = 1.0 - incoming_weight;
                let blended = outgoing[index] * outgoing_weight + new_pcm[index] * incoming_weight;
                output.push(blended);
            }

            // Remainder without crossfade.
            output.extend_from_slice(&new_pcm[crossfade_len..]);
        }

        // Append the new (post-played) buffer.
        self.buffer.extend_from_slice(&output);

        Ok(output)
    }

    /// Drain all pending (unplayed) samples and return them.
    ///
    /// This is used to retrieve the final waveform segment when synthesis is
    /// complete or cancelled. After draining, `played_count` equals `buffer.len()`.
    pub fn drain_pending(&mut self) -> Vec<f32> {
        let pending = self.buffer[self.played_count..].to_vec();
        // Keep the buffer intact; only note that these samples are now queued.
        pending
    }

    /// Total number of samples that have been reported as played.
    pub fn played_count(&self) -> usize {
        self.played_count
    }

    /// Total number of buffered samples (played + pending).
    pub fn total_samples(&self) -> usize {
        self.buffer.len()
    }

    /// Number of pending (unplayed) samples.
    pub fn pending_count(&self) -> usize {
        self.buffer.len() - self.played_count
    }
}

/// Apply a linear crossfade from `outgoing` to `incoming` over `len` samples.
///
/// Returns an error if the slices differ in length or contain non-finite
/// values. This function is exported for use in deterministic fixture tests.
pub fn crossfade_linear(outgoing: &[f32], incoming: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        outgoing.len() == incoming.len(),
        "crossfade_linear: outgoing ({}) and incoming ({}) slices must have equal length",
        outgoing.len(),
        incoming.len()
    );
    let len = outgoing.len();
    if len == 0 {
        return Ok(Vec::new());
    }
    let denominator = if len == 1 { 1.0_f32 } else { (len - 1) as f32 };
    let output = (0..len)
        .map(|index| {
            let incoming_weight = index as f32 / denominator;
            let outgoing_weight = 1.0 - incoming_weight;
            outgoing[index] * outgoing_weight + incoming[index] * incoming_weight
        })
        .collect();
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_mark_played() {
        let mut assembler = RevisionWaveformAssembler::new(4).unwrap();
        assembler.push(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assembler.mark_played(3).unwrap();
        assert_eq!(assembler.played_count(), 3);
        assert_eq!(assembler.pending_count(), 2);
        assert_eq!(assembler.total_samples(), 5);
    }

    #[test]
    fn replace_suffix_no_overlap_region() {
        let mut assembler = RevisionWaveformAssembler::new(4).unwrap();
        // Nothing played yet.
        assembler.push(&[1.0, 2.0, 3.0]).unwrap();
        let output = assembler.replace_suffix(&[10.0, 20.0, 30.0]).unwrap();
        assert_eq!(output, vec![10.0, 20.0, 30.0]);
        assert_eq!(assembler.played_count(), 0);
    }

    #[test]
    fn replace_suffix_crossfades_boundary() {
        let mut assembler = RevisionWaveformAssembler::new(4).unwrap();
        // Synthesize and play 4 samples.
        assembler.push(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assembler.mark_played(4).unwrap();

        // Replace the suffix (nothing pending) with a new waveform.
        let new_pcm = [0.0_f32, 10.0, 20.0, 30.0, 40.0, 50.0];
        let output = assembler.replace_suffix(&new_pcm).unwrap();

        // First 4 samples are the crossfade region.
        assert_eq!(output.len(), new_pcm.len());
        // At index 0: incoming_weight=0, result is fully from outgoing buffer.
        // outgoing = buffer[0..4] = [1,2,3,4]; new_pcm[0] = 0.0.
        assert!((output[0] - 1.0).abs() < 1e-6, "expected 1.0 at index 0, got {}", output[0]);
        // Last index of crossfade: fully incoming.
        assert!((output[3] - 30.0).abs() < 1e-6, "expected 30.0 at index 3, got {}", output[3]);
        // Remainder is unmodified.
        assert!((output[4] - 40.0).abs() < 1e-6);
        assert!((output[5] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn replace_suffix_is_deterministic() {
        let mut a1 = RevisionWaveformAssembler::new(4).unwrap();
        let mut a2 = RevisionWaveformAssembler::new(4).unwrap();
        let pcm = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        a1.push(&pcm).unwrap();
        a2.push(&pcm).unwrap();
        a1.mark_played(3).unwrap();
        a2.mark_played(3).unwrap();
        let new_pcm = [10.0, 20.0, 30.0, 40.0];
        let out1 = a1.replace_suffix(&new_pcm).unwrap();
        let out2 = a2.replace_suffix(&new_pcm).unwrap();
        assert_eq!(out1, out2);
    }

    #[test]
    fn mark_played_beyond_buffer_is_error() {
        let mut assembler = RevisionWaveformAssembler::new(4).unwrap();
        assembler.push(&[1.0, 2.0]).unwrap();
        assert!(assembler.mark_played(5).is_err());
    }

    #[test]
    fn non_finite_push_is_error() {
        let mut assembler = RevisionWaveformAssembler::new(4).unwrap();
        assert!(assembler.push(&[f32::INFINITY]).is_err());
        assert!(assembler.push(&[f32::NAN]).is_err());
    }

    #[test]
    fn crossfade_linear_single_sample() {
        let out = crossfade_linear(&[1.0], &[2.0]).unwrap();
        assert_eq!(out.len(), 1);
        // With len=1 denominator=1, weight=0/1=0 so outgoing is used.
        assert!((out[0] - 1.0).abs() < 1e-6, "got {}", out[0]);
    }

    #[test]
    fn crossfade_linear_mismatched_lengths() {
        assert!(crossfade_linear(&[1.0, 2.0], &[3.0]).is_err());
    }

    #[test]
    fn crossfade_linear_empty() {
        let out = crossfade_linear(&[], &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn crossfade_linear_two_samples() {
        // index 0: weight=0 → fully outgoing; index 1: weight=1 → fully incoming
        let out = crossfade_linear(&[10.0, 10.0], &[20.0, 20.0]).unwrap();
        assert!((out[0] - 10.0).abs() < 1e-6, "index 0: {}", out[0]);
        assert!((out[1] - 20.0).abs() < 1e-6, "index 1: {}", out[1]);
    }

    #[test]
    fn zero_crossfade_samples_is_error() {
        assert!(RevisionWaveformAssembler::new(0).is_err());
    }

    #[test]
    fn drain_pending_returns_unplayed() {
        let mut assembler = RevisionWaveformAssembler::new(4).unwrap();
        assembler.push(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assembler.mark_played(2).unwrap();
        let pending = assembler.drain_pending();
        assert_eq!(pending, vec![3.0, 4.0, 5.0]);
        assert_eq!(assembler.played_count(), 2);
    }
}
