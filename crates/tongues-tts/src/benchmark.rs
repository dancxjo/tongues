//! Portable benchmark corpus and measurement contracts for speech synthesis.
//!
//! This module defines the canonical benchmark fixtures and measurement types
//! used to evaluate speech pipelines across all capability tiers.  All types
//! are hardware-neutral and provider-neutral: they record *what* was measured
//! and on *which* device, but they do not encode preferred machines, host
//! roles, or fallback policy.
//!
//! ## Corpus design
//!
//! The benchmark corpus contains four fixture categories:
//!
//! - **Acknowledgment** — a very short conversational token used to measure
//!   cold and warm first-audio latency at minimum text length.
//! - **Ordinary sentence** — a typical declarative sentence covering
//!   representative phoneme and prosody variety.
//! - **Punctuation-heavy sentence** — a sentence with commas, parentheses, and
//!   other pause/prosody markers used to stress-test text handling.
//! - **Revision fixture** — a sentence pair (original + extended) designed to
//!   measure suffix-regeneration latency and crossfade cost on Tier A pipelines.
//!
//! ## Measurement contract
//!
//! [`PortableBenchmarkMeasurements`] records the six portable timings required
//! for every benchmark result plus optional revision-specific measurements for
//! Tier A pipelines.  Downstream systems (e.g. Netherwick) can consume these
//! artifacts to inform placement and fallback order without knowing Tongues
//! internals.
//!
//! ## Separation of properties
//!
//! Incremental output, low first-audio latency, and low total real-time factor
//! are *separate* properties.  A pipeline may satisfy one without satisfying
//! all three.  The fields in [`PortableBenchmarkMeasurements`] capture them
//! independently so callers can reason about each dimension on its own.

use serde::{Deserialize, Serialize};

use crate::CapabilityTier;

// ---------------------------------------------------------------------------
// Corpus fixtures
// ---------------------------------------------------------------------------

/// A single benchmark fixture sentence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkFixture {
    /// Machine-readable fixture identifier (stable across releases).
    pub id: String,
    /// Human-readable fixture category.
    pub category: BenchmarkFixtureCategory,
    /// The text to synthesize.
    pub text: String,
    /// Expected approximate duration of the synthesized audio in milliseconds.
    /// Used as a sanity check only; exact duration depends on the pipeline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_duration_ms: Option<u32>,
    /// For revision fixtures: the extended text that replaces the original
    /// after the first synthesis.  The suffix is the portion appended beyond
    /// the shared prefix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision_text: Option<String>,
}

/// Category of a benchmark fixture sentence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkFixtureCategory {
    /// Ultra-short acknowledgment token.
    Acknowledgment,
    /// Typical declarative sentence.
    OrdinarySentence,
    /// Sentence with commas, parentheses, and other prosody markers.
    PunctuationHeavy,
    /// Sentence pair for measuring suffix-regeneration on Tier A pipelines.
    RevisionFixture,
}

/// The canonical portable benchmark corpus.
///
/// Returns the four required fixture categories.  Every pipeline benchmark
/// must cover all fixtures; Tier A benchmarks must also exercise
/// [`BenchmarkFixtureCategory::RevisionFixture`].
pub fn canonical_benchmark_corpus() -> Vec<BenchmarkFixture> {
    vec![
        BenchmarkFixture {
            id: "ack-ok".into(),
            category: BenchmarkFixtureCategory::Acknowledgment,
            text: "Okay.".into(),
            expected_duration_ms: Some(350),
            revision_text: None,
        },
        BenchmarkFixture {
            id: "sentence-weather".into(),
            category: BenchmarkFixtureCategory::OrdinarySentence,
            text: "The forecast calls for partly cloudy skies and mild temperatures throughout the afternoon.".into(),
            expected_duration_ms: Some(3_500),
            revision_text: None,
        },
        BenchmarkFixture {
            id: "punct-parenthetical".into(),
            category: BenchmarkFixtureCategory::PunctuationHeavy,
            text: "The result — unexpected, at first glance — follows directly from the earlier proof (see section three, part two).".into(),
            expected_duration_ms: Some(4_200),
            revision_text: None,
        },
        BenchmarkFixture {
            id: "revision-append".into(),
            category: BenchmarkFixtureCategory::RevisionFixture,
            text: "She opened the door".into(),
            revision_text: Some("She opened the door and stepped into the hallway.".into()),
            expected_duration_ms: Some(800),
        },
    ]
}

// ---------------------------------------------------------------------------
// Hardware and execution context
// ---------------------------------------------------------------------------

/// Hardware and execution context recorded with every benchmark result.
///
/// No preferred machine, hostname, or host role is encoded here.  Downstream
/// systems supply placement policy; Tongues records only the device class and
/// thread budget observed during the run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkExecutionContext {
    /// Human-readable device label (e.g. `"cpu"`, `"cuda:0"`, `"mps"`).
    pub device: String,
    /// Number of CPU threads available to the inference engine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_threads: Option<usize>,
    /// Free-form note describing the hardware target (e.g. CPU model).
    /// Must not encode hostnames or deployment roles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware_note: Option<String>,
}

// ---------------------------------------------------------------------------
// Measurement contract
// ---------------------------------------------------------------------------

/// Portable benchmark measurements for a single pipeline run.
///
/// All timing values are in milliseconds; memory values are in mebibytes.
/// Fields that do not apply to a pipeline variant are `None`.
///
/// ## Required fields
///
/// Every pipeline benchmark must populate:
/// - [`cold_load_ms`](Self::cold_load_ms)
/// - [`warm_first_audio_ms`](Self::warm_first_audio_ms)
/// - [`real_time_factor`](Self::real_time_factor)
/// - [`peak_resident_mib`](Self::peak_resident_mib)
/// - [`cancellation_latency_ms`](Self::cancellation_latency_ms)
///
/// ## Tier A additional fields
///
/// Tier A (revision-capable) pipelines must also populate:
/// - [`suffix_regeneration_ms`](Self::suffix_regeneration_ms)
/// - [`crossfade_cost_ms`](Self::crossfade_cost_ms)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortableBenchmarkMeasurements {
    /// Pipeline backend identifier.
    pub backend: String,
    /// Pipeline model identifier.
    pub model: String,
    /// Capability tier under which this measurement was taken.
    pub capability_tier: CapabilityTier,
    /// Benchmark fixture this measurement covers.
    pub fixture_id: String,
    /// Hardware and execution context.
    pub context: BenchmarkExecutionContext,

    // --- Required timings ---
    /// Wall-clock time from process start (or model-load request) to the model
    /// being ready for first synthesis, in milliseconds.  Measured on a cold
    /// process with no prior warm-up.
    pub cold_load_ms: f64,
    /// Wall-clock time from the start of a synthesis call (on a warm, already-
    /// loaded model) to the emission of the first audio chunk, in milliseconds.
    pub warm_first_audio_ms: f64,
    /// Total synthesis wall-clock time divided by synthesized audio duration
    /// (unitless).  Values below 1.0 indicate faster-than-real-time synthesis.
    pub real_time_factor: f64,
    /// Peak resident set size observed during synthesis, in mebibytes.
    pub peak_resident_mib: f64,
    /// Time from a cancellation signal to the last audio chunk or silence, in
    /// milliseconds.  Pipelines that cannot be cancelled should record the full
    /// synthesis duration as a conservative upper bound.
    pub cancellation_latency_ms: f64,

    // --- Tier A revision fields (None for Tier B / Tier C) ---
    /// Time to regenerate only the changed suffix (i.e. the portion that
    /// differs from the original utterance), in milliseconds.
    /// Required for Tier A revision-capable pipelines; `None` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix_regeneration_ms: Option<f64>,
    /// Cost of crossfading or withholding the tail of the original audio to
    /// splice in the regenerated suffix, in milliseconds.
    /// Required for Tier A revision-capable pipelines; `None` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crossfade_cost_ms: Option<f64>,

    // --- Reference-conditioning overhead (Tier C) ---
    /// Time to process a speaker or style reference audio prior to synthesis,
    /// in milliseconds.  Reported separately from [`warm_first_audio_ms`](Self::warm_first_audio_ms)
    /// so callers can reason about steady-state synthesis cost independently.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_processing_ms: Option<f64>,
}

impl PortableBenchmarkMeasurements {
    /// Validate that all required fields are present for the declared tier and
    /// that the revision fields are populated for revision-capable Tier A runs.
    ///
    /// Returns a list of field names that are missing or inconsistent.
    pub fn validate(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if !self.cold_load_ms.is_finite() || self.cold_load_ms < 0.0 {
            missing.push("cold_load_ms");
        }
        if !self.warm_first_audio_ms.is_finite() || self.warm_first_audio_ms < 0.0 {
            missing.push("warm_first_audio_ms");
        }
        if !self.real_time_factor.is_finite() || self.real_time_factor < 0.0 {
            missing.push("real_time_factor");
        }
        if !self.peak_resident_mib.is_finite() || self.peak_resident_mib < 0.0 {
            missing.push("peak_resident_mib");
        }
        if !self.cancellation_latency_ms.is_finite() || self.cancellation_latency_ms < 0.0 {
            missing.push("cancellation_latency_ms");
        }
        if self.capability_tier.is_revision_tier() {
            if self.suffix_regeneration_ms.is_none() {
                missing.push("suffix_regeneration_ms");
            }
            if self.crossfade_cost_ms.is_none() {
                missing.push("crossfade_cost_ms");
            }
        }
        missing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_corpus_covers_all_required_categories() {
        let corpus = canonical_benchmark_corpus();
        let has = |cat: BenchmarkFixtureCategory| corpus.iter().any(|f| f.category == cat);
        assert!(has(BenchmarkFixtureCategory::Acknowledgment));
        assert!(has(BenchmarkFixtureCategory::OrdinarySentence));
        assert!(has(BenchmarkFixtureCategory::PunctuationHeavy));
        assert!(has(BenchmarkFixtureCategory::RevisionFixture));
    }

    #[test]
    fn revision_fixture_carries_both_original_and_extended_text() {
        let corpus = canonical_benchmark_corpus();
        let revision = corpus
            .iter()
            .find(|f| f.category == BenchmarkFixtureCategory::RevisionFixture)
            .expect("revision fixture must be present");
        assert!(!revision.text.is_empty());
        let extended = revision.revision_text.as_deref().expect("revision_text");
        assert!(
            extended.starts_with(&revision.text),
            "revision text must extend the original prefix"
        );
    }

    #[test]
    fn fixture_ids_are_unique() {
        let corpus = canonical_benchmark_corpus();
        let mut ids: Vec<&str> = corpus.iter().map(|f| f.id.as_str()).collect();
        let before = ids.len();
        ids.dedup();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), before, "fixture ids must be unique");
    }

    #[test]
    fn valid_tier_a_measurement_passes_validation() {
        let m = PortableBenchmarkMeasurements {
            backend: "fastpitch".into(),
            model: "fastpitch-ljspeech+hifigan-v2".into(),
            capability_tier: CapabilityTier::TierA,
            fixture_id: "revision-append".into(),
            context: BenchmarkExecutionContext {
                device: "cpu".into(),
                cpu_threads: Some(4),
                hardware_note: None,
            },
            cold_load_ms: 420.0,
            warm_first_audio_ms: 18.0,
            real_time_factor: 0.42,
            peak_resident_mib: 256.0,
            cancellation_latency_ms: 5.0,
            suffix_regeneration_ms: Some(22.0),
            crossfade_cost_ms: Some(3.0),
            reference_processing_ms: None,
        };
        assert!(m.validate().is_empty(), "should have no validation errors");
    }

    #[test]
    fn tier_a_measurement_without_revision_fields_fails_validation() {
        let m = PortableBenchmarkMeasurements {
            backend: "fastpitch".into(),
            model: "fastpitch-ljspeech+hifigan-v2".into(),
            capability_tier: CapabilityTier::TierA,
            fixture_id: "revision-append".into(),
            context: BenchmarkExecutionContext {
                device: "cpu".into(),
                cpu_threads: None,
                hardware_note: None,
            },
            cold_load_ms: 420.0,
            warm_first_audio_ms: 18.0,
            real_time_factor: 0.42,
            peak_resident_mib: 256.0,
            cancellation_latency_ms: 5.0,
            suffix_regeneration_ms: None,
            crossfade_cost_ms: None,
            reference_processing_ms: None,
        };
        let errors = m.validate();
        assert!(
            errors.contains(&"suffix_regeneration_ms"),
            "must flag missing suffix_regeneration_ms"
        );
        assert!(
            errors.contains(&"crossfade_cost_ms"),
            "must flag missing crossfade_cost_ms"
        );
    }

    #[test]
    fn tier_b_measurement_does_not_require_revision_fields() {
        let m = PortableBenchmarkMeasurements {
            backend: "vits".into(),
            model: "vits-vctk".into(),
            capability_tier: CapabilityTier::TierB,
            fixture_id: "sentence-weather".into(),
            context: BenchmarkExecutionContext {
                device: "cpu".into(),
                cpu_threads: Some(2),
                hardware_note: Some("reference CPU target".into()),
            },
            cold_load_ms: 200.0,
            warm_first_audio_ms: 150.0,
            real_time_factor: 0.9,
            peak_resident_mib: 180.0,
            cancellation_latency_ms: 200.0,
            suffix_regeneration_ms: None,
            crossfade_cost_ms: None,
            reference_processing_ms: None,
        };
        assert!(m.validate().is_empty(), "Tier B does not require revision fields");
    }

    #[test]
    fn measurements_serialize_to_stable_json_fields() {
        let m = PortableBenchmarkMeasurements {
            backend: "fastpitch".into(),
            model: "fastpitch-ljspeech+hifigan-v2".into(),
            capability_tier: CapabilityTier::TierA,
            fixture_id: "ack-ok".into(),
            context: BenchmarkExecutionContext {
                device: "cpu".into(),
                cpu_threads: Some(8),
                hardware_note: None,
            },
            cold_load_ms: 300.0,
            warm_first_audio_ms: 15.0,
            real_time_factor: 0.3,
            peak_resident_mib: 512.0,
            cancellation_latency_ms: 4.0,
            suffix_regeneration_ms: Some(20.0),
            crossfade_cost_ms: Some(2.0),
            reference_processing_ms: None,
        };
        let v = serde_json::to_value(&m).expect("serialize");
        assert_eq!(v["backend"], "fastpitch");
        assert_eq!(v["capability_tier"], "tier_a");
        assert_eq!(v["cold_load_ms"], 300.0);
        assert_eq!(v["warm_first_audio_ms"], 15.0);
        assert_eq!(v["real_time_factor"], 0.3);
        assert_eq!(v["peak_resident_mib"], 512.0);
        assert_eq!(v["cancellation_latency_ms"], 4.0);
        assert_eq!(v["suffix_regeneration_ms"], 20.0);
        assert_eq!(v["crossfade_cost_ms"], 2.0);
        assert!(v.get("reference_processing_ms").is_none(), "None fields omitted");
    }
}
