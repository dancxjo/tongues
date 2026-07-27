//! Privacy-preserving evidence contracts for synthesized and perceived speech.
//!
//! A renderer submitting PCM is not evidence that a device played it, and
//! device playback is not evidence that sound reached the room. This module
//! keeps those boundaries explicit while correlating every stage with stable
//! identifiers.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

const FEATURE_BUCKETS: usize = 8;

macro_rules! string_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);
    };
}

string_id!(SpeechPlanId);
string_id!(SynthesisRequestId);
string_id!(TargetPcmId);
string_id!(PlaybackSessionId);
string_id!(MicrophoneObservationId);

/// Correlation chain from generated text through acoustic observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechCorrelation {
    pub source_text_id: String,
    pub speech_plan_id: SpeechPlanId,
    pub synthesis_request_id: SynthesisRequestId,
    pub target_pcm_id: TargetPcmId,
    pub playback_session_id: PlaybackSessionId,
}

/// Live-only target PCM reference for device echo cancellers. Implementations
/// must consume and discard these frames; the feature verifier never stores
/// them.
#[derive(Debug, Clone, PartialEq)]
pub struct EchoReferenceFrame {
    pub playback_session_id: PlaybackSessionId,
    pub sequence: u64,
    pub start_frame: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl EchoReferenceFrame {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.sample_rate_hz > 0,
            "echo reference sample rate is zero"
        );
        anyhow::ensure!(self.channels > 0, "echo reference channel count is zero");
        anyhow::ensure!(
            !self.samples.is_empty()
                && self
                    .samples
                    .len()
                    .is_multiple_of(usize::from(self.channels)),
            "echo reference contains no complete audio frames"
        );
        anyhow::ensure!(
            self.samples.iter().all(|sample| sample.is_finite()),
            "echo reference contains a non-finite sample"
        );
        Ok(())
    }
}

/// Lifecycle stages intentionally distinguish intent, submission, device
/// activity, and perception.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechLifecycleStage {
    Planned,
    Requested,
    Started,
    Interrupted,
    Resumed,
    Aborted,
    Completed,
    Perceived,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechLifecycleEvent {
    pub sequence: u64,
    pub occurred_at_ms: u64,
    pub stage: SpeechLifecycleStage,
    pub correlation: SpeechCorrelation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observation_id: Option<MicrophoneObservationId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Bounded features that can be retained without retaining reconstructable raw
/// microphone or target audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioEvidenceFeatures {
    pub duration_ms: u64,
    pub rms: f32,
    pub peak: f32,
    pub zero_crossing_rate: f32,
    pub clipped_fraction: f32,
    /// Coarse mean-absolute-energy shape. Its length is fixed and bounded.
    pub signature: Vec<f32>,
}

impl AudioEvidenceFeatures {
    pub fn from_mono_pcm(samples: &[f32], sample_rate_hz: u32) -> anyhow::Result<Self> {
        anyhow::ensure!(sample_rate_hz > 0, "feature sample rate must be positive");
        anyhow::ensure!(
            !samples.is_empty(),
            "cannot extract features from empty PCM"
        );
        anyhow::ensure!(
            samples.iter().all(|sample| sample.is_finite()),
            "feature PCM contains a non-finite sample"
        );
        let duration_ms = (samples.len() as u64)
            .saturating_mul(1_000)
            .div_ceil(u64::from(sample_rate_hz));
        let square_sum = samples
            .iter()
            .map(|sample| f64::from(*sample) * f64::from(*sample))
            .sum::<f64>();
        let rms = (square_sum / samples.len() as f64).sqrt() as f32;
        let peak = samples
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0, f32::max);
        let crossings = samples
            .windows(2)
            .filter(|pair| pair[0].is_sign_positive() != pair[1].is_sign_positive())
            .count();
        let clipped = samples
            .iter()
            .filter(|sample| sample.abs() >= 0.999)
            .count();
        let mut signature = Vec::with_capacity(FEATURE_BUCKETS);
        for bucket in 0..FEATURE_BUCKETS {
            let start = bucket * samples.len() / FEATURE_BUCKETS;
            let end = ((bucket + 1) * samples.len() / FEATURE_BUCKETS).max(start + 1);
            let slice = &samples[start..end.min(samples.len())];
            signature
                .push(slice.iter().map(|sample| sample.abs()).sum::<f32>() / slice.len() as f32);
        }
        Ok(Self {
            duration_ms,
            rms,
            peak,
            zero_crossing_rate: crossings as f32 / samples.len().saturating_sub(1).max(1) as f32,
            clipped_fraction: clipped as f32 / samples.len() as f32,
            signature,
        })
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.duration_ms > 0, "audio evidence duration is zero");
        anyhow::ensure!(
            self.signature.len() == FEATURE_BUCKETS,
            "audio evidence signature must contain {FEATURE_BUCKETS} buckets"
        );
        anyhow::ensure!(
            self.signature
                .iter()
                .chain([
                    &self.rms,
                    &self.peak,
                    &self.zero_crossing_rate,
                    &self.clipped_fraction,
                ])
                .all(|value| value.is_finite() && *value >= 0.0),
            "audio evidence contains invalid features"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceivedAudioEvidence {
    pub observation_id: MicrophoneObservationId,
    pub playback_session_id: PlaybackSessionId,
    pub occurred_at_ms: u64,
    pub device_reported_playing: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub features: Option<AudioEvidenceFeatures>,
    #[serde(default)]
    pub external_speech_probability: f32,
    #[serde(default)]
    pub dropout_detected: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputClassification {
    LikelySelfSpeech,
    LikelyExternalSpeech,
    Overlap,
    Uncertain,
    PartialOutput,
    MissingOutput,
    PlaybackFailure,
    Clipped,
    Dropout,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputVerification {
    pub correlation: SpeechCorrelation,
    pub observation_id: MicrophoneObservationId,
    pub playback_session_id: PlaybackSessionId,
    pub classification: OutputClassification,
    pub target_similarity: f32,
    pub observed_coverage: f32,
    pub raw_audio_retained: bool,
    pub rationale: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BargeInAction {
    ContinuePlayback,
    InterruptPlayback,
    AwaitMoreEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EchoAwareDecision {
    pub action: BargeInAction,
    pub classification: OutputClassification,
    pub reason: String,
}

/// Feature-only output verifier. Target features are bounded by playback ID;
/// no PCM is stored.
#[derive(Debug, Default)]
pub struct OutputVerifier {
    targets: BTreeMap<PlaybackSessionId, TargetOutputEvidence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetOutputEvidence {
    pub correlation: SpeechCorrelation,
    pub features: AudioEvidenceFeatures,
}

impl OutputVerifier {
    pub fn register_target(&mut self, target: TargetOutputEvidence) -> anyhow::Result<()> {
        target.features.validate()?;
        let playback_session_id = target.correlation.playback_session_id.clone();
        anyhow::ensure!(
            self.targets
                .insert(playback_session_id.clone(), target)
                .is_none(),
            "duplicate playback target `{}`",
            playback_session_id.0
        );
        Ok(())
    }

    pub fn verify(&self, evidence: &PerceivedAudioEvidence) -> anyhow::Result<OutputVerification> {
        anyhow::ensure!(
            evidence.external_speech_probability.is_finite()
                && (0.0..=1.0).contains(&evidence.external_speech_probability),
            "external speech probability must be between zero and one"
        );
        let target = self
            .targets
            .get(&evidence.playback_session_id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown playback session `{}`",
                    evidence.playback_session_id.0
                )
            })?;
        let (similarity, coverage) = evidence.features.as_ref().map_or((0.0, 0.0), |observed| {
            (
                cosine_similarity(&target.features.signature, &observed.signature),
                observed.duration_ms as f32 / target.features.duration_ms as f32,
            )
        });
        let (classification, rationale) = classify(evidence, similarity, coverage);
        Ok(OutputVerification {
            correlation: target.correlation.clone(),
            observation_id: evidence.observation_id.clone(),
            playback_session_id: evidence.playback_session_id.clone(),
            classification,
            target_similarity: similarity,
            observed_coverage: coverage,
            raw_audio_retained: false,
            rationale: rationale.into(),
        })
    }

    pub fn release(&mut self, playback_session_id: &PlaybackSessionId) -> bool {
        self.targets.remove(playback_session_id).is_some()
    }

    pub fn retained_target_count(&self) -> usize {
        self.targets.len()
    }
}

fn classify(
    evidence: &PerceivedAudioEvidence,
    similarity: f32,
    coverage: f32,
) -> (OutputClassification, &'static str) {
    let Some(features) = &evidence.features else {
        return if evidence.device_reported_playing {
            (
                OutputClassification::MissingOutput,
                "device reported playback but no acoustic evidence was observed",
            )
        } else {
            (
                OutputClassification::PlaybackFailure,
                "device did not report playback and no acoustic evidence was observed",
            )
        };
    };
    if !evidence.device_reported_playing {
        return (
            OutputClassification::PlaybackFailure,
            "microphone evidence exists but the device did not report playback",
        );
    }
    if evidence.dropout_detected {
        return (
            OutputClassification::Dropout,
            "an explicit output discontinuity was observed",
        );
    }
    if features.clipped_fraction >= 0.02 {
        return (
            OutputClassification::Clipped,
            "at least two percent of observed samples were clipped",
        );
    }
    if similarity >= 0.8 && evidence.external_speech_probability >= 0.6 {
        return (
            OutputClassification::Overlap,
            "target-like output and external speech are both present",
        );
    }
    if similarity >= 0.8 && coverage < 0.8 {
        return (
            OutputClassification::PartialOutput,
            "target-like output covers only part of the planned duration",
        );
    }
    if similarity >= 0.8 {
        return (
            OutputClassification::LikelySelfSpeech,
            "observed feature shape closely matches the target output",
        );
    }
    if similarity < 0.4 && evidence.external_speech_probability >= 0.6 {
        return (
            OutputClassification::LikelyExternalSpeech,
            "observed speech does not resemble the target output",
        );
    }
    (
        OutputClassification::Uncertain,
        "available feature evidence does not support a confident attribution",
    )
}

pub fn echo_aware_barge_in(
    verification: &OutputVerification,
    playback_active: bool,
) -> EchoAwareDecision {
    let (action, reason) = if !playback_active {
        (
            BargeInAction::ContinuePlayback,
            "no active playback can be interrupted",
        )
    } else {
        match verification.classification {
            OutputClassification::LikelyExternalSpeech | OutputClassification::Overlap => (
                BargeInAction::InterruptPlayback,
                "external speech evidence overlaps active playback",
            ),
            OutputClassification::LikelySelfSpeech => (
                BargeInAction::ContinuePlayback,
                "microphone activity is attributable to target playback",
            ),
            _ => (
                BargeInAction::AwaitMoreEvidence,
                "evidence is insufficient for an echo-aware interruption",
            ),
        }
    };
    EchoAwareDecision {
        action,
        classification: verification.classification,
        reason: reason.into(),
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != FEATURE_BUCKETS || right.len() != FEATURE_BUCKETS {
        return 0.0;
    }
    let dot = left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        (dot / (left_norm * right_norm)).clamp(0.0, 1.0)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LifecycleError {
    #[error("speech lifecycle sequence must increase: current {current}, received {received}")]
    OutOfOrder { current: u64, received: u64 },
    #[error("invalid speech lifecycle transition from {from:?} to {to:?}")]
    InvalidTransition {
        from: SpeechLifecycleStage,
        to: SpeechLifecycleStage,
    },
    #[error("speech lifecycle correlation changed within one playback session")]
    CorrelationChanged,
}

/// Append-only lifecycle stream suitable for downstream agents.
#[derive(Debug, Default)]
pub struct SpeechLifecycle {
    events: Vec<SpeechLifecycleEvent>,
}

impl SpeechLifecycle {
    pub fn append(&mut self, event: SpeechLifecycleEvent) -> Result<(), LifecycleError> {
        if let Some(previous) = self.events.last() {
            if event.sequence <= previous.sequence {
                return Err(LifecycleError::OutOfOrder {
                    current: previous.sequence,
                    received: event.sequence,
                });
            }
            if event.correlation != previous.correlation {
                return Err(LifecycleError::CorrelationChanged);
            }
            if !valid_transition(previous.stage, event.stage) {
                return Err(LifecycleError::InvalidTransition {
                    from: previous.stage,
                    to: event.stage,
                });
            }
        } else if event.stage != SpeechLifecycleStage::Planned {
            return Err(LifecycleError::InvalidTransition {
                from: SpeechLifecycleStage::Planned,
                to: event.stage,
            });
        }
        self.events.push(event);
        Ok(())
    }

    pub fn events(&self) -> &[SpeechLifecycleEvent] {
        &self.events
    }

    pub fn is_terminal(&self) -> bool {
        self.events.last().is_some_and(|event| {
            matches!(
                event.stage,
                SpeechLifecycleStage::Aborted | SpeechLifecycleStage::Completed
            )
        })
    }
}

fn valid_transition(from: SpeechLifecycleStage, to: SpeechLifecycleStage) -> bool {
    use SpeechLifecycleStage as Stage;
    matches!(
        (from, to),
        (Stage::Planned, Stage::Requested)
            | (Stage::Requested, Stage::Started | Stage::Aborted)
            | (
                Stage::Started,
                Stage::Perceived | Stage::Interrupted | Stage::Completed | Stage::Aborted
            )
            | (
                Stage::Perceived,
                Stage::Perceived | Stage::Interrupted | Stage::Completed | Stage::Aborted
            )
            | (
                Stage::Interrupted,
                Stage::Resumed | Stage::Aborted | Stage::Completed
            )
            | (
                Stage::Resumed,
                Stage::Perceived | Stage::Interrupted | Stage::Completed | Stage::Aborted
            )
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct FixtureSuite {
        cases: Vec<FixtureCase>,
    }

    #[derive(Deserialize)]
    struct FixtureCase {
        name: String,
        target: AudioEvidenceFeatures,
        evidence: PerceivedAudioEvidence,
        expected: OutputClassification,
    }

    fn correlation() -> SpeechCorrelation {
        SpeechCorrelation {
            source_text_id: "text:1".into(),
            speech_plan_id: SpeechPlanId("plan:1".into()),
            synthesis_request_id: SynthesisRequestId("request:1".into()),
            target_pcm_id: TargetPcmId("pcm:1".into()),
            playback_session_id: PlaybackSessionId("playback:1".into()),
        }
    }

    #[test]
    fn feature_extraction_is_bounded_and_keeps_no_pcm() {
        let samples = (0..1_600)
            .map(|index| ((index as f32 / 10.0).sin() * 0.2).clamp(-1.0, 1.0))
            .collect::<Vec<_>>();
        let features = AudioEvidenceFeatures::from_mono_pcm(&samples, 16_000).unwrap();
        assert_eq!(features.signature.len(), FEATURE_BUCKETS);
        assert_eq!(features.duration_ms, 100);
        assert!(serde_json::to_string(&features).unwrap().len() < 512);
    }

    #[test]
    fn deterministic_output_fixtures_cover_normal_interruption_echo_missing_and_partial() {
        let suite: FixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/output-verification/scenarios_v1.json"
        ))
        .unwrap();
        assert_eq!(suite.cases.len(), 5);
        for case in suite.cases {
            let mut verifier = OutputVerifier::default();
            verifier
                .register_target(TargetOutputEvidence {
                    correlation: SpeechCorrelation {
                        playback_session_id: case.evidence.playback_session_id.clone(),
                        ..correlation()
                    },
                    features: case.target,
                })
                .unwrap();
            let result = verifier.verify(&case.evidence).unwrap();
            assert_eq!(result.classification, case.expected, "{}", case.name);
            assert!(!result.raw_audio_retained);
        }
    }

    #[test]
    fn external_or_overlapping_speech_can_barge_in_but_echo_does_not() {
        let external = OutputVerification {
            correlation: correlation(),
            observation_id: MicrophoneObservationId("obs".into()),
            playback_session_id: PlaybackSessionId("playback".into()),
            classification: OutputClassification::LikelyExternalSpeech,
            target_similarity: 0.1,
            observed_coverage: 1.0,
            raw_audio_retained: false,
            rationale: String::new(),
        };
        assert_eq!(
            echo_aware_barge_in(&external, true).action,
            BargeInAction::InterruptPlayback
        );
        let self_speech = OutputVerification {
            classification: OutputClassification::LikelySelfSpeech,
            ..external
        };
        assert_eq!(
            echo_aware_barge_in(&self_speech, true).action,
            BargeInAction::ContinuePlayback
        );
    }

    #[test]
    fn lifecycle_is_correlated_append_only_and_distinguishes_perception() {
        let mut lifecycle = SpeechLifecycle::default();
        for (sequence, stage) in [
            SpeechLifecycleStage::Planned,
            SpeechLifecycleStage::Requested,
            SpeechLifecycleStage::Started,
            SpeechLifecycleStage::Perceived,
            SpeechLifecycleStage::Interrupted,
            SpeechLifecycleStage::Resumed,
            SpeechLifecycleStage::Completed,
        ]
        .into_iter()
        .enumerate()
        {
            lifecycle
                .append(SpeechLifecycleEvent {
                    sequence: sequence as u64,
                    occurred_at_ms: sequence as u64 * 10,
                    stage,
                    correlation: correlation(),
                    observation_id: (stage == SpeechLifecycleStage::Perceived)
                        .then(|| MicrophoneObservationId("obs:1".into())),
                    detail: None,
                })
                .unwrap();
        }
        assert!(lifecycle.is_terminal());
        assert_eq!(lifecycle.events().len(), 7);
        assert_eq!(
            lifecycle
                .append(SpeechLifecycleEvent {
                    sequence: 7,
                    occurred_at_ms: 70,
                    stage: SpeechLifecycleStage::Perceived,
                    correlation: correlation(),
                    observation_id: Some(MicrophoneObservationId("obs:late".into())),
                    detail: None,
                })
                .unwrap_err(),
            LifecycleError::InvalidTransition {
                from: SpeechLifecycleStage::Completed,
                to: SpeechLifecycleStage::Perceived
            }
        );
    }

    #[test]
    fn releasing_a_target_removes_feature_state() {
        let mut verifier = OutputVerifier::default();
        let features = AudioEvidenceFeatures {
            duration_ms: 100,
            rms: 0.1,
            peak: 0.2,
            zero_crossing_rate: 0.1,
            clipped_fraction: 0.0,
            signature: vec![0.1; FEATURE_BUCKETS],
        };
        let id = PlaybackSessionId("playback:release".into());
        verifier
            .register_target(TargetOutputEvidence {
                correlation: SpeechCorrelation {
                    playback_session_id: id.clone(),
                    ..correlation()
                },
                features,
            })
            .unwrap();
        assert!(verifier.release(&id));
        assert_eq!(verifier.retained_target_count(), 0);
    }

    #[test]
    fn failure_clipping_and_dropout_remain_explicit() {
        let base = PerceivedAudioEvidence {
            observation_id: MicrophoneObservationId("obs:failure".into()),
            playback_session_id: PlaybackSessionId("playback:failure".into()),
            occurred_at_ms: 1,
            device_reported_playing: false,
            features: None,
            external_speech_probability: 0.0,
            dropout_detected: false,
        };
        let target = TargetOutputEvidence {
            correlation: SpeechCorrelation {
                playback_session_id: base.playback_session_id.clone(),
                ..correlation()
            },
            features: AudioEvidenceFeatures {
                duration_ms: 100,
                rms: 0.1,
                peak: 0.2,
                zero_crossing_rate: 0.1,
                clipped_fraction: 0.0,
                signature: vec![0.1; FEATURE_BUCKETS],
            },
        };
        let mut verifier = OutputVerifier::default();
        verifier.register_target(target.clone()).unwrap();
        assert_eq!(
            verifier.verify(&base).unwrap().classification,
            OutputClassification::PlaybackFailure
        );
        let clipped = PerceivedAudioEvidence {
            device_reported_playing: true,
            features: Some(AudioEvidenceFeatures {
                clipped_fraction: 0.1,
                ..target.features.clone()
            }),
            ..base.clone()
        };
        assert_eq!(
            verifier.verify(&clipped).unwrap().classification,
            OutputClassification::Clipped
        );
        let dropout = PerceivedAudioEvidence {
            device_reported_playing: true,
            dropout_detected: true,
            features: Some(target.features),
            ..base
        };
        assert_eq!(
            verifier.verify(&dropout).unwrap().classification,
            OutputClassification::Dropout
        );
    }

    #[test]
    fn echo_reference_is_explicitly_ephemeral_and_format_checked() {
        let reference = EchoReferenceFrame {
            playback_session_id: PlaybackSessionId("playback:echo".into()),
            sequence: 0,
            start_frame: 0,
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.0; 160],
        };
        reference.validate().unwrap();
        assert_eq!(reference.samples.len(), 160);
    }
}
