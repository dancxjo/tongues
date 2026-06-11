use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::feature::{FeatureBundle, FeatureValue};
use crate::ids::{AcousticCueId, FeatureId, PhoneId, PhonemeId};
use crate::spec::Spec;
use crate::time::TimeSpan;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticFrame {
    pub span: TimeSpan,
    pub f0_hz: Spec<f32>,
    pub energy_db: Spec<f32>,
    pub voicing_probability: Spec<f32>,
    pub periodicity: Spec<f32>,
    pub harmonicity: Spec<f32>,
    pub formants: Vec<Formant>,
    pub spectral_centroid_hz: Spec<f32>,
    pub spectral_tilt_db_per_octave: Spec<f32>,
    pub zero_crossing_rate: Spec<f32>,
    pub vectors: Vec<AcousticVector>,
}

impl AcousticFrame {
    /// Estimate a coarse vocal-tract posture from this frame's formants.
    ///
    /// This is an acoustic proxy, not anatomical measurement. It is most useful
    /// as a frame-by-frame trajectory feature beside energy, voicing, pitch, and
    /// phone likelihoods.
    pub fn estimate_vocal_tract_posture(&self) -> Spec<VocalTractEstimate> {
        estimate_vocal_tract_posture(self, &VocalTractEstimateConfig::default())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Formant {
    pub index: u8,
    pub hz: Spec<f32>,
    pub bandwidth_hz: Spec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VocalTractEstimate {
    pub jaw_open: f32,
    pub tongue_high: f32,
    pub tongue_front: f32,
    pub lip_round: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VocalTractEstimateConfig {
    pub f1_min_hz: f32,
    pub f1_max_hz: f32,
    pub f2_min_hz: f32,
    pub f2_max_hz: f32,
    pub f3_min_hz: f32,
    pub f3_max_hz: f32,
    pub f1_frontness_coupling: f32,
    pub spectral_tilt_rounding_min_db_per_octave: f32,
    pub spectral_tilt_rounding_max_db_per_octave: f32,
}

impl Default for VocalTractEstimateConfig {
    fn default() -> Self {
        Self {
            f1_min_hz: 250.0,
            f1_max_hz: 900.0,
            f2_min_hz: 600.0,
            f2_max_hz: 3000.0,
            f3_min_hz: 1400.0,
            f3_max_hz: 3600.0,
            f1_frontness_coupling: 0.35,
            spectral_tilt_rounding_min_db_per_octave: -3.0,
            spectral_tilt_rounding_max_db_per_octave: -18.0,
        }
    }
}

pub fn estimate_vocal_tract_posture(
    frame: &AcousticFrame,
    config: &VocalTractEstimateConfig,
) -> Spec<VocalTractEstimate> {
    let Some((f1_hz, f1_confidence)) = formant_hz(&frame.formants, 1) else {
        return Spec::Unknown;
    };
    let Some((f2_hz, f2_confidence)) = formant_hz(&frame.formants, 2) else {
        return Spec::Unknown;
    };

    if !valid_vocal_tract_config(config) {
        return Spec::Unknown;
    }

    let f3 = formant_hz(&frame.formants, 3);
    let spectral_tilt = spec_value(&frame.spectral_tilt_db_per_octave);
    let voicing = spec_value(&frame.voicing_probability);

    let jaw_open = normalize(f1_hz, config.f1_min_hz, config.f1_max_hz);
    let tongue_high = 1.0 - jaw_open;

    let compensated_f2 = f2_hz - config.f1_frontness_coupling * f1_hz;
    let compensated_f2_min = config.f2_min_hz - config.f1_frontness_coupling * config.f1_max_hz;
    let compensated_f2_max = config.f2_max_hz - config.f1_frontness_coupling * config.f1_min_hz;
    let tongue_front = normalize(compensated_f2, compensated_f2_min, compensated_f2_max);

    let low_f2 = 1.0 - tongue_front;
    let (low_f3, f3_confidence) = f3
        .map(|(hz, confidence)| {
            (
                1.0 - normalize(hz, config.f3_min_hz, config.f3_max_hz),
                confidence,
            )
        })
        .unwrap_or((0.0, 0.0));
    let tilt_rounding = spectral_tilt
        .map(|(tilt, _)| {
            normalize_reversed(
                tilt,
                config.spectral_tilt_rounding_max_db_per_octave,
                config.spectral_tilt_rounding_min_db_per_octave,
            )
        })
        .unwrap_or(0.0);

    let lip_round = if f3.is_some() {
        clamp01(0.4 * low_f2 + 0.5 * low_f3 + 0.1 * tilt_rounding)
    } else {
        clamp01(0.75 * low_f2 + 0.25 * tilt_rounding)
    };

    let mut confidence = (f1_confidence + f2_confidence) / 2.0;
    if f3.is_some() {
        confidence = (confidence * 2.0 + f3_confidence) / 3.0;
    } else {
        confidence *= 0.75;
    }
    if let Some((voicing_probability, voicing_confidence)) = voicing {
        confidence *= 0.5 + 0.5 * clamp01(voicing_probability) * voicing_confidence;
    }
    if let Some((_, tilt_confidence)) = spectral_tilt {
        confidence = (confidence * 4.0 + tilt_confidence) / 5.0;
    }

    Spec::Known(VocalTractEstimate {
        jaw_open,
        tongue_high,
        tongue_front,
        lip_round,
        confidence: clamp01(confidence),
    })
}

pub fn estimate_vocal_tract_trajectory(frames: &[AcousticFrame]) -> Vec<Spec<VocalTractEstimate>> {
    let config = VocalTractEstimateConfig::default();
    frames
        .iter()
        .map(|frame| estimate_vocal_tract_posture(frame, &config))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticVector {
    pub kind: String,
    pub values: Vec<f32>,
}

fn formant_hz(formants: &[Formant], index: u8) -> Option<(f32, f32)> {
    formants
        .iter()
        .find(|formant| formant.index == index)
        .and_then(|formant| spec_value(&formant.hz))
}

fn spec_value(spec: &Spec<f32>) -> Option<(f32, f32)> {
    match spec {
        Spec::Known(value) if value.is_finite() => Some((*value, 1.0)),
        Spec::Gradient { value, confidence } if value.is_finite() => {
            Some((*value, clamp01(*confidence)))
        }
        Spec::Variable(values) => finite_average(values).map(|value| (value, 0.5)),
        Spec::Unknown | Spec::Unspecified | Spec::NotApplicable | Spec::Known(_) => None,
        Spec::Gradient { .. } => None,
    }
}

fn finite_average(values: &[f32]) -> Option<f32> {
    let mut sum = 0.0;
    let mut count = 0.0;
    for value in values.iter().copied().filter(|value| value.is_finite()) {
        sum += value;
        count += 1.0;
    }
    (count > 0.0).then_some(sum / count)
}

fn valid_vocal_tract_config(config: &VocalTractEstimateConfig) -> bool {
    config.f1_min_hz.is_finite()
        && config.f1_max_hz.is_finite()
        && config.f1_min_hz < config.f1_max_hz
        && config.f2_min_hz.is_finite()
        && config.f2_max_hz.is_finite()
        && config.f2_min_hz < config.f2_max_hz
        && config.f3_min_hz.is_finite()
        && config.f3_max_hz.is_finite()
        && config.f3_min_hz < config.f3_max_hz
        && config.f1_frontness_coupling.is_finite()
        && config.spectral_tilt_rounding_min_db_per_octave.is_finite()
        && config.spectral_tilt_rounding_max_db_per_octave.is_finite()
        && config.spectral_tilt_rounding_max_db_per_octave
            < config.spectral_tilt_rounding_min_db_per_octave
}

fn normalize(value: f32, min: f32, max: f32) -> f32 {
    clamp01((value - min) / (max - min))
}

fn normalize_reversed(value: f32, min: f32, max: f32) -> f32 {
    1.0 - normalize(value, min, max)
}

fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticObservation {
    pub cue: AcousticCueId,
    pub value: Spec<FeatureValue>,
    pub span: Option<TimeSpan>,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticCueDef {
    pub id: AcousticCueId,
    pub name: String,
    pub feature: FeatureId,
    pub targets: Vec<CueTarget>,
    #[serde(default)]
    pub diagnosticity: CueDiagnosticity,
    #[serde(default)]
    pub dependencies: Vec<CueDependency>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CueDiagnosticity {
    Robust,
    #[default]
    Moderate,
    Weak,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CueDependency {
    SpeakerDependent,
    ContextDependent,
    StyleDependent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CueTarget {
    Phone(PhoneId),
    Phoneme(PhonemeId),
    Feature(FeatureId),
    Boundary,
    Stress,
    Tone,
    Speaker,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AcousticProfile {
    #[serde(default)]
    pub cues: HashMap<AcousticCueId, AcousticCueDef>,
    #[serde(default)]
    pub phone_models: HashMap<PhoneId, AcousticTargetModel>,
    #[serde(default)]
    pub phoneme_models: HashMap<PhonemeId, AcousticTargetModel>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AcousticTargetModel {
    #[serde(default)]
    pub expected_features: FeatureBundle,
    #[serde(default)]
    pub weighted_cues: Vec<WeightedCue>,
    #[serde(default)]
    pub landmarks: Vec<AcousticLandmark>,
    #[serde(default)]
    pub range_targets: Vec<AcousticRangeTarget>,
    #[serde(default)]
    pub temporal: AcousticTemporalModel,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AcousticTemporalModel {
    #[serde(default)]
    pub landmark_order: Vec<LandmarkOrderStep>,
    #[serde(default)]
    pub subsegments: Vec<SubsegmentProportion>,
    #[serde(default)]
    pub sampling_strategy: Option<SegmentSamplingStrategy>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LandmarkOrderStep {
    pub kind: AcousticLandmarkKind,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubsegmentProportion {
    pub role: SubsegmentRole,
    pub proportion: NumericRange,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubsegmentRole {
    Closure,
    Burst,
    Aspiration,
    VoiceOnsetLag,
    Frication,
    TapClosure,
    VowelOnsetTransition,
    VowelSteadyTarget,
    VowelOffsetTransition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentSamplingStrategy {
    UseMidpoint,
    UseOnsetTransition,
    UseOffsetTransition,
    UseOnsetAndOffsetTransitions,
    UseFullTrajectory,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticLandmark {
    pub id: String,
    pub kind: AcousticLandmarkKind,
    pub anchor: LandmarkAnchor,
    pub window: RelativeTimeWindow,
    #[serde(default)]
    pub expected_features: FeatureBundle,
    #[serde(default)]
    pub weighted_cues: Vec<WeightedCue>,
    #[serde(default)]
    pub range_targets: Vec<AcousticRangeTarget>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticRangeTarget {
    pub measurement: AcousticMeasurement,
    pub range: NumericRange,
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericRange {
    pub min: f32,
    pub max: f32,
    pub unit: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcousticMeasurement {
    Formant { index: u8 },
    VoiceOnsetTime,
    ClosureDuration,
    FricationDuration,
    SpectralCentroid,
    SpectralSkew,
    NasalMurmurBand,
    NasalAntiresonance,
    NasalPlaceTransition,
    FormantTransition { index: u8 },
    AffricateClosureToFrication,
    SilenceDuration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcousticLandmarkKind {
    Closure,
    ReleaseBurst,
    Aspiration,
    VoicingOnset,
    VowelTarget,
    FormantTransition,
    PeriodicVoicing,
    AperiodicNoise,
    Boundary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LandmarkAnchor {
    SegmentStart,
    SegmentCenter,
    SegmentEnd,
    Release,
    VoicingOnset,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RelativeTimeWindow {
    pub start_s: f32,
    pub end_s: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightedCue {
    pub cue: AcousticCueId,
    pub weight: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(formants: &[(u8, f32)]) -> AcousticFrame {
        AcousticFrame {
            span: TimeSpan {
                start_s: 0.0,
                end_s: 0.01,
            },
            f0_hz: Spec::Unspecified,
            energy_db: Spec::Unspecified,
            voicing_probability: Spec::Known(0.9),
            periodicity: Spec::Unspecified,
            harmonicity: Spec::Unspecified,
            formants: formants
                .iter()
                .map(|(index, hz)| Formant {
                    index: *index,
                    hz: Spec::Known(*hz),
                    bandwidth_hz: Spec::Unspecified,
                })
                .collect(),
            spectral_centroid_hz: Spec::Unspecified,
            spectral_tilt_db_per_octave: Spec::Known(-6.0),
            zero_crossing_rate: Spec::Unspecified,
            vectors: Vec::new(),
        }
    }

    fn known_estimate(spec: Spec<VocalTractEstimate>) -> VocalTractEstimate {
        match spec {
            Spec::Known(estimate) => estimate,
            other => panic!("expected known vocal tract estimate, got {other:?}"),
        }
    }

    #[test]
    fn high_front_vowel_estimates_closed_front_unrounded_posture() {
        let estimate = known_estimate(
            frame(&[(1, 300.0), (2, 2500.0), (3, 3300.0)]).estimate_vocal_tract_posture(),
        );

        assert!(estimate.jaw_open < 0.15);
        assert!(estimate.tongue_high > 0.85);
        assert!(estimate.tongue_front > 0.75);
        assert!(estimate.lip_round < 0.35);
        assert!(estimate.confidence > 0.9);
    }

    #[test]
    fn back_rounded_vowel_estimates_high_back_rounded_posture() {
        let estimate = known_estimate(
            frame(&[(1, 320.0), (2, 800.0), (3, 2200.0)]).estimate_vocal_tract_posture(),
        );

        assert!(estimate.tongue_high > 0.8);
        assert!(estimate.tongue_front < 0.2);
        assert!(estimate.lip_round > 0.55);
    }

    #[test]
    fn open_vowel_estimates_open_jaw_and_lower_tongue() {
        let estimate = known_estimate(
            frame(&[(1, 850.0), (2, 1500.0), (3, 3000.0)]).estimate_vocal_tract_posture(),
        );

        assert!(estimate.jaw_open > 0.85);
        assert!(estimate.tongue_high < 0.15);
    }

    #[test]
    fn missing_core_formants_are_unknown() {
        assert_eq!(
            frame(&[(1, 500.0), (3, 2500.0)]).estimate_vocal_tract_posture(),
            Spec::Unknown
        );
    }

    #[test]
    fn missing_f3_still_estimates_with_lower_confidence() {
        let with_f3 = known_estimate(
            frame(&[(1, 320.0), (2, 800.0), (3, 2200.0)]).estimate_vocal_tract_posture(),
        );
        let without_f3 =
            known_estimate(frame(&[(1, 320.0), (2, 800.0)]).estimate_vocal_tract_posture());

        assert!(without_f3.lip_round > 0.5);
        assert!(without_f3.confidence < with_f3.confidence);
    }

    #[test]
    fn trajectory_estimates_one_posture_per_frame() {
        let frames = vec![
            frame(&[(1, 300.0), (2, 2500.0), (3, 3300.0)]),
            frame(&[(1, 850.0), (2, 1500.0), (3, 3000.0)]),
        ];

        let trajectory = estimate_vocal_tract_trajectory(&frames);

        assert_eq!(trajectory.len(), 2);
        assert!(matches!(trajectory[0], Spec::Known(_)));
        assert!(matches!(trajectory[1], Spec::Known(_)));
    }
}
