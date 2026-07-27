//! Optional, ordered streaming cleanup that preserves audio geometry.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::{
    invalid, rms, AudioBuffer, AudioSource, AudioSourceDescriptor, AudioSourceEvent, Result,
    SourceAudioChunk,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CleanupStageConfig {
    DcRemoval {
        pole: f32,
    },
    GainControl {
        target_rms: f32,
        maximum_gain: f32,
        adaptation: f32,
    },
    LowPass {
        cutoff_hz: f32,
    },
    NoiseGate {
        threshold_rms: f32,
        attenuation: f32,
    },
    EchoCancellation {
        delay_ms: u64,
        strength: f32,
    },
    SourceSeparation {
        floor_rms: f32,
    },
}

impl CleanupStageConfig {
    pub fn defaults() -> Vec<Self> {
        vec![
            Self::DcRemoval { pole: 0.995 },
            Self::NoiseGate {
                threshold_rms: 0.012,
                attenuation: 0.15,
            },
            Self::GainControl {
                target_rms: 0.08,
                maximum_gain: 4.0,
                adaptation: 0.05,
            },
        ]
    }

    fn name(&self) -> &'static str {
        match self {
            Self::DcRemoval { .. } => "dc_removal",
            Self::GainControl { .. } => "gain_control",
            Self::LowPass { .. } => "low_pass",
            Self::NoiseGate { .. } => "noise_gate",
            Self::EchoCancellation { .. } => "echo_cancellation",
            Self::SourceSeparation { .. } => "source_separation",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CleanupCapability {
    pub kind: String,
    pub preserves_sample_rate: bool,
    pub preserves_channels: bool,
    pub preserves_frame_timeline: bool,
    pub bounded_state: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CleanupStageTrace {
    pub kind: String,
    pub bypassed: bool,
    pub algorithmic_latency_frames: usize,
    pub input_rms: f32,
    pub output_rms: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProcessedAudio {
    pub audio: AudioBuffer,
    pub stages: Vec<CleanupStageTrace>,
    pub algorithmic_latency_frames: usize,
}

pub trait CleanupStage {
    fn kind(&self) -> &'static str;
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()>;
    fn algorithmic_latency_frames(&self) -> usize {
        0
    }
    fn reset(&mut self);
}

pub struct CleanupPipeline {
    stages: Vec<(bool, Box<dyn CleanupStage>)>,
}

impl CleanupPipeline {
    pub fn new(configs: &[CleanupStageConfig]) -> Result<Self> {
        let stages = configs
            .iter()
            .map(|config| build_stage(config).map(|stage| (false, stage)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { stages })
    }

    pub fn capabilities() -> Vec<CleanupCapability> {
        CleanupStageConfig::defaults()
            .into_iter()
            .chain([
                CleanupStageConfig::LowPass { cutoff_hz: 7_000.0 },
                CleanupStageConfig::EchoCancellation {
                    delay_ms: 100,
                    strength: 0.5,
                },
                CleanupStageConfig::SourceSeparation { floor_rms: 0.01 },
            ])
            .map(|config| CleanupCapability {
                kind: config.name().into(),
                preserves_sample_rate: true,
                preserves_channels: true,
                preserves_frame_timeline: true,
                bounded_state: true,
            })
            .collect()
    }

    pub fn set_bypassed(&mut self, index: usize, bypassed: bool) -> Result<()> {
        let Some((state, _)) = self.stages.get_mut(index) else {
            return Err(invalid(format!(
                "cleanup stage index {index} is out of range"
            )));
        };
        *state = bypassed;
        Ok(())
    }

    pub fn process(&mut self, input: &AudioBuffer) -> Result<ProcessedAudio> {
        input.validate()?;
        let mut audio = input.clone();
        let mut traces = Vec::with_capacity(self.stages.len());
        let mut latency = 0;
        for (bypassed, stage) in &mut self.stages {
            let input_rms = rms(&audio.samples);
            if !*bypassed {
                stage.process(&mut audio)?;
            }
            ensure_geometry(input, &audio, stage.kind())?;
            latency += if *bypassed {
                0
            } else {
                stage.algorithmic_latency_frames()
            };
            traces.push(CleanupStageTrace {
                kind: stage.kind().into(),
                bypassed: *bypassed,
                algorithmic_latency_frames: if *bypassed {
                    0
                } else {
                    stage.algorithmic_latency_frames()
                },
                input_rms,
                output_rms: rms(&audio.samples),
            });
        }
        Ok(ProcessedAudio {
            audio,
            stages: traces,
            algorithmic_latency_frames: latency,
        })
    }

    pub fn reset(&mut self) {
        for (_, stage) in &mut self.stages {
            stage.reset();
        }
    }
}

pub struct CleanupAudioSource<S> {
    inner: S,
    descriptor: AudioSourceDescriptor,
    pipeline: CleanupPipeline,
}

impl<S: AudioSource> CleanupAudioSource<S> {
    pub fn new(inner: S, configs: &[CleanupStageConfig]) -> Result<Self> {
        let mut descriptor = inner.descriptor().clone();
        descriptor.metadata.insert(
            "server_cleanup_stages".into(),
            serde_json::to_string(configs)
                .map_err(|error| invalid(format!("serializing cleanup provenance: {error}")))?,
        );
        Ok(Self {
            inner,
            descriptor,
            pipeline: CleanupPipeline::new(configs)?,
        })
    }
}

impl<S: AudioSource> AudioSource for CleanupAudioSource<S> {
    fn descriptor(&self) -> &AudioSourceDescriptor {
        &self.descriptor
    }

    fn next_event(&mut self) -> Result<AudioSourceEvent> {
        match self.inner.next_event()? {
            AudioSourceEvent::Audio(chunk) => {
                let processed = self.pipeline.process(&chunk.audio)?;
                Ok(AudioSourceEvent::Audio(SourceAudioChunk {
                    sequence: chunk.sequence,
                    start_frame: chunk.start_frame,
                    audio: processed.audio,
                }))
            }
            AudioSourceEvent::Discontinuity(gap) => {
                self.pipeline.reset();
                Ok(AudioSourceEvent::Discontinuity(gap))
            }
            AudioSourceEvent::EndOfStream => Ok(AudioSourceEvent::EndOfStream),
        }
    }

    fn cancel(&mut self) {
        self.inner.cancel();
    }
}

fn ensure_geometry(input: &AudioBuffer, output: &AudioBuffer, stage: &str) -> Result<()> {
    output.validate()?;
    if input.sample_rate_hz != output.sample_rate_hz
        || input.channels != output.channels
        || input.frames() != output.frames()
    {
        return Err(invalid(format!(
            "cleanup stage `{stage}` changed audio geometry"
        )));
    }
    Ok(())
}

fn build_stage(config: &CleanupStageConfig) -> Result<Box<dyn CleanupStage>> {
    Ok(match *config {
        CleanupStageConfig::DcRemoval { pole } => Box::new(DcRemoval::new(pole)?),
        CleanupStageConfig::GainControl {
            target_rms,
            maximum_gain,
            adaptation,
        } => Box::new(GainControl::new(target_rms, maximum_gain, adaptation)?),
        CleanupStageConfig::LowPass { cutoff_hz } => Box::new(LowPass::new(cutoff_hz)?),
        CleanupStageConfig::NoiseGate {
            threshold_rms,
            attenuation,
        } => Box::new(NoiseGate::new(threshold_rms, attenuation)?),
        CleanupStageConfig::EchoCancellation { delay_ms, strength } => {
            Box::new(EchoCancellation::new(delay_ms, strength)?)
        }
        CleanupStageConfig::SourceSeparation { floor_rms } => {
            Box::new(SourceSeparation::new(floor_rms)?)
        }
    })
}

struct DcRemoval {
    pole: f32,
    previous_input: Vec<f32>,
    previous_output: Vec<f32>,
}

impl DcRemoval {
    fn new(pole: f32) -> Result<Self> {
        if !pole.is_finite() || !(0.0..1.0).contains(&pole) {
            return Err(invalid("DC removal pole must be finite and in [0, 1)"));
        }
        Ok(Self {
            pole,
            previous_input: Vec::new(),
            previous_output: Vec::new(),
        })
    }
}

impl CleanupStage for DcRemoval {
    fn kind(&self) -> &'static str {
        "dc_removal"
    }
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        let channels = usize::from(audio.channels);
        self.previous_input.resize(channels, 0.0);
        self.previous_output.resize(channels, 0.0);
        for frame in audio.samples.chunks_exact_mut(channels) {
            for (channel, sample) in frame.iter_mut().enumerate() {
                let input = *sample;
                let output = input - self.previous_input[channel]
                    + self.pole * self.previous_output[channel];
                self.previous_input[channel] = input;
                self.previous_output[channel] = output;
                *sample = output;
            }
        }
        Ok(())
    }
    fn reset(&mut self) {
        self.previous_input.clear();
        self.previous_output.clear();
    }
}

struct GainControl {
    target: f32,
    maximum: f32,
    adaptation: f32,
    gain: f32,
}
impl GainControl {
    fn new(target: f32, maximum: f32, adaptation: f32) -> Result<Self> {
        if !target.is_finite()
            || target <= 0.0
            || !maximum.is_finite()
            || maximum < 1.0
            || !adaptation.is_finite()
            || !(0.0..=1.0).contains(&adaptation)
        {
            return Err(invalid("invalid gain-control configuration"));
        }
        Ok(Self {
            target,
            maximum,
            adaptation,
            gain: 1.0,
        })
    }
}
impl CleanupStage for GainControl {
    fn kind(&self) -> &'static str {
        "gain_control"
    }
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        let desired = (self.target / rms(&audio.samples).max(1.0e-6)).clamp(0.1, self.maximum);
        self.gain += self.adaptation * (desired - self.gain);
        for sample in &mut audio.samples {
            *sample = (*sample * self.gain).clamp(-1.0, 1.0);
        }
        Ok(())
    }
    fn reset(&mut self) {
        self.gain = 1.0;
    }
}

struct LowPass {
    cutoff: f32,
    previous: Vec<f32>,
}
impl LowPass {
    fn new(cutoff: f32) -> Result<Self> {
        if !cutoff.is_finite() || cutoff <= 0.0 {
            return Err(invalid("low-pass cutoff must be positive"));
        }
        Ok(Self {
            cutoff,
            previous: Vec::new(),
        })
    }
}
impl CleanupStage for LowPass {
    fn kind(&self) -> &'static str {
        "low_pass"
    }
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if self.cutoff >= audio.sample_rate_hz as f32 / 2.0 {
            return Err(invalid("low-pass cutoff must be below Nyquist"));
        }
        let channels = usize::from(audio.channels);
        self.previous.resize(channels, 0.0);
        let dt = 1.0 / audio.sample_rate_hz as f32;
        let rc = 1.0 / (2.0 * std::f32::consts::PI * self.cutoff);
        let alpha = dt / (rc + dt);
        for frame in audio.samples.chunks_exact_mut(channels) {
            for (channel, sample) in frame.iter_mut().enumerate() {
                self.previous[channel] += alpha * (*sample - self.previous[channel]);
                *sample = self.previous[channel];
            }
        }
        Ok(())
    }
    fn reset(&mut self) {
        self.previous.clear();
    }
}

struct NoiseGate {
    threshold: f32,
    attenuation: f32,
}
impl NoiseGate {
    fn new(threshold: f32, attenuation: f32) -> Result<Self> {
        if !threshold.is_finite()
            || threshold <= 0.0
            || !attenuation.is_finite()
            || !(0.0..=1.0).contains(&attenuation)
        {
            return Err(invalid("invalid noise-gate configuration"));
        }
        Ok(Self {
            threshold,
            attenuation,
        })
    }
}
impl CleanupStage for NoiseGate {
    fn kind(&self) -> &'static str {
        "noise_gate"
    }
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if rms(&audio.samples) < self.threshold {
            for sample in &mut audio.samples {
                *sample *= self.attenuation;
            }
        }
        Ok(())
    }
    fn reset(&mut self) {}
}

struct EchoCancellation {
    delay_ms: u64,
    strength: f32,
    history: VecDeque<f32>,
    geometry: Option<(u32, u16)>,
}
impl EchoCancellation {
    fn new(delay_ms: u64, strength: f32) -> Result<Self> {
        if delay_ms == 0 || !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            return Err(invalid("invalid echo-cancellation configuration"));
        }
        Ok(Self {
            delay_ms,
            strength,
            history: VecDeque::new(),
            geometry: None,
        })
    }
}
impl CleanupStage for EchoCancellation {
    fn kind(&self) -> &'static str {
        "echo_cancellation"
    }
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        self.geometry
            .get_or_insert((audio.sample_rate_hz, audio.channels));
        if self.geometry != Some((audio.sample_rate_hz, audio.channels)) {
            return Err(invalid("echo canceller format changed"));
        }
        let delay = (u64::from(audio.sample_rate_hz) * self.delay_ms / 1000) as usize
            * usize::from(audio.channels);
        for sample in &mut audio.samples {
            let input = *sample;
            let echo = if self.history.len() >= delay {
                self.history.pop_front().unwrap_or(0.0)
            } else {
                0.0
            };
            *sample = (input - self.strength * echo).clamp(-1.0, 1.0);
            self.history.push_back(input);
        }
        while self.history.len() > delay {
            self.history.pop_front();
        }
        Ok(())
    }
    fn algorithmic_latency_frames(&self) -> usize {
        0
    }
    fn reset(&mut self) {
        self.history.clear();
        self.geometry = None;
    }
}

struct SourceSeparation {
    floor: f32,
}
impl SourceSeparation {
    fn new(floor: f32) -> Result<Self> {
        if !floor.is_finite() || floor < 0.0 {
            return Err(invalid("source-separation floor must be non-negative"));
        }
        Ok(Self { floor })
    }
}
impl CleanupStage for SourceSeparation {
    fn kind(&self) -> &'static str {
        "source_separation"
    }
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        for sample in &mut audio.samples {
            if sample.abs() < self.floor {
                *sample = 0.0;
            }
        }
        Ok(())
    }
    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mono(samples: Vec<f32>) -> AudioBuffer {
        AudioBuffer {
            samples,
            sample_rate_hz: 16_000,
            channels: 1,
        }
    }

    fn peak(audio: &AudioBuffer) -> f32 {
        audio
            .samples
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn clean_speech_preserves_geometry_and_provenance() {
        let input = mono(
            (0..320)
                .map(|index| ((index as f32) * 0.1).sin() * 0.2)
                .collect(),
        );
        let configs = CleanupStageConfig::defaults();
        let mut pipeline = CleanupPipeline::new(&configs).unwrap();
        let output = pipeline.process(&input).unwrap();
        assert_eq!(output.audio.frames(), input.frames());
        assert_eq!(output.audio.sample_rate_hz, input.sample_rate_hz);
        assert_eq!(output.audio.channels, input.channels);
        assert_eq!(
            output
                .stages
                .iter()
                .map(|stage| stage.kind.as_str())
                .collect::<Vec<_>>(),
            ["dc_removal", "noise_gate", "gain_control"]
        );
    }

    #[test]
    fn steady_noise_is_attenuated_by_optional_gate() {
        let input = mono(vec![0.005; 320]);
        let mut pipeline = CleanupPipeline::new(&[CleanupStageConfig::NoiseGate {
            threshold_rms: 0.01,
            attenuation: 0.1,
        }])
        .unwrap();
        let output = pipeline.process(&input).unwrap();
        assert!(rms(&output.audio.samples) < rms(&input.samples) * 0.11);
    }

    #[test]
    fn impulsive_noise_remains_finite_and_bounded() {
        let mut samples = vec![0.0; 320];
        samples[100] = 10.0;
        let input = mono(samples);
        let mut pipeline = CleanupPipeline::new(&[
            CleanupStageConfig::LowPass { cutoff_hz: 3_000.0 },
            CleanupStageConfig::GainControl {
                target_rms: 0.08,
                maximum_gain: 2.0,
                adaptation: 1.0,
            },
        ])
        .unwrap();
        let output = pipeline.process(&input).unwrap();
        assert!(output.audio.samples.iter().all(|sample| sample.is_finite()));
        assert!(peak(&output.audio) <= 1.0);
    }

    #[test]
    fn clipped_input_is_not_amplified_past_full_scale() {
        let input = mono(vec![1.0, -1.0, 1.0, -1.0]);
        let mut pipeline = CleanupPipeline::new(&[CleanupStageConfig::GainControl {
            target_rms: 0.5,
            maximum_gain: 4.0,
            adaptation: 1.0,
        }])
        .unwrap();
        assert!(peak(&pipeline.process(&input).unwrap().audio) <= 1.0);
    }

    #[test]
    fn delayed_echo_is_reduced_with_bounded_history() {
        let mut samples = vec![0.0; 3_200];
        samples[0] = 1.0;
        samples[1_600] = 0.5;
        let input = mono(samples);
        let mut pipeline = CleanupPipeline::new(&[CleanupStageConfig::EchoCancellation {
            delay_ms: 100,
            strength: 0.5,
        }])
        .unwrap();
        let output = pipeline.process(&input).unwrap();
        assert!(output.audio.samples[1_600].abs() < 1.0e-6);
    }

    #[test]
    fn bypass_and_order_are_explicit() {
        let input = mono(vec![0.005; 32]);
        let configs = [
            CleanupStageConfig::NoiseGate {
                threshold_rms: 0.01,
                attenuation: 0.0,
            },
            CleanupStageConfig::GainControl {
                target_rms: 0.1,
                maximum_gain: 4.0,
                adaptation: 1.0,
            },
        ];
        let mut pipeline = CleanupPipeline::new(&configs).unwrap();
        pipeline.set_bypassed(0, true).unwrap();
        let output = pipeline.process(&input).unwrap();
        assert!(output.audio.samples.iter().any(|sample| *sample != 0.0));
        assert!(output.stages[0].bypassed);
        assert!(!output.stages[1].bypassed);
    }

    #[test]
    fn capabilities_are_geometry_preserving_and_bounded() {
        let capabilities = CleanupPipeline::capabilities();
        assert_eq!(capabilities.len(), 6);
        assert!(capabilities.iter().all(|capability| {
            capability.preserves_sample_rate
                && capability.preserves_channels
                && capability.preserves_frame_timeline
                && capability.bounded_state
        }));
    }
}
