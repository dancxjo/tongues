//! Pluggable voice-activity detection over decoded audio frames.

use serde::{Deserialize, Serialize};

use crate::{invalid, rms, AudioBuffer, Result};

const DEFAULT_ENERGY_THRESHOLD_RMS: f32 = 0.02;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VadBackendKind {
    Energy,
    #[cfg(feature = "vad-webrtc")]
    WebRtc,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VadDecision {
    pub backend: VadBackendKind,
    pub speech_probability: f32,
    pub is_speech: bool,
    pub rms: f32,
}

/// A detector may own a thread-affine native engine.
pub trait VoiceActivityDetector {
    fn backend(&self) -> VadBackendKind;
    fn process_frame(&mut self, frame: &AudioBuffer) -> Result<VadDecision>;
}

pub fn create_vad_backend(kind: VadBackendKind) -> Box<dyn VoiceActivityDetector> {
    match kind {
        VadBackendKind::Energy => Box::new(EnergyVad::default()),
        #[cfg(feature = "vad-webrtc")]
        VadBackendKind::WebRtc => Box::new(WebRtcVad::default()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnergyVadConfig {
    pub threshold_rms: f32,
}

impl Default for EnergyVadConfig {
    fn default() -> Self {
        Self {
            threshold_rms: DEFAULT_ENERGY_THRESHOLD_RMS,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EnergyVad {
    config: EnergyVadConfig,
}

impl EnergyVad {
    pub fn new(config: EnergyVadConfig) -> Result<Self> {
        if !config.threshold_rms.is_finite() || config.threshold_rms <= 0.0 {
            return Err(invalid(
                "energy VAD threshold_rms must be finite and positive",
            ));
        }
        Ok(Self { config })
    }
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self {
            config: EnergyVadConfig::default(),
        }
    }
}

impl VoiceActivityDetector for EnergyVad {
    fn backend(&self) -> VadBackendKind {
        VadBackendKind::Energy
    }

    fn process_frame(&mut self, frame: &AudioBuffer) -> Result<VadDecision> {
        frame.validate()?;
        let mono = frame.to_mono()?;
        let frame_rms = rms(&mono);
        Ok(VadDecision {
            backend: self.backend(),
            speech_probability: (frame_rms / self.config.threshold_rms).clamp(0.0, 1.0),
            is_speech: frame_rms >= self.config.threshold_rms,
            rms: frame_rms,
        })
    }
}

#[cfg(feature = "vad-webrtc")]
const WEBRTC_DEFAULT_ENERGY_FALLBACK_RMS: f32 = 0.08;
#[cfg(feature = "vad-webrtc")]
const WEBRTC_DEFAULT_MIN_SPEECH_RMS: f32 = 0.025;
#[cfg(feature = "vad-webrtc")]
const WEBRTC_DEFAULT_NOISE_FLOOR_RMS: f32 = 0.006;

#[cfg(feature = "vad-webrtc")]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WebRtcVadConfig {
    pub energy_fallback_rms: f32,
    pub minimum_speech_rms: f32,
    pub initial_noise_floor_rms: f32,
    pub noise_gate_multiplier: f32,
    pub noise_gate_margin_rms: f32,
    pub noise_floor_alpha: f32,
}

#[cfg(feature = "vad-webrtc")]
impl Default for WebRtcVadConfig {
    fn default() -> Self {
        Self {
            energy_fallback_rms: WEBRTC_DEFAULT_ENERGY_FALLBACK_RMS,
            minimum_speech_rms: WEBRTC_DEFAULT_MIN_SPEECH_RMS,
            initial_noise_floor_rms: WEBRTC_DEFAULT_NOISE_FLOOR_RMS,
            noise_gate_multiplier: 1.8,
            noise_gate_margin_rms: 0.006,
            noise_floor_alpha: 0.05,
        }
    }
}

#[cfg(feature = "vad-webrtc")]
pub struct WebRtcVad {
    engine: webrtc_vad::Vad,
    config: WebRtcVadConfig,
    configured_sample_rate_hz: Option<u32>,
    noise_floor_rms: f32,
    energy_fallback: EnergyVad,
}

#[cfg(feature = "vad-webrtc")]
impl WebRtcVad {
    pub fn new(config: WebRtcVadConfig) -> Result<Self> {
        for (name, value) in [
            ("energy_fallback_rms", config.energy_fallback_rms),
            ("minimum_speech_rms", config.minimum_speech_rms),
            ("noise_gate_multiplier", config.noise_gate_multiplier),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(format!(
                    "WebRTC VAD {name} must be finite and positive"
                )));
            }
        }
        for (name, value) in [
            ("initial_noise_floor_rms", config.initial_noise_floor_rms),
            ("noise_gate_margin_rms", config.noise_gate_margin_rms),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(invalid(format!(
                    "WebRTC VAD {name} must be finite and non-negative"
                )));
            }
        }
        if !config.noise_floor_alpha.is_finite() || !(0.0..=1.0).contains(&config.noise_floor_alpha)
        {
            return Err(invalid(
                "WebRTC VAD noise_floor_alpha must be finite and in [0, 1]",
            ));
        }
        Ok(Self {
            engine: webrtc_vad::Vad::new_with_mode(webrtc_vad::VadMode::VeryAggressive),
            config,
            configured_sample_rate_hz: None,
            noise_floor_rms: config.initial_noise_floor_rms,
            energy_fallback: EnergyVad::new(EnergyVadConfig {
                threshold_rms: config.energy_fallback_rms,
            })?,
        })
    }

    fn speech_gate_rms(&self) -> f32 {
        self.config
            .minimum_speech_rms
            .max(self.noise_floor_rms.mul_add(
                self.config.noise_gate_multiplier,
                self.config.noise_gate_margin_rms,
            ))
    }

    fn observe_noise_floor(&mut self, frame_rms: f32) {
        self.noise_floor_rms = self.noise_floor_rms * (1.0 - self.config.noise_floor_alpha)
            + frame_rms * self.config.noise_floor_alpha;
    }
}

#[cfg(feature = "vad-webrtc")]
impl Default for WebRtcVad {
    fn default() -> Self {
        Self::new(WebRtcVadConfig::default()).expect("default WebRTC VAD configuration is valid")
    }
}

#[cfg(feature = "vad-webrtc")]
impl VoiceActivityDetector for WebRtcVad {
    fn backend(&self) -> VadBackendKind {
        VadBackendKind::WebRtc
    }

    fn process_frame(&mut self, frame: &AudioBuffer) -> Result<VadDecision> {
        frame.validate()?;
        let sample_rate = web_rtc_sample_rate(frame.sample_rate_hz)?;
        if self.configured_sample_rate_hz != Some(frame.sample_rate_hz) {
            self.engine.set_sample_rate(sample_rate);
            self.configured_sample_rate_hz = Some(frame.sample_rate_hz);
        }
        let centered = mean_centered_mono(frame)?;
        ensure_web_rtc_frame_length(frame.sample_rate_hz, centered.len())?;
        let frame_rms = rms(&centered);
        let pcm = centered
            .iter()
            .map(|sample| (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16)
            .collect::<Vec<_>>();
        let web_rtc_speech = self
            .engine
            .is_voice_segment(&pcm)
            .map_err(|_| invalid("WebRTC VAD rejected the audio frame"))?;
        let fallback = self.energy_fallback.process_frame(&AudioBuffer {
            samples: centered,
            sample_rate_hz: frame.sample_rate_hz,
            channels: 1,
        })?;
        let speech_gate_rms = self.speech_gate_rms();
        let gated_web_rtc_speech = web_rtc_speech && frame_rms >= speech_gate_rms;
        let is_speech = gated_web_rtc_speech || fallback.is_speech;
        if !is_speech {
            self.observe_noise_floor(frame_rms);
        }
        Ok(VadDecision {
            backend: self.backend(),
            speech_probability: if web_rtc_speech {
                (frame_rms / speech_gate_rms).clamp(0.0, 1.0)
            } else {
                fallback.speech_probability
            },
            is_speech,
            rms: frame_rms,
        })
    }
}

#[cfg(feature = "vad-webrtc")]
fn web_rtc_sample_rate(sample_rate_hz: u32) -> Result<webrtc_vad::SampleRate> {
    match sample_rate_hz {
        8_000 => Ok(webrtc_vad::SampleRate::Rate8kHz),
        16_000 => Ok(webrtc_vad::SampleRate::Rate16kHz),
        32_000 => Ok(webrtc_vad::SampleRate::Rate32kHz),
        48_000 => Ok(webrtc_vad::SampleRate::Rate48kHz),
        _ => Err(invalid(format!(
            "WebRTC VAD supports 8000/16000/32000/48000 Hz, not {sample_rate_hz} Hz"
        ))),
    }
}

#[cfg(feature = "vad-webrtc")]
fn ensure_web_rtc_frame_length(sample_rate_hz: u32, mono_samples: usize) -> Result<()> {
    let samples_10ms = usize::try_from(sample_rate_hz / 100).unwrap_or(0);
    if [samples_10ms, samples_10ms * 2, samples_10ms * 3].contains(&mono_samples) {
        Ok(())
    } else {
        Err(invalid(format!(
            "WebRTC VAD needs a mono 10/20/30 ms frame; received {mono_samples} samples at {sample_rate_hz} Hz"
        )))
    }
}

#[cfg(feature = "vad-webrtc")]
fn mean_centered_mono(frame: &AudioBuffer) -> Result<Vec<f32>> {
    let mut mono = frame.to_mono()?;
    if mono.is_empty() {
        return Ok(mono);
    }
    let mean = mono.iter().sum::<f32>() / mono.len() as f32;
    for sample in &mut mono {
        *sample -= mean;
    }
    Ok(mono)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(samples: Vec<f32>, sample_rate_hz: u32) -> AudioBuffer {
        AudioBuffer {
            samples,
            sample_rate_hz,
            channels: 1,
        }
    }

    #[test]
    fn energy_vad_separates_silence_and_speech() {
        let mut vad = EnergyVad::default();
        let silence = vad.process_frame(&frame(vec![0.001; 160], 16_000)).unwrap();
        let speech = vad
            .process_frame(&frame(
                (0..160)
                    .map(|index| if index % 2 == 0 { 0.1 } else { -0.1 })
                    .collect(),
                16_000,
            ))
            .unwrap();
        assert!(!silence.is_speech);
        assert!(speech.is_speech);
        assert!(speech.speech_probability > silence.speech_probability);
    }

    #[cfg(feature = "vad-webrtc")]
    #[test]
    fn web_rtc_vad_ignores_dc_offset() {
        let mut vad = WebRtcVad::default();
        let decision = vad.process_frame(&frame(vec![0.04; 160], 16_000)).unwrap();
        assert!(!decision.is_speech);
        assert!(decision.rms < 1.0e-6);
    }

    #[cfg(feature = "vad-webrtc")]
    #[test]
    fn web_rtc_vad_keeps_loud_energy_fallback() {
        let mut vad = WebRtcVad::default();
        let decision = vad
            .process_frame(&frame(
                (0..160)
                    .map(|index| if index % 2 == 0 { 0.1 } else { -0.1 })
                    .collect(),
                16_000,
            ))
            .unwrap();
        assert!(decision.is_speech);
    }

    #[cfg(feature = "vad-webrtc")]
    #[test]
    fn web_rtc_vad_rejects_unsupported_frame_geometry() {
        let mut vad = WebRtcVad::default();
        let error = vad
            .process_frame(&frame(vec![0.0; 100], 16_000))
            .unwrap_err();
        assert!(error.to_string().contains("10/20/30 ms"));
    }
}
