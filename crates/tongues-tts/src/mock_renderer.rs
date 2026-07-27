//! Deterministic mock TTS renderer for unit and integration tests.
//!
//! [`MockTtsRenderer`] implements [`SpeechSynthesisEngine`] without loading
//! any model checkpoint or opening an audio device. It generates a
//! deterministic synthetic waveform whose duration is proportional to the
//! number of phone tokens in the plan so that tests can verify timing,
//! crossfade, and ledger behaviour without expensive model inference.
//!
//! # Determinism
//!
//! Given the same [`SpeechSynthesisRequest`] the renderer always produces the
//! same PCM, making it suitable for snapshot and fixture tests.
//!
//! # Configuration
//!
//! [`MockTtsRendererConfig`] controls:
//! - `sample_rate_hz`: output sample rate (default 22 050 Hz).
//! - `samples_per_phone`: PCM frames generated per phone token (default 512).
//! - `amplitude`: peak waveform amplitude (default 0.1).
//! - `frequency_hz`: sine wave frequency used for the generated waveform
//!   (default 440.0 Hz — concert A).
//!
//! All fields have sensible defaults so tests that don't care about audio
//! content can construct `MockTtsRendererConfig::default()` and move on.

use anyhow::Result;
use speaking::UtterancePlan;

use crate::{
    AudioChunk, AudioSink, SpeechModelCapabilities, SpeechModelFamily, SpeechSynthesisEngine,
    SpeechSynthesisRequest,
};

/// Configuration for the deterministic mock TTS renderer.
#[derive(Debug, Clone, PartialEq)]
pub struct MockTtsRendererConfig {
    /// Sample rate of the generated waveform (Hz).
    pub sample_rate_hz: u32,
    /// Number of PCM samples generated per phone token in the plan.
    pub samples_per_phone: usize,
    /// Peak amplitude of the sine wave (0.0 – 1.0).
    pub amplitude: f32,
    /// Frequency of the sine wave (Hz).
    pub frequency_hz: f32,
    /// Whether to emit PCM in a single chunk (`true`) or in multiple streaming
    /// chunks of `chunk_samples` each (`false`).
    pub single_chunk: bool,
    /// Size of each streaming chunk in samples (ignored when `single_chunk`).
    pub chunk_samples: usize,
}

impl Default for MockTtsRendererConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 22_050,
            samples_per_phone: 512,
            amplitude: 0.1,
            frequency_hz: 440.0,
            single_chunk: true,
            chunk_samples: 1_024,
        }
    }
}

/// A deterministic, checkpoint-free TTS renderer for tests.
///
/// See the [module documentation](self) for a detailed description.
#[derive(Debug, Clone)]
pub struct MockTtsRenderer {
    config: MockTtsRendererConfig,
}

impl MockTtsRenderer {
    /// Create a renderer with the given configuration.
    pub fn new(config: MockTtsRendererConfig) -> Self {
        Self { config }
    }

    /// Synthesize a deterministic waveform from the phones in `plan`.
    ///
    /// The returned buffer length is
    /// `plan.target_phones.len() * config.samples_per_phone`, rounded to at
    /// least `config.samples_per_phone` when the plan has no phone tokens so
    /// that callers always receive non-empty audio.
    pub fn synthesize_plan_to_vec(&self, plan: &UtterancePlan) -> Vec<f32> {
        let phone_count = plan.target_phones.len().max(1);
        let total_samples = phone_count * self.config.samples_per_phone;
        self.generate_pcm(total_samples)
    }

    fn generate_pcm(&self, total_samples: usize) -> Vec<f32> {
        let sample_rate = self.config.sample_rate_hz as f32;
        let frequency = self.config.frequency_hz;
        let amplitude = self.config.amplitude;
        (0..total_samples)
            .map(|index| {
                let phase = 2.0 * std::f32::consts::PI * frequency * index as f32 / sample_rate;
                amplitude * phase.sin()
            })
            .collect()
    }
}

impl SpeechSynthesisEngine for MockTtsRenderer {
    fn capabilities(&self) -> SpeechModelCapabilities {
        SpeechModelCapabilities {
            family: SpeechModelFamily::EndToEndSpeech,
            supports_named_speakers: false,
            supports_languages: false,
            supports_reference_audio: false,
            supports_voice_conversion: false,
            integrated_vocoder: true,
        }
    }

    fn sample_rate_hz(&self) -> u32 {
        self.config.sample_rate_hz
    }

    fn synthesize_plan_streaming(
        &mut self,
        request: &SpeechSynthesisRequest,
        sink: &mut dyn AudioSink,
    ) -> Result<()> {
        let pcm = self.synthesize_plan_to_vec(&request.plan);

        if self.config.single_chunk || pcm.len() <= self.config.chunk_samples {
            sink.emit(AudioChunk {
                chunk_index: 0,
                is_final: true,
                pause_after_ms: 0,
                sample_rate_hz: self.config.sample_rate_hz,
                pcm_mono_f32: pcm,
            })?;
        } else {
            let chunk_size = self.config.chunk_samples;
            let chunks: Vec<_> = pcm.chunks(chunk_size).collect();
            let last_index = chunks.len().saturating_sub(1);
            for (index, chunk) in chunks.iter().enumerate() {
                sink.emit(AudioChunk {
                    chunk_index: index,
                    is_final: index == last_index,
                    pause_after_ms: 0,
                    sample_rate_hz: self.config.sample_rate_hz,
                    pcm_mono_f32: chunk.to_vec(),
                })?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use speaking::{EvidenceProvenance, EvidenceSource, ProsodyTrack, UtteranceId, VarietyId};

    fn empty_plan() -> UtterancePlan {
        UtterancePlan {
            id: UtteranceId("mock-test".into()),
            variety: VarietyId("en-US".into()),
            speaker: None,
            intended_text: None,
            intended_morphemes: Vec::new(),
            intended_phonemes: Vec::new(),
            target_phones: Vec::new(),
            target_syllables: Vec::new(),
            boundaries: Vec::new(),
            target_prosody: ProsodyTrack::default(),
            target_acoustics: Vec::new(),
            speaker_reference: None,
            style: None,
            provenance: EvidenceProvenance {
                source: EvidenceSource::Manual,
                method: "mock-test".into(),
                version: None,
            },
        }
    }

    fn plan_with_phones(count: usize) -> UtterancePlan {
        let mut plan = empty_plan();
        let provenance = EvidenceProvenance {
            source: EvidenceSource::Manual,
            method: "mock-test".into(),
            version: None,
        };
        plan.target_phones = (0..count)
            .map(|index| speaking::PhoneToken {
                phone: speaking::Spec::Known(speaking::PhoneId::from(format!("ipa.phone.{index}"))),
                span: None,
                features: speaking::FeatureBundle::default(),
                acoustic_evidence: Vec::new(),
                confidence: 1.0,
                provenance: provenance.clone(),
            })
            .collect();
        plan
    }

    #[test]
    fn mock_renderer_produces_deterministic_pcm() {
        let renderer = MockTtsRenderer::new(MockTtsRendererConfig::default());
        let plan = plan_with_phones(3);
        let pcm1 = renderer.synthesize_plan_to_vec(&plan);
        let pcm2 = renderer.synthesize_plan_to_vec(&plan);
        assert_eq!(pcm1, pcm2);
        assert_eq!(pcm1.len(), 3 * 512);
    }

    #[test]
    fn mock_renderer_empty_plan_still_produces_audio() {
        let renderer = MockTtsRenderer::new(MockTtsRendererConfig::default());
        let plan = empty_plan();
        let pcm = renderer.synthesize_plan_to_vec(&plan);
        assert!(!pcm.is_empty());
        assert_eq!(pcm.len(), 512); // 1 * samples_per_phone
    }

    #[test]
    fn mock_renderer_pcm_is_finite() {
        let renderer = MockTtsRenderer::new(MockTtsRendererConfig::default());
        let plan = plan_with_phones(2);
        let pcm = renderer.synthesize_plan_to_vec(&plan);
        assert!(pcm.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn mock_renderer_emits_single_chunk_by_default() {
        let mut renderer = MockTtsRenderer::new(MockTtsRendererConfig::default());
        let plan = plan_with_phones(2);
        let request = SpeechSynthesisRequest {
            plan: plan.clone(),
            options: Default::default(),
        };
        let mut chunks = Vec::new();
        renderer
            .synthesize_plan_streaming(&request, &mut |chunk: AudioChunk| {
                chunks.push(chunk);
                Ok(())
            })
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_final);
        assert_eq!(chunks[0].sample_rate_hz, 22_050);
    }

    #[test]
    fn mock_renderer_streaming_chunks_cover_full_waveform() {
        let config = MockTtsRendererConfig {
            single_chunk: false,
            chunk_samples: 100,
            samples_per_phone: 300,
            ..Default::default()
        };
        let mut renderer = MockTtsRenderer::new(config.clone());
        let plan = plan_with_phones(2);
        let request = SpeechSynthesisRequest {
            plan: plan.clone(),
            options: Default::default(),
        };
        let expected_pcm = renderer.synthesize_plan_to_vec(&plan);
        let mut received: Vec<f32> = Vec::new();
        let mut last_is_final = false;
        renderer
            .synthesize_plan_streaming(&request, &mut |chunk: AudioChunk| {
                last_is_final = chunk.is_final;
                received.extend(chunk.pcm_mono_f32);
                Ok(())
            })
            .unwrap();
        assert!(last_is_final);
        assert_eq!(received, expected_pcm);
    }

    #[test]
    fn mock_renderer_implements_speech_synthesis_engine() {
        let renderer = MockTtsRenderer::new(MockTtsRendererConfig::default());
        let caps = renderer.capabilities();
        assert_eq!(renderer.sample_rate_hz(), 22_050);
        assert!(!caps.supports_named_speakers);
    }
}
