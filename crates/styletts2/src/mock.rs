use crate::backend::{
    StyleTts2AudioChunk, StyleTts2AudioSink, StyleTts2Backend, StyleTts2Error,
    StyleTts2SynthesisOutput,
};
use crate::request::StyleTts2SynthesisRequest;

#[derive(Debug, Clone, PartialEq)]
pub struct MockStyleTts2Backend {
    pub sample_rate_hz: u32,
    pub amplitude: f32,
}

impl Default for MockStyleTts2Backend {
    fn default() -> Self {
        Self {
            sample_rate_hz: 24_000,
            amplitude: 0.05,
        }
    }
}

impl MockStyleTts2Backend {
    pub fn new(sample_rate_hz: u32) -> Self {
        Self {
            sample_rate_hz,
            ..Self::default()
        }
    }
}

impl StyleTts2Backend for MockStyleTts2Backend {
    fn synthesize(
        &mut self,
        request: &StyleTts2SynthesisRequest,
    ) -> Result<StyleTts2SynthesisOutput, StyleTts2Error> {
        if request.is_empty() {
            return Ok(StyleTts2SynthesisOutput {
                sample_rate_hz: self.sample_rate_hz,
                pcm_mono_f32: Vec::new(),
                realized_utterance: None,
                timings: Vec::new(),
            });
        }

        let mut pcm_mono_f32 = Vec::new();
        for chunk in &request.backend_plan.chunks {
            let offset = pcm_mono_f32.len();
            pcm_mono_f32.extend(self.generate_tone(chunk.symbols.len(), offset));
        }

        Ok(StyleTts2SynthesisOutput {
            sample_rate_hz: self.sample_rate_hz,
            pcm_mono_f32,
            realized_utterance: None,
            timings: Vec::new(),
        })
    }

    fn synthesize_streaming(
        &mut self,
        request: &StyleTts2SynthesisRequest,
        sink: &mut dyn StyleTts2AudioSink,
    ) -> Result<StyleTts2SynthesisOutput, StyleTts2Error> {
        if request.is_empty() {
            return Ok(StyleTts2SynthesisOutput {
                sample_rate_hz: self.sample_rate_hz,
                pcm_mono_f32: Vec::new(),
                realized_utterance: None,
                timings: Vec::new(),
            });
        }

        let mut pcm_mono_f32 = Vec::new();
        let chunk_count = request.backend_plan.chunks.len();
        for (chunk_index, chunk) in request.backend_plan.chunks.iter().enumerate() {
            let samples = self.generate_tone(chunk.symbols.len(), pcm_mono_f32.len());
            sink.emit(StyleTts2AudioChunk {
                chunk_index,
                is_final: chunk_index + 1 == chunk_count,
                terminal: chunk.terminal,
                source_text: chunk.source_text.clone(),
                sample_rate_hz: self.sample_rate_hz,
                pcm_mono_f32: samples.clone(),
            })?;
            pcm_mono_f32.extend(samples);
        }

        Ok(StyleTts2SynthesisOutput {
            sample_rate_hz: self.sample_rate_hz,
            pcm_mono_f32,
            realized_utterance: None,
            timings: Vec::new(),
        })
    }
}

impl MockStyleTts2Backend {
    fn generate_tone(&self, token_count: usize, sample_offset: usize) -> Vec<f32> {
        let token_count = token_count.max(1);
        let samples_per_token = (self.sample_rate_hz / 50).max(1) as usize;
        let sample_count = token_count * samples_per_token;
        let period = (self.sample_rate_hz / 200).max(2) as usize;
        let mut pcm_mono_f32 = Vec::with_capacity(sample_count);

        for i in 0..sample_count {
            let phase = ((sample_offset + i) % period) as f32 / period as f32;
            let sample = ((phase * 2.0) - 1.0) * self.amplitude;
            pcm_mono_f32.push(sample);
        }

        pcm_mono_f32
    }
}
