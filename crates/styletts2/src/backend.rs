use speech::{TerminalPunctuation, Utterance};
use thiserror::Error;

use crate::config::StyleTts2ConfigError;
use crate::request::StyleTts2SynthesisRequest;
use crate::symbols::SymbolLoweringError;

pub trait StyleTts2Backend {
    fn synthesize(
        &mut self,
        request: &StyleTts2SynthesisRequest,
    ) -> Result<StyleTts2SynthesisOutput, StyleTts2Error>;

    fn synthesize_streaming(
        &mut self,
        request: &StyleTts2SynthesisRequest,
        sink: &mut dyn StyleTts2AudioSink,
    ) -> Result<StyleTts2SynthesisOutput, StyleTts2Error> {
        let output = self.synthesize(request)?;
        if !output.pcm_mono_f32.is_empty() {
            sink.emit(StyleTts2AudioChunk {
                chunk_index: 0,
                is_final: true,
                terminal: request
                    .backend_plan
                    .chunks
                    .last()
                    .and_then(|chunk| chunk.terminal),
                source_text: request.backend_plan.text.clone(),
                sample_rate_hz: output.sample_rate_hz,
                pcm_mono_f32: output.pcm_mono_f32.clone(),
            })?;
        }
        Ok(output)
    }
}

pub trait StyleTts2AudioSink {
    fn emit(&mut self, chunk: StyleTts2AudioChunk) -> Result<(), StyleTts2Error>;
}

impl<F> StyleTts2AudioSink for F
where
    F: FnMut(StyleTts2AudioChunk) -> Result<(), StyleTts2Error>,
{
    fn emit(&mut self, chunk: StyleTts2AudioChunk) -> Result<(), StyleTts2Error> {
        self(chunk)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StyleTts2SynthesisOutput {
    pub sample_rate_hz: u32,
    pub pcm_mono_f32: Vec<f32>,
    pub realized_utterance: Option<Utterance>,
    pub timings: Vec<StyleTts2Timing>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StyleTts2AudioChunk {
    pub chunk_index: usize,
    pub is_final: bool,
    pub terminal: Option<TerminalPunctuation>,
    pub source_text: Option<String>,
    pub sample_rate_hz: u32,
    pub pcm_mono_f32: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StyleTts2Timing {
    pub stage: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Error)]
pub enum StyleTts2Error {
    #[error(transparent)]
    Config(#[from] StyleTts2ConfigError),
    #[error(transparent)]
    SymbolLowering(#[from] SymbolLoweringError),
    #[error("StyleTTS2 backend feature `{feature}` is not enabled")]
    BackendFeatureDisabled { feature: &'static str },
    #[error("StyleTTS2 backend failed: {message}")]
    Backend { message: String },
    #[error("StyleTTS2 backend returned invalid output: {reason}")]
    InvalidOutput { reason: String },
}
