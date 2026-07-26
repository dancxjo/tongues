//! Adapter from the native StyleTTS2 implementation to Tongues' public speech
//! orchestration contract.

use std::time::Instant;

use speaking::{SpeakerId, StyleRef, StyleSource};
use tongues_tts::{
    BackendCapabilities, NormalizedAudioChunk, NormalizedAudioSink, SpeakerSelection,
    SpeechRequest, StyleSelection, SynthesisContractError, SynthesisMetadata, SynthesisTiming,
    SynthesizerBackend, UnifiedSynthesisOutput, UnifiedSynthesisRequest, utterance_plan_from_text,
};

use crate::{
    DEFAULT_MAX_TTS_SYMBOLS, StyleTts2AudioChunk, StyleTts2Backend, StyleTts2DiffusionOptions,
    StyleTts2OnnxBackend, StyleTts2PlanOptions, StyleTts2SynthesisRequest, prepare_styletts2_plan,
    styletts2_en_us_symbol_set, validate_styletts2_plan,
};

pub struct StyleTts2Synthesizer {
    capabilities: BackendCapabilities,
    backend: StyleTts2OnnxBackend,
    plan_options: StyleTts2PlanOptions,
}

impl StyleTts2Synthesizer {
    pub fn new(capabilities: BackendCapabilities, backend: StyleTts2OnnxBackend) -> Self {
        Self {
            capabilities,
            backend,
            plan_options: StyleTts2PlanOptions {
                max_symbols_per_chunk: DEFAULT_MAX_TTS_SYMBOLS,
                chunking_enabled: true,
            },
        }
    }

    pub fn with_plan_options(mut self, plan_options: StyleTts2PlanOptions) -> Self {
        self.plan_options = plan_options;
        self
    }

    pub fn backend(&self) -> &StyleTts2OnnxBackend {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut StyleTts2OnnxBackend {
        &mut self.backend
    }
}

impl SynthesizerBackend for StyleTts2Synthesizer {
    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities.clone()
    }

    fn synthesize(
        &mut self,
        request: &UnifiedSynthesisRequest,
        sink: &mut dyn NormalizedAudioSink,
    ) -> Result<UnifiedSynthesisOutput, SynthesisContractError> {
        self.capabilities.validate(request)?;
        self.backend
            .set_speed(f64::from(request.speed))
            .map_err(style_error)?;
        if let Some(style) = request.style.as_ref() {
            self.backend
                .set_diffusion_options(StyleTts2DiffusionOptions {
                    diffusion_steps: style.diffusion_steps.unwrap_or(5),
                    alpha: style.speaker_blend.unwrap_or(0.3),
                    beta: style.style_blend.unwrap_or(0.1),
                    embedding_scale: style.embedding_scale.unwrap_or(1.0),
                    seed: request.seed.unwrap_or(0),
                })
                .map_err(style_error)?;
        } else if let Some(seed) = request.seed {
            self.backend.set_seed(seed);
        }

        let mut plan = utterance_plan_from_text(SpeechRequest {
            text: request.text.clone(),
            variety: request.variety.clone(),
        })
        .map_err(backend_error)?;
        plan.speaker = match request.speaker.as_ref() {
            Some(SpeakerSelection::Named(name)) => Some(SpeakerId(name.clone())),
            Some(SpeakerSelection::Numeric(_)) | None => None,
        };
        plan.style = request
            .style
            .as_ref()
            .map(|style| self.resolve_style_ref(style, request.reference_audio.style.as_deref()))
            .transpose()?;

        let backend_plan = prepare_styletts2_plan(
            &plan,
            &styletts2_en_us_symbol_set(),
            StyleTts2PlanOptions {
                max_symbols_per_chunk: request
                    .max_chunk_symbols
                    .unwrap_or(self.plan_options.max_symbols_per_chunk),
                chunking_enabled: request.chunking && self.plan_options.chunking_enabled,
            },
        )
        .map_err(style_error)?;
        validate_styletts2_plan(&backend_plan).map_err(style_error)?;
        let mut backend_request = StyleTts2SynthesisRequest::from_backend_plan(
            backend_plan,
            plan.speaker.clone(),
            plan.style.clone(),
            plan.target_prosody.clone(),
        );
        if let Some(uri) = request.reference_audio.speaker.as_ref() {
            backend_request = backend_request.with_speaker_reference_audio_uri(uri);
        }
        if let Some(uri) = request.reference_audio.style.as_ref() {
            backend_request = backend_request.with_style_reference_audio_uri(uri);
        }

        let started = Instant::now();
        let mut frames = 0_u64;
        let mut frame_offset = 0_u64;
        let mut sink_failure = None;
        let synthesis_result = self.backend.synthesize_streaming(
            &backend_request,
            &mut |chunk: StyleTts2AudioChunk| {
                let chunk_frames = chunk.pcm_mono_f32.len() as u64;
                if let Err(error) = sink.emit(NormalizedAudioChunk {
                    chunk_index: chunk.chunk_index,
                    is_final: chunk.is_final,
                    frame_offset,
                    sample_rate_hz: chunk.sample_rate_hz,
                    channels: 1,
                    pcm_f32: chunk.pcm_mono_f32,
                }) {
                    let message = error.to_string();
                    sink_failure = Some(error);
                    return Err(crate::StyleTts2Error::Backend { message });
                }
                frames += chunk_frames;
                frame_offset += chunk_frames;
                Ok(())
            },
        );
        if let Some(error) = sink_failure {
            return Err(error);
        }
        let output = synthesis_result.map_err(style_error)?;
        let mut timings = output
            .timings
            .into_iter()
            .map(|timing| SynthesisTiming {
                stage: timing.stage,
                elapsed_ms: timing.elapsed_ms,
            })
            .collect::<Vec<_>>();
        if !timings.iter().any(|timing| timing.stage == "total") {
            timings.push(SynthesisTiming {
                stage: "total".into(),
                elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
            });
        }
        Ok(UnifiedSynthesisOutput {
            metadata: SynthesisMetadata {
                backend: self.capabilities.backend.clone(),
                model: self.capabilities.model.clone(),
                sample_rate_hz: output.sample_rate_hz,
                channels: 1,
                frames,
                audio_seconds: frames as f64 / f64::from(output.sample_rate_hz),
                streaming: request.streaming,
                input_audio: Vec::new(),
                timings,
            },
        })
    }
}

impl StyleTts2Synthesizer {
    fn resolve_style_ref(
        &mut self,
        style: &StyleSelection,
        reference_uri: Option<&str>,
    ) -> Result<StyleRef, SynthesisContractError> {
        let embedding = if style.embedding_is_delta {
            let uri = reference_uri.ok_or_else(|| SynthesisContractError::InvalidRequest {
                field: "reference_audio.style",
                reason: "is required when style.embedding_is_delta is true".into(),
            })?;
            let mut values = self
                .backend
                .reference_style_vector(uri)
                .map_err(style_error)?;
            let delta = style
                .embedding
                .as_ref()
                .expect("validated style delta has an embedding");
            for (value, delta) in values.iter_mut().zip(delta) {
                *value += delta * style.strength;
            }
            Some(values)
        } else {
            style.embedding.clone()
        };
        let source = if let Some(values) = embedding {
            StyleSource::Embedding {
                kind: style
                    .name
                    .clone()
                    .unwrap_or_else(|| "styletts2.embedding.v1".into()),
                values,
            }
        } else {
            StyleSource::Manual
        };
        Ok(StyleRef {
            description: style.name.clone(),
            source,
        })
    }
}

fn backend_error(error: impl std::fmt::Display) -> SynthesisContractError {
    SynthesisContractError::Backend {
        message: error.to_string(),
    }
}

fn style_error(error: crate::StyleTts2Error) -> SynthesisContractError {
    SynthesisContractError::Backend {
        message: error.to_string(),
    }
}
