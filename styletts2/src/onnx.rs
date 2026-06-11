use std::collections::HashMap;
use std::f32::consts::TAU;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use ort::ep::ExecutionProviderDispatch;
use ort::memory::Allocator;
use ort::session::{
    IoBinding, Session,
    builder::{GraphOptimizationLevel, SessionBuilder},
};
use ort::value::{DynTensorValueType, Tensor};
use speech::{StyleRef, StyleSource};

use crate::backend::{
    StyleTts2AudioChunk, StyleTts2AudioSink, StyleTts2Backend, StyleTts2Error,
    StyleTts2SynthesisOutput, StyleTts2Timing,
};
#[cfg(test)]
use crate::plan::{styletts2_character_id, styletts2_text_for_symbol};
use crate::plan::{styletts2_token_ids_for_symbols, validate_styletts2_plan};
use crate::request::StyleTts2SynthesisRequest;
#[cfg(test)]
use crate::symbols::StyleTts2SymbolSequence;

const SAMPLE_RATE_HZ: u32 = 24_000;
const DIFFUSION_ONNX: &str =
    "14b6dd78237d223f172f8af702ed8aeb4a2c51fd0ff7e3ca03a4967d33fa13bc.onnx";
const STYLE_ENCODER_ONNX: &str =
    "4612a9dc0c0e142468f361e8e901bdccfdca45a2ae1145e5452bc98c7915302d.onnx";
const TEXT_ENCODER_ONNX: &str =
    "91473db52725b0c3b8387537979a2f42f0da82836e50902503a877c610864ad6.onnx";
const DECODER_ONNX: &str = "99e40b35027e96a247c8e1f359d2f99d3cd6e93afec2e0f4a15f72dd7b79d457.onnx";
const STYLE_VECTOR_DIMS: usize = 256;
const STYLE_HALF_DIMS: usize = STYLE_VECTOR_DIMS / 2;
const HIDDEN_DIMS: i64 = 768;
const MIN_REFERENCE_SAMPLES: usize = SAMPLE_RATE_HZ as usize;
const MAX_REFERENCE_SAMPLES: usize = SAMPLE_RATE_HZ as usize * 8;
pub const STYLETTS2_ONNX_INTRA_THREADS_ENV: &str = "STYLETTS2_ONNX_INTRA_THREADS";
pub const STYLETTS2_ONNX_INTER_THREADS_ENV: &str = "STYLETTS2_ONNX_INTER_THREADS";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StyleTts2OnnxPaths {
    pub diffusion: PathBuf,
    pub style_encoder: PathBuf,
    pub text_encoder: PathBuf,
    pub decoder: PathBuf,
}

impl StyleTts2OnnxPaths {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Self {
        let model_dir = model_dir.as_ref();
        Self {
            diffusion: model_dir.join(DIFFUSION_ONNX),
            style_encoder: model_dir.join(STYLE_ENCODER_ONNX),
            text_encoder: model_dir.join(TEXT_ENCODER_ONNX),
            decoder: model_dir.join(DECODER_ONNX),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StyleTts2DiffusionOptions {
    pub diffusion_steps: usize,
    pub alpha: f32,
    pub beta: f32,
    pub embedding_scale: f64,
    pub seed: u64,
}

impl Default for StyleTts2DiffusionOptions {
    fn default() -> Self {
        Self {
            diffusion_steps: 5,
            alpha: 0.3,
            beta: 0.1,
            embedding_scale: 1.0,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StyleTts2OnnxOptimization {
    Generation,
    Deterministic,
}

impl Default for StyleTts2OnnxOptimization {
    fn default() -> Self {
        Self::Generation
    }
}

impl StyleTts2OnnxOptimization {
    fn graph_optimization_level(self) -> GraphOptimizationLevel {
        match self {
            Self::Generation => GraphOptimizationLevel::Level3,
            Self::Deterministic => GraphOptimizationLevel::Disable,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StyleTts2OnnxOptions {
    pub optimization: StyleTts2OnnxOptimization,
    pub intra_threads: usize,
    pub inter_threads: usize,
}

impl Default for StyleTts2OnnxOptions {
    fn default() -> Self {
        Self {
            optimization: StyleTts2OnnxOptimization::Generation,
            intra_threads: 1,
            inter_threads: 1,
        }
    }
}

impl StyleTts2OnnxOptions {
    pub fn from_env() -> Result<Self, StyleTts2Error> {
        let mut options = Self::default();
        if let Some(intra_threads) = read_thread_count_env(STYLETTS2_ONNX_INTRA_THREADS_ENV)? {
            options.intra_threads = intra_threads;
        }
        if let Some(inter_threads) = read_thread_count_env(STYLETTS2_ONNX_INTER_THREADS_ENV)? {
            options.inter_threads = inter_threads;
        }
        Ok(options)
    }

    pub fn with_optimization(mut self, optimization: StyleTts2OnnxOptimization) -> Self {
        self.optimization = optimization;
        self
    }

    pub fn with_intra_threads(mut self, intra_threads: usize) -> Self {
        self.intra_threads = intra_threads;
        self
    }

    pub fn with_inter_threads(mut self, inter_threads: usize) -> Self {
        self.inter_threads = inter_threads;
        self
    }
}

pub struct StyleTts2OnnxBackend {
    diffusion: Session,
    style_encoder: Session,
    text_encoder: Session,
    decoder: Session,
    style_vector: Vec<f32>,
    speaker_reference_audio_uri: Option<String>,
    style_reference_audio_uri: Option<String>,
    reference_cache: HashMap<String, Vec<f32>>,
    diffusion_options: StyleTts2DiffusionOptions,
    speed: f64,
}

impl StyleTts2OnnxBackend {
    pub fn load(paths: StyleTts2OnnxPaths) -> Result<Self, StyleTts2Error> {
        Self::load_with_optimization(paths, StyleTts2OnnxOptimization::Generation)
    }

    pub fn load_deterministic(paths: StyleTts2OnnxPaths) -> Result<Self, StyleTts2Error> {
        Self::load_with_optimization(paths, StyleTts2OnnxOptimization::Deterministic)
    }

    pub fn load_with_optimization(
        paths: StyleTts2OnnxPaths,
        optimization: StyleTts2OnnxOptimization,
    ) -> Result<Self, StyleTts2Error> {
        let options = StyleTts2OnnxOptions::from_env()?.with_optimization(optimization);
        Self::load_with_options(paths, options)
    }

    pub fn load_with_options(
        paths: StyleTts2OnnxPaths,
        options: StyleTts2OnnxOptions,
    ) -> Result<Self, StyleTts2Error> {
        validate_thread_count("intra-op", options.intra_threads)?;
        validate_thread_count("inter-op", options.inter_threads)?;
        ensure_file(&paths.diffusion, "StyleTTS2 diffusion denoiser")?;
        ensure_file(&paths.style_encoder, "StyleTTS2 style encoder")?;
        ensure_file(&paths.text_encoder, "StyleTTS2 text encoder")?;
        ensure_file(&paths.decoder, "StyleTTS2 decoder")?;
        initialize_ort_runtime()?;

        Ok(Self {
            diffusion: load_session(&paths.diffusion, "StyleTTS2 diffusion denoiser", options)?,
            style_encoder: load_session(&paths.style_encoder, "StyleTTS2 style encoder", options)?,
            text_encoder: load_session(&paths.text_encoder, "StyleTTS2 text encoder", options)?,
            decoder: load_session(&paths.decoder, "StyleTTS2 decoder", options)?,
            style_vector: vec![0.0; STYLE_VECTOR_DIMS],
            speaker_reference_audio_uri: None,
            style_reference_audio_uri: None,
            reference_cache: HashMap::new(),
            diffusion_options: StyleTts2DiffusionOptions::default(),
            speed: 1.0,
        })
    }

    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self, StyleTts2Error> {
        Self::load(StyleTts2OnnxPaths::from_model_dir(model_dir))
    }

    pub fn from_model_dir_deterministic(
        model_dir: impl AsRef<Path>,
    ) -> Result<Self, StyleTts2Error> {
        Self::load_deterministic(StyleTts2OnnxPaths::from_model_dir(model_dir))
    }

    pub fn from_model_dir_with_optimization(
        model_dir: impl AsRef<Path>,
        optimization: StyleTts2OnnxOptimization,
    ) -> Result<Self, StyleTts2Error> {
        Self::load_with_optimization(StyleTts2OnnxPaths::from_model_dir(model_dir), optimization)
    }

    pub fn from_model_dir_with_options(
        model_dir: impl AsRef<Path>,
        options: StyleTts2OnnxOptions,
    ) -> Result<Self, StyleTts2Error> {
        Self::load_with_options(StyleTts2OnnxPaths::from_model_dir(model_dir), options)
    }

    pub fn with_style_vector(mut self, style_vector: Vec<f32>) -> Result<Self, StyleTts2Error> {
        if style_vector.len() != STYLE_VECTOR_DIMS {
            return Err(invalid_output(format!(
                "StyleTTS2 style vector must have {STYLE_VECTOR_DIMS} values, got {}",
                style_vector.len()
            )));
        }
        if !style_vector.iter().all(|value| value.is_finite()) {
            return Err(invalid_output(
                "StyleTTS2 style vector contains non-finite values",
            ));
        }
        self.style_vector = style_vector;
        Ok(self)
    }

    pub fn with_speaker_reference_audio_uri(mut self, uri: impl Into<String>) -> Self {
        self.speaker_reference_audio_uri = Some(uri.into());
        self
    }

    pub fn with_style_reference_audio_uri(mut self, uri: impl Into<String>) -> Self {
        self.style_reference_audio_uri = Some(uri.into());
        self
    }

    pub fn with_reference_audio_uri(mut self, uri: impl Into<String>) -> Self {
        let uri = uri.into();
        self.speaker_reference_audio_uri = Some(uri.clone());
        self.style_reference_audio_uri = Some(uri);
        self
    }

    pub fn with_diffusion_options(
        mut self,
        options: StyleTts2DiffusionOptions,
    ) -> Result<Self, StyleTts2Error> {
        validate_diffusion_options(&options)?;
        self.diffusion_options = options;
        Ok(self)
    }

    pub fn with_speed(mut self, speed: f64) -> Result<Self, StyleTts2Error> {
        if !speed.is_finite() || speed <= 0.0 {
            return Err(invalid_output(format!(
                "StyleTTS2 speed must be finite and positive, got {speed}"
            )));
        }
        self.speed = speed;
        Ok(self)
    }
}

impl StyleTts2Backend for StyleTts2OnnxBackend {
    fn synthesize(
        &mut self,
        request: &StyleTts2SynthesisRequest,
    ) -> Result<StyleTts2SynthesisOutput, StyleTts2Error> {
        let total_started = Instant::now();
        let preflight_started = Instant::now();
        self.preflight_request(request)?;
        let mut timings = vec![timing("preflight", preflight_started)];
        if request.is_empty() {
            timings.push(timing("total", total_started));
            return Ok(StyleTts2SynthesisOutput {
                sample_rate_hz: SAMPLE_RATE_HZ,
                pcm_mono_f32: Vec::new(),
                realized_utterance: None,
                timings,
            });
        }

        let mut pcm_mono_f32 = Vec::new();
        for (index, chunk) in request.backend_plan.chunks.iter().enumerate() {
            let chunk_started = Instant::now();
            let token_ids = styletts2_token_ids_for_symbols(&chunk.symbols)?;
            if token_ids.is_empty() {
                continue;
            }
            let output = self.synthesize_token_ids(request, token_ids)?;
            pcm_mono_f32.extend(output.pcm_mono_f32);
            let chunk_prefix = format!("chunk_{}", index + 1);
            timings.extend(
                output
                    .timings
                    .into_iter()
                    .map(|timing| prefix_timing(&chunk_prefix, timing)),
            );
            timings.push(timing(&format!("{chunk_prefix}.total"), chunk_started));
        }
        timings.push(timing("total", total_started));

        Ok(StyleTts2SynthesisOutput {
            sample_rate_hz: SAMPLE_RATE_HZ,
            pcm_mono_f32,
            realized_utterance: None,
            timings,
        })
    }

    fn synthesize_streaming(
        &mut self,
        request: &StyleTts2SynthesisRequest,
        sink: &mut dyn StyleTts2AudioSink,
    ) -> Result<StyleTts2SynthesisOutput, StyleTts2Error> {
        let total_started = Instant::now();
        let preflight_started = Instant::now();
        self.preflight_request(request)?;
        let mut timings = vec![timing("preflight", preflight_started)];
        if request.is_empty() {
            timings.push(timing("total", total_started));
            return Ok(StyleTts2SynthesisOutput {
                sample_rate_hz: SAMPLE_RATE_HZ,
                pcm_mono_f32: Vec::new(),
                realized_utterance: None,
                timings,
            });
        }

        let mut pcm_mono_f32 = Vec::new();
        let chunk_count = request.backend_plan.chunks.len();
        for (index, chunk) in request.backend_plan.chunks.iter().enumerate() {
            let chunk_started = Instant::now();
            let token_ids = styletts2_token_ids_for_symbols(&chunk.symbols)?;
            if token_ids.is_empty() {
                continue;
            }
            let output = self.synthesize_token_ids(request, token_ids)?;
            sink.emit(StyleTts2AudioChunk {
                chunk_index: index,
                is_final: index + 1 == chunk_count,
                terminal: chunk.terminal,
                source_text: chunk.source_text.clone(),
                sample_rate_hz: SAMPLE_RATE_HZ,
                pcm_mono_f32: output.pcm_mono_f32.clone(),
            })?;
            pcm_mono_f32.extend(output.pcm_mono_f32);
            let chunk_prefix = format!("chunk_{}", index + 1);
            timings.extend(
                output
                    .timings
                    .into_iter()
                    .map(|timing| prefix_timing(&chunk_prefix, timing)),
            );
            timings.push(timing(&format!("{chunk_prefix}.total"), chunk_started));
        }
        timings.push(timing("total", total_started));

        Ok(StyleTts2SynthesisOutput {
            sample_rate_hz: SAMPLE_RATE_HZ,
            pcm_mono_f32,
            realized_utterance: None,
            timings,
        })
    }
}

struct OnnxChunkSynthesisOutput {
    pcm_mono_f32: Vec<f32>,
    timings: Vec<StyleTts2Timing>,
}

impl StyleTts2OnnxBackend {
    fn synthesize_token_ids(
        &mut self,
        request: &StyleTts2SynthesisRequest,
        token_ids: Vec<i64>,
    ) -> Result<OnnxChunkSynthesisOutput, StyleTts2Error> {
        let token_len = i64::try_from(token_ids.len())
            .map_err(|_| invalid_output("StyleTTS2 token sequence is too long"))?;
        let mut timings = Vec::new();
        let text_encoder_started = Instant::now();
        let encoder_input = Tensor::from_array((vec![1_i64, token_len], token_ids.clone()))
            .map_err(|error| backend_error(format!("failed to build text encoder input: {error}")))?
            .upcast();
        let encoder_outputs = self
            .text_encoder
            .run(vec![("a".to_string(), encoder_input)])
            .map_err(|error| backend_error(format!("StyleTTS2 text encoder failed: {error}")))?;
        let (encoder_shape, encoder_values) = extract_f32_tensor(&encoder_outputs, "z")?;
        if encoder_shape.len() != 3 || encoder_shape[0] != 1 || encoder_shape[2] != HIDDEN_DIMS {
            return Err(invalid_output(format!(
                "StyleTTS2 text encoder returned unexpected shape {encoder_shape:?}"
            )));
        }
        drop(encoder_outputs);
        timings.push(timing("text_encoder", text_encoder_started));

        let style_started = Instant::now();
        let style_vector = self.resolve_style_vector(request, &encoder_shape, &encoder_values)?;
        timings.push(timing("style", style_started));

        let decoder_started = Instant::now();
        let decoder_tokens = Tensor::from_array((vec![1_i64, token_len], token_ids))
            .map_err(|error| {
                backend_error(format!("failed to build decoder token input: {error}"))
            })?
            .upcast();
        let decoder_hidden = Tensor::from_array((encoder_shape, encoder_values))
            .map_err(|error| {
                backend_error(format!("failed to build decoder hidden input: {error}"))
            })?
            .upcast();
        let style = Tensor::from_array((vec![1_i64, STYLE_VECTOR_DIMS as i64], style_vector))
            .map_err(|error| {
                backend_error(format!("failed to build decoder style input: {error}"))
            })?
            .upcast();
        let speed = Tensor::from_array((Vec::<i64>::new(), vec![self.speed]))
            .map_err(|error| {
                backend_error(format!("failed to build decoder speed input: {error}"))
            })?
            .upcast();

        let decoder_outputs = self
            .decoder
            .run(vec![
                ("a".to_string(), decoder_tokens),
                ("b".to_string(), decoder_hidden),
                ("c".to_string(), style),
                ("d".to_string(), speed),
            ])
            .map_err(|error| backend_error(format!("StyleTTS2 decoder failed: {error}")))?;
        let (_, samples) = extract_f32_tensor(&decoder_outputs, "z")?;
        if samples.is_empty() {
            return Err(invalid_output(
                "StyleTTS2 decoder returned an empty waveform",
            ));
        }
        timings.push(timing("decoder", decoder_started));

        Ok(OnnxChunkSynthesisOutput {
            pcm_mono_f32: samples,
            timings,
        })
    }

    fn preflight_request(&self, request: &StyleTts2SynthesisRequest) -> Result<(), StyleTts2Error> {
        validate_styletts2_plan(&request.backend_plan)?;
        validate_diffusion_options(&self.diffusion_options)?;

        let speaker_uri = request
            .speaker_reference_audio_uri
            .as_deref()
            .or(self.speaker_reference_audio_uri.as_deref());
        let style_uri = request
            .style_reference_audio_uri
            .as_deref()
            .or(self.style_reference_audio_uri.as_deref())
            .or_else(|| reference_audio_uri_from_style(request.style.as_ref()));

        if let Some(uri) = speaker_uri {
            ensure_reference_wav_readable(uri)?;
        }
        if let Some(uri) = style_uri
            && Some(uri) != speaker_uri
        {
            ensure_reference_wav_readable(uri)?;
        }
        Ok(())
    }

    fn resolve_style_vector(
        &mut self,
        request: &StyleTts2SynthesisRequest,
        text_embedding_shape: &[i64],
        text_embedding: &[f32],
    ) -> Result<Vec<f32>, StyleTts2Error> {
        let speaker_uri = request
            .speaker_reference_audio_uri
            .as_deref()
            .or(self.speaker_reference_audio_uri.as_deref())
            .map(str::to_string);
        let style_uri = request
            .style_reference_audio_uri
            .as_deref()
            .or(self.style_reference_audio_uri.as_deref())
            .or_else(|| reference_audio_uri_from_style(request.style.as_ref()))
            .map(str::to_string);

        let reference_features = match (speaker_uri.as_deref(), style_uri.as_deref()) {
            (Some(speaker_uri), Some(style_uri)) => {
                let speaker = self.reference_style_vector(speaker_uri)?;
                let style = self.reference_style_vector(style_uri)?;
                merge_speaker_and_style_vectors(&speaker, &style)
            }
            (Some(uri), None) | (None, Some(uri)) => self.reference_style_vector(uri)?,
            (None, None) => self.style_vector.clone(),
        };

        if !should_sample_diffusion_style(&self.diffusion_options) {
            return Ok(reference_features);
        }

        let predicted =
            self.sample_diffusion_style(text_embedding_shape, text_embedding, &reference_features)?;
        Ok(blend_predicted_and_reference_style(
            &predicted,
            &reference_features,
            self.diffusion_options.alpha,
            self.diffusion_options.beta,
        ))
    }

    fn reference_style_vector(&mut self, uri: &str) -> Result<Vec<f32>, StyleTts2Error> {
        if let Some(vector) = self.reference_cache.get(uri) {
            return Ok(vector.clone());
        }

        let path = reference_audio_path(uri)?;
        let audio = read_reference_wav_mono_24khz(&path)?;
        let sample_len = i64::try_from(audio.len())
            .map_err(|_| invalid_output("StyleTTS2 reference audio is too long"))?;
        let input = Tensor::from_array((vec![1_i64, sample_len], audio))
            .map_err(|error| {
                backend_error(format!(
                    "failed to build style encoder audio input: {error}"
                ))
            })?
            .upcast();
        let outputs = self
            .style_encoder
            .run(vec![("a".to_string(), input)])
            .map_err(|error| backend_error(format!("StyleTTS2 style encoder failed: {error}")))?;
        let (shape, values) = extract_f32_tensor(&outputs, "z")?;
        if shape != [1, STYLE_VECTOR_DIMS as i64] {
            return Err(invalid_output(format!(
                "StyleTTS2 style encoder returned unexpected shape {shape:?}"
            )));
        }

        self.reference_cache.insert(uri.to_string(), values.clone());
        Ok(values)
    }

    fn sample_diffusion_style(
        &mut self,
        text_embedding_shape: &[i64],
        text_embedding: &[f32],
        reference_features: &[f32],
    ) -> Result<Vec<f32>, StyleTts2Error> {
        validate_style_vector(reference_features)?;
        let options = self.diffusion_options.clone();
        validate_diffusion_options(&options)?;
        let mut diffusion_binding = DiffusionIoBinding::new(
            &self.diffusion,
            text_embedding_shape,
            text_embedding,
            reference_features,
            options.embedding_scale,
        )?;
        let sigmas = karras_sigmas(options.diffusion_steps, 0.0001, 3.0, 9.0);
        let mut rng = DeterministicRng::new(options.seed);
        let mut x = gaussian_vec(&mut rng, STYLE_VECTOR_DIMS)
            .into_iter()
            .map(|value| value * sigmas[0])
            .collect::<Vec<_>>();

        for index in 0..options.diffusion_steps.saturating_sub(1) {
            let sigma = sigmas[index];
            let sigma_next = sigmas[index + 1];
            x = self.adpm2_step(x, sigma, sigma_next, &mut diffusion_binding, &mut rng)?;
        }

        validate_style_vector(&x)?;
        Ok(x)
    }

    #[allow(clippy::too_many_arguments)]
    fn adpm2_step(
        &mut self,
        x: Vec<f32>,
        sigma: f32,
        sigma_next: f32,
        diffusion_binding: &mut DiffusionIoBinding,
        rng: &mut DeterministicRng,
    ) -> Result<Vec<f32>, StyleTts2Error> {
        let sigma_up = (sigma_next.powi(2) * (sigma.powi(2) - sigma_next.powi(2)) / sigma.powi(2))
            .max(0.0)
            .sqrt();
        let sigma_down = (sigma_next.powi(2) - sigma_up.powi(2)).max(0.0).sqrt();
        let sigma_mid = (sigma + sigma_down) * 0.5;
        let denoised = self.diffusion_denoise(&x, sigma, diffusion_binding)?;
        let derivative = diffusion_derivative(&x, &denoised, sigma)?;
        let midpoint = add_scaled(&x, &derivative, sigma_mid - sigma);
        let denoised_mid = self.diffusion_denoise(&midpoint, sigma_mid, diffusion_binding)?;
        let derivative_mid = diffusion_derivative(&midpoint, &denoised_mid, sigma_mid)?;
        let mut next = add_scaled(&x, &derivative_mid, sigma_down - sigma);
        if sigma_up > 0.0 {
            for (value, noise) in next.iter_mut().zip(gaussian_vec(rng, STYLE_VECTOR_DIMS)) {
                *value += noise * sigma_up;
            }
        }
        Ok(next)
    }

    fn diffusion_denoise(
        &mut self,
        x: &[f32],
        sigma: f32,
        diffusion_binding: &mut DiffusionIoBinding,
    ) -> Result<Vec<f32>, StyleTts2Error> {
        validate_style_vector(x)?;
        if sigma <= 0.0 || !sigma.is_finite() {
            return Err(invalid_output(format!(
                "StyleTTS2 diffusion sigma must be finite and positive, got {sigma}"
            )));
        }

        let noise = Tensor::from_array((vec![1_i64, 1, STYLE_VECTOR_DIMS as i64], x.to_vec()))
            .map_err(|error| {
                backend_error(format!("failed to build diffusion noise input: {error}"))
            })?;
        let sigma = Tensor::from_array((vec![1_i64], vec![sigma])).map_err(|error| {
            backend_error(format!("failed to build diffusion sigma input: {error}"))
        })?;
        diffusion_binding
            .binding
            .bind_input("a", &noise)
            .map_err(|error| {
                backend_error(format!("failed to bind diffusion noise input: {error}"))
            })?;
        diffusion_binding
            .binding
            .bind_input("b", &sigma)
            .map_err(|error| {
                backend_error(format!("failed to bind diffusion sigma input: {error}"))
            })?;
        let outputs = self
            .diffusion
            .run_binding(&diffusion_binding.binding)
            .map_err(|error| {
                backend_error(format!("StyleTTS2 diffusion denoiser failed: {error}"))
            })?;
        let (shape, values) = extract_f32_tensor(&outputs, "z")?;
        if shape != [1, 1, STYLE_VECTOR_DIMS as i64] {
            return Err(invalid_output(format!(
                "StyleTTS2 diffusion denoiser returned unexpected shape {shape:?}"
            )));
        }
        Ok(values)
    }
}

struct DiffusionIoBinding {
    binding: IoBinding,
}

impl DiffusionIoBinding {
    fn new(
        diffusion: &Session,
        text_embedding_shape: &[i64],
        text_embedding: &[f32],
        reference_features: &[f32],
        embedding_scale: f64,
    ) -> Result<Self, StyleTts2Error> {
        validate_style_vector(reference_features)?;
        let text_embedding =
            Tensor::from_array((text_embedding_shape.to_vec(), text_embedding.to_vec())).map_err(
                |error| {
                    backend_error(format!(
                        "failed to build diffusion text embedding input: {error}"
                    ))
                },
            )?;
        let embedding_scale = Tensor::from_array((Vec::<i64>::new(), vec![embedding_scale]))
            .map_err(|error| {
                backend_error(format!(
                    "failed to build diffusion embedding scale input: {error}"
                ))
            })?;
        let reference_features = Tensor::from_array((
            vec![1_i64, STYLE_VECTOR_DIMS as i64],
            reference_features.to_vec(),
        ))
        .map_err(|error| {
            backend_error(format!(
                "failed to build diffusion reference feature input: {error}"
            ))
        })?;
        let output = Tensor::<f32>::new(&Allocator::default(), [1_usize, 1, STYLE_VECTOR_DIMS])
            .map_err(|error| {
                backend_error(format!(
                    "failed to build diffusion denoiser output buffer: {error}"
                ))
            })?;
        let mut binding = diffusion.create_binding().map_err(|error| {
            backend_error(format!("failed to create diffusion I/O binding: {error}"))
        })?;
        binding.bind_input("c", &text_embedding).map_err(|error| {
            backend_error(format!(
                "failed to bind diffusion text embedding input: {error}"
            ))
        })?;
        binding.bind_input("d", &embedding_scale).map_err(|error| {
            backend_error(format!(
                "failed to bind diffusion embedding scale input: {error}"
            ))
        })?;
        binding
            .bind_input("e", &reference_features)
            .map_err(|error| {
                backend_error(format!(
                    "failed to bind diffusion reference feature input: {error}"
                ))
            })?;
        binding.bind_output("z", output).map_err(|error| {
            backend_error(format!("failed to bind diffusion denoiser output: {error}"))
        })?;

        Ok(Self { binding })
    }
}

#[cfg(test)]
fn styletts2_token_ids(sequence: &StyleTts2SymbolSequence) -> Result<Vec<i64>, StyleTts2Error> {
    styletts2_token_ids_for_symbols(&sequence.tokens)
}

fn reference_audio_uri_from_style(style: Option<&StyleRef>) -> Option<&str> {
    match style.map(|style| &style.source) {
        Some(StyleSource::ReferenceAudio { uri }) => Some(uri.as_str()),
        _ => None,
    }
}

fn should_sample_diffusion_style(options: &StyleTts2DiffusionOptions) -> bool {
    options.alpha != 0.0 || options.beta != 0.0
}

fn merge_speaker_and_style_vectors(speaker: &[f32], style: &[f32]) -> Vec<f32> {
    let mut merged = Vec::with_capacity(STYLE_VECTOR_DIMS);
    merged.extend_from_slice(&speaker[..STYLE_HALF_DIMS]);
    merged.extend_from_slice(&style[STYLE_HALF_DIMS..STYLE_VECTOR_DIMS]);
    merged
}

fn blend_predicted_and_reference_style(
    predicted: &[f32],
    reference: &[f32],
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let mut style = Vec::with_capacity(STYLE_VECTOR_DIMS);
    for index in 0..STYLE_HALF_DIMS {
        style.push(alpha * predicted[index] + (1.0 - alpha) * reference[index]);
    }
    for index in STYLE_HALF_DIMS..STYLE_VECTOR_DIMS {
        style.push(beta * predicted[index] + (1.0 - beta) * reference[index]);
    }
    style
}

fn validate_style_vector(style_vector: &[f32]) -> Result<(), StyleTts2Error> {
    if style_vector.len() != STYLE_VECTOR_DIMS {
        return Err(invalid_output(format!(
            "StyleTTS2 style vector must have {STYLE_VECTOR_DIMS} values, got {}",
            style_vector.len()
        )));
    }
    if !style_vector.iter().all(|value| value.is_finite()) {
        return Err(invalid_output(
            "StyleTTS2 style vector contains non-finite values",
        ));
    }
    Ok(())
}

fn validate_diffusion_options(options: &StyleTts2DiffusionOptions) -> Result<(), StyleTts2Error> {
    if options.diffusion_steps < 2 {
        return Err(invalid_output(format!(
            "StyleTTS2 diffusion_steps must be at least 2, got {}",
            options.diffusion_steps
        )));
    }
    for (name, value) in [("alpha", options.alpha), ("beta", options.beta)] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(invalid_output(format!(
                "StyleTTS2 diffusion {name} must be in 0..=1, got {value}"
            )));
        }
    }
    if !options.embedding_scale.is_finite() || options.embedding_scale <= 0.0 {
        return Err(invalid_output(format!(
            "StyleTTS2 diffusion embedding_scale must be finite and positive, got {}",
            options.embedding_scale
        )));
    }
    Ok(())
}

fn karras_sigmas(num_steps: usize, sigma_min: f32, sigma_max: f32, rho: f32) -> Vec<f32> {
    let rho_inv = 1.0 / rho;
    let max = sigma_max.powf(rho_inv);
    let min = sigma_min.powf(rho_inv);
    let mut sigmas = (0..num_steps)
        .map(|step| {
            let ramp = step as f32 / (num_steps - 1) as f32;
            (max + ramp * (min - max)).powf(rho)
        })
        .collect::<Vec<_>>();
    sigmas.push(0.0);
    sigmas
}

fn diffusion_derivative(
    x: &[f32],
    denoised: &[f32],
    sigma: f32,
) -> Result<Vec<f32>, StyleTts2Error> {
    if sigma <= 0.0 || !sigma.is_finite() {
        return Err(invalid_output(format!(
            "StyleTTS2 diffusion derivative sigma must be finite and positive, got {sigma}"
        )));
    }
    Ok(x.iter()
        .zip(denoised)
        .map(|(x, denoised)| (x - denoised) / sigma)
        .collect())
}

fn add_scaled(x: &[f32], derivative: &[f32], scale: f32) -> Vec<f32> {
    x.iter()
        .zip(derivative)
        .map(|(x, derivative)| x + derivative * scale)
        .collect()
}

fn gaussian_vec(rng: &mut DeterministicRng, len: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(len);
    while values.len() < len {
        let u1 = rng.next_f32().max(f32::MIN_POSITIVE);
        let u2 = rng.next_f32();
        let radius = (-2.0 * u1.ln()).sqrt();
        values.push(radius * (TAU * u2).cos());
        if values.len() < len {
            values.push(radius * (TAU * u2).sin());
        }
    }
    values
}

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.state >> 40) as u32;
        (bits as f32 + 0.5) / ((1u32 << 24) as f32)
    }
}

fn reference_audio_path(uri: &str) -> Result<PathBuf, StyleTts2Error> {
    if let Some(path) = uri.strip_prefix("file://") {
        return Ok(PathBuf::from(path));
    }
    if uri.contains("://") {
        return Err(backend_error(format!(
            "StyleTTS2 reference audio URI `{uri}` is not a local file URI"
        )));
    }
    Ok(PathBuf::from(uri))
}

fn ensure_reference_wav_readable(uri: &str) -> Result<(), StyleTts2Error> {
    let path = reference_audio_path(uri)?;
    let reader = hound::WavReader::open(&path).map_err(|error| {
        backend_error(format!(
            "failed to open StyleTTS2 reference WAV {}: {error}",
            path.display()
        ))
    })?;
    if reader.spec().channels == 0 {
        return Err(invalid_output(format!(
            "StyleTTS2 reference WAV {} has zero channels",
            path.display()
        )));
    }
    Ok(())
}

fn read_reference_wav_mono_24khz(path: &Path) -> Result<Vec<f32>, StyleTts2Error> {
    let mut reader = hound::WavReader::open(path).map_err(|error| {
        backend_error(format!(
            "failed to open StyleTTS2 reference WAV {}: {error}",
            path.display()
        ))
    })?;
    let spec = reader.spec();
    if spec.channels == 0 {
        return Err(invalid_output(format!(
            "StyleTTS2 reference WAV {} has zero channels",
            path.display()
        )));
    }
    let channels = spec.channels as usize;
    let interleaved = read_wav_samples(&mut reader, spec.bits_per_sample, spec.sample_format)?;
    let mut mono = Vec::with_capacity(interleaved.len() / channels);
    for frame in interleaved.chunks(channels) {
        mono.push(frame.iter().sum::<f32>() / frame.len() as f32);
    }
    let mono = trim_silence(mono);
    let mono = resample_linear(&mono, spec.sample_rate, SAMPLE_RATE_HZ);
    if mono.len() < MIN_REFERENCE_SAMPLES {
        return Err(invalid_output(format!(
            "StyleTTS2 reference WAV {} is too short after trimming; use at least 1 second",
            path.display()
        )));
    }
    Ok(mono.into_iter().take(MAX_REFERENCE_SAMPLES).collect())
}

fn read_wav_samples<R: std::io::Read>(
    reader: &mut hound::WavReader<R>,
    bits_per_sample: u16,
    sample_format: hound::SampleFormat,
) -> Result<Vec<f32>, StyleTts2Error> {
    match sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|sample| {
                sample
                    .map_err(|error| {
                        backend_error(format!("failed to read float WAV sample: {error}"))
                    })
                    .and_then(|value| {
                        value.is_finite().then_some(value).ok_or_else(|| {
                            invalid_output("reference WAV contains non-finite samples")
                        })
                    })
            })
            .collect(),
        hound::SampleFormat::Int if bits_per_sample <= 16 => {
            let scale = ((1_i32 << (bits_per_sample.saturating_sub(1))) - 1).max(1) as f32;
            reader
                .samples::<i16>()
                .map(|sample| {
                    sample.map(|value| value as f32 / scale).map_err(|error| {
                        backend_error(format!("failed to read integer WAV sample: {error}"))
                    })
                })
                .collect()
        }
        hound::SampleFormat::Int => {
            let scale = ((1_i64 << (bits_per_sample.saturating_sub(1))) - 1).max(1) as f32;
            reader
                .samples::<i32>()
                .map(|sample| {
                    sample.map(|value| value as f32 / scale).map_err(|error| {
                        backend_error(format!("failed to read integer WAV sample: {error}"))
                    })
                })
                .collect()
        }
    }
}

fn trim_silence(samples: Vec<f32>) -> Vec<f32> {
    let max = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max);
    if max <= f32::EPSILON {
        return samples;
    }
    let threshold = max * 10_f32.powf(-30.0 / 20.0);
    let start = samples
        .iter()
        .position(|sample| sample.abs() >= threshold)
        .unwrap_or(0);
    let end = samples
        .iter()
        .rposition(|sample| sample.abs() >= threshold)
        .map(|index| index + 1)
        .unwrap_or(samples.len());
    samples[start..end].to_vec()
}

fn resample_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let output_len =
        ((samples.len() as u128 * target_rate as u128) / source_rate as u128).max(1) as usize;
    let step = source_rate as f64 / target_rate as f64;
    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let source = index as f64 * step;
        let left = source.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let fraction = (source - left as f64) as f32;
        output.push(samples[left] * (1.0 - fraction) + samples[right] * fraction);
    }
    output
}

fn ensure_file(path: &Path, label: &str) -> Result<(), StyleTts2Error> {
    if path.is_file() {
        return Ok(());
    }
    Err(backend_error(format!(
        "{label} ONNX model file not found at {}",
        path.display()
    )))
}

fn initialize_ort_runtime() -> Result<(), StyleTts2Error> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(initialize_ort_runtime_inner)
        .clone()
        .map_err(backend_error)
}

fn initialize_ort_runtime_inner() -> Result<(), String> {
    if let Some(path) = std::env::var_os("ORT_DYLIB_PATH").filter(|value| !value.is_empty()) {
        let path = PathBuf::from(path);
        if !path.is_file() {
            return Err(format!(
                "ORT_DYLIB_PATH points to {}, but that file does not exist",
                path.display()
            ));
        }
        return initialize_ort_runtime_from(&path);
    }

    let Some(path) = find_onnxruntime_dylib() else {
        return Err(
            "StyleTTS2 ONNX requires an ONNX Runtime shared library. Install ONNX Runtime or set ORT_DYLIB_PATH to libonnxruntime.so.".into(),
        );
    };
    initialize_ort_runtime_from(&path)
}

fn initialize_ort_runtime_from(path: &Path) -> Result<(), String> {
    ort::init_from(path)
        .map_err(|error| {
            format!(
                "failed to load ONNX Runtime dynamic library from {}: {error}",
                path.display()
            )
        })?
        .commit();
    Ok(())
}

fn find_onnxruntime_dylib() -> Option<PathBuf> {
    find_home_onnxruntime_dylib().or_else(find_linker_onnxruntime_dylib)
}

fn read_thread_count_env(name: &'static str) -> Result<Option<usize>, StyleTts2Error> {
    match std::env::var(name) {
        Ok(value) => parse_thread_count_env(name, &value),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(backend_error(format!("{name} must be valid UTF-8")))
        }
    }
}

fn parse_thread_count_env(
    name: &'static str,
    value: &str,
) -> Result<Option<usize>, StyleTts2Error> {
    let value = value.trim();
    if value.is_empty() {
        return Ok(None);
    }
    let threads = value.parse::<usize>().map_err(|error| {
        backend_error(format!(
            "{name} must be a positive integer thread count, got `{value}`: {error}"
        ))
    })?;
    validate_thread_count(name, threads)?;
    Ok(Some(threads))
}

fn validate_thread_count(label: &str, threads: usize) -> Result<(), StyleTts2Error> {
    if threads == 0 {
        return Err(backend_error(format!(
            "{label} thread count must be greater than zero"
        )));
    }
    Ok(())
}

fn timing(stage: &str, started: Instant) -> StyleTts2Timing {
    StyleTts2Timing {
        stage: stage.into(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn prefix_timing(prefix: &str, timing: StyleTts2Timing) -> StyleTts2Timing {
    StyleTts2Timing {
        stage: format!("{prefix}.{}", timing.stage),
        elapsed_ms: timing.elapsed_ms,
    }
}

fn find_home_onnxruntime_dylib() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let mut dirs = Vec::new();
    let local_lib = home.join(".local/lib");
    if let Ok(entries) = std::fs::read_dir(local_lib) {
        dirs.extend(entries.flatten().filter_map(|entry| {
            let file_name = entry.file_name();
            file_name
                .to_string_lossy()
                .starts_with("python")
                .then(|| entry.path().join("site-packages/onnxruntime/capi"))
        }));
    }
    for extensions_dir in [
        home.join(".vscode/extensions"),
        home.join(".vscode-server/extensions"),
    ] {
        if let Ok(entries) = std::fs::read_dir(extensions_dir) {
            dirs.extend(
                entries
                    .flatten()
                    .filter_map(|entry| {
                        let file_name = entry.file_name();
                        if file_name.to_string_lossy().contains("windows-ai-studio") {
                            Some(vec![
                                entry.path().join("bin"),
                                entry.path().join("ai-mlstudio/bin"),
                                entry.path().join("ai-foundry/bin"),
                            ])
                        } else {
                            None
                        }
                    })
                    .flatten(),
            );
        }
    }
    find_onnxruntime_dylib_in_dirs(dirs)
}

fn find_linker_onnxruntime_dylib() -> Option<PathBuf> {
    let mut search_dirs = Vec::new();
    if let Some(paths) = std::env::var_os("LD_LIBRARY_PATH") {
        search_dirs.extend(std::env::split_paths(&paths));
    }
    search_dirs.extend([
        PathBuf::from("/usr/local/lib"),
        PathBuf::from("/usr/local/lib64"),
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        PathBuf::from("/lib/x86_64-linux-gnu"),
    ]);
    find_onnxruntime_dylib_in_dirs(search_dirs)
}

fn find_onnxruntime_dylib_in_dirs(dirs: impl IntoIterator<Item = PathBuf>) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        candidates.extend(entries.flatten().filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            (name == "libonnxruntime.so" || name.starts_with("libonnxruntime.so."))
                .then(|| entry.path())
        }));
    }
    candidates.sort();
    candidates.pop()
}

fn load_session(
    path: &Path,
    label: &str,
    options: StyleTts2OnnxOptions,
) -> Result<Session, StyleTts2Error> {
    let execution_providers = styletts2_execution_providers(options);
    if execution_providers.is_empty() {
        return load_session_with_execution_providers(path, label, options, &[]);
    }

    match load_session_with_execution_providers(path, label, options, &execution_providers) {
        Ok(session) => Ok(session),
        Err(accelerated_error) => load_session_with_execution_providers(path, label, options, &[])
            .map_err(|cpu_error| {
                backend_error(format!(
                    "failed to load {label} with StyleTTS2 ONNX accelerators ({accelerated_error}); CPU fallback also failed: {cpu_error}"
                ))
            }),
    }
}

fn styletts2_execution_providers(_options: StyleTts2OnnxOptions) -> Vec<ExecutionProviderDispatch> {
    #[allow(unused_mut)]
    let mut providers = Vec::new();

    #[cfg(feature = "styletts2-onnx-cuda")]
    providers.push(ort::ep::CUDA::default().build().fail_silently());

    #[cfg(feature = "styletts2-onnx-onednn")]
    providers.push(ort::ep::OneDNN::default().build().fail_silently());

    #[cfg(feature = "styletts2-onnx-xnnpack")]
    if let Some(threads) = std::num::NonZeroUsize::new(_options.intra_threads) {
        providers.push(
            ort::ep::XNNPACK::default()
                .with_intra_op_num_threads(threads)
                .build()
                .fail_silently(),
        );
    }

    providers
}

fn load_session_with_execution_providers(
    path: &Path,
    label: &str,
    options: StyleTts2OnnxOptions,
    execution_providers: &[ExecutionProviderDispatch],
) -> Result<Session, StyleTts2Error> {
    let builder = Session::builder()
        .map_err(|error| {
            backend_error(format!("failed to create {label} session builder: {error}"))
        })?
        .with_intra_threads(options.intra_threads)
        .map_err(|error| {
            backend_error(format!(
                "failed to configure {label} intra-op threads: {error}"
            ))
        })?
        .with_inter_threads(options.inter_threads)
        .map_err(|error| {
            backend_error(format!(
                "failed to configure {label} inter-op threads: {error}"
            ))
        })?
        .with_intra_op_spinning(false)
        .map_err(|error| {
            backend_error(format!(
                "failed to configure {label} intra-op spinning: {error}"
            ))
        })?
        .with_optimization_level(options.optimization.graph_optimization_level())
        .map_err(|error| {
            backend_error(format!("failed to configure {label} optimization: {error}"))
        })?;

    let mut builder = configure_execution_providers(builder, label, execution_providers)?;

    builder.commit_from_file(path).map_err(|error| {
        backend_error(format!(
            "failed to load {label} ONNX model from {}: {error}",
            path.display()
        ))
    })
}

fn configure_execution_providers(
    builder: SessionBuilder,
    label: &str,
    execution_providers: &[ExecutionProviderDispatch],
) -> Result<SessionBuilder, StyleTts2Error> {
    if execution_providers.is_empty() {
        return Ok(builder);
    }

    builder
        .with_execution_providers(execution_providers)
        .map_err(|error| {
            backend_error(format!(
                "failed to configure {label} execution providers: {error}"
            ))
        })
}

fn extract_f32_tensor(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<(Vec<i64>, Vec<f32>), StyleTts2Error> {
    let output = outputs
        .get(name)
        .ok_or_else(|| invalid_output(format!("StyleTTS2 inference did not return `{name}`")))?;
    let output = output
        .downcast_ref::<DynTensorValueType>()
        .map_err(|error| {
            invalid_output(format!(
                "StyleTTS2 output `{name}` is not a tensor: {error}"
            ))
        })?;
    let (shape, values) = output.try_extract_tensor::<f32>().map_err(|error| {
        invalid_output(format!("StyleTTS2 output `{name}` is not f32: {error}"))
    })?;
    if !values.iter().all(|value| value.is_finite()) {
        return Err(invalid_output(format!(
            "StyleTTS2 output `{name}` contains non-finite values"
        )));
    }
    Ok((shape.to_vec(), values.to_vec()))
}

fn backend_error(message: impl Into<String>) -> StyleTts2Error {
    StyleTts2Error::Backend {
        message: message.into(),
    }
}

fn invalid_output(reason: impl Into<String>) -> StyleTts2Error {
    StyleTts2Error::InvalidOutput {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbols::StyleTts2SymbolSource;
    use crate::symbols::StyleTts2SymbolToken;

    #[test]
    fn onnx_optimization_modes_map_to_expected_levels() {
        assert_eq!(
            StyleTts2OnnxOptimization::Generation.graph_optimization_level(),
            GraphOptimizationLevel::Level3
        );
        assert_eq!(
            StyleTts2OnnxOptimization::Deterministic.graph_optimization_level(),
            GraphOptimizationLevel::Disable
        );
        assert_eq!(
            StyleTts2OnnxOptimization::default(),
            StyleTts2OnnxOptimization::Generation
        );
    }

    #[test]
    fn onnx_options_default_to_single_threaded_generation() {
        assert_eq!(
            StyleTts2OnnxOptions::default(),
            StyleTts2OnnxOptions {
                optimization: StyleTts2OnnxOptimization::Generation,
                intra_threads: 1,
                inter_threads: 1,
            }
        );
    }

    #[test]
    fn onnx_thread_env_parsing_accepts_positive_counts() {
        assert_eq!(
            parse_thread_count_env(STYLETTS2_ONNX_INTRA_THREADS_ENV, " 4 ").expect("thread count"),
            Some(4)
        );
        assert_eq!(
            parse_thread_count_env(STYLETTS2_ONNX_INTER_THREADS_ENV, "")
                .expect("empty env ignored"),
            None
        );
    }

    #[test]
    fn onnx_thread_env_parsing_rejects_invalid_counts() {
        let error = parse_thread_count_env(STYLETTS2_ONNX_INTRA_THREADS_ENV, "0")
            .expect_err("zero thread count should fail");
        assert!(error.to_string().contains("must be greater than zero"));

        let error = parse_thread_count_env(STYLETTS2_ONNX_INTER_THREADS_ENV, "many")
            .expect_err("non-numeric thread count should fail");
        assert!(error.to_string().contains("positive integer"));
    }

    #[test]
    fn diffusion_sampling_is_skipped_only_when_reference_blend_is_exact() {
        let mut options = StyleTts2DiffusionOptions {
            alpha: 0.0,
            beta: 0.0,
            ..Default::default()
        };
        assert!(!should_sample_diffusion_style(&options));

        options.alpha = f32::MIN_POSITIVE;
        assert!(should_sample_diffusion_style(&options));

        options.alpha = 0.0;
        options.beta = f32::MIN_POSITIVE;
        assert!(should_sample_diffusion_style(&options));
    }

    #[test]
    fn style_blend_uses_alpha_for_timbre_and_beta_for_prosody() {
        let predicted = vec![1.0; STYLE_VECTOR_DIMS];
        let reference = vec![0.0; STYLE_VECTOR_DIMS];
        let blended = blend_predicted_and_reference_style(&predicted, &reference, 0.3, 0.1);

        assert_eq!(&blended[..STYLE_HALF_DIMS], vec![0.3; STYLE_HALF_DIMS]);
        assert_eq!(&blended[STYLE_HALF_DIMS..], vec![0.1; STYLE_HALF_DIMS]);
    }

    #[test]
    fn arpabet_symbols_map_to_styletts2_token_ids() {
        let ids = styletts2_token_ids(&StyleTts2SymbolSequence {
            tokens: vec![
                token("HH"),
                token("AH"),
                token("L"),
                token("OW"),
                token("|"),
                token("W"),
                token("ER"),
                token("L"),
                token("D"),
            ],
        })
        .expect("token ids");

        assert_eq!(ids[0], 0);
        assert!(ids.iter().all(|id| (0..178).contains(id)));
        assert!(ids.len() > 9);
        assert!(ids.contains(&styletts2_character_id('ɝ').expect("rhotic vowel id")));
    }

    #[test]
    fn punctuation_symbols_map_to_styletts2_token_ids() {
        let ids = styletts2_token_ids(&StyleTts2SymbolSequence {
            tokens: vec![token("HH"), token("AY"), token("!")],
        })
        .expect("token ids");

        assert_eq!(ids[0], 0);
        assert!(ids.contains(&styletts2_character_id('!').expect("punctuation id")));
    }

    #[test]
    fn affricates_lower_to_espeak_style_digraph_tokens() {
        assert_eq!(styletts2_text_for_symbol("CH").expect("CH maps"), "tʃ");
        assert_eq!(styletts2_text_for_symbol("JH").expect("JH maps"), "dʒ");

        let ids = styletts2_token_ids(&StyleTts2SymbolSequence {
            tokens: vec![token("CH"), token("EY"), token("N"), token("JH")],
        })
        .expect("token ids");

        let expected_inner = ['t', 'ʃ', 'e', 'ɪ', 'n', 'd', 'ʒ']
            .into_iter()
            .map(|character| styletts2_character_id(character).expect("character id"))
            .collect::<Vec<_>>();

        assert_eq!(&ids[1..], expected_inner.as_slice());
        assert!(!ids.contains(&styletts2_character_id('ʧ').expect("ligature CH id")));
        assert!(!ids.contains(&styletts2_character_id('ʤ').expect("ligature JH id")));
    }

    fn token(symbol: &str) -> StyleTts2SymbolToken {
        StyleTts2SymbolToken {
            symbol: symbol.into(),
            source: StyleTts2SymbolSource::Phoneme,
        }
    }
}
