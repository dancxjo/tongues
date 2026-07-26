//! Burn-native end-to-end VITS speech synthesis.
//!
//! Tongues' linguistic plan remains the shared representation. The imported
//! model vocabulary, token IDs, speaker rows, and checkpoint layout are
//! resolved only inside this adapter.
//!
//! Source provenance: `audit-required`. This file was introduced by commit
//! `8e3a9c6`, whose message combines import, adaptation, and reverse
//! engineering without identifying the exact relationship. See
//! `docs/provenance.md` before changing its license or provenance notice.

use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor, TensorData};
use speaking::{SpeakerReferenceSource, UtterancePlan};

use crate::burn_vits_flow::expand_prior_statistics_with_frames;
use crate::profiling::{
    finish_backend_stage, finish_host_stage, reborrow_profiler, record_load_stage,
};
use crate::vits_config::ImportedVitsConfig;
use crate::vits_projector::VitsLinguisticProjector;
use crate::{
    ceil_durations, normalize_embedding, AudioChunk, AudioSink, ConditioningEmbedding,
    DVectorCatalog, LanguageCatalog, ModelLoadProfileEvent, ModelLoadStage,
    NativeSpeakerEmbeddingService, ResidualCouplingFlow, ResidualCouplingFlowConfig,
    SpeakerCatalog, SpeakerEmbeddingCachePolicy, SpeakerEncoderPackageConfig,
    SpeechModelCapabilities, SpeechModelFamily, SpeechSynthesisEngine, SpeechSynthesisRequest,
    StochasticDurationConfig, StochasticDurationPredictor, SynthesisDimension, SynthesisProfiler,
    SynthesisStage, VitsInferenceConfig, VitsTextPriorConfig, VitsTextPriorEncoder,
    VitsWaveformDecoder, VitsWaveformDecoderConfig, Waveform, WaveformContract,
};
use crate::{LinguisticProjector, ModelInputContract};

const DEFAULT_MAX_OUTPUT_FRAMES: usize = 65_536;
const STREAM_LATENT_FRAMES: usize = 64;

fn speaker_embedding_tensor(path: &str, _container: &str) -> bool {
    path.starts_with("emb_g.")
}

fn language_embedding_tensor(path: &str, _container: &str) -> bool {
    path.starts_with("emb_l.")
}

#[derive(Module, Debug)]
struct SpeakerEmbedding<B: Backend> {
    emb_g: Embedding<B>,
}

impl<B: Backend> SpeakerEmbedding<B> {
    fn init(num_speakers: usize, dimensions: usize, device: &B::Device) -> Self {
        Self {
            emb_g: EmbeddingConfig::new(num_speakers, dimensions).init(device),
        }
    }

    fn load_checkpoint(mut self, checkpoint_path: &Path) -> Result<Self> {
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut self,
            checkpoint_path,
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                predicate: Some(speaker_embedding_tensor),
                map_indices_contiguous: false,
                allow_partial: true,
                skip_enum_variants: true,
                ..Default::default()
            },
        )
        .context("failed to load speaker embedding checkpoint")?;
        let unused = result
            .unused
            .iter()
            .filter(|path| path.starts_with("emb_g."))
            .collect::<Vec<_>>();
        ensure!(
            result.missing.is_empty() && result.errors.is_empty() && unused.is_empty(),
            "speaker embedding subtree does not exactly match the Burn module: {} missing, {} load errors, unused [{}]",
            result.missing.len(),
            result.errors.len(),
            unused
                .into_iter()
                .map(|path| path.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(self)
    }

    fn forward(&self, speaker_id: u32, device: &B::Device) -> Tensor<B, 3> {
        let id = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![speaker_id as i64], [1, 1]),
            device,
        );
        self.emb_g.forward(id).swap_dims(1, 2)
    }
}

#[derive(Module, Debug)]
struct LanguageEmbedding<B: Backend> {
    emb_l: Embedding<B>,
}

impl<B: Backend> LanguageEmbedding<B> {
    fn init(num_languages: usize, dimensions: usize, device: &B::Device) -> Self {
        Self {
            emb_l: EmbeddingConfig::new(num_languages, dimensions).init(device),
        }
    }

    fn load_checkpoint(mut self, checkpoint_path: &Path) -> Result<Self> {
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut self,
            checkpoint_path,
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                predicate: Some(language_embedding_tensor),
                map_indices_contiguous: false,
                allow_partial: true,
                skip_enum_variants: true,
                ..Default::default()
            },
        )
        .context("failed to load language embedding checkpoint")?;
        let unused = result
            .unused
            .iter()
            .filter(|path| path.starts_with("emb_l."))
            .collect::<Vec<_>>();
        ensure!(
            result.missing.is_empty() && result.errors.is_empty() && unused.is_empty(),
            "language embedding subtree does not exactly match the Burn module: {} missing, {} load errors, unused [{}]",
            result.missing.len(),
            result.errors.len(),
            unused
                .into_iter()
                .map(|path| path.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(self)
    }

    fn forward(&self, language_id: u32, device: &B::Device) -> Tensor<B, 3> {
        let id = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![language_id as i64], [1, 1]),
            device,
        );
        self.emb_l.forward(id).swap_dims(1, 2)
    }
}

/// Native end-to-end VITS engine using Tongues plans and named speakers.
pub struct BurnVitsSpeech<B: Backend> {
    config: VitsInferenceConfig,
    projector: VitsLinguisticProjector,
    speakers: Option<SpeakerCatalog>,
    d_vectors: Option<DVectorCatalog>,
    languages: Option<LanguageCatalog>,
    speaker_embedding: Option<SpeakerEmbedding<B>>,
    reference_encoder: Option<NativeSpeakerEmbeddingService<B>>,
    language_embedding: Option<LanguageEmbedding<B>>,
    text_prior: VitsTextPriorEncoder<B>,
    duration_predictor: StochasticDurationPredictor<B>,
    flow: ResidualCouplingFlow<B>,
    waveform_decoder: VitsWaveformDecoder<B>,
    output_contract: WaveformContract,
    device: B::Device,
}

impl<B: Backend> BurnVitsSpeech<B> {
    pub fn load(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        speaker_map_path: impl AsRef<Path>,
        device: B::Device,
    ) -> Result<Self> {
        Self::load_internal(
            config_path,
            checkpoint_path,
            speaker_map_path,
            None,
            device,
            None,
            None,
        )
    }

    pub fn load_with_languages(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        speaker_map_path: impl AsRef<Path>,
        language_map_path: impl AsRef<Path>,
        device: B::Device,
    ) -> Result<Self> {
        Self::load_internal(
            config_path,
            checkpoint_path,
            speaker_map_path,
            Some(language_map_path.as_ref()),
            device,
            None,
            None,
        )
    }

    pub fn load_profiled(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        speaker_map_path: impl AsRef<Path>,
        device: B::Device,
        profiler: &mut dyn FnMut(ModelLoadProfileEvent),
    ) -> Result<Self> {
        Self::load_internal(
            config_path,
            checkpoint_path,
            speaker_map_path,
            None,
            device,
            Some(profiler),
            None,
        )
    }

    pub fn load_profiled_with_languages(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        speaker_map_path: impl AsRef<Path>,
        language_map_path: impl AsRef<Path>,
        device: B::Device,
        profiler: &mut dyn FnMut(ModelLoadProfileEvent),
    ) -> Result<Self> {
        Self::load_internal(
            config_path,
            checkpoint_path,
            speaker_map_path,
            Some(language_map_path.as_ref()),
            device,
            Some(profiler),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_your_tts(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        d_vector_path: impl AsRef<Path>,
        language_map_path: impl AsRef<Path>,
        speaker_encoder_config_path: impl AsRef<Path>,
        speaker_encoder_checkpoint_path: impl AsRef<Path>,
        device: B::Device,
        cache_policy: SpeakerEmbeddingCachePolicy,
    ) -> Result<Self> {
        let speaker_encoder_config =
            SpeakerEncoderPackageConfig::from_file(speaker_encoder_config_path)?;
        Self::load_internal(
            config_path,
            checkpoint_path,
            d_vector_path,
            Some(language_map_path.as_ref()),
            device,
            None,
            Some((
                speaker_encoder_config,
                speaker_encoder_checkpoint_path.as_ref(),
                cache_policy,
            )),
        )
    }

    fn load_internal(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        speaker_map_path: impl AsRef<Path>,
        language_map_path: Option<&Path>,
        device: B::Device,
        profiler: Option<&mut dyn FnMut(ModelLoadProfileEvent)>,
        reference_encoder: Option<(
            SpeakerEncoderPackageConfig,
            &Path,
            SpeakerEmbeddingCachePolicy,
        )>,
    ) -> Result<Self> {
        let mut profiler = profiler;
        let started = Instant::now();
        let config_path = config_path.as_ref();
        let checkpoint_path = checkpoint_path.as_ref();
        let source = fs::read_to_string(config_path)
            .with_context(|| format!("failed to read VITS config {}", config_path.display()))?;
        let imported = ImportedVitsConfig::from_json5_str(&source)?;
        let config = imported.inference_config();
        let network = &config.network;
        ensure!(
            network.use_sdp,
            "this VITS engine requires stochastic durations"
        );
        ensure!(
            network.use_speaker_embedding ^ network.use_d_vector_file,
            "this VITS engine requires exactly one speaker-conditioning mode"
        );

        let projector = VitsLinguisticProjector::from_config(imported)?;
        let speakers = network
            .use_speaker_embedding
            .then(|| SpeakerCatalog::from_file(&speaker_map_path, network.num_speakers))
            .transpose()?;
        let d_vectors = network
            .use_d_vector_file
            .then(|| {
                DVectorCatalog::from_file(
                    &speaker_map_path,
                    network.d_vector_dim,
                    crate::COQUI_RESNET_SPEAKER_EMBEDDING_SPACE,
                )
            })
            .transpose()?;
        let languages = if network.use_language_embedding {
            let language_map_path = language_map_path
                .context("language_ids.json is required for language-conditioned VITS")?;
            Some(LanguageCatalog::from_file(
                language_map_path,
                network.num_languages,
            )?)
        } else {
            ensure!(
                language_map_path.is_none(),
                "a language map was supplied to a VITS checkpoint without learned language embeddings"
            );
            None
        };
        let speaker_channels = if network.use_speaker_embedding {
            network.speaker_embedding_channels
        } else {
            network.d_vector_dim
        };
        let text_config = VitsTextPriorConfig::from_model_config(&config)?;
        let mut duration_config =
            StochasticDurationConfig::new(text_config.encoder_channels(), 192, 3);
        duration_config.conditioning_channels = if network.condition_dp_on_speaker {
            speaker_channels
        } else {
            0
        };
        duration_config.language_conditioning_channels = text_config.language_embedding_channels;
        let flow_config = ResidualCouplingFlowConfig {
            channels: network.hidden_channels,
            hidden_channels: network.hidden_channels,
            kernel_size: network.kernel_size_flow,
            dilation_rate: network.dilation_rate_flow,
            num_layers: network.num_layers_flow,
            num_flows: 4,
            conditioning_channels: speaker_channels,
        };
        let decoder_config = VitsWaveformDecoderConfig::from_model_config(&config)?;
        let output_contract = WaveformContract::mono(config.audio.sample_rate);
        record_load_stage(
            &mut profiler,
            ModelLoadStage::ConfigCheckpointParsing,
            started,
            Some("vits"),
        );

        let started = Instant::now();
        let speaker_embedding = network.use_speaker_embedding.then(|| {
            SpeakerEmbedding::init(network.num_speakers as usize, speaker_channels, &device)
        });
        let reference_encoder = reference_encoder
            .map(|(config, checkpoint, cache_policy)| {
                ensure!(
                    network.use_d_vector_file,
                    "a reference encoder can only be attached to d-vector VITS"
                );
                ensure!(
                    config.projection_dim == network.d_vector_dim,
                    "speaker encoder emits {} dimensions but VITS expects {}",
                    config.projection_dim,
                    network.d_vector_dim
                );
                NativeSpeakerEmbeddingService::load(
                    config,
                    checkpoint,
                    device.clone(),
                    cache_policy,
                )
            })
            .transpose()?;
        let language_embedding = network.use_language_embedding.then(|| {
            LanguageEmbedding::init(
                network.num_languages as usize,
                network.embedded_language_dim,
                &device,
            )
        });
        let text_prior = text_config.init(&device)?;
        let duration_predictor = duration_config.init(&device)?;
        let flow = flow_config.init(&device)?;
        let waveform_decoder = decoder_config.init(&device)?;
        record_load_stage(
            &mut profiler,
            ModelLoadStage::ModelConstruction,
            started,
            Some("vits"),
        );

        let started = Instant::now();
        let speaker_embedding = speaker_embedding
            .map(|embedding| embedding.load_checkpoint(checkpoint_path))
            .transpose()?;
        let language_embedding = language_embedding
            .map(|embedding| embedding.load_checkpoint(checkpoint_path))
            .transpose()?;
        let text_prior = text_prior.load_checkpoint(checkpoint_path)?;
        let duration_predictor = duration_predictor.load_checkpoint(checkpoint_path)?;
        let flow = flow.load_checkpoint(checkpoint_path)?;
        let waveform_decoder = waveform_decoder.load_checkpoint(checkpoint_path)?;
        record_load_stage(
            &mut profiler,
            ModelLoadStage::WeightUpload,
            started,
            Some("vits"),
        );

        Ok(Self {
            config,
            projector,
            speakers,
            d_vectors,
            languages,
            speaker_embedding,
            reference_encoder,
            language_embedding,
            text_prior,
            duration_predictor,
            flow,
            waveform_decoder,
            output_contract,
            device,
        })
    }

    pub fn input_contract(&self) -> &ModelInputContract {
        self.projector.contract()
    }

    pub fn speaker_catalog(&self) -> &SpeakerCatalog {
        self.speakers
            .as_ref()
            .expect("this VITS engine uses d-vectors instead of learned speakers")
    }

    pub fn learned_speaker_catalog(&self) -> Option<&SpeakerCatalog> {
        self.speakers.as_ref()
    }

    pub fn d_vector_catalog(&self) -> Option<&DVectorCatalog> {
        self.d_vectors.as_ref()
    }

    pub fn language_catalog(&self) -> Option<&LanguageCatalog> {
        self.languages.as_ref()
    }

    pub fn projected_input(&self, plan: &UtterancePlan) -> Result<crate::PhonemeTokenIds> {
        self.projector.project(plan)
    }

    pub fn synthesize(&mut self, request: &SpeechSynthesisRequest) -> Result<Waveform> {
        let mut samples = Vec::new();
        self.synthesize_streaming(
            request,
            &mut |chunk: AudioChunk| {
                samples.extend(chunk.pcm_mono_f32);
                Ok(())
            },
            None,
        )?;
        let waveform = Waveform {
            contract: self.output_contract.clone(),
            samples,
        };
        waveform.validate()?;
        Ok(waveform)
    }

    fn speaker_conditioning(&mut self, request: &SpeechSynthesisRequest) -> Result<Tensor<B, 3>> {
        if let (Some(speakers), Some(embedding)) = (&self.speakers, &self.speaker_embedding) {
            ensure!(
                request.plan.speaker_reference.is_none(),
                "learned-speaker VITS does not accept a speaker reference"
            );
            let speaker_id =
                speakers.resolve(request.plan.speaker.as_ref(), request.options.speaker_id)?;
            return Ok(embedding.forward(speaker_id, &self.device));
        }

        ensure!(
            request.options.speaker_id.is_none(),
            "d-vector VITS consumes a named enrollment or speaker reference, not a numeric speaker ID"
        );
        ensure!(
            !(request.plan.speaker.is_some() && request.plan.speaker_reference.is_some()),
            "choose either a named d-vector enrollment or a speaker reference, not both"
        );
        let expected = self
            .d_vectors
            .as_ref()
            .context("d-vector VITS is missing its declared embedding catalog")?
            .contract()
            .clone();
        let embedding = if let Some(speaker) = &request.plan.speaker {
            self.d_vectors
                .as_ref()
                .expect("checked above")
                .resolve(&speaker.0)?
        } else {
            let reference = request
                .plan
                .speaker_reference
                .as_ref()
                .context("d-vector VITS requires a named enrollment or speaker reference")?;
            match &reference.source {
                SpeakerReferenceSource::Embedding { space, values } => {
                    ensure!(
                        space == &expected.space,
                        "speaker embedding space `{space}` does not match `{}`",
                        expected.space
                    );
                    let mut values = values.clone();
                    normalize_embedding(&mut values)?;
                    ConditioningEmbedding {
                        contract: expected.clone(),
                        values,
                    }
                }
                SpeakerReferenceSource::ReferenceAudio { uri } => self
                    .reference_encoder
                    .as_mut()
                    .context(
                        "reference audio requires the model's native speaker encoder checkpoint",
                    )?
                    .encode_uri(uri)?,
            }
        };
        embedding.validate()?;
        ensure!(
            embedding.contract == expected,
            "speaker embedding contract does not match this VITS checkpoint"
        );
        Ok(Tensor::from_data(
            TensorData::new(embedding.values, [1, expected.dimensions, 1]),
            &self.device,
        ))
    }

    fn prepare_latent(
        &mut self,
        request: &SpeechSynthesisRequest,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>)> {
        let mut profiler = profiler;
        self.projector.contract().ensure_supports(&request.plan)?;
        let started = Instant::now();
        let language_id = match &self.languages {
            Some(languages) => Some(languages.resolve(
                request.options.model_language.as_deref(),
                request.options.language_id,
            )?),
            None => {
                ensure!(
                    request.options.model_language.is_none()
                        && request.options.language_id.is_none(),
                    "this VITS checkpoint does not use learned language embeddings"
                );
                None
            }
        };
        let projected = self.projector.project(&request.plan)?;
        let token_count = projected.ids.len();
        finish_host_stage(
            &mut profiler,
            SynthesisStage::CheckpointProjection,
            started,
            [SynthesisDimension::new("tokens", token_count)],
        );

        let started = Instant::now();
        let token_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(projected.ids, [1, token_count]),
            &self.device,
        );
        let lengths = Tensor::<B, 1, Int>::from_data(
            TensorData::new(vec![token_count as i64], [1]),
            &self.device,
        );
        let conditioning = self.speaker_conditioning(request)?;
        let language_conditioning = language_id.map(|language_id| {
            self.language_embedding
                .as_ref()
                .expect("language catalog and embedding are constructed together")
                .forward(language_id, &self.device)
        });
        finish_backend_stage::<B>(
            &mut profiler,
            &self.device,
            SynthesisStage::HostToDevice,
            started,
            [
                SynthesisDimension::new("tokens", token_count),
                SynthesisDimension::new("speaker_channels", conditioning.dims()[1]),
            ],
        )?;

        let started = Instant::now();
        let prior = self.text_prior.forward_conditioned(
            token_ids,
            lengths,
            language_conditioning.clone(),
        )?;
        finish_backend_stage::<B>(
            &mut profiler,
            &self.device,
            SynthesisStage::TextEncoder,
            started,
            [
                SynthesisDimension::new("tokens", token_count),
                SynthesisDimension::new("channels", self.config.network.hidden_channels),
            ],
        )?;

        let duration_noise = request
            .options
            .noise_w
            .unwrap_or(self.config.network.inference_noise_scale_dp);
        ensure!(
            duration_noise.is_finite() && duration_noise >= 0.0,
            "duration noise scale must be finite and non-negative"
        );
        let duration_conditioning = self
            .config
            .network
            .condition_dp_on_speaker
            .then(|| conditioning.clone());
        let started = Instant::now();
        let log_durations = match request.options.seed {
            Some(seed) => self.duration_predictor.reverse_seeded_conditioned(
                prior.encoded,
                prior.mask.clone(),
                duration_conditioning,
                language_conditioning,
                f64::from(duration_noise),
                seed,
            )?,
            None => self.duration_predictor.reverse_conditioned(
                prior.encoded,
                prior.mask.clone(),
                duration_conditioning,
                language_conditioning,
                f64::from(duration_noise),
            )?,
        };
        finish_backend_stage::<B>(
            &mut profiler,
            &self.device,
            SynthesisStage::DurationPrediction,
            started,
            [SynthesisDimension::new("tokens", token_count)],
        )?;
        let length_scale = request
            .options
            .length_scale
            .unwrap_or(self.config.network.length_scale);
        let started = Instant::now();
        let rounded = ceil_durations(
            log_durations.exp(),
            prior.mask,
            f64::from(length_scale),
            DEFAULT_MAX_OUTPUT_FRAMES,
        )?;
        let output_frames = rounded.output_frames;
        let expanded = expand_prior_statistics_with_frames(
            prior.mean,
            prior.log_scale,
            rounded.values,
            Some(output_frames),
            DEFAULT_MAX_OUTPUT_FRAMES,
        )?;
        finish_backend_stage::<B>(
            &mut profiler,
            &self.device,
            SynthesisStage::DurationExpansion,
            started,
            [
                SynthesisDimension::new("tokens", token_count),
                SynthesisDimension::new("frames", output_frames),
            ],
        )?;

        let latent_noise = request
            .options
            .noise_scale
            .unwrap_or(self.config.network.inference_noise_scale);
        ensure!(
            latent_noise.is_finite() && latent_noise >= 0.0,
            "latent noise scale must be finite and non-negative"
        );
        if let Some(seed) = request.options.seed {
            B::seed(&self.device, seed.wrapping_add(1));
        }
        let noise = Tensor::random(
            expanded.mean.dims(),
            Distribution::Normal(0.0, 1.0),
            &self.device,
        );
        let latent_prior =
            expanded.mean + noise * expanded.log_scale.exp() * f64::from(latent_noise);
        let started = Instant::now();
        let latent = self.flow.reverse(
            latent_prior,
            expanded.frame_mask.clone(),
            Some(conditioning.clone()),
        )?;
        let latent = latent * expanded.frame_mask;
        let latent = if let Some(max_frames) = self.config.network.max_inference_len {
            let [batch, channels, frames] = latent.dims();
            latent.slice([0..batch, 0..channels, 0..frames.min(max_frames)])
        } else {
            latent
        };
        finish_backend_stage::<B>(
            &mut profiler,
            &self.device,
            SynthesisStage::VitsFlow,
            started,
            [
                SynthesisDimension::new("frames", latent.dims()[2]),
                SynthesisDimension::new("channels", latent.dims()[1]),
            ],
        )?;
        Ok((latent, conditioning))
    }

    fn synthesize_streaming(
        &mut self,
        request: &SpeechSynthesisRequest,
        sink: &mut dyn AudioSink,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<()> {
        let mut profiler = profiler;
        let (latent, conditioning) =
            self.prepare_latent(request, reborrow_profiler(&mut profiler))?;
        let [batch, _channels, frames] = latent.dims();
        ensure!(
            batch == 1,
            "streaming VITS currently requires batch size one"
        );
        let upsample = self.waveform_decoder.upsample_factor();
        let chunk_count = frames.div_ceil(STREAM_LATENT_FRAMES);

        // Decode once. The previous overlapping-window implementation launched
        // the full HiFi-GAN decoder repeatedly and recomputed 64 context frames
        // around nearly every 64-frame output chunk.
        let started = Instant::now();
        let decoded = self.waveform_decoder.forward(latent, Some(conditioning))?;
        let expected_samples = frames * upsample;
        ensure!(
            decoded.dims() == [batch, 1, expected_samples],
            "VITS decoder emitted {:?}; expected [{batch}, 1, {expected_samples}]",
            decoded.dims()
        );
        finish_backend_stage::<B>(
            &mut profiler,
            &self.device,
            SynthesisStage::WaveformDecoder,
            started,
            [
                SynthesisDimension::new("latent_frames", frames),
                SynthesisDimension::new("samples", expected_samples),
                SynthesisDimension::new("decoder_launches", 1),
            ],
        )?;

        let started = Instant::now();
        let samples = decoded
            .into_data()
            .to_vec::<f32>()
            .context("VITS waveform output is not f32")?;
        finish_host_stage(
            &mut profiler,
            SynthesisStage::DeviceToHost,
            started,
            [SynthesisDimension::new("samples", expected_samples)],
        );

        for chunk_index in 0..chunk_count {
            let frame_start = chunk_index * STREAM_LATENT_FRAMES;
            let frame_end = (frame_start + STREAM_LATENT_FRAMES).min(frames);
            let sample_start = frame_start * upsample;
            let sample_end = frame_end * upsample;
            let started = Instant::now();
            sink.emit(AudioChunk {
                chunk_index,
                is_final: chunk_index + 1 == chunk_count,
                pause_after_ms: 0,
                sample_rate_hz: self.output_contract.sample_rate_hz,
                pcm_mono_f32: samples[sample_start..sample_end].to_vec(),
            })?;
            finish_host_stage(
                &mut profiler,
                SynthesisStage::AudioSink,
                started,
                [
                    SynthesisDimension::new("chunk", chunk_index),
                    SynthesisDimension::new("samples", sample_end - sample_start),
                ],
            );
        }
        Ok(())
    }
}

impl<B: Backend> SpeechSynthesisEngine for BurnVitsSpeech<B> {
    fn capabilities(&self) -> SpeechModelCapabilities {
        SpeechModelCapabilities {
            family: SpeechModelFamily::EndToEndSpeech,
            supports_named_speakers: true,
            supports_languages: self.languages.is_some(),
            supports_reference_audio: self.reference_encoder.is_some(),
            supports_voice_conversion: false,
            integrated_vocoder: true,
        }
    }

    fn sample_rate_hz(&self) -> u32 {
        self.output_contract.sample_rate_hz
    }

    fn synthesize_plan_streaming(
        &mut self,
        request: &SpeechSynthesisRequest,
        sink: &mut dyn AudioSink,
    ) -> Result<()> {
        self.synthesize_streaming(request, sink, None)
    }

    fn synthesize_plan_streaming_profiled(
        &mut self,
        request: &SpeechSynthesisRequest,
        sink: &mut dyn AudioSink,
        profiler: &mut dyn SynthesisProfiler,
    ) -> Result<()> {
        self.synthesize_streaming(request, sink, Some(profiler))
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use speaking::SpeakerId;

    use super::*;
    use crate::{utterance_plan_from_text, SpeechRequest, SynthesisOptions};

    type TestBackend = NdArray<f32>;

    #[test]
    #[ignore = "requires pinned Coqui VITS artifacts; run scripts/speech-conformance.sh"]
    fn published_named_speakers_synthesize_streaming_when_available() {
        let config = std::env::var_os("TONGUES_TEST_COQUI_VITS_CONFIG")
            .expect("TONGUES_TEST_COQUI_VITS_CONFIG is required");
        let checkpoint = std::env::var_os("TONGUES_TEST_COQUI_VITS_CHECKPOINT")
            .expect("TONGUES_TEST_COQUI_VITS_CHECKPOINT is required");
        let speakers = std::env::var_os("TONGUES_TEST_COQUI_VITS_SPEAKERS")
            .expect("TONGUES_TEST_COQUI_VITS_SPEAKERS is required");
        let mut engine =
            BurnVitsSpeech::<TestBackend>::load(config, checkpoint, speakers, NdArrayDevice::Cpu)
                .expect("published VITS engine");

        for (name, text) in [
            ("p225", "Morning light rested on cedar trees."),
            ("p330", "Rain polished the streets, and lamps glowed."),
            ("p376", "The patient astronomer mapped three quiet moons."),
        ] {
            let mut plan = utterance_plan_from_text(SpeechRequest {
                text: text.into(),
                variety: "en-US".into(),
            })
            .expect("native linguistic plan");
            plan.speaker = Some(SpeakerId(name.into()));
            let request = SpeechSynthesisRequest {
                plan,
                options: SynthesisOptions {
                    seed: Some(27),
                    ..SynthesisOptions::default()
                },
            };
            let first = engine
                .synthesize(&request)
                .expect("named-speaker synthesis");
            let second = engine
                .synthesize(&request)
                .expect("repeatable named-speaker synthesis");

            assert!(!first.samples.is_empty());
            assert!(first.samples.iter().all(|sample| sample.is_finite()));
            assert_eq!(
                first.samples, second.samples,
                "seeded VITS inference must be repeatable for {name}"
            );
        }
    }

    #[test]
    #[ignore = "requires the pinned published YourTTS and speaker-encoder artifacts"]
    fn published_your_tts_checkpoints_load_when_available() {
        let required =
            |name: &str| std::env::var_os(name).unwrap_or_else(|| panic!("{name} is required"));
        let engine = BurnVitsSpeech::<TestBackend>::load_your_tts(
            required("TONGUES_TEST_YOURTTS_CONFIG"),
            required("TONGUES_TEST_YOURTTS_CHECKPOINT"),
            required("TONGUES_TEST_YOURTTS_SPEAKERS"),
            required("TONGUES_TEST_YOURTTS_LANGUAGES"),
            required("TONGUES_TEST_COQUI_SPEAKER_CONFIG"),
            required("TONGUES_TEST_COQUI_SPEAKER_MODEL"),
            NdArrayDevice::Cpu,
            SpeakerEmbeddingCachePolicy::Disabled,
        )
        .expect("published YourTTS graph");
        assert_eq!(
            engine.d_vector_catalog().unwrap().contract().dimensions,
            512
        );
        assert!(engine.capabilities().supports_languages);
        assert!(engine.capabilities().supports_reference_audio);
    }

    #[test]
    #[ignore = "requires the pinned published YourTTS and speaker-encoder artifacts"]
    fn published_your_tts_named_enrollment_synthesizes_when_available() {
        let required =
            |name: &str| std::env::var_os(name).unwrap_or_else(|| panic!("{name} is required"));
        let mut engine = BurnVitsSpeech::<TestBackend>::load_your_tts(
            required("TONGUES_TEST_YOURTTS_CONFIG"),
            required("TONGUES_TEST_YOURTTS_CHECKPOINT"),
            required("TONGUES_TEST_YOURTTS_SPEAKERS"),
            required("TONGUES_TEST_YOURTTS_LANGUAGES"),
            required("TONGUES_TEST_COQUI_SPEAKER_CONFIG"),
            required("TONGUES_TEST_COQUI_SPEAKER_MODEL"),
            NdArrayDevice::Cpu,
            SpeakerEmbeddingCachePolicy::Disabled,
        )
        .expect("published YourTTS graph");
        let mut plan = utterance_plan_from_text(SpeechRequest {
            text: "Hello.".into(),
            variety: "en-US".into(),
        })
        .unwrap();
        plan.speaker = Some(SpeakerId("male-en-2".into()));
        let waveform = engine
            .synthesize(&SpeechSynthesisRequest {
                plan,
                options: SynthesisOptions {
                    model_language: Some("en".into()),
                    noise_scale: Some(0.0),
                    noise_w: Some(0.0),
                    seed: Some(7),
                    ..SynthesisOptions::default()
                },
            })
            .expect("YourTTS synthesis");
        assert_eq!(waveform.contract.sample_rate_hz, 16_000);
        assert!(!waveform.samples.is_empty());
        assert!(waveform.samples.iter().all(|sample| sample.is_finite()));
    }

    fn json_numbers(value: &serde_json::Value) -> Vec<f32> {
        value
            .as_array()
            .expect("JSON array")
            .iter()
            .map(|value| value.as_f64().expect("JSON number") as f32)
            .collect()
    }

    fn assert_stage_probes(
        stage: &str,
        tensor: Tensor<TestBackend, 3>,
        reference: &serde_json::Value,
        tolerance: f32,
    ) {
        let [batch, channels, frames] = tensor.dims();
        assert_eq!(batch, 1, "{stage} batch");
        let values = tensor
            .into_data()
            .to_vec::<f32>()
            .expect("f32 conformance tensor");
        for probe in reference.as_array().expect("stage probes") {
            let probe = probe.as_array().expect("stage probe");
            let channel = probe[0].as_u64().expect("channel") as usize;
            let frame = probe[1].as_u64().expect("frame") as usize;
            let expected = probe[2].as_f64().expect("expected value") as f32;
            assert!(channel < channels && frame < frames, "{stage} probe bounds");
            let actual = values[channel * frames + frame];
            assert!(
                (actual - expected).abs() <= tolerance,
                "{stage}[{channel}, {frame}] differs: native {actual}, Coqui {expected}, tolerance {tolerance}"
            );
        }
    }

    #[test]
    #[ignore = "requires pinned Coqui VITS artifacts and reference evidence; run scripts/speech-conformance.sh"]
    fn published_checkpoint_stage_parity() {
        let config = std::env::var_os("TONGUES_TEST_COQUI_VITS_CONFIG")
            .expect("TONGUES_TEST_COQUI_VITS_CONFIG is required");
        let checkpoint = std::env::var_os("TONGUES_TEST_COQUI_VITS_CHECKPOINT")
            .expect("TONGUES_TEST_COQUI_VITS_CHECKPOINT is required");
        let speakers = std::env::var_os("TONGUES_TEST_COQUI_VITS_SPEAKERS")
            .expect("TONGUES_TEST_COQUI_VITS_SPEAKERS is required");
        let reference_path = std::env::var_os("TONGUES_TEST_COQUI_REFERENCE")
            .expect("TONGUES_TEST_COQUI_REFERENCE is required");
        let reference: serde_json::Value = serde_json::from_slice(
            &fs::read(&reference_path).expect("read Coqui reference evidence"),
        )
        .expect("parse Coqui reference evidence");
        let vits = &reference["vits"];
        assert_eq!(
            vits["noise_scale"].as_f64(),
            Some(0.0),
            "conformance reference must disable latent noise"
        );
        assert_eq!(
            vits["duration_noise_scale"].as_f64(),
            Some(0.0),
            "conformance reference must disable duration noise"
        );
        let token_ids = vits["token_ids"]
            .as_array()
            .expect("VITS token IDs")
            .iter()
            .map(|value| value.as_i64().expect("token ID"))
            .collect::<Vec<_>>();
        assert!(
            token_ids.len() > 32,
            "conformance sentence must be multiword"
        );

        let device = NdArrayDevice::Cpu;
        let engine =
            BurnVitsSpeech::<TestBackend>::load(config, checkpoint, speakers, device.clone())
                .expect("published VITS engine");
        let token_count = token_ids.len();

        for speaker in vits["speakers"].as_array().expect("speaker references") {
            let speaker_name = speaker["speaker"].as_str().expect("speaker name");
            let speaker_id = speaker["speaker_id"].as_u64().expect("speaker ID") as u32;
            assert_eq!(
                engine
                    .speakers
                    .as_ref()
                    .expect("learned speaker catalog")
                    .resolve(Some(&SpeakerId(speaker_name.into())), None)
                    .expect("resolve reference speaker"),
                speaker_id
            );
            let token_tensor = Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(token_ids.clone(), [1, token_count]),
                &device,
            );
            let lengths = Tensor::<TestBackend, 1, Int>::from_data(
                TensorData::new(vec![token_count as i64], [1]),
                &device,
            );
            let conditioning = engine
                .speaker_embedding
                .as_ref()
                .expect("learned speaker embedding")
                .forward(speaker_id, &device);
            assert_stage_probes(
                "speaker_embedding",
                conditioning.clone(),
                &speaker["stages"]["speaker_embedding"],
                2e-5,
            );

            let prior = engine
                .text_prior
                .forward(token_tensor, lengths)
                .expect("native text prior");
            assert_stage_probes(
                "encoded",
                prior.encoded.clone(),
                &speaker["stages"]["encoded"],
                2e-4,
            );
            assert_stage_probes(
                "prior_mean",
                prior.mean.clone(),
                &speaker["stages"]["prior_mean"],
                2e-4,
            );
            let zero_noise = Tensor::zeros([1, 2, token_count], &device);
            let log_durations = engine
                .duration_predictor
                .reverse_with_noise(
                    prior.encoded,
                    prior.mask.clone(),
                    Some(conditioning.clone()),
                    zero_noise,
                    0.0,
                )
                .expect("native deterministic durations");
            assert_stage_probes(
                "log_durations",
                log_durations.clone(),
                &speaker["stages"]["log_durations"],
                3e-4,
            );
            let rounded = ceil_durations(
                log_durations.exp(),
                prior.mask,
                f64::from(engine.config.network.length_scale),
                DEFAULT_MAX_OUTPUT_FRAMES,
            )
            .expect("rounded durations");
            let expected_durations = json_numbers(&speaker["durations"]);
            let actual_durations = rounded
                .values
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("duration values");
            assert_eq!(
                actual_durations, expected_durations,
                "duration mismatch for {speaker_name}"
            );
            let output_frames = speaker["output_frames"].as_u64().expect("output frames") as usize;
            assert_eq!(rounded.output_frames, output_frames);
            let expanded = expand_prior_statistics_with_frames(
                prior.mean,
                prior.log_scale,
                rounded.values,
                Some(output_frames),
                DEFAULT_MAX_OUTPUT_FRAMES,
            )
            .expect("expanded prior");
            assert_stage_probes(
                "expanded_mean",
                expanded.mean.clone(),
                &speaker["stages"]["expanded_mean"],
                3e-4,
            );
            let latent = engine
                .flow
                .reverse(
                    expanded.mean,
                    expanded.frame_mask.clone(),
                    Some(conditioning.clone()),
                )
                .expect("native reverse flow")
                * expanded.frame_mask;
            assert_stage_probes("latent", latent.clone(), &speaker["stages"]["latent"], 5e-4);
            let waveform = engine
                .waveform_decoder
                .forward(latent, Some(conditioning))
                .expect("native VITS decoder")
                .into_data()
                .to_vec::<f32>()
                .expect("native VITS samples");
            let waveform_reference = &speaker["waveform"];
            assert_eq!(
                waveform.len(),
                waveform_reference["samples"]
                    .as_u64()
                    .expect("sample count") as usize
            );
            assert!(waveform.iter().all(|sample| sample.is_finite()));
            let rms = (waveform.iter().map(|sample| sample * sample).sum::<f32>()
                / waveform.len() as f32)
                .sqrt();
            let expected_rms = waveform_reference["rms"].as_f64().expect("RMS") as f32;
            assert!(
                (rms - expected_rms).abs() <= 5e-4,
                "waveform RMS differs for {speaker_name}: native {rms}, Coqui {expected_rms}"
            );
            for probe in waveform_reference["probes"]
                .as_array()
                .expect("waveform probes")
            {
                let probe = probe.as_array().expect("waveform probe");
                let index = probe[0].as_u64().expect("sample index") as usize;
                let expected = probe[1].as_f64().expect("sample") as f32;
                let actual = waveform[index];
                assert!(
                    (actual - expected).abs() <= 2e-3,
                    "waveform[{index}] differs for {speaker_name}: native {actual}, Coqui {expected}"
                );
            }
        }
    }
}
