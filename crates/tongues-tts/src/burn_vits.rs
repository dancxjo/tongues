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
use crate::fairseq_vits::{
    FairseqVitsConfig, FairseqVitsProjector, FairseqVitsTokenizer, FAIRSEQ_MMS_CHECKPOINT,
    FAIRSEQ_MMS_CONFIG, FAIRSEQ_MMS_VOCAB,
};
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

#[derive(Debug, Clone)]
enum VitsProjector {
    Coqui(VitsLinguisticProjector),
    Fairseq(FairseqVitsProjector),
}

impl VitsProjector {
    fn contract(&self) -> &ModelInputContract {
        match self {
            Self::Coqui(projector) => projector.contract(),
            Self::Fairseq(projector) => projector.contract(),
        }
    }

    fn project(&self, plan: &UtterancePlan) -> Result<crate::PhonemeTokenIds> {
        match self {
            Self::Coqui(projector) => projector.project(plan),
            Self::Fairseq(projector) => projector.project(plan),
        }
    }
}

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
    projector: VitsProjector,
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
    /// Load an installed original Fairseq MMS VITS directory.
    ///
    /// The directory must contain the published `config.json`, `vocab.txt`,
    /// and `G_100000.pth` names. No Python or Fairseq runtime is involved.
    pub fn load_fairseq(
        model_dir: impl AsRef<Path>,
        language: impl Into<String>,
        device: B::Device,
    ) -> Result<Self> {
        Self::load_fairseq_internal(model_dir, language.into(), device, None)
    }

    pub fn load_fairseq_profiled(
        model_dir: impl AsRef<Path>,
        language: impl Into<String>,
        device: B::Device,
        profiler: &mut dyn FnMut(ModelLoadProfileEvent),
    ) -> Result<Self> {
        Self::load_fairseq_internal(model_dir, language.into(), device, Some(profiler))
    }

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

        let projector = VitsProjector::Coqui(VitsLinguisticProjector::from_config(imported)?);
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

    fn load_fairseq_internal(
        model_dir: impl AsRef<Path>,
        language: String,
        device: B::Device,
        profiler: Option<&mut dyn FnMut(ModelLoadProfileEvent)>,
    ) -> Result<Self> {
        let mut profiler = profiler;
        let started = Instant::now();
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join(FAIRSEQ_MMS_CONFIG);
        let vocab_path = model_dir.join(FAIRSEQ_MMS_VOCAB);
        let checkpoint_path = model_dir.join(FAIRSEQ_MMS_CHECKPOINT);
        let source = fs::read_to_string(&config_path).with_context(|| {
            format!(
                "failed to read Fairseq MMS config {}",
                config_path.display()
            )
        })?;
        let fairseq_config = FairseqVitsConfig::from_json_str(&source)?;
        let tokenizer = FairseqVitsTokenizer::from_file(
            language,
            &vocab_path,
            fairseq_config.add_blank(),
            fairseq_config.preprocessing(),
        )?;
        let config = fairseq_config.inference_config(tokenizer.symbols().len())?;
        let projector = VitsProjector::Fairseq(FairseqVitsProjector::new(tokenizer)?);
        let network = &config.network;
        let text_config = VitsTextPriorConfig::from_model_config(&config)?;
        let duration_config = StochasticDurationConfig::new(text_config.encoder_channels(), 192, 3);
        let flow_config = ResidualCouplingFlowConfig {
            channels: network.hidden_channels,
            hidden_channels: network.hidden_channels,
            kernel_size: network.kernel_size_flow,
            dilation_rate: network.dilation_rate_flow,
            num_layers: network.num_layers_flow,
            num_flows: 4,
            conditioning_channels: 0,
        };
        let decoder_config = VitsWaveformDecoderConfig::from_model_config(&config)?;
        let output_contract = WaveformContract::mono(config.audio.sample_rate);
        record_load_stage(
            &mut profiler,
            ModelLoadStage::ConfigCheckpointParsing,
            started,
            Some("fairseq-mms-vits"),
        );

        let started = Instant::now();
        let text_prior = text_config.init(&device)?;
        let duration_predictor = duration_config.init(&device)?;
        let flow = flow_config.init(&device)?;
        let waveform_decoder = decoder_config.init(&device)?;
        record_load_stage(
            &mut profiler,
            ModelLoadStage::ModelConstruction,
            started,
            Some("fairseq-mms-vits"),
        );

        let started = Instant::now();
        let text_prior = text_prior.load_checkpoint_with_prefix(&checkpoint_path, "enc_p")?;
        let duration_predictor = duration_predictor.load_fairseq_checkpoint(&checkpoint_path)?;
        let flow = flow.load_checkpoint(&checkpoint_path)?;
        let waveform_decoder = waveform_decoder.load_checkpoint_subtree(&checkpoint_path, "dec")?;
        record_load_stage(
            &mut profiler,
            ModelLoadStage::WeightUpload,
            started,
            Some("fairseq-mms-vits"),
        );

        Ok(Self {
            config,
            projector,
            speakers: None,
            d_vectors: None,
            languages: None,
            speaker_embedding: None,
            reference_encoder: None,
            language_embedding: None,
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

    fn speaker_conditioning(
        &mut self,
        request: &SpeechSynthesisRequest,
    ) -> Result<Option<Tensor<B, 3>>> {
        if let (Some(speakers), Some(embedding)) = (&self.speakers, &self.speaker_embedding) {
            ensure!(
                request.plan.speaker_reference.is_none(),
                "learned-speaker VITS does not accept a speaker reference"
            );
            let speaker_id =
                speakers.resolve(request.plan.speaker.as_ref(), request.options.speaker_id)?;
            return Ok(Some(embedding.forward(speaker_id, &self.device)));
        }

        if self.d_vectors.is_none() {
            ensure!(
                request.plan.speaker.is_none()
                    && request.plan.speaker_reference.is_none()
                    && request.options.speaker_id.is_none(),
                "single-speaker Fairseq MMS VITS does not accept speaker selection or reference audio"
            );
            return Ok(None);
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
        Ok(Some(Tensor::from_data(
            TensorData::new(embedding.values, [1, expected.dimensions, 1]),
            &self.device,
        )))
    }

    fn prepare_latent(
        &mut self,
        request: &SpeechSynthesisRequest,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<(Tensor<B, 3>, Option<Tensor<B, 3>>)> {
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
                SynthesisDimension::new(
                    "speaker_channels",
                    conditioning.as_ref().map_or(0, |value| value.dims()[1]),
                ),
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
            .then(|| conditioning.clone())
            .flatten();
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
            conditioning.clone(),
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
        let decoded = self.waveform_decoder.forward(latent, conditioning)?;
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
            supports_named_speakers: self.speakers.is_some() || self.d_vectors.is_some(),
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
    use std::path::PathBuf;

    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use speaking::{SpeakerId, SpeakerReference, SpeakerReferenceSource};

    use super::*;
    use crate::{cosine_similarity, utterance_plan_from_text, SpeechRequest, SynthesisOptions};

    type TestBackend = NdArray<f32>;

    #[test]
    #[ignore = "requires a pinned original Fairseq MMS model directory"]
    fn published_fairseq_mms_checkpoint_loads_and_synthesizes_without_python() {
        let model_dir = std::env::var_os("TONGUES_TEST_FAIRSEQ_MMS_MODEL_DIR")
            .expect("TONGUES_TEST_FAIRSEQ_MMS_MODEL_DIR is required");
        let language =
            std::env::var("TONGUES_TEST_FAIRSEQ_MMS_LANGUAGE").unwrap_or_else(|_| "eng".into());
        let mut engine =
            BurnVitsSpeech::<TestBackend>::load_fairseq(model_dir, language, NdArrayDevice::Cpu)
                .expect("published Fairseq MMS engine");
        let plan = utterance_plan_from_text(SpeechRequest {
            text: "This is a test.".into(),
            variety: "en-US".into(),
        })
        .expect("native linguistic plan");
        let projected = engine.projected_input(&plan).expect("MMS tokenization");
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../fixtures/speech/fairseq-mms-vits-conformance.json"
        ))
        .expect("Fairseq MMS conformance fixture");
        let expected_ids = fixture["tokenization"][0]["token_ids"]
            .as_array()
            .expect("English token ids")
            .iter()
            .map(|value| value.as_i64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(projected.ids, expected_ids);
        let request = SpeechSynthesisRequest {
            plan,
            options: SynthesisOptions {
                seed: Some(7),
                ..Default::default()
            },
        };
        let waveform = engine.synthesize(&request).expect("native MMS waveform");
        assert_eq!(waveform.contract.sample_rate_hz, 16_000);
        assert!(!waveform.samples.is_empty());
        assert!(waveform.samples.iter().all(|sample| sample.is_finite()));
        let minimum = waveform
            .samples
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let maximum = waveform
            .samples
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let rms = (waveform
            .samples
            .iter()
            .map(|sample| sample * sample)
            .sum::<f32>()
            / waveform.samples.len() as f32)
            .sqrt();
        let reference = &fixture["waveform_probe"];
        let reference_samples = reference["sample_count"].as_u64().unwrap() as f32;
        let duration_delta =
            (waveform.samples.len() as f32 - reference_samples).abs() / reference_samples;
        assert!(
            duration_delta < 0.15,
            "native/reference duration drift is {duration_delta:.3}"
        );
        for (name, actual) in [("minimum", minimum), ("maximum", maximum), ("rms", rms)] {
            let expected = reference[name].as_f64().unwrap() as f32;
            assert!(
                (actual - expected).abs() < 0.05,
                "native {name} {actual} differs from reference {expected}"
            );
        }
    }

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

    fn required_env(name: &str) -> std::ffi::OsString {
        std::env::var_os(name).unwrap_or_else(|| panic!("{name} is required"))
    }

    fn json_f32_array(value: &serde_json::Value) -> Vec<f32> {
        value
            .as_array()
            .expect("JSON float array")
            .iter()
            .map(|value| value.as_f64().expect("JSON float") as f32)
            .collect()
    }

    fn assert_embedding_parity(
        label: &str,
        actual: &ConditioningEmbedding,
        expected: &serde_json::Value,
    ) {
        const ABSOLUTE_TOLERANCE: f32 = 3.0e-4;
        const COSINE_TOLERANCE: f32 = 1.0e-4;

        let expected = ConditioningEmbedding {
            contract: actual.contract.clone(),
            values: json_f32_array(expected),
        };
        expected.validate().expect("valid golden embedding");
        assert_eq!(actual.values.len(), expected.values.len());
        let (max_index, max_absolute_error) = actual
            .values
            .iter()
            .zip(&expected.values)
            .enumerate()
            .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("non-empty embedding");
        let cosine = cosine_similarity(actual, &expected).expect("embedding cosine");
        assert!(
            max_absolute_error <= ABSOLUTE_TOLERANCE,
            "{label} embedding maximum error is {max_absolute_error} at index {max_index}, tolerance {ABSOLUTE_TOLERANCE}; cosine {cosine}"
        );
        assert!(
            1.0 - cosine <= COSINE_TOLERANCE,
            "{label} embedding cosine {cosine} is below {}",
            1.0 - COSINE_TOLERANCE
        );
    }

    fn assert_waveform_parity(label: &str, actual: &Waveform, expected: &serde_json::Value) {
        const RMS_ABSOLUTE_TOLERANCE: f32 = 5.0e-4;
        const SAMPLE_ABSOLUTE_TOLERANCE: f32 = 2.0e-3;

        assert_eq!(
            actual.contract.sample_rate_hz,
            expected["sample_rate_hz"].as_u64().expect("sample rate") as u32
        );
        assert_eq!(actual.contract.channels, 1);
        assert_eq!(
            actual.samples.len(),
            expected["samples"].as_u64().expect("sample count") as usize,
            "{label} sample count"
        );
        assert!(actual.samples.iter().all(|sample| sample.is_finite()));
        let rms = (actual
            .samples
            .iter()
            .map(|sample| sample * sample)
            .sum::<f32>()
            / actual.samples.len() as f32)
            .sqrt();
        let expected_rms = expected["rms"].as_f64().expect("waveform RMS") as f32;
        assert!(
            (rms - expected_rms).abs() <= RMS_ABSOLUTE_TOLERANCE,
            "{label} waveform RMS differs: native {rms}, Coqui {expected_rms}, tolerance {RMS_ABSOLUTE_TOLERANCE}"
        );
        for probe in expected["probes"].as_array().expect("waveform probes") {
            let probe = probe.as_array().expect("waveform probe");
            let index = probe[0].as_u64().expect("sample index") as usize;
            let expected = probe[1].as_f64().expect("sample") as f32;
            let actual = actual.samples[index];
            assert!(
                (actual - expected).abs() <= SAMPLE_ABSOLUTE_TOLERANCE,
                "{label} waveform[{index}] differs: native {actual}, Coqui {expected}, tolerance {SAMPLE_ABSOLUTE_TOLERANCE}"
            );
        }
    }

    #[test]
    #[ignore = "requires pinned YourTTS artifacts and pinned Coqui reference evidence; run scripts/speech-conformance.sh"]
    fn published_yourtts_conformance() {
        let reference: serde_json::Value = serde_json::from_slice(
            &fs::read(required_env("TONGUES_TEST_COQUI_REFERENCE"))
                .expect("read Coqui reference evidence"),
        )
        .expect("parse Coqui reference evidence");
        let yourtts = &reference["yourtts"];
        assert_eq!(yourtts["noise_scale"].as_f64(), Some(0.0));
        assert_eq!(yourtts["duration_noise_scale"].as_f64(), Some(0.0));
        assert_eq!(yourtts["embedding_dimensions"].as_u64(), Some(512));

        let reference_wav = PathBuf::from(required_env("TONGUES_TEST_YOURTTS_REFERENCE_WAV"));
        let mut engine = BurnVitsSpeech::<TestBackend>::load_your_tts(
            required_env("TONGUES_TEST_YOURTTS_CONFIG"),
            required_env("TONGUES_TEST_YOURTTS_CHECKPOINT"),
            required_env("TONGUES_TEST_YOURTTS_SPEAKERS"),
            required_env("TONGUES_TEST_YOURTTS_LANGUAGES"),
            required_env("TONGUES_TEST_COQUI_SPEAKER_CONFIG"),
            required_env("TONGUES_TEST_COQUI_SPEAKER_MODEL"),
            NdArrayDevice::Cpu,
            SpeakerEmbeddingCachePolicy::Memory { max_entries: 2 },
        )
        .expect("published YourTTS graph");

        let verification = &yourtts["verification"];
        let same_clips = verification["same_speaker"]["clips"]
            .as_array()
            .expect("same-speaker clips");
        let different_clips = verification["different_speaker"]["clips"]
            .as_array()
            .expect("different-speaker clips");
        let catalog = engine.d_vector_catalog().expect("d-vector catalog");
        let same = cosine_similarity(
            &catalog
                .embedding_for_clip(same_clips[0].as_str().expect("clip"))
                .expect("same-speaker clip one"),
            &catalog
                .embedding_for_clip(same_clips[1].as_str().expect("clip"))
                .expect("same-speaker clip two"),
        )
        .expect("same-speaker cosine");
        let different = cosine_similarity(
            &catalog
                .embedding_for_clip(different_clips[0].as_str().expect("clip"))
                .expect("different-speaker clip one"),
            &catalog
                .embedding_for_clip(different_clips[1].as_str().expect("clip"))
                .expect("different-speaker clip two"),
        )
        .expect("different-speaker cosine");
        assert!(
            (same
                - verification["same_speaker"]["cosine"]
                    .as_f64()
                    .expect("same cosine") as f32)
                .abs()
                <= 1.0e-6
        );
        assert!(
            (different
                - verification["different_speaker"]["cosine"]
                    .as_f64()
                    .expect("different cosine") as f32)
                .abs()
                <= 1.0e-6
        );
        assert!(same > different);

        for case in yourtts["cases"].as_array().expect("YourTTS cases") {
            let label = case["id"].as_str().expect("case ID");
            let text = case["text"].as_str().expect("case text");
            let variety = case["variety"].as_str().expect("case variety");
            let language = case["language"].as_str().expect("case language");
            let speaker = &case["speaker"];
            let mut plan = utterance_plan_from_text(SpeechRequest {
                text: text.into(),
                variety: variety.into(),
            })
            .expect("native linguistic plan");
            let actual_embedding = match speaker["kind"].as_str().expect("speaker kind") {
                "named" => {
                    let name = speaker["name"].as_str().expect("speaker name");
                    plan.speaker = Some(SpeakerId(name.into()));
                    engine
                        .d_vector_catalog()
                        .expect("d-vector catalog")
                        .resolve(name)
                        .expect("named golden embedding")
                }
                "reference_wav" => {
                    plan.speaker_reference = Some(SpeakerReference {
                        description: Some("pinned LJSpeech reference fixture".into()),
                        source: SpeakerReferenceSource::ReferenceAudio {
                            uri: reference_wav.display().to_string(),
                        },
                    });
                    engine
                        .reference_encoder
                        .as_mut()
                        .expect("native reference encoder")
                        .encode_path(&reference_wav)
                        .expect("native reference embedding")
                }
                kind => panic!("unknown conformance speaker kind {kind}"),
            };
            assert_embedding_parity(label, &actual_embedding, &case["embedding"]);

            let projected = engine
                .projected_input(&plan)
                .expect("YourTTS checkpoint projection");
            let expected_tokens = case["token_ids"]
                .as_array()
                .expect("token IDs")
                .iter()
                .map(|value| value.as_i64().expect("token ID"))
                .collect::<Vec<_>>();
            assert_eq!(projected.ids, expected_tokens, "{label} token IDs");

            let waveform = engine
                .synthesize(&SpeechSynthesisRequest {
                    plan,
                    options: SynthesisOptions {
                        model_language: Some(language.into()),
                        noise_scale: Some(0.0),
                        noise_w: Some(0.0),
                        seed: Some(27),
                        ..SynthesisOptions::default()
                    },
                })
                .expect("YourTTS conformance synthesis");
            assert_waveform_parity(label, &waveform, &case["waveform"]);
        }
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
