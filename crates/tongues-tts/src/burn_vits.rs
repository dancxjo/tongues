//! Burn-native end-to-end VITS speech synthesis.
//!
//! Tongues' linguistic plan remains the shared representation. The imported
//! model vocabulary, token IDs, speaker rows, and checkpoint layout are
//! resolved only inside this adapter.

use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor, TensorData};
use burn_store::{ModuleSnapshot, PytorchStore};
use speaking::UtterancePlan;

use crate::vits_config::ImportedVitsConfig;
use crate::vits_projector::VitsLinguisticProjector;
use crate::{
    ceil_durations, expand_prior_statistics, AudioChunk, AudioSink, ResidualCouplingFlow,
    ResidualCouplingFlowConfig, SpeakerCatalog, SpeechModelCapabilities, SpeechModelFamily,
    SpeechSynthesisEngine, SpeechSynthesisRequest, StochasticDurationConfig,
    StochasticDurationPredictor, VitsInferenceConfig, VitsTextPriorConfig, VitsTextPriorEncoder,
    VitsWaveformDecoder, VitsWaveformDecoderConfig, Waveform, WaveformContract,
};
use crate::{LinguisticProjector, ModelInputContract};

const DEFAULT_MAX_OUTPUT_FRAMES: usize = 65_536;
const STREAM_LATENT_FRAMES: usize = 64;
const STREAM_CONTEXT_FRAMES: usize = 32;

#[derive(Module, Debug)]
struct SpeakerEmbedding<B: Backend> {
    emb_g: Embedding<B>,
}

impl<B: Backend> SpeakerEmbedding<B> {
    fn load(
        num_speakers: usize,
        dimensions: usize,
        checkpoint_path: &Path,
        device: &B::Device,
    ) -> Result<Self> {
        let mut module = Self {
            emb_g: EmbeddingConfig::new(num_speakers, dimensions).init(device),
        };
        let mut store = PytorchStore::from_file(checkpoint_path)
            .with_top_level_key("model")
            .with_predicate(|path, _| path.starts_with("emb_g."))
            .map_indices_contiguous(false)
            .allow_partial(true)
            .skip_enum_variants(true);
        let result = module
            .load_from(&mut store)
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
        Ok(module)
    }

    fn forward(&self, speaker_id: u32, device: &B::Device) -> Tensor<B, 3> {
        let id = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![speaker_id as i64], [1, 1]),
            device,
        );
        self.emb_g.forward(id).swap_dims(1, 2)
    }
}

/// Native end-to-end VITS engine using Tongues plans and named speakers.
pub struct BurnVitsSpeech<B: Backend> {
    config: VitsInferenceConfig,
    projector: VitsLinguisticProjector,
    speakers: SpeakerCatalog,
    speaker_embedding: SpeakerEmbedding<B>,
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
            network.use_speaker_embedding,
            "this VITS engine requires learned speaker embeddings"
        );
        ensure!(
            !network.use_d_vector_file,
            "external d-vector conditioning is not supported by this VITS engine"
        );
        ensure!(
            !network.use_language_embedding,
            "language-conditioned VITS requires an explicit language input"
        );

        let projector = VitsLinguisticProjector::from_config(imported)?;
        let speakers = SpeakerCatalog::from_file(speaker_map_path, network.num_speakers)?;
        let speaker_channels = network.speaker_embedding_channels;
        let speaker_embedding = SpeakerEmbedding::load(
            network.num_speakers as usize,
            speaker_channels,
            checkpoint_path,
            &device,
        )?;
        let text_prior = VitsTextPriorConfig::from_model_config(&config)?
            .load_checkpoint(checkpoint_path, &device)?;

        let mut duration_config = StochasticDurationConfig::new(network.hidden_channels, 192, 3);
        duration_config.conditioning_channels = if network.condition_dp_on_speaker {
            speaker_channels
        } else {
            0
        };
        let duration_predictor = duration_config.load_checkpoint(checkpoint_path, &device)?;
        let flow = ResidualCouplingFlowConfig {
            channels: network.hidden_channels,
            hidden_channels: network.hidden_channels,
            kernel_size: network.kernel_size_flow,
            dilation_rate: network.dilation_rate_flow,
            num_layers: network.num_layers_flow,
            num_flows: 4,
            conditioning_channels: speaker_channels,
        }
        .load_checkpoint(checkpoint_path, &device)?;
        let waveform_decoder = VitsWaveformDecoderConfig::from_model_config(&config)?
            .load_checkpoint(checkpoint_path, &device)?;
        let output_contract = WaveformContract::mono(config.audio.sample_rate);

        Ok(Self {
            config,
            projector,
            speakers,
            speaker_embedding,
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
        &self.speakers
    }

    pub fn projected_input(&self, plan: &UtterancePlan) -> Result<crate::PhonemeTokenIds> {
        self.projector.project(plan)
    }

    pub fn synthesize(&mut self, request: &SpeechSynthesisRequest) -> Result<Waveform> {
        let mut samples = Vec::new();
        self.synthesize_streaming(request, &mut |chunk: AudioChunk| {
            samples.extend(chunk.pcm_mono_f32);
            Ok(())
        })?;
        let waveform = Waveform {
            contract: self.output_contract.clone(),
            samples,
        };
        waveform.validate()?;
        Ok(waveform)
    }

    fn prepare_latent(
        &self,
        request: &SpeechSynthesisRequest,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>)> {
        self.projector.contract().ensure_supports(&request.plan)?;
        ensure!(
            request.plan.speaker_reference.is_none(),
            "this VITS model does not accept reference audio"
        );
        let speaker_id = self
            .speakers
            .resolve(request.plan.speaker.as_ref(), request.options.speaker_id)?;
        let projected = self.projector.project(&request.plan)?;
        let token_count = projected.ids.len();
        let token_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(projected.ids, [1, token_count]),
            &self.device,
        );
        let lengths = Tensor::<B, 1, Int>::from_data(
            TensorData::new(vec![token_count as i64], [1]),
            &self.device,
        );
        let prior = self.text_prior.forward(token_ids, lengths)?;
        let conditioning = self.speaker_embedding.forward(speaker_id, &self.device);

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
        let log_durations = match request.options.seed {
            Some(seed) => self.duration_predictor.reverse_seeded(
                prior.encoded,
                prior.mask.clone(),
                duration_conditioning,
                f64::from(duration_noise),
                seed,
            )?,
            None => self.duration_predictor.reverse(
                prior.encoded,
                prior.mask.clone(),
                duration_conditioning,
                f64::from(duration_noise),
            )?,
        };
        let length_scale = request
            .options
            .length_scale
            .unwrap_or(self.config.network.length_scale);
        let rounded = ceil_durations(
            log_durations.exp(),
            prior.mask,
            f64::from(length_scale),
            DEFAULT_MAX_OUTPUT_FRAMES,
        )?;
        let expanded = expand_prior_statistics(
            prior.mean,
            prior.log_scale,
            rounded.values,
            DEFAULT_MAX_OUTPUT_FRAMES,
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
        Ok((latent, conditioning))
    }

    fn synthesize_streaming(
        &self,
        request: &SpeechSynthesisRequest,
        sink: &mut dyn AudioSink,
    ) -> Result<()> {
        let (latent, conditioning) = self.prepare_latent(request)?;
        let [batch, channels, frames] = latent.dims();
        ensure!(
            batch == 1,
            "streaming VITS currently requires batch size one"
        );
        let upsample = self.waveform_decoder.upsample_factor();
        let chunk_count = frames.div_ceil(STREAM_LATENT_FRAMES);

        for chunk_index in 0..chunk_count {
            let frame_start = chunk_index * STREAM_LATENT_FRAMES;
            let frame_end = (frame_start + STREAM_LATENT_FRAMES).min(frames);
            let context_start = frame_start.saturating_sub(STREAM_CONTEXT_FRAMES);
            let context_end = (frame_end + STREAM_CONTEXT_FRAMES).min(frames);
            let context_latent =
                latent
                    .clone()
                    .slice([0..batch, 0..channels, context_start..context_end]);
            let decoded = self
                .waveform_decoder
                .forward(context_latent, Some(conditioning.clone()))?;
            let expected_samples = (context_end - context_start) * upsample;
            ensure!(
                decoded.dims() == [1, 1, expected_samples],
                "VITS decoder emitted {:?}; expected [1, 1, {expected_samples}]",
                decoded.dims()
            );
            let sample_start = (frame_start - context_start) * upsample;
            let sample_end = (frame_end - context_start) * upsample;
            let samples = decoded
                .slice([0..1, 0..1, sample_start..sample_end])
                .into_data()
                .to_vec::<f32>()
                .context("VITS waveform output is not f32")?;
            sink.emit(AudioChunk {
                chunk_index,
                is_final: chunk_index + 1 == chunk_count,
                pause_after_ms: 0,
                sample_rate_hz: self.output_contract.sample_rate_hz,
                pcm_mono_f32: samples,
            })?;
        }
        Ok(())
    }
}

impl<B: Backend> SpeechSynthesisEngine for BurnVitsSpeech<B> {
    fn capabilities(&self) -> SpeechModelCapabilities {
        SpeechModelCapabilities {
            family: SpeechModelFamily::EndToEndSpeech,
            supports_named_speakers: true,
            supports_languages: false,
            supports_reference_audio: false,
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
        self.synthesize_streaming(request, sink)
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
    fn published_named_speakers_synthesize_streaming_when_available() {
        let (Some(config), Some(checkpoint), Some(speakers)) = (
            std::env::var_os("TONGUES_TEST_COQUI_VITS_CONFIG"),
            std::env::var_os("TONGUES_TEST_COQUI_VITS_CHECKPOINT"),
            std::env::var_os("TONGUES_TEST_COQUI_VITS_SPEAKERS"),
        ) else {
            return;
        };
        let mut engine =
            BurnVitsSpeech::<TestBackend>::load(config, checkpoint, speakers, NdArrayDevice::Cpu)
                .expect("published VITS engine");

        for (name, text) in [
            ("p225", "Morning light rested on cedar trees."),
            ("p226", "Rain polished the streets, and lamps glowed."),
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
}
