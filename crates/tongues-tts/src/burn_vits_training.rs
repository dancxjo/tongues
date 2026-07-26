//! Burn-native training-only VITS components and loss contract.
//!
//! This module deliberately builds on the inference graph instead of copying
//! it. A trainable generator owns the same text encoder, stochastic-duration
//! prior, residual flow, and waveform decoder that [`crate::BurnVitsSpeech`]
//! consumes. Only the posterior encoder and discriminators are training-only.

use std::f64::consts::PI;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, PaddingConfig1d};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor, TensorData};
use burn_store::{BurnToPyTorchAdapter, ModuleSnapshot, SafetensorsStore};

use crate::burn_vocoder_discriminators::{MultiPeriodDiscriminator, MultiScaleDiscriminator};
use crate::burn_vocoder_losses::{
    adversarial_discriminator_loss, adversarial_generator_loss, feature_matching_loss,
    mel_spectrogram_loss,
};
use crate::{
    ResidualCouplingFlow, ResidualCouplingFlowConfig, StochasticDurationConfig,
    StochasticDurationPredictor, VitsInferenceConfig, VitsTextPriorConfig, VitsTextPriorEncoder,
    VitsWaveformDecoder, VitsWaveformDecoderConfig,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VitsPosteriorEncoderConfig {
    pub input_channels: usize,
    pub latent_channels: usize,
    pub hidden_channels: usize,
    pub kernel_size: usize,
    pub dilation_rate: usize,
    pub num_layers: usize,
    pub conditioning_channels: usize,
}

impl VitsPosteriorEncoderConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.input_channels > 0 && self.latent_channels > 0 && self.hidden_channels > 0,
            "VITS posterior channel dimensions must be positive"
        );
        ensure!(
            self.kernel_size > 0 && !self.kernel_size.is_multiple_of(2),
            "VITS posterior kernel size must be positive and odd"
        );
        ensure!(
            self.dilation_rate > 0 && self.num_layers > 0,
            "VITS posterior dilation and layer count must be positive"
        );
        Ok(())
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<VitsPosteriorEncoder<B>> {
        self.validate()?;
        let pre = Conv1dConfig::new(self.input_channels, self.hidden_channels, 1).init(device);
        let mut convs = Vec::with_capacity(self.num_layers);
        for layer in 0..self.num_layers {
            let dilation = self
                .dilation_rate
                .checked_pow(layer as u32)
                .context("VITS posterior dilation overflow")?;
            let padding = dilation * (self.kernel_size - 1) / 2;
            convs.push(
                Conv1dConfig::new(self.hidden_channels, self.hidden_channels, self.kernel_size)
                    .with_dilation(dilation)
                    .with_padding(PaddingConfig1d::Explicit(padding, padding))
                    .init(device),
            );
        }
        let proj =
            Conv1dConfig::new(self.hidden_channels, self.latent_channels * 2, 1).init(device);
        let cond = (self.conditioning_channels > 0).then(|| {
            Conv1dConfig::new(self.conditioning_channels, self.hidden_channels, 1).init(device)
        });
        Ok(VitsPosteriorEncoder {
            pre,
            convs,
            proj,
            cond,
            input_channels: self.input_channels,
            latent_channels: self.latent_channels,
            hidden_channels: self.hidden_channels,
            conditioning_channels: self.conditioning_channels,
        })
    }
}

#[derive(Debug)]
pub struct VitsPosteriorOutput<B: Backend> {
    pub latent: Tensor<B, 3>,
    pub mean: Tensor<B, 3>,
    pub log_scale: Tensor<B, 3>,
    pub mask: Tensor<B, 3>,
}

/// Training-only posterior encoder from linear spectrograms to VITS latents.
#[derive(Module, Debug)]
pub struct VitsPosteriorEncoder<B: Backend> {
    pub pre: Conv1d<B>,
    pub convs: Vec<Conv1d<B>>,
    pub proj: Conv1d<B>,
    pub cond: Option<Conv1d<B>>,
    input_channels: usize,
    latent_channels: usize,
    hidden_channels: usize,
    conditioning_channels: usize,
}

impl<B: Backend> VitsPosteriorEncoder<B> {
    pub fn forward(
        &self,
        spectrogram: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 3>>,
        noise_scale: f64,
        seed: u64,
    ) -> Result<VitsPosteriorOutput<B>> {
        ensure!(
            noise_scale.is_finite() && noise_scale >= 0.0,
            "posterior noise scale must be finite and non-negative"
        );
        let [batch, channels, frames] = spectrogram.dims();
        ensure!(
            channels == self.input_channels && batch > 0 && frames > 0,
            "posterior input must be non-empty with {} channels; got [{batch}, {channels}, {frames}]",
            self.input_channels
        );
        ensure!(
            mask.dims() == [batch, 1, frames],
            "posterior mask shape {:?}; expected [{batch}, 1, {frames}]",
            mask.dims()
        );
        let mut hidden = self.pre.forward(spectrogram) * mask.clone();
        match (&self.cond, conditioning) {
            (None, None) => {}
            (None, Some(_)) => anyhow::bail!(
                "speaker conditioning was supplied to an unconditioned posterior encoder"
            ),
            (Some(_), None) => {
                anyhow::bail!("speaker conditioning is required by this posterior encoder")
            }
            (Some(projection), Some(conditioning)) => {
                let [cond_batch, cond_channels, cond_frames] = conditioning.dims();
                ensure!(
                    cond_batch == batch
                        && cond_channels == self.conditioning_channels
                        && (cond_frames == 1 || cond_frames == frames),
                    "posterior conditioning shape {:?}; expected [{batch}, {}, 1 or {frames}]",
                    conditioning.dims(),
                    self.conditioning_channels
                );
                let conditioning = projection.forward(conditioning);
                hidden = hidden
                    + if cond_frames == 1 {
                        conditioning.repeat_dim(2, frames)
                    } else {
                        conditioning
                    };
            }
        }
        for conv in &self.convs {
            hidden = (hidden.clone() + relu(conv.forward(hidden))) * mask.clone();
        }
        let stats = self.proj.forward(hidden) * mask.clone();
        let mean = stats
            .clone()
            .slice([0..batch, 0..self.latent_channels, 0..frames]);
        let log_scale = stats.slice([
            0..batch,
            self.latent_channels..self.latent_channels * 2,
            0..frames,
        ]);
        let device = mean.device();
        B::seed(&device, seed);
        let noise = Tensor::random(
            [batch, self.latent_channels, frames],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let latent = (mean.clone() + noise * log_scale.clone().exp() * noise_scale) * mask.clone();
        Ok(VitsPosteriorOutput {
            latent,
            mean,
            log_scale,
            mask,
        })
    }

    pub fn hidden_channels(&self) -> usize {
        self.hidden_channels
    }
}

/// Shared inference modules embedded in a trainable VITS generator.
///
/// Saving this module through [`Self::save_inference_safetensors`] emits the
/// exact root names expected by [`crate::BurnVitsSpeech::load`].
#[derive(Module, Debug)]
pub struct VitsInferenceExport<B: Backend> {
    pub text_encoder: VitsTextPriorEncoder<B>,
    pub duration_predictor: StochasticDurationPredictor<B>,
    pub flow: ResidualCouplingFlow<B>,
    pub waveform_decoder: VitsWaveformDecoder<B>,
    pub emb_g: Option<Embedding<B>>,
    pub emb_l: Option<Embedding<B>>,
}

impl<B: Backend> VitsInferenceExport<B> {
    /// Construct a randomly initialized graph for from-scratch training.
    pub fn init_random(config: &VitsInferenceConfig, device: &B::Device) -> Result<Self> {
        config.validate()?;
        ensure!(
            !config.network.use_d_vector_file,
            "native VITS training currently requires learned speaker IDs; reference d-vector fine-tuning is not yet supported"
        );
        let network = &config.network;
        let text_config = VitsTextPriorConfig::from_model_config(config)?;
        let mut duration_config =
            StochasticDurationConfig::new(text_config.encoder_channels(), 192, 3);
        let speaker_channels = if network.use_speaker_embedding {
            network.speaker_embedding_channels
        } else {
            0
        };
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
        let decoder_config = VitsWaveformDecoderConfig::from_model_config(config)?;
        Ok(Self {
            text_encoder: text_config.init(device)?,
            duration_predictor: duration_config.init(device)?,
            flow: flow_config.init(device)?,
            waveform_decoder: decoder_config.init(device)?,
            emb_g: network.use_speaker_embedding.then(|| {
                EmbeddingConfig::new(
                    network.num_speakers as usize,
                    network.speaker_embedding_channels,
                )
                .init(device)
            }),
            emb_l: network.use_language_embedding.then(|| {
                EmbeddingConfig::new(
                    network.num_languages as usize,
                    network.embedded_language_dim,
                )
                .init(device)
            }),
        })
    }

    /// Restore all inference-side parameters from a published Coqui VITS
    /// checkpoint. Training-only posterior/discriminator modules are
    /// intentionally constructed separately.
    pub fn load_coqui_checkpoint(
        config: &VitsInferenceConfig,
        checkpoint: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<Self> {
        config.validate()?;
        let network = &config.network;
        let text_config = VitsTextPriorConfig::from_model_config(config)?;
        let mut duration_config =
            StochasticDurationConfig::new(text_config.encoder_channels(), 192, 3);
        let speaker_channels = if network.use_speaker_embedding {
            network.speaker_embedding_channels
        } else if network.use_d_vector_file {
            network.d_vector_dim
        } else {
            0
        };
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
        let decoder_config = VitsWaveformDecoderConfig::from_model_config(config)?;
        let checkpoint = checkpoint.as_ref();
        let text_encoder = text_config.init(device)?.load_checkpoint(checkpoint)?;
        let duration_predictor = duration_config.init(device)?.load_checkpoint(checkpoint)?;
        let flow = flow_config.init(device)?.load_checkpoint(checkpoint)?;
        let waveform_decoder = decoder_config.init(device)?.load_checkpoint(checkpoint)?;
        let emb_g = if network.use_speaker_embedding {
            Some(load_speaker_embedding(
                network.num_speakers as usize,
                network.speaker_embedding_channels,
                checkpoint,
                device,
            )?)
        } else {
            None
        };
        let emb_l = if network.use_language_embedding {
            Some(load_language_embedding(
                network.num_languages as usize,
                network.embedded_language_dim,
                checkpoint,
                device,
            )?)
        } else {
            None
        };
        Ok(Self {
            text_encoder,
            duration_predictor,
            flow,
            waveform_decoder,
            emb_g,
            emb_l,
        })
    }

    pub fn speaker_conditioning(
        &self,
        speaker_ids: Option<Tensor<B, 2, Int>>,
    ) -> Result<Option<Tensor<B, 3>>> {
        match (&self.emb_g, speaker_ids) {
            (None, None) => Ok(None),
            (None, Some(_)) => anyhow::bail!(
                "speaker IDs were supplied to a VITS model without learned speaker embeddings"
            ),
            (Some(_), None) => {
                anyhow::bail!("speaker IDs are required by this VITS model during training")
            }
            (Some(embedding), Some(ids)) => Ok(Some(embedding.forward(ids).swap_dims(1, 2))),
        }
    }

    pub fn language_conditioning(
        &self,
        language_ids: Option<Tensor<B, 2, Int>>,
    ) -> Result<Option<Tensor<B, 3>>> {
        match (&self.emb_l, language_ids) {
            (None, None) => Ok(None),
            (None, Some(_)) => anyhow::bail!(
                "language IDs were supplied to a VITS model without language embeddings"
            ),
            (Some(_), None) => {
                anyhow::bail!("language IDs are required by this VITS model during training")
            }
            (Some(embedding), Some(ids)) => Ok(Some(embedding.forward(ids).swap_dims(1, 2))),
        }
    }

    pub fn save_inference_safetensors(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let mut store = SafetensorsStore::from_file(path)
            .overwrite(true)
            .skip_enum_variants(true)
            .with_to_adapter(BurnToPyTorchAdapter)
            .with_predicate(inference_export_tensor)
            .with_key_remapping(
                r"^duration_predictor\.affine\.",
                "duration_predictor.flows.0.",
            )
            .with_key_remapping(r"^waveform_decoder\.generator\.", "waveform_decoder.")
            .with_key_remapping(
                r"^waveform_decoder\.(conv_pre|conv_post)\.weight_v$",
                "waveform_decoder.$1.weight",
            );
        for index in 0..4 {
            store = store.with_key_remapping(
                format!(r"^duration_predictor\.spline_flows\.{index}\."),
                format!("duration_predictor.flows.{}.", index + 1),
            );
        }
        self.save_into(&mut store)
            .with_context(|| format!("writing VITS inference checkpoint {}", path.display()))
    }
}

#[derive(Module, Debug)]
struct SpeakerEmbeddingCheckpoint<B: Backend> {
    emb_g: Embedding<B>,
}

#[derive(Module, Debug)]
struct LanguageEmbeddingCheckpoint<B: Backend> {
    emb_l: Embedding<B>,
}

fn load_speaker_embedding<B: Backend>(
    count: usize,
    dimensions: usize,
    checkpoint: &Path,
    device: &B::Device,
) -> Result<Embedding<B>> {
    let mut module = SpeakerEmbeddingCheckpoint {
        emb_g: EmbeddingConfig::new(count, dimensions).init(device),
    };
    let result = crate::checkpoint::load_pytorch_layout_checkpoint(
        &mut module,
        checkpoint,
        crate::checkpoint::CheckpointLoadOptions {
            top_level_key: Some("model"),
            predicate: Some(speaker_embedding_tensor),
            map_indices_contiguous: false,
            allow_partial: true,
            skip_enum_variants: true,
            ..Default::default()
        },
    )?;
    ensure!(
        result.missing.is_empty()
            && result.errors.is_empty()
            && result.unused.iter().all(|path| !path.starts_with("emb_g.")),
        "published speaker embedding does not exactly match the trainable VITS graph"
    );
    Ok(module.emb_g)
}

fn load_language_embedding<B: Backend>(
    count: usize,
    dimensions: usize,
    checkpoint: &Path,
    device: &B::Device,
) -> Result<Embedding<B>> {
    let mut module = LanguageEmbeddingCheckpoint {
        emb_l: EmbeddingConfig::new(count, dimensions).init(device),
    };
    let result = crate::checkpoint::load_pytorch_layout_checkpoint(
        &mut module,
        checkpoint,
        crate::checkpoint::CheckpointLoadOptions {
            top_level_key: Some("model"),
            predicate: Some(language_embedding_tensor),
            map_indices_contiguous: false,
            allow_partial: true,
            skip_enum_variants: true,
            ..Default::default()
        },
    )?;
    ensure!(
        result.missing.is_empty()
            && result.errors.is_empty()
            && result.unused.iter().all(|path| !path.starts_with("emb_l.")),
        "published language embedding does not exactly match the trainable VITS graph"
    );
    Ok(module.emb_l)
}

fn speaker_embedding_tensor(path: &str, _container: &str) -> bool {
    path.starts_with("emb_g.")
}

fn language_embedding_tensor(path: &str, _container: &str) -> bool {
    path.starts_with("emb_l.")
}

fn inference_export_tensor(path: &str, _container: &str) -> bool {
    (path.starts_with("text_encoder.")
        || path.starts_with("duration_predictor.")
        || path.starts_with("flow.")
        || path.starts_with("waveform_decoder.")
        || path.starts_with("emb_g.")
        || path.starts_with("emb_l."))
        && !path.ends_with("conv_pre.weight_g")
        && !path.ends_with("conv_post.weight_g")
}

#[derive(Module, Debug)]
pub struct VitsTrainableGenerator<B: Backend> {
    pub inference: VitsInferenceExport<B>,
    pub posterior_encoder: VitsPosteriorEncoder<B>,
}

#[derive(Debug, Clone)]
pub struct VitsTrainingBatch<B: Backend> {
    pub token_ids: Tensor<B, 2, Int>,
    pub token_lengths: Vec<usize>,
    /// Linear spectrogram `[batch, fft_bins, frames]`.
    pub spectrogram: Tensor<B, 3>,
    pub frame_lengths: Vec<usize>,
    /// Waveform `[batch, 1, samples]`.
    pub waveform: Tensor<B, 3>,
    pub speaker_ids: Option<Tensor<B, 2, Int>>,
    pub language_ids: Option<Tensor<B, 2, Int>>,
}

#[derive(Debug)]
pub struct VitsTrainingForward<B: Backend> {
    pub generated_waveform: Tensor<B, 3>,
    pub target_waveform: Tensor<B, 3>,
    pub alignment: Tensor<B, 3>,
    pub predicted_log_duration: Tensor<B, 2>,
    pub target_log_duration: Tensor<B, 2>,
    pub duration_loss: Tensor<B, 1>,
    pub kl_loss: Tensor<B, 1>,
    pub segment_starts: Vec<usize>,
}

impl<B: Backend> VitsTrainableGenerator<B> {
    pub fn init_random(config: &VitsInferenceConfig, device: &B::Device) -> Result<Self> {
        let conditioning_channels = if config.network.use_speaker_embedding {
            config.network.speaker_embedding_channels
        } else {
            0
        };
        Ok(Self {
            inference: VitsInferenceExport::init_random(config, device)?,
            posterior_encoder: VitsPosteriorEncoderConfig {
                input_channels: config.network.out_channels,
                latent_channels: config.network.hidden_channels,
                hidden_channels: config.network.hidden_channels,
                kernel_size: config.network.kernel_size_posterior_encoder,
                dilation_rate: config.network.dilation_rate_posterior_encoder,
                num_layers: config.network.num_layers_posterior_encoder,
                conditioning_channels,
            }
            .init(device)?,
        })
    }

    pub fn training_forward(
        &self,
        batch: VitsTrainingBatch<B>,
        segment_frames: usize,
        hop_length: usize,
        seed: u64,
    ) -> Result<VitsTrainingForward<B>> {
        ensure!(
            segment_frames > 0,
            "VITS training segment must be non-empty"
        );
        ensure!(hop_length > 0, "VITS hop length must be positive");
        let [batch_size, _, frames] = batch.spectrogram.dims();
        ensure!(
            batch.token_lengths.len() == batch_size && batch.frame_lengths.len() == batch_size,
            "VITS length vectors must match batch size {batch_size}"
        );
        ensure!(
            batch.frame_lengths.iter().all(|length| *length <= frames),
            "a VITS frame length exceeds the padded spectrogram"
        );

        let speaker = self.inference.speaker_conditioning(batch.speaker_ids)?;
        let language = self.inference.language_conditioning(batch.language_ids)?;
        let device = batch.spectrogram.device();
        let token_lengths_i64 = batch
            .token_lengths
            .iter()
            .map(|length| *length as i64)
            .collect::<Vec<_>>();
        let token_lengths = Tensor::<B, 1, Int>::from_data(
            TensorData::new(token_lengths_i64, [batch_size]),
            &device,
        );
        let text = self.inference.text_encoder.forward_conditioned(
            batch.token_ids,
            token_lengths,
            language.clone(),
        )?;
        let frame_mask = lengths_mask::<B>(&batch.frame_lengths, frames, &device);
        let posterior = self.posterior_encoder.forward(
            batch.spectrogram,
            frame_mask.clone(),
            speaker.clone(),
            1.0,
            seed,
        )?;
        let transformed = self.inference.flow.forward(
            posterior.latent.clone(),
            frame_mask.clone(),
            speaker.clone(),
        )?;
        let scores = gaussian_log_likelihood(
            transformed.clone(),
            text.mean.clone(),
            text.log_scale.clone(),
        );
        let alignment = maximum_monotonic_path(scores, &batch.token_lengths, &batch.frame_lengths)?;
        let expanded_mean = text.mean.matmul(alignment.clone());
        let expanded_log_scale = text.log_scale.matmul(alignment.clone());
        let kl_loss = vits_kl_loss(
            transformed,
            expanded_mean,
            expanded_log_scale,
            posterior.log_scale,
            frame_mask,
        );

        let durations = alignment.clone().sum_dim(2).squeeze_dim::<2>(2);
        let target_log_duration = (durations + 1.0).log();
        let predicted_log_duration = self
            .inference
            .duration_predictor
            .training_log_duration(text.encoded, text.mask.clone(), speaker.clone(), language)?
            .squeeze_dim::<2>(1);
        let duration_loss = masked_duration_loss(
            predicted_log_duration.clone(),
            target_log_duration.clone(),
            text.mask.squeeze_dim::<2>(1),
        );

        let segment_starts =
            deterministic_segment_starts(&batch.frame_lengths, segment_frames, seed);
        let latent_segment = slice_segments(posterior.latent, &segment_starts, segment_frames)?;
        let generated_waveform = self
            .inference
            .waveform_decoder
            .forward(latent_segment, speaker)?;
        let target_waveform =
            slice_waveform_segments(batch.waveform, &segment_starts, segment_frames, hop_length)?;
        let [gb, gc, generated_samples] = generated_waveform.dims();
        let [tb, tc, target_samples] = target_waveform.dims();
        let samples = generated_samples.min(target_samples);
        ensure!(samples > 0, "VITS decoder produced an empty waveform");
        Ok(VitsTrainingForward {
            generated_waveform: generated_waveform.slice([0..gb, 0..gc, 0..samples]),
            target_waveform: target_waveform.slice([0..tb, 0..tc, 0..samples]),
            alignment,
            predicted_log_duration,
            target_log_duration,
            duration_loss,
            kl_loss,
            segment_starts,
        })
    }
}

#[derive(Module, Debug)]
pub struct VitsDiscriminators<B: Backend> {
    pub multi_period: MultiPeriodDiscriminator<B>,
    pub multi_scale: MultiScaleDiscriminator<B>,
}

impl<B: Backend> VitsDiscriminators<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            multi_period: MultiPeriodDiscriminator::new(device),
            multi_scale: MultiScaleDiscriminator::new(device),
        }
    }

    pub fn new_fixture(device: &B::Device) -> Self {
        Self {
            multi_period: MultiPeriodDiscriminator::new_fixture(device),
            multi_scale: MultiScaleDiscriminator::new_fixture(device),
        }
    }

    pub fn generator_losses(
        &self,
        target: Tensor<B, 3>,
        generated: Tensor<B, 3>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let period_real = self.multi_period.forward(target.clone().detach());
        let period_fake = self.multi_period.forward(generated.clone());
        let scale_real = self.multi_scale.forward(target.detach());
        let scale_fake = self.multi_scale.forward(generated);
        let mut fake_scores = period_fake.scores();
        fake_scores.extend(scale_fake.scores());
        let adversarial = adversarial_generator_loss(fake_scores);
        let mut real_features = period_real.feature_maps();
        real_features.extend(scale_real.feature_maps());
        let mut fake_features = period_fake.feature_maps();
        fake_features.extend(scale_fake.feature_maps());
        let feature_matching = feature_matching_loss(real_features, fake_features);
        (adversarial, feature_matching)
    }

    pub fn discriminator_loss(
        &self,
        target: Tensor<B, 3>,
        generated: Tensor<B, 3>,
    ) -> Tensor<B, 1> {
        let period_real = self.multi_period.forward(target.clone());
        let period_fake = self.multi_period.forward(generated.clone().detach());
        let scale_real = self.multi_scale.forward(target);
        let scale_fake = self.multi_scale.forward(generated.detach());
        let mut real_scores = period_real.scores();
        real_scores.extend(scale_real.scores());
        let mut fake_scores = period_fake.scores();
        fake_scores.extend(scale_fake.scores());
        adversarial_discriminator_loss(real_scores, fake_scores)
    }
}

#[derive(Debug)]
pub struct VitsGeneratorLosses<B: Backend> {
    pub total: Tensor<B, 1>,
    pub adversarial: Tensor<B, 1>,
    pub feature_matching: Tensor<B, 1>,
    pub mel: Tensor<B, 1>,
    pub duration: Tensor<B, 1>,
    pub kl: Tensor<B, 1>,
}

#[allow(clippy::too_many_arguments)]
pub fn combine_vits_generator_losses<B: Backend>(
    adversarial: Tensor<B, 1>,
    feature_matching: Tensor<B, 1>,
    target_mel: Tensor<B, 3>,
    generated_mel: Tensor<B, 3>,
    duration: Tensor<B, 1>,
    kl: Tensor<B, 1>,
    weights: &crate::VitsLossWeights,
) -> VitsGeneratorLosses<B> {
    let mel = mel_spectrogram_loss(target_mel, generated_mel);
    let total = adversarial.clone() * weights.adversarial
        + feature_matching.clone() * weights.feature_matching
        + mel.clone() * weights.mel
        + duration.clone() * weights.duration
        + kl.clone() * weights.kl;
    VitsGeneratorLosses {
        total,
        adversarial,
        feature_matching,
        mel,
        duration,
        kl,
    }
}

pub fn gaussian_log_likelihood<B: Backend>(
    latent: Tensor<B, 3>,
    prior_mean: Tensor<B, 3>,
    prior_log_scale: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [batch, channels, frames] = latent.dims();
    let [prior_batch, prior_channels, tokens] = prior_mean.dims();
    debug_assert_eq!([batch, channels], [prior_batch, prior_channels]);
    let latent = latent.reshape([batch, channels, 1, frames]);
    let mean = prior_mean.reshape([batch, channels, tokens, 1]);
    let log_scale = prior_log_scale.reshape([batch, channels, tokens, 1]);
    let constant = (2.0 * PI).ln();
    let score =
        ((latent - mean).square() * (log_scale.clone() * -2.0).exp() + log_scale * 2.0 + constant)
            * -0.5;
    score.sum_dim(1).squeeze_dim::<3>(1)
}

pub fn maximum_monotonic_path<B: Backend>(
    scores: Tensor<B, 3>,
    token_lengths: &[usize],
    frame_lengths: &[usize],
) -> Result<Tensor<B, 3>> {
    let [batch, padded_tokens, padded_frames] = scores.dims();
    ensure!(
        token_lengths.len() == batch && frame_lengths.len() == batch,
        "maximum-path lengths must match batch size {batch}"
    );
    let device = scores.device();
    let values = scores
        .into_data()
        .to_vec::<f32>()
        .context("maximum-path scores are not f32")?;
    let mut path = vec![0.0f32; batch * padded_tokens * padded_frames];
    for batch_index in 0..batch {
        let tokens = token_lengths[batch_index];
        let frames = frame_lengths[batch_index];
        ensure!(
            tokens > 0 && tokens <= padded_tokens,
            "invalid token length {tokens} for padded width {padded_tokens}"
        );
        ensure!(
            frames >= tokens && frames <= padded_frames,
            "maximum-path alignment requires frames >= tokens; got {frames} frames and {tokens} tokens"
        );
        let mut dp = vec![f32::NEG_INFINITY; tokens * frames];
        let mut advanced = vec![false; tokens * frames];
        let score_at = |token: usize, frame: usize| {
            values[(batch_index * padded_tokens + token) * padded_frames + frame]
        };
        dp[0] = score_at(0, 0);
        for frame in 1..frames {
            let max_token = (frame + 1).min(tokens);
            for token in 0..max_token {
                let stay = dp[token * frames + frame - 1];
                let advance = if token > 0 {
                    dp[(token - 1) * frames + frame - 1]
                } else {
                    f32::NEG_INFINITY
                };
                let take_advance = advance > stay;
                let previous = if take_advance { advance } else { stay };
                dp[token * frames + frame] = previous + score_at(token, frame);
                advanced[token * frames + frame] = take_advance;
            }
        }
        ensure!(
            dp[(tokens - 1) * frames + frames - 1].is_finite(),
            "maximum-path alignment has no complete path"
        );
        let mut token = tokens - 1;
        for frame in (0..frames).rev() {
            path[(batch_index * padded_tokens + token) * padded_frames + frame] = 1.0;
            if frame > 0 && advanced[token * frames + frame] {
                token -= 1;
            }
        }
        ensure!(
            token == 0,
            "maximum-path backtracking did not reach the first token"
        );
    }
    Ok(Tensor::from_data(
        TensorData::new(path, [batch, padded_tokens, padded_frames]),
        &device,
    ))
}

pub fn masked_duration_loss<B: Backend>(
    predicted: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let numerator = ((predicted - target).square() * mask.clone()).sum();
    numerator / mask.sum().clamp_min(1.0)
}

pub fn vits_kl_loss<B: Backend>(
    transformed_posterior: Tensor<B, 3>,
    prior_mean: Tensor<B, 3>,
    prior_log_scale: Tensor<B, 3>,
    posterior_log_scale: Tensor<B, 3>,
    mask: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let variance_ratio = (posterior_log_scale.clone() * 2.0 - prior_log_scale.clone() * 2.0).exp();
    let mean_error =
        (transformed_posterior - prior_mean).square() * (prior_log_scale.clone() * -2.0).exp();
    let kl = (prior_log_scale - posterior_log_scale - 0.5 + (variance_ratio + mean_error) * 0.5)
        * mask.clone();
    kl.sum() / mask.sum().clamp_min(1.0)
}

pub fn deterministic_segment_starts(
    frame_lengths: &[usize],
    segment_frames: usize,
    seed: u64,
) -> Vec<usize> {
    frame_lengths
        .iter()
        .enumerate()
        .map(|(index, length)| {
            let maximum = length.saturating_sub(segment_frames);
            if maximum == 0 {
                0
            } else {
                splitmix64(seed ^ index as u64) as usize % (maximum + 1)
            }
        })
        .collect()
}

pub fn slice_segments<B: Backend>(
    tensor: Tensor<B, 3>,
    starts: &[usize],
    segment_frames: usize,
) -> Result<Tensor<B, 3>> {
    let [batch, channels, frames] = tensor.dims();
    ensure!(
        starts.len() == batch,
        "segment starts must match batch size"
    );
    ensure!(segment_frames > 0, "segment frames must be positive");
    let mut segments = Vec::with_capacity(batch);
    for (batch_index, start) in starts.iter().copied().enumerate() {
        ensure!(
            start + segment_frames <= frames,
            "segment {batch_index} range {start}..{} exceeds {frames} frames",
            start + segment_frames
        );
        segments.push(tensor.clone().slice([
            batch_index..batch_index + 1,
            0..channels,
            start..start + segment_frames,
        ]));
    }
    Ok(Tensor::cat(segments, 0))
}

pub fn slice_waveform_segments<B: Backend>(
    waveform: Tensor<B, 3>,
    frame_starts: &[usize],
    segment_frames: usize,
    hop_length: usize,
) -> Result<Tensor<B, 3>> {
    let sample_starts = frame_starts
        .iter()
        .map(|start| {
            start
                .checked_mul(hop_length)
                .context("waveform segment start overflow")
        })
        .collect::<Result<Vec<_>>>()?;
    let samples = segment_frames
        .checked_mul(hop_length)
        .context("waveform segment length overflow")?;
    slice_segments(waveform, &sample_starts, samples)
}

fn lengths_mask<B: Backend>(lengths: &[usize], padded: usize, device: &B::Device) -> Tensor<B, 3> {
    let values = lengths
        .iter()
        .flat_map(|length| (0..padded).map(move |index| f32::from(index < *length)))
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, [lengths.len(), 1, padded]), device)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::backend::Autodiff;
    use burn::module::AutodiffModule;
    use burn::nn::EmbeddingConfig;
    use burn::optim::{AdamWConfig, GradientsParams, Optimizer};

    use super::*;

    type TestBackend = NdArray<f32>;
    type TrainBackend = Autodiff<TestBackend>;

    fn tiny_imported_config() -> &'static str {
        r#"{
            model: "vits",
            use_phonemes: true,
            phoneme_language: "en",
            add_blank: true,
            enable_eos_bos_chars: false,
            characters: {
                characters_class: "fixture.VitsCharacters",
                pad: "_",
                eos: "",
                bos: "",
                blank: null,
                characters: "At",
                punctuations: " ",
                phonemes: "''ʰɝʃ",
                is_unique: true,
                is_sorted: true,
            },
            model_args: {
                num_chars: 10,
                out_channels: 5,
                spec_segment_size: 2,
                hidden_channels: 4,
                hidden_channels_ffn_text_encoder: 8,
                num_heads_text_encoder: 2,
                num_layers_text_encoder: 1,
                kernel_size_text_encoder: 3,
                dropout_p_text_encoder: 0.1,
                dropout_p_duration_predictor: 0.1,
                kernel_size_posterior_encoder: 3,
                dilation_rate_posterior_encoder: 1,
                num_layers_posterior_encoder: 1,
                kernel_size_flow: 3,
                dilation_rate_flow: 1,
                num_layers_flow: 1,
                resblock_type_decoder: "1",
                resblock_kernel_sizes_decoder: [3],
                resblock_dilation_sizes_decoder: [[1, 2, 3]],
                upsample_rates_decoder: [2],
                upsample_initial_channel_decoder: 4,
                upsample_kernel_sizes_decoder: [4],
                use_sdp: true,
                inference_noise_scale: 0.667,
                length_scale: 1.0,
                inference_noise_scale_dp: 0.8,
                max_inference_len: null,
                use_speaker_embedding: true,
                num_speakers: 1,
                speaker_embedding_channels: 4,
                use_d_vector_file: false,
                d_vector_dim: 0,
                condition_dp_on_speaker: true,
                use_language_embedding: false,
                embedded_language_dim: 4,
                num_languages: 0,
            },
            audio: {
                fft_size: 8,
                win_length: 8,
                hop_length: 2,
                sample_rate: 8000,
                preemphasis: 0.0,
                log_func: "np.log10",
                num_mels: 2,
                mel_fmin: 0.0,
                mel_fmax: 4000.0,
                spec_gain: 20.0,
                signal_norm: true,
                min_level_db: -100.0,
                symmetric_norm: true,
                max_norm: 4.0,
                clip_norm: true,
                stats_path: null,
                do_amp_to_db_mel: true,
                stft_pad_mode: "reflect",
            },
        }"#
    }

    #[test]
    fn maximum_path_is_monotonic_complete_and_masks_padding() {
        let device = NdArrayDevice::Cpu;
        let scores = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    4.0, 4.0, 1.0, 0.0, 0.0, 0.0, // token 0
                    0.0, 1.0, 4.0, 4.0, 1.0, 0.0, // token 1
                    0.0, 0.0, 0.0, 1.0, 4.0, 4.0, // token 2
                    -9.0, -9.0, -9.0, -9.0, -9.0, -9.0, // padding
                ],
                [1, 4, 6],
            ),
            &device,
        );
        let path = maximum_monotonic_path(scores, &[3], &[6]).unwrap();
        let values = path.into_data().to_vec::<f32>().unwrap();
        assert_eq!(values.iter().filter(|value| **value == 1.0).count(), 6);
        assert!(values[18..].iter().all(|value| *value == 0.0));
        let selected = (0..6)
            .map(|frame| {
                (0..3)
                    .find(|token| values[token * 6 + frame] == 1.0)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert!(selected.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(selected.first(), Some(&0));
        assert_eq!(selected.last(), Some(&2));
    }

    #[test]
    fn posterior_sampling_and_losses_are_finite() {
        let device = NdArrayDevice::Cpu;
        let posterior = VitsPosteriorEncoderConfig {
            input_channels: 5,
            latent_channels: 4,
            hidden_channels: 4,
            kernel_size: 3,
            dilation_rate: 1,
            num_layers: 2,
            conditioning_channels: 0,
        }
        .init::<TestBackend>(&device)
        .unwrap();
        let spectrogram = Tensor::ones([1, 5, 6], &device);
        let mask = Tensor::ones([1, 1, 6], &device);
        let output = posterior
            .forward(spectrogram, mask.clone(), None, 0.0, 42)
            .unwrap();
        assert_eq!(output.latent.dims(), [1, 4, 6]);
        let kl = vits_kl_loss(
            output.latent.clone(),
            Tensor::zeros([1, 4, 6], &device),
            Tensor::zeros([1, 4, 6], &device),
            output.log_scale,
            mask,
        );
        assert!(kl.into_scalar().is_finite());
    }

    #[test]
    fn tiny_posterior_fixture_overfits_without_non_finite_values() {
        let device = NdArrayDevice::Cpu;
        TrainBackend::seed(&device, 7);
        let mut posterior = VitsPosteriorEncoderConfig {
            input_channels: 3,
            latent_channels: 2,
            hidden_channels: 4,
            kernel_size: 3,
            dilation_rate: 1,
            num_layers: 1,
            conditioning_channels: 0,
        }
        .init::<TrainBackend>(&device)
        .unwrap();
        let mut optimizer =
            AdamWConfig::new().init::<TrainBackend, VitsPosteriorEncoder<TrainBackend>>();
        let input = Tensor::<TrainBackend, 3>::from_data(
            TensorData::new(
                vec![
                    1.0, 0.5, 0.0, -0.5, // channel 0
                    0.0, 0.25, 0.5, 0.75, // channel 1
                    -0.5, 0.0, 0.5, 1.0, // channel 2
                ],
                [1, 3, 4],
            ),
            &device,
        );
        let target = Tensor::<TrainBackend, 3>::from_data(
            TensorData::new(
                vec![0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0],
                [1, 2, 4],
            ),
            &device,
        );
        let mask = Tensor::ones([1, 1, 4], &device);
        let mut first = None;
        let mut last = f32::INFINITY;
        for step in 0..80 {
            let output = posterior
                .forward(input.clone(), mask.clone(), None, 0.0, 7)
                .unwrap();
            let loss = (output.mean - target.clone()).square().mean();
            let value: f32 = loss.clone().into_scalar();
            assert!(value.is_finite(), "step {step} loss is non-finite");
            first.get_or_insert(value);
            last = value;
            let gradients = GradientsParams::from_grads(loss.backward(), &posterior);
            posterior = optimizer.step(1.0e-2, posterior, gradients);
        }
        assert!(
            last < first.unwrap() * 0.25,
            "tiny fixture did not overfit: first={:?}, last={last}",
            first
        );
    }

    #[test]
    fn segment_slicing_is_seeded_and_aligned_to_waveform_hops() {
        let device = NdArrayDevice::Cpu;
        let starts = deterministic_segment_starts(&[8, 5], 4, 19);
        assert_eq!(starts, deterministic_segment_starts(&[8, 5], 4, 19));
        let latent = Tensor::<TestBackend, 3>::from_data(
            TensorData::new((0..16).map(|value| value as f32).collect(), [2, 1, 8]),
            &device,
        );
        let waveform = Tensor::<TestBackend, 3>::from_data(
            TensorData::new((0..32).map(|value| value as f32).collect(), [2, 1, 16]),
            &device,
        );
        assert_eq!(
            slice_segments(latent, &starts, 4).unwrap().dims(),
            [2, 1, 4]
        );
        assert_eq!(
            slice_waveform_segments(waveform, &starts, 4, 2)
                .unwrap()
                .dims(),
            [2, 1, 8]
        );
    }

    #[test]
    fn all_generator_loss_components_are_observable() {
        let device = NdArrayDevice::Cpu;
        let scalar = || Tensor::<TestBackend, 1>::ones([1], &device);
        let mel = Tensor::<TestBackend, 3>::ones([1, 2, 3], &device);
        let losses = combine_vits_generator_losses(
            scalar(),
            scalar(),
            mel.clone(),
            mel,
            scalar(),
            scalar(),
            &crate::VitsLossWeights::default(),
        );
        assert_eq!(losses.mel.into_scalar(), 0.0);
        assert!(losses.total.into_scalar().is_finite());
    }

    #[test]
    fn training_export_loads_directly_in_burn_vits_speech() {
        let device = NdArrayDevice::Cpu;
        let imported =
            crate::vits_config::ImportedVitsConfig::from_json5_str(tiny_imported_config()).unwrap();
        let config = imported.inference_config();
        let text_config = crate::VitsTextPriorConfig::from_model_config(&config).unwrap();
        let mut duration_config =
            crate::StochasticDurationConfig::new(text_config.encoder_channels(), 192, 3);
        duration_config.conditioning_channels = 4;
        let flow_config = crate::ResidualCouplingFlowConfig {
            channels: 4,
            hidden_channels: 4,
            kernel_size: 3,
            dilation_rate: 1,
            num_layers: 1,
            num_flows: 4,
            conditioning_channels: 4,
        };
        let decoder_config = crate::VitsWaveformDecoderConfig::from_model_config(&config).unwrap();
        let export: VitsInferenceExport<TestBackend> = VitsInferenceExport {
            text_encoder: text_config.init(&device).unwrap(),
            duration_predictor: duration_config.init(&device).unwrap(),
            flow: flow_config.init(&device).unwrap(),
            waveform_decoder: decoder_config.init(&device).unwrap(),
            emb_g: Some(EmbeddingConfig::new(1, 4).init(&device)),
            emb_l: None,
        };
        let conditioning = export
            .speaker_conditioning(Some(Tensor::from_ints([[0]], &device)))
            .unwrap();
        let waveform = export
            .waveform_decoder
            .forward(Tensor::ones([1, 4, 2], &device), conditioning)
            .unwrap();
        let waveform_values = waveform.clone().into_data().to_vec::<f32>().unwrap();
        assert!(!waveform_values.is_empty());
        assert!(waveform_values.iter().all(|sample| sample.is_finite()));

        let root = tempfile::tempdir().unwrap();
        let checkpoint = root.path().join("model.safetensors");
        let config_path = root.path().join("config.json");
        let speakers = root.path().join("speaker_ids.json");
        std::fs::write(&config_path, tiny_imported_config()).unwrap();
        std::fs::write(&speakers, r#"{"fixture":0}"#).unwrap();
        export.save_inference_safetensors(&checkpoint).unwrap();

        crate::BurnVitsSpeech::<TestBackend>::load(config_path, checkpoint, speakers, device)
            .unwrap_or_else(|error| {
                panic!("training export must load through the normal inference adapter: {error:#}")
            });
    }

    #[test]
    fn published_checkpoint_short_fine_tune_improves_fixture_when_available() {
        let (Some(config_path), Some(checkpoint), Some(speakers)) = (
            std::env::var_os("TONGUES_TEST_COQUI_VITS_CONFIG"),
            std::env::var_os("TONGUES_TEST_COQUI_VITS_CHECKPOINT"),
            std::env::var_os("TONGUES_TEST_COQUI_VITS_SPEAKERS"),
        ) else {
            return;
        };
        let device = NdArrayDevice::Cpu;
        let source = std::fs::read_to_string(&config_path).unwrap();
        let imported = crate::vits_config::ImportedVitsConfig::from_json5_str(&source).unwrap();
        let config = imported.inference_config();
        let mut model = VitsInferenceExport::<TrainBackend>::load_coqui_checkpoint(
            &config,
            &checkpoint,
            &device,
        )
        .unwrap();
        let mut optimizer =
            AdamWConfig::new().init::<TrainBackend, VitsInferenceExport<TrainBackend>>();
        let tokens = Tensor::<TrainBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let lengths = Tensor::<TrainBackend, 1, Int>::from_ints([4], &device);
        TrainBackend::seed(&device, 29);
        let baseline = model
            .text_encoder
            .forward(tokens.clone(), lengths.clone())
            .unwrap()
            .mean;
        let target = baseline.detach() * 0.5;
        let mut first = None;
        let mut last = f32::INFINITY;
        for step in 0..12 {
            TrainBackend::seed(&device, 29);
            let predicted = model
                .text_encoder
                .forward(tokens.clone(), lengths.clone())
                .unwrap()
                .mean;
            let loss = (predicted - target.clone()).square().mean();
            let value: f32 = loss.clone().into_scalar();
            assert!(
                value.is_finite(),
                "published fixture step {step} is non-finite"
            );
            first.get_or_insert(value);
            last = value;
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(2.0e-4, model, gradients);
        }
        assert!(
            last < first.unwrap(),
            "published VITS fixture metric did not improve: first={first:?}, last={last}"
        );

        let root = tempfile::tempdir().unwrap();
        let exported = root.path().join("fine-tuned.safetensors");
        model.valid().save_inference_safetensors(&exported).unwrap();
        crate::BurnVitsSpeech::<TestBackend>::load(config_path, exported, speakers, device)
            .expect("fine-tuned published checkpoint must load directly in BurnVitsSpeech");
    }
}
