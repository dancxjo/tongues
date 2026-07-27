// SPDX-License-Identifier: MPL-2.0
//! Burn-native Glow-TTS acoustic inference.
//!
//! The model is deliberately acoustic-only: it emits the checkpoint's neutral
//! mel-spectrogram contract and can be composed with any compatible vocoder.
//! The relative-position encoder and duration/alignment primitives are shared
//! with the native VITS path, while Glow's ActNorm, split invertible
//! convolution, and affine coupling decoder retain their own checkpoint
//! layout.
//!
//! Source provenance: adapted from the MPL-2.0 Coqui TTS Glow-TTS inference
//! graph at revision `0cf3265a4686d7e856bd472cdaf1572d61cab2b8`
//! (`TTS/tts/models/glow_tts.py` and `TTS/tts/layers/glow_tts/`) and rewritten
//! against Burn tensors. See `THIRD_PARTY_NOTICES.md`.

use std::fmt;
use std::path::Path;

use burn::module::{Initializer, Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, PaddingConfig1d};
use burn::tensor::activation::{relu, sigmoid, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::ops::PadMode;
use burn::tensor::{Distribution, Int, Tensor, TensorData};

use crate::burn_speedy_speech::DurationPredictor;
use crate::burn_vits_duration::{StochasticDurationConfig, StochasticDurationPredictor};
use crate::burn_vits_flow::{CouplingWaveNet, FlowWeightNormConv1d};
use crate::burn_vits_text::sequence_mask;
use crate::{expand_prior_statistics, GlowTtsEncoderConfig, GlowTtsInferenceConfig};

const LAYER_NORM_EPSILON: f64 = 1e-4;
pub const DEFAULT_GLOW_MAX_OUTPUT_FRAMES: usize = 65_536;

#[derive(Debug)]
pub enum GlowTtsError {
    InvalidTopology(String),
    InvalidInput(String),
    Checkpoint(String),
}

impl fmt::Display for GlowTtsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTopology(message) => {
                write!(formatter, "invalid Glow-TTS topology: {message}")
            }
            Self::InvalidInput(message) => write!(formatter, "invalid Glow-TTS input: {message}"),
            Self::Checkpoint(message) => {
                write!(formatter, "unable to load Glow-TTS checkpoint: {message}")
            }
        }
    }
}

impl std::error::Error for GlowTtsError {}

fn topology_error(message: impl Into<String>) -> GlowTtsError {
    GlowTtsError::InvalidTopology(message.into())
}

fn input_error(message: impl Into<String>) -> GlowTtsError {
    GlowTtsError::InvalidInput(message.into())
}

#[derive(Module, Debug)]
struct GlowChannelLayerNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 3>>,
    pub beta: Param<Tensor<B, 3>>,
    epsilon: f64,
}

impl<B: Backend> GlowChannelLayerNorm<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            gamma: Initializer::Constant { value: 0.1 }.init([1, channels, 1], device),
            beta: Initializer::Zeros.init([1, channels, 1], device),
            epsilon: LAYER_NORM_EPSILON,
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mean = input.clone().mean_dim(1);
        let variance = (input.clone() - mean.clone()).square().mean_dim(1);
        (input - mean) / (variance + self.epsilon).sqrt() * self.gamma.val() + self.beta.val()
    }
}

/// Three-layer residual convolutional prenet used by released checkpoints.
#[derive(Module, Debug)]
pub struct GlowEncoderPrenet<B: Backend> {
    pub conv_layers: Vec<Conv1d<B>>,
    norm_layers: Vec<GlowChannelLayerNorm<B>>,
    pub proj: Conv1d<B>,
}

impl<B: Backend> GlowEncoderPrenet<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        let conv_layers = (0..3)
            .map(|_| {
                Conv1dConfig::new(channels, channels, 5)
                    .with_padding(PaddingConfig1d::Explicit(2, 2))
                    .init(device)
            })
            .collect();
        let norm_layers = (0..3)
            .map(|_| GlowChannelLayerNorm::init(channels, device))
            .collect();
        Self {
            conv_layers,
            norm_layers,
            proj: Conv1dConfig::new(channels, channels, 1)
                .with_padding(PaddingConfig1d::Valid)
                .with_initializer(Initializer::Zeros)
                .init(device),
        }
    }

    fn forward(&self, mut input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = input.clone();
        for (conv, norm) in self.conv_layers.iter().zip(&self.norm_layers) {
            input = relu(norm.forward(conv.forward(input * mask.clone()) * mask.clone()));
        }
        (residual + self.proj.forward(input)) * mask
    }
}

#[derive(Module, Debug)]
pub struct GlowRelativePositionTransformer<B: Backend> {
    pub attn_layers: Vec<GlowMultiHeadAttention<B>>,
    norm_layers_1: Vec<GlowChannelLayerNorm<B>>,
    pub ffn_layers: Vec<GlowFeedForwardNetwork<B>>,
    norm_layers_2: Vec<GlowChannelLayerNorm<B>>,
}

impl<B: Backend> GlowRelativePositionTransformer<B> {
    fn init(channels: usize, config: &GlowTtsEncoderConfig, device: &B::Device) -> Self {
        Self {
            attn_layers: (0..config.num_layers)
                .map(|_| {
                    GlowMultiHeadAttention::init(
                        channels,
                        config.num_heads,
                        config.rel_attn_window_size,
                        device,
                    )
                })
                .collect(),
            norm_layers_1: (0..config.num_layers)
                .map(|_| GlowChannelLayerNorm::init(channels, device))
                .collect(),
            ffn_layers: (0..config.num_layers)
                .map(|_| {
                    GlowFeedForwardNetwork::init(
                        channels,
                        config.hidden_channels_ffn,
                        config.kernel_size,
                        device,
                    )
                })
                .collect(),
            norm_layers_2: (0..config.num_layers)
                .map(|_| GlowChannelLayerNorm::init(channels, device))
                .collect(),
        }
    }

    fn forward(&self, mut input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        let attention_mask =
            mask.clone().unsqueeze_dim::<4>(2) * mask.clone().unsqueeze_dim::<4>(3);
        for layer in 0..self.attn_layers.len() {
            input = input * mask.clone();
            let attention = self.attn_layers[layer].forward(
                input.clone(),
                input.clone(),
                attention_mask.clone(),
            );
            input = self.norm_layers_1[layer].forward(input + attention);
            let ffn = self.ffn_layers[layer].forward(input.clone(), mask.clone());
            input = self.norm_layers_2[layer].forward(input + ffn);
        }
        input * mask
    }
}

#[derive(Module, Debug)]
pub struct GlowMultiHeadAttention<B: Backend> {
    pub conv_q: Conv1d<B>,
    pub conv_k: Conv1d<B>,
    pub conv_v: Conv1d<B>,
    pub conv_o: Conv1d<B>,
    pub emb_rel_k: Option<Param<Tensor<B, 3>>>,
    pub emb_rel_v: Option<Param<Tensor<B, 3>>>,
    num_heads: usize,
    channels_per_head: usize,
    relative_attention_window: Option<usize>,
}

impl<B: Backend> GlowMultiHeadAttention<B> {
    fn init(
        channels: usize,
        num_heads: usize,
        relative_attention_window: Option<usize>,
        device: &B::Device,
    ) -> Self {
        let conv = || {
            Conv1dConfig::new(channels, channels, 1)
                .with_padding(PaddingConfig1d::Valid)
                .init(device)
        };
        let channels_per_head = channels / num_heads;
        let relative_shape =
            relative_attention_window.map(|window| [1, window * 2 + 1, channels_per_head]);
        Self {
            conv_q: conv(),
            conv_k: conv(),
            conv_v: conv(),
            conv_o: conv(),
            emb_rel_k: relative_shape.map(|shape| {
                Initializer::Normal {
                    mean: 0.0,
                    std: (channels_per_head as f64).powf(-0.5),
                }
                .init(shape, device)
            }),
            emb_rel_v: relative_shape.map(|shape| {
                Initializer::Normal {
                    mean: 0.0,
                    std: (channels_per_head as f64).powf(-0.5),
                }
                .init(shape, device)
            }),
            num_heads,
            channels_per_head,
            relative_attention_window,
        }
    }

    fn forward(
        &self,
        input: Tensor<B, 3>,
        context: Tensor<B, 3>,
        attention_mask: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let query = self.conv_q.forward(input);
        let key = self.conv_k.forward(context.clone());
        let value = self.conv_v.forward(context);
        self.conv_o
            .forward(self.attention(query, key, value, attention_mask))
    }

    fn attention(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        attention_mask: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch, channels, source_tokens] = key.dims();
        let target_tokens = query.dims()[2];
        let query = query
            .reshape([batch, self.num_heads, self.channels_per_head, target_tokens])
            .swap_dims(2, 3);
        let key = key
            .reshape([batch, self.num_heads, self.channels_per_head, source_tokens])
            .swap_dims(2, 3);
        let value = value
            .reshape([batch, self.num_heads, self.channels_per_head, source_tokens])
            .swap_dims(2, 3);
        let scale = (self.channels_per_head as f64).sqrt();
        let mut scores = query.clone().matmul(key.swap_dims(2, 3)) / scale;
        if let (Some(window), Some(relative_keys)) =
            (self.relative_attention_window, &self.emb_rel_k)
        {
            let relative_keys =
                glow_relative_embeddings(relative_keys.val(), source_tokens, window);
            let relative_logits = query.clone().matmul(
                relative_keys
                    .reshape([1, 1, source_tokens * 2 - 1, self.channels_per_head])
                    .swap_dims(2, 3),
            );
            scores = scores + glow_relative_to_absolute(relative_logits) / scale;
        }
        scores = scores.mask_fill(attention_mask.equal_elem(0.0), -1.0e4);
        let attention = softmax(scores, 3);
        let mut output = attention.clone().matmul(value);
        if let (Some(window), Some(relative_values)) =
            (self.relative_attention_window, &self.emb_rel_v)
        {
            let relative_values =
                glow_relative_embeddings(relative_values.val(), source_tokens, window);
            output = output
                + glow_absolute_to_relative(attention).matmul(relative_values.reshape([
                    1,
                    1,
                    source_tokens * 2 - 1,
                    self.channels_per_head,
                ]));
        }
        output
            .swap_dims(2, 3)
            .reshape([batch, channels, target_tokens])
    }
}

fn glow_relative_embeddings<B: Backend>(
    embeddings: Tensor<B, 3>,
    length: usize,
    window: usize,
) -> Tensor<B, 3> {
    let padding = length.saturating_sub(window + 1);
    let slice_start = (window + 1).saturating_sub(length);
    let slice_end = slice_start + length * 2 - 1;
    let channels = embeddings.dims()[2];
    let embeddings = if padding > 0 {
        embeddings.pad([(0, 0), (padding, padding), (0, 0)], PadMode::Constant(0.0))
    } else {
        embeddings
    };
    embeddings.slice([0..1, slice_start..slice_end, 0..channels])
}

fn glow_relative_to_absolute<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, length, _] = input.dims();
    input
        .pad([(0, 0), (0, 0), (0, 0), (0, 1)], PadMode::Constant(0.0))
        .reshape([batch, heads, length * 2 * length])
        .pad([(0, 0), (0, 0), (0, length - 1)], PadMode::Constant(0.0))
        .reshape([batch, heads, length + 1, length * 2 - 1])
        .slice([0..batch, 0..heads, 0..length, length - 1..length * 2 - 1])
}

fn glow_absolute_to_relative<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, length, _] = input.dims();
    input
        .pad(
            [(0, 0), (0, 0), (0, 0), (0, length - 1)],
            PadMode::Constant(0.0),
        )
        .reshape([batch, heads, length * (length * 2 - 1)])
        .pad([(0, 0), (0, 0), (length, 0)], PadMode::Constant(0.0))
        .reshape([batch, heads, length, length * 2])
        .slice([0..batch, 0..heads, 0..length, 1..length * 2])
}

#[derive(Module, Debug)]
pub struct GlowFeedForwardNetwork<B: Backend> {
    pub conv_1: Conv1d<B>,
    pub conv_2: Conv1d<B>,
    pad_left: usize,
    pad_right: usize,
}

impl<B: Backend> GlowFeedForwardNetwork<B> {
    fn init(
        channels: usize,
        hidden_channels: usize,
        kernel_size: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            conv_1: Conv1dConfig::new(channels, hidden_channels, kernel_size)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            conv_2: Conv1dConfig::new(hidden_channels, channels, kernel_size)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            pad_left: (kernel_size - 1) / 2,
            pad_right: kernel_size / 2,
        }
    }

    fn forward(&self, input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        let padded = (input * mask.clone()).pad(
            [(0, 0), (0, 0), (self.pad_left, self.pad_right)],
            PadMode::Constant(0.0),
        );
        let hidden = relu(self.conv_1.forward(padded));
        let padded = (hidden * mask.clone()).pad(
            [(0, 0), (0, 0), (self.pad_left, self.pad_right)],
            PadMode::Constant(0.0),
        );
        self.conv_2.forward(padded) * mask
    }
}

/// Token-aligned Glow-TTS Gaussian parameters and duration features.
#[derive(Debug)]
pub struct GlowTextEncoderOutput<B: Backend> {
    pub encoded: Tensor<B, 3>,
    pub mean: Tensor<B, 3>,
    pub log_scale: Tensor<B, 3>,
    pub log_durations: Tensor<B, 3>,
    pub mask: Tensor<B, 3>,
}

/// Glow-TTS text encoder with checkpoint-compatible field names.
#[derive(Module, Debug)]
pub struct GlowTextEncoder<B: Backend> {
    pub emb: Embedding<B>,
    pub prenet: GlowEncoderPrenet<B>,
    pub encoder: GlowRelativePositionTransformer<B>,
    pub proj_m: Conv1d<B>,
    pub proj_s: Option<Conv1d<B>>,
    pub duration_predictor: DurationPredictor<B>,
    hidden_channels: usize,
    out_channels: usize,
    conditioning_channels: usize,
    vocabulary_size: usize,
}

impl<B: Backend> GlowTextEncoder<B> {
    fn init(config: &GlowTtsInferenceConfig, device: &B::Device) -> Result<Self, GlowTtsError> {
        config
            .validate()
            .map_err(|error| topology_error(error.to_string()))?;
        let network = &config.network;
        let embedding_std = (network.hidden_channels_enc as f64).powf(-0.5);
        let emb = EmbeddingConfig::new(network.num_chars, network.hidden_channels_enc)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: embedding_std,
            })
            .init(device);
        let conditioning_channels = network.speaker_conditioning_channels();
        Ok(Self {
            emb,
            prenet: GlowEncoderPrenet::init(network.hidden_channels_enc, device),
            encoder: GlowRelativePositionTransformer::init(
                network.hidden_channels_enc,
                &network.encoder_params,
                device,
            ),
            proj_m: Conv1dConfig::new(network.hidden_channels_enc, network.out_channels, 1)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            proj_s: (!network.mean_only).then(|| {
                Conv1dConfig::new(network.hidden_channels_enc, network.out_channels, 1)
                    .with_padding(PaddingConfig1d::Valid)
                    .init(device)
            }),
            duration_predictor: DurationPredictor::init(
                network.hidden_channels_enc + conditioning_channels,
                network.hidden_channels_dp,
                3,
                device,
            ),
            hidden_channels: network.hidden_channels_enc,
            out_channels: network.out_channels,
            conditioning_channels,
            vocabulary_size: network.num_chars,
        })
    }

    fn load_checkpoint(
        mut self,
        checkpoint_path: impl AsRef<Path>,
        deterministic_duration: bool,
    ) -> Result<Self, GlowTtsError> {
        let predicate = if deterministic_duration {
            glow_encoder_tensor
        } else {
            glow_encoder_without_duration_tensor
        };
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut self,
            checkpoint_path.as_ref(),
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                predicate: Some(predicate),
                key_remappings: vec![(r"^encoder\.".into(), String::new())],
                map_indices_contiguous: false,
                allow_partial: true,
                skip_enum_variants: true,
            },
        )
        .map_err(|error| GlowTtsError::Checkpoint(format!("{error:#}")))?;
        let missing = result
            .missing
            .iter()
            .map(|(path, _)| path.as_str())
            .collect::<Vec<_>>();
        let unused = result
            .unused
            .iter()
            .filter(|path| predicate(path, ""))
            .collect::<Vec<_>>();
        if !missing.is_empty() || !result.errors.is_empty() || !unused.is_empty() {
            return Err(GlowTtsError::Checkpoint(format!(
                "encoder subtree does not exactly match the Burn module: missing [{}], {} load errors, unused [{}]",
                missing.join(", "),
                result.errors.len(),
                unused
                    .into_iter()
                    .map(|path| path.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        Ok(self)
    }

    pub fn forward(
        &self,
        token_ids: Tensor<B, 2, Int>,
        lengths: Tensor<B, 1, Int>,
        conditioning: Option<Tensor<B, 3>>,
    ) -> Result<GlowTextEncoderOutput<B>, GlowTtsError> {
        let [batch, tokens] = token_ids.dims();
        if batch == 0 || tokens == 0 || lengths.dims() != [batch] {
            return Err(input_error(
                "tokens and lengths must have non-empty matching batches",
            ));
        }
        let mask = sequence_mask(lengths, tokens);
        let embedded = self.emb.forward(token_ids) * (self.hidden_channels as f64).sqrt();
        let hidden = self
            .prenet
            .forward(embedded.swap_dims(1, 2) * mask.clone(), mask.clone());
        let hidden = self.encoder.forward(hidden, mask.clone());
        let conditioning =
            normalize_conditioning(conditioning, batch, self.conditioning_channels, tokens)?;
        let duration_input = match conditioning {
            Some(value) => Tensor::cat(
                vec![
                    hidden.clone(),
                    value.expand([batch, self.conditioning_channels, tokens]),
                ],
                1,
            ),
            None => hidden.clone(),
        };
        let mean = self.proj_m.forward(hidden.clone()) * mask.clone();
        let log_scale = match &self.proj_s {
            Some(projection) => projection.forward(hidden.clone()) * mask.clone(),
            None => Tensor::zeros([batch, self.out_channels, tokens], &hidden.device()),
        };
        let log_durations = self
            .duration_predictor
            .forward(duration_input, mask.clone());
        Ok(GlowTextEncoderOutput {
            encoded: hidden,
            mean,
            log_scale,
            log_durations,
            mask,
        })
    }
}

fn glow_encoder_tensor(path: &str, _container: &str) -> bool {
    [
        "emb.",
        "prenet.",
        "encoder.",
        "proj_m.",
        "proj_s.",
        "duration_predictor.",
    ]
    .iter()
    .any(|prefix| path.starts_with(prefix))
}

fn glow_encoder_without_duration_tensor(path: &str, _container: &str) -> bool {
    !path.starts_with("duration_predictor.") && glow_encoder_tensor(path, "")
}

fn normalize_conditioning<B: Backend>(
    conditioning: Option<Tensor<B, 3>>,
    batch: usize,
    channels: usize,
    tokens: usize,
) -> Result<Option<Tensor<B, 3>>, GlowTtsError> {
    match (channels, conditioning) {
        (0, None) => Ok(None),
        (0, Some(_)) => Err(input_error(
            "speaker conditioning was supplied to a single-speaker Glow-TTS model",
        )),
        (_, None) => Err(input_error(
            "speaker conditioning is required by this SC-GlowTTS model",
        )),
        (_, Some(value)) => {
            let [value_batch, value_channels, frames] = value.dims();
            if value_batch != batch
                || value_channels != channels
                || (frames != 1 && frames != tokens)
            {
                return Err(input_error(format!(
                    "speaker conditioning shape {:?}; expected [{batch}, {channels}, 1 or {tokens}]",
                    value.dims()
                )));
            }
            let norm = value.clone().square().sum_dim(1).sqrt().clamp_min(1.0e-12);
            Ok(Some(value / norm))
        }
    }
}

#[derive(Module, Debug)]
pub struct GlowActNorm<B: Backend> {
    pub logs: Param<Tensor<B, 3>>,
    pub bias: Param<Tensor<B, 3>>,
}

impl<B: Backend> GlowActNorm<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            logs: Initializer::Zeros.init([1, channels, 1], device),
            bias: Initializer::Zeros.init([1, channels, 1], device),
        }
    }

    fn reverse(&self, input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        (input - self.bias.val()) * (-self.logs.val()).exp() * mask
    }
}

/// Split invertible 1x1 convolution from Glow-TTS.
#[derive(Module, Debug)]
pub struct GlowInvertibleConv<B: Backend> {
    pub weight: Param<Tensor<B, 2>>,
    #[module(skip)]
    inverse: Tensor<B, 2>,
    num_splits: usize,
}

impl<B: Backend> GlowInvertibleConv<B> {
    fn init(num_splits: usize, device: &B::Device) -> Self {
        let identity = identity_matrix::<B>(num_splits, device);
        Self {
            weight: Param::from_tensor(identity.clone()),
            inverse: identity,
            num_splits,
        }
    }

    fn refresh_inverse(mut self) -> Result<Self, GlowTtsError> {
        let device = self.weight.device();
        let values = self
            .weight
            .val()
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| GlowTtsError::Checkpoint(error.to_string()))?;
        let inverse = invert_square_matrix(&values, self.num_splits)?;
        self.inverse = Tensor::from_data(
            TensorData::new(inverse, [self.num_splits, self.num_splits]),
            &device,
        );
        Ok(self)
    }

    fn reverse(&self, input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, frames] = input.dims();
        let groups = channels / self.num_splits;
        let reshaped = input
            .reshape([batch, 2, groups, self.num_splits / 2, frames])
            .swap_dims(2, 3)
            .reshape([batch, self.num_splits, groups * frames]);
        let weight = self
            .inverse
            .clone()
            .reshape([1, self.num_splits, self.num_splits])
            .expand([batch, self.num_splits, self.num_splits]);
        weight
            .matmul(reshaped)
            .reshape([batch, 2, self.num_splits / 2, groups, frames])
            .swap_dims(2, 3)
            .reshape([batch, channels, frames])
            * mask
    }
}

fn identity_matrix<B: Backend>(size: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut values = vec![0.0f32; size * size];
    for index in 0..size {
        values[index * size + index] = 1.0;
    }
    Tensor::from_data(TensorData::new(values, [size, size]), device)
}

fn invert_square_matrix(values: &[f32], size: usize) -> Result<Vec<f32>, GlowTtsError> {
    if values.len() != size * size || size == 0 {
        return Err(GlowTtsError::Checkpoint(
            "invertible convolution has an invalid matrix shape".into(),
        ));
    }
    let width = size * 2;
    let mut augmented = vec![0.0f64; size * width];
    for row in 0..size {
        for column in 0..size {
            augmented[row * width + column] = f64::from(values[row * size + column]);
        }
        augmented[row * width + size + row] = 1.0;
    }
    for column in 0..size {
        let pivot = (column..size)
            .max_by(|left, right| {
                augmented[*left * width + column]
                    .abs()
                    .total_cmp(&augmented[*right * width + column].abs())
            })
            .expect("non-empty pivot candidates");
        if augmented[pivot * width + column].abs() < 1.0e-12 {
            return Err(GlowTtsError::Checkpoint(
                "invertible convolution weight is singular".into(),
            ));
        }
        if pivot != column {
            for entry in 0..width {
                augmented.swap(column * width + entry, pivot * width + entry);
            }
        }
        let divisor = augmented[column * width + column];
        for entry in 0..width {
            augmented[column * width + entry] /= divisor;
        }
        for row in 0..size {
            if row == column {
                continue;
            }
            let factor = augmented[row * width + column];
            for entry in 0..width {
                augmented[row * width + entry] -= factor * augmented[column * width + entry];
            }
        }
    }
    let mut inverse = Vec::with_capacity(size * size);
    for row in 0..size {
        for column in 0..size {
            inverse.push(augmented[row * width + size + column] as f32);
        }
    }
    Ok(inverse)
}

#[derive(Module, Debug)]
pub struct GlowCouplingBlock<B: Backend> {
    pub start: FlowWeightNormConv1d<B>,
    pub end: Conv1d<B>,
    pub wn: CouplingWaveNet<B>,
    half_channels: usize,
    sigmoid_scale: bool,
}

impl<B: Backend> GlowCouplingBlock<B> {
    #[allow(clippy::too_many_arguments)]
    fn init(
        channels: usize,
        hidden_channels: usize,
        kernel_size: usize,
        dilation_rate: usize,
        num_layers: usize,
        conditioning_channels: usize,
        sigmoid_scale: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            start: FlowWeightNormConv1d::new(channels / 2, hidden_channels, 1, 1, 0, 1, device),
            end: Conv1dConfig::new(hidden_channels, channels, 1)
                .with_padding(PaddingConfig1d::Valid)
                .with_initializer(Initializer::Zeros)
                .init(device),
            wn: CouplingWaveNet::new(
                hidden_channels,
                kernel_size,
                dilation_rate,
                num_layers,
                conditioning_channels,
                device,
            ),
            half_channels: channels / 2,
            sigmoid_scale,
        }
    }

    fn reverse(
        &self,
        input: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let [batch, _, frames] = input.dims();
        let first = input
            .clone()
            .slice([0..batch, 0..self.half_channels, 0..frames]);
        let second = input.slice([
            0..batch,
            self.half_channels..self.half_channels * 2,
            0..frames,
        ]);
        let hidden = self.start.forward(first.clone()) * mask.clone();
        let parameters = self
            .end
            .forward(self.wn.forward(hidden, mask.clone(), conditioning));
        let translation = parameters
            .clone()
            .slice([0..batch, 0..self.half_channels, 0..frames]);
        let mut log_scale = parameters.slice([
            0..batch,
            self.half_channels..self.half_channels * 2,
            0..frames,
        ]);
        if self.sigmoid_scale {
            log_scale = (sigmoid(log_scale + 2.0) + 1.0e-6).log();
        }
        let second = (second - translation) * (-log_scale).exp() * mask.clone();
        Tensor::cat(vec![first, second], 1) * mask
    }
}

#[derive(Module, Debug)]
pub struct GlowFlowBlock<B: Backend> {
    pub act_norm: GlowActNorm<B>,
    pub inv_conv: GlowInvertibleConv<B>,
    pub coupling: GlowCouplingBlock<B>,
}

/// Reverse-only Glow decoder used for acoustic synthesis.
#[derive(Module, Debug)]
pub struct GlowDecoder<B: Backend> {
    pub blocks: Vec<GlowFlowBlock<B>>,
    channels: usize,
    conditioning_channels: usize,
    num_squeeze: usize,
}

impl<B: Backend> GlowDecoder<B> {
    fn init(config: &GlowTtsInferenceConfig, device: &B::Device) -> Result<Self, GlowTtsError> {
        config
            .validate()
            .map_err(|error| topology_error(error.to_string()))?;
        let network = &config.network;
        let channels = network.out_channels * network.num_squeeze;
        let conditioning_channels = network.speaker_conditioning_channels();
        let blocks = (0..network.num_flow_blocks_dec)
            .map(|_| GlowFlowBlock {
                act_norm: GlowActNorm::init(channels, device),
                inv_conv: GlowInvertibleConv::init(network.num_splits, device),
                coupling: GlowCouplingBlock::init(
                    channels,
                    network.hidden_channels_dec,
                    network.kernel_size_dec,
                    network.dilation_rate,
                    network.num_block_layers,
                    conditioning_channels,
                    network.sigmoid_scale,
                    device,
                ),
            })
            .collect();
        Ok(Self {
            blocks,
            channels,
            conditioning_channels,
            num_squeeze: network.num_squeeze,
        })
    }

    fn load_checkpoint(mut self, checkpoint_path: impl AsRef<Path>) -> Result<Self, GlowTtsError> {
        let mut remappings = Vec::with_capacity(self.blocks.len() * 3);
        for index in 0..self.blocks.len() {
            remappings.push((
                format!(r"^decoder\.flows\.{}\.", index * 3),
                format!("blocks.{index}.act_norm."),
            ));
            remappings.push((
                format!(r"^decoder\.flows\.{}\.", index * 3 + 1),
                format!("blocks.{index}.inv_conv."),
            ));
            remappings.push((
                format!(r"^decoder\.flows\.{}\.", index * 3 + 2),
                format!("blocks.{index}.coupling."),
            ));
        }
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut self,
            checkpoint_path.as_ref(),
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                predicate: Some(glow_decoder_tensor),
                key_remappings: remappings,
                map_indices_contiguous: false,
                allow_partial: true,
                skip_enum_variants: true,
            },
        )
        .map_err(|error| GlowTtsError::Checkpoint(format!("{error:#}")))?;
        let missing = result
            .missing
            .iter()
            .map(|(path, _)| path.as_str())
            .collect::<Vec<_>>();
        let unused = result
            .unused
            .iter()
            .filter(|path| glow_decoder_tensor(path, ""))
            .collect::<Vec<_>>();
        if !missing.is_empty() || !result.errors.is_empty() || !unused.is_empty() {
            return Err(GlowTtsError::Checkpoint(format!(
                "decoder subtree does not exactly match the Burn module: missing [{}], {} load errors, unused [{}]",
                missing.join(", "),
                result.errors.len(),
                unused
                    .into_iter()
                    .map(|path| path.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let mut refreshed = Vec::with_capacity(self.blocks.len());
        for block in self.blocks {
            // Burn Store treats every two-dimensional PyTorch `weight` as a
            // linear-layer parameter and transposes it into Burn's layout.
            // Glow's split invertible-convolution matrix is a raw Conv2d
            // kernel, so restore its checkpoint orientation before caching
            // the inverse used by reverse inference.
            let inv_conv = GlowInvertibleConv {
                weight: Param::from_tensor(block.inv_conv.weight.val().transpose()),
                inverse: block.inv_conv.inverse,
                num_splits: block.inv_conv.num_splits,
            };
            refreshed.push(GlowFlowBlock {
                act_norm: block.act_norm,
                inv_conv: inv_conv.refresh_inverse()?,
                coupling: block.coupling,
            });
        }
        self.blocks = refreshed;
        Ok(self)
    }

    pub fn reverse(
        &self,
        input: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 3>>,
    ) -> Result<Tensor<B, 3>, GlowTtsError> {
        self.reverse_with_trace(input, mask, conditioning, false)
            .map(|(output, _)| output)
    }

    /// Runs the reverse flow and optionally retains the output of every
    /// coupling/invertible-convolution/ActNorm block for conformance evidence.
    pub fn reverse_with_trace(
        &self,
        input: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 3>>,
        capture_trace: bool,
    ) -> Result<(Tensor<B, 3>, Vec<Tensor<B, 3>>), GlowTtsError> {
        let [batch, channels, frames] = input.dims();
        if channels * self.num_squeeze != self.channels {
            return Err(input_error(format!(
                "Glow decoder input has {channels} channels; expected {} before squeeze",
                self.channels / self.num_squeeze
            )));
        }
        if mask.dims() != [batch, 1, frames] {
            return Err(input_error(format!(
                "Glow decoder mask shape {:?}; expected [{batch}, 1, {frames}]",
                mask.dims()
            )));
        }
        let conditioning =
            normalize_conditioning(conditioning, batch, self.conditioning_channels, frames)?;
        let (mut output, squeezed_mask) = squeeze(input, mask, self.num_squeeze)?;
        let mut trace = if capture_trace {
            Vec::with_capacity(self.blocks.len())
        } else {
            Vec::new()
        };
        for block in self.blocks.iter().rev() {
            output = block
                .coupling
                .reverse(output, squeezed_mask.clone(), conditioning.clone());
            output = block.inv_conv.reverse(output, squeezed_mask.clone());
            output = block.act_norm.reverse(output, squeezed_mask.clone());
            if capture_trace {
                trace.push(output.clone());
            }
        }
        Ok((unsqueeze(output, squeezed_mask, self.num_squeeze), trace))
    }
}

fn glow_decoder_tensor(path: &str, _container: &str) -> bool {
    path.starts_with("blocks.")
}

fn squeeze<B: Backend>(
    input: Tensor<B, 3>,
    mask: Tensor<B, 3>,
    factor: usize,
) -> Result<(Tensor<B, 3>, Tensor<B, 3>), GlowTtsError> {
    let [batch, channels, frames] = input.dims();
    let truncated = frames / factor * factor;
    if truncated == 0 {
        return Err(input_error(
            "Glow decoder duration path is shorter than its squeeze factor",
        ));
    }
    let input = input.slice([0..batch, 0..channels, 0..truncated]);
    let squeezed = input
        .reshape([batch, channels, truncated / factor, factor])
        .swap_dims(1, 3)
        .swap_dims(2, 3)
        .reshape([batch, channels * factor, truncated / factor]);
    let squeezed_mask = mask
        .slice([0..batch, 0..1, 0..truncated])
        .reshape([batch, 1, truncated / factor, factor])
        .slice([0..batch, 0..1, 0..truncated / factor, factor - 1..factor])
        .reshape([batch, 1, truncated / factor]);
    Ok((squeezed * squeezed_mask.clone(), squeezed_mask))
}

fn unsqueeze<B: Backend>(input: Tensor<B, 3>, mask: Tensor<B, 3>, factor: usize) -> Tensor<B, 3> {
    let [batch, channels, frames] = input.dims();
    let channels_out = channels / factor;
    let output = input
        .reshape([batch, factor, channels_out, frames])
        .swap_dims(1, 2)
        .swap_dims(2, 3)
        .reshape([batch, channels_out, frames * factor]);
    let mask = mask
        .reshape([batch, 1, frames, 1])
        .repeat_dim(3, factor)
        .reshape([batch, 1, frames * factor]);
    output * mask
}

/// Full acoustic inference outputs, retaining parity probes for tests.
#[derive(Debug)]
pub struct GlowTtsOutput<B: Backend> {
    pub mel: Tensor<B, 3>,
    pub durations: Tensor<B, 2>,
    pub alignment: Tensor<B, 3>,
    pub prior_mean: Tensor<B, 3>,
    pub prior_log_scale: Tensor<B, 3>,
    pub log_durations: Tensor<B, 3>,
}

/// Loaded deterministic-duration Glow-TTS/SC-GlowTTS acoustic model.
pub struct GlowTts<B: Backend> {
    pub encoder: GlowTextEncoder<B>,
    pub decoder: GlowDecoder<B>,
    config: GlowTtsInferenceConfig,
    device: B::Device,
}

impl<B: Backend> GlowTts<B> {
    pub fn load(
        config: GlowTtsInferenceConfig,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
    ) -> Result<Self, GlowTtsError> {
        if config.network.use_sdp {
            return Err(topology_error(
                "use_sdp requires the stochastic-duration adapter, not deterministic GlowTts",
            ));
        }
        let checkpoint_path = checkpoint_path.as_ref();
        let encoder =
            GlowTextEncoder::init(&config, &device)?.load_checkpoint(checkpoint_path, true)?;
        let decoder = GlowDecoder::init(&config, &device)?.load_checkpoint(checkpoint_path)?;
        Ok(Self {
            encoder,
            decoder,
            config,
            device,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn inference(
        &self,
        token_ids: Tensor<B, 2, Int>,
        lengths: Tensor<B, 1, Int>,
        conditioning: Option<Tensor<B, 3>>,
        explicit_durations: Option<Tensor<B, 2>>,
        length_scale: f64,
        noise_scale: f64,
        seed: Option<u64>,
    ) -> Result<GlowTtsOutput<B>, GlowTtsError> {
        if !length_scale.is_finite() || length_scale <= 0.0 {
            return Err(input_error("length_scale must be finite and positive"));
        }
        if !noise_scale.is_finite() || noise_scale < 0.0 {
            return Err(input_error(
                "acoustic noise_scale must be finite and non-negative",
            ));
        }
        let encoded = self
            .encoder
            .forward(token_ids, lengths, conditioning.clone())?;
        let [batch, _, tokens] = encoded.log_durations.dims();
        let durations = match explicit_durations {
            Some(values) => {
                validate_explicit_durations(values, batch, tokens, DEFAULT_GLOW_MAX_OUTPUT_FRAMES)?
            }
            None => glow_ceil_durations(
                encoded.log_durations.clone(),
                encoded.mask.clone(),
                length_scale,
                DEFAULT_GLOW_MAX_OUTPUT_FRAMES,
            )?,
        };
        let expanded = expand_prior_statistics(
            encoded.mean,
            encoded.log_scale,
            durations.clone(),
            DEFAULT_GLOW_MAX_OUTPUT_FRAMES,
        )
        .map_err(|error| input_error(error.to_string()))?;
        if let Some(seed) = seed {
            B::seed(&self.device, seed);
        }
        let noise = Tensor::random(
            expanded.mean.dims(),
            Distribution::Normal(0.0, 1.0),
            &self.device,
        );
        let latent = (expanded.mean.clone()
            + noise * expanded.log_scale.clone().exp() * noise_scale)
            * expanded.frame_mask.clone();
        let mel = self
            .decoder
            .reverse(latent, expanded.frame_mask, conditioning)?
            .swap_dims(1, 2);
        Ok(GlowTtsOutput {
            mel,
            durations,
            alignment: expanded.path,
            prior_mean: expanded.mean,
            prior_log_scale: expanded.log_scale,
            log_durations: encoded.log_durations,
        })
    }

    pub fn config(&self) -> &GlowTtsInferenceConfig {
        &self.config
    }
}

/// Loaded Glow-TTS-family model with a flow-based stochastic duration prior.
///
/// This is config-driven because the historical model distributed under the
/// SC-GlowTTS name is speaker-conditioned but uses deterministic durations.
pub struct StochasticGlowTts<B: Backend> {
    pub encoder: GlowTextEncoder<B>,
    pub decoder: GlowDecoder<B>,
    pub duration_predictor: StochasticDurationPredictor<B>,
    config: GlowTtsInferenceConfig,
    device: B::Device,
}

impl<B: Backend> StochasticGlowTts<B> {
    pub fn load(
        config: GlowTtsInferenceConfig,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
    ) -> Result<Self, GlowTtsError> {
        if !config.network.use_sdp {
            return Err(topology_error("stochastic Glow-TTS requires use_sdp=true"));
        }
        let checkpoint_path = checkpoint_path.as_ref();
        let encoder =
            GlowTextEncoder::init(&config, &device)?.load_checkpoint(checkpoint_path, false)?;
        let decoder = GlowDecoder::init(&config, &device)?.load_checkpoint(checkpoint_path)?;
        let mut duration_config = StochasticDurationConfig::new(
            config.network.hidden_channels_enc,
            config.network.hidden_channels_dp,
            3,
        );
        duration_config.conditioning_channels = config.network.speaker_conditioning_channels();
        let duration_predictor = duration_config
            .init(&device)
            .map_err(|error| topology_error(error.to_string()))?
            .load_checkpoint_with_prefix(checkpoint_path, "encoder.duration_predictor")
            .map_err(|error| GlowTtsError::Checkpoint(error.to_string()))?;
        Ok(Self {
            encoder,
            decoder,
            duration_predictor,
            config,
            device,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn inference(
        &self,
        token_ids: Tensor<B, 2, Int>,
        lengths: Tensor<B, 1, Int>,
        conditioning: Option<Tensor<B, 3>>,
        explicit_durations: Option<Tensor<B, 2>>,
        length_scale: f64,
        acoustic_noise_scale: f64,
        duration_noise_scale: f64,
        seed: Option<u64>,
    ) -> Result<GlowTtsOutput<B>, GlowTtsError> {
        if !length_scale.is_finite() || length_scale <= 0.0 {
            return Err(input_error("length_scale must be finite and positive"));
        }
        if !acoustic_noise_scale.is_finite() || acoustic_noise_scale < 0.0 {
            return Err(input_error(
                "acoustic noise_scale must be finite and non-negative",
            ));
        }
        if !duration_noise_scale.is_finite() || duration_noise_scale < 0.0 {
            return Err(input_error(
                "duration noise_scale must be finite and non-negative",
            ));
        }
        let encoded = self
            .encoder
            .forward(token_ids, lengths, conditioning.clone())?;
        let [batch, _, tokens] = encoded.log_durations.dims();
        let (durations, log_durations) = match explicit_durations {
            Some(values) => (
                validate_explicit_durations(values, batch, tokens, DEFAULT_GLOW_MAX_OUTPUT_FRAMES)?,
                Tensor::zeros([batch, 1, tokens], &self.device),
            ),
            None => {
                let log_durations = match seed {
                    Some(seed) => self.duration_predictor.reverse_seeded(
                        encoded.encoded.clone(),
                        encoded.mask.clone(),
                        conditioning.clone(),
                        duration_noise_scale,
                        seed,
                    ),
                    None => self.duration_predictor.reverse(
                        encoded.encoded.clone(),
                        encoded.mask.clone(),
                        conditioning.clone(),
                        duration_noise_scale,
                    ),
                }
                .map_err(|error| input_error(error.to_string()))?;
                let durations = stochastic_ceil_durations(
                    log_durations.clone(),
                    encoded.mask.clone(),
                    length_scale,
                    DEFAULT_GLOW_MAX_OUTPUT_FRAMES,
                )?;
                (durations, log_durations)
            }
        };
        let expanded = expand_prior_statistics(
            encoded.mean,
            encoded.log_scale,
            durations.clone(),
            DEFAULT_GLOW_MAX_OUTPUT_FRAMES,
        )
        .map_err(|error| input_error(error.to_string()))?;
        if let Some(seed) = seed {
            B::seed(&self.device, seed.wrapping_add(1));
        }
        let noise = Tensor::random(
            expanded.mean.dims(),
            Distribution::Normal(0.0, 1.0),
            &self.device,
        );
        let latent = (expanded.mean.clone()
            + noise * expanded.log_scale.clone().exp() * acoustic_noise_scale)
            * expanded.frame_mask.clone();
        let mel = self
            .decoder
            .reverse(latent, expanded.frame_mask, conditioning)?
            .swap_dims(1, 2);
        Ok(GlowTtsOutput {
            mel,
            durations,
            alignment: expanded.path,
            prior_mean: expanded.mean,
            prior_log_scale: expanded.log_scale,
            log_durations,
        })
    }

    pub fn config(&self) -> &GlowTtsInferenceConfig {
        &self.config
    }
}

fn glow_ceil_durations<B: Backend>(
    log_durations: Tensor<B, 3>,
    mask: Tensor<B, 3>,
    length_scale: f64,
    max_output_frames: usize,
) -> Result<Tensor<B, 2>, GlowTtsError> {
    let [batch, channels, tokens] = log_durations.dims();
    if channels != 1 || mask.dims() != [batch, 1, tokens] {
        return Err(input_error(
            "Glow duration logits and mask must use [batch, 1, tokens]",
        ));
    }
    let device = log_durations.device();
    let valid = mask
        .clone()
        .reshape([batch, tokens])
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| input_error(error.to_string()))?;
    let predicted = ((log_durations.exp() - 1.0) * mask * length_scale)
        .reshape([batch, tokens])
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| input_error(error.to_string()))?;
    let mut durations = Vec::with_capacity(predicted.len());
    for (value, valid) in predicted.into_iter().zip(valid) {
        if !value.is_finite() {
            return Err(input_error(
                "Glow duration predictor emitted a non-finite value",
            ));
        }
        durations.push(if valid > 0.0 {
            value.max(0.0).ceil().max(1.0)
        } else {
            0.0
        });
    }
    validate_duration_values(&durations, batch, tokens, max_output_frames)?;
    Ok(Tensor::from_data(
        TensorData::new(durations, [batch, tokens]),
        &device,
    ))
}

fn stochastic_ceil_durations<B: Backend>(
    log_durations: Tensor<B, 3>,
    mask: Tensor<B, 3>,
    length_scale: f64,
    max_output_frames: usize,
) -> Result<Tensor<B, 2>, GlowTtsError> {
    let [batch, channels, tokens] = log_durations.dims();
    if channels != 1 || mask.dims() != [batch, 1, tokens] {
        return Err(input_error(
            "stochastic duration logits and mask must use [batch, 1, tokens]",
        ));
    }
    let device = log_durations.device();
    let valid = mask
        .clone()
        .reshape([batch, tokens])
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| input_error(error.to_string()))?;
    let predicted = (log_durations.exp() * mask * length_scale)
        .reshape([batch, tokens])
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| input_error(error.to_string()))?;
    let mut durations = Vec::with_capacity(predicted.len());
    for (value, valid) in predicted.into_iter().zip(valid) {
        if !value.is_finite() {
            return Err(input_error(
                "stochastic duration predictor emitted a non-finite value",
            ));
        }
        durations.push(if valid > 0.0 {
            value.max(0.0).ceil().max(1.0)
        } else {
            0.0
        });
    }
    validate_duration_values(&durations, batch, tokens, max_output_frames)?;
    Ok(Tensor::from_data(
        TensorData::new(durations, [batch, tokens]),
        &device,
    ))
}

fn validate_explicit_durations<B: Backend>(
    durations: Tensor<B, 2>,
    batch: usize,
    tokens: usize,
    max_output_frames: usize,
) -> Result<Tensor<B, 2>, GlowTtsError> {
    if durations.dims() != [batch, tokens] {
        return Err(input_error(format!(
            "explicit durations shape {:?}; expected [{batch}, {tokens}]",
            durations.dims()
        )));
    }
    let values = durations
        .clone()
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| input_error(error.to_string()))?;
    validate_duration_values(&values, batch, tokens, max_output_frames)?;
    if values
        .iter()
        .any(|value| *value <= 0.0 || value.fract() != 0.0)
    {
        return Err(input_error(
            "explicit Glow-TTS durations must be positive integral frame counts",
        ));
    }
    Ok(durations)
}

fn validate_duration_values(
    values: &[f32],
    batch: usize,
    tokens: usize,
    max_output_frames: usize,
) -> Result<(), GlowTtsError> {
    if values.len() != batch * tokens
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(input_error(
            "Glow-TTS durations must be finite and non-negative",
        ));
    }
    let frames = values
        .chunks(tokens)
        .map(|row| row.iter().map(|value| *value as usize).sum::<usize>())
        .max()
        .unwrap_or(0);
    if frames == 0 || frames > max_output_frames {
        return Err(input_error(format!(
            "Glow-TTS durations request {frames} frames; valid range is 1..={max_output_frames}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use serde_json::Value;

    use super::*;
    use crate::GlowTtsInferenceConfig;

    type TestBackend = NdArray<f32>;

    fn tiny_config() -> GlowTtsInferenceConfig {
        GlowTtsInferenceConfig::from_json5_str(
            r#"{
              model: "glow_tts",
              use_phonemes: true,
              phoneme_language: "en-us",
              add_blank: false,
              enable_eos_bos_chars: false,
              characters: {
                pad: "_", eos: "~", bos: "^", blank: null,
                characters: "abc", punctuations: "! ",
                phonemes: "tk"
              },
              audio: {
                fft_size: 16, win_length: 16, hop_length: 4,
                sample_rate: 22050, num_mels: 4, mel_fmin: 0.0,
                mel_fmax: 8000.0, signal_norm: false
              },
              out_channels: 4,
              hidden_channels_enc: 4,
              hidden_channels_dec: 4,
              hidden_channels_dp: 4,
              num_flow_blocks_dec: 2,
              kernel_size_dec: 3,
              num_block_layers: 2,
              num_splits: 2,
              num_squeeze: 2,
              encoder_params: {
                kernel_size: 3, dropout_p: 0.0, num_layers: 2,
                num_heads: 2, hidden_channels_ffn: 8
              }
            }"#,
        )
        .expect("tiny config")
    }

    fn assert_fixture_probes(
        label: &str,
        tensor: Tensor<TestBackend, 3>,
        fixture: &Value,
        tolerance: f32,
    ) {
        let actual = tensor.into_data().to_vec::<f32>().expect("f32 tensor");
        for probe in fixture["probes"].as_array().expect("fixture probes") {
            let probe = probe.as_array().expect("probe row");
            let index = probe[0].as_u64().expect("probe index") as usize;
            let expected = probe[1].as_f64().expect("probe value") as f32;
            assert!(
                (actual[index] - expected).abs() <= tolerance,
                "{label} parity mismatch at flat index {index}: actual={}, expected={expected}, tolerance={tolerance}",
                actual[index]
            );
        }
    }

    #[test]
    fn split_invertible_convolution_uses_cached_matrix_inverse() {
        let device = NdArrayDevice::Cpu;
        let mut convolution = GlowInvertibleConv::<TestBackend>::init(2, &device);
        convolution.weight =
            Param::from_tensor(Tensor::from_floats([[2.0, 1.0], [1.0, 1.0]], &device));
        let convolution = convolution.refresh_inverse().expect("inverse");
        let input =
            Tensor::from_floats([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], &device);
        let output = convolution.reverse(input, Tensor::ones([1, 1, 2], &device));
        assert_eq!(output.dims(), [1, 4, 2]);
        assert_eq!(
            output.into_data().to_vec::<f32>().expect("values"),
            vec![-4.0, -4.0, -4.0, -4.0, 9.0, 10.0, 11.0, 12.0]
        );
    }

    #[test]
    fn glow_duration_rule_subtracts_one_and_is_seed_independent() {
        let device = NdArrayDevice::Cpu;
        let logits =
            Tensor::<TestBackend, 3>::from_floats([[[2.0_f32.ln(), 3.2_f32.ln()]]], &device);
        let mask = Tensor::ones([1, 1, 2], &device);
        let durations = glow_ceil_durations(logits, mask, 1.0, 100).expect("Glow durations");
        assert_eq!(
            durations.into_data().to_vec::<f32>().expect("values"),
            vec![1.0, 3.0]
        );
    }

    #[test]
    fn stochastic_duration_rule_uses_log_duration_directly() {
        let device = NdArrayDevice::Cpu;
        let logits =
            Tensor::<TestBackend, 3>::from_floats([[[2.0_f32.ln(), 3.2_f32.ln()]]], &device);
        let mask = Tensor::ones([1, 1, 2], &device);
        let durations =
            stochastic_ceil_durations(logits, mask, 1.0, 100).expect("stochastic durations");
        assert_eq!(
            durations.into_data().to_vec::<f32>().expect("values"),
            vec![2.0, 4.0]
        );
    }

    #[test]
    fn tiny_encoder_and_decoder_preserve_acoustic_shapes() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 7);
        let config = tiny_config();
        let encoder = GlowTextEncoder::<TestBackend>::init(&config, &device).expect("encoder");
        let decoder = GlowDecoder::<TestBackend>::init(&config, &device).expect("decoder");
        let tokens = Tensor::<TestBackend, 2, Int>::from_ints([[0, 1]], &device);
        let lengths = Tensor::<TestBackend, 1, Int>::from_ints([2], &device);
        let encoded = encoder.forward(tokens, lengths, None).expect("encoded");
        let durations = Tensor::from_floats([[2.0, 2.0]], &device);
        let expanded = expand_prior_statistics(
            encoded.mean,
            encoded.log_scale,
            durations,
            DEFAULT_GLOW_MAX_OUTPUT_FRAMES,
        )
        .expect("expanded");
        let mel = decoder
            .reverse(expanded.mean, expanded.frame_mask, None)
            .expect("decoded");
        assert_eq!(mel.dims(), [1, 4, 4]);
    }

    #[test]
    fn tiny_encoder_masks_padded_tokens() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 11);
        let config = tiny_config();
        let encoder = GlowTextEncoder::<TestBackend>::init(&config, &device).expect("encoder");
        let tokens =
            Tensor::<TestBackend, 2, Int>::from_ints([[0, 1, 0, 0], [0, 1, 2, 3]], &device);
        let lengths = Tensor::<TestBackend, 1, Int>::from_ints([2, 4], &device);
        let encoded = encoder.forward(tokens, lengths, None).expect("encoded");
        assert_eq!(
            encoded
                .mask
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("mask"),
            vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        );
        for (label, values, channels) in [
            ("encoded", encoded.encoded, 4),
            ("mean", encoded.mean, 4),
            ("log scale", encoded.log_scale, 4),
            ("log durations", encoded.log_durations, 1),
        ] {
            let values = values.into_data().to_vec::<f32>().expect(label);
            for channel in 0..channels {
                assert_eq!(values[channel * 4 + 2], 0.0, "{label} padding");
                assert_eq!(values[channel * 4 + 3], 0.0, "{label} padding");
            }
        }
    }

    #[test]
    fn speaker_conditioning_contract_rejects_missing_malformed_and_conflicting_inputs() {
        let device = NdArrayDevice::Cpu;
        let valid = Tensor::<TestBackend, 3>::zeros([1, 256, 1], &device);
        assert!(normalize_conditioning(Some(valid.clone()), 1, 0, 4).is_err());
        assert!(normalize_conditioning::<TestBackend>(None, 1, 256, 4).is_err());
        assert!(normalize_conditioning(
            Some(Tensor::<TestBackend, 3>::zeros([1, 255, 1], &device)),
            1,
            256,
            4
        )
        .is_err());
        assert!(normalize_conditioning(
            Some(Tensor::<TestBackend, 3>::zeros([1, 256, 2], &device)),
            1,
            256,
            4
        )
        .is_err());
        assert_eq!(
            normalize_conditioning(Some(valid), 1, 256, 4)
                .expect("valid conditioning")
                .expect("conditioning tensor")
                .dims(),
            [1, 256, 1]
        );
    }

    #[test]
    #[ignore = "requires the checksum-pinned published Glow-TTS artifact; run scripts/speech-conformance.sh (glow-tts family)"]
    fn published_glow_checkpoint_synthesizes() {
        let config_path = std::env::var_os("TONGUES_TEST_GLOW_CONFIG")
            .expect("TONGUES_TEST_GLOW_CONFIG is required");
        let checkpoint_path = std::env::var_os("TONGUES_TEST_GLOW_CHECKPOINT")
            .expect("TONGUES_TEST_GLOW_CHECKPOINT is required");
        let device = NdArrayDevice::Cpu;
        let config = GlowTtsInferenceConfig::from_file(config_path).expect("published config");
        let model =
            GlowTts::<TestBackend>::load(config, checkpoint_path, device).expect("published model");
        assert_eq!(model.config().network.out_channels, 80);
        assert_eq!(model.config().network.num_flow_blocks_dec, 12);
        let device = NdArrayDevice::Cpu;
        let token_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let lengths = Tensor::<TestBackend, 1, Int>::from_ints([4], &device);
        let durations = Tensor::<TestBackend, 2>::from_floats([[2.0, 2.0, 2.0, 2.0]], &device);
        let first = model
            .inference(
                token_ids.clone(),
                lengths.clone(),
                None,
                Some(durations.clone()),
                1.0,
                0.33,
                Some(27),
            )
            .expect("first seeded inference");
        let second = model
            .inference(
                token_ids,
                lengths,
                None,
                Some(durations),
                1.0,
                0.33,
                Some(27),
            )
            .expect("second seeded inference");
        assert_eq!(first.mel.dims(), [1, 8, 80]);
        assert_eq!(
            first.mel.into_data().to_vec::<f32>().expect("first mel"),
            second.mel.into_data().to_vec::<f32>().expect("second mel")
        );
    }

    #[test]
    #[ignore = "requires pinned Glow-TTS reference and model artifacts; run scripts/speech-conformance.sh"]
    fn published_glow_checkpoint_stage_parity() {
        let config_path = std::env::var_os("TONGUES_TEST_GLOW_CONFIG")
            .expect("TONGUES_TEST_GLOW_CONFIG is required");
        let checkpoint_path = std::env::var_os("TONGUES_TEST_GLOW_CHECKPOINT")
            .expect("TONGUES_TEST_GLOW_CHECKPOINT is required");
        let reference_path = std::env::var_os("TONGUES_TEST_COQUI_REFERENCE")
            .expect("TONGUES_TEST_COQUI_REFERENCE is required");
        let reference: Value =
            serde_json::from_slice(&std::fs::read(reference_path).expect("read reference fixture"))
                .expect("parse reference fixture");
        let fixture = &reference["glow_tts"];
        let ids = fixture["token_ids"]
            .as_array()
            .expect("Glow token IDs")
            .iter()
            .map(|value| value.as_i64().expect("token ID"))
            .collect::<Vec<_>>();
        let device = NdArrayDevice::Cpu;
        let config = GlowTtsInferenceConfig::from_file(config_path).expect("published config");
        let model = GlowTts::<TestBackend>::load(config, checkpoint_path, device).expect("model");
        let weight_fixture = &fixture["stages"]["final_invertible_weight"];
        let actual_weight = model
            .decoder
            .blocks
            .last()
            .expect("final flow block")
            .inv_conv
            .weight
            .val()
            .into_data()
            .to_vec::<f32>()
            .expect("invertible weight");
        for probe in weight_fixture["probes"]
            .as_array()
            .expect("invertible weight probes")
        {
            let probe = probe.as_array().expect("invertible weight probe");
            let index = probe[0].as_u64().expect("weight index") as usize;
            let expected = probe[1].as_f64().expect("weight value") as f32;
            assert!(
                (actual_weight[index] - expected).abs() <= 1.0e-6,
                "final invertible weight mismatch at {index}: actual={}, expected={expected}",
                actual_weight[index]
            );
        }
        let tokens = ids.len();
        let encoded = model
            .encoder
            .forward(
                Tensor::from_data(TensorData::new(ids, [1, tokens]), &device),
                Tensor::from_data(TensorData::new(vec![tokens as i64], [1]), &device),
                None,
            )
            .expect("encoder");
        assert_fixture_probes(
            "encoder mean",
            encoded.mean.clone(),
            &fixture["stages"]["encoder_mean"],
            3.0e-4,
        );
        assert_fixture_probes(
            "encoder log scale",
            encoded.log_scale.clone(),
            &fixture["stages"]["encoder_log_scale"],
            3.0e-4,
        );
        assert_fixture_probes(
            "log durations",
            encoded.log_durations.clone(),
            &fixture["stages"]["log_durations"],
            3.0e-4,
        );
        let durations = glow_ceil_durations(
            encoded.log_durations,
            encoded.mask,
            model.config().network.length_scale.into(),
            DEFAULT_GLOW_MAX_OUTPUT_FRAMES,
        )
        .expect("durations");
        let expected_durations = fixture["durations"]
            .as_array()
            .expect("duration fixture")
            .iter()
            .map(|value| value.as_f64().expect("duration") as f32)
            .collect::<Vec<_>>();
        assert_eq!(
            durations
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("duration tensor"),
            expected_durations
        );
        let expanded = expand_prior_statistics(
            encoded.mean,
            encoded.log_scale,
            durations,
            DEFAULT_GLOW_MAX_OUTPUT_FRAMES,
        )
        .expect("expanded prior");
        assert_fixture_probes(
            "alignment",
            expanded.path.clone(),
            &fixture["stages"]["alignment"],
            0.0,
        );
        let [batch, channels, frames] = expanded.mean.dims();
        let elements = batch * channels * frames;
        let pattern = (0..elements)
            .map(|index| -1.0 + 2.0 * index as f32 / (elements - 1) as f32)
            .collect::<Vec<_>>();
        let latent = (expanded.mean
            + expanded.log_scale.exp()
                * Tensor::from_data(TensorData::new(pattern, [batch, channels, frames]), &device)
                * 0.33)
            * expanded.frame_mask.clone();
        assert_fixture_probes(
            "sampled latent",
            latent.clone(),
            &fixture["stages"]["sampled_latent"],
            4.0e-4,
        );
        let (mut mel, squeezed_mask) =
            squeeze(latent, expanded.frame_mask, model.decoder.num_squeeze)
                .expect("squeeze latent");
        let step_fixture = fixture["stages"]["reverse_steps"]
            .as_array()
            .expect("reverse-step fixture");
        let trace_fixture = fixture["stages"]["reverse_flow"]
            .as_array()
            .expect("reverse-flow fixture");
        assert_eq!(step_fixture.len(), model.decoder.blocks.len() * 3);
        assert_eq!(trace_fixture.len(), model.decoder.blocks.len());
        for (index, block) in model.decoder.blocks.iter().rev().enumerate() {
            mel = block.coupling.reverse(mel, squeezed_mask.clone(), None);
            assert_fixture_probes(
                &format!("reverse coupling {}", index + 1),
                mel.clone(),
                &step_fixture[index * 3],
                7.0e-4,
            );
            mel = block.inv_conv.reverse(mel, squeezed_mask.clone());
            assert_fixture_probes(
                &format!("reverse invertible convolution {}", index + 1),
                mel.clone(),
                &step_fixture[index * 3 + 1],
                7.0e-4,
            );
            mel = block.act_norm.reverse(mel, squeezed_mask.clone());
            assert_fixture_probes(
                &format!("reverse ActNorm {}", index + 1),
                mel.clone(),
                &step_fixture[index * 3 + 2],
                7.0e-4,
            );
            assert_fixture_probes(
                &format!("reverse block {}", index + 1),
                mel.clone(),
                &trace_fixture[index],
                7.0e-4,
            );
        }
        let mel = unsqueeze(mel, squeezed_mask, model.decoder.num_squeeze);
        assert_fixture_probes(
            "mel",
            mel.swap_dims(1, 2),
            &fixture["stages"]["mel"],
            1.0e-3,
        );
    }

    #[test]
    #[ignore = "requires an external SC-GlowTTS artifact with affirmative license evidence; not part of the standard conformance lane"]
    fn external_sc_glow_checkpoint_synthesizes() {
        let config_path = std::env::var_os("TONGUES_TEST_SC_GLOW_CONFIG")
            .expect("TONGUES_TEST_SC_GLOW_CONFIG is required");
        let checkpoint_path = std::env::var_os("TONGUES_TEST_SC_GLOW_CHECKPOINT")
            .expect("TONGUES_TEST_SC_GLOW_CHECKPOINT is required");
        let device = NdArrayDevice::Cpu;
        let config = GlowTtsInferenceConfig::from_file(config_path).expect("published SC config");
        assert_eq!(config.network.speaker_conditioning_channels(), 256);
        assert!(!config.network.use_sdp);
        let model = GlowTts::<TestBackend>::load(config, checkpoint_path, device)
            .expect("published SC model");
        let output = model
            .inference(
                Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device),
                Tensor::<TestBackend, 1, Int>::from_ints([4], &device),
                Some(Tensor::zeros([1, 256, 1], &device)),
                Some(Tensor::<TestBackend, 2>::from_floats(
                    [[1.0, 1.0, 1.0, 1.0]],
                    &device,
                )),
                1.0,
                0.0,
                Some(31),
            )
            .expect("SC-GlowTTS inference");
        assert_eq!(output.mel.dims(), [1, 4, 80]);
    }
}
