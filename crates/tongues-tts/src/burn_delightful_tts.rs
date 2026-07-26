//! Native Burn inference graph for Coqui DelightfulTTS acoustic checkpoints.
//!
//! DelightfulTTS extends duration-based synthesis with Conformer encoder and
//! decoder stacks, explicit pitch/energy adaptors, predicted utterance- and
//! phoneme-level prosody, and optional speaker conditioning. Training-only
//! reference encoders and the alignment network are intentionally outside the
//! inference graph.
//!
//! Source provenance: `audit-required`. The implementation follows the
//! inference topology published in Coqui TTS v0.22.0.

use std::fmt;
use std::path::Path;

use burn::module::{Initializer, Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, PaddingConfig1d};
use burn::tensor::activation::{leaky_relu, sigmoid, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::module::embedding;
use burn::tensor::{ElementConversion, Int, Tensor};

use crate::burn_speedy_speech::expand_by_durations;
use crate::{DelightfulConformerConfig, DelightfulTtsConfig};

const NORM_EPSILON: f64 = 1e-5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DelightfulTtsError {
    InvalidConfig(String),
    InvalidInput(String),
    Checkpoint(String),
}

impl fmt::Display for DelightfulTtsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid DelightfulTTS config: {message}")
            }
            Self::InvalidInput(message) => {
                write!(formatter, "invalid DelightfulTTS input: {message}")
            }
            Self::Checkpoint(message) => {
                write!(
                    formatter,
                    "unable to load DelightfulTTS checkpoint: {message}"
                )
            }
        }
    }
}

impl std::error::Error for DelightfulTtsError {}

fn input_error(message: impl Into<String>) -> DelightfulTtsError {
    DelightfulTtsError::InvalidInput(message.into())
}

#[derive(Module, Debug)]
pub struct DelightfulLayerNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 1>>,
    pub beta: Param<Tensor<B, 1>>,
}

impl<B: Backend> DelightfulLayerNorm<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            gamma: Initializer::Ones.init([channels], device),
            beta: Initializer::Zeros.init([channels], device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let channels = input.dims()[2];
        let mean = input.clone().mean_dim(2);
        let variance = (input.clone() - mean.clone()).square().mean_dim(2);
        (input - mean) / (variance + NORM_EPSILON).sqrt()
            * self.gamma.val().reshape([1, 1, channels])
            + self.beta.val().reshape([1, 1, channels])
    }
}

#[derive(Module, Debug)]
pub struct DelightfulGroupNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 1>>,
    pub beta: Param<Tensor<B, 1>>,
}

impl<B: Backend> DelightfulGroupNorm<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            gamma: Initializer::Ones.init([channels], device),
            beta: Initializer::Zeros.init([channels], device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, frames] = input.dims();
        let flattened = input.reshape([batch, 1, channels * frames]);
        let mean = flattened.clone().mean_dim(2);
        let variance = (flattened.clone() - mean.clone()).square().mean_dim(2);
        let normalized = ((flattened - mean) / (variance + NORM_EPSILON).sqrt())
            .reshape([batch, channels, frames]);
        normalized * self.gamma.val().reshape([1, channels, 1])
            + self.beta.val().reshape([1, channels, 1])
    }
}

#[derive(Module, Debug)]
pub struct DelightfulBsConv1d<B: Backend> {
    pub pointwise: Conv1d<B>,
    pub depthwise: Conv1d<B>,
}

impl<B: Backend> DelightfulBsConv1d<B> {
    fn init(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            pointwise: Conv1dConfig::new(channels_in, channels_out, 1).init(device),
            depthwise: Conv1dConfig::new(channels_out, channels_out, kernel_size)
                .with_groups(channels_out)
                .with_padding(PaddingConfig1d::Explicit(kernel_size / 2, kernel_size / 2))
                .init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.depthwise.forward(self.pointwise.forward(input))
    }
}

#[derive(Module, Debug)]
pub struct DelightfulConditioning<B: Backend> {
    pub conv: DelightfulBsConv1d<B>,
    pub embedding_proj: Linear<B>,
}

impl<B: Backend> DelightfulConditioning<B> {
    fn init(
        channels: usize,
        embedding_channels: usize,
        kernel_size: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            conv: DelightfulBsConv1d::init(channels, channels * 2, kernel_size, device),
            embedding_proj: LinearConfig::new(embedding_channels, channels).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>, conditioning: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, frames, channels] = input.dims();
        let residual = input.clone().swap_dims(1, 2);
        let gated = self.conv.forward(residual.clone());
        let a = gated.clone().slice([0..batch, 0..channels, 0..frames]);
        let b = gated.slice([0..batch, channels..channels * 2, 0..frames]);
        let projected = self
            .embedding_proj
            .forward(conditioning)
            .reshape([batch, channels, 1]);
        let softsign = projected.clone() / (projected.abs() + 1.0);
        ((a + softsign.expand([batch, channels, frames])) * sigmoid(b) + residual)
            * std::f64::consts::FRAC_1_SQRT_2
    }
}

#[derive(Module, Debug)]
pub struct DelightfulFeedForward<B: Backend> {
    pub ln: DelightfulLayerNorm<B>,
    pub conv_1: Conv1d<B>,
    pub conv_2: Conv1d<B>,
    slope: f64,
}

impl<B: Backend> DelightfulFeedForward<B> {
    fn init(channels: usize, slope: f64, device: &B::Device) -> Self {
        Self {
            ln: DelightfulLayerNorm::init(channels, device),
            conv_1: Conv1dConfig::new(channels, channels * 4, 3)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .init(device),
            conv_2: Conv1dConfig::new(channels * 4, channels, 1).init(device),
            slope,
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let normalized = self.ln.forward(input).swap_dims(1, 2);
        let hidden = leaky_relu(self.conv_1.forward(normalized), self.slope);
        self.conv_2.forward(hidden).swap_dims(1, 2) * 0.5
    }
}

#[derive(Module, Debug)]
pub struct DelightfulConvolutionModule<B: Backend> {
    pub ln_1: DelightfulLayerNorm<B>,
    pub conv_1: Conv1d<B>,
    pub depthwise: Conv1d<B>,
    pub ln_2: DelightfulGroupNorm<B>,
    pub conv_2: Conv1d<B>,
    slope: f64,
    inner_channels: usize,
}

impl<B: Backend> DelightfulConvolutionModule<B> {
    fn init(channels: usize, kernel_size: usize, slope: f64, device: &B::Device) -> Self {
        let inner_channels = channels * 2;
        Self {
            ln_1: DelightfulLayerNorm::init(channels, device),
            conv_1: Conv1dConfig::new(channels, inner_channels * 2, 1).init(device),
            depthwise: Conv1dConfig::new(inner_channels, inner_channels, kernel_size)
                .with_groups(inner_channels)
                .with_padding(PaddingConfig1d::Explicit(kernel_size / 2, kernel_size / 2))
                .init(device),
            ln_2: DelightfulGroupNorm::init(inner_channels, device),
            conv_2: Conv1dConfig::new(inner_channels, channels, 1).init(device),
            slope,
            inner_channels,
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, frames, _] = input.dims();
        let hidden = self
            .conv_1
            .forward(self.ln_1.forward(input).swap_dims(1, 2));
        let value = hidden
            .clone()
            .slice([0..batch, 0..self.inner_channels, 0..frames]);
        let gate = hidden.slice([
            0..batch,
            self.inner_channels..self.inner_channels * 2,
            0..frames,
        ]);
        let hidden = value * leaky_relu(gate, self.slope);
        let hidden = self.depthwise.forward(hidden);
        let hidden = leaky_relu(self.ln_2.forward(hidden), self.slope);
        self.conv_2.forward(hidden).swap_dims(1, 2)
    }
}

#[derive(Module, Debug)]
pub struct DelightfulRelativeAttention<B: Backend> {
    pub query_proj: Linear<B>,
    pub key_proj: Linear<B>,
    pub value_proj: Linear<B>,
    pub pos_proj: Linear<B>,
    pub u_bias: Param<Tensor<B, 2>>,
    pub v_bias: Param<Tensor<B, 2>>,
    pub out_proj: Linear<B>,
    heads: usize,
    channels_per_head: usize,
    channels: usize,
}

impl<B: Backend> DelightfulRelativeAttention<B> {
    fn init(channels: usize, heads: usize, device: &B::Device) -> Self {
        let channels_per_head = channels / heads;
        Self {
            query_proj: LinearConfig::new(channels, channels).init(device),
            key_proj: LinearConfig::new(channels, channels)
                .with_bias(false)
                .init(device),
            value_proj: LinearConfig::new(channels, channels)
                .with_bias(false)
                .init(device),
            pos_proj: LinearConfig::new(channels, channels)
                .with_bias(false)
                .init(device),
            u_bias: Initializer::XavierUniform { gain: 1.0 }.init_with(
                [heads, channels_per_head],
                Some(channels_per_head),
                Some(channels_per_head),
                device,
            ),
            v_bias: Initializer::XavierUniform { gain: 1.0 }.init_with(
                [heads, channels_per_head],
                Some(channels_per_head),
                Some(channels_per_head),
                device,
            ),
            out_proj: LinearConfig::new(channels, channels).init(device),
            heads,
            channels_per_head,
            channels,
        }
    }

    fn forward(&self, input: Tensor<B, 3>, positions: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, frames, _] = input.dims();
        let query = self
            .query_proj
            .forward(input.clone())
            .reshape([batch, frames, self.heads, self.channels_per_head])
            .swap_dims(1, 2);
        let key = self
            .key_proj
            .forward(input.clone())
            .reshape([batch, frames, self.heads, self.channels_per_head])
            .swap_dims(1, 2);
        let value = self
            .value_proj
            .forward(input)
            .reshape([batch, frames, self.heads, self.channels_per_head])
            .swap_dims(1, 2);
        let positions = self
            .pos_proj
            .forward(positions.expand([batch, frames, self.channels]))
            .reshape([batch, frames, self.heads, self.channels_per_head])
            .swap_dims(1, 2);
        let u = self
            .u_bias
            .val()
            .reshape([1, self.heads, 1, self.channels_per_head]);
        let v = self
            .v_bias
            .val()
            .reshape([1, self.heads, 1, self.channels_per_head]);
        let content = (query.clone() + u).matmul(key.clone().swap_dims(2, 3));
        let relative = (query + v).matmul(positions.swap_dims(2, 3));
        let relative = relative_shift(relative);
        let attention = softmax((content + relative) / (self.channels as f64).sqrt(), 3);
        let output =
            attention
                .matmul(value)
                .swap_dims(1, 2)
                .reshape([batch, frames, self.channels]);
        self.out_proj.forward(output)
    }
}

fn relative_shift<B: Backend>(scores: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, frames, _] = scores.dims();
    let padded = scores.pad([(1, 0)], burn::tensor::ops::PadMode::Constant(0.0));
    padded
        .reshape([batch, heads, frames + 1, frames])
        .slice([0..batch, 0..heads, 1..frames + 1, 0..frames])
        .reshape([batch, heads, frames, frames])
}

#[derive(Module, Debug)]
pub struct DelightfulConformerBlock<B: Backend> {
    pub conditioning: Option<DelightfulConditioning<B>>,
    pub ff: DelightfulFeedForward<B>,
    pub conformer_conv_1: DelightfulConvolutionModule<B>,
    pub ln: DelightfulLayerNorm<B>,
    pub slf_attn: DelightfulSelfAttention<B>,
    pub conformer_conv_2: DelightfulConvolutionModule<B>,
}

#[derive(Module, Debug)]
pub struct DelightfulSelfAttention<B: Backend> {
    pub attention: DelightfulRelativeAttention<B>,
}

impl<B: Backend> DelightfulConformerBlock<B> {
    fn init(
        config: &DelightfulConformerConfig,
        conditioning_channels: Option<usize>,
        slope: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            conditioning: conditioning_channels.map(|conditioning_channels| {
                DelightfulConditioning::init(
                    config.hidden_channels,
                    conditioning_channels,
                    config.convolution_kernel_size,
                    device,
                )
            }),
            ff: DelightfulFeedForward::init(config.hidden_channels, slope, device),
            conformer_conv_1: DelightfulConvolutionModule::init(
                config.hidden_channels,
                config.convolution_kernel_size,
                slope,
                device,
            ),
            ln: DelightfulLayerNorm::init(config.hidden_channels, device),
            slf_attn: DelightfulSelfAttention {
                attention: DelightfulRelativeAttention::init(
                    config.hidden_channels,
                    config.heads,
                    device,
                ),
            },
            conformer_conv_2: DelightfulConvolutionModule::init(
                config.hidden_channels,
                config.convolution_kernel_size,
                slope,
                device,
            ),
        }
    }

    fn forward(
        &self,
        mut input: Tensor<B, 3>,
        positions: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 2>>,
        mask: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>, DelightfulTtsError> {
        match (&self.conditioning, conditioning) {
            (Some(layer), Some(conditioning)) => {
                input = layer.forward(input, conditioning).swap_dims(1, 2);
            }
            (None, None) => {}
            (Some(_), None) => {
                return Err(input_error(
                    "DelightfulTTS checkpoint requires speaker conditioning",
                ));
            }
            (None, Some(_)) => {
                return Err(input_error(
                    "speaker conditioning was supplied to a single-speaker DelightfulTTS checkpoint",
                ));
            }
        }
        input = input.clone() + self.ff.forward(input);
        input = input.clone() + self.conformer_conv_1.forward(input);
        let residual = input.clone();
        input = self
            .slf_attn
            .attention
            .forward(self.ln.forward(input), positions);
        input = (input + residual) * mask;
        Ok(input.clone() + self.conformer_conv_2.forward(input))
    }
}

#[derive(Module, Debug)]
pub struct DelightfulConformer<B: Backend> {
    pub layer_stack: Vec<DelightfulConformerBlock<B>>,
}

impl<B: Backend> DelightfulConformer<B> {
    fn init(
        config: &DelightfulConformerConfig,
        conditioning_channels: Option<usize>,
        slope: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            layer_stack: (0..config.layers)
                .map(|_| {
                    DelightfulConformerBlock::init(config, conditioning_channels, slope, device)
                })
                .collect(),
        }
    }

    fn forward(
        &self,
        mut input: Tensor<B, 3>,
        positions: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 2>>,
        mask: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>, DelightfulTtsError> {
        for layer in &self.layer_stack {
            input = layer.forward(input, positions.clone(), conditioning.clone(), mask.clone())?;
        }
        Ok(input)
    }
}

#[derive(Module, Debug)]
pub struct DelightfulConvTransposed<B: Backend> {
    pub conv: DelightfulBsConv1d<B>,
}

impl<B: Backend> DelightfulConvTransposed<B> {
    fn init(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            conv: DelightfulBsConv1d::init(channels_in, channels_out, kernel_size, device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.conv.forward(input.swap_dims(1, 2)).swap_dims(1, 2)
    }
}

#[derive(Module, Debug)]
pub struct DelightfulVariancePredictor<B: Backend> {
    pub conv_1: DelightfulConvTransposed<B>,
    pub norm_1: DelightfulLayerNorm<B>,
    pub conv_2: DelightfulConvTransposed<B>,
    pub norm_2: DelightfulLayerNorm<B>,
    pub linear_layer: Linear<B>,
    slope: f64,
}

impl<B: Backend> DelightfulVariancePredictor<B> {
    fn init(
        channels_in: usize,
        hidden_channels: usize,
        output_channels: usize,
        kernel_size: usize,
        slope: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            conv_1: DelightfulConvTransposed::init(
                channels_in,
                hidden_channels,
                kernel_size,
                device,
            ),
            norm_1: DelightfulLayerNorm::init(hidden_channels, device),
            conv_2: DelightfulConvTransposed::init(
                hidden_channels,
                hidden_channels,
                kernel_size,
                device,
            ),
            norm_2: DelightfulLayerNorm::init(hidden_channels, device),
            linear_layer: LinearConfig::new(hidden_channels, output_channels).init(device),
            slope,
        }
    }

    fn forward(&self, input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, tokens, _] = input.dims();
        let hidden = self
            .norm_1
            .forward(leaky_relu(self.conv_1.forward(input), self.slope));
        let hidden = self
            .norm_2
            .forward(leaky_relu(self.conv_2.forward(hidden), self.slope));
        self.linear_layer.forward(hidden).reshape([batch, tokens]) * mask.reshape([batch, tokens])
    }
}

#[derive(Module, Debug)]
pub struct DelightfulVarianceAdaptor<B: Backend> {
    pub predictor: DelightfulVariancePredictor<B>,
    pub embedding: Conv1d<B>,
}

impl<B: Backend> DelightfulVarianceAdaptor<B> {
    fn init(config: &DelightfulTtsConfig, device: &B::Device) -> Self {
        let channels = config.encoder.hidden_channels;
        Self {
            predictor: DelightfulVariancePredictor::init(
                channels,
                config.variance.hidden_channels,
                1,
                config.variance.kernel_size,
                config.leaky_relu_slope,
                device,
            ),
            embedding: Conv1dConfig::new(1, channels, config.variance.embedding_kernel_size)
                .with_padding(PaddingConfig1d::Explicit(
                    config.variance.embedding_kernel_size / 2,
                    config.variance.embedding_kernel_size / 2,
                ))
                .init(device),
        }
    }

    fn forward(
        &self,
        encoded: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        explicit: Option<Tensor<B, 3>>,
        scale: f64,
        shift: f64,
        label: &str,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>), DelightfulTtsError> {
        let [batch, tokens, _] = encoded.dims();
        let predicted = self
            .predictor
            .forward(encoded, mask.clone())
            .reshape([batch, 1, tokens]);
        let values = match explicit {
            Some(values) if values.dims() == [batch, 1, tokens] => values,
            Some(values) => {
                return Err(input_error(format!(
                    "explicit {label} has shape {:?}; expected [{batch}, 1, {tokens}]",
                    values.dims()
                )));
            }
            None => predicted,
        };
        let values = (values * scale + shift) * mask.swap_dims(1, 2);
        Ok((self.embedding.forward(values.clone()), values))
    }
}

#[derive(Module, Debug)]
pub struct DelightfulProsodyPredictor<B: Backend> {
    pub conv_1: DelightfulConvTransposed<B>,
    pub norm_1: DelightfulLayerNorm<B>,
    pub conv_2: DelightfulConvTransposed<B>,
    pub norm_2: DelightfulLayerNorm<B>,
    pub predictor_bottleneck: Linear<B>,
    slope: f64,
}

impl<B: Backend> DelightfulProsodyPredictor<B> {
    fn init(
        channels: usize,
        bottleneck: usize,
        kernel_size: usize,
        slope: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            conv_1: DelightfulConvTransposed::init(channels, channels, kernel_size, device),
            norm_1: DelightfulLayerNorm::init(channels, device),
            conv_2: DelightfulConvTransposed::init(channels, channels, kernel_size, device),
            norm_2: DelightfulLayerNorm::init(channels, device),
            predictor_bottleneck: LinearConfig::new(channels, bottleneck).init(device),
            slope,
        }
    }

    fn forward(&self, input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self
            .norm_1
            .forward(leaky_relu(self.conv_1.forward(input), self.slope));
        let hidden = self
            .norm_2
            .forward(leaky_relu(self.conv_2.forward(hidden), self.slope));
        self.predictor_bottleneck.forward(hidden) * mask
    }
}

#[derive(Module, Debug)]
pub struct DelightfulEmbedding<B: Backend> {
    pub embeddings: Param<Tensor<B, 2>>,
    padding_id: usize,
}

impl<B: Backend> DelightfulEmbedding<B> {
    fn init(vocabulary: usize, channels: usize, padding_id: usize, device: &B::Device) -> Self {
        Self {
            embeddings: Initializer::KaimingNormal {
                gain: (2.0f64).sqrt(),
                fan_out_only: false,
            }
            .init_with(
                [vocabulary, channels],
                Some(channels),
                Some(vocabulary),
                device,
            ),
            padding_id,
        }
    }

    fn forward(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut weights = self.embeddings.val();
        let [vocabulary, channels] = weights.dims();
        if self.padding_id < vocabulary {
            let device = weights.device();
            let multiplier = Tensor::<B, 2>::ones([vocabulary, 1], &device)
                .slice_assign(
                    [self.padding_id..self.padding_id + 1, 0..1],
                    Tensor::zeros([1, 1], &device),
                )
                .expand([vocabulary, channels]);
            weights = weights * multiplier;
        }
        embedding(weights, ids)
    }
}

#[derive(Debug)]
pub struct DelightfulTtsControls<B: Backend> {
    pub length_scale: f64,
    pub durations: Option<Tensor<B, 2>>,
    pub pitch_scale: f64,
    pub pitch_shift: f64,
    pub pitch: Option<Tensor<B, 3>>,
    pub energy_scale: f64,
    pub energy_shift: f64,
    pub energy: Option<Tensor<B, 3>>,
    pub speaker_ids: Option<Tensor<B, 1, Int>>,
    pub d_vectors: Option<Tensor<B, 2>>,
}

impl<B: Backend> Default for DelightfulTtsControls<B> {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            durations: None,
            pitch_scale: 1.0,
            pitch_shift: 0.0,
            pitch: None,
            energy_scale: 1.0,
            energy_shift: 0.0,
            energy: None,
            speaker_ids: None,
            d_vectors: None,
        }
    }
}

#[derive(Debug)]
pub struct DelightfulTtsOutput<B: Backend> {
    pub mel: Tensor<B, 3>,
    pub durations: Tensor<B, 2>,
    pub pitch: Tensor<B, 3>,
    pub energy: Tensor<B, 3>,
    pub utterance_prosody: Tensor<B, 3>,
    pub phoneme_prosody: Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct DelightfulTts<B: Backend> {
    pub encoder: DelightfulConformer<B>,
    pub pitch_adaptor: DelightfulVarianceAdaptor<B>,
    pub energy_adaptor: DelightfulVarianceAdaptor<B>,
    pub duration_predictor: DelightfulVariancePredictor<B>,
    pub utterance_prosody_predictor: DelightfulProsodyPredictor<B>,
    pub phoneme_prosody_predictor: DelightfulProsodyPredictor<B>,
    pub u_bottle_out: Linear<B>,
    pub p_bottle_out: Linear<B>,
    pub decoder: DelightfulConformer<B>,
    pub src_word_emb: DelightfulEmbedding<B>,
    pub to_mel: Linear<B>,
    pub emb_g: Option<Embedding<B>>,
    config: DelightfulTtsConfig,
}

impl DelightfulTtsConfig {
    pub fn init<B: Backend>(
        &self,
        padding_id: usize,
        device: &B::Device,
    ) -> Result<DelightfulTts<B>, DelightfulTtsError> {
        self.validate()
            .map_err(|error| DelightfulTtsError::InvalidConfig(error.to_string()))?;
        if padding_id >= self.num_chars {
            return Err(DelightfulTtsError::InvalidConfig(format!(
                "padding ID {padding_id} is outside vocabulary 0..{}",
                self.num_chars
            )));
        }
        let channels = self.encoder.hidden_channels;
        let conditioning_channels = self.speakers.conditioning_dimensions();
        Ok(DelightfulTts {
            encoder: DelightfulConformer::init(
                &self.encoder,
                conditioning_channels,
                self.leaky_relu_slope,
                device,
            ),
            pitch_adaptor: DelightfulVarianceAdaptor::init(self, device),
            energy_adaptor: DelightfulVarianceAdaptor::init(self, device),
            duration_predictor: DelightfulVariancePredictor::init(
                channels,
                self.variance.hidden_channels,
                1,
                self.variance.kernel_size,
                self.leaky_relu_slope,
                device,
            ),
            utterance_prosody_predictor: DelightfulProsodyPredictor::init(
                channels,
                self.prosody.utterance_bottleneck,
                self.prosody.predictor_kernel_size,
                self.leaky_relu_slope,
                device,
            ),
            phoneme_prosody_predictor: DelightfulProsodyPredictor::init(
                channels,
                self.prosody.phoneme_bottleneck,
                self.prosody.predictor_kernel_size,
                self.leaky_relu_slope,
                device,
            ),
            u_bottle_out: LinearConfig::new(self.prosody.utterance_bottleneck, channels)
                .init(device),
            p_bottle_out: LinearConfig::new(self.prosody.phoneme_bottleneck, channels).init(device),
            decoder: DelightfulConformer::init(
                &self.decoder,
                conditioning_channels,
                self.leaky_relu_slope,
                device,
            ),
            src_word_emb: DelightfulEmbedding::init(self.num_chars, channels, padding_id, device),
            to_mel: LinearConfig::new(self.decoder.hidden_channels, self.out_channels).init(device),
            emb_g: self.speakers.use_speaker_embedding.then(|| {
                EmbeddingConfig::new(
                    self.speakers.num_speakers,
                    self.speakers.speaker_embedding_channels,
                )
                .init(device)
            }),
            config: self.clone(),
        })
    }
}

impl<B: Backend> DelightfulTts<B> {
    /// Loads the inference subset from a Coqui DelightfulTTS training checkpoint.
    pub fn load_checkpoint(mut self, path: impl AsRef<Path>) -> Result<Self, DelightfulTtsError> {
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut self,
            path.as_ref(),
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                predicate: Some(delightful_inference_tensor),
                key_remappings: vec![
                    (r"^acoustic_model\.".into(), String::new()),
                    (
                        r"^src_word_emb\.weight$".into(),
                        "src_word_emb.embeddings".into(),
                    ),
                    (
                        r"^(pitch_adaptor)\.pitch_predictor\.".into(),
                        "$1.predictor.".into(),
                    ),
                    (
                        r"^(pitch_adaptor)\.pitch_emb\.".into(),
                        "$1.embedding.".into(),
                    ),
                    (
                        r"^(energy_adaptor)\.energy_predictor\.".into(),
                        "$1.predictor.".into(),
                    ),
                    (
                        r"^(energy_adaptor)\.energy_emb\.".into(),
                        "$1.embedding.".into(),
                    ),
                    (
                        r"^((?:duration_predictor|pitch_adaptor\.predictor|energy_adaptor\.predictor|utterance_prosody_predictor|phoneme_prosody_predictor))\.layers\.0\.".into(),
                        "$1.conv_1.".into(),
                    ),
                    (
                        r"^((?:duration_predictor|pitch_adaptor\.predictor|energy_adaptor\.predictor|utterance_prosody_predictor|phoneme_prosody_predictor))\.layers\.2\.".into(),
                        "$1.norm_1.".into(),
                    ),
                    (
                        r"^((?:duration_predictor|pitch_adaptor\.predictor|energy_adaptor\.predictor|utterance_prosody_predictor|phoneme_prosody_predictor))\.layers\.4\.".into(),
                        "$1.conv_2.".into(),
                    ),
                    (
                        r"^((?:duration_predictor|pitch_adaptor\.predictor|energy_adaptor\.predictor|utterance_prosody_predictor|phoneme_prosody_predictor))\.layers\.6\.".into(),
                        "$1.norm_2.".into(),
                    ),
                    (
                        r"(\.conformer_conv_[12]\.(?:conv_1|conv_2))\.conv\.".into(),
                        "$1.".into(),
                    ),
                    (
                        r"(\.conformer_conv_[12]\.depthwise)\.conv\.".into(),
                        "$1.".into(),
                    ),
                    (
                        r"(\.(?:ln|ln_[12]|norm_[12]))\.weight$".into(),
                        "$1.gamma".into(),
                    ),
                    (
                        r"(\.(?:ln|ln_[12]|norm_[12]))\.bias$".into(),
                        "$1.beta".into(),
                    ),
                ],
                map_indices_contiguous: true,
                skip_enum_variants: true,
                ..Default::default()
            },
        )
        .map_err(|error| DelightfulTtsError::Checkpoint(format!("{error:#}")))?;
        if !result.missing.is_empty() || !result.errors.is_empty() {
            return Err(DelightfulTtsError::Checkpoint(format!(
                "checkpoint inference subset does not match: {} missing tensors and {} load errors",
                result.missing.len(),
                result.errors.len()
            )));
        }
        Ok(self)
    }

    pub fn inference(
        &self,
        token_ids: Tensor<B, 2, Int>,
    ) -> Result<DelightfulTtsOutput<B>, DelightfulTtsError> {
        self.inference_with_controls(
            token_ids,
            DelightfulTtsControls {
                length_scale: self.config.length_scale,
                ..Default::default()
            },
        )
    }

    pub fn inference_with_controls(
        &self,
        token_ids: Tensor<B, 2, Int>,
        controls: DelightfulTtsControls<B>,
    ) -> Result<DelightfulTtsOutput<B>, DelightfulTtsError> {
        validate_controls(&controls)?;
        let [batch, tokens] = token_ids.dims();
        if batch == 0 || tokens == 0 {
            return Err(input_error(
                "token IDs must have non-empty [batch, tokens] dimensions",
            ));
        }
        let highest = token_ids.clone().max().into_scalar().elem::<i64>();
        if highest < 0 || highest as usize >= self.config.num_chars {
            return Err(input_error(format!(
                "token ID {highest} is outside vocabulary 0..{}",
                self.config.num_chars
            )));
        }
        let device = token_ids.device();
        let token_mask = Tensor::<B, 3>::ones([batch, tokens, 1], &device);
        let conditioning =
            self.resolve_conditioning(controls.speaker_ids, controls.d_vectors, batch)?;
        let positions =
            positional_encoding::<B>(self.config.encoder.hidden_channels, tokens, &device);
        let embedded = self.src_word_emb.forward(token_ids) * token_mask.clone();
        let mut encoded = self.encoder.forward(
            embedded,
            positions,
            conditioning.clone(),
            token_mask.clone(),
        )?;

        let utterance = self.utterance_prosody_predictor.forward(
            encoded.clone(),
            token_mask
                .clone()
                .expand([batch, tokens, self.config.prosody.utterance_bottleneck]),
        );
        let utterance = utterance.sum_dim(1) / tokens as f64;
        let utterance = normalize_last(utterance);
        encoded = encoded
            + self.u_bottle_out.forward(utterance.clone()).expand([
                batch,
                tokens,
                self.config.encoder.hidden_channels,
            ]);

        let phoneme = self.phoneme_prosody_predictor.forward(
            encoded.clone(),
            token_mask
                .clone()
                .expand([batch, tokens, self.config.prosody.phoneme_bottleneck]),
        );
        let phoneme = normalize_last(phoneme);
        encoded = encoded + self.p_bottle_out.forward(phoneme.clone());
        let duration_source = encoded.clone();

        let (pitch_embedding, pitch) = self.pitch_adaptor.forward(
            encoded.clone(),
            token_mask.clone(),
            controls.pitch,
            controls.pitch_scale,
            controls.pitch_shift,
            "pitch",
        )?;
        let (energy_embedding, energy) = self.energy_adaptor.forward(
            encoded,
            token_mask.clone(),
            controls.energy,
            controls.energy_scale,
            controls.energy_shift,
            "energy",
        )?;
        let encoded = duration_source.clone().swap_dims(1, 2) + pitch_embedding + energy_embedding;
        let durations = match controls.durations {
            Some(durations) if durations.dims() == [batch, tokens] => durations,
            Some(durations) => {
                return Err(input_error(format!(
                    "explicit durations have shape {:?}; expected [{batch}, {tokens}]",
                    durations.dims()
                )));
            }
            None => {
                let log_durations = self.duration_predictor.forward(duration_source, token_mask);
                ((log_durations.exp() - 1.0) * controls.length_scale)
                    .clamp(1.0, self.config.max_duration as f64)
                    .round()
            }
        };
        let (expanded, output_mask) =
            expand_by_durations(encoded, durations.clone(), self.config.max_output_frames)
                .map_err(|error| input_error(error.to_string()))?;
        let frames = expanded.dims()[2];
        let frame_positions =
            positional_encoding::<B>(self.config.decoder.hidden_channels, frames, &device);
        let decoded = self.decoder.forward(
            expanded.swap_dims(1, 2),
            frame_positions,
            conditioning,
            output_mask.swap_dims(1, 2),
        )?;
        let mel = self.to_mel.forward(decoded);
        Ok(DelightfulTtsOutput {
            mel,
            durations,
            pitch,
            energy,
            utterance_prosody: utterance,
            phoneme_prosody: phoneme,
        })
    }

    fn resolve_conditioning(
        &self,
        speaker_ids: Option<Tensor<B, 1, Int>>,
        d_vectors: Option<Tensor<B, 2>>,
        batch: usize,
    ) -> Result<Option<Tensor<B, 2>>, DelightfulTtsError> {
        if speaker_ids.is_some() && d_vectors.is_some() {
            return Err(input_error("use either speaker IDs or d-vectors, not both"));
        }
        if let Some(ids) = speaker_ids {
            let Some(embedding) = &self.emb_g else {
                return Err(input_error(
                    "speaker IDs were supplied to a checkpoint without learned speaker embeddings",
                ));
            };
            if ids.dims() != [batch] {
                return Err(input_error(format!(
                    "speaker IDs have shape {:?}; expected [{batch}]",
                    ids.dims()
                )));
            }
            let highest = ids.clone().max().into_scalar().elem::<i64>();
            if highest < 0 || highest as usize >= self.config.speakers.num_speakers {
                return Err(input_error(format!(
                    "speaker ID {highest} is outside 0..{}",
                    self.config.speakers.num_speakers
                )));
            }
            let values = embedding
                .forward(ids.reshape([batch, 1]))
                .reshape([batch, self.config.speakers.speaker_embedding_channels]);
            return Ok(Some(l2_normalize(values)));
        }
        if let Some(values) = d_vectors {
            let Some(dimensions) = self
                .config
                .speakers
                .use_d_vector_file
                .then_some(self.config.speakers.d_vector_dim)
            else {
                return Err(input_error(
                    "d-vectors were supplied to a checkpoint without d-vector conditioning",
                ));
            };
            if values.dims() != [batch, dimensions] {
                return Err(input_error(format!(
                    "d-vectors have shape {:?}; expected [{batch}, {dimensions}]",
                    values.dims()
                )));
            }
            return Ok(Some(l2_normalize(values)));
        }
        if self.config.speakers.conditioning_dimensions().is_some() {
            return Err(input_error(
                "this DelightfulTTS checkpoint requires speaker conditioning",
            ));
        }
        Ok(None)
    }
}

fn validate_controls<B: Backend>(
    controls: &DelightfulTtsControls<B>,
) -> Result<(), DelightfulTtsError> {
    if !controls.length_scale.is_finite() || controls.length_scale <= 0.0 {
        return Err(input_error("length_scale must be finite and positive"));
    }
    for (label, scale, shift) in [
        ("pitch", controls.pitch_scale, controls.pitch_shift),
        ("energy", controls.energy_scale, controls.energy_shift),
    ] {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(input_error(format!(
                "{label}_scale must be finite and positive"
            )));
        }
        if !shift.is_finite() {
            return Err(input_error(format!("{label}_shift must be finite")));
        }
    }
    Ok(())
}

fn normalize_last<B: Backend>(input: Tensor<B, 3>) -> Tensor<B, 3> {
    let mean = input.clone().mean_dim(2);
    let variance = (input.clone() - mean.clone()).square().mean_dim(2);
    (input - mean) / (variance + NORM_EPSILON).sqrt()
}

fn l2_normalize<B: Backend>(input: Tensor<B, 2>) -> Tensor<B, 2> {
    let norm = input.clone().square().sum_dim(1).sqrt().clamp_min(1e-12);
    input / norm
}

fn positional_encoding<B: Backend>(
    channels: usize,
    frames: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mut values = vec![0.0f32; frames * channels];
    for frame in 0..frames {
        for channel in 0..channels {
            let divisor = 10_000f32.powf((2 * (channel / 2)) as f32 / channels as f32);
            let phase = frame as f32 / divisor;
            values[frame * channels + channel] = if channel.is_multiple_of(2) {
                phase.sin()
            } else {
                phase.cos()
            };
        }
    }
    Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([1, frames, channels])
}

fn delightful_inference_tensor(path: &str, _container: &str) -> bool {
    path.starts_with("acoustic_model.")
        && ![
            ".aligner.",
            ".utterance_prosody_encoder.",
            ".phoneme_prosody_encoder.",
            ".energy_scaler.",
            ".padding_mult",
        ]
        .iter()
        .any(|ignored| path.contains(ignored))
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::TensorData;

    use super::*;
    use crate::{
        DelightfulAudioConfig, DelightfulProsodyConfig, DelightfulSpeakerConfig,
        DelightfulVarianceConfig,
    };

    type TestBackend = NdArray<f32>;

    fn tiny_config() -> DelightfulTtsConfig {
        DelightfulTtsConfig {
            num_chars: 12,
            out_channels: 3,
            encoder: DelightfulConformerConfig {
                hidden_channels: 8,
                layers: 1,
                heads: 2,
                dropout: 0.1,
                convolution_kernel_size: 3,
            },
            decoder: DelightfulConformerConfig {
                hidden_channels: 8,
                layers: 1,
                heads: 2,
                dropout: 0.1,
                convolution_kernel_size: 3,
            },
            variance: DelightfulVarianceConfig {
                hidden_channels: 8,
                kernel_size: 3,
                dropout: 0.1,
                embedding_kernel_size: 3,
            },
            prosody: DelightfulProsodyConfig {
                utterance_bottleneck: 8,
                phoneme_bottleneck: 2,
                predictor_kernel_size: 3,
            },
            speakers: DelightfulSpeakerConfig::default(),
            leaky_relu_slope: 0.3,
            length_scale: 1.0,
            max_duration: 10,
            max_output_frames: 64,
            audio: DelightfulAudioConfig {
                num_mels: 3,
                mel_fmax: 8_000.0,
                ..DelightfulAudioConfig::default()
            },
        }
    }

    #[test]
    fn explicit_variance_and_duration_controls_produce_mel_frames() {
        let device = NdArrayDevice::Cpu;
        let model = tiny_config()
            .init::<TestBackend>(0, &device)
            .expect("model");
        let tokens = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device);
        let output = model
            .inference_with_controls(
                tokens,
                DelightfulTtsControls {
                    durations: Some(Tensor::from_floats([[1.0, 2.0, 1.0]], &device)),
                    pitch: Some(Tensor::from_data(
                        TensorData::new(vec![0.2, 0.3, 0.1], [1, 1, 3]),
                        &device,
                    )),
                    energy: Some(Tensor::from_data(
                        TensorData::new(vec![0.5, 0.4, 0.6], [1, 1, 3]),
                        &device,
                    )),
                    ..Default::default()
                },
            )
            .expect("inference");
        assert_eq!(output.mel.dims(), [1, 4, 3]);
        assert_eq!(output.durations.dims(), [1, 3]);
        assert_eq!(output.pitch.dims(), [1, 1, 3]);
        assert_eq!(output.energy.dims(), [1, 1, 3]);
        assert_eq!(output.utterance_prosody.dims(), [1, 1, 8]);
        assert_eq!(output.phoneme_prosody.dims(), [1, 3, 2]);
    }

    #[test]
    fn learned_speaker_conditioning_is_required_and_bounded() {
        let device = NdArrayDevice::Cpu;
        let mut config = tiny_config();
        config.speakers = DelightfulSpeakerConfig {
            num_speakers: 2,
            use_speaker_embedding: true,
            speaker_embedding_channels: 4,
            use_d_vector_file: false,
            d_vector_dim: 0,
        };
        let model = config.init::<TestBackend>(0, &device).expect("model");
        let tokens = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device);
        assert!(model.inference(tokens.clone()).is_err());
        let output = model.inference_with_controls(
            tokens,
            DelightfulTtsControls {
                durations: Some(Tensor::from_floats([[1.0, 1.0, 1.0]], &device)),
                speaker_ids: Some(Tensor::from_ints([1], &device)),
                ..Default::default()
            },
        );
        assert!(output.is_ok());
    }
}
