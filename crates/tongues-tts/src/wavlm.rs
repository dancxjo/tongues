//! Burn-native WavLM content features used by voice-conversion backends.
//!
//! The inference topology follows Microsoft's MIT-licensed WavLM Large model.
//! Training-only masking, layer drop, and gradient scaling are intentionally
//! absent. Checkpoint names remain compatible with the published PyTorch
//! artifact used by FreeVC.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use burn::module::{Initializer, Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig1d,
};
use burn::tensor::activation::{gelu, sigmoid, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Int, Tensor, TensorData};

const WAVLM_CONV_LAYERS: &[(usize, usize, usize)] = &[
    (512, 10, 5),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 2, 2),
    (512, 2, 2),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavLmConfig {
    pub encoder_layers: usize,
    pub encoder_embed_dim: usize,
    pub encoder_ffn_embed_dim: usize,
    pub encoder_attention_heads: usize,
    pub conv_pos: usize,
    pub conv_pos_groups: usize,
    pub relative_position_embedding: bool,
    pub num_buckets: usize,
    pub max_distance: usize,
    pub gru_rel_pos: bool,
}

impl WavLmConfig {
    pub fn large() -> Self {
        Self {
            encoder_layers: 24,
            encoder_embed_dim: 1_024,
            encoder_ffn_embed_dim: 4_096,
            encoder_attention_heads: 16,
            conv_pos: 128,
            conv_pos_groups: 16,
            relative_position_embedding: true,
            num_buckets: 320,
            max_distance: 800,
            gru_rel_pos: true,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.encoder_layers > 0 && self.encoder_embed_dim > 0 && self.encoder_ffn_embed_dim > 0,
            "WavLM transformer dimensions must be positive"
        );
        ensure!(
            self.encoder_embed_dim
                .is_multiple_of(self.encoder_attention_heads),
            "WavLM embedding size must divide evenly across attention heads"
        );
        ensure!(
            self.conv_pos > 0
                && self.conv_pos_groups > 0
                && self.encoder_embed_dim.is_multiple_of(self.conv_pos_groups),
            "WavLM positional convolution topology is invalid"
        );
        if self.relative_position_embedding {
            ensure!(
                self.num_buckets >= 4
                    && self.num_buckets.is_multiple_of(2)
                    && self.max_distance > self.num_buckets / 4,
                "WavLM relative-position bucket topology is invalid"
            );
        }
        Ok(())
    }
}

#[derive(Module, Debug)]
struct WavLmFeatureBlock<B: Backend> {
    conv: Conv1d<B>,
    layer_norm: LayerNorm<B>,
}

impl<B: Backend> WavLmFeatureBlock<B> {
    fn init(
        channels_in: usize,
        channels_out: usize,
        kernel: usize,
        stride: usize,
        _first: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            conv: Conv1dConfig::new(channels_in, channels_out, kernel)
                .with_stride(stride)
                .with_bias(false)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            layer_norm: LayerNormConfig::new(channels_out).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let output = self.conv.forward(input);
        gelu(
            self.layer_norm
                .forward(output.swap_dims(1, 2))
                .swap_dims(1, 2),
        )
    }
}

#[derive(Module, Debug)]
struct WavLmFeatureExtractor<B: Backend> {
    conv_layers: Vec<WavLmFeatureBlock<B>>,
}

impl<B: Backend> WavLmFeatureExtractor<B> {
    fn init(device: &B::Device) -> Self {
        let mut channels_in = 1;
        let conv_layers = WAVLM_CONV_LAYERS
            .iter()
            .enumerate()
            .map(|(index, &(channels_out, kernel, stride))| {
                let layer = WavLmFeatureBlock::init(
                    channels_in,
                    channels_out,
                    kernel,
                    stride,
                    index == 0,
                    device,
                );
                channels_in = channels_out;
                layer
            })
            .collect();
        Self { conv_layers }
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, samples] = input.dims();
        let mut output = input.reshape([batch, 1, samples]);
        for layer in &self.conv_layers {
            output = layer.forward(output);
        }
        output
    }
}

#[derive(Module, Debug)]
struct WeightNormPositionalConv<B: Backend> {
    weight_g: Param<Tensor<B, 3>>,
    weight_v: Param<Tensor<B, 3>>,
    bias: Param<Tensor<B, 1>>,
    groups: usize,
    padding: usize,
    remove_last: bool,
}

impl<B: Backend> WeightNormPositionalConv<B> {
    fn init(channels: usize, kernel: usize, groups: usize, device: &B::Device) -> Self {
        let weight_v = Initializer::Normal {
            mean: 0.0,
            std: ((4.0 / (kernel * channels) as f64).sqrt()),
        }
        .init([channels, channels / groups, kernel], device);
        let weight_g = weight_norm_dim_two(weight_v.val()).detach();
        Self {
            weight_g: Param::from_tensor(weight_g),
            weight_v,
            bias: Initializer::Zeros.init([channels], device),
            groups,
            padding: kernel / 2,
            remove_last: kernel.is_multiple_of(2),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, frames] = input.dims();
        let weight_v = self.weight_v.val();
        let weight = weight_v.clone() * self.weight_g.val() / weight_norm_dim_two(weight_v);
        let output = conv1d(
            input,
            weight,
            Some(self.bias.val()),
            ConvOptions::new([1], [self.padding], [1], self.groups),
        );
        let output = if self.remove_last && output.dims()[2] > frames {
            output.slice([0..batch, 0..channels, 0..frames])
        } else {
            output
        };
        gelu(output)
    }
}

fn weight_norm_dim_two<B: Backend>(weight: Tensor<B, 3>) -> Tensor<B, 3> {
    weight.powf_scalar(2.0).sum_dims(&[0usize, 1usize]).sqrt()
}

#[derive(Module, Debug)]
struct WavLmAttention<B: Backend> {
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    q_proj: Linear<B>,
    out_proj: Linear<B>,
    relative_attention_bias: Option<Embedding<B>>,
    grep_linear: Option<Linear<B>>,
    grep_a: Option<Param<Tensor<B, 4>>>,
    num_heads: usize,
    head_dim: usize,
    num_buckets: usize,
    max_distance: usize,
}

impl<B: Backend> WavLmAttention<B> {
    fn init(config: &WavLmConfig, relative_bias: bool, device: &B::Device) -> Self {
        let embed = config.encoder_embed_dim;
        let num_heads = config.encoder_attention_heads;
        let head_dim = embed / num_heads;
        Self {
            k_proj: LinearConfig::new(embed, embed).init(device),
            v_proj: LinearConfig::new(embed, embed).init(device),
            q_proj: LinearConfig::new(embed, embed).init(device),
            out_proj: LinearConfig::new(embed, embed).init(device),
            relative_attention_bias: relative_bias
                .then(|| EmbeddingConfig::new(config.num_buckets, num_heads).init(device)),
            grep_linear: config
                .gru_rel_pos
                .then(|| LinearConfig::new(head_dim, 8).init(device)),
            grep_a: config
                .gru_rel_pos
                .then(|| Initializer::Ones.init([1, num_heads, 1, 1], device)),
            num_heads,
            head_dim,
            num_buckets: config.num_buckets,
            max_distance: config.max_distance,
        }
    }

    fn relative_bias(&self, frames: usize, device: &B::Device) -> Option<Tensor<B, 4>> {
        let embedding = self.relative_attention_bias.as_ref()?;
        let buckets =
            relative_position_buckets(frames, frames, self.num_buckets, self.max_distance);
        let buckets =
            Tensor::<B, 2, Int>::from_data(TensorData::new(buckets, [frames, frames]), device);
        Some(
            embedding
                .forward(buckets)
                .swap_dims(0, 2)
                .swap_dims(1, 2)
                .reshape([1, self.num_heads, frames, frames]),
        )
    }

    fn forward(
        &self,
        input: Tensor<B, 3>,
        position_bias: Option<Tensor<B, 4>>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 4>>) {
        let [batch, frames, embed] = input.dims();
        let position_bias = position_bias.or_else(|| self.relative_bias(frames, &input.device()));
        let q = self
            .q_proj
            .forward(input.clone())
            .reshape([batch, frames, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = self
            .k_proj
            .forward(input.clone())
            .reshape([batch, frames, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(input)
            .reshape([batch, frames, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let mut scores = q.clone().matmul(k.transpose()) / (self.head_dim as f64).sqrt();
        if let Some(bias) = position_bias.clone() {
            let scaled_bias = match (&self.grep_linear, &self.grep_a) {
                (Some(gate), Some(gate_a)) => {
                    let gates = sigmoid(
                        gate.forward(q)
                            .reshape([batch, self.num_heads, frames, 2, 4])
                            .sum_dim(4)
                            .reshape([batch, self.num_heads, frames, 2]),
                    );
                    let first = gates
                        .clone()
                        .slice([0..batch, 0..self.num_heads, 0..frames, 0..1]);
                    let second = gates.slice([0..batch, 0..self.num_heads, 0..frames, 1..2]);
                    let scale = first * (second * gate_a.val() - 1.0) + 2.0;
                    scale * bias.expand([batch, self.num_heads, frames, frames])
                }
                _ => bias.expand([batch, self.num_heads, frames, frames]),
            };
            scores = scores + scaled_bias;
        }
        let attended = softmax(scores, 3).matmul(v);
        let attended = attended.swap_dims(1, 2).reshape([batch, frames, embed]);
        (self.out_proj.forward(attended), position_bias)
    }
}

fn relative_position_buckets(
    query: usize,
    key: usize,
    buckets: usize,
    max_distance: usize,
) -> Vec<i64> {
    let half = buckets / 2;
    let exact = half / 2;
    let log_denominator = (max_distance as f64 / exact as f64).ln();
    let mut output = Vec::with_capacity(query * key);
    for q in 0..query {
        for k in 0..key {
            let relative = k as isize - q as isize;
            let direction = usize::from(relative > 0) * half;
            let distance = relative.unsigned_abs();
            let bucket = if distance < exact {
                distance
            } else {
                let logarithmic = exact
                    + ((distance as f64 / exact as f64).ln() / log_denominator
                        * (half - exact) as f64) as usize;
                logarithmic.min(half - 1)
            };
            output.push((direction + bucket) as i64);
        }
    }
    output
}

#[derive(Module, Debug)]
struct WavLmTransformerLayer<B: Backend> {
    self_attn: WavLmAttention<B>,
    self_attn_layer_norm: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    final_layer_norm: LayerNorm<B>,
}

impl<B: Backend> WavLmTransformerLayer<B> {
    fn init(config: &WavLmConfig, index: usize, device: &B::Device) -> Self {
        Self {
            self_attn: WavLmAttention::init(
                config,
                config.relative_position_embedding && index == 0,
                device,
            ),
            self_attn_layer_norm: LayerNormConfig::new(config.encoder_embed_dim).init(device),
            fc1: LinearConfig::new(config.encoder_embed_dim, config.encoder_ffn_embed_dim)
                .init(device),
            fc2: LinearConfig::new(config.encoder_ffn_embed_dim, config.encoder_embed_dim)
                .init(device),
            final_layer_norm: LayerNormConfig::new(config.encoder_embed_dim).init(device),
        }
    }

    fn forward(
        &self,
        input: Tensor<B, 3>,
        position_bias: Option<Tensor<B, 4>>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 4>>) {
        // WavLM Large uses pre-normalization. Dropout is disabled in inference.
        let residual = input.clone();
        let (attention, position_bias) = self
            .self_attn
            .forward(self.self_attn_layer_norm.forward(input), position_bias);
        let output = residual + attention;
        let residual = output.clone();
        let output = self.final_layer_norm.forward(output);
        let output = self.fc2.forward(gelu(self.fc1.forward(output)));
        (residual + output, position_bias)
    }
}

#[derive(Module, Debug)]
struct WavLmTransformerEncoder<B: Backend> {
    pos_conv: WeightNormPositionalConv<B>,
    layers: Vec<WavLmTransformerLayer<B>>,
    layer_norm: LayerNorm<B>,
}

impl<B: Backend> WavLmTransformerEncoder<B> {
    fn init(config: &WavLmConfig, device: &B::Device) -> Self {
        Self {
            pos_conv: WeightNormPositionalConv::init(
                config.encoder_embed_dim,
                config.conv_pos,
                config.conv_pos_groups,
                device,
            ),
            layers: (0..config.encoder_layers)
                .map(|index| WavLmTransformerLayer::init(config, index, device))
                .collect(),
            layer_norm: LayerNormConfig::new(config.encoder_embed_dim).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut output =
            input.clone() + self.pos_conv.forward(input.swap_dims(1, 2)).swap_dims(1, 2);
        let mut position_bias = None;
        for layer in &self.layers {
            (output, position_bias) = layer.forward(output, position_bias);
        }
        self.layer_norm.forward(output)
    }
}

#[derive(Module, Debug)]
pub struct WavLm<B: Backend> {
    feature_extractor: WavLmFeatureExtractor<B>,
    post_extract_proj: Linear<B>,
    layer_norm: LayerNorm<B>,
    encoder: WavLmTransformerEncoder<B>,
    config: WavLmConfig,
}

impl<B: Backend> WavLm<B> {
    pub fn init(config: WavLmConfig, device: &B::Device) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            feature_extractor: WavLmFeatureExtractor::init(device),
            post_extract_proj: LinearConfig::new(512, config.encoder_embed_dim).init(device),
            layer_norm: LayerNormConfig::new(512).init(device),
            encoder: WavLmTransformerEncoder::init(&config, device),
            config,
        })
    }

    pub fn load_large(checkpoint_path: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        let mut model = Self::init(WavLmConfig::large(), device)?;
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut model,
            checkpoint_path.as_ref(),
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                predicate: None,
                key_remappings: wavlm_checkpoint_remappings(),
                map_indices_contiguous: false,
                allow_partial: true,
                skip_enum_variants: true,
            },
        )
        .context("failed to load WavLM Large checkpoint")?;
        let unused = result
            .unused
            .iter()
            .filter(|name| !wavlm_training_only_tensor(name))
            .collect::<Vec<_>>();
        ensure!(
            result.missing.is_empty() && result.errors.is_empty() && unused.is_empty(),
            "WavLM checkpoint mismatch: {} missing, {} load errors, unused [{}]",
            result.missing.len(),
            result.errors.len(),
            unused
                .into_iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(model)
    }

    /// Extract `[batch, 1024, frames]` content features from 16 kHz mono PCM.
    pub fn extract_features(&self, input: Tensor<B, 2>) -> Result<Tensor<B, 3>> {
        let [batch, samples] = input.dims();
        ensure!(batch > 0 && samples >= 400, "WavLM input is too short");
        let features = self.feature_extractor.forward(input).swap_dims(1, 2);
        let features = self.layer_norm.forward(features);
        let features = self.post_extract_proj.forward(features);
        Ok(self.encoder.forward(features).swap_dims(1, 2))
    }

    pub fn output_dimensions(&self) -> usize {
        self.config.encoder_embed_dim
    }
}

fn wavlm_checkpoint_remappings() -> Vec<(String, String)> {
    vec![
        (
            r"^feature_extractor\.conv_layers\.(\d+)\.0\.".into(),
            "feature_extractor.conv_layers.$1.conv.".into(),
        ),
        (
            r"^feature_extractor\.conv_layers\.(\d+)\.2\.1\.".into(),
            "feature_extractor.conv_layers.$1.layer_norm.".into(),
        ),
        (
            r"^encoder\.pos_conv\.0\.".into(),
            "encoder.pos_conv.".into(),
        ),
        (
            r"(^|\.)(layer_norm|self_attn_layer_norm|final_layer_norm)\.weight$".into(),
            "$1$2.gamma".into(),
        ),
        (
            r"(^|\.)(layer_norm|self_attn_layer_norm|final_layer_norm)\.bias$".into(),
            "$1$2.beta".into(),
        ),
    ]
}

fn wavlm_training_only_tensor(name: &str) -> bool {
    name == "mask_emb"
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    use super::*;

    type TestBackend = NdArray<f32>;

    #[test]
    fn relative_buckets_are_directional_and_bounded() {
        let buckets = relative_position_buckets(4, 4, 320, 800);
        assert_eq!(buckets.len(), 16);
        assert_eq!(buckets[0], 0);
        assert_ne!(buckets[1], buckets[4]);
        assert!(buckets.into_iter().all(|bucket| (0..320).contains(&bucket)));
    }

    #[test]
    fn feature_extractor_has_expected_frame_rate() {
        let device = NdArrayDevice::Cpu;
        let extractor = WavLmFeatureExtractor::<TestBackend>::init(&device);
        let output = extractor.forward(Tensor::zeros([1, 16_000], &device));
        assert_eq!(output.dims(), [1, 512, 49]);
    }
}
