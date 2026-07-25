//! Burn-native VITS text prior encoder.
//!
//! This module implements the inference graph used by the published VCTK VITS
//! checkpoint: token embedding, masked relative-position Transformer blocks,
//! and the prior mean/log-scale projection. Tokenization and linguistic
//! lowering remain outside this checkpoint boundary.

use std::fmt;
use std::path::Path;

use burn::module::{Initializer, Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, PaddingConfig1d};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::ops::PadMode;
use burn::tensor::{Int, Tensor};
use burn_store::{ModuleSnapshot, PytorchStore};

use crate::VitsInferenceConfig;

const LAYER_NORM_EPSILON: f64 = 1e-5;
const RELATIVE_ATTENTION_WINDOW: usize = 4;

#[derive(Debug)]
pub enum VitsTextPriorError {
    InvalidTopology(String),
    InvalidInput(String),
    Checkpoint(String),
}

impl fmt::Display for VitsTextPriorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTopology(message) => {
                write!(formatter, "invalid VITS text prior topology: {message}")
            }
            Self::InvalidInput(message) => {
                write!(formatter, "invalid VITS text prior input: {message}")
            }
            Self::Checkpoint(message) => {
                write!(
                    formatter,
                    "unable to load VITS text prior checkpoint: {message}"
                )
            }
        }
    }
}

impl std::error::Error for VitsTextPriorError {}

fn topology_error(message: impl Into<String>) -> VitsTextPriorError {
    VitsTextPriorError::InvalidTopology(message.into())
}

fn input_error(message: impl Into<String>) -> VitsTextPriorError {
    VitsTextPriorError::InvalidInput(message.into())
}

/// Checkpoint-independent topology of a VITS text prior encoder.
#[derive(Debug, Clone, PartialEq)]
pub struct VitsTextPriorConfig {
    pub vocabulary_size: usize,
    pub prior_channels: usize,
    pub hidden_channels: usize,
    pub ffn_channels: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub ffn_kernel_size: usize,
    pub dropout: f64,
    pub relative_attention_window: usize,
}

impl VitsTextPriorConfig {
    pub fn from_model_config(config: &VitsInferenceConfig) -> Result<Self, VitsTextPriorError> {
        config
            .validate()
            .map_err(|error| topology_error(error.to_string()))?;
        if config.network.use_language_embedding {
            return Err(topology_error(
                "language-conditioned text encoders require an explicit language input",
            ));
        }

        let config = Self {
            vocabulary_size: config.network.num_chars,
            prior_channels: config.network.hidden_channels,
            hidden_channels: config.network.hidden_channels,
            ffn_channels: config.network.hidden_channels_ffn_text_encoder,
            num_heads: config.network.num_heads_text_encoder,
            num_layers: config.network.num_layers_text_encoder,
            ffn_kernel_size: config.network.kernel_size_text_encoder,
            dropout: f64::from(config.network.dropout_p_text_encoder),
            relative_attention_window: RELATIVE_ATTENTION_WINDOW,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), VitsTextPriorError> {
        if self.vocabulary_size == 0 {
            return Err(topology_error("vocabulary_size must be positive"));
        }
        if self.prior_channels == 0
            || self.hidden_channels == 0
            || self.ffn_channels == 0
            || self.num_heads == 0
            || self.num_layers == 0
        {
            return Err(topology_error(
                "prior, hidden, FFN, head, and layer dimensions must be positive",
            ));
        }
        if !self.hidden_channels.is_multiple_of(self.num_heads) {
            return Err(topology_error(
                "hidden_channels must divide evenly across attention heads",
            ));
        }
        if self.ffn_kernel_size == 0 {
            return Err(topology_error("ffn_kernel_size must be positive"));
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(topology_error("dropout must be in [0, 1)"));
        }
        if self.relative_attention_window == 0 {
            return Err(topology_error("relative_attention_window must be positive"));
        }
        Ok(())
    }

    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<VitsTextPriorEncoder<B>, VitsTextPriorError> {
        self.validate()?;

        let embedding_std = (self.hidden_channels as f64).powf(-0.5);
        let emb = EmbeddingConfig::new(self.vocabulary_size, self.hidden_channels)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: embedding_std,
            })
            .init(device);
        let encoder = RelativePositionTransformer::init(self, device);
        let proj = Conv1dConfig::new(self.hidden_channels, self.prior_channels * 2, 1)
            .with_padding(PaddingConfig1d::Valid)
            .init(device);

        Ok(VitsTextPriorEncoder {
            emb,
            encoder,
            proj,
            hidden_channels: self.hidden_channels,
            prior_channels: self.prior_channels,
            vocabulary_size: self.vocabulary_size,
        })
    }

    pub fn load_checkpoint<B: Backend>(
        &self,
        checkpoint_path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<VitsTextPriorEncoder<B>, VitsTextPriorError> {
        self.init(device)?.load_checkpoint(checkpoint_path)
    }
}

/// Encoded text states and Gaussian prior statistics.
#[derive(Debug)]
pub struct VitsTextPriorOutput<B: Backend> {
    pub encoded: Tensor<B, 3>,
    pub mean: Tensor<B, 3>,
    pub log_scale: Tensor<B, 3>,
    pub mask: Tensor<B, 3>,
}

/// VITS text prior, with field names matching its checkpoint subtree.
#[derive(Module, Debug)]
pub struct VitsTextPriorEncoder<B: Backend> {
    pub emb: Embedding<B>,
    pub encoder: RelativePositionTransformer<B>,
    pub proj: Conv1d<B>,
    hidden_channels: usize,
    prior_channels: usize,
    vocabulary_size: usize,
}

impl<B: Backend> VitsTextPriorEncoder<B> {
    /// Loads exactly the `text_encoder` subtree from a complete VITS checkpoint.
    pub fn load_checkpoint(
        mut self,
        checkpoint_path: impl AsRef<Path>,
    ) -> Result<Self, VitsTextPriorError> {
        let mut store = PytorchStore::from_file(checkpoint_path.as_ref())
            .with_top_level_key("model")
            .with_key_remapping(r"^text_encoder\.", "")
            .with_predicate(text_prior_tensor)
            .map_indices_contiguous(false)
            .allow_partial(true)
            .skip_enum_variants(true);
        let result = self
            .load_from(&mut store)
            .map_err(|error| VitsTextPriorError::Checkpoint(error.to_string()))?;

        let mut missing = result
            .missing
            .iter()
            .map(|(path, _)| path.as_str())
            .collect::<Vec<_>>();
        missing.sort_unstable();
        let mut unused = result
            .unused
            .iter()
            .filter(|path| text_prior_tensor(path, ""))
            .cloned()
            .collect::<Vec<_>>();
        unused.sort_unstable();
        if !missing.is_empty() || !result.errors.is_empty() || !unused.is_empty() {
            return Err(VitsTextPriorError::Checkpoint(format!(
                "text prior subtree does not exactly match the Burn module: missing [{}], {} load errors, unused [{}]",
                missing.join(", "),
                result.errors.len(),
                unused.join(", ")
            )));
        }
        Ok(self)
    }

    /// Encodes model-local token IDs shaped `[batch, tokens]`.
    ///
    /// `lengths` contains the valid token count for each batch item. Padding is
    /// excluded from attention and all returned tensors by the sequence mask.
    pub fn forward(
        &self,
        token_ids: Tensor<B, 2, Int>,
        lengths: Tensor<B, 1, Int>,
    ) -> Result<VitsTextPriorOutput<B>, VitsTextPriorError> {
        let [batch, tokens] = token_ids.dims();
        let [length_batch] = lengths.dims();
        if batch == 0 {
            return Err(input_error("batch must contain at least one sequence"));
        }
        if tokens == 0 {
            return Err(input_error("each sequence must contain at least one token"));
        }
        if length_batch != batch {
            return Err(input_error(format!(
                "length batch {length_batch} does not match token batch {batch}"
            )));
        }

        let mask = sequence_mask(lengths, tokens);
        let embedded = self.emb.forward(token_ids) * (self.hidden_channels as f64).sqrt();
        let encoded = self
            .encoder
            .forward(embedded.swap_dims(1, 2) * mask.clone(), mask.clone());
        let stats = self.proj.forward(encoded.clone()) * mask.clone();
        let mean = stats
            .clone()
            .slice([0..batch, 0..self.prior_channels, 0..tokens]);
        let log_scale = stats.slice([
            0..batch,
            self.prior_channels..self.prior_channels * 2,
            0..tokens,
        ]);

        Ok(VitsTextPriorOutput {
            encoded,
            mean,
            log_scale,
            mask,
        })
    }

    pub fn hidden_channels(&self) -> usize {
        self.hidden_channels
    }

    pub fn prior_channels(&self) -> usize {
        self.prior_channels
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }
}

/// Builds `[batch, 1, max_length]` with ones before each sequence length.
pub fn sequence_mask<B: Backend>(lengths: Tensor<B, 1, Int>, max_length: usize) -> Tensor<B, 3> {
    let [batch] = lengths.dims();
    let device = lengths.device();
    let positions = Tensor::<B, 1, Int>::arange(0..max_length as i64, &device)
        .reshape([1, max_length])
        .repeat_dim(0, batch);
    let lengths = lengths.reshape([batch, 1]).repeat_dim(1, max_length);
    positions
        .lower(lengths)
        .float()
        .reshape([batch, 1, max_length])
}

#[derive(Module, Debug)]
pub struct RelativePositionTransformer<B: Backend> {
    pub attn_layers: Vec<RelativePositionMultiHeadAttention<B>>,
    pub norm_layers_1: Vec<ChannelLayerNorm<B>>,
    pub ffn_layers: Vec<FeedForwardNetwork<B>>,
    pub norm_layers_2: Vec<ChannelLayerNorm<B>>,
    dropout: Dropout,
}

impl<B: Backend> RelativePositionTransformer<B> {
    fn init(config: &VitsTextPriorConfig, device: &B::Device) -> Self {
        let mut attn_layers = Vec::with_capacity(config.num_layers);
        let mut norm_layers_1 = Vec::with_capacity(config.num_layers);
        let mut ffn_layers = Vec::with_capacity(config.num_layers);
        let mut norm_layers_2 = Vec::with_capacity(config.num_layers);

        for _ in 0..config.num_layers {
            attn_layers.push(RelativePositionMultiHeadAttention::init(
                config.hidden_channels,
                config.num_heads,
                config.relative_attention_window,
                config.dropout,
                device,
            ));
            norm_layers_1.push(ChannelLayerNorm::init(config.hidden_channels, device));
            ffn_layers.push(FeedForwardNetwork::init(
                config.hidden_channels,
                config.ffn_channels,
                config.ffn_kernel_size,
                config.dropout,
                device,
            ));
            norm_layers_2.push(ChannelLayerNorm::init(config.hidden_channels, device));
        }

        Self {
            attn_layers,
            norm_layers_1,
            ffn_layers,
            norm_layers_2,
            dropout: DropoutConfig::new(config.dropout).init(),
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
            input = self.norm_layers_1[layer].forward(input + self.dropout.forward(attention));

            let ffn = self.ffn_layers[layer].forward(input.clone(), mask.clone());
            input = self.norm_layers_2[layer].forward(input + self.dropout.forward(ffn));
        }
        input * mask
    }
}

#[derive(Module, Debug)]
pub struct RelativePositionMultiHeadAttention<B: Backend> {
    pub conv_q: Conv1d<B>,
    pub conv_k: Conv1d<B>,
    pub conv_v: Conv1d<B>,
    pub conv_o: Conv1d<B>,
    pub emb_rel_k: Param<Tensor<B, 3>>,
    pub emb_rel_v: Param<Tensor<B, 3>>,
    dropout: Dropout,
    num_heads: usize,
    channels_per_head: usize,
    relative_attention_window: usize,
}

impl<B: Backend> RelativePositionMultiHeadAttention<B> {
    fn init(
        channels: usize,
        num_heads: usize,
        relative_attention_window: usize,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        let conv = || {
            Conv1dConfig::new(channels, channels, 1)
                .with_padding(PaddingConfig1d::Valid)
                .init(device)
        };
        let channels_per_head = channels / num_heads;
        let relative_std = (channels_per_head as f64).powf(-0.5);
        let relative_shape = [1, relative_attention_window * 2 + 1, channels_per_head];

        Self {
            conv_q: conv(),
            conv_k: conv(),
            conv_v: conv(),
            conv_o: conv(),
            emb_rel_k: Initializer::Normal {
                mean: 0.0,
                std: relative_std,
            }
            .init(relative_shape, device),
            emb_rel_v: Initializer::Normal {
                mean: 0.0,
                std: relative_std,
            }
            .init(relative_shape, device),
            dropout: DropoutConfig::new(dropout).init(),
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
        debug_assert_eq!(source_tokens, target_tokens);

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
        let relative_keys = self.relative_embeddings(self.emb_rel_k.val(), source_tokens);
        let relative_logits = query.clone().matmul(
            relative_keys
                .reshape([1, 1, source_tokens * 2 - 1, self.channels_per_head])
                .swap_dims(2, 3),
        );
        let relative_logits = relative_position_to_absolute(relative_logits) / scale;
        let mut scores = query.matmul(key.swap_dims(2, 3)) / scale + relative_logits;
        scores = scores.mask_fill(attention_mask.equal_elem(0.0), -1.0e4);

        let attention = self.dropout.forward(softmax(scores, 3));
        let mut output = attention.clone().matmul(value);
        let relative_weights = absolute_position_to_relative(attention);
        let relative_values = self.relative_embeddings(self.emb_rel_v.val(), source_tokens);
        output = output
            + relative_weights.matmul(relative_values.reshape([
                1,
                1,
                source_tokens * 2 - 1,
                self.channels_per_head,
            ]));

        output
            .swap_dims(2, 3)
            .reshape([batch, channels, target_tokens])
    }

    fn relative_embeddings(&self, embeddings: Tensor<B, 3>, length: usize) -> Tensor<B, 3> {
        let padding = length.saturating_sub(self.relative_attention_window + 1);
        let slice_start = (self.relative_attention_window + 1).saturating_sub(length);
        let slice_end = slice_start + length * 2 - 1;
        let embeddings = if padding > 0 {
            embeddings.pad([(0, 0), (padding, padding), (0, 0)], PadMode::Constant(0.0))
        } else {
            embeddings
        };
        embeddings.slice([0..1, slice_start..slice_end, 0..self.channels_per_head])
    }
}

fn relative_position_to_absolute<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, length, _] = input.dims();
    input
        .pad([(0, 0), (0, 0), (0, 0), (0, 1)], PadMode::Constant(0.0))
        .reshape([batch, heads, length * 2 * length])
        .pad([(0, 0), (0, 0), (0, length - 1)], PadMode::Constant(0.0))
        .reshape([batch, heads, length + 1, length * 2 - 1])
        .slice([0..batch, 0..heads, 0..length, length - 1..length * 2 - 1])
}

fn absolute_position_to_relative<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
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
pub struct FeedForwardNetwork<B: Backend> {
    pub conv_1: Conv1d<B>,
    pub conv_2: Conv1d<B>,
    dropout: Dropout,
    pad_left: usize,
    pad_right: usize,
}

impl<B: Backend> FeedForwardNetwork<B> {
    fn init(
        channels: usize,
        hidden_channels: usize,
        kernel_size: usize,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            conv_1: Conv1dConfig::new(channels, hidden_channels, kernel_size)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            conv_2: Conv1dConfig::new(hidden_channels, channels, kernel_size)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            dropout: DropoutConfig::new(dropout).init(),
            pad_left: (kernel_size - 1) / 2,
            pad_right: kernel_size / 2,
        }
    }

    fn forward(&self, input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        let padded = (input * mask.clone()).pad(
            [(0, 0), (0, 0), (self.pad_left, self.pad_right)],
            PadMode::Constant(0.0),
        );
        let hidden = self.dropout.forward(relu(self.conv_1.forward(padded)));
        let padded = (hidden * mask.clone()).pad(
            [(0, 0), (0, 0), (self.pad_left, self.pad_right)],
            PadMode::Constant(0.0),
        );
        self.conv_2.forward(padded) * mask
    }
}

/// Layer normalization over the channel axis with checkpoint names
/// `gamma`/`beta`.
#[derive(Module, Debug)]
pub struct ChannelLayerNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 1>>,
    pub beta: Param<Tensor<B, 1>>,
    epsilon: f64,
}

impl<B: Backend> ChannelLayerNorm<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            gamma: Initializer::Ones.init([channels], device),
            beta: Initializer::Zeros.init([channels], device),
            epsilon: LAYER_NORM_EPSILON,
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mean = input.clone().mean_dim(1);
        let variance = (input.clone() - mean.clone()).square().mean_dim(1);
        let normalized = (input - mean) / (variance + self.epsilon).sqrt();
        normalized * self.gamma.val().reshape([1, self.gamma.dims()[0], 1])
            + self.beta.val().reshape([1, self.beta.dims()[0], 1])
    }
}

fn text_prior_tensor(path: &str, _container: &str) -> bool {
    [
        "emb.",
        "encoder.attn_layers.",
        "encoder.norm_layers_1.",
        "encoder.ffn_layers.",
        "encoder.norm_layers_2.",
        "proj.",
    ]
    .iter()
    .any(|prefix| path.starts_with(prefix))
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    use super::*;

    type TestBackend = NdArray<f32>;

    fn tiny_config() -> VitsTextPriorConfig {
        VitsTextPriorConfig {
            vocabulary_size: 12,
            prior_channels: 4,
            hidden_channels: 4,
            ffn_channels: 8,
            num_heads: 2,
            num_layers: 2,
            ffn_kernel_size: 3,
            dropout: 0.1,
            relative_attention_window: 4,
        }
    }

    #[test]
    fn sequence_mask_tracks_each_batch_length() {
        let device = NdArrayDevice::Cpu;
        let lengths = Tensor::<TestBackend, 1, Int>::from_ints([2, 4], &device);

        let mask = sequence_mask(lengths, 5);

        assert_eq!(mask.dims(), [2, 1, 5]);
        assert_eq!(
            mask.to_data().to_vec::<f32>().expect("mask values"),
            vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        );
    }

    #[test]
    fn tiny_prior_is_deterministic_and_masks_padding() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 37);
        let encoder = tiny_config().init::<TestBackend>(&device).expect("encoder");
        let tokens =
            Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4], [4, 3, 0, 0]], &device);
        let lengths = Tensor::<TestBackend, 1, Int>::from_ints([4, 2], &device);

        let first = encoder
            .forward(tokens.clone(), lengths.clone())
            .expect("first forward");
        let second = encoder.forward(tokens, lengths).expect("second forward");

        assert_eq!(first.encoded.dims(), [2, 4, 4]);
        assert_eq!(first.mean.dims(), [2, 4, 4]);
        assert_eq!(first.log_scale.dims(), [2, 4, 4]);
        assert_eq!(first.mask.dims(), [2, 1, 4]);
        assert_eq!(first.mean.to_data(), second.mean.to_data());

        let masked_mean = first.mean.slice([1..2, 0..4, 2..4]);
        assert_eq!(
            masked_mean.to_data().to_vec::<f32>().expect("masked mean"),
            vec![0.0; 8]
        );
    }

    #[test]
    fn rejects_mismatched_length_batch() {
        let device = NdArrayDevice::Cpu;
        let encoder = tiny_config().init::<TestBackend>(&device).expect("encoder");
        let tokens = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2], [2, 3]], &device);
        let lengths = Tensor::<TestBackend, 1, Int>::from_ints([2], &device);

        let error = encoder
            .forward(tokens, lengths)
            .expect_err("length batch must fail");

        assert!(error.to_string().contains("does not match token batch"));
    }

    #[test]
    fn published_checkpoint_loads_and_runs_when_provided() {
        let Some(config_path) = std::env::var_os("TONGUES_TEST_COQUI_VITS_CONFIG") else {
            return;
        };
        let Some(checkpoint_path) = std::env::var_os("TONGUES_TEST_COQUI_VITS_CHECKPOINT") else {
            return;
        };
        let model_config = VitsInferenceConfig::from_file(config_path).expect("published config");
        let config =
            VitsTextPriorConfig::from_model_config(&model_config).expect("published topology");

        assert_eq!(config.vocabulary_size, 179);
        assert_eq!(config.prior_channels, 192);
        assert_eq!(config.hidden_channels, 192);
        assert_eq!(config.ffn_channels, 768);
        assert_eq!(config.num_heads, 2);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.relative_attention_window, 4);

        let device = NdArrayDevice::Cpu;
        let encoder = config
            .load_checkpoint::<TestBackend>(checkpoint_path, &device)
            .expect("strict text prior load");
        let tokens = Tensor::<TestBackend, 2, Int>::from_ints([[0, 18, 178, 1]], &device);
        let lengths = Tensor::<TestBackend, 1, Int>::from_ints([3], &device);

        let output = encoder.forward(tokens, lengths).expect("published forward");

        assert_eq!(output.encoded.dims(), [1, 192, 4]);
        assert_eq!(output.mean.dims(), [1, 192, 4]);
        assert_eq!(output.log_scale.dims(), [1, 192, 4]);
        assert_eq!(output.mask.dims(), [1, 1, 4]);
        assert_eq!(
            output.mask.to_data().to_vec::<f32>().expect("mask"),
            vec![1.0, 1.0, 1.0, 0.0]
        );
        assert!(output
            .mean
            .to_data()
            .to_vec::<f32>()
            .expect("prior mean")
            .iter()
            .all(|value| value.is_finite()));
    }
}
