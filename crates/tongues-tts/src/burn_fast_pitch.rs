//! Native Burn implementation of Coqui's released LJSpeech FastPitch model.
//!
//! The v0.6.1 checkpoint uses the original `FastPitch` container: feed-forward
//! Transformer encoder and decoder stacks, parallel duration and pitch
//! predictors, and a token-level pitch embedding. Text normalization and
//! checkpoint-local vocabulary projection remain outside this module.
//!
//! Source provenance: `audit-required`. This module targets published Coqui
//! checkpoint structure and behavior; no claim of independent implementation
//! or source adaptation should be made until the ledger in
//! `docs/provenance.md` records a file-by-file comparison.

use std::fmt;
use std::path::Path;
use std::time::Instant;

use burn::module::{Initializer, Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, PaddingConfig1d};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::burn_speedy_speech::{
    expand_by_durations, AlignmentNetwork, DurationPredictor, PositionalEncoding,
};
use crate::profiling::finish_backend_stage;
use crate::{SynthesisDimension, SynthesisProfiler, SynthesisStage};

const LAYER_NORM_EPSILON: f64 = 1e-5;
const MAX_POSITIONAL_FRAMES: usize = 5_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FastPitchError {
    InvalidConfig(String),
    InvalidInput(String),
    Checkpoint(String),
}

impl fmt::Display for FastPitchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid FastPitch config: {message}")
            }
            Self::InvalidInput(message) => write!(formatter, "invalid FastPitch input: {message}"),
            Self::Checkpoint(message) => {
                write!(formatter, "unable to load FastPitch checkpoint: {message}")
            }
        }
    }
}

impl std::error::Error for FastPitchError {}

fn config_error(message: impl Into<String>) -> FastPitchError {
    FastPitchError::InvalidConfig(message.into())
}

fn input_error(message: impl Into<String>) -> FastPitchError {
    FastPitchError::InvalidInput(message.into())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedForwardTransformerConfig {
    pub hidden_channels_ffn: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dropout: f64,
}

impl FeedForwardTransformerConfig {
    pub(crate) fn from_value(value: &Value, label: &str) -> Result<Self, FastPitchError> {
        let config = Self {
            hidden_channels_ffn: usize_at(value, &["hidden_channels_ffn"])?,
            num_heads: usize_at(value, &["num_heads"])?,
            num_layers: usize_at(value, &["num_layers"])?,
            dropout: number_at(value, &["dropout_p"])?,
        };
        config.validate(label)?;
        Ok(config)
    }

    pub(crate) fn validate(&self, label: &str) -> Result<(), FastPitchError> {
        if self.hidden_channels_ffn == 0 || self.num_heads == 0 || self.num_layers == 0 {
            return Err(config_error(format!(
                "{label} FFN channels, heads, and layers must be positive"
            )));
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(config_error(format!("{label} dropout must be in [0, 1)")));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FastPitchConfig {
    pub num_chars: usize,
    pub out_channels: usize,
    pub hidden_channels: usize,
    pub positional_encoding: bool,
    pub length_scale: f64,
    pub encoder: FeedForwardTransformerConfig,
    pub decoder: FeedForwardTransformerConfig,
    pub duration_predictor_hidden_channels: usize,
    pub duration_predictor_kernel_size: usize,
    pub duration_predictor_dropout: f64,
    pub pitch_predictor_hidden_channels: usize,
    pub pitch_predictor_kernel_size: usize,
    pub pitch_predictor_dropout: f64,
    pub pitch_embedding_kernel_size: usize,
    pub use_aligner: bool,
    pub max_duration: usize,
    pub max_output_frames: usize,
}

impl FastPitchConfig {
    pub fn ljspeech() -> Self {
        let transformer = FeedForwardTransformerConfig {
            hidden_channels_ffn: 1_024,
            num_heads: 1,
            num_layers: 6,
            dropout: 0.1,
        };
        Self {
            num_chars: 130,
            out_channels: 80,
            hidden_channels: 384,
            positional_encoding: true,
            length_scale: 1.0,
            encoder: transformer.clone(),
            decoder: transformer,
            duration_predictor_hidden_channels: 256,
            duration_predictor_kernel_size: 3,
            duration_predictor_dropout: 0.1,
            pitch_predictor_hidden_channels: 256,
            pitch_predictor_kernel_size: 3,
            pitch_predictor_dropout: 0.1,
            pitch_embedding_kernel_size: 3,
            use_aligner: true,
            max_duration: 75,
            max_output_frames: 20_000,
        }
    }

    pub fn from_json_value(root: &Value) -> Result<Self, FastPitchError> {
        let model = string_at(root, &["model"])?;
        if model != "fast_pitch" {
            return Err(config_error(format!(
                "expected model \"fast_pitch\", got {model:?}"
            )));
        }
        let args = object_at(root, &["model_args"])?;
        for (field, expected) in [
            ("encoder_type", "fftransformer"),
            ("decoder_type", "fftransformer"),
        ] {
            let actual = string_at(args, &[field])?;
            if actual != expected {
                return Err(config_error(format!(
                    "unsupported {field} {actual:?}; the released checkpoint uses {expected}"
                )));
            }
        }
        if usize_at(args, &["num_speakers"])? > 1
            || bool_at(args, &["use_d_vector"])?
            || usize_at(args, &["d_vector_dim"])? != 0
        {
            return Err(config_error(
                "the released LJSpeech FastPitch checkpoint is single-speaker",
            ));
        }
        let num_chars = match args.get("num_chars").and_then(Value::as_u64) {
            Some(value) => usize::try_from(value)
                .map_err(|_| config_error("model_args.num_chars does not fit usize"))?,
            None => published_vocabulary_size(root)?,
        };
        let config = Self {
            num_chars,
            out_channels: usize_at(args, &["out_channels"])?,
            hidden_channels: usize_at(args, &["hidden_channels"])?,
            positional_encoding: bool_at(args, &["positional_encoding"])?,
            length_scale: number_at(args, &["length_scale"])?,
            encoder: FeedForwardTransformerConfig::from_value(
                object_at(args, &["encoder_params"])?,
                "encoder",
            )?,
            decoder: FeedForwardTransformerConfig::from_value(
                object_at(args, &["decoder_params"])?,
                "decoder",
            )?,
            duration_predictor_hidden_channels: usize_at(
                args,
                &["duration_predictor_hidden_channels"],
            )?,
            duration_predictor_kernel_size: usize_at(args, &["duration_predictor_kernel_size"])?,
            duration_predictor_dropout: number_at(args, &["duration_predictor_dropout_p"])?,
            pitch_predictor_hidden_channels: usize_at(args, &["pitch_predictor_hidden_channels"])?,
            pitch_predictor_kernel_size: usize_at(args, &["pitch_predictor_kernel_size"])?,
            pitch_predictor_dropout: number_at(args, &["pitch_predictor_dropout_p"])?,
            pitch_embedding_kernel_size: usize_at(args, &["pitch_embedding_kernel_size"])?,
            use_aligner: bool_at(args, &["use_aligner"])?,
            max_duration: usize_at(args, &["max_duration"])?,
            max_output_frames: 20_000,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), FastPitchError> {
        if self.num_chars == 0 || self.out_channels == 0 || self.hidden_channels == 0 {
            return Err(config_error(
                "character, output, and hidden dimensions must be positive",
            ));
        }
        if !self.hidden_channels.is_multiple_of(self.encoder.num_heads)
            || !self.hidden_channels.is_multiple_of(self.decoder.num_heads)
        {
            return Err(config_error(
                "hidden_channels must divide evenly across encoder and decoder heads",
            ));
        }
        if self.positional_encoding && !self.hidden_channels.is_multiple_of(2) {
            return Err(config_error(
                "hidden_channels must be even when positional encoding is enabled",
            ));
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(config_error("length_scale must be finite and positive"));
        }
        for (label, hidden, kernel, dropout) in [
            (
                "duration predictor",
                self.duration_predictor_hidden_channels,
                self.duration_predictor_kernel_size,
                self.duration_predictor_dropout,
            ),
            (
                "pitch predictor",
                self.pitch_predictor_hidden_channels,
                self.pitch_predictor_kernel_size,
                self.pitch_predictor_dropout,
            ),
        ] {
            if hidden == 0 || kernel == 0 || kernel.is_multiple_of(2) {
                return Err(config_error(format!(
                    "{label} hidden channels must be positive and kernel size must be positive and odd"
                )));
            }
            if !(0.0..1.0).contains(&dropout) {
                return Err(config_error(format!("{label} dropout must be in [0, 1)")));
            }
        }
        if self.pitch_embedding_kernel_size == 0
            || self.pitch_embedding_kernel_size.is_multiple_of(2)
        {
            return Err(config_error(
                "pitch embedding kernel size must be positive and odd",
            ));
        }
        if self.max_duration == 0 || self.max_output_frames == 0 {
            return Err(config_error(
                "maximum duration and output frame limit must be positive",
            ));
        }
        self.encoder.validate("encoder")?;
        self.decoder.validate("decoder")?;
        Ok(())
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<FastPitch<B>, FastPitchError> {
        self.validate()?;
        let emb = EmbeddingConfig::new(self.num_chars, self.hidden_channels).init(device);
        let encoder = FastPitchEncoder {
            encoder: FeedForwardTransformerBlock::init(self.hidden_channels, &self.encoder, device),
        };
        let decoder = FastPitchDecoderContainer {
            decoder: FastPitchDecoder {
                transformer_block: FeedForwardTransformerBlock::init(
                    self.hidden_channels,
                    &self.decoder,
                    device,
                ),
                postnet: Conv1dConfig::new(self.hidden_channels, self.out_channels, 1)
                    .with_padding(PaddingConfig1d::Valid)
                    .init(device),
            },
        };
        let duration_predictor = DurationPredictor::init(
            self.hidden_channels,
            self.duration_predictor_hidden_channels,
            self.duration_predictor_kernel_size,
            device,
        );
        let pitch_predictor = DurationPredictor::init(
            self.hidden_channels,
            self.pitch_predictor_hidden_channels,
            self.pitch_predictor_kernel_size,
            device,
        );
        let pitch_emb =
            Conv1dConfig::new(1, self.hidden_channels, self.pitch_embedding_kernel_size)
                .with_padding(PaddingConfig1d::Explicit(
                    self.pitch_embedding_kernel_size / 2,
                    self.pitch_embedding_kernel_size / 2,
                ))
                .init(device);
        let pos_encoder = self
            .positional_encoding
            .then(|| PositionalEncoding::init(self.hidden_channels, device));
        let aligner = self.use_aligner.then(|| {
            AlignmentNetwork::init(
                self.out_channels,
                self.hidden_channels,
                self.out_channels,
                device,
            )
        });
        Ok(FastPitch {
            emb,
            encoder,
            pos_encoder,
            decoder,
            duration_predictor,
            pitch_predictor,
            pitch_emb,
            aligner,
            length_scale: self.length_scale,
            num_chars: self.num_chars,
            out_channels: self.out_channels,
            max_duration: self.max_duration,
            max_output_frames: self.max_output_frames,
        })
    }
}

#[derive(Module, Debug)]
pub struct PytorchMultiHeadAttention<B: Backend> {
    /// PyTorch stores the combined QKV matrix as `[output, input]`. This is a
    /// raw parameter rather than a Burn `Linear`, so inference transposes it
    /// explicitly.
    pub in_proj_weight: Param<Tensor<B, 2>>,
    pub in_proj_bias: Param<Tensor<B, 1>>,
    pub out_proj: burn::nn::Linear<B>,
    num_heads: usize,
    channels_per_head: usize,
}

impl<B: Backend> PytorchMultiHeadAttention<B> {
    fn init(channels: usize, num_heads: usize, device: &B::Device) -> Self {
        Self {
            in_proj_weight: Initializer::XavierUniform { gain: 1.0 }.init_with(
                [channels * 3, channels],
                Some(channels),
                Some(channels * 3),
                device,
            ),
            in_proj_bias: Initializer::Zeros.init([channels * 3], device),
            out_proj: burn::nn::LinearConfig::new(channels, channels).init(device),
            num_heads,
            channels_per_head: channels / num_heads,
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, tokens] = input.dims();
        let input = input.swap_dims(1, 2);
        let weight = self
            .in_proj_weight
            .val()
            .swap_dims(0, 1)
            .reshape([1, channels, channels * 3])
            .expand([batch, channels, channels * 3]);
        let projected =
            input.matmul(weight) + self.in_proj_bias.val().reshape([1, 1, channels * 3]);
        let query = projected
            .clone()
            .slice([0..batch, 0..tokens, 0..channels])
            .reshape([batch, tokens, self.num_heads, self.channels_per_head])
            .swap_dims(1, 2);
        let key = projected
            .clone()
            .slice([0..batch, 0..tokens, channels..channels * 2])
            .reshape([batch, tokens, self.num_heads, self.channels_per_head])
            .swap_dims(1, 2);
        let value = projected
            .slice([0..batch, 0..tokens, channels * 2..channels * 3])
            .reshape([batch, tokens, self.num_heads, self.channels_per_head])
            .swap_dims(1, 2);
        let scores = query.matmul(key.swap_dims(2, 3)) / (self.channels_per_head as f64).sqrt();
        let attended = softmax(scores, 3)
            .matmul(value)
            .swap_dims(1, 2)
            .reshape([batch, tokens, channels]);
        self.out_proj.forward(attended).swap_dims(1, 2)
    }
}

#[derive(Module, Debug)]
pub struct TransformerLayerNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 1>>,
    pub beta: Param<Tensor<B, 1>>,
}

impl<B: Backend> TransformerLayerNorm<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            gamma: Initializer::Ones.init([channels], device),
            beta: Initializer::Zeros.init([channels], device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mean = input.clone().mean_dim(1);
        let variance = (input.clone() - mean.clone()).square().mean_dim(1);
        (input - mean) / (variance + LAYER_NORM_EPSILON).sqrt()
            * self.gamma.val().reshape([1, self.gamma.dims()[0], 1])
            + self.beta.val().reshape([1, self.beta.dims()[0], 1])
    }
}

#[derive(Module, Debug)]
pub struct FeedForwardTransformerLayer<B: Backend> {
    pub self_attn: PytorchMultiHeadAttention<B>,
    pub conv1: Conv1d<B>,
    pub conv2: Conv1d<B>,
    pub norm1: TransformerLayerNorm<B>,
    pub norm2: TransformerLayerNorm<B>,
}

impl<B: Backend> FeedForwardTransformerLayer<B> {
    fn init(channels: usize, config: &FeedForwardTransformerConfig, device: &B::Device) -> Self {
        Self {
            self_attn: PytorchMultiHeadAttention::init(channels, config.num_heads, device),
            conv1: Conv1dConfig::new(channels, config.hidden_channels_ffn, 3)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .init(device),
            conv2: Conv1dConfig::new(config.hidden_channels_ffn, channels, 3)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .init(device),
            norm1: TransformerLayerNorm::init(channels, device),
            norm2: TransformerLayerNorm::init(channels, device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let attention = self.self_attn.forward(input.clone());
        // Coqui v0.6.1's reference layer adds the attention result once before
        // adding it again inside norm1. Reproduce that released graph exactly.
        let output = self.norm1.forward(input + attention.clone() * 2.0);
        let feed_forward = self.conv2.forward(relu(self.conv1.forward(output.clone())));
        self.norm2.forward(output + feed_forward)
    }
}

#[derive(Module, Debug)]
pub struct FeedForwardTransformerBlock<B: Backend> {
    pub fft_layers: Vec<FeedForwardTransformerLayer<B>>,
}

impl<B: Backend> FeedForwardTransformerBlock<B> {
    pub(crate) fn init(
        channels: usize,
        config: &FeedForwardTransformerConfig,
        device: &B::Device,
    ) -> Self {
        Self {
            fft_layers: (0..config.num_layers)
                .map(|_| FeedForwardTransformerLayer::init(channels, config, device))
                .collect(),
        }
    }

    pub(crate) fn forward(&self, mut input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in &self.fft_layers {
            input = layer.forward(input) * mask.clone();
        }
        input
    }
}

#[derive(Module, Debug)]
pub struct FastPitchEncoder<B: Backend> {
    pub encoder: FeedForwardTransformerBlock<B>,
}

#[derive(Module, Debug)]
pub struct FastPitchDecoder<B: Backend> {
    pub transformer_block: FeedForwardTransformerBlock<B>,
    pub postnet: Conv1d<B>,
}

#[derive(Module, Debug)]
pub struct FastPitchDecoderContainer<B: Backend> {
    pub decoder: FastPitchDecoder<B>,
}

#[derive(Debug)]
pub struct FastPitchOutput<B: Backend> {
    pub mel: Tensor<B, 3>,
    pub durations: Tensor<B, 2>,
    pub pitch: Tensor<B, 3>,
}

#[derive(Debug)]
pub struct FastPitchControls<B: Backend> {
    pub length_scale: f64,
    pub pitch_scale: f64,
    pub pitch_shift: f64,
    pub durations: Option<Tensor<B, 2>>,
    pub pitch: Option<Tensor<B, 3>>,
}

impl<B: Backend> Default for FastPitchControls<B> {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            pitch_scale: 1.0,
            pitch_shift: 0.0,
            durations: None,
            pitch: None,
        }
    }
}

#[derive(Module, Debug)]
pub struct FastPitch<B: Backend> {
    pub emb: Embedding<B>,
    pub encoder: FastPitchEncoder<B>,
    pub pos_encoder: Option<PositionalEncoding<B>>,
    pub decoder: FastPitchDecoderContainer<B>,
    pub duration_predictor: DurationPredictor<B>,
    pub pitch_predictor: DurationPredictor<B>,
    pub pitch_emb: Conv1d<B>,
    pub aligner: Option<AlignmentNetwork<B>>,
    length_scale: f64,
    num_chars: usize,
    out_channels: usize,
    max_duration: usize,
    max_output_frames: usize,
}

impl<B: Backend> FastPitch<B> {
    pub fn load_checkpoint(mut self, path: impl AsRef<Path>) -> Result<Self, FastPitchError> {
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut self,
            path.as_ref(),
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                predicate: Some(checkpoint_tensor),
                key_remappings: vec![
                    (r"(\.norm_[12])\.weight$".into(), "$1.gamma".into()),
                    (r"(\.norm_[12])\.bias$".into(), "$1.beta".into()),
                    (r"(\.norm[12])\.weight$".into(), "$1.gamma".into()),
                    (r"(\.norm[12])\.bias$".into(), "$1.beta".into()),
                ],
                skip_enum_variants: true,
                ..Default::default()
            },
        )
        .map_err(|error| FastPitchError::Checkpoint(format!("{error:#}")))?;
        let unexpected_unused = result
            .unused
            .iter()
            .filter(|path| !path.ends_with(".num_batches_tracked"))
            .cloned()
            .collect::<Vec<_>>();
        if !result.missing.is_empty() || !result.errors.is_empty() || !unexpected_unused.is_empty()
        {
            return Err(FastPitchError::Checkpoint(format!(
                "checkpoint does not exactly match the Burn model: {} missing, {} load errors, unexpected tensors: {}",
                result.missing.len(),
                result.errors.len(),
                unexpected_unused.join(", ")
            )));
        }
        if let Some(positional) = &self.pos_encoder {
            let device = positional.pe.val().device();
            self.pos_encoder = Some(PositionalEncoding::init(positional.channels, &device));
        }
        Ok(self)
    }

    pub fn inference(
        &self,
        token_ids: Tensor<B, 2, Int>,
    ) -> Result<FastPitchOutput<B>, FastPitchError> {
        self.inference_with_controls(
            token_ids,
            FastPitchControls {
                length_scale: self.length_scale,
                ..Default::default()
            },
            false,
            None,
        )
    }

    pub(crate) fn inference_projected_with_controls(
        &self,
        token_ids: Tensor<B, 2, Int>,
        controls: FastPitchControls<B>,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<FastPitchOutput<B>, FastPitchError> {
        self.inference_with_controls(token_ids, controls, true, profiler)
    }

    fn inference_with_controls(
        &self,
        token_ids: Tensor<B, 2, Int>,
        controls: FastPitchControls<B>,
        ids_validated_on_host: bool,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<FastPitchOutput<B>, FastPitchError> {
        let mut profiler = profiler;
        if !controls.length_scale.is_finite() || controls.length_scale <= 0.0 {
            return Err(input_error("length_scale must be finite and positive"));
        }
        if !controls.pitch_scale.is_finite() || controls.pitch_scale <= 0.0 {
            return Err(input_error("pitch_scale must be finite and positive"));
        }
        if !controls.pitch_shift.is_finite() {
            return Err(input_error("pitch_shift must be finite"));
        }
        let [batch, tokens] = token_ids.dims();
        if batch == 0 || tokens == 0 {
            return Err(input_error(
                "token_ids must have non-empty [batch, tokens] dimensions",
            ));
        }
        if !ids_validated_on_host {
            let highest_id = token_ids.clone().max().into_scalar().elem::<i64>();
            if highest_id < 0 || highest_id as usize >= self.num_chars {
                return Err(input_error(format!(
                    "token ID {highest_id} is outside vocabulary 0..{}",
                    self.num_chars
                )));
            }
        }
        let device = token_ids.device();
        let mask = Tensor::<B, 3>::ones([batch, 1, tokens], &device);
        let started = Instant::now();
        let embedded = self.emb.forward(token_ids).swap_dims(1, 2);
        let encoded = self.encoder.encoder.forward(embedded, mask.clone());
        finish_backend_stage::<B>(
            &mut profiler,
            &device,
            SynthesisStage::TextEncoder,
            started,
            [SynthesisDimension::new("tokens", tokens)],
        )
        .map_err(|error| input_error(error.to_string()))?;

        let started = Instant::now();
        let durations = match controls.durations {
            Some(durations) => {
                if durations.dims() != [batch, tokens] {
                    return Err(input_error(format!(
                        "explicit durations have shape {:?}; expected [{batch}, {tokens}]",
                        durations.dims()
                    )));
                }
                durations
            }
            None => {
                let duration_log = self
                    .duration_predictor
                    .forward(encoded.clone(), mask.clone());
                ((duration_log.exp() - 1.0) * mask.clone() * controls.length_scale)
                    .clamp(1.0, self.max_duration as f64)
                    .round()
                    .reshape([batch, tokens])
            }
        };
        finish_backend_stage::<B>(
            &mut profiler,
            &device,
            SynthesisStage::DurationPrediction,
            started,
            [SynthesisDimension::new("tokens", tokens)],
        )
        .map_err(|error| input_error(error.to_string()))?;

        let started = Instant::now();
        let predicted_pitch = self.pitch_predictor.forward(encoded.clone(), mask.clone());
        let pitch = (match controls.pitch {
            Some(pitch) => {
                if pitch.dims() != [batch, 1, tokens] {
                    return Err(input_error(format!(
                        "explicit pitch has shape {:?}; expected [{batch}, 1, {tokens}]",
                        pitch.dims()
                    )));
                }
                pitch
            }
            None => predicted_pitch,
        } * controls.pitch_scale
            + controls.pitch_shift)
            * mask.clone();
        let pitch_conditioned = encoded + self.pitch_emb.forward(pitch.clone());

        let (mut expanded, output_mask) =
            expand_by_durations(pitch_conditioned, durations.clone(), self.max_output_frames)
                .map_err(|error| input_error(error.to_string()))?;
        let output_frames = expanded.dims()[2];
        if output_frames > MAX_POSITIONAL_FRAMES {
            return Err(input_error(format!(
                "duration controls requested {output_frames} frames, exceeding positional limit {MAX_POSITIONAL_FRAMES}"
            )));
        }
        finish_backend_stage::<B>(
            &mut profiler,
            &device,
            SynthesisStage::DurationExpansion,
            started,
            [
                SynthesisDimension::new("tokens", tokens),
                SynthesisDimension::new("frames", output_frames),
            ],
        )
        .map_err(|error| input_error(error.to_string()))?;
        if let Some(positional) = &self.pos_encoder {
            expanded = positional
                .forward(expanded, output_mask.clone())
                .map_err(|error| input_error(error.to_string()))?;
        }
        let started = Instant::now();
        let decoded = self
            .decoder
            .decoder
            .transformer_block
            .forward(expanded, output_mask.clone());
        let mel = self.decoder.decoder.postnet.forward(decoded) * output_mask;
        let mel = mel.swap_dims(1, 2);
        finish_backend_stage::<B>(
            &mut profiler,
            &device,
            SynthesisStage::AcousticDecoder,
            started,
            [
                SynthesisDimension::new("frames", output_frames),
                SynthesisDimension::new("mel_bins", self.out_channels),
            ],
        )
        .map_err(|error| input_error(error.to_string()))?;
        Ok(FastPitchOutput {
            mel,
            durations,
            pitch,
        })
    }
}

fn checkpoint_tensor(path: &str, _container: &str) -> bool {
    !path.ends_with(".num_batches_tracked")
}

fn object_at<'a>(root: &'a Value, path: &[&str]) -> Result<&'a Value, FastPitchError> {
    let value = path.iter().try_fold(root, |value, key| {
        value
            .get(*key)
            .ok_or_else(|| config_error(format!("missing {}", path.join("."))))
    })?;
    value
        .as_object()
        .map(|_| value)
        .ok_or_else(|| config_error(format!("{} must be an object", path.join("."))))
}

fn string_at<'a>(root: &'a Value, path: &[&str]) -> Result<&'a str, FastPitchError> {
    let value = path.iter().try_fold(root, |value, key| {
        value
            .get(*key)
            .ok_or_else(|| config_error(format!("missing {}", path.join("."))))
    })?;
    value
        .as_str()
        .ok_or_else(|| config_error(format!("{} must be a string", path.join("."))))
}

fn bool_at(root: &Value, path: &[&str]) -> Result<bool, FastPitchError> {
    let value = path.iter().try_fold(root, |value, key| {
        value
            .get(*key)
            .ok_or_else(|| config_error(format!("missing {}", path.join("."))))
    })?;
    value
        .as_bool()
        .ok_or_else(|| config_error(format!("{} must be a boolean", path.join("."))))
}

fn usize_at(root: &Value, path: &[&str]) -> Result<usize, FastPitchError> {
    let value = path.iter().try_fold(root, |value, key| {
        value
            .get(*key)
            .ok_or_else(|| config_error(format!("missing {}", path.join("."))))
    })?;
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| config_error(format!("{} must be an unsigned integer", path.join("."))))
}

fn number_at(root: &Value, path: &[&str]) -> Result<f64, FastPitchError> {
    let value = path.iter().try_fold(root, |value, key| {
        value
            .get(*key)
            .ok_or_else(|| config_error(format!("missing {}", path.join("."))))
    })?;
    value
        .as_f64()
        .ok_or_else(|| config_error(format!("{} must be numeric", path.join("."))))
}

fn published_vocabulary_size(root: &Value) -> Result<usize, FastPitchError> {
    let characters = object_at(root, &["characters"])?;
    ["pad", "eos", "bos", "phonemes", "punctuations"]
        .iter()
        .try_fold(0usize, |total, field| {
            let value = characters
                .get(*field)
                .and_then(Value::as_str)
                .ok_or_else(|| config_error(format!("characters.{field} must be a string")))?;
            Ok(total + value.chars().count())
        })
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::TensorData;

    use super::*;

    type TestBackend = NdArray<f32>;

    fn tiny_config() -> FastPitchConfig {
        FastPitchConfig {
            num_chars: 8,
            out_channels: 3,
            hidden_channels: 4,
            positional_encoding: true,
            length_scale: 1.0,
            encoder: FeedForwardTransformerConfig {
                hidden_channels_ffn: 8,
                num_heads: 1,
                num_layers: 1,
                dropout: 0.1,
            },
            decoder: FeedForwardTransformerConfig {
                hidden_channels_ffn: 8,
                num_heads: 1,
                num_layers: 1,
                dropout: 0.1,
            },
            duration_predictor_hidden_channels: 4,
            duration_predictor_kernel_size: 3,
            duration_predictor_dropout: 0.1,
            pitch_predictor_hidden_channels: 4,
            pitch_predictor_kernel_size: 3,
            pitch_predictor_dropout: 0.1,
            pitch_embedding_kernel_size: 3,
            use_aligner: false,
            max_duration: 10,
            max_output_frames: 64,
        }
    }

    #[test]
    fn parses_released_ljspeech_shape() {
        let root = serde_json::json!({
            "model": "fast_pitch",
            "model_args": {
                "num_chars": 130,
                "out_channels": 80,
                "hidden_channels": 384,
                "num_speakers": 0,
                "duration_predictor_hidden_channels": 256,
                "duration_predictor_kernel_size": 3,
                "duration_predictor_dropout_p": 0.1,
                "pitch_predictor_hidden_channels": 256,
                "pitch_predictor_kernel_size": 3,
                "pitch_predictor_dropout_p": 0.1,
                "pitch_embedding_kernel_size": 3,
                "positional_encoding": true,
                "length_scale": 1,
                "encoder_type": "fftransformer",
                "encoder_params": {
                    "hidden_channels_ffn": 1024,
                    "num_heads": 1,
                    "num_layers": 6,
                    "dropout_p": 0.1
                },
                "decoder_type": "fftransformer",
                "decoder_params": {
                    "hidden_channels_ffn": 1024,
                    "num_heads": 1,
                    "num_layers": 6,
                    "dropout_p": 0.1
                },
                "use_d_vector": false,
                "d_vector_dim": 0,
                "max_duration": 75,
                "use_aligner": true
            }
        });
        let config = FastPitchConfig::from_json_value(&root).expect("FastPitch config");
        assert_eq!(config, FastPitchConfig::ljspeech());
    }

    #[test]
    fn explicit_duration_and_pitch_controls_determine_shapes() {
        let device = NdArrayDevice::Cpu;
        let model = tiny_config().init::<TestBackend>(&device).expect("model");
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3], [1, 3]),
            &device,
        );
        let durations = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 1.0], [1, 3]),
            &device,
        );
        let pitch = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![100.0_f32, 120.0, 110.0], [1, 1, 3]),
            &device,
        );
        let output = model
            .inference_projected_with_controls(
                tokens,
                FastPitchControls {
                    durations: Some(durations),
                    pitch: Some(pitch),
                    ..Default::default()
                },
                None,
            )
            .expect("inference");
        assert_eq!(output.durations.dims(), [1, 3]);
        assert_eq!(output.pitch.dims(), [1, 1, 3]);
        assert_eq!(output.mel.dims(), [1, 4, 3]);
    }

    fn published_model() -> Option<(FastPitch<TestBackend>, NdArrayDevice)> {
        let model_path = std::env::var_os("TONGUES_TEST_COQUI_FASTPITCH_MODEL")?;
        let config_path = std::env::var_os("TONGUES_TEST_COQUI_FASTPITCH_CONFIG")
            .expect("TONGUES_TEST_COQUI_FASTPITCH_CONFIG must accompany the model");
        let source = std::fs::read_to_string(config_path).expect("config");
        let value: Value = serde_json::from_str(&source).expect("JSON config");
        let config = FastPitchConfig::from_json_value(&value).expect("model config");
        let device = NdArrayDevice::Cpu;
        let model = config
            .init::<TestBackend>(&device)
            .expect("model")
            .load_checkpoint(model_path)
            .expect("checkpoint");
        Some((model, device))
    }

    #[test]
    fn loads_and_runs_the_published_checkpoint_when_available() {
        let Some((model, device)) = published_model() else {
            return;
        };
        let input = Tensor::<TestBackend, 2, Int>::from_ints(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]],
            &device,
        );
        let output = model.inference(input).expect("inference");
        assert_eq!(output.mel.dims()[0], 1);
        assert_eq!(output.mel.dims()[2], 80);
        assert_eq!(output.durations.dims(), [1, 13]);
        assert_eq!(output.pitch.dims(), [1, 1, 13]);
    }

    #[test]
    #[ignore = "requires pinned Coqui FastPitch model artifacts; run scripts/speech-conformance.sh"]
    fn published_checkpoint_stage_parity() {
        let (model, device) =
            published_model().expect("TONGUES_TEST_COQUI_FASTPITCH_CONFIG and MODEL are required");
        let ids = [
            14, 43, 77, 15, 63, 33, 129, 13, 3, 63, 21, 129, 77, 50, 20, 21, 63, 6, 129, 43, 15,
            129, 30, 48, 129, 20, 10, 6, 49, 129, 21, 77, 10, 27, 129, 24, 3, 63, 13, 129, 30, 48,
            129, 12, 50, 21, 48, 13, 129, 4, 63, 55, 28, 15, 129, 21, 48, 129, 20, 63, 33, 125,
        ];
        let input = Tensor::<TestBackend, 1, Int>::from_ints(ids, &device).reshape([1, 62]);
        let output = model.inference(input).expect("inference");
        assert_eq!(output.mel.dims(), [1, 323, 80]);
        assert_eq!(
            output
                .durations
                .into_data()
                .to_vec::<f32>()
                .expect("f32 durations"),
            vec![
                5.0, 8.0, 4.0, 4.0, 5.0, 6.0, 6.0, 2.0, 6.0, 10.0, 2.0, 7.0, 4.0, 6.0, 10.0, 5.0,
                5.0, 1.0, 5.0, 2.0, 4.0, 1.0, 1.0, 2.0, 14.0, 3.0, 11.0, 3.0, 4.0, 7.0, 8.0, 5.0,
                10.0, 7.0, 6.0, 2.0, 3.0, 4.0, 2.0, 2.0, 1.0, 2.0, 6.0, 8.0, 6.0, 2.0, 4.0, 5.0,
                3.0, 2.0, 9.0, 5.0, 11.0, 7.0, 3.0, 4.0, 3.0, 14.0, 4.0, 8.0, 12.0, 2.0,
            ]
        );
        let pitch = output.pitch.into_data().to_vec::<f32>().expect("f32 pitch");
        for (token, expected) in [
            (0, 0.75493604),
            (1, 1.2340915),
            (31, -0.31366393),
            (61, 0.06257132),
        ] {
            assert!(
                (pitch[token] - expected).abs() <= 2e-4,
                "pitch parity mismatch at token {token}: actual={}, expected={expected}",
                pitch[token]
            );
        }
        let mel = output.mel.into_data().to_vec::<f32>().expect("f32 mel");
        for (frame, bin, expected) in [
            (0, 0, -6.321159),
            (0, 79, -9.630943),
            (1, 7, -3.1317515),
            (5, 23, -4.5700483),
            (20, 40, -3.2360125),
            (161, 40, -3.9115438),
            (322, 79, -9.365692),
        ] {
            let actual = mel[frame * 80 + bin];
            assert!(
                (actual - expected).abs() <= 3e-4,
                "mel parity mismatch at frame {frame}, bin {bin}: actual={actual}, expected={expected}"
            );
        }
    }
}
