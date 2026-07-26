//! Native Burn implementation of the released LJSpeech SpeedySpeech model.
//!
//! The release named `tts_models/en/ljspeech/speedy-speech` is configured as a
//! residual-convolution SpeedySpeech model through the reference `ForwardTTS`
//! container. This module therefore follows the published checkpoint rather
//! than the older, subsequently removed standalone container:
//!
//! - `emb`
//! - `encoder.encoder`
//! - `pos_encoder`
//! - `decoder.decoder`
//! - `duration_predictor`
//! - `aligner` (training-only, retained so the complete checkpoint loads)
//!
//! Input is already-projected model token IDs. Text cleaning, phonemization,
//! IPA handling, and vocabulary projection belong outside this module.
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
use burn::nn::{BatchNorm, BatchNormConfig, Embedding, EmbeddingConfig, PaddingConfig1d};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::profiling::finish_backend_stage;
use crate::{SynthesisDimension, SynthesisProfiler, SynthesisStage};

const BATCH_NORM_EPSILON: f64 = 1e-5;
const DURATION_NORM_EPSILON: f64 = 1e-4;
const POSITIONAL_ENCODING_LIMIT: usize = 5_000;

/// Errors detected while parsing, constructing, loading, or running SpeedySpeech.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeedySpeechError {
    InvalidConfig(String),
    InvalidInput(String),
    Checkpoint(String),
}

impl fmt::Display for SpeedySpeechError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid SpeedySpeech config: {message}")
            }
            Self::InvalidInput(message) => {
                write!(formatter, "invalid SpeedySpeech input: {message}")
            }
            Self::Checkpoint(message) => {
                write!(
                    formatter,
                    "unable to load SpeedySpeech checkpoint: {message}"
                )
            }
        }
    }
}

impl std::error::Error for SpeedySpeechError {}

fn config_error(message: impl Into<String>) -> SpeedySpeechError {
    SpeedySpeechError::InvalidConfig(message.into())
}

fn input_error(message: impl Into<String>) -> SpeedySpeechError {
    SpeedySpeechError::InvalidInput(message.into())
}

/// Parameters for one residual-convolution stack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidualConvConfig {
    pub kernel_size: usize,
    pub dilations: Vec<usize>,
    pub num_conv_blocks: usize,
    pub num_res_blocks: usize,
}

impl ResidualConvConfig {
    fn validate(&self, label: &str) -> Result<(), SpeedySpeechError> {
        if self.kernel_size == 0 {
            return Err(config_error(format!(
                "{label} kernel_size must be positive"
            )));
        }
        if self.num_conv_blocks == 0 {
            return Err(config_error(format!(
                "{label} num_conv_blocks must be positive"
            )));
        }
        if self.num_res_blocks == 0 {
            return Err(config_error(format!(
                "{label} num_res_blocks must be positive"
            )));
        }
        if self.dilations.len() != self.num_res_blocks {
            return Err(config_error(format!(
                "{label} has {} dilations but num_res_blocks is {}",
                self.dilations.len(),
                self.num_res_blocks
            )));
        }
        if self.dilations.contains(&0) {
            return Err(config_error(format!(
                "{label} dilations must all be positive"
            )));
        }
        Ok(())
    }
}

/// Model parameters required by the released residual-convolution graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeedySpeechConfig {
    pub num_chars: usize,
    pub out_channels: usize,
    pub hidden_channels: usize,
    pub positional_encoding: bool,
    pub length_scale: f64,
    pub encoder: ResidualConvConfig,
    pub decoder: ResidualConvConfig,
    pub duration_predictor_hidden_channels: usize,
    pub duration_predictor_kernel_size: usize,
    pub duration_predictor_dropout: f64,
    pub use_aligner: bool,
    pub max_duration: usize,
    /// Allocation guard applied after duration prediction.
    pub max_output_frames: usize,
}

impl SpeedySpeechConfig {
    /// Published `tts_models/en/ljspeech/speedy-speech` architecture.
    pub fn ljspeech() -> Self {
        Self {
            num_chars: 130,
            out_channels: 80,
            hidden_channels: 128,
            positional_encoding: true,
            length_scale: 1.0,
            encoder: ResidualConvConfig {
                kernel_size: 4,
                dilations: vec![1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1],
                num_conv_blocks: 2,
                num_res_blocks: 13,
            },
            decoder: ResidualConvConfig {
                kernel_size: 4,
                dilations: vec![1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1],
                num_conv_blocks: 2,
                num_res_blocks: 17,
            },
            duration_predictor_hidden_channels: 256,
            duration_predictor_kernel_size: 3,
            duration_predictor_dropout: 0.1,
            use_aligner: true,
            max_duration: 75,
            max_output_frames: 20_000,
        }
    }

    /// Parse the model-specific values from a compatible `config.json` value.
    ///
    /// The caller may use JSON5 before invoking this method when the source
    /// file contains comments.
    pub fn from_json_value(root: &Value) -> Result<Self, SpeedySpeechError> {
        let model = string_at(root, &["model"])?;
        if model != "speedy_speech" {
            return Err(config_error(format!(
                "expected model \"speedy_speech\", got {model:?}"
            )));
        }

        let args = object_at(root, &["model_args"])?;
        let encoder_params = object_at(args, &["encoder_params"])?;
        let decoder_params = object_at(args, &["decoder_params"])?;

        let encoder_type = string_at(args, &["encoder_type"])?;
        if encoder_type != "residual_conv_bn" {
            return Err(config_error(format!(
                "unsupported encoder_type {encoder_type:?}; only residual_conv_bn matches the released model"
            )));
        }
        let decoder_type = string_at(args, &["decoder_type"])?;
        if decoder_type != "residual_conv_bn" {
            return Err(config_error(format!(
                "unsupported decoder_type {decoder_type:?}; only residual_conv_bn matches the released model"
            )));
        }
        if bool_at(args, &["use_pitch"])? {
            return Err(config_error(
                "this SpeedySpeech implementation does not accept FastPitch checkpoints",
            ));
        }
        if usize_at(args, &["num_speakers"])? > 1
            || bool_at(args, &["use_d_vector"])?
            || usize_at(args, &["d_vector_dim"])? != 0
        {
            return Err(config_error(
                "the released LJSpeech model is single-speaker; speaker-conditioned ForwardTTS is not this checkpoint",
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
            encoder: residual_config(encoder_params, "encoder_params")?,
            decoder: residual_config(decoder_params, "decoder_params")?,
            duration_predictor_hidden_channels: usize_at(
                args,
                &["duration_predictor_hidden_channels"],
            )?,
            duration_predictor_kernel_size: usize_at(args, &["duration_predictor_kernel_size"])?,
            duration_predictor_dropout: number_at(args, &["duration_predictor_dropout_p"])?,
            use_aligner: bool_at(args, &["use_aligner"])?,
            max_duration: usize_at(args, &["max_duration"])?,
            max_output_frames: 20_000,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), SpeedySpeechError> {
        if self.num_chars == 0 {
            return Err(config_error("num_chars must be positive"));
        }
        if self.out_channels == 0 {
            return Err(config_error("out_channels must be positive"));
        }
        if self.hidden_channels == 0 {
            return Err(config_error("hidden_channels must be positive"));
        }
        if self.positional_encoding && !self.hidden_channels.is_multiple_of(2) {
            return Err(config_error(
                "hidden_channels must be even when positional encoding is enabled",
            ));
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(config_error("length_scale must be finite and positive"));
        }
        if self.duration_predictor_hidden_channels == 0 {
            return Err(config_error(
                "duration_predictor_hidden_channels must be positive",
            ));
        }
        if self.duration_predictor_kernel_size == 0
            || self.duration_predictor_kernel_size.is_multiple_of(2)
        {
            return Err(config_error(
                "duration_predictor_kernel_size must be positive and odd",
            ));
        }
        if !(0.0..1.0).contains(&self.duration_predictor_dropout) {
            return Err(config_error("duration_predictor_dropout must be in [0, 1)"));
        }
        if self.max_duration == 0 {
            return Err(config_error("max_duration must be positive"));
        }
        if self.max_output_frames == 0 {
            return Err(config_error("max_output_frames must be positive"));
        }
        self.encoder.validate("encoder")?;
        self.decoder.validate("decoder")?;
        Ok(())
    }

    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<SpeedySpeech<B>, SpeedySpeechError> {
        self.validate()?;

        let emb = EmbeddingConfig::new(self.num_chars, self.hidden_channels).init(device);
        let encoder = Encoder {
            encoder: ResidualConvEncoder::init(self.hidden_channels, &self.encoder, device),
        };
        let pos_encoder = self
            .positional_encoding
            .then(|| PositionalEncoding::init(self.hidden_channels, device));
        let decoder = Decoder {
            decoder: ResidualConvDecoder::init(
                self.hidden_channels,
                self.out_channels,
                &self.decoder,
                device,
            ),
        };
        let duration_predictor = DurationPredictor::init(
            self.hidden_channels,
            self.duration_predictor_hidden_channels,
            self.duration_predictor_kernel_size,
            device,
        );
        let aligner = self.use_aligner.then(|| {
            AlignmentNetwork::init(
                self.out_channels,
                self.hidden_channels,
                self.out_channels,
                device,
            )
        });

        Ok(SpeedySpeech {
            emb,
            encoder,
            pos_encoder,
            decoder,
            duration_predictor,
            aligner,
            length_scale: self.length_scale,
            num_chars: self.num_chars,
            out_channels: self.out_channels,
            minimum_input_tokens: minimum_stack_input_size(&self.encoder),
            minimum_output_frames: minimum_stack_input_size(&self.decoder),
            max_output_frames: self.max_output_frames,
        })
    }
}

fn minimum_stack_input_size(config: &ResidualConvConfig) -> usize {
    config
        .dilations
        .iter()
        .copied()
        .max()
        .map(|dilation| dilation * (config.kernel_size - 1) + 1)
        .unwrap_or(1)
}

fn residual_config(value: &Value, label: &str) -> Result<ResidualConvConfig, SpeedySpeechError> {
    let dilations = value
        .get("dilations")
        .and_then(Value::as_array)
        .ok_or_else(|| config_error(format!("{label}.dilations must be an array")))?
        .iter()
        .enumerate()
        .map(|(index, item)| {
            item.as_u64()
                .and_then(|item| usize::try_from(item).ok())
                .ok_or_else(|| {
                    config_error(format!(
                        "{label}.dilations[{index}] must be an unsigned integer"
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ResidualConvConfig {
        kernel_size: usize_at(value, &["kernel_size"])?,
        dilations,
        num_conv_blocks: usize_at(value, &["num_conv_blocks"])?,
        num_res_blocks: usize_at(value, &["num_res_blocks"])?,
    })
}

fn object_at<'a>(root: &'a Value, path: &[&str]) -> Result<&'a Value, SpeedySpeechError> {
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

fn string_at<'a>(root: &'a Value, path: &[&str]) -> Result<&'a str, SpeedySpeechError> {
    let value = path.iter().try_fold(root, |value, key| {
        value
            .get(*key)
            .ok_or_else(|| config_error(format!("missing {}", path.join("."))))
    })?;
    value
        .as_str()
        .ok_or_else(|| config_error(format!("{} must be a string", path.join("."))))
}

fn bool_at(root: &Value, path: &[&str]) -> Result<bool, SpeedySpeechError> {
    let value = path.iter().try_fold(root, |value, key| {
        value
            .get(*key)
            .ok_or_else(|| config_error(format!("missing {}", path.join("."))))
    })?;
    value
        .as_bool()
        .ok_or_else(|| config_error(format!("{} must be a boolean", path.join("."))))
}

fn usize_at(root: &Value, path: &[&str]) -> Result<usize, SpeedySpeechError> {
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

fn number_at(root: &Value, path: &[&str]) -> Result<f64, SpeedySpeechError> {
    let value = path.iter().try_fold(root, |value, key| {
        value
            .get(*key)
            .ok_or_else(|| config_error(format!("missing {}", path.join("."))))
    })?;
    value
        .as_f64()
        .ok_or_else(|| config_error(format!("{} must be numeric", path.join("."))))
}

fn published_vocabulary_size(root: &Value) -> Result<usize, SpeedySpeechError> {
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

fn conv1d<B: Backend>(
    channels_in: usize,
    channels_out: usize,
    kernel_size: usize,
    dilation: usize,
    padding: PaddingConfig1d,
    device: &B::Device,
) -> Conv1d<B> {
    Conv1dConfig::new(channels_in, channels_out, kernel_size)
        .with_dilation(dilation)
        .with_padding(padding)
        .init(device)
}

/// Checkpoint-compatible `Conv1d -> uneven zero pad -> ReLU -> BatchNorm1d`.
#[derive(Module, Debug)]
pub struct Conv1dBn<B: Backend> {
    pub conv1d: Conv1d<B>,
    pub norm: BatchNorm<B>,
    pad_left: usize,
    pad_right: usize,
}

impl<B: Backend> Conv1dBn<B> {
    pub(crate) fn init(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        dilation: usize,
        device: &B::Device,
    ) -> Self {
        let total_padding = dilation * (kernel_size - 1);
        let pad_left = total_padding / 2;
        let pad_right = total_padding - pad_left;
        Self {
            conv1d: conv1d(
                channels_in,
                channels_out,
                kernel_size,
                dilation,
                PaddingConfig1d::Valid,
                device,
            ),
            norm: BatchNormConfig::new(channels_out)
                .with_epsilon(BATCH_NORM_EPSILON)
                .init(device),
            pad_left,
            pad_right,
        }
    }

    pub(crate) fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let output = relu(self.conv1d.forward(input).pad(
            [(self.pad_left, self.pad_right)],
            burn::tensor::ops::PadMode::Constant(0.0),
        ));
        self.norm.forward(output)
    }
}

#[derive(Module, Debug)]
pub struct Conv1dBnBlock<B: Backend> {
    pub conv_bn_blocks: Vec<Conv1dBn<B>>,
}

impl<B: Backend> Conv1dBnBlock<B> {
    fn init(
        channels_in: usize,
        channels_out: usize,
        hidden_channels: usize,
        kernel_size: usize,
        dilation: usize,
        count: usize,
        device: &B::Device,
    ) -> Self {
        let conv_bn_blocks = (0..count)
            .map(|index| {
                Conv1dBn::init(
                    if index == 0 {
                        channels_in
                    } else {
                        hidden_channels
                    },
                    if index + 1 == count {
                        channels_out
                    } else {
                        hidden_channels
                    },
                    kernel_size,
                    dilation,
                    device,
                )
            })
            .collect();
        Self { conv_bn_blocks }
    }

    fn forward(&self, mut input: Tensor<B, 3>) -> Tensor<B, 3> {
        for block in &self.conv_bn_blocks {
            input = block.forward(input);
        }
        input
    }
}

#[derive(Module, Debug)]
pub struct ResidualConv1dBnBlock<B: Backend> {
    pub res_blocks: Vec<Conv1dBnBlock<B>>,
}

impl<B: Backend> ResidualConv1dBnBlock<B> {
    fn init(channels: usize, config: &ResidualConvConfig, device: &B::Device) -> Self {
        let res_blocks = config
            .dilations
            .iter()
            .map(|&dilation| {
                Conv1dBnBlock::init(
                    channels,
                    channels,
                    channels,
                    config.kernel_size,
                    dilation,
                    config.num_conv_blocks,
                    device,
                )
            })
            .collect();
        Self { res_blocks }
    }

    fn forward(&self, input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut output = input * mask.clone();
        for block in &self.res_blocks {
            let residual = output.clone();
            output = (block.forward(output) + residual) * mask.clone();
        }
        output
    }
}

/// Parameter-bearing entries of a PyTorch `nn.Sequential`.
///
/// Burn Store's contiguous-index mapping removes the gaps occupied by ReLU
/// modules in the source checkpoint while preserving all parameter paths.
#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Sequential1d<B: Backend> {
    Conv(Conv1d<B>),
    Norm(BatchNorm<B>),
    ConvBn(Conv1dBnBlock<B>),
}

impl<B: Backend> Sequential1d<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Conv(layer) => layer.forward(input),
            Self::Norm(layer) => layer.forward(input),
            Self::ConvBn(layer) => layer.forward(input),
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualConvEncoder<B: Backend> {
    pub prenet: Vec<Conv1d<B>>,
    pub res_conv_block: ResidualConv1dBnBlock<B>,
    pub postnet: Vec<Sequential1d<B>>,
}

impl<B: Backend> ResidualConvEncoder<B> {
    fn init(channels: usize, config: &ResidualConvConfig, device: &B::Device) -> Self {
        Self {
            prenet: vec![conv1d(
                channels,
                channels,
                1,
                1,
                PaddingConfig1d::Valid,
                device,
            )],
            res_conv_block: ResidualConv1dBnBlock::init(channels, config, device),
            postnet: vec![
                Sequential1d::Conv(conv1d(
                    channels,
                    channels,
                    1,
                    1,
                    PaddingConfig1d::Valid,
                    device,
                )),
                Sequential1d::Norm(
                    BatchNormConfig::new(channels)
                        .with_epsilon(BATCH_NORM_EPSILON)
                        .init(device),
                ),
                Sequential1d::Conv(conv1d(
                    channels,
                    channels,
                    1,
                    1,
                    PaddingConfig1d::Valid,
                    device,
                )),
            ],
        }
    }

    fn forward(&self, input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        let embedded = input.clone();
        let mut output = relu(self.prenet[0].forward(input)) * mask.clone();
        output = self.res_conv_block.forward(output, mask.clone());
        output = output + embedded;
        output = relu(self.postnet[0].forward(output));
        output = self.postnet[1].forward(output);
        output = self.postnet[2].forward(output) * mask.clone();
        output * mask
    }
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    pub encoder: ResidualConvEncoder<B>,
}

impl<B: Backend> Encoder<B> {
    fn forward(&self, input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(input, mask)
    }
}

#[derive(Module, Debug)]
pub struct ResidualConvDecoder<B: Backend> {
    pub res_conv_block: ResidualConv1dBnBlock<B>,
    pub post_conv: Conv1d<B>,
    pub postnet: Vec<Sequential1d<B>>,
}

impl<B: Backend> ResidualConvDecoder<B> {
    fn init(
        hidden_channels: usize,
        out_channels: usize,
        config: &ResidualConvConfig,
        device: &B::Device,
    ) -> Self {
        Self {
            res_conv_block: ResidualConv1dBnBlock::init(hidden_channels, config, device),
            post_conv: conv1d(
                hidden_channels,
                hidden_channels,
                1,
                1,
                PaddingConfig1d::Valid,
                device,
            ),
            postnet: vec![
                Sequential1d::ConvBn(Conv1dBnBlock::init(
                    hidden_channels,
                    hidden_channels,
                    hidden_channels,
                    config.kernel_size,
                    1,
                    2,
                    device,
                )),
                Sequential1d::Conv(conv1d(
                    hidden_channels,
                    out_channels,
                    1,
                    1,
                    PaddingConfig1d::Valid,
                    device,
                )),
            ],
        }
    }

    fn forward(&self, input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = input.clone();
        let mut output = self.res_conv_block.forward(input, mask.clone());
        output = self.post_conv.forward(output) + residual;
        output = self.postnet[0].forward(output);
        self.postnet[1].forward(output) * mask
    }
}

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    pub decoder: ResidualConvDecoder<B>,
}

impl<B: Backend> Decoder<B> {
    fn forward(&self, input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        self.decoder.forward(input, mask)
    }
}

/// Channel-axis layer normalization with checkpoint names `gamma`/`beta`.
#[derive(Module, Debug)]
pub struct ChannelLayerNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 3>>,
    pub beta: Param<Tensor<B, 3>>,
    epsilon: f64,
}

impl<B: Backend> ChannelLayerNorm<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            gamma: Initializer::Constant { value: 0.1 }.init([1, channels, 1], device),
            beta: Initializer::Zeros.init([1, channels, 1], device),
            epsilon: DURATION_NORM_EPSILON,
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mean = input.clone().mean_dim(1);
        let variance = (input.clone() - mean.clone()).square().mean_dim(1);
        (input - mean) / (variance + self.epsilon).sqrt() * self.gamma.val() + self.beta.val()
    }
}

/// Glow-TTS duration predictor used by the released ForwardTTS checkpoint.
#[derive(Module, Debug)]
pub struct DurationPredictor<B: Backend> {
    pub conv_1: Conv1d<B>,
    pub norm_1: ChannelLayerNorm<B>,
    pub conv_2: Conv1d<B>,
    pub norm_2: ChannelLayerNorm<B>,
    pub proj: Conv1d<B>,
}

impl<B: Backend> DurationPredictor<B> {
    pub(crate) fn init(
        channels_in: usize,
        hidden_channels: usize,
        kernel_size: usize,
        device: &B::Device,
    ) -> Self {
        let padding = PaddingConfig1d::Explicit(kernel_size / 2, kernel_size / 2);
        Self {
            conv_1: conv1d(
                channels_in,
                hidden_channels,
                kernel_size,
                1,
                padding.clone(),
                device,
            ),
            norm_1: ChannelLayerNorm::init(hidden_channels, device),
            conv_2: conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size,
                1,
                padding,
                device,
            ),
            norm_2: ChannelLayerNorm::init(hidden_channels, device),
            proj: conv1d(hidden_channels, 1, 1, 1, PaddingConfig1d::Valid, device),
        }
    }

    pub(crate) fn forward(&self, input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut output = relu(self.conv_1.forward(input * mask.clone()));
        output = self.norm_1.forward(output);
        // Dropout is deliberately inactive during deterministic inference.
        output = relu(self.conv_2.forward(output * mask.clone()));
        output = self.norm_2.forward(output);
        self.proj.forward(output * mask.clone()) * mask
    }
}

#[derive(Module, Debug)]
pub struct PositionalEncoding<B: Backend> {
    pub(crate) pe: Param<Tensor<B, 3>>,
    pub(crate) channels: usize,
}

impl<B: Backend> PositionalEncoding<B> {
    pub(crate) fn init(channels: usize, device: &B::Device) -> Self {
        let mut values = vec![0.0f32; channels * POSITIONAL_ENCODING_LIMIT];
        for channel in 0..channels {
            let pair = channel / 2;
            // Coqui constructs this table in torch.float32. Keeping the
            // exponent, power, and phase in f32 matters at high channels:
            // small divisor rounding differences become large phase shifts
            // over a 5,000-position table.
            let divisor = 10_000f32.powf((2 * pair) as f32 / channels as f32);
            for position in 0..POSITIONAL_ENCODING_LIMIT {
                let angle = position as f32 * divisor;
                values[channel * POSITIONAL_ENCODING_LIMIT + position] =
                    if channel.is_multiple_of(2) {
                        angle.sin()
                    } else {
                        angle.cos()
                    };
            }
        }
        let pe = Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([
            1,
            channels,
            POSITIONAL_ENCODING_LIMIT,
        ]);
        Self {
            pe: Param::from_tensor(pe),
            channels,
        }
    }

    pub(crate) fn forward(
        &self,
        input: Tensor<B, 3>,
        mask: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>, SpeedySpeechError> {
        let frames = input.dims()[2];
        if frames > POSITIONAL_ENCODING_LIMIT {
            return Err(input_error(format!(
                "predicted {frames} frames but positional encoding is limited to {POSITIONAL_ENCODING_LIMIT}"
            )));
        }
        let positional = self.pe.val().slice([0..1, 0..self.channels, 0..frames]) * mask;
        Ok(input * (self.channels as f64).sqrt() + positional)
    }
}

/// Training-only Gaussian aligner retained for complete checkpoint loading.
#[derive(Module, Debug)]
pub struct AlignmentNetwork<B: Backend> {
    pub key_layer: Vec<Conv1d<B>>,
    pub query_layer: Vec<Conv1d<B>>,
}

impl<B: Backend> AlignmentNetwork<B> {
    pub(crate) fn init(
        query_channels: usize,
        key_channels: usize,
        attention_channels: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            key_layer: vec![
                conv1d(
                    key_channels,
                    key_channels * 2,
                    3,
                    1,
                    PaddingConfig1d::Explicit(1, 1),
                    device,
                ),
                conv1d(
                    key_channels * 2,
                    attention_channels,
                    1,
                    1,
                    PaddingConfig1d::Valid,
                    device,
                ),
            ],
            query_layer: vec![
                conv1d(
                    query_channels,
                    query_channels * 2,
                    3,
                    1,
                    PaddingConfig1d::Explicit(1, 1),
                    device,
                ),
                conv1d(
                    query_channels * 2,
                    query_channels,
                    1,
                    1,
                    PaddingConfig1d::Valid,
                    device,
                ),
                conv1d(
                    query_channels,
                    attention_channels,
                    1,
                    1,
                    PaddingConfig1d::Valid,
                    device,
                ),
            ],
        }
    }
}

/// Burn tensors returned by deterministic acoustic inference.
#[derive(Debug)]
pub struct SpeedySpeechOutput<B: Backend> {
    /// Mel spectrogram in `[batch, frames, mel_channels]` layout.
    pub mel: Tensor<B, 3>,
    /// Rounded per-token durations in `[batch, tokens]` layout.
    pub durations: Tensor<B, 2>,
}

/// Native Burn acoustic model matching the published checkpoint hierarchy.
#[derive(Module, Debug)]
pub struct SpeedySpeech<B: Backend> {
    pub emb: Embedding<B>,
    pub encoder: Encoder<B>,
    pub pos_encoder: Option<PositionalEncoding<B>>,
    pub decoder: Decoder<B>,
    pub duration_predictor: DurationPredictor<B>,
    pub aligner: Option<AlignmentNetwork<B>>,
    length_scale: f64,
    num_chars: usize,
    out_channels: usize,
    minimum_input_tokens: usize,
    minimum_output_frames: usize,
    max_output_frames: usize,
}

impl<B: Backend> SpeedySpeech<B> {
    /// Load the unmodified `.pth` checkpoint directly through Burn Store.
    pub fn load_checkpoint(mut self, path: impl AsRef<Path>) -> Result<Self, SpeedySpeechError> {
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut self,
            path.as_ref(),
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                predicate: Some(checkpoint_tensor),
                key_remappings: vec![
                    (r"(\.norm)\.weight$".into(), "$1.gamma".into()),
                    (r"(\.norm)\.bias$".into(), "$1.beta".into()),
                    (
                        r"^(encoder\.encoder\.postnet\.2)\.weight$".into(),
                        "$1.gamma".into(),
                    ),
                    (
                        r"^(encoder\.encoder\.postnet\.2)\.bias$".into(),
                        "$1.beta".into(),
                    ),
                ],
                skip_enum_variants: true,
                ..Default::default()
            },
        )
        .map_err(|error| SpeedySpeechError::Checkpoint(error.to_string()))?;
        let unexpected_unused = result
            .unused
            .iter()
            .filter(|path| !path.ends_with(".num_batches_tracked"))
            .cloned()
            .collect::<Vec<_>>();
        if !result.missing.is_empty() || !result.errors.is_empty() || !unexpected_unused.is_empty()
        {
            return Err(SpeedySpeechError::Checkpoint(format!(
                "checkpoint does not exactly match the Burn model: {} missing, {} load errors, unexpected tensors: {}",
                result.missing.len(),
                result.errors.len(),
                unexpected_unused.join(", ")
            )));
        }
        // `pe` is a deterministic buffer, not a learned weight. Burn Store's
        // PyTorch conversion transposes its last two dimensions as though it
        // were a linear weight, which preserves frame zero but corrupts every
        // later position. Rebuild it from the published architecture after
        // validating that the checkpoint otherwise matches exactly.
        if let Some(positional) = &self.pos_encoder {
            let device = positional.pe.val().device();
            self.pos_encoder = Some(PositionalEncoding::init(positional.channels, &device));
        }
        Ok(self)
    }

    /// Run deterministic inference from already-projected model token IDs.
    pub fn inference(
        &self,
        token_ids: Tensor<B, 2, Int>,
    ) -> Result<SpeedySpeechOutput<B>, SpeedySpeechError> {
        self.inference_with_length_scale(token_ids, self.length_scale)
    }

    /// Run deterministic inference with a request-local duration scale.
    pub fn inference_with_length_scale(
        &self,
        token_ids: Tensor<B, 2, Int>,
        length_scale: f64,
    ) -> Result<SpeedySpeechOutput<B>, SpeedySpeechError> {
        self.inference_inner(token_ids, length_scale, false, None)
    }

    pub(crate) fn inference_projected_with_length_scale(
        &self,
        token_ids: Tensor<B, 2, Int>,
        length_scale: f64,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<SpeedySpeechOutput<B>, SpeedySpeechError> {
        self.inference_inner(token_ids, length_scale, true, profiler)
    }

    fn inference_inner(
        &self,
        token_ids: Tensor<B, 2, Int>,
        length_scale: f64,
        ids_validated_on_host: bool,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<SpeedySpeechOutput<B>, SpeedySpeechError> {
        let mut profiler = profiler;
        if !length_scale.is_finite() || length_scale <= 0.0 {
            return Err(input_error("length_scale must be finite and positive"));
        }
        let [batch, tokens] = token_ids.dims();
        if batch == 0 || tokens == 0 {
            return Err(input_error(
                "token_ids must have non-empty [batch, tokens] dimensions",
            ));
        }
        if tokens < self.minimum_input_tokens {
            return Err(input_error(format!(
                "token sequence contains {tokens} tokens but this checkpoint requires at least {} for its dilated encoder",
                self.minimum_input_tokens
            )));
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
        let x_mask = Tensor::<B, 3>::ones([batch, 1, tokens], &device);
        let started = Instant::now();
        let embedded = self.emb.forward(token_ids).swap_dims(1, 2);
        let encoded = self.encoder.forward(embedded, x_mask.clone());
        finish_backend_stage::<B>(
            &mut profiler,
            &device,
            SynthesisStage::TextEncoder,
            started,
            [SynthesisDimension::new("tokens", tokens)],
        )
        .map_err(|error| input_error(error.to_string()))?;

        let started = Instant::now();
        let duration_log = self
            .duration_predictor
            .forward(encoded.clone(), x_mask.clone());
        let durations = ((duration_log.exp() - 1.0) * x_mask * length_scale)
            .clamp_min(1.0)
            .round()
            .reshape([batch, tokens]);
        finish_backend_stage::<B>(
            &mut profiler,
            &device,
            SynthesisStage::DurationPrediction,
            started,
            [SynthesisDimension::new("tokens", tokens)],
        )
        .map_err(|error| input_error(error.to_string()))?;

        let started = Instant::now();
        let (expanded, output_mask) =
            expand_by_durations(encoded, durations.clone(), self.max_output_frames)?;
        let output_frames = expanded.dims()[2];
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
        if output_frames < self.minimum_output_frames {
            return Err(input_error(format!(
                "duration predictor produced {output_frames} frames but this checkpoint requires at least {} for its dilated decoder",
                self.minimum_output_frames
            )));
        }
        let expanded = match &self.pos_encoder {
            Some(positional) => positional.forward(expanded, output_mask.clone())?,
            None => expanded,
        };
        let started = Instant::now();
        let mel = self.decoder.forward(expanded, output_mask).swap_dims(1, 2);
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
        debug_assert_eq!(mel.dims()[2], self.out_channels);

        Ok(SpeedySpeechOutput { mel, durations })
    }
}

fn checkpoint_tensor(path: &str, _container: &str) -> bool {
    !path.ends_with(".num_batches_tracked")
}

pub(crate) fn expand_by_durations<B: Backend>(
    encoded: Tensor<B, 3>,
    durations: Tensor<B, 2>,
    max_output_frames: usize,
) -> Result<(Tensor<B, 3>, Tensor<B, 3>), SpeedySpeechError> {
    let [batch, channels, tokens] = encoded.dims();
    let ends = durations.clone().cumsum(1);
    let output_frames = ends.clone().max().into_scalar().elem::<f32>() as usize;
    if output_frames == 0 {
        return Err(input_error(
            "duration predictor produced zero output frames",
        ));
    }
    if output_frames > max_output_frames {
        return Err(input_error(format!(
            "duration predictor requested {output_frames} frames, exceeding configured limit {max_output_frames}"
        )));
    }
    if output_frames > POSITIONAL_ENCODING_LIMIT {
        return Err(input_error(format!(
            "duration predictor requested {output_frames} frames, exceeding positional limit {POSITIONAL_ENCODING_LIMIT}"
        )));
    }

    let device = encoded.device();
    let starts = (ends.clone() - durations.clone())
        .reshape([batch, tokens, 1])
        .expand([batch, tokens, output_frames]);
    let ends = ends
        .reshape([batch, tokens, 1])
        .expand([batch, tokens, output_frames]);
    let positions = Tensor::<B, 1, Int>::arange(0..output_frames as i64, &device)
        .float()
        .reshape([1, 1, output_frames])
        .expand([batch, tokens, output_frames]);
    let attention = positions
        .clone()
        .greater_equal(starts)
        .bool_and(positions.lower(ends))
        .float();
    let expanded = encoded.matmul(attention);

    let lengths = durations
        .sum_dim(1)
        .reshape([batch, 1, 1])
        .expand([batch, 1, output_frames]);
    let output_mask = Tensor::<B, 1, Int>::arange(0..output_frames as i64, &device)
        .float()
        .reshape([1, 1, output_frames])
        .expand([batch, 1, output_frames])
        .lower(lengths)
        .float();
    debug_assert_eq!(expanded.dims(), [batch, channels, output_frames]);
    Ok((expanded, output_mask))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    fn tiny_config() -> SpeedySpeechConfig {
        SpeedySpeechConfig {
            num_chars: 8,
            out_channels: 3,
            hidden_channels: 4,
            positional_encoding: true,
            length_scale: 1.0,
            encoder: ResidualConvConfig {
                kernel_size: 2,
                dilations: vec![1],
                num_conv_blocks: 1,
                num_res_blocks: 1,
            },
            decoder: ResidualConvConfig {
                kernel_size: 2,
                dilations: vec![1],
                num_conv_blocks: 1,
                num_res_blocks: 1,
            },
            duration_predictor_hidden_channels: 4,
            duration_predictor_kernel_size: 3,
            duration_predictor_dropout: 0.1,
            use_aligner: true,
            max_duration: 10,
            max_output_frames: 64,
        }
    }

    fn published_json() -> Value {
        serde_json::json!({
            "model": "speedy_speech",
            "characters": {
                "pad": "_",
                "eos": "~",
                "bos": "^",
                "phonemes": "abcd",
                "punctuations": " !"
            },
            "model_args": {
                "num_chars": null,
                "out_channels": 80,
                "hidden_channels": 128,
                "num_speakers": 0,
                "use_aligner": true,
                "use_pitch": false,
                "duration_predictor_hidden_channels": 256,
                "duration_predictor_kernel_size": 3,
                "duration_predictor_dropout_p": 0.1,
                "positional_encoding": true,
                "length_scale": 1,
                "encoder_type": "residual_conv_bn",
                "encoder_params": {
                    "kernel_size": 4,
                    "dilations": [1, 2],
                    "num_conv_blocks": 2,
                    "num_res_blocks": 2
                },
                "decoder_type": "residual_conv_bn",
                "decoder_params": {
                    "kernel_size": 4,
                    "dilations": [1, 2, 4],
                    "num_conv_blocks": 2,
                    "num_res_blocks": 3
                },
                "use_d_vector": false,
                "d_vector_dim": 0,
                "max_duration": 75
            }
        })
    }

    #[test]
    fn parses_the_released_model_section_without_text_lowering() {
        let config = SpeedySpeechConfig::from_json_value(&published_json()).unwrap();
        assert_eq!(config.num_chars, 9);
        assert_eq!(config.hidden_channels, 128);
        assert_eq!(config.out_channels, 80);
        assert_eq!(config.encoder.dilations, vec![1, 2]);
        assert_eq!(config.decoder.dilations, vec![1, 2, 4]);
        assert_eq!(config.duration_predictor_hidden_channels, 256);
    }

    #[test]
    fn rejects_a_fastpitch_or_speaker_conditioned_checkpoint() {
        let mut value = published_json();
        value["model_args"]["use_pitch"] = Value::Bool(true);
        assert!(SpeedySpeechConfig::from_json_value(&value)
            .unwrap_err()
            .to_string()
            .contains("FastPitch"));

        let mut value = published_json();
        value["model_args"]["num_speakers"] = Value::from(2);
        assert!(SpeedySpeechConfig::from_json_value(&value)
            .unwrap_err()
            .to_string()
            .contains("single-speaker"));
    }

    #[test]
    fn expands_token_encodings_by_opaque_predicted_durations() {
        let device = Default::default();
        let encoded =
            Tensor::<TestBackend, 3>::from_floats([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]], &device);
        let durations = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 1.0]], &device);
        let (expanded, mask) = expand_by_durations(encoded, durations, 8).unwrap();
        assert_eq!(expanded.dims(), [1, 2, 4]);
        assert_eq!(mask.dims(), [1, 1, 4]);
        assert_eq!(
            expanded.into_data().as_slice::<f32>().unwrap(),
            &[1.0, 2.0, 2.0, 3.0, 10.0, 20.0, 20.0, 30.0]
        );
    }

    #[test]
    fn tiny_inference_is_deterministic_finite_and_shape_consistent() {
        let device = Default::default();
        let model = tiny_config().init::<TestBackend>(&device).unwrap();
        let input = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device);

        let first = model.inference(input.clone()).unwrap();
        let second = model.inference(input).unwrap();
        let [batch, frames, channels] = first.mel.dims();
        assert_eq!(batch, 1);
        assert_eq!(channels, 3);
        assert!(frames >= 3);
        assert_eq!(first.durations.dims(), [1, 3]);
        assert_eq!(first.mel.to_data(), second.mel.to_data());
        assert!(first
            .mel
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn inference_rejects_out_of_vocabulary_ids() {
        let device = Default::default();
        let model = tiny_config().init::<TestBackend>(&device).unwrap();
        let input = Tensor::<TestBackend, 2, Int>::from_ints([[1, 8]], &device);
        let error = model.inference(input).unwrap_err();
        assert!(error.to_string().contains("outside vocabulary"));
    }

    #[test]
    fn positional_table_matches_coqui_float32_layout_without_a_checkpoint() {
        let device = Default::default();
        let positional = PositionalEncoding::<TestBackend>::init(128, &device);
        let data = positional
            .pe
            .val()
            .into_data()
            .to_vec::<f32>()
            .expect("f32 positional table");

        for (channel, position, expected) in [
            (0, 0, 0.0),
            (7, 1, 0.030864894),
            (64, 168, -0.9449728),
            (127, 338, 0.8975993),
        ] {
            let actual = data[channel * POSITIONAL_ENCODING_LIMIT + position];
            assert!(
                (actual - expected).abs() <= 1e-6,
                "positional[{channel},{position}] mismatch: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn loads_and_runs_the_published_checkpoint_when_available() {
        let Some(model_path) = std::env::var_os("TONGUES_TEST_COQUI_SPEEDY_MODEL") else {
            return;
        };
        let config_path = std::env::var_os("TONGUES_TEST_COQUI_SPEEDY_CONFIG")
            .expect("TONGUES_TEST_COQUI_SPEEDY_CONFIG must accompany the model");
        let source = std::fs::read_to_string(config_path).expect("config");
        let value: Value = serde_json::from_str(&source).expect("JSON config");
        let config = SpeedySpeechConfig::from_json_value(&value).expect("model config");
        let device = Default::default();
        let model = config
            .init::<TestBackend>(&device)
            .expect("model")
            .load_checkpoint(model_path)
            .expect("checkpoint");
        let input = Tensor::<TestBackend, 2, Int>::from_ints(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]],
            &device,
        );

        let output = model.inference(input).expect("inference");

        assert_eq!(output.mel.dims()[0], 1);
        assert_eq!(output.mel.dims()[2], config.out_channels);
        assert_eq!(output.durations.dims(), [1, 13]);
        assert!(output
            .mel
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|sample| sample.is_finite()));
    }

    #[test]
    #[ignore = "requires pinned Coqui SpeedySpeech model artifacts; run scripts/speech-conformance.sh"]
    fn published_checkpoint_stage_parity() {
        let model_path = std::env::var_os("TONGUES_TEST_COQUI_SPEEDY_MODEL")
            .expect("TONGUES_TEST_COQUI_SPEEDY_MODEL is required");
        let config_path = std::env::var_os("TONGUES_TEST_COQUI_SPEEDY_CONFIG")
            .expect("TONGUES_TEST_COQUI_SPEEDY_CONFIG must accompany the model");
        let source = std::fs::read_to_string(config_path).expect("config");
        let value: Value = serde_json::from_str(&source).expect("JSON config");
        let config = SpeedySpeechConfig::from_json_value(&value).expect("model config");
        let device = Default::default();
        let model = config
            .init::<TestBackend>(&device)
            .expect("model")
            .load_checkpoint(model_path)
            .expect("checkpoint");
        let ids = [
            14, 43, 77, 15, 63, 33, 129, 13, 3, 63, 21, 129, 77, 50, 20, 21, 63, 6, 129, 43, 15,
            129, 30, 48, 129, 20, 10, 6, 49, 129, 21, 77, 10, 27, 129, 24, 3, 63, 13, 129, 30, 48,
            129, 12, 50, 21, 48, 13, 129, 4, 63, 55, 28, 15, 129, 21, 48, 129, 20, 63, 33, 125,
        ];
        let token_ids = Tensor::<TestBackend, 1, Int>::from_ints(ids, &device).reshape([1, 62]);
        let x_mask = Tensor::<TestBackend, 3>::ones([1, 1, 62], &device);

        let embedded = model.emb.forward(token_ids).swap_dims(1, 2);
        let encoded = model.encoder.forward(embedded, x_mask.clone());
        let duration_log = model
            .duration_predictor
            .forward(encoded.clone(), x_mask.clone());
        let durations = ((duration_log.exp() - 1.0) * x_mask * model.length_scale)
            .clamp_min(1.0)
            .round()
            .reshape([1, 62]);
        let (expanded, output_mask) =
            expand_by_durations(encoded.clone(), durations.clone(), model.max_output_frames)
                .expect("duration expansion");
        let positioned = model
            .pos_encoder
            .as_ref()
            .expect("published positional encoder")
            .forward(expanded.clone(), output_mask.clone())
            .expect("positional encoding");
        let mel = model
            .decoder
            .forward(positioned.clone(), output_mask)
            .swap_dims(1, 2);

        let encoded_data = encoded
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("f32 encoded");
        let expanded_before_position_data =
            expanded.into_data().to_vec::<f32>().expect("f32 expanded");
        let positioned_data = positioned
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("f32 positioned");
        let positional_data = model
            .pos_encoder
            .as_ref()
            .expect("published positional encoder")
            .pe
            .val()
            .into_data()
            .to_vec::<f32>()
            .expect("f32 positional");
        for (actual, expected, stage) in [
            (encoded_data[0], 0.28330755, "encoded[0,0]"),
            (encoded_data[7 * 62 + 1], -0.017651677, "encoded[7,1]"),
            (encoded_data[64 * 62 + 20], 0.014495345, "encoded[64,20]"),
            (encoded_data[127 * 62 + 61], -0.11383733, "encoded[127,61]"),
            (
                expanded_before_position_data[0],
                0.28330755,
                "expanded[0,0]",
            ),
            (
                expanded_before_position_data[7 * 339 + 1],
                -0.47962627,
                "expanded[7,1]",
            ),
            (
                expanded_before_position_data[64 * 339 + 168],
                0.26118344,
                "expanded[64,168]",
            ),
            (
                expanded_before_position_data[127 * 339 + 338],
                -0.11383733,
                "expanded[127,338]",
            ),
            (positioned_data[0], 3.205259, "positioned[0,0]"),
            (positioned_data[7 * 339 + 1], -5.395487, "positioned[7,1]"),
            (
                positioned_data[64 * 339 + 168],
                2.0099804,
                "positioned[64,168]",
            ),
            (
                positioned_data[127 * 339 + 338],
                -0.3903231,
                "positioned[127,338]",
            ),
            (positional_data[0], 0.0, "positional[0,0]"),
            (
                positional_data[7 * POSITIONAL_ENCODING_LIMIT + 1],
                0.030864894,
                "positional[7,1]",
            ),
            (
                positional_data[64 * POSITIONAL_ENCODING_LIMIT + 168],
                -0.9449728,
                "positional[64,168]",
            ),
            (
                positional_data[127 * POSITIONAL_ENCODING_LIMIT + 338],
                0.8975993,
                "positional[127,338]",
            ),
        ] {
            assert!(
                (actual - expected).abs() <= 2e-5,
                "{stage} parity mismatch: actual={actual}, expected={expected}"
            );
        }

        let actual_durations = durations
            .into_data()
            .to_vec::<f32>()
            .expect("f32 durations");
        assert_eq!(
            actual_durations,
            vec![
                3.0, 7.0, 6.0, 5.0, 3.0, 8.0, 5.0, 2.0, 6.0, 7.0, 12.0, 5.0, 3.0, 6.0, 11.0, 3.0,
                6.0, 3.0, 9.0, 4.0, 5.0, 2.0, 1.0, 2.0, 12.0, 4.0, 5.0, 5.0, 9.0, 7.0, 8.0, 4.0,
                9.0, 7.0, 7.0, 2.0, 4.0, 13.0, 4.0, 3.0, 1.0, 2.0, 7.0, 5.0, 5.0, 7.0, 4.0, 6.0,
                4.0, 1.0, 4.0, 6.0, 6.0, 10.0, 3.0, 3.0, 2.0, 12.0, 3.0, 6.0, 11.0, 4.0,
            ]
        );
        assert_eq!(mel.dims(), [1, 339, 80]);
        let mel_data = mel.clone().into_data().to_vec::<f32>().expect("f32 mel");
        let reference = [
            (0, 0, -7.4673758),
            (0, 79, -9.108585),
            (1, 7, -3.5108964),
            (5, 23, -5.008403),
            (20, 40, -4.3483267),
            (50, 79, -5.0085955),
            (100, 23, -5.913802),
            (168, 40, -4.2294984),
            (250, 7, -3.2281606),
            (338, 79, -9.108585),
        ];
        for (frame, bin, expected) in reference {
            let actual = mel_data[frame * config.out_channels + bin];
            assert!(
                (actual - expected).abs() <= 2e-4,
                "mel parity mismatch at frame {frame}, bin {bin}: actual={actual}, expected={expected}"
            );
        }

        let vocoder_model = std::env::var_os("TONGUES_TEST_COQUI_HIFIGAN_MODEL")
            .expect("TONGUES_TEST_COQUI_HIFIGAN_MODEL is required");
        let vocoder_config = std::env::var_os("TONGUES_TEST_COQUI_HIFIGAN_CONFIG")
            .expect("TONGUES_TEST_COQUI_HIFIGAN_CONFIG is required");
        let generator = crate::HifiganBundleConfig::from_file(vocoder_config)
            .expect("HiFi-GAN config")
            .load_burn_generator(vocoder_model, &device)
            .expect("HiFi-GAN checkpoint");
        let waveform = generator
            .inference(mel.swap_dims(1, 2))
            .expect("HiFi-GAN inference");
        assert_eq!(waveform.dims(), [1, 1, 89_344]);
        let waveform = waveform.into_data().to_vec::<f32>().expect("f32 waveform");
        assert!(waveform.iter().all(|sample| sample.is_finite()));
        let rms = (waveform.iter().map(|sample| sample * sample).sum::<f32>()
            / waveform.len() as f32)
            .sqrt();
        assert!((rms - 0.05458769).abs() <= 2e-5, "waveform RMS: {rms}");
        for (index, expected) in [
            (0, -0.0014194329),
            (1, -0.0012611531),
            (255, -0.00044831855),
            (256, -0.00046205145),
            (1_000, 0.0004874973),
            (44_672, -0.011885947),
            (89_342, 0.00084172835),
            (89_343, 0.0003609802),
        ] {
            let actual = waveform[index];
            assert!(
                (actual - expected).abs() <= 2e-4,
                "waveform parity mismatch at sample {index}: actual={actual}, expected={expected}"
            );
        }
    }
}
