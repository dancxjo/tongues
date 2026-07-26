//! Native Burn implementation of a HiFi-GAN generator.
//!
//! The module layout deliberately follows a published
//! `TTS/vocoder/models/hifigan_generator.py` checkpoint hierarchy. The fields named
//! `conv_pre`, `ups`, `resblocks`, `conv_post`, `convs1`, and `convs2`, plus
//! the `weight_g`, `weight_v`, and `bias` parameters below, produce the same
//! paths as older weight-normalized PyTorch checkpoints when used
//! with Burn's `PytorchStore`.
//!
//! Configuration can be deserialized with Burn's [`Config`] implementation
//! when the parent has emitted ordinary JSON. [`HifiganGeneratorConfig::from_json_value`]
//! is also provided for a parent that has already parsed JSON5 and wants
//! checkpoint-compatible defaults applied before model construction.
//!
//! Source provenance: `audit-required`. This module targets published Coqui
//! checkpoint structure and behavior; no claim of independent implementation
//! or source adaptation should be made until the ledger in
//! `docs/provenance.md` records a file-by-file comparison.

use std::fmt;

use burn::config::Config;
use burn::module::{Initializer, Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::PaddingConfig1d;
use burn::tensor::activation::leaky_relu;
use burn::tensor::backend::Backend;
use burn::tensor::module::{conv1d, conv_transpose1d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions, PadMode};
use burn::tensor::Tensor;
use serde_json::{Map, Value};

const LRELU_SLOPE: f64 = 0.1;
const POST_LRELU_SLOPE: f64 = 0.01;
const PYTORCH_CONV_GAIN: f64 = 0.577_350_269_189_625_8;

/// Errors detected before dispatching an invalid tensor operation to Burn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HifiganError {
    InvalidConfig(String),
    InvalidInput(String),
}

impl fmt::Display for HifiganError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(formatter, "invalid HiFi-GAN config: {message}"),
            Self::InvalidInput(message) => write!(formatter, "invalid HiFi-GAN input: {message}"),
        }
    }
}

impl std::error::Error for HifiganError {}

fn config_error(message: impl Into<String>) -> HifiganError {
    HifiganError::InvalidConfig(message.into())
}

fn input_error(message: impl Into<String>) -> HifiganError {
    HifiganError::InvalidInput(message.into())
}

fn pytorch_conv_initializer() -> Initializer {
    // Equivalent to torch.nn.Conv1d.reset_parameters(): kaiming_uniform_(a=sqrt(5)).
    Initializer::KaimingUniform {
        gain: PYTORCH_CONV_GAIN,
        fan_out_only: false,
    }
}

fn same_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size * dilation - dilation) / 2
}

/// Configuration for the first HiFi-GAN residual-block variant.
#[derive(Config, Debug, PartialEq)]
pub struct ResBlock1Config {
    pub channels: usize,
    #[config(default = 3)]
    pub kernel_size: usize,
    #[config(default = "vec![1, 3, 5]")]
    pub dilations: Vec<usize>,
}

impl ResBlock1Config {
    pub fn validate(&self) -> Result<(), HifiganError> {
        if self.channels == 0 {
            return Err(config_error("residual-block channels must be positive"));
        }
        if self.kernel_size == 0 || self.kernel_size.is_multiple_of(2) {
            return Err(config_error(format!(
                "ResBlock1 kernel size {} must be positive and odd",
                self.kernel_size
            )));
        }
        if self.dilations.len() != 3 {
            return Err(config_error(format!(
                "ResBlock1 requires exactly three dilations, got {}",
                self.dilations.len()
            )));
        }
        if self.dilations.contains(&0) {
            return Err(config_error("ResBlock1 dilations must be positive"));
        }
        Ok(())
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<ResBlock1<B>, HifiganError> {
        self.validate()?;

        let convs1 = self
            .dilations
            .iter()
            .map(|&dilation| {
                WeightNormConv1d::new(
                    self.channels,
                    self.channels,
                    self.kernel_size,
                    1,
                    dilation,
                    same_padding(self.kernel_size, dilation),
                    true,
                    device,
                )
            })
            .collect();
        let convs2 = (0..self.dilations.len())
            .map(|_| {
                WeightNormConv1d::new(
                    self.channels,
                    self.channels,
                    self.kernel_size,
                    1,
                    1,
                    same_padding(self.kernel_size, 1),
                    true,
                    device,
                )
            })
            .collect();

        Ok(ResBlock1 {
            convs1,
            convs2,
            convs: Vec::new(),
            resblock_type: "1".into(),
        })
    }

    fn init_type2<B: Backend>(&self, device: &B::Device) -> Result<ResBlock1<B>, HifiganError> {
        if self.channels == 0
            || self.kernel_size == 0
            || self.kernel_size.is_multiple_of(2)
            || self.dilations.len() < 2
            || self.dilations[..2].contains(&0)
        {
            return Err(config_error(
                "ResBlock2 requires positive channels, an odd kernel, and two positive dilations",
            ));
        }
        let convs = self.dilations[..2]
            .iter()
            .map(|&dilation| {
                WeightNormConv1d::new(
                    self.channels,
                    self.channels,
                    self.kernel_size,
                    1,
                    dilation,
                    same_padding(self.kernel_size, dilation),
                    true,
                    device,
                )
            })
            .collect();
        Ok(ResBlock1 {
            convs1: Vec::new(),
            convs2: Vec::new(),
            convs,
            resblock_type: "2".into(),
        })
    }
}

/// A weight-normalized 1-D convolution with PyTorch-compatible parameter names.
///
/// `weight_g` has shape `[channels_out, 1, 1]` and `weight_v` has shape
/// `[channels_out, channels_in, kernel_size]`, matching PyTorch weight norm
/// with its default `dim = 0`.
#[derive(Module, Debug)]
pub struct WeightNormConv1d<B: Backend> {
    pub weight_g: Param<Tensor<B, 3>>,
    pub weight_v: Param<Tensor<B, 3>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl<B: Backend> WeightNormConv1d<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        padding: usize,
        bias: bool,
        device: &B::Device,
    ) -> Self {
        let fan_in = channels_in * kernel_size;
        let weight_v = pytorch_conv_initializer().init_with(
            [channels_out, channels_in, kernel_size],
            Some(fan_in),
            None,
            device,
        );
        let weight_g = weight_norm_dim_zero(weight_v.val()).detach();
        let bias = bias.then(|| {
            pytorch_conv_initializer().init_with([channels_out], Some(fan_in), None, device)
        });

        Self {
            weight_g: Param::from_tensor(weight_g),
            weight_v,
            bias,
            stride,
            padding,
            dilation,
        }
    }

    pub fn weight(&self) -> Tensor<B, 3> {
        let weight_v = self.weight_v.val();
        let norm = weight_norm_dim_zero(weight_v.clone());
        weight_v * self.weight_g.val() / norm
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        conv1d(
            input,
            self.weight(),
            self.bias.as_ref().map(Param::val),
            ConvOptions::new([self.stride], [self.padding], [self.dilation], 1),
        )
    }
}

/// A weight-normalized transposed 1-D convolution with checkpoint-compatible names.
///
/// PyTorch transposed-convolution weights are laid out
/// `[channels_in, channels_out, kernel_size]`. Weight norm still normalizes
/// every dimension except dimension zero.
#[derive(Module, Debug)]
pub struct WeightNormConvTranspose1d<B: Backend> {
    pub weight_g: Param<Tensor<B, 3>>,
    pub weight_v: Param<Tensor<B, 3>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub dilation: usize,
}

impl<B: Backend> WeightNormConvTranspose1d<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        padding: usize,
        bias: bool,
        device: &B::Device,
    ) -> Self {
        Self::new_with_output_padding(
            channels_in,
            channels_out,
            kernel_size,
            stride,
            dilation,
            padding,
            0,
            bias,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_output_padding(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        padding: usize,
        output_padding: usize,
        bias: bool,
        device: &B::Device,
    ) -> Self {
        let fan_in = channels_out * kernel_size;
        let weight_v = pytorch_conv_initializer().init_with(
            [channels_in, channels_out, kernel_size],
            Some(fan_in),
            None,
            device,
        );
        let weight_g = weight_norm_dim_zero(weight_v.val()).detach();
        let bias = bias.then(|| {
            pytorch_conv_initializer().init_with([channels_out], Some(fan_in), None, device)
        });

        Self {
            weight_g: Param::from_tensor(weight_g),
            weight_v,
            bias,
            stride,
            padding,
            output_padding,
            dilation,
        }
    }

    pub fn weight(&self) -> Tensor<B, 3> {
        let weight_v = self.weight_v.val();
        let norm = weight_norm_dim_zero(weight_v.clone());
        weight_v * self.weight_g.val() / norm
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        conv_transpose1d(
            input,
            self.weight(),
            self.bias.as_ref().map(Param::val),
            ConvTransposeOptions::new(
                [self.stride],
                [self.padding],
                [self.output_padding],
                [self.dilation],
                1,
            ),
        )
    }
}

fn weight_norm_dim_zero<B: Backend>(weight: Tensor<B, 3>) -> Tensor<B, 3> {
    weight.powf_scalar(2.0).sum_dims(&[1usize, 2usize]).sqrt()
}

/// HiFi-GAN residual block type 1.
#[derive(Module, Debug)]
pub struct ResBlock1<B: Backend> {
    pub convs1: Vec<WeightNormConv1d<B>>,
    pub convs2: Vec<WeightNormConv1d<B>>,
    pub convs: Vec<WeightNormConv1d<B>>,
    pub resblock_type: String,
}

impl<B: Backend> ResBlock1<B> {
    pub fn forward(&self, mut input: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.resblock_type == "2" {
            for conv in &self.convs {
                let residual = input.clone();
                input = conv.forward(leaky_relu(input, LRELU_SLOPE)) + residual;
            }
            return input;
        }
        for (conv1, conv2) in self.convs1.iter().zip(&self.convs2) {
            let residual = input.clone();
            let hidden = conv1.forward(leaky_relu(input, LRELU_SLOPE));
            let hidden = conv2.forward(leaky_relu(hidden, LRELU_SLOPE));
            input = hidden + residual;
        }
        input
    }
}

/// Configurable HiFi-GAN generator parameters.
///
/// Both upstream checkpoint layouts remain exact:
/// `resblocks.N.convs{1,2}.N.*` for type 1 and
/// `resblocks.N.convs.N.*` for type 2.
#[derive(Config, Debug, PartialEq)]
pub struct HifiganGeneratorConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub resblock_type: String,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_initial_channel: usize,
    pub upsample_factors: Vec<usize>,
    #[config(default = 5)]
    pub inference_padding: usize,
    #[config(default = 0)]
    pub cond_channels: usize,
    #[config(default = true)]
    pub conv_pre_weight_norm: bool,
    #[config(default = true)]
    pub conv_post_weight_norm: bool,
    #[config(default = true)]
    pub conv_post_bias: bool,
}

impl HifiganGeneratorConfig {
    /// Builds a config from ordinary JSON produced by a JSON5-aware parent.
    ///
    /// Common constructor defaults are applied for fields omitted from the
    /// model JSON. The parent remains responsible for selecting the generator
    /// object from a larger configuration document.
    pub fn from_json_value(value: &Value) -> Result<Self, HifiganError> {
        let object = value
            .as_object()
            .ok_or_else(|| config_error("generator config must be a JSON object"))?;

        let config = Self {
            in_channels: required_usize(object, "in_channels")?,
            out_channels: required_usize(object, "out_channels")?,
            resblock_type: required_string(object, "resblock_type")?,
            resblock_dilation_sizes: required_nested_usizes(object, "resblock_dilation_sizes")?,
            resblock_kernel_sizes: required_usizes(object, "resblock_kernel_sizes")?,
            upsample_kernel_sizes: required_usizes(object, "upsample_kernel_sizes")?,
            upsample_initial_channel: required_usize(object, "upsample_initial_channel")?,
            upsample_factors: required_usizes(object, "upsample_factors")?,
            inference_padding: optional_usize(object, "inference_padding", 5)?,
            cond_channels: optional_usize(object, "cond_channels", 0)?,
            conv_pre_weight_norm: optional_bool(object, "conv_pre_weight_norm", true)?,
            conv_post_weight_norm: optional_bool(object, "conv_post_weight_norm", true)?,
            conv_post_bias: optional_bool(object, "conv_post_bias", true)?,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), HifiganError> {
        if self.in_channels == 0 {
            return Err(config_error("in_channels must be positive"));
        }
        if self.out_channels == 0 {
            return Err(config_error("out_channels must be positive"));
        }
        if !matches!(self.resblock_type.as_str(), "1" | "2") {
            return Err(config_error(format!(
                "resblock_type `{}` is unsupported; expected `1` or `2`",
                self.resblock_type
            )));
        }
        if self.resblock_kernel_sizes.is_empty() {
            return Err(config_error(
                "at least one residual-block kernel is required",
            ));
        }
        if self.resblock_kernel_sizes.len() != self.resblock_dilation_sizes.len() {
            return Err(config_error(format!(
                "resblock_kernel_sizes has {} entries but resblock_dilation_sizes has {}",
                self.resblock_kernel_sizes.len(),
                self.resblock_dilation_sizes.len()
            )));
        }
        for (&kernel_size, dilations) in self
            .resblock_kernel_sizes
            .iter()
            .zip(&self.resblock_dilation_sizes)
        {
            let config = ResBlock1Config {
                channels: 1,
                kernel_size,
                dilations: dilations.clone(),
            };
            if self.resblock_type == "1" {
                config.validate()?;
            } else if config.kernel_size == 0
                || config.kernel_size.is_multiple_of(2)
                || config.dilations.len() < 2
                || config.dilations[..2].contains(&0)
            {
                return Err(config_error(
                    "ResBlock2 requires an odd kernel and at least two positive dilations",
                ));
            }
        }

        if self.upsample_factors.is_empty() {
            return Err(config_error("at least one upsampling stage is required"));
        }
        if self.upsample_factors.len() != self.upsample_kernel_sizes.len() {
            return Err(config_error(format!(
                "upsample_factors has {} entries but upsample_kernel_sizes has {}",
                self.upsample_factors.len(),
                self.upsample_kernel_sizes.len()
            )));
        }
        if self.upsample_initial_channel == 0 {
            return Err(config_error("upsample_initial_channel must be positive"));
        }
        for (stage, (&factor, &kernel_size)) in self
            .upsample_factors
            .iter()
            .zip(&self.upsample_kernel_sizes)
            .enumerate()
        {
            if factor == 0 {
                return Err(config_error(format!(
                    "upsample factor at stage {stage} must be positive"
                )));
            }
            if kernel_size < factor || !(kernel_size - factor).is_multiple_of(2) {
                return Err(config_error(format!(
                    "upsample kernel {kernel_size} at stage {stage} must be at least its factor {factor} and differ from it by an even number"
                )));
            }
            let channels_in = self.upsample_initial_channel >> stage;
            let channels_out = self.upsample_initial_channel >> (stage + 1);
            if channels_in == 0 || channels_out == 0 {
                return Err(config_error(format!(
                    "upsample_initial_channel {} is too small for {} stages",
                    self.upsample_initial_channel,
                    self.upsample_factors.len()
                )));
            }
        }
        if !self.conv_pre_weight_norm || !self.conv_post_weight_norm {
            return Err(config_error(
                "conv_pre_weight_norm and conv_post_weight_norm must be true for weight_g/weight_v checkpoint compatibility",
            ));
        }
        Ok(())
    }

    pub fn upsample_factor(&self) -> usize {
        self.upsample_factors.iter().product()
    }

    pub fn output_frames(&self, input_frames: usize) -> Option<usize> {
        input_frames.checked_mul(self.upsample_factor())
    }

    pub fn inference_output_frames(&self, input_frames: usize) -> Option<usize> {
        input_frames
            .checked_add(self.inference_padding.checked_mul(2)?)
            .and_then(|frames| frames.checked_mul(self.upsample_factor()))
    }

    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<HifiganGenerator<B>, HifiganError> {
        self.validate()?;

        let conv_pre = WeightNormConv1d::new(
            self.in_channels,
            self.upsample_initial_channel,
            7,
            1,
            1,
            3,
            true,
            device,
        );

        let mut ups = Vec::with_capacity(self.upsample_factors.len());
        let mut resblocks =
            Vec::with_capacity(self.upsample_factors.len() * self.resblock_kernel_sizes.len());
        for (stage, (&factor, &kernel_size)) in self
            .upsample_factors
            .iter()
            .zip(&self.upsample_kernel_sizes)
            .enumerate()
        {
            let channels_in = self.upsample_initial_channel >> stage;
            let channels_out = self.upsample_initial_channel >> (stage + 1);
            ups.push(WeightNormConvTranspose1d::new(
                channels_in,
                channels_out,
                kernel_size,
                factor,
                1,
                (kernel_size - factor) / 2,
                true,
                device,
            ));
            for (&resblock_kernel, dilations) in self
                .resblock_kernel_sizes
                .iter()
                .zip(&self.resblock_dilation_sizes)
            {
                let config = ResBlock1Config {
                    channels: channels_out,
                    kernel_size: resblock_kernel,
                    dilations: dilations.clone(),
                };
                resblocks.push(if self.resblock_type == "1" {
                    config.init(device)?
                } else {
                    config.init_type2(device)?
                });
            }
        }

        let final_channels = self.upsample_initial_channel >> self.upsample_factors.len();
        let conv_post = WeightNormConv1d::new(
            final_channels,
            self.out_channels,
            7,
            1,
            1,
            3,
            self.conv_post_bias,
            device,
        );
        let cond_layer = (self.cond_channels > 0).then(|| {
            Conv1dConfig::new(self.cond_channels, self.upsample_initial_channel, 1)
                .with_padding(PaddingConfig1d::Valid)
                .with_initializer(pytorch_conv_initializer())
                .init(device)
        });

        Ok(HifiganGenerator {
            conv_pre,
            ups,
            resblocks,
            conv_post,
            cond_layer,
            num_kernels: self.resblock_kernel_sizes.len(),
            num_upsamples: self.upsample_factors.len(),
            inference_padding: self.inference_padding,
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            cond_channels: self.cond_channels,
            upsample_factor: self.upsample_factor(),
        })
    }
}

/// Burn-native HiFi-GAN waveform generator.
#[derive(Module, Debug)]
pub struct HifiganGenerator<B: Backend> {
    pub conv_pre: WeightNormConv1d<B>,
    pub ups: Vec<WeightNormConvTranspose1d<B>>,
    pub resblocks: Vec<ResBlock1<B>>,
    pub conv_post: WeightNormConv1d<B>,
    pub cond_layer: Option<Conv1d<B>>,
    pub num_kernels: usize,
    pub num_upsamples: usize,
    pub inference_padding: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub cond_channels: usize,
    pub upsample_factor: usize,
}

impl<B: Backend> HifiganGenerator<B> {
    /// Synthesizes `[batch, out_channels, frames * upsample_factor]`.
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 3>>,
    ) -> Result<Tensor<B, 3>, HifiganError> {
        let [batch, channels, frames] = input.dims();
        if channels != self.in_channels {
            return Err(input_error(format!(
                "expected {} feature channels, got {channels}",
                self.in_channels
            )));
        }
        if frames == 0 {
            return Err(input_error("feature input must contain at least one frame"));
        }

        let mut output = self.conv_pre.forward(input);
        match (&self.cond_layer, conditioning) {
            (Some(cond_layer), Some(conditioning)) => {
                let [cond_batch, cond_channels, cond_frames] = conditioning.dims();
                if cond_batch != batch
                    || cond_channels != self.cond_channels
                    || cond_frames != frames
                {
                    return Err(input_error(format!(
                        "conditioning shape [{cond_batch}, {cond_channels}, {cond_frames}] must be [{batch}, {}, {frames}]",
                        self.cond_channels
                    )));
                }
                output = output + cond_layer.forward(conditioning);
            }
            (Some(_), None) => {
                return Err(input_error(format!(
                    "generator requires conditioning with {} channels",
                    self.cond_channels
                )));
            }
            (None, Some(_)) => {
                return Err(input_error(
                    "conditioning was supplied to an unconditioned generator",
                ));
            }
            (None, None) => {}
        }

        for stage in 0..self.num_upsamples {
            output = self.ups[stage].forward(leaky_relu(output, LRELU_SLOPE));

            let first = stage * self.num_kernels;
            let mut fused = self.resblocks[first].forward(output.clone());
            for block in &self.resblocks[first + 1..first + self.num_kernels] {
                fused = fused + block.forward(output.clone());
            }
            output = fused / self.num_kernels as f64;
        }

        output = leaky_relu(output, POST_LRELU_SLOPE);
        Ok(self.conv_post.forward(output).tanh())
    }

    /// Inference with replicate padding on the feature-frame axis.
    pub fn inference(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, HifiganError> {
        self.inference_with_conditioning(input, None)
    }

    /// Conditioned inference with the same replicate padding applied to both
    /// frame-aligned inputs.
    pub fn inference_with_conditioning(
        &self,
        input: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 3>>,
    ) -> Result<Tensor<B, 3>, HifiganError> {
        if input.dims()[2] == 0 {
            return Err(input_error("feature input must contain at least one frame"));
        }
        let padding = self.inference_padding;
        let input = input.pad([(padding, padding)], PadMode::Edge);
        let conditioning =
            conditioning.map(|tensor| tensor.pad([(padding, padding)], PadMode::Edge));
        self.forward(input, conditioning)
    }

    pub fn input_channels(&self) -> usize {
        self.in_channels
    }

    pub fn output_channels(&self) -> usize {
        self.out_channels
    }

    pub fn upsample_factor(&self) -> usize {
        self.upsample_factor
    }

    pub fn output_frames(&self, input_frames: usize) -> Option<usize> {
        input_frames.checked_mul(self.upsample_factor)
    }

    pub fn inference_output_frames(&self, input_frames: usize) -> Option<usize> {
        input_frames
            .checked_add(self.inference_padding.checked_mul(2)?)
            .and_then(|frames| frames.checked_mul(self.upsample_factor))
    }
}

fn required_value<'a>(
    object: &'a Map<String, Value>,
    key: &str,
) -> Result<&'a Value, HifiganError> {
    object
        .get(key)
        .ok_or_else(|| config_error(format!("missing `{key}`")))
}

fn required_usize(object: &Map<String, Value>, key: &str) -> Result<usize, HifiganError> {
    json_usize(required_value(object, key)?, key)
}

fn optional_usize(
    object: &Map<String, Value>,
    key: &str,
    default: usize,
) -> Result<usize, HifiganError> {
    object
        .get(key)
        .map_or(Ok(default), |value| json_usize(value, key))
}

fn json_usize(value: &Value, key: &str) -> Result<usize, HifiganError> {
    let value = value
        .as_u64()
        .ok_or_else(|| config_error(format!("`{key}` must be a non-negative integer")))?;
    usize::try_from(value).map_err(|_| config_error(format!("`{key}` does not fit usize")))
}

fn required_string(object: &Map<String, Value>, key: &str) -> Result<String, HifiganError> {
    required_value(object, key)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| config_error(format!("`{key}` must be a string")))
}

fn optional_bool(
    object: &Map<String, Value>,
    key: &str,
    default: bool,
) -> Result<bool, HifiganError> {
    object.get(key).map_or(Ok(default), |value| {
        value
            .as_bool()
            .ok_or_else(|| config_error(format!("`{key}` must be a boolean")))
    })
}

fn required_usizes(object: &Map<String, Value>, key: &str) -> Result<Vec<usize>, HifiganError> {
    required_value(object, key)?
        .as_array()
        .ok_or_else(|| config_error(format!("`{key}` must be an array")))?
        .iter()
        .enumerate()
        .map(|(index, value)| json_usize(value, &format!("{key}[{index}]")))
        .collect()
}

fn required_nested_usizes(
    object: &Map<String, Value>,
    key: &str,
) -> Result<Vec<Vec<usize>>, HifiganError> {
    required_value(object, key)?
        .as_array()
        .ok_or_else(|| config_error(format!("`{key}` must be an array")))?
        .iter()
        .enumerate()
        .map(|(outer, value)| {
            value
                .as_array()
                .ok_or_else(|| config_error(format!("`{key}[{outer}]` must be an array")))?
                .iter()
                .enumerate()
                .map(|(inner, value)| json_usize(value, &format!("{key}[{outer}][{inner}]")))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    fn tiny_config() -> HifiganGeneratorConfig {
        HifiganGeneratorConfig {
            in_channels: 4,
            out_channels: 1,
            resblock_type: "1".to_owned(),
            resblock_dilation_sizes: vec![vec![1, 2, 3], vec![1, 2, 3]],
            resblock_kernel_sizes: vec![3, 5],
            upsample_kernel_sizes: vec![4, 4],
            upsample_initial_channel: 16,
            upsample_factors: vec![2, 2],
            inference_padding: 1,
            cond_channels: 0,
            conv_pre_weight_norm: true,
            conv_post_weight_norm: true,
            conv_post_bias: true,
        }
    }

    #[test]
    fn forward_and_inference_shapes_match_upsampling_contract() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 11);
        let config = tiny_config();
        let model = config.init::<TestBackend>(&device).expect("valid config");

        let features = Tensor::<TestBackend, 3>::ones([2, 4, 3], &device);
        let output = model.forward(features.clone(), None).expect("forward");
        assert_eq!(output.dims(), [2, 1, 12]);
        assert_eq!(model.output_frames(3), Some(12));

        let output = model.inference(features).expect("inference");
        assert_eq!(output.dims(), [2, 1, 20]);
        assert_eq!(model.inference_output_frames(3), Some(20));
    }

    #[test]
    fn config_validation_rejects_incompatible_topologies() {
        let mut config = tiny_config();
        config.resblock_kernel_sizes[0] = 4;
        assert!(config.validate().is_err());

        let mut config = tiny_config();
        config.upsample_kernel_sizes[0] = 3;
        assert!(config.validate().is_err());

        let mut config = tiny_config();
        config.resblock_type = "2".to_owned();
        config.validate().expect("ResBlock2 topology");
        let model = config
            .init::<TestBackend>(&NdArrayDevice::Cpu)
            .expect("ResBlock2 model");
        assert_eq!(model.resblocks[0].convs.len(), 2);
    }

    #[test]
    fn initialized_model_output_is_deterministic_and_finite() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 19);
        let model = tiny_config()
            .init::<TestBackend>(&device)
            .expect("valid config");
        let features = Tensor::<TestBackend, 3>::ones([1, 4, 4], &device);

        let first = model
            .forward(features.clone(), None)
            .expect("first forward")
            .into_data()
            .to_vec::<f32>()
            .expect("f32 output");
        let second = model
            .forward(features, None)
            .expect("second forward")
            .into_data()
            .to_vec::<f32>()
            .expect("f32 output");

        assert_eq!(first, second);
        assert!(!first.is_empty());
        assert!(first.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn normalized_json_config_applies_coqui_defaults() {
        let value = serde_json::json!({
            "in_channels": 80,
            "out_channels": 1,
            "resblock_type": "1",
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "upsample_initial_channel": 512,
            "upsample_factors": [8, 8, 2, 2]
        });

        let config =
            HifiganGeneratorConfig::from_json_value(&value).expect("Coqui generator config");
        assert_eq!(config.inference_padding, 5);
        assert_eq!(config.cond_channels, 0);
        assert!(config.conv_pre_weight_norm);
        assert!(config.conv_post_weight_norm);
        assert!(config.conv_post_bias);
        assert_eq!(config.upsample_factor(), 256);
    }
}
