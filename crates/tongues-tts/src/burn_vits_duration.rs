//! Burn-native stochastic duration prediction for VITS-family inference.
//!
//! The public boundary accepts text-prior features, a sequence mask, optional
//! speaker conditioning, and either caller-provided or seeded Gaussian noise.
//! It implements only the reverse (synthesis) graph. Posterior duration
//! conditioning is training-only and is deliberately absent.

use std::fmt;
use std::path::Path;

use burn::module::{Initializer, Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::PaddingConfig1d;
use burn::tensor::activation::{gelu, softmax, softplus};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};
use burn_store::{ModuleSnapshot, PytorchStore};

const LAYER_NORM_EPSILON: f64 = 1e-5;
const MIN_BIN_WIDTH: f64 = 1e-3;
const MIN_BIN_HEIGHT: f64 = 1e-3;
const MIN_DERIVATIVE: f64 = 1e-3;

/// Checkpoint subtrees used only by variational duration training.
pub const TRAINING_ONLY_DURATION_TENSOR_PREFIXES: &[&str] =
    &["post_pre.", "post_convs.", "post_proj.", "post_flows."];

#[derive(Debug, Clone, PartialEq)]
pub enum StochasticDurationError {
    InvalidConfig(String),
    InvalidInput(String),
    Checkpoint(String),
}

impl fmt::Display for StochasticDurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid stochastic duration config: {message}")
            }
            Self::InvalidInput(message) => {
                write!(formatter, "invalid stochastic duration input: {message}")
            }
            Self::Checkpoint(message) => {
                write!(
                    formatter,
                    "unable to load stochastic duration checkpoint: {message}"
                )
            }
        }
    }
}

impl std::error::Error for StochasticDurationError {}

/// Inference topology for a VITS-family stochastic duration predictor.
#[derive(Debug, Clone, PartialEq)]
pub struct StochasticDurationConfig {
    pub input_channels: usize,
    pub hidden_channels: usize,
    pub kernel_size: usize,
    pub num_flows: usize,
    pub conditioning_channels: usize,
    pub num_bins: usize,
    pub tail_bound: f64,
}

impl StochasticDurationConfig {
    pub fn new(input_channels: usize, hidden_channels: usize, kernel_size: usize) -> Self {
        Self {
            input_channels,
            hidden_channels,
            kernel_size,
            num_flows: 4,
            conditioning_channels: 0,
            num_bins: 10,
            tail_bound: 5.0,
        }
    }

    pub fn validate(&self) -> Result<(), StochasticDurationError> {
        if self.input_channels == 0 {
            return Err(config_error("input_channels must be positive"));
        }
        if self.hidden_channels == 0 {
            return Err(config_error("hidden_channels must be positive"));
        }
        if self.kernel_size == 0 || self.kernel_size.is_multiple_of(2) {
            return Err(config_error("kernel_size must be positive and odd"));
        }
        if self.num_flows == 0 {
            return Err(config_error("num_flows must be positive"));
        }
        if self.num_bins < 2 {
            return Err(config_error("num_bins must be at least two"));
        }
        if self.tail_bound <= 0.0 || !self.tail_bound.is_finite() {
            return Err(config_error("tail_bound must be finite and positive"));
        }
        if MIN_BIN_WIDTH * self.num_bins as f64 > 1.0 {
            return Err(config_error("minimum bin width exceeds the spline domain"));
        }
        if MIN_BIN_HEIGHT * self.num_bins as f64 > 1.0 {
            return Err(config_error("minimum bin height exceeds the spline range"));
        }
        Ok(())
    }

    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<StochasticDurationPredictor<B>, StochasticDurationError> {
        self.validate()?;

        let pre = pointwise_conv(self.input_channels, self.hidden_channels, device);
        let convs =
            DilatedDepthSeparableConv::init(self.hidden_channels, self.kernel_size, 3, device);
        let proj = pointwise_conv(self.hidden_channels, self.hidden_channels, device);
        let affine = ElementwiseAffine::init(2, device);
        let spline_flows = (0..self.num_flows)
            .map(|_| {
                ConvFlow::init(
                    2,
                    self.hidden_channels,
                    self.kernel_size,
                    3,
                    self.num_bins,
                    self.tail_bound,
                    device,
                )
            })
            .collect();
        let cond = (self.conditioning_channels > 0)
            .then(|| pointwise_conv(self.conditioning_channels, self.hidden_channels, device));

        Ok(StochasticDurationPredictor {
            pre,
            convs,
            proj,
            affine,
            spline_flows,
            cond,
            input_channels: self.input_channels,
            hidden_channels: self.hidden_channels,
            conditioning_channels: self.conditioning_channels,
        })
    }

    pub fn load_checkpoint<B: Backend>(
        &self,
        checkpoint_path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<StochasticDurationPredictor<B>, StochasticDurationError> {
        self.init(device)?.load_checkpoint(checkpoint_path)
    }
}

fn config_error(message: impl Into<String>) -> StochasticDurationError {
    StochasticDurationError::InvalidConfig(message.into())
}

fn input_error(message: impl Into<String>) -> StochasticDurationError {
    StochasticDurationError::InvalidInput(message.into())
}

fn pointwise_conv<B: Backend>(
    channels_in: usize,
    channels_out: usize,
    device: &B::Device,
) -> Conv1d<B> {
    Conv1dConfig::new(channels_in, channels_out, 1).init(device)
}

/// Channel-wise layer normalization for tensors shaped `[batch, channels, time]`.
///
#[derive(Module, Debug)]
struct ChannelLayerNorm<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
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
        let [_, channels, _] = input.dims();
        let mean = input.clone().mean_dim(1);
        let centered = input - mean;
        let variance = centered.clone().powf_scalar(2.0).mean_dim(1);
        centered / (variance + self.epsilon).sqrt() * self.gamma.val().reshape([1, channels, 1])
            + self.beta.val().reshape([1, channels, 1])
    }
}

/// Dilated depth-separable convolution stack used by the duration flows.
#[derive(Module, Debug)]
pub struct DilatedDepthSeparableConv<B: Backend> {
    convs_sep: Vec<Conv1d<B>>,
    convs_1x1: Vec<Conv1d<B>>,
    norms_1: Vec<ChannelLayerNorm<B>>,
    norms_2: Vec<ChannelLayerNorm<B>>,
}

impl<B: Backend> DilatedDepthSeparableConv<B> {
    pub fn init(
        channels: usize,
        kernel_size: usize,
        num_layers: usize,
        device: &B::Device,
    ) -> Self {
        let mut convs_sep = Vec::with_capacity(num_layers);
        let mut convs_1x1 = Vec::with_capacity(num_layers);
        let mut norms_1 = Vec::with_capacity(num_layers);
        let mut norms_2 = Vec::with_capacity(num_layers);

        for index in 0..num_layers {
            let dilation = kernel_size.pow(index as u32);
            let padding = (kernel_size * dilation - dilation) / 2;
            convs_sep.push(
                Conv1dConfig::new(channels, channels, kernel_size)
                    .with_dilation(dilation)
                    .with_groups(channels)
                    .with_padding(PaddingConfig1d::Explicit(padding, padding))
                    .init(device),
            );
            convs_1x1.push(pointwise_conv(channels, channels, device));
            norms_1.push(ChannelLayerNorm::init(channels, device));
            norms_2.push(ChannelLayerNorm::init(channels, device));
        }

        Self {
            convs_sep,
            convs_1x1,
            norms_1,
            norms_2,
        }
    }

    /// Inference forward pass. Dropout from the training graph is intentionally
    /// inactive, matching an evaluated reference module.
    pub fn forward(
        &self,
        mut input: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        if let Some(conditioning) = conditioning {
            input = input + conditioning;
        }
        for index in 0..self.convs_sep.len() {
            let residual = input.clone();
            let value = self.convs_sep[index].forward(input * mask.clone());
            let value = gelu(self.norms_1[index].forward(value));
            let value = self.convs_1x1[index].forward(value);
            let value = gelu(self.norms_2[index].forward(value));
            input = residual + value;
        }
        input * mask
    }
}

/// Elementwise affine flow at the base of the duration flow stack.
#[derive(Module, Debug)]
pub struct ElementwiseAffine<B: Backend> {
    translation: Param<Tensor<B, 2>>,
    log_scale: Param<Tensor<B, 2>>,
}

impl<B: Backend> ElementwiseAffine<B> {
    pub fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            translation: Initializer::Zeros.init([channels, 1], device),
            log_scale: Initializer::Zeros.init([channels, 1], device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, channels, _] = input.dims();
        (input * self.log_scale.val().reshape([1, channels, 1]).exp()
            + self.translation.val().reshape([1, channels, 1]))
            * mask
    }

    pub fn reverse(&self, input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, channels, _] = input.dims();
        (input - self.translation.val().reshape([1, channels, 1]))
            * (-self.log_scale.val().reshape([1, channels, 1])).exp()
            * mask
    }
}

/// Dilated convolutional rational-quadratic spline coupling flow.
#[derive(Module, Debug)]
pub struct ConvFlow<B: Backend> {
    pre: Conv1d<B>,
    convs: DilatedDepthSeparableConv<B>,
    proj: Conv1d<B>,
    half_channels: usize,
    hidden_channels: usize,
    num_bins: usize,
    tail_bound: f64,
}

impl<B: Backend> ConvFlow<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn init(
        input_channels: usize,
        hidden_channels: usize,
        kernel_size: usize,
        num_layers: usize,
        num_bins: usize,
        tail_bound: f64,
        device: &B::Device,
    ) -> Self {
        assert!(
            input_channels > 0 && input_channels.is_multiple_of(2),
            "ConvFlow input channels must be positive and even"
        );
        let half_channels = input_channels / 2;
        let output_channels = half_channels * (num_bins * 3 - 1);
        Self {
            pre: pointwise_conv(half_channels, hidden_channels, device),
            convs: DilatedDepthSeparableConv::init(
                hidden_channels,
                kernel_size,
                num_layers,
                device,
            ),
            proj: Conv1dConfig::new(hidden_channels, output_channels, 1)
                .with_initializer(Initializer::Zeros)
                .init(device),
            half_channels,
            hidden_channels,
            num_bins,
            tail_bound,
        }
    }

    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        conditioning: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.transform(input, mask, conditioning, false)
    }

    pub fn reverse(
        &self,
        input: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        conditioning: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.transform(input, mask, conditioning, true)
    }

    fn transform(
        &self,
        input: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        conditioning: Tensor<B, 3>,
        inverse: bool,
    ) -> Tensor<B, 3> {
        let [batch, channels, frames] = input.dims();
        assert_eq!(channels, self.half_channels * 2);
        let x0 = input
            .clone()
            .slice([0..batch, 0..self.half_channels, 0..frames]);
        let x1 = input.slice([
            0..batch,
            self.half_channels..self.half_channels * 2,
            0..frames,
        ]);

        let hidden = self.pre.forward(x0.clone());
        let hidden = self.convs.forward(hidden, mask.clone(), Some(conditioning));
        let parameters = self.proj.forward(hidden) * mask.clone();
        let parameters = parameters
            .reshape([batch, self.half_channels, self.num_bins * 3 - 1, frames])
            .swap_dims(2, 3);

        let scale = (self.hidden_channels as f64).sqrt();
        let widths = parameters.clone().slice([
            0..batch,
            0..self.half_channels,
            0..frames,
            0..self.num_bins,
        ]) / scale;
        let heights = parameters.clone().slice([
            0..batch,
            0..self.half_channels,
            0..frames,
            self.num_bins..self.num_bins * 2,
        ]) / scale;
        let derivatives = parameters.slice([
            0..batch,
            0..self.half_channels,
            0..frames,
            self.num_bins * 2..self.num_bins * 3 - 1,
        ]);

        let (x1, _) = unconstrained_rational_quadratic_spline(
            x1,
            widths,
            heights,
            derivatives,
            inverse,
            self.tail_bound,
        );
        Tensor::cat(vec![x0, x1], 1) * mask
    }
}

/// Inference-only stochastic duration predictor.
#[derive(Module, Debug)]
pub struct StochasticDurationPredictor<B: Backend> {
    pre: Conv1d<B>,
    convs: DilatedDepthSeparableConv<B>,
    proj: Conv1d<B>,
    affine: ElementwiseAffine<B>,
    spline_flows: Vec<ConvFlow<B>>,
    cond: Option<Conv1d<B>>,
    input_channels: usize,
    hidden_channels: usize,
    conditioning_channels: usize,
}

impl<B: Backend> StochasticDurationPredictor<B> {
    /// Strictly loads the inference subtree from a full model checkpoint.
    ///
    /// The posterior `post_*` tensors are filtered as training-only. All prior
    /// flows are loaded, including the first spline flow that the published
    /// reverse algorithm deliberately bypasses.
    pub fn load_checkpoint(
        mut self,
        checkpoint_path: impl AsRef<Path>,
    ) -> Result<Self, StochasticDurationError> {
        let mut store = PytorchStore::from_file(checkpoint_path.as_ref())
            .with_top_level_key("model")
            .with_key_remapping(r"^duration_predictor\.", "")
            .with_key_remapping(r"^flows\.0\.", "affine.")
            .with_predicate(duration_inference_tensor)
            .map_indices_contiguous(false)
            .allow_partial(true)
            .skip_enum_variants(true);
        for index in 0..self.spline_flows.len() {
            store = store.with_key_remapping(
                format!(r"^flows\.{}\.", index + 1),
                format!("spline_flows.{index}."),
            );
        }

        let result = self
            .load_from(&mut store)
            .map_err(|error| StochasticDurationError::Checkpoint(error.to_string()))?;
        let mut missing = result
            .missing
            .iter()
            .map(|(path, _)| path.as_str())
            .collect::<Vec<_>>();
        missing.sort_unstable();
        let mut unused = result
            .unused
            .iter()
            .filter(|path| duration_inference_tensor(path, ""))
            .cloned()
            .collect::<Vec<_>>();
        unused.sort_unstable();
        if !missing.is_empty() || !result.errors.is_empty() || !unused.is_empty() {
            return Err(StochasticDurationError::Checkpoint(format!(
                "duration inference subtree does not exactly match the Burn module: missing [{}], {} load errors, unused [{}]",
                missing.join(", "),
                result.errors.len(),
                unused.join(", ")
            )));
        }
        Ok(self)
    }

    /// Reverse inference using caller-provided standard-normal noise.
    pub fn reverse_with_noise(
        &self,
        input: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 3>>,
        noise: Tensor<B, 3>,
        noise_scale: f64,
    ) -> Result<Tensor<B, 3>, StochasticDurationError> {
        if !noise_scale.is_finite() || noise_scale < 0.0 {
            return Err(input_error("noise_scale must be finite and non-negative"));
        }
        let [batch, input_channels, frames] = input.dims();
        if input_channels != self.input_channels {
            return Err(input_error(format!(
                "input has {input_channels} channels; expected {}",
                self.input_channels
            )));
        }
        if mask.dims() != [batch, 1, frames] {
            return Err(input_error(format!(
                "mask shape {:?}; expected [{batch}, 1, {frames}]",
                mask.dims()
            )));
        }
        if noise.dims() != [batch, 2, frames] {
            return Err(input_error(format!(
                "noise shape {:?}; expected [{batch}, 2, {frames}]",
                noise.dims()
            )));
        }

        let mut hidden = self.pre.forward(input);
        let speaker = self.resolve_conditioning(conditioning, batch, frames)?;
        if let Some(speaker) = speaker {
            hidden = hidden + speaker;
        }
        hidden = self.convs.forward(hidden, mask.clone(), None);
        hidden = self.proj.forward(hidden) * mask.clone();

        // Reverse the prior stack exactly: reverse all spline flows except the
        // first ("useless vflow" in the reference), then the base affine.
        let mut latent = noise * noise_scale;
        for index in (1..self.spline_flows.len()).rev() {
            latent = latent.flip([1]);
            latent = self.spline_flows[index].reverse(latent, mask.clone(), hidden.clone());
        }
        latent = latent.flip([1]);
        latent = self.affine.reverse(latent, mask.clone());

        Ok(latent.slice([0..batch, 0..1, 0..frames]) * mask)
    }

    /// Reverse inference with deterministic backend-seeded Gaussian noise.
    pub fn reverse_seeded(
        &self,
        input: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 3>>,
        noise_scale: f64,
        seed: u64,
    ) -> Result<Tensor<B, 3>, StochasticDurationError> {
        let [batch, _, frames] = input.dims();
        let device = input.device();
        B::seed(&device, seed);
        let noise = Tensor::random([batch, 2, frames], Distribution::Normal(0.0, 1.0), &device);
        self.reverse_with_noise(input, mask, conditioning, noise, noise_scale)
    }

    /// Reverse inference with backend-generated Gaussian noise.
    pub fn reverse(
        &self,
        input: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 3>>,
        noise_scale: f64,
    ) -> Result<Tensor<B, 3>, StochasticDurationError> {
        let [batch, _, frames] = input.dims();
        let device = input.device();
        let noise = Tensor::random([batch, 2, frames], Distribution::Normal(0.0, 1.0), &device);
        self.reverse_with_noise(input, mask, conditioning, noise, noise_scale)
    }

    fn resolve_conditioning(
        &self,
        conditioning: Option<Tensor<B, 3>>,
        batch: usize,
        frames: usize,
    ) -> Result<Option<Tensor<B, 3>>, StochasticDurationError> {
        match (&self.cond, conditioning) {
            (None, None) => Ok(None),
            (None, Some(_)) => Err(input_error(
                "speaker conditioning was supplied to an unconditioned predictor",
            )),
            (Some(_), None) => Err(input_error(
                "speaker conditioning is required by this predictor",
            )),
            (Some(projection), Some(conditioning)) => {
                let [cond_batch, cond_channels, cond_frames] = conditioning.dims();
                if cond_batch != batch || cond_channels != self.conditioning_channels {
                    return Err(input_error(format!(
                        "conditioning shape {:?}; expected [{batch}, {}, 1 or {frames}]",
                        conditioning.dims(),
                        self.conditioning_channels
                    )));
                }
                if cond_frames != 1 && cond_frames != frames {
                    return Err(input_error(format!(
                        "conditioning has {cond_frames} frames; expected 1 or {frames}"
                    )));
                }
                let projected = projection.forward(conditioning);
                Ok(Some(if cond_frames == 1 {
                    projected.repeat_dim(2, frames)
                } else {
                    projected
                }))
            }
        }
    }

    pub fn hidden_channels(&self) -> usize {
        self.hidden_channels
    }
}

fn duration_inference_tensor(path: &str, _container: &str) -> bool {
    [
        "pre.",
        "convs.",
        "proj.",
        "affine.",
        "spline_flows.",
        "cond.",
    ]
    .iter()
    .any(|prefix| path.starts_with(prefix))
}

fn unconstrained_rational_quadratic_spline<B: Backend>(
    inputs: Tensor<B, 3>,
    unnormalized_widths: Tensor<B, 4>,
    unnormalized_heights: Tensor<B, 4>,
    unnormalized_derivatives: Tensor<B, 4>,
    inverse: bool,
    tail_bound: f64,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let [batch, channels, frames] = inputs.dims();
    let device = inputs.device();
    let boundary_value = ((1.0 - MIN_DERIVATIVE).exp() - 1.0).ln();
    let boundary = Tensor::full([batch, channels, frames, 1], boundary_value, &device);
    let derivatives = Tensor::cat(
        vec![boundary.clone(), unnormalized_derivatives, boundary],
        3,
    );

    let clipped = inputs.clone().clamp(-tail_bound, tail_bound);
    let (inside_outputs, inside_logdet) = rational_quadratic_spline(
        clipped,
        unnormalized_widths,
        unnormalized_heights,
        derivatives,
        inverse,
        -tail_bound,
        tail_bound,
        -tail_bound,
        tail_bound,
    );
    let inside = inputs
        .clone()
        .greater_equal_elem(-tail_bound)
        .bool_and(inputs.clone().lower_equal_elem(tail_bound));
    let outputs = inputs.clone().mask_where(inside.clone(), inside_outputs);
    let logdet = Tensor::zeros_like(&inputs).mask_where(inside, inside_logdet);
    (outputs, logdet)
}

#[allow(clippy::too_many_arguments)]
fn rational_quadratic_spline<B: Backend>(
    inputs: Tensor<B, 3>,
    unnormalized_widths: Tensor<B, 4>,
    unnormalized_heights: Tensor<B, 4>,
    unnormalized_derivatives: Tensor<B, 4>,
    inverse: bool,
    left: f64,
    right: f64,
    bottom: f64,
    top: f64,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let [batch, channels, frames, num_bins] = unnormalized_widths.dims();
    let device = inputs.device();

    let widths =
        softmax(unnormalized_widths, 3) * (1.0 - MIN_BIN_WIDTH * num_bins as f64) + MIN_BIN_WIDTH;
    let cumulative_widths = widths.clone().cumsum(3);
    let cumulative_widths = Tensor::cat(
        vec![
            Tensor::zeros([batch, channels, frames, 1], &device),
            cumulative_widths,
        ],
        3,
    ) * (right - left)
        + left;
    let widths =
        cumulative_widths
            .clone()
            .slice([0..batch, 0..channels, 0..frames, 1..num_bins + 1])
            - cumulative_widths
                .clone()
                .slice([0..batch, 0..channels, 0..frames, 0..num_bins]);

    let heights = softmax(unnormalized_heights, 3) * (1.0 - MIN_BIN_HEIGHT * num_bins as f64)
        + MIN_BIN_HEIGHT;
    let cumulative_heights = heights.clone().cumsum(3);
    let cumulative_heights = Tensor::cat(
        vec![
            Tensor::zeros([batch, channels, frames, 1], &device),
            cumulative_heights,
        ],
        3,
    ) * (top - bottom)
        + bottom;
    let heights =
        cumulative_heights
            .clone()
            .slice([0..batch, 0..channels, 0..frames, 1..num_bins + 1])
            - cumulative_heights
                .clone()
                .slice([0..batch, 0..channels, 0..frames, 0..num_bins]);

    let derivatives = softplus(unnormalized_derivatives, 1.0) + MIN_DERIVATIVE;
    let search_locations = if inverse {
        cumulative_heights.clone()
    } else {
        cumulative_widths.clone()
    };
    let bin_indices = inputs
        .clone()
        .unsqueeze_dim::<4>(3)
        .greater_equal(search_locations)
        .int()
        .sum_dim(3)
        .sub_scalar(1)
        .clamp(0, num_bins as i64 - 1);

    let input_cumulative_widths = cumulative_widths
        .gather(3, bin_indices.clone())
        .reshape([batch, channels, frames]);
    let input_bin_widths = widths
        .clone()
        .gather(3, bin_indices.clone())
        .reshape([batch, channels, frames]);
    let input_cumulative_heights = cumulative_heights
        .gather(3, bin_indices.clone())
        .reshape([batch, channels, frames]);
    let input_heights = heights
        .clone()
        .gather(3, bin_indices.clone())
        .reshape([batch, channels, frames]);
    let delta = heights / widths;
    let input_delta = delta
        .gather(3, bin_indices.clone())
        .reshape([batch, channels, frames]);
    let input_derivatives = derivatives
        .clone()
        .gather(3, bin_indices.clone())
        .reshape([batch, channels, frames]);
    let input_derivatives_plus_one = derivatives
        .slice([0..batch, 0..channels, 0..frames, 1..num_bins + 1])
        .gather(3, bin_indices)
        .reshape([batch, channels, frames]);

    if inverse {
        let offset = inputs - input_cumulative_heights.clone();
        let derivative_sum = input_derivatives.clone() + input_derivatives_plus_one.clone()
            - input_delta.clone() * 2.0;
        let a = offset.clone() * derivative_sum.clone()
            + input_heights.clone() * (input_delta.clone() - input_derivatives.clone());
        let b = input_heights.clone() * input_derivatives.clone() - offset.clone() * derivative_sum;
        let c = -input_delta.clone() * offset;
        let discriminant =
            (b.clone().powf_scalar(2.0) - a.clone() * c.clone() * 4.0).clamp_min(0.0);
        let root = c * 2.0 / (-b - discriminant.sqrt());
        let outputs = root.clone() * input_bin_widths + input_cumulative_widths;
        let theta_one_minus_theta = root.clone() * (Tensor::ones_like(&root) - root.clone());
        let denominator = input_delta.clone()
            + (input_derivatives.clone() + input_derivatives_plus_one.clone()
                - input_delta.clone() * 2.0)
                * theta_one_minus_theta.clone();
        let derivative_numerator = input_delta.clone().powf_scalar(2.0)
            * (input_derivatives_plus_one * root.clone().powf_scalar(2.0)
                + input_delta * theta_one_minus_theta * 2.0
                + input_derivatives * (Tensor::ones_like(&root) - root).powf_scalar(2.0));
        let logdet = -(derivative_numerator.log() - denominator.log() * 2.0);
        (outputs, logdet)
    } else {
        let theta = (inputs - input_cumulative_widths) / input_bin_widths;
        let theta_one_minus_theta = theta.clone() * (Tensor::ones_like(&theta) - theta.clone());
        let denominator = input_delta.clone()
            + (input_derivatives.clone() + input_derivatives_plus_one.clone()
                - input_delta.clone() * 2.0)
                * theta_one_minus_theta.clone();
        let numerator = input_heights
            * (input_delta.clone() * theta.clone().powf_scalar(2.0)
                + input_derivatives.clone() * theta_one_minus_theta);
        let outputs = input_cumulative_heights + numerator / denominator.clone();
        let derivative_numerator = input_delta.clone().powf_scalar(2.0)
            * (input_derivatives_plus_one * theta.clone().powf_scalar(2.0)
                + input_delta * theta.clone() * (Tensor::ones_like(&theta) - theta.clone()) * 2.0
                + input_derivatives * (Tensor::ones_like(&theta) - theta).powf_scalar(2.0));
        let logdet = derivative_numerator.log() - denominator.log() * 2.0;
        (outputs, logdet)
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::TensorData;

    use super::*;

    type TestBackend = NdArray<f32>;

    fn values<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
        tensor
            .into_data()
            .into_vec::<f32>()
            .expect("f32 tensor data")
    }

    fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (index, (left, right)) in left.iter().zip(right).enumerate() {
            assert!(
                (left - right).abs() <= tolerance,
                "value {index}: {left} != {right} within {tolerance}"
            );
        }
    }

    #[test]
    fn affine_flow_round_trips_and_applies_mask() {
        let device = NdArrayDevice::Cpu;
        let flow = ElementwiseAffine::<TestBackend>::init(2, &device);
        let input = Tensor::from_floats([[[0.25, -0.5, 3.0], [2.0, 1.0, -4.0]]], &device);
        let mask = Tensor::from_floats([[[1.0, 1.0, 0.0]]], &device);

        let output = flow.forward(input.clone(), mask.clone());
        let restored = flow.reverse(output, mask);

        assert_close(&values(restored), &[0.25, -0.5, 0.0, 2.0, 1.0, 0.0], 1e-6);
    }

    #[test]
    fn rational_quadratic_spline_round_trips_and_has_linear_tails() {
        let device = NdArrayDevice::Cpu;
        let inputs = Tensor::from_floats([[[-7.0, -2.25, -0.1, 1.75, 7.0]]], &device);
        let widths = Tensor::from_data(
            TensorData::new(
                vec![
                    0.3f32, -0.2, 0.7, -0.4, 0.1, 0.2, -0.8, 0.4, 0.6, -0.3, 0.2, 0.1, -0.5, 0.9,
                    -0.1, 0.4, 0.3, -0.7, 0.8, -0.2,
                ],
                [1, 1, 5, 4],
            ),
            &device,
        );
        let heights = widths.clone().flip([3]);
        let derivatives = Tensor::zeros([1, 1, 5, 3], &device);

        let (mapped, _) = unconstrained_rational_quadratic_spline(
            inputs.clone(),
            widths.clone(),
            heights.clone(),
            derivatives.clone(),
            false,
            5.0,
        );
        let (restored, _) = unconstrained_rational_quadratic_spline(
            mapped.clone(),
            widths,
            heights,
            derivatives,
            true,
            5.0,
        );

        assert_close(&values(restored), &values(inputs), 2e-4);
        let mapped = values(mapped);
        assert_eq!(mapped[0], -7.0);
        assert_eq!(mapped[4], 7.0);
    }

    #[test]
    fn controlled_noise_is_repeatable_and_masked() {
        let device = NdArrayDevice::Cpu;
        let config = StochasticDurationConfig {
            input_channels: 4,
            hidden_channels: 4,
            kernel_size: 3,
            num_flows: 2,
            conditioning_channels: 0,
            num_bins: 4,
            tail_bound: 5.0,
        };
        TestBackend::seed(&device, 13);
        let predictor = config.init::<TestBackend>(&device).expect("predictor");
        let input = Tensor::ones([1, 4, 4], &device);
        let mask = Tensor::from_floats([[[1.0, 1.0, 1.0, 0.0]]], &device);
        let noise = Tensor::from_floats([[[0.2, -0.4, 0.7, 8.0], [0.9, -0.1, 0.5, -3.0]]], &device);

        let first = predictor
            .reverse_with_noise(input.clone(), mask.clone(), None, noise.clone(), 0.8)
            .expect("first");
        let second = predictor
            .reverse_with_noise(input, mask, None, noise, 0.8)
            .expect("second");

        assert_eq!(first.dims(), [1, 1, 4]);
        assert_close(&values(first.clone()), &values(second), 1e-6);
        assert_eq!(values(first)[3], 0.0);
    }

    #[test]
    fn seeded_noise_and_speaker_conditioning_are_supported() {
        let device = NdArrayDevice::Cpu;
        let config = StochasticDurationConfig {
            input_channels: 4,
            hidden_channels: 4,
            kernel_size: 3,
            num_flows: 2,
            conditioning_channels: 3,
            num_bins: 4,
            tail_bound: 5.0,
        };
        TestBackend::seed(&device, 29);
        let predictor = config.init::<TestBackend>(&device).expect("predictor");
        let input = Tensor::ones([2, 4, 3], &device);
        let mask = Tensor::ones([2, 1, 3], &device);
        let speaker = Tensor::ones([2, 3, 1], &device);
        let noise = Tensor::ones([2, 2, 3], &device);

        let first = predictor
            .reverse_with_noise(
                input.clone(),
                mask.clone(),
                Some(speaker.clone()),
                noise.clone(),
                0.8,
            )
            .expect("first");
        let second = predictor
            .reverse_with_noise(
                input.clone(),
                mask.clone(),
                Some(speaker.clone()),
                noise,
                0.8,
            )
            .expect("second");
        let seeded = predictor
            .reverse_seeded(input, mask, Some(speaker), 0.8, 71)
            .expect("seeded");

        assert_eq!(first.dims(), [2, 1, 3]);
        assert_close(&values(first), &values(second), 1e-6);
        assert!(values(seeded).iter().all(|value| value.is_finite()));
    }

    #[test]
    fn published_checkpoint_loads_and_runs_when_provided() {
        let checkpoint = std::env::var_os("TONGUES_TEST_VITS_CHECKPOINT")
            .or_else(|| std::env::var_os("TONGUES_TEST_COQUI_VITS_CHECKPOINT"));
        let Some(checkpoint) = checkpoint else {
            return;
        };
        let device = NdArrayDevice::Cpu;
        let config = StochasticDurationConfig {
            input_channels: 192,
            hidden_channels: 192,
            kernel_size: 3,
            num_flows: 4,
            conditioning_channels: 256,
            num_bins: 10,
            tail_bound: 5.0,
        };
        let predictor = config
            .load_checkpoint::<TestBackend>(checkpoint, &device)
            .expect("strict duration inference subtree load");
        let input = Tensor::zeros([1, 192, 3], &device);
        let mask = Tensor::ones([1, 1, 3], &device);
        let speaker = Tensor::zeros([1, 256, 1], &device);
        let noise = Tensor::zeros([1, 2, 3], &device);

        let output = predictor
            .reverse_with_noise(input, mask, Some(speaker), noise, 0.8)
            .expect("published duration forward");

        assert_eq!(output.dims(), [1, 1, 3]);
        assert!(values(output).iter().all(|value| value.is_finite()));
    }
}
