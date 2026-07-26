//! Burn-native monotonic alignment and residual coupling flow components.
//!
//! These operations form the architecture-neutral boundary between predicted
//! token durations, frame-aligned prior statistics, and a VITS-family latent
//! decoder. The flow field names deliberately match the established
//! `flow.flows.N.*` checkpoint hierarchy after the outer `flow.` prefix is
//! removed by [`ResidualCouplingFlow::load_checkpoint`].
//!
//! Source provenance: `audit-required`. This file was introduced by commit
//! `8e3a9c6`, whose message combines import, adaptation, and reverse
//! engineering without identifying the exact relationship. See
//! `docs/provenance.md` before changing its license or provenance notice.

use std::fmt;
use std::path::Path;

use burn::module::{Initializer, Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::PaddingConfig1d;
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{ElementConversion, Int, Tensor};

const PYTORCH_CONV_GAIN: f64 = 0.577_350_269_189_625_8;

#[derive(Debug)]
pub enum VitsFlowError {
    InvalidTopology(String),
    InvalidInput(String),
    Checkpoint(String),
}

impl fmt::Display for VitsFlowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTopology(message) => {
                write!(
                    formatter,
                    "invalid residual coupling flow topology: {message}"
                )
            }
            Self::InvalidInput(message) => {
                write!(formatter, "invalid residual coupling flow input: {message}")
            }
            Self::Checkpoint(message) => {
                write!(
                    formatter,
                    "unable to load residual coupling flow checkpoint: {message}"
                )
            }
        }
    }
}

impl std::error::Error for VitsFlowError {}

fn topology_error(message: impl Into<String>) -> VitsFlowError {
    VitsFlowError::InvalidTopology(message.into())
}

fn input_error(message: impl Into<String>) -> VitsFlowError {
    VitsFlowError::InvalidInput(message.into())
}

/// Rounded token durations and the maximum frame count in their batch.
#[derive(Debug)]
pub struct CeiledDurations<B: Backend> {
    /// Integral-valued durations in `[batch, tokens]` layout.
    pub values: Tensor<B, 2>,
    /// Maximum sum of token durations across the batch.
    pub output_frames: usize,
}

/// Scales, masks, and rounds predicted positive durations for path generation.
///
/// `durations` and `token_mask` use `[batch, 1, tokens]` layout. Padding is
/// forced to zero before the ceiling operation. At least one output frame is
/// required, and `max_output_frames` bounds allocations driven by model output.
pub fn ceil_durations<B: Backend>(
    durations: Tensor<B, 3>,
    token_mask: Tensor<B, 3>,
    length_scale: f64,
    max_output_frames: usize,
) -> Result<CeiledDurations<B>, VitsFlowError> {
    let [batch, channels, tokens] = durations.dims();
    if batch == 0 || channels != 1 || tokens == 0 {
        return Err(input_error(format!(
            "durations must have non-empty [batch, 1, tokens] dimensions, got [{batch}, {channels}, {tokens}]"
        )));
    }
    if token_mask.dims() != [batch, 1, tokens] {
        return Err(input_error(format!(
            "token mask dimensions {:?} do not match durations [{batch}, 1, {tokens}]",
            token_mask.dims()
        )));
    }
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(input_error(
            "length_scale must be finite and strictly positive",
        ));
    }
    if max_output_frames == 0 {
        return Err(input_error("max_output_frames must be positive"));
    }
    let device = durations.device();
    let scaled = (durations * token_mask * length_scale).reshape([batch, tokens]);
    // Shape decisions require a host value. Read the small duration vector once
    // and derive validation plus every per-batch frame count from that single
    // synchronization, instead of issuing separate min/sum/max scalar reads.
    let mut host_values = scaled
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| input_error(format!("rounded durations are not f32: {error}")))?;
    if host_values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(input_error(
            "predicted durations must be finite and non-negative",
        ));
    }
    for value in &mut host_values {
        *value = value.ceil();
    }
    let output_frames = host_values
        .chunks(tokens)
        .map(|row| row.iter().map(|value| *value as usize).sum::<usize>())
        .max()
        .unwrap_or(0);
    if output_frames == 0 {
        return Err(input_error("rounded durations produce zero output frames"));
    }
    if output_frames > max_output_frames {
        return Err(input_error(format!(
            "rounded durations request {output_frames} frames, exceeding configured limit {max_output_frames}"
        )));
    }

    Ok(CeiledDurations {
        values: Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(host_values, [batch, tokens]),
            &device,
        ),
        output_frames,
    })
}

/// Generates a hard monotonic token-to-frame path.
///
/// `durations` has shape `[batch, tokens]`; `mask` has shape
/// `[batch, tokens, frames]`. Speaker or model-specific token meanings do not
/// enter this operation. A zero-duration token receives no frames.
pub fn generate_path<B: Backend>(
    durations: Tensor<B, 2>,
    mask: Tensor<B, 3>,
) -> Result<Tensor<B, 3>, VitsFlowError> {
    let [batch, tokens] = durations.dims();
    let [mask_batch, mask_tokens, frames] = mask.dims();
    if batch == 0 || tokens == 0 || frames == 0 {
        return Err(input_error(
            "path inputs must have non-empty batch, token, and frame dimensions",
        ));
    }
    if [mask_batch, mask_tokens] != [batch, tokens] {
        return Err(input_error(format!(
            "path mask dimensions [{mask_batch}, {mask_tokens}, {frames}] do not match durations [{batch}, {tokens}]"
        )));
    }

    let ends = durations.clone().cumsum(1);
    let starts = ends.clone() - durations;
    let starts = starts
        .reshape([batch, tokens, 1])
        .expand([batch, tokens, frames]);
    let ends = ends
        .reshape([batch, tokens, 1])
        .expand([batch, tokens, frames]);
    let positions = Tensor::<B, 1, Int>::arange(0..frames as i64, &mask.device())
        .float()
        .reshape([1, 1, frames])
        .expand([batch, tokens, frames]);

    Ok(positions
        .clone()
        .greater_equal(starts)
        .bool_and(positions.lower(ends))
        .float()
        * mask)
}

/// Frame-aligned prior statistics and their monotonic alignment metadata.
#[derive(Debug)]
pub struct ExpandedPrior<B: Backend> {
    /// Prior means in `[batch, channels, frames]` layout.
    pub mean: Tensor<B, 3>,
    /// Prior log-scales in `[batch, channels, frames]` layout.
    pub log_scale: Tensor<B, 3>,
    /// Hard alignment in `[batch, tokens, frames]` layout.
    pub path: Tensor<B, 3>,
    /// Valid-frame mask in `[batch, 1, frames]` layout.
    pub frame_mask: Tensor<B, 3>,
    /// Per-sample output lengths in `[batch]` layout.
    pub frame_lengths: Tensor<B, 1>,
}

/// Expands token-aligned prior means and log-scales along a monotonic path.
pub fn expand_prior_statistics<B: Backend>(
    mean: Tensor<B, 3>,
    log_scale: Tensor<B, 3>,
    durations: Tensor<B, 2>,
    max_output_frames: usize,
) -> Result<ExpandedPrior<B>, VitsFlowError> {
    expand_prior_statistics_with_frames(mean, log_scale, durations, None, max_output_frames)
}

pub(crate) fn expand_prior_statistics_with_frames<B: Backend>(
    mean: Tensor<B, 3>,
    log_scale: Tensor<B, 3>,
    durations: Tensor<B, 2>,
    known_output_frames: Option<usize>,
    max_output_frames: usize,
) -> Result<ExpandedPrior<B>, VitsFlowError> {
    let [batch, channels, tokens] = mean.dims();
    if batch == 0 || channels == 0 || tokens == 0 {
        return Err(input_error(
            "prior statistics must have non-empty [batch, channels, tokens] dimensions",
        ));
    }
    if log_scale.dims() != [batch, channels, tokens] {
        return Err(input_error(format!(
            "prior log-scale dimensions {:?} do not match mean [{batch}, {channels}, {tokens}]",
            log_scale.dims()
        )));
    }
    if durations.dims() != [batch, tokens] {
        return Err(input_error(format!(
            "duration dimensions {:?} do not match prior [{batch}, {channels}, {tokens}]",
            durations.dims()
        )));
    }
    if max_output_frames == 0 {
        return Err(input_error("max_output_frames must be positive"));
    }

    let frame_lengths = durations.clone().sum_dim(1).reshape([batch]);
    let frames = match known_output_frames {
        Some(frames) => frames,
        None => frame_lengths.clone().max().into_scalar().elem::<f32>() as usize,
    };
    if frames == 0 {
        return Err(input_error("durations produce zero output frames"));
    }
    if frames > max_output_frames {
        return Err(input_error(format!(
            "durations request {frames} frames, exceeding configured limit {max_output_frames}"
        )));
    }

    let positions = Tensor::<B, 1, Int>::arange(0..frames as i64, &mean.device())
        .float()
        .reshape([1, 1, frames])
        .expand([batch, 1, frames]);
    let frame_mask = positions
        .lower(
            frame_lengths
                .clone()
                .reshape([batch, 1, 1])
                .expand([batch, 1, frames]),
        )
        .float();
    let path_mask = frame_mask.clone().expand([batch, tokens, frames]);
    let path = generate_path(durations, path_mask)?;
    let expanded_mean = mean.matmul(path.clone());
    let expanded_log_scale = log_scale.matmul(path.clone());

    Ok(ExpandedPrior {
        mean: expanded_mean,
        log_scale: expanded_log_scale,
        path,
        frame_mask,
        frame_lengths,
    })
}

/// Topology of a stack of mean-only residual coupling layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualCouplingFlowConfig {
    pub channels: usize,
    pub hidden_channels: usize,
    pub kernel_size: usize,
    pub dilation_rate: usize,
    pub num_layers: usize,
    pub num_flows: usize,
    pub conditioning_channels: usize,
}

impl ResidualCouplingFlowConfig {
    pub fn validate(&self) -> Result<(), VitsFlowError> {
        if self.channels == 0 || !self.channels.is_multiple_of(2) {
            return Err(topology_error(
                "flow channels must be positive and divisible by two",
            ));
        }
        if self.hidden_channels == 0 || !self.hidden_channels.is_multiple_of(2) {
            return Err(topology_error(
                "WaveNet hidden channels must be positive and divisible by two",
            ));
        }
        if self.kernel_size == 0 || self.kernel_size.is_multiple_of(2) {
            return Err(topology_error(
                "WaveNet kernel size must be positive and odd",
            ));
        }
        if self.dilation_rate == 0 || self.num_layers == 0 || self.num_flows == 0 {
            return Err(topology_error(
                "dilation rate, layer count, and flow count must be positive",
            ));
        }
        self.dilation_rate
            .checked_pow((self.num_layers - 1) as u32)
            .ok_or_else(|| topology_error("WaveNet dilation overflows usize"))?;
        Ok(())
    }

    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<ResidualCouplingFlow<B>, VitsFlowError> {
        self.validate()?;
        let flows = (0..self.num_flows)
            .map(|_| {
                ResidualCouplingLayer::new(
                    self.channels,
                    self.hidden_channels,
                    self.kernel_size,
                    self.dilation_rate,
                    self.num_layers,
                    self.conditioning_channels,
                    device,
                )
            })
            .collect();
        Ok(ResidualCouplingFlow {
            flows,
            channels: self.channels,
            conditioning_channels: self.conditioning_channels,
        })
    }

    pub fn load_checkpoint<B: Backend>(
        &self,
        checkpoint_path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<ResidualCouplingFlow<B>, VitsFlowError> {
        self.init(device)?.load_checkpoint(checkpoint_path)
    }
}

fn pytorch_conv_initializer() -> Initializer {
    Initializer::KaimingUniform {
        gain: PYTORCH_CONV_GAIN,
        fan_out_only: false,
    }
}

fn plain_conv1d<B: Backend>(
    channels_in: usize,
    channels_out: usize,
    kernel_size: usize,
    device: &B::Device,
) -> Conv1d<B> {
    Conv1dConfig::new(channels_in, channels_out, kernel_size)
        .with_padding(PaddingConfig1d::Valid)
        .with_initializer(pytorch_conv_initializer())
        .init(device)
}

/// Weight-normalized one-dimensional convolution with checkpoint-compatible
/// `weight_g`, `weight_v`, and `bias` parameter names.
#[derive(Module, Debug)]
pub struct FlowWeightNormConv1d<B: Backend> {
    pub weight_g: Param<Tensor<B, 3>>,
    pub weight_v: Param<Tensor<B, 3>>,
    pub bias: Param<Tensor<B, 1>>,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl<B: Backend> FlowWeightNormConv1d<B> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
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
        let bias = pytorch_conv_initializer().init_with([channels_out], Some(fan_in), None, device);
        Self {
            weight_g: Param::from_tensor(weight_g),
            weight_v,
            bias,
            stride,
            padding,
            dilation,
        }
    }

    fn weight(&self) -> Tensor<B, 3> {
        let weight_v = self.weight_v.val();
        weight_v.clone() * self.weight_g.val() / weight_norm_dim_zero(weight_v)
    }

    pub(crate) fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        conv1d(
            input,
            self.weight(),
            Some(self.bias.val()),
            ConvOptions::new([self.stride], [self.padding], [self.dilation], 1),
        )
    }
}

fn weight_norm_dim_zero<B: Backend>(weight: Tensor<B, 3>) -> Tensor<B, 3> {
    weight.powf_scalar(2.0).sum_dims(&[1usize, 2usize]).sqrt()
}

/// Conditioned, non-causal WaveNet used by each coupling layer.
#[derive(Module, Debug)]
pub struct CouplingWaveNet<B: Backend> {
    pub in_layers: Vec<FlowWeightNormConv1d<B>>,
    pub res_skip_layers: Vec<FlowWeightNormConv1d<B>>,
    pub cond_layer: Option<FlowWeightNormConv1d<B>>,
    hidden_channels: usize,
}

impl<B: Backend> CouplingWaveNet<B> {
    pub(crate) fn new(
        hidden_channels: usize,
        kernel_size: usize,
        dilation_rate: usize,
        num_layers: usize,
        conditioning_channels: usize,
        device: &B::Device,
    ) -> Self {
        let mut in_layers = Vec::with_capacity(num_layers);
        let mut res_skip_layers = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let dilation = dilation_rate.pow(layer as u32);
            let padding = (kernel_size * dilation - dilation) / 2;
            in_layers.push(FlowWeightNormConv1d::new(
                hidden_channels,
                hidden_channels * 2,
                kernel_size,
                1,
                padding,
                dilation,
                device,
            ));
            let output_channels = if layer + 1 < num_layers {
                hidden_channels * 2
            } else {
                hidden_channels
            };
            res_skip_layers.push(FlowWeightNormConv1d::new(
                hidden_channels,
                output_channels,
                1,
                1,
                0,
                1,
                device,
            ));
        }
        let cond_layer = (conditioning_channels > 0).then(|| {
            FlowWeightNormConv1d::new(
                conditioning_channels,
                hidden_channels * 2 * num_layers,
                1,
                1,
                0,
                1,
                device,
            )
        });
        Self {
            in_layers,
            res_skip_layers,
            cond_layer,
            hidden_channels,
        }
    }

    pub(crate) fn forward(
        &self,
        mut input: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let [batch, _, frames] = input.dims();
        let device = input.device();
        let projected_conditioning = match (&self.cond_layer, conditioning) {
            (Some(layer), Some(value)) => Some(layer.forward(value)),
            _ => None,
        };
        let mut output = Tensor::<B, 3>::zeros([batch, self.hidden_channels, frames], &device);
        for (layer_index, (input_layer, res_skip_layer)) in
            self.in_layers.iter().zip(&self.res_skip_layers).enumerate()
        {
            let activation_input = input_layer.forward(input.clone());
            let activation_input = match &projected_conditioning {
                Some(projected) => {
                    let start = layer_index * 2 * self.hidden_channels;
                    activation_input
                        + projected
                            .clone()
                            .slice([
                                0..batch,
                                start..start + 2 * self.hidden_channels,
                                0..projected.dims()[2],
                            ])
                            .expand([batch, 2 * self.hidden_channels, frames])
                }
                None => activation_input,
            };
            let tanh_gate = activation_input
                .clone()
                .slice([0..batch, 0..self.hidden_channels, 0..frames])
                .tanh();
            let sigmoid_gate = sigmoid(activation_input.slice([
                0..batch,
                self.hidden_channels..2 * self.hidden_channels,
                0..frames,
            ]));
            let residual_skip = res_skip_layer.forward(tanh_gate * sigmoid_gate);
            if layer_index + 1 < self.in_layers.len() {
                input = (input
                    + residual_skip
                        .clone()
                        .slice([0..batch, 0..self.hidden_channels, 0..frames]))
                    * mask.clone();
                output = output
                    + residual_skip.slice([
                        0..batch,
                        self.hidden_channels..2 * self.hidden_channels,
                        0..frames,
                    ]);
            } else {
                output = output + residual_skip;
            }
        }
        output * mask
    }
}

/// Mean-only residual coupling transform.
#[derive(Module, Debug)]
pub struct ResidualCouplingLayer<B: Backend> {
    pub pre: Conv1d<B>,
    pub enc: CouplingWaveNet<B>,
    pub post: Conv1d<B>,
    half_channels: usize,
}

impl<B: Backend> ResidualCouplingLayer<B> {
    fn new(
        channels: usize,
        hidden_channels: usize,
        kernel_size: usize,
        dilation_rate: usize,
        num_layers: usize,
        conditioning_channels: usize,
        device: &B::Device,
    ) -> Self {
        let half_channels = channels / 2;
        Self {
            pre: plain_conv1d(half_channels, hidden_channels, 1, device),
            enc: CouplingWaveNet::new(
                hidden_channels,
                kernel_size,
                dilation_rate,
                num_layers,
                conditioning_channels,
                device,
            ),
            post: plain_conv1d(hidden_channels, half_channels, 1, device),
            half_channels,
        }
    }

    fn mean(
        &self,
        first_half: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let hidden = self.pre.forward(first_half) * mask.clone();
        self.post
            .forward(self.enc.forward(hidden, mask.clone(), conditioning))
            * mask
    }

    fn forward_transform(
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
            self.half_channels..2 * self.half_channels,
            0..frames,
        ]);
        let mean = self.mean(first.clone(), mask.clone(), conditioning);
        Tensor::cat(vec![first, (mean + second) * mask.clone()], 1) * mask
    }

    fn reverse_transform(
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
            self.half_channels..2 * self.half_channels,
            0..frames,
        ]);
        let mean = self.mean(first.clone(), mask.clone(), conditioning);
        Tensor::cat(vec![first, (second - mean) * mask.clone()], 1) * mask
    }
}

/// Stack of residual coupling layers with channel reversal between layers.
#[derive(Module, Debug)]
pub struct ResidualCouplingFlow<B: Backend> {
    pub flows: Vec<ResidualCouplingLayer<B>>,
    channels: usize,
    conditioning_channels: usize,
}

impl<B: Backend> ResidualCouplingFlow<B> {
    /// Loads only `flow.*` tensors and rejects missing, erroneous, or unused
    /// tensors in that inference subtree.
    pub fn load_checkpoint(
        mut self,
        checkpoint_path: impl AsRef<Path>,
    ) -> Result<Self, VitsFlowError> {
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut self,
            checkpoint_path.as_ref(),
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                predicate: Some(flow_tensor),
                key_remappings: vec![(r"^flow\.".into(), String::new())],
                map_indices_contiguous: false,
                allow_partial: true,
                skip_enum_variants: true,
            },
        )
        .map_err(|error| VitsFlowError::Checkpoint(error.to_string()))?;
        let unexpected_unused = result
            .unused
            .iter()
            .filter(|path| flow_tensor(path, ""))
            .cloned()
            .collect::<Vec<_>>();
        if !result.missing.is_empty() || !result.errors.is_empty() || !unexpected_unused.is_empty()
        {
            return Err(VitsFlowError::Checkpoint(format!(
                "flow subtree does not exactly match the Burn module: {} missing, {} load errors, unused [{}]",
                result.missing.len(),
                result.errors.len(),
                unexpected_unused.join(", ")
            )));
        }
        Ok(self)
    }

    /// Applies the training-direction coupling stack.
    ///
    /// This exists to verify the bijection and can also support future native
    /// training. Speech synthesis normally calls [`Self::reverse`].
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 3>>,
    ) -> Result<Tensor<B, 3>, VitsFlowError> {
        self.validate_inputs(&input, &mask, conditioning.as_ref())?;
        let mut output = input * mask.clone();
        for flow in &self.flows {
            output = flow.forward_transform(output, mask.clone(), conditioning.clone());
            output = output.flip([1]);
        }
        Ok(output * mask)
    }

    /// Applies reverse residual coupling inference to prior latent features.
    pub fn reverse(
        &self,
        input: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 3>>,
    ) -> Result<Tensor<B, 3>, VitsFlowError> {
        self.validate_inputs(&input, &mask, conditioning.as_ref())?;
        let mut output = input * mask.clone();
        for flow in self.flows.iter().rev() {
            output = output.flip([1]);
            output = flow.reverse_transform(output, mask.clone(), conditioning.clone());
        }
        Ok(output * mask)
    }

    fn validate_inputs(
        &self,
        input: &Tensor<B, 3>,
        mask: &Tensor<B, 3>,
        conditioning: Option<&Tensor<B, 3>>,
    ) -> Result<(), VitsFlowError> {
        let [batch, channels, frames] = input.dims();
        if batch == 0 || channels != self.channels || frames == 0 {
            return Err(input_error(format!(
                "latent input must be non-empty with {} channels, got [{batch}, {channels}, {frames}]",
                self.channels
            )));
        }
        if mask.dims() != [batch, 1, frames] {
            return Err(input_error(format!(
                "latent mask dimensions {:?} do not match [{batch}, 1, {frames}]",
                mask.dims()
            )));
        }
        match (self.conditioning_channels, conditioning) {
            (0, Some(_)) => {
                return Err(input_error(
                    "conditioning was supplied to an unconditioned flow",
                ));
            }
            (expected, Some(value)) => {
                let [cond_batch, cond_channels, cond_frames] = value.dims();
                if cond_batch != batch
                    || cond_channels != expected
                    || (cond_frames != 1 && cond_frames != frames)
                {
                    return Err(input_error(format!(
                        "conditioning must have dimensions [{batch}, {expected}, 1 or {frames}], got [{cond_batch}, {cond_channels}, {cond_frames}]"
                    )));
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn flow_tensor(path: &str, _container: &str) -> bool {
    path.starts_with("flows.")
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::TensorData;

    use super::*;

    type TestBackend = NdArray<f32>;

    #[test]
    fn duration_ceiling_and_path_are_monotonic_and_complete() {
        let device = NdArrayDevice::Cpu;
        let durations = Tensor::<TestBackend, 3>::from_floats([[[1.2, 0.0, 2.1]]], &device);
        let token_mask = Tensor::<TestBackend, 3>::ones([1, 1, 3], &device);
        let rounded = ceil_durations(durations, token_mask, 1.0, 10).expect("rounded durations");
        rounded
            .values
            .clone()
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([[2.0, 0.0, 3.0]]), Default::default());
        assert_eq!(rounded.output_frames, 5);

        let mask = Tensor::<TestBackend, 3>::ones([1, 3, 5], &device);
        let path = generate_path(rounded.values, mask).expect("monotonic path");
        path.into_data().assert_approx_eq::<f32>(
            &TensorData::from([[
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0],
            ]]),
            Default::default(),
        );
    }

    #[test]
    fn prior_statistics_expand_along_the_hard_path() {
        let device = NdArrayDevice::Cpu;
        let mean = Tensor::<TestBackend, 3>::from_floats([[[10.0, 20.0, 30.0]]], &device);
        let log_scale = Tensor::<TestBackend, 3>::from_floats([[[0.1, 0.2, 0.3]]], &device);
        let durations = Tensor::<TestBackend, 2>::from_floats([[2.0, 0.0, 1.0]], &device);

        let expanded =
            expand_prior_statistics(mean, log_scale, durations, 8).expect("expanded prior");

        assert_eq!(expanded.mean.dims(), [1, 1, 3]);
        expanded.mean.into_data().assert_approx_eq::<f32>(
            &TensorData::from([[[10.0, 10.0, 30.0]]]),
            Default::default(),
        );
        expanded
            .log_scale
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([[[0.1, 0.1, 0.3]]]), Default::default());
        expanded
            .frame_mask
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([[[1.0, 1.0, 1.0]]]), Default::default());
    }

    #[test]
    fn coupling_stack_is_invertible_and_preserves_shape() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 73);
        let config = ResidualCouplingFlowConfig {
            channels: 4,
            hidden_channels: 4,
            kernel_size: 3,
            dilation_rate: 2,
            num_layers: 2,
            num_flows: 3,
            conditioning_channels: 2,
        };
        let flow = config.init::<TestBackend>(&device).expect("tiny flow");
        let input = Tensor::<TestBackend, 3>::random(
            [2, 4, 5],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let mask = Tensor::<TestBackend, 3>::ones([2, 1, 5], &device);
        let conditioning = Tensor::<TestBackend, 3>::random(
            [2, 2, 1],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );

        let transformed = flow
            .forward(input.clone(), mask.clone(), Some(conditioning.clone()))
            .expect("forward flow");
        assert_eq!(transformed.dims(), [2, 4, 5]);
        let restored = flow
            .reverse(transformed, mask, Some(conditioning))
            .expect("reverse flow");

        restored
            .into_data()
            .assert_approx_eq::<f32>(&input.into_data(), burn::tensor::Tolerance::absolute(1e-4));
    }

    #[test]
    fn published_flow_checkpoint_loads_and_runs_when_provided() {
        let Some(checkpoint_path) = std::env::var_os("TONGUES_TEST_COQUI_VITS_CHECKPOINT") else {
            return;
        };
        let device = NdArrayDevice::Cpu;
        let config = ResidualCouplingFlowConfig {
            channels: 192,
            hidden_channels: 192,
            kernel_size: 5,
            dilation_rate: 1,
            num_layers: 4,
            num_flows: 4,
            conditioning_channels: 256,
        };
        let flow = config
            .load_checkpoint::<TestBackend>(checkpoint_path, &device)
            .expect("strict flow subtree load");
        let input = Tensor::<TestBackend, 3>::zeros([1, 192, 3], &device);
        let mask = Tensor::<TestBackend, 3>::ones([1, 1, 3], &device);
        let conditioning = Tensor::<TestBackend, 3>::zeros([1, 256, 1], &device);

        let output = flow
            .reverse(input, mask, Some(conditioning))
            .expect("published reverse flow");

        assert_eq!(output.dims(), [1, 192, 3]);
    }
}
