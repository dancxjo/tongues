//! Native Burn implementations of Coqui MelGAN and MultiBand-MelGAN.
//!
//! The parameter-bearing layer vectors deliberately mirror the sparse
//! `nn.Sequential` paths used by Coqui. Checkpoint loading enables contiguous
//! index mapping, so source paths such as `layers.1`, `layers.3`, and
//! `layers.4.blocks.0.2` map to the compact native module vectors without
//! retaining parameter-free activation and padding modules.

use std::f64::consts::PI;
use std::fmt;

use burn::module::{Module, Param};
use burn::tensor::activation::leaky_relu;
use burn::tensor::backend::Backend;
use burn::tensor::module::{conv1d, conv_transpose1d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions, PadMode};
use burn::tensor::{Tensor, TensorData};

use crate::burn_hifigan::{WeightNormConv1d, WeightNormConvTranspose1d};

const ACTIVATION_SLOPE: f64 = 0.2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MelganError {
    InvalidConfig(String),
    InvalidInput(String),
}

impl fmt::Display for MelganError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(formatter, "invalid MelGAN config: {message}"),
            Self::InvalidInput(message) => write!(formatter, "invalid MelGAN input: {message}"),
        }
    }
}

impl std::error::Error for MelganError {}

fn config_error(message: impl Into<String>) -> MelganError {
    MelganError::InvalidConfig(message.into())
}

fn input_error(message: impl Into<String>) -> MelganError {
    MelganError::InvalidInput(message.into())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MelganGeneratorConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub projection_kernel_size: usize,
    pub base_channels: usize,
    pub upsample_factors: Vec<usize>,
    pub residual_kernel_size: usize,
    pub residual_blocks: usize,
    pub inference_padding: usize,
}

impl MelganGeneratorConfig {
    pub fn validate(&self) -> Result<(), MelganError> {
        if self.in_channels == 0 || self.out_channels == 0 || self.base_channels == 0 {
            return Err(config_error(
                "input, output, and base channel counts must be positive",
            ));
        }
        if self.projection_kernel_size == 0 || self.projection_kernel_size.is_multiple_of(2) {
            return Err(config_error(
                "projection kernel size must be positive and odd",
            ));
        }
        if self.residual_kernel_size == 0 || self.residual_kernel_size.is_multiple_of(2) {
            return Err(config_error(
                "residual kernel size must be positive and odd",
            ));
        }
        if self.residual_blocks == 0 {
            return Err(config_error("at least one residual block is required"));
        }
        if self.upsample_factors.is_empty() || self.upsample_factors.contains(&0) {
            return Err(config_error(
                "upsample factors must be non-empty and positive",
            ));
        }
        for stage in 0..self.upsample_factors.len() {
            let channels_in = self.base_channels >> stage;
            let channels_out = self.base_channels >> (stage + 1);
            if channels_in == 0 || channels_out == 0 {
                return Err(config_error(format!(
                    "base channel count {} is too small for {} upsample stages",
                    self.base_channels,
                    self.upsample_factors.len()
                )));
            }
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

    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<MelganGenerator<B>, MelganError> {
        self.validate()?;
        Ok(MelganGenerator {
            layers: init_layers(self, device)?,
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            upsample_factor: self.upsample_factor(),
            inference_padding: self.inference_padding,
        })
    }

    pub fn init_multiband<B: Backend>(
        &self,
        pqmf: PqmfConfig,
        device: &B::Device,
    ) -> Result<MultibandMelganGenerator<B>, MelganError> {
        self.validate()?;
        if self.out_channels != pqmf.bands {
            return Err(config_error(format!(
                "MultiBand-MelGAN emits {} channels but PQMF requires {} bands",
                self.out_channels, pqmf.bands
            )));
        }
        Ok(MultibandMelganGenerator {
            layers: init_layers(self, device)?,
            pqmf_layer: pqmf.init(device)?,
            in_channels: self.in_channels,
            subbands: self.out_channels,
            upsample_factor: self.upsample_factor(),
            inference_padding: self.inference_padding,
        })
    }
}

fn init_layers<B: Backend>(
    config: &MelganGeneratorConfig,
    device: &B::Device,
) -> Result<Vec<MelganLayer<B>>, MelganError> {
    let mut layers = Vec::with_capacity(2 + config.upsample_factors.len() * 2);
    layers.push(MelganLayer::Conv(WeightNormConv1d::new(
        config.in_channels,
        config.base_channels,
        config.projection_kernel_size,
        1,
        1,
        0,
        true,
        device,
    )));
    for (stage, &factor) in config.upsample_factors.iter().enumerate() {
        let channels_in = config.base_channels >> stage;
        let channels_out = config.base_channels >> (stage + 1);
        let output_padding = factor % 2;
        layers.push(MelganLayer::Upsample(
            WeightNormConvTranspose1d::new_with_output_padding(
                channels_in,
                channels_out,
                factor * 2,
                factor,
                1,
                factor / 2 + output_padding,
                output_padding,
                true,
                device,
            ),
        ));
        layers.push(MelganLayer::Residual(ResidualStack::new(
            channels_out,
            config.residual_blocks,
            config.residual_kernel_size,
            device,
        )?));
    }
    let final_channels = config.base_channels >> config.upsample_factors.len();
    layers.push(MelganLayer::Conv(WeightNormConv1d::new(
        final_channels,
        config.out_channels,
        config.projection_kernel_size,
        1,
        1,
        0,
        true,
        device,
    )));
    Ok(layers)
}

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum MelganLayer<B: Backend> {
    Conv(WeightNormConv1d<B>),
    Upsample(WeightNormConvTranspose1d<B>),
    Residual(ResidualStack<B>),
}

#[derive(Module, Debug)]
pub struct ResidualStack<B: Backend> {
    /// Each source `nn.Sequential` contributes its two parameter-bearing
    /// convolutions after sparse indices are mapped contiguously.
    pub blocks: Vec<Vec<WeightNormConv1d<B>>>,
    pub shortcuts: Vec<WeightNormConv1d<B>>,
    pub kernel_size: usize,
}

impl<B: Backend> ResidualStack<B> {
    fn new(
        channels: usize,
        blocks: usize,
        kernel_size: usize,
        device: &B::Device,
    ) -> Result<Self, MelganError> {
        let mut block_layers = Vec::with_capacity(blocks);
        let mut shortcuts = Vec::with_capacity(blocks);
        for index in 0..blocks {
            let dilation = kernel_size
                .checked_pow(index as u32)
                .ok_or_else(|| config_error("residual dilation overflow"))?;
            block_layers.push(vec![
                WeightNormConv1d::new(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation,
                    0,
                    true,
                    device,
                ),
                WeightNormConv1d::new(channels, channels, 1, 1, 1, 0, true, device),
            ]);
            shortcuts.push(WeightNormConv1d::new(
                channels, channels, 1, 1, 1, 0, true, device,
            ));
        }
        Ok(Self {
            blocks: block_layers,
            shortcuts,
            kernel_size,
        })
    }

    fn forward(&self, mut input: Tensor<B, 3>) -> Tensor<B, 3> {
        let base_padding = (self.kernel_size - 1) / 2;
        for (index, (block, shortcut)) in self.blocks.iter().zip(&self.shortcuts).enumerate() {
            let dilation = self.kernel_size.pow(index as u32);
            let padding = base_padding * dilation;
            let hidden = leaky_relu(input.clone(), ACTIVATION_SLOPE)
                .pad([(padding, padding)], PadMode::Reflect);
            let hidden = block[0].forward(hidden);
            let hidden = block[1].forward(leaky_relu(hidden, ACTIVATION_SLOPE));
            input = shortcut.forward(input) + hidden;
        }
        input
    }
}

fn forward_layers<B: Backend>(
    layers: &[MelganLayer<B>],
    input: Tensor<B, 3>,
) -> Result<Tensor<B, 3>, MelganError> {
    let first = layers
        .first()
        .ok_or_else(|| input_error("generator has no layers"))?;
    let last = layers
        .last()
        .ok_or_else(|| input_error("generator has no layers"))?;
    let MelganLayer::Conv(first) = first else {
        return Err(input_error("generator does not start with a projection"));
    };
    let MelganLayer::Conv(last) = last else {
        return Err(input_error("generator does not end with a projection"));
    };
    let projection_padding = (first.weight_v.dims()[2] - 1) / 2;
    let mut output =
        first.forward(input.pad([(projection_padding, projection_padding)], PadMode::Reflect));
    for pair in layers[1..layers.len() - 1].chunks_exact(2) {
        let [MelganLayer::Upsample(upsample), MelganLayer::Residual(residual)] = pair else {
            return Err(input_error(
                "generator upsample/residual layer order is invalid",
            ));
        };
        output = upsample.forward(leaky_relu(output, ACTIVATION_SLOPE));
        output = residual.forward(output);
    }
    output = leaky_relu(output, ACTIVATION_SLOPE);
    Ok(last
        .forward(output.pad([(projection_padding, projection_padding)], PadMode::Reflect))
        .tanh())
}

#[derive(Module, Debug)]
pub struct MelganGenerator<B: Backend> {
    pub layers: Vec<MelganLayer<B>>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub upsample_factor: usize,
    pub inference_padding: usize,
}

impl<B: Backend> MelganGenerator<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, MelganError> {
        let [_, channels, frames] = input.dims();
        if channels != self.in_channels || frames == 0 {
            return Err(input_error(format!(
                "expected non-empty input with {} channels, got [{channels}, {frames}]",
                self.in_channels
            )));
        }
        forward_layers(&self.layers, input)
    }

    pub fn inference(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, MelganError> {
        if input.dims()[2] == 0 {
            return Err(input_error("feature input must contain at least one frame"));
        }
        self.forward(input.pad(
            [(self.inference_padding, self.inference_padding)],
            PadMode::Edge,
        ))
    }

    pub fn inference_output_frames(&self, frames: usize) -> Option<usize> {
        frames
            .checked_add(self.inference_padding.checked_mul(2)?)
            .and_then(|frames| frames.checked_mul(self.upsample_factor))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PqmfConfig {
    pub bands: usize,
    pub taps: usize,
    pub cutoff: f64,
    pub beta: f64,
}

impl Default for PqmfConfig {
    fn default() -> Self {
        Self {
            bands: 4,
            taps: 62,
            cutoff: 0.15,
            beta: 9.0,
        }
    }
}

impl PqmfConfig {
    pub fn validate(&self) -> Result<(), MelganError> {
        if self.bands == 0 || self.taps == 0 || !self.taps.is_multiple_of(2) {
            return Err(config_error(
                "PQMF requires positive bands and a positive even tap order",
            ));
        }
        if !self.cutoff.is_finite() || !(0.0..1.0).contains(&self.cutoff) {
            return Err(config_error("PQMF cutoff must be finite and in 0..1"));
        }
        if !self.beta.is_finite() || self.beta < 0.0 {
            return Err(config_error(
                "PQMF Kaiser beta must be finite and non-negative",
            ));
        }
        Ok(())
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<Pqmf<B>, MelganError> {
        self.validate()?;
        let prototype = firwin_lowpass(self.taps + 1, self.cutoff, self.beta);
        let center = (self.taps as f64 - 1.0) / 2.0;
        let mut analysis = vec![0.0f32; self.bands * (self.taps + 1)];
        let mut synthesis = vec![0.0f32; self.bands * (self.taps + 1)];
        for band in 0..self.bands {
            let phase = if band.is_multiple_of(2) {
                PI / 4.0
            } else {
                -PI / 4.0
            };
            for tap in 0..=self.taps {
                let angle =
                    (2 * band + 1) as f64 * PI / (2 * self.bands) as f64 * (tap as f64 - center);
                analysis[band * (self.taps + 1) + tap] =
                    (2.0 * prototype[tap] * (angle + phase).cos()) as f32;
                synthesis[band * (self.taps + 1) + tap] =
                    (2.0 * prototype[tap] * (angle - phase).cos()) as f32;
            }
        }
        let mut updown = vec![0.0f32; self.bands * self.bands * self.bands];
        for band in 0..self.bands {
            updown[(band * self.bands + band) * self.bands] = 1.0;
        }
        Ok(Pqmf {
            analysis_filter: Param::from_tensor(Tensor::from_data(
                TensorData::new(analysis, [self.bands, 1, self.taps + 1]),
                device,
            )),
            synthesis_filter: Param::from_tensor(Tensor::from_data(
                TensorData::new(synthesis, [1, self.bands, self.taps + 1]),
                device,
            )),
            updown_filter: Param::from_tensor(Tensor::from_data(
                TensorData::new(updown, [self.bands, self.bands, self.bands]),
                device,
            )),
            bands: self.bands,
            taps: self.taps,
        })
    }
}

fn firwin_lowpass(length: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let alpha = (length - 1) as f64 / 2.0;
    let denominator = bessel_i0(beta);
    let mut coefficients = (0..length)
        .map(|index| {
            let offset = index as f64 - alpha;
            let sinc = if offset == 0.0 {
                cutoff
            } else {
                (PI * cutoff * offset).sin() / (PI * offset)
            };
            let ratio = offset / alpha;
            let window = bessel_i0(beta * (1.0 - ratio * ratio).max(0.0).sqrt()) / denominator;
            sinc * window
        })
        .collect::<Vec<_>>();
    let scale = coefficients.iter().sum::<f64>();
    for value in &mut coefficients {
        *value /= scale;
    }
    coefficients
}

fn bessel_i0(value: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let quarter = value * value / 4.0;
    for index in 1..=64 {
        term *= quarter / (index * index) as f64;
        sum += term;
        if term.abs() <= sum.abs() * f64::EPSILON {
            break;
        }
    }
    sum
}

#[derive(Module, Debug)]
pub struct Pqmf<B: Backend> {
    pub analysis_filter: Param<Tensor<B, 3>>,
    pub synthesis_filter: Param<Tensor<B, 3>>,
    pub updown_filter: Param<Tensor<B, 3>>,
    pub bands: usize,
    pub taps: usize,
}

impl<B: Backend> Pqmf<B> {
    pub fn analysis(&self, waveform: Tensor<B, 3>) -> Result<Tensor<B, 3>, MelganError> {
        if waveform.dims()[1] != 1 {
            return Err(input_error("PQMF analysis requires mono waveform input"));
        }
        Ok(conv1d(
            waveform,
            self.analysis_filter.val(),
            None,
            ConvOptions::new([self.bands], [self.taps / 2], [1], 1),
        ))
    }

    pub fn synthesis(&self, subbands: Tensor<B, 3>) -> Result<Tensor<B, 3>, MelganError> {
        if subbands.dims()[1] != self.bands {
            return Err(input_error(format!(
                "PQMF synthesis requires {} subbands, got {}",
                self.bands,
                subbands.dims()[1]
            )));
        }
        let upsampled = conv_transpose1d(
            subbands,
            self.updown_filter.val() * self.bands as f32,
            None,
            ConvTransposeOptions::new([self.bands], [0], [0], [1], 1),
        );
        Ok(conv1d(
            upsampled,
            self.synthesis_filter.val(),
            None,
            ConvOptions::new([1], [self.taps / 2], [1], 1),
        ))
    }
}

#[derive(Module, Debug)]
pub struct MultibandMelganGenerator<B: Backend> {
    pub layers: Vec<MelganLayer<B>>,
    pub pqmf_layer: Pqmf<B>,
    pub in_channels: usize,
    pub subbands: usize,
    pub upsample_factor: usize,
    pub inference_padding: usize,
}

impl<B: Backend> MultibandMelganGenerator<B> {
    pub fn forward_subbands(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, MelganError> {
        let [_, channels, frames] = input.dims();
        if channels != self.in_channels || frames == 0 {
            return Err(input_error(format!(
                "expected non-empty input with {} channels, got [{channels}, {frames}]",
                self.in_channels
            )));
        }
        forward_layers(&self.layers, input)
    }

    pub fn inference(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, MelganError> {
        if input.dims()[2] == 0 {
            return Err(input_error("feature input must contain at least one frame"));
        }
        let subbands = self.forward_subbands(input.pad(
            [(self.inference_padding, self.inference_padding)],
            PadMode::Edge,
        ))?;
        self.pqmf_layer.synthesis(subbands)
    }

    pub fn inference_output_frames(&self, frames: usize) -> Option<usize> {
        frames
            .checked_add(self.inference_padding.checked_mul(2)?)
            .and_then(|frames| frames.checked_mul(self.upsample_factor))
            .and_then(|subband_frames| subband_frames.checked_mul(self.subbands))
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    use super::*;

    type TestBackend = NdArray<f32>;

    fn tiny_config(out_channels: usize) -> MelganGeneratorConfig {
        MelganGeneratorConfig {
            in_channels: 4,
            out_channels,
            projection_kernel_size: 3,
            base_channels: 16,
            upsample_factors: vec![2, 2],
            residual_kernel_size: 3,
            residual_blocks: 2,
            inference_padding: 2,
        }
    }

    #[test]
    fn melgan_forward_and_inference_follow_upsample_contract() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 17);
        let model = tiny_config(1).init::<TestBackend>(&device).unwrap();
        let input = Tensor::ones([2, 4, 5], &device);

        assert_eq!(model.forward(input.clone()).unwrap().dims(), [2, 1, 20]);
        assert_eq!(model.inference(input).unwrap().dims(), [2, 1, 36]);
        assert_eq!(model.inference_output_frames(5), Some(36));
    }

    #[test]
    fn multiband_output_is_synthesized_to_full_rate() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 23);
        let model = tiny_config(4)
            .init_multiband::<TestBackend>(PqmfConfig::default(), &device)
            .unwrap();
        let input = Tensor::ones([1, 4, 5], &device);

        assert_eq!(
            model.forward_subbands(input.clone()).unwrap().dims(),
            [1, 4, 20]
        );
        assert_eq!(model.inference(input).unwrap().dims(), [1, 1, 144]);
        assert_eq!(model.inference_output_frames(5), Some(144));
    }

    #[test]
    fn pqmf_reconstruction_preserves_length_and_has_bounded_boundaries() {
        let device = NdArrayDevice::Cpu;
        let pqmf = PqmfConfig::default().init::<TestBackend>(&device).unwrap();
        let samples = (0..1024)
            .map(|index| (std::f32::consts::TAU * index as f32 / 71.0).sin() * 0.25)
            .collect::<Vec<_>>();
        let waveform = Tensor::from_data(
            TensorData::new(samples.clone(), [1, 1, samples.len()]),
            &device,
        );
        let reconstructed = pqmf
            .synthesis(pqmf.analysis(waveform).unwrap())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        assert_eq!(reconstructed.len(), samples.len());
        assert!(reconstructed.iter().all(|sample| sample.is_finite()));
        assert!(reconstructed.iter().all(|sample| sample.abs() <= 1.0));
        let upstream_boundaries = [
            (0, -0.008_270_583),
            (1, -0.009_795_113),
            (2, 0.010_811_696),
            (1021, 0.095_793_23),
            (1022, 0.027_590_973),
            (1023, -0.021_864_146),
        ];
        for (index, expected) in upstream_boundaries {
            let actual = reconstructed[index];
            assert!(
                (actual - expected).abs() <= 2e-5,
                "PQMF boundary sample {index} differs: native {actual}, Coqui {expected}"
            );
        }
        let interior_error = samples[64..samples.len() - 64]
            .iter()
            .zip(&reconstructed[64..reconstructed.len() - 64])
            .map(|(expected, actual)| (expected - actual).abs())
            .sum::<f32>()
            / (samples.len() - 128) as f32;
        // Coqui's short 63-tap prototype is intentionally approximate; the
        // important regression bounds are stable length, finite edges, and a
        // small interior reconstruction error.
        assert!(interior_error < 0.05, "interior MAE was {interior_error}");
    }
}
