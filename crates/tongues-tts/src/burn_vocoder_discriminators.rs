//! Multi-period and multi-scale discriminators for native vocoder training.
//!
//! [`MultiPeriodDiscriminator`] (MPD) and [`MultiScaleDiscriminator`] (MSD)
//! are the training-only networks required by HiFi-GAN. The MSD alone is used
//! for MelGAN and MultiBand-MelGAN training.
//!
//! Discriminators are never exported to inference packages; they exist solely
//! to supply gradients to the generator during adversarial training.

use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn::nn::{PaddingConfig1d, PaddingConfig2d};
use burn::tensor::activation::leaky_relu;
use burn::tensor::backend::Backend;
use burn::tensor::module::avg_pool1d;
use burn::tensor::Tensor;

const LRELU_SLOPE: f64 = 0.1;

/// Intermediate activations and unbounded score from one sub-discriminator.
///
/// `feature_maps` are the inputs to the loss-side of feature-matching loss.
/// `score` is `[batch, 1, time_reduced]` – real/fake distinction logits.
#[derive(Debug)]
pub struct SubDiscriminatorOutput<B: Backend> {
    /// One tensor per intermediate layer: used by [`crate::burn_vocoder_losses::feature_matching_loss`].
    pub feature_maps: Vec<Tensor<B, 3>>,
    /// Unbounded discriminator score shaped `[batch, 1, reduced_time]`.
    pub score: Tensor<B, 3>,
}

/// Collected outputs from every sub-discriminator in a stack.
#[derive(Debug)]
pub struct DiscriminatorStackOutput<B: Backend> {
    pub sub_outputs: Vec<SubDiscriminatorOutput<B>>,
}

impl<B: Backend> DiscriminatorStackOutput<B> {
    /// Score tensor from every sub-discriminator.
    pub fn scores(&self) -> Vec<Tensor<B, 3>> {
        self.sub_outputs.iter().map(|o| o.score.clone()).collect()
    }

    /// Feature-map vectors from every sub-discriminator.
    pub fn feature_maps(&self) -> Vec<Vec<Tensor<B, 3>>> {
        self.sub_outputs
            .iter()
            .map(|o| o.feature_maps.clone())
            .collect()
    }
}

// ── Multi-period discriminator (MPD) ─────────────────────────────────────────

/// Periods used by the HiFi-GAN multi-period discriminator.
pub const MPD_PERIODS: [usize; 5] = [2, 3, 5, 7, 11];

/// One period sub-discriminator from the HiFi-GAN MPD.
///
/// Folds the waveform `[B, 1, T]` into a 2-D view `[B, 1, T/p, p]`, then
/// applies a stack of strided 2-D convolutions. Feature maps from all but the
/// final layer are returned for feature-matching loss.
#[derive(Module, Debug)]
pub struct PeriodSubDiscriminator<B: Backend> {
    convs: Vec<Conv2d<B>>,
    conv_post: Conv2d<B>,
    period: usize,
}

impl<B: Backend> PeriodSubDiscriminator<B> {
    /// Construct a sub-discriminator for a given `period`.
    pub fn new(period: usize, device: &B::Device) -> Self {
        // Channel progression: 1→32→128→512→1024→1024
        let channel_pairs: &[(usize, usize)] =
            &[(1, 32), (32, 128), (128, 512), (512, 1024), (1024, 1024)];
        let convs = channel_pairs
            .iter()
            .map(|&(c_in, c_out)| {
                Conv2dConfig::new([c_in, c_out], [5, 1])
                    .with_stride([3, 1])
                    .with_padding(PaddingConfig2d::Explicit(2, 0, 2, 0))
                    .init(device)
            })
            .collect();
        let conv_post = Conv2dConfig::new([1024, 1], [3, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .init(device);
        Self {
            convs,
            conv_post,
            period,
        }
    }

    /// Run the sub-discriminator.
    ///
    /// `waveform` must be shaped `[batch, 1, samples]`.
    pub fn forward(&self, waveform: Tensor<B, 3>) -> SubDiscriminatorOutput<B> {
        let [batch, _channels, time] = waveform.dims();
        // Pad the time dimension to a multiple of the period.
        let remainder = time % self.period;
        let x = if remainder != 0 {
            let pad = self.period - remainder;
            waveform.pad([(0, pad)], burn::tensor::ops::PadMode::Edge)
        } else {
            waveform
        };
        let padded_time = x.dims()[2];
        // Reshape [B, 1, T] → [B, 1, T/p, p]
        let mut x = x.reshape([batch, 1, padded_time / self.period, self.period]);

        let mut feature_maps = Vec::new();
        for conv in &self.convs {
            x = leaky_relu(conv.forward(x), LRELU_SLOPE);
            let [fb, fc, fh, fw] = x.dims();
            // Flatten the spatial dimensions into a single time axis for uniform
            // representation in feature-matching loss computation.
            feature_maps.push(x.clone().reshape([fb, fc, fh * fw]));
        }
        let score = self.conv_post.forward(x);
        let [sb, sc, sh, sw] = score.dims();
        SubDiscriminatorOutput {
            feature_maps,
            score: score.reshape([sb, sc, sh * sw]),
        }
    }
}

/// Multi-period discriminator (MPD) used in HiFi-GAN training.
///
/// Aggregates five [`PeriodSubDiscriminator`]s with periods `[2, 3, 5, 7, 11]`
/// and returns their combined feature maps and scores.
#[derive(Module, Debug)]
pub struct MultiPeriodDiscriminator<B: Backend> {
    sub_discriminators: Vec<PeriodSubDiscriminator<B>>,
}

impl<B: Backend> MultiPeriodDiscriminator<B> {
    /// Initialize with the canonical HiFi-GAN period set.
    pub fn new(device: &B::Device) -> Self {
        let sub_discriminators = MPD_PERIODS
            .iter()
            .map(|&period| PeriodSubDiscriminator::new(period, device))
            .collect();
        Self { sub_discriminators }
    }

    /// Forward pass over `[batch, 1, samples]`.
    pub fn forward(&self, waveform: Tensor<B, 3>) -> DiscriminatorStackOutput<B> {
        let sub_outputs = self
            .sub_discriminators
            .iter()
            .map(|disc| disc.forward(waveform.clone()))
            .collect();
        DiscriminatorStackOutput { sub_outputs }
    }
}

// ── Multi-scale discriminator (MSD) ──────────────────────────────────────────

/// One scale sub-discriminator from the HiFi-GAN / MelGAN MSD.
///
/// Applies grouped Conv1d layers to a (possibly downsampled) waveform and
/// returns intermediate activations for feature matching.
#[derive(Module, Debug)]
pub struct ScaleSubDiscriminator<B: Backend> {
    conv_pre: Conv1d<B>,
    convs: Vec<Conv1d<B>>,
    conv_post: Conv1d<B>,
}

impl<B: Backend> ScaleSubDiscriminator<B> {
    /// Initialize a scale sub-discriminator with the HiFi-GAN MSD topology.
    pub fn new(device: &B::Device) -> Self {
        // Input projection: 1 → 128 with a wide receptive field.
        let conv_pre = Conv1dConfig::new(1, 128, 15)
            .with_padding(PaddingConfig1d::Explicit(7, 7))
            .init(device);

        // Strided + grouped convolution stack.
        // (channels_in, channels_out, kernel, stride, groups)
        let layer_specs: &[(usize, usize, usize, usize, usize)] = &[
            (128, 128, 41, 2, 4),
            (128, 256, 41, 2, 16),
            (256, 512, 41, 4, 16),
            (512, 1024, 41, 4, 16),
            (1024, 1024, 5, 1, 1),
            (1024, 1024, 3, 1, 1),
        ];
        let convs = layer_specs
            .iter()
            .map(|&(c_in, c_out, kernel, stride, groups)| {
                let pad = kernel / 2;
                Conv1dConfig::new(c_in, c_out, kernel)
                    .with_stride(stride)
                    .with_groups(groups)
                    .with_padding(PaddingConfig1d::Explicit(pad, pad))
                    .init(device)
            })
            .collect();

        let conv_post = Conv1dConfig::new(1024, 1, 3)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .init(device);

        Self {
            conv_pre,
            convs,
            conv_post,
        }
    }

    /// Run the sub-discriminator over `[batch, 1, samples]`.
    pub fn forward(&self, waveform: Tensor<B, 3>) -> SubDiscriminatorOutput<B> {
        let mut x = leaky_relu(self.conv_pre.forward(waveform), LRELU_SLOPE);
        let mut feature_maps = Vec::new();
        for conv in &self.convs {
            x = leaky_relu(conv.forward(x), LRELU_SLOPE);
            feature_maps.push(x.clone());
        }
        let score = self.conv_post.forward(x);
        SubDiscriminatorOutput {
            feature_maps,
            score,
        }
    }
}

/// Multi-scale discriminator (MSD) used in HiFi-GAN and MelGAN training.
///
/// Applies a [`ScaleSubDiscriminator`] at three resolutions – original,
/// 2× average-pooled, and 4× average-pooled – and collects their outputs.
#[derive(Module, Debug)]
pub struct MultiScaleDiscriminator<B: Backend> {
    sub_discriminators: Vec<ScaleSubDiscriminator<B>>,
}

impl<B: Backend> MultiScaleDiscriminator<B> {
    /// Initialize three scale sub-discriminators.
    pub fn new(device: &B::Device) -> Self {
        let sub_discriminators = (0..3)
            .map(|_| ScaleSubDiscriminator::new(device))
            .collect();
        Self { sub_discriminators }
    }

    /// Forward pass over `[batch, 1, samples]` at three scales.
    pub fn forward(&self, waveform: Tensor<B, 3>) -> DiscriminatorStackOutput<B> {
        let mut sub_outputs = Vec::with_capacity(3);
        let mut x = waveform;
        for (index, disc) in self.sub_discriminators.iter().enumerate() {
            if index > 0 {
                // 2× average-pool for each additional scale.
                x = avg_pool1d(x, 4, 2, 2, false, false);
            }
            sub_outputs.push(disc.forward(x.clone()));
        }
        DiscriminatorStackOutput { sub_outputs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    fn ones_waveform(batch: usize, samples: usize) -> Tensor<TestBackend, 3> {
        let device = NdArrayDevice::Cpu;
        Tensor::ones([batch, 1, samples], &device)
    }

    #[test]
    fn period_sub_discriminator_produces_finite_outputs() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 1);
        let disc = PeriodSubDiscriminator::<TestBackend>::new(3, &device);
        let output = disc.forward(ones_waveform(2, 64));
        assert!(!output.feature_maps.is_empty());
        assert_eq!(output.score.dims()[0], 2);
        assert_eq!(output.score.dims()[1], 1);
        let values = output
            .score
            .into_data()
            .to_vec::<f32>()
            .expect("score values");
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mpd_has_five_sub_discriminators() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 2);
        let mpd = MultiPeriodDiscriminator::<TestBackend>::new(&device);
        let waveform = ones_waveform(1, 48);
        let output = mpd.forward(waveform);
        assert_eq!(output.sub_outputs.len(), 5);
        for sub in &output.sub_outputs {
            assert!(!sub.feature_maps.is_empty());
            let vals = sub
                .score
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("score");
            assert!(vals.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn scale_sub_discriminator_produces_finite_outputs() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 3);
        let disc = ScaleSubDiscriminator::<TestBackend>::new(&device);
        let output = disc.forward(ones_waveform(2, 128));
        assert_eq!(output.feature_maps.len(), 6);
        assert_eq!(output.score.dims()[0], 2);
        assert_eq!(output.score.dims()[1], 1);
        let vals = output
            .score
            .into_data()
            .to_vec::<f32>()
            .expect("score values");
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn msd_has_three_sub_discriminators_with_shrinking_time_axes() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 4);
        let msd = MultiScaleDiscriminator::<TestBackend>::new(&device);
        let waveform = ones_waveform(1, 256);
        let output = msd.forward(waveform);
        assert_eq!(output.sub_outputs.len(), 3);
        let t0 = output.sub_outputs[0].score.dims()[2];
        let t1 = output.sub_outputs[1].score.dims()[2];
        let t2 = output.sub_outputs[2].score.dims()[2];
        // Downsampled scales must have progressively shorter time axes.
        assert!(
            t0 >= t1 && t1 >= t2,
            "expected t0 >= t1 >= t2 but got {t0} >= {t1} >= {t2}"
        );
        for sub in &output.sub_outputs {
            let vals = sub
                .score
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("score");
            assert!(vals.iter().all(|v| v.is_finite()));
        }
    }
}
