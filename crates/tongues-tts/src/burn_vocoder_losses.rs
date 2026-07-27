//! Loss functions for native vocoder adversarial training.
//!
//! All functions operate on Burn tensors and are backend-agnostic. They do
//! not allocate Burn modules; the caller is responsible for wrapping results
//! in optimiser steps.
//!
//! ## Loss components
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`feature_matching_loss`] | Mean L1 between discriminator feature maps. |
//! | [`adversarial_generator_loss`] | MSE of discriminator scores against 1 (generator objective). |
//! | [`adversarial_discriminator_loss`] | MSE of real scores→1, fake scores→0. |
//! | [`mel_spectrogram_loss`] | L1 in the mel-spectrogram domain. |
//! | [`waveform_reconstruction_loss`] | L1 between target and predicted waveforms. |

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Configurable weights for the components of the combined vocoder loss.
#[derive(Debug, Clone)]
pub struct VocoderLossWeights {
    /// Weight applied to the feature-matching loss (default: 10.0, HiFi-GAN paper §3).
    pub feature_matching: f64,
    /// Weight applied to the mel-spectrogram reconstruction loss (default: 45.0).
    pub mel_spectrogram: f64,
    /// Weight applied to the waveform L1 reconstruction loss (default: 0.0 for HiFi-GAN,
    /// non-zero for MelGAN).
    pub waveform_reconstruction: f64,
    /// Weight applied to the generator adversarial loss (default: 1.0).
    pub adversarial_generator: f64,
}

impl Default for VocoderLossWeights {
    fn default() -> Self {
        Self {
            feature_matching: 10.0,
            mel_spectrogram: 45.0,
            waveform_reconstruction: 0.0,
            adversarial_generator: 1.0,
        }
    }
}

impl VocoderLossWeights {
    /// Weights matching the MelGAN paper: no mel loss, higher reconstruction weight.
    pub fn melgan() -> Self {
        Self {
            feature_matching: 10.0,
            mel_spectrogram: 0.0,
            waveform_reconstruction: 1.0,
            adversarial_generator: 1.0,
        }
    }
}

/// Mean L1 distance between corresponding feature maps.
///
/// `real_fmaps` and `fake_fmaps` are each a list-of-lists where the outer
/// dimension indexes sub-discriminators and the inner dimension indexes layers.
/// Both must have the same structure.
///
/// When real and fake feature maps have different time-axis lengths (which can
/// occur when the generator output length differs from the target waveform
/// length), the longer tensor is truncated to match the shorter one before
/// computing the L1 distance.
///
/// Returns a scalar loss tensor.
pub fn feature_matching_loss<B: Backend>(
    real_fmaps: Vec<Vec<Tensor<B, 3>>>,
    fake_fmaps: Vec<Vec<Tensor<B, 3>>>,
) -> Tensor<B, 1> {
    let mut loss: Option<Tensor<B, 1>> = None;
    let mut count = 0usize;
    for (real_layers, fake_layers) in real_fmaps.into_iter().zip(fake_fmaps) {
        for (real_map, fake_map) in real_layers.into_iter().zip(fake_layers) {
            // Align time axes before computing L1 to tolerate minor length differences
            // that arise when the generator segment length differs from the target.
            let (real_aligned, fake_aligned) = align_time_axes(real_map.detach(), fake_map);
            let diff = (real_aligned - fake_aligned).abs().mean();
            loss = Some(match loss {
                Some(acc) => acc + diff,
                None => diff,
            });
            count += 1;
        }
    }
    let total = loss.unwrap_or_else(|| {
        let device = Default::default();
        Tensor::zeros([1], &device)
    });
    if count > 1 {
        total / count as f64
    } else {
        total
    }
}

/// Truncate two 3-D tensors along the time axis (dim 2) to the same length.
fn align_time_axes<B: Backend>(a: Tensor<B, 3>, b: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let [ab, ac, at] = a.dims();
    let [bb, bc, bt] = b.dims();
    let min_t = at.min(bt);
    let a = if at > min_t {
        a.slice([0..ab, 0..ac, 0..min_t])
    } else {
        a
    };
    let b = if bt > min_t {
        b.slice([0..bb, 0..bc, 0..min_t])
    } else {
        b
    };
    (a, b)
}

/// Generator adversarial loss: mean MSE of discriminator scores against 1.
///
/// `fake_scores` is a list of score tensors (one per sub-discriminator),
/// each shaped `[batch, 1, reduced_time]`. The generator minimises this loss
/// by producing outputs that the discriminator rates as real.
///
/// Returns a scalar loss tensor.
pub fn adversarial_generator_loss<B: Backend>(fake_scores: Vec<Tensor<B, 3>>) -> Tensor<B, 1> {
    let mut loss: Option<Tensor<B, 1>> = None;
    let mut count = 0usize;
    for score in fake_scores {
        let ones = Tensor::ones_like(&score);
        let mse = (score - ones).powi_scalar(2).mean();
        loss = Some(match loss {
            Some(acc) => acc + mse,
            None => mse,
        });
        count += 1;
    }
    let total = loss.unwrap_or_else(|| {
        let device = Default::default();
        Tensor::zeros([1], &device)
    });
    if count > 1 {
        total / count as f64
    } else {
        total
    }
}

/// Discriminator adversarial loss: real scores→1, fake scores→0 (MSE / LSGAN).
///
/// `real_scores` and `fake_scores` are parallel lists of score tensors.
///
/// Returns a scalar loss tensor.
pub fn adversarial_discriminator_loss<B: Backend>(
    real_scores: Vec<Tensor<B, 3>>,
    fake_scores: Vec<Tensor<B, 3>>,
) -> Tensor<B, 1> {
    let mut loss: Option<Tensor<B, 1>> = None;
    let mut count = 0usize;
    for (real, fake) in real_scores.into_iter().zip(fake_scores) {
        let real_loss = (real - Tensor::ones_like(&fake)).powi_scalar(2).mean();
        let fake_loss = fake.powi_scalar(2).mean();
        let sub_loss = (real_loss + fake_loss) / 2.0f64;
        loss = Some(match loss {
            Some(acc) => acc + sub_loss,
            None => sub_loss,
        });
        count += 1;
    }
    let total = loss.unwrap_or_else(|| {
        let device = Default::default();
        Tensor::zeros([1], &device)
    });
    if count > 1 {
        total / count as f64
    } else {
        total
    }
}

/// L1 loss in the mel-spectrogram domain.
///
/// Both tensors must be shaped `[batch, mel_bins, frames]`. Returns a scalar.
pub fn mel_spectrogram_loss<B: Backend>(
    target_mel: Tensor<B, 3>,
    predicted_mel: Tensor<B, 3>,
) -> Tensor<B, 1> {
    (target_mel - predicted_mel).abs().mean()
}

/// L1 loss between target and predicted waveforms.
///
/// Both tensors must be shaped `[batch, channels, samples]`. Returns a scalar.
pub fn waveform_reconstruction_loss<B: Backend>(
    target: Tensor<B, 3>,
    predicted: Tensor<B, 3>,
) -> Tensor<B, 1> {
    (target - predicted).abs().mean()
}

/// Combine weighted loss components into a single scalar for the generator update.
///
/// Returns `(total_generator_loss, adversarial, feature_match, mel, reconstruction)`.
#[allow(clippy::too_many_arguments)]
pub fn combined_generator_loss<B: Backend>(
    adv_loss: Tensor<B, 1>,
    fm_loss: Tensor<B, 1>,
    mel_loss: Tensor<B, 1>,
    recon_loss: Tensor<B, 1>,
    weights: &VocoderLossWeights,
) -> Tensor<B, 1> {
    adv_loss * weights.adversarial_generator
        + fm_loss * weights.feature_matching
        + mel_loss * weights.mel_spectrogram
        + recon_loss * weights.waveform_reconstruction
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    fn tensor(values: &[f32], time: usize) -> Tensor<TestBackend, 3> {
        let device = NdArrayDevice::Cpu;
        Tensor::from_data(
            burn::tensor::TensorData::new(values.to_vec(), [1, 1, time]),
            &device,
        )
    }

    #[test]
    fn feature_matching_loss_is_zero_for_identical_maps() {
        let device = NdArrayDevice::Cpu;
        let map = Tensor::<TestBackend, 3>::ones([1, 8, 16], &device);
        let real = vec![vec![map.clone()]];
        let fake = vec![vec![map]];
        let loss = feature_matching_loss(real, fake)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(loss.abs() < 1e-6, "expected ~0 but got {loss}");
    }

    #[test]
    fn feature_matching_loss_is_positive_for_differing_maps() {
        let device = NdArrayDevice::Cpu;
        let real_map = Tensor::<TestBackend, 3>::ones([1, 8, 16], &device);
        let fake_map = Tensor::<TestBackend, 3>::zeros([1, 8, 16], &device);
        let real = vec![vec![real_map]];
        let fake = vec![vec![fake_map]];
        let loss = feature_matching_loss(real, fake)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(loss > 0.0, "expected positive loss but got {loss}");
    }

    #[test]
    fn adversarial_generator_loss_is_zero_when_scores_are_one() {
        let device = NdArrayDevice::Cpu;
        let scores = vec![Tensor::<TestBackend, 3>::ones([1, 1, 4], &device)];
        let loss = adversarial_generator_loss(scores)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(loss.abs() < 1e-6, "expected ~0 but got {loss}");
    }

    #[test]
    fn adversarial_discriminator_loss_is_zero_when_real_one_fake_zero() {
        let device = NdArrayDevice::Cpu;
        let real = vec![Tensor::<TestBackend, 3>::ones([1, 1, 4], &device)];
        let fake = vec![Tensor::<TestBackend, 3>::zeros([1, 1, 4], &device)];
        let loss = adversarial_discriminator_loss(real, fake)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(loss.abs() < 1e-6, "expected ~0 but got {loss}");
    }

    #[test]
    fn mel_spectrogram_loss_matches_manual_l1() {
        let target = tensor(&[0.0, 1.0, 2.0, 3.0], 4);
        let pred = tensor(&[1.0, 1.0, 1.0, 1.0], 4);
        let loss = mel_spectrogram_loss(target, pred)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        // Mean absolute error: (1+0+1+2)/4 = 1.0
        assert!((loss - 1.0).abs() < 1e-5, "expected 1.0 but got {loss}");
    }

    #[test]
    fn waveform_reconstruction_loss_is_zero_for_identical_tensors() {
        let device = NdArrayDevice::Cpu;
        let waveform = Tensor::<TestBackend, 3>::ones([1, 1, 16], &device);
        let loss = waveform_reconstruction_loss(waveform.clone(), waveform)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(loss.abs() < 1e-6, "expected ~0 but got {loss}");
    }

    #[test]
    fn default_loss_weights_match_hifigan_paper() {
        let weights = VocoderLossWeights::default();
        assert_eq!(weights.feature_matching, 10.0);
        assert_eq!(weights.mel_spectrogram, 45.0);
        assert_eq!(weights.waveform_reconstruction, 0.0);
        assert_eq!(weights.adversarial_generator, 1.0);
    }

    #[test]
    fn melgan_weights_have_no_mel_loss() {
        let weights = VocoderLossWeights::melgan();
        assert_eq!(weights.mel_spectrogram, 0.0);
        assert!(weights.waveform_reconstruction > 0.0);
    }
}
