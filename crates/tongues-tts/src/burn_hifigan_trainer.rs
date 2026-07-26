//! HiFi-GAN training hooks: generator, MPD, and MSD working together.
//!
//! [`HifiganTrainer`] owns a generator and both discriminators and implements
//! [`BurnVocoderTrainingHooks`].  The alternating generator / discriminator
//! schedule follows the original HiFi-GAN paper (Jungil Kong et al., 2020).
//!
//! ## Training schedule
//!
//! - **Generator step** (even global steps by default):
//!   1. Forward-pass through the generator.
//!   2. Forward-pass through MPD and MSD for both the real and generated
//!      waveforms.
//!   3. Compute adversarial generator loss + feature-matching loss.
//!   4. Return combined generator loss in [`BurnVocoderTrainingOutput::generator_loss`].
//!
//! - **Discriminator step** (odd global steps by default):
//!   1. Forward-pass through the generator (detached for discriminator update).
//!   2. Forward-pass through MPD and MSD for both real and generated.
//!   3. Compute LSGAN discriminator loss (real→1, fake→0).
//!   4. Return discriminator loss in [`BurnVocoderTrainingOutput::discriminator_loss`].

use anyhow::{anyhow, Result};
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::burn_hifigan::HifiganGenerator;
use crate::burn_vocoder_discriminators::{MultiPeriodDiscriminator, MultiScaleDiscriminator};
use crate::burn_vocoder_losses::{
    adversarial_discriminator_loss, adversarial_generator_loss, combined_generator_loss,
    feature_matching_loss, VocoderLossWeights,
};
use crate::burn_vocoder_training::{
    BurnVocoderTrainingBatch, BurnVocoderTrainingHooks, BurnVocoderTrainingOutput,
    VocoderTrainingPhase,
};

/// Bundled HiFi-GAN generator and discriminators for adversarial training.
///
/// The trainer does not own an optimiser; the outer training loop is
/// responsible for calling `.backward()` on the returned loss and stepping
/// the appropriate parameter group.
#[derive(Module, Debug)]
pub struct HifiganTrainer<B: Backend> {
    pub generator: HifiganGenerator<B>,
    pub mpd: MultiPeriodDiscriminator<B>,
    pub msd: MultiScaleDiscriminator<B>,
    /// Relative frequency of discriminator updates.  A value of `1` means both
    /// networks are updated every step; `n` means the discriminator is updated
    /// once per `n` generator updates.
    disc_update_interval: u64,
    loss_weights: VocoderLossWeightsHolder,
}

/// A non-`Module` wrapper for [`VocoderLossWeights`] so the trainer can carry
/// loss configuration without those weights appearing in checkpoint saves.
#[derive(Module, Debug, Clone)]
struct VocoderLossWeightsHolder {
    feature_matching: f64,
    mel_spectrogram: f64,
    waveform_reconstruction: f64,
    adversarial_generator: f64,
}

impl VocoderLossWeightsHolder {
    fn new(w: &VocoderLossWeights) -> Self {
        Self {
            feature_matching: w.feature_matching,
            mel_spectrogram: w.mel_spectrogram,
            waveform_reconstruction: w.waveform_reconstruction,
            adversarial_generator: w.adversarial_generator,
        }
    }

    fn to_weights(&self) -> VocoderLossWeights {
        VocoderLossWeights {
            feature_matching: self.feature_matching,
            mel_spectrogram: self.mel_spectrogram,
            waveform_reconstruction: self.waveform_reconstruction,
            adversarial_generator: self.adversarial_generator,
        }
    }
}

impl<B: Backend> HifiganTrainer<B> {
    /// Construct a trainer from pre-initialised modules.
    pub fn new(
        generator: HifiganGenerator<B>,
        device: &B::Device,
        loss_weights: VocoderLossWeights,
        disc_update_interval: u64,
    ) -> Self {
        Self {
            mpd: MultiPeriodDiscriminator::new(device),
            msd: MultiScaleDiscriminator::new(device),
            generator,
            disc_update_interval: disc_update_interval.max(1),
            loss_weights: VocoderLossWeightsHolder::new(&loss_weights),
        }
    }

    /// Construct a trainer with HiFi-GAN paper default loss weights.
    pub fn with_defaults(generator: HifiganGenerator<B>, device: &B::Device) -> Self {
        Self::new(
            generator,
            device,
            VocoderLossWeights::default(),
            1,
        )
    }

    /// Run the generator forward pass.
    ///
    /// `conditioning_mel` is `[batch, frames, mel_bins]` (frame-major).
    /// Returns the generated waveform `[batch, 1, samples]`.
    pub fn generate(&self, conditioning_mel: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        // Generator expects [batch, mel_bins, frames].
        let mel = conditioning_mel.swap_dims(1, 2);
        self.generator
            .forward(mel, None)
            .map_err(|e| anyhow!("HiFi-GAN generator forward failed: {e}"))
    }

    fn generator_step(
        &self,
        batch: BurnVocoderTrainingBatch<B>,
    ) -> Result<BurnVocoderTrainingOutput<B>> {
        let target_waveform = batch.target_waveform;
        let predicted_waveform = self.generate(batch.conditioning_mel)?;

        // Discriminator forward on real (detached from generator graph) and fake.
        let real = target_waveform.clone().detach();
        let mpd_real = self.mpd.forward(real.clone());
        let mpd_fake = self.mpd.forward(predicted_waveform.clone());
        let msd_real = self.msd.forward(real);
        let msd_fake = self.msd.forward(predicted_waveform.clone());

        // Adversarial generator loss.
        let mut fake_scores = mpd_fake.scores();
        fake_scores.extend(msd_fake.scores());
        let adv_loss = adversarial_generator_loss(fake_scores);

        // Feature-matching loss.
        let mut real_fmaps = mpd_real.feature_maps();
        real_fmaps.extend(msd_real.feature_maps());
        let mut fake_fmaps = mpd_fake.feature_maps();
        fake_fmaps.extend(msd_fake.feature_maps());
        let fm_loss = feature_matching_loss(real_fmaps, fake_fmaps);

        // Mel loss is omitted here because it requires a Burn-native STFT.
        // The caller can compute it externally using the returned predicted_waveform
        // and `burn_vocoder_losses::mel_spectrogram_loss`.
        let mel_zero = Tensor::zeros_like(&adv_loss);
        let recon_zero = Tensor::zeros_like(&adv_loss);
        let weights = self.loss_weights.to_weights();
        let gen_loss = combined_generator_loss(adv_loss, fm_loss, mel_zero, recon_zero, &weights);

        Ok(BurnVocoderTrainingOutput {
            phase: VocoderTrainingPhase::Generator,
            predicted_waveform,
            discriminator_outputs: None,
            generator_loss: Some(gen_loss),
            discriminator_loss: None,
        })
    }

    fn discriminator_step(
        &self,
        batch: BurnVocoderTrainingBatch<B>,
    ) -> Result<BurnVocoderTrainingOutput<B>> {
        let target_waveform = batch.target_waveform;
        // Generator output is detached so that discriminator gradients do not
        // flow back through the generator.
        let predicted_waveform = self.generate(batch.conditioning_mel)?.detach();

        let mpd_real = self.mpd.forward(target_waveform.clone());
        let mpd_fake = self.mpd.forward(predicted_waveform.clone());
        let msd_real = self.msd.forward(target_waveform.clone());
        let msd_fake = self.msd.forward(predicted_waveform.clone());

        let mut real_scores = mpd_real.scores();
        real_scores.extend(msd_real.scores());
        let mut fake_scores = mpd_fake.scores();
        fake_scores.extend(msd_fake.scores());

        let disc_loss = adversarial_discriminator_loss(real_scores, fake_scores);

        Ok(BurnVocoderTrainingOutput {
            phase: VocoderTrainingPhase::Discriminator,
            predicted_waveform,
            discriminator_outputs: None,
            generator_loss: None,
            discriminator_loss: Some(disc_loss),
        })
    }
}

impl<B: Backend> BurnVocoderTrainingHooks<B> for HifiganTrainer<B> {
    /// Return the phase for a given global training step.
    ///
    /// The discriminator is updated every `disc_update_interval` steps.  On all
    /// other steps the generator is updated.
    fn training_phase(&self, global_step: u64) -> VocoderTrainingPhase {
        if global_step % self.disc_update_interval == 0 {
            VocoderTrainingPhase::Discriminator
        } else {
            VocoderTrainingPhase::Generator
        }
    }

    fn training_forward(
        &self,
        batch: BurnVocoderTrainingBatch<B>,
        global_step: u64,
    ) -> Result<BurnVocoderTrainingOutput<B>> {
        match self.training_phase(global_step) {
            VocoderTrainingPhase::Generator => self.generator_step(batch),
            VocoderTrainingPhase::Discriminator => self.discriminator_step(batch),
            VocoderTrainingPhase::Joint => {
                // Not used by HiFi-GAN; fall back to generator step.
                self.generator_step(batch)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    fn tiny_generator(device: &NdArrayDevice) -> HifiganGenerator<TestBackend> {
        use crate::burn_hifigan::HifiganGeneratorConfig;
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
        .init(device)
        .expect("tiny HiFi-GAN generator")
    }

    fn make_batch(
        batch: usize,
        frames: usize,
        mel_bins: usize,
        samples: usize,
        device: &NdArrayDevice,
    ) -> BurnVocoderTrainingBatch<TestBackend> {
        BurnVocoderTrainingBatch {
            conditioning_mel: Tensor::ones([batch, frames, mel_bins], device),
            target_waveform: Tensor::ones([batch, 1, samples], device),
        }
    }

    #[test]
    fn trainer_phase_alternates_correctly() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 1);
        let gen = tiny_generator(&device);
        let trainer = HifiganTrainer::with_defaults(gen, &device);
        // disc_update_interval = 1 → every step is discriminator step
        assert_eq!(
            trainer.training_phase(0),
            VocoderTrainingPhase::Discriminator
        );
        assert_eq!(
            trainer.training_phase(1),
            VocoderTrainingPhase::Discriminator
        );
    }

    #[test]
    fn trainer_phase_skips_discriminator_with_interval() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 2);
        let gen = tiny_generator(&device);
        let trainer = HifiganTrainer::new(
            gen,
            &device,
            VocoderLossWeights::default(),
            2, // discriminator every 2 steps
        );
        assert_eq!(
            trainer.training_phase(0),
            VocoderTrainingPhase::Discriminator
        );
        assert_eq!(trainer.training_phase(1), VocoderTrainingPhase::Generator);
        assert_eq!(
            trainer.training_phase(2),
            VocoderTrainingPhase::Discriminator
        );
    }

    #[test]
    fn generator_step_returns_finite_loss() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 3);
        let gen = tiny_generator(&device);
        // Use interval 2 so step 1 is a generator step.
        let trainer = HifiganTrainer::new(gen, &device, VocoderLossWeights::default(), 2);
        let batch = make_batch(1, 3, 4, 12, &device);
        let output = trainer.training_forward(batch, 1).expect("forward");
        assert_eq!(output.phase, VocoderTrainingPhase::Generator);
        assert!(output.generator_loss.is_some());
        let loss = output
            .generator_loss
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(loss.is_finite(), "generator loss must be finite, got {loss}");
        assert!(loss >= 0.0, "generator loss must be non-negative, got {loss}");
    }

    #[test]
    fn discriminator_step_returns_finite_loss() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 4);
        let gen = tiny_generator(&device);
        let trainer = HifiganTrainer::with_defaults(gen, &device);
        let batch = make_batch(1, 3, 4, 12, &device);
        let output = trainer.training_forward(batch, 0).expect("forward");
        assert_eq!(output.phase, VocoderTrainingPhase::Discriminator);
        assert!(output.discriminator_loss.is_some());
        let loss = output
            .discriminator_loss
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            loss.is_finite(),
            "discriminator loss must be finite, got {loss}"
        );
    }

    #[test]
    fn predicted_waveform_has_correct_upsampled_shape() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 5);
        let gen = tiny_generator(&device);
        let trainer = HifiganTrainer::with_defaults(gen, &device);
        // 3 frames × (2×2) upsample = 12 samples
        let batch = make_batch(1, 3, 4, 12, &device);
        let output = trainer.training_forward(batch, 0).expect("forward");
        let [b, c, t] = output.predicted_waveform.dims();
        assert_eq!(b, 1);
        assert_eq!(c, 1);
        assert_eq!(t, 12);
    }
}
