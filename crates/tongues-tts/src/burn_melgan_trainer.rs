//! MelGAN and MultiBand-MelGAN training hooks.
//!
//! [`MelganTrainer`] and [`MultibandMelganTrainer`] own a generator and a
//! multi-scale discriminator and implement [`BurnVocoderTrainingHooks`].  Both
//! follow the MelGAN training schedule (Kumar et al., 2019):
//!
//! - **Generator step**: generate waveform, compute adversarial + feature-
//!   matching losses.
//! - **Discriminator step**: compute LSGAN discriminator loss on real vs. fake.
//!
//! MultiBand-MelGAN additionally applies the PQMF synthesis bank so that the
//! discriminator always sees full-bandwidth audio.

use anyhow::{anyhow, Result};
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::burn_melgan::{MelganGenerator, MultibandMelganGenerator};
use crate::burn_vocoder_discriminators::MultiScaleDiscriminator;
use crate::burn_vocoder_losses::{
    adversarial_discriminator_loss, adversarial_generator_loss, combined_generator_loss,
    feature_matching_loss, waveform_reconstruction_loss, VocoderLossWeights,
};
use crate::burn_vocoder_training::{
    BurnVocoderTrainingBatch, BurnVocoderTrainingHooks, BurnVocoderTrainingOutput,
    VocoderTrainingPhase,
};

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Opaque wrapper so `VocoderLossWeights` can live inside a `Module` derive.
#[derive(Module, Debug, Clone)]
struct LossWeightsHolder {
    feature_matching: f64,
    mel_spectrogram: f64,
    waveform_reconstruction: f64,
    adversarial_generator: f64,
}

impl LossWeightsHolder {
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

fn generator_output<B: Backend>(
    msd: &MultiScaleDiscriminator<B>,
    target_waveform: Tensor<B, 3>,
    predicted_waveform: Tensor<B, 3>,
    loss_weights: &LossWeightsHolder,
) -> Result<BurnVocoderTrainingOutput<B>> {
    let weights = loss_weights.to_weights();
    let real = target_waveform.clone().detach();
    let msd_real = msd.forward(real.clone());
    let msd_fake = msd.forward(predicted_waveform.clone());

    let adv_loss = adversarial_generator_loss(msd_fake.scores());
    let fm_loss = feature_matching_loss(msd_real.feature_maps(), msd_fake.feature_maps());

    // Waveform reconstruction loss (L1).  Requires the same length; the
    // caller is responsible for ensuring both tensors are the same shape.
    let recon_loss = if weights.waveform_reconstruction > 0.0 {
        let real_dims = real.dims();
        let fake_dims = predicted_waveform.dims();
        let min_len = real_dims[2].min(fake_dims[2]);
        let real_slice = real.slice([0..real_dims[0], 0..1, 0..min_len]);
        let fake_slice = predicted_waveform
            .clone()
            .slice([0..fake_dims[0], 0..1, 0..min_len]);
        waveform_reconstruction_loss(real_slice, fake_slice)
    } else {
        Tensor::zeros_like(&adv_loss)
    };

    let mel_zero = Tensor::zeros_like(&adv_loss);
    let gen_loss = combined_generator_loss(adv_loss, fm_loss, mel_zero, recon_loss, &weights);

    Ok(BurnVocoderTrainingOutput {
        phase: VocoderTrainingPhase::Generator,
        predicted_waveform,
        discriminator_outputs: None,
        generator_loss: Some(gen_loss),
        discriminator_loss: None,
    })
}

fn discriminator_output<B: Backend>(
    msd: &MultiScaleDiscriminator<B>,
    target_waveform: Tensor<B, 3>,
    predicted_waveform: Tensor<B, 3>,
) -> Result<BurnVocoderTrainingOutput<B>> {
    let msd_real = msd.forward(target_waveform);
    let msd_fake = msd.forward(predicted_waveform.clone().detach());

    let disc_loss = adversarial_discriminator_loss(msd_real.scores(), msd_fake.scores());

    Ok(BurnVocoderTrainingOutput {
        phase: VocoderTrainingPhase::Discriminator,
        predicted_waveform,
        discriminator_outputs: None,
        generator_loss: None,
        discriminator_loss: Some(disc_loss),
    })
}

// ── Plain MelGAN ──────────────────────────────────────────────────────────────

/// Bundled plain MelGAN generator and multi-scale discriminator.
#[derive(Module, Debug)]
pub struct MelganTrainer<B: Backend> {
    pub generator: MelganGenerator<B>,
    pub msd: MultiScaleDiscriminator<B>,
    disc_update_interval: u64,
    loss_weights: LossWeightsHolder,
}

impl<B: Backend> MelganTrainer<B> {
    /// Construct a trainer from a pre-initialised generator.
    pub fn new(
        generator: MelganGenerator<B>,
        device: &B::Device,
        loss_weights: VocoderLossWeights,
        disc_update_interval: u64,
    ) -> Self {
        Self {
            msd: MultiScaleDiscriminator::new(device),
            generator,
            disc_update_interval: disc_update_interval.max(1),
            loss_weights: LossWeightsHolder::new(&loss_weights),
        }
    }

    /// Construct a trainer with MelGAN paper default loss weights.
    pub fn with_defaults(generator: MelganGenerator<B>, device: &B::Device) -> Self {
        Self::new(generator, device, VocoderLossWeights::melgan(), 1)
    }

    fn generate(&self, conditioning_mel: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        // Generator expects [batch, mel_bins, frames].
        let mel = conditioning_mel.swap_dims(1, 2);
        self.generator
            .forward(mel)
            .map_err(|e| anyhow!("MelGAN generator forward failed: {e}"))
    }
}

impl<B: Backend> BurnVocoderTrainingHooks<B> for MelganTrainer<B> {
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
        let predicted = self.generate(batch.conditioning_mel)?;
        match self.training_phase(global_step) {
            VocoderTrainingPhase::Generator | VocoderTrainingPhase::Joint => generator_output(
                &self.msd,
                batch.target_waveform,
                predicted,
                &self.loss_weights,
            ),
            VocoderTrainingPhase::Discriminator => {
                discriminator_output(&self.msd, batch.target_waveform, predicted)
            }
        }
    }
}

// ── MultiBand-MelGAN ──────────────────────────────────────────────────────────

/// Bundled MultiBand-MelGAN generator and multi-scale discriminator.
///
/// The generator produces subbands that are synthesised to full-bandwidth audio
/// by the embedded PQMF bank before the discriminator sees them.
#[derive(Module, Debug)]
pub struct MultibandMelganTrainer<B: Backend> {
    pub generator: MultibandMelganGenerator<B>,
    pub msd: MultiScaleDiscriminator<B>,
    disc_update_interval: u64,
    loss_weights: LossWeightsHolder,
}

impl<B: Backend> MultibandMelganTrainer<B> {
    /// Construct a trainer from a pre-initialised MultiBand-MelGAN generator.
    pub fn new(
        generator: MultibandMelganGenerator<B>,
        device: &B::Device,
        loss_weights: VocoderLossWeights,
        disc_update_interval: u64,
    ) -> Self {
        Self {
            msd: MultiScaleDiscriminator::new(device),
            generator,
            disc_update_interval: disc_update_interval.max(1),
            loss_weights: LossWeightsHolder::new(&loss_weights),
        }
    }

    /// Construct a trainer with MelGAN paper default loss weights.
    pub fn with_defaults(generator: MultibandMelganGenerator<B>, device: &B::Device) -> Self {
        Self::new(generator, device, VocoderLossWeights::melgan(), 1)
    }

    /// Generate full-bandwidth waveform via PQMF synthesis.
    fn generate(&self, conditioning_mel: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let mel = conditioning_mel.swap_dims(1, 2);
        self.generator
            .inference(mel)
            .map_err(|e| anyhow!("MultiBand-MelGAN inference failed: {e}"))
    }
}

impl<B: Backend> BurnVocoderTrainingHooks<B> for MultibandMelganTrainer<B> {
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
        let predicted = self.generate(batch.conditioning_mel)?;
        match self.training_phase(global_step) {
            VocoderTrainingPhase::Generator | VocoderTrainingPhase::Joint => generator_output(
                &self.msd,
                batch.target_waveform,
                predicted,
                &self.loss_weights,
            ),
            VocoderTrainingPhase::Discriminator => {
                discriminator_output(&self.msd, batch.target_waveform, predicted)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    use crate::burn_melgan::{MelganGeneratorConfig, PqmfConfig};

    type TestBackend = NdArray<f32>;

    fn tiny_melgan_config(out_channels: usize) -> MelganGeneratorConfig {
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
    fn melgan_generator_step_returns_finite_loss() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 10);
        let gen = tiny_melgan_config(1)
            .init::<TestBackend>(&device)
            .expect("generator");
        let trainer = MelganTrainer::new(gen, &device, VocoderLossWeights::melgan(), 2);

        // Step 1 → generator phase (interval 2 → step 0 is disc, step 1 is gen).
        // upsample = 2×2 = 4; 3 frames × 4 = 12 samples.
        let batch = make_batch(1, 3, 4, 12, &device);
        let output = trainer.training_forward(batch, 1).expect("forward");
        assert_eq!(output.phase, VocoderTrainingPhase::Generator);
        let loss = output
            .generator_loss
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            loss.is_finite(),
            "generator loss must be finite, got {loss}"
        );
    }

    #[test]
    fn melgan_discriminator_step_returns_finite_loss() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 11);
        let gen = tiny_melgan_config(1)
            .init::<TestBackend>(&device)
            .expect("generator");
        let trainer = MelganTrainer::with_defaults(gen, &device);
        // upsample = 2×2 = 4; 3 frames × 4 = 12 samples.
        let batch = make_batch(1, 3, 4, 12, &device);
        let output = trainer.training_forward(batch, 0).expect("forward");
        assert_eq!(output.phase, VocoderTrainingPhase::Discriminator);
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
    fn multiband_melgan_discriminator_step_returns_finite_loss() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 12);
        let gen = tiny_melgan_config(4)
            .init_multiband::<TestBackend>(PqmfConfig::default(), &device)
            .expect("generator");
        let trainer = MultibandMelganTrainer::with_defaults(gen, &device);
        // batch with samples = (frames + 2*padding) * upsample * subbands
        // (3 + 4) * 4 * 4 = 112
        let batch = make_batch(1, 3, 4, 112, &device);
        let output = trainer.training_forward(batch, 0).expect("forward");
        assert_eq!(output.phase, VocoderTrainingPhase::Discriminator);
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
    fn melgan_phase_schedule_respects_disc_interval() {
        let device = NdArrayDevice::Cpu;
        let gen = tiny_melgan_config(1)
            .init::<TestBackend>(&device)
            .expect("generator");
        let trainer = MelganTrainer::new(gen, &device, VocoderLossWeights::melgan(), 3);
        assert_eq!(
            trainer.training_phase(0),
            VocoderTrainingPhase::Discriminator
        );
        assert_eq!(trainer.training_phase(1), VocoderTrainingPhase::Generator);
        assert_eq!(trainer.training_phase(2), VocoderTrainingPhase::Generator);
        assert_eq!(
            trainer.training_phase(3),
            VocoderTrainingPhase::Discriminator
        );
    }
}
