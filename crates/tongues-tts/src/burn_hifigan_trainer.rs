//! HiFi-GAN training hooks: generator, MPD, and MSD working together.
//!
//! [`HifiganTrainer`] owns a generator and both discriminators and implements
//! [`BurnVocoderTrainingHooks`]. The default schedule updates generator and
//! discriminator once per batch; callers can also request a deterministic
//! discriminator-then-generator cycle.
//!
//! ## Training schedule
//!
//! - **Default (`EveryBatch`)**:
//!   1. Compute generator loss.
//!   2. Compute discriminator loss from a detached generator output.
//!   3. Return both losses with [`VocoderTrainingPhase::Joint`].
//! - **Alternating (`Cycle { discriminator_steps: 1, generator_steps: 1 }`)**:
//!   discriminator, generator, discriminator, generator, ...
//! - **Generator step**:
//!   1. Forward-pass through the generator.
//!   2. Forward-pass through MPD and MSD for both the real and generated
//!      waveforms.
//!   3. Compute adversarial generator loss + feature-matching loss.
//!   4. Return combined generator loss in [`BurnVocoderTrainingOutput::generator_loss`].
//!
//! - **Discriminator step**:
//!   1. Forward-pass through the generator (detached for discriminator update).
//!   2. Forward-pass through MPD and MSD for both real and generated.
//!   3. Compute LSGAN discriminator loss (real→1, fake→0).
//!   4. Return discriminator loss in [`BurnVocoderTrainingOutput::discriminator_loss`].

use anyhow::{anyhow, Context, Result};
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
    VocoderAdversarialUpdateSchedule, VocoderTrainingPhase, VocoderTrainingProgress,
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
    adversarial_schedule: VocoderAdversarialUpdateSchedule,
    loss_weights: VocoderLossWeightsHolder,
    mel_loss: Option<crate::VocoderMelLossConfig>,
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
        adversarial_schedule: VocoderAdversarialUpdateSchedule,
    ) -> Self {
        Self {
            mpd: MultiPeriodDiscriminator::new(device),
            msd: MultiScaleDiscriminator::new(device),
            generator,
            adversarial_schedule,
            loss_weights: VocoderLossWeightsHolder::new(&loss_weights),
            mel_loss: None,
        }
    }

    pub fn new_complete(
        generator: HifiganGenerator<B>,
        device: &B::Device,
        loss_weights: VocoderLossWeights,
        adversarial_schedule: VocoderAdversarialUpdateSchedule,
        audio: &crate::AudioFeatureConfig,
    ) -> Self {
        let mut trainer = Self::new(generator, device, loss_weights, adversarial_schedule);
        trainer.mel_loss = Some(crate::VocoderMelLossConfig::from_audio(audio));
        trainer
    }

    /// Construct a trainer with HiFi-GAN paper default loss weights.
    pub fn with_defaults(generator: HifiganGenerator<B>, device: &B::Device) -> Self {
        Self::new(
            generator,
            device,
            VocoderLossWeights::default(),
            VocoderAdversarialUpdateSchedule::EveryBatch,
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
        progress: VocoderTrainingProgress,
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

        let weights = self.loss_weights.to_weights();
        let mel_loss = if weights.mel_spectrogram > 0.0 {
            let config = self
                .mel_loss
                .as_ref()
                .context("enabled HiFi-GAN mel loss requires a differentiable mel contract")?
                .audio_config();
            let target_mel =
                crate::vits_trainer::differentiable_mel(target_waveform.clone(), &config)?;
            let generated_mel =
                crate::vits_trainer::differentiable_mel(predicted_waveform.clone(), &config)?;
            crate::mel_spectrogram_loss(target_mel, generated_mel)
        } else {
            Tensor::zeros_like(&adv_loss)
        };
        let recon_zero = Tensor::zeros_like(&adv_loss);
        let gen_loss = combined_generator_loss(adv_loss, fm_loss, mel_loss, recon_zero, &weights);

        Ok(BurnVocoderTrainingOutput {
            progress,
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
        progress: VocoderTrainingProgress,
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
            progress,
            phase: VocoderTrainingPhase::Discriminator,
            predicted_waveform,
            discriminator_outputs: None,
            generator_loss: None,
            discriminator_loss: Some(disc_loss),
        })
    }

    fn joint_step(
        &self,
        batch: BurnVocoderTrainingBatch<B>,
        progress: VocoderTrainingProgress,
    ) -> Result<BurnVocoderTrainingOutput<B>> {
        let generator = self.generator_step(batch.clone(), progress)?;
        let discriminator = self.discriminator_step(batch, progress)?;
        Ok(BurnVocoderTrainingOutput {
            progress,
            phase: VocoderTrainingPhase::Joint,
            predicted_waveform: generator.predicted_waveform,
            discriminator_outputs: None,
            generator_loss: generator.generator_loss,
            discriminator_loss: discriminator.discriminator_loss,
        })
    }
}

impl<B: Backend> BurnVocoderTrainingHooks<B> for HifiganTrainer<B> {
    fn training_phase(&self, global_step: u64) -> VocoderTrainingPhase {
        self.adversarial_schedule.training_phase(global_step)
    }

    fn training_forward(
        &self,
        batch: BurnVocoderTrainingBatch<B>,
        global_step: u64,
    ) -> Result<BurnVocoderTrainingOutput<B>> {
        let progress = self.adversarial_schedule.progress(global_step);
        match progress.phase {
            VocoderTrainingPhase::Generator => self.generator_step(batch, progress),
            VocoderTrainingPhase::Discriminator => self.discriminator_step(batch, progress),
            VocoderTrainingPhase::Joint => self.joint_step(batch, progress),
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

    fn test_audio() -> crate::AudioFeatureConfig {
        crate::AudioFeatureConfig {
            fft_size: 8,
            win_length: 8,
            hop_length: 4,
            sample_rate: 8_000,
            preemphasis: 0.0,
            log_func: "np.log".into(),
            num_mels: 4,
            mel_fmin: 0.0,
            mel_fmax: Some(4_000.0),
            spec_gain: 1.0,
            signal_norm: false,
            min_level_db: -100.0,
            ref_level_db: Some(20.0),
            symmetric_norm: true,
            max_norm: 4.0,
            clip_norm: true,
            stats_path: None,
            stats_sha256: None,
            do_amp_to_db_mel: true,
            stft_pad_mode: "reflect".into(),
            centered: true,
            stft_manual_padding: None,
        }
    }

    #[test]
    fn trainer_default_schedule_updates_both_parameter_groups() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 1);
        let gen = tiny_generator(&device);
        let trainer = HifiganTrainer::with_defaults(gen, &device);
        assert_eq!(trainer.training_phase(0), VocoderTrainingPhase::Joint);
        assert_eq!(trainer.training_phase(1), VocoderTrainingPhase::Joint);
    }

    #[test]
    fn trainer_phase_respects_alternating_cycle() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 2);
        let gen = tiny_generator(&device);
        let trainer = HifiganTrainer::new(
            gen,
            &device,
            VocoderLossWeights::default(),
            VocoderAdversarialUpdateSchedule::Cycle {
                discriminator_steps: 1,
                generator_steps: 1,
            },
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
        let trainer = HifiganTrainer::new_complete(
            gen,
            &device,
            VocoderLossWeights::default(),
            VocoderAdversarialUpdateSchedule::Cycle {
                discriminator_steps: 1,
                generator_steps: 1,
            },
            &test_audio(),
        );
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
        assert!(
            loss.is_finite(),
            "generator loss must be finite, got {loss}"
        );
        assert!(
            loss >= 0.0,
            "generator loss must be non-negative, got {loss}"
        );
    }

    #[test]
    fn discriminator_step_returns_finite_loss() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 4);
        let gen = tiny_generator(&device);
        let trainer = HifiganTrainer::new(
            gen,
            &device,
            VocoderLossWeights::default(),
            VocoderAdversarialUpdateSchedule::Cycle {
                discriminator_steps: 1,
                generator_steps: 1,
            },
        );
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
    fn default_schedule_returns_both_losses() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 5);
        let gen = tiny_generator(&device);
        let trainer = HifiganTrainer::new_complete(
            gen,
            &device,
            VocoderLossWeights::default(),
            VocoderAdversarialUpdateSchedule::EveryBatch,
            &test_audio(),
        );
        let batch = make_batch(1, 3, 4, 12, &device);
        let output = trainer.training_forward(batch, 0).expect("forward");
        assert_eq!(output.phase, VocoderTrainingPhase::Joint);
        assert!(output.generator_loss.is_some());
        assert!(output.discriminator_loss.is_some());
        assert_eq!(output.progress.generator_updates, 1);
        assert_eq!(output.progress.discriminator_updates, 1);
    }

    #[test]
    fn predicted_waveform_has_correct_upsampled_shape() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 6);
        let gen = tiny_generator(&device);
        let trainer = HifiganTrainer::new_complete(
            gen,
            &device,
            VocoderLossWeights::default(),
            VocoderAdversarialUpdateSchedule::EveryBatch,
            &test_audio(),
        );
        // 3 frames × (2×2) upsample = 12 samples
        let batch = make_batch(1, 3, 4, 12, &device);
        let output = trainer.training_forward(batch, 0).expect("forward");
        let [b, c, t] = output.predicted_waveform.dims();
        assert_eq!(b, 1);
        assert_eq!(c, 1);
        assert_eq!(t, 12);
    }
}
