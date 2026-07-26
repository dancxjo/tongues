use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::profiling::{finish_backend_stage, record_load_stage};
use crate::{
    HifiganBundleConfig, HifiganGenerator, InferenceRuntime, MelganBundleConfig, MelganGenerator,
    ModelLoadProfileEvent, ModelLoadStage, MultibandMelganGenerator, NeuralVocoder, Spectrogram,
    SpectrogramContract, SpectrogramLayout, SynthesisDimension, SynthesisProfiler, SynthesisStage,
    Waveform, WaveformContract,
};

pub trait BurnTensorVocoder<B: Backend>: NeuralVocoder {
    fn synthesize_tensor(
        &self,
        spectrogram: Tensor<B, 3>,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<Tensor<B, 3>>;
}

pub enum BurnVocoder<B: Backend> {
    Hifigan(BurnHifiganVocoder<B>),
    Melgan(BurnMelganVocoder<B>),
    MultibandMelgan(BurnMultibandMelganVocoder<B>),
}

impl<B: Backend> BurnVocoder<B> {
    pub fn load(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
    ) -> Result<Self> {
        Self::load_internal(config_path, checkpoint_path, device, None)
    }

    pub fn load_profiled(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
        profiler: &mut dyn FnMut(ModelLoadProfileEvent),
    ) -> Result<Self> {
        Self::load_internal(config_path, checkpoint_path, device, Some(profiler))
    }

    fn load_internal(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
        profiler: Option<&mut dyn FnMut(ModelLoadProfileEvent)>,
    ) -> Result<Self> {
        let config_path = config_path.as_ref();
        let checkpoint_path = checkpoint_path.as_ref();
        let source = fs::read_to_string(config_path)
            .with_context(|| format!("failed to read vocoder config {}", config_path.display()))?;
        let root: serde_json::Value =
            json5::from_str(&source).context("failed to inspect vocoder config")?;
        let generator = root
            .get("generator_model")
            .and_then(serde_json::Value::as_str)
            .context("vocoder config has no string `generator_model`")?;
        match (generator, profiler) {
            ("hifigan_generator", Some(profiler)) => Ok(Self::Hifigan(
                BurnHifiganVocoder::load_profiled(config_path, checkpoint_path, device, profiler)?,
            )),
            ("hifigan_generator", None) => Ok(Self::Hifigan(BurnHifiganVocoder::load(
                config_path,
                checkpoint_path,
                device,
            )?)),
            ("melgan_generator", Some(profiler)) => Ok(Self::Melgan(
                BurnMelganVocoder::load_profiled(config_path, checkpoint_path, device, profiler)?,
            )),
            ("melgan_generator", None) => Ok(Self::Melgan(BurnMelganVocoder::load(
                config_path,
                checkpoint_path,
                device,
            )?)),
            ("multiband_melgan_generator", Some(profiler)) => Ok(Self::MultibandMelgan(
                BurnMultibandMelganVocoder::load_profiled(
                    config_path,
                    checkpoint_path,
                    device,
                    profiler,
                )?,
            )),
            ("multiband_melgan_generator", None) => Ok(Self::MultibandMelgan(
                BurnMultibandMelganVocoder::load(config_path, checkpoint_path, device)?,
            )),
            (other, _) => anyhow::bail!("unsupported native vocoder generator `{other}`"),
        }
    }
}

impl<B: Backend> NeuralVocoder for BurnVocoder<B> {
    fn runtime(&self) -> InferenceRuntime {
        InferenceRuntime::Burn
    }

    fn input_contract(&self) -> &SpectrogramContract {
        match self {
            Self::Hifigan(vocoder) => vocoder.input_contract(),
            Self::Melgan(vocoder) => vocoder.input_contract(),
            Self::MultibandMelgan(vocoder) => vocoder.input_contract(),
        }
    }

    fn output_contract(&self) -> WaveformContract {
        match self {
            Self::Hifigan(vocoder) => vocoder.output_contract(),
            Self::Melgan(vocoder) => vocoder.output_contract(),
            Self::MultibandMelgan(vocoder) => vocoder.output_contract(),
        }
    }

    fn synthesize(&mut self, spectrogram: &Spectrogram) -> Result<Waveform> {
        match self {
            Self::Hifigan(vocoder) => vocoder.synthesize(spectrogram),
            Self::Melgan(vocoder) => vocoder.synthesize(spectrogram),
            Self::MultibandMelgan(vocoder) => vocoder.synthesize(spectrogram),
        }
    }
}

impl<B: Backend> BurnTensorVocoder<B> for BurnVocoder<B> {
    fn synthesize_tensor(
        &self,
        spectrogram: Tensor<B, 3>,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<Tensor<B, 3>> {
        match self {
            Self::Hifigan(vocoder) => vocoder.synthesize_tensor(spectrogram, profiler),
            Self::Melgan(vocoder) => vocoder.synthesize_tensor(spectrogram, profiler),
            Self::MultibandMelgan(vocoder) => vocoder.synthesize_tensor(spectrogram, profiler),
        }
    }
}

/// Burn-native HiFi-GAN adapter at the shared acoustic boundary.
pub struct BurnHifiganVocoder<B: Backend> {
    generator: HifiganGenerator<B>,
    input_contract: SpectrogramContract,
    output_contract: WaveformContract,
    device: B::Device,
}

impl<B: Backend> BurnHifiganVocoder<B> {
    pub fn from_generator(
        config: HifiganBundleConfig,
        generator: HifiganGenerator<B>,
        device: B::Device,
    ) -> Result<Self> {
        let input_contract = config.input_contract()?;
        ensure!(
            input_contract.layout == SpectrogramLayout::FramesByBins,
            "Burn HiFi-GAN adapter requires frame-major shared spectrograms"
        );
        Ok(Self {
            generator,
            output_contract: WaveformContract::mono(input_contract.sample_rate_hz),
            input_contract,
            device,
        })
    }

    pub fn load(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
    ) -> Result<Self> {
        Self::load_internal(config_path, checkpoint_path, device, None)
    }

    pub fn load_profiled(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
        profiler: &mut dyn FnMut(ModelLoadProfileEvent),
    ) -> Result<Self> {
        Self::load_internal(config_path, checkpoint_path, device, Some(profiler))
    }

    fn load_internal(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
        profiler: Option<&mut dyn FnMut(ModelLoadProfileEvent)>,
    ) -> Result<Self> {
        let mut profiler = profiler;
        let started = Instant::now();
        let config = HifiganBundleConfig::from_file(config_path)?;
        let input_contract = config.input_contract()?;
        ensure!(
            input_contract.layout == SpectrogramLayout::FramesByBins,
            "Burn HiFi-GAN adapter requires frame-major shared spectrograms"
        );
        let output_contract = WaveformContract::mono(input_contract.sample_rate_hz);
        record_load_stage(
            &mut profiler,
            ModelLoadStage::ConfigCheckpointParsing,
            started,
            Some("hifigan"),
        );

        let started = Instant::now();
        let generator: HifiganGenerator<B> = config.init_burn_generator(&device)?;
        record_load_stage(
            &mut profiler,
            ModelLoadStage::ModelConstruction,
            started,
            Some("hifigan"),
        );

        let started = Instant::now();
        let generator = config.load_burn_generator_checkpoint(generator, checkpoint_path)?;
        record_load_stage(
            &mut profiler,
            ModelLoadStage::WeightUpload,
            started,
            Some("hifigan"),
        );
        let vocoder = Self::from_generator(config, generator, device)?;
        ensure!(
            vocoder.output_contract == output_contract && vocoder.input_contract == input_contract,
            "HiFi-GAN constructed contracts changed during loading"
        );
        Ok(vocoder)
    }

    pub fn generator(&self) -> &HifiganGenerator<B> {
        &self.generator
    }

    pub fn synthesize_tensor(
        &self,
        spectrogram: Tensor<B, 3>,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<Tensor<B, 3>> {
        let mut profiler = profiler;
        let [batch, frames, bins] = spectrogram.dims();
        ensure!(
            batch == 1,
            "Burn HiFi-GAN currently requires batch size one"
        );
        ensure!(
            bins == self.input_contract.bins,
            "Burn HiFi-GAN expected {} mel bins, got {bins}",
            self.input_contract.bins
        );
        ensure!(frames > 0, "Burn HiFi-GAN requires at least one mel frame");

        let started = Instant::now();
        let waveform = self
            .generator
            .inference(spectrogram.swap_dims(1, 2))
            .map_err(anyhow::Error::new)
            .context("Burn HiFi-GAN inference failed")?;
        finish_backend_stage::<B>(
            &mut profiler,
            &self.device,
            SynthesisStage::WaveformDecoder,
            started,
            [
                SynthesisDimension::new("mel_frames", frames),
                SynthesisDimension::new("samples", waveform.dims()[2]),
            ],
        )?;
        Ok(waveform)
    }
}

impl<B: Backend> NeuralVocoder for BurnHifiganVocoder<B> {
    fn runtime(&self) -> InferenceRuntime {
        InferenceRuntime::Burn
    }

    fn input_contract(&self) -> &SpectrogramContract {
        &self.input_contract
    }

    fn output_contract(&self) -> WaveformContract {
        self.output_contract.clone()
    }

    fn synthesize(&mut self, spectrogram: &Spectrogram) -> Result<Waveform> {
        spectrogram
            .contract
            .ensure_compatible_with(&self.input_contract)?;
        spectrogram.validate()?;

        let features = Tensor::<B, 3>::from_data(
            TensorData::new(
                spectrogram.values.clone(),
                [1, spectrogram.frames, self.input_contract.bins],
            ),
            &self.device,
        )
        .swap_dims(1, 2);
        let waveform = self
            .generator
            .inference(features)
            .map_err(anyhow::Error::new)
            .context("Burn HiFi-GAN inference failed")?;
        let samples = waveform
            .into_data()
            .to_vec::<f32>()
            .context("Burn HiFi-GAN output is not f32")?;
        let waveform = Waveform {
            contract: self.output_contract.clone(),
            samples,
        };
        waveform.validate()?;
        Ok(waveform)
    }
}

impl<B: Backend> BurnTensorVocoder<B> for BurnHifiganVocoder<B> {
    fn synthesize_tensor(
        &self,
        spectrogram: Tensor<B, 3>,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<Tensor<B, 3>> {
        BurnHifiganVocoder::synthesize_tensor(self, spectrogram, profiler)
    }
}

/// Burn-native Coqui MelGAN adapter at the shared acoustic boundary.
pub struct BurnMelganVocoder<B: Backend> {
    generator: MelganGenerator<B>,
    input_contract: SpectrogramContract,
    output_contract: WaveformContract,
    device: B::Device,
}

impl<B: Backend> BurnMelganVocoder<B> {
    pub fn load(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
    ) -> Result<Self> {
        Self::load_internal(config_path, checkpoint_path, device, None)
    }

    pub fn load_profiled(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
        profiler: &mut dyn FnMut(ModelLoadProfileEvent),
    ) -> Result<Self> {
        Self::load_internal(config_path, checkpoint_path, device, Some(profiler))
    }

    fn load_internal(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
        profiler: Option<&mut dyn FnMut(ModelLoadProfileEvent)>,
    ) -> Result<Self> {
        let mut profiler = profiler;
        let started = Instant::now();
        let config = MelganBundleConfig::from_file(config_path)?;
        ensure!(
            config.variant()? == crate::MelganVariant::Melgan,
            "expected a plain MelGAN config"
        );
        let input_contract = config.input_contract()?;
        ensure!(
            input_contract.layout == SpectrogramLayout::FramesByBins,
            "Burn MelGAN adapter requires frame-major shared spectrograms"
        );
        let output_contract = WaveformContract::mono(input_contract.sample_rate_hz);
        record_load_stage(
            &mut profiler,
            ModelLoadStage::ConfigCheckpointParsing,
            started,
            Some("melgan"),
        );

        let started = Instant::now();
        let generator: MelganGenerator<B> = config.init_burn_generator(&device)?;
        record_load_stage(
            &mut profiler,
            ModelLoadStage::ModelConstruction,
            started,
            Some("melgan"),
        );

        let started = Instant::now();
        let generator = config.load_burn_generator_checkpoint(generator, checkpoint_path)?;
        record_load_stage(
            &mut profiler,
            ModelLoadStage::WeightUpload,
            started,
            Some("melgan"),
        );
        Ok(Self {
            generator,
            input_contract,
            output_contract,
            device,
        })
    }

    pub fn generator(&self) -> &MelganGenerator<B> {
        &self.generator
    }
}

impl<B: Backend> BurnTensorVocoder<B> for BurnMelganVocoder<B> {
    fn synthesize_tensor(
        &self,
        spectrogram: Tensor<B, 3>,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<Tensor<B, 3>> {
        synthesize_melgan_tensor(
            &self.input_contract,
            &self.device,
            spectrogram,
            profiler,
            |features| {
                self.generator
                    .inference(features)
                    .map_err(anyhow::Error::new)
            },
        )
    }
}

impl<B: Backend> NeuralVocoder for BurnMelganVocoder<B> {
    fn runtime(&self) -> InferenceRuntime {
        InferenceRuntime::Burn
    }

    fn input_contract(&self) -> &SpectrogramContract {
        &self.input_contract
    }

    fn output_contract(&self) -> WaveformContract {
        self.output_contract.clone()
    }

    fn synthesize(&mut self, spectrogram: &Spectrogram) -> Result<Waveform> {
        synthesize_host_tensor(
            &self.input_contract,
            &self.output_contract,
            &self.device,
            spectrogram,
            |features| <Self as BurnTensorVocoder<B>>::synthesize_tensor(self, features, None),
        )
    }
}

/// Burn-native Coqui MultiBand-MelGAN adapter including PQMF synthesis.
pub struct BurnMultibandMelganVocoder<B: Backend> {
    generator: MultibandMelganGenerator<B>,
    input_contract: SpectrogramContract,
    output_contract: WaveformContract,
    device: B::Device,
}

impl<B: Backend> BurnMultibandMelganVocoder<B> {
    pub fn load(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
    ) -> Result<Self> {
        Self::load_internal(config_path, checkpoint_path, device, None)
    }

    pub fn load_profiled(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
        profiler: &mut dyn FnMut(ModelLoadProfileEvent),
    ) -> Result<Self> {
        Self::load_internal(config_path, checkpoint_path, device, Some(profiler))
    }

    fn load_internal(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
        profiler: Option<&mut dyn FnMut(ModelLoadProfileEvent)>,
    ) -> Result<Self> {
        let mut profiler = profiler;
        let started = Instant::now();
        let config = MelganBundleConfig::from_file(config_path)?;
        ensure!(
            config.variant()? == crate::MelganVariant::Multiband,
            "expected a MultiBand-MelGAN config"
        );
        let input_contract = config.input_contract()?;
        ensure!(
            input_contract.layout == SpectrogramLayout::FramesByBins,
            "Burn MultiBand-MelGAN adapter requires frame-major shared spectrograms"
        );
        let output_contract = WaveformContract::mono(input_contract.sample_rate_hz);
        record_load_stage(
            &mut profiler,
            ModelLoadStage::ConfigCheckpointParsing,
            started,
            Some("multiband-melgan"),
        );

        let started = Instant::now();
        let generator: MultibandMelganGenerator<B> =
            config.init_burn_multiband_generator(&device)?;
        record_load_stage(
            &mut profiler,
            ModelLoadStage::ModelConstruction,
            started,
            Some("multiband-melgan"),
        );

        let started = Instant::now();
        let generator =
            config.load_burn_multiband_generator_checkpoint(generator, checkpoint_path)?;
        record_load_stage(
            &mut profiler,
            ModelLoadStage::WeightUpload,
            started,
            Some("multiband-melgan"),
        );
        Ok(Self {
            generator,
            input_contract,
            output_contract,
            device,
        })
    }

    pub fn generator(&self) -> &MultibandMelganGenerator<B> {
        &self.generator
    }
}

impl<B: Backend> BurnTensorVocoder<B> for BurnMultibandMelganVocoder<B> {
    fn synthesize_tensor(
        &self,
        spectrogram: Tensor<B, 3>,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<Tensor<B, 3>> {
        synthesize_melgan_tensor(
            &self.input_contract,
            &self.device,
            spectrogram,
            profiler,
            |features| {
                self.generator
                    .inference(features)
                    .map_err(anyhow::Error::new)
            },
        )
    }
}

impl<B: Backend> NeuralVocoder for BurnMultibandMelganVocoder<B> {
    fn runtime(&self) -> InferenceRuntime {
        InferenceRuntime::Burn
    }

    fn input_contract(&self) -> &SpectrogramContract {
        &self.input_contract
    }

    fn output_contract(&self) -> WaveformContract {
        self.output_contract.clone()
    }

    fn synthesize(&mut self, spectrogram: &Spectrogram) -> Result<Waveform> {
        synthesize_host_tensor(
            &self.input_contract,
            &self.output_contract,
            &self.device,
            spectrogram,
            |features| <Self as BurnTensorVocoder<B>>::synthesize_tensor(self, features, None),
        )
    }
}

fn synthesize_melgan_tensor<B: Backend>(
    input_contract: &SpectrogramContract,
    device: &B::Device,
    spectrogram: Tensor<B, 3>,
    mut profiler: Option<&mut dyn SynthesisProfiler>,
    inference: impl FnOnce(Tensor<B, 3>) -> Result<Tensor<B, 3>>,
) -> Result<Tensor<B, 3>> {
    let [batch, frames, bins] = spectrogram.dims();
    ensure!(batch == 1, "Burn MelGAN currently requires batch size one");
    ensure!(
        bins == input_contract.bins,
        "Burn MelGAN expected {} mel bins, got {bins}",
        input_contract.bins
    );
    ensure!(frames > 0, "Burn MelGAN requires at least one mel frame");
    let started = Instant::now();
    let waveform =
        inference(spectrogram.swap_dims(1, 2)).context("Burn MelGAN inference failed")?;
    finish_backend_stage::<B>(
        &mut profiler,
        device,
        SynthesisStage::WaveformDecoder,
        started,
        [
            SynthesisDimension::new("mel_frames", frames),
            SynthesisDimension::new("samples", waveform.dims()[2]),
        ],
    )?;
    Ok(waveform)
}

fn synthesize_host_tensor<B: Backend>(
    input_contract: &SpectrogramContract,
    output_contract: &WaveformContract,
    device: &B::Device,
    spectrogram: &Spectrogram,
    inference: impl FnOnce(Tensor<B, 3>) -> Result<Tensor<B, 3>>,
) -> Result<Waveform> {
    spectrogram
        .contract
        .ensure_compatible_with(input_contract)?;
    spectrogram.validate()?;
    let features = Tensor::<B, 3>::from_data(
        TensorData::new(
            spectrogram.values.clone(),
            [1, spectrogram.frames, input_contract.bins],
        ),
        device,
    );
    let samples = inference(features)?
        .into_data()
        .to_vec::<f32>()
        .context("Burn MelGAN output is not f32")?;
    let waveform = Waveform {
        contract: output_contract.clone(),
        samples,
    };
    waveform.validate()?;
    Ok(waveform)
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    use super::*;

    type TestBackend = NdArray<f32>;

    #[test]
    fn loads_and_synthesizes_real_checkpoint_when_provided() {
        let Some(model_path) = std::env::var_os("TONGUES_TEST_COQUI_HIFIGAN_MODEL") else {
            return;
        };
        let config_path = std::env::var_os("TONGUES_TEST_COQUI_HIFIGAN_CONFIG")
            .expect("TONGUES_TEST_COQUI_HIFIGAN_CONFIG must accompany the model");
        let mut vocoder =
            BurnHifiganVocoder::<TestBackend>::load(config_path, model_path, NdArrayDevice::Cpu)
                .expect("vocoder");
        let contract = vocoder.input_contract().clone();
        let spectrogram = Spectrogram {
            values: vec![0.0; contract.bins * 2],
            contract,
            frames: 2,
        };

        let waveform = vocoder.synthesize(&spectrogram).expect("waveform");

        assert_eq!(waveform.contract, WaveformContract::mono(22_050));
        assert_eq!(
            waveform.samples.len(),
            vocoder.generator().inference_output_frames(2).unwrap()
        );
    }

    #[test]
    fn loads_and_synthesizes_real_multiband_melgan_checkpoint_when_provided() {
        let Some(model_path) = std::env::var_os("TONGUES_TEST_COQUI_MULTIBAND_MELGAN_MODEL") else {
            return;
        };
        let config_path = std::env::var_os("TONGUES_TEST_COQUI_MULTIBAND_MELGAN_CONFIG")
            .expect("TONGUES_TEST_COQUI_MULTIBAND_MELGAN_CONFIG must accompany the model");
        let mut vocoder = BurnMultibandMelganVocoder::<TestBackend>::load(
            config_path,
            model_path,
            NdArrayDevice::Cpu,
        )
        .expect("MultiBand-MelGAN vocoder");
        let contract = vocoder.input_contract().clone();
        let spectrogram = Spectrogram {
            values: vec![0.0; contract.bins * 2],
            contract,
            frames: 2,
        };

        let waveform = vocoder.synthesize(&spectrogram).expect("waveform");

        assert_eq!(waveform.contract, WaveformContract::mono(22_050));
        assert_eq!(
            waveform.samples.len(),
            vocoder
                .generator()
                .inference_output_frames(2)
                .expect("output sample count")
        );
        assert!(waveform.samples.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    #[ignore = "requires the pinned Coqui MultiBand-MelGAN artifact and reference evidence; run scripts/speech-conformance.sh"]
    fn published_multiband_melgan_checkpoint_parity() {
        let config_path = std::env::var_os("TONGUES_TEST_COQUI_MULTIBAND_MELGAN_CONFIG")
            .expect("TONGUES_TEST_COQUI_MULTIBAND_MELGAN_CONFIG is required");
        let model_path = std::env::var_os("TONGUES_TEST_COQUI_MULTIBAND_MELGAN_MODEL")
            .expect("TONGUES_TEST_COQUI_MULTIBAND_MELGAN_MODEL is required");
        let reference_path = std::env::var_os("TONGUES_TEST_COQUI_REFERENCE")
            .expect("TONGUES_TEST_COQUI_REFERENCE is required");
        let reference: serde_json::Value = serde_json::from_slice(
            &fs::read(reference_path).expect("read Coqui reference evidence"),
        )
        .expect("parse Coqui reference evidence");
        let reference = &reference["multiband_melgan"];
        assert_eq!(
            reference["input_pattern"].as_str(),
            Some("channel-major-linspace-negative-one-to-one")
        );
        let input_shape = reference["input_shape"]
            .as_array()
            .expect("MultiBand-MelGAN input shape");
        let bins = input_shape[1].as_u64().expect("mel bins") as usize;
        let frames = input_shape[2].as_u64().expect("mel frames") as usize;
        let denominator = (bins * frames - 1) as f32;
        let mut values = Vec::with_capacity(bins * frames);
        for frame in 0..frames {
            for bin in 0..bins {
                values.push(-1.0 + 2.0 * (bin * frames + frame) as f32 / denominator);
            }
        }

        let mut vocoder =
            BurnVocoder::<TestBackend>::load(config_path, model_path, NdArrayDevice::Cpu)
                .expect("published MultiBand-MelGAN vocoder");
        assert!(matches!(&vocoder, BurnVocoder::MultibandMelgan(_)));
        let contract = vocoder.input_contract().clone();
        assert_eq!(contract.bins, bins);
        let waveform = vocoder
            .synthesize(&Spectrogram {
                contract,
                frames,
                values,
            })
            .expect("native MultiBand-MelGAN waveform");
        let expected = &reference["waveform"];
        assert_eq!(
            waveform.samples.len(),
            expected["samples"].as_u64().expect("sample count") as usize
        );
        assert!(waveform.samples.iter().all(|sample| sample.is_finite()));
        let rms = (waveform
            .samples
            .iter()
            .map(|sample| sample * sample)
            .sum::<f32>()
            / waveform.samples.len() as f32)
            .sqrt();
        let expected_rms = expected["rms"].as_f64().expect("reference RMS") as f32;
        assert!(
            (rms - expected_rms).abs() <= 5e-4,
            "MultiBand-MelGAN RMS differs: native {rms}, Coqui {expected_rms}"
        );
        for probe in expected["probes"].as_array().expect("waveform probes") {
            let probe = probe.as_array().expect("waveform probe");
            let index = probe[0].as_u64().expect("sample index") as usize;
            let expected = probe[1].as_f64().expect("sample") as f32;
            let actual = waveform.samples[index];
            assert!(
                (actual - expected).abs() <= 2e-3,
                "MultiBand-MelGAN waveform[{index}] differs: native {actual}, Coqui {expected}"
            );
        }
    }

    #[test]
    #[ignore = "requires the pinned Descript MelGAN artifact and reference evidence; run scripts/speech-conformance.sh"]
    fn published_melgan_checkpoint_parity() {
        let config_path = std::env::var_os("TONGUES_TEST_DESCRIPT_MELGAN_CONFIG")
            .expect("TONGUES_TEST_DESCRIPT_MELGAN_CONFIG is required");
        let model_path = std::env::var_os("TONGUES_TEST_DESCRIPT_MELGAN_MODEL")
            .expect("TONGUES_TEST_DESCRIPT_MELGAN_MODEL is required");
        let reference_path = std::env::var_os("TONGUES_TEST_COQUI_REFERENCE")
            .expect("TONGUES_TEST_COQUI_REFERENCE is required");
        let reference: serde_json::Value = serde_json::from_slice(
            &fs::read(reference_path).expect("read MelGAN reference evidence"),
        )
        .expect("parse MelGAN reference evidence");
        let reference = &reference["melgan"];
        assert_eq!(
            reference["input_pattern"].as_str(),
            Some("channel-major-linspace-negative-one-to-one")
        );
        let input_shape = reference["input_shape"]
            .as_array()
            .expect("MelGAN input shape");
        let bins = input_shape[1].as_u64().expect("mel bins") as usize;
        let frames = input_shape[2].as_u64().expect("mel frames") as usize;
        let denominator = (bins * frames - 1) as f32;
        let mut values = Vec::with_capacity(bins * frames);
        for frame in 0..frames {
            for bin in 0..bins {
                values.push(-1.0 + 2.0 * (bin * frames + frame) as f32 / denominator);
            }
        }

        let mut vocoder =
            BurnVocoder::<TestBackend>::load(config_path, model_path, NdArrayDevice::Cpu)
                .expect("published MelGAN vocoder");
        assert!(matches!(&vocoder, BurnVocoder::Melgan(_)));
        let contract = vocoder.input_contract().clone();
        assert_eq!(contract.bins, bins);
        assert_eq!(contract.sample_rate_hz, 22_050);
        assert_eq!(contract.hop_size, 256);
        assert_eq!(contract.frame_padding, Some((384, 384)));
        let waveform = vocoder
            .synthesize(&Spectrogram {
                contract,
                frames,
                values,
            })
            .expect("native MelGAN waveform");
        let expected = &reference["waveform"];
        assert_eq!(
            waveform.samples.len(),
            expected["samples"].as_u64().expect("sample count") as usize
        );
        assert!(waveform.samples.iter().all(|sample| sample.is_finite()));
        let rms = (waveform
            .samples
            .iter()
            .map(|sample| sample * sample)
            .sum::<f32>()
            / waveform.samples.len() as f32)
            .sqrt();
        let expected_rms = expected["rms"].as_f64().expect("reference RMS") as f32;
        assert!(
            (rms - expected_rms).abs() <= 5e-4,
            "MelGAN RMS differs: native {rms}, upstream {expected_rms}"
        );
        for probe in expected["probes"].as_array().expect("waveform probes") {
            let probe = probe.as_array().expect("waveform probe");
            let index = probe[0].as_u64().expect("sample index") as usize;
            let expected = probe[1].as_f64().expect("sample") as f32;
            let actual = waveform.samples[index];
            assert!(
                (actual - expected).abs() <= 2e-3,
                "MelGAN waveform[{index}] differs: native {actual}, upstream {expected}"
            );
        }
    }

    #[test]
    fn loads_and_synthesizes_real_melgan_checkpoint_when_provided() {
        let Some(model_path) = std::env::var_os("TONGUES_TEST_COQUI_MELGAN_MODEL") else {
            return;
        };
        let config_path = std::env::var_os("TONGUES_TEST_COQUI_MELGAN_CONFIG")
            .expect("TONGUES_TEST_COQUI_MELGAN_CONFIG must accompany the model");
        let mut vocoder =
            BurnMelganVocoder::<TestBackend>::load(config_path, model_path, NdArrayDevice::Cpu)
                .expect("MelGAN vocoder");
        let contract = vocoder.input_contract().clone();
        let spectrogram = Spectrogram {
            values: vec![0.0; contract.bins * 2],
            contract,
            frames: 2,
        };

        let waveform = vocoder.synthesize(&spectrogram).expect("waveform");

        assert_eq!(waveform.contract, WaveformContract::mono(22_050));
        assert_eq!(
            waveform.samples.len(),
            vocoder
                .generator()
                .inference_output_frames(2)
                .expect("output sample count")
        );
        assert!(waveform.samples.iter().all(|sample| sample.is_finite()));
    }
}
