use std::path::Path;

use anyhow::{ensure, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::{
    HifiganBundleConfig, HifiganGenerator, InferenceRuntime, NeuralVocoder, Spectrogram,
    SpectrogramContract, SpectrogramLayout, Waveform, WaveformContract,
};

/// Burn-native Coqui HiFi-GAN adapter at the shared acoustic boundary.
pub struct BurnHifiganVocoder<B: Backend> {
    generator: HifiganGenerator<B>,
    input_contract: SpectrogramContract,
    output_contract: WaveformContract,
    device: B::Device,
}

impl<B: Backend> BurnHifiganVocoder<B> {
    pub fn load(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
    ) -> Result<Self> {
        let config = HifiganBundleConfig::from_file(config_path)?;
        let input_contract = config.input_contract()?;
        ensure!(
            input_contract.layout == SpectrogramLayout::FramesByBins,
            "Burn HiFi-GAN adapter requires frame-major shared spectrograms"
        );
        let output_contract = WaveformContract::mono(input_contract.sample_rate_hz);
        let generator = config.load_burn_generator(checkpoint_path, &device)?;
        Ok(Self {
            generator,
            input_contract,
            output_contract,
            device,
        })
    }

    pub fn generator(&self) -> &HifiganGenerator<B> {
        &self.generator
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
}
