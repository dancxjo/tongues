//! Named, package-pinned spectrogram feature conversions.
//!
//! A converter is part of an explicit acoustic/vocoder composition. It never
//! changes analysis geometry, layout, sample rate, or logarithmic scale.

use std::time::Instant;

use anyhow::{ensure, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use serde::Deserialize;

use crate::profiling::{finish_backend_stage, reborrow_profiler};
use crate::{
    BurnTensorVocoder, InferenceRuntime, NeuralVocoder, Spectrogram, SpectrogramContract,
    SpectrogramNormalization, SynthesisDimension, SynthesisProfiler, SynthesisStage, Waveform,
    WaveformContract,
};

pub const GLOW_MULTIBAND_STANDARDIZER_ID: &str = "coqui-ljspeech-multiband-melgan-standardize-v1";
pub const GLOW_MULTIBAND_STANDARDIZER_JSON: &str =
    include_str!("../catalog/glow-tts-multiband-melgan-standardization-v1.json");

#[derive(Debug, Clone, Deserialize)]
pub struct FeatureStandardizationConfig {
    pub schema_version: u32,
    pub id: String,
    pub source_artifact: FeatureStatisticsArtifact,
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FeatureStatisticsArtifact {
    pub path: String,
    pub sha256: String,
}

impl FeatureStandardizationConfig {
    pub fn glow_multiband() -> Result<Self> {
        let config: Self = serde_json::from_str(GLOW_MULTIBAND_STANDARDIZER_JSON)
            .context("invalid embedded Glow-TTS feature-conversion package")?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1,
            "unsupported feature-conversion schema {}",
            self.schema_version
        );
        ensure!(
            self.id == GLOW_MULTIBAND_STANDARDIZER_ID,
            "unexpected feature-conversion identity `{}`",
            self.id
        );
        ensure!(
            self.source_artifact.path == "scale_stats.npy",
            "feature-conversion statistics must identify scale_stats.npy"
        );
        ensure!(
            self.source_artifact.sha256.len() == 64
                && self
                    .source_artifact
                    .sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
            "feature-conversion statistics have an invalid SHA-256"
        );
        ensure!(
            !self.mean.is_empty() && self.mean.len() == self.scale.len(),
            "feature-conversion mean and scale lengths differ"
        );
        ensure!(
            self.mean.iter().all(|value| value.is_finite())
                && self
                    .scale
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0),
            "feature-conversion statistics contain invalid values"
        );
        Ok(())
    }
}

/// A Burn vocoder preceded by an exact, named per-bin standardization.
pub struct BurnStandardizingVocoder<B: Backend, V> {
    inner: V,
    config: FeatureStandardizationConfig,
    source_contract: SpectrogramContract,
    target_contract: SpectrogramContract,
    device: B::Device,
}

impl<B: Backend, V: BurnTensorVocoder<B>> BurnStandardizingVocoder<B, V> {
    pub fn new(
        inner: V,
        config: FeatureStandardizationConfig,
        source_contract: SpectrogramContract,
        device: B::Device,
    ) -> Result<Self> {
        config.validate()?;
        source_contract.validate()?;
        let target_contract = inner.input_contract().clone();
        target_contract.validate()?;
        ensure!(
            config.mean.len() == source_contract.bins,
            "feature-conversion has {} bins, acoustic contract has {}",
            config.mean.len(),
            source_contract.bins
        );
        let mut expected_source = target_contract.clone();
        let SpectrogramNormalization::OpaqueStandardized { sha256 } =
            &target_contract.normalization
        else {
            anyhow::bail!("feature-converted vocoder must require opaque standardization");
        };
        ensure!(
            sha256 == &config.source_artifact.sha256,
            "feature-conversion statistics checksum `{}` does not match vocoder `{sha256}`",
            config.source_artifact.sha256
        );
        expected_source.normalization = SpectrogramNormalization::None;
        source_contract
            .ensure_compatible_with(&expected_source)
            .context("feature conversion changes more than normalization")?;
        Ok(Self {
            inner,
            config,
            source_contract,
            target_contract,
            device,
        })
    }

    pub fn conversion_id(&self) -> &str {
        &self.config.id
    }

    pub fn inner(&self) -> &V {
        &self.inner
    }

    fn standardize_tensor(&self, spectrogram: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [batch, frames, bins] = spectrogram.dims();
        ensure!(
            batch == 1 && frames > 0 && bins == self.config.mean.len(),
            "feature conversion expected [1, frames, {}], got {:?}",
            self.config.mean.len(),
            spectrogram.dims()
        );
        let mean = Tensor::<B, 3>::from_data(
            TensorData::new(self.config.mean.clone(), [1, 1, bins]),
            &self.device,
        );
        let scale = Tensor::<B, 3>::from_data(
            TensorData::new(self.config.scale.clone(), [1, 1, bins]),
            &self.device,
        );
        Ok((spectrogram - mean) / scale)
    }
}

impl<B: Backend, V: BurnTensorVocoder<B>> NeuralVocoder for BurnStandardizingVocoder<B, V> {
    fn runtime(&self) -> InferenceRuntime {
        self.inner.runtime()
    }

    fn input_contract(&self) -> &SpectrogramContract {
        &self.source_contract
    }

    fn output_contract(&self) -> WaveformContract {
        self.inner.output_contract()
    }

    fn synthesize(&mut self, spectrogram: &Spectrogram) -> Result<Waveform> {
        spectrogram
            .contract
            .ensure_compatible_with(&self.source_contract)?;
        spectrogram.validate()?;
        let mut values = spectrogram.values.clone();
        for frame in values.chunks_exact_mut(self.config.mean.len()) {
            for (bin, value) in frame.iter_mut().enumerate() {
                *value = (*value - self.config.mean[bin]) / self.config.scale[bin];
            }
        }
        let converted = Spectrogram {
            contract: self.target_contract.clone(),
            frames: spectrogram.frames,
            values,
        };
        self.inner.synthesize(&converted)
    }
}

impl<B: Backend, V: BurnTensorVocoder<B>> BurnTensorVocoder<B> for BurnStandardizingVocoder<B, V> {
    fn synthesize_tensor(
        &self,
        spectrogram: Tensor<B, 3>,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<Tensor<B, 3>> {
        let mut profiler = profiler;
        let frames = spectrogram.dims()[1];
        let started = Instant::now();
        let converted = self.standardize_tensor(spectrogram)?;
        finish_backend_stage::<B>(
            &mut profiler,
            &self.device,
            SynthesisStage::FeatureConversion,
            started,
            [
                SynthesisDimension::new("mel_frames", frames),
                SynthesisDimension::new("mel_bins", self.config.mean.len()),
            ],
        )?;
        self.inner
            .synthesize_tensor(converted, reborrow_profiler(&mut profiler))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_glow_conversion_is_pinned_and_well_formed() {
        let config = FeatureStandardizationConfig::glow_multiband().expect("config");
        assert_eq!(config.mean.len(), 80);
        assert_eq!(
            config.source_artifact.sha256,
            "8c4a45b935563157509ddbff09f59e4ffea35e1d07f3bbf87ec21484cb275c4a"
        );
    }

    #[test]
    fn glow_conversion_matches_pinned_reference_probes() {
        let config = FeatureStandardizationConfig::glow_multiband().expect("config");
        for (bin, mel, expected) in [
            (0, -2.2727494_f32, 0.7144379_f32),
            (79, -4.1605105, -1.3568813),
            (0, -2.3858812, 0.38489646),
            (0, -2.5310504, -0.03796704),
            (0, -3.1077003, -1.7176903),
            (79, -3.9948568, -1.180381),
        ] {
            let actual = (mel - config.mean[bin]) / config.scale[bin];
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "standardized probe mismatch for bin {bin}: actual={actual}, expected={expected}"
            );
        }
    }
}
