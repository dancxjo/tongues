use std::fs;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

use crate::{
    HifiganGenerator, HifiganGeneratorConfig, MelFilterBank, SpectrogramContract,
    SpectrogramDomain, SpectrogramKind, SpectrogramLayout, SpectrogramNormalization,
    SpectrogramPadMode, SpectrogramScale,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioFeatureConfig {
    pub fft_size: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub sample_rate: u32,
    #[serde(default)]
    pub preemphasis: f32,
    #[serde(default = "default_log_func")]
    pub log_func: String,
    pub num_mels: usize,
    #[serde(default)]
    pub mel_fmin: f32,
    pub mel_fmax: Option<f32>,
    #[serde(default = "default_spec_gain")]
    pub spec_gain: f32,
    #[serde(default)]
    pub signal_norm: bool,
    #[serde(default = "default_min_level_db")]
    pub min_level_db: f32,
    #[serde(default = "default_true")]
    pub symmetric_norm: bool,
    #[serde(default = "default_max_norm")]
    pub max_norm: f32,
    #[serde(default = "default_true")]
    pub clip_norm: bool,
    pub stats_path: Option<String>,
    #[serde(default = "default_true")]
    pub do_amp_to_db_mel: bool,
    #[serde(default = "default_stft_pad_mode")]
    pub stft_pad_mode: String,
}

impl AudioFeatureConfig {
    pub fn from_json5_str(source: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct Config {
            audio: AudioFeatureConfig,
        }

        let config: Config =
            json5::from_str(source).context("failed to parse audio feature config")?;
        config.audio.mel_contract()?;
        Ok(config.audio)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read model bundle config {}", path.display()))?;
        Self::from_json5_str(&source)
            .with_context(|| format!("invalid audio feature config {}", path.display()))
    }

    pub fn mel_contract(&self) -> Result<SpectrogramContract> {
        ensure!(
            self.preemphasis.is_finite() && (0.0..1.0).contains(&self.preemphasis),
            "audio feature preemphasis must be finite and in 0..1"
        );
        let normalization = if !self.signal_norm {
            SpectrogramNormalization::None
        } else if let Some(stats_path) = &self.stats_path {
            bail!(
                "mean/variance spectrogram normalization requires loading stats from `{stats_path}`"
            )
        } else {
            SpectrogramNormalization::Range {
                min_db: self.min_level_db,
                max_norm: self.max_norm,
                symmetric: self.symmetric_norm,
                clipped: self.clip_norm,
            }
        };
        let scale = if !self.do_amp_to_db_mel {
            SpectrogramScale::Linear
        } else {
            match self.log_func.as_str() {
                "np.log" | "log" => SpectrogramScale::NaturalLog {
                    gain: self.spec_gain,
                },
                "np.log10" | "log10" => SpectrogramScale::Log10 {
                    gain: self.spec_gain,
                },
                other => bail!("unsupported audio feature log function `{other}`"),
            }
        };
        let pad_mode = match self.stft_pad_mode.as_str() {
            "reflect" => SpectrogramPadMode::Reflect,
            "constant" => SpectrogramPadMode::Constant,
            other => SpectrogramPadMode::Other(other.to_string()),
        };
        let contract = SpectrogramContract {
            kind: SpectrogramKind::Mel {
                min_frequency_hz: self.mel_fmin,
                max_frequency_hz: self.mel_fmax,
            },
            domain: SpectrogramDomain::Amplitude,
            scale,
            normalization,
            sample_rate_hz: self.sample_rate,
            fft_size: self.fft_size,
            window_size: self.win_length,
            hop_size: self.hop_length,
            bins: self.num_mels,
            centered: true,
            pad_mode,
            preemphasis: (self.preemphasis != 0.0).then_some(self.preemphasis),
            mel_filter_bank: Some(MelFilterBank::Slaney),
            layout: SpectrogramLayout::FramesByBins,
        };
        contract.validate()?;
        Ok(contract)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HifiganGeneratorParams {
    pub resblock_type: String,
    pub upsample_factors: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_initial_channel: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
}

impl HifiganGeneratorParams {
    pub fn validate(&self, hop_size: usize) -> Result<()> {
        ensure!(
            matches!(self.resblock_type.as_str(), "1" | "2"),
            "unsupported HiFi-GAN residual block type `{}`",
            self.resblock_type
        );
        ensure!(
            !self.upsample_factors.is_empty(),
            "HiFi-GAN requires upsample stages"
        );
        ensure!(
            self.upsample_factors.len() == self.upsample_kernel_sizes.len(),
            "HiFi-GAN upsample factor and kernel lists differ in length"
        );
        ensure!(
            !self.resblock_kernel_sizes.is_empty()
                && self.resblock_kernel_sizes.len() == self.resblock_dilation_sizes.len(),
            "HiFi-GAN residual kernel and dilation lists differ in length"
        );
        ensure!(
            self.resblock_dilation_sizes
                .iter()
                .all(|dilations| !dilations.is_empty()),
            "HiFi-GAN residual blocks require dilation stages"
        );
        let total_upsample = self
            .upsample_factors
            .iter()
            .try_fold(1usize, |total, factor| total.checked_mul(*factor))
            .context("HiFi-GAN upsample product overflow")?;
        ensure!(
            total_upsample == hop_size,
            "HiFi-GAN upsample product {total_upsample} does not match spectrogram hop size {hop_size}"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HifiganBundleConfig {
    pub audio: AudioFeatureConfig,
    pub generator_model: String,
    pub generator_model_params: HifiganGeneratorParams,
}

impl HifiganBundleConfig {
    pub fn from_json5_str(source: &str) -> Result<Self> {
        let config: Self =
            json5::from_str(source).context("failed to parse model bundle config")?;
        ensure!(
            config.generator_model == "hifigan_generator",
            "expected `hifigan_generator`, found `{}`",
            config.generator_model
        );
        config
            .generator_model_params
            .validate(config.audio.hop_length)?;
        config.audio.mel_contract()?;
        Ok(config)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read model bundle config {}", path.display()))?;
        Self::from_json5_str(&source)
            .with_context(|| format!("invalid model bundle config {}", path.display()))
    }

    pub fn input_contract(&self) -> Result<SpectrogramContract> {
        self.audio.mel_contract()
    }

    pub fn burn_generator_config(&self) -> Result<HifiganGeneratorConfig> {
        let params = &self.generator_model_params;
        let config = HifiganGeneratorConfig {
            in_channels: self.audio.num_mels,
            out_channels: 1,
            resblock_type: params.resblock_type.clone(),
            resblock_dilation_sizes: params.resblock_dilation_sizes.clone(),
            resblock_kernel_sizes: params.resblock_kernel_sizes.clone(),
            upsample_kernel_sizes: params.upsample_kernel_sizes.clone(),
            upsample_initial_channel: params.upsample_initial_channel,
            upsample_factors: params.upsample_factors.clone(),
            inference_padding: 5,
            cond_channels: 0,
            conv_pre_weight_norm: true,
            conv_post_weight_norm: true,
            conv_post_bias: true,
        };
        config.validate().map_err(anyhow::Error::new)?;
        Ok(config)
    }

    pub fn load_burn_generator<B: Backend>(
        &self,
        checkpoint_path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<HifiganGenerator<B>> {
        let generator = self.init_burn_generator(device)?;
        self.load_burn_generator_checkpoint(generator, checkpoint_path)
    }

    pub fn init_burn_generator<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<HifiganGenerator<B>> {
        self.burn_generator_config()?
            .init(device)
            .map_err(anyhow::Error::new)
    }

    pub fn load_burn_generator_checkpoint<B: Backend>(
        &self,
        mut generator: HifiganGenerator<B>,
        checkpoint_path: impl AsRef<Path>,
    ) -> Result<HifiganGenerator<B>> {
        let checkpoint_path = checkpoint_path.as_ref();
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut generator,
            checkpoint_path,
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                map_indices_contiguous: false,
                ..Default::default()
            },
        )
        .with_context(|| {
            format!(
                "failed to load HiFi-GAN checkpoint {}",
                checkpoint_path.display()
            )
        })?;
        ensure!(
            result.unused.is_empty(),
            "HiFi-GAN checkpoint contains tensors not consumed by the Burn generator: {}",
            result.unused.join(", ")
        );
        Ok(generator)
    }
}

fn default_true() -> bool {
    true
}

fn default_log_func() -> String {
    "np.log10".into()
}

fn default_spec_gain() -> f32 {
    20.0
}

fn default_min_level_db() -> f32 {
    -100.0
}

fn default_max_norm() -> f32 {
    4.0
}

fn default_stft_pad_mode() -> String {
    "reflect".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    const HIFIGAN_CONFIG: &str = r#"
    {
      // Coqui files are JSON-with-comments, not strict JSON.
      "audio": {
        "fft_size": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "sample_rate": 22050,
        "preemphasis": 0.0,
        "log_func": "np.log",
        "num_mels": 80,
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0,
        "spec_gain": 1.0,
        "signal_norm": false,
        "min_level_db": -100,
        "symmetric_norm": true,
        "max_norm": 4.0,
        "clip_norm": true,
        "stats_path": null
      },
      "generator_model": "hifigan_generator",
      "generator_model_params": {
        "resblock_type": "1",
        "upsample_factors": [8, 8, 2, 2],
        "upsample_kernel_sizes": [16, 16, 4, 4],
        "upsample_initial_channel": 128,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
      }
    }
    "#;

    #[test]
    fn parses_commented_coqui_hifigan_config_into_exact_mel_contract() {
        let config = HifiganBundleConfig::from_json5_str(HIFIGAN_CONFIG).expect("config");
        let contract = config.input_contract().expect("contract");

        assert_eq!(contract.sample_rate_hz, 22_050);
        assert_eq!(contract.hop_size, 256);
        assert_eq!(contract.bins, 80);
        assert_eq!(contract.scale, SpectrogramScale::NaturalLog { gain: 1.0 });
        assert_eq!(contract.normalization, SpectrogramNormalization::None);
        assert_eq!(contract.mel_filter_bank, Some(MelFilterBank::Slaney));
    }

    #[test]
    fn rejects_hifigan_whose_upsample_product_does_not_match_hop() {
        let source = HIFIGAN_CONFIG.replace("[8, 8, 2, 2]", "[8, 8, 2, 1]");
        let error = HifiganBundleConfig::from_json5_str(&source).expect_err("mismatch");

        assert!(error.to_string().contains("does not match"));
    }

    #[test]
    fn rejects_external_normalization_stats_until_they_are_loaded() {
        let source = HIFIGAN_CONFIG
            .replace("\"signal_norm\": false", "\"signal_norm\": true")
            .replace("\"stats_path\": null", "\"stats_path\": \"scale.npy\"");
        let error = HifiganBundleConfig::from_json5_str(&source).expect_err("stats");

        assert!(error.to_string().contains("scale.npy"));
    }

    #[test]
    fn derives_burn_generator_defaults_from_coqui_config() {
        let config = HifiganBundleConfig::from_json5_str(HIFIGAN_CONFIG).expect("config");
        let burn = config.burn_generator_config().expect("Burn config");

        assert_eq!(burn.in_channels, 80);
        assert_eq!(burn.out_channels, 1);
        assert_eq!(burn.inference_padding, 5);
        assert_eq!(burn.upsample_factor(), 256);
    }

    #[test]
    fn loads_real_coqui_hifigan_checkpoint_when_provided() {
        use burn::backend::ndarray::{NdArray, NdArrayDevice};
        use burn::tensor::Tensor;

        type TestBackend = NdArray<f32>;

        let Some(model_path) = std::env::var_os("TONGUES_TEST_COQUI_HIFIGAN_MODEL") else {
            return;
        };
        let config_path = std::env::var_os("TONGUES_TEST_COQUI_HIFIGAN_CONFIG")
            .expect("TONGUES_TEST_COQUI_HIFIGAN_CONFIG must accompany the model");
        let config = HifiganBundleConfig::from_file(config_path).expect("model bundle config");
        let device = NdArrayDevice::Cpu;
        let generator = config
            .load_burn_generator::<TestBackend>(model_path, &device)
            .expect("direct Burn checkpoint load");
        let features = Tensor::<TestBackend, 3>::zeros([1, config.audio.num_mels, 2], &device);
        let waveform = generator.inference(features).expect("Burn inference");
        let expected = generator
            .inference_output_frames(2)
            .expect("output frame count");

        assert_eq!(waveform.dims(), [1, 1, expected]);
        assert!(waveform
            .into_data()
            .to_vec::<f32>()
            .expect("f32 waveform")
            .into_iter()
            .all(f32::is_finite));
    }
}
