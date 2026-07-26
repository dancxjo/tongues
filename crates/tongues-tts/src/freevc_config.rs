//! Provider-neutral configuration for FreeVC-compatible voice conversion.

use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::burn_hifigan::HifiganGeneratorConfig;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FreeVcAudioConfig {
    pub input_sample_rate: u32,
    pub output_sample_rate: u32,
    pub filter_length: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mel_channels: usize,
    pub mel_fmin: f32,
    pub mel_fmax: Option<f32>,
}

impl Default for FreeVcAudioConfig {
    fn default() -> Self {
        Self {
            input_sample_rate: 16_000,
            output_sample_rate: 24_000,
            filter_length: 1_280,
            hop_length: 320,
            win_length: 1_280,
            n_mel_channels: 80,
            mel_fmin: 0.0,
            mel_fmax: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FreeVcNetworkConfig {
    pub inter_channels: usize,
    pub hidden_channels: usize,
    pub resblock: String,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub upsample_rates: Vec<usize>,
    pub upsample_initial_channel: usize,
    pub upsample_kernel_sizes: Vec<usize>,
    pub gin_channels: usize,
    pub ssl_dim: usize,
    pub use_spk: bool,
}

impl Default for FreeVcNetworkConfig {
    fn default() -> Self {
        Self {
            inter_channels: 192,
            hidden_channels: 192,
            resblock: "1".into(),
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            upsample_rates: vec![10, 6, 4, 2],
            upsample_initial_channel: 512,
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            gin_channels: 256,
            ssl_dim: 1_024,
            use_spk: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FreeVcConfig {
    pub model: String,
    pub model_args: FreeVcNetworkConfig,
    pub audio: FreeVcAudioConfig,
}

impl Default for FreeVcConfig {
    fn default() -> Self {
        Self {
            model: "freevc".into(),
            model_args: FreeVcNetworkConfig::default(),
            audio: FreeVcAudioConfig::default(),
        }
    }
}

impl FreeVcConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read FreeVC config {}", path.display()))?;
        Self::from_json5_str(&source)
            .with_context(|| format!("invalid FreeVC config {}", path.display()))
    }

    pub fn from_json5_str(source: &str) -> Result<Self> {
        let config: Self = json5::from_str(source).context("failed to parse FreeVC JSON/JSON5")?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model.eq_ignore_ascii_case("freevc"),
            "expected model `freevc`, got `{}`",
            self.model
        );
        ensure!(
            self.audio.input_sample_rate > 0 && self.audio.output_sample_rate > 0,
            "FreeVC sample rates must be positive"
        );
        ensure!(
            self.audio.filter_length > 0
                && self.audio.hop_length > 0
                && self.audio.win_length > 0
                && self.audio.win_length <= self.audio.filter_length,
            "FreeVC STFT dimensions are invalid"
        );
        ensure!(
            self.audio.n_mel_channels > 0,
            "FreeVC target mel channel count must be positive"
        );
        ensure!(
            self.model_args.inter_channels > 0
                && self.model_args.inter_channels.is_multiple_of(2)
                && self.model_args.hidden_channels > 0
                && self.model_args.gin_channels > 0
                && self.model_args.ssl_dim > 0,
            "FreeVC network dimensions must be positive and latent channels must be even"
        );
        ensure!(
            self.model_args.use_spk,
            "the published FreeVC24 runtime requires its external speaker encoder"
        );
        ensure!(
            self.model_args.upsample_rates.len() == self.model_args.upsample_kernel_sizes.len(),
            "FreeVC upsample rate/kernel counts differ"
        );
        ensure!(
            self.model_args.resblock_kernel_sizes.len()
                == self.model_args.resblock_dilation_sizes.len(),
            "FreeVC residual kernel/dilation counts differ"
        );
        crate::VitsWaveformDecoderConfig::from_generator_config(self.decoder_config())
            .map_err(anyhow::Error::from)?;
        Ok(())
    }

    pub fn decoder_config(&self) -> HifiganGeneratorConfig {
        HifiganGeneratorConfig {
            in_channels: self.model_args.inter_channels,
            out_channels: 1,
            resblock_type: self.model_args.resblock.clone(),
            resblock_dilation_sizes: self.model_args.resblock_dilation_sizes.clone(),
            resblock_kernel_sizes: self.model_args.resblock_kernel_sizes.clone(),
            upsample_kernel_sizes: self.model_args.upsample_kernel_sizes.clone(),
            upsample_initial_channel: self.model_args.upsample_initial_channel,
            upsample_factors: self.model_args.upsample_rates.clone(),
            inference_padding: 0,
            cond_channels: self.model_args.gin_channels,
            conv_pre_weight_norm: false,
            conv_post_weight_norm: false,
            conv_post_bias: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_defaults_describe_24_khz_freevc() {
        let config = FreeVcConfig::default();
        config.validate().unwrap();
        assert_eq!(config.audio.input_sample_rate, 16_000);
        assert_eq!(config.audio.output_sample_rate, 24_000);
        assert_eq!(
            config.model_args.upsample_rates.iter().product::<usize>(),
            480
        );
        assert_eq!(config.decoder_config().cond_channels, 256);
    }
}
