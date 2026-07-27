use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, ensure, Context, Result};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    HifiganGenerator, HifiganGeneratorConfig, MelFilterBank, MelganGenerator,
    MelganGeneratorConfig, MultibandMelganGenerator, PqmfConfig, SpectrogramContract,
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
    #[serde(default = "default_ref_level_db")]
    pub ref_level_db: Option<f32>,
    #[serde(default = "default_true")]
    pub symmetric_norm: bool,
    #[serde(default = "default_max_norm")]
    pub max_norm: f32,
    #[serde(default = "default_true")]
    pub clip_norm: bool,
    pub stats_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stats_sha256: Option<String>,
    #[serde(default = "default_true")]
    pub do_amp_to_db_mel: bool,
    #[serde(default = "default_stft_pad_mode")]
    pub stft_pad_mode: String,
    #[serde(default = "default_true")]
    pub centered: bool,
    #[serde(default)]
    pub stft_manual_padding: Option<usize>,
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
        #[derive(Deserialize)]
        struct Config {
            audio: AudioFeatureConfig,
        }

        let path = path.as_ref();
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read model bundle config {}", path.display()))?;
        let mut config: Config = json5::from_str(&source)
            .with_context(|| format!("invalid audio feature config {}", path.display()))?;
        resolve_stats_digest(path, &mut config.audio)?;
        config.audio.mel_contract()?;
        Ok(config.audio)
    }

    pub fn mel_contract(&self) -> Result<SpectrogramContract> {
        ensure!(
            self.preemphasis.is_finite() && (0.0..1.0).contains(&self.preemphasis),
            "audio feature preemphasis must be finite and in 0..1"
        );
        let normalization = if !self.signal_norm {
            SpectrogramNormalization::None
        } else if let Some(stats_path) = &self.stats_path {
            let sha256 = self.stats_sha256.clone().with_context(|| {
                format!(
                    "mean/variance spectrogram normalization requires hashing stats from `{stats_path}`"
                )
            })?;
            SpectrogramNormalization::OpaqueStandardized { sha256 }
        } else {
            let reference_db = self
                .ref_level_db
                .context("range spectrogram normalization requires ref_level_db")?;
            SpectrogramNormalization::Range {
                min_db: self.min_level_db,
                reference_db,
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
            centered: self.centered,
            frame_padding: self.stft_manual_padding.map(|padding| (padding, padding)),
            pad_mode,
            preemphasis: (self.preemphasis != 0.0).then_some(self.preemphasis),
            mel_filter_bank: Some(MelFilterBank::Slaney),
            layout: SpectrogramLayout::FramesByBins,
        };
        contract.validate()?;
        Ok(contract)
    }

    /// Translate Coqui's field names into the model-neutral native DSP
    /// contract. External naming and defaults stop at this adapter boundary.
    pub fn native_spectrogram_config(&self) -> Result<tongues_audio::SpectrogramConfig> {
        self.mel_contract()?;
        if let Some(stats_path) = &self.stats_path {
            bail!(
                "native feature extraction cannot safely deserialize opaque normalization stats from `{stats_path}`"
            );
        }
        if let Some(padding) = self.stft_manual_padding {
            bail!(
                "native feature extraction does not yet implement explicit {padding}-sample STFT input padding"
            );
        }
        let pad_mode = match self.stft_pad_mode.as_str() {
            "reflect" => tongues_audio::PadMode::Reflect,
            "constant" => tongues_audio::PadMode::Constant,
            other => bail!("unsupported native STFT pad mode `{other}`"),
        };
        let scale = if !self.do_amp_to_db_mel {
            tongues_audio::SpectralScale::Linear
        } else {
            match self.log_func.as_str() {
                "np.log" | "log" => tongues_audio::SpectralScale::NaturalLog {
                    gain: self.spec_gain,
                    floor: tongues_audio::DEFAULT_SPECTRAL_FLOOR,
                },
                "np.log10" | "log10" => tongues_audio::SpectralScale::Log10 {
                    gain: self.spec_gain,
                    floor: tongues_audio::DEFAULT_SPECTRAL_FLOOR,
                },
                other => bail!("unsupported audio feature log function `{other}`"),
            }
        };
        let normalization = if self.signal_norm {
            tongues_audio::SpectrogramNormalization::Range {
                min_db: self.min_level_db,
                reference_db: self
                    .ref_level_db
                    .context("range spectrogram normalization requires ref_level_db")?,
                max_norm: self.max_norm,
                symmetric: self.symmetric_norm,
                clipped: self.clip_norm,
            }
        } else {
            tongues_audio::SpectrogramNormalization::None
        };
        let config = tongues_audio::SpectrogramConfig {
            sample_rate_hz: self.sample_rate,
            stft: tongues_audio::StftConfig {
                fft_size: self.fft_size,
                window_size: self.win_length,
                hop_size: self.hop_length,
                center: self.centered,
                pad_mode,
                window: tongues_audio::Window::Hann,
            },
            output: tongues_audio::SpectrogramOutput::Mel(tongues_audio::MelConfig {
                bins: self.num_mels,
                min_frequency_hz: self.mel_fmin,
                max_frequency_hz: self.mel_fmax,
                scale: tongues_audio::MelScale::Slaney,
                normalization: tongues_audio::MelNormalization::Slaney,
            }),
            domain: tongues_audio::SpectralDomain::Amplitude,
            scale,
            normalization,
            preemphasis: (self.preemphasis != 0.0).then_some(self.preemphasis),
        };
        config
            .validate()
            .map_err(anyhow::Error::from)
            .context("invalid native audio feature contract")?;
        Ok(config)
    }

    pub fn extract_native_spectrogram(
        &self,
        mono_samples: &[f32],
    ) -> Result<tongues_audio::Spectrogram> {
        let config = self.native_spectrogram_config()?;
        tongues_audio::spectrogram(mono_samples, &config)
            .map_err(anyhow::Error::from)
            .context("native audio feature extraction failed")
    }
}

fn resolve_stats_digest(config_path: &Path, audio: &mut AudioFeatureConfig) -> Result<()> {
    let Some(stats_path) = audio.stats_path.as_deref() else {
        return Ok(());
    };
    let filename = Path::new(stats_path)
        .file_name()
        .context("normalization stats path has no filename")?;
    let sibling = config_path
        .parent()
        .context("model config path has no parent directory")?
        .join(filename);
    let resolved = if sibling.is_file() {
        sibling
    } else {
        Path::new(stats_path).to_path_buf()
    };
    let bytes = fs::read(&resolved).with_context(|| {
        format!(
            "failed to read normalization stats `{}` (resolved from `{stats_path}`)",
            resolved.display()
        )
    })?;
    audio.stats_sha256 = Some(
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    );
    Ok(())
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
        let mut config: Self = json5::from_str(&source)
            .with_context(|| format!("invalid model bundle config {}", path.display()))?;
        ensure!(
            config.generator_model == "hifigan_generator",
            "expected `hifigan_generator`, found `{}`",
            config.generator_model
        );
        resolve_stats_digest(path, &mut config.audio)?;
        config
            .generator_model_params
            .validate(config.audio.hop_length)?;
        config.input_contract()?;
        Ok(config)
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MelganGeneratorParams {
    #[serde(default)]
    pub in_channels: Option<usize>,
    #[serde(default)]
    pub out_channels: Option<usize>,
    #[serde(default)]
    pub proj_kernel: Option<usize>,
    #[serde(default)]
    pub base_channels: Option<usize>,
    #[serde(default)]
    pub upsample_factors: Option<Vec<usize>>,
    #[serde(default)]
    pub res_kernel: Option<usize>,
    #[serde(default)]
    pub num_res_blocks: Option<usize>,
    #[serde(default)]
    pub inference_padding: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelganVariant {
    Melgan,
    Multiband,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MelganBundleConfig {
    pub audio: AudioFeatureConfig,
    pub generator_model: String,
    #[serde(default)]
    pub generator_model_params: MelganGeneratorParams,
    #[serde(default)]
    pub use_pqmf: bool,
}

impl MelganBundleConfig {
    pub fn from_json5_str(source: &str) -> Result<Self> {
        let config: Self =
            json5::from_str(source).context("failed to parse MelGAN bundle config")?;
        config.validate()?;
        Ok(config)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read model bundle config {}", path.display()))?;
        let mut config: Self = json5::from_str(&source)
            .with_context(|| format!("invalid MelGAN bundle config {}", path.display()))?;
        resolve_stats_digest(path, &mut config.audio)?;
        config.validate()?;
        Ok(config)
    }

    pub fn variant(&self) -> Result<MelganVariant> {
        match self.generator_model.as_str() {
            "melgan_generator" => {
                ensure!(
                    !self.use_pqmf,
                    "plain MelGAN must not declare `use_pqmf=true`"
                );
                Ok(MelganVariant::Melgan)
            }
            "multiband_melgan_generator" => {
                ensure!(self.use_pqmf, "MultiBand-MelGAN requires `use_pqmf=true`");
                Ok(MelganVariant::Multiband)
            }
            other => bail!("unsupported MelGAN generator model `{other}`"),
        }
    }

    pub fn validate(&self) -> Result<()> {
        let variant = self.variant()?;
        let generator = self.burn_generator_config()?;
        generator.validate().map_err(anyhow::Error::new)?;
        let generated_samples = generator
            .upsample_factor()
            .checked_mul(match variant {
                MelganVariant::Melgan => 1,
                MelganVariant::Multiband => PqmfConfig::default().bands,
            })
            .context("MelGAN output factor overflow")?;
        ensure!(
            generated_samples == self.audio.hop_length,
            "MelGAN output factor {generated_samples} does not match spectrogram hop size {}",
            self.audio.hop_length
        );
        self.audio.mel_contract()?;
        Ok(())
    }

    pub fn input_contract(&self) -> Result<SpectrogramContract> {
        self.audio.mel_contract()
    }

    pub fn burn_generator_config(&self) -> Result<MelganGeneratorConfig> {
        let variant = self.variant()?;
        let params = &self.generator_model_params;
        let multiband = variant == MelganVariant::Multiband;
        let config = MelganGeneratorConfig {
            in_channels: params.in_channels.unwrap_or(self.audio.num_mels),
            out_channels: params.out_channels.unwrap_or(if multiband { 4 } else { 1 }),
            projection_kernel_size: params.proj_kernel.unwrap_or(7),
            base_channels: params
                .base_channels
                .unwrap_or(if multiband { 384 } else { 512 }),
            upsample_factors: params.upsample_factors.clone().unwrap_or_else(|| {
                if multiband {
                    vec![8, 4, 2]
                } else {
                    vec![8, 8, 2, 2]
                }
            }),
            residual_kernel_size: params.res_kernel.unwrap_or(3),
            residual_blocks: params
                .num_res_blocks
                .unwrap_or(if multiband { 4 } else { 3 }),
            inference_padding: params.inference_padding.unwrap_or(2),
        };
        ensure!(
            config.in_channels == self.audio.num_mels,
            "MelGAN input channels {} do not match {} mel bins",
            config.in_channels,
            self.audio.num_mels
        );
        config.validate().map_err(anyhow::Error::new)?;
        Ok(config)
    }

    pub fn init_burn_generator<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<MelganGenerator<B>> {
        ensure!(
            self.variant()? == MelganVariant::Melgan,
            "requested plain generator from MultiBand-MelGAN config"
        );
        self.burn_generator_config()?
            .init(device)
            .map_err(anyhow::Error::new)
    }

    pub fn init_burn_multiband_generator<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<MultibandMelganGenerator<B>> {
        ensure!(
            self.variant()? == MelganVariant::Multiband,
            "requested multiband generator from plain MelGAN config"
        );
        self.burn_generator_config()?
            .init_multiband(PqmfConfig::default(), device)
            .map_err(anyhow::Error::new)
    }

    pub fn load_burn_generator<B: Backend>(
        &self,
        checkpoint_path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<MelganGenerator<B>> {
        let generator = self.init_burn_generator(device)?;
        self.load_burn_generator_checkpoint(generator, checkpoint_path)
    }

    pub fn load_burn_generator_checkpoint<B: Backend>(
        &self,
        mut generator: MelganGenerator<B>,
        checkpoint_path: impl AsRef<Path>,
    ) -> Result<MelganGenerator<B>> {
        load_melgan_checkpoint(&mut generator, checkpoint_path.as_ref(), "MelGAN", true)?;
        Ok(generator)
    }

    pub fn load_burn_multiband_generator<B: Backend>(
        &self,
        checkpoint_path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<MultibandMelganGenerator<B>> {
        let generator = self.init_burn_multiband_generator(device)?;
        self.load_burn_multiband_generator_checkpoint(generator, checkpoint_path)
    }

    pub fn load_burn_multiband_generator_checkpoint<B: Backend>(
        &self,
        mut generator: MultibandMelganGenerator<B>,
        checkpoint_path: impl AsRef<Path>,
    ) -> Result<MultibandMelganGenerator<B>> {
        load_melgan_checkpoint(
            &mut generator,
            checkpoint_path.as_ref(),
            "MultiBand-MelGAN",
            false,
        )?;
        Ok(generator)
    }
}

fn load_melgan_checkpoint<B: Backend, M: burn_store::ModuleSnapshot<B>>(
    generator: &mut M,
    checkpoint_path: &Path,
    label: &str,
    allow_descript_layout: bool,
) -> Result<()> {
    let load_coqui = |generator: &mut M, path: &Path| {
        crate::checkpoint::load_pytorch_layout_checkpoint(
            generator,
            path,
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                key_remappings: vec![
                    (
                        r"^pqmf_layer\.H$".into(),
                        "pqmf_layer.analysis_filter".into(),
                    ),
                    (
                        r"^pqmf_layer\.G$".into(),
                        "pqmf_layer.synthesis_filter".into(),
                    ),
                ],
                map_indices_contiguous: true,
                skip_enum_variants: true,
                ..Default::default()
            },
        )
    };
    let load_descript = |generator: &mut M, path: &Path| {
        crate::checkpoint::load_pytorch_layout_checkpoint(
            generator,
            path,
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: None,
                key_remappings: descript_melgan_key_remappings(),
                map_indices_contiguous: true,
                skip_enum_variants: true,
                ..Default::default()
            },
        )
    };
    let result = (|| -> Result<_> {
        match load_coqui(generator, checkpoint_path) {
            Ok(result) => return Ok(result),
            Err(error) if format!("{error:#}").contains("collections.Counter") => {
                let sanitized = SanitizedLegacyCheckpoint::create(checkpoint_path)?;
                return load_coqui(generator, &sanitized.path).with_context(|| {
                    format!(
                        "failed to load {label} checkpoint after safely treating legacy training-only collections.Counter metadata as an ordered dictionary"
                    )
                });
            }
            Err(error) if !allow_descript_layout => return Err(error),
            Err(_) => {}
        }
        load_descript(generator, checkpoint_path)
            .context("checkpoint is neither a Coqui nor Descript MelGAN tensor layout")
    })()
    .with_context(|| {
        format!(
            "failed to load {label} checkpoint {}",
            checkpoint_path.display()
        )
    })?;
    ensure!(
        result.unused.is_empty(),
        "{label} checkpoint contains tensors not consumed by the Burn generator: {}",
        result.unused.join(", ")
    );
    Ok(())
}

fn descript_melgan_key_remappings() -> Vec<(String, String)> {
    let mut mappings = vec![
        (r"^model\.1\.".into(), "layers.1.".into()),
        (r"^model\.3\.".into(), "layers.3.".into()),
        (r"^model\.8\.".into(), "layers.6.".into()),
        (r"^model\.13\.".into(), "layers.9.".into()),
        (r"^model\.18\.".into(), "layers.12.".into()),
        (r"^model\.24\.".into(), "layers.16.".into()),
    ];
    for (source_start, target_stack) in [(4, 4), (9, 7), (14, 10), (19, 13)] {
        for block in 0..3 {
            let source = source_start + block;
            mappings.push((
                format!(r"^model\.{source}\.block\.2\."),
                format!("layers.{target_stack}.blocks.{block}.2."),
            ));
            mappings.push((
                format!(r"^model\.{source}\.block\.4\."),
                format!("layers.{target_stack}.blocks.{block}.4."),
            ));
            mappings.push((
                format!(r"^model\.{source}\.shortcut\."),
                format!("layers.{target_stack}.shortcuts.{block}."),
            ));
        }
    }
    mappings
}

pub(crate) struct SanitizedLegacyCheckpoint {
    pub(crate) path: std::path::PathBuf,
}

impl SanitizedLegacyCheckpoint {
    pub(crate) fn create(source: &Path) -> Result<Self> {
        const COUNTER: &[u8] = b"ccollections\nCounter\n";
        const ORDERED_DICT: &[u8] = b"ccollections\nOrderedDict\n";

        let bytes = fs::read(source).with_context(|| {
            format!(
                "failed to read legacy MelGAN checkpoint {}",
                source.display()
            )
        })?;
        ensure!(
            bytes.starts_with(&[0x80, 0x02]),
            "collections.Counter fallback is limited to legacy protocol-2 PyTorch checkpoints"
        );
        let occurrences = bytes
            .windows(COUNTER.len())
            .filter(|window| *window == COUNTER)
            .count();
        ensure!(
            occurrences > 0,
            "legacy checkpoint reported collections.Counter but contains no matching GLOBAL opcode"
        );
        let mut sanitized =
            Vec::with_capacity(bytes.len() + occurrences * (ORDERED_DICT.len() - COUNTER.len()));
        let mut cursor = 0;
        while let Some(offset) = bytes[cursor..]
            .windows(COUNTER.len())
            .position(|window| window == COUNTER)
        {
            let start = cursor + offset;
            sanitized.extend_from_slice(&bytes[cursor..start]);
            sanitized.extend_from_slice(ORDERED_DICT);
            cursor = start + COUNTER.len();
        }
        sanitized.extend_from_slice(&bytes[cursor..]);

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before the Unix epoch")?
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("tongues-melgan-{}-{nonce}.pth", std::process::id()));
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| {
                format!(
                    "failed to create temporary sanitized checkpoint {}",
                    path.display()
                )
            })?;
        output.write_all(&sanitized)?;
        output.flush()?;
        Ok(Self { path })
    }
}

impl Drop for SanitizedLegacyCheckpoint {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
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

fn default_ref_level_db() -> Option<f32> {
    Some(20.0)
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

    fn melgan_config(generator: &str, parameters: &str, use_pqmf: bool) -> String {
        format!(
            r#"{{
              "audio": {{
                "fft_size": 1024,
                "win_length": 1024,
                "hop_length": 256,
                "sample_rate": 22050,
                "preemphasis": 0.0,
                "log_func": "np.log10",
                "num_mels": 80,
                "mel_fmin": 50.0,
                "mel_fmax": 7600.0,
                "spec_gain": 1.0,
                "signal_norm": false,
                "min_level_db": -100,
                "ref_level_db": 0,
                "symmetric_norm": true,
                "max_norm": 4.0,
                "clip_norm": true,
                "stats_path": null
              }},
              "generator_model": "{generator}",
              "generator_model_params": {parameters},
              "use_pqmf": {use_pqmf}
            }}"#
        )
    }

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
    fn coqui_adapter_extracts_native_features_with_exact_metadata() {
        let config = HifiganBundleConfig::from_json5_str(HIFIGAN_CONFIG).expect("config");
        let samples = (0..2_048)
            .map(|index| (std::f32::consts::TAU * index as f32 / 37.0).sin() * 0.25)
            .collect::<Vec<_>>();
        let features = config
            .audio
            .extract_native_spectrogram(&samples)
            .expect("native features");

        assert!(features.frames > 0);
        assert_eq!(features.config.sample_rate_hz, 22_050);
        assert_eq!(features.config.output_bins(), 80);
        assert!(features.values.iter().all(|value| value.is_finite()));
        assert_eq!(
            features.config,
            config.audio.native_spectrogram_config().unwrap()
        );
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

    #[test]
    fn parses_plain_and_multiband_melgan_topologies() {
        let plain = MelganBundleConfig::from_json5_str(&melgan_config(
            "melgan_generator",
            r#"{"upsample_factors": [8, 8, 2, 2], "num_res_blocks": 3}"#,
            false,
        ))
        .expect("plain MelGAN config");
        assert_eq!(plain.variant().unwrap(), MelganVariant::Melgan);
        assert_eq!(plain.burn_generator_config().unwrap().out_channels, 1);
        assert_eq!(
            plain.burn_generator_config().unwrap().upsample_factor(),
            256
        );

        let multiband = MelganBundleConfig::from_json5_str(&melgan_config(
            "multiband_melgan_generator",
            r#"{"upsample_factors": [8, 4, 2], "num_res_blocks": 4}"#,
            true,
        ))
        .expect("MultiBand-MelGAN config");
        assert_eq!(multiband.variant().unwrap(), MelganVariant::Multiband);
        assert_eq!(multiband.burn_generator_config().unwrap().out_channels, 4);
        assert_eq!(
            multiband.burn_generator_config().unwrap().upsample_factor(),
            64
        );

        let multiband_defaults = MelganBundleConfig::from_json5_str(&melgan_config(
            "multiband_melgan_generator",
            "{}",
            true,
        ))
        .expect("default MultiBand-MelGAN config");
        let generator = multiband_defaults.burn_generator_config().unwrap();
        assert_eq!(generator.upsample_factors, vec![8, 4, 2]);
        assert_eq!(generator.residual_blocks, 4);
    }

    #[test]
    fn multiband_stats_digest_is_part_of_the_composition_contract() {
        let plain = MelganBundleConfig::from_json5_str(&melgan_config(
            "melgan_generator",
            r#"{"upsample_factors": [8, 8, 2, 2]}"#,
            false,
        ))
        .unwrap();
        let mut multiband: MelganBundleConfig = json5::from_str(&melgan_config(
            "multiband_melgan_generator",
            r#"{"upsample_factors": [8, 4, 2]}"#,
            true,
        ))
        .unwrap();
        multiband.audio.signal_norm = true;
        multiband.audio.stats_path = Some("scale_stats.npy".into());
        multiband.audio.stats_sha256 = Some("a".repeat(64));

        let produced = plain.input_contract().unwrap();
        let required = multiband.input_contract().unwrap();
        let error = produced
            .ensure_compatible_with(&required)
            .expect_err("normalization identity must prevent unsafe composition");
        assert!(error.to_string().contains("contract mismatch"));
    }
}
