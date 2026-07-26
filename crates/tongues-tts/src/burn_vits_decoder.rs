//! Burn-native adapter for the waveform decoder embedded in a VITS model.
//!
//! VITS uses a conditionable HiFi-GAN generator internally, but its published
//! checkpoint topology differs from the standalone vocoder in two details:
//! inference adds no frame padding, and the pre/post convolutions are stored
//! without weight normalization. The reusable generator represents those two
//! convolutions with weight-normalized parameters, so checkpoint loading maps
//! each plain weight to `weight_v` and derives `weight_g = norm(weight_v)`.
//! This is exactly equivalent to the stored plain convolution at inference.

use std::fmt;
use std::path::Path;

use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::burn_hifigan::{HifiganError, HifiganGenerator, HifiganGeneratorConfig};
use crate::VitsInferenceConfig;

#[derive(Debug)]
pub enum VitsWaveformDecoderError {
    InvalidTopology(String),
    InvalidInput(String),
    Generator(HifiganError),
    Checkpoint(String),
}

impl fmt::Display for VitsWaveformDecoderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTopology(message) => {
                write!(
                    formatter,
                    "invalid VITS waveform decoder topology: {message}"
                )
            }
            Self::InvalidInput(message) => {
                write!(formatter, "invalid VITS waveform decoder input: {message}")
            }
            Self::Generator(error) => error.fmt(formatter),
            Self::Checkpoint(message) => {
                write!(
                    formatter,
                    "unable to load VITS waveform decoder checkpoint: {message}"
                )
            }
        }
    }
}

impl std::error::Error for VitsWaveformDecoderError {}

impl From<HifiganError> for VitsWaveformDecoderError {
    fn from(error: HifiganError) -> Self {
        Self::Generator(error)
    }
}

/// Topology of the waveform decoder embedded in a VITS model.
#[derive(Debug, Clone, PartialEq)]
pub struct VitsWaveformDecoderConfig {
    generator: HifiganGeneratorConfig,
}

impl VitsWaveformDecoderConfig {
    pub fn from_generator_config(
        generator: HifiganGeneratorConfig,
    ) -> Result<Self, VitsWaveformDecoderError> {
        generator
            .validate()
            .map_err(VitsWaveformDecoderError::Generator)?;
        Ok(Self { generator })
    }

    pub fn from_model_config(
        config: &VitsInferenceConfig,
    ) -> Result<Self, VitsWaveformDecoderError> {
        config
            .validate()
            .map_err(|error| VitsWaveformDecoderError::InvalidTopology(error.to_string()))?;

        let args = &config.network;
        let cond_channels = if args.use_speaker_embedding {
            args.speaker_embedding_channels
        } else if args.use_d_vector_file {
            args.d_vector_dim
        } else {
            0
        };
        let generator = HifiganGeneratorConfig {
            in_channels: args.hidden_channels,
            out_channels: 1,
            resblock_type: args.resblock_type_decoder.clone(),
            resblock_dilation_sizes: args.resblock_dilation_sizes_decoder.clone(),
            resblock_kernel_sizes: args.resblock_kernel_sizes_decoder.clone(),
            upsample_kernel_sizes: args.upsample_kernel_sizes_decoder.clone(),
            upsample_initial_channel: args.upsample_initial_channel_decoder,
            upsample_factors: args.upsample_rates_decoder.clone(),
            inference_padding: 0,
            cond_channels,
            conv_pre_weight_norm: false,
            conv_post_weight_norm: false,
            conv_post_bias: false,
        };

        if generator.upsample_factor() != config.audio.hop_length {
            return Err(VitsWaveformDecoderError::InvalidTopology(format!(
                "decoder upsample factor {} does not match audio hop length {}",
                generator.upsample_factor(),
                config.audio.hop_length
            )));
        }

        Ok(Self { generator })
    }

    /// Describes the checkpoint topology, including its plain pre/post layers.
    pub fn generator_config(&self) -> &HifiganGeneratorConfig {
        &self.generator
    }

    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<VitsWaveformDecoder<B>, VitsWaveformDecoderError> {
        // HifiganGenerator currently stores pre/post convolutions in
        // weight-normalized form. The adapter keeps that implementation detail
        // private and reconstructs plain checkpoint weights exactly on load.
        let storage_config = HifiganGeneratorConfig {
            in_channels: self.generator.in_channels,
            out_channels: self.generator.out_channels,
            resblock_type: self.generator.resblock_type.clone(),
            resblock_dilation_sizes: self.generator.resblock_dilation_sizes.clone(),
            resblock_kernel_sizes: self.generator.resblock_kernel_sizes.clone(),
            upsample_kernel_sizes: self.generator.upsample_kernel_sizes.clone(),
            upsample_initial_channel: self.generator.upsample_initial_channel,
            upsample_factors: self.generator.upsample_factors.clone(),
            inference_padding: 0,
            cond_channels: self.generator.cond_channels,
            conv_pre_weight_norm: true,
            conv_post_weight_norm: true,
            conv_post_bias: false,
        };
        let generator = storage_config.init(device)?;
        Ok(VitsWaveformDecoder { generator })
    }

    pub fn load_checkpoint<B: Backend>(
        &self,
        checkpoint_path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<VitsWaveformDecoder<B>, VitsWaveformDecoderError> {
        self.init(device)?.load_checkpoint(checkpoint_path)
    }
}

/// Conditionable, integrated VITS waveform decoder.
#[derive(Module, Debug)]
pub struct VitsWaveformDecoder<B: Backend> {
    generator: HifiganGenerator<B>,
}

impl<B: Backend> VitsWaveformDecoder<B> {
    /// Loads only the `waveform_decoder` subtree from a full VITS checkpoint.
    pub fn load_checkpoint(
        self,
        checkpoint_path: impl AsRef<Path>,
    ) -> Result<Self, VitsWaveformDecoderError> {
        self.load_checkpoint_subtree(checkpoint_path, "waveform_decoder")
    }

    pub(crate) fn load_checkpoint_subtree(
        mut self,
        checkpoint_path: impl AsRef<Path>,
        subtree: &str,
    ) -> Result<Self, VitsWaveformDecoderError> {
        let (predicate, prefix) = match subtree {
            "waveform_decoder" => (
                waveform_decoder_tensor as fn(&str, &str) -> bool,
                r"^waveform_decoder\.",
            ),
            "dec" => (freevc_decoder_tensor as fn(&str, &str) -> bool, r"^dec\."),
            other => {
                return Err(VitsWaveformDecoderError::Checkpoint(format!(
                    "unsupported decoder checkpoint subtree `{other}`"
                )));
            }
        };
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut self.generator,
            checkpoint_path.as_ref(),
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                predicate: Some(predicate),
                key_remappings: vec![
                    (prefix.into(), String::new()),
                    (r"^cond\.".into(), "cond_layer.".into()),
                    (
                        r"^(conv_pre|conv_post)\.weight$".into(),
                        "$1.weight_v".into(),
                    ),
                ],
                map_indices_contiguous: false,
                allow_partial: true,
                skip_enum_variants: true,
            },
        )
        .map_err(|error| VitsWaveformDecoderError::Checkpoint(error.to_string()))?;

        let mut missing = result
            .missing
            .iter()
            .map(|(path, _)| path.as_str())
            .collect::<Vec<_>>();
        missing.sort_unstable();
        let expected_missing = ["conv_post.weight_g", "conv_pre.weight_g"];
        let unexpected_unused = result
            .unused
            .iter()
            .filter(|path| predicate(path, ""))
            .cloned()
            .collect::<Vec<_>>();
        if missing != expected_missing || !result.errors.is_empty() || !unexpected_unused.is_empty()
        {
            return Err(VitsWaveformDecoderError::Checkpoint(format!(
                "decoder subtree does not exactly match the Burn module: missing [{}], {} load errors, unused [{}]",
                missing.join(", "),
                result.errors.len(),
                unexpected_unused.join(", ")
            )));
        }

        self.generator.conv_pre.weight_g = Param::from_tensor(
            weight_norm_dim_zero(self.generator.conv_pre.weight_v.val()).detach(),
        );
        self.generator.conv_post.weight_g = Param::from_tensor(
            weight_norm_dim_zero(self.generator.conv_post.weight_v.val()).detach(),
        );
        Ok(self)
    }

    /// Decodes latent features shaped `[batch, hidden_channels, frames]`.
    ///
    /// Speaker conditioning may be frame-aligned or `[batch, cond_channels, 1]`;
    /// the latter is expanded across the latent sequence as VITS requires.
    pub fn forward(
        &self,
        latent: Tensor<B, 3>,
        conditioning: Option<Tensor<B, 3>>,
    ) -> Result<Tensor<B, 3>, VitsWaveformDecoderError> {
        let [batch, _, frames] = latent.dims();
        let conditioning = match conditioning {
            Some(conditioning) => {
                let [cond_batch, _, cond_frames] = conditioning.dims();
                if cond_batch != batch {
                    return Err(VitsWaveformDecoderError::InvalidInput(format!(
                        "conditioning batch {cond_batch} does not match latent batch {batch}"
                    )));
                }
                if cond_frames == 1 {
                    Some(conditioning.repeat_dim(2, frames))
                } else if cond_frames == frames {
                    Some(conditioning)
                } else {
                    return Err(VitsWaveformDecoderError::InvalidInput(format!(
                        "conditioning has {cond_frames} frames; expected 1 or {frames}"
                    )));
                }
            }
            None => None,
        };
        self.generator
            .forward(latent, conditioning)
            .map_err(Into::into)
    }

    pub fn input_channels(&self) -> usize {
        self.generator.input_channels()
    }

    pub fn conditioning_channels(&self) -> usize {
        self.generator.cond_channels
    }

    pub fn upsample_factor(&self) -> usize {
        self.generator.upsample_factor()
    }

    pub fn output_frames(&self, latent_frames: usize) -> Option<usize> {
        self.generator.output_frames(latent_frames)
    }
}

fn waveform_decoder_tensor(path: &str, _container: &str) -> bool {
    [
        "conv_pre.",
        "ups.",
        "resblocks.",
        "conv_post.",
        "cond_layer.",
    ]
    .iter()
    .any(|prefix| path.starts_with(prefix))
}

fn freevc_decoder_tensor(path: &str, _container: &str) -> bool {
    path.strip_prefix("dec.")
        .is_some_and(|path| path.starts_with("cond.") || waveform_decoder_tensor(path, ""))
}

fn weight_norm_dim_zero<B: Backend>(weight: Tensor<B, 3>) -> Tensor<B, 3> {
    weight.powf_scalar(2.0).sum_dims(&[1usize, 2usize]).sqrt()
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    use super::*;

    type TestBackend = NdArray<f32>;

    fn tiny_model_config() -> VitsInferenceConfig {
        VitsInferenceConfig::from_json5_str(
            r#"{
                model: "vits",
                use_phonemes: true,
                phoneme_language: "en",
                add_blank: true,
                enable_eos_bos_chars: false,
                characters: {
                    characters_class: "fixture.VitsCharacters",
                    pad: "_",
                    eos: "",
                    bos: "",
                    blank: null,
                    characters: "At",
                    punctuations: " ",
                    phonemes: "''ʰɝʃ",
                    is_unique: true,
                    is_sorted: true,
                },
                model_args: {
                    num_chars: 10,
                    out_channels: 5,
                    spec_segment_size: 2,
                    hidden_channels: 4,
                    hidden_channels_ffn_text_encoder: 8,
                    num_heads_text_encoder: 2,
                    num_layers_text_encoder: 1,
                    kernel_size_text_encoder: 3,
                    dropout_p_text_encoder: 0.1,
                    dropout_p_duration_predictor: 0.1,
                    kernel_size_posterior_encoder: 3,
                    dilation_rate_posterior_encoder: 1,
                    num_layers_posterior_encoder: 1,
                    kernel_size_flow: 3,
                    dilation_rate_flow: 1,
                    num_layers_flow: 1,
                    resblock_type_decoder: "1",
                    resblock_kernel_sizes_decoder: [3],
                    resblock_dilation_sizes_decoder: [[1, 2, 3]],
                    upsample_rates_decoder: [2],
                    upsample_initial_channel_decoder: 4,
                    upsample_kernel_sizes_decoder: [4],
                    use_sdp: true,
                    inference_noise_scale: 0.667,
                    length_scale: 1.0,
                    inference_noise_scale_dp: 0.8,
                    max_inference_len: null,
                    use_speaker_embedding: true,
                    num_speakers: 3,
                    speaker_embedding_channels: 4,
                    use_d_vector_file: false,
                    d_vector_dim: 0,
                    condition_dp_on_speaker: true,
                    use_language_embedding: false,
                    embedded_language_dim: 4,
                    num_languages: 0,
                },
                audio: {
                    fft_size: 8,
                    win_length: 8,
                    hop_length: 2,
                    sample_rate: 8000,
                    preemphasis: 0.0,
                    log_func: "np.log10",
                    num_mels: 2,
                    mel_fmin: 0.0,
                    mel_fmax: 4000.0,
                    spec_gain: 20.0,
                    signal_norm: true,
                    min_level_db: -100.0,
                    symmetric_norm: true,
                    max_norm: 4.0,
                    clip_norm: true,
                    stats_path: null,
                    do_amp_to_db_mel: true,
                    stft_pad_mode: "reflect",
                },
            }"#,
        )
        .expect("tiny VITS fixture")
    }

    #[test]
    fn derives_integrated_decoder_topology_from_model_config() {
        let config =
            VitsWaveformDecoderConfig::from_model_config(&tiny_model_config()).expect("topology");
        let generator = config.generator_config();

        assert_eq!(generator.in_channels, 4);
        assert_eq!(generator.cond_channels, 4);
        assert_eq!(generator.out_channels, 1);
        assert_eq!(generator.inference_padding, 0);
        assert!(!generator.conv_pre_weight_norm);
        assert!(!generator.conv_post_weight_norm);
        assert!(!generator.conv_post_bias);
        assert_eq!(generator.upsample_factor(), 2);
    }

    #[test]
    fn decoder_expands_global_conditioning_and_preserves_shape_contract() {
        let device = NdArrayDevice::Cpu;
        TestBackend::seed(&device, 41);
        let decoder = VitsWaveformDecoderConfig::from_model_config(&tiny_model_config())
            .expect("topology")
            .init::<TestBackend>(&device)
            .expect("decoder");
        let latent = Tensor::<TestBackend, 3>::ones([2, 4, 3], &device);
        let speaker = Tensor::<TestBackend, 3>::ones([2, 4, 1], &device);

        let waveform = decoder.forward(latent, Some(speaker)).expect("waveform");

        assert_eq!(waveform.dims(), [2, 1, 6]);
        assert_eq!(decoder.input_channels(), 4);
        assert_eq!(decoder.conditioning_channels(), 4);
        assert_eq!(decoder.output_frames(3), Some(6));
    }

    #[test]
    fn published_topology_and_checkpoint_load_when_provided() {
        let Some(config_path) = std::env::var_os("TONGUES_TEST_COQUI_VITS_CONFIG") else {
            return;
        };
        let config = VitsInferenceConfig::from_file(config_path).expect("published config");
        let topology =
            VitsWaveformDecoderConfig::from_model_config(&config).expect("published topology");
        let generator = topology.generator_config();

        assert_eq!(generator.in_channels, 192);
        assert_eq!(generator.cond_channels, 256);
        assert_eq!(generator.upsample_factor(), 256);
        assert_eq!(generator.inference_padding, 0);
        assert!(!generator.conv_post_bias);

        let Some(checkpoint_path) = std::env::var_os("TONGUES_TEST_COQUI_VITS_CHECKPOINT") else {
            return;
        };
        let device = NdArrayDevice::Cpu;
        topology
            .load_checkpoint::<TestBackend>(checkpoint_path, &device)
            .expect("strict waveform decoder subtree load");
    }
}
