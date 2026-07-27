//! Configuration adapter for Coqui DelightfulTTS checkpoints.

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const DEFAULT_DELIGHTFUL_MAX_OUTPUT_FRAMES: usize = 20_000;

fn default_num_chars() -> usize {
    100
}
fn default_mel_bins() -> usize {
    100
}
fn default_hidden() -> usize {
    512
}
fn default_layers() -> usize {
    6
}
fn default_heads() -> usize {
    8
}
fn default_dropout() -> f64 {
    0.1
}
fn default_encoder_kernel() -> usize {
    7
}
fn default_decoder_kernel() -> usize {
    11
}
fn default_lrelu_slope() -> f64 {
    0.3
}
fn default_variance_kernel() -> usize {
    5
}
fn default_variance_dropout() -> f64 {
    0.5
}
fn default_variance_embedding_kernel() -> usize {
    3
}
fn default_phoneme_bottleneck() -> usize {
    4
}
fn default_utterance_bottleneck() -> usize {
    512
}
fn default_prosody_kernel() -> usize {
    5
}
fn default_length_scale() -> f64 {
    1.0
}
fn default_max_duration() -> usize {
    75
}
fn default_sample_rate() -> u32 {
    22_050
}
fn default_fft_size() -> usize {
    1_024
}
fn default_hop_length() -> usize {
    256
}
fn default_mel_fmax() -> f32 {
    8_000.0
}
fn default_speaker_embedding_channels() -> usize {
    384
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DelightfulConformerConfig {
    #[serde(default = "default_hidden")]
    pub hidden_channels: usize,
    #[serde(default = "default_layers")]
    pub layers: usize,
    #[serde(default = "default_heads")]
    pub heads: usize,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    pub convolution_kernel_size: usize,
}

impl DelightfulConformerConfig {
    fn validate(&self, label: &str) -> Result<()> {
        ensure!(
            self.hidden_channels > 0 && self.layers > 0 && self.heads > 0,
            "{label} hidden channels, layers, and heads must be positive"
        );
        ensure!(
            self.hidden_channels.is_multiple_of(self.heads),
            "{label} hidden channels must divide evenly across attention heads"
        );
        ensure!(
            (0.0..1.0).contains(&self.dropout),
            "{label} dropout must be in [0, 1)"
        );
        ensure!(
            self.convolution_kernel_size > 0 && !self.convolution_kernel_size.is_multiple_of(2),
            "{label} convolution kernel must be positive and odd"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DelightfulVarianceConfig {
    #[serde(default = "default_hidden")]
    pub hidden_channels: usize,
    #[serde(default = "default_variance_kernel")]
    pub kernel_size: usize,
    #[serde(default = "default_variance_dropout")]
    pub dropout: f64,
    #[serde(default = "default_variance_embedding_kernel")]
    pub embedding_kernel_size: usize,
}

impl Default for DelightfulVarianceConfig {
    fn default() -> Self {
        Self {
            hidden_channels: default_hidden(),
            kernel_size: default_variance_kernel(),
            dropout: default_variance_dropout(),
            embedding_kernel_size: default_variance_embedding_kernel(),
        }
    }
}

impl DelightfulVarianceConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.hidden_channels > 0,
            "DelightfulTTS variance hidden channels must be positive"
        );
        ensure!(
            self.kernel_size > 0 && !self.kernel_size.is_multiple_of(2),
            "DelightfulTTS variance kernel must be positive and odd"
        );
        ensure!(
            self.embedding_kernel_size > 0 && !self.embedding_kernel_size.is_multiple_of(2),
            "DelightfulTTS variance embedding kernel must be positive and odd"
        );
        ensure!(
            (0.0..1.0).contains(&self.dropout),
            "DelightfulTTS variance dropout must be in [0, 1)"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DelightfulProsodyConfig {
    #[serde(default = "default_utterance_bottleneck")]
    pub utterance_bottleneck: usize,
    #[serde(default = "default_phoneme_bottleneck")]
    pub phoneme_bottleneck: usize,
    #[serde(default = "default_prosody_kernel")]
    pub predictor_kernel_size: usize,
}

impl Default for DelightfulProsodyConfig {
    fn default() -> Self {
        Self {
            utterance_bottleneck: default_utterance_bottleneck(),
            phoneme_bottleneck: default_phoneme_bottleneck(),
            predictor_kernel_size: default_prosody_kernel(),
        }
    }
}

impl DelightfulProsodyConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.utterance_bottleneck > 0 && self.phoneme_bottleneck > 0,
            "DelightfulTTS prosody bottlenecks must be positive"
        );
        ensure!(
            self.predictor_kernel_size > 0 && !self.predictor_kernel_size.is_multiple_of(2),
            "DelightfulTTS prosody predictor kernel must be positive and odd"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelightfulSpeakerConfig {
    #[serde(default)]
    pub num_speakers: usize,
    #[serde(default)]
    pub use_speaker_embedding: bool,
    #[serde(default = "default_speaker_embedding_channels")]
    pub speaker_embedding_channels: usize,
    #[serde(default)]
    pub use_d_vector_file: bool,
    #[serde(default)]
    pub d_vector_dim: usize,
}

impl Default for DelightfulSpeakerConfig {
    fn default() -> Self {
        Self {
            num_speakers: 0,
            use_speaker_embedding: false,
            speaker_embedding_channels: default_speaker_embedding_channels(),
            use_d_vector_file: false,
            d_vector_dim: 0,
        }
    }
}

impl DelightfulSpeakerConfig {
    pub fn conditioning_dimensions(&self) -> Option<usize> {
        if self.use_d_vector_file {
            Some(self.d_vector_dim)
        } else if self.use_speaker_embedding {
            Some(self.speaker_embedding_channels)
        } else {
            None
        }
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            !(self.use_speaker_embedding && self.use_d_vector_file),
            "DelightfulTTS cannot use learned speaker IDs and d-vectors simultaneously"
        );
        if self.use_speaker_embedding {
            ensure!(
                self.num_speakers > 0,
                "speaker embeddings require num_speakers > 0"
            );
            ensure!(
                self.speaker_embedding_channels > 0,
                "speaker embedding channels must be positive"
            );
        }
        if self.use_d_vector_file {
            ensure!(
                self.d_vector_dim > 0,
                "d-vector conditioning requires d_vector_dim > 0"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DelightfulAudioConfig {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_hop_length")]
    pub hop_length: usize,
    #[serde(default = "default_fft_size")]
    pub win_length: usize,
    #[serde(default = "default_fft_size")]
    pub fft_size: usize,
    #[serde(default)]
    pub mel_fmin: f32,
    #[serde(default = "default_mel_fmax")]
    pub mel_fmax: f32,
    #[serde(default = "default_mel_bins")]
    pub num_mels: usize,
}

impl Default for DelightfulAudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: default_sample_rate(),
            hop_length: default_hop_length(),
            win_length: default_fft_size(),
            fft_size: default_fft_size(),
            mel_fmin: 0.0,
            mel_fmax: default_mel_fmax(),
            num_mels: default_mel_bins(),
        }
    }
}

impl DelightfulAudioConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.sample_rate > 0
                && self.hop_length > 0
                && self.win_length > 0
                && self.fft_size > 0
                && self.num_mels > 0,
            "DelightfulTTS audio dimensions must be positive"
        );
        ensure!(
            self.win_length <= self.fft_size,
            "DelightfulTTS window length cannot exceed FFT size"
        );
        ensure!(
            self.mel_fmin.is_finite()
                && self.mel_fmax.is_finite()
                && self.mel_fmin >= 0.0
                && self.mel_fmax > self.mel_fmin
                && self.mel_fmax <= self.sample_rate as f32 / 2.0,
            "DelightfulTTS mel frequency bounds are invalid"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DelightfulTtsConfig {
    #[serde(default = "default_num_chars")]
    pub num_chars: usize,
    #[serde(default = "default_mel_bins")]
    pub out_channels: usize,
    pub encoder: DelightfulConformerConfig,
    pub decoder: DelightfulConformerConfig,
    #[serde(default)]
    pub variance: DelightfulVarianceConfig,
    #[serde(default)]
    pub prosody: DelightfulProsodyConfig,
    #[serde(default)]
    pub speakers: DelightfulSpeakerConfig,
    #[serde(default = "default_lrelu_slope")]
    pub leaky_relu_slope: f64,
    #[serde(default = "default_length_scale")]
    pub length_scale: f64,
    #[serde(default = "default_max_duration")]
    pub max_duration: usize,
    #[serde(default = "default_delightful_max_output_frames")]
    pub max_output_frames: usize,
    #[serde(default)]
    pub audio: DelightfulAudioConfig,
}

fn default_delightful_max_output_frames() -> usize {
    DEFAULT_DELIGHTFUL_MAX_OUTPUT_FRAMES
}

impl Default for DelightfulTtsConfig {
    fn default() -> Self {
        Self {
            num_chars: default_num_chars(),
            out_channels: default_mel_bins(),
            encoder: DelightfulConformerConfig {
                hidden_channels: default_hidden(),
                layers: default_layers(),
                heads: default_heads(),
                dropout: default_dropout(),
                convolution_kernel_size: default_encoder_kernel(),
            },
            decoder: DelightfulConformerConfig {
                hidden_channels: default_hidden(),
                layers: default_layers(),
                heads: default_heads(),
                dropout: default_dropout(),
                convolution_kernel_size: default_decoder_kernel(),
            },
            variance: DelightfulVarianceConfig::default(),
            prosody: DelightfulProsodyConfig::default(),
            speakers: DelightfulSpeakerConfig::default(),
            leaky_relu_slope: default_lrelu_slope(),
            length_scale: default_length_scale(),
            max_duration: default_max_duration(),
            max_output_frames: DEFAULT_DELIGHTFUL_MAX_OUTPUT_FRAMES,
            audio: DelightfulAudioConfig::default(),
        }
    }
}

impl DelightfulTtsConfig {
    pub fn from_json_value(root: &Value) -> Result<Self> {
        ensure!(
            root.get("model").and_then(Value::as_str) == Some("delightful_tts"),
            "expected model `delightful_tts`"
        );
        let args = root
            .get("model_args")
            .context("missing DelightfulTTS model_args")?;
        let audio: DelightfulAudioConfig = serde_json::from_value(
            root.get("audio")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({})),
        )
        .context("invalid DelightfulTTS audio config")?;

        let mut config = Self {
            num_chars: usize_field(args, "num_chars")?.unwrap_or_else(|| {
                published_vocabulary_size(root).unwrap_or_else(default_num_chars)
            }),
            out_channels: audio.num_mels,
            encoder: DelightfulConformerConfig {
                hidden_channels: usize_field(args, "n_hidden_conformer_encoder")?
                    .unwrap_or_else(default_hidden),
                layers: usize_field(args, "n_layers_conformer_encoder")?
                    .unwrap_or_else(default_layers),
                heads: usize_field(args, "n_heads_conformer_encoder")?
                    .unwrap_or_else(default_heads),
                dropout: number_field(args, "dropout_conformer_encoder")?
                    .unwrap_or_else(default_dropout),
                convolution_kernel_size: usize_field(
                    args,
                    "kernel_size_conv_mod_conformer_encoder",
                )?
                .unwrap_or_else(default_encoder_kernel),
            },
            decoder: DelightfulConformerConfig {
                hidden_channels: usize_field(args, "n_hidden_conformer_decoder")?
                    .unwrap_or_else(default_hidden),
                layers: usize_field(args, "n_layers_conformer_decoder")?
                    .unwrap_or_else(default_layers),
                heads: usize_field(args, "n_heads_conformer_decoder")?
                    .unwrap_or_else(default_heads),
                dropout: number_field(args, "dropout_conformer_decoder")?
                    .unwrap_or_else(default_dropout),
                convolution_kernel_size: usize_field(
                    args,
                    "kernel_size_conv_mod_conformer_decoder",
                )?
                .unwrap_or_else(default_decoder_kernel),
            },
            variance: DelightfulVarianceConfig {
                hidden_channels: usize_field(args, "n_hidden_variance_adaptor")?
                    .unwrap_or_else(default_hidden),
                kernel_size: usize_field(args, "kernel_size_variance_adaptor")?
                    .unwrap_or_else(default_variance_kernel),
                dropout: number_field(args, "dropout_variance_adaptor")?
                    .unwrap_or_else(default_variance_dropout),
                embedding_kernel_size: usize_field(args, "emb_kernel_size_variance_adaptor")?
                    .unwrap_or_else(default_variance_embedding_kernel),
            },
            prosody: DelightfulProsodyConfig {
                utterance_bottleneck: usize_field(args, "bottleneck_size_u_reference_encoder")?
                    .unwrap_or_else(default_utterance_bottleneck),
                phoneme_bottleneck: usize_field(args, "bottleneck_size_p_reference_encoder")?
                    .unwrap_or_else(default_phoneme_bottleneck),
                predictor_kernel_size: usize_field(
                    args,
                    "predictor_kernel_size_reference_encoder",
                )?
                .unwrap_or_else(default_prosody_kernel),
            },
            speakers: DelightfulSpeakerConfig {
                num_speakers: usize_field(args, "num_speakers")?
                    .or(usize_field(root, "num_speakers")?)
                    .unwrap_or(0),
                use_speaker_embedding: bool_field(args, "use_speaker_embedding")?
                    .or(bool_field(root, "use_speaker_embedding")?)
                    .unwrap_or(false),
                speaker_embedding_channels: usize_field(args, "speaker_embedding_channels")?
                    .unwrap_or_else(default_speaker_embedding_channels),
                use_d_vector_file: bool_field(args, "use_d_vector_file")?
                    .or(bool_field(root, "use_d_vector_file")?)
                    .unwrap_or(false),
                d_vector_dim: usize_field(args, "d_vector_dim")?
                    .or(usize_field(root, "d_vector_dim")?)
                    .unwrap_or(0),
            },
            leaky_relu_slope: number_field(args, "lrelu_slope")?
                .unwrap_or_else(default_lrelu_slope),
            length_scale: number_field(args, "length_scale")?.unwrap_or_else(default_length_scale),
            max_duration: usize_field(args, "max_duration")?.unwrap_or_else(default_max_duration),
            max_output_frames: DEFAULT_DELIGHTFUL_MAX_OUTPUT_FRAMES,
            audio,
        };
        if let Some(num_chars) = usize_field(args, "num_chars")? {
            config.num_chars = num_chars;
        }
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.num_chars > 0 && self.out_channels > 0,
            "DelightfulTTS character and output dimensions must be positive"
        );
        self.encoder.validate("DelightfulTTS encoder")?;
        self.decoder.validate("DelightfulTTS decoder")?;
        ensure!(
            self.encoder.hidden_channels == self.decoder.hidden_channels,
            "native DelightfulTTS currently requires equal encoder and decoder hidden channels"
        );
        ensure!(
            self.variance.hidden_channels == self.encoder.hidden_channels,
            "DelightfulTTS variance hidden channels must match encoder hidden channels"
        );
        ensure!(
            self.out_channels == self.audio.num_mels,
            "DelightfulTTS output channels must match audio mel bins"
        );
        self.variance.validate()?;
        self.prosody.validate()?;
        self.speakers.validate()?;
        ensure!(
            self.leaky_relu_slope.is_finite() && self.leaky_relu_slope >= 0.0,
            "DelightfulTTS leaky-ReLU slope must be finite and non-negative"
        );
        ensure!(
            self.length_scale.is_finite() && self.length_scale > 0.0,
            "DelightfulTTS length scale must be finite and positive"
        );
        ensure!(
            self.max_duration > 0 && self.max_output_frames > 0,
            "DelightfulTTS duration guards must be positive"
        );
        self.audio.validate()?;
        Ok(())
    }
}

fn usize_field(root: &Value, name: &str) -> Result<Option<usize>> {
    root.get(name)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .with_context(|| format!("{name} must be an unsigned integer"))
        })
        .transpose()
}

fn number_field(root: &Value, name: &str) -> Result<Option<f64>> {
    root.get(name)
        .map(|value| {
            value
                .as_f64()
                .with_context(|| format!("{name} must be numeric"))
        })
        .transpose()
}

fn bool_field(root: &Value, name: &str) -> Result<Option<bool>> {
    root.get(name)
        .map(|value| {
            value
                .as_bool()
                .with_context(|| format!("{name} must be a boolean"))
        })
        .transpose()
}

fn published_vocabulary_size(root: &Value) -> Option<usize> {
    let characters = root.get("characters")?;
    ["pad", "eos", "bos", "phonemes", "punctuations"]
        .iter()
        .try_fold(0usize, |total, field| {
            Some(total + characters.get(*field)?.as_str()?.chars().count())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_upstream_defaults_and_audio_contract() {
        let root = serde_json::json!({
            "model": "delightful_tts",
            "audio": {
                "sample_rate": 22050,
                "hop_length": 256,
                "win_length": 1024,
                "fft_size": 1024,
                "mel_fmin": 0.0,
                "mel_fmax": 8000.0,
                "num_mels": 100
            },
            "model_args": {
                "num_chars": 130,
                "n_hidden_conformer_encoder": 512,
                "n_layers_conformer_encoder": 6,
                "n_heads_conformer_encoder": 8,
                "dropout_conformer_encoder": 0.1,
                "kernel_size_conv_mod_conformer_encoder": 7,
                "n_hidden_conformer_decoder": 512,
                "n_layers_conformer_decoder": 6,
                "n_heads_conformer_decoder": 8,
                "dropout_conformer_decoder": 0.1,
                "kernel_size_conv_mod_conformer_decoder": 11,
                "bottleneck_size_p_reference_encoder": 4,
                "bottleneck_size_u_reference_encoder": 512,
                "predictor_kernel_size_reference_encoder": 5,
                "n_hidden_variance_adaptor": 512,
                "kernel_size_variance_adaptor": 5,
                "dropout_variance_adaptor": 0.5,
                "emb_kernel_size_variance_adaptor": 3,
                "use_speaker_embedding": false,
                "num_speakers": 0,
                "use_d_vector_file": false,
                "d_vector_dim": 0,
                "lrelu_slope": 0.3,
                "length_scale": 1.0
            }
        });
        let config = DelightfulTtsConfig::from_json_value(&root).expect("config");
        assert_eq!(
            config,
            DelightfulTtsConfig {
                num_chars: 130,
                ..DelightfulTtsConfig::default()
            }
        );
    }

    #[test]
    fn validates_conditioning_modes() {
        let mut config = DelightfulTtsConfig::default();
        config.speakers.use_speaker_embedding = true;
        config.speakers.num_speakers = 4;
        config.validate().expect("speaker IDs");
        assert_eq!(config.speakers.conditioning_dimensions(), Some(384));

        config.speakers.use_d_vector_file = true;
        config.speakers.d_vector_dim = 256;
        assert!(config.validate().is_err());
    }
}
