// SPDX-License-Identifier: MPL-2.0
//! Configuration adapter for Coqui Glow-TTS-family acoustic checkpoints.
//!
//! Coqui shipped two materially different configuration generations for
//! Glow-TTS.  The original LJSpeech release used names such as
//! `hidden_channels_encoder`; later releases use `hidden_channels_enc`.  This
//! adapter accepts both spellings and lowers them into one inference-only
//! topology.
//!
//! Source provenance: adapted from the configuration contract in Coqui TTS
//! revision `0cf3265a4686d7e856bd472cdaf1572d61cab2b8`. See
//! `THIRD_PARTY_NOTICES.md`.

use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    AudioFeatureConfig, PhonemeTokenizerConfig, PhonemeVocabularyProjector, SpectrogramContract,
};

fn default_hidden_channels() -> usize {
    192
}

fn default_duration_channels() -> usize {
    256
}

fn default_out_channels() -> usize {
    80
}

fn default_flow_blocks() -> usize {
    12
}

fn default_decoder_kernel() -> usize {
    5
}

fn default_one() -> usize {
    1
}

fn default_coupling_layers() -> usize {
    4
}

fn default_num_splits() -> usize {
    4
}

fn default_num_squeeze() -> usize {
    2
}

fn default_length_scale() -> f32 {
    1.0
}

fn default_duration_noise_scale() -> f32 {
    0.8
}

fn default_encoder_type() -> String {
    "rel_pos_transformer".into()
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlowTtsEncoderConfig {
    #[serde(default = "default_decoder_kernel")]
    pub kernel_size: usize,
    #[serde(default = "default_encoder_dropout")]
    pub dropout_p: f32,
    #[serde(default = "default_encoder_layers")]
    pub num_layers: usize,
    #[serde(default = "default_encoder_heads")]
    pub num_heads: usize,
    #[serde(default = "default_encoder_ffn")]
    pub hidden_channels_ffn: usize,
    #[serde(default)]
    pub input_length: Option<usize>,
    #[serde(default)]
    pub rel_attn_window_size: Option<usize>,
    #[serde(default = "default_layer_norm_type")]
    pub layer_norm_type: String,
}

fn default_layer_norm_type() -> String {
    "1".into()
}

fn default_encoder_dropout() -> f32 {
    0.1
}

fn default_encoder_layers() -> usize {
    6
}

fn default_encoder_heads() -> usize {
    2
}

fn default_encoder_ffn() -> usize {
    768
}

impl Default for GlowTtsEncoderConfig {
    fn default() -> Self {
        Self {
            kernel_size: default_decoder_kernel(),
            dropout_p: default_encoder_dropout(),
            num_layers: default_encoder_layers(),
            num_heads: default_encoder_heads(),
            hidden_channels_ffn: default_encoder_ffn(),
            input_length: None,
            rel_attn_window_size: None,
            layer_norm_type: default_layer_norm_type(),
        }
    }
}

/// Inference topology shared by Glow-TTS and speaker-conditioned SC-GlowTTS.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlowTtsNetworkConfig {
    pub num_chars: usize,
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,
    #[serde(default = "default_hidden_channels")]
    pub hidden_channels_enc: usize,
    #[serde(default = "default_hidden_channels")]
    pub hidden_channels_dec: usize,
    #[serde(default = "default_duration_channels")]
    pub hidden_channels_dp: usize,
    #[serde(default = "default_encoder_dropout")]
    pub dropout_p_dp: f32,
    #[serde(default = "default_decoder_dropout")]
    pub dropout_p_dec: f32,
    #[serde(default = "default_true")]
    pub mean_only: bool,
    #[serde(default = "default_flow_blocks")]
    pub num_flow_blocks_dec: usize,
    #[serde(default = "default_decoder_kernel")]
    pub kernel_size_dec: usize,
    #[serde(default = "default_one")]
    pub dilation_rate: usize,
    #[serde(default = "default_coupling_layers")]
    pub num_block_layers: usize,
    #[serde(default = "default_num_splits")]
    pub num_splits: usize,
    #[serde(default = "default_num_squeeze")]
    pub num_squeeze: usize,
    #[serde(default)]
    pub sigmoid_scale: bool,
    #[serde(default = "default_encoder_type")]
    pub encoder_type: String,
    #[serde(default)]
    pub encoder_params: GlowTtsEncoderConfig,
    #[serde(default = "default_true")]
    pub use_encoder_prenet: bool,
    #[serde(default)]
    pub use_speaker_embedding: bool,
    #[serde(default)]
    pub use_d_vector_file: bool,
    #[serde(default)]
    pub d_vector_dim: usize,
    #[serde(default)]
    pub num_speakers: usize,
    #[serde(default)]
    pub use_sdp: bool,
    #[serde(default)]
    pub inference_noise_scale: f32,
    #[serde(default = "default_duration_noise_scale")]
    pub inference_noise_scale_dp: f32,
    #[serde(default = "default_length_scale")]
    pub length_scale: f32,
}

fn default_decoder_dropout() -> f32 {
    0.05
}

impl GlowTtsNetworkConfig {
    pub fn speaker_conditioning_channels(&self) -> usize {
        if self.use_d_vector_file {
            self.d_vector_dim
        } else if self.use_speaker_embedding {
            self.hidden_channels_enc
        } else {
            0
        }
    }

    pub fn is_speaker_conditioned(&self) -> bool {
        self.speaker_conditioning_channels() > 0
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.num_chars > 0
                && self.out_channels > 0
                && self.hidden_channels_enc > 0
                && self.hidden_channels_dec > 0
                && self.hidden_channels_dp > 0,
            "Glow-TTS character, mel, encoder, decoder, and duration dimensions must be positive"
        );
        ensure!(
            self.encoder_type
                .eq_ignore_ascii_case("rel_pos_transformer"),
            "unsupported Glow-TTS encoder type `{}`",
            self.encoder_type
        );
        ensure!(
            self.use_encoder_prenet,
            "Glow-TTS checkpoints without the encoder prenet are not supported"
        );
        ensure!(
            self.encoder_params.num_heads > 0
                && self.encoder_params.num_layers > 0
                && self.encoder_params.hidden_channels_ffn > 0
                && self.encoder_params.kernel_size > 0,
            "Glow-TTS encoder dimensions must be positive"
        );
        ensure!(
            self.hidden_channels_enc
                .is_multiple_of(self.encoder_params.num_heads),
            "Glow-TTS encoder channels must divide evenly across attention heads"
        );
        ensure!(
            self.encoder_params.kernel_size % 2 == 1,
            "Glow-TTS encoder FFN kernel must be odd"
        );
        ensure!(
            self.encoder_params.layer_norm_type == "1",
            "Glow-TTS layer_norm_type `{}` is unsupported; published Glow/SC-Glow checkpoints use type 1",
            self.encoder_params.layer_norm_type
        );
        ensure!(
            (0.0..1.0).contains(&self.encoder_params.dropout_p)
                && (0.0..1.0).contains(&self.dropout_p_dp)
                && (0.0..1.0).contains(&self.dropout_p_dec),
            "Glow-TTS dropout values must be in [0, 1)"
        );
        ensure!(
            self.num_flow_blocks_dec > 0
                && self.kernel_size_dec > 0
                && self.kernel_size_dec % 2 == 1
                && self.dilation_rate > 0
                && self.num_block_layers > 0,
            "Glow-TTS decoder block counts, dilation, and odd kernel must be positive"
        );
        ensure!(
            self.hidden_channels_dec % 2 == 0,
            "Glow-TTS decoder hidden channels must be even"
        );
        ensure!(
            self.num_squeeze > 0
                && self.num_splits > 0
                && self.num_splits % 2 == 0
                && (self.out_channels * self.num_squeeze).is_multiple_of(self.num_splits),
            "Glow-TTS squeeze channels must divide evenly across an even invertible-convolution split"
        );
        ensure!(
            self.out_channels * self.num_squeeze >= 2
                && (self.out_channels * self.num_squeeze).is_multiple_of(2),
            "Glow-TTS squeezed mel channels must be positive and even"
        );
        ensure!(
            !(self.use_d_vector_file && !self.use_speaker_embedding),
            "Glow-TTS d-vector conditioning requires use_speaker_embedding=true"
        );
        ensure!(
            !self.use_d_vector_file || self.d_vector_dim > 0,
            "SC-GlowTTS d-vector dimensions must be positive"
        );
        ensure!(
            self.use_d_vector_file || !self.use_speaker_embedding || self.num_speakers > 0,
            "learned Glow-TTS speaker embeddings require num_speakers > 0"
        );
        for (label, value) in [
            ("inference_noise_scale", self.inference_noise_scale),
            ("inference_noise_scale_dp", self.inference_noise_scale_dp),
        ] {
            ensure!(
                value.is_finite() && value >= 0.0,
                "Glow-TTS {label} must be finite and non-negative"
            );
        }
        ensure!(
            self.length_scale.is_finite() && self.length_scale > 0.0,
            "Glow-TTS length_scale must be finite and positive"
        );
        Ok(())
    }
}

/// Canonical inference configuration retained in Tongues model packages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlowTtsInferenceConfig {
    pub network: GlowTtsNetworkConfig,
    pub audio: AudioFeatureConfig,
    pub tokenizer: PhonemeTokenizerConfig,
}

impl GlowTtsInferenceConfig {
    pub fn from_json5_str(source: &str) -> Result<Self> {
        let imported: ImportedGlowTtsConfig =
            json5::from_str(source).context("failed to parse imported Glow-TTS config")?;
        imported.inference_config()
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read Glow-TTS config {}", path.display()))?;
        Self::from_json5_str(&source)
            .with_context(|| format!("invalid Glow-TTS config {}", path.display()))
    }

    pub fn output_contract(&self) -> Result<SpectrogramContract> {
        self.audio.mel_contract()
    }

    pub fn validate(&self) -> Result<()> {
        self.network.validate()?;
        let projector =
            PhonemeVocabularyProjector::from_legacy_config_with_duplicates(self.tokenizer.clone())?;
        ensure!(
            projector.vocabulary().len() == self.network.num_chars,
            "Glow-TTS vocabulary has {} entries but the model expects {}",
            projector.vocabulary().len(),
            self.network.num_chars
        );
        let contract = self.output_contract()?;
        ensure!(
            contract.bins == self.network.out_channels,
            "Glow-TTS emits {} mel bins but audio config declares {}",
            self.network.out_channels,
            contract.bins
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ImportedGlowTtsConfig {
    model: String,
    #[serde(flatten)]
    tokenizer: PhonemeTokenizerConfig,
    audio: AudioFeatureConfig,
    num_chars: Option<usize>,
    #[serde(default = "default_out_channels")]
    out_channels: usize,
    #[serde(default = "default_hidden_channels", alias = "hidden_channels_encoder")]
    hidden_channels_enc: usize,
    #[serde(default = "default_hidden_channels", alias = "hidden_channels_decoder")]
    hidden_channels_dec: usize,
    #[serde(
        default = "default_duration_channels",
        alias = "hidden_channels_duration_predictor"
    )]
    hidden_channels_dp: usize,
    #[serde(default = "default_encoder_dropout")]
    dropout_p_dp: f32,
    #[serde(default = "default_decoder_dropout")]
    dropout_p_dec: f32,
    #[serde(default = "default_true")]
    mean_only: bool,
    #[serde(default = "default_flow_blocks")]
    num_flow_blocks_dec: usize,
    #[serde(default = "default_decoder_kernel")]
    kernel_size_dec: usize,
    #[serde(default = "default_one")]
    dilation_rate: usize,
    #[serde(default = "default_coupling_layers")]
    num_block_layers: usize,
    #[serde(default = "default_num_splits")]
    num_splits: usize,
    #[serde(default = "default_num_squeeze")]
    num_squeeze: usize,
    #[serde(default)]
    sigmoid_scale: bool,
    #[serde(default = "default_encoder_type")]
    encoder_type: String,
    #[serde(default)]
    encoder_params: GlowTtsEncoderConfig,
    #[serde(default = "default_true")]
    use_encoder_prenet: bool,
    #[serde(default)]
    use_speaker_embedding: bool,
    #[serde(default, alias = "use_external_speaker_embedding_file")]
    use_d_vector_file: bool,
    #[serde(default)]
    d_vector_dim: usize,
    #[serde(default)]
    num_speakers: usize,
    #[serde(default)]
    use_sdp: bool,
    #[serde(default)]
    inference_noise_scale: f32,
    #[serde(default = "default_duration_noise_scale")]
    inference_noise_scale_dp: f32,
    #[serde(default = "default_length_scale")]
    length_scale: f32,
}

impl ImportedGlowTtsConfig {
    fn inference_config(self) -> Result<GlowTtsInferenceConfig> {
        ensure!(
            self.model.eq_ignore_ascii_case("glow_tts")
                || self.model.eq_ignore_ascii_case("glow-tts"),
            "expected Glow-TTS model, found `{}`",
            self.model
        );
        let projector =
            PhonemeVocabularyProjector::from_legacy_config_with_duplicates(self.tokenizer.clone())?;
        let network = GlowTtsNetworkConfig {
            num_chars: self.num_chars.unwrap_or(projector.vocabulary().len()),
            out_channels: self.out_channels,
            hidden_channels_enc: self.hidden_channels_enc,
            hidden_channels_dec: self.hidden_channels_dec,
            hidden_channels_dp: self.hidden_channels_dp,
            dropout_p_dp: self.dropout_p_dp,
            dropout_p_dec: self.dropout_p_dec,
            mean_only: self.mean_only,
            num_flow_blocks_dec: self.num_flow_blocks_dec,
            kernel_size_dec: self.kernel_size_dec,
            dilation_rate: self.dilation_rate,
            num_block_layers: self.num_block_layers,
            num_splits: self.num_splits,
            num_squeeze: self.num_squeeze,
            sigmoid_scale: self.sigmoid_scale,
            encoder_type: self.encoder_type,
            encoder_params: self.encoder_params,
            use_encoder_prenet: self.use_encoder_prenet,
            use_speaker_embedding: self.use_speaker_embedding,
            use_d_vector_file: self.use_d_vector_file,
            d_vector_dim: self.d_vector_dim,
            num_speakers: self.num_speakers,
            use_sdp: self.use_sdp,
            inference_noise_scale: self.inference_noise_scale,
            inference_noise_scale_dp: self.inference_noise_scale_dp,
            length_scale: self.length_scale,
        };
        let config = GlowTtsInferenceConfig {
            network,
            audio: self.audio,
            tokenizer: self.tokenizer,
        };
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEGACY_CONFIG: &str = r#"{
      model: "glow_tts",
      use_phonemes: true,
      phoneme_language: "en-us",
      add_blank: false,
      enable_eos_bos_chars: false,
      characters: {
        pad: "_", eos: "~", bos: "^", blank: null,
        characters: "abc", punctuations: "! ",
        phonemes: "tk"
      },
      audio: {
        fft_size: 1024, win_length: 1024, hop_length: 256,
        sample_rate: 22050, num_mels: 80, mel_fmin: 50.0,
        mel_fmax: 7600.0, spec_gain: 1.0, signal_norm: false
      },
      hidden_channels_encoder: 192,
      hidden_channels_decoder: 192,
      hidden_channels_duration_predictor: 256,
      use_encoder_prenet: true,
      encoder_type: "rel_pos_transformer",
      encoder_params: {
        kernel_size: 3, dropout_p: 0.1, num_layers: 6,
        num_heads: 2, hidden_channels_ffn: 768
      },
      use_speaker_embedding: false
    }"#;

    #[test]
    fn legacy_field_names_lower_to_canonical_topology() {
        let config =
            GlowTtsInferenceConfig::from_json5_str(LEGACY_CONFIG).expect("legacy Glow-TTS config");
        assert_eq!(config.network.num_chars, 7);
        assert_eq!(config.network.hidden_channels_enc, 192);
        assert_eq!(config.network.hidden_channels_dec, 192);
        assert_eq!(config.network.hidden_channels_dp, 256);
        assert!(!config.network.is_speaker_conditioned());
        assert_eq!(config.output_contract().expect("mel contract").bins, 80);
    }

    #[test]
    fn speaker_conditioning_and_stochastic_duration_are_explicit() {
        let source = LEGACY_CONFIG.replace(
            "use_speaker_embedding: false",
            "use_speaker_embedding: true, use_d_vector_file: true, d_vector_dim: 256, use_sdp: true",
        );
        let config = GlowTtsInferenceConfig::from_json5_str(&source).expect("SC-GlowTTS config");
        assert_eq!(config.network.speaker_conditioning_channels(), 256);
        assert!(config.network.use_sdp);
    }
}
