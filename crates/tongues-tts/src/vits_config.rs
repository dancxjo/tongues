use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

use crate::AudioFeatureConfig;

pub const VITS_BLANK_TOKEN: &str = "<BLNK>";

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct VitsCharactersConfig {
    pub characters_class: Option<String>,
    pub pad: String,
    pub eos: Option<String>,
    pub bos: Option<String>,
    pub blank: Option<String>,
    pub characters: String,
    pub punctuations: String,
    pub phonemes: Option<String>,
    #[serde(default)]
    pub is_unique: bool,
    #[serde(default = "default_true")]
    pub is_sorted: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct VitsModelArgs {
    pub num_chars: usize,
    pub out_channels: usize,
    pub spec_segment_size: usize,
    pub hidden_channels: usize,
    pub hidden_channels_ffn_text_encoder: usize,
    pub num_heads_text_encoder: usize,
    pub num_layers_text_encoder: usize,
    pub kernel_size_text_encoder: usize,
    pub dropout_p_text_encoder: f32,
    pub dropout_p_duration_predictor: f32,
    pub kernel_size_posterior_encoder: usize,
    pub dilation_rate_posterior_encoder: usize,
    pub num_layers_posterior_encoder: usize,
    pub kernel_size_flow: usize,
    pub dilation_rate_flow: usize,
    pub num_layers_flow: usize,
    pub resblock_type_decoder: String,
    pub resblock_kernel_sizes_decoder: Vec<usize>,
    pub resblock_dilation_sizes_decoder: Vec<Vec<usize>>,
    pub upsample_rates_decoder: Vec<usize>,
    pub upsample_initial_channel_decoder: usize,
    pub upsample_kernel_sizes_decoder: Vec<usize>,
    pub use_sdp: bool,
    pub inference_noise_scale: f32,
    pub length_scale: f32,
    pub inference_noise_scale_dp: f32,
    pub max_inference_len: Option<usize>,
    pub use_speaker_embedding: bool,
    pub num_speakers: u32,
    pub speaker_embedding_channels: usize,
    pub use_d_vector_file: bool,
    pub d_vector_dim: usize,
    pub condition_dp_on_speaker: bool,
    pub use_language_embedding: bool,
    pub embedded_language_dim: usize,
    pub num_languages: u32,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct VitsModelConfig {
    pub model: String,
    pub use_phonemes: bool,
    pub phoneme_language: Option<String>,
    pub add_blank: bool,
    pub enable_eos_bos_chars: bool,
    pub characters: VitsCharactersConfig,
    pub model_args: VitsModelArgs,
    pub audio: AudioFeatureConfig,
}

impl VitsModelConfig {
    pub fn from_json5_str(source: &str) -> Result<Self> {
        let config: Self =
            json5::from_str(source).context("failed to parse imported VITS config")?;
        config.validate()?;
        Ok(config)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read VITS config {}", path.display()))?;
        Self::from_json5_str(&source)
            .with_context(|| format!("invalid VITS config {}", path.display()))
    }

    pub fn vocabulary(&self) -> Vec<String> {
        let mut model_characters = self.characters.characters.chars().collect::<Vec<_>>();
        if let Some(phonemes) = &self.characters.phonemes {
            model_characters.extend(phonemes.chars());
        }
        if self.characters.is_sorted {
            model_characters.sort_unstable();
        }

        let mut vocabulary = Vec::with_capacity(
            2 + self.characters.punctuations.chars().count() + model_characters.len(),
        );
        vocabulary.push(self.characters.pad.clone());
        vocabulary.extend(
            self.characters
                .punctuations
                .chars()
                .map(|symbol| symbol.to_string()),
        );
        vocabulary.extend(
            model_characters
                .into_iter()
                .map(|symbol| symbol.to_string()),
        );
        vocabulary.push(VITS_BLANK_TOKEN.to_string());
        vocabulary
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model.eq_ignore_ascii_case("vits"),
            "expected VITS model, found `{}`",
            self.model
        );
        ensure!(
            self.use_phonemes,
            "this VITS model configuration requires phoneme input"
        );
        ensure!(
            !self.enable_eos_bos_chars,
            "this VITS vocabulary layout does not use BOS/EOS tokens"
        );
        ensure!(
            !self.characters.pad.is_empty(),
            "VITS padding token must not be empty"
        );
        ensure!(
            self.characters.is_sorted,
            "this VITS vocabulary layout requires sorted model characters"
        );
        if let Some(class) = &self.characters.characters_class {
            ensure!(
                class.ends_with(".VitsCharacters"),
                "unsupported imported VITS character class `{class}`"
            );
        }

        let args = &self.model_args;
        ensure!(
            self.vocabulary().len() == args.num_chars,
            "VITS vocabulary has {} entries but the model expects {}",
            self.vocabulary().len(),
            args.num_chars
        );
        ensure!(
            args.out_channels == self.audio.fft_size / 2 + 1,
            "VITS output channels {} do not match FFT size {}",
            args.out_channels,
            self.audio.fft_size
        );
        ensure!(
            args.hidden_channels > 0
                && args.hidden_channels_ffn_text_encoder > 0
                && args.num_heads_text_encoder > 0
                && args.num_layers_text_encoder > 0,
            "VITS text encoder dimensions must be positive"
        );
        ensure!(
            args.hidden_channels % args.num_heads_text_encoder == 0,
            "VITS hidden channels must divide evenly across attention heads"
        );
        ensure!(
            args.kernel_size_text_encoder > 0
                && args.kernel_size_posterior_encoder > 0
                && args.kernel_size_flow > 0,
            "VITS convolution kernels must be positive"
        );
        ensure!(
            args.dilation_rate_posterior_encoder > 0
                && args.dilation_rate_flow > 0
                && args.num_layers_posterior_encoder > 0
                && args.num_layers_flow > 0,
            "VITS flow and posterior encoder dimensions must be positive"
        );
        ensure!(
            args.resblock_type_decoder == "1" || args.resblock_type_decoder == "2",
            "unsupported VITS decoder residual block type `{}`",
            args.resblock_type_decoder
        );
        ensure!(
            !args.resblock_kernel_sizes_decoder.is_empty()
                && args.resblock_kernel_sizes_decoder.len()
                    == args.resblock_dilation_sizes_decoder.len(),
            "VITS decoder residual kernel and dilation lists differ"
        );
        ensure!(
            args.resblock_dilation_sizes_decoder
                .iter()
                .all(|dilations| !dilations.is_empty()),
            "VITS decoder residual blocks require dilation stages"
        );
        ensure!(
            !args.upsample_rates_decoder.is_empty()
                && args.upsample_rates_decoder.len() == args.upsample_kernel_sizes_decoder.len(),
            "VITS decoder upsample rate and kernel lists differ"
        );
        let upsample = args
            .upsample_rates_decoder
            .iter()
            .try_fold(1usize, |product, rate| product.checked_mul(*rate))
            .context("VITS decoder upsample product overflow")?;
        ensure!(
            upsample == self.audio.hop_length,
            "VITS decoder upsample product {upsample} does not match hop length {}",
            self.audio.hop_length
        );
        ensure!(
            args.spec_segment_size > 0 && args.upsample_initial_channel_decoder > 0,
            "VITS decoder dimensions must be positive"
        );
        ensure!(
            args.inference_noise_scale.is_finite() && args.inference_noise_scale >= 0.0,
            "VITS inference noise scale must be finite and non-negative"
        );
        ensure!(
            args.inference_noise_scale_dp.is_finite() && args.inference_noise_scale_dp >= 0.0,
            "VITS duration noise scale must be finite and non-negative"
        );
        ensure!(
            args.length_scale.is_finite() && args.length_scale > 0.0,
            "VITS length scale must be finite and positive"
        );
        if args.use_speaker_embedding {
            ensure!(
                args.num_speakers > 0 && args.speaker_embedding_channels > 0,
                "VITS speaker embedding dimensions must be positive"
            );
        }
        if args.use_d_vector_file {
            ensure!(
                args.d_vector_dim > 0,
                "VITS d-vector mode requires a positive dimension"
            );
        }
        if args.use_language_embedding {
            ensure!(
                args.num_languages > 0 && args.embedded_language_dim > 0,
                "VITS language embedding dimensions must be positive"
            );
        }
        self.audio.mel_contract()?;
        Ok(())
    }
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
pub(crate) fn test_vits_config() -> VitsModelConfig {
    VitsModelConfig {
        model: "vits".into(),
        use_phonemes: true,
        phoneme_language: Some("en".into()),
        add_blank: true,
        enable_eos_bos_chars: false,
        characters: VitsCharactersConfig {
            characters_class: Some("TTS.tts.models.vits.VitsCharacters".into()),
            pad: "_".into(),
            eos: Some(String::new()),
            bos: Some(String::new()),
            blank: None,
            characters: "At".into(),
            punctuations: " ".into(),
            phonemes: Some("''ʰɝʃ".into()),
            is_unique: true,
            is_sorted: true,
        },
        model_args: VitsModelArgs {
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
            resblock_type_decoder: "1".into(),
            resblock_kernel_sizes_decoder: vec![3],
            resblock_dilation_sizes_decoder: vec![vec![1]],
            upsample_rates_decoder: vec![2],
            upsample_initial_channel_decoder: 4,
            upsample_kernel_sizes_decoder: vec![4],
            use_sdp: true,
            inference_noise_scale: 0.667,
            length_scale: 1.0,
            inference_noise_scale_dp: 0.8,
            max_inference_len: None,
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
        audio: AudioFeatureConfig {
            fft_size: 8,
            win_length: 8,
            hop_length: 2,
            sample_rate: 8_000,
            preemphasis: 0.0,
            log_func: "np.log10".into(),
            num_mels: 2,
            mel_fmin: 0.0,
            mel_fmax: Some(4_000.0),
            spec_gain: 20.0,
            signal_norm: true,
            min_level_db: -100.0,
            symmetric_norm: true,
            max_norm: 4.0,
            clip_norm: true,
            stats_path: None,
            do_amp_to_db_mel: true,
            stft_pad_mode: "reflect".into(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_vocabulary_keeps_duplicate_rows_and_terminal_blank() {
        let config = test_vits_config();

        config.validate().expect("valid VITS config");
        assert_eq!(
            config.vocabulary(),
            vec!["_", " ", "'", "'", "A", "t", "ɝ", "ʃ", "ʰ", "<BLNK>"]
        );
    }

    #[test]
    fn parses_the_published_vctk_config_when_available() {
        let Some(path) = std::env::var_os("TONGUES_TEST_COQUI_VITS_CONFIG") else {
            return;
        };
        let config = VitsModelConfig::from_file(path).expect("published VCTK VITS config");

        assert_eq!(config.model_args.num_chars, 179);
        assert_eq!(config.model_args.out_channels, 513);
        assert_eq!(config.model_args.hidden_channels, 192);
        assert_eq!(config.model_args.num_speakers, 109);
        assert_eq!(config.model_args.speaker_embedding_channels, 256);
        assert_eq!(config.audio.sample_rate, 22_050);
        assert_eq!(config.audio.hop_length, 256);
        assert!(config.model_args.use_sdp);
        assert!(config.model_args.condition_dp_on_speaker);
    }
}
