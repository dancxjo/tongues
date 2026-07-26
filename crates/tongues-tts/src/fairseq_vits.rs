//! Native tokenizer and configuration boundary for Fairseq MMS VITS models.
//!
//! MMS checkpoints use the original VITS module names (`enc_p`, `dp`, `flow`,
//! and `dec`) and a model-local `vocab.txt`.  This module deliberately keeps
//! that external contract separate from Coqui's renamed VITS package layout.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use speaking::UtterancePlan;

use crate::{
    AudioFeatureConfig, LinguisticInputKind, LinguisticIntent, LinguisticProjector,
    ModelInputContract, PhonemeTokenIds, VitsInferenceConfig, VitsNetworkConfig,
};

pub const FAIRSEQ_MMS_CHECKPOINT: &str = "G_100000.pth";
pub const FAIRSEQ_MMS_CONFIG: &str = "config.json";
pub const FAIRSEQ_MMS_VOCAB: &str = "vocab.txt";
pub const FAIRSEQ_MMS_LICENSE: &str = "CC-BY-NC-4.0";
pub const FAIRSEQ_MMS_LICENSE_EVIDENCE: &str =
    "https://huggingface.co/facebook/mms-tts/blob/44cc7fb408064ef9ea6e7c59130d88cac1274671/README.md";
pub const FAIRSEQ_MMS_SOURCE: &str = "https://huggingface.co/facebook/mms-tts";
pub const UROMAN_ENVIRONMENT_VARIABLE: &str = "TONGUES_UROMAN";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FairseqPreprocessingRequirement {
    None,
    Uroman,
}

impl FairseqPreprocessingRequirement {
    pub fn as_catalog_value(self) -> &'static str {
        match self {
            Self::None => "lowercase-and-filter-vocab",
            Self::Uroman => "uroman",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct FairseqTrainConfig {
    segment_size: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct FairseqDataConfig {
    training_files: String,
    sampling_rate: u32,
    filter_length: usize,
    hop_length: usize,
    win_length: usize,
    n_mel_channels: usize,
    mel_fmin: f32,
    mel_fmax: Option<f32>,
    add_blank: bool,
    n_speakers: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct FairseqModelConfig {
    inter_channels: usize,
    hidden_channels: usize,
    filter_channels: usize,
    n_heads: usize,
    n_layers: usize,
    kernel_size: usize,
    p_dropout: f32,
    resblock: String,
    resblock_kernel_sizes: Vec<usize>,
    resblock_dilation_sizes: Vec<Vec<usize>>,
    upsample_rates: Vec<usize>,
    upsample_initial_channel: usize,
    upsample_kernel_sizes: Vec<usize>,
}

/// The inference-relevant subset of the original MMS `config.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FairseqVitsConfig {
    train: FairseqTrainConfig,
    data: FairseqDataConfig,
    model: FairseqModelConfig,
}

impl FairseqVitsConfig {
    pub fn from_json_str(source: &str) -> Result<Self> {
        let config: Self =
            serde_json::from_str(source).context("failed to parse Fairseq MMS VITS config")?;
        config.validate()?;
        Ok(config)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read Fairseq MMS config {}", path.display()))?;
        Self::from_json_str(&source)
            .with_context(|| format!("invalid Fairseq MMS config {}", path.display()))
    }

    pub fn preprocessing(&self) -> FairseqPreprocessingRequirement {
        if self
            .data
            .training_files
            .rsplit_once('.')
            .is_some_and(|(_, suffix)| suffix.eq_ignore_ascii_case("uroman"))
        {
            FairseqPreprocessingRequirement::Uroman
        } else {
            FairseqPreprocessingRequirement::None
        }
    }

    pub fn sample_rate_hz(&self) -> u32 {
        self.data.sampling_rate
    }

    pub fn add_blank(&self) -> bool {
        self.data.add_blank
    }

    pub fn inference_config(&self, vocabulary_size: usize) -> Result<VitsInferenceConfig> {
        ensure!(vocabulary_size > 0, "Fairseq MMS vocabulary is empty");
        let config = VitsInferenceConfig {
            network: VitsNetworkConfig {
                num_chars: vocabulary_size,
                out_channels: self.data.filter_length / 2 + 1,
                spec_segment_size: self.train.segment_size / self.data.hop_length,
                hidden_channels: self.model.inter_channels,
                hidden_channels_ffn_text_encoder: self.model.filter_channels,
                num_heads_text_encoder: self.model.n_heads,
                num_layers_text_encoder: self.model.n_layers,
                kernel_size_text_encoder: self.model.kernel_size,
                dropout_p_text_encoder: self.model.p_dropout,
                dropout_p_duration_predictor: 0.5,
                // These posterior settings are part of the training graph and
                // are not loaded for inference, but retain the published VITS
                // topology in neutral metadata.
                kernel_size_posterior_encoder: 5,
                dilation_rate_posterior_encoder: 1,
                num_layers_posterior_encoder: 16,
                kernel_size_flow: 5,
                dilation_rate_flow: 1,
                num_layers_flow: 4,
                resblock_type_decoder: self.model.resblock.clone(),
                resblock_kernel_sizes_decoder: self.model.resblock_kernel_sizes.clone(),
                resblock_dilation_sizes_decoder: self.model.resblock_dilation_sizes.clone(),
                upsample_rates_decoder: self.model.upsample_rates.clone(),
                upsample_initial_channel_decoder: self.model.upsample_initial_channel,
                upsample_kernel_sizes_decoder: self.model.upsample_kernel_sizes.clone(),
                use_sdp: true,
                inference_noise_scale: 0.667,
                length_scale: 1.0,
                inference_noise_scale_dp: 0.8,
                max_inference_len: None,
                use_speaker_embedding: false,
                num_speakers: 0,
                speaker_embedding_channels: 0,
                use_d_vector_file: false,
                d_vector_dim: 0,
                condition_dp_on_speaker: false,
                use_language_embedding: false,
                embedded_language_dim: 0,
                num_languages: 0,
            },
            audio: AudioFeatureConfig {
                fft_size: self.data.filter_length,
                win_length: self.data.win_length,
                hop_length: self.data.hop_length,
                sample_rate: self.data.sampling_rate,
                preemphasis: 0.0,
                log_func: "np.log".into(),
                num_mels: self.data.n_mel_channels,
                mel_fmin: self.data.mel_fmin,
                mel_fmax: self.data.mel_fmax,
                spec_gain: 1.0,
                signal_norm: false,
                min_level_db: -100.0,
                ref_level_db: None,
                symmetric_norm: false,
                max_norm: 1.0,
                clip_norm: false,
                stats_path: None,
                stats_sha256: None,
                do_amp_to_db_mel: false,
                stft_pad_mode: "reflect".into(),
                centered: true,
                stft_manual_padding: None,
            },
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.data.n_speakers == 0,
            "Fairseq MMS adapter supports the published single-speaker checkpoints; config declares {} speakers",
            self.data.n_speakers
        );
        ensure!(
            self.model.inter_channels == self.model.hidden_channels,
            "Fairseq MMS inter_channels {} differs from hidden_channels {}",
            self.model.inter_channels,
            self.model.hidden_channels
        );
        ensure!(
            self.data.filter_length > 0
                && self.data.filter_length.is_multiple_of(2)
                && self.data.hop_length > 0
                && self.data.win_length > 0
                && self.data.sampling_rate > 0,
            "Fairseq MMS audio dimensions must be positive and filter_length must be even"
        );
        ensure!(
            self.train.segment_size > 0
                && self.train.segment_size.is_multiple_of(self.data.hop_length),
            "Fairseq MMS segment size must be a positive multiple of hop length"
        );
        ensure!(
            !self.model.resblock_kernel_sizes.is_empty()
                && self.model.resblock_kernel_sizes.len()
                    == self.model.resblock_dilation_sizes.len(),
            "Fairseq MMS residual block kernels and dilations differ"
        );
        ensure!(
            !self.model.upsample_rates.is_empty()
                && self.model.upsample_rates.len() == self.model.upsample_kernel_sizes.len(),
            "Fairseq MMS upsample rates and kernels differ"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FairseqTokenization {
    pub normalized_text: String,
    pub ids: Vec<i64>,
    #[serde(default)]
    pub filtered_symbols: Vec<String>,
}

/// Exact `vocab.txt` tokenizer used by the published MMS inference script.
#[derive(Debug, Clone)]
pub struct FairseqVitsTokenizer {
    language: String,
    symbols: Vec<String>,
    symbol_to_id: BTreeMap<String, i64>,
    add_blank: bool,
    preprocessing: FairseqPreprocessingRequirement,
}

impl FairseqVitsTokenizer {
    pub fn from_vocab_str(
        language: impl Into<String>,
        source: &str,
        add_blank: bool,
        preprocessing: FairseqPreprocessingRequirement,
    ) -> Result<Self> {
        let language = language.into();
        ensure!(
            !language.trim().is_empty(),
            "Fairseq MMS language id is empty"
        );
        let symbols = source
            .lines()
            .map(|line| line.strip_suffix('\r').unwrap_or(line).to_string())
            .collect::<Vec<_>>();
        ensure!(!symbols.is_empty(), "Fairseq MMS vocabulary is empty");
        // The published mapper hard-codes row 0 as the interspersed blank,
        // even though some vocabularies also assign that row to a printable
        // character. Preserve the checkpoint's exact row identities.
        let mut symbol_to_id = BTreeMap::new();
        for (id, symbol) in symbols.iter().enumerate() {
            ensure!(
                symbol.chars().count() <= 1,
                "Fairseq MMS vocabulary row {id} is not a single Unicode scalar: {symbol:?}"
            );
            ensure!(
                !symbol_to_id.contains_key(symbol),
                "Fairseq MMS vocabulary repeats symbol {symbol:?}"
            );
            symbol_to_id.insert(
                symbol.clone(),
                i64::try_from(id).context("Fairseq MMS vocabulary is too large")?,
            );
        }
        Ok(Self {
            language,
            symbols,
            symbol_to_id,
            add_blank,
            preprocessing,
        })
    }

    pub fn from_file(
        language: impl Into<String>,
        path: impl AsRef<Path>,
        add_blank: bool,
        preprocessing: FairseqPreprocessingRequirement,
    ) -> Result<Self> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read Fairseq MMS vocab {}", path.display()))?;
        Self::from_vocab_str(language, &source, add_blank, preprocessing)
            .with_context(|| format!("invalid Fairseq MMS vocab {}", path.display()))
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }

    pub fn preprocessing(&self) -> FairseqPreprocessingRequirement {
        self.preprocessing
    }

    /// Tokenize text that has already satisfied the declared preprocessing
    /// contract. This is useful for conformance fixtures and callers that
    /// manage Uroman themselves.
    pub fn encode_preprocessed(&self, text: &str) -> Result<FairseqTokenization> {
        let mut normalized = text.trim().to_lowercase();
        if base_language(&self.language) == "ron" {
            normalized = normalized.replace('ț', "ţ");
        }
        normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");

        let mut ids = Vec::new();
        let mut filtered_symbols = Vec::new();
        for symbol in normalized.chars() {
            let symbol = symbol.to_string();
            if let Some(id) = self.symbol_to_id.get(&symbol) {
                ids.push(*id);
            } else {
                filtered_symbols.push(symbol);
            }
        }
        ensure!(
            !ids.is_empty(),
            "Fairseq MMS preprocessing left no in-vocabulary symbols for language `{}`",
            self.language
        );
        if self.add_blank {
            let mut interspersed = vec![0; ids.len() * 2 + 1];
            for (slot, id) in interspersed.iter_mut().skip(1).step_by(2).zip(ids) {
                *slot = id;
            }
            ids = interspersed;
        }
        Ok(FairseqTokenization {
            normalized_text: normalized,
            ids,
            filtered_symbols,
        })
    }

    /// Apply any model-declared preprocessing and then reproduce the upstream
    /// lower-case, OOV-filter, and blank-intersperse tokenizer.
    pub fn encode(&self, text: &str) -> Result<FairseqTokenization> {
        let preprocessed = match self.preprocessing {
            FairseqPreprocessingRequirement::None => text.to_string(),
            FairseqPreprocessingRequirement::Uroman => self.run_uroman(text)?,
        };
        self.encode_preprocessed(&preprocessed)
    }

    fn run_uroman(&self, text: &str) -> Result<String> {
        let configured = std::env::var_os(UROMAN_ENVIRONMENT_VARIABLE).with_context(|| {
            format!(
                "Fairseq MMS model `{}` requires Uroman preprocessing; set {UROMAN_ENVIRONMENT_VARIABLE} to uroman.pl or a compatible executable",
                self.language
            )
        })?;
        let path = PathBuf::from(configured);
        ensure!(
            path.is_file(),
            "{UROMAN_ENVIRONMENT_VARIABLE} points to missing file {}",
            path.display()
        );
        let mut command = if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("pl"))
        {
            let mut command = Command::new("perl");
            command.arg(&path);
            command
        } else {
            Command::new(&path)
        };
        let mut child = command
            // The published reference uses `xxx`; preserve that behavior
            // instead of inventing language-specific Uroman normalization.
            .args(["-l", "xxx"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("failed to start Uroman from {}", path.display()))?;
        child
            .stdin
            .take()
            .context("failed to open Uroman stdin")?
            .write_all(text.as_bytes())
            .context("failed to write text to Uroman")?;
        let output = child
            .wait_with_output()
            .context("failed while waiting for Uroman")?;
        ensure!(
            output.status.success(),
            "Uroman preprocessing failed for `{}`: {}",
            self.language,
            String::from_utf8_lossy(&output.stderr).trim()
        );
        let output = String::from_utf8(output.stdout).context("Uroman emitted non-UTF-8 text")?;
        let output = output.split_whitespace().collect::<Vec<_>>().join(" ");
        ensure!(
            !output.is_empty(),
            "Uroman preprocessing produced empty text for `{}`",
            self.language
        );
        Ok(output)
    }
}

fn base_language(language: &str) -> &str {
    language.split("-script_").next().unwrap_or(language)
}

/// Terminal projector from a Tongues plan to a model-local MMS vocabulary.
#[derive(Debug, Clone)]
pub struct FairseqVitsProjector {
    tokenizer: FairseqVitsTokenizer,
    contract: ModelInputContract,
}

impl FairseqVitsProjector {
    pub fn new(tokenizer: FairseqVitsTokenizer) -> Result<Self> {
        let contract = ModelInputContract {
            kind: LinguisticInputKind::Graphemes,
            vocabulary_fingerprint: format!(
                "fairseq-mms-vits-v1:{}:{:?}",
                tokenizer.language, tokenizer.symbols
            ),
            // Catalog metadata carries the exact variety mapping. The
            // checkpoint projector consumes its already-selected model's text
            // rather than inferring a model from a variety prefix.
            supported_varieties: vec!["*".into()],
            consumes: BTreeSet::from([LinguisticIntent::Text]),
        };
        contract.validate()?;
        Ok(Self {
            tokenizer,
            contract,
        })
    }

    pub fn tokenizer(&self) -> &FairseqVitsTokenizer {
        &self.tokenizer
    }
}

impl LinguisticProjector for FairseqVitsProjector {
    type ModelInput = PhonemeTokenIds;

    fn contract(&self) -> &ModelInputContract {
        &self.contract
    }

    fn project(&self, plan: &UtterancePlan) -> Result<Self::ModelInput> {
        self.contract.ensure_supports(plan)?;
        let text = plan
            .intended_text
            .as_deref()
            .context("Fairseq MMS VITS requires intended text in the utterance plan")?;
        let tokenization = self.tokenizer.encode(text)?;
        Ok(PhonemeTokenIds {
            ids: tokenization.ids,
            projected_symbols: tokenization.normalized_text,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENGLISH_VOCAB: &str =
        "k\n'\nz\ny\nu\nd\nh\ne\ns\nw\n–\n3\nc\np\n-\n1\nj\nm\ni\n \nf\nl\no\n0\nb\nr\na\n4\n2\nn\n_\nx\nv\nt\nq\n5\n6\ng\n";
    const AMHARIC_UROMAN_VOCAB: &str =
        "c\n_\nl\nf\np\ne\nm\nj\nr\nh\no\nz\n \ns\n'\nt\nn\nu\nq\nb\nw\na\nk\nx\ni\ny\nd\ng\n";
    const THAI_VOCAB: &str =
        "า\nน\n่\nร\nเ\n้\nอ\nง\nก\nว\nะ\nั\nม\nท\nพ\nย\nล\nจ\nี\nค\nต\nด\nห\nข\nิ\nแ\nส\nบ\nป\nไ\nู\nใ\n็\nื\n์\nช\nุ\nึ\nํ\nโ\nผ\nถ\nญ\nซ\nธ\nศ\nณ\nษ\nฟ\nภ\nฉ\nฝ\nฐ\nฤ\nฏ\nฮ\nฆ\n๋\nฎ\n'\n0\n๊\nฑ\n1\n4\n2\n-\nฬ\nฒ\nฌ\n \n";

    fn english_config(training_files: &str) -> String {
        format!(
            r#"{{
              "train": {{"segment_size": 8192}},
              "data": {{
                "training_files": "{training_files}",
                "sampling_rate": 16000,
                "filter_length": 1024,
                "hop_length": 256,
                "win_length": 1024,
                "n_mel_channels": 80,
                "mel_fmin": 0.0,
                "mel_fmax": null,
                "add_blank": true,
                "n_speakers": 0
              }},
              "model": {{
                "inter_channels": 192,
                "hidden_channels": 192,
                "filter_channels": 768,
                "n_heads": 2,
                "n_layers": 6,
                "kernel_size": 3,
                "p_dropout": 0.1,
                "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [8, 8, 2, 2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [16, 16, 4, 4]
              }}
            }}"#
        )
    }

    #[test]
    fn parses_published_topology_and_preprocessing_requirement() {
        let config = FairseqVitsConfig::from_json_str(&english_config("train.ltr")).unwrap();
        let inference = config.inference_config(40).unwrap();
        assert_eq!(
            config.preprocessing(),
            FairseqPreprocessingRequirement::None
        );
        assert_eq!(inference.network.num_chars, 40);
        assert_eq!(inference.network.hidden_channels, 192);
        assert_eq!(inference.network.num_layers_flow, 4);
        assert_eq!(inference.audio.sample_rate, 16_000);
        assert_eq!(inference.audio.hop_length, 256);

        let uroman = FairseqVitsConfig::from_json_str(&english_config("train.uroman")).unwrap();
        assert_eq!(
            uroman.preprocessing(),
            FairseqPreprocessingRequirement::Uroman
        );
    }

    #[test]
    fn tokenization_matches_upstream_blank_and_oov_behavior() {
        let tokenizer = FairseqVitsTokenizer::from_vocab_str(
            "eng",
            ENGLISH_VOCAB,
            true,
            FairseqPreprocessingRequirement::None,
        )
        .unwrap();
        let tokenized = tokenizer.encode("This!").unwrap();

        assert_eq!(tokenized.normalized_text, "this!");
        assert_eq!(tokenized.filtered_symbols, vec!["!"]);
        // t=33, h=6, i=18, s=8 with blank row 0 interspersed.
        assert_eq!(tokenized.ids, vec![0, 33, 0, 6, 0, 18, 0, 8, 0]);
    }

    #[test]
    fn already_romanized_fixture_can_be_checked_without_external_uroman() {
        let tokenizer = FairseqVitsTokenizer::from_vocab_str(
            "amh",
            AMHARIC_UROMAN_VOCAB,
            true,
            FairseqPreprocessingRequirement::Uroman,
        )
        .unwrap();
        let tokenized = tokenizer.encode_preprocessed("selam").unwrap();
        assert_eq!(tokenized.ids, vec![0, 13, 0, 5, 0, 2, 0, 21, 0, 6, 0]);
    }

    #[test]
    fn native_script_fixture_matches_published_thai_vocabulary() {
        let tokenizer = FairseqVitsTokenizer::from_vocab_str(
            "tha",
            THAI_VOCAB,
            true,
            FairseqPreprocessingRequirement::None,
        )
        .unwrap();
        let tokenized = tokenizer.encode("สวัสดี").unwrap();
        assert_eq!(
            tokenized.ids,
            vec![0, 26, 0, 9, 0, 11, 0, 26, 0, 21, 0, 18, 0]
        );
    }

    #[test]
    fn uroman_requirement_fails_clearly_when_not_configured() {
        let tokenizer = FairseqVitsTokenizer::from_vocab_str(
            "amh",
            AMHARIC_UROMAN_VOCAB,
            true,
            FairseqPreprocessingRequirement::Uroman,
        )
        .unwrap();
        if std::env::var_os(UROMAN_ENVIRONMENT_VARIABLE).is_none() {
            let error = tokenizer.encode("ሰላም").unwrap_err().to_string();
            assert!(error.contains("requires Uroman preprocessing"));
            assert!(error.contains(UROMAN_ENVIRONMENT_VARIABLE));
        }
    }

    #[test]
    fn romanian_compatibility_substitution_matches_reference() {
        let tokenizer = FairseqVitsTokenizer::from_vocab_str(
            "ron",
            "\nţ\ne\ns\nt\n",
            false,
            FairseqPreprocessingRequirement::None,
        )
        .unwrap();
        let tokenized = tokenizer.encode("Țesț").unwrap();
        assert_eq!(tokenized.normalized_text, "ţesţ");
        assert_eq!(tokenized.ids, vec![1, 2, 3, 1]);
    }
}
