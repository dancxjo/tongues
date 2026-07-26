//! Native XTTS v2 configuration, tokenizer, and streaming contracts.
//!
//! The checkpoint-compatible configuration and streaming overlap semantics
//! follow Coqui TTS revision `dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e`
//! (`TTS/tts/models/xtts.py` and `TTS/tts/configs/xtts_config.py`, MPL-2.0).
//! This file is an MPL-2.0 covered modification. Model weights remain subject
//! to their separately recorded artifact license.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokenizers::Tokenizer;

pub const XTTS_V2_CONDITIONING_MEL_BINS: usize = 80;
pub const XTTS_V2_STREAM_OVERLAP_SAMPLES: usize = 1_024;
pub const XTTS_V2_DEFAULT_STREAM_CODE_CHUNK: usize = 20;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XttsV2Config {
    pub model: String,
    pub audio: XttsAudioConfig,
    pub model_args: XttsModelArgs,
    pub languages: Vec<String>,
    pub temperature: f32,
    pub length_penalty: f32,
    pub repetition_penalty: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub gpt_cond_len: usize,
    pub gpt_cond_chunk_len: usize,
    pub max_ref_len: usize,
    pub sound_norm_refs: bool,
    /// Canonical package-local tokenizer path. The source config's empty or
    /// machine-local `tokenizer_file` is never retained.
    pub tokenizer: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct XttsAudioConfig {
    pub input_sample_rate: u32,
    pub output_sample_rate: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XttsModelArgs {
    pub kv_cache: bool,
    pub gpt_max_audio_tokens: usize,
    pub gpt_max_text_tokens: usize,
    pub gpt_max_prompt_tokens: usize,
    pub gpt_layers: usize,
    pub gpt_n_model_channels: usize,
    pub gpt_n_heads: usize,
    pub gpt_number_text_tokens: usize,
    pub gpt_start_text_token: Option<u32>,
    pub gpt_stop_text_token: Option<u32>,
    pub gpt_num_audio_tokens: usize,
    pub gpt_start_audio_token: u32,
    pub gpt_stop_audio_token: u32,
    pub gpt_code_stride_len: usize,
    pub gpt_use_masking_gt_prompt_approach: bool,
    pub gpt_use_perceiver_resampler: bool,
    pub output_hop_length: usize,
    pub decoder_input_dim: usize,
    pub d_vector_dim: usize,
    pub cond_d_vector_in_each_upsampling_layer: bool,
    pub duration_const: usize,
}

impl XttsV2Config {
    pub fn from_file(
        config_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let config_path = config_path.as_ref();
        let source = std::fs::read_to_string(config_path)
            .with_context(|| format!("failed to read XTTS config {}", config_path.display()))?;
        let root: Value = json5::from_str(&source)
            .with_context(|| format!("invalid XTTS JSON/JSON5 config {}", config_path.display()))?;
        Self::from_json_value(&root, tokenizer_path)
    }

    pub fn from_json_value(root: &Value, tokenizer_path: impl AsRef<Path>) -> Result<Self> {
        let object = root
            .as_object()
            .context("XTTS config root must be an object")?;
        ensure!(
            object
                .get("model")
                .and_then(Value::as_str)
                .is_some_and(|model| model.eq_ignore_ascii_case("xtts")),
            "XTTS config requires model `xtts`"
        );
        let args = object
            .get("model_args")
            .and_then(Value::as_object)
            .context("XTTS config requires model_args")?;
        reject_unknown_model_args(args)?;

        let input_sample_rate = u32_field(args, "input_sample_rate")?;
        let output_sample_rate = u32_field(args, "output_sample_rate")?;
        let audio = object
            .get("audio")
            .and_then(Value::as_object)
            .context("XTTS config requires audio")?;
        ensure!(
            audio.get("sample_rate").and_then(Value::as_u64) == Some(u64::from(input_sample_rate)),
            "XTTS audio.sample_rate must match model_args.input_sample_rate"
        );
        ensure!(
            audio.get("output_sample_rate").and_then(Value::as_u64)
                == Some(u64::from(output_sample_rate)),
            "XTTS audio.output_sample_rate must match model_args.output_sample_rate"
        );

        let tokenizer = XttsTokenizer::from_file(tokenizer_path)?;
        let languages = string_array(object, "languages")?;
        ensure!(!languages.is_empty(), "XTTS languages must not be empty");
        let unique = languages.iter().collect::<BTreeSet<_>>();
        ensure!(
            unique.len() == languages.len(),
            "XTTS languages contain duplicates"
        );
        for language in &languages {
            tokenizer.require_language(language)?;
        }

        let gpt_number_text_tokens = usize_field(args, "gpt_number_text_tokens")?;
        ensure!(
            tokenizer.vocab_size() == gpt_number_text_tokens,
            "XTTS tokenizer has {} token IDs but config declares gpt_number_text_tokens={gpt_number_text_tokens}",
            tokenizer.vocab_size()
        );
        let configured_start = optional_u32_field(args, "gpt_start_text_token")?;
        let configured_stop = optional_u32_field(args, "gpt_stop_text_token")?;
        if let Some(id) = configured_start {
            ensure!(
                id == tokenizer.start_token_id(),
                "XTTS gpt_start_text_token {id} does not match tokenizer [START] id {}",
                tokenizer.start_token_id()
            );
        }
        if let Some(id) = configured_stop {
            ensure!(
                id == tokenizer.stop_token_id(),
                "XTTS gpt_stop_text_token {id} does not match tokenizer [STOP] id {}",
                tokenizer.stop_token_id()
            );
        }

        let gpt_cond_len = usize_root_field(object, "gpt_cond_len")?;
        let gpt_cond_chunk_len = usize_root_field(object, "gpt_cond_chunk_len")?;
        ensure!(
            gpt_cond_chunk_len > 0 && gpt_cond_chunk_len <= gpt_cond_len,
            "XTTS gpt_cond_chunk_len must be positive and no greater than gpt_cond_len"
        );
        let top_p = f32_root_field(object, "top_p")?;
        let temperature = f32_root_field(object, "temperature")?;
        let repetition_penalty = f32_root_field(object, "repetition_penalty")?;
        ensure!(
            temperature.is_finite() && temperature > 0.0,
            "XTTS temperature must be finite and positive"
        );
        ensure!(
            top_p.is_finite() && top_p > 0.0 && top_p <= 1.0,
            "XTTS top_p must be in (0, 1]"
        );
        ensure!(
            repetition_penalty.is_finite() && repetition_penalty > 0.0,
            "XTTS repetition_penalty must be finite and positive"
        );

        let model_args = XttsModelArgs {
            kv_cache: bool_field(args, "kv_cache")?,
            gpt_max_audio_tokens: usize_field(args, "gpt_max_audio_tokens")?,
            gpt_max_text_tokens: usize_field(args, "gpt_max_text_tokens")?,
            gpt_max_prompt_tokens: usize_field(args, "gpt_max_prompt_tokens")?,
            gpt_layers: usize_field(args, "gpt_layers")?,
            gpt_n_model_channels: usize_field(args, "gpt_n_model_channels")?,
            gpt_n_heads: usize_field(args, "gpt_n_heads")?,
            gpt_number_text_tokens,
            // Published XTTS v2 configs leave these null and resolve them
            // from vocab.json at load time. Persist the resolved IDs in the
            // neutral package so runtime never needs to guess them.
            gpt_start_text_token: Some(configured_start.unwrap_or(tokenizer.start_token_id())),
            gpt_stop_text_token: Some(configured_stop.unwrap_or(tokenizer.stop_token_id())),
            gpt_num_audio_tokens: usize_field(args, "gpt_num_audio_tokens")?,
            gpt_start_audio_token: u32_field(args, "gpt_start_audio_token")?,
            gpt_stop_audio_token: u32_field(args, "gpt_stop_audio_token")?,
            gpt_code_stride_len: usize_field(args, "gpt_code_stride_len")?,
            gpt_use_masking_gt_prompt_approach: bool_field(
                args,
                "gpt_use_masking_gt_prompt_approach",
            )?,
            gpt_use_perceiver_resampler: bool_field(args, "gpt_use_perceiver_resampler")?,
            output_hop_length: usize_field(args, "output_hop_length")?,
            decoder_input_dim: usize_field(args, "decoder_input_dim")?,
            d_vector_dim: usize_field(args, "d_vector_dim")?,
            cond_d_vector_in_each_upsampling_layer: bool_field(
                args,
                "cond_d_vector_in_each_upsampling_layer",
            )?,
            duration_const: usize_field(args, "duration_const")?,
        };
        ensure!(
            model_args.gpt_layers > 0
                && model_args.gpt_n_heads > 0
                && model_args.gpt_n_model_channels % model_args.gpt_n_heads == 0,
            "XTTS GPT channels must be divisible by its positive head count"
        );
        ensure!(
            model_args.gpt_use_perceiver_resampler,
            "XTTS v2 import requires gpt_use_perceiver_resampler=true"
        );
        ensure!(
            usize::try_from(model_args.gpt_start_audio_token)? < model_args.gpt_num_audio_tokens
                && usize::try_from(model_args.gpt_stop_audio_token)?
                    < model_args.gpt_num_audio_tokens
                && model_args.gpt_start_audio_token != model_args.gpt_stop_audio_token,
            "XTTS audio start/stop tokens must be distinct members of the audio vocabulary"
        );

        Ok(Self {
            model: "xtts".into(),
            audio: XttsAudioConfig {
                input_sample_rate,
                output_sample_rate,
            },
            model_args,
            languages,
            temperature,
            length_penalty: f32_root_field(object, "length_penalty")?,
            repetition_penalty,
            top_k: usize_root_field(object, "top_k")?,
            top_p,
            gpt_cond_len,
            gpt_cond_chunk_len,
            max_ref_len: usize_root_field(object, "max_ref_len")?,
            sound_norm_refs: object
                .get("sound_norm_refs")
                .and_then(Value::as_bool)
                .context("XTTS config requires boolean sound_norm_refs")?,
            tokenizer: "vocab.json".into(),
        })
    }
}

#[derive(Clone)]
pub struct XttsTokenizer {
    path: PathBuf,
    tokenizer: Tokenizer,
    vocab_size: usize,
    start_token_id: u32,
    stop_token_id: u32,
}

impl std::fmt::Debug for XttsTokenizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("XttsTokenizer")
            .field("path", &self.path)
            .field("vocab_size", &self.vocab_size)
            .field("start_token_id", &self.start_token_id)
            .field("stop_token_id", &self.stop_token_id)
            .finish()
    }
}

impl XttsTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|error| anyhow::anyhow!("{error}"))
            .with_context(|| format!("failed to load XTTS tokenizer {}", path.display()))?;
        let vocabulary = tokenizer.get_vocab(true);
        let vocab_size = vocabulary
            .values()
            .copied()
            .max()
            .and_then(|id| usize::try_from(id).ok())
            .and_then(|id| id.checked_add(1))
            .context("XTTS tokenizer vocabulary is empty or too large")?;
        let start_token_id = tokenizer
            .token_to_id("[START]")
            .context("XTTS tokenizer has no [START] token")?;
        let stop_token_id = tokenizer
            .token_to_id("[STOP]")
            .context("XTTS tokenizer has no [STOP] token")?;
        ensure!(
            tokenizer.token_to_id("[SPACE]").is_some(),
            "XTTS tokenizer has no [SPACE] token"
        );
        ensure!(
            tokenizer.token_to_id("[UNK]").is_some(),
            "XTTS tokenizer has no [UNK] token"
        );
        Ok(Self {
            path: path.to_path_buf(),
            tokenizer,
            vocab_size,
            start_token_id,
            stop_token_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn start_token_id(&self) -> u32 {
        self.start_token_id
    }

    pub fn stop_token_id(&self) -> u32 {
        self.stop_token_id
    }

    pub fn require_language(&self, language: &str) -> Result<()> {
        let tag = canonical_tokenizer_language(language);
        ensure!(
            self.tokenizer.token_to_id(&format!("[{tag}]")).is_some(),
            "XTTS tokenizer has no [{tag}] language token"
        );
        Ok(())
    }

    /// Encode text that has already passed through the language-specific XTTS
    /// cleaner. Keeping this boundary explicit prevents a partial cleaner from
    /// silently changing numbers or transliteration before inference.
    pub fn encode_preprocessed(&self, text: &str, language: &str) -> Result<Vec<u32>> {
        ensure!(
            !text.trim().is_empty(),
            "XTTS text must not be empty after preprocessing"
        );
        self.require_language(language)?;
        let language = canonical_tokenizer_language(language);
        let tagged = format!("[{language}]{}", text.replace(' ', "[SPACE]"));
        let encoded = self
            .tokenizer
            .encode(tagged, false)
            .map_err(|error| anyhow::anyhow!("{error}"))
            .context("XTTS BPE tokenization failed")?;
        ensure!(!encoded.is_empty(), "XTTS tokenizer produced no tokens");
        Ok(encoded.get_ids().to_vec())
    }
}

fn canonical_tokenizer_language(language: &str) -> &str {
    if language.eq_ignore_ascii_case("zh") || language.eq_ignore_ascii_case("zh-cn") {
        "zh-cn"
    } else {
        language.split('-').next().unwrap_or(language)
    }
}

/// Incrementally turns cumulative XTTS decoder waveforms into ordered chunks.
///
/// Each decoder call recomputes the waveform for all generated GPT latents.
/// This state emits only the newly stable suffix, crossfades the recomputed
/// overlap, and flushes the withheld tail on finalization.
#[derive(Debug, Clone)]
pub struct XttsStreamAssembler {
    overlap_samples: usize,
    previous_waveform_len: usize,
    pending_overlap: Vec<f32>,
    finalized: bool,
}

impl XttsStreamAssembler {
    pub fn new(overlap_samples: usize) -> Result<Self> {
        ensure!(overlap_samples > 0, "XTTS overlap must be positive");
        Ok(Self {
            overlap_samples,
            previous_waveform_len: 0,
            pending_overlap: Vec::new(),
            finalized: false,
        })
    }

    pub fn push_cumulative(&mut self, waveform: &[f32], is_final: bool) -> Result<Vec<f32>> {
        ensure!(!self.finalized, "XTTS stream is already finalized");
        ensure!(
            waveform.len() >= self.previous_waveform_len,
            "XTTS cumulative waveform shrank from {} to {} samples",
            self.previous_waveform_len,
            waveform.len()
        );
        ensure!(
            waveform.iter().all(|sample| sample.is_finite()),
            "XTTS decoder produced non-finite PCM"
        );

        let start = self
            .previous_waveform_len
            .saturating_sub(self.overlap_samples);
        let stable_end = if is_final {
            waveform.len()
        } else {
            waveform.len().saturating_sub(self.overlap_samples)
        };
        let mut chunk = if stable_end > start {
            waveform[start..stable_end].to_vec()
        } else {
            Vec::new()
        };
        crossfade_prefix(&mut chunk, &self.pending_overlap);

        self.pending_overlap = if is_final {
            Vec::new()
        } else {
            waveform[stable_end..].to_vec()
        };
        self.previous_waveform_len = waveform.len();
        self.finalized = is_final;
        Ok(chunk)
    }

    pub fn is_finalized(&self) -> bool {
        self.finalized
    }
}

fn crossfade_prefix(chunk: &mut [f32], previous: &[f32]) {
    let samples = chunk.len().min(previous.len());
    if samples == 0 {
        return;
    }
    if samples == 1 {
        chunk[0] = (chunk[0] + previous[0]) * 0.5;
        return;
    }
    let denominator = (samples - 1) as f32;
    for index in 0..samples {
        let incoming = index as f32 / denominator;
        chunk[index] = previous[index] * (1.0 - incoming) + chunk[index] * incoming;
    }
}

fn reject_unknown_model_args(args: &serde_json::Map<String, Value>) -> Result<()> {
    let allowed = [
        "gpt_batch_size",
        "enable_redaction",
        "kv_cache",
        "gpt_checkpoint",
        "clvp_checkpoint",
        "decoder_checkpoint",
        "num_chars",
        "tokenizer_file",
        "gpt_max_audio_tokens",
        "gpt_max_text_tokens",
        "gpt_max_prompt_tokens",
        "gpt_layers",
        "gpt_n_model_channels",
        "gpt_n_heads",
        "gpt_number_text_tokens",
        "gpt_start_text_token",
        "gpt_stop_text_token",
        "gpt_num_audio_tokens",
        "gpt_start_audio_token",
        "gpt_stop_audio_token",
        "gpt_code_stride_len",
        "gpt_use_masking_gt_prompt_approach",
        "gpt_use_perceiver_resampler",
        "input_sample_rate",
        "output_sample_rate",
        "output_hop_length",
        "decoder_input_dim",
        "d_vector_dim",
        "cond_d_vector_in_each_upsampling_layer",
        "duration_const",
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    let unknown = args
        .keys()
        .filter(|key| !allowed.contains(key.as_str()))
        .map(|key| format!("model_args.{key}"))
        .collect::<Vec<_>>();
    ensure!(
        unknown.is_empty(),
        "unsupported XTTS config field(s): {}",
        unknown.join(", ")
    );
    Ok(())
}

fn usize_field(object: &serde_json::Map<String, Value>, name: &str) -> Result<usize> {
    object
        .get(name)
        .and_then(Value::as_u64)
        .with_context(|| format!("XTTS model_args.{name} must be an unsigned integer"))?
        .try_into()
        .with_context(|| format!("XTTS model_args.{name} does not fit usize"))
}

fn u32_field(object: &serde_json::Map<String, Value>, name: &str) -> Result<u32> {
    object
        .get(name)
        .and_then(Value::as_u64)
        .with_context(|| format!("XTTS model_args.{name} must be an unsigned integer"))?
        .try_into()
        .with_context(|| format!("XTTS model_args.{name} does not fit u32"))
}

fn optional_u32_field(object: &serde_json::Map<String, Value>, name: &str) -> Result<Option<u32>> {
    object
        .get(name)
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_u64()
                .with_context(|| format!("XTTS model_args.{name} must be an unsigned integer"))?
                .try_into()
                .with_context(|| format!("XTTS model_args.{name} does not fit u32"))
        })
        .transpose()
}

fn bool_field(object: &serde_json::Map<String, Value>, name: &str) -> Result<bool> {
    object
        .get(name)
        .and_then(Value::as_bool)
        .with_context(|| format!("XTTS model_args.{name} must be boolean"))
}

fn usize_root_field(object: &serde_json::Map<String, Value>, name: &str) -> Result<usize> {
    object
        .get(name)
        .and_then(Value::as_u64)
        .with_context(|| format!("XTTS {name} must be an unsigned integer"))?
        .try_into()
        .with_context(|| format!("XTTS {name} does not fit usize"))
}

fn f32_root_field(object: &serde_json::Map<String, Value>, name: &str) -> Result<f32> {
    let value = object
        .get(name)
        .and_then(Value::as_f64)
        .with_context(|| format!("XTTS {name} must be numeric"))?;
    ensure!(
        value.is_finite() && value >= f64::from(f32::MIN) && value <= f64::from(f32::MAX),
        "XTTS {name} does not fit finite f32"
    );
    Ok(value as f32)
}

fn string_array(object: &serde_json::Map<String, Value>, name: &str) -> Result<Vec<String>> {
    object
        .get(name)
        .and_then(Value::as_array)
        .with_context(|| format!("XTTS {name} must be an array"))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .map(str::to_string)
                .with_context(|| format!("XTTS {name} entries must be non-empty strings"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::bpe::BPE;

    #[test]
    fn cumulative_streaming_matches_one_shot_waveform() {
        let waveform = (0..24).map(|value| value as f32 / 24.0).collect::<Vec<_>>();
        let mut stream = XttsStreamAssembler::new(4).unwrap();
        let mut actual = Vec::new();
        actual.extend(stream.push_cumulative(&waveform[..8], false).unwrap());
        actual.extend(stream.push_cumulative(&waveform[..16], false).unwrap());
        actual.extend(stream.push_cumulative(&waveform, true).unwrap());
        assert_eq!(actual, waveform);
        assert!(stream.is_finalized());
    }

    #[test]
    fn cumulative_streaming_crossfades_recomputed_overlap() {
        let mut stream = XttsStreamAssembler::new(4).unwrap();
        assert_eq!(
            stream
                .push_cumulative(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], false)
                .unwrap(),
            [0.0, 1.0]
        );
        let chunk = stream
            .push_cumulative(&[0.0, 1.0, 20.0, 30.0, 40.0, 50.0, 6.0, 7.0], true)
            .unwrap();
        assert_eq!(chunk[0], 2.0);
        assert_eq!(chunk[3], 50.0);
        assert_eq!(&chunk[4..], &[6.0, 7.0]);
    }

    #[test]
    fn xtts_v2_config_is_bound_to_tokenizer_and_languages() {
        let path = std::env::temp_dir().join(format!(
            "tongues-xtts-tokenizer-{}.json",
            std::process::id()
        ));
        let vocabulary = [
            ("[UNK]".to_string(), 0),
            ("[START]".to_string(), 1),
            ("[STOP]".to_string(), 2),
            ("[SPACE]".to_string(), 3),
            ("[en]".to_string(), 4),
            ("[fr]".to_string(), 5),
        ];
        let model = BPE::builder()
            .vocab_and_merges(vocabulary, Vec::new())
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        Tokenizer::new(model).save(&path, false).unwrap();
        let config = serde_json::json!({
            "model": "xtts",
            "audio": {"sample_rate": 22050, "output_sample_rate": 24000},
            "model_args": {
                "gpt_batch_size": 1,
                "enable_redaction": false,
                "kv_cache": true,
                "gpt_checkpoint": null,
                "clvp_checkpoint": null,
                "decoder_checkpoint": null,
                "num_chars": 255,
                "tokenizer_file": "",
                "gpt_max_audio_tokens": 605,
                "gpt_max_text_tokens": 402,
                "gpt_max_prompt_tokens": 70,
                "gpt_layers": 30,
                "gpt_n_model_channels": 1024,
                "gpt_n_heads": 16,
                "gpt_number_text_tokens": 6,
                "gpt_start_text_token": null,
                "gpt_stop_text_token": null,
                "gpt_num_audio_tokens": 1026,
                "gpt_start_audio_token": 1024,
                "gpt_stop_audio_token": 1025,
                "gpt_code_stride_len": 1024,
                "gpt_use_masking_gt_prompt_approach": true,
                "gpt_use_perceiver_resampler": true,
                "input_sample_rate": 22050,
                "output_sample_rate": 24000,
                "output_hop_length": 256,
                "decoder_input_dim": 1024,
                "d_vector_dim": 512,
                "cond_d_vector_in_each_upsampling_layer": true,
                "duration_const": 102400
            },
            "languages": ["en", "fr"],
            "temperature": 0.75,
            "length_penalty": 1.0,
            "repetition_penalty": 5.0,
            "top_k": 50,
            "top_p": 0.85,
            "gpt_cond_len": 30,
            "gpt_cond_chunk_len": 4,
            "max_ref_len": 30,
            "sound_norm_refs": false
        });
        let parsed = XttsV2Config::from_json_value(&config, &path).unwrap();
        assert_eq!(parsed.languages, ["en", "fr"]);
        assert_eq!(parsed.audio.output_sample_rate, 24_000);
        assert_eq!(parsed.model_args.d_vector_dim, 512);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn published_xtts_v2_config_and_vocab_are_compatible_when_available() {
        let (Some(config), Some(vocab)) = (
            std::env::var_os("TONGUES_TEST_XTTS_CONFIG"),
            std::env::var_os("TONGUES_TEST_XTTS_VOCAB"),
        ) else {
            return;
        };
        let config = XttsV2Config::from_file(config, vocab).expect("published XTTS v2 contract");
        assert_eq!(config.audio.input_sample_rate, 22_050);
        assert_eq!(config.audio.output_sample_rate, 24_000);
        assert_eq!(config.model_args.gpt_layers, 30);
        assert_eq!(config.model_args.gpt_n_model_channels, 1_024);
        assert_eq!(config.model_args.d_vector_dim, 512);
        assert!(config.languages.iter().any(|language| language == "en"));
        assert!(config.languages.iter().any(|language| language == "fr"));
    }
}
