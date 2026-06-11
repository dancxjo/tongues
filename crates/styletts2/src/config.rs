use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::symbols::SymbolSet;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StyleTts2Config {
    pub sample_rate_hz: u32,
    pub symbol_set: SymbolSet,
    pub supports_reference_audio: bool,
    pub supports_speaker_embedding: bool,
    pub model_paths: StyleTts2ModelPaths,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct StyleTts2ModelPaths {
    pub acoustic: Option<PathBuf>,
    pub style_encoder: Option<PathBuf>,
    pub decoder: Option<PathBuf>,
    pub diffusion: Option<PathBuf>,
    pub vocoder: Option<PathBuf>,
    pub speaker_embeddings: Option<PathBuf>,
    pub config: Option<PathBuf>,
}

#[derive(Debug, Error)]
pub enum StyleTts2ConfigError {
    #[error("failed to parse StyleTTS2 config JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
    #[error("missing required StyleTTS2 config field `{field}`")]
    MissingField { field: &'static str },
    #[error("invalid StyleTTS2 config field `{field}`: {reason}")]
    InvalidField { field: &'static str, reason: String },
}

impl StyleTts2Config {
    pub fn from_json_str(json: &str) -> Result<Self, StyleTts2ConfigError> {
        let value: Value = serde_json::from_str(json)?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &Value) -> Result<Self, StyleTts2ConfigError> {
        let sample_rate_hz = parse_required_u32(
            value,
            &[
                &["audio", "sample_rate_hz"],
                &["audio", "sample_rate"],
                &["sample_rate_hz"],
                &["sample_rate"],
            ],
            "sample_rate_hz",
        )?;
        let symbol_set = SymbolSet::from_config_value(
            find_value(value, &[&["symbol_set"], &["symbols"], &["vocab"]]).ok_or(
                StyleTts2ConfigError::MissingField {
                    field: "symbol_set",
                },
            )?,
        )
        .map_err(|reason| StyleTts2ConfigError::InvalidField {
            field: "symbol_set",
            reason,
        })?;
        let model_paths = parse_model_paths(find_value(
            value,
            &[&["model_paths"], &["paths"], &["models"]],
        ));
        let supports_reference_audio = parse_optional_bool(
            value,
            &[
                &["supports_reference_audio"],
                &["capabilities", "reference_audio"],
                &["style", "reference_audio"],
            ],
            "supports_reference_audio",
        )?
        .unwrap_or(false);
        let supports_speaker_embedding = parse_optional_bool(
            value,
            &[
                &["supports_speaker_embedding"],
                &["capabilities", "speaker_embedding"],
                &["speaker", "embedding"],
            ],
            "supports_speaker_embedding",
        )?
        .unwrap_or_else(|| model_paths.speaker_embeddings.is_some());

        Ok(Self {
            sample_rate_hz,
            symbol_set,
            supports_reference_audio,
            supports_speaker_embedding,
            model_paths,
        })
    }
}

fn find_value<'a>(root: &'a Value, paths: &[&[&str]]) -> Option<&'a Value> {
    paths.iter().find_map(|path| {
        let mut current = root;
        for segment in *path {
            current = current.get(*segment)?;
        }
        Some(current)
    })
}

fn parse_required_u32(
    root: &Value,
    paths: &[&[&str]],
    field: &'static str,
) -> Result<u32, StyleTts2ConfigError> {
    let value = find_value(root, paths).ok_or(StyleTts2ConfigError::MissingField { field })?;
    let number = value
        .as_u64()
        .ok_or_else(|| invalid_field(field, "expected an unsigned integer"))?;
    u32::try_from(number).map_err(|_| invalid_field(field, "value exceeds u32 range"))
}

fn parse_optional_bool(
    root: &Value,
    paths: &[&[&str]],
    field: &'static str,
) -> Result<Option<bool>, StyleTts2ConfigError> {
    find_value(root, paths)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| invalid_field(field, "expected a boolean"))
        })
        .transpose()
}

fn parse_model_paths(value: Option<&Value>) -> StyleTts2ModelPaths {
    let Some(Value::Object(paths)) = value else {
        return StyleTts2ModelPaths::default();
    };

    StyleTts2ModelPaths {
        acoustic: path_field(paths, &["acoustic", "model", "onnx"]),
        style_encoder: path_field(paths, &["style_encoder", "style"]),
        decoder: path_field(paths, &["decoder"]),
        diffusion: path_field(paths, &["diffusion"]),
        vocoder: path_field(paths, &["vocoder"]),
        speaker_embeddings: path_field(paths, &["speaker_embeddings", "speaker_embedding"]),
        config: path_field(paths, &["config"]),
    }
}

fn path_field(paths: &serde_json::Map<String, Value>, names: &[&str]) -> Option<PathBuf> {
    names
        .iter()
        .find_map(|name| paths.get(*name).and_then(Value::as_str).map(PathBuf::from))
}

fn invalid_field(field: &'static str, reason: impl Into<String>) -> StyleTts2ConfigError {
    StyleTts2ConfigError::InvalidField {
        field,
        reason: reason.into(),
    }
}
