//! Backend-neutral configuration for the legacy Tacotron family.
//!
//! Coqui shipped several generations of configuration files.  Older released
//! checkpoints put inference fields at the root and use mixed-case model
//! names, while newer Coqpit files use lower-case names and may materialize
//! defaults that were absent from the old JSON5.  This parser deliberately
//! accepts both layouts, but rejects neural variants that the native runtime
//! cannot reproduce.
//!
//! Source provenance: this MPL-2.0 covered configuration adaptation targets
//! Coqui TTS v0.6.1 revision
//! `0cf3265a4686d7e856bd472cdaf1572d61cab2b8`, especially
//! `TTS/tts/configs/tacotron_config.py` and
//! `TTS/tts/configs/tacotron2_config.py`. See `THIRD_PARTY_NOTICES.md`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const DEFAULT_TACOTRON_MAX_DECODER_STEPS: usize = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TacotronArchitecture {
    Tacotron,
    Tacotron2,
}

impl TacotronArchitecture {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tacotron => "tacotron",
            Self::Tacotron2 => "tacotron2",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TacotronVariant {
    Plain,
    DoubleDecoderConsistency,
    Capacitron,
    CapacitronDoubleDecoderConsistency,
}

impl TacotronVariant {
    pub fn uses_ddc(self) -> bool {
        matches!(
            self,
            Self::DoubleDecoderConsistency | Self::CapacitronDoubleDecoderConsistency
        )
    }

    pub fn uses_capacitron(self) -> bool {
        matches!(
            self,
            Self::Capacitron | Self::CapacitronDoubleDecoderConsistency
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TacotronAttentionNormalization {
    Softmax,
    Sigmoid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapacitronInferenceConfig {
    pub embedding_dim: usize,
    pub use_text_summary_embedding: bool,
    pub text_summary_embedding_dim: usize,
    pub use_speaker_embedding: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TacotronInferenceConfig {
    pub architecture: TacotronArchitecture,
    pub variant: TacotronVariant,
    pub num_chars: usize,
    pub out_channels: usize,
    pub encoder_channels: usize,
    pub decoder_channels: usize,
    pub reduction_factor: usize,
    pub ddc_reduction_factor: Option<usize>,
    pub max_decoder_steps: usize,
    pub stop_threshold: f32,
    pub location_attention: bool,
    pub attention_normalization: TacotronAttentionNormalization,
    pub attention_windowing: bool,
    pub forward_attention: bool,
    pub forward_attention_mask: bool,
    pub transition_agent: bool,
    pub prenet_dropout_at_inference: bool,
    pub separate_stopnet: bool,
    pub capacitron: Option<CapacitronInferenceConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TacotronConfigError {
    #[error("invalid Tacotron config: {0}")]
    Invalid(String),
    #[error("unsupported historical Tacotron variant: {0}")]
    Unsupported(String),
}

impl TacotronInferenceConfig {
    pub fn from_json_value(root: &Value) -> Result<Self, TacotronConfigError> {
        let object = root
            .as_object()
            .ok_or_else(|| invalid("config root must be an object"))?;
        let args = object
            .get("model_args")
            .and_then(Value::as_object)
            .unwrap_or(object);
        let model = string_in(args, object, "model")
            .ok_or_else(|| invalid("missing model"))?
            .to_ascii_lowercase()
            .replace(['-', '_'], "");
        let architecture = match model.as_str() {
            "tacotron" => TacotronArchitecture::Tacotron,
            "tacotron2" => TacotronArchitecture::Tacotron2,
            other => {
                return Err(invalid(format!(
                    "expected `tacotron` or `tacotron2`, got `{other}`"
                )));
            }
        };

        reject_enabled(args, object, "use_gst", "GST conditioning")?;
        reject_enabled(
            args,
            object,
            "bidirectional_decoder",
            "bidirectional training decoder",
        )?;
        let attention_type = string_in(args, object, "attention_type").unwrap_or("original");
        if !attention_type.eq_ignore_ascii_case("original") {
            return Err(unsupported(format!(
                "attention_type={attention_type:?}; native import currently requires original location-sensitive attention"
            )));
        }
        let prenet_type = string_in(args, object, "prenet_type").unwrap_or("original");
        if !prenet_type.eq_ignore_ascii_case("original") {
            return Err(unsupported(format!(
                "prenet_type={prenet_type:?}; batch-normalized Tacotron prenets have a different checkpoint topology"
            )));
        }
        if bool_in(args, object, "use_d_vector_file").unwrap_or(false) {
            return Err(unsupported(
                "external d-vector speaker conditioning is not yet available for Tacotron",
            ));
        }
        if bool_in(args, object, "use_speaker_embedding").unwrap_or(false)
            || usize_in(args, object, "num_speakers").unwrap_or(1) > 1
        {
            return Err(unsupported(
                "learned multi-speaker conditioning is not yet available for Tacotron",
            ));
        }

        let num_chars = usize_in(args, object, "num_chars")
            .filter(|count| *count > 0)
            .or_else(|| vocabulary_size(root))
            .ok_or_else(|| {
                invalid("num_chars is absent and characters do not define a vocabulary")
            })?;
        let out_channels = usize_in(args, object, "out_channels")
            .or_else(|| {
                root.get("audio")
                    .and_then(|audio| audio.get("num_mels"))
                    .and_then(Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
            })
            .ok_or_else(|| invalid("missing out_channels/audio.num_mels"))?;
        let encoder_channels =
            usize_in(args, object, "encoder_in_features").unwrap_or(match architecture {
                TacotronArchitecture::Tacotron => 256,
                TacotronArchitecture::Tacotron2 => 512,
            });
        let decoder_channels =
            usize_in(args, object, "decoder_in_features").unwrap_or(encoder_channels);
        let reduction_factor = usize_in(args, object, "r").unwrap_or(2);
        let max_decoder_steps = usize_in(args, object, "max_decoder_steps")
            .unwrap_or(DEFAULT_TACOTRON_MAX_DECODER_STEPS);
        let ddc = bool_in(args, object, "double_decoder_consistency").unwrap_or(false)
            || metadata_mentions_ddc(root);
        let use_capacitron = bool_in(args, object, "use_capacitron_vae").unwrap_or(false);
        let capacitron = if use_capacitron {
            Some(parse_capacitron(root)?)
        } else {
            None
        };
        let variant = match (ddc, use_capacitron) {
            (false, false) => TacotronVariant::Plain,
            (true, false) => TacotronVariant::DoubleDecoderConsistency,
            (false, true) => TacotronVariant::Capacitron,
            (true, true) => TacotronVariant::CapacitronDoubleDecoderConsistency,
        };
        let attention_normalization =
            match string_in(args, object, "attention_norm").unwrap_or("sigmoid") {
                value if value.eq_ignore_ascii_case("softmax") => {
                    TacotronAttentionNormalization::Softmax
                }
                value if value.eq_ignore_ascii_case("sigmoid") => {
                    TacotronAttentionNormalization::Sigmoid
                }
                value => {
                    return Err(unsupported(format!(
                        "attention_norm={value:?}; expected softmax or sigmoid"
                    )));
                }
            };
        let config = Self {
            architecture,
            variant,
            num_chars,
            out_channels,
            encoder_channels,
            decoder_channels,
            reduction_factor,
            ddc_reduction_factor: ddc.then(|| usize_in(args, object, "ddc_r").unwrap_or(6)),
            max_decoder_steps,
            stop_threshold: number_in(args, object, "stop_threshold").unwrap_or(0.5) as f32,
            location_attention: bool_in(args, object, "location_attn").unwrap_or(true),
            attention_normalization,
            attention_windowing: bool_in(args, object, "attention_win")
                .or_else(|| bool_in(args, object, "windowing"))
                .unwrap_or(false),
            forward_attention: bool_in(args, object, "use_forward_attn").unwrap_or(false),
            forward_attention_mask: bool_in(args, object, "forward_attn_mask").unwrap_or(false),
            transition_agent: bool_in(args, object, "transition_agent").unwrap_or(false),
            prenet_dropout_at_inference: bool_in(args, object, "prenet_dropout_at_inference")
                .unwrap_or(false),
            separate_stopnet: bool_in(args, object, "separate_stopnet").unwrap_or(true),
            capacitron,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), TacotronConfigError> {
        if self.num_chars == 0
            || self.out_channels == 0
            || self.encoder_channels == 0
            || self.decoder_channels == 0
        {
            return Err(invalid("model dimensions must be positive"));
        }
        if self.reduction_factor == 0 || self.max_decoder_steps == 0 {
            return Err(invalid(
                "reduction factor and maximum decoder steps must be positive",
            ));
        }
        if !self.stop_threshold.is_finite() || !(0.0..=1.0).contains(&self.stop_threshold) {
            return Err(invalid("stop threshold must be finite and in [0, 1]"));
        }
        if self.variant.uses_ddc() && self.ddc_reduction_factor.is_none_or(|factor| factor == 0) {
            return Err(invalid("DDC requires a positive coarse reduction factor"));
        }
        if self.variant.uses_capacitron() != self.capacitron.is_some() {
            return Err(invalid(
                "Capacitron variant and conditioning config disagree",
            ));
        }
        if let Some(capacitron) = &self.capacitron {
            if capacitron.embedding_dim == 0 {
                return Err(invalid("Capacitron embedding dimension must be positive"));
            }
            if capacitron.use_speaker_embedding {
                return Err(unsupported(
                    "Capacitron speaker-conditioned posterior artifacts are not yet supported",
                ));
            }
        }
        Ok(())
    }
}

fn parse_capacitron(root: &Value) -> Result<CapacitronInferenceConfig, TacotronConfigError> {
    let value = root
        .get("capacitron_vae")
        .or_else(|| {
            root.get("model_args")
                .and_then(|args| args.get("capacitron_vae"))
        })
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("use_capacitron_vae=true requires capacitron_vae"))?;
    let embedding_dim = usize_alias(
        value,
        &[
            "capacitron_VAE_embedding_dim",
            "capacitron_vae_embedding_dim",
        ],
    )
    .ok_or_else(|| invalid("Capacitron config is missing VAE embedding dimension"))?;
    Ok(CapacitronInferenceConfig {
        embedding_dim,
        use_text_summary_embedding: bool_alias(
            value,
            &[
                "capacitron_use_text_summary_embeddings",
                "use_text_summary_embeddings",
            ],
        )
        .unwrap_or(true),
        text_summary_embedding_dim: usize_alias(
            value,
            &[
                "capacitron_text_summary_embedding_dim",
                "text_summary_embedding_dim",
            ],
        )
        .unwrap_or(128),
        use_speaker_embedding: bool_alias(
            value,
            &["capacitron_use_speaker_embedding", "use_speaker_embedding"],
        )
        .unwrap_or(false),
    })
}

fn metadata_mentions_ddc(root: &Value) -> bool {
    ["run_name", "run_description"]
        .into_iter()
        .filter_map(|field| root.get(field).and_then(Value::as_str))
        .map(str::to_ascii_lowercase)
        .any(|value| value.contains("ddc") || value.contains("double decoder"))
}

fn vocabulary_size(root: &Value) -> Option<usize> {
    let characters = root.get("characters")?.as_object()?;
    let use_phonemes = root
        .get("use_phonemes")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let primary = if use_phonemes {
        characters
            .get("phonemes")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .or_else(|| characters.get("characters").and_then(Value::as_str))
    } else {
        characters.get("characters").and_then(Value::as_str)
    }?;
    let mut count = primary.chars().count();
    count += characters
        .get("punctuations")
        .and_then(Value::as_str)
        .map(str::chars)
        .map(Iterator::count)
        .unwrap_or(0);
    for special in ["blank", "bos", "eos", "pad"] {
        count += characters
            .get(special)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::chars)
            .map(Iterator::count)
            .unwrap_or(0);
    }
    (count > 0).then_some(count)
}

fn reject_enabled(
    args: &serde_json::Map<String, Value>,
    root: &serde_json::Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<(), TacotronConfigError> {
    if bool_in(args, root, key).unwrap_or(false) {
        Err(unsupported(format!("{label} ({key}=true)")))
    } else {
        Ok(())
    }
}

fn string_in<'a>(
    args: &'a serde_json::Map<String, Value>,
    root: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Option<&'a str> {
    args.get(key)
        .or_else(|| root.get(key))
        .and_then(Value::as_str)
}

fn bool_in(
    args: &serde_json::Map<String, Value>,
    root: &serde_json::Map<String, Value>,
    key: &str,
) -> Option<bool> {
    args.get(key)
        .or_else(|| root.get(key))
        .and_then(Value::as_bool)
}

fn usize_in(
    args: &serde_json::Map<String, Value>,
    root: &serde_json::Map<String, Value>,
    key: &str,
) -> Option<usize> {
    args.get(key)
        .or_else(|| root.get(key))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn number_in(
    args: &serde_json::Map<String, Value>,
    root: &serde_json::Map<String, Value>,
    key: &str,
) -> Option<f64> {
    args.get(key)
        .or_else(|| root.get(key))
        .and_then(Value::as_f64)
}

fn usize_alias(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<usize> {
    keys.iter().find_map(|key| {
        object
            .get(*key)
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
    })
}

fn bool_alias(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<bool> {
    keys.iter()
        .find_map(|key| object.get(*key).and_then(Value::as_bool))
}

fn invalid(message: impl Into<String>) -> TacotronConfigError {
    TacotronConfigError::Invalid(message.into())
}

fn unsupported(message: impl Into<String>) -> TacotronConfigError {
    TacotronConfigError::Unsupported(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_released_ljspeech_ddc_shape_and_pruned_training_decoder() {
        let root: Value = json5::from_str(
            r#"{
                model: "Tacotron2",
                run_name: "ljspeech-ddc",
                run_description: "Tacotron2 with DDC; second decoder pruned for inference.",
                audio: { num_mels: 80 },
                characters: {
                    pad: "", eos: "", bos: "",
                    characters: "_-!'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                    punctuations: "", phonemes: ""
                },
                r: 1,
                attention_type: "original",
                attention_norm: "softmax",
                location_attn: true,
                double_decoder_consistency: false,
                ddc_r: 7
            }"#,
        )
        .unwrap();
        let config = TacotronInferenceConfig::from_json_value(&root).unwrap();
        assert_eq!(config.architecture, TacotronArchitecture::Tacotron2);
        assert_eq!(config.variant, TacotronVariant::DoubleDecoderConsistency);
        assert_eq!(config.num_chars, 64);
        assert_eq!(config.out_channels, 80);
        assert_eq!(config.encoder_channels, 512);
        assert_eq!(config.ddc_reduction_factor, Some(7));
    }

    #[test]
    fn parses_capacitron_conditioning_contract() {
        let root: Value = serde_json::json!({
            "model": "tacotron2",
            "num_chars": 70,
            "out_channels": 80,
            "use_capacitron_vae": true,
            "capacitron_vae": {
                "capacitron_VAE_embedding_dim": 50,
                "capacitron_use_text_summary_embeddings": true,
                "capacitron_text_summary_embedding_dim": 128,
                "capacitron_use_speaker_embedding": false
            }
        });
        let config = TacotronInferenceConfig::from_json_value(&root).unwrap();
        assert_eq!(config.variant, TacotronVariant::Capacitron);
        assert_eq!(config.capacitron.unwrap().embedding_dim, 50);
    }

    #[test]
    fn rejects_historical_attention_with_actionable_diagnostic() {
        let root: Value = serde_json::json!({
            "model": "tacotron2",
            "num_chars": 64,
            "out_channels": 80,
            "attention_type": "dynamic_convolution"
        });
        let error = TacotronInferenceConfig::from_json_value(&root).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("dynamic_convolution"));
        assert!(message.contains("location-sensitive"));
    }
}
