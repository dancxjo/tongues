//! Stable component identities and executable speech-pipeline selections.
//!
//! The registry is intentionally artifact-location independent. Catalog and
//! runtime layers enrich these descriptors with installation, verification,
//! and load state.

use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};

use crate::components::AudioDecoder;
use crate::{AcousticOutputContract, CapabilityTier, NativeSpeechComponentReadiness};

pub const TEXT_INPUT_COMPONENT_ID: &str = "text";
pub const WAV_OUTPUT_COMPONENT_ID: &str = "wav";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechPipelineSelection {
    pub input: String,
    pub projector: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acoustic_model: Option<String>,
    #[serde(default)]
    pub conditioners: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocoder: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_to_end: Option<String>,
    pub output: String,
}

impl SpeechPipelineSelection {
    pub fn acoustic(
        projector: impl Into<String>,
        acoustic_model: impl Into<String>,
        vocoder: impl Into<String>,
    ) -> Self {
        Self {
            input: TEXT_INPUT_COMPONENT_ID.into(),
            projector: projector.into(),
            acoustic_model: Some(acoustic_model.into()),
            conditioners: Vec::new(),
            vocoder: Some(vocoder.into()),
            end_to_end: None,
            output: WAV_OUTPUT_COMPONENT_ID.into(),
        }
    }

    pub fn end_to_end(
        projector: impl Into<String>,
        end_to_end: impl Into<String>,
        conditioners: Vec<String>,
    ) -> Self {
        Self {
            input: TEXT_INPUT_COMPONENT_ID.into(),
            projector: projector.into(),
            acoustic_model: None,
            conditioners,
            vocoder: None,
            end_to_end: Some(end_to_end.into()),
            output: WAV_OUTPUT_COMPONENT_ID.into(),
        }
    }

    pub fn validate_shape(&self) -> Result<()> {
        ensure!(
            self.input == TEXT_INPUT_COMPONENT_ID,
            "pipeline input must be `{TEXT_INPUT_COMPONENT_ID}`"
        );
        ensure!(
            self.output == WAV_OUTPUT_COMPONENT_ID,
            "pipeline output must be `{WAV_OUTPUT_COMPONENT_ID}`"
        );
        ensure!(
            !self.projector.trim().is_empty(),
            "pipeline projector is required"
        );
        match (
            self.acoustic_model.as_deref(),
            self.vocoder.as_deref(),
            self.end_to_end.as_deref(),
        ) {
            (Some(_), Some(_), None) | (None, None, Some(_)) => {}
            (Some(_), None, None) => bail!("acoustic pipeline requires a vocoder"),
            (None, Some(_), None) => bail!("vocoder requires an acoustic model"),
            (Some(_), _, Some(_)) | (_, Some(_), Some(_)) => {
                bail!("pipeline cannot combine end-to-end and acoustic/vocoder stages")
            }
            (None, None, None) => {
                bail!("pipeline requires an acoustic/vocoder pair or an end-to-end model")
            }
        }
        ensure!(
            self.conditioners
                .iter()
                .all(|conditioner| !conditioner.trim().is_empty()),
            "pipeline contains an empty conditioner identity"
        );
        Ok(())
    }

    pub fn canonical_id(&self) -> Result<String> {
        self.validate_shape()?;
        let generator = if let Some(end_to_end) = self.end_to_end.as_deref() {
            format!("e2e={end_to_end}")
        } else {
            format!(
                "acoustic={};vocoder={}",
                self.acoustic_model.as_deref().unwrap_or_default(),
                self.vocoder.as_deref().unwrap_or_default()
            )
        };
        let conditioners = if self.conditioners.is_empty() {
            "none".into()
        } else {
            self.conditioners.join(",")
        };
        Ok(format!(
            "input={};projector={};{};conditioners={};output={}",
            self.input, self.projector, generator, conditioners, self.output
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechPipelineStage {
    Input,
    Projector,
    AcousticModel,
    Conditioner,
    Vocoder,
    EndToEnd,
    Output,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechPortContract {
    pub kind: String,
    pub key: String,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechPipelineComponent {
    pub id: String,
    pub display_name: String,
    pub architecture: String,
    pub stage: SpeechPipelineStage,
    #[serde(default)]
    pub spans: Vec<SpeechPipelineStage>,
    pub readiness: NativeSpeechComponentReadiness,
    #[serde(default)]
    pub accepts: Vec<SpeechPortContract>,
    #[serde(default)]
    pub produces: Vec<SpeechPortContract>,
    #[serde(default)]
    pub controls: Vec<String>,
    pub explanation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechPipelineCompatibility {
    pub from_component_id: String,
    pub to_component_id: String,
    pub compatible: bool,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisteredSpeechComposition {
    pub id: String,
    pub display_name: String,
    pub backend: String,
    pub model: String,
    pub pipeline: SpeechPipelineSelection,
    pub recommended: bool,
    /// Portable capability tier for this composition.
    ///
    /// Tier A: low-latency conversational (revision-capable, streaming).
    /// Tier B: language coverage (committed-phrase, offline preprocessing).
    /// Tier C: expressive accelerated (reference-conditioned, high-parameter).
    /// Unassigned: under evaluation or no tier confirmed yet.
    #[serde(default)]
    pub capability_tier: CapabilityTier,
    /// Whether this pipeline supports suffix regeneration (revision-capable).
    /// All Tier A pipelines set this to `true`.
    #[serde(default)]
    pub revision_capable: bool,
}

impl RegisteredSpeechComposition {
    fn new(
        display_name: &str,
        backend: &str,
        model: &str,
        pipeline: SpeechPipelineSelection,
        recommended: bool,
    ) -> Self {
        let id = pipeline
            .canonical_id()
            .expect("registered speech composition must be structurally valid");
        Self {
            id,
            display_name: display_name.into(),
            backend: backend.into(),
            model: model.into(),
            pipeline,
            recommended,
            capability_tier: CapabilityTier::Unassigned,
            revision_capable: false,
        }
    }

    fn with_tier(mut self, tier: CapabilityTier) -> Self {
        self.revision_capable = tier.is_revision_tier();
        self.capability_tier = tier;
        self
    }
}

pub fn registered_speech_compositions() -> Vec<RegisteredSpeechComposition> {
    vec![
        // --- Tier A: low-latency conversational ---
        // SpeedySpeech + HiFi-GAN: lean native path
        RegisteredSpeechComposition::new(
            "SpeedySpeech → HiFi-GAN",
            "burn",
            "speedyspeech-ljspeech+hifigan-v2",
            SpeechPipelineSelection::acoustic(
                "projector/speedy-speech-ljspeech",
                "speedy-speech-ljspeech",
                "hifigan-v2-ljspeech",
            ),
            true,
        )
        .with_tier(CapabilityTier::TierA),
        // FastPitch + HiFi-GAN: canonical Tier A, revision-capable
        RegisteredSpeechComposition::new(
            "FastPitch → HiFi-GAN",
            "fastpitch",
            "fastpitch-ljspeech+hifigan-v2",
            SpeechPipelineSelection::acoustic(
                "projector/fastpitch-ljspeech",
                "fastpitch-ljspeech",
                "hifigan-v2-ljspeech",
            ),
            true,
        )
        .with_tier(CapabilityTier::TierA),
        // Glow-TTS + MultiBand-MelGAN: lean path; Tier A candidacy
        // pending measured suffix-regeneration confirmation.
        RegisteredSpeechComposition::new(
            "Glow-TTS → pinned standardizer → MultiBand-MelGAN",
            "glow",
            "glow-tts-ljspeech+standardizer+multiband-melgan",
            SpeechPipelineSelection::acoustic(
                "projector/glow-tts-ljspeech",
                "glow-tts-ljspeech",
                "glow-standardized-multiband-melgan-ljspeech",
            ),
            true,
        )
        .with_tier(CapabilityTier::TierA),
        // --- Tier B: language coverage ---
        // Native VITS: committed-phrase, no suffix revision
        RegisteredSpeechComposition::new(
            "VITS VCTK",
            "vits",
            "vits-vctk",
            SpeechPipelineSelection::end_to_end("projector/vits-vctk", "vits-vctk", Vec::new()),
            true,
        )
        .with_tier(CapabilityTier::TierB),
        // YourTTS: multilingual committed-phrase coverage
        RegisteredSpeechComposition::new(
            "YourTTS Multilingual",
            "yourtts",
            "yourtts-multilingual",
            SpeechPipelineSelection::end_to_end(
                "projector/yourtts-multilingual",
                "yourtts-multilingual",
                vec!["speaker-encoder".into()],
            ),
            true,
        )
        .with_tier(CapabilityTier::TierB),
        // --- Unassigned: voice conversion (not a TTS tier) ---
        RegisteredSpeechComposition::new(
            "FreeVC24 Voice Conversion",
            "freevc",
            "freevc24-vctk",
            SpeechPipelineSelection::end_to_end(
                "projector/freevc-content",
                "freevc24-vctk",
                vec!["wavlm-large".into(), "freevc-speaker-encoder".into()],
            ),
            true,
        ),
        // --- Tier C: expressive accelerated ---
        // StyleTTS2: reference-conditioned; warmup measured separately
        RegisteredSpeechComposition::new(
            "StyleTTS2 English",
            "styletts2",
            "styletts2-en-us",
            SpeechPipelineSelection::end_to_end(
                "projector/styletts2-en-us",
                "styletts2-en-us",
                vec!["style-reference-encoder".into()],
            ),
            true,
        )
        .with_tier(CapabilityTier::TierC),
        // --- Unassigned: deterministic mock for testing ---
        RegisteredSpeechComposition::new(
            "Deterministic Mock",
            "mock",
            "deterministic-mock",
            SpeechPipelineSelection::end_to_end(
                "projector/deterministic-mock",
                "deterministic-mock",
                Vec::new(),
            ),
            false,
        ),
        // User-supplied voice database. Runtime availability is resolved by the
        // host from TONGUES_MBROLA_VOICE rather than a redistributed artifact.
        RegisteredSpeechComposition::new(
            "Native MBROLA TD-PSOLA",
            "mbrola",
            "mbrola-us3",
            SpeechPipelineSelection::end_to_end(
                "projector/mbrola-phone-timing",
                "mbrola-native-td-psola",
                Vec::new(),
            ),
            false,
        ),
    ]
}

fn port(kind: &str, key: impl Into<String>, summary: impl Into<String>) -> SpeechPortContract {
    SpeechPortContract {
        kind: kind.into(),
        key: key.into(),
        summary: summary.into(),
    }
}

/// Component-level view of the built-in pipeline registry.
///
/// Artifact state is deliberately absent; consumers join these stable
/// descriptors with model-catalog and resident-runtime state.
pub fn registered_speech_pipeline_components() -> Vec<SpeechPipelineComponent> {
    let text = port(
        "text",
        "tongues/text-v1",
        "UTF-8 text planned through Tongues linguistic IR.",
    );
    let plan = port(
        "linguistic_plan",
        "tongues/utterance-plan-v1",
        "Backend-neutral Tongues utterance plan.",
    );
    let waveform = port(
        "waveform",
        "audio/wav-mono-f32",
        "Mono waveform rendered as downloadable WAV audio.",
    );
    let neutral_mel = port(
        "mel_spectrogram",
        "mel/coqui-ljspeech-neutral-v1",
        "80-bin LJSpeech mel features with the complete published analysis contract.",
    );
    let mut components = vec![
        SpeechPipelineComponent {
            id: TEXT_INPUT_COMPONENT_ID.into(),
            display_name: "Text".into(),
            architecture: "tongues-text-input".into(),
            stage: SpeechPipelineStage::Input,
            spans: Vec::new(),
            readiness: NativeSpeechComponentReadiness::Runtime,
            accepts: Vec::new(),
            produces: vec![text],
            controls: Vec::new(),
            explanation: "Text input planned into Tongues linguistic IR.".into(),
        },
        SpeechPipelineComponent {
            id: WAV_OUTPUT_COMPONENT_ID.into(),
            display_name: "WAV output".into(),
            architecture: "wav".into(),
            stage: SpeechPipelineStage::Output,
            spans: Vec::new(),
            readiness: NativeSpeechComponentReadiness::Runtime,
            accepts: vec![waveform.clone()],
            produces: Vec::new(),
            controls: Vec::new(),
            explanation: "Playback, download, and waveform metadata output.".into(),
        },
        SpeechPipelineComponent {
            id: "speedy-speech-ljspeech".into(),
            display_name: "SpeedySpeech LJSpeech".into(),
            architecture: "speedy-speech".into(),
            stage: SpeechPipelineStage::AcousticModel,
            spans: Vec::new(),
            readiness: NativeSpeechComponentReadiness::Runtime,
            accepts: vec![port(
                "model_tokens",
                "tokens/speedy-speech-ljspeech",
                "Checkpoint-private projected tokens.",
            )],
            produces: vec![neutral_mel.clone()],
            controls: vec!["speed".into(), "seed".into()],
            explanation: "Native acoustic model requiring an exact-contract neural vocoder.".into(),
        },
        SpeechPipelineComponent {
            id: "fastpitch-ljspeech".into(),
            display_name: "FastPitch LJSpeech".into(),
            architecture: "fastpitch".into(),
            stage: SpeechPipelineStage::AcousticModel,
            spans: Vec::new(),
            readiness: NativeSpeechComponentReadiness::Runtime,
            accepts: vec![port(
                "model_tokens",
                "tokens/fastpitch-ljspeech",
                "Checkpoint-private projected tokens.",
            )],
            produces: vec![neutral_mel.clone()],
            controls: vec![
                "speed".into(),
                "pitch_scale".into(),
                "pitch_shift".into(),
                "pitch".into(),
                "durations".into(),
            ],
            explanation:
                "Pitch-conditioned native acoustic model requiring an exact-contract vocoder."
                    .into(),
        },
        SpeechPipelineComponent {
            id: "glow-tts-ljspeech".into(),
            display_name: "Glow-TTS LJSpeech".into(),
            architecture: "glow-tts".into(),
            stage: SpeechPipelineStage::AcousticModel,
            spans: Vec::new(),
            readiness: NativeSpeechComponentReadiness::Runtime,
            accepts: vec![port(
                "model_tokens",
                "tokens/glow-tts-ljspeech",
                "Checkpoint-private projected tokens.",
            )],
            produces: vec![port(
                "mel_spectrogram",
                "mel/glow-tts-ljspeech-log10-v1",
                "Unstandardized 80-bin Glow-TTS LJSpeech mel features.",
            )],
            controls: vec![
                "speed".into(),
                "durations".into(),
                "noise_scale".into(),
                "seed".into(),
            ],
            explanation: "Native flow acoustic inference with a checkpoint-exact mel contract."
                .into(),
        },
        SpeechPipelineComponent {
            id: "hifigan-v2-ljspeech".into(),
            display_name: "HiFi-GAN v2 LJSpeech".into(),
            architecture: "hifigan-v2".into(),
            stage: SpeechPipelineStage::Vocoder,
            spans: Vec::new(),
            readiness: NativeSpeechComponentReadiness::Runtime,
            accepts: vec![neutral_mel],
            produces: vec![waveform.clone()],
            controls: Vec::new(),
            explanation: "Native waveform decoder for its exact published mel contract.".into(),
        },
        SpeechPipelineComponent {
            id: "glow-standardized-multiband-melgan-ljspeech".into(),
            display_name: "Glow-TTS standardizer → MultiBand-MelGAN".into(),
            architecture: "spectrogram-standardizer+multiband-melgan".into(),
            stage: SpeechPipelineStage::Vocoder,
            spans: Vec::new(),
            readiness: NativeSpeechComponentReadiness::Runtime,
            accepts: vec![port(
                "mel_spectrogram",
                "mel/glow-tts-ljspeech-log10-v1",
                "Unstandardized Glow-TTS features with exact matching analysis geometry.",
            )],
            produces: vec![waveform.clone()],
            controls: Vec::new(),
            explanation: format!(
                "Named `{}` per-bin conversion pinned to scale_stats.npy, followed by native MultiBand-MelGAN.",
                crate::GLOW_MULTIBAND_STANDARDIZER_ID
            ),
        },
        SpeechPipelineComponent {
            id: "speaker-encoder".into(),
            display_name: "Coqui ResNet Speaker Encoder".into(),
            architecture: "speaker-encoder".into(),
            stage: SpeechPipelineStage::Conditioner,
            spans: Vec::new(),
            readiness: NativeSpeechComponentReadiness::Runtime,
            accepts: vec![port(
                "reference_audio",
                "audio/reference-waveform",
                "Reference speaker audio.",
            )],
            produces: vec![port(
                "speaker_embedding",
                "embedding/coqui-resnet-256",
                "Normalized 256-value Coqui speaker embedding.",
            )],
            controls: vec!["speaker".into(), "voice_sample".into()],
            explanation: "Reference-audio speaker conditioning for YourTTS.".into(),
        },
        SpeechPipelineComponent {
            id: "style-reference-encoder".into(),
            display_name: "Style reference encoder".into(),
            architecture: "styletts2-reference-encoder".into(),
            stage: SpeechPipelineStage::Conditioner,
            spans: Vec::new(),
            readiness: NativeSpeechComponentReadiness::Experimental,
            accepts: vec![port(
                "reference_audio",
                "audio/reference-waveform",
                "Speaker or style reference audio.",
            )],
            produces: vec![port(
                "style_embedding",
                "embedding/styletts2-256",
                "StyleTTS2 speaker/style embedding.",
            )],
            controls: vec![
                "voice_sample".into(),
                "style_sample".into(),
                "emotion".into(),
            ],
            explanation: "Reference conditioning intrinsic to the StyleTTS2 composition.".into(),
        },
        SpeechPipelineComponent {
            id: "wavlm-large".into(),
            display_name: "WavLM Large content encoder".into(),
            architecture: "wavlm-large".into(),
            stage: SpeechPipelineStage::Conditioner,
            spans: Vec::new(),
            readiness: NativeSpeechComponentReadiness::Runtime,
            accepts: vec![port(
                "source_audio",
                "audio/source-waveform-16khz",
                "Source speech waveform at the model analysis rate.",
            )],
            produces: vec![port(
                "content_embedding",
                "embedding/wavlm-large-layer-12",
                "WavLM content representation for voice conversion.",
            )],
            controls: vec!["source_sample".into()],
            explanation: "Source-speech content conditioning for FreeVC24.".into(),
        },
        SpeechPipelineComponent {
            id: "freevc-speaker-encoder".into(),
            display_name: "FreeVC speaker encoder".into(),
            architecture: "freevc-speaker-encoder".into(),
            stage: SpeechPipelineStage::Conditioner,
            spans: Vec::new(),
            readiness: NativeSpeechComponentReadiness::Runtime,
            accepts: vec![port(
                "reference_audio",
                "audio/reference-waveform-16khz",
                "Target speaker reference waveform.",
            )],
            produces: vec![port(
                "speaker_embedding",
                "embedding/freevc-speaker",
                "FreeVC target-speaker conditioning.",
            )],
            controls: vec!["voice_sample".into()],
            explanation: "Target-speaker reference conditioning for FreeVC24.".into(),
        },
    ];
    for composition in registered_speech_compositions() {
        if let Some(end_to_end) = composition.pipeline.end_to_end.as_ref() {
            if !components
                .iter()
                .any(|component| component.id == *end_to_end)
            {
                components.push(SpeechPipelineComponent {
                    id: end_to_end.clone(),
                    display_name: composition.display_name.clone(),
                    architecture: if composition.backend == "yourtts" {
                        "vits".into()
                    } else {
                        composition.backend.clone()
                    },
                    stage: SpeechPipelineStage::EndToEnd,
                    spans: vec![
                        SpeechPipelineStage::AcousticModel,
                        SpeechPipelineStage::Vocoder,
                    ],
                    readiness: if composition.backend == "styletts2" {
                        NativeSpeechComponentReadiness::Experimental
                    } else {
                        NativeSpeechComponentReadiness::Runtime
                    },
                    accepts: vec![port(
                        "model_tokens",
                        format!("tokens/{}", composition.model),
                        "Checkpoint-private projected tokens and declared conditioning.",
                    )],
                    produces: vec![waveform.clone()],
                    controls: match composition.backend.as_str() {
                        "mbrola" => vec![
                            "voice_database".into(),
                            "symbol_map".into(),
                            "speed".into(),
                            "pitch".into(),
                            "durations".into(),
                            "pho".into(),
                        ],
                        "vits" => vec!["speaker".into(), "speed".into(), "seed".into()],
                        "yourtts" => vec![
                            "speaker".into(),
                            "model_language".into(),
                            "voice_sample".into(),
                            "speed".into(),
                            "seed".into(),
                        ],
                        "freevc" => vec!["source_sample".into(), "voice_sample".into()],
                        "styletts2" => vec![
                            "voice_sample".into(),
                            "style_sample".into(),
                            "emotion".into(),
                            "speed".into(),
                            "seed".into(),
                        ],
                        _ => Vec::new(),
                    },
                    explanation: if composition.backend == "mbrola" {
                        "Native user-supplied MBROLA database rendering with typed phone timing, F0 targets, `.pho` inspection, and mono f32 waveform output.".into()
                    } else {
                        "End-to-end model spanning acoustic generation and waveform decoding."
                            .into()
                    },
                });
            }
        }
        let projector_owner = composition
            .display_name
            .split('→')
            .next()
            .unwrap_or(&composition.display_name)
            .trim();
        components.push(SpeechPipelineComponent {
            id: composition.pipeline.projector.clone(),
            display_name: format!("{projector_owner} projector"),
            architecture: if composition.backend == "mbrola" {
                "mbrola-phone-timing".into()
            } else {
                "checkpoint-projector".into()
            },
            stage: SpeechPipelineStage::Projector,
            spans: Vec::new(),
            readiness: NativeSpeechComponentReadiness::Runtime,
            accepts: vec![plan.clone()],
            produces: vec![if composition.backend == "mbrola" {
                port(
                    "phone_timed_plan",
                    "mbrola/phone-timed-plan-v1",
                    "Voice-mapped phones with explicit millisecond durations and within-phone F0 targets; inspectable as `.pho`.",
                )
            } else {
                port(
                    "model_tokens",
                    format!("tokens/{}", composition.model),
                    format!(
                        "Checkpoint-local vocabulary and tokenization for {}.",
                        composition.display_name
                    ),
                )
            }],
            controls: if composition.backend == "mbrola" {
                vec!["speed".into(), "pitch".into(), "durations".into(), "pho".into()]
            } else {
                Vec::new()
            },
            explanation: if composition.backend == "mbrola" {
                "Deterministic UtterancePlan lowering using explicit voice symbol mapping, phone spans or documented timing defaults, breaks, and pitch source/fallback provenance.".into()
            } else {
                "Terminal projection into a checkpoint-private vocabulary; it cannot be substituted across models."
                    .into()
            },
        });
    }
    components.sort_by(|left, right| {
        format!("{:?}", left.stage)
            .cmp(&format!("{:?}", right.stage))
            .then_with(|| left.display_name.cmp(&right.display_name))
    });
    components.dedup_by(|left, right| left.id == right.id);
    components
}

pub fn registered_composition_for_pipeline(
    pipeline: &SpeechPipelineSelection,
) -> Result<RegisteredSpeechComposition> {
    let id = pipeline.canonical_id()?;
    registered_speech_compositions()
        .into_iter()
        .find(|composition| composition.id == id)
        .ok_or_else(|| anyhow::anyhow!("speech pipeline `{id}` is not registered"))
}

pub fn registered_composition_for_legacy(
    backend: &str,
    model: &str,
) -> Option<RegisteredSpeechComposition> {
    registered_speech_compositions()
        .into_iter()
        .find(|composition| composition.backend == backend && composition.model == model)
}

/// Validate a real acoustic/decoder boundary and retain its precise mismatch
/// as the explanation presented by discovery and user interfaces.
pub fn acoustic_decoder_compatibility(
    from_component_id: impl Into<String>,
    to_component_id: impl Into<String>,
    produced: &AcousticOutputContract,
    decoder: &dyn AudioDecoder,
) -> SpeechPipelineCompatibility {
    let from_component_id = from_component_id.into();
    let to_component_id = to_component_id.into();
    match decoder.validate_input_contract(produced) {
        Ok(()) => SpeechPipelineCompatibility {
            from_component_id,
            to_component_id,
            compatible: true,
            reason: "The declared acoustic and decoder contracts match exactly.".into(),
        },
        Err(error) => SpeechPipelineCompatibility {
            from_component_id,
            to_component_id,
            compatible: false,
            reason: error.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_shape_requires_exactly_one_generator_form() {
        let acoustic = SpeechPipelineSelection::acoustic("projector/a", "a", "v");
        assert!(acoustic.validate_shape().is_ok());
        assert_eq!(
            registered_composition_for_pipeline(
                &registered_speech_compositions()
                    .into_iter()
                    .find(|composition| composition.backend == "fastpitch")
                    .unwrap()
                    .pipeline
            )
            .unwrap()
            .backend,
            "fastpitch"
        );

        let mut ambiguous = acoustic;
        ambiguous.end_to_end = Some("e2e".into());
        assert!(ambiguous
            .validate_shape()
            .unwrap_err()
            .to_string()
            .contains("cannot combine"));
    }

    #[test]
    fn canonical_ids_include_every_model_affecting_stage() {
        let left = SpeechPipelineSelection::acoustic("projector/a", "a", "v1");
        let right = SpeechPipelineSelection::acoustic("projector/a", "a", "v2");
        assert_ne!(left.canonical_id().unwrap(), right.canonical_id().unwrap());
    }

    #[test]
    fn mbrola_is_shared_discovery_with_pho_stage_contract() {
        let composition = registered_speech_compositions()
            .into_iter()
            .find(|composition| composition.backend == "mbrola")
            .expect("MBROLA composition");
        assert_eq!(composition.model, "mbrola-us3");
        assert_eq!(
            composition.pipeline.projector,
            "projector/mbrola-phone-timing"
        );
        assert_eq!(
            composition.pipeline.end_to_end.as_deref(),
            Some("mbrola-native-td-psola")
        );
        let components = registered_speech_pipeline_components();
        let projector = components
            .iter()
            .find(|component| component.id == "projector/mbrola-phone-timing")
            .expect("MBROLA projector");
        assert_eq!(projector.produces[0].kind, "phone_timed_plan");
        assert!(projector.produces[0].summary.contains("`.pho`"));
        let renderer = components
            .iter()
            .find(|component| component.id == "mbrola-native-td-psola")
            .expect("MBROLA renderer");
        assert!(renderer.controls.iter().any(|control| control == "pho"));
    }

    #[test]
    fn shared_registry_exposes_ports_controls_and_spanning_models() {
        let components = registered_speech_pipeline_components();
        let fastpitch = components
            .iter()
            .find(|component| component.id == "fastpitch-ljspeech")
            .expect("FastPitch descriptor");
        assert_eq!(fastpitch.stage, SpeechPipelineStage::AcousticModel);
        assert!(fastpitch.controls.iter().any(|field| field == "pitch"));
        assert_eq!(fastpitch.produces[0].kind, "mel_spectrogram");

        let vits = components
            .iter()
            .find(|component| component.id == "vits-vctk")
            .expect("VITS descriptor");
        assert_eq!(vits.stage, SpeechPipelineStage::EndToEnd);
        assert_eq!(
            vits.spans,
            [
                SpeechPipelineStage::AcousticModel,
                SpeechPipelineStage::Vocoder
            ]
        );
    }

    #[test]
    fn compatibility_preserves_the_typed_contract_mismatch_reason() {
        let decoder = crate::IdentityAudioDecoder::new(crate::WaveformContract::mono(22_050))
            .expect("valid decoder");
        let compatible = acoustic_decoder_compatibility(
            "waveform-model",
            "wav",
            &AcousticOutputContract::Waveform(crate::WaveformContract::mono(22_050)),
            &decoder,
        );
        assert!(compatible.compatible);

        let incompatible = acoustic_decoder_compatibility(
            "waveform-model",
            "wav",
            &AcousticOutputContract::Waveform(crate::WaveformContract::mono(24_000)),
            &decoder,
        );
        assert!(!incompatible.compatible);
        assert!(incompatible.reason.contains("waveform contract mismatch"));
        assert!(incompatible.reason.contains("24000"));
        assert!(incompatible.reason.contains("22050"));
    }

    #[test]
    fn fastpitch_hifigan_is_canonical_tier_a_and_revision_capable() {
        let compositions = registered_speech_compositions();
        let fastpitch = compositions
            .iter()
            .find(|c| c.backend == "fastpitch")
            .expect("FastPitch composition must be registered");
        assert_eq!(
            fastpitch.capability_tier,
            CapabilityTier::TierA,
            "FastPitch + HiFi-GAN must be Tier A"
        );
        assert!(
            fastpitch.revision_capable,
            "FastPitch + HiFi-GAN must declare revision_capable"
        );
    }

    #[test]
    fn speedy_speech_is_tier_a_and_vits_is_tier_b() {
        let compositions = registered_speech_compositions();
        let speedy = compositions
            .iter()
            .find(|c| c.backend == "burn")
            .expect("SpeedySpeech composition must be registered");
        assert_eq!(
            speedy.capability_tier,
            CapabilityTier::TierA,
            "SpeedySpeech + HiFi-GAN must be Tier A"
        );

        let vits = compositions
            .iter()
            .find(|c| c.backend == "vits")
            .expect("VITS composition must be registered");
        assert_eq!(
            vits.capability_tier,
            CapabilityTier::TierB,
            "VITS must be Tier B"
        );
        assert!(
            !vits.revision_capable,
            "VITS must not declare revision_capable"
        );
    }

    #[test]
    fn styletts2_is_tier_c_and_freevc_is_unassigned() {
        let compositions = registered_speech_compositions();
        let styletts2 = compositions
            .iter()
            .find(|c| c.backend == "styletts2")
            .expect("StyleTTS2 composition must be registered");
        assert_eq!(
            styletts2.capability_tier,
            CapabilityTier::TierC,
            "StyleTTS2 must be Tier C"
        );

        let freevc = compositions
            .iter()
            .find(|c| c.backend == "freevc")
            .expect("FreeVC composition must be registered");
        assert_eq!(
            freevc.capability_tier,
            CapabilityTier::Unassigned,
            "FreeVC (voice conversion) must be Unassigned"
        );
    }

    #[test]
    fn tier_a_compositions_are_revision_capable() {
        let compositions = registered_speech_compositions();
        for composition in &compositions {
            if composition.capability_tier == CapabilityTier::TierA {
                assert!(
                    composition.revision_capable,
                    "Tier A composition `{}` must declare revision_capable",
                    composition.display_name
                );
            }
        }
    }

    #[test]
    fn capability_tier_serializes_with_snake_case_keys() {
        let v = serde_json::to_value(CapabilityTier::TierA).expect("serialize tier");
        assert_eq!(v, "tier_a");
        let v = serde_json::to_value(CapabilityTier::TierB).expect("serialize tier");
        assert_eq!(v, "tier_b");
        let v = serde_json::to_value(CapabilityTier::TierC).expect("serialize tier");
        assert_eq!(v, "tier_c");
        let v = serde_json::to_value(CapabilityTier::Unassigned).expect("serialize tier");
        assert_eq!(v, "unassigned");
    }
}
