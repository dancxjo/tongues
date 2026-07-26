//! Native Burn implementation of Coqui `ForwardTTS` FastSpeech variants.
//!
//! Coqui represents FastSpeech, FastPitch, and FastSpeech 2 as configurations
//! of one feed-forward duration model. The transformer and duration-expansion
//! primitives are therefore shared with the native FastPitch and SpeedySpeech
//! implementations. This module adds the optional pitch and energy variance
//! paths which distinguish FastSpeech 2 from the original FastSpeech graph.
//!
//! Input IDs are checkpoint-local. Text normalization, phonemization, and
//! vocabulary projection remain adapter responsibilities.
//!
//! Source provenance: `audit-required`. This module targets published Coqui
//! checkpoint structure and behavior; `docs/provenance.md` records the source
//! files that must be audited before an independent-implementation claim.

use std::fmt;
use std::path::Path;
use std::time::Instant;

use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, PaddingConfig1d};
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::burn_fast_pitch::{
    FastPitchDecoder, FastPitchDecoderContainer, FastPitchEncoder, FeedForwardTransformerBlock,
    FeedForwardTransformerConfig,
};
use crate::burn_speedy_speech::{
    expand_by_durations, AlignmentNetwork, DurationPredictor, PositionalEncoding,
};
use crate::profiling::finish_backend_stage;
use crate::{
    EnergyCapabilities, PitchCapabilities, SynthesisDimension, SynthesisProfiler, SynthesisStage,
};

const MAX_POSITIONAL_FRAMES: usize = 5_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastSpeechVariant {
    FastSpeech,
    FastSpeech2,
}

impl FastSpeechVariant {
    pub fn supports_pitch(self) -> bool {
        matches!(self, Self::FastSpeech2)
    }

    pub fn supports_energy(self) -> bool {
        matches!(self, Self::FastSpeech2)
    }

    pub fn pitch_capabilities(self) -> PitchCapabilities {
        let supported = self.supports_pitch();
        PitchCapabilities {
            scale: supported,
            shift: supported,
            explicit_values: supported,
        }
    }

    pub fn energy_capabilities(self) -> EnergyCapabilities {
        let supported = self.supports_energy();
        EnergyCapabilities {
            scale: supported,
            shift: supported,
            explicit_values: supported,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FastSpeechError {
    InvalidConfig(String),
    InvalidInput(String),
    Checkpoint(String),
}

impl fmt::Display for FastSpeechError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid FastSpeech config: {message}")
            }
            Self::InvalidInput(message) => {
                write!(formatter, "invalid FastSpeech input: {message}")
            }
            Self::Checkpoint(message) => {
                write!(formatter, "unable to load FastSpeech checkpoint: {message}")
            }
        }
    }
}

impl std::error::Error for FastSpeechError {}

fn config_error(message: impl Into<String>) -> FastSpeechError {
    FastSpeechError::InvalidConfig(message.into())
}

fn input_error(message: impl Into<String>) -> FastSpeechError {
    FastSpeechError::InvalidInput(message.into())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariancePredictorConfig {
    pub hidden_channels: usize,
    pub kernel_size: usize,
    pub dropout: f64,
    pub embedding_kernel_size: usize,
}

impl VariancePredictorConfig {
    fn from_args(args: &Value, prefix: &str) -> Result<Self, FastSpeechError> {
        let config = Self {
            hidden_channels: usize_at(args, &[&format!("{prefix}_predictor_hidden_channels")])?,
            kernel_size: usize_at(args, &[&format!("{prefix}_predictor_kernel_size")])?,
            dropout: number_at(args, &[&format!("{prefix}_predictor_dropout_p")])?,
            embedding_kernel_size: usize_at(args, &[&format!("{prefix}_embedding_kernel_size")])?,
        };
        config.validate(prefix)?;
        Ok(config)
    }

    fn validate(&self, label: &str) -> Result<(), FastSpeechError> {
        if self.hidden_channels == 0 {
            return Err(config_error(format!(
                "{label} predictor hidden channels must be positive"
            )));
        }
        if self.kernel_size == 0 || self.kernel_size.is_multiple_of(2) {
            return Err(config_error(format!(
                "{label} predictor kernel size must be positive and odd"
            )));
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(config_error(format!(
                "{label} predictor dropout must be in [0, 1)"
            )));
        }
        if self.embedding_kernel_size == 0 || self.embedding_kernel_size.is_multiple_of(2) {
            return Err(config_error(format!(
                "{label} embedding kernel size must be positive and odd"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FastSpeechConfig {
    pub variant: FastSpeechVariant,
    pub num_chars: usize,
    pub out_channels: usize,
    pub hidden_channels: usize,
    pub positional_encoding: bool,
    pub length_scale: f64,
    pub encoder: FeedForwardTransformerConfig,
    pub decoder: FeedForwardTransformerConfig,
    pub duration_predictor_hidden_channels: usize,
    pub duration_predictor_kernel_size: usize,
    pub duration_predictor_dropout: f64,
    pub pitch: Option<VariancePredictorConfig>,
    pub energy: Option<VariancePredictorConfig>,
    pub use_aligner: bool,
    pub max_duration: usize,
    pub max_output_frames: usize,
}

impl FastSpeechConfig {
    pub fn fastspeech(num_chars: usize) -> Self {
        Self::with_variant(FastSpeechVariant::FastSpeech, num_chars)
    }

    pub fn fastspeech2(num_chars: usize) -> Self {
        Self::with_variant(FastSpeechVariant::FastSpeech2, num_chars)
    }

    fn with_variant(variant: FastSpeechVariant, num_chars: usize) -> Self {
        let transformer = FeedForwardTransformerConfig {
            hidden_channels_ffn: 1_024,
            num_heads: 1,
            num_layers: 6,
            dropout: 0.1,
        };
        let variance = VariancePredictorConfig {
            hidden_channels: 256,
            kernel_size: 3,
            dropout: 0.1,
            embedding_kernel_size: 3,
        };
        Self {
            variant,
            num_chars,
            out_channels: 80,
            hidden_channels: 384,
            positional_encoding: true,
            length_scale: 1.0,
            encoder: transformer.clone(),
            decoder: transformer,
            duration_predictor_hidden_channels: 256,
            duration_predictor_kernel_size: 3,
            duration_predictor_dropout: 0.1,
            pitch: variant.supports_pitch().then(|| variance.clone()),
            energy: variant.supports_energy().then_some(variance),
            use_aligner: true,
            max_duration: 75,
            max_output_frames: 20_000,
        }
    }

    /// Parse a Coqui `ForwardTTS` FastSpeech or FastSpeech 2 configuration.
    pub fn from_json_value(root: &Value) -> Result<Self, FastSpeechError> {
        let model = string_at(root, &["model"])?;
        let variant = match model {
            "fastspeech" | "fast_speech" => FastSpeechVariant::FastSpeech,
            "fastspeech2" | "fast_speech2" => FastSpeechVariant::FastSpeech2,
            other => {
                return Err(config_error(format!(
                    "expected FastSpeech model, got {other:?}"
                )));
            }
        };
        if let Some(base_model) = root.get("base_model").and_then(Value::as_str) {
            if base_model != "forward_tts" {
                return Err(config_error(format!(
                    "unsupported base_model {base_model:?}; FastSpeech checkpoints require \"forward_tts\""
                )));
            }
        }
        let args = object_at(root, &["model_args"])?;
        for (field, expected) in [
            ("encoder_type", "fftransformer"),
            ("decoder_type", "fftransformer"),
        ] {
            let actual = string_at(args, &[field])?;
            if actual != expected {
                return Err(config_error(format!(
                    "unsupported {field} {actual:?}; FastSpeech requires {expected}"
                )));
            }
        }
        reject_speaker_conditioning(root, args)?;

        let use_pitch = optional_bool_at(args, "use_pitch")?.unwrap_or(false);
        let use_energy = optional_bool_at(args, "use_energy")?.unwrap_or(false);
        match variant {
            FastSpeechVariant::FastSpeech if use_pitch || use_energy => {
                return Err(config_error(
                    "FastSpeech must disable pitch and energy variance predictors",
                ));
            }
            FastSpeechVariant::FastSpeech2 if !use_pitch || !use_energy => {
                return Err(config_error(
                    "FastSpeech 2 requires both pitch and energy variance predictors",
                ));
            }
            _ => {}
        }

        let num_chars = match args.get("num_chars").and_then(Value::as_u64) {
            Some(value) => usize::try_from(value)
                .map_err(|_| config_error("model_args.num_chars does not fit usize"))?,
            None => published_vocabulary_size(root)?,
        };
        let config = Self {
            variant,
            num_chars,
            out_channels: usize_at(args, &["out_channels"])?,
            hidden_channels: usize_at(args, &["hidden_channels"])?,
            positional_encoding: bool_at(args, &["positional_encoding"])?,
            length_scale: number_at(args, &["length_scale"])?,
            encoder: FeedForwardTransformerConfig::from_value(
                object_at(args, &["encoder_params"])?,
                "encoder",
            )
            .map_err(|error| config_error(error.to_string()))?,
            decoder: FeedForwardTransformerConfig::from_value(
                object_at(args, &["decoder_params"])?,
                "decoder",
            )
            .map_err(|error| config_error(error.to_string()))?,
            duration_predictor_hidden_channels: usize_at(
                args,
                &["duration_predictor_hidden_channels"],
            )?,
            duration_predictor_kernel_size: usize_at(args, &["duration_predictor_kernel_size"])?,
            duration_predictor_dropout: number_at(args, &["duration_predictor_dropout_p"])?,
            pitch: use_pitch
                .then(|| VariancePredictorConfig::from_args(args, "pitch"))
                .transpose()?,
            energy: use_energy
                .then(|| VariancePredictorConfig::from_args(args, "energy"))
                .transpose()?,
            use_aligner: bool_at(args, &["use_aligner"])?,
            max_duration: usize_at(args, &["max_duration"])?,
            max_output_frames: 20_000,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), FastSpeechError> {
        if self.num_chars == 0 || self.out_channels == 0 || self.hidden_channels == 0 {
            return Err(config_error(
                "character, output, and hidden dimensions must be positive",
            ));
        }
        if !self.hidden_channels.is_multiple_of(self.encoder.num_heads)
            || !self.hidden_channels.is_multiple_of(self.decoder.num_heads)
        {
            return Err(config_error(
                "hidden_channels must divide evenly across encoder and decoder heads",
            ));
        }
        if self.positional_encoding && !self.hidden_channels.is_multiple_of(2) {
            return Err(config_error(
                "hidden_channels must be even when positional encoding is enabled",
            ));
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(config_error("length_scale must be finite and positive"));
        }
        if self.duration_predictor_hidden_channels == 0
            || self.duration_predictor_kernel_size == 0
            || self.duration_predictor_kernel_size.is_multiple_of(2)
        {
            return Err(config_error(
                "duration predictor hidden channels must be positive and kernel size must be positive and odd",
            ));
        }
        if !(0.0..1.0).contains(&self.duration_predictor_dropout) {
            return Err(config_error("duration predictor dropout must be in [0, 1)"));
        }
        match self.variant {
            FastSpeechVariant::FastSpeech => {
                if self.pitch.is_some() || self.energy.is_some() {
                    return Err(config_error(
                        "FastSpeech cannot contain pitch or energy variance predictors",
                    ));
                }
            }
            FastSpeechVariant::FastSpeech2 => {
                if self.pitch.is_none() || self.energy.is_none() {
                    return Err(config_error(
                        "FastSpeech 2 requires pitch and energy variance predictors",
                    ));
                }
            }
        }
        if let Some(config) = &self.pitch {
            config.validate("pitch")?;
        }
        if let Some(config) = &self.energy {
            config.validate("energy")?;
        }
        if self.max_duration == 0 || self.max_output_frames == 0 {
            return Err(config_error(
                "maximum duration and output frame limit must be positive",
            ));
        }
        self.encoder
            .validate("encoder")
            .map_err(|error| config_error(error.to_string()))?;
        self.decoder
            .validate("decoder")
            .map_err(|error| config_error(error.to_string()))?;
        Ok(())
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<FastSpeech<B>, FastSpeechError> {
        self.validate()?;
        let emb = EmbeddingConfig::new(self.num_chars, self.hidden_channels).init(device);
        let encoder = FastPitchEncoder {
            encoder: FeedForwardTransformerBlock::init(self.hidden_channels, &self.encoder, device),
        };
        let decoder = FastPitchDecoderContainer {
            decoder: FastPitchDecoder {
                transformer_block: FeedForwardTransformerBlock::init(
                    self.hidden_channels,
                    &self.decoder,
                    device,
                ),
                postnet: Conv1dConfig::new(self.hidden_channels, self.out_channels, 1)
                    .with_padding(PaddingConfig1d::Valid)
                    .init(device),
            },
        };
        let duration_predictor = DurationPredictor::init(
            self.hidden_channels,
            self.duration_predictor_hidden_channels,
            self.duration_predictor_kernel_size,
            device,
        );
        let (pitch_predictor, pitch_emb) =
            init_variance_path(self.hidden_channels, self.pitch.as_ref(), device);
        let (energy_predictor, energy_emb) =
            init_variance_path(self.hidden_channels, self.energy.as_ref(), device);
        let pos_encoder = self
            .positional_encoding
            .then(|| PositionalEncoding::init(self.hidden_channels, device));
        let aligner = self.use_aligner.then(|| {
            AlignmentNetwork::init(
                self.out_channels,
                self.hidden_channels,
                self.out_channels,
                device,
            )
        });
        Ok(FastSpeech {
            emb,
            encoder,
            pos_encoder,
            decoder,
            duration_predictor,
            pitch_predictor,
            pitch_emb,
            energy_predictor,
            energy_emb,
            aligner,
            variant: self.variant,
            length_scale: self.length_scale,
            num_chars: self.num_chars,
            out_channels: self.out_channels,
            max_duration: self.max_duration,
            max_output_frames: self.max_output_frames,
        })
    }
}

fn init_variance_path<B: Backend>(
    hidden_channels: usize,
    config: Option<&VariancePredictorConfig>,
    device: &B::Device,
) -> (Option<DurationPredictor<B>>, Option<Conv1d<B>>) {
    match config {
        Some(config) => (
            Some(DurationPredictor::init(
                hidden_channels,
                config.hidden_channels,
                config.kernel_size,
                device,
            )),
            Some(
                Conv1dConfig::new(1, hidden_channels, config.embedding_kernel_size)
                    .with_padding(PaddingConfig1d::Explicit(
                        config.embedding_kernel_size / 2,
                        config.embedding_kernel_size / 2,
                    ))
                    .init(device),
            ),
        ),
        None => (None, None),
    }
}

#[derive(Debug)]
pub struct FastSpeechOutput<B: Backend> {
    /// Mel spectrogram in `[batch, frames, mel_channels]`.
    pub mel: Tensor<B, 3>,
    /// Rounded frame durations in `[batch, tokens]`.
    pub durations: Tensor<B, 2>,
    /// Token-level pitch conditioning in `[batch, 1, tokens]`, when supported.
    pub pitch: Option<Tensor<B, 3>>,
    /// Token-level energy conditioning in `[batch, 1, tokens]`, when supported.
    pub energy: Option<Tensor<B, 3>>,
}

#[derive(Debug)]
pub struct FastSpeechControls<B: Backend> {
    pub length_scale: f64,
    pub durations: Option<Tensor<B, 2>>,
    pub pitch_scale: f64,
    pub pitch_shift: f64,
    pub pitch: Option<Tensor<B, 3>>,
    pub energy_scale: f64,
    pub energy_shift: f64,
    pub energy: Option<Tensor<B, 3>>,
}

impl<B: Backend> Default for FastSpeechControls<B> {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            durations: None,
            pitch_scale: 1.0,
            pitch_shift: 0.0,
            pitch: None,
            energy_scale: 1.0,
            energy_shift: 0.0,
            energy: None,
        }
    }
}

#[derive(Module, Debug)]
pub struct FastSpeech<B: Backend> {
    pub emb: Embedding<B>,
    pub encoder: FastPitchEncoder<B>,
    pub pos_encoder: Option<PositionalEncoding<B>>,
    pub decoder: FastPitchDecoderContainer<B>,
    pub duration_predictor: DurationPredictor<B>,
    pub pitch_predictor: Option<DurationPredictor<B>>,
    pub pitch_emb: Option<Conv1d<B>>,
    pub energy_predictor: Option<DurationPredictor<B>>,
    pub energy_emb: Option<Conv1d<B>>,
    pub aligner: Option<AlignmentNetwork<B>>,
    variant: FastSpeechVariant,
    length_scale: f64,
    num_chars: usize,
    out_channels: usize,
    max_duration: usize,
    max_output_frames: usize,
}

impl<B: Backend> FastSpeech<B> {
    pub fn variant(&self) -> FastSpeechVariant {
        self.variant
    }

    pub fn load_checkpoint(mut self, path: impl AsRef<Path>) -> Result<Self, FastSpeechError> {
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut self,
            path.as_ref(),
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                predicate: Some(checkpoint_tensor),
                key_remappings: vec![
                    (r"(\.norm_[12])\.weight$".into(), "$1.gamma".into()),
                    (r"(\.norm_[12])\.bias$".into(), "$1.beta".into()),
                    (r"(\.norm[12])\.weight$".into(), "$1.gamma".into()),
                    (r"(\.norm[12])\.bias$".into(), "$1.beta".into()),
                ],
                skip_enum_variants: true,
                ..Default::default()
            },
        )
        .map_err(|error| FastSpeechError::Checkpoint(format!("{error:#}")))?;
        let unexpected_unused = result
            .unused
            .iter()
            .filter(|path| !path.ends_with(".num_batches_tracked"))
            .cloned()
            .collect::<Vec<_>>();
        if !result.missing.is_empty() || !result.errors.is_empty() || !unexpected_unused.is_empty()
        {
            return Err(FastSpeechError::Checkpoint(format!(
                "checkpoint does not exactly match the Burn model: {} missing, {} load errors, unexpected tensors: {}",
                result.missing.len(),
                result.errors.len(),
                unexpected_unused.join(", ")
            )));
        }
        if let Some(positional) = &self.pos_encoder {
            let device = positional.pe.val().device();
            self.pos_encoder = Some(PositionalEncoding::init(positional.channels, &device));
        }
        Ok(self)
    }

    pub fn inference(
        &self,
        token_ids: Tensor<B, 2, Int>,
    ) -> Result<FastSpeechOutput<B>, FastSpeechError> {
        self.inference_with_controls(
            token_ids,
            FastSpeechControls {
                length_scale: self.length_scale,
                ..Default::default()
            },
            false,
            None,
        )
    }

    pub(crate) fn inference_projected_with_controls(
        &self,
        token_ids: Tensor<B, 2, Int>,
        controls: FastSpeechControls<B>,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<FastSpeechOutput<B>, FastSpeechError> {
        self.inference_with_controls(token_ids, controls, true, profiler)
    }

    fn inference_with_controls(
        &self,
        token_ids: Tensor<B, 2, Int>,
        controls: FastSpeechControls<B>,
        ids_validated_on_host: bool,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<FastSpeechOutput<B>, FastSpeechError> {
        let mut profiler = profiler;
        validate_controls(&controls)?;
        let [batch, tokens] = token_ids.dims();
        if batch == 0 || tokens == 0 {
            return Err(input_error(
                "token_ids must have non-empty [batch, tokens] dimensions",
            ));
        }
        if !ids_validated_on_host {
            let highest_id = token_ids.clone().max().into_scalar().elem::<i64>();
            if highest_id < 0 || highest_id as usize >= self.num_chars {
                return Err(input_error(format!(
                    "token ID {highest_id} is outside vocabulary 0..{}",
                    self.num_chars
                )));
            }
        }
        reject_unsupported_variance_controls(
            "pitch",
            self.pitch_predictor.is_some(),
            controls.pitch_scale,
            controls.pitch_shift,
            controls.pitch.is_some(),
        )?;
        reject_unsupported_variance_controls(
            "energy",
            self.energy_predictor.is_some(),
            controls.energy_scale,
            controls.energy_shift,
            controls.energy.is_some(),
        )?;

        let device = token_ids.device();
        let mask = Tensor::<B, 3>::ones([batch, 1, tokens], &device);
        let started = Instant::now();
        let embedded = self.emb.forward(token_ids).swap_dims(1, 2);
        let mut encoded = self.encoder.encoder.forward(embedded, mask.clone());
        finish_backend_stage::<B>(
            &mut profiler,
            &device,
            SynthesisStage::TextEncoder,
            started,
            [SynthesisDimension::new("tokens", tokens)],
        )
        .map_err(|error| input_error(error.to_string()))?;

        let started = Instant::now();
        let durations = match controls.durations {
            Some(durations) => {
                validate_duration_shape(durations.dims(), batch, tokens)?;
                durations
            }
            None => {
                let duration_log = self
                    .duration_predictor
                    .forward(encoded.clone(), mask.clone());
                ((duration_log.exp() - 1.0) * mask.clone() * controls.length_scale)
                    .clamp(1.0, self.max_duration as f64)
                    .round()
                    .reshape([batch, tokens])
            }
        };
        finish_backend_stage::<B>(
            &mut profiler,
            &device,
            SynthesisStage::DurationPrediction,
            started,
            [SynthesisDimension::new("tokens", tokens)],
        )
        .map_err(|error| input_error(error.to_string()))?;

        let pitch = apply_variance_path(
            "pitch",
            &mut encoded,
            &mask,
            controls.pitch,
            controls.pitch_scale,
            controls.pitch_shift,
            self.pitch_predictor.as_ref(),
            self.pitch_emb.as_ref(),
            batch,
            tokens,
        )?;
        // Coqui FastSpeech 2 predicts energy after pitch conditioning, so the
        // energy predictor sees the updated encoder state.
        let energy = apply_variance_path(
            "energy",
            &mut encoded,
            &mask,
            controls.energy,
            controls.energy_scale,
            controls.energy_shift,
            self.energy_predictor.as_ref(),
            self.energy_emb.as_ref(),
            batch,
            tokens,
        )?;

        let started = Instant::now();
        let (mut expanded, output_mask) =
            expand_by_durations(encoded, durations.clone(), self.max_output_frames)
                .map_err(|error| input_error(error.to_string()))?;
        let output_frames = expanded.dims()[2];
        if output_frames > MAX_POSITIONAL_FRAMES {
            return Err(input_error(format!(
                "duration controls requested {output_frames} frames, exceeding positional limit {MAX_POSITIONAL_FRAMES}"
            )));
        }
        finish_backend_stage::<B>(
            &mut profiler,
            &device,
            SynthesisStage::DurationExpansion,
            started,
            [
                SynthesisDimension::new("tokens", tokens),
                SynthesisDimension::new("frames", output_frames),
            ],
        )
        .map_err(|error| input_error(error.to_string()))?;
        if let Some(positional) = &self.pos_encoder {
            expanded = positional
                .forward(expanded, output_mask.clone())
                .map_err(|error| input_error(error.to_string()))?;
        }
        let started = Instant::now();
        let decoded = self
            .decoder
            .decoder
            .transformer_block
            .forward(expanded, output_mask.clone());
        let mel = self
            .decoder
            .decoder
            .postnet
            .forward(decoded)
            .mul(output_mask)
            .swap_dims(1, 2);
        finish_backend_stage::<B>(
            &mut profiler,
            &device,
            SynthesisStage::AcousticDecoder,
            started,
            [
                SynthesisDimension::new("frames", output_frames),
                SynthesisDimension::new("mel_bins", self.out_channels),
            ],
        )
        .map_err(|error| input_error(error.to_string()))?;
        Ok(FastSpeechOutput {
            mel,
            durations,
            pitch,
            energy,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_variance_path<B: Backend>(
    label: &str,
    encoded: &mut Tensor<B, 3>,
    mask: &Tensor<B, 3>,
    explicit: Option<Tensor<B, 3>>,
    scale: f64,
    shift: f64,
    predictor: Option<&DurationPredictor<B>>,
    embedding: Option<&Conv1d<B>>,
    batch: usize,
    tokens: usize,
) -> Result<Option<Tensor<B, 3>>, FastSpeechError> {
    let (Some(predictor), Some(embedding)) = (predictor, embedding) else {
        return Ok(None);
    };
    let predicted = predictor.forward(encoded.clone(), mask.clone());
    let values = match explicit {
        Some(values) => {
            if values.dims() != [batch, 1, tokens] {
                return Err(input_error(format!(
                    "explicit {label} has shape {:?}; expected [{batch}, 1, {tokens}]",
                    values.dims()
                )));
            }
            values
        }
        None => predicted,
    };
    let values = (values * scale + shift) * mask.clone();
    *encoded = encoded.clone() + embedding.forward(values.clone());
    Ok(Some(values))
}

fn validate_controls<B: Backend>(controls: &FastSpeechControls<B>) -> Result<(), FastSpeechError> {
    if !controls.length_scale.is_finite() || controls.length_scale <= 0.0 {
        return Err(input_error("length_scale must be finite and positive"));
    }
    for (label, scale, shift) in [
        ("pitch", controls.pitch_scale, controls.pitch_shift),
        ("energy", controls.energy_scale, controls.energy_shift),
    ] {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(input_error(format!(
                "{label}_scale must be finite and positive"
            )));
        }
        if !shift.is_finite() {
            return Err(input_error(format!("{label}_shift must be finite")));
        }
    }
    Ok(())
}

fn reject_unsupported_variance_controls(
    label: &str,
    supported: bool,
    scale: f64,
    shift: f64,
    explicit: bool,
) -> Result<(), FastSpeechError> {
    if !supported && (scale != 1.0 || shift != 0.0 || explicit) {
        return Err(input_error(format!(
            "{label} controls are not supported by this FastSpeech variant"
        )));
    }
    Ok(())
}

fn validate_duration_shape(
    actual: [usize; 2],
    batch: usize,
    tokens: usize,
) -> Result<(), FastSpeechError> {
    if actual != [batch, tokens] {
        return Err(input_error(format!(
            "explicit durations have shape {actual:?}; expected [{batch}, {tokens}]"
        )));
    }
    Ok(())
}

fn checkpoint_tensor(path: &str, _container: &str) -> bool {
    !path.ends_with(".num_batches_tracked")
}

fn reject_speaker_conditioning(root: &Value, args: &Value) -> Result<(), FastSpeechError> {
    let top_level_speakers = root
        .get("num_speakers")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let argument_speakers = args
        .get("num_speakers")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let speaker_embedding = root
        .get("use_speaker_embedding")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || args
            .get("use_speaker_embedding")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let d_vectors = root
        .get("use_d_vector_file")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || args
            .get("use_d_vector_file")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        || args
            .get("use_d_vector")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    if top_level_speakers > 1 || argument_speakers > 1 || speaker_embedding || d_vectors {
        return Err(config_error(
            "speaker-conditioned FastSpeech checkpoints require a conditioning adapter",
        ));
    }
    Ok(())
}

fn object_at<'a>(root: &'a Value, path: &[&str]) -> Result<&'a Value, FastSpeechError> {
    let value = path.iter().try_fold(root, |value, key| {
        value
            .get(*key)
            .ok_or_else(|| config_error(format!("missing {}", path.join("."))))
    })?;
    value
        .as_object()
        .map(|_| value)
        .ok_or_else(|| config_error(format!("{} must be an object", path.join("."))))
}

fn string_at<'a>(root: &'a Value, path: &[&str]) -> Result<&'a str, FastSpeechError> {
    let value = path.iter().try_fold(root, |value, key| {
        value
            .get(*key)
            .ok_or_else(|| config_error(format!("missing {}", path.join("."))))
    })?;
    value
        .as_str()
        .ok_or_else(|| config_error(format!("{} must be a string", path.join("."))))
}

fn bool_at(root: &Value, path: &[&str]) -> Result<bool, FastSpeechError> {
    let value = path.iter().try_fold(root, |value, key| {
        value
            .get(*key)
            .ok_or_else(|| config_error(format!("missing {}", path.join("."))))
    })?;
    value
        .as_bool()
        .ok_or_else(|| config_error(format!("{} must be a boolean", path.join("."))))
}

fn optional_bool_at(root: &Value, key: &str) -> Result<Option<bool>, FastSpeechError> {
    match root.get(key) {
        Some(value) => value
            .as_bool()
            .map(Some)
            .ok_or_else(|| config_error(format!("{key} must be a boolean"))),
        None => Ok(None),
    }
}

fn usize_at(root: &Value, path: &[&str]) -> Result<usize, FastSpeechError> {
    let value = path.iter().try_fold(root, |value, key| {
        value
            .get(*key)
            .ok_or_else(|| config_error(format!("missing {}", path.join("."))))
    })?;
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| config_error(format!("{} must be an unsigned integer", path.join("."))))
}

fn number_at(root: &Value, path: &[&str]) -> Result<f64, FastSpeechError> {
    let value = path.iter().try_fold(root, |value, key| {
        value
            .get(*key)
            .ok_or_else(|| config_error(format!("missing {}", path.join("."))))
    })?;
    value
        .as_f64()
        .ok_or_else(|| config_error(format!("{} must be numeric", path.join("."))))
}

fn published_vocabulary_size(root: &Value) -> Result<usize, FastSpeechError> {
    let characters = object_at(root, &["characters"])?;
    ["pad", "eos", "bos", "phonemes", "punctuations"]
        .iter()
        .try_fold(0usize, |total, field| {
            let value = characters
                .get(*field)
                .and_then(Value::as_str)
                .ok_or_else(|| config_error(format!("characters.{field} must be a string")))?;
            Ok(total + value.chars().count())
        })
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::TensorData;

    use super::*;

    type TestBackend = NdArray<f32>;

    fn tiny_config(variant: FastSpeechVariant) -> FastSpeechConfig {
        let variance = VariancePredictorConfig {
            hidden_channels: 4,
            kernel_size: 3,
            dropout: 0.1,
            embedding_kernel_size: 3,
        };
        FastSpeechConfig {
            variant,
            num_chars: 8,
            out_channels: 3,
            hidden_channels: 4,
            positional_encoding: true,
            length_scale: 1.0,
            encoder: FeedForwardTransformerConfig {
                hidden_channels_ffn: 8,
                num_heads: 1,
                num_layers: 1,
                dropout: 0.1,
            },
            decoder: FeedForwardTransformerConfig {
                hidden_channels_ffn: 8,
                num_heads: 1,
                num_layers: 1,
                dropout: 0.1,
            },
            duration_predictor_hidden_channels: 4,
            duration_predictor_kernel_size: 3,
            duration_predictor_dropout: 0.1,
            pitch: variant.supports_pitch().then(|| variance.clone()),
            energy: variant.supports_energy().then_some(variance),
            use_aligner: false,
            max_duration: 10,
            max_output_frames: 64,
        }
    }

    fn config_json(model: &str, use_pitch: bool, use_energy: bool) -> Value {
        serde_json::json!({
            "model": model,
            "base_model": "forward_tts",
            "model_args": {
                "num_chars": 130,
                "out_channels": 80,
                "hidden_channels": 384,
                "num_speakers": 1,
                "use_speaker_embedding": false,
                "use_d_vector_file": false,
                "use_pitch": use_pitch,
                "pitch_predictor_hidden_channels": 256,
                "pitch_predictor_kernel_size": 3,
                "pitch_predictor_dropout_p": 0.1,
                "pitch_embedding_kernel_size": 3,
                "use_energy": use_energy,
                "energy_predictor_hidden_channels": 256,
                "energy_predictor_kernel_size": 3,
                "energy_predictor_dropout_p": 0.1,
                "energy_embedding_kernel_size": 3,
                "duration_predictor_hidden_channels": 256,
                "duration_predictor_kernel_size": 3,
                "duration_predictor_dropout_p": 0.1,
                "positional_encoding": true,
                "length_scale": 1,
                "encoder_type": "fftransformer",
                "encoder_params": {
                    "hidden_channels_ffn": 1024,
                    "num_heads": 1,
                    "num_layers": 6,
                    "dropout_p": 0.1
                },
                "decoder_type": "fftransformer",
                "decoder_params": {
                    "hidden_channels_ffn": 1024,
                    "num_heads": 1,
                    "num_layers": 6,
                    "dropout_p": 0.1
                },
                "max_duration": 75,
                "use_aligner": true
            }
        })
    }

    #[test]
    fn parses_fastspeech2_as_forward_tts_with_pitch_and_energy() {
        let config = FastSpeechConfig::from_json_value(&config_json("fastspeech2", true, true))
            .expect("FastSpeech 2 config");
        assert_eq!(config, FastSpeechConfig::fastspeech2(130));
    }

    #[test]
    fn parses_original_fastspeech_without_variance_paths() {
        let config = FastSpeechConfig::from_json_value(&config_json("fastspeech", false, false))
            .expect("FastSpeech config");
        assert_eq!(config, FastSpeechConfig::fastspeech(130));
    }

    #[test]
    fn rejects_variant_and_variance_mismatches() {
        let error = FastSpeechConfig::from_json_value(&config_json("fastspeech2", true, false))
            .expect_err("missing energy must fail");
        assert!(error.to_string().contains("both pitch and energy"));

        let error = FastSpeechConfig::from_json_value(&config_json("fastspeech", true, false))
            .expect_err("FastSpeech pitch path must fail");
        assert!(error.to_string().contains("must disable"));
    }

    #[test]
    fn controls_are_discoverable_only_for_fastspeech2() {
        assert_eq!(
            FastSpeechVariant::FastSpeech.pitch_capabilities(),
            PitchCapabilities::default()
        );
        assert_eq!(
            FastSpeechVariant::FastSpeech.energy_capabilities(),
            EnergyCapabilities::default()
        );
        assert!(
            FastSpeechVariant::FastSpeech2
                .pitch_capabilities()
                .explicit_values
        );
        assert!(
            FastSpeechVariant::FastSpeech2
                .energy_capabilities()
                .explicit_values
        );
    }

    #[test]
    fn fastspeech2_explicit_variance_controls_determine_shapes() {
        let device = NdArrayDevice::Cpu;
        let model = tiny_config(FastSpeechVariant::FastSpeech2)
            .init::<TestBackend>(&device)
            .expect("model");
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3], [1, 3]),
            &device,
        );
        let durations = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 1.0], [1, 3]),
            &device,
        );
        let pitch = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![100.0_f32, 120.0, 110.0], [1, 1, 3]),
            &device,
        );
        let energy = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.5_f32, 0.8, 0.4], [1, 1, 3]),
            &device,
        );
        let output = model
            .inference_projected_with_controls(
                tokens,
                FastSpeechControls {
                    durations: Some(durations),
                    pitch: Some(pitch),
                    energy: Some(energy),
                    ..Default::default()
                },
                None,
            )
            .expect("inference");
        assert_eq!(output.durations.dims(), [1, 3]);
        assert_eq!(output.pitch.expect("pitch").dims(), [1, 1, 3]);
        assert_eq!(output.energy.expect("energy").dims(), [1, 1, 3]);
        assert_eq!(output.mel.dims(), [1, 4, 3]);
    }

    #[test]
    fn original_fastspeech_rejects_variance_controls() {
        let device = NdArrayDevice::Cpu;
        let model = tiny_config(FastSpeechVariant::FastSpeech)
            .init::<TestBackend>(&device)
            .expect("model");
        let tokens = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device);
        let error = model
            .inference_projected_with_controls(
                tokens,
                FastSpeechControls {
                    pitch_scale: 1.1,
                    ..Default::default()
                },
                None,
            )
            .expect_err("unsupported pitch control");
        assert!(error
            .to_string()
            .contains("pitch controls are not supported"));
    }
}
