//! Native Burn inference for Coqui Align-TTS checkpoints.
//!
//! The inference graph follows Coqui TTS v0.22.0's `AlignTTS`: text embedding,
//! feed-forward transformer encoder, convolutional duration predictor, duration
//! expansion, positional encoding, transformer decoder, and mel projection.
//! The MDN alignment block and modulation layer are retained so training
//! checkpoints load completely, although inference does not execute them.
//!
//! Source provenance: `source-adapted`, from MPL-2.0 Coqui TTS revision
//! `dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e`, principally
//! `TTS/tts/models/align_tts.py` and `TTS/tts/layers/align_tts/mdn.py`.

use std::fmt;
use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, PaddingConfig1d};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::burn_fast_pitch::{
    FastPitchDecoder, FastPitchDecoderContainer, FastPitchEncoder, FeedForwardTransformerBlock,
    FeedForwardTransformerConfig,
};
use crate::burn_speedy_speech::{expand_by_durations, Conv1dBn, PositionalEncoding};
use crate::burn_variance_acoustic::tensor_to_artifact;
use crate::{
    AcousticArtifact, AcousticModel, AcousticOutputContract, AcousticTrainingPhase,
    AudioFeatureConfig, BurnAcousticTrainingBatch, BurnAcousticTrainingHooks,
    BurnAcousticTrainingOutput, EmbeddingContract, InferenceRuntime, LinguisticProjector,
    ModelInputContract, PhonemeVocabularyProjector, SpectrogramContract, SpectrogramLayout,
    SpeechModelCapabilities, SpeechModelFamily, SpeechSynthesisRequest,
};

const MAX_DURATION: usize = 75;
const MAX_OUTPUT_FRAMES: usize = 20_000;
const MAX_POSITIONAL_FRAMES: usize = 5_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlignTtsError {
    InvalidConfig(String),
    InvalidInput(String),
    Checkpoint(String),
}

impl fmt::Display for AlignTtsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid Align-TTS config: {message}"),
            Self::InvalidInput(message) => write!(f, "invalid Align-TTS input: {message}"),
            Self::Checkpoint(message) => {
                write!(f, "unable to load Align-TTS checkpoint: {message}")
            }
        }
    }
}

impl std::error::Error for AlignTtsError {}

fn config_error(message: impl Into<String>) -> AlignTtsError {
    AlignTtsError::InvalidConfig(message.into())
}

fn input_error(message: impl Into<String>) -> AlignTtsError {
    AlignTtsError::InvalidInput(message.into())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignTtsTrainingConfig {
    pub phase_start_steps: Option<Vec<u64>>,
    pub ssim_alpha: f64,
    pub spec_loss_alpha: f64,
    pub duration_loss_alpha: f64,
    pub mdn_alpha: f64,
}

impl AlignTtsTrainingConfig {
    fn from_json_value(root: &Value) -> Result<Self, AlignTtsError> {
        let number = |name: &str, default: f64| {
            root.get(name)
                .map(|value| {
                    value
                        .as_f64()
                        .ok_or_else(|| config_error(format!("{name} must be numeric")))
                })
                .unwrap_or(Ok(default))
        };
        let phase_start_steps = match root.get("phase_start_steps") {
            None | Some(Value::Null) => None,
            Some(Value::Array(values)) => Some(
                values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        value.as_u64().ok_or_else(|| {
                            config_error(format!(
                                "phase_start_steps[{index}] must be a non-negative integer"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Some(_) => {
                return Err(config_error(
                    "phase_start_steps must be null or an array of four steps",
                ))
            }
        };
        let config = Self {
            phase_start_steps,
            ssim_alpha: number("ssim_alpha", 1.0)?,
            spec_loss_alpha: number("spec_loss_alpha", 1.0)?,
            duration_loss_alpha: number("dur_loss_alpha", 1.0)?,
            mdn_alpha: number("mdn_alpha", 1.0)?,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), AlignTtsError> {
        if let Some(steps) = &self.phase_start_steps {
            if steps.len() != 4 || steps.windows(2).any(|window| window[0] > window[1]) {
                return Err(config_error(
                    "phase_start_steps must contain four non-decreasing steps",
                ));
            }
        }
        for (name, value) in [
            ("ssim_alpha", self.ssim_alpha),
            ("spec_loss_alpha", self.spec_loss_alpha),
            ("dur_loss_alpha", self.duration_loss_alpha),
            ("mdn_alpha", self.mdn_alpha),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(config_error(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }

    pub fn phase(&self, global_step: u64) -> AcousticTrainingPhase {
        let Some(steps) = &self.phase_start_steps else {
            return AcousticTrainingPhase::Joint;
        };
        match steps.iter().filter(|step| **step < global_step).count() {
            0 => AcousticTrainingPhase::Alignment,
            1 => AcousticTrainingPhase::Decoder,
            2 => AcousticTrainingPhase::Acoustic,
            3 => AcousticTrainingPhase::DurationPredictor,
            _ => AcousticTrainingPhase::Joint,
        }
    }
}

impl Default for AlignTtsTrainingConfig {
    fn default() -> Self {
        Self {
            phase_start_steps: None,
            ssim_alpha: 1.0,
            spec_loss_alpha: 1.0,
            duration_loss_alpha: 1.0,
            mdn_alpha: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignTtsConfig {
    pub num_chars: usize,
    pub out_channels: usize,
    pub hidden_channels: usize,
    pub hidden_channels_dp: usize,
    pub encoder: FeedForwardTransformerConfig,
    pub decoder: FeedForwardTransformerConfig,
    pub length_scale: f64,
    pub max_duration: usize,
    pub max_output_frames: usize,
    pub training: AlignTtsTrainingConfig,
}

impl AlignTtsConfig {
    pub fn from_json_value(root: &Value) -> Result<Self, AlignTtsError> {
        let model = root
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if model != "align_tts" && model != "align-tts" {
            return Err(config_error(format!(
                "expected align_tts model, got {model:?}"
            )));
        }
        let args = root
            .get("model_args")
            .and_then(Value::as_object)
            .ok_or_else(|| config_error("model_args must be an object"))?;
        for field in ["encoder_type", "decoder_type"] {
            if args.get(field).and_then(Value::as_str) != Some("fftransformer") {
                return Err(config_error(format!("{field} must be \"fftransformer\"")));
            }
        }
        for field in ["use_speaker_embedding", "use_d_vector_file"] {
            if args.get(field).and_then(Value::as_bool).unwrap_or(false)
                || root.get(field).and_then(Value::as_bool).unwrap_or(false)
            {
                return Err(config_error(
                    "speaker-conditioned Align-TTS checkpoints are not yet supported",
                ));
            }
        }
        let usize_field = |name: &str| {
            args.get(name)
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    config_error(format!("model_args.{name} must be a positive integer"))
                })
        };
        let num_chars = match args.get("num_chars").and_then(Value::as_u64) {
            Some(value) => usize::try_from(value)
                .map_err(|_| config_error("model_args.num_chars does not fit usize"))?,
            None => published_vocabulary_size(root)?,
        };
        let config = Self {
            num_chars,
            out_channels: usize_field("out_channels")?,
            hidden_channels: usize_field("hidden_channels")?,
            hidden_channels_dp: usize_field("hidden_channels_dp")?,
            encoder: FeedForwardTransformerConfig::from_value(
                args.get("encoder_params")
                    .ok_or_else(|| config_error("model_args.encoder_params is required"))?,
                "encoder",
            )
            .map_err(|error| config_error(error.to_string()))?,
            decoder: FeedForwardTransformerConfig::from_value(
                args.get("decoder_params")
                    .ok_or_else(|| config_error("model_args.decoder_params is required"))?,
                "decoder",
            )
            .map_err(|error| config_error(error.to_string()))?,
            length_scale: args
                .get("length_scale")
                .and_then(Value::as_f64)
                .ok_or_else(|| config_error("model_args.length_scale must be numeric"))?,
            max_duration: MAX_DURATION,
            max_output_frames: MAX_OUTPUT_FRAMES,
            training: AlignTtsTrainingConfig::from_json_value(root)?,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), AlignTtsError> {
        if self.num_chars == 0
            || self.out_channels == 0
            || self.hidden_channels == 0
            || self.hidden_channels_dp == 0
        {
            return Err(config_error("model dimensions must be positive"));
        }
        if self.hidden_channels_dp != self.hidden_channels {
            return Err(config_error(
                "Coqui Align-TTS requires hidden_channels_dp to equal hidden_channels",
            ));
        }
        if !self.hidden_channels.is_multiple_of(self.encoder.num_heads)
            || !self.hidden_channels.is_multiple_of(self.decoder.num_heads)
            || !self.hidden_channels.is_multiple_of(2)
        {
            return Err(config_error(
                "hidden_channels must be even and divisible by all attention head counts",
            ));
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(config_error("length_scale must be finite and positive"));
        }
        Ok(())
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<AlignTts<B>, AlignTtsError> {
        self.validate()?;
        Ok(AlignTts {
            emb: EmbeddingConfig::new(self.num_chars, self.hidden_channels).init(device),
            encoder: FastPitchEncoder {
                encoder: FeedForwardTransformerBlock::init(
                    self.hidden_channels,
                    &self.encoder,
                    device,
                ),
            },
            pos_encoder: PositionalEncoding::init(self.hidden_channels, device),
            decoder: FastPitchDecoderContainer {
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
            },
            duration_predictor: AlignDurationPredictor::init(
                self.hidden_channels,
                self.hidden_channels_dp,
                device,
            ),
            mod_layer: Conv1dConfig::new(self.hidden_channels, self.hidden_channels, 1)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            mdn_block: MdnBlock::init(self.hidden_channels, self.out_channels * 2, device),
            length_scale: self.length_scale,
            num_chars: self.num_chars,
            out_channels: self.out_channels,
            max_duration: self.max_duration,
            max_output_frames: self.max_output_frames,
            training: self.training.clone(),
        })
    }
}

fn published_vocabulary_size(root: &Value) -> Result<usize, AlignTtsError> {
    let characters = root
        .get("characters")
        .and_then(Value::as_object)
        .ok_or_else(|| config_error("characters must be an object when num_chars is omitted"))?;
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

#[derive(Module, Debug)]
pub struct AlignDurationPredictor<B: Backend> {
    pub layers: Vec<Conv1dBn<B>>,
    pub proj: Conv1d<B>,
}

impl<B: Backend> AlignDurationPredictor<B> {
    fn init(channels_in: usize, hidden: usize, device: &B::Device) -> Self {
        Self {
            layers: vec![
                Conv1dBn::init(channels_in, hidden, 4, 1, device),
                Conv1dBn::init(hidden, hidden, 3, 1, device),
                Conv1dBn::init(hidden, hidden, 1, 1, device),
            ],
            proj: Conv1dConfig::new(hidden, 1, 1)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
        }
    }

    fn forward(&self, mut input: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in &self.layers {
            input = layer.forward(input) * mask.clone();
        }
        self.proj.forward(input) * mask
    }
}

#[derive(Module, Debug)]
pub struct MdnBlock<B: Backend> {
    pub conv1: Conv1d<B>,
    pub norm: LayerNorm<B>,
    pub conv2: Conv1d<B>,
}

impl<B: Backend> MdnBlock<B> {
    fn init(channels_in: usize, channels_out: usize, device: &B::Device) -> Self {
        Self {
            conv1: Conv1dConfig::new(channels_in, channels_in, 1).init(device),
            norm: LayerNormConfig::new(channels_in).init(device),
            conv2: Conv1dConfig::new(channels_in, channels_out, 1).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let output = self.conv1.forward(input).swap_dims(1, 2);
        let output = relu(self.norm.forward(output)).swap_dims(1, 2);
        let output = self.conv2.forward(output);
        let [batch, output_channels, tokens] = output.dims();
        let channels = output_channels / 2;
        (
            output.clone().slice([0..batch, 0..channels, 0..tokens]),
            output.slice([0..batch, channels..channels * 2, 0..tokens]),
        )
    }
}

#[derive(Debug)]
pub struct AlignTtsControls<B: Backend> {
    pub length_scale: f64,
    pub durations: Option<Tensor<B, 2>>,
}

impl<B: Backend> Default for AlignTtsControls<B> {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            durations: None,
        }
    }
}

#[derive(Debug)]
pub struct AlignTtsOutput<B: Backend> {
    pub mel: Tensor<B, 3>,
    pub durations: Tensor<B, 2>,
    pub alignment: Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct AlignTts<B: Backend> {
    pub emb: Embedding<B>,
    pub encoder: FastPitchEncoder<B>,
    pub pos_encoder: PositionalEncoding<B>,
    pub decoder: FastPitchDecoderContainer<B>,
    pub duration_predictor: AlignDurationPredictor<B>,
    pub mod_layer: Conv1d<B>,
    pub mdn_block: MdnBlock<B>,
    length_scale: f64,
    num_chars: usize,
    out_channels: usize,
    max_duration: usize,
    max_output_frames: usize,
    training: AlignTtsTrainingConfig,
}

impl<B: Backend> AlignTts<B> {
    pub fn load_checkpoint(mut self, path: impl AsRef<Path>) -> Result<Self, AlignTtsError> {
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut self,
            path.as_ref(),
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                predicate: Some(|path, _| !path.ends_with(".num_batches_tracked")),
                key_remappings: vec![
                    (
                        r"^duration_predictor\.layers\.3\.".into(),
                        "duration_predictor.proj.".into(),
                    ),
                    (r"(\.norm_[12])\.weight$".into(), "$1.gamma".into()),
                    (r"(\.norm_[12])\.bias$".into(), "$1.beta".into()),
                    (r"(\.norm[12])\.weight$".into(), "$1.gamma".into()),
                    (r"(\.norm[12])\.bias$".into(), "$1.beta".into()),
                    (r"(\.norm)\.weight$".into(), "$1.gamma".into()),
                    (r"(\.norm)\.bias$".into(), "$1.beta".into()),
                ],
                skip_enum_variants: true,
                ..Default::default()
            },
        )
        .map_err(|error| AlignTtsError::Checkpoint(format!("{error:#}")))?;
        let unused = result
            .unused
            .iter()
            .filter(|path| !path.ends_with(".num_batches_tracked"))
            .cloned()
            .collect::<Vec<_>>();
        if !result.missing.is_empty() || !result.errors.is_empty() || !unused.is_empty() {
            return Err(AlignTtsError::Checkpoint(format!(
                "checkpoint does not exactly match the native model: {} missing, {} load errors, unexpected tensors: {}",
                result.missing.len(),
                result.errors.len(),
                unused.join(", ")
            )));
        }
        let device = self.pos_encoder.pe.val().device();
        self.pos_encoder = PositionalEncoding::init(self.pos_encoder.channels, &device);
        Ok(self)
    }

    pub fn inference(
        &self,
        token_ids: Tensor<B, 2, Int>,
    ) -> Result<AlignTtsOutput<B>, AlignTtsError> {
        self.inference_with_controls(
            token_ids,
            AlignTtsControls {
                length_scale: self.length_scale,
                durations: None,
            },
            false,
        )
    }

    pub fn training_config(&self) -> &AlignTtsTrainingConfig {
        &self.training
    }

    pub(crate) fn inference_projected_with_controls(
        &self,
        token_ids: Tensor<B, 2, Int>,
        controls: AlignTtsControls<B>,
    ) -> Result<AlignTtsOutput<B>, AlignTtsError> {
        self.inference_with_controls(token_ids, controls, true)
    }

    fn inference_with_controls(
        &self,
        token_ids: Tensor<B, 2, Int>,
        controls: AlignTtsControls<B>,
        ids_validated: bool,
    ) -> Result<AlignTtsOutput<B>, AlignTtsError> {
        if !controls.length_scale.is_finite() || controls.length_scale <= 0.0 {
            return Err(input_error("length_scale must be finite and positive"));
        }
        let [batch, tokens] = token_ids.dims();
        if batch == 0 || tokens == 0 {
            return Err(input_error("token_ids must be non-empty"));
        }
        if !ids_validated {
            let highest = token_ids.clone().max().into_scalar().elem::<i64>();
            if highest < 0 || highest as usize >= self.num_chars {
                return Err(input_error(format!(
                    "token ID {highest} is outside the vocabulary"
                )));
            }
        }
        let device = token_ids.device();
        let mask = Tensor::<B, 3>::ones([batch, 1, tokens], &device);
        let encoded = self
            .encoder
            .encoder
            .forward(self.emb.forward(token_ids).swap_dims(1, 2), mask.clone());
        let durations = match controls.durations {
            Some(value) => {
                if value.dims() != [batch, tokens] {
                    return Err(input_error(
                        "explicit durations must have shape [batch, tokens]",
                    ));
                }
                value
            }
            None => ((self
                .duration_predictor
                .forward(encoded.clone(), mask.clone())
                .exp()
                - 1.0)
                * mask.clone()
                * controls.length_scale)
                .clamp(1.0, self.max_duration as f64)
                .round()
                .reshape([batch, tokens]),
        };
        let (expanded, output_mask) =
            expand_by_durations(encoded, durations.clone(), self.max_output_frames)
                .map_err(|error| input_error(error.to_string()))?;
        let frames = expanded.dims()[2];
        if frames > MAX_POSITIONAL_FRAMES {
            return Err(input_error(format!(
                "predicted {frames} frames, exceeding the positional limit"
            )));
        }
        let expanded = self
            .pos_encoder
            .forward(expanded, output_mask.clone())
            .map_err(|error| input_error(error.to_string()))?;
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
        let alignment = durations_to_alignment(durations.clone(), frames, &device)?;
        Ok(AlignTtsOutput {
            mel,
            durations,
            alignment,
        })
    }
}

impl<B: Backend> BurnAcousticTrainingHooks<B> for AlignTts<B> {
    fn training_phase(&self, global_step: u64) -> AcousticTrainingPhase {
        self.training.phase(global_step)
    }

    fn training_forward(
        &self,
        batch: BurnAcousticTrainingBatch<B>,
        global_step: u64,
    ) -> Result<BurnAcousticTrainingOutput<B>> {
        let [batch_size, tokens] = batch.token_ids.dims();
        let [mel_batch, frames, mel_bins] = batch.target_mel.dims();
        ensure!(
            batch_size > 0 && tokens > 0 && frames > 0,
            "Align-TTS training batch dimensions must be non-empty"
        );
        ensure!(
            mel_batch == batch_size && mel_bins == self.out_channels,
            "Align-TTS target mel shape does not match batch/model dimensions"
        );
        ensure!(
            batch.token_lengths.len() == batch_size && batch.mel_lengths.len() == batch_size,
            "Align-TTS training length vectors must match batch size"
        );
        ensure!(
            batch
                .token_lengths
                .iter()
                .all(|length| *length > 0 && *length <= tokens)
                && batch
                    .mel_lengths
                    .iter()
                    .all(|length| *length > 0 && *length <= frames),
            "Align-TTS training lengths are outside their padded dimensions"
        );
        ensure!(
            batch
                .token_lengths
                .iter()
                .zip(&batch.mel_lengths)
                .all(|(token_length, mel_length)| token_length <= mel_length),
            "Align-TTS monotonic alignment requires at least one frame per token"
        );
        let highest = batch.token_ids.clone().max().into_scalar().elem::<i64>();
        ensure!(
            highest >= 0 && (highest as usize) < self.num_chars,
            "Align-TTS training token ID {highest} is outside the vocabulary"
        );

        let device = batch.token_ids.device();
        let token_mask = length_mask::<B>(&batch.token_lengths, tokens, &device);
        let frame_mask = length_mask::<B>(&batch.mel_lengths, frames, &device);
        let encoded = self.encoder.encoder.forward(
            self.emb.forward(batch.token_ids).swap_dims(1, 2),
            token_mask.clone(),
        );
        let (mean, log_scale) = self.mdn_block.forward(encoded.clone());
        let log_prob = alignment_log_prob(mean, log_scale, batch.target_mel);
        let alignment = maximum_monotonic_alignment(
            log_prob.clone(),
            &batch.token_lengths,
            &batch.mel_lengths,
            &device,
        )?;
        let durations = alignment.clone().sum_dim(1).reshape([batch_size, tokens]);
        let aligned_duration_log = (durations.clone() + 1.0).log();
        let phase = self.training_phase(global_step);

        let predicted_duration_log = matches!(
            phase,
            AcousticTrainingPhase::DurationPredictor | AcousticTrainingPhase::Joint
        )
        .then(|| {
            self.duration_predictor
                .forward(encoded.clone().detach(), token_mask.clone())
                .reshape([batch_size, tokens])
        });
        let predicted_mel = if phase == AcousticTrainingPhase::Alignment {
            None
        } else {
            let decoder_input = if phase == AcousticTrainingPhase::Decoder {
                encoded.detach()
            } else {
                encoded
            };
            let expanded = alignment
                .clone()
                .matmul(decoder_input.swap_dims(1, 2))
                .swap_dims(1, 2);
            let expanded = self
                .pos_encoder
                .forward(expanded, frame_mask.clone())
                .map_err(anyhow::Error::new)?;
            let decoded = self
                .decoder
                .decoder
                .transformer_block
                .forward(expanded, frame_mask.clone());
            Some(
                self.decoder
                    .decoder
                    .postnet
                    .forward(decoded)
                    .mul(frame_mask)
                    .swap_dims(1, 2),
            )
        };
        Ok(BurnAcousticTrainingOutput {
            phase,
            predicted_mel,
            alignment,
            predicted_duration_log,
            aligned_duration_log,
            alignment_log_prob: Some(log_prob),
        })
    }
}

fn length_mask<B: Backend>(lengths: &[usize], padded: usize, device: &B::Device) -> Tensor<B, 3> {
    let values = lengths
        .iter()
        .flat_map(|length| (0..padded).map(move |index| f32::from(index < *length)))
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, [lengths.len(), 1, padded]), device)
}

fn alignment_log_prob<B: Backend>(
    mean: Tensor<B, 3>,
    log_scale: Tensor<B, 3>,
    target_mel: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [batch, bins, tokens] = mean.dims();
    let frames = target_mel.dims()[1];
    let mean = mean.swap_dims(1, 2).reshape([batch, tokens, 1, bins]);
    let log_scale = log_scale.swap_dims(1, 2).reshape([batch, tokens, 1, bins]);
    let target = target_mel.reshape([batch, 1, frames, bins]);
    let squared_error: Tensor<B, 4> = (target - mean).square() / (log_scale.clone() * 2.0).exp();
    let exponential: Tensor<B, 4> = squared_error.mean_dim(3) * -0.5;
    let log_scale_penalty: Tensor<B, 4> = log_scale.mean_dim(3) * 0.5;
    (exponential - log_scale_penalty).reshape([batch, tokens, frames])
}

fn maximum_monotonic_alignment<B: Backend>(
    log_prob: Tensor<B, 3>,
    token_lengths: &[usize],
    mel_lengths: &[usize],
    device: &B::Device,
) -> Result<Tensor<B, 3>> {
    let [batch, padded_tokens, padded_frames] = log_prob.dims();
    let scores = log_prob
        .into_data()
        .to_vec::<f32>()
        .context("Align-TTS alignment likelihoods are not f32")?;
    let mut output = vec![0.0f32; batch * padded_frames * padded_tokens];
    for batch_index in 0..batch {
        let tokens = token_lengths[batch_index];
        let frames = mel_lengths[batch_index];
        let mut values = vec![f32::NEG_INFINITY; tokens * frames];
        let score = |token: usize, frame: usize| {
            scores[(batch_index * padded_tokens + token) * padded_frames + frame]
        };
        values[0] = score(0, 0);
        for frame in 1..frames {
            let first_token = tokens.saturating_sub(frames - frame);
            let last_token = frame.min(tokens - 1);
            for token in first_token..=last_token {
                let stay = values[token * frames + frame - 1];
                let advance = if token > 0 {
                    values[(token - 1) * frames + frame - 1]
                } else {
                    f32::NEG_INFINITY
                };
                values[token * frames + frame] = stay.max(advance) + score(token, frame);
            }
        }
        let mut token = tokens - 1;
        for frame in (0..frames).rev() {
            output[(batch_index * padded_frames + frame) * padded_tokens + token] = 1.0;
            if frame > 0
                && token > 0
                && values[(token - 1) * frames + frame - 1] > values[token * frames + frame - 1]
            {
                token -= 1;
            }
        }
    }
    Ok(Tensor::from_data(
        TensorData::new(output, [batch, padded_frames, padded_tokens]),
        device,
    ))
}

fn durations_to_alignment<B: Backend>(
    durations: Tensor<B, 2>,
    frames: usize,
    device: &B::Device,
) -> Result<Tensor<B, 3>, AlignTtsError> {
    let [batch, tokens] = durations.dims();
    let values = durations
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| input_error(error.to_string()))?;
    let mut alignment = vec![0.0f32; batch * frames * tokens];
    for batch_index in 0..batch {
        let mut frame = 0;
        for token in 0..tokens {
            let duration = values[batch_index * tokens + token].max(0.0) as usize;
            for _ in 0..duration {
                if frame < frames {
                    alignment[(batch_index * frames + frame) * tokens + token] = 1.0;
                    frame += 1;
                }
            }
        }
    }
    Ok(Tensor::from_data(
        TensorData::new(alignment, [batch, frames, tokens]),
        device,
    ))
}

pub struct BurnAlignTtsAcoustic<B: Backend> {
    model: AlignTts<B>,
    projector: PhonemeVocabularyProjector,
    output_contract: SpectrogramContract,
    device: B::Device,
}

impl<B: Backend> BurnAlignTtsAcoustic<B> {
    pub fn load(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
    ) -> Result<Self> {
        let source = fs::read_to_string(config_path.as_ref())
            .with_context(|| format!("failed to read {}", config_path.as_ref().display()))?;
        let root: Value = json5::from_str(&source).context("invalid Align-TTS config")?;
        let config = AlignTtsConfig::from_json_value(&root).map_err(anyhow::Error::new)?;
        let projector = PhonemeVocabularyProjector::from_json5_str(&source)?;
        ensure!(
            projector.vocabulary().len() == config.num_chars,
            "symbol count does not match num_chars"
        );
        let output_contract = AudioFeatureConfig::from_json5_str(&source)?.mel_contract()?;
        ensure!(
            output_contract.layout == SpectrogramLayout::FramesByBins,
            "Align-TTS requires frame-major spectrograms"
        );
        ensure!(
            output_contract.bins == config.out_channels,
            "mel bin count does not match out_channels"
        );
        let model = config
            .init::<B>(&device)
            .map_err(anyhow::Error::new)?
            .load_checkpoint(checkpoint_path)
            .map_err(anyhow::Error::new)?;
        Ok(Self {
            model,
            projector,
            output_contract,
            device,
        })
    }

    pub fn model(&self) -> &AlignTts<B> {
        &self.model
    }

    pub fn synthesize_tensor(&self, request: &SpeechSynthesisRequest) -> Result<Tensor<B, 3>> {
        ensure!(
            request.plan.speaker.is_none() && request.options.speaker_id.is_none(),
            "this Align-TTS backend is single-speaker"
        );
        ensure!(
            request.plan.speaker_reference.is_none(),
            "Align-TTS does not accept reference audio"
        );
        let projected = self.projector.project(&request.plan)?;
        let tokens = projected.ids.len();
        let token_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(projected.ids, [1, tokens]),
            &self.device,
        );
        let durations = request
            .options
            .durations
            .as_ref()
            .map(|values| {
                ensure!(
                    values.len() == tokens,
                    "explicit duration count must match token count"
                );
                Ok(Tensor::<B, 2>::from_data(
                    TensorData::new(values.clone(), [1, tokens]),
                    &self.device,
                ))
            })
            .transpose()?;
        Ok(self
            .model
            .inference_projected_with_controls(
                token_ids,
                AlignTtsControls {
                    length_scale: request.options.length_scale.map(f64::from).unwrap_or(1.0),
                    durations,
                },
            )
            .map_err(anyhow::Error::new)?
            .mel)
    }
}

impl<B: Backend> AcousticModel for BurnAlignTtsAcoustic<B> {
    fn runtime(&self) -> InferenceRuntime {
        InferenceRuntime::Burn
    }
    fn capabilities(&self) -> SpeechModelCapabilities {
        SpeechModelCapabilities {
            family: SpeechModelFamily::AcousticModel,
            supports_named_speakers: false,
            supports_languages: false,
            supports_reference_audio: false,
            supports_voice_conversion: false,
            integrated_vocoder: false,
        }
    }
    fn input_contract(&self) -> &ModelInputContract {
        self.projector.contract()
    }
    fn conditioning_contracts(&self) -> &[EmbeddingContract] {
        &[]
    }
    fn output_contract(&self) -> AcousticOutputContract {
        AcousticOutputContract::Spectrogram(self.output_contract.clone())
    }
    fn synthesize(&mut self, request: &SpeechSynthesisRequest) -> Result<AcousticArtifact> {
        tensor_to_artifact(self.synthesize_tensor(request)?, &self.output_contract)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/speech/align-tts-mpl-fixture")
            .join(name)
    }

    fn config() -> AlignTtsConfig {
        AlignTtsConfig {
            num_chars: 8,
            out_channels: 4,
            hidden_channels: 8,
            hidden_channels_dp: 8,
            encoder: FeedForwardTransformerConfig {
                hidden_channels_ffn: 16,
                num_heads: 2,
                num_layers: 1,
                dropout: 0.1,
            },
            decoder: FeedForwardTransformerConfig {
                hidden_channels_ffn: 16,
                num_heads: 2,
                num_layers: 1,
                dropout: 0.1,
            },
            length_scale: 1.0,
            max_duration: 10,
            max_output_frames: 100,
            training: AlignTtsTrainingConfig::default(),
        }
    }

    #[test]
    fn explicit_durations_control_alignment_and_mel_length() {
        let device = NdArrayDevice::Cpu;
        let model = config().init::<NdArray>(&device).expect("model");
        let ids =
            Tensor::<NdArray, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);
        let durations =
            Tensor::<NdArray, 2>::from_data(TensorData::new(vec![1.0, 2.0, 1.0], [1, 3]), &device);
        let output = model
            .inference_projected_with_controls(
                ids,
                AlignTtsControls {
                    length_scale: 1.0,
                    durations: Some(durations),
                },
            )
            .expect("inference");
        assert_eq!(output.mel.dims(), [1, 4, 4]);
        assert_eq!(output.alignment.dims(), [1, 4, 3]);
        assert_eq!(
            output.alignment.into_data().to_vec::<f32>().unwrap(),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn parses_upstream_defaults() {
        let root: Value = serde_json::json!({
            "model": "align_tts",
            "model_args": {
                "num_chars": 100, "out_channels": 80, "hidden_channels": 256,
                "hidden_channels_dp": 256, "encoder_type": "fftransformer",
                "encoder_params": {"hidden_channels_ffn": 1024, "num_heads": 2, "num_layers": 6, "dropout_p": 0.1},
                "decoder_type": "fftransformer",
                "decoder_params": {"hidden_channels_ffn": 1024, "num_heads": 2, "num_layers": 6, "dropout_p": 0.1},
                "length_scale": 1.0
            }
        });
        assert_eq!(
            AlignTtsConfig::from_json_value(&root)
                .unwrap()
                .hidden_channels_dp,
            256
        );
    }

    #[test]
    fn licensed_upstream_layout_matches_duration_alignment_and_mel_fixture_on_cpu() {
        let config_source = fs::read_to_string(fixture_path("config.json")).expect("config");
        let root: Value = json5::from_str(&config_source).expect("JSON5 config");
        let config = AlignTtsConfig::from_json_value(&root).expect("Align-TTS config");
        let device = NdArrayDevice::Cpu;
        let model = config
            .init::<TestBackend>(&device)
            .expect("model")
            .load_checkpoint(fixture_path("model_file.pth"))
            .expect("MPL fixture checkpoint");
        let reference: Value = serde_json::from_str(include_str!(
            "../../../fixtures/speech/align-tts-mpl-fixture/reference.json"
        ))
        .expect("reference");
        let ids = reference["token_ids"]
            .as_array()
            .expect("token ids")
            .iter()
            .map(|value| value.as_i64().expect("integer token"))
            .collect::<Vec<_>>();
        let output = model
            .inference(
                Tensor::<TestBackend, 1, Int>::from_ints(ids.as_slice(), &device)
                    .reshape([1, ids.len()]),
            )
            .expect("CPU inference");
        let expected_durations = reference["durations"]
            .as_array()
            .expect("durations")
            .iter()
            .map(|value| value.as_f64().expect("duration") as f32)
            .collect::<Vec<_>>();
        assert_eq!(
            output.durations.into_data().to_vec::<f32>().unwrap(),
            expected_durations
        );
        assert_eq!(
            output.alignment.dims(),
            [
                reference["alignment_shape"][0].as_u64().unwrap() as usize,
                reference["alignment_shape"][1].as_u64().unwrap() as usize,
                reference["alignment_shape"][2].as_u64().unwrap() as usize,
            ]
        );
        let alignment = output.alignment.into_data().to_vec::<f32>().unwrap();
        assert_eq!(
            alignment.iter().filter(|value| **value == 1.0).count(),
            ids.len()
        );
        assert!(alignment.iter().all(|value| matches!(*value, 0.0 | 1.0)));

        let mel_shape = [
            reference["mel_shape"][0].as_u64().unwrap() as usize,
            reference["mel_shape"][1].as_u64().unwrap() as usize,
            reference["mel_shape"][2].as_u64().unwrap() as usize,
        ];
        assert_eq!(output.mel.dims(), mel_shape);
        let mel = output.mel.into_data().to_vec::<f32>().unwrap();
        for probe in reference["mel_probes"].as_array().expect("mel probes") {
            let index = probe[0].as_u64().unwrap() as usize;
            let expected = probe[1].as_f64().unwrap() as f32;
            assert!(
                (mel[index] - expected).abs() <= 3e-4,
                "mel parity mismatch at {index}: actual={}, expected={expected}",
                mel[index]
            );
        }
    }

    #[test]
    fn model_neutral_training_hooks_expose_all_phases_and_joint_outputs() {
        let source = fs::read_to_string(fixture_path("config.json")).expect("config");
        let root: Value = json5::from_str(&source).expect("JSON5 config");
        let device = NdArrayDevice::Cpu;
        let model = AlignTtsConfig::from_json_value(&root)
            .expect("config")
            .init::<TestBackend>(&device)
            .expect("model")
            .load_checkpoint(fixture_path("model_file.pth"))
            .expect("checkpoint");
        assert_eq!(model.training_phase(0), AcousticTrainingPhase::Alignment);
        assert_eq!(model.training_phase(11), AcousticTrainingPhase::Decoder);
        assert_eq!(model.training_phase(21), AcousticTrainingPhase::Acoustic);
        assert_eq!(
            model.training_phase(31),
            AcousticTrainingPhase::DurationPredictor
        );
        assert_eq!(model.training_phase(41), AcousticTrainingPhase::Joint);

        let output = model
            .training_forward(
                BurnAcousticTrainingBatch {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                        [[15, 110, 44, 112]],
                        &device,
                    ),
                    token_lengths: vec![4],
                    target_mel: Tensor::<TestBackend, 3>::zeros([1, 6, 80], &device),
                    mel_lengths: vec![6],
                },
                41,
            )
            .expect("joint training hook");
        assert_eq!(output.phase, AcousticTrainingPhase::Joint);
        assert_eq!(output.alignment.dims(), [1, 6, 4]);
        assert_eq!(output.predicted_mel.expect("mel").dims(), [1, 6, 80]);
        assert_eq!(
            output.predicted_duration_log.expect("durations").dims(),
            [1, 4]
        );
        assert_eq!(output.aligned_duration_log.dims(), [1, 4]);
        assert_eq!(
            output.alignment_log_prob.expect("MDN likelihood").dims(),
            [1, 4, 6]
        );
    }
}
