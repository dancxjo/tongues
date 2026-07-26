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
    AcousticArtifact, AcousticModel, AcousticOutputContract, AudioFeatureConfig, EmbeddingContract,
    InferenceRuntime, LinguisticProjector, ModelInputContract, PhonemeVocabularyProjector,
    SpectrogramContract, SpectrogramLayout, SpeechModelCapabilities, SpeechModelFamily,
    SpeechSynthesisRequest,
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
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

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
}
