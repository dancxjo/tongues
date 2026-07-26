//! Model-neutral acoustic adapters for FastSpeech-family and DelightfulTTS.

use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use serde_json::Value;

use crate::{
    AcousticArtifact, AcousticModel, AcousticOutputContract, AudioFeatureConfig, DelightfulTts,
    DelightfulTtsConfig, DelightfulTtsControls, EmbeddingContract, EnergyCapabilities, FastSpeech,
    FastSpeechConfig, FastSpeechControls, FastSpeechVariant, InferenceRuntime, LinguisticProjector,
    ModelInputContract, PhonemeVocabularyProjector, PitchCapabilities, Spectrogram,
    SpectrogramContract, SpectrogramLayout, SpeechModelCapabilities, SpeechModelFamily,
    SpeechSynthesisRequest,
};

pub struct BurnFastSpeechAcoustic<B: Backend> {
    model: FastSpeech<B>,
    projector: PhonemeVocabularyProjector,
    output_contract: SpectrogramContract,
    device: B::Device,
}

impl<B: Backend> BurnFastSpeechAcoustic<B> {
    pub fn load(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
    ) -> Result<Self> {
        let config_path = config_path.as_ref();
        let source = fs::read_to_string(config_path)
            .with_context(|| format!("failed to read model config {}", config_path.display()))?;
        let root: Value = json5::from_str(&source)
            .with_context(|| format!("invalid model config {}", config_path.display()))?;
        let config = FastSpeechConfig::from_json_value(&root).map_err(anyhow::Error::new)?;
        let projector = PhonemeVocabularyProjector::from_json5_str(&source)?;
        ensure!(
            projector.vocabulary().len() == config.num_chars,
            "phoneme vocabulary has {} entries but FastSpeech expects {}",
            projector.vocabulary().len(),
            config.num_chars
        );
        let output_contract = AudioFeatureConfig::from_json5_str(&source)?.mel_contract()?;
        ensure!(
            output_contract.layout == SpectrogramLayout::FramesByBins,
            "Burn FastSpeech adapter requires frame-major shared spectrograms"
        );
        ensure!(
            output_contract.bins == config.out_channels,
            "audio feature config declares {} mel bins but FastSpeech emits {}",
            output_contract.bins,
            config.out_channels
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

    pub fn model(&self) -> &FastSpeech<B> {
        &self.model
    }

    pub fn variant(&self) -> FastSpeechVariant {
        self.model.variant()
    }

    pub fn projector(&self) -> &PhonemeVocabularyProjector {
        &self.projector
    }

    pub fn pitch_capabilities(&self) -> PitchCapabilities {
        self.variant().pitch_capabilities()
    }

    pub fn energy_capabilities(&self) -> EnergyCapabilities {
        self.variant().energy_capabilities()
    }

    pub fn supports_explicit_durations(&self) -> bool {
        true
    }

    pub fn synthesize_tensor(&self, request: &SpeechSynthesisRequest) -> Result<Tensor<B, 3>> {
        ensure!(
            request.plan.speaker.is_none() && request.options.speaker_id.is_none(),
            "this FastSpeech adapter is single-speaker"
        );
        ensure!(
            request.plan.speaker_reference.is_none(),
            "FastSpeech does not accept reference audio"
        );
        let projected = self.projector.project(&request.plan)?;
        let token_count = projected.ids.len();
        let token_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(projected.ids, [1, token_count]),
            &self.device,
        );
        let durations = optional_duration_tensor(
            request.options.durations.as_deref(),
            token_count,
            &self.device,
        )?;
        let pitch = optional_variance_tensor(
            "pitch",
            request.options.pitch.as_deref(),
            token_count,
            &self.device,
        )?;
        let energy = optional_variance_tensor(
            "energy",
            request.options.energy.as_deref(),
            token_count,
            &self.device,
        )?;
        let output = self
            .model
            .inference_projected_with_controls(
                token_ids,
                FastSpeechControls {
                    length_scale: request.options.length_scale.map(f64::from).unwrap_or(1.0),
                    durations,
                    pitch_scale: request.options.pitch_scale.map(f64::from).unwrap_or(1.0),
                    pitch_shift: request.options.pitch_shift.map(f64::from).unwrap_or(0.0),
                    pitch,
                    energy_scale: request.options.energy_scale.map(f64::from).unwrap_or(1.0),
                    energy_shift: request.options.energy_shift.map(f64::from).unwrap_or(0.0),
                    energy,
                },
                None,
            )
            .map_err(anyhow::Error::new)
            .context("Burn FastSpeech inference failed")?;
        Ok(output.mel)
    }
}

impl<B: Backend> AcousticModel for BurnFastSpeechAcoustic<B> {
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

pub struct BurnDelightfulTtsAcoustic<B: Backend> {
    model: DelightfulTts<B>,
    config: DelightfulTtsConfig,
    projector: PhonemeVocabularyProjector,
    output_contract: SpectrogramContract,
    conditioning_contracts: Vec<EmbeddingContract>,
    device: B::Device,
}

impl<B: Backend> BurnDelightfulTtsAcoustic<B> {
    pub fn load(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
    ) -> Result<Self> {
        let config_path = config_path.as_ref();
        let source = fs::read_to_string(config_path)
            .with_context(|| format!("failed to read model config {}", config_path.display()))?;
        let root: Value = json5::from_str(&source)
            .with_context(|| format!("invalid model config {}", config_path.display()))?;
        let config = DelightfulTtsConfig::from_json_value(&root)?;
        let projector = PhonemeVocabularyProjector::from_json5_str(&source)?;
        ensure!(
            projector.vocabulary().len() == config.num_chars,
            "phoneme vocabulary has {} entries but DelightfulTTS expects {}",
            projector.vocabulary().len(),
            config.num_chars
        );
        let padding_symbol = root
            .get("characters")
            .and_then(|value| value.get("pad"))
            .and_then(Value::as_str)
            .and_then(|value| value.chars().next())
            .context("DelightfulTTS config requires a one-character padding symbol")?;
        let padding_id = projector
            .symbol_id(padding_symbol)
            .context("padding symbol is absent from checkpoint vocabulary")?;
        let padding_id =
            usize::try_from(padding_id).context("padding symbol ID must be non-negative")?;

        let output_contract = AudioFeatureConfig::from_json5_str(&source)?.mel_contract()?;
        ensure!(
            output_contract.layout == SpectrogramLayout::FramesByBins,
            "Burn DelightfulTTS adapter requires frame-major shared spectrograms"
        );
        ensure!(
            output_contract.bins == config.out_channels,
            "audio feature config declares {} mel bins but DelightfulTTS emits {}",
            output_contract.bins,
            config.out_channels
        );
        let model = config
            .init::<B>(padding_id, &device)
            .map_err(anyhow::Error::new)?
            .load_checkpoint(checkpoint_path)
            .map_err(anyhow::Error::new)?;
        let conditioning_contracts = config
            .speakers
            .use_d_vector_file
            .then(|| EmbeddingContract {
                kind: crate::ConditioningKind::Speaker,
                space: "coqui-delightful-tts-d-vector".into(),
                dimensions: config.speakers.d_vector_dim,
                l2_normalized: true,
            })
            .into_iter()
            .collect();
        Ok(Self {
            model,
            config,
            projector,
            output_contract,
            conditioning_contracts,
            device,
        })
    }

    pub fn model(&self) -> &DelightfulTts<B> {
        &self.model
    }

    pub fn config(&self) -> &DelightfulTtsConfig {
        &self.config
    }

    pub fn pitch_capabilities(&self) -> PitchCapabilities {
        PitchCapabilities {
            scale: true,
            shift: true,
            explicit_values: true,
        }
    }

    pub fn energy_capabilities(&self) -> EnergyCapabilities {
        EnergyCapabilities {
            scale: true,
            shift: true,
            explicit_values: true,
        }
    }

    pub fn supports_explicit_durations(&self) -> bool {
        true
    }

    pub fn synthesize_tensor(&self, request: &SpeechSynthesisRequest) -> Result<Tensor<B, 3>> {
        ensure!(
            request.plan.speaker_reference.is_none(),
            "DelightfulTTS d-vector conditioning requires a precomputed embedding adapter, not raw reference audio"
        );
        let projected = self.projector.project(&request.plan)?;
        let token_count = projected.ids.len();
        let token_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(projected.ids, [1, token_count]),
            &self.device,
        );
        let speaker_id = request.options.speaker_id.map(|speaker_id| {
            Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![i64::from(speaker_id)], [1]),
                &self.device,
            )
        });
        ensure!(
            request.plan.speaker.is_none() || speaker_id.is_some(),
            "named DelightfulTTS speakers must be resolved to a checkpoint speaker ID"
        );
        let output = self
            .model
            .inference_with_controls(
                token_ids,
                DelightfulTtsControls {
                    length_scale: request.options.length_scale.map(f64::from).unwrap_or(1.0),
                    durations: optional_duration_tensor(
                        request.options.durations.as_deref(),
                        token_count,
                        &self.device,
                    )?,
                    pitch_scale: request.options.pitch_scale.map(f64::from).unwrap_or(1.0),
                    pitch_shift: request.options.pitch_shift.map(f64::from).unwrap_or(0.0),
                    pitch: optional_variance_tensor(
                        "pitch",
                        request.options.pitch.as_deref(),
                        token_count,
                        &self.device,
                    )?,
                    energy_scale: request.options.energy_scale.map(f64::from).unwrap_or(1.0),
                    energy_shift: request.options.energy_shift.map(f64::from).unwrap_or(0.0),
                    energy: optional_variance_tensor(
                        "energy",
                        request.options.energy.as_deref(),
                        token_count,
                        &self.device,
                    )?,
                    speaker_ids: speaker_id,
                    // Embedding production is deliberately a separate native
                    // conditioning adapter under the shared contract.
                    d_vectors: None,
                },
            )
            .map_err(anyhow::Error::new)
            .context("Burn DelightfulTTS inference failed")?;
        Ok(output.mel)
    }
}

impl<B: Backend> AcousticModel for BurnDelightfulTtsAcoustic<B> {
    fn runtime(&self) -> InferenceRuntime {
        InferenceRuntime::Burn
    }

    fn capabilities(&self) -> SpeechModelCapabilities {
        SpeechModelCapabilities {
            family: SpeechModelFamily::AcousticModel,
            supports_named_speakers: self.config.speakers.use_speaker_embedding,
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
        &self.conditioning_contracts
    }

    fn output_contract(&self) -> AcousticOutputContract {
        AcousticOutputContract::Spectrogram(self.output_contract.clone())
    }

    fn synthesize(&mut self, request: &SpeechSynthesisRequest) -> Result<AcousticArtifact> {
        tensor_to_artifact(self.synthesize_tensor(request)?, &self.output_contract)
    }
}

fn optional_duration_tensor<B: Backend>(
    values: Option<&[u32]>,
    token_count: usize,
    device: &B::Device,
) -> Result<Option<Tensor<B, 2>>> {
    values
        .map(|values| {
            ensure!(
                values.len() == token_count,
                "explicit durations contain {} values but projection produced {token_count} tokens",
                values.len()
            );
            ensure!(
                values.iter().all(|value| *value > 0),
                "explicit durations must all be positive"
            );
            Ok(Tensor::from_data(
                TensorData::new(
                    values.iter().map(|value| *value as f32).collect(),
                    [1, token_count],
                ),
                device,
            ))
        })
        .transpose()
}

fn optional_variance_tensor<B: Backend>(
    label: &str,
    values: Option<&[f32]>,
    token_count: usize,
    device: &B::Device,
) -> Result<Option<Tensor<B, 3>>> {
    values
        .map(|values| {
            ensure!(
                values.len() == token_count,
                "explicit {label} contains {} values but projection produced {token_count} tokens",
                values.len()
            );
            ensure!(
                values.iter().all(|value| value.is_finite()),
                "explicit {label} values must be finite"
            );
            Ok(Tensor::from_data(
                TensorData::new(values.to_vec(), [1, 1, token_count]),
                device,
            ))
        })
        .transpose()
}

fn tensor_to_artifact<B: Backend>(
    mel: Tensor<B, 3>,
    contract: &SpectrogramContract,
) -> Result<AcousticArtifact> {
    let frames = mel.dims()[1];
    let values = mel
        .into_data()
        .to_vec::<f32>()
        .context("Burn acoustic output is not f32")?;
    let spectrogram = Spectrogram {
        contract: contract.clone(),
        frames,
        values,
    };
    spectrogram.validate()?;
    Ok(AcousticArtifact::Spectrogram(spectrogram))
}
