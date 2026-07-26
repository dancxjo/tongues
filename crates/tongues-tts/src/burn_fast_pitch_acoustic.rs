use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use serde_json::Value;

use crate::profiling::{
    finish_backend_stage, finish_host_stage, reborrow_profiler, record_load_stage,
};
use crate::{
    AcousticArtifact, AcousticModel, AcousticOutputContract, AudioFeatureConfig, EmbeddingContract,
    FastPitch, FastPitchConfig, FastPitchControls, InferenceRuntime, LinguisticProjector,
    ModelInputContract, ModelLoadProfileEvent, ModelLoadStage, PhonemeVocabularyProjector,
    Spectrogram, SpectrogramContract, SpectrogramLayout, SpeechModelCapabilities,
    SpeechModelFamily, SpeechSynthesisRequest, SynthesisDimension, SynthesisProfiler,
    SynthesisStage,
};

/// Native adapter from Tongues' linguistic plan to the released LJSpeech
/// FastPitch checkpoint.
pub struct BurnFastPitchAcoustic<B: Backend> {
    model: FastPitch<B>,
    projector: PhonemeVocabularyProjector,
    output_contract: SpectrogramContract,
    device: B::Device,
}

impl<B: Backend> BurnFastPitchAcoustic<B> {
    pub fn load(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
    ) -> Result<Self> {
        Self::load_internal(config_path, checkpoint_path, device, None)
    }

    pub fn load_profiled(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
        profiler: &mut dyn FnMut(ModelLoadProfileEvent),
    ) -> Result<Self> {
        Self::load_internal(config_path, checkpoint_path, device, Some(profiler))
    }

    fn load_internal(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
        profiler: Option<&mut dyn FnMut(ModelLoadProfileEvent)>,
    ) -> Result<Self> {
        let mut profiler = profiler;
        let started = Instant::now();
        let config_path = config_path.as_ref();
        let source = fs::read_to_string(config_path)
            .with_context(|| format!("failed to read model config {}", config_path.display()))?;
        let root: Value = json5::from_str(&source)
            .with_context(|| format!("invalid model config {}", config_path.display()))?;
        let model_config = FastPitchConfig::from_json_value(&root).map_err(anyhow::Error::new)?;
        let projector = PhonemeVocabularyProjector::from_json5_str(&source)?;
        ensure!(
            projector.vocabulary().len() == model_config.num_chars,
            "phoneme vocabulary has {} entries but FastPitch expects {}",
            projector.vocabulary().len(),
            model_config.num_chars
        );
        let output_contract = AudioFeatureConfig::from_json5_str(&source)?.mel_contract()?;
        ensure!(
            output_contract.layout == SpectrogramLayout::FramesByBins,
            "Burn FastPitch adapter requires frame-major shared spectrograms"
        );
        ensure!(
            output_contract.bins == model_config.out_channels,
            "audio feature config declares {} mel bins but FastPitch emits {}",
            output_contract.bins,
            model_config.out_channels
        );
        record_load_stage(
            &mut profiler,
            ModelLoadStage::ConfigCheckpointParsing,
            started,
            Some("fast_pitch"),
        );

        let started = Instant::now();
        let model = model_config
            .init::<B>(&device)
            .map_err(anyhow::Error::new)?;
        record_load_stage(
            &mut profiler,
            ModelLoadStage::ModelConstruction,
            started,
            Some("fast_pitch"),
        );
        let started = Instant::now();
        let model = model
            .load_checkpoint(checkpoint_path)
            .map_err(anyhow::Error::new)?;
        record_load_stage(
            &mut profiler,
            ModelLoadStage::WeightUpload,
            started,
            Some("fast_pitch"),
        );
        Ok(Self {
            model,
            projector,
            output_contract,
            device,
        })
    }

    pub fn projector(&self) -> &PhonemeVocabularyProjector {
        &self.projector
    }

    pub fn model(&self) -> &FastPitch<B> {
        &self.model
    }

    pub fn synthesize_tensor(
        &self,
        request: &SpeechSynthesisRequest,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<Tensor<B, 3>> {
        let mut profiler = profiler;
        ensure!(
            request.plan.speaker.is_none() && request.options.speaker_id.is_none(),
            "the released LJSpeech FastPitch checkpoint is single-speaker"
        );
        ensure!(
            request.plan.speaker_reference.is_none(),
            "the released LJSpeech FastPitch checkpoint does not accept reference audio"
        );

        let started = Instant::now();
        let projected = self.projector.project(&request.plan)?;
        let token_count = projected.ids.len();
        ensure!(
            projected
                .ids
                .iter()
                .all(|id| *id >= 0 && (*id as usize) < self.projector.vocabulary().len()),
            "projected FastPitch token ID is outside the checkpoint vocabulary"
        );
        finish_host_stage(
            &mut profiler,
            SynthesisStage::CheckpointProjection,
            started,
            [SynthesisDimension::new("tokens", token_count)],
        );

        let length_scale = request.options.length_scale.map(f64::from).unwrap_or(1.0);
        let pitch_scale = request.options.pitch_scale.map(f64::from).unwrap_or(1.0);
        let pitch_shift = request.options.pitch_shift.map(f64::from).unwrap_or(0.0);
        ensure!(
            pitch_scale.is_finite() && pitch_scale > 0.0,
            "pitch_scale must be finite and positive"
        );
        ensure!(pitch_shift.is_finite(), "pitch_shift must be finite");
        let durations = request
            .options
            .durations
            .as_ref()
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
                Ok(Tensor::<B, 2>::from_data(
                    TensorData::new(
                        values.iter().map(|value| *value as f32).collect::<Vec<_>>(),
                        [1, token_count],
                    ),
                    &self.device,
                ))
            })
            .transpose()?;
        let pitch = request
            .options
            .pitch
            .as_ref()
            .map(|values| {
                ensure!(
                    values.len() == token_count,
                    "explicit pitch contains {} values but projection produced {token_count} tokens",
                    values.len()
                );
                ensure!(
                    values.iter().all(|value| value.is_finite()),
                    "explicit pitch values must be finite"
                );
                Ok(Tensor::<B, 3>::from_data(
                    TensorData::new(values.clone(), [1, 1, token_count]),
                    &self.device,
                ))
            })
            .transpose()?;

        let started = Instant::now();
        let token_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(projected.ids, [1, token_count]),
            &self.device,
        );
        finish_backend_stage::<B>(
            &mut profiler,
            &self.device,
            SynthesisStage::HostToDevice,
            started,
            [SynthesisDimension::new("tokens", token_count)],
        )?;
        let output = self
            .model
            .inference_projected_with_controls(
                token_ids,
                FastPitchControls {
                    length_scale,
                    pitch_scale,
                    pitch_shift,
                    durations,
                    pitch,
                },
                reborrow_profiler(&mut profiler),
            )
            .map_err(anyhow::Error::new)
            .context("Burn FastPitch inference failed")?;
        Ok(output.mel)
    }
}

impl<B: Backend> AcousticModel for BurnFastPitchAcoustic<B> {
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
        let mel = self.synthesize_tensor(request, None)?;
        let frames = mel.dims()[1];
        let values = mel
            .into_data()
            .to_vec::<f32>()
            .context("Burn FastPitch output is not f32")?;
        let spectrogram = Spectrogram {
            contract: self.output_contract.clone(),
            frames,
            values,
        };
        spectrogram.validate()?;
        Ok(AcousticArtifact::Spectrogram(spectrogram))
    }
}
