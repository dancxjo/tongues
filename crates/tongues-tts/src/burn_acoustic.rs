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
    InferenceRuntime, LinguisticProjector, ModelInputContract, ModelLoadProfileEvent,
    ModelLoadStage, PhonemeVocabularyProjector, Spectrogram, SpectrogramContract,
    SpectrogramLayout, SpeechModelCapabilities, SpeechModelFamily, SpeechSynthesisRequest,
    SpeedySpeech, SpeedySpeechConfig, SynthesisDimension, SynthesisProfiler, SynthesisStage,
};

/// Burn-native adapter for a released SpeedySpeech acoustic checkpoint.
///
/// The shared input remains Tongues' linguistic plan. Checkpoint-local
/// character IDs exist only between this adapter's projector and model.
pub struct BurnSpeedySpeechAcoustic<B: Backend> {
    model: SpeedySpeech<B>,
    projector: PhonemeVocabularyProjector,
    output_contract: SpectrogramContract,
    device: B::Device,
}

impl<B: Backend> BurnSpeedySpeechAcoustic<B> {
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
        let model_config =
            SpeedySpeechConfig::from_json_value(&root).map_err(anyhow::Error::new)?;
        let projector = PhonemeVocabularyProjector::from_json5_str(&source)?;
        ensure!(
            projector.vocabulary().len() == model_config.num_chars,
            "phoneme vocabulary has {} entries but SpeedySpeech expects {}",
            projector.vocabulary().len(),
            model_config.num_chars
        );
        let output_contract = AudioFeatureConfig::from_json5_str(&source)?.mel_contract()?;
        ensure!(
            output_contract.layout == SpectrogramLayout::FramesByBins,
            "Burn SpeedySpeech adapter requires frame-major shared spectrograms"
        );
        ensure!(
            output_contract.bins == model_config.out_channels,
            "audio feature config declares {} mel bins but SpeedySpeech emits {}",
            output_contract.bins,
            model_config.out_channels
        );
        record_load_stage(
            &mut profiler,
            ModelLoadStage::ConfigCheckpointParsing,
            started,
            Some("speedy_speech"),
        );

        let started = Instant::now();
        let model = model_config
            .init::<B>(&device)
            .map_err(anyhow::Error::new)?;
        record_load_stage(
            &mut profiler,
            ModelLoadStage::ModelConstruction,
            started,
            Some("speedy_speech"),
        );

        let started = Instant::now();
        let model = model
            .load_checkpoint(checkpoint_path)
            .map_err(anyhow::Error::new)?;
        record_load_stage(
            &mut profiler,
            ModelLoadStage::WeightUpload,
            started,
            Some("speedy_speech"),
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

    pub fn model(&self) -> &SpeedySpeech<B> {
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
            "the released LJSpeech SpeedySpeech checkpoint is single-speaker"
        );
        ensure!(
            request.plan.speaker_reference.is_none(),
            "the released LJSpeech SpeedySpeech checkpoint does not accept reference audio"
        );

        let started = Instant::now();
        let projected = self.projector.project(&request.plan)?;
        let token_count = projected.ids.len();
        ensure!(
            projected
                .ids
                .iter()
                .all(|id| *id >= 0 && (*id as usize) < self.projector.vocabulary().len()),
            "projected SpeedySpeech token ID is outside the checkpoint vocabulary"
        );
        finish_host_stage(
            &mut profiler,
            SynthesisStage::CheckpointProjection,
            started,
            [SynthesisDimension::new("tokens", token_count)],
        );

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
        let length_scale = request.options.length_scale.map(f64::from).unwrap_or(1.0);
        let output = self
            .model
            .inference_projected_with_length_scale(
                token_ids,
                length_scale,
                reborrow_profiler(&mut profiler),
            )
            .map_err(anyhow::Error::new)
            .context("Burn SpeedySpeech inference failed")?;
        Ok(output.mel)
    }
}

impl<B: Backend> AcousticModel for BurnSpeedySpeechAcoustic<B> {
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
            .context("Burn SpeedySpeech output is not f32")?;
        let spectrogram = Spectrogram {
            contract: self.output_contract.clone(),
            frames,
            values,
        };
        spectrogram.validate()?;
        Ok(AcousticArtifact::Spectrogram(spectrogram))
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    use super::*;
    use crate::{
        utterance_plan_from_text, BurnHifiganVocoder, SpeechPipeline, SpeechRequest,
        SynthesisOptions, VocoderDecoder, WaveformContract,
    };

    type TestBackend = NdArray<f32>;

    fn plan() -> speaking::UtterancePlan {
        utterance_plan_from_text(SpeechRequest {
            text: "Morning light rested on cedar trees.".into(),
            variety: "en-US".into(),
        })
        .expect("native multi-word linguistic plan")
    }

    #[test]
    fn published_components_compose_from_native_plan_when_available() {
        let Some(acoustic_model) = std::env::var_os("TONGUES_TEST_COQUI_SPEEDY_MODEL") else {
            return;
        };
        let acoustic_config = std::env::var_os("TONGUES_TEST_COQUI_SPEEDY_CONFIG")
            .expect("TONGUES_TEST_COQUI_SPEEDY_CONFIG must accompany the acoustic model");
        let Some(vocoder_model) = std::env::var_os("TONGUES_TEST_COQUI_HIFIGAN_MODEL") else {
            return;
        };
        let vocoder_config = std::env::var_os("TONGUES_TEST_COQUI_HIFIGAN_CONFIG")
            .expect("TONGUES_TEST_COQUI_HIFIGAN_CONFIG must accompany the vocoder model");
        let device = NdArrayDevice::Cpu;
        let acoustic =
            BurnSpeedySpeechAcoustic::<TestBackend>::load(acoustic_config, acoustic_model, device)
                .expect("acoustic model");
        let vocoder =
            BurnHifiganVocoder::<TestBackend>::load(vocoder_config, vocoder_model, device)
                .expect("vocoder");
        let mut pipeline =
            SpeechPipeline::new(acoustic, VocoderDecoder::new(vocoder)).expect("pipeline");
        let request = SpeechSynthesisRequest {
            plan: plan(),
            options: SynthesisOptions::default(),
        };

        let waveform = pipeline.synthesize(&request).expect("waveform");

        assert_eq!(waveform.contract, WaveformContract::mono(22_050));
        assert!(!waveform.samples.is_empty());
        assert!(waveform.samples.iter().all(|sample| sample.is_finite()));
        assert!(waveform.samples.iter().any(|sample| sample.abs() > 1e-6));
    }
}
