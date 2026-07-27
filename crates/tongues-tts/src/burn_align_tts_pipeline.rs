//! Tensor-preserving native Align-TTS plus Burn-vocoder inference.

use std::time::Instant;

use anyhow::{ensure, Context, Result};
use burn::tensor::backend::Backend;

use crate::profiling::{finish_host_stage, reborrow_profiler};
use crate::{
    AcousticModel, AcousticOutputContract, AudioChunk, AudioSink, BurnAlignTtsAcoustic,
    BurnHifiganVocoder, BurnTensorVocoder, SpeechModelCapabilities, SpeechSynthesisEngine,
    SpeechSynthesisRequest, SynthesisDimension, SynthesisProfiler, SynthesisStage,
    WaveformContract,
};

pub struct BurnAlignTtsPipeline<B: Backend, V = BurnHifiganVocoder<B>> {
    acoustic: BurnAlignTtsAcoustic<B>,
    vocoder: V,
    output_contract: WaveformContract,
}

impl<B: Backend, V: BurnTensorVocoder<B>> BurnAlignTtsPipeline<B, V> {
    pub fn new(acoustic: BurnAlignTtsAcoustic<B>, vocoder: V) -> Result<Self> {
        let AcousticOutputContract::Spectrogram(contract) = acoustic.output_contract() else {
            anyhow::bail!("Align-TTS must emit a spectrogram");
        };
        contract.ensure_compatible_with(vocoder.input_contract())?;
        let output_contract = vocoder.output_contract();
        output_contract.validate()?;
        Ok(Self {
            acoustic,
            vocoder,
            output_contract,
        })
    }

    pub fn acoustic_model(&self) -> &BurnAlignTtsAcoustic<B> {
        &self.acoustic
    }

    pub fn vocoder(&self) -> &V {
        &self.vocoder
    }

    fn synthesize_internal(
        &mut self,
        request: &SpeechSynthesisRequest,
        sink: &mut dyn AudioSink,
        profiler: Option<&mut dyn SynthesisProfiler>,
    ) -> Result<()> {
        let mut profiler = profiler;
        self.acoustic
            .input_contract()
            .ensure_supports(&request.plan)?;
        let mel = self.acoustic.synthesize_tensor(request)?;
        let waveform = self
            .vocoder
            .synthesize_tensor(mel, reborrow_profiler(&mut profiler))?;
        let sample_count = waveform.dims()[2];
        let started = Instant::now();
        let samples = waveform
            .into_data()
            .to_vec::<f32>()
            .context("Burn vocoder output is not f32")?;
        finish_host_stage(
            &mut profiler,
            SynthesisStage::DeviceToHost,
            started,
            [SynthesisDimension::new("samples", sample_count)],
        );
        ensure!(
            samples.len() == sample_count && samples.iter().all(|sample| sample.is_finite()),
            "Align-TTS vocoder produced an invalid waveform"
        );
        sink.emit(AudioChunk {
            chunk_index: 0,
            is_final: true,
            pause_after_ms: 0,
            sample_rate_hz: self.output_contract.sample_rate_hz,
            pcm_mono_f32: samples,
        })
    }
}

impl<B: Backend, V: BurnTensorVocoder<B>> SpeechSynthesisEngine for BurnAlignTtsPipeline<B, V> {
    fn capabilities(&self) -> SpeechModelCapabilities {
        self.acoustic.capabilities()
    }

    fn sample_rate_hz(&self) -> u32 {
        self.output_contract.sample_rate_hz
    }

    fn synthesize_plan_streaming(
        &mut self,
        request: &SpeechSynthesisRequest,
        sink: &mut dyn AudioSink,
    ) -> Result<()> {
        self.synthesize_internal(request, sink, None)
    }

    fn synthesize_plan_streaming_profiled(
        &mut self,
        request: &SpeechSynthesisRequest,
        sink: &mut dyn AudioSink,
        profiler: &mut dyn SynthesisProfiler,
    ) -> Result<()> {
        self.synthesize_internal(request, sink, Some(profiler))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    use super::*;
    use crate::{
        utterance_plan_from_text, AudioFeatureConfig, HifiganBundleConfig, HifiganGeneratorParams,
        SpeechRequest, SynthesisOptions,
    };

    type TestBackend = NdArray<f32>;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/speech/align-tts-mpl-fixture")
            .join(name)
    }

    #[test]
    fn cpu_align_tts_composes_and_synthesizes_with_native_hifigan() {
        let device = NdArrayDevice::Cpu;
        let acoustic = BurnAlignTtsAcoustic::<TestBackend>::load(
            fixture_path("config.json"),
            fixture_path("model_file.pth"),
            device,
        )
        .expect("Align-TTS fixture");
        let config = HifiganBundleConfig {
            audio: AudioFeatureConfig::from_file(fixture_path("config.json"))
                .expect("shared feature contract"),
            generator_model: "hifigan_generator".into(),
            generator_model_params: HifiganGeneratorParams {
                resblock_type: "2".into(),
                upsample_factors: vec![4, 4, 4, 4],
                upsample_kernel_sizes: vec![8, 8, 8, 8],
                upsample_initial_channel: 16,
                resblock_kernel_sizes: vec![3],
                resblock_dilation_sizes: vec![vec![1, 3]],
            },
        };
        let generator = config
            .init_burn_generator::<TestBackend>(&device)
            .expect("tiny native HiFi-GAN");
        let vocoder =
            BurnHifiganVocoder::from_generator(config, generator, device).expect("vocoder");
        let mut pipeline = BurnAlignTtsPipeline::new(acoustic, vocoder).expect("composition");
        let request = SpeechSynthesisRequest {
            plan: utterance_plan_from_text(SpeechRequest {
                text: "Morning light rested on the cedar trees.".into(),
                variety: "en-US".into(),
            })
            .expect("plan"),
            options: SynthesisOptions::default(),
        };
        let mut chunks = Vec::new();
        pipeline
            .synthesize_plan_streaming(&request, &mut |chunk| {
                chunks.push(chunk);
                Ok(())
            })
            .expect("native CPU synthesis");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].sample_rate_hz, 22_050);
        assert!(!chunks[0].pcm_mono_f32.is_empty());
        assert!(chunks[0]
            .pcm_mono_f32
            .iter()
            .all(|sample| sample.is_finite()));
    }
}
