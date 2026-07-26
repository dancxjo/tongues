//! Tensor-preserving native Glow-TTS-family acoustic inference plus vocoder.

use std::time::Instant;

use anyhow::{ensure, Context, Result};
use burn::tensor::backend::Backend;

use crate::profiling::{finish_host_stage, reborrow_profiler};
use crate::{
    AcousticModel, AcousticOutputContract, AudioChunk, AudioSink, BurnGlowTtsAcoustic,
    BurnTensorVocoder, BurnVocoder, SpeechModelCapabilities, SpeechSynthesisEngine,
    SpeechSynthesisRequest, SynthesisDimension, SynthesisProfiler, SynthesisStage,
    WaveformContract,
};

/// Composes Glow-TTS or SC-GlowTTS with any contract-compatible Burn vocoder.
pub struct BurnGlowTtsPipeline<B: Backend, V = BurnVocoder<B>> {
    acoustic: BurnGlowTtsAcoustic<B>,
    vocoder: V,
    output_contract: WaveformContract,
}

impl<B: Backend, V: BurnTensorVocoder<B>> BurnGlowTtsPipeline<B, V> {
    pub fn new(acoustic: BurnGlowTtsAcoustic<B>, vocoder: V) -> Result<Self> {
        let AcousticOutputContract::Spectrogram(spectrogram_contract) = acoustic.output_contract()
        else {
            anyhow::bail!("Glow-TTS must emit a spectrogram");
        };
        spectrogram_contract.ensure_compatible_with(vocoder.input_contract())?;
        let output_contract = vocoder.output_contract();
        output_contract.validate()?;
        Ok(Self {
            acoustic,
            vocoder,
            output_contract,
        })
    }

    pub fn acoustic_model(&self) -> &BurnGlowTtsAcoustic<B> {
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
        let mel = match reborrow_profiler(&mut profiler) {
            Some(profiler) => self
                .acoustic
                .synthesize_tensor_profiled(request, profiler)?,
            None => self.acoustic.synthesize_tensor(request)?,
        };
        let waveform = self
            .vocoder
            .synthesize_tensor(mel, reborrow_profiler(&mut profiler))?;
        let sample_count = waveform.dims()[2];
        let started = Instant::now();
        let samples = waveform
            .into_data()
            .to_vec::<f32>()
            .context("Glow-TTS vocoder output is not f32")?;
        finish_host_stage(
            &mut profiler,
            SynthesisStage::DeviceToHost,
            started,
            [SynthesisDimension::new("samples", sample_count)],
        );
        ensure!(
            samples.len() == sample_count,
            "Glow-TTS vocoder returned {} samples, expected {sample_count}",
            samples.len()
        );
        ensure!(
            samples.iter().all(|sample| sample.is_finite()),
            "Glow-TTS vocoder waveform contains non-finite samples"
        );
        let started = Instant::now();
        sink.emit(AudioChunk {
            chunk_index: 0,
            is_final: true,
            pause_after_ms: 0,
            sample_rate_hz: self.output_contract.sample_rate_hz,
            pcm_mono_f32: samples,
        })?;
        finish_host_stage(
            &mut profiler,
            SynthesisStage::AudioSink,
            started,
            [SynthesisDimension::new("samples", sample_count)],
        );
        Ok(())
    }
}

impl<B: Backend, V: BurnTensorVocoder<B>> SpeechSynthesisEngine for BurnGlowTtsPipeline<B, V> {
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
