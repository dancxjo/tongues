//! Deterministic, CPU-native audio loading and feature extraction.
//!
//! This crate deliberately uses model-neutral names. Compatibility adapters
//! (for example, Coqui configuration parsing) belong in the model crate that
//! owns that external format.

use std::f32::consts::PI;
use std::io::Read;
use std::path::Path;

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod cleanup;
mod phone_alignment;
mod phonetic_segmentation;
mod segmentation;
mod source;
mod speech_pipeline;
mod system_input;
mod system_output;
mod transform;
mod vad;

pub use cleanup::{
    CleanupAudioSource, CleanupCapability, CleanupPipeline, CleanupStage, CleanupStageConfig,
    CleanupStageTrace, ProcessedAudio,
};
pub use phone_alignment::{
    check_alignment_conformance, evaluate_alignment, AlignedUnit, AlignmentBackendCapabilities,
    AlignmentBackendConformanceReport, AlignmentCancellation, AlignmentCorrection, AlignmentDelta,
    AlignmentDeltaKind, AlignmentDiagnostic, AlignmentEvaluationReference,
    AlignmentEvaluationReport, AlignmentHypothesis, AlignmentLifecycle, AlignmentLimits,
    AlignmentMode, AlignmentProjection, AlignmentReadiness, AlignmentScoreBreakdown,
    AlignmentUnitRelation, AlignmentUnitSpec, AudioAlignmentInput, BoundaryEstimate, BoundaryHint,
    CtcPosteriorBackend, CtcPosteriorMatrix, DurationPrior, PhoneAlignmentArtifact,
    PhoneAlignmentBackend, PhoneAlignmentEngine, PhoneAlignmentRequest, ProjectionKind,
    ProjectionLoss, PronunciationPath, ReferenceAlignmentUnit, ReferenceAlignmentWord,
    StreamingAlignmentMetrics, StreamingAlignmentUpdate, StreamingPhoneAligner, TimingAuthority,
    TranscriptLattice, TranscriptToken, PHONE_ALIGNMENT_ALGORITHM_VERSION,
    PHONE_ALIGNMENT_SCHEMA_VERSION,
};
pub use phonetic_segmentation::{
    audio_sha256, AlignmentCandidate, AlignmentEvidence, AlignmentRecipe, AlignmentSourceIdentity,
    ExpectedSegment, FrameInterval, HintAlignmentSource, InventoryMembership,
    PhoneticAlignmentSource, PhoneticBoundaryOrigin, PhoneticEvidenceLinks, PhoneticSegment,
    PhoneticSegmentArtifact, PhoneticSegmentStatus, PhoneticSegmentationContext,
    PhoneticSegmentationEngine, PhoneticSegmentationIssue, PhoneticSegmentationReadiness,
    SegmentKind, UnalignedRegion, ALIGNMENT_RECIPE_SCHEMA_VERSION,
    PHONETIC_SEGMENTATION_ALGORITHM_VERSION, PHONETIC_SEGMENTATION_ARTIFACT_SCHEMA_VERSION,
};
pub use segmentation::{
    AudioSegment, AuthoritativeBoundary, BoundaryEvidenceKind, SegmentCloseReason,
    SegmentationConfig, SegmentationEvent, SegmentationFrame, SegmentationMetrics,
    UtteranceSegmenter,
};
pub use source::{
    bounded_audio_input, AudioDiscontinuity, AudioSource, AudioSourceDescriptor, AudioSourceEvent,
    AudioSourceKind, BoundedAudioInput, BoundedAudioInputSender, PcmEncoding, PcmReaderSource,
    PushedAudioChunk, SourceAudioChunk, WavAudioSource,
};
pub use speech_pipeline::{VadPipelineEvent, VadSegmentationPipeline};
pub use system_input::{
    input_capabilities, input_device_inventory, AudioInputCapabilities, CpalAudioSource,
    InputDeviceInfo,
};
pub use system_output::{
    output_device_inventory, write_wav_output, CpalAudioSink, OutputDeviceInfo,
};
pub use transform::NormalizedAudioSource;
pub use vad::{
    create_vad_backend, EnergyVad, EnergyVadConfig, VadBackendKind, VadDecision,
    VoiceActivityDetector,
};
#[cfg(feature = "vad-webrtc")]
pub use vad::{WebRtcVad, WebRtcVadConfig};

pub const DEFAULT_SPECTRAL_FLOOR: f32 = 1.0e-8;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("failed to read WAV: {0}")]
    Wav(#[from] hound::Error),
    #[error("invalid audio: {0}")]
    Invalid(String),
    #[error("failed to read audio input: {0}")]
    Io(#[from] std::io::Error),
    #[error("audio input was cancelled")]
    Cancelled,
    #[error("audio input backpressure limit of {capacity} chunks was reached")]
    Backpressure { capacity: usize },
}

pub type Result<T> = std::result::Result<T, AudioError>;

/// Interleaved, normalized floating-point PCM and its exact stream metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
    pub channels: u16,
}

impl AudioBuffer {
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate_hz == 0 {
            return Err(invalid("sample rate must be positive"));
        }
        if self.channels == 0 {
            return Err(invalid("channel count must be positive"));
        }
        if !self
            .samples
            .len()
            .is_multiple_of(usize::from(self.channels))
        {
            return Err(invalid(format!(
                "{} samples are not divisible by {} channels",
                self.samples.len(),
                self.channels
            )));
        }
        if self.samples.iter().any(|sample| !sample.is_finite()) {
            return Err(invalid("PCM contains a non-finite sample"));
        }
        Ok(())
    }

    pub fn frames(&self) -> usize {
        self.samples.len() / usize::from(self.channels.max(1))
    }

    /// Convert between mono and interleaved multi-channel audio.
    ///
    /// Downmixing averages all input channels. Upmixing is defined only for a
    /// mono input and duplicates that channel, keeping the operation explicit
    /// and deterministic.
    pub fn convert_channels(&self, target_channels: u16) -> Result<Self> {
        self.validate()?;
        if target_channels == 0 {
            return Err(invalid("target channel count must be positive"));
        }
        if target_channels == self.channels {
            return Ok(self.clone());
        }

        let input_channels = usize::from(self.channels);
        let output_channels = usize::from(target_channels);
        let samples = if target_channels == 1 {
            self.samples
                .chunks_exact(input_channels)
                .map(|frame| frame.iter().sum::<f32>() / input_channels as f32)
                .collect()
        } else if self.channels == 1 {
            let mut output = Vec::with_capacity(self.samples.len() * output_channels);
            for &sample in &self.samples {
                output.extend(std::iter::repeat_n(sample, output_channels));
            }
            output
        } else {
            return Err(invalid(format!(
                "conversion from {} to {target_channels} channels is ambiguous",
                self.channels
            )));
        };

        Ok(Self {
            samples,
            sample_rate_hz: self.sample_rate_hz,
            channels: target_channels,
        })
    }

    pub fn to_mono(&self) -> Result<Vec<f32>> {
        Ok(self.convert_channels(1)?.samples)
    }

    /// Deterministic linear resampling, performed independently per channel.
    ///
    /// The endpoint convention is `output_index * source_rate / target_rate`.
    /// This makes lengths and cached features stable across platforms.
    pub fn resample_linear(&self, target_rate_hz: u32) -> Result<Self> {
        self.validate()?;
        if target_rate_hz == 0 {
            return Err(invalid("target sample rate must be positive"));
        }
        if target_rate_hz == self.sample_rate_hz || self.samples.is_empty() {
            let mut output = self.clone();
            output.sample_rate_hz = target_rate_hz;
            return Ok(output);
        }

        let channels = usize::from(self.channels);
        let input_frames = self.frames();
        let output_frames = ((input_frames as u128 * u128::from(target_rate_hz)
            + u128::from(self.sample_rate_hz) / 2)
            / u128::from(self.sample_rate_hz))
        .max(1) as usize;
        let step = self.sample_rate_hz as f64 / target_rate_hz as f64;
        let mut samples = Vec::with_capacity(output_frames * channels);
        for output_frame in 0..output_frames {
            let source = output_frame as f64 * step;
            let left = (source.floor() as usize).min(input_frames - 1);
            let right = (left + 1).min(input_frames - 1);
            let fraction = (source - left as f64) as f32;
            for channel in 0..channels {
                let a = self.samples[left * channels + channel];
                let b = self.samples[right * channels + channel];
                samples.push(a + (b - a) * fraction);
            }
        }
        Ok(Self {
            samples,
            sample_rate_hz: target_rate_hz,
            channels: self.channels,
        })
    }
}

pub fn read_wav(path: impl AsRef<Path>) -> Result<AudioBuffer> {
    read_wav_reader(hound::WavReader::open(path)?)
}

pub fn read_wav_bytes(bytes: &[u8]) -> Result<AudioBuffer> {
    read_wav_reader(hound::WavReader::new(std::io::Cursor::new(bytes))?)
}

fn read_wav_reader<R: Read>(mut reader: hound::WavReader<R>) -> Result<AudioBuffer> {
    let spec = reader.spec();
    if spec.channels == 0 {
        return Err(invalid("WAV has zero channels"));
    }
    if spec.sample_rate == 0 {
        return Err(invalid("WAV has a zero sample rate"));
    }
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|sample| sample.map_err(AudioError::from))
            .collect::<Result<Vec<_>>>()?,
        hound::SampleFormat::Int => {
            if spec.bits_per_sample == 0 || spec.bits_per_sample > 32 {
                return Err(invalid(format!(
                    "unsupported {}-bit integer WAV",
                    spec.bits_per_sample
                )));
            }
            let divisor = 2_f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / divisor))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    let audio = AudioBuffer {
        samples,
        sample_rate_hz: spec.sample_rate,
        channels: spec.channels,
    };
    audio.validate()?;
    Ok(audio)
}

pub fn preemphasis(samples: &[f32], coefficient: f32) -> Result<Vec<f32>> {
    validate_emphasis(coefficient)?;
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    let mut output = Vec::with_capacity(samples.len());
    output.push(samples[0]);
    output.extend(
        samples
            .windows(2)
            .map(|pair| pair[1] - coefficient * pair[0]),
    );
    Ok(output)
}

pub fn deemphasis(samples: &[f32], coefficient: f32) -> Result<Vec<f32>> {
    validate_emphasis(coefficient)?;
    let mut output = Vec::with_capacity(samples.len());
    for &sample in samples {
        let restored = sample + coefficient * output.last().copied().unwrap_or(0.0);
        output.push(restored);
    }
    Ok(output)
}

fn validate_emphasis(coefficient: f32) -> Result<()> {
    if !coefficient.is_finite() || !(0.0..1.0).contains(&coefficient) {
        return Err(invalid(
            "emphasis coefficient must be finite and in the range [0, 1)",
        ));
    }
    Ok(())
}

pub fn peak_normalize(samples: &[f32], target_peak: f32) -> Result<Vec<f32>> {
    if !target_peak.is_finite() || !(0.0..=1.0).contains(&target_peak) {
        return Err(invalid("target peak must be finite and in [0, 1]"));
    }
    let peak = samples.iter().map(|value| value.abs()).fold(0.0, f32::max);
    if peak <= f32::EPSILON {
        return Ok(samples.to_vec());
    }
    Ok(samples
        .iter()
        .map(|sample| sample * target_peak / peak)
        .collect())
}

pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt()
}

pub fn rms_normalize_db(samples: &[f32], target_db: f32) -> Result<Vec<f32>> {
    if !target_db.is_finite() || !(-99.0..=0.0).contains(&target_db) {
        return Err(invalid("target RMS dB must be in [-99, 0]"));
    }
    let current = rms(samples);
    if current <= f32::EPSILON {
        return Ok(samples.to_vec());
    }
    let target = 10_f32.powf(target_db / 20.0);
    Ok(samples
        .iter()
        .map(|sample| sample * target / current)
        .collect())
}

/// Trim leading and trailing frames whose peak is below `top_db` from the
/// signal peak. One frame of context is retained on either side.
pub fn trim_silence(
    samples: &[f32],
    top_db: f32,
    frame_length: usize,
    hop_length: usize,
) -> Result<Vec<f32>> {
    if !top_db.is_finite() || top_db < 0.0 {
        return Err(invalid("silence threshold must be a non-negative dB value"));
    }
    if frame_length == 0 || hop_length == 0 {
        return Err(invalid("silence frame and hop lengths must be positive"));
    }
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max);
    if peak <= f32::EPSILON {
        return Ok(Vec::new());
    }
    let threshold = peak * 10_f32.powf(-top_db / 20.0);
    let frame_count = 1 + samples.len().saturating_sub(1) / hop_length;
    let active = (0..frame_count)
        .map(|frame| {
            let center = frame * hop_length;
            let start = center.saturating_sub(frame_length / 2);
            let end = (start + frame_length).min(samples.len());
            samples[start..end]
                .iter()
                .any(|sample| sample.abs() >= threshold)
        })
        .collect::<Vec<_>>();
    let Some(first) = active.iter().position(|value| *value) else {
        return Ok(Vec::new());
    };
    let last = active.iter().rposition(|value| *value).unwrap_or(first);
    let start = first.saturating_sub(1) * hop_length;
    let end = ((last + 2) * hop_length).min(samples.len());
    Ok(samples[start..end].to_vec())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PadMode {
    Reflect,
    Constant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Window {
    Hann,
    Hamming,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StftConfig {
    pub fft_size: usize,
    pub window_size: usize,
    pub hop_size: usize,
    pub center: bool,
    pub pad_mode: PadMode,
    pub window: Window,
}

impl StftConfig {
    pub fn validate(&self) -> Result<()> {
        if self.fft_size == 0 {
            return Err(invalid("FFT size must be positive"));
        }
        if self.window_size == 0 || self.window_size > self.fft_size {
            return Err(invalid("window size must be in 1..=FFT size"));
        }
        if self.hop_size == 0 {
            return Err(invalid("hop size must be positive"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ComplexBin {
    pub re: f32,
    pub im: f32,
}

impl From<Complex32> for ComplexBin {
    fn from(value: Complex32) -> Self {
        Self {
            re: value.re,
            im: value.im,
        }
    }
}

impl From<ComplexBin> for Complex32 {
    fn from(value: ComplexBin) -> Self {
        Self::new(value.re, value.im)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stft {
    pub config: StftConfig,
    pub frames: usize,
    /// Frame-major one-sided complex bins.
    pub bins: Vec<ComplexBin>,
}

impl Stft {
    pub fn bins_per_frame(&self) -> usize {
        self.config.fft_size / 2 + 1
    }

    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        if self.bins.len() != self.frames * self.bins_per_frame() {
            return Err(invalid("STFT bin count does not match its metadata"));
        }
        if self
            .bins
            .iter()
            .any(|bin| !bin.re.is_finite() || !bin.im.is_finite())
        {
            return Err(invalid("STFT contains a non-finite value"));
        }
        Ok(())
    }
}

pub fn stft(samples: &[f32], config: &StftConfig) -> Result<Stft> {
    config.validate()?;
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(invalid("STFT input contains a non-finite sample"));
    }
    let pad = if config.center {
        config.fft_size / 2
    } else {
        0
    };
    let padded = pad_signal(samples, pad, config.pad_mode);
    if padded.len() < config.fft_size {
        return Ok(Stft {
            config: config.clone(),
            frames: 0,
            bins: Vec::new(),
        });
    }
    let frames = 1 + (padded.len() - config.fft_size) / config.hop_size;
    let window = padded_window(config);
    let output_bins = config.fft_size / 2 + 1;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(config.fft_size);
    let mut buffer = vec![Complex32::new(0.0, 0.0); config.fft_size];
    let mut bins = Vec::with_capacity(frames * output_bins);
    for frame in 0..frames {
        let start = frame * config.hop_size;
        for index in 0..config.fft_size {
            buffer[index] = Complex32::new(padded[start + index] * window[index], 0.0);
        }
        fft.process(&mut buffer);
        bins.extend(buffer[..output_bins].iter().copied().map(ComplexBin::from));
    }
    Ok(Stft {
        config: config.clone(),
        frames,
        bins,
    })
}

pub fn istft(spectrum: &Stft) -> Result<Vec<f32>> {
    spectrum.validate()?;
    if spectrum.frames == 0 {
        return Ok(Vec::new());
    }
    let config = &spectrum.config;
    let bins_per_frame = spectrum.bins_per_frame();
    let output_len = config.fft_size + (spectrum.frames - 1) * config.hop_size;
    let window = padded_window(config);
    let mut output = vec![0.0f32; output_len];
    let mut window_sum = vec![0.0f32; output_len];
    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(config.fft_size);
    let mut buffer = vec![Complex32::new(0.0, 0.0); config.fft_size];
    for frame in 0..spectrum.frames {
        buffer.fill(Complex32::new(0.0, 0.0));
        let one_sided = &spectrum.bins[frame * bins_per_frame..(frame + 1) * bins_per_frame];
        for (index, &bin) in one_sided.iter().enumerate() {
            buffer[index] = bin.into();
        }
        let mirror_end = if config.fft_size.is_multiple_of(2) {
            bins_per_frame - 1
        } else {
            bins_per_frame
        };
        for index in 1..mirror_end {
            buffer[config.fft_size - index] = buffer[index].conj();
        }
        ifft.process(&mut buffer);
        let start = frame * config.hop_size;
        for index in 0..config.fft_size {
            let value = buffer[index].re / config.fft_size as f32;
            output[start + index] += value * window[index];
            window_sum[start + index] += window[index] * window[index];
        }
    }
    for (sample, weight) in output.iter_mut().zip(window_sum) {
        if weight > 1.0e-11 {
            *sample /= weight;
        }
    }
    if config.center {
        let pad = config.fft_size / 2;
        let end = output.len().saturating_sub(pad);
        return Ok(output[pad.min(end)..end].to_vec());
    }
    Ok(output)
}

fn pad_signal(samples: &[f32], pad: usize, mode: PadMode) -> Vec<f32> {
    if pad == 0 {
        return samples.to_vec();
    }
    let mut output = Vec::with_capacity(samples.len() + 2 * pad);
    for position in -(pad as isize)..samples.len() as isize + pad as isize {
        let sample = match mode {
            PadMode::Constant => {
                if (0..samples.len() as isize).contains(&position) {
                    samples[position as usize]
                } else {
                    0.0
                }
            }
            PadMode::Reflect => {
                if samples.is_empty() {
                    0.0
                } else {
                    samples[reflect_index(position, samples.len())]
                }
            }
        };
        output.push(sample);
    }
    output
}

fn reflect_index(position: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len - 1) as isize;
    let wrapped = position.rem_euclid(period);
    if wrapped < len as isize {
        wrapped as usize
    } else {
        (period - wrapped) as usize
    }
}

fn padded_window(config: &StftConfig) -> Vec<f32> {
    let mut output = vec![0.0; config.fft_size];
    let offset = (config.fft_size - config.window_size) / 2;
    match config.window {
        Window::Hann => {
            for index in 0..config.window_size {
                output[offset + index] =
                    0.5 - 0.5 * (2.0 * PI * index as f32 / config.window_size as f32).cos();
            }
        }
        Window::Hamming => {
            for index in 0..config.window_size {
                output[offset + index] =
                    0.54 - 0.46 * (2.0 * PI * index as f32 / config.window_size as f32).cos();
            }
        }
    }
    output
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SpectralDomain {
    Amplitude,
    Power,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MelScale {
    Slaney,
    Htk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MelNormalization {
    None,
    Slaney,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MelConfig {
    pub bins: usize,
    pub min_frequency_hz: f32,
    pub max_frequency_hz: Option<f32>,
    pub scale: MelScale,
    pub normalization: MelNormalization,
}

impl MelConfig {
    pub fn validate(&self, sample_rate_hz: u32) -> Result<()> {
        if self.bins == 0 {
            return Err(invalid("mel bin count must be positive"));
        }
        let max = self.max_frequency_hz.unwrap_or(sample_rate_hz as f32 / 2.0);
        if !self.min_frequency_hz.is_finite()
            || !max.is_finite()
            || self.min_frequency_hz < 0.0
            || max <= self.min_frequency_hz
            || max > sample_rate_hz as f32 / 2.0
        {
            return Err(invalid("mel frequency range is outside Nyquist"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum SpectrogramOutput {
    Linear,
    Mel(MelConfig),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum SpectralScale {
    Linear,
    NaturalLog { gain: f32, floor: f32 },
    Log10 { gain: f32, floor: f32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum SpectrogramNormalization {
    None,
    Range {
        min_db: f32,
        reference_db: f32,
        max_norm: f32,
        symmetric: bool,
        clipped: bool,
    },
    Standardized {
        mean: Vec<f32>,
        scale: Vec<f32>,
    },
}

/// Exact preprocessing metadata stored alongside cached tensors/checkpoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpectrogramConfig {
    pub sample_rate_hz: u32,
    pub stft: StftConfig,
    pub output: SpectrogramOutput,
    pub domain: SpectralDomain,
    pub scale: SpectralScale,
    pub normalization: SpectrogramNormalization,
    pub preemphasis: Option<f32>,
}

impl SpectrogramConfig {
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate_hz == 0 {
            return Err(invalid("spectrogram sample rate must be positive"));
        }
        self.stft.validate()?;
        if let SpectrogramOutput::Mel(config) = &self.output {
            config.validate(self.sample_rate_hz)?;
        }
        if let Some(coefficient) = self.preemphasis {
            validate_emphasis(coefficient)?;
        }
        match self.scale {
            SpectralScale::Linear => {}
            SpectralScale::NaturalLog { gain, floor } | SpectralScale::Log10 { gain, floor } => {
                if !gain.is_finite() || gain <= 0.0 || !floor.is_finite() || floor <= 0.0 {
                    return Err(invalid(
                        "logarithmic gain and floor must be finite and positive",
                    ));
                }
            }
        }
        let bins = self.output_bins();
        match &self.normalization {
            SpectrogramNormalization::None => {}
            SpectrogramNormalization::Range {
                min_db,
                reference_db,
                max_norm,
                ..
            } => {
                if !min_db.is_finite()
                    || *min_db >= 0.0
                    || !reference_db.is_finite()
                    || !max_norm.is_finite()
                    || *max_norm <= 0.0
                {
                    return Err(invalid("invalid range normalization metadata"));
                }
            }
            SpectrogramNormalization::Standardized { mean, scale } => {
                if mean.len() != bins
                    || scale.len() != bins
                    || mean.iter().any(|value| !value.is_finite())
                    || scale
                        .iter()
                        .any(|value| !value.is_finite() || *value == 0.0)
                {
                    return Err(invalid("invalid standardization metadata"));
                }
            }
        }
        Ok(())
    }

    pub fn output_bins(&self) -> usize {
        match &self.output {
            SpectrogramOutput::Linear => self.stft.fft_size / 2 + 1,
            SpectrogramOutput::Mel(config) => config.bins,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Spectrogram {
    pub config: SpectrogramConfig,
    pub frames: usize,
    /// Frame-major values.
    pub values: Vec<f32>,
}

impl Spectrogram {
    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        if self.values.len() != self.frames * self.config.output_bins() {
            return Err(invalid(
                "spectrogram tensor shape does not match its metadata",
            ));
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err(invalid("spectrogram contains a non-finite value"));
        }
        Ok(())
    }

    pub fn frame(&self, index: usize) -> Option<&[f32]> {
        let bins = self.config.output_bins();
        self.values.get(index * bins..(index + 1) * bins)
    }
}

pub fn spectrogram(samples: &[f32], config: &SpectrogramConfig) -> Result<Spectrogram> {
    config.validate()?;
    let emphasized;
    let samples = if let Some(coefficient) = config.preemphasis {
        emphasized = preemphasis(samples, coefficient)?;
        &emphasized
    } else {
        samples
    };
    let spectrum = stft(samples, &config.stft)?;
    let linear_bins = spectrum.bins_per_frame();
    let mut values = spectrum
        .bins
        .iter()
        .map(|bin| {
            let power = bin.re * bin.re + bin.im * bin.im;
            match config.domain {
                SpectralDomain::Amplitude => power.sqrt(),
                SpectralDomain::Power => power,
            }
        })
        .collect::<Vec<_>>();
    if let SpectrogramOutput::Mel(mel) = &config.output {
        let weights = mel_filter_bank(config.sample_rate_hz, config.stft.fft_size, mel)?;
        let mut projected = vec![0.0; spectrum.frames * mel.bins];
        for frame in 0..spectrum.frames {
            for mel_bin in 0..mel.bins {
                projected[frame * mel.bins + mel_bin] = weights
                    [mel_bin * linear_bins..(mel_bin + 1) * linear_bins]
                    .iter()
                    .zip(&values[frame * linear_bins..(frame + 1) * linear_bins])
                    .map(|(weight, value)| weight * value)
                    .sum();
            }
        }
        values = projected;
    }
    apply_scale(&mut values, &config.scale);
    apply_normalization(&mut values, config.output_bins(), &config.normalization);
    let output = Spectrogram {
        config: config.clone(),
        frames: spectrum.frames,
        values,
    };
    output.validate()?;
    Ok(output)
}

pub fn mel_filter_bank(
    sample_rate_hz: u32,
    fft_size: usize,
    config: &MelConfig,
) -> Result<Vec<f32>> {
    config.validate(sample_rate_hz)?;
    if fft_size == 0 {
        return Err(invalid("FFT size must be positive"));
    }
    let max_hz = config
        .max_frequency_hz
        .unwrap_or(sample_rate_hz as f32 / 2.0);
    let min_mel = hz_to_mel(config.min_frequency_hz, config.scale);
    let max_mel = hz_to_mel(max_hz, config.scale);
    let points = (0..config.bins + 2)
        .map(|index| {
            let mel = min_mel + (max_mel - min_mel) * index as f32 / (config.bins + 1) as f32;
            mel_to_hz(mel, config.scale)
        })
        .collect::<Vec<_>>();
    let frequencies = (0..fft_size / 2 + 1)
        .map(|bin| bin as f32 * sample_rate_hz as f32 / fft_size as f32)
        .collect::<Vec<_>>();
    let mut weights = vec![0.0; config.bins * frequencies.len()];
    for mel_bin in 0..config.bins {
        let lower = points[mel_bin];
        let center = points[mel_bin + 1];
        let upper = points[mel_bin + 2];
        let normalization = match config.normalization {
            MelNormalization::None => 1.0,
            MelNormalization::Slaney => 2.0 / (upper - lower),
        };
        for (frequency_bin, &frequency) in frequencies.iter().enumerate() {
            let lower_slope = (frequency - lower) / (center - lower);
            let upper_slope = (upper - frequency) / (upper - center);
            weights[mel_bin * frequencies.len() + frequency_bin] =
                lower_slope.min(upper_slope).max(0.0) * normalization;
        }
    }
    Ok(weights)
}

fn hz_to_mel(frequency: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + frequency / 700.0).log10(),
        MelScale::Slaney => {
            const F_SP: f32 = 200.0 / 3.0;
            const MIN_LOG_HZ: f32 = 1000.0;
            const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
            let log_step = 6.4_f32.ln() / 27.0;
            if frequency >= MIN_LOG_HZ {
                MIN_LOG_MEL + (frequency / MIN_LOG_HZ).ln() / log_step
            } else {
                frequency / F_SP
            }
        }
    }
}

fn mel_to_hz(mel: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 700.0 * (10_f32.powf(mel / 2595.0) - 1.0),
        MelScale::Slaney => {
            const F_SP: f32 = 200.0 / 3.0;
            const MIN_LOG_HZ: f32 = 1000.0;
            const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
            let log_step = 6.4_f32.ln() / 27.0;
            if mel >= MIN_LOG_MEL {
                MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * log_step).exp()
            } else {
                mel * F_SP
            }
        }
    }
}

fn apply_scale(values: &mut [f32], scale: &SpectralScale) {
    match *scale {
        SpectralScale::Linear => {}
        SpectralScale::NaturalLog { gain, floor } => {
            values
                .iter_mut()
                .for_each(|value| *value = gain * value.max(floor).ln());
        }
        SpectralScale::Log10 { gain, floor } => {
            values
                .iter_mut()
                .for_each(|value| *value = gain * value.max(floor).log10());
        }
    }
}

fn apply_normalization(values: &mut [f32], bins: usize, normalization: &SpectrogramNormalization) {
    match normalization {
        SpectrogramNormalization::None => {}
        SpectrogramNormalization::Range {
            min_db,
            reference_db,
            max_norm,
            symmetric,
            clipped,
        } => {
            for value in values {
                let unit = (*value - *reference_db - *min_db) / -*min_db;
                *value = if *symmetric {
                    2.0 * *max_norm * unit - *max_norm
                } else {
                    *max_norm * unit
                };
                if *clipped {
                    *value = if *symmetric {
                        value.clamp(-*max_norm, *max_norm)
                    } else {
                        value.clamp(0.0, *max_norm)
                    };
                }
            }
        }
        SpectrogramNormalization::Standardized { mean, scale } => {
            for (index, value) in values.iter_mut().enumerate() {
                let bin = index % bins;
                *value = (*value - mean[bin]) / scale[bin];
            }
        }
    }
}

fn invalid(message: impl Into<String>) -> AudioError {
    AudioError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stft_config(center: bool) -> StftConfig {
        StftConfig {
            fft_size: 32,
            window_size: 32,
            hop_size: 8,
            center,
            pad_mode: PadMode::Reflect,
            window: Window::Hann,
        }
    }

    #[test]
    fn preemphasis_round_trips() {
        let samples = [0.25, -0.5, 0.75, -1.0, 0.125];
        let emphasized = preemphasis(&samples, 0.97).unwrap();
        let restored = deemphasis(&emphasized, 0.97).unwrap();
        for (actual, expected) in restored.iter().zip(samples) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn stft_round_trips_centered_audio() {
        let samples = (0..257)
            .map(|index| (2.0 * PI * index as f32 / 31.0).sin() * 0.7)
            .collect::<Vec<_>>();
        let spectrum = stft(&samples, &stft_config(true)).unwrap();
        let restored = istft(&spectrum).unwrap();
        assert!(restored.len() >= samples.len() - 8);
        for (actual, expected) in restored.iter().zip(&samples) {
            assert!((actual - expected).abs() < 2.0e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn short_and_silent_inputs_are_finite() {
        let config = SpectrogramConfig {
            sample_rate_hz: 16_000,
            stft: stft_config(true),
            output: SpectrogramOutput::Mel(MelConfig {
                bins: 8,
                min_frequency_hz: 0.0,
                max_frequency_hz: Some(8_000.0),
                scale: MelScale::Slaney,
                normalization: MelNormalization::Slaney,
            }),
            domain: SpectralDomain::Amplitude,
            scale: SpectralScale::NaturalLog {
                gain: 1.0,
                floor: DEFAULT_SPECTRAL_FLOOR,
            },
            normalization: SpectrogramNormalization::None,
            preemphasis: None,
        };
        for samples in [&[][..], &[0.0][..], &[0.0; 7][..]] {
            let features = spectrogram(samples, &config).unwrap();
            assert!(features.values.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn stereo_downmix_resample_and_clipping_boundaries() {
        let audio = AudioBuffer {
            samples: vec![-1.0, 1.0, -0.5, 0.5, 1.0, 1.0],
            sample_rate_hz: 8_000,
            channels: 2,
        };
        assert_eq!(audio.to_mono().unwrap(), vec![0.0, 0.0, 1.0]);
        let resampled = audio.resample_linear(16_000).unwrap();
        assert_eq!(resampled.channels, 2);
        assert_eq!(resampled.frames(), 6);
        assert!(resampled
            .samples
            .iter()
            .all(|sample| (-1.0..=1.0).contains(sample)));
    }

    #[test]
    fn wav_pcm_loading_preserves_rate_channels_and_full_scale_bounds() {
        let mut bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut bytes);
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 44_100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::new(cursor, spec).unwrap();
            for sample in [i16::MIN, i16::MAX, 0, 16_384] {
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        let audio = read_wav_bytes(&bytes).unwrap();
        assert_eq!(audio.sample_rate_hz, 44_100);
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.frames(), 2);
        assert_eq!(audio.samples[0], -1.0);
        assert!(audio.samples[1] < 1.0);
        assert_eq!(audio.samples[2], 0.0);
        assert_eq!(audio.samples[3], 0.5);
    }

    #[test]
    fn all_silence_trims_without_nan() {
        assert!(trim_silence(&[0.0; 128], 60.0, 32, 8).unwrap().is_empty());
        assert_eq!(rms_normalize_db(&[0.0; 8], -27.0).unwrap(), vec![0.0; 8]);
    }

    #[test]
    fn non_power_of_two_fft_supports_freevc_speaker_features() {
        let spectrum = stft(
            &[0.1; 1_600],
            &StftConfig {
                fft_size: 400,
                window_size: 400,
                hop_size: 160,
                center: true,
                pad_mode: PadMode::Reflect,
                window: Window::Hann,
            },
        )
        .expect("rustfft supports the 400-point FreeVC speaker transform");
        assert_eq!(spectrum.bins_per_frame(), 201);
        assert!(spectrum
            .bins
            .iter()
            .all(|bin| bin.re.is_finite() && bin.im.is_finite()));
    }

    #[test]
    fn exact_feature_metadata_serializes_with_tensor() {
        let config = SpectrogramConfig {
            sample_rate_hz: 22_050,
            stft: StftConfig {
                fft_size: 1024,
                window_size: 1024,
                hop_size: 256,
                center: true,
                pad_mode: PadMode::Reflect,
                window: Window::Hann,
            },
            output: SpectrogramOutput::Mel(MelConfig {
                bins: 80,
                min_frequency_hz: 0.0,
                max_frequency_hz: Some(8_000.0),
                scale: MelScale::Slaney,
                normalization: MelNormalization::Slaney,
            }),
            domain: SpectralDomain::Amplitude,
            scale: SpectralScale::Log10 {
                gain: 20.0,
                floor: DEFAULT_SPECTRAL_FLOOR,
            },
            normalization: SpectrogramNormalization::Range {
                min_db: -100.0,
                reference_db: 20.0,
                max_norm: 4.0,
                symmetric: true,
                clipped: true,
            },
            preemphasis: Some(0.97),
        };
        let encoded = serde_json::to_string(&config).unwrap();
        let decoded: SpectrogramConfig = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, config);
    }
}
