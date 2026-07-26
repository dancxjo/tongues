//! Native FreeVC24 voice conversion.
//!
//! The network topology and inference ordering follow the MIT-licensed
//! FreeVC implementation used by Coqui TTS. PyTorch files are treated only as
//! tensor containers by the shared safe checkpoint reader; Python is not part
//! of inference.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use burn::module::{Initializer, Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig1d};
use burn::tensor::activation::{relu, sigmoid, tanh};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor, TensorData};
use tongues_audio::{
    read_wav, rms, spectrogram, trim_silence, AudioBuffer, MelConfig, MelNormalization, MelScale,
    PadMode, SpectralDomain, SpectralScale, SpectrogramConfig, SpectrogramNormalization,
    SpectrogramOutput, StftConfig, Window,
};

use crate::burn_vits_flow::CouplingWaveNet;
use crate::{
    BackendCapabilities, CapabilityValue, FreeVcConfig, InputAudioMetadata, LanguageCapabilities,
    NormalizedAudioChunk, NormalizedAudioSink, OutputAudioContract, ReferenceAudioCapabilities,
    ResidualCouplingFlow, ResidualCouplingFlowConfig, SpeakerCapabilities, SpeechDeviceRequest,
    SpeechModelFamily, StyleCapabilities, SynthesisContractError, SynthesisMetadata,
    SynthesisTiming, SynthesizerBackend, UnifiedSynthesisOutput, UnifiedSynthesisRequest,
    VitsWaveformDecoder, VitsWaveformDecoderConfig, WavLm,
};

pub const FREEVC_BACKEND_ID: &str = "freevc";
pub const FREEVC_MODEL_ID: &str = "freevc24-vctk";
pub const FREEVC_SPEAKER_EMBEDDING_SPACE: &str = "freevc-resemblyzer-speaker-v1";

#[derive(Module, Debug)]
struct FreeVcPriorEncoder<B: Backend> {
    pre: Conv1d<B>,
    enc: CouplingWaveNet<B>,
    proj: Conv1d<B>,
    inter_channels: usize,
}

impl<B: Backend> FreeVcPriorEncoder<B> {
    fn init(config: &FreeVcConfig, device: &B::Device) -> Self {
        let network = &config.model_args;
        Self {
            pre: Conv1dConfig::new(network.ssl_dim, network.hidden_channels, 1)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            enc: CouplingWaveNet::new(network.hidden_channels, 5, 1, 16, 0, device),
            proj: Conv1dConfig::new(network.hidden_channels, network.inter_channels * 2, 1)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            inter_channels: network.inter_channels,
        }
    }

    fn load_checkpoint(mut self, checkpoint: &Path) -> Result<Self> {
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut self,
            checkpoint,
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                predicate: Some(|path, _| path.starts_with("enc_p.")),
                key_remappings: vec![(r"^enc_p\.".into(), String::new())],
                map_indices_contiguous: false,
                allow_partial: true,
                skip_enum_variants: true,
            },
        )
        .context("failed to load FreeVC content encoder")?;
        let unused = result
            .unused
            .iter()
            .filter(|path| path.starts_with("enc_p."))
            .collect::<Vec<_>>();
        ensure!(
            result.missing.is_empty() && result.errors.is_empty() && unused.is_empty(),
            "FreeVC content encoder checkpoint mismatch: {} missing, {} load errors, {} unused",
            result.missing.len(),
            result.errors.len(),
            unused.len()
        );
        Ok(self)
    }

    fn sample(
        &self,
        features: Tensor<B, 3>,
        noise_scale: f64,
        seed: Option<u64>,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>)> {
        ensure!(
            noise_scale.is_finite() && noise_scale >= 0.0,
            "FreeVC noise scale must be finite and non-negative"
        );
        let [batch, _, frames] = features.dims();
        let device = features.device();
        let mask = Tensor::<B, 3>::ones([batch, 1, frames], &device);
        let hidden = self.pre.forward(features) * mask.clone();
        let statistics = self
            .proj
            .forward(self.enc.forward(hidden, mask.clone(), None))
            * mask.clone();
        let mean = statistics
            .clone()
            .slice([0..batch, 0..self.inter_channels, 0..frames]);
        let log_scale = statistics.slice([
            0..batch,
            self.inter_channels..self.inter_channels * 2,
            0..frames,
        ]);
        if let Some(seed) = seed {
            B::seed(&device, seed);
        }
        let noise = Tensor::random(
            [batch, self.inter_channels, frames],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        Ok((
            (mean + noise * log_scale.exp() * noise_scale) * mask.clone(),
            mask,
        ))
    }
}

#[derive(Module, Debug)]
struct FreeVcLstmLayer<B: Backend> {
    weight_ih: Param<Tensor<B, 2>>,
    weight_hh: Param<Tensor<B, 2>>,
    bias_ih: Param<Tensor<B, 1>>,
    bias_hh: Param<Tensor<B, 1>>,
    hidden_channels: usize,
}

impl<B: Backend> FreeVcLstmLayer<B> {
    fn init(input_channels: usize, hidden_channels: usize, device: &B::Device) -> Self {
        let initializer = Initializer::XavierUniform { gain: 1.0 };
        Self {
            weight_ih: initializer.clone().init_with(
                [hidden_channels * 4, input_channels],
                Some(input_channels),
                Some(hidden_channels * 4),
                device,
            ),
            weight_hh: initializer.init_with(
                [hidden_channels * 4, hidden_channels],
                Some(hidden_channels),
                Some(hidden_channels * 4),
                device,
            ),
            bias_ih: Initializer::Zeros.init([hidden_channels * 4], device),
            bias_hh: Initializer::Zeros.init([hidden_channels * 4], device),
            hidden_channels,
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [batch, frames, _] = input.dims();
        let device = input.device();
        let mut hidden = Tensor::zeros([batch, self.hidden_channels], &device);
        let mut cell = Tensor::zeros([batch, self.hidden_channels], &device);
        let mut outputs = Vec::with_capacity(frames);
        for frame in 0..frames {
            let value = input
                .clone()
                .slice([0..batch, frame..frame + 1, 0..input.dims()[2]])
                .reshape([batch, input.dims()[2]]);
            let gates = value.matmul(self.weight_ih.val().transpose())
                + hidden.clone().matmul(self.weight_hh.val().transpose())
                + self
                    .bias_ih
                    .val()
                    .reshape([1, self.hidden_channels * 4])
                    .repeat_dim(0, batch)
                + self
                    .bias_hh
                    .val()
                    .reshape([1, self.hidden_channels * 4])
                    .repeat_dim(0, batch);
            let channels = self.hidden_channels;
            let input_gate = sigmoid(gates.clone().slice([0..batch, 0..channels]));
            let forget_gate = sigmoid(gates.clone().slice([0..batch, channels..channels * 2]));
            let candidate = tanh(gates.clone().slice([0..batch, channels * 2..channels * 3]));
            let output_gate = sigmoid(gates.slice([0..batch, channels * 3..channels * 4]));
            cell = forget_gate * cell + input_gate * candidate;
            hidden = output_gate * tanh(cell.clone());
            outputs.push(hidden.clone().reshape([batch, 1, channels]));
        }
        (Tensor::cat(outputs, 1), hidden)
    }
}

/// FreeVC's MIT-licensed three-layer speaker encoder.
#[derive(Module, Debug)]
pub struct FreeVcSpeakerEncoder<B: Backend> {
    lstm: Vec<FreeVcLstmLayer<B>>,
    linear: Linear<B>,
    device: B::Device,
}

impl<B: Backend> FreeVcSpeakerEncoder<B> {
    pub fn load(checkpoint: impl AsRef<Path>, device: B::Device) -> Result<Self> {
        let checkpoint = checkpoint.as_ref();
        let mut model = Self {
            lstm: vec![
                FreeVcLstmLayer::init(40, 256, &device),
                FreeVcLstmLayer::init(256, 256, &device),
                FreeVcLstmLayer::init(256, 256, &device),
            ],
            linear: LinearConfig::new(256, 256).init(&device),
            device,
        };
        let mut remappings = Vec::new();
        for index in 0..3 {
            for name in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"] {
                remappings.push((
                    format!(r"^lstm\.{name}_l{index}$"),
                    format!("lstm.{index}.{name}"),
                ));
            }
        }
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut model,
            checkpoint,
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model_state"),
                predicate: Some(speaker_encoder_tensor),
                key_remappings: remappings,
                map_indices_contiguous: false,
                allow_partial: true,
                skip_enum_variants: true,
            },
        )
        .context("failed to load FreeVC speaker encoder")?;
        let unused = result
            .unused
            .iter()
            .filter(|path| speaker_encoder_tensor(path, ""))
            .collect::<Vec<_>>();
        ensure!(
            result.missing.is_empty() && result.errors.is_empty() && unused.is_empty(),
            "FreeVC speaker encoder checkpoint mismatch: {} missing, {} load errors, {} unused",
            result.missing.len(),
            result.errors.len(),
            unused.len()
        );
        Ok(model)
    }

    pub fn encode_audio(&self, audio: &AudioBuffer) -> Result<Vec<f32>> {
        audio
            .validate()
            .context("invalid speaker reference audio")?;
        let mono = audio.convert_channels(1)?.resample_linear(16_000)?;
        let trimmed = trim_silence(&mono.samples, 20.0, 2_048, 512)?;
        ensure!(!trimmed.is_empty(), "speaker reference is silent");
        let crops = speaker_partial_crops(&trimmed);
        let config = speaker_mel_config();
        let mut features = Vec::with_capacity(crops.len() * 160 * 40);
        for crop in &crops {
            let mel = spectrogram(crop, &config)?;
            ensure!(
                mel.frames >= 160,
                "speaker mel crop is shorter than 160 frames"
            );
            features.extend_from_slice(&mel.values[..160 * 40]);
        }
        let mut hidden = Tensor::from_data(
            TensorData::new(features, [crops.len(), 160, 40]),
            &self.device,
        );
        let mut final_hidden = None;
        for layer in &self.lstm {
            let (output, state) = layer.forward(hidden);
            hidden = output;
            final_hidden = Some(state);
        }
        let embeddings = relu(
            self.linear
                .forward(final_hidden.context("speaker LSTM has no layers")?),
        )
        .into_data()
        .to_vec::<f32>()
        .context("speaker embedding output is not f32")?;
        let mut mean = vec![0.0_f32; 256];
        for embedding in embeddings.chunks_exact(256) {
            for (target, value) in mean.iter_mut().zip(embedding) {
                *target += *value / crops.len() as f32;
            }
        }
        l2_normalize(&mut mean)?;
        Ok(mean)
    }
}

fn speaker_encoder_tensor(path: &str, _container: &str) -> bool {
    path.starts_with("lstm.") || path.starts_with("linear.")
}

fn speaker_partial_crops(samples: &[f32]) -> Vec<Vec<f32>> {
    const CROP_SAMPLES: usize = 160 * 160;
    const STEP_FRAMES: usize = 123;
    let frames = samples.len().div_ceil(160);
    let count = if frames <= 160 {
        1
    } else {
        (frames - 160).div_ceil(STEP_FRAMES) + 1
    };
    (0..count)
        .map(|index| {
            let start = index * STEP_FRAMES * 160;
            let mut crop = vec![0.0; CROP_SAMPLES];
            let available = samples.len().saturating_sub(start).min(CROP_SAMPLES);
            crop[..available].copy_from_slice(&samples[start..start + available]);
            crop
        })
        .collect()
}

fn speaker_mel_config() -> SpectrogramConfig {
    SpectrogramConfig {
        sample_rate_hz: 16_000,
        stft: StftConfig {
            fft_size: 400,
            window_size: 400,
            hop_size: 160,
            center: true,
            pad_mode: PadMode::Reflect,
            window: Window::Hann,
        },
        output: SpectrogramOutput::Mel(MelConfig {
            bins: 40,
            min_frequency_hz: 0.0,
            max_frequency_hz: 8_000.0.into(),
            scale: MelScale::Slaney,
            normalization: MelNormalization::Slaney,
        }),
        domain: SpectralDomain::Power,
        scale: SpectralScale::Linear,
        normalization: SpectrogramNormalization::None,
        preemphasis: None,
    }
}

fn l2_normalize(values: &mut [f32]) -> Result<()> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    ensure!(
        norm.is_finite() && norm > f32::EPSILON,
        "speaker embedding has zero or invalid norm"
    );
    for value in values {
        *value /= norm;
    }
    Ok(())
}

/// Fully native FreeVC runtime.
#[derive(Module, Debug)]
pub struct FreeVc<B: Backend> {
    config: FreeVcConfig,
    wavlm: WavLm<B>,
    prior: FreeVcPriorEncoder<B>,
    flow: ResidualCouplingFlow<B>,
    decoder: VitsWaveformDecoder<B>,
    speaker_encoder: FreeVcSpeakerEncoder<B>,
    device: B::Device,
}

impl<B: Backend> FreeVc<B> {
    pub fn load(model_dir: impl AsRef<Path>, device: B::Device) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let model_path = model_dir.join("model.pth");
        let wavlm_path = model_dir.join("WavLM-Large.pt");
        let speaker_path = model_dir.join("speaker_encoder.pt");
        for path in [&config_path, &model_path, &wavlm_path, &speaker_path] {
            ensure!(
                path.is_file(),
                "required FreeVC file is missing: {}",
                path.display()
            );
        }
        let config = FreeVcConfig::from_file(config_path)?;
        let wavlm = WavLm::load_large(wavlm_path, &device)?;
        let prior = FreeVcPriorEncoder::init(&config, &device).load_checkpoint(&model_path)?;
        let flow = ResidualCouplingFlowConfig {
            channels: config.model_args.inter_channels,
            hidden_channels: config.model_args.hidden_channels,
            kernel_size: 5,
            dilation_rate: 1,
            num_layers: 4,
            num_flows: 4,
            conditioning_channels: config.model_args.gin_channels,
        }
        .load_checkpoint(&model_path, &device)?;
        let decoder = VitsWaveformDecoderConfig::from_generator_config(config.decoder_config())?
            .init(&device)?
            .load_checkpoint_subtree(&model_path, "dec")?;
        let speaker_encoder = FreeVcSpeakerEncoder::load(speaker_path, device.clone())?;
        Ok(Self {
            config,
            wavlm,
            prior,
            flow,
            decoder,
            speaker_encoder,
            device,
        })
    }

    pub fn convert_audio(
        &self,
        source: &AudioBuffer,
        target: &AudioBuffer,
        noise_scale: f64,
        seed: Option<u64>,
    ) -> Result<Vec<f32>> {
        source.validate().context("invalid source audio")?;
        target.validate().context("invalid target speaker audio")?;
        let source = source
            .convert_channels(1)?
            .resample_linear(self.config.audio.input_sample_rate)?;
        ensure!(source.samples.len() >= 400, "source audio is too short");
        let source_frames = source.frames();
        let source_tensor = Tensor::from_data(
            TensorData::new(source.samples, [1, source_frames]),
            &self.device,
        );
        let content = self.wavlm.extract_features(source_tensor)?;
        let speaker = self.speaker_encoder.encode_audio(target)?;
        let conditioning = Tensor::from_data(
            TensorData::new(speaker, [1, self.config.model_args.gin_channels, 1]),
            &self.device,
        );
        let (latent, mask) = self.prior.sample(content, noise_scale, seed)?;
        let latent = self
            .flow
            .reverse(latent, mask, Some(conditioning.clone()))?;
        let output = self.decoder.forward(latent, Some(conditioning))?;
        let output_values: usize = output.dims().iter().product();
        let output = output
            .reshape([output_values])
            .into_data()
            .to_vec::<f32>()
            .context("FreeVC waveform output is not f32")?;
        ensure!(
            !output.is_empty() && output.iter().all(|value| value.is_finite()),
            "FreeVC produced empty or non-finite audio"
        );
        Ok(output)
    }

    pub fn output_sample_rate_hz(&self) -> u32 {
        self.config.audio.output_sample_rate
    }
}

impl<B> SynthesizerBackend for FreeVc<B>
where
    B: Backend + Send,
    B::Device: Send,
{
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            backend: FREEVC_BACKEND_ID.into(),
            model: FREEVC_MODEL_ID.into(),
            family: SpeechModelFamily::VoiceConversion,
            varieties: CapabilityValue::Any,
            languages: LanguageCapabilities::unsupported(),
            speakers: SpeakerCapabilities::unsupported(),
            styles: StyleCapabilities::unsupported(),
            reference_audio: ReferenceAudioCapabilities {
                speaker: true,
                source: true,
                speaker_required: true,
                source_required: true,
                ..Default::default()
            },
            speed: false,
            pitch: Default::default(),
            energy: Default::default(),
            durations: false,
            seed: true,
            devices: vec![SpeechDeviceRequest::Cpu],
            output: OutputAudioContract {
                sample_rate_hz: self.config.audio.output_sample_rate,
                channels: 1,
                streaming: false,
            },
            provenance: vec![
                "coqui-ai/TTS@fa28f99f1508b5b5366539b2149963edcb80ba62".into(),
                "microsoft/unilm WavLM".into(),
                "OlaWod/FreeVC".into(),
            ],
        }
    }

    fn synthesize(
        &mut self,
        request: &UnifiedSynthesisRequest,
        sink: &mut dyn NormalizedAudioSink,
    ) -> Result<UnifiedSynthesisOutput, SynthesisContractError> {
        let capabilities = self.capabilities();
        capabilities.validate(request)?;
        let source_uri = request.reference_audio.source.as_deref().ok_or_else(|| {
            SynthesisContractError::MissingRequiredFeature {
                backend: FREEVC_BACKEND_ID.into(),
                feature: "reference_audio.source",
            }
        })?;
        let target_uri = request.reference_audio.speaker.as_deref().ok_or_else(|| {
            SynthesisContractError::MissingRequiredFeature {
                backend: FREEVC_BACKEND_ID.into(),
                feature: "reference_audio.speaker",
            }
        })?;
        let source_path = local_audio_path(source_uri).map_err(contract_backend)?;
        let target_path = local_audio_path(target_uri).map_err(contract_backend)?;
        let source = read_wav(&source_path).map_err(contract_backend)?;
        let target = read_wav(&target_path).map_err(contract_backend)?;
        let input_audio = vec![
            audio_metadata("source", &source),
            audio_metadata("target-speaker", &target),
        ];
        let started = Instant::now();
        let pcm_f32 = self
            .convert_audio(
                &source,
                &target,
                f64::from(request.noise_scale.unwrap_or(1.0)),
                request.seed,
            )
            .map_err(contract_backend)?;
        let frames = pcm_f32.len() as u64;
        sink.emit(NormalizedAudioChunk {
            chunk_index: 0,
            is_final: true,
            frame_offset: 0,
            sample_rate_hz: self.config.audio.output_sample_rate,
            channels: 1,
            pcm_f32,
        })?;
        Ok(UnifiedSynthesisOutput {
            metadata: SynthesisMetadata {
                backend: FREEVC_BACKEND_ID.into(),
                model: FREEVC_MODEL_ID.into(),
                sample_rate_hz: self.config.audio.output_sample_rate,
                channels: 1,
                frames,
                audio_seconds: frames as f64 / f64::from(self.config.audio.output_sample_rate),
                streaming: false,
                input_audio,
                timings: vec![SynthesisTiming {
                    stage: "total".into(),
                    elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
                }],
            },
        })
    }
}

fn local_audio_path(uri: &str) -> Result<PathBuf> {
    if let Some(path) = uri.strip_prefix("file://") {
        return Ok(PathBuf::from(path));
    }
    if uri.contains("://") {
        bail!("reference-audio URI scheme is unsupported: {uri}");
    }
    Ok(PathBuf::from(uri))
}

fn audio_metadata(role: &str, audio: &AudioBuffer) -> InputAudioMetadata {
    let level = rms(&audio.samples);
    InputAudioMetadata {
        role: role.into(),
        sample_rate_hz: audio.sample_rate_hz,
        channels: audio.channels,
        frames: audio.frames() as u64,
        rms_dbfs: if level > f32::EPSILON {
            20.0 * level.log10()
        } else {
            -120.0
        },
    }
}

fn contract_backend(error: impl std::fmt::Display) -> SynthesisContractError {
    SynthesisContractError::Backend {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    use super::*;

    #[test]
    fn partial_crops_are_fixed_and_cover_short_audio() {
        let crops = speaker_partial_crops(&vec![0.25; 8_000]);
        assert_eq!(crops.len(), 1);
        assert_eq!(crops[0].len(), 25_600);
        assert_eq!(crops[0][7_999], 0.25);
        assert_eq!(crops[0][8_000], 0.0);
    }

    #[test]
    fn input_metadata_preserves_original_audio_properties() {
        let audio = AudioBuffer {
            samples: vec![0.5; 960],
            sample_rate_hz: 48_000,
            channels: 2,
        };
        let metadata = audio_metadata("source", &audio);
        assert_eq!(metadata.sample_rate_hz, 48_000);
        assert_eq!(metadata.channels, 2);
        assert_eq!(metadata.frames, 480);
        assert!((metadata.rms_dbfs + 6.0206).abs() < 0.001);
    }

    #[test]
    #[ignore = "requires the licensed FreeVC speaker_encoder.pt artifact"]
    fn published_speaker_encoder_loads_and_separates_fixtures() {
        let checkpoint =
            std::env::var("TONGUES_FREEVC_SPEAKER_CHECKPOINT").expect("speaker checkpoint");
        let same = std::env::var("TONGUES_FREEVC_SAME_SPEAKER_WAV").expect("same-speaker WAV");
        let different =
            std::env::var("TONGUES_FREEVC_DIFFERENT_SPEAKER_WAV").expect("cross-speaker WAV");
        let encoder =
            FreeVcSpeakerEncoder::<NdArray<f32>>::load(checkpoint, NdArrayDevice::Cpu).unwrap();
        let first = encoder.encode_audio(&read_wav(&same).unwrap()).unwrap();
        let repeated = encoder.encode_audio(&read_wav(&same).unwrap()).unwrap();
        let cross = encoder
            .encode_audio(&read_wav(&different).unwrap())
            .unwrap();
        let same_score = first.iter().zip(repeated).map(|(a, b)| a * b).sum::<f32>();
        let cross_score = first.iter().zip(cross).map(|(a, b)| a * b).sum::<f32>();
        assert!(same_score > 0.999);
        assert!(
            same_score > cross_score,
            "same-speaker similarity {same_score} must exceed cross-speaker {cross_score}"
        );
    }

    #[test]
    #[ignore = "requires the complete licensed FreeVC24 artifact set"]
    fn published_artifacts_convert_without_python() {
        let model_dir = std::env::var("TONGUES_FREEVC_MODEL_DIR").expect("FreeVC model directory");
        let source = std::env::var("TONGUES_FREEVC_SOURCE_WAV").expect("source WAV");
        let target = std::env::var("TONGUES_FREEVC_TARGET_WAV").expect("target WAV");
        let runtime = FreeVc::<NdArray<f32>>::load(model_dir, NdArrayDevice::Cpu).unwrap();
        let output = runtime
            .convert_audio(
                &read_wav(&source).unwrap(),
                &read_wav(&target).unwrap(),
                0.0,
                Some(38),
            )
            .unwrap();
        assert!(output.len() > 24_000);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(rms(&output) > 1.0e-5);

        let source_embedding = runtime
            .speaker_encoder
            .encode_audio(&read_wav(&source).unwrap())
            .unwrap();
        let target_embedding = runtime
            .speaker_encoder
            .encode_audio(&read_wav(&target).unwrap())
            .unwrap();
        let converted_embedding = runtime
            .speaker_encoder
            .encode_audio(&AudioBuffer {
                samples: output.clone(),
                sample_rate_hz: runtime.output_sample_rate_hz(),
                channels: 1,
            })
            .unwrap();
        let converted_to_target = converted_embedding
            .iter()
            .zip(&target_embedding)
            .map(|(a, b)| a * b)
            .sum::<f32>();
        let converted_to_source = converted_embedding
            .iter()
            .zip(&source_embedding)
            .map(|(a, b)| a * b)
            .sum::<f32>();
        assert!(
            converted_to_target > converted_to_source,
            "converted target similarity {converted_to_target} must exceed source similarity {converted_to_source}"
        );

        let source_content = content_summary(&runtime, &read_wav(&source).unwrap());
        let unrelated_content = content_summary(&runtime, &read_wav(&target).unwrap());
        let converted_content = content_summary(
            &runtime,
            &AudioBuffer {
                samples: output,
                sample_rate_hz: runtime.output_sample_rate_hz(),
                channels: 1,
            },
        );
        let source_content_similarity = source_content
            .iter()
            .zip(&converted_content)
            .map(|(a, b)| a * b)
            .sum::<f32>();
        let unrelated_content_similarity = unrelated_content
            .iter()
            .zip(converted_content)
            .map(|(a, b)| a * b)
            .sum::<f32>();
        assert!(
            source_content_similarity > 0.25
                && source_content_similarity > unrelated_content_similarity,
            "converted source-content similarity {source_content_similarity} must exceed the unrelated reference {unrelated_content_similarity}"
        );
    }

    fn content_summary(runtime: &FreeVc<NdArray<f32>>, audio: &AudioBuffer) -> Vec<f32> {
        let mono = audio
            .convert_channels(1)
            .unwrap()
            .resample_linear(16_000)
            .unwrap();
        let frames = mono.frames();
        let features = runtime
            .wavlm
            .extract_features(Tensor::from_data(
                TensorData::new(mono.samples, [1, frames]),
                &runtime.device,
            ))
            .unwrap()
            .mean_dim(2)
            .reshape([1_024])
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let mut features = features;
        l2_normalize(&mut features).unwrap();
        features
    }
}
