//! Burn-native Coqui ResNet speaker encoder and neutral enrollment utilities.
//!
//! The checkpoint-compatible topology and crop semantics follow Coqui TTS
//! v0.6.1 `TTS/speaker_encoder/models/resnet.py` (MPL-2.0), revision
//! `0cf3265a4686d7e856bd472cdaf1572d61cab2b8`. This file is an MPL-2.0
//! covered modification. The embedding-space identity remains explicit so a
//! d-vector from another encoder cannot silently condition YourTTS.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn::nn::{
    BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig1d, PaddingConfig2d,
};
use burn::tensor::activation::{relu, sigmoid, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use sha2::{Digest, Sha256};
use tongues_audio::{
    read_wav, rms_normalize_db, spectrogram, AudioBuffer, MelConfig, MelNormalization, MelScale,
    PadMode, SpectralDomain, SpectralScale, SpectrogramConfig, SpectrogramNormalization,
    SpectrogramOutput, StftConfig, Window,
};

use crate::d_vectors::{l2_normalize, COQUI_RESNET_SPEAKER_EMBEDDING_SPACE};
use crate::{
    ConditioningEmbedding, ConditioningKind, EmbeddingContract, SpeakerEncoderPackageConfig,
};

const DEFAULT_CROP_FRAMES: usize = 250;
const DEFAULT_EVAL_CROPS: usize = 10;

#[derive(Module, Debug)]
struct SeLayer<B: Backend> {
    fc_0: Linear<B>,
    fc_2: Linear<B>,
}

impl<B: Backend> SeLayer<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            fc_0: LinearConfig::new(channels, channels / 8).init(device),
            fc_2: LinearConfig::new(channels / 8, channels).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, _, _] = input.dims();
        let pooled = input.clone().mean_dims(&[2, 3]).reshape([batch, channels]);
        let scale = sigmoid(self.fc_2.forward(relu(self.fc_0.forward(pooled))))
            .reshape([batch, channels, 1, 1]);
        input * scale
    }
}

#[derive(Module, Debug)]
struct Downsample<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
}

impl<B: Backend> Downsample<B> {
    fn init(channels_in: usize, channels_out: usize, stride: usize, device: &B::Device) -> Self {
        Self {
            conv: Conv2dConfig::new([channels_in, channels_out], [1, 1])
                .with_stride([stride, stride])
                .with_bias(false)
                .init(device),
            bn: BatchNormConfig::new(channels_out).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.bn.forward(self.conv.forward(input))
    }
}

#[derive(Module, Debug)]
struct SeBasicBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    se: SeLayer<B>,
    downsample: Option<Downsample<B>>,
}

impl<B: Backend> SeBasicBlock<B> {
    fn init(channels_in: usize, channels_out: usize, stride: usize, device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([channels_in, channels_out], [3, 3])
                .with_stride([stride, stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false)
                .init(device),
            bn1: BatchNormConfig::new(channels_out).init(device),
            conv2: Conv2dConfig::new([channels_out, channels_out], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false)
                .init(device),
            bn2: BatchNormConfig::new(channels_out).init(device),
            se: SeLayer::init(channels_out, device),
            downsample: (stride != 1 || channels_in != channels_out)
                .then(|| Downsample::init(channels_in, channels_out, stride, device)),
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = match &self.downsample {
            Some(downsample) => downsample.forward(input.clone()),
            None => input.clone(),
        };
        let output = self.bn1.forward(relu(self.conv1.forward(input)));
        let output = self.bn2.forward(self.conv2.forward(output));
        relu(self.se.forward(output) + residual)
    }
}

fn init_layer<B: Backend>(
    blocks: usize,
    channels_in: usize,
    channels_out: usize,
    stride: usize,
    device: &B::Device,
) -> Vec<SeBasicBlock<B>> {
    let mut layer = Vec::with_capacity(blocks);
    layer.push(SeBasicBlock::init(
        channels_in,
        channels_out,
        stride,
        device,
    ));
    for _ in 1..blocks {
        layer.push(SeBasicBlock::init(channels_out, channels_out, 1, device));
    }
    layer
}

fn forward_layer<B: Backend>(layer: &[SeBasicBlock<B>], mut input: Tensor<B, 4>) -> Tensor<B, 4> {
    for block in layer {
        input = block.forward(input);
    }
    input
}

/// Coqui's ResNet H/ASP encoder with checkpoint-compatible module names.
#[derive(Module, Debug)]
pub(crate) struct CoquiResNetSpeakerEncoder<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    layer1: Vec<SeBasicBlock<B>>,
    layer2: Vec<SeBasicBlock<B>>,
    layer3: Vec<SeBasicBlock<B>>,
    layer4: Vec<SeBasicBlock<B>>,
    attention_0: Conv1d<B>,
    attention_2: BatchNorm<B>,
    attention_3: Conv1d<B>,
    fc: Linear<B>,
    input_dim: usize,
    projection_dim: usize,
}

impl<B: Backend> CoquiResNetSpeakerEncoder<B> {
    pub fn init(config: &SpeakerEncoderPackageConfig, device: &B::Device) -> Result<Self> {
        ensure!(
            config.model_name.eq_ignore_ascii_case("resnet"),
            "native ResNet speaker encoder cannot load model_name `{}`",
            config.model_name
        );
        ensure!(
            config.layers.len() == 4 && config.num_filters.len() == 4,
            "ResNet speaker encoder requires four layer/filter stages"
        );
        let filters = &config.num_filters;
        let attention_channels = filters[3] * (config.input_dim / 8);
        ensure!(
            attention_channels > 0,
            "speaker encoder input dimension is too small"
        );
        Ok(Self {
            conv1: Conv2dConfig::new([1, filters[0]], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            bn1: BatchNormConfig::new(filters[0]).init(device),
            layer1: init_layer(config.layers[0], filters[0], filters[0], 1, device),
            layer2: init_layer(config.layers[1], filters[0], filters[1], 2, device),
            layer3: init_layer(config.layers[2], filters[1], filters[2], 2, device),
            layer4: init_layer(config.layers[3], filters[2], filters[3], 2, device),
            attention_0: Conv1dConfig::new(attention_channels, 128, 1)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            attention_2: BatchNormConfig::new(128).init(device),
            attention_3: Conv1dConfig::new(128, attention_channels, 1)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            fc: LinearConfig::new(attention_channels * 2, config.projection_dim).init(device),
            input_dim: config.input_dim,
            projection_dim: config.projection_dim,
        })
    }

    pub fn load(
        config: &SpeakerEncoderPackageConfig,
        checkpoint_path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<Self> {
        let mut model = Self::init(config, device)?;
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut model,
            checkpoint_path.as_ref(),
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                predicate: Some(speaker_model_tensor),
                key_remappings: vec![
                    (r"\.se\.fc\.0\.".into(), ".se.fc_0.".into()),
                    (r"\.se\.fc\.2\.".into(), ".se.fc_2.".into()),
                    (r"\.downsample\.0\.".into(), ".downsample.conv.".into()),
                    (r"\.downsample\.1\.".into(), ".downsample.bn.".into()),
                    (r"^attention\.0\.".into(), "attention_0.".into()),
                    (r"^attention\.2\.".into(), "attention_2.".into()),
                    (r"^attention\.3\.".into(), "attention_3.".into()),
                    (r"(^|.*\.)(bn1|bn2)\.weight$".into(), "$1$2.gamma".into()),
                    (r"(^|.*\.)(bn1|bn2)\.bias$".into(), "$1$2.beta".into()),
                    (r"(\.downsample\.bn)\.weight$".into(), "$1.gamma".into()),
                    (r"(\.downsample\.bn)\.bias$".into(), "$1.beta".into()),
                    (r"^attention_2\.weight$".into(), "attention_2.gamma".into()),
                    (r"^attention_2\.bias$".into(), "attention_2.beta".into()),
                ],
                map_indices_contiguous: true,
                allow_partial: true,
                skip_enum_variants: true,
            },
        )
        .context("failed to load Coqui ResNet speaker encoder checkpoint")?;
        let unused = result
            .unused
            .iter()
            .filter(|name| speaker_model_tensor(name, ""))
            .collect::<Vec<_>>();
        ensure!(
            result.missing.is_empty() && result.errors.is_empty() && unused.is_empty(),
            "speaker encoder checkpoint mismatch: {} missing, {} load errors, unused [{}]",
            result.missing.len(),
            result.errors.len(),
            unused
                .into_iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(model)
    }

    /// Encode a batch of `[batch, mel, frames]` power-mel spectrograms.
    pub fn forward(&self, input: Tensor<B, 3>, l2_norm: bool) -> Tensor<B, 2> {
        let [batch, mel, frames] = input.dims();
        assert_eq!(mel, self.input_dim);
        let mean = input.clone().mean_dim(2);
        let variance = (input.clone() - mean.clone()).square().mean_dim(2);
        let input = (input - mean) / (variance + 1.0e-5).sqrt();

        let mut output = self.bn1.forward(relu(
            self.conv1.forward(input.reshape([batch, 1, mel, frames])),
        ));
        output = forward_layer(&self.layer1, output);
        output = forward_layer(&self.layer2, output);
        output = forward_layer(&self.layer3, output);
        output = forward_layer(&self.layer4, output);

        let [batch, channels, frequency, frames] = output.dims();
        let output = output.reshape([batch, channels * frequency, frames]);
        let weights = softmax(
            self.attention_3.forward(
                self.attention_2
                    .forward(relu(self.attention_0.forward(output.clone()))),
            ),
            2,
        );
        let mean = (output.clone() * weights.clone())
            .sum_dim(2)
            .reshape([batch, channels * frequency]);
        let deviation = ((output.square() * weights)
            .sum_dim(2)
            .reshape([batch, channels * frequency])
            - mean.clone().square())
        .clamp_min(1.0e-5)
        .sqrt();
        let embedding = self.fc.forward(Tensor::cat(vec![mean, deviation], 1));
        if l2_norm {
            let norm = embedding
                .clone()
                .square()
                .sum_dim(1)
                .sqrt()
                .clamp_min(1.0e-12);
            embedding / norm
        } else {
            embedding
        }
    }

    #[cfg(test)]
    fn projection_dim(&self) -> usize {
        self.projection_dim
    }
}

fn speaker_model_tensor(path: &str, _container: &str) -> bool {
    !path.starts_with("torch_spec.") && !path.ends_with("num_batches_tracked")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerEmbeddingCachePolicy {
    Disabled,
    Memory { max_entries: usize },
}

/// Runtime wrapper that owns preprocessing, enrollment, and an embedding-only
/// cache. Reference PCM is never retained after encoding.
pub struct NativeSpeakerEmbeddingService<B: Backend> {
    encoder: CoquiResNetSpeakerEncoder<B>,
    config: SpeakerEncoderPackageConfig,
    contract: EmbeddingContract,
    cache_policy: SpeakerEmbeddingCachePolicy,
    cache: HashMap<[u8; 32], ConditioningEmbedding>,
    cache_order: Vec<[u8; 32]>,
    device: B::Device,
}

impl<B: Backend> NativeSpeakerEmbeddingService<B> {
    pub fn load(
        config: SpeakerEncoderPackageConfig,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
        cache_policy: SpeakerEmbeddingCachePolicy,
    ) -> Result<Self> {
        ensure!(
            config.use_torch_spec,
            "reference-audio service requires a speaker encoder with `use_torch_spec=true`"
        );
        let encoder = CoquiResNetSpeakerEncoder::load(&config, checkpoint_path, &device)?;
        let contract = EmbeddingContract {
            kind: ConditioningKind::Speaker,
            space: COQUI_RESNET_SPEAKER_EMBEDDING_SPACE.into(),
            dimensions: config.projection_dim,
            l2_normalized: true,
        };
        contract.validate()?;
        Ok(Self {
            encoder,
            config,
            contract,
            cache_policy,
            cache: HashMap::new(),
            cache_order: Vec::new(),
            device,
        })
    }

    pub fn output_contract(&self) -> &EmbeddingContract {
        &self.contract
    }

    pub fn encode_path(&mut self, path: impl AsRef<Path>) -> Result<ConditioningEmbedding> {
        let path = path.as_ref();
        let audio = read_wav(path)
            .with_context(|| format!("failed to read speaker reference {}", path.display()))?;
        self.encode_audio(&audio)
            .with_context(|| format!("failed to encode speaker reference {}", path.display()))
    }

    pub fn encode_uri(&mut self, uri: &str) -> Result<ConditioningEmbedding> {
        let path = if let Some(path) = uri.strip_prefix("file://") {
            PathBuf::from(path)
        } else if uri.contains("://") {
            bail!("reference-audio URI scheme is unsupported: {uri}");
        } else {
            PathBuf::from(uri)
        };
        self.encode_path(path)
    }

    pub fn encode_audio(&mut self, audio: &AudioBuffer) -> Result<ConditioningEmbedding> {
        audio
            .validate()
            .context("invalid speaker reference audio")?;
        let key = audio_cache_key(audio, &self.config);
        if let Some(embedding) = self.cache.get(&key) {
            return Ok(embedding.clone());
        }
        let samples = audio
            .convert_channels(1)?
            .resample_linear(self.config.sample_rate_hz)?
            .samples;
        ensure!(!samples.is_empty(), "speaker reference audio is empty");
        let samples = rms_normalize_db(&samples, -27.0)?;
        let batch = speaker_crops(&samples, self.config.hop_size.unwrap_or(160));
        let mut features = Vec::new();
        let mut frames = None;
        for crop in &batch {
            let feature = speaker_spectrogram(crop, &self.config)?;
            frames = Some(feature.frames);
            for mel in 0..self.config.input_dim {
                for frame in 0..feature.frames {
                    let value = feature.values[frame * self.config.input_dim + mel];
                    features.push(if self.config.log_input {
                        (value + 1.0e-6).ln()
                    } else {
                        value
                    });
                }
            }
        }
        let frames = frames.context("speaker encoder produced no crops")?;
        let input = Tensor::from_data(
            TensorData::new(features, [batch.len(), self.config.input_dim, frames]),
            &self.device,
        );
        let embeddings = self
            .encoder
            .forward(input, true)
            .into_data()
            .to_vec::<f32>()
            .context("speaker encoder output is not f32")?;
        let mut values = vec![0.0; self.contract.dimensions];
        for embedding in embeddings.chunks_exact(self.contract.dimensions) {
            for (mean, value) in values.iter_mut().zip(embedding) {
                *mean += *value / batch.len() as f32;
            }
        }
        l2_normalize(&mut values)?;
        let embedding = ConditioningEmbedding {
            contract: self.contract.clone(),
            values,
        };
        embedding.validate()?;
        self.cache_embedding(key, embedding.clone());
        Ok(embedding)
    }

    pub fn enroll_paths(
        &mut self,
        paths: impl IntoIterator<Item = impl AsRef<Path>>,
    ) -> Result<ConditioningEmbedding> {
        let embeddings = paths
            .into_iter()
            .map(|path| self.encode_path(path))
            .collect::<Result<Vec<_>>>()?;
        average_embeddings(&embeddings, &self.contract)
    }

    fn cache_embedding(&mut self, key: [u8; 32], embedding: ConditioningEmbedding) {
        let SpeakerEmbeddingCachePolicy::Memory { max_entries } = self.cache_policy else {
            return;
        };
        if max_entries == 0 {
            return;
        }
        while self.cache_order.len() >= max_entries {
            let oldest = self.cache_order.remove(0);
            self.cache.remove(&oldest);
        }
        self.cache_order.push(key);
        self.cache.insert(key, embedding);
    }
}

fn speaker_crops(samples: &[f32], hop_size: usize) -> Vec<Vec<f32>> {
    let crop_len = (DEFAULT_CROP_FRAMES * hop_size).min(samples.len());
    let span = samples.len() - crop_len;
    (0..DEFAULT_EVAL_CROPS)
        .map(|index| {
            let offset = if DEFAULT_EVAL_CROPS == 1 {
                0
            } else {
                (index as f64 * span as f64 / (DEFAULT_EVAL_CROPS - 1) as f64) as usize
            };
            samples[offset..offset + crop_len].to_vec()
        })
        .collect()
}

fn speaker_spectrogram(
    samples: &[f32],
    config: &SpeakerEncoderPackageConfig,
) -> Result<tongues_audio::Spectrogram> {
    let fft_size = config.fft_size.unwrap_or(512);
    let window_size = config.window_size.unwrap_or(400);
    let hop_size = config.hop_size.unwrap_or(160);
    let emphasized = reflect_preemphasis(samples, 0.97);
    spectrogram(
        &emphasized,
        &SpectrogramConfig {
            sample_rate_hz: config.sample_rate_hz,
            stft: StftConfig {
                fft_size,
                window_size,
                hop_size,
                center: true,
                pad_mode: PadMode::Reflect,
                window: Window::Hamming,
            },
            output: SpectrogramOutput::Mel(MelConfig {
                bins: config.input_dim,
                min_frequency_hz: 0.0,
                max_frequency_hz: Some(config.sample_rate_hz as f32 / 2.0),
                scale: MelScale::Htk,
                normalization: MelNormalization::None,
            }),
            domain: SpectralDomain::Power,
            scale: SpectralScale::Linear,
            normalization: SpectrogramNormalization::None,
            preemphasis: None,
        },
    )
    .context("failed to compute Coqui speaker spectrogram")
}

fn reflect_preemphasis(samples: &[f32], coefficient: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let reflected = samples.get(1).copied().unwrap_or(samples[0]);
    let mut output = Vec::with_capacity(samples.len());
    output.push(samples[0] - coefficient * reflected);
    output.extend(
        samples
            .windows(2)
            .map(|window| window[1] - coefficient * window[0]),
    );
    output
}

fn audio_cache_key(audio: &AudioBuffer, config: &SpeakerEncoderPackageConfig) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(audio.sample_rate_hz.to_le_bytes());
    digest.update(audio.channels.to_le_bytes());
    digest.update(config.sample_rate_hz.to_le_bytes());
    digest.update(config.projection_dim.to_le_bytes());
    for sample in &audio.samples {
        digest.update(sample.to_bits().to_le_bytes());
    }
    digest.finalize().into()
}

pub fn average_embeddings(
    embeddings: &[ConditioningEmbedding],
    expected: &EmbeddingContract,
) -> Result<ConditioningEmbedding> {
    ensure!(!embeddings.is_empty(), "speaker enrollment has no clips");
    let mut values = vec![0.0; expected.dimensions];
    for embedding in embeddings {
        embedding.validate()?;
        ensure!(
            &embedding.contract == expected,
            "speaker embedding contract does not match enrollment encoder"
        );
        for (mean, value) in values.iter_mut().zip(&embedding.values) {
            *mean += *value / embeddings.len() as f32;
        }
    }
    l2_normalize(&mut values)?;
    Ok(ConditioningEmbedding {
        contract: expected.clone(),
        values,
    })
}

pub fn cosine_similarity(
    left: &ConditioningEmbedding,
    right: &ConditioningEmbedding,
) -> Result<f32> {
    left.validate()?;
    right.validate()?;
    ensure!(
        left.contract == right.contract,
        "cannot compare embeddings from different contracts"
    );
    let mut left_values = left.values.clone();
    let mut right_values = right.values.clone();
    l2_normalize(&mut left_values)?;
    l2_normalize(&mut right_values)?;
    Ok(left_values
        .iter()
        .zip(right_values)
        .map(|(left, right)| left * right)
        .sum())
}

/// Generalized end-to-end angular prototypical loss over already encoded
/// `[speaker][utterance][dimension]` embeddings.
pub fn angular_prototypical_loss(embeddings: &[Vec<Vec<f32>>]) -> Result<f32> {
    ensure!(
        embeddings.len() >= 2,
        "angular prototypical loss requires at least two speakers"
    );
    let utterances = embeddings[0].len();
    ensure!(
        utterances >= 2,
        "angular prototypical loss requires at least two utterances per speaker"
    );
    let dimensions = embeddings[0][0].len();
    ensure!(dimensions > 0, "speaker embeddings cannot be empty");
    for speaker in embeddings {
        ensure!(
            speaker.len() == utterances
                && speaker
                    .iter()
                    .all(|embedding| embedding.len() == dimensions),
            "angular prototypical batches must be rectangular"
        );
    }

    let centroids = embeddings
        .iter()
        .map(|speaker| {
            let mut centroid = vec![0.0; dimensions];
            for embedding in speaker.iter().take(utterances - 1) {
                for (mean, value) in centroid.iter_mut().zip(embedding) {
                    *mean += *value / (utterances - 1) as f32;
                }
            }
            l2_normalize(&mut centroid)?;
            Ok(centroid)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut loss = 0.0;
    for (target, speaker) in embeddings.iter().enumerate() {
        let mut query = speaker[utterances - 1].clone();
        l2_normalize(&mut query)?;
        let logits = centroids
            .iter()
            .map(|centroid| {
                query
                    .iter()
                    .zip(centroid)
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator = logits
            .iter()
            .map(|value| (*value - maximum).exp())
            .sum::<f32>();
        loss -= logits[target] - maximum - denominator.ln();
    }
    Ok(loss / embeddings.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    fn embedding(values: Vec<f32>) -> ConditioningEmbedding {
        ConditioningEmbedding {
            contract: EmbeddingContract {
                kind: ConditioningKind::Speaker,
                space: "fixture".into(),
                dimensions: values.len(),
                l2_normalized: true,
            },
            values,
        }
    }

    #[test]
    fn crop_offsets_match_evenly_spaced_coqui_semantics() {
        let samples = (0..50_000).map(|value| value as f32).collect::<Vec<_>>();
        let crops = speaker_crops(&samples, 160);
        assert_eq!(crops.len(), 10);
        assert_eq!(crops[0][0], 0.0);
        assert_eq!(crops[9][0], 10_000.0);
        assert!(crops.iter().all(|crop| crop.len() == 40_000));
    }

    #[test]
    fn enrollment_and_verification_keep_contracts_explicit() {
        let first = embedding(vec![1.0, 0.0]);
        let second = embedding(vec![0.0, 1.0]);
        let enrolled = average_embeddings(&[first.clone(), second], &first.contract).unwrap();
        assert!((enrolled.values[0] - 2_f32.sqrt().recip()).abs() < 1.0e-6);
        assert!(
            (cosine_similarity(&first, &enrolled).unwrap() - 2_f32.sqrt().recip()).abs() < 1.0e-6
        );
    }

    #[test]
    fn angular_loss_prefers_correct_speaker_centroids() {
        let separated = vec![
            vec![vec![1.0, 0.0], vec![0.9, 0.1]],
            vec![vec![0.0, 1.0], vec![0.1, 0.9]],
        ];
        let confused = vec![
            vec![vec![1.0, 0.0], vec![0.1, 0.9]],
            vec![vec![0.0, 1.0], vec![0.9, 0.1]],
        ];
        assert!(
            angular_prototypical_loss(&separated).unwrap()
                < angular_prototypical_loss(&confused).unwrap()
        );
    }

    #[test]
    fn published_resnet_checkpoint_loads_when_available() {
        let (Some(config), Some(checkpoint)) = (
            std::env::var_os("TONGUES_TEST_COQUI_SPEAKER_CONFIG"),
            std::env::var_os("TONGUES_TEST_COQUI_SPEAKER_MODEL"),
        ) else {
            return;
        };
        let config = SpeakerEncoderPackageConfig::from_file(config).unwrap();
        let model =
            CoquiResNetSpeakerEncoder::<NdArray>::load(&config, checkpoint, &NdArrayDevice::Cpu)
                .unwrap();
        assert_eq!(model.projection_dim(), 512);
    }

    #[test]
    #[ignore = "requires the pinned published Coqui speaker encoder"]
    fn published_resnet_encodes_reference_audio_when_available() {
        let config_path = std::env::var_os("TONGUES_TEST_COQUI_SPEAKER_CONFIG")
            .expect("TONGUES_TEST_COQUI_SPEAKER_CONFIG is required");
        let checkpoint = std::env::var_os("TONGUES_TEST_COQUI_SPEAKER_MODEL")
            .expect("TONGUES_TEST_COQUI_SPEAKER_MODEL is required");
        let config = SpeakerEncoderPackageConfig::from_file(config_path).unwrap();
        let mut service = NativeSpeakerEmbeddingService::<NdArray>::load(
            config,
            checkpoint,
            NdArrayDevice::Cpu,
            SpeakerEmbeddingCachePolicy::Memory { max_entries: 2 },
        )
        .unwrap();
        let audio = AudioBuffer {
            samples: (0..16_000)
                .map(|index| {
                    (2.0 * std::f32::consts::PI * 220.0 * index as f32 / 16_000.0).sin() * 0.1
                })
                .collect(),
            sample_rate_hz: 16_000,
            channels: 1,
        };
        let first = service.encode_audio(&audio).unwrap();
        let second = service.encode_audio(&audio).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.values.len(), 512);
        assert!((first.values.iter().map(|value| value * value).sum::<f32>() - 1.0).abs() < 1e-4);
    }
}
