//! Burn-native XTTS v2 inference.
//!
//! The checkpoint-compatible graph follows Coqui TTS revision
//! `dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e`, principally
//! `TTS/tts/models/xtts.py` and `TTS/tts/layers/xtts/{gpt,
//! gpt_inference,latent_encoder,perceiver_encoder,hifigan_decoder}.py`.
//! This file is an MPL-2.0 covered modification. Model weights and generated
//! output retain their separately recorded artifact terms.

use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, GroupNorm, GroupNormConfig, LayerNorm, LayerNormConfig, Linear,
    LinearConfig, PaddingConfig1d,
};
use burn::nn::interpolate::{Interpolate1dConfig, InterpolateMode};
use burn::tensor::activation::{leaky_relu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use speaking::SpeakerReferenceSource;
use tongues_audio::{
    read_wav, spectrogram, AudioBuffer, MelConfig, MelNormalization, MelScale, PadMode,
    SpectralDomain, SpectralScale, SpectrogramConfig, SpectrogramNormalization,
    SpectrogramOutput, StftConfig, Window,
};

use crate::model_package::{
    ModelPackageArchitecture, NeutralModelConfig, SpeakerEncoderPackageConfig,
    MODEL_PACKAGE_CONFIG, MODEL_PACKAGE_WEIGHTS,
};
use crate::speaker_encoder::CoquiResNetSpeakerEncoder;
use crate::burn_hifigan::{WeightNormConv1d, WeightNormConvTranspose1d};
use crate::{
    AudioChunk, AudioSink, SpeechModelCapabilities, SpeechModelFamily, SpeechSynthesisEngine,
    SpeechSynthesisRequest, SynthesisDimension, SynthesisProfileEvent, SynthesisProfiler,
    SynthesisStage, XttsStreamAssembler, XttsTokenizer, XttsV2Config,
    XTTS_V2_DEFAULT_STREAM_CODE_CHUNK, XTTS_V2_STREAM_OVERLAP_SAMPLES,
};

const CONDITIONING_ATTENTION_BLOCKS: usize = 6;
const CONDITIONING_MIN_SECONDS: f32 = 0.33;
const CONDITIONING_SAMPLE_RATE: u32 = 22_050;
const SPEAKER_SAMPLE_RATE: u32 = 16_000;
const SPEAKER_MEL_BINS: usize = 64;
const PERCEIVER_DEPTH: usize = 2;
const PERCEIVER_HEADS: usize = 8;
const PERCEIVER_HEAD_DIM: usize = 64;
const PERCEIVER_LATENTS: usize = 32;
const GPT_LAYER_NORM_EPSILON: f64 = 1.0e-5;
const SPEAKER_EMBEDDING_CACHE_ENTRIES: usize = 16;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XttsGenerationControls {
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub stream_code_chunk: usize,
    pub overlap_samples: usize,
    pub seed: u64,
}

impl XttsGenerationControls {
    pub fn from_config(config: &XttsV2Config, seed: Option<u64>) -> Self {
        Self {
            temperature: config.temperature,
            repetition_penalty: config.repetition_penalty,
            top_k: config.top_k,
            top_p: config.top_p,
            stream_code_chunk: XTTS_V2_DEFAULT_STREAM_CODE_CHUNK,
            overlap_samples: XTTS_V2_STREAM_OVERLAP_SAMPLES,
            seed: seed.unwrap_or(0),
        }
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.temperature.is_finite() && self.temperature > 0.0,
            "XTTS temperature must be finite and positive"
        );
        ensure!(
            self.repetition_penalty.is_finite() && self.repetition_penalty > 0.0,
            "XTTS repetition penalty must be finite and positive"
        );
        ensure!(self.top_k > 0, "XTTS top-k must be positive");
        ensure!(
            self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0,
            "XTTS top-p must be in (0, 1]"
        );
        ensure!(
            self.stream_code_chunk > 0,
            "XTTS stream code chunk must be positive"
        );
        ensure!(
            self.overlap_samples > 0,
            "XTTS stream overlap must be positive"
        );
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct XttsConditioning<B: Backend> {
    /// `[1, 32, model_channels]`
    pub gpt: Tensor<B, 3>,
    /// `[1, d_vector_dim, 1]`
    pub speaker: Tensor<B, 3>,
}

#[derive(Module, Debug)]
struct HfConv1d<B: Backend> {
    /// Hugging Face GPT-2 Conv1D stores `[input, output]`.
    weight: Param<Tensor<B, 2>>,
    bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> HfConv1d<B> {
    fn init(input: usize, output: usize, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::zeros([input, output], device)),
            bias: Param::from_tensor(Tensor::zeros([output], device)),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, input_channels, _] = input.dims();
        let output = self.bias.dims()[0];
        input.matmul(
            self.weight
                .val()
                .reshape([1, input_channels, output])
                .expand([batch, input_channels, output]),
        ) + self.bias.val().reshape([1, 1, output])
    }
}

#[derive(Module, Debug)]
struct ConditioningAttention<B: Backend> {
    norm: GroupNorm<B>,
    qkv: Conv1d<B>,
    proj_out: Conv1d<B>,
    heads: usize,
}

impl<B: Backend> ConditioningAttention<B> {
    fn init(channels: usize, heads: usize, device: &B::Device) -> Self {
        let mut groups = if channels <= 16 {
            8
        } else if channels <= 64 {
            16
        } else {
            32
        };
        while channels % groups != 0 {
            groups /= 2;
        }
        Self {
            norm: GroupNormConfig::new(groups, channels).init(device),
            qkv: Conv1dConfig::new(channels, channels * 3, 1)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            proj_out: Conv1dConfig::new(channels, channels, 1)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            heads,
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, frames] = input.dims();
        let per_head = channels / self.heads;
        let qkv = self
            .qkv
            .forward(self.norm.forward(input.clone()))
            .reshape([batch, self.heads, per_head * 3, frames]);
        let query = qkv
            .clone()
            .slice([
                0..batch,
                0..self.heads,
                0..per_head,
                0..frames,
            ])
            .reshape([batch * self.heads, per_head, frames]);
        let key = qkv
            .clone()
            .slice([
                0..batch,
                0..self.heads,
                per_head..per_head * 2,
                0..frames,
            ])
            .reshape([batch * self.heads, per_head, frames]);
        let value = qkv
            .slice([
                0..batch,
                0..self.heads,
                per_head * 2..per_head * 3,
                0..frames,
            ])
            .reshape([batch * self.heads, per_head, frames]);
        let scores = query.swap_dims(1, 2).matmul(key) / (per_head as f64).sqrt();
        let attended = softmax(scores, 2)
            .matmul(value.swap_dims(1, 2))
            .swap_dims(1, 2)
            .reshape([batch, channels, frames]);
        input + self.proj_out.forward(attended)
    }
}

#[derive(Module, Debug)]
struct ConditioningEncoder<B: Backend> {
    init: Conv1d<B>,
    attn: Vec<ConditioningAttention<B>>,
}

impl<B: Backend> ConditioningEncoder<B> {
    fn init(channels: usize, heads: usize, device: &B::Device) -> Self {
        Self {
            init: Conv1dConfig::new(80, channels, 1)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            attn: (0..CONDITIONING_ATTENTION_BLOCKS)
                .map(|_| ConditioningAttention::init(channels, heads, device))
                .collect(),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut output = self.init.forward(input);
        for block in &self.attn {
            output = block.forward(output);
        }
        output
    }
}

#[derive(Module, Debug)]
struct PerceiverAttention<B: Backend> {
    to_q: Linear<B>,
    to_kv: Linear<B>,
    to_out: Linear<B>,
}

impl<B: Backend> PerceiverAttention<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        let inner = PERCEIVER_HEADS * PERCEIVER_HEAD_DIM;
        Self {
            to_q: LinearConfig::new(channels, inner)
                .with_bias(false)
                .init(device),
            to_kv: LinearConfig::new(channels, inner * 2)
                .with_bias(false)
                .init(device),
            to_out: LinearConfig::new(inner, channels)
                .with_bias(false)
                .init(device),
        }
    }

    fn forward(&self, latents: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, latent_count, _] = latents.dims();
        let context = Tensor::cat(vec![latents.clone(), context], 1);
        let context_count = context.dims()[1];
        let query = self
            .to_q
            .forward(latents)
            .reshape([
                batch,
                latent_count,
                PERCEIVER_HEADS,
                PERCEIVER_HEAD_DIM,
            ])
            .swap_dims(1, 2);
        let kv = self.to_kv.forward(context).reshape([
            batch,
            context_count,
            2,
            PERCEIVER_HEADS,
            PERCEIVER_HEAD_DIM,
        ]);
        let key = kv
            .clone()
            .slice([
                0..batch,
                0..context_count,
                0..1,
                0..PERCEIVER_HEADS,
                0..PERCEIVER_HEAD_DIM,
            ])
            .squeeze::<4>()
            .swap_dims(1, 2);
        let value = kv
            .slice([
                0..batch,
                0..context_count,
                1..2,
                0..PERCEIVER_HEADS,
                0..PERCEIVER_HEAD_DIM,
            ])
            .squeeze::<4>()
            .swap_dims(1, 2);
        let attended = softmax(
            query.matmul(key.swap_dims(2, 3)) / (PERCEIVER_HEAD_DIM as f64).sqrt(),
            3,
        )
        .matmul(value)
        .swap_dims(1, 2)
        .reshape([
            batch,
            latent_count,
            PERCEIVER_HEADS * PERCEIVER_HEAD_DIM,
        ]);
        self.to_out.forward(attended)
    }
}

#[derive(Module, Debug)]
struct PerceiverFeedForward<B: Backend> {
    first: Linear<B>,
    second: Linear<B>,
    inner: usize,
}

impl<B: Backend> PerceiverFeedForward<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        let inner = channels * 8 / 3;
        Self {
            first: LinearConfig::new(channels, inner * 2).init(device),
            second: LinearConfig::new(inner, channels).init(device),
            inner,
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, frames, _] = input.dims();
        let projected = self.first.forward(input);
        let values = projected.clone().slice([
            0..batch,
            0..frames,
            0..self.inner,
        ]);
        let gates = projected.slice([
            0..batch,
            0..frames,
            self.inner..self.inner * 2,
        ]);
        self.second.forward(values * gelu_new(gates))
    }
}

#[derive(Module, Debug)]
struct PerceiverLayer<B: Backend> {
    attention: PerceiverAttention<B>,
    feed_forward: PerceiverFeedForward<B>,
}

#[derive(Module, Debug)]
struct PerceiverResampler<B: Backend> {
    latents: Param<Tensor<B, 2>>,
    layers: Vec<PerceiverLayer<B>>,
    norm: RmsNorm<B>,
}

impl<B: Backend> PerceiverResampler<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            latents: Param::from_tensor(Tensor::zeros(
                [PERCEIVER_LATENTS, channels],
                device,
            )),
            layers: (0..PERCEIVER_DEPTH)
                .map(|_| PerceiverLayer {
                    attention: PerceiverAttention::init(channels, device),
                    feed_forward: PerceiverFeedForward::init(channels, device),
                })
                .collect(),
            norm: RmsNorm::init(channels, device),
        }
    }

    fn forward(&self, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, _, channels] = context.dims();
        let mut latents = self
            .latents
            .val()
            .reshape([1, PERCEIVER_LATENTS, channels])
            .expand([batch, PERCEIVER_LATENTS, channels]);
        for layer in &self.layers {
            latents = layer
                .attention
                .forward(latents.clone(), context.clone())
                + latents;
            latents = layer.feed_forward.forward(latents.clone()) + latents;
        }
        self.norm.forward(latents)
    }
}

#[derive(Module, Debug)]
struct RmsNorm<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
    scale: f64,
}

impl<B: Backend> RmsNorm<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::ones([channels], device)),
            scale: (channels as f64).sqrt(),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let channels = self.gamma.dims()[0];
        let norm = input
            .clone()
            .square()
            .sum_dim(2)
            .sqrt()
            .clamp_min(1.0e-12);
        input / norm * self.scale * self.gamma.val().reshape([1, 1, channels])
    }
}

#[derive(Debug, Clone)]
struct GptLayerCache<B: Backend> {
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
}

#[derive(Module, Debug)]
struct Gpt2Attention<B: Backend> {
    c_attn: HfConv1d<B>,
    c_proj: HfConv1d<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> Gpt2Attention<B> {
    fn init(channels: usize, heads: usize, device: &B::Device) -> Self {
        Self {
            c_attn: HfConv1d::init(channels, channels * 3, device),
            c_proj: HfConv1d::init(channels, channels, device),
            heads,
            head_dim: channels / heads,
        }
    }

    fn forward(
        &self,
        input: Tensor<B, 3>,
        past: Option<GptLayerCache<B>>,
    ) -> (Tensor<B, 3>, GptLayerCache<B>) {
        let [batch, query_len, channels] = input.dims();
        let qkv = self.c_attn.forward(input);
        let query = qkv
            .clone()
            .slice([0..batch, 0..query_len, 0..channels])
            .reshape([batch, query_len, self.heads, self.head_dim])
            .swap_dims(1, 2);
        let mut key = qkv
            .clone()
            .slice([
                0..batch,
                0..query_len,
                channels..channels * 2,
            ])
            .reshape([batch, query_len, self.heads, self.head_dim])
            .swap_dims(1, 2);
        let mut value = qkv
            .slice([
                0..batch,
                0..query_len,
                channels * 2..channels * 3,
            ])
            .reshape([batch, query_len, self.heads, self.head_dim])
            .swap_dims(1, 2);
        let past_len = past.as_ref().map_or(0, |cache| cache.key.dims()[2]);
        if let Some(past) = past {
            key = Tensor::cat(vec![past.key, key], 2);
            value = Tensor::cat(vec![past.value, value], 2);
        }
        let key_len = key.dims()[2];
        let mut scores =
            query.matmul(key.clone().swap_dims(2, 3)) / (self.head_dim as f64).sqrt();
        if query_len > 1 {
            let device = scores.device();
            scores = scores + causal_mask::<B>(query_len, key_len, past_len, &device);
        }
        let attended = softmax(scores, 3)
            .matmul(value.clone())
            .swap_dims(1, 2)
            .reshape([batch, query_len, channels]);
        (
            self.c_proj.forward(attended),
            GptLayerCache { key, value },
        )
    }
}

#[derive(Module, Debug)]
struct Gpt2Mlp<B: Backend> {
    c_fc: HfConv1d<B>,
    c_proj: HfConv1d<B>,
}

impl<B: Backend> Gpt2Mlp<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            c_fc: HfConv1d::init(channels, channels * 4, device),
            c_proj: HfConv1d::init(channels * 4, channels, device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.c_proj.forward(gelu_new(self.c_fc.forward(input)))
    }
}

#[derive(Module, Debug)]
struct Gpt2Block<B: Backend> {
    ln_1: LayerNorm<B>,
    attn: Gpt2Attention<B>,
    ln_2: LayerNorm<B>,
    mlp: Gpt2Mlp<B>,
}

impl<B: Backend> Gpt2Block<B> {
    fn init(channels: usize, heads: usize, device: &B::Device) -> Self {
        Self {
            ln_1: LayerNormConfig::new(channels)
                .with_epsilon(GPT_LAYER_NORM_EPSILON)
                .init(device),
            attn: Gpt2Attention::init(channels, heads, device),
            ln_2: LayerNormConfig::new(channels)
                .with_epsilon(GPT_LAYER_NORM_EPSILON)
                .init(device),
            mlp: Gpt2Mlp::init(channels, device),
        }
    }

    fn forward(
        &self,
        input: Tensor<B, 3>,
        past: Option<GptLayerCache<B>>,
    ) -> (Tensor<B, 3>, GptLayerCache<B>) {
        let (attention, cache) = self.attn.forward(self.ln_1.forward(input.clone()), past);
        let output = input + attention;
        let output = output.clone() + self.mlp.forward(self.ln_2.forward(output));
        (output, cache)
    }
}

#[derive(Module, Debug)]
struct Gpt2Transformer<B: Backend> {
    h: Vec<Gpt2Block<B>>,
    ln_f: LayerNorm<B>,
}

impl<B: Backend> Gpt2Transformer<B> {
    fn init(config: &XttsV2Config, device: &B::Device) -> Self {
        let channels = config.model_args.gpt_n_model_channels;
        Self {
            h: (0..config.model_args.gpt_layers)
                .map(|_| Gpt2Block::init(channels, config.model_args.gpt_n_heads, device))
                .collect(),
            ln_f: LayerNormConfig::new(channels)
                .with_epsilon(GPT_LAYER_NORM_EPSILON)
                .init(device),
        }
    }

    fn forward(
        &self,
        mut input: Tensor<B, 3>,
        past: Option<Vec<GptLayerCache<B>>>,
    ) -> (Tensor<B, 3>, Vec<GptLayerCache<B>>) {
        let mut caches = Vec::with_capacity(self.h.len());
        let mut past = past.unwrap_or_default().into_iter();
        for block in &self.h {
            let (output, cache) = block.forward(input, past.next());
            input = output;
            caches.push(cache);
        }
        (self.ln_f.forward(input), caches)
    }
}

#[derive(Module, Debug)]
struct LearnedPositionEmbeddings<B: Backend> {
    emb: Embedding<B>,
}

impl<B: Backend> LearnedPositionEmbeddings<B> {
    fn init(length: usize, channels: usize, device: &B::Device) -> Self {
        Self {
            emb: EmbeddingConfig::new(length, channels).init(device),
        }
    }

    fn positions(&self, length: usize, device: &B::Device) -> Tensor<B, 3> {
        let ids = (0..length).map(|id| id as i64).collect::<Vec<_>>();
        self.emb.forward(Tensor::<B, 2, Int>::from_data(
            TensorData::new(ids, [1, length]),
            device,
        ))
    }

    fn position(&self, index: usize, device: &B::Device) -> Tensor<B, 3> {
        self.emb.forward(Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![index as i64], [1, 1]),
            device,
        ))
    }
}

#[derive(Module, Debug)]
struct XttsGpt<B: Backend> {
    conditioning_encoder: ConditioningEncoder<B>,
    conditioning_perceiver: PerceiverResampler<B>,
    text_embedding: Embedding<B>,
    mel_embedding: Embedding<B>,
    gpt: Gpt2Transformer<B>,
    mel_pos_embedding: LearnedPositionEmbeddings<B>,
    text_pos_embedding: LearnedPositionEmbeddings<B>,
    final_norm: LayerNorm<B>,
    text_head: Linear<B>,
    mel_head: Linear<B>,
}

impl<B: Backend> XttsGpt<B> {
    fn init(config: &XttsV2Config, device: &B::Device) -> Self {
        let channels = config.model_args.gpt_n_model_channels;
        Self {
            conditioning_encoder: ConditioningEncoder::init(
                channels,
                config.model_args.gpt_n_heads,
                device,
            ),
            conditioning_perceiver: PerceiverResampler::init(channels, device),
            text_embedding: EmbeddingConfig::new(
                config.model_args.gpt_number_text_tokens,
                channels,
            )
            .init(device),
            mel_embedding: EmbeddingConfig::new(
                config.model_args.gpt_num_audio_tokens,
                channels,
            )
            .init(device),
            gpt: Gpt2Transformer::init(config, device),
            mel_pos_embedding: LearnedPositionEmbeddings::init(
                config.model_args.gpt_max_audio_tokens + 3,
                channels,
                device,
            ),
            text_pos_embedding: LearnedPositionEmbeddings::init(
                config.model_args.gpt_max_text_tokens + 2,
                channels,
                device,
            ),
            final_norm: LayerNormConfig::new(channels).init(device),
            text_head: LinearConfig::new(
                channels,
                config.model_args.gpt_number_text_tokens,
            )
            .init(device),
            mel_head: LinearConfig::new(channels, config.model_args.gpt_num_audio_tokens)
                .init(device),
        }
    }

    fn style_embedding(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        self.conditioning_perceiver
            .forward(self.conditioning_encoder.forward(mel).swap_dims(1, 2))
    }

    fn prefix(
        &self,
        conditioning: Tensor<B, 3>,
        text_ids: &[u32],
        config: &XttsV2Config,
        device: &B::Device,
    ) -> Result<Tensor<B, 3>> {
        ensure!(
            text_ids.len() < config.model_args.gpt_max_text_tokens,
            "XTTS text has {} tokens; checkpoint limit is {}",
            text_ids.len(),
            config.model_args.gpt_max_text_tokens - 1
        );
        let start = config
            .model_args
            .gpt_start_text_token
            .context("XTTS tokenizer start token was not resolved")?;
        let stop = config
            .model_args
            .gpt_stop_text_token
            .context("XTTS tokenizer stop token was not resolved")?;
        let mut text = Vec::with_capacity(text_ids.len() + 2);
        text.push(i64::from(start));
        text.extend(text_ids.iter().map(|id| i64::from(*id)));
        text.push(i64::from(stop));
        let text = Tensor::<B, 2, Int>::from_data(
            TensorData::new(text, [1, text_ids.len() + 2]),
            device,
        );
        let text_embedding = self.text_embedding.forward(text)
            + self.text_pos_embedding.positions(text_ids.len() + 2, device);
        let start_audio = Tensor::<B, 2, Int>::from_data(
            TensorData::new(
                vec![i64::from(config.model_args.gpt_start_audio_token)],
                [1, 1],
            ),
            device,
        );
        let audio_embedding =
            self.mel_embedding.forward(start_audio) + self.mel_pos_embedding.position(0, device);
        Ok(Tensor::cat(
            vec![conditioning, text_embedding, audio_embedding],
            1,
        ))
    }

    fn next(
        &self,
        input: Tensor<B, 3>,
        cache: Option<Vec<GptLayerCache<B>>>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Vec<GptLayerCache<B>>) {
        let (hidden, cache) = self.gpt.forward(input, cache);
        let hidden_dims = hidden.dims();
        let last = hidden.slice([
            0..1,
            hidden_dims[1] - 1..hidden_dims[1],
            0..hidden_dims[2],
        ]);
        let latent = self.final_norm.forward(last).squeeze::<2>();
        let logits = self.mel_head.forward(latent.clone());
        (logits, latent, cache)
    }

    fn code_embedding(&self, code: u32, position: usize, device: &B::Device) -> Tensor<B, 3> {
        let code = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![i64::from(code)], [1, 1]),
            device,
        );
        self.mel_embedding.forward(code) + self.mel_pos_embedding.position(position, device)
    }
}

#[derive(Module, Debug)]
struct XttsResBlock<B: Backend> {
    convs1: Vec<WeightNormConv1d<B>>,
    convs2: Vec<WeightNormConv1d<B>>,
}

impl<B: Backend> XttsResBlock<B> {
    fn init(channels: usize, kernel: usize, device: &B::Device) -> Self {
        let dilations = [1, 3, 5];
        Self {
            convs1: dilations
                .into_iter()
                .map(|dilation| {
                    WeightNormConv1d::new(
                        channels,
                        channels,
                        kernel,
                        1,
                        dilation,
                        (kernel * dilation - dilation) / 2,
                        true,
                        device,
                    )
                })
                .collect(),
            convs2: (0..3)
                .map(|_| {
                    WeightNormConv1d::new(
                        channels,
                        channels,
                        kernel,
                        1,
                        1,
                        (kernel - 1) / 2,
                        true,
                        device,
                    )
                })
                .collect(),
        }
    }

    fn forward(&self, mut input: Tensor<B, 3>) -> Tensor<B, 3> {
        for (first, second) in self.convs1.iter().zip(&self.convs2) {
            let output = first.forward(leaky_relu(input.clone(), 0.1));
            let output = second.forward(leaky_relu(output, 0.1));
            input = input + output;
        }
        input
    }
}

#[derive(Module, Debug)]
struct XttsWaveformDecoder<B: Backend> {
    conv_pre: Conv1d<B>,
    ups: Vec<WeightNormConvTranspose1d<B>>,
    resblocks: Vec<XttsResBlock<B>>,
    conv_post: Conv1d<B>,
    cond_layer: Conv1d<B>,
    conds: Vec<Conv1d<B>>,
}

impl<B: Backend> XttsWaveformDecoder<B> {
    fn init(config: &XttsV2Config, device: &B::Device) -> Self {
        let initial = 512;
        let factors = [8, 8, 2, 2];
        let kernels = [16, 16, 4, 4];
        let residual_kernels = [3, 7, 11];
        let mut ups = Vec::with_capacity(factors.len());
        let mut resblocks = Vec::with_capacity(factors.len() * residual_kernels.len());
        let mut conds = Vec::with_capacity(factors.len());
        for (stage, (&factor, &kernel)) in factors.iter().zip(&kernels).enumerate() {
            let input = initial >> stage;
            let output = initial >> (stage + 1);
            ups.push(WeightNormConvTranspose1d::new(
                input,
                output,
                kernel,
                factor,
                1,
                (kernel - factor) / 2,
                true,
                device,
            ));
            for residual_kernel in residual_kernels {
                resblocks.push(XttsResBlock::init(output, residual_kernel, device));
            }
            conds.push(
                Conv1dConfig::new(config.model_args.d_vector_dim, output, 1)
                    .with_padding(PaddingConfig1d::Valid)
                    .init(device),
            );
        }
        Self {
            conv_pre: Conv1dConfig::new(config.model_args.decoder_input_dim, initial, 7)
                .with_padding(PaddingConfig1d::Explicit(3, 3))
                .init(device),
            ups,
            resblocks,
            conv_post: Conv1dConfig::new(initial >> factors.len(), 1, 7)
                .with_padding(PaddingConfig1d::Explicit(3, 3))
                .with_bias(false)
                .init(device),
            cond_layer: Conv1dConfig::new(config.model_args.d_vector_dim, initial, 1)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            conds,
        }
    }

    fn forward(&self, input: Tensor<B, 3>, speaker: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut output = self.conv_pre.forward(input) + self.cond_layer.forward(speaker.clone());
        for stage in 0..self.ups.len() {
            output = self.ups[stage].forward(leaky_relu(output, 0.1))
                + self.conds[stage].forward(speaker.clone());
            let first = stage * 3;
            let mut fused = self.resblocks[first].forward(output.clone());
            fused = fused + self.resblocks[first + 1].forward(output.clone());
            fused = fused + self.resblocks[first + 2].forward(output);
            output = fused / 3.0;
        }
        self.conv_post.forward(leaky_relu(output, 0.1)).tanh()
    }
}

#[derive(Module, Debug)]
struct XttsHifiDecoder<B: Backend> {
    waveform_decoder: XttsWaveformDecoder<B>,
    speaker_encoder: CoquiResNetSpeakerEncoder<B>,
}

impl<B: Backend> XttsHifiDecoder<B> {
    fn init(config: &XttsV2Config, device: &B::Device) -> Result<Self> {
        Ok(Self {
            waveform_decoder: XttsWaveformDecoder::init(config, device),
            speaker_encoder: CoquiResNetSpeakerEncoder::init(
                &xtts_speaker_encoder_config(config),
                device,
            )?,
        })
    }

    fn forward(
        &self,
        latents: Tensor<B, 3>,
        speaker: Tensor<B, 3>,
        config: &XttsV2Config,
    ) -> Tensor<B, 3> {
        let input = latents.swap_dims(1, 2);
        let first = Interpolate1dConfig::new()
            .with_scale_factor(Some(
                config.model_args.gpt_code_stride_len as f32
                    / config.model_args.output_hop_length as f32,
            ))
            .with_mode(InterpolateMode::Linear)
            .with_align_corners(false)
            .init()
            .forward(input);
        let second = if config.audio.output_sample_rate != config.audio.input_sample_rate {
            Interpolate1dConfig::new()
                .with_scale_factor(Some(
                    config.audio.output_sample_rate as f32
                        / config.audio.input_sample_rate as f32,
                ))
                .with_mode(InterpolateMode::Linear)
                .with_align_corners(false)
                .init()
                .forward(first)
        } else {
            first
        };
        self.waveform_decoder.forward(second, speaker)
    }
}

#[derive(Module, Debug)]
struct XttsNativeModel<B: Backend> {
    gpt: XttsGpt<B>,
    hifigan_decoder: XttsHifiDecoder<B>,
    mel_stats: Param<Tensor<B, 1>>,
}

impl<B: Backend> XttsNativeModel<B> {
    fn init(config: &XttsV2Config, device: &B::Device) -> Result<Self> {
        Ok(Self {
            gpt: XttsGpt::init(config, device),
            hifigan_decoder: XttsHifiDecoder::init(config, device)?,
            mel_stats: Param::from_tensor(Tensor::ones([80], device)),
        })
    }

    fn load(
        config: &XttsV2Config,
        checkpoint: &Path,
        device: &B::Device,
    ) -> Result<Self> {
        let mut model = Self::init(config, device)?;
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut model,
            checkpoint,
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                predicate: Some(xtts_runtime_tensor),
                key_remappings: vec![
                    (
                        r"^gpt\.conditioning_perceiver\.layers\.(\d+)\.0\.".into(),
                        "gpt.conditioning_perceiver.layers.$1.attention.".into(),
                    ),
                    (
                        r"^gpt\.conditioning_perceiver\.layers\.(\d+)\.1\.0\.".into(),
                        "gpt.conditioning_perceiver.layers.$1.feed_forward.first.".into(),
                    ),
                    (
                        r"^gpt\.conditioning_perceiver\.layers\.(\d+)\.1\.2\.".into(),
                        "gpt.conditioning_perceiver.layers.$1.feed_forward.second.".into(),
                    ),
                    (
                        r"^(hifigan_decoder\.waveform_decoder\..*)\.parametrizations\.weight\.original0$"
                            .into(),
                        "$1.weight_g".into(),
                    ),
                    (
                        r"^(hifigan_decoder\.waveform_decoder\..*)\.parametrizations\.weight\.original1$"
                            .into(),
                        "$1.weight_v".into(),
                    ),
                    (
                        r"\.speaker_encoder\.layer(\d+)\.(\d+)\.se\.fc\.0\.".into(),
                        ".speaker_encoder.layer$1.$2.se.fc_0.".into(),
                    ),
                    (
                        r"\.speaker_encoder\.layer(\d+)\.(\d+)\.se\.fc\.2\.".into(),
                        ".speaker_encoder.layer$1.$2.se.fc_2.".into(),
                    ),
                    (
                        r"\.speaker_encoder\.layer(\d+)\.(\d+)\.downsample\.0\.".into(),
                        ".speaker_encoder.layer$1.$2.downsample.conv.".into(),
                    ),
                    (
                        r"\.speaker_encoder\.layer(\d+)\.(\d+)\.downsample\.1\.".into(),
                        ".speaker_encoder.layer$1.$2.downsample.bn.".into(),
                    ),
                    (
                        r"\.speaker_encoder\.attention\.0\.".into(),
                        ".speaker_encoder.attention_0.".into(),
                    ),
                    (
                        r"\.speaker_encoder\.attention\.2\.".into(),
                        ".speaker_encoder.attention_2.".into(),
                    ),
                    (
                        r"\.speaker_encoder\.attention\.3\.".into(),
                        ".speaker_encoder.attention_3.".into(),
                    ),
                ],
                map_indices_contiguous: true,
                allow_partial: true,
                skip_enum_variants: true,
            },
        )
        .context("failed to load native XTTS checkpoint")?;
        ensure!(
            result.missing.is_empty() && result.errors.is_empty(),
            "XTTS checkpoint mismatch: {} missing, {} load errors; missing [{}]",
            result.missing.len(),
            result.errors.len(),
            result
                .missing
                .iter()
                .map(|(path, _)| path.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(model)
    }
}

pub struct BurnXtts<B: Backend> {
    config: XttsV2Config,
    tokenizer: XttsTokenizer,
    model: XttsNativeModel<B>,
    device: B::Device,
    conditioning_cache: HashMap<[u8; 32], XttsConditioning<B>>,
    conditioning_cache_order: Vec<[u8; 32]>,
}

impl<B: Backend> BurnXtts<B> {
    pub fn load(package_dir: impl AsRef<Path>, device: B::Device) -> Result<Self> {
        let package_dir = package_dir.as_ref();
        let neutral: NeutralModelConfig = serde_json::from_slice(
            &std::fs::read(package_dir.join(MODEL_PACKAGE_CONFIG)).with_context(|| {
                format!(
                    "failed to read XTTS package config {}",
                    package_dir.join(MODEL_PACKAGE_CONFIG).display()
                )
            })?,
        )
        .context("invalid XTTS package model.json")?;
        ensure!(
            neutral.architecture == ModelPackageArchitecture::XttsV2,
            "model package architecture is {}, expected xtts_v2",
            neutral.architecture.as_str()
        );
        let config: XttsV2Config = serde_json::from_value(neutral.parameters)
            .context("invalid canonical XTTS package parameters")?;
        let tokenizer = XttsTokenizer::from_file(package_dir.join(&config.tokenizer))?;
        let model = XttsNativeModel::load(
            &config,
            &package_dir.join(MODEL_PACKAGE_WEIGHTS),
            &device,
        )?;
        Ok(Self {
            config,
            tokenizer,
            model,
            device,
            conditioning_cache: HashMap::new(),
            conditioning_cache_order: Vec::new(),
        })
    }

    pub fn config(&self) -> &XttsV2Config {
        &self.config
    }

    pub fn tokenizer(&self) -> &XttsTokenizer {
        &self.tokenizer
    }

    pub fn condition_reference(
        &mut self,
        references: &[PathBuf],
    ) -> Result<XttsConditioning<B>> {
        ensure!(
            !references.is_empty(),
            "XTTS requires at least one reference WAV"
        );
        let key = conditioning_cache_key(references, &self.config)?;
        if let Some(conditioning) = self.conditioning_cache.get(&key) {
            return Ok(conditioning.clone());
        }
        let mut loaded = Vec::with_capacity(references.len());
        for path in references {
            loaded.push(
                read_wav(path)
                    .with_context(|| format!("failed to read XTTS reference {}", path.display()))?,
            );
        }
        let conditioning = self.condition_audio(&loaded)?;
        while self.conditioning_cache_order.len() >= SPEAKER_EMBEDDING_CACHE_ENTRIES {
            let oldest = self.conditioning_cache_order.remove(0);
            self.conditioning_cache.remove(&oldest);
        }
        self.conditioning_cache_order.push(key);
        self.conditioning_cache.insert(key, conditioning.clone());
        Ok(conditioning)
    }

    pub fn condition_audio(&self, references: &[AudioBuffer]) -> Result<XttsConditioning<B>> {
        ensure!(
            !references.is_empty(),
            "XTTS requires at least one reference waveform"
        );
        let mut speaker_embeddings = Vec::with_capacity(references.len());
        let mut gpt_audio = Vec::new();
        for reference in references {
            reference.validate().context("invalid XTTS reference audio")?;
            let mut mono = reference.convert_channels(1)?;
            let max_samples = usize::try_from(self.config.max_ref_len)?
                .checked_mul(mono.sample_rate_hz as usize)
                .context("XTTS maximum reference length overflow")?;
            mono.samples.truncate(max_samples);
            ensure!(!mono.samples.is_empty(), "XTTS reference audio is empty");
            if self.config.sound_norm_refs {
                let peak = mono
                    .samples
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0_f32, f32::max);
                if peak > 0.0 {
                    for sample in &mut mono.samples {
                        *sample = *sample / peak * 0.75;
                    }
                }
            }
            let speaker_audio = mono.resample_linear(SPEAKER_SAMPLE_RATE)?;
            speaker_embeddings.push(self.speaker_embedding(&speaker_audio.samples)?);
            let conditioning_audio = mono.resample_linear(CONDITIONING_SAMPLE_RATE)?;
            gpt_audio.extend(conditioning_audio.samples);
        }
        let speaker = Tensor::cat(speaker_embeddings, 0)
            .mean_dim(0)
            .reshape([1, self.config.model_args.d_vector_dim, 1]);
        let gpt = self.gpt_conditioning(&gpt_audio)?;
        Ok(XttsConditioning { gpt, speaker })
    }

    fn speaker_embedding(&self, samples: &[f32]) -> Result<Tensor<B, 2>> {
        let feature = xtts_speaker_spectrogram(samples)?;
        let mut values = Vec::with_capacity(feature.values.len());
        for mel in 0..SPEAKER_MEL_BINS {
            for frame in 0..feature.frames {
                values.push((feature.values[frame * SPEAKER_MEL_BINS + mel] + 1.0e-6).ln());
            }
        }
        let input = Tensor::from_data(
            TensorData::new(values, [1, SPEAKER_MEL_BINS, feature.frames]),
            &self.device,
        );
        Ok(self
            .model
            .hifigan_decoder
            .speaker_encoder
            .forward(input, true))
    }

    fn gpt_conditioning(&self, samples: &[f32]) -> Result<Tensor<B, 3>> {
        let maximum = self
            .config
            .gpt_cond_len
            .checked_mul(CONDITIONING_SAMPLE_RATE as usize)
            .context("XTTS GPT conditioning length overflow")?
            .min(samples.len());
        let chunk = self
            .config
            .gpt_cond_chunk_len
            .checked_mul(CONDITIONING_SAMPLE_RATE as usize)
            .context("XTTS GPT conditioning chunk overflow")?;
        let minimum = (CONDITIONING_MIN_SECONDS * CONDITIONING_SAMPLE_RATE as f32) as usize;
        let mel_stats = self
            .model
            .mel_stats
            .val()
            .reshape([1, 80, 1]);
        let mut embeddings = Vec::new();
        for audio in samples[..maximum].chunks(chunk) {
            if audio.len() < minimum {
                continue;
            }
            let feature = xtts_conditioning_spectrogram(audio)?;
            let mut values = Vec::with_capacity(feature.values.len());
            for mel in 0..80 {
                for frame in 0..feature.frames {
                    values.push(
                        feature.values[frame * 80 + mel]
                            .max(1.0e-5)
                            .ln(),
                    );
                }
            }
            let mel = Tensor::from_data(
                TensorData::new(values, [1, 80, feature.frames]),
                &self.device,
            ) / mel_stats.clone();
            embeddings.push(self.model.gpt.style_embedding(mel));
        }
        ensure!(
            !embeddings.is_empty(),
            "XTTS reference must contain at least {CONDITIONING_MIN_SECONDS:.2}s of usable audio"
        );
        Ok(Tensor::cat(embeddings, 0).mean_dim(0))
    }

    fn generate_streaming(
        &mut self,
        request: &SpeechSynthesisRequest,
        sink: &mut dyn AudioSink,
        profiler: &mut Option<&mut dyn SynthesisProfiler>,
    ) -> Result<()> {
        let text = request
            .plan
            .intended_text
            .as_deref()
            .context("XTTS requires intended grapheme text")?;
        let language = request
            .options
            .model_language
            .as_deref()
            .context("XTTS requires an explicit checkpoint-local --model-language")?;
        ensure!(
            self.config
                .languages
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(language)),
            "XTTS language `{language}` is unsupported; available: {}",
            self.config.languages.join(", ")
        );
        let reference = request
            .plan
            .speaker_reference
            .as_ref()
            .context("XTTS requires a speaker reference")?;
        let SpeakerReferenceSource::ReferenceAudio { uri } = &reference.source else {
            bail!("XTTS currently requires reference-audio speaker conditioning");
        };
        let reference_path = local_audio_path(uri)?;
        let conditioning_started = Instant::now();
        let conditioning = self.condition_reference(&[reference_path])?;
        record_profile::<B>(
            profiler,
            &self.device,
            SynthesisStage::ReferenceConditioning,
            conditioning_started,
            [SynthesisDimension::new("references", 1)],
        )?;

        let cleaned = xtts_clean_text(text, language)?;
        let projection_started = Instant::now();
        let text_ids = self.tokenizer.encode_preprocessed(&cleaned, language)?;
        record_profile::<B>(
            profiler,
            &self.device,
            SynthesisStage::CheckpointProjection,
            projection_started,
            [SynthesisDimension::new("tokens", text_ids.len())],
        )?;

        let controls = XttsGenerationControls::from_config(&self.config, request.options.seed);
        controls.validate()?;
        let prefix_started = Instant::now();
        let prefix = self.model.gpt.prefix(
            conditioning.gpt,
            &text_ids,
            &self.config,
            &self.device,
        )?;
        let (mut logits, mut current_latent, mut cache) = self.model.gpt.next(prefix, None);
        record_profile::<B>(
            profiler,
            &self.device,
            SynthesisStage::AutoregressiveFirstCode,
            prefix_started,
            [SynthesisDimension::new("prefix_tokens", text_ids.len() + 35)],
        )?;

        let generation_started = Instant::now();
        let mut rng = StdRng::seed_from_u64(controls.seed);
        let mut generated = Vec::new();
        let mut latents = Vec::new();
        let mut assembler = XttsStreamAssembler::new(controls.overlap_samples)?;
        let mut chunk_index = 0;
        let mut last_decoded_codes = 0;
        for position in 0..self.config.model_args.gpt_max_audio_tokens {
            let code = sample_code(
                &logits
                    .into_data()
                    .to_vec::<f32>()
                    .context("XTTS logits are not f32")?,
                &generated,
                &controls,
                &mut rng,
            )?;
            if code == self.config.model_args.gpt_stop_audio_token {
                break;
            }
            generated.push(code);
            latents.push(current_latent);
            let embedding = self
                .model
                .gpt
                .code_embedding(code, position + 1, &self.device);
            let (next_logits, next_latent, next_cache) =
                self.model.gpt.next(embedding, Some(cache));
            logits = next_logits;
            current_latent = next_latent;
            cache = next_cache;
            if latents.len().is_multiple_of(controls.stream_code_chunk) {
                self.decode_emit(
                    &latents,
                    &conditioning.speaker,
                    false,
                    &mut assembler,
                    &mut chunk_index,
                    sink,
                    profiler,
                )?;
                last_decoded_codes = latents.len();
            }
        }
        ensure!(!latents.is_empty(), "XTTS GPT produced no audio codes");
        if latents.len() != last_decoded_codes {
            self.decode_emit(
                &latents,
                &conditioning.speaker,
                false,
                &mut assembler,
                &mut chunk_index,
                sink,
                profiler,
            )?;
        }
        self.decode_emit(
            &latents,
            &conditioning.speaker,
            true,
            &mut assembler,
            &mut chunk_index,
            sink,
            profiler,
        )?;
        record_profile::<B>(
            profiler,
            &self.device,
            SynthesisStage::AutoregressiveGeneration,
            generation_started,
            [SynthesisDimension::new("audio_codes", latents.len())],
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_emit(
        &self,
        latents: &[Tensor<B, 2>],
        speaker: &Tensor<B, 3>,
        is_final: bool,
        assembler: &mut XttsStreamAssembler,
        chunk_index: &mut usize,
        sink: &mut dyn AudioSink,
        profiler: &mut Option<&mut dyn SynthesisProfiler>,
    ) -> Result<()> {
        let decode_started = Instant::now();
        let latent = Tensor::stack(latents.to_vec(), 1);
        let waveform = self
            .model
            .hifigan_decoder
            .forward(latent, speaker.clone(), &self.config)
            .into_data()
            .to_vec::<f32>()
            .context("XTTS waveform is not f32")?;
        record_profile::<B>(
            profiler,
            &self.device,
            SynthesisStage::WaveformDecoder,
            decode_started,
            [
                SynthesisDimension::new("audio_codes", latents.len()),
                SynthesisDimension::new("cumulative_samples", waveform.len()),
            ],
        )?;
        let assembly_started = Instant::now();
        let pcm = assembler.push_cumulative(&waveform, is_final)?;
        record_profile::<B>(
            profiler,
            &self.device,
            SynthesisStage::StreamAssembly,
            assembly_started,
            [SynthesisDimension::new("emitted_samples", pcm.len())],
        )?;
        if !pcm.is_empty() || is_final {
            sink.emit(AudioChunk {
                chunk_index: *chunk_index,
                is_final,
                pause_after_ms: 0,
                sample_rate_hz: self.config.audio.output_sample_rate,
                pcm_mono_f32: pcm,
            })?;
            *chunk_index += 1;
        }
        Ok(())
    }
}

impl<B: Backend> SpeechSynthesisEngine for BurnXtts<B> {
    fn capabilities(&self) -> SpeechModelCapabilities {
        SpeechModelCapabilities {
            family: SpeechModelFamily::CrossLingualVoiceClone,
            supports_named_speakers: false,
            supports_languages: true,
            supports_reference_audio: true,
            supports_voice_conversion: false,
            integrated_vocoder: true,
        }
    }

    fn sample_rate_hz(&self) -> u32 {
        self.config.audio.output_sample_rate
    }

    fn synthesize_plan_streaming(
        &mut self,
        request: &SpeechSynthesisRequest,
        sink: &mut dyn AudioSink,
    ) -> Result<()> {
        self.generate_streaming(request, sink, &mut None)
    }

    fn synthesize_plan_streaming_profiled(
        &mut self,
        request: &SpeechSynthesisRequest,
        sink: &mut dyn AudioSink,
        profiler: &mut dyn SynthesisProfiler,
    ) -> Result<()> {
        self.generate_streaming(request, sink, &mut Some(profiler))
    }
}

fn sample_code(
    logits: &[f32],
    generated: &[u32],
    controls: &XttsGenerationControls,
    rng: &mut StdRng,
) -> Result<u32> {
    ensure!(!logits.is_empty(), "XTTS produced empty logits");
    let mut adjusted = logits
        .iter()
        .map(|value| *value / controls.temperature)
        .collect::<Vec<_>>();
    for &code in generated {
        if let Some(logit) = adjusted.get_mut(code as usize) {
            *logit = if *logit < 0.0 {
                *logit * controls.repetition_penalty
            } else {
                *logit / controls.repetition_penalty
            };
        }
    }
    let mut ranked = adjusted.into_iter().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(controls.top_k.min(ranked.len()));
    let max = ranked[0].1;
    let mut probabilities = ranked
        .iter()
        .map(|(_, logit)| (*logit - max).exp())
        .collect::<Vec<_>>();
    let sum: f32 = probabilities.iter().sum();
    ensure!(sum.is_finite() && sum > 0.0, "XTTS sampling logits are invalid");
    for probability in &mut probabilities {
        *probability /= sum;
    }
    let mut cumulative = 0.0;
    let keep = probabilities
        .iter()
        .position(|probability| {
            cumulative += *probability;
            cumulative >= controls.top_p
        })
        .map_or(probabilities.len(), |index| index + 1)
        .max(1);
    ranked.truncate(keep);
    probabilities.truncate(keep);
    let distribution =
        WeightedIndex::new(&probabilities).context("invalid XTTS sampling distribution")?;
    u32::try_from(ranked[distribution.sample(rng)].0).context("XTTS token ID does not fit u32")
}

fn causal_mask<B: Backend>(
    query: usize,
    key: usize,
    past: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let mut values = Vec::with_capacity(query * key);
    for query_index in 0..query {
        let maximum = past + query_index;
        for key_index in 0..key {
            values.push(if key_index <= maximum { 0.0 } else { -1.0e4 });
        }
    }
    Tensor::from_data(
        TensorData::new(values, [1, 1, query, key]),
        device,
    )
}

fn gelu_new<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    let inner = input.clone() + input.clone().powf_scalar(3.0) * 0.044_715;
    input * 0.5 * (inner * (2.0 / PI).sqrt() as f64).tanh().add_scalar(1.0)
}

fn xtts_runtime_tensor(path: &str, _container: &str) -> bool {
    path == "mel_stats"
        || path.starts_with("gpt.")
        || path.starts_with("hifigan_decoder.")
}

fn xtts_speaker_encoder_config(config: &XttsV2Config) -> SpeakerEncoderPackageConfig {
    SpeakerEncoderPackageConfig {
        model_name: "resnet".into(),
        input_dim: SPEAKER_MEL_BINS,
        projection_dim: config.model_args.d_vector_dim,
        lstm_dim: None,
        num_lstm_layers: None,
        use_lstm_with_projection: false,
        use_torch_spec: true,
        log_input: true,
        encoder_type: "ASP".into(),
        layers: vec![3, 4, 6, 3],
        num_filters: vec![32, 64, 128, 256],
        sample_rate_hz: SPEAKER_SAMPLE_RATE,
        fft_size: Some(512),
        window_size: Some(400),
        hop_size: Some(160),
    }
}

fn xtts_speaker_spectrogram(samples: &[f32]) -> Result<tongues_audio::Spectrogram> {
    let emphasized = reflect_preemphasis(samples, 0.97);
    spectrogram(
        &emphasized,
        &SpectrogramConfig {
            sample_rate_hz: SPEAKER_SAMPLE_RATE,
            stft: StftConfig {
                fft_size: 512,
                window_size: 400,
                hop_size: 160,
                center: true,
                pad_mode: PadMode::Reflect,
                window: Window::Hamming,
            },
            output: SpectrogramOutput::Mel(MelConfig {
                bins: SPEAKER_MEL_BINS,
                min_frequency_hz: 0.0,
                max_frequency_hz: Some(8_000.0),
                scale: MelScale::Htk,
                normalization: MelNormalization::None,
            }),
            domain: SpectralDomain::Power,
            scale: SpectralScale::Linear,
            normalization: SpectrogramNormalization::None,
            preemphasis: None,
        },
    )
    .context("failed to compute XTTS speaker spectrogram")
}

fn xtts_conditioning_spectrogram(samples: &[f32]) -> Result<tongues_audio::Spectrogram> {
    spectrogram(
        samples,
        &SpectrogramConfig {
            sample_rate_hz: CONDITIONING_SAMPLE_RATE,
            stft: StftConfig {
                fft_size: 2_048,
                window_size: 1_024,
                hop_size: 256,
                center: true,
                pad_mode: PadMode::Reflect,
                window: Window::Hann,
            },
            output: SpectrogramOutput::Mel(MelConfig {
                bins: 80,
                min_frequency_hz: 0.0,
                max_frequency_hz: Some(8_000.0),
                // torchaudio MelSpectrogram defaults to the HTK frequency
                // scale even when `norm="slaney"` is selected.
                scale: MelScale::Htk,
                normalization: MelNormalization::Slaney,
            }),
            domain: SpectralDomain::Power,
            scale: SpectralScale::Linear,
            normalization: SpectrogramNormalization::None,
            preemphasis: None,
        },
    )
    .context("failed to compute XTTS conditioning spectrogram")
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

fn conditioning_cache_key(
    references: &[PathBuf],
    config: &XttsV2Config,
) -> Result<[u8; 32]> {
    let mut digest = Sha256::new();
    digest.update(config.gpt_cond_len.to_le_bytes());
    digest.update(config.gpt_cond_chunk_len.to_le_bytes());
    digest.update(config.max_ref_len.to_le_bytes());
    digest.update([u8::from(config.sound_norm_refs)]);
    for path in references {
        let canonical = path
            .canonicalize()
            .with_context(|| format!("failed to resolve XTTS reference {}", path.display()))?;
        digest.update(canonical.as_os_str().as_encoded_bytes());
        let metadata = std::fs::metadata(&canonical)
            .with_context(|| format!("failed to stat XTTS reference {}", canonical.display()))?;
        digest.update(metadata.len().to_le_bytes());
        if let Ok(modified) = metadata.modified() {
            if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                digest.update(duration.as_nanos().to_le_bytes());
            }
        }
    }
    Ok(digest.finalize().into())
}

fn local_audio_path(uri: &str) -> Result<PathBuf> {
    if let Some(path) = uri.strip_prefix("file://") {
        Ok(PathBuf::from(path))
    } else if uri.contains("://") {
        bail!("XTTS reference-audio URI scheme is unsupported: {uri}")
    } else {
        Ok(PathBuf::from(uri))
    }
}

fn xtts_clean_text(text: &str, language: &str) -> Result<String> {
    let language = language.to_ascii_lowercase();
    ensure!(
        !matches!(language.as_str(), "ar" | "zh" | "zh-cn" | "ko" | "ja"),
        "XTTS language `{language}` requires an upstream-compatible transliterator that is not available; provide a supported Latin/Cyrillic language"
    );
    let mut output = text
        .trim()
        .to_lowercase()
        .replace('“', "\"")
        .replace('”', "\"")
        .replace('‘', "'")
        .replace('’', "'")
        .replace('…', "...")
        .replace('–', "-")
        .replace('—', "-");
    output = output.split_whitespace().collect::<Vec<_>>().join(" ");
    ensure!(!output.is_empty(), "XTTS text is empty after cleaning");
    ensure!(
        !output.chars().any(|character| character.is_ascii_digit()),
        "XTTS number expansion is language-specific; spell out digits before synthesis"
    );
    Ok(output)
}

fn record_profile<B: Backend>(
    profiler: &mut Option<&mut dyn SynthesisProfiler>,
    device: &B::Device,
    stage: SynthesisStage,
    started: Instant,
    dimensions: impl IntoIterator<Item = SynthesisDimension>,
) -> Result<()> {
    let Some(profiler) = profiler.as_deref_mut() else {
        return Ok(());
    };
    B::sync(device)
        .map_err(anyhow::Error::new)
        .with_context(|| format!("failed to synchronize after {stage}"))?;
    profiler.record(SynthesisProfileEvent::new(
        stage,
        started.elapsed(),
        dimensions,
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleaner_is_explicit_about_unported_language_rules() {
        assert_eq!(
            xtts_clean_text("  Bonjour — monde… ", "fr").unwrap(),
            "bonjour - monde..."
        );
        assert!(xtts_clean_text("Room 2", "en")
            .unwrap_err()
            .to_string()
            .contains("number expansion"));
        assert!(xtts_clean_text("مرحبا", "ar")
            .unwrap_err()
            .to_string()
            .contains("transliterator"));
    }

    #[test]
    fn top_k_top_p_sampling_is_seeded_and_repetition_aware() {
        let controls = XttsGenerationControls {
            temperature: 1.0,
            repetition_penalty: 5.0,
            top_k: 3,
            top_p: 0.9,
            stream_code_chunk: 2,
            overlap_samples: 4,
            seed: 17,
        };
        let mut left = StdRng::seed_from_u64(17);
        let mut right = StdRng::seed_from_u64(17);
        let logits = [0.0, 4.0, 3.0, 2.0];
        let a = sample_code(&logits, &[1], &controls, &mut left).unwrap();
        let b = sample_code(&logits, &[1], &controls, &mut right).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, 1);
    }
}
