//! Burn-native Tacotron 2 acoustic inference.
//!
//! The autoregressive decoder intentionally exposes its stop and attention
//! traces.  Attention-based synthesis can otherwise return plausible-sized
//! garbage after silently hitting a decoder limit.
//!
//! DDC's coarse decoder is a training regularizer and is not used by upstream
//! inference.  Checkpoints may therefore omit it (released DDC bundles do) or
//! contain it; this module loads only inference-reachable weights.
//!
//! Source provenance: this is an MPL-2.0 covered source adaptation of the
//! Coqui TTS v0.6.1 Tacotron 2 inference graph at revision
//! `0cf3265a4686d7e856bd472cdaf1572d61cab2b8`, principally
//! `TTS/tts/layers/tacotron/tacotron2.py`,
//! `TTS/tts/layers/tacotron/attentions.py`,
//! `TTS/tts/layers/tacotron/common_layers.py`, and
//! `TTS/tts/models/tacotron2.py`. See `THIRD_PARTY_NOTICES.md`.

use std::fmt;
use std::path::Path;

use burn::module::{Initializer, Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear,
    LinearConfig, PaddingConfig1d,
};
use burn::tensor::activation::{relu, sigmoid, softmax, tanh};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, ElementConversion, Int, Tensor};

use crate::{TacotronArchitecture, TacotronAttentionNormalization, TacotronInferenceConfig};

const ENCODER_CONVOLUTIONS: usize = 3;
const POSTNET_CONVOLUTIONS: usize = 5;
const TACOTRON2_RNN_CHANNELS: usize = 1_024;
const PRENET_CHANNELS: usize = 256;
const ATTENTION_CHANNELS: usize = 128;
const LOCATION_FILTERS: usize = 32;
const LOCATION_KERNEL: usize = 31;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TacotronControls {
    pub stop_threshold: f32,
    pub max_decoder_steps: usize,
    pub minimum_decoder_steps: usize,
}

impl TacotronControls {
    pub fn from_config(config: &TacotronInferenceConfig) -> Self {
        Self {
            stop_threshold: config.stop_threshold,
            max_decoder_steps: config.max_decoder_steps,
            // Coqui only permits a stop after at least one completed step.
            minimum_decoder_steps: 1,
        }
    }

    pub fn validate(self) -> Result<Self, TacotronError> {
        if !self.stop_threshold.is_finite() || !(0.0..=1.0).contains(&self.stop_threshold) {
            return Err(input_error(
                "stop threshold must be finite and in the inclusive range [0, 1]",
            ));
        }
        if self.max_decoder_steps == 0 {
            return Err(input_error("maximum decoder steps must be positive"));
        }
        if self.minimum_decoder_steps >= self.max_decoder_steps {
            return Err(input_error(
                "minimum decoder steps must be smaller than the maximum",
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone)]
pub struct TacotronConditioning<B: Backend> {
    /// Explicit Capacitron latent `[batch, latent_channels]`.  When omitted for
    /// a Capacitron checkpoint, inference samples the standard-normal prior.
    pub style_embedding: Option<Tensor<B, 2>>,
}

impl<B: Backend> Default for TacotronConditioning<B> {
    fn default() -> Self {
        Self {
            style_embedding: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TacotronTermination {
    StopToken,
    MaximumSteps,
}

#[derive(Debug, Clone)]
pub struct TacotronOutput<B: Backend> {
    /// Postnet-refined mel spectrogram `[batch, frames, mel_bins]`.
    pub mel: Tensor<B, 3>,
    /// Raw decoder spectrogram `[batch, frames, mel_bins]`.
    pub decoder_mel: Tensor<B, 3>,
    /// Attention alignment `[batch, decoder_steps, input_tokens]`.
    pub alignments: Tensor<B, 3>,
    /// Stop probabilities `[batch, decoder_steps]`.
    pub stop_probabilities: Tensor<B, 2>,
    pub termination: TacotronTermination,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TacotronError {
    InvalidConfig(String),
    InvalidInput(String),
    Checkpoint(String),
    AttentionFailure {
        steps: usize,
        input_tokens: usize,
        last_focus: usize,
    },
}

impl fmt::Display for TacotronError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(formatter, "invalid Tacotron config: {message}"),
            Self::InvalidInput(message) => write!(formatter, "invalid Tacotron input: {message}"),
            Self::Checkpoint(message) => {
                write!(formatter, "unable to load Tacotron checkpoint: {message}")
            }
            Self::AttentionFailure {
                steps,
                input_tokens,
                last_focus,
            } => write!(
                formatter,
                "Tacotron attention did not emit a stop token after {steps} decoder steps \
                 (input tokens: {input_tokens}, last attention focus: {last_focus}); \
                 shorten or rephrase the text, or raise max_decoder_steps only after inspecting the alignment"
            ),
        }
    }
}

impl std::error::Error for TacotronError {}

fn config_error(message: impl Into<String>) -> TacotronError {
    TacotronError::InvalidConfig(message.into())
}

fn input_error(message: impl Into<String>) -> TacotronError {
    TacotronError::InvalidInput(message.into())
}

#[derive(Module, Debug)]
pub struct PytorchLstmCell<B: Backend> {
    pub weight_ih: Param<Tensor<B, 2>>,
    pub weight_hh: Param<Tensor<B, 2>>,
    pub bias_ih: Param<Tensor<B, 1>>,
    pub bias_hh: Param<Tensor<B, 1>>,
    hidden_channels: usize,
}

impl<B: Backend> PytorchLstmCell<B> {
    fn init(input_channels: usize, hidden_channels: usize, device: &B::Device) -> Self {
        let init = Initializer::XavierUniform { gain: 1.0 };
        Self {
            weight_ih: init.clone().init_with(
                [hidden_channels * 4, input_channels],
                Some(input_channels),
                Some(hidden_channels * 4),
                device,
            ),
            weight_hh: init.init_with(
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

    fn step(
        &self,
        input: Tensor<B, 2>,
        hidden: Tensor<B, 2>,
        cell: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let batch = input.dims()[0];
        let gates = input.matmul(self.weight_ih.val().transpose())
            + hidden.matmul(self.weight_hh.val().transpose())
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
        let hidden_channels = self.hidden_channels;
        // PyTorch LSTM gate order is input, forget, cell, output.
        let input_gate = sigmoid(gates.clone().slice([0..batch, 0..hidden_channels]));
        let forget_gate = sigmoid(
            gates
                .clone()
                .slice([0..batch, hidden_channels..hidden_channels * 2]),
        );
        let candidate = tanh(
            gates
                .clone()
                .slice([0..batch, hidden_channels * 2..hidden_channels * 3]),
        );
        let output_gate =
            sigmoid(gates.slice([0..batch, hidden_channels * 3..hidden_channels * 4]));
        let cell = forget_gate * cell + input_gate * candidate;
        let hidden = output_gate * tanh(cell.clone());
        (hidden, cell)
    }
}

#[derive(Module, Debug)]
pub struct PytorchBiLstm<B: Backend> {
    pub weight_ih_l0: Param<Tensor<B, 2>>,
    pub weight_hh_l0: Param<Tensor<B, 2>>,
    pub bias_ih_l0: Param<Tensor<B, 1>>,
    pub bias_hh_l0: Param<Tensor<B, 1>>,
    pub weight_ih_l0_reverse: Param<Tensor<B, 2>>,
    pub weight_hh_l0_reverse: Param<Tensor<B, 2>>,
    pub bias_ih_l0_reverse: Param<Tensor<B, 1>>,
    pub bias_hh_l0_reverse: Param<Tensor<B, 1>>,
    input_channels: usize,
    hidden_channels: usize,
}

impl<B: Backend> PytorchBiLstm<B> {
    fn init(input_channels: usize, hidden_channels: usize, device: &B::Device) -> Self {
        let init_input = || {
            Initializer::XavierUniform { gain: 1.0 }.init_with(
                [hidden_channels * 4, input_channels],
                Some(input_channels),
                Some(hidden_channels * 4),
                device,
            )
        };
        let init_hidden = || {
            Initializer::XavierUniform { gain: 1.0 }.init_with(
                [hidden_channels * 4, hidden_channels],
                Some(hidden_channels),
                Some(hidden_channels * 4),
                device,
            )
        };
        Self {
            weight_ih_l0: init_input(),
            weight_hh_l0: init_hidden(),
            bias_ih_l0: Initializer::Zeros.init([hidden_channels * 4], device),
            bias_hh_l0: Initializer::Zeros.init([hidden_channels * 4], device),
            weight_ih_l0_reverse: init_input(),
            weight_hh_l0_reverse: init_hidden(),
            bias_ih_l0_reverse: Initializer::Zeros.init([hidden_channels * 4], device),
            bias_hh_l0_reverse: Initializer::Zeros.init([hidden_channels * 4], device),
            input_channels,
            hidden_channels,
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, tokens, _] = input.dims();
        let device = input.device();
        let mut forward_hidden = Tensor::zeros([batch, self.hidden_channels], &device);
        let mut forward_cell = Tensor::zeros([batch, self.hidden_channels], &device);
        let mut forward_outputs = Vec::with_capacity(tokens);
        for token in 0..tokens {
            let current = input
                .clone()
                .slice([0..batch, token..token + 1, 0..self.input_channels])
                .reshape([batch, self.input_channels]);
            (forward_hidden, forward_cell) = lstm_step(
                current,
                forward_hidden,
                forward_cell,
                self.weight_ih_l0.val(),
                self.weight_hh_l0.val(),
                self.bias_ih_l0.val(),
                self.bias_hh_l0.val(),
                self.hidden_channels,
            );
            forward_outputs.push(forward_hidden.clone().unsqueeze_dim::<3>(1));
        }

        let mut reverse_hidden = Tensor::zeros([batch, self.hidden_channels], &device);
        let mut reverse_cell = Tensor::zeros([batch, self.hidden_channels], &device);
        let mut reverse_outputs = (0..tokens)
            .map(|_| Tensor::zeros([batch, 1, self.hidden_channels], &device))
            .collect::<Vec<_>>();
        for token in (0..tokens).rev() {
            let current = input
                .clone()
                .slice([0..batch, token..token + 1, 0..self.input_channels])
                .reshape([batch, self.input_channels]);
            (reverse_hidden, reverse_cell) = lstm_step(
                current,
                reverse_hidden,
                reverse_cell,
                self.weight_ih_l0_reverse.val(),
                self.weight_hh_l0_reverse.val(),
                self.bias_ih_l0_reverse.val(),
                self.bias_hh_l0_reverse.val(),
                self.hidden_channels,
            );
            reverse_outputs[token] = reverse_hidden.clone().unsqueeze_dim::<3>(1);
        }
        Tensor::cat(
            vec![
                Tensor::cat(forward_outputs, 1),
                Tensor::cat(reverse_outputs, 1),
            ],
            2,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn lstm_step<B: Backend>(
    input: Tensor<B, 2>,
    hidden: Tensor<B, 2>,
    cell: Tensor<B, 2>,
    weight_ih: Tensor<B, 2>,
    weight_hh: Tensor<B, 2>,
    bias_ih: Tensor<B, 1>,
    bias_hh: Tensor<B, 1>,
    hidden_channels: usize,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let batch = input.dims()[0];
    let gates = input.matmul(weight_ih.transpose())
        + hidden.matmul(weight_hh.transpose())
        + bias_ih
            .reshape([1, hidden_channels * 4])
            .repeat_dim(0, batch)
        + bias_hh
            .reshape([1, hidden_channels * 4])
            .repeat_dim(0, batch);
    let input_gate = sigmoid(gates.clone().slice([0..batch, 0..hidden_channels]));
    let forget_gate = sigmoid(
        gates
            .clone()
            .slice([0..batch, hidden_channels..hidden_channels * 2]),
    );
    let candidate = tanh(
        gates
            .clone()
            .slice([0..batch, hidden_channels * 2..hidden_channels * 3]),
    );
    let output_gate = sigmoid(gates.slice([0..batch, hidden_channels * 3..hidden_channels * 4]));
    let cell = forget_gate * cell + input_gate * candidate;
    let hidden = output_gate * tanh(cell.clone());
    (hidden, cell)
}

#[derive(Module, Debug)]
pub struct TacotronConvBnBlock<B: Backend> {
    pub convolution1d: Conv1d<B>,
    pub batch_normalization: BatchNorm<B>,
    activation: usize,
}

const CONV_ACTIVATION_LINEAR: usize = 0;
const CONV_ACTIVATION_RELU: usize = 1;
const CONV_ACTIVATION_TANH: usize = 2;

impl<B: Backend> TacotronConvBnBlock<B> {
    fn init(
        input_channels: usize,
        output_channels: usize,
        kernel_size: usize,
        activation: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            convolution1d: Conv1dConfig::new(input_channels, output_channels, kernel_size)
                .with_padding(PaddingConfig1d::Explicit(kernel_size / 2, kernel_size / 2))
                .init(device),
            batch_normalization: BatchNormConfig::new(output_channels)
                .with_momentum(0.1)
                .with_epsilon(1e-5)
                .init(device),
            activation,
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let output = self
            .batch_normalization
            .forward(self.convolution1d.forward(input));
        match self.activation {
            CONV_ACTIVATION_RELU => relu(output),
            CONV_ACTIVATION_TANH => tanh(output),
            _ => output,
        }
    }
}

#[derive(Module, Debug)]
pub struct Tacotron2Encoder<B: Backend> {
    pub convolutions: Vec<TacotronConvBnBlock<B>>,
    pub lstm: PytorchBiLstm<B>,
}

impl<B: Backend> Tacotron2Encoder<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            convolutions: (0..ENCODER_CONVOLUTIONS)
                .map(|_| {
                    TacotronConvBnBlock::init(channels, channels, 5, CONV_ACTIVATION_RELU, device)
                })
                .collect(),
            lstm: PytorchBiLstm::init(channels, channels / 2, device),
        }
    }

    fn forward(&self, mut input: Tensor<B, 3>) -> Tensor<B, 3> {
        for convolution in &self.convolutions {
            input = convolution.forward(input);
        }
        self.lstm.forward(input.swap_dims(1, 2))
    }
}

#[derive(Module, Debug)]
pub struct LegacyLinear<B: Backend> {
    pub linear_layer: Linear<B>,
}

impl<B: Backend> LegacyLinear<B> {
    fn init(input: usize, output: usize, bias: bool, device: &B::Device) -> Self {
        Self {
            linear_layer: LinearConfig::new(input, output)
                .with_bias(bias)
                .init(device),
        }
    }

    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        self.linear_layer.forward(input)
    }
}

#[derive(Module, Debug)]
pub struct TacotronPrenet<B: Backend> {
    pub linear_layers: Vec<LegacyLinear<B>>,
    dropout: Dropout,
    dropout_at_inference: bool,
}

impl<B: Backend> TacotronPrenet<B> {
    fn init(input_channels: usize, dropout_at_inference: bool, device: &B::Device) -> Self {
        let _ = device;
        Self {
            linear_layers: vec![
                LegacyLinear::init(input_channels, PRENET_CHANNELS, false, device),
                LegacyLinear::init(PRENET_CHANNELS, PRENET_CHANNELS, false, device),
            ],
            dropout: DropoutConfig::new(0.5).init(),
            dropout_at_inference,
        }
    }

    fn forward(&self, mut input: Tensor<B, 2>) -> Tensor<B, 2> {
        for layer in &self.linear_layers {
            input = relu(layer.forward(input));
            if self.dropout_at_inference {
                // Burn's Dropout intentionally becomes a no-op on inference
                // backends. Tacotron prenets are the exception: released
                // Coqui checkpoints explicitly request Monte-Carlo dropout
                // during inference, so apply the Bernoulli mask directly.
                let mask = input.random_like(Distribution::Bernoulli(0.5));
                input = input * mask * 2.0;
            } else if B::ad_enabled(&input.device()) {
                input = self.dropout.forward(input);
            }
        }
        input
    }
}

#[derive(Module, Debug)]
pub struct TacotronLocationLayer<B: Backend> {
    pub location_conv1d: Conv1d<B>,
    pub location_dense: LegacyLinear<B>,
}

impl<B: Backend> TacotronLocationLayer<B> {
    fn init(device: &B::Device) -> Self {
        Self {
            location_conv1d: Conv1dConfig::new(2, LOCATION_FILTERS, LOCATION_KERNEL)
                .with_bias(false)
                .with_padding(PaddingConfig1d::Explicit(
                    LOCATION_KERNEL / 2,
                    LOCATION_KERNEL / 2,
                ))
                .init(device),
            location_dense: LegacyLinear::init(LOCATION_FILTERS, ATTENTION_CHANNELS, false, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct TacotronOriginalAttention<B: Backend> {
    pub query_layer: LegacyLinear<B>,
    pub inputs_layer: LegacyLinear<B>,
    pub v: LegacyLinear<B>,
    pub location_layer: TacotronLocationLayer<B>,
    normalization: TacotronAttentionNormalization,
    location_attention: bool,
}

struct AttentionState<B: Backend> {
    weights: Tensor<B, 2>,
    cumulative: Tensor<B, 2>,
}

impl<B: Backend> TacotronOriginalAttention<B> {
    fn init(
        encoder_channels: usize,
        normalization: TacotronAttentionNormalization,
        location_attention: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            query_layer: LegacyLinear::init(
                TACOTRON2_RNN_CHANNELS,
                ATTENTION_CHANNELS,
                false,
                device,
            ),
            inputs_layer: LegacyLinear::init(encoder_channels, ATTENTION_CHANNELS, false, device),
            v: LegacyLinear::init(ATTENTION_CHANNELS, 1, true, device),
            location_layer: TacotronLocationLayer::init(device),
            normalization,
            location_attention,
        }
    }

    fn preprocess(&self, inputs: Tensor<B, 3>) -> Tensor<B, 3> {
        self.inputs_layer.forward(inputs)
    }

    fn forward(
        &self,
        query: Tensor<B, 2>,
        inputs: Tensor<B, 3>,
        processed_inputs: Tensor<B, 3>,
        mut state: AttentionState<B>,
    ) -> (Tensor<B, 2>, AttentionState<B>) {
        let [batch, tokens, encoder_channels] = inputs.dims();
        let processed_query = self
            .query_layer
            .forward(query)
            .reshape([batch, 1, ATTENTION_CHANNELS])
            .repeat_dim(1, tokens);
        let location = if self.location_attention {
            let attention_cat = Tensor::cat(
                vec![
                    state.weights.clone().unsqueeze_dim::<3>(1),
                    state.cumulative.clone().unsqueeze_dim::<3>(1),
                ],
                1,
            );
            self.location_dense(
                self.location_layer
                    .location_conv1d
                    .forward(attention_cat)
                    .swap_dims(1, 2),
            )
        } else {
            Tensor::zeros(
                [batch, tokens, ATTENTION_CHANNELS],
                &processed_inputs.device(),
            )
        };
        let energies = self
            .v
            .forward(tanh(processed_query + processed_inputs + location))
            .reshape([batch, tokens]);
        let weights = match self.normalization {
            TacotronAttentionNormalization::Softmax => softmax(energies, 1),
            TacotronAttentionNormalization::Sigmoid => {
                let probabilities = sigmoid(energies);
                probabilities.clone() / probabilities.sum_dim(1)
            }
        };
        state.cumulative = state.cumulative + weights.clone();
        state.weights = weights.clone();
        let context = weights
            .unsqueeze_dim::<3>(1)
            .matmul(inputs)
            .reshape([batch, encoder_channels]);
        (context, state)
    }

    fn location_dense(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.location_layer.location_dense.forward(input)
    }
}

#[derive(Module, Debug)]
pub struct TacotronStopnet<B: Backend> {
    pub linear_layer: Linear<B>,
}

impl<B: Backend> TacotronStopnet<B> {
    fn init(input_channels: usize, device: &B::Device) -> Self {
        Self {
            linear_layer: LinearConfig::new(input_channels, 1).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Tacotron2Decoder<B: Backend> {
    pub prenet: TacotronPrenet<B>,
    pub attention_rnn: PytorchLstmCell<B>,
    pub attention: TacotronOriginalAttention<B>,
    pub decoder_rnn: PytorchLstmCell<B>,
    pub linear_projection: LegacyLinear<B>,
    pub stopnet: TacotronStopnet<B>,
    encoder_channels: usize,
    frame_channels: usize,
    reduction_factor: usize,
}

struct DecoderState<B: Backend> {
    query: Tensor<B, 2>,
    attention_cell: Tensor<B, 2>,
    decoder_hidden: Tensor<B, 2>,
    decoder_cell: Tensor<B, 2>,
    context: Tensor<B, 2>,
    attention: AttentionState<B>,
}

struct DecoderStep<B: Backend> {
    frames: Tensor<B, 2>,
    stop_probability: Tensor<B, 2>,
    state: DecoderState<B>,
}

impl<B: Backend> Tacotron2Decoder<B> {
    fn init(
        encoder_channels: usize,
        frame_channels: usize,
        config: &TacotronInferenceConfig,
        device: &B::Device,
    ) -> Self {
        let reduction_channels = frame_channels * config.reduction_factor;
        Self {
            prenet: TacotronPrenet::init(
                frame_channels,
                config.prenet_dropout_at_inference,
                device,
            ),
            attention_rnn: PytorchLstmCell::init(
                PRENET_CHANNELS + encoder_channels,
                TACOTRON2_RNN_CHANNELS,
                device,
            ),
            attention: TacotronOriginalAttention::init(
                encoder_channels,
                config.attention_normalization,
                config.location_attention,
                device,
            ),
            decoder_rnn: PytorchLstmCell::init(
                TACOTRON2_RNN_CHANNELS + encoder_channels,
                TACOTRON2_RNN_CHANNELS,
                device,
            ),
            linear_projection: LegacyLinear::init(
                TACOTRON2_RNN_CHANNELS + encoder_channels,
                reduction_channels,
                true,
                device,
            ),
            stopnet: TacotronStopnet::init(TACOTRON2_RNN_CHANNELS + reduction_channels, device),
            encoder_channels,
            frame_channels,
            reduction_factor: config.reduction_factor,
        }
    }

    fn initial_state(&self, inputs: &Tensor<B, 3>) -> DecoderState<B> {
        let [batch, tokens, _] = inputs.dims();
        let device = inputs.device();
        DecoderState {
            query: Tensor::zeros([batch, TACOTRON2_RNN_CHANNELS], &device),
            attention_cell: Tensor::zeros([batch, TACOTRON2_RNN_CHANNELS], &device),
            decoder_hidden: Tensor::zeros([batch, TACOTRON2_RNN_CHANNELS], &device),
            decoder_cell: Tensor::zeros([batch, TACOTRON2_RNN_CHANNELS], &device),
            context: Tensor::zeros([batch, self.encoder_channels], &device),
            attention: AttentionState {
                weights: Tensor::zeros([batch, tokens], &device),
                cumulative: Tensor::zeros([batch, tokens], &device),
            },
        }
    }

    fn step(
        &self,
        memory: Tensor<B, 2>,
        inputs: Tensor<B, 3>,
        processed_inputs: Tensor<B, 3>,
        mut state: DecoderState<B>,
    ) -> DecoderStep<B> {
        let query_input = Tensor::cat(vec![self.prenet.forward(memory), state.context], 1);
        (state.query, state.attention_cell) =
            self.attention_rnn
                .step(query_input, state.query, state.attention_cell);
        (state.context, state.attention) = self.attention.forward(
            state.query.clone(),
            inputs,
            processed_inputs,
            state.attention,
        );
        let decoder_input = Tensor::cat(vec![state.query.clone(), state.context.clone()], 1);
        (state.decoder_hidden, state.decoder_cell) =
            self.decoder_rnn
                .step(decoder_input, state.decoder_hidden, state.decoder_cell);
        let hidden_context =
            Tensor::cat(vec![state.decoder_hidden.clone(), state.context.clone()], 1);
        let frames = self.linear_projection.forward(hidden_context);
        let stop_input = Tensor::cat(vec![state.decoder_hidden.clone(), frames.clone()], 1);
        let stop_probability = sigmoid(self.stopnet.linear_layer.forward(stop_input));
        DecoderStep {
            frames,
            stop_probability,
            state,
        }
    }

    fn next_memory(&self, frames: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch = frames.dims()[0];
        frames.slice([
            0..batch,
            self.frame_channels * (self.reduction_factor - 1)
                ..self.frame_channels * self.reduction_factor,
        ])
    }
}

#[derive(Module, Debug)]
pub struct Tacotron2Postnet<B: Backend> {
    pub convolutions: Vec<TacotronConvBnBlock<B>>,
}

impl<B: Backend> Tacotron2Postnet<B> {
    fn init(frame_channels: usize, device: &B::Device) -> Self {
        let mut convolutions = Vec::with_capacity(POSTNET_CONVOLUTIONS);
        convolutions.push(TacotronConvBnBlock::init(
            frame_channels,
            512,
            5,
            CONV_ACTIVATION_TANH,
            device,
        ));
        for _ in 1..POSTNET_CONVOLUTIONS - 1 {
            convolutions.push(TacotronConvBnBlock::init(
                512,
                512,
                5,
                CONV_ACTIVATION_TANH,
                device,
            ));
        }
        convolutions.push(TacotronConvBnBlock::init(
            512,
            frame_channels,
            5,
            CONV_ACTIVATION_LINEAR,
            device,
        ));
        Self { convolutions }
    }

    fn forward(&self, mut input: Tensor<B, 3>) -> Tensor<B, 3> {
        for convolution in &self.convolutions {
            input = convolution.forward(input);
        }
        input
    }
}

#[derive(Module, Debug)]
pub struct Tacotron2<B: Backend> {
    pub embedding: Embedding<B>,
    pub encoder: Tacotron2Encoder<B>,
    pub decoder: Tacotron2Decoder<B>,
    pub postnet: Tacotron2Postnet<B>,
    config: TacotronInferenceConfig,
    base_encoder_channels: usize,
}

impl TacotronInferenceConfig {
    pub fn init_tacotron2<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<Tacotron2<B>, TacotronError> {
        self.validate()
            .map_err(|error| config_error(error.to_string()))?;
        if self.architecture != TacotronArchitecture::Tacotron2 {
            return Err(config_error(
                "this runtime implements Tacotron 2; Tacotron 1 has a different CBHG encoder and decoder topology",
            ));
        }
        if self.forward_attention
            || self.forward_attention_mask
            || self.transition_agent
            || self.attention_windowing
        {
            return Err(config_error(
                "forward/transition/windowed attention is not implemented; import a location-sensitive original-attention checkpoint",
            ));
        }
        let style_channels = self
            .capacitron
            .as_ref()
            .map(|config| config.embedding_dim)
            .unwrap_or(0);
        let decoder_encoder_channels = self.encoder_channels + style_channels;
        Ok(Tacotron2 {
            embedding: EmbeddingConfig::new(self.num_chars, self.encoder_channels).init(device),
            encoder: Tacotron2Encoder::init(self.encoder_channels, device),
            decoder: Tacotron2Decoder::init(
                decoder_encoder_channels,
                self.out_channels,
                self,
                device,
            ),
            postnet: Tacotron2Postnet::init(self.out_channels, device),
            config: self.clone(),
            base_encoder_channels: self.encoder_channels,
        })
    }
}

impl<B: Backend> Tacotron2<B> {
    pub fn load_checkpoint(mut self, path: impl AsRef<Path>) -> Result<Self, TacotronError> {
        let result = crate::checkpoint::load_pytorch_layout_checkpoint(
            &mut self,
            path.as_ref(),
            crate::checkpoint::CheckpointLoadOptions {
                top_level_key: Some("model"),
                predicate: Some(tacotron_inference_tensor),
                key_remappings: vec![
                    (
                        r"(\.batch_normalization)\.weight$".into(),
                        "$1.gamma".into(),
                    ),
                    (r"(\.batch_normalization)\.bias$".into(), "$1.beta".into()),
                    (
                        r"^decoder\.stopnet\.1\.linear_layer\.".into(),
                        "decoder.stopnet.linear_layer.".into(),
                    ),
                ],
                skip_enum_variants: true,
                ..Default::default()
            },
        )
        .map_err(|error| TacotronError::Checkpoint(format!("{error:#}")))?;
        let unexpected_unused = result
            .unused
            .iter()
            .filter(|path| !path.ends_with(".num_batches_tracked"))
            .cloned()
            .collect::<Vec<_>>();
        if !result.missing.is_empty() || !result.errors.is_empty() || !unexpected_unused.is_empty()
        {
            return Err(TacotronError::Checkpoint(format!(
                "checkpoint does not exactly match the native inference graph: missing [{}], \
                 load errors [{}], unexpected tensors [{}]",
                result
                    .missing
                    .iter()
                    .map(|(source, target)| format!("{source}->{target}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                result
                    .errors
                    .iter()
                    .map(|error| error.to_string())
                    .collect::<Vec<_>>()
                    .join("; "),
                unexpected_unused.join(", ")
            )));
        }
        Ok(self)
    }

    pub fn inference(
        &self,
        token_ids: Tensor<B, 2, Int>,
        conditioning: TacotronConditioning<B>,
        controls: Option<TacotronControls>,
    ) -> Result<TacotronOutput<B>, TacotronError> {
        let [batch, input_tokens] = token_ids.dims();
        if batch != 1 {
            return Err(input_error(
                "autoregressive Tacotron inference currently requires batch size 1",
            ));
        }
        if input_tokens == 0 {
            return Err(input_error("token sequence must not be empty"));
        }
        let highest = token_ids.clone().max().into_scalar().elem::<i64>();
        if highest < 0 || highest as usize >= self.config.num_chars {
            return Err(input_error(format!(
                "token ID {highest} is outside the {}-symbol checkpoint vocabulary",
                self.config.num_chars
            )));
        }
        let controls = controls
            .unwrap_or_else(|| TacotronControls::from_config(&self.config))
            .validate()?;
        let mut encoder_outputs = self
            .encoder
            .forward(self.embedding.forward(token_ids).swap_dims(1, 2));
        encoder_outputs = self.apply_conditioning(encoder_outputs, conditioning)?;
        let processed_inputs = self.decoder.attention.preprocess(encoder_outputs.clone());
        let mut state = self.decoder.initial_state(&encoder_outputs);
        let device = encoder_outputs.device();
        let mut memory = Tensor::zeros([batch, self.config.out_channels], &device);
        let mut frame_groups = Vec::new();
        let mut alignments = Vec::new();
        let mut stop_probabilities = Vec::new();
        let mut termination = TacotronTermination::MaximumSteps;

        for step_index in 0..controls.max_decoder_steps {
            let step = self.decoder.step(
                memory,
                encoder_outputs.clone(),
                processed_inputs.clone(),
                state,
            );
            let stop = step
                .stop_probability
                .clone()
                .max()
                .into_scalar()
                .elem::<f32>();
            frame_groups.push(step.frames.clone().reshape([
                batch,
                self.config.reduction_factor,
                self.config.out_channels,
            ]));
            alignments.push(step.state.attention.weights.clone().unsqueeze_dim::<3>(1));
            stop_probabilities.push(step.stop_probability.clone());
            memory = self.decoder.next_memory(step.frames);
            state = step.state;
            if step_index + 1 >= controls.minimum_decoder_steps && stop > controls.stop_threshold {
                termination = TacotronTermination::StopToken;
                break;
            }
        }

        let decoder_mel = Tensor::cat(frame_groups, 1);
        let residual = self
            .postnet
            .forward(decoder_mel.clone().swap_dims(1, 2))
            .swap_dims(1, 2);
        let output = TacotronOutput {
            mel: decoder_mel.clone() + residual,
            decoder_mel,
            alignments: Tensor::cat(alignments, 1),
            stop_probabilities: Tensor::cat(stop_probabilities, 1),
            termination: termination.clone(),
        };
        if termination == TacotronTermination::MaximumSteps {
            let last_focus = output
                .alignments
                .clone()
                .slice([
                    0..1,
                    output.alignments.dims()[1] - 1..output.alignments.dims()[1],
                    0..input_tokens,
                ])
                .reshape([1, input_tokens])
                .argmax(1)
                .into_scalar()
                .elem::<i64>() as usize;
            return Err(TacotronError::AttentionFailure {
                steps: controls.max_decoder_steps,
                input_tokens,
                last_focus,
            });
        }
        Ok(output)
    }

    fn apply_conditioning(
        &self,
        encoder_outputs: Tensor<B, 3>,
        conditioning: TacotronConditioning<B>,
    ) -> Result<Tensor<B, 3>, TacotronError> {
        let [batch, tokens, channels] = encoder_outputs.dims();
        if channels != self.base_encoder_channels {
            return Err(input_error("encoder output channel mismatch"));
        }
        match (&self.config.capacitron, conditioning.style_embedding) {
            (None, None) => Ok(encoder_outputs),
            (None, Some(_)) => Err(input_error(
                "style embedding was supplied to a non-Capacitron checkpoint",
            )),
            (Some(config), Some(style)) => {
                if style.dims() != [batch, config.embedding_dim] {
                    return Err(input_error(format!(
                        "Capacitron style embedding has shape {:?}; expected [{batch}, {}]",
                        style.dims(),
                        config.embedding_dim
                    )));
                }
                Ok(Tensor::cat(
                    vec![
                        encoder_outputs,
                        style
                            .reshape([batch, 1, config.embedding_dim])
                            .repeat_dim(1, tokens),
                    ],
                    2,
                ))
            }
            (Some(config), None) => {
                let style = Tensor::<B, 2>::random(
                    [batch, config.embedding_dim],
                    Distribution::Normal(0.0, 1.0),
                    &encoder_outputs.device(),
                )
                .reshape([batch, 1, config.embedding_dim])
                .repeat_dim(1, tokens);
                Ok(Tensor::cat(vec![encoder_outputs, style], 2))
            }
        }
    }
}

fn tacotron_inference_tensor(source: &str, _target: &str) -> bool {
    !source.starts_with("coarse_decoder.")
        && !source.starts_with("decoder_backward.")
        && !source.starts_with("capacitron_vae_layer.")
        && !source.starts_with("gst_layer.")
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::TensorData;

    use super::*;
    use crate::TacotronVariant;

    type TestBackend = NdArray<f32>;

    fn tiny_config() -> TacotronInferenceConfig {
        TacotronInferenceConfig {
            architecture: TacotronArchitecture::Tacotron2,
            variant: TacotronVariant::Plain,
            num_chars: 16,
            out_channels: 4,
            encoder_channels: 8,
            decoder_channels: 8,
            reduction_factor: 2,
            ddc_reduction_factor: None,
            max_decoder_steps: 4,
            stop_threshold: 0.5,
            location_attention: true,
            attention_normalization: TacotronAttentionNormalization::Softmax,
            attention_windowing: false,
            forward_attention: false,
            forward_attention_mask: false,
            transition_agent: false,
            prenet_dropout_at_inference: false,
            separate_stopnet: true,
            capacitron: None,
        }
    }

    #[test]
    fn location_attention_normalizes_and_accumulates_repeated_positions() {
        let device = NdArrayDevice::Cpu;
        let attention = TacotronOriginalAttention::<TestBackend>::init(
            8,
            TacotronAttentionNormalization::Softmax,
            true,
            &device,
        );
        let inputs = Tensor::from_data(
            TensorData::new(
                (0..40).map(|value| value as f32 / 40.0).collect(),
                [1, 5, 8],
            ),
            &device,
        );
        let processed = attention.preprocess(inputs.clone());
        let state = AttentionState {
            weights: Tensor::zeros([1, 5], &device),
            cumulative: Tensor::zeros([1, 5], &device),
        };
        let query = Tensor::zeros([1, TACOTRON2_RNN_CHANNELS], &device);
        let (_, state) = attention.forward(query, inputs, processed, state);
        let weights = state.weights.into_data().to_vec::<f32>().unwrap();
        let cumulative = state.cumulative.into_data().to_vec::<f32>().unwrap();
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(weights.len(), 5);
        for (weight, cumulative) in weights.iter().zip(cumulative) {
            assert!((weight - cumulative).abs() < 1e-6);
        }
    }

    #[test]
    fn stop_controls_reject_unbounded_or_invalid_thresholds() {
        assert!(TacotronControls {
            stop_threshold: f32::NAN,
            max_decoder_steps: 100,
            minimum_decoder_steps: 1,
        }
        .validate()
        .is_err());
        assert!(TacotronControls {
            stop_threshold: 0.5,
            max_decoder_steps: 1,
            minimum_decoder_steps: 1,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn short_repeated_and_long_token_shapes_are_accepted_by_encoder() {
        let device = NdArrayDevice::Cpu;
        let model = tiny_config()
            .init_tacotron2::<TestBackend>(&device)
            .expect("tiny Tacotron2");
        for ids in [
            vec![1],
            vec![2; 8],
            (0..128).map(|index| index % 16).collect(),
        ] {
            let length = ids.len();
            let input = Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(ids.into_iter().map(i64::from).collect(), [1, length]),
                &device,
            );
            let output = model
                .encoder
                .forward(model.embedding.forward(input).swap_dims(1, 2));
            assert_eq!(output.dims(), [1, length, 8]);
        }
    }

    #[test]
    fn attention_limit_is_an_error_instead_of_a_truncated_success() {
        let device = NdArrayDevice::Cpu;
        let model = tiny_config()
            .init_tacotron2::<TestBackend>(&device)
            .expect("tiny Tacotron2");
        let input = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 1, 1], [1, 3]),
            &device,
        );
        let error = model
            .inference(
                input,
                TacotronConditioning::default(),
                Some(TacotronControls {
                    stop_threshold: 1.0,
                    max_decoder_steps: 2,
                    minimum_decoder_steps: 1,
                }),
            )
            .expect_err("a stop probability cannot exceed one");
        assert!(matches!(
            error,
            TacotronError::AttentionFailure {
                steps: 2,
                input_tokens: 3,
                ..
            }
        ));
    }

    #[test]
    fn released_ddc_checkpoint_loads_when_fixture_is_available() {
        let Some(config_path) = std::env::var_os("TONGUES_TEST_COQUI_TACOTRON2_CONFIG") else {
            return;
        };
        let checkpoint_path = std::env::var_os("TONGUES_TEST_COQUI_TACOTRON2_MODEL")
            .expect("TONGUES_TEST_COQUI_TACOTRON2_MODEL must accompany config");
        let source = std::fs::read_to_string(config_path).unwrap();
        let root: serde_json::Value = json5::from_str(&source).unwrap();
        let config = TacotronInferenceConfig::from_json_value(&root).unwrap();
        assert!(config.variant.uses_ddc());
        config
            .init_tacotron2::<TestBackend>(&NdArrayDevice::Cpu)
            .unwrap()
            .load_checkpoint(checkpoint_path)
            .expect("released Tacotron2-DDC checkpoint");
    }
}
