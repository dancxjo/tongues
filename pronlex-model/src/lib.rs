//! Burn-based sequence-to-sequence translation model and training loop.
//!
//! Maps between Spelling (Graphemes), Phonemes (Broad IPA), and Phones (Narrow IPA)
//! using task prefix tokens.

use std::path::Path;

use anyhow::{Context, Result};
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig,
    transformer::{
        TransformerEncoder, TransformerEncoderConfig,
        TransformerDecoder, TransformerDecoderConfig,
        TransformerDecoderInput,
    },
};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;
use rand::Rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use pronlex_core::{PAD_ID, Vocab, BOS_ID, EOS_ID};
use pronlex_data::{
    Lexeme, Task, Seq2SeqExample, collate_batch, make_seq2seq_example,
};

// ── Model configuration ────────────────────────────────────────────────────

/// Architecture hyper-parameters (serialized alongside the model).
#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Unified vocabulary size (set by prepare).
    pub vocab_size: usize,
    /// Model dimension.
    #[config(default = 128)]
    pub d_model: usize,
    /// Number of attention heads.
    #[config(default = 4)]
    pub n_heads: usize,
    /// Number of Transformer layers (both encoder and decoder).
    #[config(default = 3)]
    pub n_layers: usize,
    /// Feed-forward hidden dimension.
    #[config(default = 512)]
    pub d_ff: usize,
    /// Dropout rate.
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Maximum sequence length for position embeddings.
    #[config(default = 128)]
    pub max_seq_len: usize,
}

impl ModelConfig {
    /// Initialize a new model on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Seq2SeqModel<B> {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let pos_embedding = EmbeddingConfig::new(self.max_seq_len, self.d_model).init(device);

        let encoder = TransformerEncoderConfig::new(
            self.d_model,
            self.d_ff,
            self.n_heads,
            self.n_layers,
        )
        .with_dropout(self.dropout)
        .init(device);

        let decoder = TransformerDecoderConfig::new(
            self.d_model,
            self.d_ff,
            self.n_heads,
            self.n_layers,
        )
        .with_dropout(self.dropout)
        .init(device);

        let classifier = LinearConfig::new(self.d_model, self.vocab_size).init(device);
        let dropout = DropoutConfig::new(self.dropout).init();

        Seq2SeqModel {
            embedding,
            pos_embedding,
            encoder,
            decoder,
            classifier,
            dropout,
        }
    }
}

// ── Training configuration ─────────────────────────────────────────────────

/// Training hyper-parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Initial learning rate (AdamW). Default: 3e-4.
    pub learning_rate: f64,
    /// AdamW weight decay. Default: 1e-4.
    pub weight_decay: f32,
    /// Dropout rate (must match `ModelConfig::dropout`). Default: 0.1.
    pub dropout: f64,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Maximum number of training epochs.
    pub epochs: usize,
    /// Early stopping: stop if validation loss does not improve for this
    /// many consecutive epochs.
    pub early_stopping_patience: usize,
    /// Optional training direction task (None = Both).
    #[serde(default)]
    pub task: Option<Task>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            learning_rate: 3e-4,
            weight_decay: 1e-4,
            dropout: 0.1,
            batch_size: 64,
            epochs: 20,
            early_stopping_patience: 5,
            task: None,
        }
    }
}

// ── Model ──────────────────────────────────────────────────────────────────

/// Sequence-to-sequence Transformer model.
#[derive(Module, Debug)]
pub struct Seq2SeqModel<B: Backend> {
    embedding: Embedding<B>,
    pos_embedding: Embedding<B>,
    encoder: TransformerEncoder<B>,
    decoder: TransformerDecoder<B>,
    classifier: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> Seq2SeqModel<B> {
    /// Forward pass.
    ///
    /// # Arguments
    /// * `src_ids`      - `[batch, src_len]` source token IDs.
    /// * `tgt_in_ids`   - `[batch, tgt_len]` target input token IDs (starts with BOS).
    /// * `src_pad_mask` - `[batch, src_len]` padding mask (true for PAD).
    /// * `tgt_pad_mask` - `[batch, tgt_len]` padding mask (true for PAD).
    ///
    /// # Returns
    /// Logits `[batch, tgt_len, vocab_size]`.
    pub fn forward(
        &self,
        src_ids: Tensor<B, 2, Int>,
        tgt_in_ids: Tensor<B, 2, Int>,
        src_pad_mask: Tensor<B, 2, Bool>,
        tgt_pad_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        let [batch, src_len] = src_ids.dims();
        let [_, tgt_len] = tgt_in_ids.dims();
        let device = src_ids.device();

        // 1. Embed source
        let src_pos = Tensor::arange(0..src_len as i64, &device)
            .unsqueeze_dim::<2>(0)
            .repeat_dim(0, batch);
        let src_emb = self.embedding.forward(src_ids) + self.pos_embedding.forward(src_pos);
        let src_emb = self.dropout.forward(src_emb);

        // 2. Encode source
        let encoder_input = burn::nn::transformer::TransformerEncoderInput::new(src_emb)
            .mask_pad(src_pad_mask.clone());
        let memory = self.encoder.forward(encoder_input);

        // 3. Embed target input
        let tgt_pos = Tensor::arange(0..tgt_len as i64, &device)
            .unsqueeze_dim::<2>(0)
            .repeat_dim(0, batch);
        let tgt_emb = self.embedding.forward(tgt_in_ids) + self.pos_embedding.forward(tgt_pos);
        let tgt_emb = self.dropout.forward(tgt_emb);

        // 4. Generate causal self-attention mask for target
        let tgt_attn_mask = burn::nn::attention::generate_autoregressive_mask(batch, tgt_len, &device);

        // 5. Decode target
        let decoder_input = TransformerDecoderInput::new(tgt_emb, memory)
            .target_mask_pad(tgt_pad_mask)
            .memory_mask_pad(src_pad_mask)
            .target_mask_attn(tgt_attn_mask);
        let out = self.decoder.forward(decoder_input);

        // 6. Classify
        self.classifier.forward(out)
    }

    /// Autoregressively decode a target sequence given a source sequence.
    pub fn generate(
        &self,
        src_ids: Tensor<B, 2, Int>,
        max_tgt_len: usize,
    ) -> Vec<u32> {
        let device = src_ids.device();
        let [batch, src_len] = src_ids.dims();
        assert_eq!(batch, 1, "Only batch size 1 supported for inference generation");

        // 1. Encode source
        let src_pos = Tensor::arange(0..src_len as i64, &device).unsqueeze_dim::<2>(0);
        let src_emb = self.embedding.forward(src_ids.clone()) + self.pos_embedding.forward(src_pos);
        let src_pad_mask = src_ids.equal_elem(PAD_ID as i32);

        let encoder_input = burn::nn::transformer::TransformerEncoderInput::new(src_emb)
            .mask_pad(src_pad_mask.clone());
        let memory = self.encoder.forward(encoder_input);

        // 2. Autoregressive loop
        let mut generated = vec![BOS_ID];

        for _ in 0..max_tgt_len {
            let tgt_len = generated.len();

            let tgt_in_ids = Tensor::<B, 2, Int>::from_data(
                TensorData::new(generated.iter().map(|&x| x as i32).collect::<Vec<_>>(), [1, tgt_len]),
                &device,
            );

            let tgt_pos = Tensor::arange(0..tgt_len as i64, &device).unsqueeze_dim::<2>(0);
            let tgt_emb = self.embedding.forward(tgt_in_ids.clone()) + self.pos_embedding.forward(tgt_pos);

            let tgt_pad_mask = tgt_in_ids.equal_elem(PAD_ID as i32);
            let tgt_attn_mask = burn::nn::attention::generate_autoregressive_mask(1, tgt_len, &device);

            let decoder_input = TransformerDecoderInput::new(tgt_emb, memory.clone())
                .target_mask_pad(tgt_pad_mask)
                .memory_mask_pad(src_pad_mask.clone())
                .target_mask_attn(tgt_attn_mask);

            let out = self.decoder.forward(decoder_input);

            // Get logits for the LAST token
            let classifier_device = self.classifier.clone().to_device(&device);
            let d_out = out.dims()[2];
            let last_out = out.slice([0..1, (tgt_len - 1)..tgt_len, 0..d_out]);
            let logits = classifier_device.forward(last_out).squeeze_dim::<2>(1);

            let scores: Vec<f32> = logits.into_data().to_vec().unwrap();
            let next_token = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(EOS_ID);

            if next_token == EOS_ID {
                break;
            }
            generated.push(next_token);
        }

        generated.into_iter().skip(1).collect()
    }
}

// ── Checkpoint State ───────────────────────────────────────────────────────

/// Saved checkpoint training state to pick up where training left off.
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainState {
    /// Last completed epoch.
    pub current_epoch: usize,
    /// Best validation loss achieved.
    pub best_val_loss: f32,
}

// ── Helpers ────────────────────────────────────────────────────────────────

type FileRecorder = BinFileRecorder<FullPrecisionSettings>;

fn make_recorder() -> FileRecorder {
    FileRecorder::new()
}

/// Compute cross-entropy loss over all target tokens.
pub fn seq2seq_loss<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    let [batch, seq_len, vocab] = logits.dims();
    let device = logits.device();
    let ce = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![PAD_ID as usize]))
        .init::<B>(&device);

    let logits_flat = logits.reshape([batch * seq_len, vocab]);
    let targets_flat = targets.reshape([batch * seq_len]);
    ce.forward(logits_flat, targets_flat)
}

// ── Training ───────────────────────────────────────────────────────────────

/// Train the model for one epoch. Returns the mean training loss.
pub fn train_epoch<B: AutodiffBackend, R: Rng>(
    model: &mut Seq2SeqModel<B>,
    optimizer: &mut impl Optimizer<Seq2SeqModel<B>, B>,
    lexemes: &[Lexeme],
    vocab: &Vocab,
    config: &TrainConfig,
    _epoch: usize,
    device: &B::Device,
    rng: &mut R,
    pb: &indicatif::ProgressBar,
) -> f32 {
    let mut indices: Vec<usize> = (0..lexemes.len()).collect();
    indices.shuffle(rng);

    let mut total_loss = 0f32;
    let mut n_batches = 0usize;

    for chunk in indices.chunks(config.batch_size) {
        let examples: Vec<Seq2SeqExample> = chunk
            .iter()
            .map(|&i| {
                let lex = &lexemes[i];
                let task = config.task.unwrap_or_else(|| Task::sample(rng));
                make_seq2seq_example(lex, task, vocab)
            })
            .collect();

        if examples.is_empty() {
            continue;
        }

        let max_src = examples.iter().map(|ex| ex.src_ids.len()).max().unwrap_or(1);
        let max_tgt = examples.iter().map(|ex| ex.tgt_in_ids.len()).max().unwrap_or(1);
        let batch = collate_batch(&examples, max_src, max_tgt);

        let b = batch.size;
        let src_flat: Vec<i32> = batch.src_ids.into_iter().flatten().collect();
        let tgt_in_flat: Vec<i32> = batch.tgt_in_ids.into_iter().flatten().collect();
        let tgt_out_flat: Vec<i32> = batch.tgt_out_ids.into_iter().flatten().collect();
        let src_pad_flat: Vec<bool> = batch.src_pad_mask.into_iter().flatten().collect();
        let tgt_pad_flat: Vec<bool> = batch.tgt_pad_mask.into_iter().flatten().collect();

        let src_ids = Tensor::<B, 2, Int>::from_data(TensorData::new(src_flat, [b, max_src]), device);
        let tgt_in_ids = Tensor::<B, 2, Int>::from_data(TensorData::new(tgt_in_flat, [b, max_tgt]), device);
        let tgt_out_ids = Tensor::<B, 2, Int>::from_data(TensorData::new(tgt_out_flat, [b, max_tgt]), device);
        let src_pad_mask = Tensor::<B, 2, Bool>::from_data(TensorData::new(src_pad_flat, [b, max_src]), device);
        let tgt_pad_mask = Tensor::<B, 2, Bool>::from_data(TensorData::new(tgt_pad_flat, [b, max_tgt]), device);

        let logits = model.forward(src_ids, tgt_in_ids, src_pad_mask, tgt_pad_mask);
        let loss = seq2seq_loss(logits, tgt_out_ids);

        let grads = GradientsParams::from_grads(loss.backward(), model);
        *model = optimizer.step(config.learning_rate, model.clone(), grads);

        let loss_val: f32 = loss.into_scalar().elem();
        total_loss += loss_val;
        n_batches += 1;
        pb.set_message(format!("{:.4}", total_loss / n_batches as f32));
        pb.inc(1);
    }

    if n_batches == 0 {
        0.0
    } else {
        total_loss / n_batches as f32
    }
}

/// Evaluate the model on a set of lexemes.
pub fn evaluate<B: Backend, R: Rng>(
    model: &Seq2SeqModel<B>,
    lexemes: &[Lexeme],
    vocab: &Vocab,
    task_filter: Option<Task>,
    device: &B::Device,
    _rng: &mut R,
) -> (f32, f32, f32) {
    if lexemes.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut total_loss = 0f32;
    let mut exact_matches = 0usize;
    let mut n_batches = 0usize;
    let mut total_tokens = 0usize;
    let mut matched_tokens = 0usize;

    // Use a subset of up to 1000 items to keep validation fast
    let eval_lexemes = if lexemes.len() > 1000 {
        &lexemes[0..1000]
    } else {
        lexemes
    };

    let mut total_examples = 0usize;
    let examples: Vec<Seq2SeqExample> = eval_lexemes
        .iter()
        .map(|lex| {
            let task = match task_filter {
                Some(t) => t,
                None => {
                    let t = if total_examples % 2 == 0 {
                        Task::S2Pm
                    } else {
                        Task::Pm2S
                    };
                    total_examples += 1;
                    t
                }
            };
            make_seq2seq_example(lex, task, vocab)
        })
        .collect();

    for chunk in examples.chunks(64) {
        let max_src = chunk.iter().map(|ex| ex.src_ids.len()).max().unwrap_or(1);
        let max_tgt = chunk.iter().map(|ex| ex.tgt_in_ids.len()).max().unwrap_or(1);
        let batch = collate_batch(chunk, max_src, max_tgt);

        let b = batch.size;
        let src_flat: Vec<i32> = batch.src_ids.into_iter().flatten().collect();
        let tgt_in_flat: Vec<i32> = batch.tgt_in_ids.into_iter().flatten().collect();
        let tgt_out_flat: Vec<i32> = batch.tgt_out_ids.into_iter().flatten().collect();
        let src_pad_flat: Vec<bool> = batch.src_pad_mask.into_iter().flatten().collect();
        let tgt_pad_flat: Vec<bool> = batch.tgt_pad_mask.into_iter().flatten().collect();

        let src_ids = Tensor::<B, 2, Int>::from_data(TensorData::new(src_flat, [b, max_src]), device);
        let tgt_in_ids = Tensor::<B, 2, Int>::from_data(TensorData::new(tgt_in_flat, [b, max_tgt]), device);
        let tgt_out_ids = Tensor::<B, 2, Int>::from_data(TensorData::new(tgt_out_flat, [b, max_tgt]), device);
        let src_pad_mask = Tensor::<B, 2, Bool>::from_data(TensorData::new(src_pad_flat, [b, max_src]), device);
        let tgt_pad_mask = Tensor::<B, 2, Bool>::from_data(TensorData::new(tgt_pad_flat, [b, max_tgt]), device);

        let logits = model.forward(src_ids, tgt_in_ids, src_pad_mask, tgt_pad_mask);
        let loss = seq2seq_loss(logits.clone(), tgt_out_ids.clone());
        total_loss += loss.into_scalar().elem::<f32>();
        n_batches += 1;

        let [_, tgt_len, vocab_size] = logits.dims();
        for i in 0..b {
            let mut matched = true;
            for j in 0..tgt_len {
                let tgt_id = chunk[i].tgt_out_ids.get(j).copied().unwrap_or(PAD_ID);
                if tgt_id == PAD_ID {
                    break;
                }
                total_tokens += 1;
                let pos_logits = logits
                    .clone()
                    .slice([i..i+1, j..j+1, 0..vocab_size])
                    .reshape([vocab_size]);
                let pos_logits_vec: Vec<f32> = pos_logits.into_data().to_vec().unwrap();
                let pred = pos_logits_vec
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0);
                if pred == tgt_id {
                    matched_tokens += 1;
                } else {
                    matched = false;
                }
            }
            if matched {
                exact_matches += 1;
            }
        }
    }

    let mean_loss = if n_batches > 0 {
        total_loss / n_batches as f32
    } else {
        0.0
    };
    let acc = if eval_lexemes.is_empty() {
        0.0
    } else {
        exact_matches as f32 / eval_lexemes.len() as f32
    };
    let token_acc = if total_tokens > 0 {
        matched_tokens as f32 / total_tokens as f32
    } else {
        0.0
    };

    (mean_loss, acc, token_acc)
}

// ── Full training loop ─────────────────────────────────────────────────────

/// Run the complete training loop with early stopping, loading state from disk if present.
pub fn train<B: AutodiffBackend, R: Rng>(
    model_config: &ModelConfig,
    train_config: &TrainConfig,
    train_lexemes: &[Lexeme],
    valid_lexemes: &[Lexeme],
    vocab: &Vocab,
    model_path: &Path,
    device: &B::Device,
    rng: &mut R,
) -> Result<f32>
where
    <Seq2SeqModel<B> as Module<B>>::Record: Send,
{
    let out_dir = model_path.parent().unwrap_or(Path::new("."));
    let state_path = out_dir.join("train_state.json");
    let model_file = model_path.with_extension("bin");

    let mut start_epoch = 1usize;
    let mut best_val_loss = f32::INFINITY;

    let mut model: Seq2SeqModel<B> = if model_file.exists() && state_path.exists() {
        println!("Resuming training from checkpoint: {}", model_file.display());
        let loaded_model = model_config
            .init(device)
            .load_file(model_path, &make_recorder(), device)
            .context("loading model weights")?;

        let state_data = std::fs::read_to_string(&state_path).context("reading train_state.json")?;
        let state: TrainState = serde_json::from_str(&state_data).context("parsing train_state.json")?;

        start_epoch = state.current_epoch + 1;
        best_val_loss = state.best_val_loss;
        println!("Resuming from epoch {} (previous best val loss: {:.4})", start_epoch, best_val_loss);

        loaded_model
    } else {
        println!("No existing checkpoint found. Initializing new model weights...");
        model_config.init(device)
    };

    if start_epoch > train_config.epochs {
        println!("Model has already completed all requested {} epochs.", train_config.epochs);
        return Ok(best_val_loss);
    }

    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(train_config.weight_decay)
        .init::<B, Seq2SeqModel<B>>();
    let mut patience_counter = 0usize;

    let mut last_train_loss = None;
    let mut last_val_loss = None;
    let mut last_val_acc = None;
    let mut last_val_token_acc = None;

    for epoch in start_epoch..=train_config.epochs {
        let n_batches = (train_lexemes.len() + train_config.batch_size - 1) / train_config.batch_size;
        let pb = indicatif::ProgressBar::new(n_batches as u64);
        let template = if let (Some(tl), Some(vl), Some(va), Some(vt)) = (last_train_loss, last_val_loss, last_val_acc, last_val_token_acc) {
            format!(
                "{{spinner:.green}} Epoch {}/{} [{{elapsed_precise}}] [{{bar:40.cyan/blue}}] {{pos}}/{{len}} Loss: {{msg}} (prev: train={:.4} val={:.4} exact={:.3} token={:.3})",
                epoch,
                train_config.epochs,
                tl,
                vl,
                va,
                vt
            )
        } else {
            format!(
                "{{spinner:.green}} Epoch {}/{} [{{elapsed_precise}}] [{{bar:40.cyan/blue}}] {{pos}}/{{len}} Loss: {{msg}} ({} train / {} valid)",
                epoch,
                train_config.epochs,
                train_lexemes.len(),
                valid_lexemes.len()
            )
        };
        pb.set_style(
            indicatif::ProgressStyle::default_bar()
                .template(&template)
                .expect("valid template")
                .progress_chars("#>-")
        );
        pb.set_message("...");

        let train_loss = train_epoch(
            &mut model,
            &mut optimizer,
            train_lexemes,
            vocab,
            train_config,
            epoch,
            device,
            rng,
            &pb,
        );

        pb.set_message("evaluating...");
        let eval_model: Seq2SeqModel<B::InnerBackend> = model.valid();
        let (val_loss, val_acc, val_token_acc) = evaluate(&eval_model, valid_lexemes, vocab, train_config.task, device, rng);

        pb.finish_and_clear();

        println!(
            "Epoch {:3} | train_loss={:.4}  val_loss={:.4}  val_exact_match={:.3}  val_token_acc={:.3}",
            epoch, train_loss, val_loss, val_acc, val_token_acc
        );

        last_train_loss = Some(train_loss);
        last_val_loss = Some(val_loss);
        last_val_acc = Some(val_acc);
        last_val_token_acc = Some(val_token_acc);

        // Save progress for resume
        let current_state = TrainState {
            current_epoch: epoch,
            best_val_loss,
        };
        let state_json = serde_json::to_string_pretty(&current_state)?;
        std::fs::write(&state_path, state_json)?;

        // Early stopping & saving best model
        if val_loss < best_val_loss - 1e-5 {
            best_val_loss = val_loss;
            patience_counter = 0;

            eval_model
                .save_file(model_path, &make_recorder())
                .context("saving best model weights")?;
            println!("  ✓ New best model saved (val_loss={:.4})", best_val_loss);

            // Update best val loss in train state
            let best_state = TrainState {
                current_epoch: epoch,
                best_val_loss,
            };
            std::fs::write(&state_path, serde_json::to_string_pretty(&best_state)?)?;
        } else {
            patience_counter += 1;
            println!(
                "  (no improvement, patience {}/{})",
                patience_counter, train_config.early_stopping_patience
            );
            if patience_counter >= train_config.early_stopping_patience {
                println!("Early stopping at epoch {}", epoch);
                break;
            }
        }
    }

    Ok(best_val_loss)
}

// ── Evaluation report ──────────────────────────────────────────────────────

/// Sequence translation evaluation report.
#[derive(Debug)]
pub struct EvalReport {
    pub exact_match_accuracy: f32,
    pub val_loss: f32,
    pub token_accuracy: f32,
}

/// Produce sequence-to-sequence evaluation metrics.
pub fn eval_report<B: Backend, R: Rng>(
    model: &Seq2SeqModel<B>,
    test_lexemes: &[Lexeme],
    _train_lexemes: &[Lexeme],
    vocab: &Vocab,
    task_filter: Option<Task>,
    device: &B::Device,
    rng: &mut R,
) -> EvalReport {
    let (loss, acc, token_acc) = evaluate(model, test_lexemes, vocab, task_filter, device, rng);
    EvalReport {
        exact_match_accuracy: acc,
        val_loss: loss,
        token_accuracy: token_acc,
    }
}

// ── Prediction ─────────────────────────────────────────────────────────────

/// Perform sequence translation prediction.
pub fn predict<B: Backend>(
    model: &Seq2SeqModel<B>,
    input_str: &str,
    task: Task,
    vocab: &Vocab,
    device: &B::Device,
) -> String {
    // 1. Format the source sequence with task token
    let mut src_ids = vec![task.get_prefix_id()];
    src_ids.extend(vocab.encode_string(input_str));

    let src_len = src_ids.len();
    let src_tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(src_ids.iter().map(|&x| x as i32).collect::<Vec<_>>(), [1, src_len]),
        device,
    );

    // 2. Decode autoregressively
    let pred_ids = model.generate(src_tensor, 128);

    // 3. Decode vocabulary IDs back to text
    vocab.decode_ids(&pred_ids)
}

// ── Load / save helpers ────────────────────────────────────────────────────

/// Save a model to path.
pub fn save_model<B: Backend>(model: &Seq2SeqModel<B>, path: &Path) -> Result<()> {
    model
        .clone()
        .save_file(path, &make_recorder())
        .with_context(|| format!("saving model to {}", path.display()))
}

/// Load a model from path.
pub fn load_model<B: Backend>(
    model_config: &ModelConfig,
    path: &Path,
    device: &B::Device,
) -> Result<Seq2SeqModel<B>> {
    model_config
        .init::<B>(device)
        .load_file(path, &make_recorder(), device)
        .with_context(|| format!("loading model from {}", path.display()))
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn model_forward_shape() {
        let vocab = Vocab::build(
            &vec!["cat".to_string(), "dog".to_string()],
            &vec!["kæt".to_string(), "dɔɡ".to_string()],
            &[],
        );
        let device = Default::default();
        let config = ModelConfig::new(vocab.size()).with_d_model(16).with_n_heads(2).with_n_layers(1);
        let model = config.init::<TestBackend>(&device);

        let src = Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![5i32, 11i32, 12i32], [1, 3]), &device);
        let tgt = Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![2i32, 13i32, 14i32], [1, 3]), &device);
        let src_mask = Tensor::<TestBackend, 2, Bool>::from_data(TensorData::new(vec![false, false, false], [1, 3]), &device);
        let tgt_mask = Tensor::<TestBackend, 2, Bool>::from_data(TensorData::new(vec![false, false, false], [1, 3]), &device);

        let logits = model.forward(src, tgt, src_mask, tgt_mask);
        assert_eq!(logits.dims(), [1, 3, vocab.size()]);
    }
}
