//! Burn-based masked-phone prediction model and training loop.
//!
//! ## Architecture
//!
//! ```text
//! Input:
//!   char_ids   [B, W]  ─── embedding ──────┐
//!                       + pos embedding      │ masked mean-pool → [B, D]
//!                                            │
//!   phone_ids  [B, P]  ─── embedding ────┐  │
//!                       + pos embedding   │  └─ broadcast-add ──→ [B, P, D]
//!                                         │
//!                                   Transformer encoder (2 layers)
//!                                         │
//!                                   Linear → [B, P, phone_vocab]
//! ```
//!
//! Loss is computed only at masked positions (targets ≠ PAD_ID) using
//! CrossEntropyLoss with `pad_tokens = [0]`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig,
    transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;
use rand::Rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use pronlex_core::{PAD_ID, Vocab};
use pronlex_data::{
    Batch, Lexeme, MaskPolicy, MaskedExample, apply_mask, collate_batch,
    generate_single_mask_examples, most_common_phone, most_common_phone_by_position,
    sample_mask_spec,
};

// ── Model configuration ────────────────────────────────────────────────────

/// Architecture hyper-parameters (serialised alongside the model).
#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Char vocabulary size (set by prepare).
    pub char_vocab_size: usize,
    /// Phone vocabulary size (set by prepare).
    pub phone_vocab_size: usize,
    /// Model dimension.
    #[config(default = 64)]
    pub d_model: usize,
    /// Number of attention heads.
    #[config(default = 2)]
    pub n_heads: usize,
    /// Number of Transformer layers.
    #[config(default = 2)]
    pub n_layers: usize,
    /// Feed-forward hidden dimension.
    #[config(default = 256)]
    pub d_ff: usize,
    /// Dropout rate.
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Maximum spelling length (chars) – used for position embeddings.
    #[config(default = 64)]
    pub max_word_chars: usize,
    /// Maximum phone sequence length – used for position embeddings.
    #[config(default = 32)]
    pub max_phones: usize,
}

impl ModelConfig {
    /// Initialise a new model on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PhonePredictorModel<B> {
        let transformer = TransformerEncoderConfig::new(
            self.d_model,
            self.d_ff,
            self.n_heads,
            self.n_layers,
        )
        .with_dropout(self.dropout)
        .init(device);

        PhonePredictorModel {
            char_embedding: EmbeddingConfig::new(self.char_vocab_size, self.d_model).init(device),
            phone_embedding: EmbeddingConfig::new(self.phone_vocab_size, self.d_model)
                .init(device),
            char_pos_embedding: EmbeddingConfig::new(self.max_word_chars, self.d_model)
                .init(device),
            phone_pos_embedding: EmbeddingConfig::new(self.max_phones, self.d_model).init(device),
            transformer,
            classifier: LinearConfig::new(self.d_model, self.phone_vocab_size).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            d_model: self.d_model,
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
    /// Masking policy (Single | Variable).
    #[serde(skip)]
    pub mask_policy: MaskPolicy,
    /// Maximum fraction of phones to mask in random-pct mode.
    pub max_mask_rate: f64,
    /// Probability weight given to span masking.
    pub span_mask_prob: f64,
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
            mask_policy: MaskPolicy::default(),
            max_mask_rate: 0.4,
            span_mask_prob: 0.15,
        }
    }
}

// ── Model ──────────────────────────────────────────────────────────────────

/// Masked-phone predictor using a 2-layer Transformer encoder.
#[derive(Module, Debug)]
pub struct PhonePredictorModel<B: Backend> {
    char_embedding: Embedding<B>,
    phone_embedding: Embedding<B>,
    char_pos_embedding: Embedding<B>,
    phone_pos_embedding: Embedding<B>,
    transformer: TransformerEncoder<B>,
    classifier: Linear<B>,
    dropout: Dropout,
    /// Stored for broadcasting (not a learned parameter).
    #[module(skip)]
    d_model: usize,
}

impl<B: Backend> PhonePredictorModel<B> {
    /// Forward pass.
    ///
    /// # Arguments
    /// * `char_ids`  – `[batch, max_word_len]` char token IDs (PAD=0).
    /// * `phone_ids` – `[batch, max_phone_len]` phone token IDs with `MASK_ID`
    ///                  at masked positions.
    ///
    /// # Returns
    /// Logits `[batch, max_phone_len, phone_vocab]`.
    pub fn forward(
        &self,
        char_ids: Tensor<B, 2, Int>,
        phone_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [batch, word_len] = char_ids.dims();
        let [_, phone_len] = phone_ids.dims();
        let device = char_ids.device();

        // ── Char embeddings + positions ────────────────────────────────
        let char_pos: Tensor<B, 2, Int> = Tensor::arange(0..word_len as i64, &device)
            .unsqueeze_dim::<2>(0)
            .repeat_dim(0, batch);
        let char_emb = self.char_embedding.forward(char_ids.clone())
            + self.char_pos_embedding.forward(char_pos);
        let char_emb = self.dropout.forward(char_emb); // [batch, word_len, d_model]

        // ── Masked mean-pool over non-PAD chars ────────────────────────
        let not_pad_char: Tensor<B, 2, Bool> = char_ids.not_equal_elem(0i32);
        let mask_f: Tensor<B, 2> = not_pad_char.float();
        // count: [batch, 1]
        let count: Tensor<B, 2> = mask_f.clone().sum_dim(1).clamp_min(1.0f32);
        // zero PAD contributions: [batch, word_len, 1]
        let mask_3d: Tensor<B, 3> = mask_f.unsqueeze_dim::<3>(2);
        // sum: [batch, 1, d_model] → [batch, d_model]
        let sum: Tensor<B, 3> = (char_emb * mask_3d).sum_dim(1);
        let sum: Tensor<B, 2> = sum.squeeze_dim::<2>(1);
        // char_repr: [batch, d_model]
        let char_repr: Tensor<B, 2> = sum / count;

        // ── Phone embeddings + positions ───────────────────────────────
        let phone_pos: Tensor<B, 2, Int> = Tensor::arange(0..phone_len as i64, &device)
            .unsqueeze_dim::<2>(0)
            .repeat_dim(0, batch);
        let phone_emb = self.phone_embedding.forward(phone_ids.clone())
            + self.phone_pos_embedding.forward(phone_pos);

        // ── Add char context to each phone position ────────────────────
        // [batch, 1, d_model] → [batch, phone_len, d_model]
        let char_ctx: Tensor<B, 3> = char_repr.unsqueeze_dim::<3>(1).repeat_dim(1, phone_len);
        let phone_combined = self.dropout.forward(phone_emb + char_ctx);

        // ── Transformer encoder ─────────────────────────────────────────
        let phone_pad_mask: Tensor<B, 2, Bool> = phone_ids.equal_elem(0i32);
        let input = TransformerEncoderInput::new(phone_combined).mask_pad(phone_pad_mask);
        let phone_out: Tensor<B, 3> = self.transformer.forward(input);

        // ── Per-position classifier ─────────────────────────────────────
        // Linear applies to last dim: [batch, phone_len, d_model] → [batch, phone_len, phone_vocab]
        self.classifier.forward(phone_out)
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

type FileRecorder = BinFileRecorder<FullPrecisionSettings>;

fn make_recorder() -> FileRecorder {
    FileRecorder::new()
}

/// Convert a `Batch` to a pair of 2-D Int tensors on `device`.
pub fn batch_to_tensors<B: Backend>(
    batch: &Batch,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let b = batch.size;
    let w = batch.char_ids[0].len();
    let p = batch.phone_ids[0].len();

    let char_flat: Vec<i32> = batch.char_ids.iter().flatten().copied().collect();
    let phone_flat: Vec<i32> = batch.phone_ids.iter().flatten().copied().collect();
    let target_flat: Vec<i32> = batch.targets.iter().flatten().copied().collect();

    let chars = Tensor::<B, 2, Int>::from_data(
        TensorData::new(char_flat, [b, w]),
        device,
    );
    let phones = Tensor::<B, 2, Int>::from_data(
        TensorData::new(phone_flat, [b, p]),
        device,
    );
    let targets = Tensor::<B, 2, Int>::from_data(
        TensorData::new(target_flat, [b, p]),
        device,
    );
    (chars, phones, targets)
}

/// Compute cross-entropy loss over all masked positions.
///
/// `logits`:  `[batch, seq_len, vocab]`
/// `targets`: `[batch, seq_len]` with PAD_ID=0 at non-masked positions.
///
/// The loss is computed only where `target != 0`, using CrossEntropyLoss
/// with `pad_tokens = [0]`.
pub fn masked_ce_loss<B: Backend>(
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
    model: &mut PhonePredictorModel<B>,
    optimizer: &mut impl Optimizer<PhonePredictorModel<B>, B>,
    lexemes: &[Lexeme],
    vocab: &Vocab,
    config: &TrainConfig,
    epoch: usize,
    max_word_len: usize,
    max_phone_len: usize,
    device: &B::Device,
    rng: &mut R,
) -> f32 {
    // Shuffle lexemes
    let mut indices: Vec<usize> = (0..lexemes.len()).collect();
    indices.shuffle(rng);

    let mut total_loss = 0f32;
    let mut n_batches = 0usize;

    for chunk in indices.chunks(config.batch_size) {
        // Build masked examples for this batch
        let examples: Vec<MaskedExample> = chunk
            .iter()
            .filter_map(|&i| {
                let lex = &lexemes[i];
                let spec = sample_mask_spec(&config.mask_policy, epoch, lex.phones.len(), rng);
                apply_mask(lex, spec, vocab, rng)
            })
            .collect();

        if examples.is_empty() {
            continue;
        }

        let batch = collate_batch(&examples, max_word_len, max_phone_len);
        let (chars, phones, targets) = batch_to_tensors::<B>(&batch, device);

        let logits = model.forward(chars, phones);
        let loss = masked_ce_loss(logits, targets);

        // Backward pass
        let grads = GradientsParams::from_grads(loss.backward(), model);
        *model = optimizer.step(config.learning_rate, model.clone(), grads);

        // Accumulate scalar loss
        let loss_val: f32 = loss.into_scalar().elem();
        total_loss += loss_val;
        n_batches += 1;
    }

    if n_batches == 0 {
        0.0
    } else {
        total_loss / n_batches as f32
    }
}

/// Evaluate the model on a set of lexemes (no gradients).
///
/// Returns `(mean_loss, top1_accuracy, top3_accuracy)`.
pub fn evaluate<B: Backend, R: Rng>(
    model: &PhonePredictorModel<B>,
    lexemes: &[Lexeme],
    vocab: &Vocab,
    max_word_len: usize,
    max_phone_len: usize,
    device: &B::Device,
    rng: &mut R,
) -> (f32, f32, f32) {
    // Use single-mask evaluation for consistent metrics
    let examples: Vec<MaskedExample> = lexemes
        .iter()
        .flat_map(|lex| generate_single_mask_examples(lex, vocab))
        .collect();

    if examples.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut total_loss = 0f32;
    let mut top1_correct = 0usize;
    let mut top3_correct = 0usize;
    let mut total = 0usize;

    for chunk in examples.chunks(64) {
        let batch = collate_batch(chunk, max_word_len, max_phone_len);
        let (chars, phones, targets) = batch_to_tensors::<B>(&batch, device);

        let logits = model.forward(chars, phones.clone()); // [B, P, V]
        let loss = masked_ce_loss(logits.clone(), targets.clone());
        total_loss += loss.into_scalar().elem::<f32>();

        // Compute accuracy for each masked position
        let [b, p, v] = logits.dims();
        for i in 0..b {
            for j in 0..p {
                let tgt: i32 = batch.targets[i][j];
                if tgt == 0 {
                    continue; // non-masked
                }
                // Extract logits for position [i, j, :]
                let pos_logits = logits
                    .clone()
                    .slice([i..i + 1, j..j + 1, 0..v])
                    .reshape([v]);
                let pos_logits_vec: Vec<f32> = pos_logits.into_data().to_vec().unwrap();

                // Top-1
                let top1 = argmax_f32(&pos_logits_vec);
                if top1 as i32 == tgt {
                    top1_correct += 1;
                }
                // Top-3
                let top3 = topk_f32(&pos_logits_vec, 3);
                if top3.contains(&(tgt as usize)) {
                    top3_correct += 1;
                }
                total += 1;
            }
        }
        let _ = rng; // not used here
    }

    let n_batches = (examples.len() as f32 / 64.0).ceil();
    let mean_loss = if n_batches > 0.0 {
        total_loss / n_batches
    } else {
        0.0
    };
    let top1 = if total > 0 {
        top1_correct as f32 / total as f32
    } else {
        0.0
    };
    let top3 = if total > 0 {
        top3_correct as f32 / total as f32
    } else {
        0.0
    };

    (mean_loss, top1, top3)
}

fn argmax_f32(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn topk_f32(v: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = v.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|&(i, _)| i).collect()
}

// ── Full training loop ─────────────────────────────────────────────────────

/// Run the complete training loop with early stopping.
///
/// Saves the best model (by validation loss) to `model_path`.
/// Returns the best validation loss achieved.
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
    <PhonePredictorModel<B> as Module<B>>::Record: Send,
{
    let max_word_len = model_config.max_word_chars;
    let max_phone_len = model_config.max_phones;

    let mut best_val_loss = f32::INFINITY;
    let model_file = model_path.with_extension("bin");
    let mut model: PhonePredictorModel<B> = if model_file.exists() {
        println!("Loading existing model from {} to resume training...", model_file.display());
        let loaded = model_config
            .init(device)
            .load_file(model_path, &make_recorder(), device)
            .with_context(|| format!("loading model to resume from {}", model_path.display()))?;

        // Evaluate on validation set to establish baseline validation loss
        let eval_model: PhonePredictorModel<B::InnerBackend> = loaded.valid();
        let (val_loss, _, _) = evaluate(
            &eval_model,
            valid_lexemes,
            vocab,
            max_word_len,
            max_phone_len,
            device,
            rng,
        );
        println!("Loaded model baseline validation loss: {:.4}", val_loss);
        best_val_loss = val_loss;
        loaded
    } else {
        println!("No existing model found at {}. Initializing new model...", model_file.display());
        model_config.init(device)
    };

    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(train_config.weight_decay)
        .init::<B, PhonePredictorModel<B>>();
    let mut patience_counter = 0usize;

    for epoch in 1..=train_config.epochs {
        let train_loss = train_epoch(
            &mut model,
            &mut optimizer,
            train_lexemes,
            vocab,
            train_config,
            epoch,
            max_word_len,
            max_phone_len,
            device,
            rng,
        );

        // Evaluate on validation set (no gradients needed)
        let eval_model: PhonePredictorModel<B::InnerBackend> = model.valid();
        let (val_loss, val_top1, val_top3) = evaluate(
            &eval_model,
            valid_lexemes,
            vocab,
            max_word_len,
            max_phone_len,
            device,
            rng,
        );

        println!(
            "Epoch {:3} | train_loss={:.4}  val_loss={:.4}  val_top1={:.3}  val_top3={:.3}",
            epoch, train_loss, val_loss, val_top1, val_top3
        );

        // Early stopping
        if val_loss < best_val_loss - 1e-5 {
            best_val_loss = val_loss;
            patience_counter = 0;
            // Save best model
            eval_model
                .save_file(model_path, &make_recorder())
                .with_context(|| format!("saving model to {}", model_path.display()))?;
            println!("  ✓ New best model saved (val_loss={:.4})", best_val_loss);
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

/// Full evaluation metrics struct.
#[derive(Debug)]
pub struct EvalReport {
    pub top1_accuracy: f32,
    pub top3_accuracy: f32,
    pub val_loss: f32,
    pub baseline_overall_top1: f32,
    pub baseline_by_position_top1: f32,
    pub per_phone_accuracy: HashMap<String, (usize, usize)>, // phone -> (correct, total)
}

/// Produce a full `EvalReport` on `test_lexemes`.
pub fn eval_report<B: Backend, R: Rng>(
    model: &PhonePredictorModel<B>,
    test_lexemes: &[Lexeme],
    train_lexemes: &[Lexeme],
    vocab: &Vocab,
    max_word_len: usize,
    max_phone_len: usize,
    device: &B::Device,
    rng: &mut R,
) -> EvalReport {
    // ── Baselines ───────────────────────────────────────────────────────
    let overall_best = most_common_phone(train_lexemes, &vocab.phones);
    let by_position = most_common_phone_by_position(train_lexemes, &vocab.phones);

    let examples: Vec<MaskedExample> = test_lexemes
        .iter()
        .flat_map(|lex| generate_single_mask_examples(lex, vocab))
        .collect();

    let mut top1_correct = 0usize;
    let mut top3_correct = 0usize;
    let mut total = 0usize;
    let mut total_loss = 0f32;
    let mut n_batches = 0usize;
    let mut bl_overall_correct = 0usize;
    let mut bl_pos_correct = 0usize;
    let mut per_phone: HashMap<String, (usize, usize)> = HashMap::new();

    for chunk in examples.chunks(64) {
        let batch = collate_batch(chunk, max_word_len, max_phone_len);
        let (chars, phones, targets) = batch_to_tensors::<B>(&batch, device);

        let logits = model.forward(chars, phones);
        let loss = masked_ce_loss(logits.clone(), targets.clone());
        total_loss += loss.into_scalar().elem::<f32>();
        n_batches += 1;

        let [b, p, v] = logits.dims();
        for i in 0..b {
            for j in 0..p {
                let tgt = batch.targets[i][j] as u32;
                if tgt == 0 {
                    continue;
                }
                let pos_logits = logits
                    .clone()
                    .slice([i..i + 1, j..j + 1, 0..v])
                    .reshape([v]);
                let pos_logits_vec: Vec<f32> = pos_logits.into_data().to_vec().unwrap();

                let top1 = argmax_f32(&pos_logits_vec);
                let top3 = topk_f32(&pos_logits_vec, 3);
                let phone_str = vocab.phones.get_phone(tgt).to_string();

                let entry = per_phone.entry(phone_str).or_insert((0, 0));
                entry.1 += 1;
                if top1 as u32 == tgt {
                    top1_correct += 1;
                    entry.0 += 1;
                }
                if top3.contains(&(tgt as usize)) {
                    top3_correct += 1;
                }

                // Baselines
                if overall_best == tgt {
                    bl_overall_correct += 1;
                }
                let pos_best = *by_position.get(&j).unwrap_or(&overall_best);
                if pos_best == tgt {
                    bl_pos_correct += 1;
                }

                total += 1;
            }
        }
        let _ = rng;
    }

    let mean_loss = if n_batches > 0 {
        total_loss / n_batches as f32
    } else {
        0.0
    };

    EvalReport {
        top1_accuracy: if total > 0 {
            top1_correct as f32 / total as f32
        } else {
            0.0
        },
        top3_accuracy: if total > 0 {
            top3_correct as f32 / total as f32
        } else {
            0.0
        },
        val_loss: mean_loss,
        baseline_overall_top1: if total > 0 {
            bl_overall_correct as f32 / total as f32
        } else {
            0.0
        },
        baseline_by_position_top1: if total > 0 {
            bl_pos_correct as f32 / total as f32
        } else {
            0.0
        },
        per_phone_accuracy: per_phone,
    }
}

// ── Prediction ─────────────────────────────────────────────────────────────

/// Run inference and return ranked phone predictions for each MASK position.
///
/// `phone_tokens` is a list like `["SH", "AA1", "MASK", "L", "MASK", "T"]`.
/// Returns `Vec<(mask_position, Vec<(phone_str, score)>)>`.
pub fn predict<B: Backend>(
    model: &PhonePredictorModel<B>,
    word: &str,
    phone_tokens: &[&str],
    vocab: &Vocab,
    top_k: usize,
    max_word_len: usize,
    max_phone_len: usize,
    device: &B::Device,
) -> Vec<(usize, Vec<(String, f32)>)> {
    use pronlex_core::MASK_ID;

    // Encode chars
    let mut char_ids = vec![PAD_ID as i32; max_word_len];
    for (i, c) in word.chars().enumerate().take(max_word_len) {
        char_ids[i] = vocab.chars.get_id(c) as i32;
    }

    // Encode phones (replace "MASK" with MASK_ID)
    let mut phone_ids = vec![PAD_ID as i32; max_phone_len];
    let mut mask_positions = Vec::new();
    for (j, &tok) in phone_tokens.iter().enumerate().take(max_phone_len) {
        if tok.eq_ignore_ascii_case("MASK") || tok == "<MASK>" {
            phone_ids[j] = MASK_ID as i32;
            mask_positions.push(j);
        } else {
            phone_ids[j] = vocab.phones.get_id(tok) as i32;
        }
    }

    // Tensors (batch = 1)
    let chars = Tensor::<B, 2, Int>::from_data(
        TensorData::new(char_ids, [1, max_word_len]),
        device,
    );
    let phones = Tensor::<B, 2, Int>::from_data(
        TensorData::new(phone_ids, [1, max_phone_len]),
        device,
    );

    let logits = model.forward(chars, phones); // [1, max_phone_len, vocab]
    let [_, _, v] = logits.dims();

    let mut results = Vec::new();
    for pos in mask_positions {
        let pos_logits = logits
            .clone()
            .slice([0..1, pos..pos + 1, 0..v])
            .reshape([v]);
        let scores: Vec<f32> = pos_logits.into_data().to_vec().unwrap();
        let ranked = topk_f32(&scores, top_k);
        let ranked_phones: Vec<(String, f32)> = ranked
            .iter()
            .map(|&id| (vocab.phones.get_phone(id as u32).to_string(), scores[id]))
            .collect();
        results.push((pos, ranked_phones));
    }
    results
}

// ── Load / save helpers ────────────────────────────────────────────────────

/// Save a model (NdArray backend, no autodiff overhead) to `path`.
pub fn save_model<B: Backend>(model: &PhonePredictorModel<B>, path: &Path) -> Result<()> {
    model
        .clone()
        .save_file(path, &make_recorder())
        .with_context(|| format!("saving model to {}", path.display()))
}

/// Load a model from `path` using `model_config` for the initial weights.
pub fn load_model<B: Backend>(
    model_config: &ModelConfig,
    path: &Path,
    device: &B::Device,
) -> Result<PhonePredictorModel<B>> {
    model_config
        .init::<B>(device)
        .load_file(path, &make_recorder(), device)
        .with_context(|| format!("loading model from {}", path.display()))
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::backend::ndarray::NdArrayDevice;
    use pronlex_data::{build_vocab, parse_cmudict};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type TestBackend = NdArray<f32>;
    type TrainTestBackend = Autodiff<TestBackend>;

    fn tiny_vocab_and_lexemes() -> (Vocab, Vec<Lexeme>) {
        let dict = "\
            BUTTER  B AH1 T ER0\n\
            CAT  K AE1 T\n\
            DOG  D AO1 G\n\
            EGG  EH1 G\n\
            FIG  F IH1 G\n";
        let lexemes = parse_cmudict(dict);
        let vocab = build_vocab(&lexemes);
        (vocab, lexemes)
    }

    #[test]
    fn model_forward_shape() {
        let (vocab, _) = tiny_vocab_and_lexemes();
        let device = Default::default();
        let config = ModelConfig::new(vocab.chars.size(), vocab.phones.size())
            .with_d_model(16)
            .with_n_heads(2)
            .with_n_layers(1)
            .with_d_ff(32)
            .with_dropout(0.0)
            .with_max_word_chars(16)
            .with_max_phones(8);
        let model = config.init::<TestBackend>(&device);

        let chars = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![2i32, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 16]),
            &device,
        );
        let phones = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![5i32, 1, 7, 8, 0, 0, 0, 0], [1, 8]),
            &device,
        );
        let logits = model.forward(chars, phones);
        let [b, p, v] = logits.dims();
        assert_eq!(b, 1);
        assert_eq!(p, 8);
        assert_eq!(v, vocab.phones.size());
    }

    #[test]
    fn tiny_training_fixture_no_panic() {
        let (vocab, lexemes) = tiny_vocab_and_lexemes();
        let mut rng = StdRng::seed_from_u64(0);
        let device = NdArrayDevice::default();

        let model_config = ModelConfig::new(vocab.chars.size(), vocab.phones.size())
            .with_d_model(16)
            .with_n_heads(2)
            .with_n_layers(1)
            .with_d_ff(32)
            .with_dropout(0.0)
            .with_max_word_chars(16)
            .with_max_phones(8);

        let train_config = TrainConfig {
            learning_rate: 3e-4,
            weight_decay: 1e-4,
            dropout: 0.0,
            batch_size: 4,
            epochs: 2,
            early_stopping_patience: 5,
            mask_policy: MaskPolicy::Single,
            max_mask_rate: 0.4,
            span_mask_prob: 0.15,
        };

        let mut model = model_config.init::<TrainTestBackend>(&device);
        let mut optimizer = AdamWConfig::new()
            .with_weight_decay(train_config.weight_decay)
            .init::<TrainTestBackend, PhonePredictorModel<TrainTestBackend>>();

        // One training step must not panic
        let loss = train_epoch(
            &mut model,
            &mut optimizer,
            &lexemes,
            &vocab,
            &train_config,
            1,
            16,
            8,
            &device,
            &mut rng,
        );
        assert!(loss.is_finite(), "training loss should be finite");

        // Eval must not panic
        let eval_model: PhonePredictorModel<TestBackend> = model.valid();
        let (val_loss, _top1, _top3) = evaluate(
            &eval_model,
            &lexemes,
            &vocab,
            16,
            8,
            &device,
            &mut rng,
        );
        assert!(val_loss.is_finite(), "validation loss should be finite");
    }

    #[test]
    fn predict_returns_ranked_phones() {
        let (vocab, _) = tiny_vocab_and_lexemes();
        let device = NdArrayDevice::default();

        let model_config = ModelConfig::new(vocab.chars.size(), vocab.phones.size())
            .with_d_model(16)
            .with_n_heads(2)
            .with_n_layers(1)
            .with_d_ff(32)
            .with_dropout(0.0)
            .with_max_word_chars(16)
            .with_max_phones(8);

        let model = model_config.init::<TestBackend>(&device);
        let results = predict(
            &model,
            "butter",
            &["B", "MASK", "T", "ER0"],
            &vocab,
            3,
            16,
            8,
            &device,
        );
        assert_eq!(results.len(), 1, "one MASK position");
        let (pos, ranked) = &results[0];
        assert_eq!(*pos, 1);
        assert_eq!(ranked.len(), 3, "top-3 predictions");
    }
}
