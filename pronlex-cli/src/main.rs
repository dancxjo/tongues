//! `pronlex` CLI – masked-phone prediction with CMUdict ARPABET.
//!
//! # Commands
//!
//! ```text
//! pronlex fetch-cmudict --out data/cmudict.dict
//! pronlex prepare --input data/cmudict.dict --out runs/cmudict-v0
//! pronlex train  --data runs/cmudict-v0 --out models/cmudict-v0
//!                [--mask-policy variable] [--max-mask-rate 0.4]
//!                [--span-mask-prob 0.15]
//!                [--learning-rate 3e-4] [--weight-decay 1e-4]
//!                [--dropout 0.1] [--epochs 20] [--patience 5]
//! pronlex eval   --model models/cmudict-v0 --split test
//!                --data runs/cmudict-v0
//! pronlex predict --model models/cmudict-v0
//!                 --word charlotte --phones "SH AA1 R L MASK T"
//! ```

use std::fs;
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use rand::SeedableRng;
use rand::rngs::StdRng;

use burn::backend::{Autodiff, NdArray};
use burn::backend::ndarray::NdArrayDevice;

use pronlex_core::Vocab;
use pronlex_data::{
    Lexeme, MaskPolicy, build_vocab, check_split_leakage, parse_cmudict, split_by_base_word,
};
use pronlex_model::{ModelConfig, TrainConfig, eval_report, load_model, predict, train};

// ── Backend aliases ────────────────────────────────────────────────────────

type InferBackend = NdArray<f32>;
type TrainBackend = Autodiff<InferBackend>;

// ── CLI definition ─────────────────────────────────────────────────────────

/// pronlex – ARPABET masked-phone predictor (v0, CMUdict)
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Download CMUdict from GitHub
    FetchCmudict {
        /// Output path for the downloaded file
        #[arg(long, default_value = "data/cmudict.dict")]
        out: PathBuf,
    },

    /// Parse CMUdict, build vocabulary, and create train/valid/test splits
    Prepare {
        /// Path to CMUdict .dict file (local or downloaded)
        #[arg(long)]
        input: PathBuf,

        /// Output directory for splits and vocabulary
        #[arg(long, default_value = "runs/cmudict-v0")]
        out: PathBuf,

        /// Fraction of base words for training
        #[arg(long, default_value_t = 0.8)]
        train_frac: f64,

        /// Fraction of base words for validation
        #[arg(long, default_value_t = 0.1)]
        valid_frac: f64,

        /// Random seed for reproducible splits
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },

    /// Train the masked-phone predictor
    Train {
        /// Prepared data directory (output of `prepare`)
        #[arg(long)]
        data: PathBuf,

        /// Output directory for the model
        #[arg(long, default_value = "models/cmudict-v0")]
        out: PathBuf,

        /// Masking policy: single (always one mask) or variable (curriculum)
        #[arg(long, value_enum, default_value = "variable")]
        mask_policy: MaskPolicyArg,

        /// Max fraction of phones to mask in variable mode
        #[arg(long, default_value_t = 0.4)]
        max_mask_rate: f64,

        /// Span mask probability weight
        #[arg(long, default_value_t = 0.15)]
        span_mask_prob: f64,

        /// AdamW learning rate
        #[arg(long, default_value_t = 3e-4)]
        learning_rate: f64,

        /// AdamW weight decay
        #[arg(long, default_value_t = 1e-4)]
        weight_decay: f32,

        /// Dropout rate
        #[arg(long, default_value_t = 0.1)]
        dropout: f64,

        /// Maximum training epochs
        #[arg(long, default_value_t = 20)]
        epochs: usize,

        /// Early stopping patience (epochs with no improvement)
        #[arg(long, default_value_t = 5)]
        patience: usize,

        /// Mini-batch size
        #[arg(long, default_value_t = 64)]
        batch_size: usize,

        /// Random seed
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },

    /// Evaluate a trained model
    Eval {
        /// Directory containing the trained model
        #[arg(long)]
        model: PathBuf,

        /// Split to evaluate on: train, valid, or test
        #[arg(long, default_value = "test")]
        split: String,

        /// Prepared data directory
        #[arg(long)]
        data: PathBuf,
    },

    /// Run masked-phone prediction for a single word
    Predict {
        /// Directory containing the trained model
        #[arg(long)]
        model: PathBuf,

        /// Spelling of the word (lowercase)
        #[arg(long)]
        word: String,

        /// Phone sequence with MASK at unknown positions, e.g. "SH AA1 R L MASK T"
        #[arg(long)]
        phones: String,

        /// Number of top predictions to show per mask position
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
}

#[derive(Debug, Clone, ValueEnum)]
enum MaskPolicyArg {
    Single,
    Variable,
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::FetchCmudict { out } => cmd_fetch_cmudict(&out),
        Commands::Prepare {
            input,
            out,
            train_frac,
            valid_frac,
            seed,
        } => cmd_prepare(&input, &out, train_frac, valid_frac, seed),
        Commands::Train {
            data,
            out,
            mask_policy,
            max_mask_rate,
            span_mask_prob,
            learning_rate,
            weight_decay,
            dropout,
            epochs,
            patience,
            batch_size,
            seed,
        } => cmd_train(
            &data,
            &out,
            mask_policy,
            max_mask_rate,
            span_mask_prob,
            learning_rate,
            weight_decay,
            dropout,
            epochs,
            patience,
            batch_size,
            seed,
        ),
        Commands::Eval { model, split, data } => cmd_eval(&model, &split, &data),
        Commands::Predict {
            model,
            word,
            phones,
            top_k,
        } => cmd_predict(&model, &word, &phones, top_k),
    }
}

// ── fetch-cmudict ──────────────────────────────────────────────────────────

fn cmd_fetch_cmudict(out: &Path) -> Result<()> {
    const URL: &str =
        "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict";
    println!("Fetching CMUdict from {}", URL);

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).context("creating output directory")?;
    }

    // Use curl if available (standard on Linux/macOS), fall back to wget
    let status = std::process::Command::new("curl")
        .args(["-fsSL", "-o", out.to_str().unwrap_or("cmudict.dict"), URL])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("Saved to {}", out.display());
            Ok(())
        }
        _ => {
            // Try wget
            let s = std::process::Command::new("wget")
                .args(["-qO", out.to_str().unwrap_or("cmudict.dict"), URL])
                .status()
                .context("neither curl nor wget succeeded")?;
            if s.success() {
                println!("Saved to {}", out.display());
                Ok(())
            } else {
                anyhow::bail!(
                    "Could not download CMUdict. \
                     Please download manually from:\n  {}\nand save to {}",
                    URL,
                    out.display()
                )
            }
        }
    }
}

// ── prepare ────────────────────────────────────────────────────────────────

fn cmd_prepare(
    input: &Path,
    out: &Path,
    train_frac: f64,
    valid_frac: f64,
    seed: u64,
) -> Result<()> {
    println!("Parsing CMUdict from {}", input.display());
    let text = fs::read_to_string(input).context("reading CMUdict")?;
    let lexemes = parse_cmudict(&text);
    println!("  {} entries parsed", lexemes.len());

    println!("Building vocabulary ...");
    let vocab = build_vocab(&lexemes);
    println!(
        "  {} phones, {} chars",
        vocab.phones.size(),
        vocab.chars.size()
    );

    println!("Splitting by base_word (train={} valid={}) ...", train_frac, valid_frac);
    let mut rng = StdRng::seed_from_u64(seed);
    let (train, valid, test) = split_by_base_word(&lexemes, train_frac, valid_frac, &mut rng);

    let leaking = check_split_leakage(&train, &valid, &test);
    if !leaking.is_empty() {
        eprintln!(
            "WARNING: {} base_words appear in more than one split: {:?}",
            leaking.len(),
            &leaking[..leaking.len().min(5)]
        );
    }

    println!(
        "  train={} valid={} test={}",
        train.len(),
        valid.len(),
        test.len()
    );

    fs::create_dir_all(out).context("creating output directory")?;

    // Save vocabulary
    let vocab_path = out.join("vocab.json");
    let vocab_json = serde_json::to_string_pretty(&vocab)?;
    fs::write(&vocab_path, &vocab_json).context("writing vocab.json")?;
    println!("  vocab saved to {}", vocab_path.display());

    // Save splits as JSONL
    for (name, split_data) in [("train", &train), ("valid", &valid), ("test", &test)] {
        let path = out.join(format!("{}.jsonl", name));
        write_jsonl(&path, split_data)?;
        println!("  {} saved ({} entries)", path.display(), split_data.len());
    }

    // Save word lists (for anti-leakage auditing)
    for (name, split_data) in [("train", &train), ("valid", &valid), ("test", &test)] {
        let path = out.join(format!("{}_words.txt", name));
        let words: Vec<&str> = split_data.iter().map(|l| l.base_word.as_str()).collect();
        let mut deduped = words.clone();
        deduped.sort_unstable();
        deduped.dedup();
        fs::write(&path, deduped.join("\n"))?;
    }

    println!("Prepare complete.");
    Ok(())
}

fn write_jsonl(path: &Path, lexemes: &[Lexeme]) -> Result<()> {
    let f = fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(f);
    for lex in lexemes {
        let line = serde_json::to_string(lex)?;
        writeln!(w, "{}", line)?;
    }
    Ok(())
}

fn read_jsonl(path: &Path) -> Result<Vec<Lexeme>> {
    let f = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = std::io::BufReader::new(f);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let lex: Lexeme = serde_json::from_str(&line)
            .with_context(|| format!("parsing JSONL line: {}", &line[..line.len().min(80)]))?;
        out.push(lex);
    }
    Ok(out)
}

// ── train ──────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn cmd_train(
    data: &Path,
    out: &Path,
    mask_policy_arg: MaskPolicyArg,
    max_mask_rate: f64,
    span_mask_prob: f64,
    learning_rate: f64,
    weight_decay: f32,
    dropout: f64,
    epochs: usize,
    patience: usize,
    batch_size: usize,
    seed: u64,
) -> Result<()> {
    let vocab: Vocab = {
        let s = fs::read_to_string(data.join("vocab.json")).context("reading vocab.json")?;
        serde_json::from_str(&s)?
    };

    let train_lexemes = read_jsonl(&data.join("train.jsonl"))?;
    let valid_lexemes = read_jsonl(&data.join("valid.jsonl"))?;

    println!(
        "Loaded {} train / {} valid lexemes",
        train_lexemes.len(),
        valid_lexemes.len()
    );

    let mask_policy = match mask_policy_arg {
        MaskPolicyArg::Single => MaskPolicy::Single,
        MaskPolicyArg::Variable => MaskPolicy::Variable {
            max_mask_rate,
            span_mask_prob,
        },
    };

    let model_config = ModelConfig::new(vocab.chars.size(), vocab.phones.size())
        .with_dropout(dropout);

    let train_config = TrainConfig {
        learning_rate,
        weight_decay,
        dropout,
        batch_size,
        epochs,
        early_stopping_patience: patience,
        mask_policy,
        max_mask_rate,
        span_mask_prob,
    };

    fs::create_dir_all(out).context("creating model directory")?;

    // Save model config and train config for later use by eval/predict
    let model_config_path = out.join("model_config.json");
    fs::write(
        &model_config_path,
        serde_json::to_string_pretty(&model_config)?,
    )?;
    let train_config_path = out.join("train_config.json");
    fs::write(
        &train_config_path,
        serde_json::to_string_pretty(&train_config)?,
    )?;

    let model_path = out.join("model");
    let device: NdArrayDevice = Default::default();
    let mut rng = StdRng::seed_from_u64(seed);

    println!("Starting training...");
    println!("  mask_policy: {:?}", train_config.mask_policy);
    println!("  lr={} wd={} dropout={}", learning_rate, weight_decay, dropout);
    println!("  epochs={} patience={} batch_size={}", epochs, patience, batch_size);

    let best_loss = train::<TrainBackend, _>(
        &model_config,
        &train_config,
        &train_lexemes,
        &valid_lexemes,
        &vocab,
        &model_path,
        &device,
        &mut rng,
    )?;

    println!("\nTraining complete. Best validation loss: {:.4}", best_loss);
    println!("Model saved to {}", model_path.display());
    Ok(())
}

// ── eval ───────────────────────────────────────────────────────────────────

fn cmd_eval(model_dir: &Path, split: &str, data: &Path) -> Result<()> {
    let vocab: Vocab = {
        let s = fs::read_to_string(data.join("vocab.json")).context("reading vocab.json")?;
        serde_json::from_str(&s)?
    };
    let model_config: ModelConfig = {
        let s = fs::read_to_string(model_dir.join("model_config.json"))
            .context("reading model_config.json")?;
        serde_json::from_str(&s)?
    };

    let test_lexemes = read_jsonl(&data.join(format!("{}.jsonl", split)))?;
    let train_lexemes = read_jsonl(&data.join("train.jsonl"))?;

    println!(
        "Evaluating on {} split ({} lexemes) ...",
        split,
        test_lexemes.len()
    );

    let device: NdArrayDevice = Default::default();
    let model = load_model::<InferBackend>(&model_config, &model_dir.join("model"), &device)?;
    let mut rng = StdRng::seed_from_u64(0);

    let report = eval_report(
        &model,
        &test_lexemes,
        &train_lexemes,
        &vocab,
        model_config.max_word_chars,
        model_config.max_phones,
        &device,
        &mut rng,
    );

    println!("\n── Evaluation Results ──");
    println!("  Loss          : {:.4}", report.val_loss);
    println!("  Top-1 accuracy: {:.3}", report.top1_accuracy);
    println!("  Top-3 accuracy: {:.3}", report.top3_accuracy);
    println!(
        "  Baseline (overall most common phone)    : {:.3}",
        report.baseline_overall_top1
    );
    println!(
        "  Baseline (most common by position index): {:.3}",
        report.baseline_by_position_top1
    );

    println!("\n── Per-phone accuracy (top-20 by total count) ──");
    let mut per_phone_vec: Vec<(&String, &(usize, usize))> =
        report.per_phone_accuracy.iter().collect();
    per_phone_vec.sort_by_key(|(_, &(_, total))| std::cmp::Reverse(total));
    for (phone, &(correct, total)) in per_phone_vec.iter().take(20) {
        println!(
            "  {:6}  {:.3}  ({}/{})",
            phone,
            correct as f32 / total as f32,
            correct,
            total
        );
    }

    Ok(())
}

// ── predict ────────────────────────────────────────────────────────────────

fn cmd_predict(model_dir: &Path, word: &str, phones_str: &str, top_k: usize) -> Result<()> {
    // Load vocab from the model directory (co-located during train)
    let vocab_path = model_dir.parent().unwrap_or(model_dir).join("vocab.json");
    // Also check if vocab is next to the data (common layout)
    let vocab: Vocab = {
        let search_paths = [
            model_dir.join("vocab.json"),
            vocab_path.clone(),
        ];
        let mut found = None;
        for p in &search_paths {
            if p.exists() {
                found = Some(p.clone());
                break;
            }
        }
        // Fall back: look in sibling data dir
        let path = found.context(
            "vocab.json not found next to the model. \
             Pass --data to specify the prepared data directory, or co-locate vocab.json.",
        )?;
        let s = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&s)?
    };

    let model_config: ModelConfig = {
        let s = fs::read_to_string(model_dir.join("model_config.json"))
            .context("reading model_config.json")?;
        serde_json::from_str(&s)?
    };

    let device: NdArrayDevice = Default::default();
    let model = load_model::<InferBackend>(&model_config, &model_dir.join("model"), &device)?;

    let phone_tokens: Vec<&str> = phones_str.split_ascii_whitespace().collect();

    println!("Predicting for word='{}' phones='{}'", word, phones_str);

    let results = predict(
        &model,
        &word.to_lowercase(),
        &phone_tokens,
        &vocab,
        top_k,
        model_config.max_word_chars,
        model_config.max_phones,
        &device,
    );

    if results.is_empty() {
        println!("No MASK positions found in the phone sequence.");
        return Ok(());
    }

    for (pos, ranked) in &results {
        println!("\nMASK at position {} (0-indexed):", pos);
        for (i, (phone, score)) in ranked.iter().enumerate() {
            println!("  {}. {:6}  logit={:.3}", i + 1, phone, score);
        }
    }

    Ok(())
}
