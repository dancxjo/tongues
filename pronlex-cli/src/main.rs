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

mod speak;
mod piper;
pub mod models;

use std::fs;
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use rand::SeedableRng;
use rand::rngs::StdRng;

use burn::backend::{Autodiff, NdArray};
use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn_cuda::{Cuda, CudaDevice};

use pronlex_core::Vocab;
use pronlex_data::{
    Lexeme, build_vocab, check_split_leakage, parse_cmudict, split_by_base_word,
    phonemicize_lexemes, Task,
};
use pronlex_model::{ModelConfig, TrainConfig, eval_report, load_model, predict, train, Seq2SeqModel};

// ── Backend aliases ────────────────────────────────────────────────────────

type CpuInferBackend = NdArray<f32>;
type CpuTrainBackend = Autodiff<CpuInferBackend>;

type CudaInferBackend = Cuda<f32, i32>;
type CudaTrainBackend = Autodiff<CudaInferBackend>;

#[derive(ValueEnum, Clone, Debug, Copy, PartialEq, Eq)]
enum DeviceArg {
    Cpu,
    Cuda,
}

// ── CLI definition ─────────────────────────────────────────────────────────

/// pronlex – ARPABET masked-phone predictor (v0, CMUdict)
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Device to use (cuda or cpu)
    #[arg(long, global = true, default_value = "cuda")]
    device: DeviceArg,

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

    /// Run translation prediction (Seq2Seq)
    #[command(alias = "infer")]
    Predict {
        /// The input sequence to translate
        input: String,

        /// Direction of translation: s2pm, s2ph, pm2s, ph2s, pm2ph, ph2pm
        #[arg(long, default_value = "s2pm")]
        task: String,

        /// Directory containing the trained model
        #[arg(long, default_value = "models/cmudict-v0")]
        model: PathBuf,

        /// Optional path to the prepared data directory containing vocab.json
        #[arg(long)]
        data: Option<PathBuf>,
    },

    /// Speak/synthesize text into a WAV file using speech plans
    Speak(speak::SpeakCommand),

    /// Phonemize text into a broad IPA phoneme sequence
    Phonemes {
        /// The text to phonemize
        text: String,
    },

    /// Phonemize text into a narrow IPA phone sequence
    Phones {
        /// The text to phonemize
        text: String,
    },

    /// Manage local models
    Models {
        #[command(subcommand)]
        command: Option<models::ModelsCommand>,
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
            cli.device,
        ),
        Commands::Eval { model, split, data } => cmd_eval(&model, &split, &data, cli.device),
        Commands::Predict {
            model,
            input,
            task,
            data,
        } => cmd_predict(&model, &task, &input, cli.device, data.as_deref()),
        Commands::Speak(command) => speak::run_speak(command),
        Commands::Phonemes { text } => cmd_phonemes(&text),
        Commands::Phones { text } => cmd_phones(&text),
        Commands::Models { command } => models::run(command),
    }
}

fn cmd_phonemes(text: &str) -> Result<()> {
    use speech::{EnglishPhonemicizer, PhonemicizeRequest, Phonemicizer, VarietyId};

    let phonemicizer = EnglishPhonemicizer;
    let phonemicized = phonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: text.to_string(),
            variety: VarietyId("en-US".to_string()),
            style: None,
        })
        .map_err(|e| anyhow::anyhow!("Failed to phonemicize: {:?}", e))?;

    let mut words: Vec<(usize, Vec<speech::Syllable>)> = Vec::new();
    for syllable in phonemicized.syllables.iter() {
        if let Some(first_phone) = syllable.phones.first() {
            if let Some(word_idx) = token_word_index(&first_phone.features) {
                if let Some(last_word) = words.last_mut() {
                    if last_word.0 == word_idx {
                        last_word.1.push(syllable.clone());
                        continue;
                    }
                }
                words.push((word_idx, vec![syllable.clone()]));
            }
        }
    }

    let mut ipa_words = Vec::new();
    for (_, word_syllables) in words {
        let ipa = syllables_to_phonemes_ipa(&word_syllables, &phonemicized.phonemes, &phonemicized.variety);
        if !ipa.is_empty() {
            ipa_words.push(ipa);
        }
    }

    println!("/{}/", ipa_words.join(" "));
    Ok(())
}

fn cmd_phones(text: &str) -> Result<()> {
    use speech::{EnglishPhonemicizer, PhonemicizeRequest, Phonemicizer, VarietyId};

    let phonemicizer = EnglishPhonemicizer;
    let phonemicized = phonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: text.to_string(),
            variety: VarietyId("en-US".to_string()),
            style: None,
        })
        .map_err(|e| anyhow::anyhow!("Failed to phonemicize: {:?}", e))?;

    let mut words: Vec<(usize, Vec<speech::Syllable>)> = Vec::new();
    for syllable in phonemicized.syllables.iter() {
        if let Some(first_phone) = syllable.phones.first() {
            if let Some(word_idx) = token_word_index(&first_phone.features) {
                if let Some(last_word) = words.last_mut() {
                    if last_word.0 == word_idx {
                        last_word.1.push(syllable.clone());
                        continue;
                    }
                }
                words.push((word_idx, vec![syllable.clone()]));
            }
        }
    }

    let mut ipa_words = Vec::new();
    for (_, word_syllables) in words {
        let ipa = syllables_to_ipa_formatted(&word_syllables);
        if !ipa.is_empty() {
            ipa_words.push(ipa);
        }
    }

    println!("[{}]", ipa_words.join(" "));
    Ok(())
}

fn find_phoneme_for_phone(
    phone: &speech::PhoneToken,
    phonemes: &[speech::PhonemeToken],
) -> Option<speech::PhonemeId> {
    for phoneme_token in phonemes {
        for realized_phone in &phoneme_token.realized_as {
            if realized_phone.phone == phone.phone
                && realized_phone.features == phone.features
                && realized_phone.span == phone.span
            {
                if let speech::Spec::Known(ref id) = phoneme_token.phoneme {
                    return Some(id.clone());
                }
            }
        }
    }
    None
}

fn phone_ipa(phone: &speech::PhoneToken) -> &str {
    match &phone.phone {
        speech::Spec::Known(id) => id
            .as_str()
            .strip_prefix("ipa.phone.")
            .unwrap_or(id.as_str()),
        _ => "",
    }
}

fn syllables_to_phonemes_ipa(
    syllables: &[speech::Syllable],
    phonemes: &[speech::PhonemeToken],
    variety: &speech::VarietyId,
) -> String {
    syllables
        .iter()
        .map(|syllable| {
            let mut text = String::new();
            match syllable.stress {
                speech::Spec::Known(speech::Stress::Primary) => text.push('ˈ'),
                speech::Spec::Known(speech::Stress::Secondary) => text.push('ˌ'),
                _ => {}
            }
            for phone in &syllable.phones {
                if let Some(phoneme_id) = find_phoneme_for_phone(phone, phonemes) {
                    let symbol = speech::phoneme_default_phone_display_symbol(&phoneme_id, variety);
                    text.push_str(&symbol);
                } else {
                    text.push_str(phone_ipa(phone));
                }
            }
            text
        })
        .collect()
}

fn syllables_to_ipa_formatted(syllables: &[speech::Syllable]) -> String {
    syllables
        .iter()
        .enumerate()
        .map(|(index, syllable)| {
            let mut text = String::new();
            let mut has_stress_mark = false;
            let stress_char = match syllable.stress {
                speech::Spec::Known(speech::Stress::Primary) => {
                    has_stress_mark = true;
                    Some('ˈ')
                }
                speech::Spec::Known(speech::Stress::Secondary) => {
                    has_stress_mark = true;
                    Some('ˌ')
                }
                _ => None,
            };

            if index > 0 && !has_stress_mark {
                text.push('.');
            }
            if let Some(c) = stress_char {
                text.push(c);
            }
            for phone in &syllable.phones {
                text.push_str(phone_ipa(phone));
            }
            text
        })
        .collect()
}

fn token_word_index(features: &speech::FeatureBundle) -> Option<usize> {
    let value = features
        .values
        .get(&speech::FeatureId("orthography.word_index".into()))?;
    match value {
        speech::Spec::Known(speech::FeatureValue::Number(value)) if value.is_finite() && *value >= 0.0 => {
            Some(*value as usize)
        }
        _ => None,
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
    let base_words = parse_cmudict(&text);
    println!("  {} base words parsed", base_words.len());

    println!("Phonemicizing base words in parallel ...");
    let lexemes = phonemicize_lexemes(base_words);
    println!("  {} valid lexemes created", lexemes.len());

    println!("Building vocabulary ...");
    let vocab = build_vocab(&lexemes);
    println!(
        "  Unified vocabulary size: {}",
        vocab.size()
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
    _mask_policy_arg: MaskPolicyArg,
    _max_mask_rate: f64,
    _span_mask_prob: f64,
    learning_rate: f64,
    weight_decay: f32,
    dropout: f64,
    epochs: usize,
    patience: usize,
    batch_size: usize,
    seed: u64,
    device_arg: DeviceArg,
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

    let model_config = ModelConfig::new(vocab.size())
        .with_dropout(dropout);

    let train_config = TrainConfig {
        learning_rate,
        weight_decay,
        dropout,
        batch_size,
        epochs,
        early_stopping_patience: patience,
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

    // Copy vocab.json to model output directory to make it self-contained
    let vocab_src = data.join("vocab.json");
    let vocab_dst = out.join("vocab.json");
    if vocab_src.exists() {
        fs::copy(&vocab_src, &vocab_dst)
            .context("copying vocab.json to model directory")?;
    }

    let model_path = out.join("model");

    println!("Starting training...");
    println!("  lr={} wd={} dropout={}", learning_rate, weight_decay, dropout);
    println!("  epochs={} patience={} batch_size={}", epochs, patience, batch_size);

    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            println!("  device: CPU (ndarray)");
            run_train::<CpuTrainBackend>(
                &device,
                &model_config,
                &train_config,
                &train_lexemes,
                &valid_lexemes,
                &vocab,
                &model_path,
                seed,
            )?;
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            println!("  device: CUDA GPU");
            run_train::<CudaTrainBackend>(
                &device,
                &model_config,
                &train_config,
                &train_lexemes,
                &valid_lexemes,
                &vocab,
                &model_path,
                seed,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_train<B: AutodiffBackend>(
    device: &B::Device,
    model_config: &ModelConfig,
    train_config: &TrainConfig,
    train_lexemes: &[Lexeme],
    valid_lexemes: &[Lexeme],
    vocab: &Vocab,
    model_path: &Path,
    seed: u64,
) -> Result<()>
where
    <Seq2SeqModel<B> as burn::module::Module<B>>::Record: Send,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let best_loss = train::<B, _>(
        model_config,
        train_config,
        train_lexemes,
        valid_lexemes,
        vocab,
        model_path,
        device,
        &mut rng,
    )?;

    println!("\nTraining complete. Best validation loss: {:.4}", best_loss);
    println!("Model saved to {}", model_path.display());
    Ok(())
}

// ── eval ───────────────────────────────────────────────────────────────────

fn cmd_eval(model_dir: &Path, split: &str, data: &Path, device_arg: DeviceArg) -> Result<()> {
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

    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            println!("  device: CPU (ndarray)");
            run_eval::<CpuInferBackend>(
                &device,
                &model_config,
                model_dir,
                split,
                &vocab,
                &test_lexemes,
                &train_lexemes,
            )?;
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            println!("  device: CUDA GPU");
            run_eval::<CudaInferBackend>(
                &device,
                &model_config,
                model_dir,
                split,
                &vocab,
                &test_lexemes,
                &train_lexemes,
            )?;
        }
    }
    Ok(())
}

fn run_eval<B: Backend>(
    device: &B::Device,
    model_config: &ModelConfig,
    model_dir: &Path,
    _split: &str,
    vocab: &Vocab,
    test_lexemes: &[Lexeme],
    train_lexemes: &[Lexeme],
) -> Result<()> {
    let model = load_model::<B>(model_config, &model_dir.join("model"), device)?;
    let mut rng = StdRng::seed_from_u64(0);

    let report = eval_report(
        &model,
        test_lexemes,
        train_lexemes,
        vocab,
        device,
        &mut rng,
    );

    println!("\n── Evaluation Results ──");
    println!("  Loss          : {:.4}", report.val_loss);
    println!("  Exact match   : {:.3}", report.exact_match_accuracy);

    Ok(())
}

// ── predict ────────────────────────────────────────────────────────────────

fn cmd_predict(
    model_dir: &Path,
    task_str: &str,
    input: &str,
    device_arg: DeviceArg,
    data_arg: Option<&Path>,
) -> Result<()> {
    // Load vocab
    let vocab: Vocab = {
        let mut found = None;

        // 1. Check if data_arg was passed
        if let Some(data_path) = data_arg {
            let p = data_path.join("vocab.json");
            if p.exists() {
                found = Some(p);
            }
        }

        // 2. Check next to the model file
        if found.is_none() {
            let p = model_dir.join("vocab.json");
            if p.exists() {
                found = Some(p);
            }
        }

        // 3. Check model parent dir
        if found.is_none() {
            let p = model_dir.parent().unwrap_or(model_dir).join("vocab.json");
            if p.exists() {
                found = Some(p);
            }
        }

        // 4. Try sibling folder (substituting "models" for "runs" or next to model_dir)
        if found.is_none() {
            let p = model_dir.parent().unwrap_or(model_dir).parent().unwrap_or(model_dir)
                .join("runs")
                .join(model_dir.file_name().unwrap_or_default())
                .join("vocab.json");
            if p.exists() {
                found = Some(p);
            }
        }

        let path = found.context(
            "vocab.json not found. Pass --data to specify the prepared data directory containing vocab.json, or copy vocab.json to the model directory.",
        )?;
        let s = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&s)?
    };

    let task = Task::from_str(task_str)
        .ok_or_else(|| anyhow::anyhow!("Invalid task. Supported: s2pm, s2ph, pm2s, ph2s, pm2ph, ph2pm"))?;

    let model_config: ModelConfig = {
        let s = fs::read_to_string(model_dir.join("model_config.json"))
            .context("reading model_config.json")?;
        serde_json::from_str(&s)?
    };

    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            println!("  device: CPU (ndarray)");
            run_predict::<CpuInferBackend>(
                &device,
                &model_config,
                model_dir,
                &vocab,
                task,
                input,
            )?;
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            println!("  device: CUDA GPU");
            run_predict::<CudaInferBackend>(
                &device,
                &model_config,
                model_dir,
                &vocab,
                task,
                input,
            )?;
        }
    }
    Ok(())
}

fn run_predict<B: Backend>(
    device: &B::Device,
    model_config: &ModelConfig,
    model_dir: &Path,
    vocab: &Vocab,
    task: Task,
    input: &str,
) -> Result<()> {
    let model = load_model::<B>(model_config, &model_dir.join("model"), device)?;

    println!("Translating input='{}' with task={:?}", input, task);

    let output = predict(
        &model,
        input,
        task,
        vocab,
        device,
    );

    println!("\nPrediction output:\n  {}", output);

    Ok(())
}
