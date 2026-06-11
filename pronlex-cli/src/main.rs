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

pub mod models;
mod piper;
mod speak;

use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn_cuda::{Cuda, CudaDevice};

use pronlex_core::{Vocab, UNK_ID};
use pronlex_data::{parse_cmudict, Lexeme, Task};
use pronlex_model::{
    eval_report, load_model, predict, train, ModelConfig, Seq2SeqModel, TrainConfig,
};
use speech::data::notation::openepd::normalize_openepd_ipa;

// ── Backend aliases ────────────────────────────────────────────────────────

type CpuInferBackend = NdArray<f32>;
type CpuTrainBackend = Autodiff<CpuInferBackend>;

type CudaInferBackend = Cuda<f32, i32>;
type CudaTrainBackend = Autodiff<CudaInferBackend>;

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
enum DeviceArg {
    Cpu,
    Cuda,
}

// ── CLI definition ─────────────────────────────────────────────────────────

/// pronlex – ARPABET masked-phone predictor (v0, CMUdict)
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Use CPU instead of CUDA GPU
    #[arg(long, global = true)]
    cpu: bool,

    #[command(subcommand)]
    command: Option<Commands>,
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

        /// Direction of translation to train: g2p, p2g, or both
        #[arg(long, default_value = "both")]
        task: String,
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

        /// Direction of translation to evaluate: g2p, p2g, both, or auto (detect from train_config)
        #[arg(long, default_value = "auto")]
        task: String,
    },

    /// Fine-tune a model on validation/test discrepancies
    Refine {
        /// Directory containing the trained source model
        #[arg(long)]
        model: PathBuf,

        /// Prepared data directory
        #[arg(long)]
        data: PathBuf,

        /// Output directory for the refined model
        #[arg(long)]
        out: PathBuf,

        /// Comma-separated splits to mine for discrepancies
        #[arg(long, default_value = "valid,test")]
        splits: String,

        /// Refinement source: held-out discrepancies or the built-in sight-word list
        #[arg(long, value_enum, default_value = "discrepancies")]
        source: RefinementSourceArg,

        /// Direction to refine: g2p, p2g, or both
        #[arg(long, default_value = "g2p")]
        task: String,

        /// AdamW learning rate for refinement
        #[arg(long, default_value_t = 1e-4)]
        learning_rate: f64,

        /// AdamW weight decay
        #[arg(long, default_value_t = 1e-4)]
        weight_decay: f32,

        /// Maximum refinement epochs
        #[arg(long, default_value_t = 5)]
        epochs: usize,

        /// Early stopping patience
        #[arg(long, default_value_t = 2)]
        patience: usize,

        /// Mini-batch size
        #[arg(long, default_value_t = 32)]
        batch_size: usize,

        /// Random seed
        #[arg(long, default_value_t = 0)]
        seed: u64,

        /// Print each discrepant word and detailed mining/training context
        #[arg(long)]
        verbose: bool,
    },

    /// Interactive REPL for sequence translation
    Repl {
        /// Direction of translation: g2p, p2g, auto
        #[arg(long, default_value = "auto")]
        task: String,

        /// Directory containing the trained model
        #[arg(long, default_value = "models/cmudict-v0")]
        model: PathBuf,

        /// Optional path to the prepared data directory containing vocab.json
        #[arg(long)]
        data: Option<PathBuf>,
    },

    /// Run translation prediction (Seq2Seq)
    #[command(alias = "infer")]
    Predict {
        /// The input sequence to translate
        input: String,

        /// Direction of translation: g2p, p2g, auto
        #[arg(long, default_value = "auto")]
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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RefinementSourceArg {
    /// Mine held-out split examples where model predictions disagree with OpenEPD.
    Discrepancies,
    /// Fine-tune on the built-in Dolch sight-word list using OpenEPD gold pronunciations.
    SightWords,
}

fn is_cuda_available() -> bool {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(|| {
        let device = CudaDevice::default();
        type B = Cuda<f32, i32>;
        let _tensor = burn::tensor::Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &device);
    });
    std::panic::set_hook(default_hook);
    result.is_ok()
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    let command = cli.command.unwrap_or_else(|| Commands::Repl {
        task: "auto".to_string(),
        model: PathBuf::from("models/cmudict-v0"),
        data: None,
    });

    // Determine target device (CUDA with fallback to CPU, or forced CPU)
    let device_arg = if cli.cpu {
        DeviceArg::Cpu
    } else if is_cuda_available() {
        DeviceArg::Cuda
    } else {
        // Only warn for commands that actually run model computations on the device
        match &command {
            Commands::Train { .. }
            | Commands::Eval { .. }
            | Commands::Refine { .. }
            | Commands::Predict { .. }
            | Commands::Repl { .. } => {
                println!("Warning: CUDA is not available. Falling back to CPU.");
            }
            _ => {}
        }
        DeviceArg::Cpu
    };

    match command {
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
            task,
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
            task,
            device_arg,
        ),
        Commands::Eval {
            model,
            split,
            data,
            task,
        } => cmd_eval(&model, &split, &data, &task, device_arg),
        Commands::Refine {
            model,
            data,
            out,
            splits,
            source,
            task,
            learning_rate,
            weight_decay,
            epochs,
            patience,
            batch_size,
            seed,
            verbose,
        } => cmd_refine(
            &model,
            &data,
            &out,
            &splits,
            source,
            &task,
            learning_rate,
            weight_decay,
            epochs,
            patience,
            batch_size,
            seed,
            verbose,
            device_arg,
        ),
        Commands::Predict {
            model,
            input,
            task,
            data,
        } => cmd_predict(&model, &task, &input, device_arg, data.as_deref()),
        Commands::Repl { model, task, data } => {
            cmd_repl(&model, &task, device_arg, data.as_deref())
        }
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
        let ipa = syllables_to_phonemes_ipa(
            &word_syllables,
            &phonemicized.phonemes,
            &phonemicized.variety,
        );
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
        speech::Spec::Known(speech::FeatureValue::Number(value))
            if value.is_finite() && *value >= 0.0 =>
        {
            Some(*value as usize)
        }
        _ => None,
    }
}

// ── fetch-cmudict ──────────────────────────────────────────────────────────

fn cmd_fetch_cmudict(out: &Path) -> Result<()> {
    const URL: &str = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict";
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
    _seed: u64,
) -> Result<()> {
    println!("Parsing CMUdict from {}", input.display());
    let text = fs::read_to_string(input).context("reading CMUdict")?;
    let base_words = parse_cmudict(&text);
    let total_words = base_words.len();
    println!("  {} base words parsed", total_words);

    fs::create_dir_all(out).context("creating output directory")?;

    // Open output files
    let train_path = out.join("train.jsonl");
    let valid_path = out.join("valid.jsonl");
    let test_path = out.join("test.jsonl");

    let train_file = fs::File::create(&train_path)?;
    let valid_file = fs::File::create(&valid_path)?;
    let test_file = fs::File::create(&test_path)?;

    use indicatif::{ProgressBar, ProgressStyle};
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    let train_writer = Arc::new(Mutex::new(std::io::BufWriter::new(train_file)));
    let valid_writer = Arc::new(Mutex::new(std::io::BufWriter::new(valid_file)));
    let test_writer = Arc::new(Mutex::new(std::io::BufWriter::new(test_file)));

    // Track word lists for anti-leakage auditing
    let train_words = Arc::new(Mutex::new(Vec::new()));
    let valid_words = Arc::new(Mutex::new(Vec::new()));
    let test_words = Arc::new(Mutex::new(Vec::new()));

    // Vocab character/symbol accumulation
    let seen_word_chars = Arc::new(Mutex::new(std::collections::BTreeSet::new()));
    let seen_phoneme_chars = Arc::new(Mutex::new(std::collections::BTreeSet::new()));

    // Shared thread-safe list of base words
    let base_words = Arc::new(Mutex::new(base_words));

    println!("Phonemicizing and writing data splits on-the-fly ...");

    // Setup indicatif progress bar!
    let pb = ProgressBar::new(total_words as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}")?
            .progress_chars("#>-")
    );

    let num_threads = 20;
    let mut handles = Vec::new();

    // Deterministic FNV-1a hash function for thread-safe split assignment
    fn fnv1a_hash(s: &str) -> u64 {
        let mut hash = 0xcbf29ce484222325;
        for byte in s.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    for _ in 0..num_threads {
        let base_words = Arc::clone(&base_words);
        let train_writer = Arc::clone(&train_writer);
        let valid_writer = Arc::clone(&valid_writer);
        let test_writer = Arc::clone(&test_writer);

        let train_words = Arc::clone(&train_words);
        let valid_words = Arc::clone(&valid_words);
        let test_words = Arc::clone(&test_words);

        let seen_word_chars = Arc::clone(&seen_word_chars);
        let seen_phoneme_chars = Arc::clone(&seen_phoneme_chars);

        let pb = pb.clone();

        let handle = std::thread::spawn(move || {
            loop {
                // Pop next base word
                let word = {
                    let mut guard = base_words.lock().unwrap();
                    guard.pop()
                };

                let word = match word {
                    Some(w) => w,
                    None => break,
                };

                if let Some((phonemes, _phones)) = pronlex_data::phonemicize_word(&word) {
                    let lex = Lexeme {
                        base_word: word.clone(),
                        phonemes: phonemes.clone(),
                    };

                    // Add to vocab sets
                    {
                        let mut w_chars = seen_word_chars.lock().unwrap();
                        for c in lex.base_word.chars() {
                            w_chars.insert(c.to_string());
                        }
                        let mut pm_chars = seen_phoneme_chars.lock().unwrap();
                        for c in lex.phonemes.chars() {
                            pm_chars.insert(c.to_string());
                        }
                    }

                    // Split deterministically via FNV-1a hash
                    let hash_val = fnv1a_hash(&lex.base_word);
                    let fraction = (hash_val as f64) / (std::u64::MAX as f64);

                    let line = serde_json::to_string(&lex).unwrap();

                    if fraction < train_frac {
                        let mut w = train_writer.lock().unwrap();
                        let _ = writeln!(w, "{}", line);
                        let mut words = train_words.lock().unwrap();
                        words.push(lex.base_word);
                    } else if fraction < train_frac + valid_frac {
                        let mut w = valid_writer.lock().unwrap();
                        let _ = writeln!(w, "{}", line);
                        let mut words = valid_words.lock().unwrap();
                        words.push(lex.base_word);
                    } else {
                        let mut w = test_writer.lock().unwrap();
                        let _ = writeln!(w, "{}", line);
                        let mut words = test_words.lock().unwrap();
                        words.push(lex.base_word);
                    }
                }

                pb.inc(1);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for h in handles {
        let _ = h.join();
    }

    pb.finish_with_message("Done!");

    // Flush writers
    train_writer.lock().unwrap().flush()?;
    valid_writer.lock().unwrap().flush()?;
    test_writer.lock().unwrap().flush()?;

    let t_words = train_words.lock().unwrap().clone();
    let v_words = valid_words.lock().unwrap().clone();
    let te_words = test_words.lock().unwrap().clone();

    println!(
        "Data splits generated on-the-fly:\n  train={} valid={} test={}",
        t_words.len(),
        v_words.len(),
        te_words.len()
    );

    // Save word lists
    for (name, words) in [
        ("train", &t_words),
        ("valid", &v_words),
        ("test", &te_words),
    ] {
        let path = out.join(format!("{}_words.txt", name));
        let mut deduped = words.clone();
        deduped.sort_unstable();
        deduped.dedup();
        fs::write(&path, deduped.join("\n"))?;
    }

    // Build & save vocabulary
    println!("Building vocabulary from seen characters ...");
    let vocab = {
        let w_list: Vec<String> = seen_word_chars.lock().unwrap().iter().cloned().collect();
        let pm_list: Vec<String> = seen_phoneme_chars.lock().unwrap().iter().cloned().collect();
        Vocab::build(&w_list, &pm_list, &[])
    };

    println!("  Unified vocabulary size: {}", vocab.size());
    let vocab_path = out.join("vocab.json");
    let vocab_json = serde_json::to_string_pretty(&vocab)?;
    fs::write(&vocab_path, &vocab_json).context("writing vocab.json")?;
    println!("  Vocab saved to {}", vocab_path.display());

    println!("Prepare complete.");
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
    task_str: String,
    device_arg: DeviceArg,
) -> Result<()> {
    if !data.join("vocab.json").exists()
        || !data.join("train.jsonl").exists()
        || !data.join("valid.jsonl").exists()
    {
        println!(
            "Data directory or required splits not found at {}. Automatically preparing...",
            data.display()
        );
        let dict_path = Path::new("data/cmudict.dict");
        if !dict_path.exists() {
            println!("CMUdict file not found at data/cmudict.dict. Fetching...");
            cmd_fetch_cmudict(dict_path)?;
        }
        cmd_prepare(dict_path, data, 0.8, 0.1, 42)?;
    }

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

    let model_config = ModelConfig::new(vocab.size()).with_dropout(dropout);

    let task_opt = match task_str.to_lowercase().as_str() {
        "g2p" => Some(Task::G2P),
        "p2g" => Some(Task::P2G),
        "both" => None,
        _ => anyhow::bail!("Invalid task. Supported: g2p, p2g, both"),
    };

    let train_config = TrainConfig {
        learning_rate,
        weight_decay,
        dropout,
        batch_size,
        epochs,
        early_stopping_patience: patience,
        task: task_opt,
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
        fs::copy(&vocab_src, &vocab_dst).context("copying vocab.json to model directory")?;
    }

    let model_path = out.join("model");

    println!("Starting training...");
    println!(
        "  lr={} wd={} dropout={}",
        learning_rate, weight_decay, dropout
    );
    println!(
        "  epochs={} patience={} batch_size={}",
        epochs, patience, batch_size
    );

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

    println!(
        "\nTraining complete. Best validation loss: {:.4}",
        best_loss
    );
    println!("Model saved to {}", model_path.display());
    Ok(())
}

// ── eval ───────────────────────────────────────────────────────────────────

fn cmd_eval(
    model_dir: &Path,
    split: &str,
    data: &Path,
    task_str: &str,
    device_arg: DeviceArg,
) -> Result<()> {
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

    let resolved_task = if task_str.to_lowercase() == "auto" {
        let config_path = model_dir.join("train_config.json");
        if config_path.exists() {
            let s = fs::read_to_string(&config_path).context("reading train_config.json")?;
            let train_config: TrainConfig = serde_json::from_str(&s)?;
            train_config.task
        } else {
            None
        }
    } else {
        match task_str.to_lowercase().as_str() {
            "g2p" => Some(Task::G2P),
            "p2g" => Some(Task::P2G),
            "both" => None,
            _ => anyhow::bail!("Invalid task. Supported: g2p, p2g, both, auto"),
        }
    };

    println!(
        "Evaluating on {} split ({} lexemes) ...",
        split,
        test_lexemes.len()
    );
    if let Some(task) = resolved_task {
        println!("  task: {:?}", task);
    } else {
        println!("  task: both");
    }

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
                resolved_task,
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
                resolved_task,
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
    task_filter: Option<Task>,
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
        task_filter,
        device,
        &mut rng,
    );

    println!("\n── Evaluation Results ──");
    println!("  Loss          : {:.4}", report.val_loss);
    println!("  Exact match   : {:.3}", report.exact_match_accuracy);
    println!("  Token accuracy: {:.3}", report.token_accuracy);

    Ok(())
}

// ── refine ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct DiscrepancyRecord {
    split: String,
    task: String,
    gold_source: String,
    base_word: String,
    input: String,
    gold: String,
    prediction: String,
    gold_compare: String,
    prediction_compare: String,
    edit_distance: usize,
}

const SIGHT_WORDS: &[&str] = &[
    "a",
    "about",
    "after",
    "again",
    "all",
    "always",
    "am",
    "an",
    "and",
    "any",
    "apple",
    "are",
    "around",
    "as",
    "ask",
    "at",
    "ate",
    "away",
    "baby",
    "back",
    "ball",
    "be",
    "bear",
    "because",
    "bed",
    "been",
    "before",
    "bell",
    "best",
    "better",
    "big",
    "bird",
    "birthday",
    "black",
    "blue",
    "boat",
    "both",
    "box",
    "boy",
    "bread",
    "bring",
    "brown",
    "but",
    "buy",
    "by",
    "cake",
    "call",
    "came",
    "can",
    "car",
    "carry",
    "cat",
    "chair",
    "chicken",
    "children",
    "christmas",
    "clean",
    "coat",
    "cold",
    "come",
    "corn",
    "could",
    "cow",
    "cut",
    "day",
    "did",
    "do",
    "does",
    "dog",
    "doll",
    "done",
    "door",
    "down",
    "draw",
    "drink",
    "duck",
    "eat",
    "egg",
    "eight",
    "every",
    "eye",
    "fall",
    "far",
    "farm",
    "farmer",
    "fast",
    "father",
    "feet",
    "find",
    "fire",
    "first",
    "fish",
    "five",
    "floor",
    "flower",
    "fly",
    "for",
    "found",
    "four",
    "from",
    "full",
    "funny",
    "game",
    "garden",
    "gave",
    "get",
    "girl",
    "give",
    "go",
    "goes",
    "going",
    "good",
    "goodbye",
    "got",
    "grass",
    "green",
    "ground",
    "grow",
    "had",
    "hand",
    "has",
    "have",
    "he",
    "head",
    "help",
    "her",
    "here",
    "hill",
    "him",
    "his",
    "hold",
    "home",
    "horse",
    "hot",
    "house",
    "how",
    "hurt",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "jump",
    "just",
    "keep",
    "kind",
    "kitty",
    "know",
    "laugh",
    "leg",
    "let",
    "letter",
    "light",
    "like",
    "little",
    "live",
    "long",
    "look",
    "made",
    "make",
    "man",
    "many",
    "may",
    "me",
    "men",
    "milk",
    "money",
    "morning",
    "mother",
    "much",
    "must",
    "my",
    "myself",
    "name",
    "nest",
    "never",
    "new",
    "night",
    "no",
    "not",
    "now",
    "of",
    "off",
    "old",
    "on",
    "once",
    "one",
    "only",
    "open",
    "or",
    "our",
    "out",
    "over",
    "own",
    "paper",
    "party",
    "picture",
    "pick",
    "pig",
    "play",
    "please",
    "pretty",
    "pull",
    "put",
    "rabbit",
    "rain",
    "ran",
    "read",
    "red",
    "ride",
    "right",
    "ring",
    "robin",
    "round",
    "run",
    "said",
    "santa",
    "saw",
    "say",
    "school",
    "see",
    "seed",
    "seven",
    "shall",
    "she",
    "sheep",
    "shoe",
    "show",
    "sing",
    "sister",
    "sit",
    "six",
    "sleep",
    "small",
    "snow",
    "so",
    "some",
    "song",
    "soon",
    "squirrel",
    "start",
    "stick",
    "stop",
    "street",
    "sun",
    "table",
    "take",
    "tell",
    "ten",
    "thank",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "thing",
    "think",
    "this",
    "those",
    "three",
    "time",
    "to",
    "today",
    "together",
    "too",
    "top",
    "toy",
    "tree",
    "try",
    "two",
    "under",
    "up",
    "upon",
    "us",
    "use",
    "very",
    "walk",
    "warm",
    "was",
    "wash",
    "watch",
    "water",
    "way",
    "we",
    "well",
    "went",
    "were",
    "what",
    "when",
    "where",
    "which",
    "white",
    "who",
    "why",
    "will",
    "wind",
    "window",
    "wish",
    "with",
    "wood",
    "work",
    "would",
    "write",
    "yellow",
    "yes",
    "you",
    "your",
];

#[allow(clippy::too_many_arguments)]
fn cmd_refine(
    model_dir: &Path,
    data: &Path,
    out: &Path,
    splits: &str,
    source: RefinementSourceArg,
    task_str: &str,
    learning_rate: f64,
    weight_decay: f32,
    epochs: usize,
    patience: usize,
    batch_size: usize,
    seed: u64,
    verbose: bool,
    device_arg: DeviceArg,
) -> Result<()> {
    let vocab: Vocab = {
        let s = fs::read_to_string(data.join("vocab.json")).context("reading vocab.json")?;
        serde_json::from_str(&s)?
    };
    let model_config: ModelConfig = {
        let s = fs::read_to_string(model_dir.join("model_config.json"))
            .context("reading model_config.json")?;
        serde_json::from_str(&s)?
    };

    let task_filter = match task_str.to_lowercase().as_str() {
        "g2p" => Some(Task::G2P),
        "p2g" => Some(Task::P2G),
        "both" => None,
        _ => anyhow::bail!("Invalid task. Supported: g2p, p2g, both"),
    };

    let split_names: Vec<String> = splits
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if matches!(source, RefinementSourceArg::Discrepancies) && split_names.is_empty() {
        anyhow::bail!("At least one split is required");
    }

    if out.exists() && model_dir.exists() {
        let out_canon = out
            .canonicalize()
            .context("canonicalizing output directory")?;
        let model_canon = model_dir
            .canonicalize()
            .context("canonicalizing model directory")?;
        if out_canon == model_canon {
            anyhow::bail!(
                "Refinement output must be separate from the source model directory: {}",
                out.display()
            );
        }
    }

    let mut split_lexemes = Vec::new();
    if matches!(source, RefinementSourceArg::Discrepancies) {
        for split in &split_names {
            let path = data.join(format!("{}.jsonl", split));
            let lexemes = read_jsonl(&path)?;
            split_lexemes.push((split.clone(), lexemes));
        }
    }

    fs::create_dir_all(out).context("creating refinement output directory")?;
    if out.join("train_state.json").exists() {
        println!(
            "Existing refinement state found in {}; training will resume there",
            out.display()
        );
    } else {
        fs::copy(
            model_dir.join("model_config.json"),
            out.join("model_config.json"),
        )
        .context("copying model_config.json")?;
        fs::copy(data.join("vocab.json"), out.join("vocab.json")).context("copying vocab.json")?;
        fs::copy(model_dir.join("model.bin"), out.join("model.bin"))
            .context("copying base model")?;
    }

    println!("Mining discrepancies from {}", model_dir.display());
    println!("  gold source: OpenEPD preferred IPA");
    match source {
        RefinementSourceArg::Discrepancies => {
            println!("  source: held-out discrepancies");
            println!("  splits: {}", split_names.join(","));
            for (split, lexemes) in &split_lexemes {
                println!("  {}: {} lexemes", split, lexemes.len());
            }
        }
        RefinementSourceArg::SightWords => {
            println!(
                "  source: built-in Dolch sight words ({} words before OpenEPD/vocab filtering)",
                SIGHT_WORDS.len()
            );
        }
    }
    if let Some(task) = task_filter {
        println!("  task: {:?}", task);
    } else {
        println!("  task: both");
    }
    println!(
        "  output: {}{}",
        out.display(),
        if verbose { " (verbose)" } else { "" }
    );

    let (records, refine_lexemes) = match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            println!("  device: CPU (ndarray)");
            match source {
                RefinementSourceArg::Discrepancies => collect_discrepancies::<CpuInferBackend>(
                    &device,
                    &model_config,
                    model_dir,
                    &vocab,
                    task_filter,
                    &split_lexemes,
                    verbose,
                )?,
                RefinementSourceArg::SightWords => {
                    collect_sight_word_refinement::<CpuInferBackend>(
                        &device,
                        &model_config,
                        model_dir,
                        &vocab,
                        task_filter,
                        verbose,
                    )?
                }
            }
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            println!("  device: CUDA GPU");
            match source {
                RefinementSourceArg::Discrepancies => collect_discrepancies::<CudaInferBackend>(
                    &device,
                    &model_config,
                    model_dir,
                    &vocab,
                    task_filter,
                    &split_lexemes,
                    verbose,
                )?,
                RefinementSourceArg::SightWords => {
                    collect_sight_word_refinement::<CudaInferBackend>(
                        &device,
                        &model_config,
                        model_dir,
                        &vocab,
                        task_filter,
                        verbose,
                    )?
                }
            }
        }
    };

    let discrepancies_path = out.join("discrepancies.jsonl");
    write_discrepancies(&discrepancies_path, &records)?;
    println!(
        "Stored {} discrepancies at {}",
        records.len(),
        discrepancies_path.display()
    );
    print_discrepancy_summary(&records);

    if refine_lexemes.is_empty() {
        println!("No refinement examples found. Refinement skipped.");
        return Ok(());
    }

    let total_edit_distance: usize = records.iter().map(|r| r.edit_distance).sum();
    let mean_edit_distance = if records.is_empty() {
        0.0
    } else {
        total_edit_distance as f32 / records.len() as f32
    };
    println!(
        "Refinement set: {} lexemes, mean edit distance {:.2}",
        refine_lexemes.len(),
        mean_edit_distance
    );
    println!(
        "Refinement training: lr={} wd={} epochs={} patience={} batch_size={}",
        learning_rate, weight_decay, epochs, patience, batch_size
    );

    let train_config = TrainConfig {
        learning_rate,
        weight_decay,
        dropout: model_config.dropout,
        batch_size,
        epochs,
        early_stopping_patience: patience,
        task: task_filter,
    };
    fs::write(
        out.join("train_config.json"),
        serde_json::to_string_pretty(&train_config)?,
    )?;

    let model_path = out.join("model");
    match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            run_train::<CpuTrainBackend>(
                &device,
                &model_config,
                &train_config,
                &refine_lexemes,
                &refine_lexemes,
                &vocab,
                &model_path,
                seed,
            )?;
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            run_train::<CudaTrainBackend>(
                &device,
                &model_config,
                &train_config,
                &refine_lexemes,
                &refine_lexemes,
                &vocab,
                &model_path,
                seed,
            )?;
        }
    }

    Ok(())
}

fn collect_discrepancies<B: Backend>(
    device: &B::Device,
    model_config: &ModelConfig,
    model_dir: &Path,
    vocab: &Vocab,
    task_filter: Option<Task>,
    split_lexemes: &[(String, Vec<Lexeme>)],
    verbose: bool,
) -> Result<(Vec<DiscrepancyRecord>, Vec<Lexeme>)> {
    let model = load_model::<B>(model_config, &model_dir.join("model"), device)?;
    println!("Loading OpenEPD corpus...");
    let openepd = open_english_pronouncing_dictionary::load()
        .map_err(|err| anyhow::anyhow!("loading OpenEPD corpus: {}", err))?;
    println!("  OpenEPD words: {}", openepd.word_count());

    let tasks: Vec<Task> = match task_filter {
        Some(task) => vec![task],
        None => vec![Task::G2P, Task::P2G],
    };

    let total: usize = split_lexemes
        .iter()
        .map(|(_, lexemes)| lexemes.len() * tasks.len())
        .sum();
    let pb = indicatif::ProgressBar::new(total as u64);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}")?
            .progress_chars("#>-"),
    );

    let mut records = Vec::new();
    let mut refine_lexemes = Vec::new();
    let mut refine_seen = std::collections::BTreeSet::new();
    let mut skipped_missing_openepd = 0usize;
    let mut skipped_parse_error = 0usize;
    let mut skipped_unknown_vocab = 0usize;
    for (split, lexemes) in split_lexemes {
        let mut split_checked = 0usize;
        let mut split_discrepancies = 0usize;
        let mut split_skipped_missing_openepd = 0usize;
        let mut split_skipped_parse_error = 0usize;
        let mut split_skipped_unknown_vocab = 0usize;
        for lex in lexemes {
            let base_word = lex.base_word.to_lowercase();
            let Some(raw_openepd_ipa) = openepd.preferred_ipa(&base_word) else {
                skipped_missing_openepd += tasks.len();
                split_skipped_missing_openepd += tasks.len();
                if verbose {
                    pb.println(format!(
                        "SKIP split={} word={} reason=no-openepd-entry",
                        split, base_word
                    ));
                }
                pb.inc(tasks.len() as u64);
                continue;
            };
            let openepd_ipa = match normalize_openepd_ipa(raw_openepd_ipa) {
                Ok(normalized) => normalized,
                Err(err) => {
                    skipped_parse_error += tasks.len();
                    split_skipped_parse_error += tasks.len();
                    if verbose {
                        pb.println(format!(
                            "SKIP split={} word={} reason=openepd-parse-error raw={} error={}",
                            split, base_word, raw_openepd_ipa, err
                        ));
                    }
                    pb.inc(tasks.len() as u64);
                    continue;
                }
            };
            let openepd_lexeme = Lexeme {
                base_word: base_word.clone(),
                phonemes: openepd_ipa.clone(),
            };

            if has_unknown_vocab(vocab, &openepd_ipa) {
                skipped_unknown_vocab += tasks.len();
                split_skipped_unknown_vocab += tasks.len();
                if verbose {
                    pb.println(format!(
                        "SKIP split={} word={} reason=openepd-gold-not-in-vocab gold={}",
                        split, base_word, openepd_ipa
                    ));
                }
                pb.inc(tasks.len() as u64);
                continue;
            }

            for &task in &tasks {
                let (input, gold, task_name) = match task {
                    Task::G2P => (base_word.clone(), openepd_ipa.clone(), "g2p".to_string()),
                    Task::P2G => (openepd_ipa.clone(), base_word.clone(), "p2g".to_string()),
                };
                pb.set_message(format!("{} {}", split, base_word));
                let prediction = predict(&model, &input, task, vocab, device);
                let gold_compare = comparison_key(&gold, task);
                let prediction_compare = comparison_key(&prediction, task);
                let edit_distance = edit_distance_chars(&prediction_compare, &gold_compare);
                split_checked += 1;
                if edit_distance > 0 {
                    split_discrepancies += 1;
                    let record = DiscrepancyRecord {
                        split: split.clone(),
                        task: task_name,
                        gold_source: "openepd".to_string(),
                        base_word: base_word.clone(),
                        input,
                        gold,
                        prediction,
                        gold_compare,
                        prediction_compare,
                        edit_distance,
                    };
                    if verbose {
                        pb.println(format_discrepancy(&record));
                    }
                    records.push(record);
                    if refine_seen.insert(base_word.clone()) {
                        refine_lexemes.push(openepd_lexeme.clone());
                    }
                }
                pb.inc(1);
            }
        }
        pb.println(format!(
            "Completed split {}: checked {} examples, found {} discrepancies, skipped {} missing OpenEPD, skipped {} parse errors, skipped {} unknown-vocab golds",
            split,
            split_checked,
            split_discrepancies,
            split_skipped_missing_openepd,
            split_skipped_parse_error,
            split_skipped_unknown_vocab
        ));
    }
    pb.finish_and_clear();
    if skipped_missing_openepd > 0 || skipped_parse_error > 0 || skipped_unknown_vocab > 0 {
        println!(
            "Skipped during OpenEPD mining: {} missing OpenEPD entries, {} parse errors, {} OpenEPD golds with chars outside vocab",
            skipped_missing_openepd, skipped_parse_error, skipped_unknown_vocab
        );
    }

    Ok((records, refine_lexemes))
}

fn collect_sight_word_refinement<B: Backend>(
    device: &B::Device,
    model_config: &ModelConfig,
    model_dir: &Path,
    vocab: &Vocab,
    task_filter: Option<Task>,
    verbose: bool,
) -> Result<(Vec<DiscrepancyRecord>, Vec<Lexeme>)> {
    let model = load_model::<B>(model_config, &model_dir.join("model"), device)?;
    println!("Loading OpenEPD corpus...");
    let openepd = open_english_pronouncing_dictionary::load()
        .map_err(|err| anyhow::anyhow!("loading OpenEPD corpus: {}", err))?;
    println!("  OpenEPD words: {}", openepd.word_count());

    let tasks: Vec<Task> = match task_filter {
        Some(task) => vec![task],
        None => vec![Task::G2P, Task::P2G],
    };

    let mut sight_words = std::collections::BTreeSet::new();
    for word in SIGHT_WORDS {
        sight_words.insert((*word).to_string());
    }

    let pb = indicatif::ProgressBar::new((sight_words.len() * tasks.len()) as u64);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}")?
            .progress_chars("#>-"),
    );

    let mut records = Vec::new();
    let mut refine_lexemes = Vec::new();
    let mut skipped_missing_openepd = 0usize;
    let mut skipped_parse_error = 0usize;
    let mut skipped_unknown_vocab = 0usize;
    let mut checked = 0usize;

    for base_word in sight_words {
        let Some(raw_openepd_ipa) = openepd.preferred_ipa(&base_word) else {
            skipped_missing_openepd += tasks.len();
            if verbose {
                pb.println(format!(
                    "SKIP split=sight-words word={} reason=no-openepd-entry",
                    base_word
                ));
            }
            pb.inc(tasks.len() as u64);
            continue;
        };
        let openepd_ipa = match normalize_openepd_ipa(raw_openepd_ipa) {
            Ok(normalized) => normalized,
            Err(err) => {
                skipped_parse_error += tasks.len();
                if verbose {
                    pb.println(format!(
                        "SKIP split=sight-words word={} reason=openepd-parse-error raw={} error={}",
                        base_word, raw_openepd_ipa, err
                    ));
                }
                pb.inc(tasks.len() as u64);
                continue;
            }
        };

        if has_unknown_vocab(vocab, &base_word) || has_unknown_vocab(vocab, &openepd_ipa) {
            skipped_unknown_vocab += tasks.len();
            if verbose {
                pb.println(format!(
                    "SKIP split=sight-words word={} reason=gold-not-in-vocab phonemes={}",
                    base_word, openepd_ipa
                ));
            }
            pb.inc(tasks.len() as u64);
            continue;
        }

        refine_lexemes.push(Lexeme {
            base_word: base_word.clone(),
            phonemes: openepd_ipa.clone(),
        });

        for &task in &tasks {
            let (input, gold, task_name) = match task {
                Task::G2P => (base_word.clone(), openepd_ipa.clone(), "g2p".to_string()),
                Task::P2G => (openepd_ipa.clone(), base_word.clone(), "p2g".to_string()),
            };
            pb.set_message(format!("sight-words {}", base_word));
            let prediction = predict(&model, &input, task, vocab, device);
            let gold_compare = comparison_key(&gold, task);
            let prediction_compare = comparison_key(&prediction, task);
            let edit_distance = edit_distance_chars(&prediction_compare, &gold_compare);
            checked += 1;
            if edit_distance > 0 {
                let record = DiscrepancyRecord {
                    split: "sight-words".to_string(),
                    task: task_name,
                    gold_source: "openepd-dolch".to_string(),
                    base_word: base_word.clone(),
                    input,
                    gold,
                    prediction,
                    gold_compare,
                    prediction_compare,
                    edit_distance,
                };
                if verbose {
                    pb.println(format_discrepancy(&record));
                }
                records.push(record);
            }
            pb.inc(1);
        }
    }
    pb.println(format!(
        "Completed sight-word source: checked {} examples, found {} discrepancies, selected {} training lexemes, skipped {} missing OpenEPD, skipped {} parse errors, skipped {} unknown-vocab forms",
        checked,
        records.len(),
        refine_lexemes.len(),
        skipped_missing_openepd,
        skipped_parse_error,
        skipped_unknown_vocab
    ));
    pb.finish_and_clear();

    Ok((records, refine_lexemes))
}

fn format_discrepancy(record: &DiscrepancyRecord) -> String {
    let mut text = format!(
        "EXCEPTION split={} task={} gold_source={} word={} edit_distance={}\n  input: {}\n  gold : {}\n  pred : {}",
        record.split,
        record.task,
        record.gold_source,
        record.base_word,
        record.edit_distance,
        record.input,
        record.gold,
        record.prediction
    );
    if record.gold_compare != record.gold || record.prediction_compare != record.prediction {
        text.push_str(&format!(
            "\n  cmp gold: {}\n  cmp pred: {}",
            record.gold_compare, record.prediction_compare
        ));
    }
    text
}

fn has_unknown_vocab(vocab: &Vocab, text: &str) -> bool {
    vocab.encode_string(text).into_iter().any(|id| id == UNK_ID)
}

fn comparison_key(value: &str, task: Task) -> String {
    match task {
        Task::G2P => pronunciation_comparison_key(value),
        Task::P2G => value.to_lowercase(),
    }
}

fn pronunciation_comparison_key(value: &str) -> String {
    let no_length = value.replace('ː', "");
    let no_syllable_marks = no_length.replace('.', "");
    no_syllable_marks
        .chars()
        .filter(|c| !matches!(c, 'ˈ' | 'ˌ'))
        .collect::<String>()
        .replace('ɝ', "ɚ")
        .replace("iə", "iɚ")
        .replace("uə", "uɚ")
        .replace("əɹ", "ɚ")
        .replace("lɹ", "lɚ")
}

fn print_discrepancy_summary(records: &[DiscrepancyRecord]) {
    if records.is_empty() {
        return;
    }

    let mut by_split_task = std::collections::BTreeMap::<(String, String), usize>::new();
    for record in records {
        *by_split_task
            .entry((record.split.clone(), record.task.clone()))
            .or_default() += 1;
    }

    println!("Discrepancy counts:");
    for ((split, task), count) in by_split_task {
        println!("  {} {}: {}", split, task, count);
    }

    let mut worst = records.to_vec();
    worst.sort_by(|a, b| {
        b.edit_distance
            .cmp(&a.edit_distance)
            .then_with(|| a.base_word.cmp(&b.base_word))
    });

    println!("Largest edit distances:");
    for record in worst.iter().take(10) {
        println!(
            "  {} {} {} edit_distance={} gold={} pred={}",
            record.split,
            record.task,
            record.base_word,
            record.edit_distance,
            record.gold,
            record.prediction
        );
    }
}

fn write_discrepancies(path: &Path, records: &[DiscrepancyRecord]) -> Result<()> {
    use std::io::Write;

    let file = fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    for record in records {
        writeln!(writer, "{}", serde_json::to_string(record)?)?;
    }
    writer.flush()?;
    Ok(())
}

fn edit_distance_chars(left: &str, right: &str) -> usize {
    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();
    let mut prev: Vec<usize> = (0..=right.len()).collect();
    let mut curr = vec![0; right.len() + 1];

    for (i, lc) in left.iter().enumerate() {
        curr[0] = i + 1;
        for (j, rc) in right.iter().enumerate() {
            let substitution = prev[j] + usize::from(lc != rc);
            let insertion = curr[j] + 1;
            let deletion = prev[j + 1] + 1;
            curr[j + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[right.len()]
}

#[cfg(test)]
mod refinement_tests {
    use super::*;

    #[test]
    fn pronunciation_comparison_ignores_length_stress_and_syllable_marks() {
        assert_eq!(
            pronunciation_comparison_key("ˈziː.ə"),
            pronunciation_comparison_key("ˈziə")
        );
        assert_eq!(
            pronunciation_comparison_key("ˈʒuː"),
            pronunciation_comparison_key("ˈʒu")
        );
    }

    #[test]
    fn pronunciation_comparison_collapses_common_r_colored_spellings() {
        assert_eq!(
            pronunciation_comparison_key("ˈziː.ɡɚ"),
            pronunciation_comparison_key("ˈziɡəɹ")
        );
        assert_eq!(
            pronunciation_comparison_key("ˈziː.ɡlɚ"),
            pronunciation_comparison_key("ˈziɡlɹ")
        );
    }
}

// ── predict ────────────────────────────────────────────────────────────────

fn cmd_predict(
    model_dir: &Path,
    task_str: &str,
    input: &str,
    device_arg: DeviceArg,
    data_arg: Option<&Path>,
) -> Result<()> {
    let start_total = std::time::Instant::now();

    println!("Loading vocabulary...");
    let start_vocab = std::time::Instant::now();
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
            let p = model_dir
                .parent()
                .unwrap_or(model_dir)
                .parent()
                .unwrap_or(model_dir)
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
    println!("  ✓ Loaded vocabulary in {:?}", start_vocab.elapsed());

    let task = if task_str.to_lowercase() == "auto" {
        detect_task(input)
    } else {
        Task::from_str(task_str)
            .ok_or_else(|| anyhow::anyhow!("Invalid task. Supported: g2p, p2g, auto"))?
    };

    let model_config: ModelConfig = {
        let s = fs::read_to_string(model_dir.join("model_config.json"))
            .context("reading model_config.json")?;
        serde_json::from_str(&s)?
    };

    match device_arg {
        DeviceArg::Cpu => {
            println!("Initializing CPU device (ndarray)...");
            let start_dev = std::time::Instant::now();
            let device = NdArrayDevice::Cpu;
            println!("  ✓ Initialized CPU device in {:?}", start_dev.elapsed());
            run_predict::<CpuInferBackend>(
                &device,
                &model_config,
                model_dir,
                &vocab,
                task,
                input,
                start_total,
            )?;
        }
        DeviceArg::Cuda => {
            println!("Initializing CUDA GPU device...");
            let start_dev = std::time::Instant::now();
            let device = CudaDevice::default();
            println!(
                "  ✓ Initialized CUDA GPU device in {:?}",
                start_dev.elapsed()
            );
            run_predict::<CudaInferBackend>(
                &device,
                &model_config,
                model_dir,
                &vocab,
                task,
                input,
                start_total,
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
    start_total: std::time::Instant,
) -> Result<()> {
    println!("Loading model config & weights...");
    let start_load = std::time::Instant::now();
    let model = load_model::<B>(model_config, &model_dir.join("model"), device)?;
    println!("  ✓ Loaded model weights in {:?}", start_load.elapsed());

    println!("Translating input='{}' with task={:?}...", input, task);
    let start_pred = std::time::Instant::now();
    let output = predict(&model, input, task, vocab, device);
    println!("  ✓ Finished prediction in {:?}", start_pred.elapsed());

    println!("\nPrediction output:\n  {}", output);
    println!("Total time elapsed: {:?}", start_total.elapsed());

    Ok(())
}

fn cmd_repl(
    model_dir: &Path,
    task_str: &str,
    device_arg: DeviceArg,
    data_arg: Option<&Path>,
) -> Result<()> {
    println!("Loading vocabulary...");
    let start_vocab = std::time::Instant::now();
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
            let p = model_dir
                .parent()
                .unwrap_or(model_dir)
                .parent()
                .unwrap_or(model_dir)
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
    println!("  ✓ Loaded vocabulary in {:?}", start_vocab.elapsed());

    let model_config: ModelConfig = {
        let s = fs::read_to_string(model_dir.join("model_config.json"))
            .context("reading model_config.json")?;
        serde_json::from_str(&s)?
    };

    match device_arg {
        DeviceArg::Cpu => {
            println!("Initializing CPU device (ndarray)...");
            let start_dev = std::time::Instant::now();
            let device = NdArrayDevice::Cpu;
            println!("  ✓ Initialized CPU device in {:?}", start_dev.elapsed());
            run_repl::<CpuInferBackend>(&device, &model_config, model_dir, &vocab, task_str)?;
        }
        DeviceArg::Cuda => {
            println!("Initializing CUDA GPU device...");
            let start_dev = std::time::Instant::now();
            let device = CudaDevice::default();
            println!(
                "  ✓ Initialized CUDA GPU device in {:?}",
                start_dev.elapsed()
            );
            run_repl::<CudaInferBackend>(&device, &model_config, model_dir, &vocab, task_str)?;
        }
    }
    Ok(())
}

fn run_repl<B: Backend>(
    device: &B::Device,
    model_config: &ModelConfig,
    model_dir: &Path,
    vocab: &Vocab,
    initial_task_str: &str,
) -> Result<()> {
    println!("Loading model config & weights...");
    let start_load = std::time::Instant::now();
    let model = load_model::<B>(model_config, &model_dir.join("model"), device)?;
    println!("  ✓ Loaded model weights in {:?}", start_load.elapsed());

    let mut current_task = if initial_task_str.to_lowercase() == "auto" {
        None
    } else {
        Some(
            Task::from_str(initial_task_str)
                .ok_or_else(|| anyhow::anyhow!("Invalid task. Supported: g2p, p2g, auto"))?,
        )
    };

    let mut timings_enabled = true;

    println!("\nREPL ready! Enter input, or type :help for commands.");

    use std::io::{self, Write};
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut line = String::new();

    loop {
        print!("pronlex> ");
        io::stdout().flush().context("flushing stdout")?;

        line.clear();
        let bytes_read = reader.read_line(&mut line).context("reading from stdin")?;
        if bytes_read == 0 {
            // EOF (Ctrl-D)
            println!();
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.starts_with(':') {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            match parts[0] {
                ":quit" | ":q" => {
                    break;
                }
                ":task" => {
                    if parts.len() < 2 {
                        println!("Error: specify task (g2p or p2g)");
                    } else {
                        match parts[1].to_lowercase().as_str() {
                            "g2p" => {
                                current_task = Some(Task::G2P);
                                println!("Task forced to grapheme-to-phoneme (G2P)");
                            }
                            "p2g" => {
                                current_task = Some(Task::P2G);
                                println!("Task forced to phoneme-to-grapheme (P2G)");
                            }
                            _ => {
                                println!("Error: invalid task. Supported: g2p, p2g");
                            }
                        }
                    }
                }
                ":auto" => {
                    current_task = None;
                    println!("Task auto-detect enabled");
                }
                ":timings" => {
                    timings_enabled = !timings_enabled;
                    if timings_enabled {
                        println!("Timing output enabled");
                    } else {
                        println!("Timing output disabled");
                    }
                }
                ":help" => {
                    println!("Commands:");
                    println!("  :quit / :q / Ctrl-D   Exits the REPL");
                    println!("  :task g2p            Forces grapheme-to-phoneme");
                    println!("  :task p2g            Forces phoneme-to-grapheme");
                    println!("  :auto                 Returns to auto-detect task");
                    println!("  :timings              Toggles timing output");
                    println!("  :help                 Prints this help message");
                }
                _ => {
                    println!(
                        "Unknown command: {}. Type :help for list of commands",
                        parts[0]
                    );
                }
            }
            continue;
        }

        let task = match current_task {
            Some(t) => t,
            None => detect_task(trimmed),
        };

        if timings_enabled {
            println!("Translating input='{}' with task={:?}...", trimmed, task);
        }

        let start_pred = std::time::Instant::now();
        let output = predict(&model, trimmed, task, vocab, device);
        let elapsed_pred = start_pred.elapsed();

        if timings_enabled {
            println!("  ✓ Finished prediction in {:?}", elapsed_pred);
            println!("\nPrediction output:\n  {}", output);
        } else {
            println!("{}", output);
        }
        println!();
    }

    Ok(())
}

/// Auto-detect the task based on the input text.
/// If all characters are ASCII alphabetic, apostrophes, or hyphens, we assume G2P.
/// Otherwise, we assume P2G.
pub fn detect_task(input: &str) -> Task {
    let is_spelling = !input.is_empty()
        && input
            .chars()
            .all(|c| c.is_ascii_alphabetic() || c == '\'' || c == '-');
    if is_spelling {
        Task::G2P
    } else {
        Task::P2G
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_task() {
        assert_eq!(detect_task("farkle"), Task::G2P);
        assert_eq!(detect_task("farkle's"), Task::G2P);
        assert_eq!(detect_task("fark-le"), Task::G2P);
        assert_eq!(detect_task("ˈfɑɹ.kəl"), Task::P2G);
        assert_eq!(detect_task("kæt"), Task::P2G); // non-ASCII chars
        assert_eq!(detect_task(""), Task::P2G);
    }
}
