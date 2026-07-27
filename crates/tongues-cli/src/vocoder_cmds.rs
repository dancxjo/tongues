use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn_cuda::{Cuda, CudaDevice};
use clap::{Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tongues_tts::{
    evaluate_vocoder, export_vocoder, load_vocoder_examples, train_vocoder, BurnVocoder,
    HifiganGeneratorConfig, HifiganTrainingRecipe, MelganGeneratorConfig, MelganTrainingRecipe,
    NativeVocoderKind, NativeVocoderRecipe, PqmfConfig, RecipeMelContract, ResolvedSpeechDevice,
    SerializableLossWeights, VocoderAdversarialUpdateSchedule, VocoderPreparedExample,
    VocoderTrainingHyperparams, NativeVocoderTrainingProgress, VocoderTrainingState,
    RECIPE_SCHEMA_VERSION,
};

type Cpu = NdArray<f32>;
type CpuTrain = Autodiff<Cpu>;
type CudaBackend = Cuda<f32, i32>;
type CudaTrain = Autodiff<CudaBackend>;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum VocoderKindArg {
    Hifigan,
    Melgan,
    MultibandMelgan,
    All,
}

#[derive(Subcommand, Debug)]
pub enum VocoderCommands {
    /// Initialize a durable run over cached mel/audio data
    Initialize {
        #[arg(long, value_enum)]
        kind: VocoderKindArg,
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        recipe: PathBuf,
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        source_checkpoint: Option<PathBuf>,
        #[arg(long, default_value = "user-supplied")]
        source_license: String,
        #[arg(long, default_value = "user-supplied native vocoder source")]
        source_provenance: String,
        #[arg(long, default_value = "user-supplied")]
        dataset_license: String,
        #[arg(long, default_value = "prepared speech-corpus mel cache")]
        dataset_provenance: String,
    },
    /// Train or continue a native vocoder run
    Train {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        max_steps: Option<u64>,
    },
    /// Resume from the exact batch cursor and both optimizer records
    Resume {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        max_steps: Option<u64>,
    },
    /// Evaluate spectral, waveform, finite-audio, RTF, and memory metrics
    Evaluate {
        #[arg(long)]
        run: PathBuf,
        #[arg(long, default_value = "valid")]
        split: String,
    },
    /// Export and load through the normal native vocoder adapter
    Export {
        #[arg(long)]
        run: PathBuf,
    },
    /// Run compact CPU overfit/resume/export fixtures
    Fixture {
        #[arg(long, value_enum, default_value = "all")]
        kind: VocoderKindArg,
        #[arg(long, default_value = "target/vocoder-fixture")]
        out: PathBuf,
        #[arg(long, default_value_t = 2)]
        epochs: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VocoderRunManifest {
    kind: NativeVocoderKind,
    train_split: PathBuf,
    validation_split: PathBuf,
    test_split: PathBuf,
    feature_cache: PathBuf,
    source_config: Option<PathBuf>,
    source_checkpoint: Option<PathBuf>,
    source_checkpoint_sha256: Option<String>,
    source_license: String,
    source_provenance: String,
    dataset_license: String,
    dataset_provenance: String,
}

pub fn run(command: VocoderCommands, device: ResolvedSpeechDevice) -> Result<()> {
    match command {
        VocoderCommands::Initialize {
            kind,
            data,
            recipe,
            config,
            out,
            source_checkpoint,
            source_license,
            source_provenance,
            dataset_license,
            dataset_provenance,
        } => initialize(
            kind,
            &data,
            &recipe,
            &config,
            &out,
            source_checkpoint.as_deref(),
            &source_license,
            &source_provenance,
            &dataset_license,
            &dataset_provenance,
        ),
        VocoderCommands::Train { run, max_steps }
        | VocoderCommands::Resume { run, max_steps } => {
            train(&run, max_steps, false, device)
        }
        VocoderCommands::Evaluate { run, split } => evaluate(&run, &split, false, device),
        VocoderCommands::Export { run } => export(&run, false, device),
        VocoderCommands::Fixture { kind, out, epochs } => fixture(kind, &out, epochs),
    }
}

#[allow(clippy::too_many_arguments)]
fn initialize(
    kind: VocoderKindArg,
    data: &Path,
    recipe_path: &Path,
    config: &Path,
    out: &Path,
    source_checkpoint: Option<&Path>,
    source_license: &str,
    source_provenance: &str,
    dataset_license: &str,
    dataset_provenance: &str,
) -> Result<()> {
    ensure!(
        !matches!(kind, VocoderKindArg::All),
        "`all` is only valid for fixture mode"
    );
    let recipe: NativeVocoderRecipe = read_json(recipe_path)?;
    ensure!(recipe.kind() == kind.into_kind()?);
    recipe.validate()?;
    let cache = data.join("vocoder-features");
    ensure!(
        cache.is_dir(),
        "vocoder feature cache is missing: {}",
        cache.display()
    );
    fs::create_dir_all(out)?;
    fs::copy(config, out.join("config.json"))?;
    write_json_atomic(&out.join("recipe.json"), &recipe)?;
    let manifest = VocoderRunManifest {
        kind: recipe.kind(),
        train_split: data.join("train.jsonl"),
        validation_split: data.join("valid.jsonl"),
        test_split: data.join("test.jsonl"),
        feature_cache: cache,
        source_config: source_checkpoint.map(|_| out.join("source-config.json")),
        source_checkpoint: source_checkpoint.map(Path::to_path_buf),
        source_checkpoint_sha256: source_checkpoint.map(sha256_file).transpose()?,
        source_license: source_license.into(),
        source_provenance: source_provenance.into(),
        dataset_license: dataset_license.into(),
        dataset_provenance: dataset_provenance.into(),
    };
    if source_checkpoint.is_some() {
        fs::copy(config, out.join("source-config.json"))?;
    }
    write_json_atomic(&out.join("run-manifest.json"), &manifest)?;
    let mut state = VocoderTrainingState::initial();
    state.generator_learning_rate = recipe.hyperparams().learning_rate;
    state.discriminator_learning_rate = recipe.hyperparams().discriminator_learning_rate;
    write_json_atomic(&out.join("train_state.json"), &state)?;
    fs::create_dir_all(out.join("samples"))?;
    print_paths(out);
    print_compute(false);
    Ok(())
}

fn train(
    run: &Path,
    max_steps: Option<u64>,
    fixture: bool,
    device: ResolvedSpeechDevice,
) -> Result<()> {
    let recipe: NativeVocoderRecipe = read_json(&run.join("recipe.json"))?;
    let manifest: VocoderRunManifest = read_json(&run.join("run-manifest.json"))?;
    let state: VocoderTrainingState = read_json(&run.join("train_state.json"))?;
    let train = load_vocoder_examples(
        &manifest.train_split,
        &manifest.feature_cache,
        recipe.mel_contract().sample_rate_hz,
    )?;
    let valid = load_vocoder_examples(
        &manifest.validation_split,
        &manifest.feature_cache,
        recipe.mel_contract().sample_rate_hz,
    )?;
    print_paths(run);
    print_compute(fixture);
    let report = match device {
        ResolvedSpeechDevice::Cpu => train_vocoder::<CpuTrain>(
            &recipe,
            run,
            state,
            &train,
            &valid,
            manifest.source_config.as_deref(),
            manifest.source_checkpoint.as_deref(),
            &NdArrayDevice::Cpu,
            fixture,
            max_steps,
            print_progress,
        )?,
        ResolvedSpeechDevice::Cuda { index } => train_vocoder::<CudaTrain>(
            &recipe,
            run,
            state,
            &train,
            &valid,
            manifest.source_config.as_deref(),
            manifest.source_checkpoint.as_deref(),
            &CudaDevice::new(index),
            fixture,
            max_steps,
            print_progress,
        )?,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn evaluate(
    run: &Path,
    split: &str,
    fixture: bool,
    device: ResolvedSpeechDevice,
) -> Result<()> {
    let recipe: NativeVocoderRecipe = read_json(&run.join("recipe.json"))?;
    let manifest: VocoderRunManifest = read_json(&run.join("run-manifest.json"))?;
    let path = match split {
        "train" => &manifest.train_split,
        "valid" | "validation" => &manifest.validation_split,
        "test" => &manifest.test_split,
        other => anyhow::bail!("invalid vocoder split `{other}`"),
    };
    let examples = load_vocoder_examples(
        path,
        &manifest.feature_cache,
        recipe.mel_contract().sample_rate_hz,
    )?;
    let report = match device {
        ResolvedSpeechDevice::Cpu => evaluate_vocoder::<Cpu>(
            &recipe,
            run,
            &examples,
            &NdArrayDevice::Cpu,
            fixture,
        )?,
        ResolvedSpeechDevice::Cuda { index } => evaluate_vocoder::<CudaBackend>(
            &recipe,
            run,
            &examples,
            &CudaDevice::new(index),
            fixture,
        )?,
    };
    ensure!(report.loss.is_finite() && report.finite_audio);
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn export(run: &Path, fixture: bool, device: ResolvedSpeechDevice) -> Result<()> {
    let recipe: NativeVocoderRecipe = read_json(&run.join("recipe.json"))?;
    let output = match device {
        ResolvedSpeechDevice::Cpu => {
            let device = NdArrayDevice::Cpu;
            let path = export_vocoder::<Cpu>(&recipe, run, &device, fixture)?;
            BurnVocoder::<Cpu>::load(run.join("config.json"), &path, device)
                .context("loading export through normal native vocoder adapter")?;
            path
        }
        ResolvedSpeechDevice::Cuda { index } => {
            let device = CudaDevice::new(index);
            let path = export_vocoder::<CudaBackend>(&recipe, run, &device, fixture)?;
            BurnVocoder::<CudaBackend>::load(run.join("config.json"), &path, device)
                .context("loading export through normal native vocoder adapter")?;
            path
        }
    };
    println!(
        "Exported and loaded {} checkpoint: {}",
        kind_name(recipe.kind()),
        output.display()
    );
    Ok(())
}

fn fixture(kind: VocoderKindArg, root: &Path, epochs: u64) -> Result<()> {
    let kinds = match kind {
        VocoderKindArg::All => vec![
            NativeVocoderKind::Hifigan,
            NativeVocoderKind::Melgan,
            NativeVocoderKind::MultibandMelgan,
        ],
        other => vec![other.into_kind()?],
    };
    for kind in kinds {
        fixture_one(kind, &root.join(kind_name(kind)), epochs)?;
    }
    Ok(())
}

fn fixture_one(kind: NativeVocoderKind, run: &Path, epochs: u64) -> Result<()> {
    fs::create_dir_all(run)?;
    let recipe = fixture_recipe(kind, epochs);
    let config = fixture_config(kind);
    fs::write(run.join("config.json"), config)?;
    write_json_atomic(&run.join("recipe.json"), &recipe)?;
    let manifest = VocoderRunManifest {
        kind,
        train_split: PathBuf::new(),
        validation_split: PathBuf::new(),
        test_split: PathBuf::new(),
        feature_cache: PathBuf::new(),
        source_config: None,
        source_checkpoint: None,
        source_checkpoint_sha256: None,
        source_license: "random fixture initialization".into(),
        source_provenance: "tongues compact CPU fixture".into(),
        dataset_license: "CC0 fixture".into(),
        dataset_provenance: "deterministic mel/sine fixture".into(),
    };
    write_json_atomic(&run.join("run-manifest.json"), &manifest)?;
    let mut state = VocoderTrainingState::initial();
    state.generator_learning_rate = recipe.hyperparams().learning_rate;
    state.discriminator_learning_rate = recipe.hyperparams().discriminator_learning_rate;
    write_json_atomic(&run.join("train_state.json"), &state)?;
    let examples = fixture_examples(recipe.mel_contract().mel_bins);
    print_paths(run);
    print_compute(true);
    let first = train_vocoder::<CpuTrain>(
        &recipe,
        run,
        state,
        &examples[..2],
        &examples[2..],
        None,
        None,
        &NdArrayDevice::Cpu,
        true,
        Some(1),
        print_progress,
    )?;
    ensure!(first.interrupted);
    let state: VocoderTrainingState = read_json(&run.join("train_state.json"))?;
    ensure!(state.batch_in_epoch == 1);
    let baseline = evaluate_vocoder::<Cpu>(
        &recipe,
        run,
        &examples[2..],
        &NdArrayDevice::Cpu,
        true,
    )?;
    let resumed = train_vocoder::<CpuTrain>(
        &recipe,
        run,
        state,
        &examples[..2],
        &examples[2..],
        None,
        None,
        &NdArrayDevice::Cpu,
        true,
        None,
        print_progress,
    )?;
    let final_report = evaluate_vocoder::<Cpu>(
        &recipe,
        run,
        &examples[2..],
        &NdArrayDevice::Cpu,
        true,
    )?;
    ensure!(
        final_report.finite_audio && final_report.generated_samples > 0,
        "{} fixture produced invalid audio",
        kind_name(kind)
    );
    ensure!(
        final_report.loss < baseline.loss,
        "{} fixture did not overfit: baseline={} final={}",
        kind_name(kind),
        baseline.loss,
        final_report.loss
    );
    export(run, true, ResolvedSpeechDevice::Cpu)?;
    println!(
        "{} fixture complete: baseline={} final={} resumed_step={} rtf={:.3} memory_bytes={}",
        kind_name(kind),
        baseline.loss,
        final_report.loss,
        resumed.global_step,
        final_report.realtime_factor,
        final_report.parameter_memory_bytes
    );
    Ok(())
}

fn fixture_recipe(kind: NativeVocoderKind, epochs: u64) -> NativeVocoderRecipe {
    let mel = RecipeMelContract {
        mel_bins: 4,
        hop_size: 8,
        sample_rate_hz: 8_000,
        win_length: 16,
        fft_size: 16,
        mel_fmin: 0.0,
        mel_fmax: Some(4_000.0),
    };
    let hyper = VocoderTrainingHyperparams {
        learning_rate: 1.0e-3,
        discriminator_learning_rate: 1.0e-3,
        batch_size: 1,
        epochs,
        segment_size: 128,
        adversarial_schedule: VocoderAdversarialUpdateSchedule::EveryBatch,
        checkpoint_interval_steps: 1,
        eval_interval_steps: 1,
        ..Default::default()
    };
    match kind {
        NativeVocoderKind::Hifigan => NativeVocoderRecipe::Hifigan(HifiganTrainingRecipe {
            schema_version: RECIPE_SCHEMA_VERSION,
            mel_contract: mel,
            generator: HifiganGeneratorConfig {
                in_channels: 4,
                out_channels: 1,
                resblock_type: "1".into(),
                resblock_dilation_sizes: vec![vec![1, 2, 3]],
                resblock_kernel_sizes: vec![3],
                upsample_kernel_sizes: vec![4, 4, 4],
                upsample_initial_channel: 8,
                upsample_factors: vec![2, 2, 2],
                inference_padding: 1,
                cond_channels: 0,
                conv_pre_weight_norm: true,
                conv_post_weight_norm: true,
                conv_post_bias: true,
            },
            hyperparams: hyper,
            loss_weights: SerializableLossWeights::default(),
            description: Some("compact CPU fixture".into()),
        }),
        NativeVocoderKind::Melgan => NativeVocoderRecipe::Melgan(MelganTrainingRecipe {
            schema_version: RECIPE_SCHEMA_VERSION,
            mel_contract: mel,
            generator: MelganGeneratorConfig {
                in_channels: 4,
                out_channels: 1,
                projection_kernel_size: 3,
                base_channels: 32,
                upsample_factors: vec![2, 2, 2],
                residual_kernel_size: 3,
                residual_blocks: 2,
                inference_padding: 1,
            },
            pqmf: None,
            hyperparams: hyper,
            loss_weights: SerializableLossWeights::melgan(),
            description: Some("compact CPU fixture".into()),
        }),
        NativeVocoderKind::MultibandMelgan => {
            NativeVocoderRecipe::MultibandMelgan(MelganTrainingRecipe {
                schema_version: RECIPE_SCHEMA_VERSION,
                mel_contract: mel,
                generator: MelganGeneratorConfig {
                    in_channels: 4,
                    out_channels: 4,
                    projection_kernel_size: 3,
                    base_channels: 8,
                    upsample_factors: vec![2],
                    residual_kernel_size: 3,
                    residual_blocks: 1,
                    inference_padding: 1,
                },
                pqmf: Some(PqmfConfig::default()),
                hyperparams: hyper,
                loss_weights: SerializableLossWeights::melgan(),
                description: Some("compact CPU fixture".into()),
            })
        }
    }
}

fn fixture_examples(bins: usize) -> Vec<VocoderPreparedExample> {
    (0..3)
        .map(|index| VocoderPreparedExample {
            record_id: format!("fixture-{index}"),
            mel: (0..16)
                .map(|frame| {
                    (0..bins)
                        .map(|bin| ((frame + 1 + index) * (bin + 1)) as f32 / 64.0)
                        .collect()
                })
                .collect(),
            waveform: (0..128)
                .map(|sample| {
                    (2.0 * std::f32::consts::PI * (sample + index) as f32 / 16.0).sin() * 0.2
                })
                .collect(),
        })
        .collect()
}

fn fixture_config(kind: NativeVocoderKind) -> String {
    let generator_model = match kind {
        NativeVocoderKind::Hifigan => "hifigan_generator",
        NativeVocoderKind::Melgan => "melgan_generator",
        NativeVocoderKind::MultibandMelgan => "multiband_melgan_generator",
    };
    let params = match kind {
        NativeVocoderKind::Hifigan => r#"{
          resblock_type:"1", upsample_factors:[2,2,2],
          upsample_kernel_sizes:[4,4,4], upsample_initial_channel:8,
          resblock_kernel_sizes:[3], resblock_dilation_sizes:[[1,2,3]]
        }"#,
        NativeVocoderKind::Melgan => r#"{
          in_channels:4, out_channels:1, proj_kernel:3, base_channels:32,
          upsample_factors:[2,2,2], res_kernel:3, num_res_blocks:2,
          inference_padding:1
        }"#,
        NativeVocoderKind::MultibandMelgan => r#"{
          in_channels:4, out_channels:4, proj_kernel:3, base_channels:8,
          upsample_factors:[2], res_kernel:3, num_res_blocks:1,
          inference_padding:1
        }"#,
    };
    format!(
        r#"{{
          audio: {{
            fft_size:16, win_length:16, hop_length:8, sample_rate:8000,
            preemphasis:0.0, log_func:"np.log", num_mels:4,
            mel_fmin:0.0, mel_fmax:4000.0, spec_gain:1.0,
            signal_norm:false, min_level_db:-100.0, ref_level_db:20.0,
            symmetric_norm:true, max_norm:4.0, clip_norm:true,
            stats_path:null, do_amp_to_db_mel:true, stft_pad_mode:"reflect",
            centered:true, stft_manual_padding:null
          }},
          generator_model:"{generator_model}",
          generator_model_params:{params},
          use_pqmf:{}
        }}"#,
        matches!(kind, NativeVocoderKind::MultibandMelgan)
    )
}

impl VocoderKindArg {
    fn into_kind(self) -> Result<NativeVocoderKind> {
        match self {
            Self::Hifigan => Ok(NativeVocoderKind::Hifigan),
            Self::Melgan => Ok(NativeVocoderKind::Melgan),
            Self::MultibandMelgan => Ok(NativeVocoderKind::MultibandMelgan),
            Self::All => anyhow::bail!("`all` does not identify one vocoder"),
        }
    }
}

fn kind_name(kind: NativeVocoderKind) -> &'static str {
    match kind {
        NativeVocoderKind::Hifigan => "hifigan",
        NativeVocoderKind::Melgan => "melgan",
        NativeVocoderKind::MultibandMelgan => "multiband-melgan",
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path).with_context(|| format!("reading {}", path.display()))?)
        .with_context(|| format!("parsing {}", path.display()))
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let part = path.with_extension("json.part");
    fs::write(&part, serde_json::to_vec_pretty(value)?)?;
    fs::rename(part, path)?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn print_paths(run: &Path) {
    println!("Native vocoder durable paths:");
    println!("  train state: {}", run.join("train_state.json").display());
    println!("  model state: {}", run.join("trainer-latest.bin").display());
    println!(
        "  generator optimizer: {}",
        run.join("optim-generator-latest.bin").display()
    );
    println!(
        "  discriminator optimizer: {}",
        run.join("optim-discriminator-latest.bin").display()
    );
    println!("  best inference export: {}", run.join("model.safetensors").display());
    println!("  samples: {}", run.join("samples").display());
}

fn print_compute(fixture: bool) {
    if fixture {
        println!("Compute: compact CPU fixture; intended for resume/overfit correctness.");
    } else {
        println!(
            "Compute: CPU is feasible for fixtures/debugging; practical vocoder training and fine-tuning require CUDA."
        );
    }
}

fn print_progress(progress: NativeVocoderTrainingProgress) {
    match progress {
        NativeVocoderTrainingProgress::Resume {
            epoch,
            batch,
            global_step,
        } => println!("Resume epoch={epoch} batch={batch} global_step={global_step}"),
        NativeVocoderTrainingProgress::Epoch { epoch, epochs } => {
            println!("Vocoder epoch {epoch}/{epochs}")
        }
        NativeVocoderTrainingProgress::Batch {
            epoch,
            batch,
            batches,
            global_step,
            phase,
        } if batch <= 3 || batch == batches || batch % 25 == 0 => println!(
            "Vocoder epoch={epoch} batch={batch}/{batches} global_step={global_step} phase={phase:?}"
        ),
        NativeVocoderTrainingProgress::Batch { .. } => {}
        NativeVocoderTrainingProgress::Checkpoint { global_step, path } => {
            println!("Checkpoint global_step={global_step} path={}", path.display())
        }
        NativeVocoderTrainingProgress::Sample { global_step, path } => {
            println!("Sample global_step={global_step} path={}", path.display())
        }
        NativeVocoderTrainingProgress::Complete {
            best_loss_micros,
            path,
        } => println!(
            "Vocoder complete best_loss={} path={}",
            best_loss_micros as f64 / 1_000_000.0,
            path.display()
        ),
    }
}
