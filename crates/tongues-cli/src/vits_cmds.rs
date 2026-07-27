use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn_cuda::{Cuda, CudaDevice};
use clap::Subcommand;
use sha2::{Digest, Sha256};
use tongues_tts::{
    evaluate_vits, export_vits, initialize_vits_run_with_progress, load_vits_examples,
    load_vits_training_model_config, train_vits, write_vits_training_manifest, BurnVitsSpeech,
    ResolvedSpeechDevice, VitsCheckpointPolicy, VitsDatasetManifest, VitsRunLayout,
    VitsTrainOptions, VitsTrainingBackend, VitsTrainingManifest, VitsTrainingProgress,
    VitsTrainingRecipe, VITS_TRAINING_MANIFEST_SCHEMA_VERSION,
};

type Cpu = NdArray<f32>;
type CpuTrain = Autodiff<Cpu>;
type CudaBackend = Cuda<f32, i32>;
type CudaTrain = Autodiff<CudaBackend>;

#[derive(Subcommand, Debug)]
pub enum VitsCommands {
    /// Create a durable run from a prepared speech corpus and model config
    Initialize {
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        source_checkpoint: Option<PathBuf>,
        #[arg(long)]
        baseline_metric: Option<f64>,
        #[arg(long, default_value = "user-supplied")]
        source_license: String,
        #[arg(long, default_value = "user-supplied native VITS initialization")]
        source_provenance: String,
        #[arg(long, default_value = "user-supplied")]
        dataset_license: String,
        #[arg(long, default_value = "prepared speech-corpus manifest")]
        dataset_provenance: String,
        #[arg(long, default_value_t = 1000)]
        epochs: u64,
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
        #[arg(long, default_value_t = 32)]
        segment_frames: usize,
        #[arg(long, default_value_t = 1000)]
        checkpoint_every: u64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Train a new or initialized run
    Train {
        #[arg(long)]
        run: PathBuf,
        /// Stop after N newly completed batches, leaving an exact resumable cursor
        #[arg(long)]
        max_steps: Option<u64>,
    },
    /// Resume from the exact epoch/batch cursor and both optimizer records
    Resume {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        max_steps: Option<u64>,
    },
    /// Evaluate the durable latest training checkpoint
    Evaluate {
        #[arg(long)]
        run: PathBuf,
        #[arg(long, default_value = "valid")]
        split: String,
    },
    /// Export the latest generator and verify it with BurnVitsSpeech
    Export {
        #[arg(long)]
        run: PathBuf,
    },
    /// Train and resume a compact deterministic CPU fixture end to end
    Fixture {
        #[arg(long, default_value = "target/vits-fixture")]
        out: PathBuf,
        #[arg(long, default_value_t = 4)]
        epochs: u64,
    },
}

pub fn run(command: VitsCommands, device: ResolvedSpeechDevice) -> Result<()> {
    match command {
        VitsCommands::Initialize {
            data,
            config,
            out,
            source_checkpoint,
            baseline_metric,
            source_license,
            source_provenance,
            dataset_license,
            dataset_provenance,
            epochs,
            batch_size,
            segment_frames,
            checkpoint_every,
            seed,
        } => initialize(
            &data,
            &config,
            &out,
            source_checkpoint.as_deref(),
            baseline_metric,
            &source_license,
            &source_provenance,
            &dataset_license,
            &dataset_provenance,
            epochs,
            batch_size,
            segment_frames,
            checkpoint_every,
            seed,
            device,
        ),
        VitsCommands::Train { run, max_steps } | VitsCommands::Resume { run, max_steps } => {
            run_training(&run, max_steps, false, device)
        }
        VitsCommands::Evaluate { run, split } => evaluate(&run, &split, device),
        VitsCommands::Export { run } => export(&run, device),
        VitsCommands::Fixture { out, epochs } => fixture(&out, epochs),
    }
}

#[allow(clippy::too_many_arguments)]
fn initialize(
    data: &Path,
    config: &Path,
    out: &Path,
    source_checkpoint: Option<&Path>,
    baseline_metric: Option<f64>,
    source_license: &str,
    source_provenance: &str,
    dataset_license: &str,
    dataset_provenance: &str,
    epochs: u64,
    batch_size: usize,
    segment_frames: usize,
    checkpoint_every: u64,
    seed: u64,
    device: ResolvedSpeechDevice,
) -> Result<()> {
    if source_checkpoint.is_some() {
        ensure!(
            baseline_metric.is_some(),
            "--baseline-metric is required for published-checkpoint fine-tuning"
        );
    }
    let model = load_vits_training_model_config(config)?;
    let feature_cache = data.join("vits-features");
    ensure!(
        feature_cache.is_dir(),
        "VITS feature cache is missing: {}. Populate CachedSpeechFeatures with checkpoint-local token IDs and channel-major-compatible linear spectrogram frames before initializing.",
        feature_cache.display()
    );
    let train = count_jsonl(&data.join("train.jsonl"))?;
    let valid = count_jsonl(&data.join("valid.jsonl"))?;
    let test = count_jsonl(&data.join("test.jsonl"))?;
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    fs::copy(config, out.join("config.json"))
        .with_context(|| format!("copying {}", config.display()))?;
    ensure_map(out, "speaker_ids.json");
    ensure_map(out, "language_ids.json");
    let dataset = VitsDatasetManifest {
        normalized_manifest: data.join("manifest.jsonl"),
        train_split: data.join("train.jsonl"),
        validation_split: data.join("valid.jsonl"),
        test_split: data.join("test.jsonl"),
        feature_cache,
        sample_rate_hz: model.inference.audio.sample_rate,
        audio_channels: 1,
        train_records: train,
        validation_records: valid,
        test_records: test,
        split_seed: seed,
        license: dataset_license.to_string(),
        provenance: dataset_provenance.to_string(),
    };
    let recipe = VitsTrainingRecipe {
        epochs,
        batch_size,
        segment_frames,
        seed,
        backend: match device {
            ResolvedSpeechDevice::Cpu => VitsTrainingBackend::Cpu,
            ResolvedSpeechDevice::Cuda { .. } => VitsTrainingBackend::Cuda,
        },
        checkpoints: VitsCheckpointPolicy {
            every_steps: checkpoint_every,
            ..Default::default()
        },
        ..Default::default()
    };
    let manifest = VitsTrainingManifest {
        schema_version: VITS_TRAINING_MANIFEST_SCHEMA_VERSION,
        architecture: "vits".into(),
        source_checkpoint: source_checkpoint.map(Path::to_path_buf),
        source_checkpoint_sha256: source_checkpoint.map(sha256_file).transpose()?,
        source_license: source_license.into(),
        source_provenance: source_provenance.into(),
        dataset,
        target_metric: "validation-generator-loss".into(),
        baseline_metric,
        best_metric: None,
    };
    let (layout, _) = initialize_vits_run_with_progress(out, &recipe, &manifest, print_progress)?;
    print_paths(&layout);
    print_compute(recipe.backend, false);
    Ok(())
}

fn run_training(
    run: &Path,
    max_steps: Option<u64>,
    fixture: bool,
    device: ResolvedSpeechDevice,
) -> Result<()> {
    let (recipe, manifest, state, model) = load_run(run)?;
    print_paths(&VitsRunLayout::new(run));
    print_compute(recipe.backend, fixture);
    let speakers = read_map(&run.join("speaker_ids.json"))?;
    let languages = read_map(&run.join("language_ids.json"))?;
    let train = load_vits_examples(
        &manifest.dataset.train_split,
        &manifest.dataset.feature_cache,
        manifest.dataset.sample_rate_hz,
        &speakers,
        &languages,
    )?;
    let valid = load_vits_examples(
        &manifest.dataset.validation_split,
        &manifest.dataset.feature_cache,
        manifest.dataset.sample_rate_hz,
        &speakers,
        &languages,
    )?;
    let options = VitsTrainOptions { max_steps, fixture };
    let layout = VitsRunLayout::new(run);
    let report = match device {
        ResolvedSpeechDevice::Cpu => train_vits::<CpuTrain>(
            &model.inference,
            &recipe,
            &layout,
            state,
            &train,
            &valid,
            manifest.source_checkpoint.as_deref(),
            &NdArrayDevice::Cpu,
            &options,
            print_progress,
        )?,
        ResolvedSpeechDevice::Cuda { index } => train_vits::<CudaTrain>(
            &model.inference,
            &recipe,
            &layout,
            state,
            &train,
            &valid,
            manifest.source_checkpoint.as_deref(),
            &CudaDevice::new(index),
            &options,
            print_progress,
        )?,
    };
    if let Some(metric) = report.best_validation_loss {
        let mut updated_manifest = manifest;
        if updated_manifest.record_metric(metric)? {
            write_vits_training_manifest(&layout, &recipe, &updated_manifest)?;
            println!("Recorded improved target metric validation-generator-loss={metric}");
        }
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn evaluate(run: &Path, split: &str, device: ResolvedSpeechDevice) -> Result<()> {
    let (recipe, manifest, _state, model) = load_run(run)?;
    let path = match split {
        "train" => &manifest.dataset.train_split,
        "valid" | "validation" => &manifest.dataset.validation_split,
        "test" => &manifest.dataset.test_split,
        other => anyhow::bail!("invalid VITS split `{other}`; expected train, valid, or test"),
    };
    let speakers = read_map(&run.join("speaker_ids.json"))?;
    let languages = read_map(&run.join("language_ids.json"))?;
    let examples = load_vits_examples(
        path,
        &manifest.dataset.feature_cache,
        manifest.dataset.sample_rate_hz,
        &speakers,
        &languages,
    )?;
    let layout = VitsRunLayout::new(run);
    let report = match device {
        ResolvedSpeechDevice::Cpu => evaluate_vits::<Cpu>(
            &model.inference,
            &recipe,
            &layout,
            &examples,
            &NdArrayDevice::Cpu,
        )?,
        ResolvedSpeechDevice::Cuda { index } => evaluate_vits::<CudaBackend>(
            &model.inference,
            &recipe,
            &layout,
            &examples,
            &CudaDevice::new(index),
        )?,
    };
    ensure!(
        report.loss.is_finite() && report.finite_audio && report.generated_samples > 0,
        "VITS evaluation did not produce finite non-empty audio"
    );
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn export(run: &Path, device: ResolvedSpeechDevice) -> Result<()> {
    let (_recipe, _manifest, _state, model) = load_run(run)?;
    let layout = VitsRunLayout::new(run);
    let output = match device {
        ResolvedSpeechDevice::Cpu => {
            let device = NdArrayDevice::Cpu;
            let output = export_vits::<Cpu>(&model.inference, &layout, &device)?;
            BurnVitsSpeech::<Cpu>::load(
                run.join("config.json"),
                &output,
                run.join("speaker_ids.json"),
                device,
            )
            .context("loading final VITS export through BurnVitsSpeech")?;
            output
        }
        ResolvedSpeechDevice::Cuda { index } => {
            let device = CudaDevice::new(index);
            let output = export_vits::<CudaBackend>(&model.inference, &layout, &device)?;
            BurnVitsSpeech::<CudaBackend>::load(
                run.join("config.json"),
                &output,
                run.join("speaker_ids.json"),
                device,
            )
            .context("loading final VITS export through BurnVitsSpeech")?;
            output
        }
    };
    println!(
        "Exported and loaded native VITS inference checkpoint: {}",
        output.display()
    );
    Ok(())
}

fn fixture(out: &Path, epochs: u64) -> Result<()> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let config_path = out.join("config.json");
    fs::write(&config_path, fixture_config())
        .with_context(|| format!("writing {}", config_path.display()))?;
    fs::write(out.join("speaker_ids.json"), b"{\"fixture\":0}\n")?;
    ensure_map(out, "language_ids.json");
    let model = load_vits_training_model_config(&config_path)?;
    let data = out.join("fixture-data");
    fs::create_dir_all(&data)?;
    let dataset = VitsDatasetManifest {
        normalized_manifest: data.join("manifest.jsonl"),
        train_split: data.join("train.jsonl"),
        validation_split: data.join("valid.jsonl"),
        test_split: data.join("test.jsonl"),
        feature_cache: data.join("vits-features"),
        sample_rate_hz: model.inference.audio.sample_rate,
        audio_channels: 1,
        train_records: 2,
        validation_records: 1,
        test_records: 1,
        split_seed: 7,
        license: "CC0 fixture".into(),
        provenance: "deterministic generated sine fixture".into(),
    };
    let recipe = VitsTrainingRecipe {
        epochs,
        batch_size: 1,
        segment_frames: 16,
        seed: 7,
        backend: VitsTrainingBackend::Cpu,
        checkpoints: VitsCheckpointPolicy {
            every_steps: 1,
            sample_every_steps: 1,
            keep_last_epochs: 2,
        },
        ..Default::default()
    };
    let manifest = VitsTrainingManifest {
        schema_version: VITS_TRAINING_MANIFEST_SCHEMA_VERSION,
        architecture: "vits".into(),
        source_checkpoint: None,
        source_checkpoint_sha256: None,
        source_license: "random fixture initialization".into(),
        source_provenance: "tongues built-in VITS CPU fixture".into(),
        dataset,
        target_metric: "validation-generator-loss".into(),
        baseline_metric: None,
        best_metric: None,
    };
    let (layout, state) =
        initialize_vits_run_with_progress(out, &recipe, &manifest, print_progress)?;
    let examples = fixture_examples(model.inference.network.out_channels);
    print_paths(&layout);
    print_compute(VitsTrainingBackend::Cpu, true);
    let first = train_vits::<CpuTrain>(
        &model.inference,
        &recipe,
        &layout,
        state,
        &examples[..2],
        &examples[2..],
        None,
        &NdArrayDevice::Cpu,
        &VitsTrainOptions {
            max_steps: Some(1),
            fixture: true,
        },
        print_progress,
    )?;
    ensure!(
        first.interrupted,
        "fixture interruption checkpoint was not exercised"
    );
    let state: tongues_tts::VitsTrainingState = read_json(&layout.train_state())?;
    ensure!(
        state.batch_in_epoch > 0,
        "fixture step checkpoint did not record a batch cursor"
    );
    let baseline = evaluate_vits::<Cpu>(
        &model.inference,
        &recipe,
        &layout,
        &examples[2..],
        &NdArrayDevice::Cpu,
    )?
    .loss;
    let resumed = train_vits::<CpuTrain>(
        &model.inference,
        &recipe,
        &layout,
        state,
        &examples[..2],
        &examples[2..],
        None,
        &NdArrayDevice::Cpu,
        &VitsTrainOptions {
            max_steps: None,
            fixture: true,
        },
        print_progress,
    )?;
    ensure!(!resumed.interrupted, "fixture did not finish after resume");
    let final_report = evaluate_vits::<Cpu>(
        &model.inference,
        &recipe,
        &layout,
        &examples[2..],
        &NdArrayDevice::Cpu,
    )?;
    ensure!(
        final_report.finite_audio && final_report.generated_samples > 0,
        "fixture produced empty or non-finite audio"
    );
    ensure!(
        final_report.loss < baseline,
        "VITS fixture did not overfit: baseline={baseline}, final={}",
        final_report.loss
    );
    export(out, ResolvedSpeechDevice::Cpu)?;
    println!(
        "VITS CPU fixture complete: baseline={baseline}, final={}, resumed_step={}",
        final_report.loss, resumed.global_step
    );
    Ok(())
}

fn fixture_examples(bins: usize) -> Vec<tongues_tts::VitsPreparedExample> {
    (0..3)
        .map(|index| {
            let frames = 16;
            let samples = 128;
            let waveform = (0..samples)
                .map(|sample| {
                    (2.0 * std::f32::consts::PI * (sample + index) as f32 / 16.0).sin() * 0.2
                })
                .collect();
            let spectrogram = (0..bins)
                .map(|bin| {
                    (0..frames)
                        .map(|frame| ((bin + 1) * (frame + 1 + index)) as f32 / 256.0)
                        .collect()
                })
                .collect();
            tongues_tts::VitsPreparedExample {
                record_id: format!("fixture-{index}"),
                token_ids: vec![2, 3],
                spectrogram,
                waveform,
                speaker_id: Some(0),
                language_id: None,
            }
        })
        .collect()
}

fn fixture_config() -> &'static str {
    r#"{
      model: "vits",
      use_phonemes: false,
      phoneme_language: null,
      add_blank: true,
      enable_eos_bos_chars: false,
      characters: {
        characters_class: "fixture.VitsCharacters",
        pad: "_", eos: "", bos: "", blank: null,
        characters: "ab", punctuations: " ",
        phonemes: null, is_unique: true, is_sorted: true
      },
      model_args: {
        num_chars: 5, out_channels: 9, spec_segment_size: 16,
        hidden_channels: 4, hidden_channels_ffn_text_encoder: 8,
        num_heads_text_encoder: 2, num_layers_text_encoder: 1,
        kernel_size_text_encoder: 3, dropout_p_text_encoder: 0.0,
        dropout_p_duration_predictor: 0.0,
        kernel_size_posterior_encoder: 3, dilation_rate_posterior_encoder: 1,
        num_layers_posterior_encoder: 1, kernel_size_flow: 3,
        dilation_rate_flow: 1, num_layers_flow: 1,
        resblock_type_decoder: "1", resblock_kernel_sizes_decoder: [3],
        resblock_dilation_sizes_decoder: [[1,2,3]],
        upsample_rates_decoder: [2,2,2], upsample_initial_channel_decoder: 8,
        upsample_kernel_sizes_decoder: [4,4,4],
        use_sdp: true, inference_noise_scale: 0.0, length_scale: 1.0,
        inference_noise_scale_dp: 0.0, max_inference_len: null,
        use_speaker_embedding: true, num_speakers: 1,
        speaker_embedding_channels: 4, use_d_vector_file: false,
        d_vector_dim: 0, condition_dp_on_speaker: true,
        use_language_embedding: false, embedded_language_dim: 0, num_languages: 0
      },
      audio: {
        fft_size: 16, win_length: 16, hop_length: 8, sample_rate: 8000,
        preemphasis: 0.0, log_func: "np.log10", num_mels: 4,
        mel_fmin: 0.0, mel_fmax: 4000.0, spec_gain: 20.0,
        signal_norm: true, min_level_db: -100.0, ref_level_db: 20.0,
        symmetric_norm: true, max_norm: 4.0, clip_norm: true,
        stats_path: null, do_amp_to_db_mel: true,
        stft_pad_mode: "reflect", centered: true, stft_manual_padding: null
      }
    }"#
}

fn load_run(
    run: &Path,
) -> Result<(
    VitsTrainingRecipe,
    VitsTrainingManifest,
    tongues_tts::VitsTrainingState,
    tongues_tts::VitsTrainingModelConfig,
)> {
    Ok((
        read_json(&run.join("recipe.json"))?,
        read_json(&run.join("training-manifest.json"))?,
        read_json(&run.join("train_state.json"))?,
        load_vits_training_model_config(run.join("config.json"))?,
    ))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path).with_context(|| format!("reading {}", path.display()))?)
        .with_context(|| format!("parsing {}", path.display()))
}

fn read_map(path: &Path) -> Result<HashMap<String, u32>> {
    read_json(path)
}

fn ensure_map(out: &Path, name: &str) {
    let path = out.join(name);
    if !path.exists() {
        let _ = fs::write(path, b"{}\n");
    }
}

fn count_jsonl(path: &Path) -> Result<usize> {
    let source = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(source
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count())
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(
        Sha256::digest(fs::read(path).with_context(|| format!("reading {}", path.display()))?)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    )
}

fn print_paths(layout: &VitsRunLayout) {
    println!("Native VITS durable paths:");
    println!("  train state: {}", layout.train_state().display());
    println!(
        "  generator state: {}",
        layout.root().join("trainer-generator-latest.bin").display()
    );
    println!(
        "  discriminator state: {}",
        layout
            .root()
            .join("trainer-discriminator-latest.bin")
            .display()
    );
    println!(
        "  generator optimizer: {}",
        layout.latest_generator_optimizer().display()
    );
    println!(
        "  discriminator optimizer: {}",
        layout.latest_discriminator_optimizer().display()
    );
    println!(
        "  best inference export: {}",
        layout.best_checkpoint().display()
    );
    println!("  validation samples: {}", layout.sample_dir().display());
}

fn print_compute(backend: VitsTrainingBackend, fixture: bool) {
    match (backend, fixture) {
        (VitsTrainingBackend::Cpu, true) => {
            println!("Compute: compact CPU fixture mode; intended for correctness and overfit checks.")
        }
        (VitsTrainingBackend::Cpu, false) => println!(
            "Compute: CPU selected. This is supported for debugging, but practical VITS training/fine-tuning requires CUDA."
        ),
        (VitsTrainingBackend::Cuda, _) => println!(
            "Compute: CUDA selected. Ensure device memory fits the configured length-aware batch and segment geometry."
        ),
    }
}

fn print_progress(progress: VitsTrainingProgress) {
    match progress {
        VitsTrainingProgress::Initialize { output } => {
            println!("Initializing native VITS run at {}", output.display())
        }
        VitsTrainingProgress::Write { path } => println!("Wrote {}", path.display()),
        VitsTrainingProgress::Resume {
            epoch,
            global_step,
            checkpoint,
        } => println!(
            "Resuming epoch={epoch} global_step={global_step} checkpoint={}",
            checkpoint.display()
        ),
        VitsTrainingProgress::Epoch { epoch, epochs } => {
            println!("VITS epoch {epoch}/{epochs}")
        }
        VitsTrainingProgress::Batch {
            epoch,
            batch,
            batches,
            global_step,
        } if batch <= 3 || batch == batches || batch % 25 == 0 => {
            println!("VITS epoch {epoch} batch {batch}/{batches} global_step={global_step}")
        }
        VitsTrainingProgress::Batch { .. } => {}
        VitsTrainingProgress::Checkpoint {
            epoch,
            global_step,
            path,
        } => println!(
            "Checkpoint epoch={epoch} global_step={global_step} path={}",
            path.display()
        ),
        VitsTrainingProgress::Sample { global_step, path } => println!(
            "Validation sample global_step={global_step} path={}",
            path.display()
        ),
        VitsTrainingProgress::Complete {
            best_epoch,
            best_model,
        } => println!(
            "VITS complete best_epoch={best_epoch} best_model={}",
            best_model.display()
        ),
    }
}
