use std::fs::{self, File};
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde_json::json;

use burn::backend::ndarray::NdArrayDevice;
use burn_cuda::CudaDevice;

use speaking::PhoneToken;
use speaking::Spec;
use speaking::UtteranceId;
use speaking::UtterancePlan;
use speaking::VarietyId;

use styletts2::prepare_styletts2_plan;
use styletts2::styletts2_en_us_symbol_set;
use styletts2::styletts2_text_for_symbols;
use styletts2::StyleTts2DiffusionOptions;
use styletts2::StyleTts2OnnxBackend;
use styletts2::StyleTts2PlanOptions;
use styletts2::StyleTts2SynthesisRequest;

use styletts2::StyleTts2Backend;

use tongues_core::Vocab;
use tongues_g2p2g::ModelConfig;

use crate::models::ensure_styletts2_model_available;
use crate::speak::write_wav_mono_f32;
use crate::{DeviceArg, Styletts2Commands};

pub fn run_styletts2_command(command: Styletts2Commands, device_arg: DeviceArg) -> Result<()> {
    match command {
        Styletts2Commands::Discover {
            text,
            out_dir,
            num_samples,
            head2phones_model,
            variety,
            seed,
            tier,
            references_dir,
        } => run_discover(
            &text,
            &out_dir,
            num_samples,
            &head2phones_model,
            &variety,
            seed,
            tier,
            references_dir.as_deref(),
            device_arg,
        ),
        Styletts2Commands::EncodeStyle { refs, out, labels } => {
            run_encode_style(&refs, &labels, &out)
        }
        Styletts2Commands::EmotionSignatures {
            style_vectors,
            method,
            out,
        } => run_emotion_signatures(&style_vectors, &method, &out),
    }
}

fn run_discover(
    text: &str,
    out_dir: &Path,
    num_samples: usize,
    head2phones_model: &Path,
    variety: &str,
    seed: u64,
    tier: u8,
    references_dir: Option<&Path>,
    device_arg: DeviceArg,
) -> Result<()> {
    // 1. Run text through head2phones parser
    println!(
        "Loading head2phones model from {}...",
        head2phones_model.display()
    );
    let manifest_path = head2phones_model.join(tongues_neural::ARTIFACT_MANIFEST_FILE);
    let manifest = tongues_neural::read_manifest(&manifest_path)?;
    anyhow::ensure!(
        manifest.family == tongues_head2phones::FAMILY,
        "expected head2phones manifest, found `{}`",
        manifest.family
    );

    let model_config_path = head2phones_model.join("model_config.json");
    let model_config: ModelConfig = crate::read_json_file(&model_config_path)?;
    let vocab_path = head2phones_model.join("vocab.json");
    let vocab: Vocab = crate::read_json_file(&vocab_path)?;

    let input = tongues_head2phones::format_input_for_variety(variety, text);

    println!("Predicting sentence boundary and phones...");
    let output = match device_arg {
        DeviceArg::Cpu => {
            let device = NdArrayDevice::Cpu;
            let model = tongues_g2p2g::load_model::<crate::CpuInferBackend>(
                &model_config,
                &head2phones_model.join("model"),
                &device,
            )?;
            crate::predict_sentence_boundary(&model, &input, &vocab, &device)
        }
        DeviceArg::Cuda => {
            let device = CudaDevice::default();
            let model = tongues_g2p2g::load_model::<crate::CudaInferBackend>(
                &model_config,
                &head2phones_model.join("model"),
                &device,
            )?;
            crate::predict_sentence_boundary(&model, &input, &vocab, &device)
        }
    };

    println!("head2phones output: {}", output);

    // Extract phones
    let phones_open = tongues_head2phones::PHONES_OPEN;
    let phones_close = tongues_head2phones::PHONES_CLOSE;

    let phones_str = if let Some(start) = output.find(phones_open) {
        if let Some(end) = output.find(phones_close) {
            output[start + phones_open.len()..end].trim()
        } else {
            anyhow::bail!("Found {} but no closing tag", phones_open)
        }
    } else {
        anyhow::bail!("No {} tag found in head2phones output", phones_open)
    };

    let phone_tokens: Vec<PhoneToken> = phones_str
        .split_whitespace()
        .filter(|&s| s != "|")
        .map(|s| PhoneToken {
            phone: Spec::Known(speaking::ids::PhoneId(s.to_string().into())),
            span: None,
            features: Default::default(),
            acoustic_evidence: vec![],
            confidence: 1.0,
            provenance: speaking::EvidenceProvenance {
                source: speaking::EvidenceSource::Inference,
                method: "tongues_head2phones".into(),
                version: Some("0.1".into()),
            },
        })
        .collect();

    if phone_tokens.is_empty() {
        anyhow::bail!("No phones parsed from output.");
    }

    let utterance_plan = UtterancePlan {
        id: UtteranceId("styletts2.discover".to_string()),
        variety: VarietyId(variety.to_string()),
        speaker: None,
        intended_text: Some(text.to_string()),
        intended_morphemes: vec![],
        intended_phonemes: vec![],
        target_phones: phone_tokens,
        target_syllables: vec![],
        boundaries: vec![],
        target_prosody: Default::default(),
        target_acoustics: vec![],
        style: None,
        provenance: speaking::EvidenceProvenance {
            source: speaking::EvidenceSource::TtsPlan,
            method: "tongues styletts2 discover".into(),
            version: Some("0.1".into()),
        },
    };

    // Load StyleTTS2 Backend
    println!("Loading StyleTTS2 ONNX Backend...");
    let primary_model = ensure_styletts2_model_available()?;
    let model_dir = primary_model
        .parent()
        .context("StyleTTS2 primary model path has no parent directory")?;

    let mut backend = StyleTts2OnnxBackend::from_model_dir(model_dir)
        .context("failed to load native StyleTTS2 ONNX backend")?;

    fs::create_dir_all(out_dir).context("failed to create output directory")?;
    let manifest_out_path = out_dir.join("manifest.jsonl");
    let mut manifest_out = BufWriter::new(File::create(&manifest_out_path)?);

    let mut reference_vectors: Vec<Vec<f32>> = Vec::new();
    if tier == 2 {
        if let Some(refs_dir) = references_dir {
            println!("Loading reference styles from {}...", refs_dir.display());
            for entry in fs::read_dir(refs_dir).context("failed to read references dir")? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("wav") {
                    let uri = format!("file://{}", path.display());
                    match backend.reference_style_vector(&uri) {
                        Ok(vector) => reference_vectors.push(vector),
                        Err(e) => println!("Warning: failed to encode {}: {}", path.display(), e),
                    }
                }
            }
            if reference_vectors.is_empty() {
                anyhow::bail!("No valid WAV references found in {}", refs_dir.display());
            }
            println!("Loaded {} reference styles.", reference_vectors.len());
        } else {
            anyhow::bail!("Tier 2 requires a --references-dir");
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);

    let styletts2_plan = prepare_styletts2_plan(
        &utterance_plan,
        &styletts2_en_us_symbol_set(),
        StyleTts2PlanOptions::default(),
    )?;

    // Extract predicted symbols for logging
    let mut predicted_symbols_list = Vec::new();
    for chunk in &styletts2_plan.chunks {
        if let Ok(sym_str) = styletts2_text_for_symbols(&chunk.symbols) {
            predicted_symbols_list.push(sym_str.trim().to_string());
        }
    }
    let predicted_symbols = predicted_symbols_list.join(" || ");

    for i in 0..num_samples {
        let sample_seed: u64 = rng.gen();
        let alpha: f32 = rng.gen_range(0.0..1.0);
        let beta: f32 = rng.gen_range(0.0..1.0);
        let speed: f64 = rng.gen_range(0.85..1.2);

        let diffusion_opts = StyleTts2DiffusionOptions {
            diffusion_steps: 5,
            alpha,
            beta,
            embedding_scale: 1.0,
            seed: sample_seed,
        };

        backend = backend.with_diffusion_options(diffusion_opts)?;
        backend = backend.with_speed(speed)?;

        let mut style_ref = None;
        if tier == 2 {
            let v1 = &reference_vectors[rng.gen_range(0..reference_vectors.len())];
            let v2 = &reference_vectors[rng.gen_range(0..reference_vectors.len())];
            let w: f32 = rng.gen();
            let mut mixed = Vec::with_capacity(256);
            for (a, b) in v1.iter().zip(v2.iter()) {
                mixed.push(w * a + (1.0 - w) * b);
            }
            style_ref = Some(speaking::StyleRef {
                description: None,
                source: speaking::StyleSource::Embedding {
                    kind: "styletts2.style_vector.v1".into(),
                    values: mixed,
                },
            });
        } else if tier == 3 {
            let mut feral = Vec::with_capacity(256);
            for _ in 0..128 {
                let u1: f32 = rng.gen_range(1e-6..1.0);
                let u2: f32 = rng.gen_range(0.0..1.0);
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).sin();
                feral.push(z0);
                feral.push(z1);
            }
            style_ref = Some(speaking::StyleRef {
                description: None,
                source: speaking::StyleSource::Embedding {
                    kind: "styletts2.style_vector.v1".into(),
                    values: feral,
                },
            });
        }

        let request = StyleTts2SynthesisRequest::from_backend_plan(
            styletts2_plan.clone(),
            None,
            style_ref,
            Default::default(),
        );

        let mut pcm_mono_f32 = Vec::new();
        let output = backend
            .synthesize_streaming(&request, &mut |chunk: styletts2::StyleTts2AudioChunk| {
                pcm_mono_f32.extend(&chunk.pcm_mono_f32);
                Ok(())
            })
            .context("native StyleTTS2 ONNX synthesis failed")?;

        let wav_filename = format!("sample_{:04}.wav", i);
        let wav_path = out_dir.join(&wav_filename);
        write_wav_mono_f32(&wav_path, output.sample_rate_hz, &pcm_mono_f32)?;

        let meta = json!({
            "id": i,
            "text": text,
            "wav_path": wav_filename,
            "seed": sample_seed,
            "alpha": alpha,
            "beta": beta,
            "speed": speed,
            "tier": tier,
            "predicted_symbols": predicted_symbols,
            "style_vector": output.style_vector,
        });

        writeln!(manifest_out, "{}", serde_json::to_string(&meta)?)?;
        println!("Generated sample {} -> {}", i, wav_path.display());
    }

    manifest_out.flush()?;
    println!(
        "Discovery complete! Manifest saved to {}",
        manifest_out_path.display()
    );
    Ok(())
}

fn run_encode_style(refs: &Path, labels: &Path, out: &Path) -> Result<()> {
    use serde::Deserialize;
    use std::io::BufRead;

    println!("Loading labels from {}...", labels.display());
    let labels_file = fs::File::open(labels).context("failed to open labels file")?;
    let reader = std::io::BufReader::new(labels_file);

    #[derive(Deserialize)]
    struct LabelEntry {
        path: String,
        emotion: String,
        speaker: String,
    }

    let mut label_map = std::collections::HashMap::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: LabelEntry = serde_json::from_str(&line)?;
        label_map.insert(entry.path.clone(), entry);
    }
    println!("Loaded {} labels.", label_map.len());

    let primary_model = ensure_styletts2_model_available()?;
    let model_dir = primary_model.parent().unwrap();
    let mut backend = StyleTts2OnnxBackend::from_model_dir(model_dir)?;

    let out_file = fs::File::create(out)?;
    let mut out_writer = std::io::BufWriter::new(out_file);

    let mut success_count = 0;

    // Instead of using walkdir directly, we just iterate through the mapped files if refs is not a dir,
    // actually refs might be the directory and we iterate over walkdir, checking if they exist in label_map.
    for entry in walkdir::WalkDir::new(refs)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.path().extension().and_then(|e| e.to_str()) == Some("wav") {
            let path_str = entry
                .path()
                .canonicalize()
                .unwrap_or(entry.path().to_path_buf())
                .display()
                .to_string();
            if let Some(lbl) = label_map.get(&path_str) {
                let uri = format!("file://{}", path_str);
                match backend.reference_style_vector(&uri) {
                    Ok(vector) => {
                        let out_json = serde_json::json!({
                            "id": success_count,
                            "path": path_str,
                            "emotion": lbl.emotion,
                            "speaker": lbl.speaker,
                            "vector": vector,
                        });
                        writeln!(out_writer, "{}", serde_json::to_string(&out_json)?)?;
                        success_count += 1;
                        if success_count % 100 == 0 {
                            println!("Encoded {} files...", success_count);
                        }
                    }
                    Err(e) => println!("Warning: failed to encode {}: {}", path_str, e),
                }
            }
        }
    }

    out_writer.flush()?;
    println!(
        "Successfully encoded {} reference styles to {}",
        success_count,
        out.display()
    );

    Ok(())
}

fn run_emotion_signatures(style_vectors_path: &Path, method: &str, out: &Path) -> Result<()> {
    if method != "speaker-neutral-delta" {
        anyhow::bail!("Unsupported method: {}", method);
    }

    use serde::Deserialize;
    use std::io::BufRead;

    let in_file = fs::File::open(style_vectors_path)?;
    let reader = std::io::BufReader::new(in_file);

    #[derive(Deserialize)]
    struct VectorEntry {
        emotion: String,
        speaker: String,
        vector: Vec<f32>,
    }

    // Group by speaker -> emotion -> Vec<Vec<f32>>
    let mut speaker_map: std::collections::HashMap<
        String,
        std::collections::HashMap<String, Vec<Vec<f32>>>,
    > = std::collections::HashMap::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: VectorEntry = serde_json::from_str(&line)?;
        speaker_map
            .entry(entry.speaker)
            .or_default()
            .entry(entry.emotion)
            .or_default()
            .push(entry.vector);
    }

    let mut emotion_deltas: std::collections::HashMap<String, Vec<Vec<f32>>> =
        std::collections::HashMap::new();

    for (speaker, emotions) in &speaker_map {
        if let Some(neutrals) = emotions.get("neutral") {
            // Mean of neutral
            let mut neutral_mean = vec![0.0f32; 256];
            for v in neutrals {
                for (i, val) in v.iter().enumerate() {
                    neutral_mean[i] += val;
                }
            }
            for val in &mut neutral_mean {
                *val /= neutrals.len() as f32;
            }

            for (emotion, vectors) in emotions {
                if emotion == "neutral" {
                    continue;
                }

                let mut emotion_mean = vec![0.0f32; 256];
                for v in vectors {
                    for (i, val) in v.iter().enumerate() {
                        emotion_mean[i] += val;
                    }
                }
                for val in &mut emotion_mean {
                    *val /= vectors.len() as f32;
                }

                let mut delta = vec![0.0f32; 256];
                for i in 0..256 {
                    delta[i] = emotion_mean[i] - neutral_mean[i];
                }

                emotion_deltas
                    .entry(emotion.clone())
                    .or_default()
                    .push(delta);
            }
        } else {
            println!(
                "Warning: speaker {} has no 'neutral' emotion to compute deltas.",
                speaker
            );
        }
    }

    // Output signatures
    let out_file = fs::File::create(out)?;
    let mut out_writer = std::io::BufWriter::new(out_file);
    let mut signatures = serde_json::Map::new();

    for (emotion, deltas) in emotion_deltas {
        let mut final_delta = vec![0.0f32; 256];
        for d in &deltas {
            for i in 0..256 {
                final_delta[i] += d[i];
            }
        }
        for val in &mut final_delta {
            *val /= deltas.len() as f32;
        }

        let sig = serde_json::json!({
            "kind": "styletts2.emotion_signature.v1",
            "emotion": emotion,
            "method": method,
            "dims": 256,
            "vector": final_delta,
            "stats": {
                "n_speakers": deltas.len(),
            },
            "recommended_strength": {
                "subtle": 0.25,
                "normal": 0.65,
                "strong": 1.10
            }
        });

        signatures.insert(emotion, sig);
    }

    writeln!(out_writer, "{}", serde_json::to_string_pretty(&signatures)?)?;
    println!(
        "Saved signatures for {} emotions to {}",
        signatures.len(),
        out.display()
    );

    Ok(())
}
