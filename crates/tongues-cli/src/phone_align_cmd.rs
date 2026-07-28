use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use tongues_audio::{
    evaluate_alignment, AlignmentEvaluationReference, AlignmentEvaluationReport,
    CtcPosteriorBackend, CtcPosteriorMatrix, PhoneAlignmentBackend, PhoneAlignmentRequest,
};

#[derive(Args, Debug)]
pub struct PhoneAlignCommand {
    /// WAV whose original frame timebase is retained in the artifact
    #[arg(long)]
    pub wav: PathBuf,

    /// Versioned schema-v2 transcript/pronunciation/projection request
    #[arg(long)]
    pub request: PathBuf,

    /// Recorded acoustic CTC posterior matrix
    #[arg(long, conflicts_with = "common_phone_model")]
    pub posteriors: Option<PathBuf>,

    /// Native Common Phone model directory used to produce CTC posteriors
    #[arg(long, conflicts_with = "posteriors")]
    pub common_phone_model: Option<PathBuf>,

    /// Versioned schema-v2 alignment artifact written atomically
    #[arg(long)]
    pub out: PathBuf,

    /// Optional trusted reference used to write a reproducible metric report
    #[arg(long)]
    pub evaluation_reference: Option<PathBuf>,

    /// Evaluation report path; required with --evaluation-reference
    #[arg(long, requires = "evaluation_reference")]
    pub evaluation_out: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct PhoneAlignmentEvalCommand {
    /// Redistributable synthetic suite or a compatible local suite
    #[arg(
        long,
        default_value = "fixtures/phone-alignment/multilingual-synthetic-v1.json"
    )]
    pub suite: PathBuf,

    /// Reproducible per-case metric report written atomically
    #[arg(long)]
    pub out: PathBuf,
}

pub fn run(command: PhoneAlignCommand) -> Result<()> {
    anyhow::ensure!(
        command.posteriors.is_some() || command.common_phone_model.is_some(),
        "phone-align requires --posteriors or --common-phone-model"
    );
    let request: PhoneAlignmentRequest = read_json(&command.request)?;
    let audio = tongues_audio::read_wav(&command.wav)
        .with_context(|| format!("reading waveform {}", command.wav.display()))?;
    let posterior = if let Some(path) = &command.posteriors {
        read_json::<CtcPosteriorMatrix>(path)?
    } else {
        let model = command
            .common_phone_model
            .as_deref()
            .expect("one posterior source was required");
        let decoder = tongues_common_phone::CommonPhoneLiveDecoder::load(
            model,
            tongues_common_phone::CommonPhoneTask::Frames2Phones,
        )
        .with_context(|| format!("loading Common Phone model {}", model.display()))?;
        let mono = audio
            .to_mono()
            .context("converting alignment audio to mono")?;
        let config = decoder.config(audio.sample_rate_hz, 25.0, 10.0);
        let languages = request
            .pronunciations
            .iter()
            .map(|path| path.language_tag.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        decoder
            .phone_alignment_posteriors(&mono, audio.sample_rate_hz, &config, languages)
            .context("running Common Phone acoustic posterior inference")?
    };
    let artifact = CtcPosteriorBackend {
        posteriors: posterior,
    }
    .align(&audio, &request)
    .with_context(|| format!("aligning {}", command.wav.display()))?;
    write_json_atomically(&command.out, &artifact)?;

    if let (Some(reference), Some(out)) = (
        command.evaluation_reference.as_deref(),
        command.evaluation_out.as_deref(),
    ) {
        let reference: AlignmentEvaluationReference = read_json(reference)?;
        let tolerance = reference.annotator_tolerance_frames.max(1);
        let report = evaluate_alignment(
            &artifact,
            &reference,
            &[tolerance, tolerance * 2, tolerance * 4],
        );
        write_json_atomically(out, &report)?;
    }
    println!("{}", serde_json::to_string_pretty(&artifact)?);
    Ok(())
}

#[derive(Debug, serde::Deserialize)]
struct AlignmentFixtureSuite {
    schema_version: u32,
    license: String,
    description: String,
    cases: Vec<AlignmentFixtureCase>,
}

#[derive(Debug, serde::Deserialize)]
struct AlignmentFixtureCase {
    id: String,
    audio_frames: usize,
    request: PhoneAlignmentRequest,
    posteriors: CtcPosteriorMatrix,
    reference: AlignmentEvaluationReference,
}

#[derive(Debug, serde::Serialize)]
struct AlignmentFixtureCaseReport {
    id: String,
    readiness: tongues_audio::AlignmentReadiness,
    backend: tongues_audio::AlignmentSourceIdentity,
    metrics: AlignmentEvaluationReport,
}

#[derive(Debug, serde::Serialize)]
struct AlignmentFixtureSuiteReport {
    schema_version: u32,
    source_schema_version: u32,
    license: String,
    description: String,
    cases: Vec<AlignmentFixtureCaseReport>,
}

pub fn run_eval(command: PhoneAlignmentEvalCommand) -> Result<()> {
    let suite: AlignmentFixtureSuite = read_json(&command.suite)?;
    anyhow::ensure!(
        suite.schema_version == 1,
        "alignment evaluation suite schema {} is unsupported; expected 1",
        suite.schema_version
    );
    let mut reports = Vec::with_capacity(suite.cases.len());
    for case in suite.cases {
        let audio = tongues_audio::AudioBuffer {
            samples: vec![0.05; case.audio_frames],
            sample_rate_hz: case.posteriors.sample_rate_hz,
            channels: 1,
        };
        let artifact = CtcPosteriorBackend {
            posteriors: case.posteriors,
        }
        .align(&audio, &case.request)
        .with_context(|| format!("aligning evaluation case {}", case.id))?;
        let tolerance = case.reference.annotator_tolerance_frames.max(1);
        reports.push(AlignmentFixtureCaseReport {
            id: case.id,
            readiness: artifact.readiness,
            backend: artifact.backend.clone(),
            metrics: evaluate_alignment(
                &artifact,
                &case.reference,
                &[tolerance, tolerance * 2, tolerance * 4],
            ),
        });
    }
    let report = AlignmentFixtureSuiteReport {
        schema_version: 1,
        source_schema_version: suite.schema_version,
        license: suite.license,
        description: suite.description,
        cases: reports,
    };
    write_json_atomically(&command.out, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

fn write_json_atomically(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating output directory {}", parent.display()))?;
    }
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("phone-alignment.json");
    let part = path.with_file_name(format!("{filename}.part"));
    let file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .with_context(|| format!("serializing {}", part.display()))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&part, path)
        .with_context(|| format!("committing {} -> {}", part.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tongues_audio::{
        AlignmentLimits, AlignmentMode, AlignmentSourceIdentity, AlignmentUnitSpec,
        AudioAlignmentInput, PhoneAlignmentArtifact, PhoneticSegmentationContext,
        PronunciationPath, SegmentKind, PHONE_ALIGNMENT_SCHEMA_VERSION,
    };

    #[test]
    fn recorded_ctc_fixture_aligns_end_to_end_and_commits_atomically() {
        let root = std::env::temp_dir().join(format!(
            "tongues-phone-align-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let wav = root.join("fixture.wav");
        let request_path = root.join("request.json");
        let posterior_path = root.join("posterior.json");
        let out = root.join("alignment.json");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 1_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&wav, spec).unwrap();
        for _ in 0..100 {
            writer.write_sample(1_000_i16).unwrap();
        }
        writer.finalize().unwrap();
        let unit = |id: &str, symbol: &str| AlignmentUnitSpec {
            id: id.into(),
            symbol: symbol.into(),
            kind: SegmentKind::Phone,
            language_tag: "en".into(),
            inventory_id: "fixture-ipa".into(),
            utterance_ids: vec!["utterance:0".into()],
            transcript_token_ids: vec!["token:0".into()],
            word_ids: vec!["word:0".into()],
            morpheme_ids: Vec::new(),
            syllable_ids: Vec::new(),
            phoneme_ids: vec![format!("phoneme:{id}")],
            speaker_span_ids: Vec::new(),
        };
        let request = PhoneAlignmentRequest {
            schema_version: PHONE_ALIGNMENT_SCHEMA_VERSION,
            mode: AlignmentMode::PronunciationConstrained,
            audio: AudioAlignmentInput {
                artifact_id: "fixture.wav".into(),
                expected_sha256: None,
                channel: 0,
                selected_regions: Vec::new(),
                preprocessing_artifacts: Vec::new(),
            },
            transcript: None,
            pronunciations: vec![PronunciationPath {
                id: "path:cat".into(),
                lexical_source: "fixture".into(),
                language_tag: "en".into(),
                inventory_id: "fixture-ipa".into(),
                prior_probability: 1.0,
                units: vec![unit("k", "k"), unit("ae", "æ"), unit("t", "t")],
            }],
            timing_hints: Vec::new(),
            duration_priors: Vec::new(),
            corrections: Vec::new(),
            projections: Vec::new(),
            limits: AlignmentLimits {
                minimum_path_posterior: 0.0,
                minimum_selection_margin: 0.0,
                ..Default::default()
            },
            context: PhoneticSegmentationContext {
                graph_id: "graph:fixture".into(),
                graph_revision: 1,
                recipe_id: "recipe:fixture".into(),
                execution_record_id: "run:fixture".into(),
                session_id: None,
                audio_span_id: None,
                runtime: "tongues-cli-test".into(),
                runtime_version: env!("CARGO_PKG_VERSION").into(),
            },
        };
        let mut rows = vec![vec![0.97, 0.01, 0.01, 0.01]; 10];
        for (range, symbol) in [(1..3, 1), (3..6, 2), (6..9, 3)] {
            for frame in range {
                rows[frame] = vec![0.02; 4];
                rows[frame][symbol] = 0.94;
            }
        }
        let posterior = CtcPosteriorMatrix {
            schema_version: 1,
            source: AlignmentSourceIdentity {
                provider: "common-phone".into(),
                model: "recorded-fixture".into(),
                version: "1".into(),
                artifact_id: Some("fixture-posteriors.json".into()),
            },
            language_tags: vec!["en".into()],
            inventory_id: "fixture-ipa".into(),
            sample_rate_hz: 1_000,
            frame_start: 0,
            frame_stride: 10,
            frame_width: 10,
            blank_index: 0,
            symbols: vec!["<blank>".into(), "k".into(), "æ".into(), "t".into()],
            probabilities: rows,
            model_checksum: Some("sha256:fixture".into()),
        };
        fs::write(&request_path, serde_json::to_vec_pretty(&request).unwrap()).unwrap();
        fs::write(
            &posterior_path,
            serde_json::to_vec_pretty(&posterior).unwrap(),
        )
        .unwrap();
        run(PhoneAlignCommand {
            wav,
            request: request_path,
            posteriors: Some(posterior_path),
            common_phone_model: None,
            out: out.clone(),
            evaluation_reference: None,
            evaluation_out: None,
        })
        .unwrap();
        let artifact: PhoneAlignmentArtifact =
            serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
        assert!(artifact.selected_hypothesis_id.is_some());
        assert_eq!(artifact.selected_hypothesis().unwrap().units.len(), 3);
        assert!(!out.with_file_name("alignment.json.part").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn multilingual_evaluation_command_writes_language_broken_out_metrics() {
        let root = std::env::temp_dir().join(format!(
            "tongues-phone-align-eval-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let out = root.join("report.json");
        run_eval(PhoneAlignmentEvalCommand {
            suite: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/phone-alignment/multilingual-synthetic-v1.json"),
            out: out.clone(),
        })
        .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
        assert_eq!(value["cases"].as_array().unwrap().len(), 3);
        let languages = value["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|case| case["metrics"]["language_tag"].as_str().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            languages,
            std::collections::BTreeSet::from(["en", "ja", "mul"])
        );
        let cases = value["cases"].as_array().unwrap();
        assert_eq!(
            cases[0]["metrics"]["breakdowns"]["phone_class"]["stop"]["reference_units"],
            2
        );
        assert_eq!(
            cases[0]["metrics"]["breakdowns"]["evidence_source"]["forced_alignment"]
                ["aligned_units"],
            3
        );
        assert_eq!(
            cases[0]["metrics"]["breakdowns"]["backend"]["fixture-common-phone/recorded-ctc/1"]
                ["aligned_units"],
            3
        );
        assert_eq!(
            cases[2]["metrics"]["breakdowns"]["language"]["sw"]["reference_units"],
            2
        );
        assert!(!out.with_file_name("report.json.part").exists());
        fs::remove_dir_all(root).unwrap();
    }
}
