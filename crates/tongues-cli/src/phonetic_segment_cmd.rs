use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use tongues_audio::{AlignmentRecipe, PhoneticSegmentationEngine};

#[derive(Args, Debug)]
pub struct PhoneticSegmentCommand {
    /// WAV file whose original frame timebase is used by alignment candidates
    #[arg(long)]
    pub wav: PathBuf,

    /// Versioned JSON alignment recipe containing expected symbols and hints
    #[arg(long)]
    pub recipe: PathBuf,

    /// Versioned phonetic-segment artifact to write atomically
    #[arg(long)]
    pub out: PathBuf,
}

pub fn run(command: PhoneticSegmentCommand) -> Result<()> {
    let recipe_bytes = fs::read(&command.recipe)
        .with_context(|| format!("reading alignment recipe {}", command.recipe.display()))?;
    let recipe: AlignmentRecipe = serde_json::from_slice(&recipe_bytes)
        .with_context(|| format!("parsing alignment recipe {}", command.recipe.display()))?;
    let audio = tongues_audio::read_wav(&command.wav)
        .with_context(|| format!("reading waveform {}", command.wav.display()))?;
    let artifact = PhoneticSegmentationEngine::default()
        .segment_recipe(&audio, &recipe)
        .with_context(|| format!("segmenting {}", command.wav.display()))?;

    write_json_atomically(&command.out, &artifact)?;
    println!("{}", serde_json::to_string_pretty(&artifact)?);
    Ok(())
}

fn write_json_atomically(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating output directory {}", parent.display()))?;
    }
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("phonetic-segments.json");
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
        AlignmentCandidate, AlignmentEvidence, AlignmentSourceIdentity, ExpectedSegment,
        InventoryMembership, PhoneticSegmentArtifact, PhoneticSegmentationContext,
        PhoneticSegmentationReadiness, SegmentKind, ALIGNMENT_RECIPE_SCHEMA_VERSION,
    };

    #[test]
    fn command_runs_fixture_end_to_end_and_commits_versioned_artifact() {
        let root = std::env::temp_dir().join(format!(
            "tongues-phonetic-segment-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let wav = root.join("fixture.wav");
        let recipe_path = root.join("recipe.json");
        let out = root.join("segments.json");

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 1_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&wav, spec).unwrap();
        for _ in 0..500 {
            writer.write_sample(1_000_i16).unwrap();
        }
        writer.finalize().unwrap();

        let source = AlignmentSourceIdentity {
            provider: "fixture".into(),
            model: "manual-boundaries".into(),
            version: "1".into(),
            artifact_id: Some("fixture-hints.json".into()),
        };
        let recipe = AlignmentRecipe {
            schema_version: ALIGNMENT_RECIPE_SCHEMA_VERSION,
            audio_artifact_id: "fixture.wav".into(),
            expected_audio_sha256: None,
            transcript: Some("ta".into()),
            expected: vec![
                ExpectedSegment {
                    symbol: "t".into(),
                    kind: SegmentKind::Phone,
                    inventory_membership: InventoryMembership::Known,
                    language_tag: "mul".into(),
                    inventory_id: "fixture-ipa".into(),
                    pronunciation_source: "fixture-pronunciation".into(),
                },
                ExpectedSegment {
                    symbol: "a".into(),
                    kind: SegmentKind::Phone,
                    inventory_membership: InventoryMembership::Known,
                    language_tag: "mul".into(),
                    inventory_id: "fixture-ipa".into(),
                    pronunciation_source: "fixture-pronunciation".into(),
                },
            ],
            candidates: vec![
                AlignmentCandidate {
                    expected_index: 0,
                    start_frame: 100,
                    end_frame: 180,
                    confidence: 0.9,
                    source: source.clone(),
                    evidence: AlignmentEvidence::default(),
                },
                AlignmentCandidate {
                    expected_index: 1,
                    start_frame: 180,
                    end_frame: 300,
                    confidence: 0.9,
                    source,
                    evidence: AlignmentEvidence::default(),
                },
            ],
            context: PhoneticSegmentationContext {
                graph_id: "fixture-graph".into(),
                graph_revision: 1,
                recipe_id: "fixture-recipe".into(),
                execution_record_id: "fixture-execution".into(),
                runtime: "tongues-cli-test".into(),
                runtime_version: env!("CARGO_PKG_VERSION").into(),
            },
        };
        fs::write(&recipe_path, serde_json::to_vec_pretty(&recipe).unwrap()).unwrap();

        run(PhoneticSegmentCommand {
            wav,
            recipe: recipe_path,
            out: out.clone(),
        })
        .unwrap();

        let artifact: PhoneticSegmentArtifact =
            serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
        assert_eq!(artifact.readiness, PhoneticSegmentationReadiness::Ready);
        assert_eq!(artifact.segments.len(), 2);
        assert!(artifact.recipe_sha256.starts_with("sha256:"));
        assert!(!out.with_file_name("segments.json.part").exists());

        fs::remove_dir_all(root).unwrap();
    }
}
