use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use speaking::{UtteranceId, VarietyId};
use tongues_duplex::{
    prepare_dataset_with_progress, replay_journal, DuplexFixtureSuite, DuplexPrepareProgress,
    DuplexSimulator, EvidenceModality, FixtureCompletionProvider, NormalizedCompletionHypothesis,
    ObservedEvidence, OracleCompletionProvider, SimulatorConfig, SimulatorEventKind,
    SimulatorJournal, SimulatorState,
};

const DEFAULT_FIXTURES_PATH: &str = "fixtures/duplex/completion_scenarios_v1.json";
const DEFAULT_DUPLEX_DATA_DIR: &str = "datasets/duplex/v0";

#[derive(Subcommand, Debug)]
pub enum DuplexCommands {
    /// Run deterministic completion beams over fixture or chunked evidence
    Demo(DuplexDemoCommand),
    /// Prepare a prefix/completion/rollback/repair training dataset from fixtures
    Prepare(DuplexPrepareCommand),
}

#[derive(Args, Debug)]
pub struct DuplexDemoCommand {
    /// Built-in fixture ID to run
    #[arg(long, default_value = "who-shot-john-f")]
    fixture: String,

    /// Completion fixture suite
    #[arg(long, default_value = DEFAULT_FIXTURES_PATH)]
    fixtures: PathBuf,

    /// List fixture IDs and exit
    #[arg(long)]
    list: bool,

    /// Feed a direct text chunk to the deterministic oracle; repeat for streaming
    #[arg(long = "chunk", conflicts_with = "list")]
    chunks: Vec<String>,

    /// Feed a mock/prerecorded acoustic transcript chunk; repeat for streaming
    #[arg(long = "mock-acoustic", conflicts_with = "list")]
    mock_acoustics: Vec<String>,

    /// Variety used for oracle chunks
    #[arg(long, default_value = "en-US-GA")]
    variety: String,

    /// Override the fixture/oracle posterior-mass threshold
    #[arg(long)]
    posterior_mass: Option<f64>,

    /// Replayable journal output; defaults to target/duplex/<run>.journal.json
    #[arg(long)]
    journal: Option<PathBuf>,

    /// Human timeline or complete JSON result
    #[arg(long, value_enum, default_value = "text")]
    format: DuplexOutputFormat,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DuplexOutputFormat {
    Text,
    Json,
}

/// Arguments for `tongues duplex prepare`.
#[derive(Args, Debug)]
pub struct DuplexPrepareCommand {
    /// Duplex fixture suite JSON to prepare from
    #[arg(long, default_value = DEFAULT_FIXTURES_PATH)]
    fixtures: PathBuf,

    /// Output directory for the prepared dataset
    #[arg(long, default_value = DEFAULT_DUPLEX_DATA_DIR)]
    out: PathBuf,
}

pub fn run(command: DuplexCommands) -> Result<()> {
    match command {
        DuplexCommands::Demo(command) => run_demo(command),
        DuplexCommands::Prepare(command) => run_prepare(command),
    }
}

fn run_prepare(command: DuplexPrepareCommand) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};
    use std::time::Duration;

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} {msg}").expect("valid spinner template"),
    );
    pb.enable_steady_tick(Duration::from_millis(120));
    pb.set_message(format!(
        "Preparing duplex dataset at {}",
        command.out.display()
    ));

    let report = prepare_dataset_with_progress(&command.out, &command.fixtures, {
        let pb = pb.clone();
        move |progress| match progress {
            DuplexPrepareProgress::Stage { message } => pb.set_message(message),
            DuplexPrepareProgress::Fixture { fixture_id, rows } => {
                pb.set_message(format!("Processed fixture '{fixture_id}' → {rows} rows"));
            }
            DuplexPrepareProgress::Write { path, rows } => {
                pb.set_message(format!("Wrote {rows} rows → {path}"));
            }
        }
    })?;

    pb.finish_and_clear();
    println!(
        "Prepared duplex dataset at {}: {} train / {} valid / {} test rows \
         from {} fixtures ({} prefix, {} completion, {} rollback, {} repair)",
        command.out.display(),
        report.train_rows,
        report.valid_rows,
        report.test_rows,
        report.fixtures,
        report.prefix_rows,
        report.completion_rows,
        report.rollback_rows,
        report.repair_rows,
    );
    Ok(())
}

fn run_demo(command: DuplexDemoCommand) -> Result<()> {
    let suite = load_suite(&command.fixtures)?;
    if command.list {
        for fixture in &suite.fixtures {
            println!("{}\t{}", fixture.id, fixture.description);
        }
        return Ok(());
    }

    let custom_evidence = !command.chunks.is_empty() || !command.mock_acoustics.is_empty();
    let (run_id, journal, state) = if custom_evidence {
        run_oracle_chunks(&command)?
    } else {
        let fixture = suite.fixture(&command.fixture).ok_or_else(|| {
            anyhow!(
                "unknown duplex fixture '{}'; use `tongues duplex demo --list`",
                command.fixture
            )
        })?;
        let mut config = fixture.config.clone();
        if let Some(posterior_mass) = command.posterior_mass {
            config.posterior_mass = posterior_mass;
        }
        let provider = FixtureCompletionProvider::new(fixture);
        let mut simulator = DuplexSimulator::new(
            fixture.utterance_id.clone(),
            fixture.variety.clone(),
            config,
            provider,
        )?;
        for step in &fixture.steps {
            simulator.observe(step.evidence.clone())?;
        }
        let (journal, state) = simulator.into_parts();
        (fixture.id.clone(), journal, state)
    };

    let replayed = replay_journal(&journal).context("replaying generated duplex journal")?;
    if replayed != state {
        bail!("generated duplex journal did not reproduce the live state");
    }
    let journal_path = command
        .journal
        .unwrap_or_else(|| PathBuf::from(format!("target/duplex/{run_id}.journal.json")));
    write_journal_atomic(&journal_path, &journal)?;

    match command.format {
        DuplexOutputFormat::Text => print_timeline(&run_id, &journal_path, &journal, &state),
        DuplexOutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "run": run_id,
                    "journal_path": journal_path,
                    "journal": journal,
                    "final_state": state,
                    "replay_verified": true,
                }))?
            );
        }
    }
    Ok(())
}

fn run_oracle_chunks(
    command: &DuplexDemoCommand,
) -> Result<(String, SimulatorJournal, SimulatorState)> {
    let mut config = SimulatorConfig::default();
    if let Some(posterior_mass) = command.posterior_mass {
        config.posterior_mass = posterior_mass;
    }
    let run_id = "oracle-chunks".to_string();
    let mut simulator = DuplexSimulator::new(
        UtteranceId(run_id.clone()),
        VarietyId(command.variety.clone()),
        config,
        OracleCompletionProvider,
    )?;
    for (index, chunk) in command.chunks.iter().enumerate() {
        simulator.observe(ObservedEvidence::text(format!("text:{index}"), chunk))?;
    }
    for (index, transcript) in command.mock_acoustics.iter().enumerate() {
        simulator.observe(ObservedEvidence::acoustics(
            format!("acoustics:{index}"),
            transcript,
        ))?;
    }
    let (journal, state) = simulator.into_parts();
    Ok((run_id, journal, state))
}

fn load_suite(path: &Path) -> Result<DuplexFixtureSuite> {
    let bytes = fs::read(path)
        .with_context(|| format!("reading duplex fixture suite {}", path.display()))?;
    let suite: DuplexFixtureSuite = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing duplex fixture suite {}", path.display()))?;
    suite
        .validate()
        .with_context(|| format!("validating duplex fixture suite {}", path.display()))?;
    Ok(suite)
}

fn write_journal_atomic(path: &Path, journal: &SimulatorJournal) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("creating duplex journal directory {}", parent.display())
            })?;
        }
    }
    let part = part_path(path);
    let file = File::create(&part)
        .with_context(|| format!("creating duplex journal {}", part.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, journal)
        .with_context(|| format!("writing duplex journal {}", part.display()))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&part, path).with_context(|| {
        format!(
            "renaming completed duplex journal {} to {}",
            part.display(),
            path.display()
        )
    })
}

fn part_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_else(|| "duplex-journal.json".into());
    name.push(".part");
    path.with_file_name(name)
}

fn print_timeline(
    run_id: &str,
    journal_path: &Path,
    journal: &SimulatorJournal,
    state: &SimulatorState,
) {
    println!(
        "duplex demo: {run_id} (posterior mass {:.3})",
        journal.config.posterior_mass
    );
    for event in &journal.events {
        match &event.event {
            SimulatorEventKind::EvidenceObserved { evidence } => println!(
                "[evidence] #{:02} {} {}: {}",
                event.sequence,
                modality_name(evidence.modality),
                evidence.id,
                evidence.content
            ),
            SimulatorEventKind::HypothesisProposed { hypothesis } => {
                println!(
                    "[prediction] #{:02} propose {} p={:.3}: {}",
                    event.sequence,
                    hypothesis.id.0,
                    hypothesis.probability,
                    hypothesis_text(hypothesis)
                );
            }
            SimulatorEventKind::HypothesisWithdrawn { hypothesis, reason } => println!(
                "[inference] #{:02} withdraw {}: {}",
                event.sequence, hypothesis.id.0, reason
            ),
            SimulatorEventKind::HypothesisRepaired {
                previous,
                replacement,
                reason,
            } => println!(
                "[inference] #{:02} repair {} p={:.3}->{:.3}: {}",
                event.sequence,
                previous.id.0,
                previous.probability,
                replacement.probability,
                reason
            ),
            SimulatorEventKind::BeamInferred {
                selected,
                covered_probability,
                shared_prefix,
            } => println!(
                "[inference] #{:02} posterior={:.3} selected=[{}] shared=[{}]",
                event.sequence,
                covered_probability,
                selected
                    .iter()
                    .map(|id| id.0.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                shared_prefix.join(" ")
            ),
            SimulatorEventKind::CommitFrontierAdvanced {
                from,
                to,
                committed,
            } => println!(
                "[commitment] #{:02} frontier {}->{}: {}",
                event.sequence,
                from,
                to,
                committed
                    .iter()
                    .map(|morpheme| morpheme.surface.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            SimulatorEventKind::VerificationEvaluated { result } => println!(
                "[verification] #{:02} status={} score={:.3}: {}",
                event.sequence, result.status, result.score, result.reason
            ),
        }
    }
    let predicted_suffix = state.predicted_suffix();
    let prediction = predicted_suffix
        .iter()
        .map(|morpheme| morpheme.surface.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "[commitment] final: {}",
        if state.committed.is_empty() {
            "<none>".into()
        } else {
            state.committed_text()
        }
    );
    println!(
        "[prediction] provisional: {}",
        if prediction.is_empty() {
            "<none>"
        } else {
            &prediction
        }
    );
    println!(
        "journal: {} ({} events, replay verified)",
        journal_path.display(),
        journal.events.len()
    );
}

fn hypothesis_text(hypothesis: &NormalizedCompletionHypothesis) -> String {
    hypothesis
        .morphemes
        .iter()
        .map(|morpheme| morpheme.surface.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn modality_name(modality: EvidenceModality) -> &'static str {
    match modality {
        EvidenceModality::Text => "text",
        EvidenceModality::Acoustics => "acoustics",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct TestCli {
        #[command(subcommand)]
        command: DuplexCommands,
    }

    #[test]
    fn parses_chunked_duplex_demo() {
        let cli = TestCli::try_parse_from([
            "test",
            "demo",
            "--chunk",
            "Who shot John F.",
            "--chunk",
            "Kennedy?",
            "--posterior-mass",
            "0.9",
        ])
        .unwrap();
        let DuplexCommands::Demo(command) = cli.command else {
            panic!("expected duplex demo command");
        };
        assert_eq!(command.chunks.len(), 2);
        assert_eq!(command.posterior_mass, Some(0.9));
    }
}
