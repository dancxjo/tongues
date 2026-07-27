#![cfg(unix)]

use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use speaking::{
    GrammarAnalysisStatus, GrammarBackend, GrammarBackendState, GrammarFallbackReason,
    GrammarParser, GrammarParserBackend, PartOfSpeech, TerminalPunctuation, UdPipeExecutionLimits,
    UdPipeGrammarParser, VarietyGrammarParser, VarietyId, builtin_varieties,
    grammar_backend_catalog,
};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

struct BackendFixture {
    root: PathBuf,
    model: PathBuf,
    command: PathBuf,
}

impl BackendFixture {
    fn new(script: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "tongues-grammar-backend-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let model = root.join("fixture.udpipe");
        fs::write(&model, b"bounded fixture model").unwrap();
        let command = root.join("fake-udpipe");
        let mut command_file = fs::File::create(&command).unwrap();
        command_file.write_all(script.as_bytes()).unwrap();
        command_file.sync_all().unwrap();
        drop(command_file);
        let mut permissions = fs::metadata(&command).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&command, permissions).unwrap();
        thread::sleep(Duration::from_millis(2));
        Self {
            root,
            model,
            command,
        }
    }

    fn parser(&self) -> UdPipeGrammarParser {
        UdPipeGrammarParser::with_command(
            self.model.display().to_string(),
            self.command.display().to_string(),
        )
    }

    fn parser_with_timeout(&self, timeout: Duration) -> UdPipeGrammarParser {
        self.parser().with_limits(UdPipeExecutionLimits {
            timeout,
            poll_interval: Duration::from_millis(2),
            ..UdPipeExecutionLimits::default()
        })
    }
}

impl Drop for BackendFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn words(sentence: &str) -> Vec<String> {
    sentence.split_whitespace().map(str::to_string).collect()
}

fn accepted_script() -> &'static str {
    r#"#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "fake-udpipe 1.2.3"
  exit 0
fi
cat >/dev/null
printf '1\tthey\t_\tPRON\t_\t_\t2\tnsubj\t_\t_\n2\tlook\t_\tVERB\t_\t_\t0\troot\t_\t_\n3\tat\t_\tADP\t_\t_\t4\tcase\t_\t_\n4\tus\t_\tPRON\t_\t_\t2\tobl\t_\t_\n'
"#
}

#[test]
fn accepted_backend_reports_identity_checksum_and_complete_projection() {
    let fixture = BackendFixture::new(accepted_script());
    let analysis = fixture
        .parser()
        .parse(&words("they look at us"), Some(TerminalPunctuation::Period));

    assert_eq!(
        analysis.status,
        GrammarAnalysisStatus::Complete,
        "{analysis:#?}"
    );
    assert_eq!(
        analysis.backend_report.selected,
        Some(GrammarBackend::UdPipe)
    );
    let attempt = &analysis.backend_report.attempts[0];
    assert_eq!(attempt.state, GrammarBackendState::Accepted);
    assert_eq!(attempt.exit_code, Some(0));
    assert!(attempt.projection.as_ref().unwrap().is_complete());
    let identity = attempt.identity.as_ref().unwrap();
    assert_eq!(
        identity.command.as_deref(),
        Some(fixture.command.to_str().unwrap())
    );
    assert_eq!(identity.version.as_deref(), Some("fake-udpipe 1.2.3"));
    assert_eq!(identity.model_sha256.as_ref().unwrap().len(), 64);
}

#[test]
fn token_mismatch_preserves_unmatched_input_and_partial_projection() {
    let fixture = BackendFixture::new(
        r#"#!/bin/sh
if [ "$1" = "--version" ]; then echo "fake-udpipe mismatch"; exit 0; fi
cat >/dev/null
printf '1\tthey\t_\tPRON\t_\t_\t2\tnsubj\t_\t_\n2\tlook\t_\tVERB\t_\t_\t0\troot\t_\t_\n3\tus\t_\tPRON\t_\t_\t2\tobl\t_\t_\n'
"#,
    );
    let analysis = fixture.parser().parse(&words("they look at us"), None);

    assert_eq!(
        analysis.status,
        GrammarAnalysisStatus::Partial,
        "{analysis:#?}"
    );
    assert_eq!(analysis.tokens.len(), 4);
    assert_eq!(analysis.tokens[2].text, "at");
    assert_eq!(analysis.tokens[2].pos, PartOfSpeech::Unknown);
    assert!(!analysis.backend_parses[0].accepted);
    let projection = analysis.backend_report.attempts[0]
        .projection
        .as_ref()
        .unwrap();
    assert_eq!(projection.aligned_tokens, 3);
    assert_eq!(projection.unmatched_input_indices, [2]);
    assert_eq!(
        analysis.backend_report.attempts[0].state,
        GrammarBackendState::TokenAlignmentLoss
    );
}

#[test]
fn timeout_and_cancellation_are_bounded_and_stderr_is_redacted() {
    let fixture = BackendFixture::new(
        r#"#!/bin/sh
if [ "$1" = "--version" ]; then echo "fake-udpipe slow"; exit 0; fi
echo "$4 token=super-secret" >&2
exec sleep 5
"#,
    );
    let parser = fixture.parser_with_timeout(Duration::from_millis(40));
    let timeout = parser.parse(&words("they wait"), None);
    let timeout_attempt = &timeout.backend_report.attempts[0];
    assert_eq!(timeout_attempt.state, GrammarBackendState::Timeout);
    assert!(
        !timeout_attempt
            .diagnostic
            .as_deref()
            .unwrap()
            .contains(fixture.model.to_str().unwrap())
    );
    assert!(
        timeout_attempt
            .diagnostic
            .as_deref()
            .unwrap()
            .contains("[model]")
    );
    assert!(
        !timeout_attempt
            .diagnostic
            .as_deref()
            .unwrap()
            .contains("super-secret")
    );
    assert!(
        timeout_attempt
            .diagnostic
            .as_deref()
            .unwrap()
            .contains("[redacted]")
    );

    let large_input = std::iter::repeat_n("abcdefgh", 5_000)
        .map(str::to_string)
        .collect::<Vec<_>>();
    let started = Instant::now();
    let blocked_stdin = fixture
        .parser_with_timeout(Duration::from_millis(40))
        .parse(&large_input, None);
    assert_eq!(
        blocked_stdin.backend_report.attempts[0].state,
        GrammarBackendState::Timeout
    );
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "a child that never reads stdin must still honor the deadline"
    );

    let cancelled = Arc::new(AtomicBool::new(false));
    let cancel_from_thread = cancelled.clone();
    let handle = thread::spawn(move || {
        thread::sleep(Duration::from_millis(20));
        cancel_from_thread.store(true, Ordering::Release);
    });
    let cancellation = fixture
        .parser_with_timeout(Duration::from_secs(1))
        .parse_detailed(&words("they wait"), None, Some(&cancelled));
    handle.join().unwrap();
    assert_eq!(
        cancellation.backend_report.attempts[0].state,
        GrammarBackendState::Cancelled
    );
}

#[test]
fn output_limit_and_malformed_output_are_structured_failures() {
    let output_fixture = BackendFixture::new(
        r#"#!/bin/sh
if [ "$1" = "--version" ]; then echo "fake-udpipe large"; exit 0; fi
cat >/dev/null
printf 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
"#,
    );
    let output = output_fixture
        .parser()
        .with_limits(UdPipeExecutionLimits {
            max_stdout_bytes: 16,
            ..UdPipeExecutionLimits::default()
        })
        .parse(&words("too much"), None);
    assert_eq!(
        output.backend_report.attempts[0].state,
        GrammarBackendState::OutputTooLarge,
        "{output:#?}"
    );
    let oversized_input = output_fixture
        .parser()
        .with_limits(UdPipeExecutionLimits {
            max_input_bytes: 3,
            ..UdPipeExecutionLimits::default()
        })
        .parse(&words("too much"), None);
    assert_eq!(
        oversized_input.backend_report.attempts[0].state,
        GrammarBackendState::InputTooLarge
    );

    let malformed_fixture = BackendFixture::new(
        r#"#!/bin/sh
if [ "$1" = "--version" ]; then echo "fake-udpipe malformed"; exit 0; fi
cat >/dev/null
echo 'not conllu'
"#,
    );
    let malformed = malformed_fixture.parser().parse(&words("not parsed"), None);
    assert_eq!(
        malformed.backend_report.attempts[0].state,
        GrammarBackendState::MalformedOutput
    );

    let rejected_fixture = BackendFixture::new(
        r#"#!/bin/sh
if [ "$1" = "--version" ]; then echo "fake-udpipe rejected"; exit 0; fi
cat >/dev/null
echo 'model rejected the request' >&2
exit 7
"#,
    );
    let rejected = rejected_fixture
        .parser()
        .parse(&words("rejected input"), None);
    assert_eq!(
        rejected.backend_report.attempts[0].state,
        GrammarBackendState::Rejected
    );
    assert_eq!(rejected.backend_report.attempts[0].exit_code, Some(7));
    assert!(
        rejected.backend_report.attempts[0]
            .diagnostic
            .as_deref()
            .unwrap()
            .contains("model rejected")
    );
}

#[test]
fn readiness_catalog_reports_native_without_running_a_parse() {
    let catalog = grammar_backend_catalog(VarietyId("en-US-GA".into()));
    let native = catalog
        .backends
        .iter()
        .find(|backend| backend.backend == GrammarBackend::TonguesRules)
        .unwrap();

    assert_eq!(native.state, GrammarBackendState::Ready);
    assert!(native.identity.is_some());
    assert!(
        catalog
            .backends
            .iter()
            .any(|backend| backend.backend == GrammarBackend::UdPipe)
    );
}

#[test]
fn every_builtin_variety_declares_an_honest_native_backend_state() {
    for variety in builtin_varieties() {
        let catalog = grammar_backend_catalog(variety.id.clone());
        let native = catalog
            .backends
            .iter()
            .find(|backend| backend.backend == GrammarBackend::TonguesRules)
            .unwrap();
        assert_eq!(
            native.state,
            GrammarBackendState::Ready,
            "{} did not declare a native grammar profile: {native:#?}",
            variety.id.0
        );
    }
}

#[test]
fn auto_retains_external_failure_and_explains_native_fallback() {
    let fixture = BackendFixture::new(
        r#"#!/bin/sh
if [ "$1" = "--version" ]; then echo "fake-udpipe malformed"; exit 0; fi
cat >/dev/null
echo 'not conllu'
"#,
    );
    let parser = VarietyGrammarParser::with_backend(
        VarietyId("en-US-GA".into()),
        GrammarParserBackend::Auto,
    )
    .with_udpipe_parser(fixture.parser());
    let analysis = parser.parse(&words("I saw the man"), None);

    assert_eq!(
        analysis.backend_report.requested,
        GrammarParserBackend::Auto
    );
    assert_eq!(
        analysis.backend_report.selected,
        Some(GrammarBackend::TonguesRules)
    );
    assert_eq!(analysis.backend_report.attempts.len(), 2);
    assert_eq!(
        analysis.backend_report.attempts[0].state,
        GrammarBackendState::MalformedOutput
    );
    let native_coverage = analysis.backend_report.attempts[1]
        .coverage
        .as_ref()
        .unwrap();
    assert_eq!(native_coverage.input_tokens, 4);
    assert_eq!(native_coverage.linked_tokens, 4);
    assert!(native_coverage.unsupported_token_indices.is_empty());
    assert_eq!(
        analysis.backend_report.fallback_reason,
        Some(GrammarFallbackReason::ExternalFailure)
    );
}

#[test]
fn forced_spawn_failure_is_not_a_grammatical_fragment() {
    let parser = UdPipeGrammarParser::with_command(
        Path::new("/tmp/does-not-exist.udpipe")
            .display()
            .to_string(),
        "/definitely/not/a/udpipe-command",
    );
    let analysis = parser.parse(&words("fragment"), None);

    assert_eq!(analysis.status, GrammarAnalysisStatus::Failed);
    assert!(analysis.tokens.is_empty());
    assert_eq!(
        analysis.backend_report.attempts[0].state,
        GrammarBackendState::SpawnFailure
    );
}
