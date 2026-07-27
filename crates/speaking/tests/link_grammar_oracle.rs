#![cfg(unix)]

use speaking::{
    GrammarBackend, GrammarBackendState, TerminalPunctuation, VarietyId, compare_grammar_backends,
};
#[cfg(not(feature = "link-grammar-oracle"))]
use speaking::{evaluate_curated_grammar_parity, grammar_backend_catalog};

fn words(sentence: &str) -> Vec<String> {
    sentence.split_whitespace().map(str::to_string).collect()
}

#[cfg(not(feature = "link-grammar-oracle"))]
#[test]
fn default_build_reports_oracle_as_feature_disabled() {
    let catalog = grammar_backend_catalog(VarietyId("en-US".into()));
    let oracle = catalog
        .backends
        .iter()
        .find(|backend| backend.backend == GrammarBackend::LinkGrammarOracle)
        .expect("Link Grammar readiness should be explicit");
    assert_eq!(oracle.state, GrammarBackendState::FeatureDisabled);

    let report = compare_grammar_backends(
        VarietyId("en-US".into()),
        &words("The fox jumps"),
        Some(TerminalPunctuation::Period),
        None,
    );
    assert_eq!(
        report.link_grammar.state,
        GrammarBackendState::FeatureDisabled
    );
    assert!(
        report
            .link_grammar
            .diagnostic
            .as_deref()
            .unwrap()
            .contains("--features link-grammar-oracle")
    );
    assert_eq!(report.native.backend, GrammarBackend::TonguesRules);
    assert_eq!(report.udpipe.backend, GrammarBackend::UdPipe);

    let curated = evaluate_curated_grammar_parity(VarietyId("en-US".into()), None);
    assert_eq!(curated.reports.len(), 5);
    assert!(
        curated
            .interpretation
            .contains("no backend is ground truth")
    );
}

#[cfg(feature = "link-grammar-oracle")]
mod enabled {
    use std::fs;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use speaking::{
        GrammarAnalysisStatus, LinkGrammarOracleConfig, SyntacticLinkKind, UdPipeExecutionLimits,
        run_link_grammar_oracle,
    };

    use super::*;

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    struct OracleFixture {
        root: PathBuf,
        command: PathBuf,
    }

    impl OracleFixture {
        fn new(output: &str) -> Self {
            let root = std::env::temp_dir().join(format!(
                "tongues-link-grammar-oracle-{}-{}",
                std::process::id(),
                NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&root).unwrap();
            let command = root.join("fake-link-parser");
            let script = format!(
                "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then\n  echo 'link-grammar-5.13.0 fixture'\n  exit 0\nfi\ncat >/dev/null\nprintf '%s' '{}'\n",
                output.replace('\'', "'\"'\"'")
            );
            let mut file = fs::File::create(&command).unwrap();
            file.write_all(script.as_bytes()).unwrap();
            file.sync_all().unwrap();
            drop(file);
            let mut permissions = fs::metadata(&command).unwrap().permissions();
            permissions.set_mode(0o700);
            fs::set_permissions(&command, permissions).unwrap();
            Self { root, command }
        }

        fn config(&self) -> LinkGrammarOracleConfig {
            LinkGrammarOracleConfig {
                command: self.command.display().to_string(),
                dictionary: "en".into(),
                dictionary_path: None,
                configured_varieties: vec!["en-US".into()],
                limits: UdPipeExecutionLimits {
                    timeout: Duration::from_millis(500),
                    poll_interval: Duration::from_millis(2),
                    ..UdPipeExecutionLimits::default()
                },
                max_linkages: 8,
            }
        }
    }

    impl Drop for OracleFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    const ACCEPTED_OUTPUT: &str = "\
\tUnique linkage, cost vector = (UNUSED=0 DIS= 0.25 LEN=4)\n\
LEFT-WALL Xp ----Xp----> Xp The\n\
The Ds ----D----> Ds fox\n\
quick A ----A----> A fox\n\
fox Ss ----S----> Ss jumps\n\
quick ZZ ----ZZ----> ZZ jumps\n";

    #[test]
    fn installed_process_retains_raw_unknown_labels_and_projects_known_families() {
        let fixture = OracleFixture::new(ACCEPTED_OUTPUT);
        let run = run_link_grammar_oracle(
            &fixture.config(),
            &words("The quick fox jumps"),
            Some(TerminalPunctuation::Period),
            None,
        );

        assert_eq!(run.state, GrammarBackendState::Accepted, "{run:#?}");
        let raw = run.raw.as_ref().unwrap();
        assert_eq!(raw.provenance.upstream_license, "LGPL-2.1");
        assert_eq!(
            raw.provenance.dictionary_source,
            "installed_link_grammar_data"
        );
        assert!(raw.stdout.contains("cost vector"));
        assert_eq!(raw.unknown_labels, vec!["ZZ"]);
        assert!(
            raw.parses[0]
                .links
                .iter()
                .any(|link| { link.label == "ZZ" && link.projected_kind.is_none() })
        );

        let analysis = run.analysis.as_ref().unwrap();
        assert_eq!(analysis.status, GrammarAnalysisStatus::Complete);
        assert_eq!(
            analysis.backend_parses[0].backend,
            GrammarBackend::LinkGrammarOracle
        );
        let kinds = analysis
            .best_parse()
            .unwrap()
            .links
            .iter()
            .map(|link| link.kind)
            .collect::<Vec<_>>();
        assert!(kinds.contains(&SyntacticLinkKind::Determiner));
        assert!(kinds.contains(&SyntacticLinkKind::Modifier));
        assert!(kinds.contains(&SyntacticLinkKind::Subject));
        assert!(
            analysis.backend_parses[0]
                .links
                .iter()
                .any(|link| link.label == "ZZ")
        );
    }

    #[test]
    fn one_report_contains_raw_projection_and_pairwise_metrics() {
        let fixture = OracleFixture::new(ACCEPTED_OUTPUT);
        let report = compare_grammar_backends(
            VarietyId("en-US".into()),
            &words("The quick fox jumps"),
            Some(TerminalPunctuation::Period),
            Some(fixture.config()),
        );

        assert_eq!(report.link_grammar.state, GrammarBackendState::Accepted);
        assert!(report.link_grammar.raw.is_some());
        assert!(report.link_grammar.analysis.is_some());
        assert_eq!(report.comparisons.len(), 3);
        assert!(report.reference_policy.contains("not ground truth"));
        let native_oracle = report
            .comparisons
            .iter()
            .find(|comparison| comparison.right == GrammarBackend::LinkGrammarOracle)
            .unwrap();
        assert!(native_oracle.downstream.compared_tokens > 0);
    }

    #[test]
    fn rejected_output_is_distinct_from_unavailable_or_malformed() {
        let fixture = OracleFixture::new("No complete linkages found.\n");
        let run =
            run_link_grammar_oracle(&fixture.config(), &words("unlikely fragment"), None, None);
        assert_eq!(run.state, GrammarBackendState::Rejected);
        assert!(run.raw.as_ref().unwrap().parses.is_empty());
        assert!(run.analysis.is_none());

        let malformed = OracleFixture::new("unexpected protocol output\n");
        let malformed_run = run_link_grammar_oracle(
            &malformed.config(),
            &words("unexpected protocol"),
            None,
            None,
        );
        assert_eq!(malformed_run.state, GrammarBackendState::MalformedOutput);
        assert!(
            malformed_run
                .raw
                .as_ref()
                .unwrap()
                .stdout
                .contains("unexpected")
        );
    }

    #[test]
    fn unused_words_are_reported_as_partial_not_accepted() {
        let fixture = OracleFixture::new(
            "\tLinkage 1, cost vector = (UNUSED=1 DIS= 1.00 LEN=1)\n\
             known S ----S----> S fragment\n",
        );
        let run = run_link_grammar_oracle(
            &fixture.config(),
            &words("known partial fragment"),
            None,
            None,
        );
        assert_eq!(run.state, GrammarBackendState::TokenAlignmentLoss);
        assert_eq!(
            run.analysis.as_ref().unwrap().status,
            GrammarAnalysisStatus::Partial
        );
        assert!(run.raw.as_ref().unwrap().parses[0].partial);
        assert!(!run.raw.as_ref().unwrap().parses[0].accepted);
    }
}
