//! Optional Link Grammar process oracle and backend-neutral parity reports.
//!
//! This module deliberately uses the separately installed `link-parser`
//! executable. Tongues does not link the LGPL library or redistribute its
//! dictionaries and rules.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use serde::{Deserialize, Serialize};

use crate::ids::VarietyId;
use crate::segment::TerminalPunctuation;
use crate::syntax::{
    BackendCost, GrammarAnalysis, GrammarAnalysisStatus, GrammarBackend, GrammarParser,
    GrammarParserBackend, SyntacticLinkKind, VarietyGrammarParser,
};
#[cfg(feature = "link-grammar-oracle")]
use crate::syntax::{
    BackendLink, BackendParse, GrammarParseId, GrammarParseProvenance, GrammarParseStatus,
    GrammarParseVariant, PartOfSpeech, ProsodicRole, RankedGrammarParse, SyntacticLink,
    SyntacticLinkSource, SyntaxToken,
};
use crate::syntax_backend::{
    GrammarBackendAttempt, GrammarBackendIdentity, GrammarBackendReadiness, GrammarBackendReport,
    GrammarBackendState, UdPipeExecutionLimits, command_version, model_sha256,
};
#[cfg(feature = "link-grammar-oracle")]
use crate::syntax_backend::{GrammarProjectionReport, execute_bounded_command};

pub const LINK_GRAMMAR_UPSTREAM: &str = "https://github.com/opencog/link-grammar";
pub const LINK_GRAMMAR_LICENSE: &str = "LGPL-2.1";
pub const LINK_GRAMMAR_PROTOCOL: &str = "link-parser 5.12/5.13 complete-links text protocol";
pub const DEFAULT_LINK_GRAMMAR_MAX_LINKAGES: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkGrammarOracleConfig {
    pub command: String,
    /// A Link Grammar language code (for example `en`) or dictionary path.
    pub dictionary: String,
    pub dictionary_path: Option<PathBuf>,
    pub configured_varieties: Vec<String>,
    pub limits: UdPipeExecutionLimits,
    pub max_linkages: usize,
}

impl LinkGrammarOracleConfig {
    pub fn identity(&self) -> GrammarBackendIdentity {
        GrammarBackendIdentity {
            backend: GrammarBackend::LinkGrammarOracle,
            command: Some(self.command.clone()),
            version: command_version(&self.command, self.limits),
            model_path: Some(self.dictionary.clone()),
            model_sha256: self
                .dictionary_path
                .as_deref()
                .filter(|path| path.is_file())
                .and_then(model_sha256),
            configured_varieties: self.configured_varieties.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkGrammarProvenance {
    pub adapter: String,
    pub protocol: String,
    pub upstream: String,
    pub upstream_license: String,
    pub executable: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    pub dictionary: String,
    pub dictionary_source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dictionary_sha256: Option<String>,
    pub redistribution: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkGrammarRawEnvelope {
    pub stdout: String,
    pub stderr: String,
    pub provenance: LinkGrammarProvenance,
    pub parses: Vec<LinkGrammarRawParse>,
    #[serde(default)]
    pub backend_tokens: Vec<String>,
    #[serde(default)]
    pub unknown_labels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkGrammarRawParse {
    pub rank: usize,
    pub cost: BackendCost,
    pub accepted: bool,
    pub partial: bool,
    #[serde(default)]
    pub links: Vec<LinkGrammarRawLink>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkGrammarRawLink {
    pub left_token: String,
    pub right_token: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub left_input_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub right_input_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projected_kind: Option<SyntacticLinkKind>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkGrammarOracleRun {
    pub state: GrammarBackendState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<GrammarBackendIdentity>,
    pub duration_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<LinkGrammarRawEnvelope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analysis: Option<GrammarAnalysis>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrammarParityReport {
    pub fixture: String,
    pub variety: VarietyId,
    pub reference_policy: String,
    pub native: GrammarParityBackendResult,
    pub udpipe: GrammarParityBackendResult,
    pub link_grammar: LinkGrammarOracleRun,
    #[serde(default)]
    pub comparisons: Vec<GrammarPairwiseMetrics>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrammarParityBackendResult {
    pub backend: GrammarBackend,
    pub state: GrammarBackendState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<String>,
    pub analysis: GrammarAnalysis,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrammarPairwiseMetrics {
    pub left: GrammarBackend,
    pub right: GrammarBackend,
    pub both_accepted: bool,
    pub typed_link_agreement: AgreementMetrics,
    pub attachment_agreement: AgreementMetrics,
    pub ambiguity: AmbiguityParity,
    pub downstream: DownstreamDecisionParity,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AgreementMetrics {
    pub left_count: usize,
    pub right_count: usize,
    pub shared_count: usize,
    pub precision_against_left: f32,
    pub recall_against_left: f32,
    pub f1: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AmbiguityParity {
    pub left_parse_count: usize,
    pub right_parse_count: usize,
    pub left_top_rank: Option<f32>,
    pub right_top_rank: Option<f32>,
    pub count_delta: isize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DownstreamDecisionParity {
    pub compared_tokens: usize,
    pub pronunciation_context_matches: usize,
    pub pronunciation_context_agreement: f32,
    pub prosody_role_matches: usize,
    pub prosody_role_agreement: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrammarParityEvaluationReport {
    pub interpretation: String,
    pub reports: Vec<GrammarParityReport>,
}

pub fn curated_grammar_fixtures() -> &'static [&'static str] {
    &[
        "The quick brown fox jumps.",
        "I saw the man with the telescope.",
        "She wants to sing.",
        "Alice and Bob read the book.",
        "However, the small robot did not move.",
    ]
}

pub fn link_grammar_readiness(variety: &VarietyId) -> GrammarBackendReadiness {
    #[cfg(not(feature = "link-grammar-oracle"))]
    {
        let _ = variety;
        GrammarBackendReadiness {
            backend: GrammarBackend::LinkGrammarOracle,
            state: GrammarBackendState::FeatureDisabled,
            diagnostic: Some(
                "rebuild tongues-cli with --features link-grammar-oracle to enable the optional process oracle"
                    .into(),
            ),
            identity: None,
        }
    }
    #[cfg(feature = "link-grammar-oracle")]
    {
        match discover_link_grammar_config(variety) {
            Ok(config) => {
                let identity = config.identity();
                let state = if identity.version.is_some() {
                    GrammarBackendState::Ready
                } else {
                    GrammarBackendState::UnavailableExecutable
                };
                GrammarBackendReadiness {
                    backend: GrammarBackend::LinkGrammarOracle,
                    state,
                    diagnostic: (state == GrammarBackendState::UnavailableExecutable).then(|| {
                        "link-parser was not found or did not return a bounded --version response"
                            .into()
                    }),
                    identity: Some(identity),
                }
            }
            Err(readiness) => *readiness,
        }
    }
}

#[cfg(feature = "link-grammar-oracle")]
pub fn discover_link_grammar_config(
    variety: &VarietyId,
) -> Result<LinkGrammarOracleConfig, Box<GrammarBackendReadiness>> {
    let configured_dictionary = std::env::var("TONGUES_LINK_GRAMMAR_DICTIONARY")
        .ok()
        .filter(|value| !value.trim().is_empty());
    let dictionary = match configured_dictionary {
        Some(dictionary) => dictionary,
        None if variety.0.starts_with("en-") || variety.0 == "en" => "en".into(),
        None => {
            return Err(Box::new(GrammarBackendReadiness {
                backend: GrammarBackend::LinkGrammarOracle,
                state: GrammarBackendState::UnsupportedVariety,
                diagnostic: Some(format!(
                    "set TONGUES_LINK_GRAMMAR_DICTIONARY to a language code or dictionary path for {}",
                    variety.0
                )),
                identity: None,
            }));
        }
    };
    let path = PathBuf::from(&dictionary);
    let looks_like_path = path.is_absolute()
        || dictionary.contains(std::path::MAIN_SEPARATOR)
        || dictionary.starts_with('.');
    if looks_like_path && !path.exists() {
        return Err(Box::new(GrammarBackendReadiness {
            backend: GrammarBackend::LinkGrammarOracle,
            state: GrammarBackendState::UnavailableDictionary,
            diagnostic: Some(format!(
                "configured Link Grammar dictionary path is unavailable: {}",
                path.display()
            )),
            identity: None,
        }));
    }
    Ok(LinkGrammarOracleConfig {
        command: std::env::var("TONGUES_LINK_GRAMMAR_COMMAND")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "link-parser".into()),
        dictionary_path: looks_like_path.then_some(path),
        dictionary,
        configured_varieties: vec![variety.0.clone()],
        limits: UdPipeExecutionLimits::default(),
        max_linkages: DEFAULT_LINK_GRAMMAR_MAX_LINKAGES,
    })
}

pub fn compare_grammar_backends(
    variety: VarietyId,
    words: &[String],
    terminal: Option<TerminalPunctuation>,
    link_grammar_config: Option<LinkGrammarOracleConfig>,
) -> GrammarParityReport {
    let fixture = render_fixture(words, terminal);
    let native =
        VarietyGrammarParser::with_backend(variety.clone(), GrammarParserBackend::TonguesRules)
            .parse(words, terminal);
    let udpipe = VarietyGrammarParser::with_backend(variety.clone(), GrammarParserBackend::UdPipe)
        .parse(words, terminal);
    let link_grammar = run_discovered_link_grammar(
        &variety,
        words,
        terminal,
        link_grammar_config.as_ref(),
        None,
    );

    let native_result = parity_backend_result(GrammarBackend::TonguesRules, native);
    let udpipe_result = parity_backend_result(GrammarBackend::UdPipe, udpipe);
    let mut comparisons = vec![pairwise_metrics(
        &native_result.analysis,
        &udpipe_result.analysis,
        GrammarBackend::TonguesRules,
        GrammarBackend::UdPipe,
    )];
    if let Some(link_analysis) = link_grammar.analysis.as_ref() {
        comparisons.push(pairwise_metrics(
            &native_result.analysis,
            link_analysis,
            GrammarBackend::TonguesRules,
            GrammarBackend::LinkGrammarOracle,
        ));
        comparisons.push(pairwise_metrics(
            &udpipe_result.analysis,
            link_analysis,
            GrammarBackend::UdPipe,
            GrammarBackend::LinkGrammarOracle,
        ));
    }
    GrammarParityReport {
        fixture,
        variety,
        reference_policy:
            "pairwise diagnostic parity only; Link Grammar is an oracle, not ground truth".into(),
        native: native_result,
        udpipe: udpipe_result,
        link_grammar,
        comparisons,
    }
}

pub fn evaluate_curated_grammar_parity(
    variety: VarietyId,
    link_grammar_config: Option<LinkGrammarOracleConfig>,
) -> GrammarParityEvaluationReport {
    let reports = curated_grammar_fixtures()
        .iter()
        .map(|fixture| {
            let (words, terminal) = fixture_words(fixture);
            compare_grammar_backends(
                variety.clone(),
                &words,
                terminal,
                link_grammar_config.clone(),
            )
        })
        .collect();
    GrammarParityEvaluationReport {
        interpretation:
            "bounded curated parity sample; pairwise agreement is diagnostic and no backend is ground truth"
                .into(),
        reports,
    }
}

pub(crate) fn parse_link_grammar_for_variety(
    variety: &VarietyId,
    words: &[String],
    terminal: Option<TerminalPunctuation>,
) -> GrammarAnalysis {
    let run = run_discovered_link_grammar(variety, words, terminal, None, None);
    if let Some(mut analysis) = run.analysis {
        analysis.backend_report.requested = GrammarParserBackend::LinkGrammarOracle;
        return analysis;
    }
    let diagnostic = run
        .diagnostic
        .unwrap_or_else(|| "Link Grammar oracle did not produce an analysis".into());
    let mut analysis = GrammarAnalysis::failed(terminal, diagnostic.clone());
    analysis.backend_report = GrammarBackendReport {
        requested: GrammarParserBackend::LinkGrammarOracle,
        selected: None,
        attempts: vec![GrammarBackendAttempt {
            backend: GrammarBackend::LinkGrammarOracle,
            state: run.state,
            diagnostic: Some(diagnostic),
            identity: run.identity,
            projection: None,
            coverage: None,
            duration_ms: run.duration_ms,
            exit_code: run.exit_code,
        }],
        fallback_reason: None,
    };
    analysis
}

fn run_discovered_link_grammar(
    variety: &VarietyId,
    words: &[String],
    terminal: Option<TerminalPunctuation>,
    config: Option<&LinkGrammarOracleConfig>,
    cancelled: Option<&AtomicBool>,
) -> LinkGrammarOracleRun {
    #[cfg(not(feature = "link-grammar-oracle"))]
    {
        let _ = (variety, words, terminal, config, cancelled);
        LinkGrammarOracleRun {
            state: GrammarBackendState::FeatureDisabled,
            diagnostic: Some(
                "Link Grammar oracle is feature-gated; rebuild tongues-cli with --features link-grammar-oracle"
                    .into(),
            ),
            identity: None,
            duration_ms: 0,
            exit_code: None,
            raw: None,
            analysis: None,
        }
    }
    #[cfg(feature = "link-grammar-oracle")]
    {
        let owned;
        let config = match config {
            Some(config) => config,
            None => {
                owned = match discover_link_grammar_config(variety) {
                    Ok(config) => config,
                    Err(readiness) => {
                        return LinkGrammarOracleRun {
                            state: readiness.state,
                            diagnostic: readiness.diagnostic,
                            identity: readiness.identity,
                            duration_ms: 0,
                            exit_code: None,
                            raw: None,
                            analysis: None,
                        };
                    }
                };
                &owned
            }
        };
        run_link_grammar_oracle(config, words, terminal, cancelled)
    }
}

#[cfg(feature = "link-grammar-oracle")]
pub fn run_link_grammar_oracle(
    config: &LinkGrammarOracleConfig,
    words: &[String],
    terminal: Option<TerminalPunctuation>,
    cancelled: Option<&AtomicBool>,
) -> LinkGrammarOracleRun {
    let identity = config.identity();
    if identity.version.is_none() {
        return LinkGrammarOracleRun {
            state: GrammarBackendState::UnavailableExecutable,
            diagnostic: Some(
                "link-parser was not found or did not return a bounded --version response".into(),
            ),
            identity: Some(identity),
            duration_ms: 0,
            exit_code: None,
            raw: None,
            analysis: None,
        };
    }
    let input = format!("{}\n", render_fixture(words, terminal));
    let args = vec![
        config.dictionary.clone(),
        "--quiet".into(),
        "-graphics=0".into(),
        "-links=1".into(),
        "-constituents=0".into(),
        "-verbosity=1".into(),
        format!("-limit={}", config.max_linkages),
        format!("-test=auto-next-linkage:{}", config.max_linkages),
        "-walls=1".into(),
        "-echo=0".into(),
    ];
    let redacted_paths = config
        .dictionary_path
        .as_deref()
        .into_iter()
        .collect::<Vec<_>>();
    let execution = execute_bounded_command(
        &config.command,
        &args,
        input.as_bytes(),
        &redacted_paths,
        config.limits,
        cancelled,
        "Link Grammar",
    );
    if execution.state != GrammarBackendState::Accepted {
        return LinkGrammarOracleRun {
            state: execution.state,
            diagnostic: Some(if execution.stderr.is_empty() {
                format!("Link Grammar ended in {:?}", execution.state)
            } else {
                execution.stderr
            }),
            identity: Some(identity),
            duration_ms: execution.duration_ms,
            exit_code: execution.status.and_then(|status| status.code()),
            raw: None,
            analysis: None,
        };
    }
    let stdout = match String::from_utf8(execution.stdout) {
        Ok(stdout) => stdout,
        Err(error) => {
            return LinkGrammarOracleRun {
                state: GrammarBackendState::MalformedOutput,
                diagnostic: Some(format!("Link Grammar returned non-UTF-8 output: {error}")),
                identity: Some(identity),
                duration_ms: execution.duration_ms,
                exit_code: execution.status.and_then(|status| status.code()),
                raw: None,
                analysis: None,
            };
        }
    };
    let mut raw = parse_link_grammar_output(config, words, &stdout, &execution.stderr);
    let exit_code = execution.status.and_then(|status| status.code());
    let projection = link_grammar_analysis(
        words,
        terminal,
        &raw,
        identity.clone(),
        execution.duration_ms,
        exit_code,
    );
    let (state, diagnostic, analysis) = match projection {
        Some((analysis, projection)) => {
            let state =
                if analysis.status == GrammarAnalysisStatus::Complete && projection.is_complete() {
                    GrammarBackendState::Accepted
                } else if !projection.unmatched_input_indices.is_empty()
                    || !projection.unmatched_backend_tokens.is_empty()
                {
                    GrammarBackendState::TokenAlignmentLoss
                } else {
                    GrammarBackendState::PartialProjection
                };
            let diagnostic = analysis.diagnostic.clone();
            (state, diagnostic, Some(analysis))
        }
        None => {
            let rejected =
                stdout.contains("No complete linkages") || stdout.contains("No linkages found");
            let diagnostic = if rejected {
                "Link Grammar rejected the fixture with no linkage".into()
            } else {
                "Link Grammar output contained no parseable complete-link rows".into()
            };
            (
                if rejected {
                    GrammarBackendState::Rejected
                } else {
                    GrammarBackendState::MalformedOutput
                },
                Some(diagnostic),
                None,
            )
        }
    };
    raw.unknown_labels.sort();
    raw.unknown_labels.dedup();
    LinkGrammarOracleRun {
        state,
        diagnostic,
        identity: Some(identity),
        duration_ms: execution.duration_ms,
        exit_code,
        raw: Some(raw),
        analysis,
    }
}

#[cfg(feature = "link-grammar-oracle")]
fn parse_link_grammar_output(
    config: &LinkGrammarOracleConfig,
    words: &[String],
    stdout: &str,
    stderr: &str,
) -> LinkGrammarRawEnvelope {
    let mut parses = Vec::<LinkGrammarRawParse>::new();
    for line in stdout.lines() {
        if line.contains("cost vector =") {
            parses.push(LinkGrammarRawParse {
                rank: parses.len(),
                cost: parse_cost_vector(line),
                accepted: true,
                partial: false,
                links: Vec::new(),
            });
            continue;
        }
        let Some((left_token, label, right_token)) = parse_complete_link_row(line) else {
            continue;
        };
        if parses.is_empty() {
            parses.push(LinkGrammarRawParse {
                rank: 0,
                cost: BackendCost {
                    unused: None,
                    disjunct: None,
                    length: None,
                },
                accepted: true,
                partial: false,
                links: Vec::new(),
            });
        }
        let left_input_index = match_input_token(words, &left_token);
        let right_input_index = match_input_token(words, &right_token);
        let projected_kind = link_grammar_link_kind(&label);
        parses
            .last_mut()
            .expect("parse was created")
            .links
            .push(LinkGrammarRawLink {
                left_token,
                right_token,
                label,
                left_input_index,
                right_input_index,
                projected_kind,
            });
    }
    for parse in &mut parses {
        parse.partial = parse.cost.unused.is_some_and(|unused| unused > 0.0)
            || parse.links.iter().any(|link| {
                !is_wall(&link.left_token)
                    && !is_wall(&link.right_token)
                    && (link.left_input_index.is_none() || link.right_input_index.is_none())
            });
        parse.accepted &= !parse.partial;
    }
    let mut backend_tokens = parses
        .iter()
        .flat_map(|parse| &parse.links)
        .flat_map(|link| [&link.left_token, &link.right_token])
        .filter(|token| !is_wall(token))
        .cloned()
        .collect::<Vec<_>>();
    backend_tokens.sort_by_key(|token| match_input_token(words, token).unwrap_or(usize::MAX));
    backend_tokens.dedup_by(|left, right| {
        normalize_link_grammar_token(left) == normalize_link_grammar_token(right)
    });
    let unknown_labels = parses
        .iter()
        .flat_map(|parse| &parse.links)
        .filter(|link| {
            link.projected_kind.is_none()
                && !is_wall(&link.left_token)
                && !is_wall(&link.right_token)
        })
        .map(|link| link.label.clone())
        .collect::<Vec<_>>();
    let identity = config.identity();
    LinkGrammarRawEnvelope {
        stdout: stdout.into(),
        stderr: stderr.into(),
        provenance: LinkGrammarProvenance {
            adapter: "separate_process".into(),
            protocol: LINK_GRAMMAR_PROTOCOL.into(),
            upstream: LINK_GRAMMAR_UPSTREAM.into(),
            upstream_license: LINK_GRAMMAR_LICENSE.into(),
            executable: config.command.clone(),
            version: identity.version,
            dictionary: config.dictionary.clone(),
            dictionary_source: if config.dictionary_path.is_some() {
                "operator_configured_path"
            } else {
                "installed_link_grammar_data"
            }
            .into(),
            dictionary_sha256: identity.model_sha256,
            redistribution: "not redistributed by Tongues".into(),
        },
        parses,
        backend_tokens,
        unknown_labels,
    }
}

#[cfg(feature = "link-grammar-oracle")]
fn link_grammar_analysis(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
    raw: &LinkGrammarRawEnvelope,
    identity: GrammarBackendIdentity,
    duration_ms: u64,
    exit_code: Option<i32>,
) -> Option<(GrammarAnalysis, GrammarProjectionReport)> {
    let first = raw.parses.first()?;
    let mut ranked_parses = Vec::new();
    let mut backend_parses = Vec::new();
    let mut dropped_backend_links = 0;
    for parse in &raw.parses {
        let mut projected = Vec::new();
        let mut backend_links = Vec::new();
        for link in &parse.links {
            let (Some(left), Some(right)) = (link.left_input_index, link.right_input_index) else {
                if !is_wall(&link.left_token) && !is_wall(&link.right_token) {
                    dropped_backend_links += 1;
                }
                continue;
            };
            if left == right {
                dropped_backend_links += 1;
                continue;
            }
            backend_links.push(BackendLink {
                left: left.min(right),
                right: left.max(right),
                label: link.label.clone(),
            });
            let Some(kind) = link.projected_kind else {
                continue;
            };
            let typed = SyntacticLink {
                left: left.min(right),
                right: left.max(right),
                kind,
                confidence: if parse.accepted { 0.8 } else { 0.55 },
                source: SyntacticLinkSource::LinkGrammarOracleProjection,
            };
            if !projected.iter().any(|existing: &SyntacticLink| {
                existing.left == typed.left
                    && existing.right == typed.right
                    && existing.kind == typed.kind
            }) {
                projected.push(typed);
            }
        }
        projected.sort_by_key(|link| (link.left, link.right, link.kind as u8));
        backend_links.sort_by(|left, right| {
            (left.left, left.right, &left.label).cmp(&(right.left, right.right, &right.label))
        });
        let confidence = if projected.is_empty() {
            0.0
        } else {
            projected.iter().map(|link| link.confidence).sum::<f32>() / projected.len() as f32
        };
        let linked = projected
            .iter()
            .flat_map(|link| [link.left, link.right])
            .collect::<BTreeSet<_>>()
            .len();
        let coverage = if words.is_empty() {
            0.0
        } else {
            linked as f32 / words.len() as f32
        };
        let status = if parse.accepted && !projected.is_empty() {
            GrammarParseStatus::Complete
        } else {
            GrammarParseStatus::Partial
        };
        ranked_parses.push(RankedGrammarParse {
            id: GrammarParseId(format!("link-grammar-oracle-{}-primary", parse.rank)),
            links: projected,
            rank: (0.65 * confidence
                + 0.25 * coverage
                + 0.10 * f32::from(status == GrammarParseStatus::Complete))
            .clamp(0.0, 1.0),
            confidence,
            status,
            provenance: GrammarParseProvenance {
                backend: GrammarBackend::LinkGrammarOracle,
                backend_parse_index: parse.rank,
                variant: GrammarParseVariant::BackendPrimary,
            },
        });
        backend_parses.push(BackendParse {
            links: backend_links,
            cost: Some(parse.cost),
            accepted: parse.accepted,
            backend: GrammarBackend::LinkGrammarOracle,
        });
    }
    ranked_parses.sort_by(|left, right| {
        right
            .rank
            .total_cmp(&left.rank)
            .then_with(|| left.id.cmp(&right.id))
    });
    let best = ranked_parses.first()?;
    let tokens = words
        .iter()
        .enumerate()
        .map(|(word_index, text)| {
            let mut syntactic_links = best
                .links
                .iter()
                .filter(|link| link.left == word_index || link.right == word_index)
                .map(|link| link.kind)
                .collect::<Vec<_>>();
            syntactic_links.sort_unstable_by_key(|kind| *kind as u8);
            syntactic_links.dedup();
            SyntaxToken {
                word_index,
                text: text.clone(),
                pos: PartOfSpeech::Unknown,
                prosodic_role: link_grammar_prosodic_role(&syntactic_links),
                syntactic_links,
            }
        })
        .collect::<Vec<_>>();
    let matched = first
        .links
        .iter()
        .flat_map(|link| [link.left_input_index, link.right_input_index])
        .flatten()
        .collect::<BTreeSet<_>>();
    let unmatched_input_indices = (0..words.len())
        .filter(|index| !matched.contains(index))
        .collect::<Vec<_>>();
    let unmatched_backend_tokens = raw
        .backend_tokens
        .iter()
        .enumerate()
        .filter(|(_, token)| match_input_token(words, token).is_none())
        .map(|(id, form)| crate::syntax_backend::BackendTokenIdentity {
            id,
            form: form.clone(),
        })
        .collect::<Vec<_>>();
    let projection = GrammarProjectionReport {
        input_tokens: words.len(),
        backend_tokens: raw.backend_tokens.len(),
        aligned_tokens: words.len().saturating_sub(unmatched_input_indices.len()),
        unmatched_input_indices,
        unmatched_backend_tokens,
        dropped_backend_links,
    };
    let complete =
        first.accepted && projection.is_complete() && best.status == GrammarParseStatus::Complete;
    let diagnostic = (!complete).then(|| {
        format!(
            "Link Grammar projection is partial: {} unknown labels, {} dropped links",
            raw.unknown_labels.len(),
            projection.dropped_backend_links
        )
    });
    let state = if complete {
        GrammarAnalysisStatus::Complete
    } else {
        GrammarAnalysisStatus::Partial
    };
    let analysis = GrammarAnalysis {
        tokens,
        ranked_parses,
        backend_parses,
        terminal,
        status: state,
        diagnostic: diagnostic.clone(),
        backend_report: GrammarBackendReport {
            requested: GrammarParserBackend::LinkGrammarOracle,
            selected: Some(GrammarBackend::LinkGrammarOracle),
            attempts: vec![GrammarBackendAttempt {
                backend: GrammarBackend::LinkGrammarOracle,
                state: if complete {
                    GrammarBackendState::Accepted
                } else {
                    GrammarBackendState::PartialProjection
                },
                diagnostic,
                identity: Some(identity),
                projection: Some(projection.clone()),
                coverage: None,
                duration_ms,
                exit_code,
            }],
            fallback_reason: None,
        },
    };
    Some((analysis, projection))
}

fn parity_backend_result(
    backend: GrammarBackend,
    analysis: GrammarAnalysis,
) -> GrammarParityBackendResult {
    let state = analysis
        .backend_report
        .attempts
        .last()
        .map(|attempt| attempt.state)
        .unwrap_or_else(|| match analysis.status {
            GrammarAnalysisStatus::Complete => GrammarBackendState::Accepted,
            GrammarAnalysisStatus::Partial => GrammarBackendState::PartialProjection,
            GrammarAnalysisStatus::Failed => GrammarBackendState::Rejected,
        });
    GrammarParityBackendResult {
        backend,
        state,
        diagnostic: analysis.diagnostic.clone(),
        analysis,
    }
}

fn pairwise_metrics(
    left: &GrammarAnalysis,
    right: &GrammarAnalysis,
    left_backend: GrammarBackend,
    right_backend: GrammarBackend,
) -> GrammarPairwiseMetrics {
    let left_typed = typed_link_set(left);
    let right_typed = typed_link_set(right);
    let left_attachments = attachment_set(left);
    let right_attachments = attachment_set(right);
    GrammarPairwiseMetrics {
        left: left_backend,
        right: right_backend,
        both_accepted: left.status == GrammarAnalysisStatus::Complete
            && right.status == GrammarAnalysisStatus::Complete,
        typed_link_agreement: agreement(&left_typed, &right_typed),
        attachment_agreement: agreement(&left_attachments, &right_attachments),
        ambiguity: AmbiguityParity {
            left_parse_count: left.ranked_parses.len(),
            right_parse_count: right.ranked_parses.len(),
            left_top_rank: left.ranked_parses.first().map(|parse| parse.rank),
            right_top_rank: right.ranked_parses.first().map(|parse| parse.rank),
            count_delta: right.ranked_parses.len() as isize - left.ranked_parses.len() as isize,
        },
        downstream: downstream_parity(left, right),
    }
}

fn typed_link_set(analysis: &GrammarAnalysis) -> BTreeSet<(usize, usize, SyntacticLinkKind)> {
    analysis
        .best_parse()
        .into_iter()
        .flat_map(|parse| &parse.links)
        .map(|link| (link.left, link.right, link.kind))
        .collect()
}

fn attachment_set(analysis: &GrammarAnalysis) -> BTreeSet<(usize, usize)> {
    analysis
        .best_parse()
        .into_iter()
        .flat_map(|parse| &parse.links)
        .map(|link| (link.left, link.right))
        .collect()
}

fn agreement<T: Ord>(left: &BTreeSet<T>, right: &BTreeSet<T>) -> AgreementMetrics {
    let shared = left.intersection(right).count();
    let precision = ratio(shared, right.len());
    let recall = ratio(shared, left.len());
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    AgreementMetrics {
        left_count: left.len(),
        right_count: right.len(),
        shared_count: shared,
        precision_against_left: precision,
        recall_against_left: recall,
        f1,
    }
}

fn downstream_parity(left: &GrammarAnalysis, right: &GrammarAnalysis) -> DownstreamDecisionParity {
    let compared = left.tokens.len().min(right.tokens.len());
    let pronunciation_context_matches = (0..compared)
        .filter(|index| {
            left.tokens[*index].pos == right.tokens[*index].pos
                && left.tokens[*index].syntactic_links == right.tokens[*index].syntactic_links
        })
        .count();
    let prosody_role_matches = (0..compared)
        .filter(|index| left.tokens[*index].prosodic_role == right.tokens[*index].prosodic_role)
        .count();
    DownstreamDecisionParity {
        compared_tokens: compared,
        pronunciation_context_matches,
        pronunciation_context_agreement: ratio(pronunciation_context_matches, compared),
        prosody_role_matches,
        prosody_role_agreement: ratio(prosody_role_matches, compared),
    }
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        f32::from(numerator == 0)
    } else {
        numerator as f32 / denominator as f32
    }
}

#[cfg(feature = "link-grammar-oracle")]
fn parse_cost_vector(line: &str) -> BackendCost {
    BackendCost {
        unused: parse_cost_field(line, "UNUSED="),
        disjunct: parse_cost_field(line, "DIS="),
        length: parse_cost_field(line, "LEN="),
    }
}

#[cfg(feature = "link-grammar-oracle")]
fn parse_cost_field(line: &str, marker: &str) -> Option<f32> {
    line.split_once(marker)?
        .1
        .trim_start()
        .split(|character: char| character.is_whitespace() || character == ')')
        .next()?
        .parse()
        .ok()
}

#[cfg(feature = "link-grammar-oracle")]
fn parse_complete_link_row(line: &str) -> Option<(String, String, String)> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    let arrow_index = fields.iter().position(|field| {
        field.contains("---") && field.chars().any(|character| character.is_alphabetic())
    })?;
    if arrow_index < 2 || arrow_index + 2 >= fields.len() {
        return None;
    }
    let label = fields[arrow_index]
        .trim_matches(|character| matches!(character, '-' | '<' | '>'))
        .to_string();
    (!label.is_empty()).then(|| {
        (
            fields[arrow_index - 2].to_string(),
            label,
            fields[fields.len() - 1].to_string(),
        )
    })
}

#[cfg(feature = "link-grammar-oracle")]
fn match_input_token(words: &[String], backend: &str) -> Option<usize> {
    let backend = normalize_link_grammar_token(backend);
    let matches = words
        .iter()
        .enumerate()
        .filter(|(_, word)| normalize_link_grammar_token(word) == backend)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if matches.len() == 1 {
        matches.first().copied()
    } else {
        None
    }
}

#[cfg(feature = "link-grammar-oracle")]
fn normalize_link_grammar_token(token: &str) -> String {
    token
        .trim_matches(|character: char| !character.is_alphanumeric() && character != '\'')
        .split('.')
        .next()
        .unwrap_or(token)
        .trim_end_matches([']', '!', '?', '~', '&'])
        .to_lowercase()
}

#[cfg(feature = "link-grammar-oracle")]
fn is_wall(token: &str) -> bool {
    matches!(token, "LEFT-WALL" | "RIGHT-WALL")
}

#[cfg(feature = "link-grammar-oracle")]
fn link_grammar_link_kind(label: &str) -> Option<SyntacticLinkKind> {
    let label = label.to_ascii_uppercase();
    if label.starts_with("SI") || label.starts_with("PP") || label.starts_with("PG") {
        Some(SyntacticLinkKind::Auxiliary)
    } else if label.starts_with('S') {
        Some(SyntacticLinkKind::Subject)
    } else if label.starts_with('O') {
        Some(SyntacticLinkKind::Object)
    } else if label.starts_with("TO") || label.starts_with('I') {
        Some(SyntacticLinkKind::InfinitivalMarker)
    } else if label.starts_with('D') {
        Some(SyntacticLinkKind::Determiner)
    } else if label.starts_with('J') || label.starts_with("IN") {
        Some(SyntacticLinkKind::Preposition)
    } else if label.starts_with("CO") {
        Some(SyntacticLinkKind::Coordination)
    } else if label.starts_with('C') || label.starts_with('B') {
        Some(SyntacticLinkKind::Complement)
    } else if label.starts_with('A') || label.starts_with('M') || label.starts_with('R') {
        Some(SyntacticLinkKind::Modifier)
    } else {
        None
    }
}

#[cfg(feature = "link-grammar-oracle")]
fn link_grammar_prosodic_role(links: &[SyntacticLinkKind]) -> ProsodicRole {
    if links.iter().any(|kind| {
        matches!(
            kind,
            SyntacticLinkKind::Object | SyntacticLinkKind::Complement
        )
    }) {
        ProsodicRole::Focus
    } else if !links.is_empty()
        && links.iter().all(|kind| {
            matches!(
                kind,
                SyntacticLinkKind::Determiner
                    | SyntacticLinkKind::Auxiliary
                    | SyntacticLinkKind::Preposition
                    | SyntacticLinkKind::Coordination
                    | SyntacticLinkKind::InfinitivalMarker
            )
        })
    {
        ProsodicRole::FunctionWeak
    } else {
        ProsodicRole::Content
    }
}

fn fixture_words(fixture: &str) -> (Vec<String>, Option<TerminalPunctuation>) {
    let terminal = terminal_from_text(fixture);
    let words = fixture
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|character: char| {
                matches!(character, '.' | '?' | '!' | ',' | ';' | ':')
            })
            .to_string()
        })
        .filter(|word| !word.is_empty())
        .collect();
    (words, terminal)
}

fn terminal_from_text(text: &str) -> Option<TerminalPunctuation> {
    match text.trim_end().chars().last() {
        Some('?') => Some(TerminalPunctuation::Question),
        Some('!') => Some(TerminalPunctuation::Exclamation),
        Some('.') => Some(TerminalPunctuation::Period),
        _ => None,
    }
}

fn render_fixture(words: &[String], terminal: Option<TerminalPunctuation>) -> String {
    let mut text = words
        .iter()
        .map(|word| word.replace(['\r', '\n'], " "))
        .collect::<Vec<_>>()
        .join(" ");
    let punctuation = match terminal {
        Some(TerminalPunctuation::Question) => "?",
        Some(TerminalPunctuation::Exclamation) => "!",
        Some(TerminalPunctuation::Period) => ".",
        _ => "",
    };
    text.push_str(punctuation);
    text
}
