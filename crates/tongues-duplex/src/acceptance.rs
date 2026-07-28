//! Reproducible multilingual acceptance corpus and evaluation report.
//!
//! The harness deliberately calls the production native grammar parser,
//! phonemicizer/claim builder, and Duplex replay path. Optional learned and
//! external backends are reported separately and may skip with a concrete
//! reason; they never become a prerequisite for the bounded CI profile.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use speaking::{
    phonemicizer_for_variety, syllables_to_ipa, GrammarAnalysisStatus, GrammarBackendState,
    GrammarParser, GrammarParserBackend, PhonemicizeRequest, PhonemicizeStyle, Spec, Stress,
    SyntacticLinkKind, VarietyGrammarParser, VarietyId,
};

use crate::{run_fixture, DuplexFixtureSuite, LearnedDuplexModel, SimulatorEventKind};

pub const INTERPRETATION_ACCEPTANCE_SCHEMA_VERSION: u32 = 1;
pub const INTERPRETATION_ACCEPTANCE_REPORT_VERSION: u32 = 1;
pub const INTERPRETATION_EPIC_CHILDREN: &[u32] = &[176, 178, 177, 179, 185, 183, 182, 181, 184];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceptanceProfile {
    Ci,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypologyRole {
    Romance,
    CaseRichFreeWordOrder,
    LowerResource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceContribution {
    NativeRule,
    Learned,
    ExternalBackend,
    Combined,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptanceSource {
    pub id: String,
    pub name: String,
    pub license: String,
    pub provenance: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    pub evaluation_only: bool,
    pub training_exclusion_key: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptanceSplitPolicy {
    pub purpose: String,
    pub grouping_key: String,
    #[serde(default)]
    pub training_exclusion_keys: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamingExpectation {
    pub fixture_id: String,
    pub final_committed_text: String,
    #[serde(default)]
    pub requires_repair: bool,
    #[serde(default)]
    pub requires_withdrawal: bool,
    #[serde(default)]
    pub requires_abstention: bool,
    #[serde(default)]
    pub stable_prefix_retained: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AcceptanceExpectation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar_status: Option<GrammarAnalysisStatus>,
    #[serde(default)]
    pub required_links: Vec<SyntacticLinkKind>,
    #[serde(default)]
    pub required_parse_variants: Vec<String>,
    #[serde(default)]
    pub min_ranked_parses: usize,
    #[serde(default)]
    pub min_claims: usize,
    #[serde(default)]
    pub min_resolutions: usize,
    #[serde(default)]
    pub min_conflicts: usize,
    #[serde(default)]
    pub min_lifecycle_events: usize,
    #[serde(default)]
    pub min_pronunciation_alternatives: usize,
    #[serde(default)]
    pub required_candidate_phonemes: Vec<Vec<String>>,
    #[serde(default)]
    pub required_selected_phonemes: Vec<Vec<String>>,
    #[serde(default)]
    pub min_boundaries: usize,
    #[serde(default)]
    pub min_primary_stress: usize,
    #[serde(default)]
    pub explicit_unsupported: Vec<String>,
    #[serde(default)]
    pub required_provenance_sources: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streaming: Option<StreamingExpectation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterpretationAcceptanceCase {
    pub id: String,
    pub description: String,
    pub source_id: String,
    pub leakage_group: String,
    pub variety: VarietyId,
    pub text: String,
    #[serde(default)]
    pub style: PhonemicizeStyle,
    #[serde(default)]
    pub ci: bool,
    #[serde(default)]
    pub child_issues: Vec<u32>,
    #[serde(default)]
    pub phenomena: Vec<String>,
    #[serde(default)]
    pub typology_roles: Vec<TypologyRole>,
    #[serde(default)]
    pub contributions: Vec<EvidenceContribution>,
    pub expected: AcceptanceExpectation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendProbe {
    pub id: String,
    pub description: String,
    pub variety: VarietyId,
    pub text: String,
    pub backend: GrammarParserBackend,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub injected_state: Option<GrammarBackendState>,
    #[serde(default)]
    pub optional: bool,
    #[serde(default)]
    pub allowed_dispositions: Vec<BackendDisposition>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterpretationAcceptanceCorpus {
    pub schema_version: u32,
    pub corpus_id: String,
    pub description: String,
    pub split_policy: AcceptanceSplitPolicy,
    pub sources: Vec<AcceptanceSource>,
    pub cases: Vec<InterpretationAcceptanceCase>,
    #[serde(default)]
    pub backend_probes: Vec<BackendProbe>,
}

impl InterpretationAcceptanceCorpus {
    pub fn validate(&self, duplex_fixtures: &DuplexFixtureSuite) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == INTERPRETATION_ACCEPTANCE_SCHEMA_VERSION,
            "acceptance schema {} is unsupported; expected {}",
            self.schema_version,
            INTERPRETATION_ACCEPTANCE_SCHEMA_VERSION
        );
        anyhow::ensure!(
            self.split_policy.purpose == "evaluation_only",
            "acceptance corpus must be evaluation_only"
        );
        let sources = self
            .sources
            .iter()
            .map(|source| (source.id.as_str(), source))
            .collect::<BTreeMap<_, _>>();
        anyhow::ensure!(
            sources.len() == self.sources.len(),
            "acceptance source IDs must be unique"
        );
        for source in &self.sources {
            anyhow::ensure!(
                source.evaluation_only,
                "source '{}' is not reserved for evaluation",
                source.id
            );
            anyhow::ensure!(
                !source.license.trim().is_empty()
                    && !source.provenance.trim().is_empty()
                    && !source.training_exclusion_key.trim().is_empty(),
                "source '{}' is missing license/provenance/leakage metadata",
                source.id
            );
            anyhow::ensure!(
                self.split_policy
                    .training_exclusion_keys
                    .contains(&source.training_exclusion_key),
                "source '{}' exclusion key is absent from split policy",
                source.id
            );
        }

        let mut case_ids = BTreeSet::new();
        let mut child_issues = BTreeSet::new();
        let mut typology = BTreeSet::new();
        for case in &self.cases {
            anyhow::ensure!(
                case_ids.insert(case.id.as_str()),
                "duplicate acceptance case '{}'",
                case.id
            );
            anyhow::ensure!(
                sources.contains_key(case.source_id.as_str()),
                "case '{}' references unknown source '{}'",
                case.id,
                case.source_id
            );
            anyhow::ensure!(
                !case.leakage_group.trim().is_empty(),
                "case '{}' has no leakage group",
                case.id
            );
            child_issues.extend(case.child_issues.iter().copied());
            typology.extend(case.typology_roles.iter().copied());
            if !case.expected.explicit_unsupported.is_empty() {
                anyhow::ensure!(
                    case.expected.required_links.is_empty(),
                    "case '{}' must not force link categories onto explicitly unsupported syntax",
                    case.id
                );
            }
            if let Some(streaming) = &case.expected.streaming {
                anyhow::ensure!(
                    duplex_fixtures.fixture(&streaming.fixture_id).is_some(),
                    "case '{}' references missing Duplex fixture '{}'",
                    case.id,
                    streaming.fixture_id
                );
            }
        }
        for child in INTERPRETATION_EPIC_CHILDREN {
            anyhow::ensure!(
                child_issues.contains(child),
                "no end-to-end acceptance case covers child issue #{child}"
            );
        }
        for required in [
            TypologyRole::Romance,
            TypologyRole::CaseRichFreeWordOrder,
            TypologyRole::LowerResource,
        ] {
            anyhow::ensure!(
                typology.contains(&required),
                "acceptance corpus is missing {required:?} coverage"
            );
        }
        anyhow::ensure!(
            self.cases.iter().any(|case| case.ci),
            "acceptance corpus has no bounded CI cases"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptanceDiff {
    pub path: String,
    pub expected: String,
    pub actual: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PronunciationAlternativeActual {
    pub token: String,
    pub selected_candidate_id: Option<String>,
    pub candidates: Vec<Vec<String>>,
    pub alternative_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamingActual {
    pub fixture_id: String,
    pub final_committed_text: String,
    pub repairs: usize,
    pub withdrawals: usize,
    pub abstentions: usize,
    pub frontier_advances: usize,
    pub stable_prefix_retained: bool,
    pub replay_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcceptanceCaseActual {
    pub grammar_status: GrammarAnalysisStatus,
    pub parse_ids: Vec<String>,
    pub parse_ranks: Vec<f32>,
    pub parse_variants: Vec<String>,
    pub link_kinds: Vec<SyntacticLinkKind>,
    pub backend_attempts: Vec<speaking::GrammarBackendAttempt>,
    pub claim_ids: Vec<String>,
    pub resolution_ids: Vec<String>,
    pub conflict_edges: usize,
    pub lifecycle_events: usize,
    pub pronunciation_alternatives: Vec<PronunciationAlternativeActual>,
    #[serde(default)]
    pub selected_phonemes: Vec<Vec<String>>,
    #[serde(default)]
    pub selected_confidences: Vec<f64>,
    pub broad_ipa: String,
    pub phoneme_count: usize,
    pub phone_count: usize,
    pub boundary_count: usize,
    pub primary_stress_count: usize,
    pub provenance_sources: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streaming: Option<StreamingActual>,
    pub latency_micros: u128,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcceptanceCaseResult {
    pub id: String,
    pub variety: VarietyId,
    pub phenomena: Vec<String>,
    pub child_issues: Vec<u32>,
    pub source_id: String,
    pub leakage_group: String,
    pub passed: bool,
    pub expected: AcceptanceExpectation,
    pub actual: AcceptanceCaseActual,
    #[serde(default)]
    pub diffs: Vec<AcceptanceDiff>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendDisposition {
    Accepted,
    Conservative,
    Fallback,
    Abstain,
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendProbeResult {
    pub id: String,
    pub backend: GrammarParserBackend,
    pub observed_state: GrammarBackendState,
    pub disposition: BackendDisposition,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skip_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AcceptanceMetrics {
    pub parse_link_agreement: f64,
    pub ambiguity_recall: f64,
    pub top_k_lexical_accuracy: f64,
    pub homophone_heteronym_accuracy: f64,
    pub calibration_brier: Option<f64>,
    pub repair_precision: f64,
    pub repair_recall: f64,
    pub pronunciation_selection_accuracy: f64,
    pub boundary_stress_accuracy: f64,
    pub latency_mean_micros: f64,
    pub latency_p95_micros: u128,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ContributionReport {
    pub attempted: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterpretationAcceptanceReport {
    pub schema_version: u32,
    pub corpus_id: String,
    pub corpus_sha256: String,
    pub profile: AcceptanceProfile,
    pub passed: bool,
    pub selected_cases: usize,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub metrics: AcceptanceMetrics,
    pub contributions: BTreeMap<EvidenceContribution, ContributionReport>,
    pub cases: Vec<AcceptanceCaseResult>,
    pub backend_probes: Vec<BackendProbeResult>,
    pub leakage_policy_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcceptanceProgress {
    CaseStarted {
        index: usize,
        total: usize,
        id: String,
    },
    CaseCompleted {
        index: usize,
        total: usize,
        id: String,
        passed: bool,
    },
    BackendProbe {
        index: usize,
        total: usize,
        id: String,
    },
}

pub fn load_interpretation_acceptance_corpus(
    path: &Path,
) -> Result<InterpretationAcceptanceCorpus> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading acceptance corpus {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing acceptance corpus {}", path.display()))
}

pub fn evaluate_interpretation_acceptance(
    corpus: &InterpretationAcceptanceCorpus,
    duplex_fixtures: &DuplexFixtureSuite,
    profile: AcceptanceProfile,
    learned_model: Option<&mut LearnedDuplexModel>,
) -> Result<InterpretationAcceptanceReport> {
    evaluate_interpretation_acceptance_with_progress(
        corpus,
        duplex_fixtures,
        profile,
        learned_model,
        |_| {},
    )
}

pub fn evaluate_interpretation_acceptance_with_progress(
    corpus: &InterpretationAcceptanceCorpus,
    duplex_fixtures: &DuplexFixtureSuite,
    profile: AcceptanceProfile,
    mut learned_model: Option<&mut LearnedDuplexModel>,
    mut progress: impl FnMut(AcceptanceProgress),
) -> Result<InterpretationAcceptanceReport> {
    corpus.validate(duplex_fixtures)?;
    let selected = corpus
        .cases
        .iter()
        .filter(|case| profile == AcceptanceProfile::Full || case.ci)
        .collect::<Vec<_>>();
    let total = selected.len();
    let mut results = Vec::with_capacity(total);
    let mut learned_attempts = 0usize;
    let mut learned_passed = 0usize;
    for (offset, case) in selected.into_iter().enumerate() {
        let index = offset + 1;
        progress(AcceptanceProgress::CaseStarted {
            index,
            total,
            id: case.id.clone(),
        });
        let result = evaluate_case(case, duplex_fixtures)?;
        if case.contributions.contains(&EvidenceContribution::Learned)
            && let Some(model) = learned_model.as_deref_mut()
        {
            learned_attempts += 1;
            let unstable = case
                .text
                .split_whitespace()
                .map(str::to_string)
                .collect::<Vec<_>>();
            let inference = model.infer_uncached(&[], &unstable);
            if inference.decision_confidence.is_finite()
                && inference
                    .candidates
                    .iter()
                    .all(|candidate| candidate.probability.is_finite())
            {
                learned_passed += 1;
            }
        }
        progress(AcceptanceProgress::CaseCompleted {
            index,
            total,
            id: case.id.clone(),
            passed: result.passed,
        });
        results.push(result);
    }

    let mut backend_probes = Vec::with_capacity(corpus.backend_probes.len());
    for (offset, probe) in corpus.backend_probes.iter().enumerate() {
        progress(AcceptanceProgress::BackendProbe {
            index: offset + 1,
            total: corpus.backend_probes.len(),
            id: probe.id.clone(),
        });
        backend_probes.push(evaluate_backend_probe(probe));
    }

    let metrics = acceptance_metrics(&results);
    let passed_cases = results.iter().filter(|result| result.passed).count();
    let failed_cases = results.len() - passed_cases;
    let mut contributions = BTreeMap::new();
    contributions.insert(
        EvidenceContribution::NativeRule,
        ContributionReport {
            attempted: results.len(),
            passed: passed_cases,
            failed: failed_cases,
            skipped: 0,
            notes: vec!["Production Tongues native grammar parser.".into()],
        },
    );
    contributions.insert(
        EvidenceContribution::Combined,
        ContributionReport {
            attempted: results.len(),
            passed: passed_cases,
            failed: failed_cases,
            skipped: 0,
            notes: vec![
                "Production phonemicizer, claim ledger, grammar, prosody, and Duplex replay."
                    .into(),
            ],
        },
    );
    let external_attempted = backend_probes
        .iter()
        .filter(|probe| {
            matches!(
                probe.backend,
                GrammarParserBackend::UdPipe | GrammarParserBackend::LinkGrammarOracle
            )
        })
        .count();
    let external_skipped = backend_probes
        .iter()
        .filter(|probe| {
            matches!(
                probe.backend,
                GrammarParserBackend::UdPipe | GrammarParserBackend::LinkGrammarOracle
            ) && probe.disposition == BackendDisposition::Skipped
        })
        .count();
    let external_failed = backend_probes
        .iter()
        .filter(|probe| {
            matches!(
                probe.backend,
                GrammarParserBackend::UdPipe | GrammarParserBackend::LinkGrammarOracle
            ) && !probe.passed
        })
        .count();
    contributions.insert(
        EvidenceContribution::ExternalBackend,
        ContributionReport {
            attempted: external_attempted - external_skipped,
            passed: external_attempted - external_skipped - external_failed,
            failed: external_failed,
            skipped: external_skipped,
            notes: backend_probes
                .iter()
                .filter_map(|probe| probe.skip_reason.clone())
                .collect(),
        },
    );
    let learned_required = results
        .iter()
        .filter(|result| {
            corpus
                .cases
                .iter()
                .find(|case| case.id == result.id)
                .is_some_and(|case| case.contributions.contains(&EvidenceContribution::Learned))
        })
        .count();
    contributions.insert(
        EvidenceContribution::Learned,
        ContributionReport {
            attempted: learned_attempts,
            passed: learned_passed,
            failed: learned_attempts.saturating_sub(learned_passed),
            skipped: learned_required.saturating_sub(learned_attempts),
            notes: if learned_model.is_none() && learned_required > 0 {
                vec![
                    "No learned Duplex checkpoint was supplied; learned contribution skipped."
                        .into(),
                ]
            } else {
                Vec::new()
            },
        },
    );

    let corpus_bytes = serde_json::to_vec(corpus)?;
    let corpus_sha256 = Sha256::digest(corpus_bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let probes_passed = backend_probes.iter().all(|probe| probe.passed);
    Ok(InterpretationAcceptanceReport {
        schema_version: INTERPRETATION_ACCEPTANCE_REPORT_VERSION,
        corpus_id: corpus.corpus_id.clone(),
        corpus_sha256,
        profile,
        passed: failed_cases == 0 && probes_passed,
        selected_cases: results.len(),
        passed_cases,
        failed_cases,
        metrics,
        contributions,
        cases: results,
        backend_probes,
        leakage_policy_verified: true,
    })
}

fn evaluate_case(
    case: &InterpretationAcceptanceCase,
    duplex_fixtures: &DuplexFixtureSuite,
) -> Result<AcceptanceCaseResult> {
    let started = Instant::now();
    let phonemicizer = phonemicizer_for_variety(&case.variety)
        .with_context(|| format!("loading phonemicizer for case '{}'", case.id))?;
    let output = phonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: case.text.clone(),
            variety: case.variety.clone(),
            style: Some(case.style.clone()),
        })
        .with_context(|| format!("phonemicizing acceptance case '{}'", case.id))?;
    let words = output
        .lexical_candidates
        .iter()
        .map(|candidate| candidate.token.clone())
        .collect::<Vec<_>>();
    let native = VarietyGrammarParser::with_backend(
        case.variety.clone(),
        GrammarParserBackend::TonguesRules,
    )
    .parse(&words, output.syntax.terminal);
    let parse_variants = native
        .ranked_parses
        .iter()
        .filter_map(|parse| {
            serde_json::to_value(&parse.provenance.variant)
                .ok()?
                .get("kind")?
                .as_str()
                .map(str::to_string)
        })
        .collect::<Vec<_>>();
    let link_kinds = native
        .ranked_parses
        .first()
        .into_iter()
        .flat_map(|parse| parse.links.iter().map(|link| link.kind))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let artifact = &output.linguistic_evidence;
    let pronunciation_alternatives = output
        .lexical_candidates
        .iter()
        .map(|candidate| PronunciationAlternativeActual {
            token: candidate.token.clone(),
            selected_candidate_id: candidate.selected_candidate_id.clone(),
            candidates: candidate
                .candidates
                .iter()
                .map(|phones| phones.iter().map(|phone| phone.0.clone()).collect())
                .collect(),
            alternative_ids: candidate
                .alternatives
                .iter()
                .map(|alternative| alternative.id.clone())
                .collect(),
        })
        .collect::<Vec<_>>();
    let selected_phonemes = output
        .lexical_candidates
        .iter()
        .filter_map(|candidate| {
            candidate
                .alternatives
                .iter()
                .find(|alternative| alternative.selected)
                .map(|alternative| {
                    alternative
                        .phonemes
                        .iter()
                        .map(|phoneme| phoneme.0.clone())
                        .collect()
                })
        })
        .collect::<Vec<_>>();
    let claims_by_id = artifact
        .claims
        .iter()
        .map(|claim| (&claim.id, claim))
        .collect::<BTreeMap<_, _>>();
    let selected_confidences = artifact
        .resolutions
        .iter()
        .filter_map(|resolution| {
            resolution
                .winner
                .as_ref()
                .and_then(|winner| claims_by_id.get(winner))
                .map(|claim| claim.confidence.probability)
        })
        .collect::<Vec<_>>();
    let provenance_sources = artifact
        .claims
        .iter()
        .filter_map(|claim| serde_name(&claim.provenance.source))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let streaming = case
        .expected
        .streaming
        .as_ref()
        .map(|expectation| {
            let fixture = duplex_fixtures
                .fixture(&expectation.fixture_id)
                .expect("validated fixture reference");
            let (journal, state) = run_fixture(fixture)?;
            let replay_verified = crate::replay_journal(&journal)? == state;
            let repairs = journal
                .events
                .iter()
                .filter(|event| {
                    matches!(event.event, SimulatorEventKind::HypothesisRepaired { .. })
                })
                .count();
            let withdrawals = journal
                .events
                .iter()
                .filter(|event| {
                    matches!(event.event, SimulatorEventKind::HypothesisWithdrawn { .. })
                })
                .count();
            let abstentions = journal
                .events
                .iter()
                .filter(|event| {
                    matches!(
                        &event.event,
                        SimulatorEventKind::HypothesesReranked {
                            abstained: true,
                            ..
                        }
                    )
                })
                .count();
            let frontiers = journal
                .events
                .iter()
                .filter_map(|event| match &event.event {
                    SimulatorEventKind::CommitFrontierAdvanced { from, to, .. } => {
                        Some((*from, *to))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            let stable_prefix_retained = frontiers.iter().all(|(from, to)| to >= from)
                && frontiers.windows(2).all(|pair| pair[1].0 >= pair[0].1);
            Ok::<_, anyhow::Error>(StreamingActual {
                fixture_id: expectation.fixture_id.clone(),
                final_committed_text: state.committed_text(),
                repairs,
                withdrawals,
                abstentions,
                frontier_advances: frontiers.len(),
                stable_prefix_retained,
                replay_verified,
            })
        })
        .transpose()?;
    let actual = AcceptanceCaseActual {
        grammar_status: native.status,
        parse_ids: native
            .ranked_parses
            .iter()
            .map(|parse| parse.id.0.clone())
            .collect(),
        parse_ranks: native
            .ranked_parses
            .iter()
            .map(|parse| parse.rank)
            .collect(),
        parse_variants,
        link_kinds,
        backend_attempts: native.backend_report.attempts,
        claim_ids: artifact
            .claims
            .iter()
            .map(|claim| claim.id.0.clone())
            .collect(),
        resolution_ids: artifact
            .resolutions
            .iter()
            .map(|resolution| resolution.id.0.clone())
            .collect(),
        conflict_edges: artifact
            .claims
            .iter()
            .map(|claim| claim.conflicts_with.len())
            .sum(),
        lifecycle_events: artifact.lifecycle.len(),
        pronunciation_alternatives,
        selected_phonemes,
        selected_confidences,
        broad_ipa: syllables_to_ipa(&output.syllables),
        phoneme_count: output.phonemes.len(),
        phone_count: output.phones.len(),
        boundary_count: output.boundaries.len(),
        primary_stress_count: output
            .syllables
            .iter()
            .filter(|syllable| matches!(syllable.stress, Spec::Known(Stress::Primary)))
            .count(),
        provenance_sources,
        streaming,
        latency_micros: started.elapsed().as_micros(),
    };
    let diffs = case_diffs(case, &actual);
    Ok(AcceptanceCaseResult {
        id: case.id.clone(),
        variety: case.variety.clone(),
        phenomena: case.phenomena.clone(),
        child_issues: case.child_issues.clone(),
        source_id: case.source_id.clone(),
        leakage_group: case.leakage_group.clone(),
        passed: diffs.is_empty(),
        expected: case.expected.clone(),
        actual,
        diffs,
    })
}

fn case_diffs(
    case: &InterpretationAcceptanceCase,
    actual: &AcceptanceCaseActual,
) -> Vec<AcceptanceDiff> {
    let expected = &case.expected;
    let mut diffs = Vec::new();
    if let Some(status) = expected.grammar_status {
        compare(
            &mut diffs,
            "grammar.status",
            status == actual.grammar_status,
            serde_name(&status).unwrap_or_default(),
            serde_name(&actual.grammar_status).unwrap_or_default(),
        );
    }
    for required in &expected.required_links {
        compare(
            &mut diffs,
            "grammar.required_links",
            actual.link_kinds.contains(required),
            serde_name(required).unwrap_or_default(),
            format!("{:?}", actual.link_kinds),
        );
    }
    for required in &expected.required_parse_variants {
        compare(
            &mut diffs,
            "grammar.required_parse_variants",
            actual.parse_variants.contains(required),
            required.clone(),
            format!("{:?}", actual.parse_variants),
        );
    }
    minimum(
        &mut diffs,
        "grammar.ranked_parses",
        expected.min_ranked_parses,
        actual.parse_ids.len(),
    );
    minimum(
        &mut diffs,
        "claims.total",
        expected.min_claims,
        actual.claim_ids.len(),
    );
    minimum(
        &mut diffs,
        "claims.resolutions",
        expected.min_resolutions,
        actual.resolution_ids.len(),
    );
    minimum(
        &mut diffs,
        "claims.conflict_edges",
        expected.min_conflicts,
        actual.conflict_edges,
    );
    minimum(
        &mut diffs,
        "claims.lifecycle_events",
        expected.min_lifecycle_events,
        actual.lifecycle_events,
    );
    for source in &expected.required_provenance_sources {
        compare(
            &mut diffs,
            "claims.required_provenance_sources",
            actual.provenance_sources.contains(source),
            source.clone(),
            format!("{:?}", actual.provenance_sources),
        );
    }
    let alternatives = actual
        .pronunciation_alternatives
        .iter()
        .map(|candidate| candidate.alternative_ids.len())
        .sum();
    minimum(
        &mut diffs,
        "pronunciation.alternatives",
        expected.min_pronunciation_alternatives,
        alternatives,
    );
    let all_candidates = actual
        .pronunciation_alternatives
        .iter()
        .flat_map(|candidate| candidate.candidates.iter())
        .collect::<Vec<_>>();
    for required in &expected.required_candidate_phonemes {
        compare(
            &mut diffs,
            "pronunciation.required_candidate_phonemes",
            all_candidates.contains(&required),
            format!("{required:?}"),
            format!("{all_candidates:?}"),
        );
    }
    for required in &expected.required_selected_phonemes {
        compare(
            &mut diffs,
            "pronunciation.required_selected_phonemes",
            actual
                .selected_phonemes
                .iter()
                .any(|selected| selected == required),
            format!("{required:?}"),
            format!("{:?}", actual.selected_phonemes),
        );
    }
    minimum(
        &mut diffs,
        "prosody.boundaries",
        expected.min_boundaries,
        actual.boundary_count,
    );
    minimum(
        &mut diffs,
        "prosody.primary_stress",
        expected.min_primary_stress,
        actual.primary_stress_count,
    );
    if let Some(streaming) = &expected.streaming {
        if let Some(actual_streaming) = &actual.streaming {
            compare(
                &mut diffs,
                "streaming.final_committed_text",
                actual_streaming.final_committed_text == streaming.final_committed_text,
                streaming.final_committed_text.clone(),
                actual_streaming.final_committed_text.clone(),
            );
            compare(
                &mut diffs,
                "streaming.repair",
                !streaming.requires_repair || actual_streaming.repairs > 0,
                streaming.requires_repair.to_string(),
                actual_streaming.repairs.to_string(),
            );
            compare(
                &mut diffs,
                "streaming.withdrawal",
                !streaming.requires_withdrawal || actual_streaming.withdrawals > 0,
                streaming.requires_withdrawal.to_string(),
                actual_streaming.withdrawals.to_string(),
            );
            compare(
                &mut diffs,
                "streaming.abstention",
                !streaming.requires_abstention || actual_streaming.abstentions > 0,
                streaming.requires_abstention.to_string(),
                actual_streaming.abstentions.to_string(),
            );
            compare(
                &mut diffs,
                "streaming.stable_prefix_retained",
                !streaming.stable_prefix_retained || actual_streaming.stable_prefix_retained,
                streaming.stable_prefix_retained.to_string(),
                actual_streaming.stable_prefix_retained.to_string(),
            );
            compare(
                &mut diffs,
                "streaming.replay_verified",
                actual_streaming.replay_verified,
                "true",
                actual_streaming.replay_verified.to_string(),
            );
        } else {
            diffs.push(AcceptanceDiff {
                path: "streaming".into(),
                expected: streaming.fixture_id.clone(),
                actual: "missing".into(),
            });
        }
    }
    diffs
}

fn evaluate_backend_probe(probe: &BackendProbe) -> BackendProbeResult {
    let (state, diagnostic) = if let Some(state) = probe.injected_state {
        (
            state,
            Some(format!(
                "deterministic injected backend outcome: {}",
                serde_name(&state).unwrap_or_default()
            )),
        )
    } else {
        let words = words_and_terminal(&probe.text).0;
        let analysis = VarietyGrammarParser::with_backend(probe.variety.clone(), probe.backend)
            .parse(&words, words_and_terminal(&probe.text).1);
        let attempt = analysis.backend_report.attempts.first();
        (
            attempt
                .map(|attempt| attempt.state)
                .unwrap_or(GrammarBackendState::Rejected),
            attempt
                .and_then(|attempt| attempt.diagnostic.clone())
                .or(analysis.diagnostic),
        )
    };
    let mut disposition = backend_disposition(state);
    let unavailable = matches!(
        state,
        GrammarBackendState::FeatureDisabled
            | GrammarBackendState::UnsupportedVariety
            | GrammarBackendState::UnavailableExecutable
            | GrammarBackendState::UnavailableDictionary
            | GrammarBackendState::UnavailableModel
            | GrammarBackendState::SpawnFailure
    );
    let skip_reason = (probe.optional && unavailable).then(|| {
        format!(
            "{} skipped honestly: {}",
            probe.id,
            diagnostic
                .as_deref()
                .unwrap_or("optional backend is unavailable")
        )
    });
    if skip_reason.is_some() {
        disposition = BackendDisposition::Skipped;
    }
    BackendProbeResult {
        id: probe.id.clone(),
        backend: probe.backend,
        observed_state: state,
        disposition,
        passed: probe.allowed_dispositions.contains(&disposition),
        diagnostic,
        skip_reason,
    }
}

fn backend_disposition(state: GrammarBackendState) -> BackendDisposition {
    match state {
        GrammarBackendState::Ready | GrammarBackendState::Accepted => BackendDisposition::Accepted,
        GrammarBackendState::TokenAlignmentLoss | GrammarBackendState::PartialProjection => {
            BackendDisposition::Conservative
        }
        GrammarBackendState::Timeout
        | GrammarBackendState::MalformedOutput
        | GrammarBackendState::Cancelled
        | GrammarBackendState::InputTooLarge
        | GrammarBackendState::OutputTooLarge => BackendDisposition::Fallback,
        GrammarBackendState::Rejected => BackendDisposition::Abstain,
        GrammarBackendState::FeatureDisabled
        | GrammarBackendState::UnsupportedVariety
        | GrammarBackendState::UnavailableExecutable
        | GrammarBackendState::UnavailableDictionary
        | GrammarBackendState::UnavailableModel
        | GrammarBackendState::SpawnFailure => BackendDisposition::Fallback,
    }
}

fn acceptance_metrics(results: &[AcceptanceCaseResult]) -> AcceptanceMetrics {
    let required_links = results
        .iter()
        .map(|result| result.expected.required_links.len())
        .sum::<usize>();
    let matched_links = results
        .iter()
        .flat_map(|result| {
            result
                .expected
                .required_links
                .iter()
                .map(|link| result.actual.link_kinds.contains(link))
        })
        .filter(|matched| *matched)
        .count();
    let ambiguity_cases = results
        .iter()
        .filter(|result| result.expected.min_ranked_parses > 1)
        .collect::<Vec<_>>();
    let lexical_expectations = results
        .iter()
        .flat_map(|result| {
            result
                .expected
                .required_candidate_phonemes
                .iter()
                .map(move |required| (result, required))
        })
        .collect::<Vec<_>>();
    let lexical_matches = lexical_expectations
        .iter()
        .filter(|(result, required)| {
            result
                .actual
                .pronunciation_alternatives
                .iter()
                .flat_map(|candidate| candidate.candidates.iter())
                .any(|candidate| candidate == *required)
        })
        .count();
    let ambiguity_accuracy_cases = results
        .iter()
        .filter(|result| {
            result
                .phenomena
                .iter()
                .any(|value| value == "homophone" || value == "heteronym")
        })
        .collect::<Vec<_>>();
    let expected_repairs = results
        .iter()
        .filter(|result| {
            result
                .expected
                .streaming
                .as_ref()
                .is_some_and(|streaming| streaming.requires_repair)
        })
        .count();
    let actual_repairs = results
        .iter()
        .filter(|result| {
            result
                .actual
                .streaming
                .as_ref()
                .is_some_and(|streaming| streaming.repairs > 0)
        })
        .count();
    let repair_true_positive = results
        .iter()
        .filter(|result| {
            result
                .expected
                .streaming
                .as_ref()
                .is_some_and(|streaming| streaming.requires_repair)
                && result
                    .actual
                    .streaming
                    .as_ref()
                    .is_some_and(|streaming| streaming.repairs > 0)
        })
        .count();
    let boundary_stress_cases = results
        .iter()
        .filter(|result| {
            result.expected.min_boundaries > 0 || result.expected.min_primary_stress > 0
        })
        .collect::<Vec<_>>();
    let calibration = results
        .iter()
        .flat_map(|result| result.actual.selected_confidences.iter().copied())
        // Every retained selected claim is a curated positive in this
        // evaluation-only corpus. This is positive-class Brier, not a claim of
        // population calibration.
        .map(|probability| (1.0 - probability).powi(2))
        .collect::<Vec<_>>();
    let mut latencies = results
        .iter()
        .map(|result| result.actual.latency_micros)
        .collect::<Vec<_>>();
    latencies.sort_unstable();
    let p95 = percentile(&latencies, 0.95);
    AcceptanceMetrics {
        parse_link_agreement: ratio(matched_links, required_links),
        ambiguity_recall: ratio(
            ambiguity_cases
                .iter()
                .filter(|result| result.actual.parse_ids.len() >= result.expected.min_ranked_parses)
                .count(),
            ambiguity_cases.len(),
        ),
        top_k_lexical_accuracy: ratio(lexical_matches, lexical_expectations.len()),
        homophone_heteronym_accuracy: ratio(
            ambiguity_accuracy_cases
                .iter()
                .filter(|result| result.passed)
                .count(),
            ambiguity_accuracy_cases.len(),
        ),
        calibration_brier: (!calibration.is_empty())
            .then(|| calibration.iter().sum::<f64>() / calibration.len() as f64),
        repair_precision: ratio(repair_true_positive, actual_repairs),
        repair_recall: ratio(repair_true_positive, expected_repairs),
        pronunciation_selection_accuracy: ratio(
            results
                .iter()
                .flat_map(|result| {
                    result
                        .expected
                        .required_selected_phonemes
                        .iter()
                        .map(move |required| (result, required))
                })
                .filter(|(result, required)| {
                    result
                        .actual
                        .selected_phonemes
                        .iter()
                        .any(|selected| selected == *required)
                })
                .count(),
            results
                .iter()
                .map(|result| result.expected.required_selected_phonemes.len())
                .sum(),
        ),
        boundary_stress_accuracy: ratio(
            boundary_stress_cases
                .iter()
                .filter(|result| result.passed)
                .count(),
            boundary_stress_cases.len(),
        ),
        latency_mean_micros: if latencies.is_empty() {
            0.0
        } else {
            latencies.iter().sum::<u128>() as f64 / latencies.len() as f64
        },
        latency_p95_micros: p95,
    }
}

fn words_and_terminal(text: &str) -> (Vec<String>, Option<speaking::segment::TerminalPunctuation>) {
    let terminal = text.trim_end().chars().last().and_then(|ch| match ch {
        '.' => Some(speaking::segment::TerminalPunctuation::Period),
        '?' => Some(speaking::segment::TerminalPunctuation::Question),
        '!' => Some(speaking::segment::TerminalPunctuation::Exclamation),
        _ => None,
    });
    let words = text
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|ch: char| ch.is_ascii_punctuation() && ch != '\'' && ch != '-')
                .to_string()
        })
        .filter(|word| !word.is_empty())
        .collect();
    (words, terminal)
}

fn compare(
    diffs: &mut Vec<AcceptanceDiff>,
    path: &str,
    matches: bool,
    expected: impl Into<String>,
    actual: impl Into<String>,
) {
    if !matches {
        diffs.push(AcceptanceDiff {
            path: path.into(),
            expected: expected.into(),
            actual: actual.into(),
        });
    }
}

fn minimum(diffs: &mut Vec<AcceptanceDiff>, path: &str, expected: usize, actual: usize) {
    compare(
        diffs,
        path,
        actual >= expected,
        format!(">= {expected}"),
        actual.to_string(),
    );
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn percentile(values: &[u128], percentile: f64) -> u128 {
    if values.is_empty() {
        return 0;
    }
    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[index.min(values.len() - 1)]
}

fn serde_name<T: Serialize>(value: &T) -> Option<String> {
    serde_json::to_value(value)
        .ok()?
        .as_str()
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> InterpretationAcceptanceCorpus {
        serde_json::from_str(include_str!(
            "../../../fixtures/interpretation/ambiguity-acceptance-v1.json"
        ))
        .expect("acceptance corpus parses")
    }

    fn duplex_fixtures() -> DuplexFixtureSuite {
        serde_json::from_str(include_str!(
            "../../../fixtures/duplex/completion_scenarios_v1.json"
        ))
        .expect("Duplex fixtures parse")
    }

    #[test]
    fn corpus_covers_every_epic_child_typology_and_leakage_boundary() {
        corpus()
            .validate(&duplex_fixtures())
            .expect("acceptance corpus validates");
    }

    #[test]
    fn bounded_ci_profile_is_deterministic_and_passes_without_downloads() {
        let corpus = corpus();
        let fixtures = duplex_fixtures();
        let first =
            evaluate_interpretation_acceptance(&corpus, &fixtures, AcceptanceProfile::Ci, None)
                .expect("first acceptance run");
        let second =
            evaluate_interpretation_acceptance(&corpus, &fixtures, AcceptanceProfile::Ci, None)
                .expect("second acceptance run");

        let failures = first
            .cases
            .iter()
            .filter(|case| !case.passed)
            .map(|case| (&case.id, &case.diffs))
            .collect::<Vec<_>>();
        assert!(first.passed, "acceptance diffs: {failures:#?}");
        assert_eq!(first.selected_cases, second.selected_cases);
        assert_eq!(first.passed_cases, second.passed_cases);
        assert_eq!(first.corpus_sha256, second.corpus_sha256);
        assert_eq!(
            first
                .cases
                .iter()
                .map(|result| (&result.id, &result.actual.parse_ids, &result.diffs))
                .collect::<Vec<_>>(),
            second
                .cases
                .iter()
                .map(|result| (&result.id, &result.actual.parse_ids, &result.diffs))
                .collect::<Vec<_>>()
        );
        assert!(first.backend_probes.iter().all(|probe| probe.passed));
        assert!(first
            .backend_probes
            .iter()
            .any(|probe| probe.disposition == BackendDisposition::Skipped));
    }

    #[test]
    fn failure_report_is_structured_and_diffable() {
        let mut corpus = corpus();
        corpus.cases[0].expected.min_claims = usize::MAX;
        let report = evaluate_interpretation_acceptance(
            &corpus,
            &duplex_fixtures(),
            AcceptanceProfile::Ci,
            None,
        )
        .expect("acceptance report");
        assert!(!report.passed);
        let failed = report
            .cases
            .iter()
            .find(|case| !case.passed)
            .expect("failed case");
        assert!(failed.diffs.iter().any(|diff| diff.path == "claims.total"));
        assert!(serde_json::to_string_pretty(&report).is_ok());
    }
}
