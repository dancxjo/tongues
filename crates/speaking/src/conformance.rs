use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

#[cfg(test)]
use crate::VarietyId;
use crate::{
    EvidenceProvenance, EvidenceSource, PhonemicizeError, PhonemicizeOutput, PhonemicizeRequest,
    Spec, UtterancePlan, display_plan_connected_speech, display_plan_phonemes, display_plan_phones,
    phone_display_symbol, phoneme_default_phone_display_symbol, phonemicizer_for_variety,
};

pub const PRONUNCIATION_ANALYSIS_SCHEMA_VERSION: u32 = 2;
pub const PRONUNCIATION_CONFORMANCE_SCHEMA_VERSION: u32 = 1;
const PRONUNCIATION_CONFORMANCE_CORPUS_JSON: &str =
    include_str!("../../../fixtures/pronunciation/conformance-v1.json");

/// Versioned pronunciation diagnostics.
///
/// Schema v2 stores one typed [`UtterancePlan`]. IPA strings are pure computed
/// projections exposed by the accessor methods below.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PronunciationAnalysis {
    pub schema_version: u32,
    pub normalized_text: String,
    pub lexical_candidates: Vec<LexicalCandidateAnalysis>,
    pub trace: Vec<PronunciationTraceStep>,
    pub plan: UtterancePlan,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LexicalCandidateAnalysis {
    pub word_index: usize,
    pub token: String,
    pub accepted: Vec<String>,
    pub accepted_ids: Vec<String>,
    pub confidence: f32,
    pub provenance: EvidenceProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PronunciationTraceStep {
    pub stage: PronunciationStage,
    pub source: EvidenceSource,
    pub method: String,
    pub before: String,
    pub after: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PronunciationStage {
    TextNormalization,
    LexicalLookup,
    Phonemicization,
    VarietyRealization,
    IpaNormalization,
    CheckpointProjection,
    AcousticInferenceBoundary,
}

impl fmt::Display for PronunciationStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::TextNormalization => "text_normalization",
            Self::LexicalLookup => "lexical_lookup",
            Self::Phonemicization => "phonemicization",
            Self::VarietyRealization => "variety_realization",
            Self::IpaNormalization => "ipa_normalization",
            Self::CheckpointProjection => "checkpoint_projection",
            Self::AcousticInferenceBoundary => "acoustic_inference_boundary",
        })
    }
}

pub fn analyze_pronunciation(
    request: &PhonemicizeRequest,
) -> Result<PronunciationAnalysis, PhonemicizeError> {
    let canonical_variety = crate::canonical_variety_id(&request.variety.0).ok_or_else(|| {
        PhonemicizeError::UnsupportedVariety {
            variety: request.variety.clone(),
        }
    })?;
    let output = phonemicizer_for_variety(&canonical_variety)?.phonemicize(request)?;
    Ok(PronunciationAnalysis::from_output(output))
}

impl PronunciationAnalysis {
    pub fn from_output(output: PhonemicizeOutput) -> Self {
        let plan = UtterancePlan::from(&output);
        let broad_phonemes = display_plan_phonemes(&plan);
        let broad_phoneme_ids = format_phoneme_ids(&plan);
        let lexical_candidates = output
            .lexical_candidates
            .iter()
            .map(|candidate| LexicalCandidateAnalysis {
                word_index: candidate.word_index,
                token: candidate.token.clone(),
                accepted: candidate
                    .candidates
                    .iter()
                    .map(|sequence| {
                        sequence
                            .iter()
                            .map(|id| phoneme_default_phone_display_symbol(id, &output.variety))
                            .collect::<String>()
                    })
                    .collect(),
                accepted_ids: candidate
                    .candidates
                    .iter()
                    .map(|sequence| {
                        sequence
                            .iter()
                            .map(|id| crate::phoneme_display_symbol(id))
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .collect(),
                confidence: candidate.confidence,
                provenance: candidate.provenance.clone(),
            })
            .collect::<Vec<_>>();

        let mut trace = vec![PronunciationTraceStep {
            stage: PronunciationStage::TextNormalization,
            source: EvidenceSource::Rule,
            method: format!("{} text normalization", output.variety.0),
            before: output.text.clone(),
            after: output.normalized_text.clone(),
            confidence: 1.0,
        }];
        trace.extend(
            lexical_candidates
                .iter()
                .map(|candidate| PronunciationTraceStep {
                    stage: PronunciationStage::LexicalLookup,
                    source: candidate.provenance.source.clone(),
                    method: candidate.provenance.method.clone(),
                    before: candidate.token.clone(),
                    after: candidate.accepted.join(" | "),
                    confidence: candidate.confidence,
                }),
        );
        trace.push(PronunciationTraceStep {
            stage: PronunciationStage::Phonemicization,
            source: output.provenance.source.clone(),
            method: output.provenance.method.clone(),
            before: output.normalized_text.clone(),
            after: broad_phonemes.clone(),
            confidence: minimum_phoneme_confidence(&output),
        });
        trace.extend(realization_trace(&output));
        trace.push(PronunciationTraceStep {
            stage: PronunciationStage::IpaNormalization,
            source: EvidenceSource::Rule,
            method: "speaking canonical IPA display".into(),
            before: broad_phoneme_ids.clone(),
            after: broad_phonemes.clone(),
            confidence: minimum_phoneme_confidence(&output),
        });

        Self {
            schema_version: PRONUNCIATION_ANALYSIS_SCHEMA_VERSION,
            normalized_text: output.normalized_text.clone(),
            lexical_candidates,
            trace,
            plan,
        }
    }

    pub fn broad_phonemes(&self) -> String {
        display_plan_phonemes(&self.plan)
    }

    pub fn broad_phoneme_ids(&self) -> String {
        format_phoneme_ids(&self.plan)
    }

    pub fn realized_phones(&self) -> String {
        display_plan_phones(&self.plan)
    }

    pub fn connected_speech(&self) -> String {
        display_plan_connected_speech(&self.plan)
    }
}

fn realization_trace(output: &PhonemicizeOutput) -> Vec<PronunciationTraceStep> {
    let mut steps = Vec::new();
    for (index, phoneme) in output.phonemes.iter().enumerate() {
        let Spec::Known(phoneme_id) = &phoneme.phoneme else {
            continue;
        };
        let before = phoneme_default_phone_display_symbol(phoneme_id, &output.variety);
        for phone in &phoneme.realized_as {
            let Spec::Known(phone_id) = &phone.phone else {
                continue;
            };
            let after = phone_display_symbol(phone_id).to_string();
            if before != after || phone.provenance.source == EvidenceSource::Rule {
                steps.push(PronunciationTraceStep {
                    stage: PronunciationStage::VarietyRealization,
                    source: phone.provenance.source.clone(),
                    method: phone.provenance.method.clone(),
                    before: format!("{index}:{before}"),
                    after,
                    confidence: phone.confidence,
                });
            }
        }
    }
    if steps.is_empty() {
        let plan = UtterancePlan::from(output);
        steps.push(PronunciationTraceStep {
            stage: PronunciationStage::VarietyRealization,
            source: output.provenance.source.clone(),
            method: format!("{} default phone realization", output.variety.0),
            before: display_plan_phonemes(&plan),
            after: display_plan_phones(&plan),
            confidence: minimum_phone_confidence(output),
        });
    }
    steps
}

fn minimum_phoneme_confidence(output: &PhonemicizeOutput) -> f32 {
    output
        .phonemes
        .iter()
        .map(|token| token.confidence)
        .reduce(f32::min)
        .unwrap_or(1.0)
}

fn minimum_phone_confidence(output: &PhonemicizeOutput) -> f32 {
    output
        .phones
        .iter()
        .map(|token| token.confidence)
        .reduce(f32::min)
        .unwrap_or(1.0)
}

fn format_phoneme_ids(plan: &UtterancePlan) -> String {
    plan.intended_phonemes
        .iter()
        .filter_map(|token| {
            let Spec::Known(id) = &token.phoneme else {
                return None;
            };
            Some(crate::phoneme_display_symbol(id))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PronunciationConformanceCorpus {
    pub schema_version: u32,
    pub cases: Vec<PronunciationConformanceCase>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PronunciationConformanceCase {
    pub id: String,
    pub input_text: String,
    pub variety: String,
    #[serde(default)]
    pub careful_style: bool,
    pub expected: PronunciationExpectation,
    #[serde(default)]
    pub checkpoints: BTreeMap<String, CheckpointExpectation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PronunciationExpectation {
    pub normalized_text: AcceptedAlternatives<String>,
    pub lexical_candidates: AcceptedAlternatives<Vec<String>>,
    pub broad_phonemes: AcceptedAlternatives<String>,
    pub broad_phoneme_ids: AcceptedAlternatives<String>,
    pub realized_phones: AcceptedAlternatives<String>,
    #[serde(default)]
    pub connected_speech: Option<AcceptedAlternatives<String>>,
    #[serde(default)]
    pub required_trace_methods: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointExpectation {
    pub symbols: AcceptedAlternatives<String>,
    pub token_ids: AcceptedAlternatives<Vec<i64>>,
    #[serde(default)]
    pub declared_collapses: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcceptedAlternatives<T> {
    pub required: T,
    #[serde(default)]
    pub accepted_alternatives: Vec<T>,
}

impl<T: PartialEq> AcceptedAlternatives<T> {
    pub fn accepts(&self, actual: &T) -> bool {
        &self.required == actual
            || self
                .accepted_alternatives
                .iter()
                .any(|value| value == actual)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConformanceFailure {
    pub case_id: String,
    pub first_divergent_stage: PronunciationStage,
    pub expected: String,
    pub actual: String,
}

pub fn compare_conformance_case(
    case: &PronunciationConformanceCase,
    analysis: &PronunciationAnalysis,
) -> Result<(), ConformanceFailure> {
    compare_stage(
        case,
        PronunciationStage::TextNormalization,
        &case.expected.normalized_text,
        &analysis.normalized_text,
    )?;
    let lexical = analysis
        .lexical_candidates
        .iter()
        .flat_map(|candidate| candidate.accepted.clone())
        .collect::<Vec<_>>();
    compare_stage(
        case,
        PronunciationStage::LexicalLookup,
        &case.expected.lexical_candidates,
        &lexical,
    )?;
    compare_stage(
        case,
        PronunciationStage::Phonemicization,
        &case.expected.broad_phonemes,
        &analysis.broad_phonemes(),
    )?;
    compare_stage(
        case,
        PronunciationStage::Phonemicization,
        &case.expected.broad_phoneme_ids,
        &analysis.broad_phoneme_ids(),
    )?;
    compare_stage(
        case,
        PronunciationStage::VarietyRealization,
        &case.expected.realized_phones,
        &analysis.realized_phones(),
    )?;
    if let Some(expected) = &case.expected.connected_speech {
        compare_stage(
            case,
            PronunciationStage::VarietyRealization,
            expected,
            &analysis.connected_speech(),
        )?;
    }
    for required in &case.expected.required_trace_methods {
        if !analysis
            .trace
            .iter()
            .any(|step| step.method.contains(required))
        {
            return Err(ConformanceFailure {
                case_id: case.id.clone(),
                first_divergent_stage: PronunciationStage::VarietyRealization,
                expected: format!("trace method containing {required:?}"),
                actual: analysis
                    .trace
                    .iter()
                    .map(|step| step.method.as_str())
                    .collect::<Vec<_>>()
                    .join(" | "),
            });
        }
    }
    Ok(())
}

fn compare_stage<T>(
    case: &PronunciationConformanceCase,
    stage: PronunciationStage,
    expected: &AcceptedAlternatives<T>,
    actual: &T,
) -> Result<(), ConformanceFailure>
where
    T: PartialEq + fmt::Debug,
{
    if expected.accepts(actual) {
        return Ok(());
    }
    Err(ConformanceFailure {
        case_id: case.id.clone(),
        first_divergent_stage: stage,
        expected: format!(
            "{:?} or {:?}",
            expected.required, expected.accepted_alternatives
        ),
        actual: format!("{actual:?}"),
    })
}

pub fn load_pronunciation_conformance_corpus()
-> Result<PronunciationConformanceCorpus, serde_json::Error> {
    serde_json::from_str(PRONUNCIATION_CONFORMANCE_CORPUS_JSON)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PhonemicizeStyle;

    fn corpus() -> PronunciationConformanceCorpus {
        load_pronunciation_conformance_corpus().expect("pronunciation conformance corpus")
    }

    #[test]
    fn core_pronunciation_conformance_is_checkpoint_free() {
        let corpus = corpus();
        assert_eq!(
            corpus.schema_version,
            PRONUNCIATION_CONFORMANCE_SCHEMA_VERSION
        );
        assert!(
            !corpus.cases.is_empty(),
            "the conformance corpus must exercise real pronunciation cases"
        );
        for case in &corpus.cases {
            let analysis = analyze_pronunciation(&PhonemicizeRequest {
                text: case.input_text.clone(),
                variety: VarietyId(case.variety.clone()),
                style: case.careful_style.then_some(PhonemicizeStyle {
                    careful_style: true,
                }),
            })
            .unwrap_or_else(|error| panic!("{} could not be analyzed: {error}", case.id));
            compare_conformance_case(case, &analysis).unwrap_or_else(|failure| {
                panic!("{}", serde_json::to_string_pretty(&failure).unwrap())
            });
        }
    }

    #[test]
    fn reports_the_first_divergent_stage() {
        let analysis = analyze_pronunciation(&PhonemicizeRequest {
            text: "hello".into(),
            variety: VarietyId("en-US".into()),
            style: None,
        })
        .expect("analysis");
        let lexical_candidates = analysis
            .lexical_candidates
            .iter()
            .flat_map(|candidate| candidate.accepted.clone())
            .collect();
        let mut case = PronunciationConformanceCase {
            id: "deliberate-normalization-divergence".into(),
            input_text: "hello".into(),
            variety: "en-US".into(),
            careful_style: false,
            expected: PronunciationExpectation {
                normalized_text: AcceptedAlternatives {
                    required: analysis.normalized_text.clone(),
                    accepted_alternatives: Vec::new(),
                },
                lexical_candidates: AcceptedAlternatives {
                    required: lexical_candidates,
                    accepted_alternatives: Vec::new(),
                },
                broad_phonemes: AcceptedAlternatives {
                    required: analysis.broad_phonemes(),
                    accepted_alternatives: Vec::new(),
                },
                broad_phoneme_ids: AcceptedAlternatives {
                    required: analysis.broad_phoneme_ids(),
                    accepted_alternatives: Vec::new(),
                },
                realized_phones: AcceptedAlternatives {
                    required: analysis.realized_phones(),
                    accepted_alternatives: Vec::new(),
                },
                connected_speech: Some(AcceptedAlternatives {
                    required: analysis.connected_speech(),
                    accepted_alternatives: Vec::new(),
                }),
                required_trace_methods: Vec::new(),
            },
            checkpoints: BTreeMap::new(),
        };
        case.expected.normalized_text.required = "deliberately wrong".into();
        let failure = compare_conformance_case(&case, &analysis).expect_err("must diverge");
        assert_eq!(
            failure.first_divergent_stage,
            PronunciationStage::TextNormalization
        );
    }

    #[test]
    fn accepted_alternatives_are_not_required_equivalence() {
        let expectation = AcceptedAlternatives {
            required: "one".to_string(),
            accepted_alternatives: vec!["another".to_string()],
        };
        assert!(expectation.accepts(&"one".to_string()));
        assert!(expectation.accepts(&"another".to_string()));
        assert!(!expectation.accepts(&"neither".to_string()));
    }

    #[test]
    fn analysis_serializes_one_linguistic_ir() {
        let analysis = analyze_pronunciation(&PhonemicizeRequest {
            text: "Hello, world?".into(),
            variety: VarietyId("en-US".into()),
            style: None,
        })
        .expect("analysis");
        let json = serde_json::to_value(analysis).expect("analysis JSON");
        assert!(json.get("plan").is_some());
        for duplicate in [
            "output",
            "input_text",
            "variety",
            "broad_phonemes",
            "broad_phoneme_ids",
            "realized_phones",
        ] {
            assert!(
                json.get(duplicate).is_none(),
                "{duplicate} must remain a projection of plan"
            );
        }
    }

    #[test]
    fn analysis_keeps_intended_and_connected_speech_projections_distinct() {
        let analysis = analyze_pronunciation(&PhonemicizeRequest {
            text: "umbrella up".into(),
            variety: VarietyId("en-GB-RP".into()),
            style: None,
        })
        .expect("analysis");

        assert_eq!(analysis.broad_phonemes(), "əmˈbɹe.lə ˈʌp");
        assert_eq!(analysis.connected_speech(), "əmˈbɹe.ləɹ | ˈʌp ↘ .");
    }
}
