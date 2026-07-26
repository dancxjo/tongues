use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    phone_display_symbol, phoneme_default_phone_display_symbol, phonemicizer_for_variety,
    token_stress, EvidenceProvenance, EvidenceSource, FeatureBundle, FeatureId, FeatureValue,
    PhonemicizeError, PhonemicizeOutput, PhonemicizeRequest, Spec, Stress, VarietyId,
};

pub const PRONUNCIATION_ANALYSIS_SCHEMA_VERSION: u32 = 1;
pub const PRONUNCIATION_CONFORMANCE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PronunciationAnalysis {
    pub schema_version: u32,
    pub input_text: String,
    pub normalized_text: String,
    pub variety: VarietyId,
    pub lexical_candidates: Vec<LexicalCandidateAnalysis>,
    pub broad_phonemes: String,
    pub broad_phoneme_ids: String,
    pub realized_phones: String,
    pub trace: Vec<PronunciationTraceStep>,
    pub output: PhonemicizeOutput,
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
        let broad_phonemes = format_phonemes(&output);
        let broad_phoneme_ids = format_phoneme_ids(&output);
        let realized_phones = format_phones(&output);
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
                            .map(|id| {
                                phoneme_default_phone_display_symbol(id, &output.variety)
                            })
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
            input_text: output.text.clone(),
            normalized_text: output.normalized_text.clone(),
            variety: output.variety.clone(),
            lexical_candidates,
            broad_phonemes,
            broad_phoneme_ids,
            realized_phones,
            trace,
            output,
        }
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
        steps.push(PronunciationTraceStep {
            stage: PronunciationStage::VarietyRealization,
            source: output.provenance.source.clone(),
            method: format!("{} default phone realization", output.variety.0),
            before: format_phonemes(output),
            after: format_phones(output),
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

fn format_phonemes(output: &PhonemicizeOutput) -> String {
    format_indexed_symbols(
        output.phonemes.iter().filter_map(|token| {
            let Spec::Known(id) = &token.phoneme else {
                return None;
            };
            let mut symbol = phoneme_default_phone_display_symbol(id, &output.variety);
            if let Some(stress) = token_stress(token) {
                symbol.insert_str(
                    0,
                    match stress {
                        Stress::Primary => "ˈ",
                        Stress::Secondary => "ˌ",
                        Stress::Unstressed | Stress::Reduced => "",
                    },
                );
            }
            Some((symbol, token_word_index(&token.features)))
        }),
    )
}

fn format_phoneme_ids(output: &PhonemicizeOutput) -> String {
    format_indexed_symbols(output.phonemes.iter().filter_map(|token| {
        let Spec::Known(id) = &token.phoneme else {
            return None;
        };
        Some((
            crate::phoneme_display_symbol(id).to_string(),
            token_word_index(&token.features),
        ))
    }))
}

fn format_phones(output: &PhonemicizeOutput) -> String {
    format_indexed_symbols(output.phones.iter().filter_map(|token| {
        let Spec::Known(id) = &token.phone else {
            return None;
        };
        (!id.as_str().starts_with("boundary.")).then(|| {
            (
                phone_display_symbol(id).to_string(),
                token_word_index(&token.features),
            )
        })
    }))
}

fn format_indexed_symbols(
    symbols: impl IntoIterator<Item = (String, Option<usize>)>,
) -> String {
    let mut output = String::new();
    let mut previous_word = None;
    for (symbol, word_index) in symbols {
        if previous_word.is_some() && word_index != previous_word {
            output.push(' ');
        }
        output.push_str(&symbol);
        if word_index.is_some() {
            previous_word = word_index;
        }
    }
    output
}

fn token_word_index(features: &FeatureBundle) -> Option<usize> {
    match features
        .values
        .get(&FeatureId("orthography.word_index".into()))
    {
        Some(Spec::Known(FeatureValue::Number(value))) if value.is_finite() && *value >= 0.0 => {
            Some(*value as usize)
        }
        _ => None,
    }
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
        &self.required == actual || self.accepted_alternatives.iter().any(|value| value == actual)
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
        &analysis.broad_phonemes,
    )?;
    compare_stage(
        case,
        PronunciationStage::Phonemicization,
        &case.expected.broad_phoneme_ids,
        &analysis.broad_phoneme_ids,
    )?;
    compare_stage(
        case,
        PronunciationStage::VarietyRealization,
        &case.expected.realized_phones,
        &analysis.realized_phones,
    )?;
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
        expected: format!("{:?} or {:?}", expected.required, expected.accepted_alternatives),
        actual: format!("{actual:?}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PhonemicizeStyle;

    fn corpus() -> PronunciationConformanceCorpus {
        serde_json::from_str(include_str!(
            "../../../fixtures/pronunciation/conformance-v1.json"
        ))
        .expect("pronunciation conformance corpus")
    }

    #[test]
    fn core_pronunciation_conformance_is_checkpoint_free() {
        let corpus = corpus();
        assert_eq!(
            corpus.schema_version,
            PRONUNCIATION_CONFORMANCE_SCHEMA_VERSION
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
            compare_conformance_case(case, &analysis)
                .unwrap_or_else(|failure| panic!("{}", serde_json::to_string_pretty(&failure).unwrap()));
        }
    }

    #[test]
    fn reports_the_first_divergent_stage() {
        let mut case = corpus().cases[0].clone();
        case.expected.normalized_text.required = "deliberately wrong".into();
        let analysis = analyze_pronunciation(&PhonemicizeRequest {
            text: case.input_text.clone(),
            variety: VarietyId(case.variety.clone()),
            style: None,
        })
        .expect("analysis");
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
}
