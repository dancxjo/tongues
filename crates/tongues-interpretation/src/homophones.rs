//! Bounded, provenance-preserving homophone and confusable-word hypotheses.
//!
//! Pronunciation equivalence is derived from the configured pronunciation
//! resources. Context can rank spellings, but it never rewrites or duplicates
//! the acoustic observation shared by those spellings.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use speaking::data::lexicons::{cmudict, PronunciationStatus};
use speaking::syntax::{GrammarAnalysisStatus, GrammarParser, VarietyGrammarParser};
use speaking::{
    phonemicizer_for_variety, ClaimConfidence, ClaimRationale, ClaimResolutionId,
    EvidenceProvenance, EvidenceSource, LinguisticClaim, LinguisticClaimId, LinguisticClaimKind,
    LinguisticClaimValue, LinguisticEvidenceArtifact, LinguisticTarget, PhonemeId,
    PhonemicizeRequest, TextRange, UtteranceId, VarietyId,
};

/// Small shared lexicon used to make common ambiguity classes available even
/// when a particular runtime vocabulary contains only one spelling.
pub const COMMON_CONFUSABLE_WORDS: &[&str] = &[
    "to", "two", "too", "there", "their", "they're", "for", "four", "right", "write", "hear",
    "here", "its", "it's", "your", "you're", "one", "won", "eight", "ate", "no", "know",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfusableRelationKind {
    Homophone,
    NearHomophone,
    Contraction,
    NumberForm,
    Deletion,
    Insertion,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PronunciationProvenance {
    pub resource: String,
    pub variety: String,
    pub normalized_phone_sequence: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfusableLexeme {
    pub spelling: String,
    pub phonemes: Vec<String>,
    pub provenance: PronunciationProvenance,
}

/// Adapter payload for Wiktionary or another caller-owned pronunciation
/// resource. Candidate symbols must already use that resource's normalized
/// phone/phoneme inventory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PronunciationResourceEntry {
    pub spelling: String,
    pub candidates: Vec<Vec<String>>,
    pub resource: String,
    pub variety: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfusableMatch {
    pub lexeme: ConfusableLexeme,
    pub relation: ConfusableRelationKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfusablePronunciationIndex {
    pub variety: String,
    by_pronunciation: BTreeMap<String, Vec<ConfusableLexeme>>,
}

impl ConfusablePronunciationIndex {
    /// Builds an index from CMUdict plus the variety phonemicizer. Only words
    /// supplied by the caller are indexed, so dataset splits can be isolated.
    pub fn from_words<I, S>(words: I, variety: &str) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let variety_id = VarietyId(variety.to_string());
        let phonemicizer = phonemicizer_for_variety(&variety_id)?;
        let mut unique = words
            .into_iter()
            .filter_map(|word| normalize_spelling(word.as_ref()))
            .collect::<BTreeSet<_>>();
        unique.extend(
            COMMON_CONFUSABLE_WORDS
                .iter()
                .map(|word| (*word).to_string()),
        );

        let mut by_pronunciation = BTreeMap::<String, Vec<ConfusableLexeme>>::new();
        for spelling in unique {
            let mut seen = BTreeSet::new();
            let cmu_matches = cmudict::homophones(&spelling);
            let cmu_found = !cmu_matches.is_empty();
            for entry in cmu_matches {
                if entry.status == PronunciationStatus::Missing {
                    continue;
                }
                for candidate in entry.candidates {
                    let phonemes = candidate
                        .into_iter()
                        .map(|phone| strip_cmu_stress(&phone.raw_symbol()))
                        .collect::<Vec<_>>();
                    insert_lexeme(
                        &mut by_pronunciation,
                        &mut seen,
                        &entry.lookup,
                        phonemes,
                        format!("cmudict:{}", entry.source),
                        "en-US",
                    );
                }
            }

            if cmu_found && variety.eq_ignore_ascii_case("en-US") {
                continue;
            }
            let output = phonemicizer
                .phonemicize(&PhonemicizeRequest {
                    text: spelling.clone(),
                    variety: variety_id.clone(),
                    style: None,
                })
                .with_context(|| format!("phonemicizing homophone lexeme `{spelling}`"))?;
            for candidate in output
                .lexical_candidates
                .first()
                .into_iter()
                .flat_map(|entry| entry.candidates.iter())
            {
                insert_lexeme(
                    &mut by_pronunciation,
                    &mut seen,
                    &spelling,
                    candidate.iter().map(|phoneme| phoneme.0.clone()).collect(),
                    format!(
                        "{:?}:{}",
                        output.provenance.source, output.provenance.method
                    )
                    .to_ascii_lowercase(),
                    variety,
                );
            }
        }
        for lexemes in by_pronunciation.values_mut() {
            lexemes.sort_by(|left, right| left.spelling.cmp(&right.spelling));
            lexemes.dedup_by(|left, right| {
                left.spelling == right.spelling && left.phonemes == right.phonemes
            });
        }
        Ok(Self {
            variety: variety.to_string(),
            by_pronunciation,
        })
    }

    /// Adds caller-owned lexicon entries, such as prepared Wiktionary
    /// pronunciations, without discarding their resource and variety labels.
    pub fn extend_resource_entries<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = PronunciationResourceEntry>,
    {
        for entry in entries {
            let Some(spelling) = normalize_spelling(&entry.spelling) else {
                continue;
            };
            let mut seen = BTreeSet::new();
            for candidate in entry.candidates {
                insert_lexeme(
                    &mut self.by_pronunciation,
                    &mut seen,
                    &spelling,
                    candidate,
                    entry.resource.clone(),
                    &entry.variety,
                );
            }
        }
        for lexemes in self.by_pronunciation.values_mut() {
            lexemes.sort_by(|left, right| left.spelling.cmp(&right.spelling));
            lexemes.dedup_by(|left, right| {
                left.spelling == right.spelling && left.phonemes == right.phonemes
            });
        }
    }

    pub fn candidates(&self, word: &str, include_near: bool) -> Vec<ConfusableMatch> {
        let Some(word) = normalize_spelling(word) else {
            return Vec::new();
        };
        let keys = self
            .by_pronunciation
            .iter()
            .filter(|(_, lexemes)| lexemes.iter().any(|lexeme| lexeme.spelling == word))
            .map(|(key, _)| key.clone())
            .collect::<BTreeSet<_>>();
        let mut out = Vec::new();
        for (key, lexemes) in &self.by_pronunciation {
            let distance = keys
                .iter()
                .map(|source| phone_edit_distance(source, key))
                .min()
                .unwrap_or(usize::MAX);
            if distance > usize::from(include_near) {
                continue;
            }
            for lexeme in lexemes {
                if lexeme.spelling == word {
                    continue;
                }
                out.push(ConfusableMatch {
                    relation: if distance == 0 {
                        classify_exact_relation(&word, &lexeme.spelling)
                    } else {
                        ConfusableRelationKind::NearHomophone
                    },
                    lexeme: lexeme.clone(),
                });
            }
        }
        out.sort_by(|left, right| {
            relation_rank(left.relation)
                .cmp(&relation_rank(right.relation))
                .then_with(|| left.lexeme.spelling.cmp(&right.lexeme.spelling))
                .then_with(|| left.lexeme.phonemes.cmp(&right.lexeme.phonemes))
        });
        out.dedup_by(|left, right| left.lexeme.spelling == right.lexeme.spelling);
        out
    }
}

fn insert_lexeme(
    index: &mut BTreeMap<String, Vec<ConfusableLexeme>>,
    seen: &mut BTreeSet<String>,
    spelling: &str,
    phonemes: Vec<String>,
    resource: String,
    variety: &str,
) {
    if phonemes.is_empty() {
        return;
    }
    let key = phonemes.join(" ");
    if !seen.insert(format!("{spelling}|{key}")) {
        return;
    }
    index
        .entry(key.clone())
        .or_default()
        .push(ConfusableLexeme {
            spelling: spelling.to_string(),
            phonemes,
            provenance: PronunciationProvenance {
                resource,
                variety: variety.to_string(),
                normalized_phone_sequence: key,
            },
        });
}

fn strip_cmu_stress(phone: &str) -> String {
    phone
        .strip_suffix(['0', '1', '2'])
        .unwrap_or(phone)
        .to_string()
}

fn phone_edit_distance(left: &str, right: &str) -> usize {
    let left = left.split_whitespace().collect::<Vec<_>>();
    let right = right.split_whitespace().collect::<Vec<_>>();
    sequence_edit_distance(&left, &right)
}

fn sequence_edit_distance(left: &[&str], right: &[&str]) -> usize {
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    for (left_index, left_phone) in left.iter().enumerate() {
        let mut current = vec![left_index + 1];
        for (right_index, right_phone) in right.iter().enumerate() {
            current.push(
                (previous[right_index + 1] + 1)
                    .min(current[right_index] + 1)
                    .min(previous[right_index] + usize::from(left_phone != right_phone)),
            );
        }
        previous = current;
    }
    previous[right.len()]
}

fn relation_rank(relation: ConfusableRelationKind) -> u8 {
    match relation {
        ConfusableRelationKind::Contraction => 0,
        ConfusableRelationKind::NumberForm => 1,
        ConfusableRelationKind::Homophone => 2,
        ConfusableRelationKind::NearHomophone => 3,
        ConfusableRelationKind::Deletion => 4,
        ConfusableRelationKind::Insertion => 5,
    }
}

fn classify_exact_relation(left: &str, right: &str) -> ConfusableRelationKind {
    if left.contains('\'') || right.contains('\'') {
        ConfusableRelationKind::Contraction
    } else if is_number_form(left) || is_number_form(right) {
        ConfusableRelationKind::NumberForm
    } else {
        ConfusableRelationKind::Homophone
    }
}

fn is_number_form(word: &str) -> bool {
    matches!(
        word,
        "zero"
            | "one"
            | "two"
            | "three"
            | "four"
            | "five"
            | "six"
            | "seven"
            | "eight"
            | "nine"
            | "ten"
    ) || word.chars().all(|character| character.is_ascii_digit())
}

fn normalize_spelling(word: &str) -> Option<String> {
    let normalized = word
        .trim_matches(|character: char| !character.is_alphanumeric() && character != '\'')
        .to_ascii_lowercase();
    (!normalized.is_empty()).then_some(normalized)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HomophoneScoringWeights {
    pub acoustic: f32,
    pub previous_word_head: f32,
    pub current_word_head: f32,
    pub next_word_head: f32,
    pub masked_cloze: f32,
    pub grammar: f32,
    pub lexical_frequency: f32,
    pub explicit_context: f32,
}

impl Default for HomophoneScoringWeights {
    fn default() -> Self {
        Self {
            acoustic: 1.0,
            previous_word_head: 0.35,
            current_word_head: 0.75,
            next_word_head: 0.35,
            masked_cloze: 0.75,
            grammar: 0.5,
            lexical_frequency: 0.2,
            explicit_context: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HomophoneHypothesisConfig {
    pub max_candidates_per_word: usize,
    pub max_total_candidates: usize,
    pub include_near_homophones: bool,
    pub commit_posterior: f32,
    pub clarification_margin: f32,
    pub weights: HomophoneScoringWeights,
}

impl Default for HomophoneHypothesisConfig {
    fn default() -> Self {
        Self {
            max_candidates_per_word: 5,
            max_total_candidates: 32,
            include_near_homophones: true,
            commit_posterior: 0.72,
            clarification_margin: 0.12,
            weights: HomophoneScoringWeights::default(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HomophoneContextScores {
    pub previous_word_head: BTreeMap<String, f32>,
    pub current_word_head: BTreeMap<String, f32>,
    pub next_word_head: BTreeMap<String, f32>,
    pub masked_cloze: BTreeMap<String, f32>,
    pub lexical_frequency: BTreeMap<String, f32>,
    pub explicit_context: BTreeMap<String, f32>,
    pub lexical_frequency_licensed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticWordAlternative {
    pub spelling: String,
    pub phonemes: Vec<String>,
    /// Calibrated acoustic likelihood from the upstream word/phone lattice.
    pub acoustic_likelihood: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignedWordObservation {
    pub word: String,
    pub word_index: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub start_frame: usize,
    pub end_frame: usize,
    pub acoustic_evidence_id: String,
    #[serde(default)]
    pub observed_phonemes: Vec<String>,
    #[serde(default)]
    pub lattice_alternatives: Vec<AcousticWordAlternative>,
    /// Calibrated acoustic likelihood, or an explicitly neutral value when
    /// the upstream recognizer has not calibrated word posteriors.
    pub acoustic_score: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HomophoneScoreComponents {
    pub acoustic: f32,
    pub previous_word_head: f32,
    pub current_word_head: f32,
    pub next_word_head: f32,
    pub masked_cloze: f32,
    pub grammar: f32,
    pub lexical_frequency: Option<f32>,
    pub explicit_context: f32,
    pub combined: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignedSpellingHypothesis {
    pub spelling: String,
    pub relation: ConfusableRelationKind,
    pub pronunciation: Vec<String>,
    pub pronunciation_provenance: PronunciationProvenance,
    pub start_char: usize,
    pub end_char: usize,
    pub start_frame: usize,
    pub end_frame: usize,
    pub acoustic_evidence_id: String,
    /// False for exact homophones: context selected a spelling, not acoustics.
    pub acoustically_distinguishable: bool,
    pub scores: HomophoneScoreComponents,
    pub posterior: f32,
    pub selected: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HomophoneDecision {
    Committed,
    Provisional,
    ClarificationRequired,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignedHomophoneLattice {
    pub utterance_id: String,
    pub transcript: String,
    pub alternatives: Vec<Vec<AlignedSpellingHypothesis>>,
    pub decision: HomophoneDecision,
    pub selected_transcript: Option<String>,
    pub clarification: Option<String>,
    pub linguistic_evidence: LinguisticEvidenceArtifact,
}

/// Creates aligned alternatives while retaining one acoustic observation per
/// word slot. Callers supply calibrated model/context contributions separately.
pub fn rank_homophone_hypotheses(
    utterance_id: &str,
    transcript: &str,
    observations: &[AlignedWordObservation],
    index: &ConfusablePronunciationIndex,
    context: &HomophoneContextScores,
    config: &HomophoneHypothesisConfig,
) -> Result<AlignedHomophoneLattice> {
    let parser = VarietyGrammarParser::new(VarietyId(index.variety.clone()));
    let original_words = observations
        .iter()
        .map(|observation| observation.word.clone())
        .collect::<Vec<_>>();
    let utterance_id_value = UtteranceId(utterance_id.to_string());
    let mut evidence = LinguisticEvidenceArtifact::new(utterance_id_value.clone());
    let mut alternatives = Vec::new();
    let mut total = 0usize;

    for observation in observations {
        if total >= config.max_total_candidates {
            break;
        }
        let original = normalize_spelling(&observation.word).unwrap_or_default();
        let mut candidates = index.candidates(&original, config.include_near_homophones);
        for lattice_candidate in &observation.lattice_alternatives {
            let spelling = match normalize_spelling(&lattice_candidate.spelling) {
                Some(spelling) => spelling,
                None if lattice_candidate.phonemes.is_empty() => "<deletion>".to_string(),
                None => continue,
            };
            let distance = sequence_edit_distance(
                &observation
                    .observed_phonemes
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                &lattice_candidate
                    .phonemes
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            );
            let relation = if lattice_candidate.phonemes.is_empty() {
                ConfusableRelationKind::Deletion
            } else if observation.observed_phonemes.is_empty() {
                ConfusableRelationKind::Insertion
            } else if distance == 0 {
                classify_exact_relation(&original, &spelling)
            } else {
                ConfusableRelationKind::NearHomophone
            };
            candidates.push(ConfusableMatch {
                lexeme: ConfusableLexeme {
                    spelling,
                    phonemes: lattice_candidate.phonemes.clone(),
                    provenance: PronunciationProvenance {
                        resource: "asr-word-phone-lattice".to_string(),
                        variety: index.variety.clone(),
                        normalized_phone_sequence: lattice_candidate.phonemes.join(" "),
                    },
                },
                relation,
            });
        }
        let original_lexeme =
            pronunciation_for_word(index, &original).unwrap_or_else(|| ConfusableLexeme {
                spelling: original.clone(),
                phonemes: observation.observed_phonemes.clone(),
                provenance: PronunciationProvenance {
                    resource: "asr-lattice".to_string(),
                    variety: index.variety.clone(),
                    normalized_phone_sequence: String::new(),
                },
            });
        candidates.push(ConfusableMatch {
            lexeme: original_lexeme.clone(),
            relation: ConfusableRelationKind::Homophone,
        });
        candidates.sort_by(|left, right| {
            (left.lexeme.spelling != original)
                .cmp(&(right.lexeme.spelling != original))
                .then_with(|| relation_rank(left.relation).cmp(&relation_rank(right.relation)))
                .then_with(|| left.lexeme.spelling.cmp(&right.lexeme.spelling))
        });
        candidates.dedup_by(|left, right| left.lexeme.spelling == right.lexeme.spelling);
        candidates.truncate(
            config
                .max_candidates_per_word
                .min(config.max_total_candidates.saturating_sub(total)),
        );
        if candidates.len() <= 1 {
            continue;
        }

        let shared_pronunciation = candidates
            .iter()
            .find(|candidate| candidate.lexeme.spelling == original)
            .map(|candidate| candidate.lexeme.phonemes.clone())
            .unwrap_or_default();
        let target = LinguisticTarget::word(
            utterance_id_value.clone(),
            format!("word-{}", observation.word_index),
            TextRange {
                start: u32::try_from(observation.start_char).unwrap_or(u32::MAX),
                end: u32::try_from(observation.end_char).unwrap_or(u32::MAX),
            },
        );
        let acoustic_claim_id = LinguisticClaimId(format!(
            "homophone:{}:{}:acoustic",
            utterance_id, observation.word_index
        ));
        evidence.insert_claim(LinguisticClaim::acoustics(
            acoustic_claim_id.clone(),
            target.clone(),
            LinguisticClaimValue::Pronunciation {
                phonemes: shared_pronunciation
                    .iter()
                    .cloned()
                    .map(PhonemeId)
                    .collect(),
            },
            false,
            f64::from(observation.acoustic_score.clamp(0.0, 1.0)),
            ClaimRationale::new(
                "shared_acoustic_pronunciation",
                "One aligned acoustic observation supports every spelling alternative",
            )
            .with_attribute("acoustic_evidence_id", &observation.acoustic_evidence_id),
        )?)?;

        let mut slot = candidates
            .into_iter()
            .map(|candidate| {
                let mut sentence = original_words.clone();
                if observation.word_index < sentence.len() {
                    sentence[observation.word_index] = candidate.lexeme.spelling.clone();
                }
                let grammar = grammar_score(&parser, &sentence);
                let spelling = &candidate.lexeme.spelling;
                let lexical_frequency = context
                    .lexical_frequency_licensed
                    .then(|| lookup_score(&context.lexical_frequency, spelling));
                let acoustically_distinguishable = matches!(
                    candidate.relation,
                    ConfusableRelationKind::NearHomophone
                        | ConfusableRelationKind::Deletion
                        | ConfusableRelationKind::Insertion
                );
                let acoustic = if acoustically_distinguishable {
                    observation
                        .lattice_alternatives
                        .iter()
                        .find(|alternative| {
                            (spelling == "<deletion>"
                                && alternative.spelling.trim().is_empty()
                                && alternative.phonemes.is_empty())
                                || normalize_spelling(&alternative.spelling).as_deref()
                                    == Some(spelling.as_str())
                        })
                        .map(|alternative| alternative.acoustic_likelihood)
                        .unwrap_or(observation.acoustic_score)
                } else {
                    observation.acoustic_score
                };
                let components = HomophoneScoreComponents {
                    acoustic,
                    previous_word_head: lookup_score(&context.previous_word_head, spelling),
                    current_word_head: lookup_score(&context.current_word_head, spelling),
                    next_word_head: lookup_score(&context.next_word_head, spelling),
                    masked_cloze: lookup_score(&context.masked_cloze, spelling),
                    grammar,
                    lexical_frequency,
                    explicit_context: lookup_score(&context.explicit_context, spelling),
                    combined: 0.0,
                };
                let combined = weighted_score(&components, &config.weights);
                AlignedSpellingHypothesis {
                    spelling: spelling.clone(),
                    relation: candidate.relation,
                    pronunciation: candidate.lexeme.phonemes,
                    pronunciation_provenance: candidate.lexeme.provenance,
                    start_char: observation.start_char,
                    end_char: observation.end_char,
                    start_frame: observation.start_frame,
                    end_frame: observation.end_frame,
                    acoustic_evidence_id: observation.acoustic_evidence_id.clone(),
                    acoustically_distinguishable,
                    scores: HomophoneScoreComponents {
                        combined,
                        ..components
                    },
                    posterior: 0.0,
                    selected: false,
                }
            })
            .collect::<Vec<_>>();
        normalize_posteriors(&mut slot);
        slot.sort_by(|left, right| {
            right
                .posterior
                .total_cmp(&left.posterior)
                .then_with(|| left.spelling.cmp(&right.spelling))
        });

        let claim_ids = slot
            .iter()
            .map(|candidate| {
                LinguisticClaimId(format!(
                    "homophone:{}:{}:{}",
                    utterance_id,
                    observation.word_index,
                    stable_component(&candidate.spelling)
                ))
            })
            .collect::<Vec<_>>();
        for (candidate_index, candidate) in slot.iter().enumerate() {
            let claim_id = claim_ids[candidate_index].clone();
            let mut claim = LinguisticClaim::new(
                claim_id.clone(),
                target.clone(),
                LinguisticClaimKind::LexicalIdentity,
                LinguisticClaimValue::LexicalIdentity {
                    lexeme_id: candidate.spelling.clone(),
                },
                EvidenceProvenance {
                    source: EvidenceSource::Inference,
                    method: "combined-homophone-scorer".to_string(),
                    version: Some("1".to_string()),
                },
                ClaimConfidence::new(f64::from(candidate.posterior), Some("softmax-v1".into()))?,
                ClaimRationale::new(
                    "homophone_word_identity",
                    "Spelling ranked from shared acoustics and independent context evidence",
                )
                .with_attribute("acoustically_recognized", "false")
                .with_attribute("acoustic_evidence_id", &observation.acoustic_evidence_id),
            )?
            .with_support(acoustic_claim_id.clone());
            for conflict in &claim_ids {
                if conflict != &claim_id {
                    claim = claim.with_conflict(conflict.clone());
                }
            }
            evidence.insert_claim(claim)?;
        }
        total += slot.len();
        alternatives.push(slot);
    }

    let mut selected_words = original_words;
    let mut closest_margin = f32::INFINITY;
    let mut minimum_posterior = 1.0f32;
    for slot in &alternatives {
        if let Some(best) = slot.first() {
            if let Some(word_index) = observations.iter().position(|observation| {
                observation.start_char == best.start_char && observation.end_char == best.end_char
            }) {
                selected_words[word_index] = best.spelling.clone();
            }
            minimum_posterior = minimum_posterior.min(best.posterior);
            if let Some(second) = slot.get(1) {
                closest_margin = closest_margin.min(best.posterior - second.posterior);
            }
        }
    }
    let decision = if alternatives.is_empty() {
        HomophoneDecision::Committed
    } else if closest_margin < config.clarification_margin {
        HomophoneDecision::ClarificationRequired
    } else if minimum_posterior < config.commit_posterior {
        HomophoneDecision::Provisional
    } else {
        HomophoneDecision::Committed
    };
    if decision == HomophoneDecision::Committed {
        for slot in &mut alternatives {
            if let Some(best) = slot.first_mut() {
                best.selected = true;
            }
        }
        for (slot_index, slot) in alternatives.iter().enumerate() {
            let Some(best) = slot.first() else {
                continue;
            };
            let observation = observations
                .iter()
                .find(|observation| {
                    observation.start_char == best.start_char
                        && observation.end_char == best.end_char
                })
                .context("locating aligned homophone observation")?;
            let target = LinguisticTarget::word(
                utterance_id_value.clone(),
                format!("word-{}", observation.word_index),
                TextRange {
                    start: u32::try_from(best.start_char).unwrap_or(u32::MAX),
                    end: u32::try_from(best.end_char).unwrap_or(u32::MAX),
                },
            );
            evidence.resolve(
                ClaimResolutionId(format!("homophone:{utterance_id}:{slot_index}:resolution")),
                &target,
                LinguisticClaimKind::LexicalIdentity,
            )?;
        }
    }
    evidence.validate()?;
    Ok(AlignedHomophoneLattice {
        utterance_id: utterance_id.to_string(),
        transcript: transcript.to_string(),
        alternatives,
        decision,
        selected_transcript: (decision == HomophoneDecision::Committed)
            .then(|| selected_words.join(" ")),
        clarification: (decision == HomophoneDecision::ClarificationRequired).then(|| {
            "Multiple spellings remain plausible; request clarification before commitment."
                .to_string()
        }),
        linguistic_evidence: evidence,
    })
}

fn pronunciation_for_word(
    index: &ConfusablePronunciationIndex,
    spelling: &str,
) -> Option<ConfusableLexeme> {
    index
        .by_pronunciation
        .values()
        .flatten()
        .find(|lexeme| lexeme.spelling == spelling)
        .cloned()
}

fn lookup_score(scores: &BTreeMap<String, f32>, word: &str) -> f32 {
    scores
        .get(word)
        .copied()
        .unwrap_or_default()
        .clamp(0.0, 1.0)
}

fn grammar_score(parser: &VarietyGrammarParser, words: &[String]) -> f32 {
    let analysis = parser.parse(words, None);
    match analysis.status {
        GrammarAnalysisStatus::Failed => 0.0,
        GrammarAnalysisStatus::Partial => analysis
            .ranked_parses
            .first()
            .map(|parse| parse.rank * 0.75)
            .unwrap_or(0.25),
        GrammarAnalysisStatus::Complete => analysis
            .ranked_parses
            .first()
            .map(|parse| parse.rank)
            .unwrap_or(0.5),
    }
    .clamp(0.0, 1.0)
}

fn weighted_score(components: &HomophoneScoreComponents, weights: &HomophoneScoringWeights) -> f32 {
    components.acoustic * weights.acoustic
        + components.previous_word_head * weights.previous_word_head
        + components.current_word_head * weights.current_word_head
        + components.next_word_head * weights.next_word_head
        + components.masked_cloze * weights.masked_cloze
        + components.grammar * weights.grammar
        + components.lexical_frequency.unwrap_or_default() * weights.lexical_frequency
        + components.explicit_context * weights.explicit_context
}

fn normalize_posteriors(candidates: &mut [AlignedSpellingHypothesis]) {
    let max = candidates
        .iter()
        .map(|candidate| candidate.scores.combined)
        .fold(f32::NEG_INFINITY, f32::max);
    let denominator = candidates
        .iter()
        .map(|candidate| (candidate.scores.combined - max).exp())
        .sum::<f32>();
    for candidate in candidates {
        candidate.posterior = if denominator > 0.0 {
            (candidate.scores.combined - max).exp() / denominator
        } else {
            0.0
        };
    }
}

fn stable_component(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '-'
            }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HomophoneAugmentationConfig {
    pub seed: u64,
    pub max_candidates_per_sentence: usize,
    pub include_near_homophones: bool,
    pub include_deletion: bool,
    pub include_insertion: bool,
}

impl Default for HomophoneAugmentationConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            max_candidates_per_sentence: 4,
            include_near_homophones: false,
            include_deletion: true,
            include_insertion: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HomophoneAugmentationProvenance {
    pub split_key: String,
    pub source_word: Option<String>,
    pub candidate_word: Option<String>,
    pub relation: ConfusableRelationKind,
    pub pronunciation: Option<PronunciationProvenance>,
    pub generation_seed: u64,
    pub candidate_rank: usize,
    pub acoustic_evidence_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AugmentedSentence {
    pub text: String,
    pub provenance: HomophoneAugmentationProvenance,
}

pub fn augment_sentence(
    sentence_id: &str,
    text: &str,
    split_key: &str,
    index: &ConfusablePronunciationIndex,
    config: &HomophoneAugmentationConfig,
) -> Vec<AugmentedSentence> {
    let spans = word_spans(text);
    let mut candidates = Vec::new();
    for (word_index, (start, end, word)) in spans.iter().enumerate() {
        for candidate in index.candidates(word, config.include_near_homophones) {
            let mut edited = text.to_string();
            edited.replace_range(*start..*end, &candidate.lexeme.spelling);
            let hash = stable_hash(&format!(
                "{}|{}|{}|{}|{}",
                config.seed, split_key, sentence_id, word_index, candidate.lexeme.spelling
            ));
            candidates.push((
                hash,
                edited,
                Some(word.to_ascii_lowercase()),
                Some(candidate.lexeme.spelling.clone()),
                candidate.relation,
                Some(candidate.lexeme.provenance),
                word_index,
            ));
        }
    }
    if config.include_deletion && spans.len() >= 4 {
        let word_index = spans.len() / 2;
        let (start, end, word) = &spans[word_index];
        let mut edited = text.to_string();
        let remove_start = if *start > 0 && edited.as_bytes()[start - 1] == b' ' {
            start - 1
        } else {
            *start
        };
        edited.replace_range(remove_start..*end, "");
        candidates.push((
            stable_hash(&format!(
                "{}|{}|{}|{}|deletion",
                config.seed, split_key, sentence_id, word_index
            )),
            edited,
            Some(word.to_ascii_lowercase()),
            None,
            ConfusableRelationKind::Deletion,
            None,
            word_index,
        ));
    }
    if config.include_insertion && !spans.is_empty() {
        let word_index = spans.len() / 2;
        let insertion = "to";
        let at = spans[word_index].0;
        let mut edited = text.to_string();
        edited.insert_str(at, &format!("{insertion} "));
        candidates.push((
            stable_hash(&format!(
                "{}|{}|{}|{}|insertion",
                config.seed, split_key, sentence_id, word_index
            )),
            edited,
            None,
            Some(insertion.to_string()),
            ConfusableRelationKind::Insertion,
            None,
            word_index,
        ));
    }
    candidates.sort_by_key(|candidate| candidate.0);
    candidates.truncate(config.max_candidates_per_sentence);
    candidates
        .into_iter()
        .enumerate()
        .map(
            |(
                rank,
                (_, text, source_word, candidate_word, relation, pronunciation, word_index),
            )| AugmentedSentence {
                text,
                provenance: HomophoneAugmentationProvenance {
                    split_key: split_key.to_string(),
                    source_word,
                    candidate_word,
                    relation,
                    pronunciation,
                    generation_seed: config.seed,
                    candidate_rank: rank,
                    acoustic_evidence_id: format!("{sentence_id}:word-{word_index}"),
                },
            },
        )
        .collect()
}

fn word_spans(text: &str) -> Vec<(usize, usize, String)> {
    let mut out = Vec::new();
    let mut start = None;
    for (index, character) in text.char_indices() {
        if character.is_alphanumeric() || character == '\'' {
            start.get_or_insert(index);
        } else if let Some(word_start) = start.take() {
            out.push((word_start, index, text[word_start..index].to_string()));
        }
    }
    if let Some(word_start) = start {
        out.push((word_start, text.len(), text[word_start..].to_string()));
    }
    out
}

fn stable_hash(value: &str) -> u64 {
    value
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HomophoneEvaluationReport {
    pub examples: usize,
    pub top_1_accuracy: f32,
    pub top_k_accuracy: f32,
    pub expected_calibration_error: f32,
    pub oracle_recall: f32,
    pub repair_precision: f32,
    pub repair_recall: f32,
    pub acoustic_only_accuracy: f32,
    pub context_only_accuracy: f32,
    pub grammar_only_accuracy: f32,
    pub combined_accuracy: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HomophoneEvaluationCase {
    pub expected: String,
    pub original: String,
    pub alternatives: Vec<AlignedSpellingHypothesis>,
}

pub fn evaluate_homophone_cases(
    cases: &[HomophoneEvaluationCase],
    k: usize,
) -> HomophoneEvaluationReport {
    if cases.is_empty() {
        return HomophoneEvaluationReport::default();
    }
    let mut top_1 = 0usize;
    let mut top_k = 0usize;
    let mut oracle = 0usize;
    let mut repairs_predicted = 0usize;
    let mut repairs_expected = 0usize;
    let mut repairs_correct = 0usize;
    let mut calibration_error = 0.0f32;
    let mut acoustic = 0usize;
    let mut context = 0usize;
    let mut grammar = 0usize;
    let mut combined = 0usize;
    for case in cases {
        let mut alternatives = case.alternatives.clone();
        alternatives.sort_by(|left, right| right.posterior.total_cmp(&left.posterior));
        let selected = alternatives.first();
        let selected_word = selected.map(|candidate| candidate.spelling.as_str());
        let is_correct = selected_word == Some(case.expected.as_str());
        top_1 += usize::from(is_correct);
        combined += usize::from(is_correct);
        top_k += usize::from(
            alternatives
                .iter()
                .take(k.max(1))
                .any(|candidate| candidate.spelling == case.expected),
        );
        oracle += usize::from(
            alternatives
                .iter()
                .any(|candidate| candidate.spelling == case.expected),
        );
        let expected_repair = case.expected != case.original;
        let predicted_repair = selected_word.is_some_and(|word| word != case.original);
        repairs_expected += usize::from(expected_repair);
        repairs_predicted += usize::from(predicted_repair);
        repairs_correct += usize::from(expected_repair && is_correct);
        if let Some(selected) = selected {
            calibration_error += (selected.posterior - if is_correct { 1.0 } else { 0.0 }).abs();
        }
        acoustic += usize::from(
            best_by(&alternatives, |candidate| candidate.scores.acoustic)
                == Some(case.expected.as_str()),
        );
        context += usize::from(
            best_by(&alternatives, |candidate| {
                candidate.scores.previous_word_head
                    + candidate.scores.current_word_head
                    + candidate.scores.next_word_head
                    + candidate.scores.masked_cloze
                    + candidate.scores.explicit_context
            }) == Some(case.expected.as_str()),
        );
        grammar += usize::from(
            best_by(&alternatives, |candidate| candidate.scores.grammar)
                == Some(case.expected.as_str()),
        );
    }
    let examples = cases.len();
    let ratio = |count: usize, total: usize| {
        if total == 0 {
            0.0
        } else {
            count as f32 / total as f32
        }
    };
    HomophoneEvaluationReport {
        examples,
        top_1_accuracy: ratio(top_1, examples),
        top_k_accuracy: ratio(top_k, examples),
        expected_calibration_error: calibration_error / examples as f32,
        oracle_recall: ratio(oracle, examples),
        repair_precision: ratio(repairs_correct, repairs_predicted),
        repair_recall: ratio(repairs_correct, repairs_expected),
        acoustic_only_accuracy: ratio(acoustic, examples),
        context_only_accuracy: ratio(context, examples),
        grammar_only_accuracy: ratio(grammar, examples),
        combined_accuracy: ratio(combined, examples),
    }
}

fn best_by(
    alternatives: &[AlignedSpellingHypothesis],
    score: impl Fn(&AlignedSpellingHypothesis) -> f32,
) -> Option<&str> {
    alternatives
        .iter()
        .max_by(|left, right| {
            score(left)
                .total_cmp(&score(right))
                .then_with(|| right.spelling.cmp(&left.spelling))
        })
        .map(|candidate| candidate.spelling.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_index() -> ConfusablePronunciationIndex {
        ConfusablePronunciationIndex::from_words(COMMON_CONFUSABLE_WORDS, "en-US").unwrap()
    }

    #[test]
    fn required_common_sets_are_derived_as_alternatives() {
        let index = fixture_index();
        for (source, expected) in [
            ("to", &["two", "too"][..]),
            ("there", &["their", "they're"][..]),
            ("for", &["four"][..]),
            ("right", &["write"][..]),
            ("hear", &["here"][..]),
            ("its", &["it's"][..]),
            ("your", &["you're"][..]),
        ] {
            let candidates = index.candidates(source, false);
            for expected in expected {
                assert!(
                    candidates
                        .iter()
                        .any(|candidate| candidate.lexeme.spelling == *expected),
                    "{source} should include {expected}: {candidates:?}"
                );
            }
        }
    }

    #[test]
    fn required_runtime_fixtures_expose_aligned_lattices() {
        let index = fixture_index();
        for (source, expected) in [
            ("to", &["two", "too"][..]),
            ("there", &["their", "they're"][..]),
            ("for", &["four"][..]),
            ("right", &["write"][..]),
            ("hear", &["here"][..]),
            ("its", &["it's"][..]),
            ("your", &["you're"][..]),
        ] {
            let observation = AlignedWordObservation {
                word: source.into(),
                word_index: 0,
                start_char: 2,
                end_char: 2 + source.len(),
                start_frame: 11,
                end_frame: 19,
                acoustic_evidence_id: format!("fixture:{source}:frames-11-19"),
                observed_phonemes: Vec::new(),
                lattice_alternatives: Vec::new(),
                acoustic_score: 0.75,
            };
            let lattice = rank_homophone_hypotheses(
                &format!("fixture-{source}"),
                source,
                &[observation],
                &index,
                &HomophoneContextScores::default(),
                &HomophoneHypothesisConfig::default(),
            )
            .unwrap();
            let slot = &lattice.alternatives[0];
            for expected in expected {
                assert!(
                    slot.iter().any(|candidate| candidate.spelling == *expected),
                    "runtime lattice for {source} should include {expected}: {slot:?}"
                );
            }
            assert!(slot.len() > 1);
            assert!(slot.iter().all(|candidate| {
                candidate.start_char == 2
                    && candidate.end_char == 2 + source.len()
                    && candidate.start_frame == 11
                    && candidate.end_frame == 19
                    && candidate.acoustic_evidence_id == format!("fixture:{source}:frames-11-19")
            }));
        }
    }

    #[test]
    fn context_reranks_without_changing_shared_acoustic_evidence() {
        let index = fixture_index();
        let observations = vec![AlignedWordObservation {
            word: "there".into(),
            word_index: 0,
            start_char: 0,
            end_char: 5,
            start_frame: 10,
            end_frame: 20,
            acoustic_evidence_id: "frames:10-20".into(),
            observed_phonemes: Vec::new(),
            lattice_alternatives: Vec::new(),
            acoustic_score: 0.8,
        }];
        let mut context = HomophoneContextScores::default();
        context.explicit_context.insert("their".into(), 1.0);
        let lattice = rank_homophone_hypotheses(
            "utt",
            "there book",
            &observations,
            &index,
            &context,
            &HomophoneHypothesisConfig::default(),
        )
        .unwrap();
        let slot = &lattice.alternatives[0];
        assert_eq!(slot[0].spelling, "their");
        assert!(slot.iter().all(|candidate| {
            candidate.acoustic_evidence_id == "frames:10-20"
                && candidate.start_frame == 10
                && candidate.end_frame == 20
                && candidate.scores.acoustic == 0.8
        }));
        assert!(
            slot.iter()
                .filter(|candidate| !candidate.acoustically_distinguishable)
                .count()
                >= 2
        );
        assert!(
            lattice
                .linguistic_evidence
                .claims
                .iter()
                .filter(|claim| claim.kind == LinguisticClaimKind::Pronunciation)
                .count()
                == 1
        );
    }

    #[test]
    fn close_scores_require_clarification_and_do_not_resolve_identity() {
        let index = fixture_index();
        let observations = vec![AlignedWordObservation {
            word: "right".into(),
            word_index: 0,
            start_char: 0,
            end_char: 5,
            start_frame: 0,
            end_frame: 8,
            acoustic_evidence_id: "same".into(),
            observed_phonemes: Vec::new(),
            lattice_alternatives: Vec::new(),
            acoustic_score: 0.9,
        }];
        let config = HomophoneHypothesisConfig {
            clarification_margin: 0.2,
            ..HomophoneHypothesisConfig::default()
        };
        let lattice = rank_homophone_hypotheses(
            "utt",
            "right",
            &observations,
            &index,
            &HomophoneContextScores::default(),
            &config,
        )
        .unwrap();
        assert_eq!(lattice.decision, HomophoneDecision::ClarificationRequired);
        assert!(lattice.selected_transcript.is_none());
        assert!(lattice
            .alternatives
            .iter()
            .flatten()
            .all(|candidate| !candidate.selected));
        assert!(lattice.linguistic_evidence.resolutions.is_empty());
    }

    #[test]
    fn augmentation_is_bounded_reproducible_and_split_scoped() {
        let index = fixture_index();
        let config = HomophoneAugmentationConfig {
            max_candidates_per_sentence: 3,
            ..HomophoneAugmentationConfig::default()
        };
        let first = augment_sentence("utt", "GO TO THEIR HOUSE NOW", "train", &index, &config);
        let second = augment_sentence("utt", "GO TO THEIR HOUSE NOW", "train", &index, &config);
        assert_eq!(first, second);
        assert_eq!(first.len(), 3);
        assert!(first
            .iter()
            .all(|candidate| candidate.provenance.split_key == "train"));
        let validation = augment_sentence("utt", "GO TO THEIR HOUSE NOW", "valid", &index, &config);
        assert!(validation
            .iter()
            .all(|candidate| candidate.provenance.split_key == "valid"));
        assert_ne!(first, validation);
    }

    #[test]
    fn relation_types_and_metric_contributions_stay_separate() {
        let index = fixture_index();
        assert_eq!(
            index
                .candidates("its", false)
                .into_iter()
                .find(|candidate| candidate.lexeme.spelling == "it's")
                .unwrap()
                .relation,
            ConfusableRelationKind::Contraction
        );
        assert_eq!(
            index
                .candidates("for", false)
                .into_iter()
                .find(|candidate| candidate.lexeme.spelling == "four")
                .unwrap()
                .relation,
            ConfusableRelationKind::NumberForm
        );
        let near_index = ConfusablePronunciationIndex::from_words(["cat", "cap"], "en-US").unwrap();
        assert_eq!(
            near_index
                .candidates("cat", true)
                .into_iter()
                .find(|candidate| candidate.lexeme.spelling == "cap")
                .unwrap()
                .relation,
            ConfusableRelationKind::NearHomophone
        );

        let alternatives = ["there", "their"]
            .into_iter()
            .enumerate()
            .map(|(index, spelling)| AlignedSpellingHypothesis {
                spelling: spelling.into(),
                relation: ConfusableRelationKind::Homophone,
                pronunciation: vec!["DH".into(), "EH".into(), "R".into()],
                pronunciation_provenance: PronunciationProvenance {
                    resource: "fixture".into(),
                    variety: "en-US".into(),
                    normalized_phone_sequence: "DH EH R".into(),
                },
                start_char: 0,
                end_char: 5,
                start_frame: 0,
                end_frame: 5,
                acoustic_evidence_id: "shared".into(),
                acoustically_distinguishable: false,
                scores: HomophoneScoreComponents {
                    acoustic: if index == 0 { 0.9 } else { 0.8 },
                    previous_word_head: 0.0,
                    current_word_head: 0.0,
                    next_word_head: 0.0,
                    masked_cloze: if index == 1 { 1.0 } else { 0.0 },
                    grammar: if index == 1 { 1.0 } else { 0.0 },
                    lexical_frequency: None,
                    explicit_context: 0.0,
                    combined: if index == 1 { 2.0 } else { 0.0 },
                },
                posterior: if index == 1 { 0.9 } else { 0.1 },
                selected: index == 1,
            })
            .collect();
        let report = evaluate_homophone_cases(
            &[HomophoneEvaluationCase {
                expected: "their".into(),
                original: "there".into(),
                alternatives,
            }],
            2,
        );
        assert_eq!(report.top_1_accuracy, 1.0);
        assert_eq!(report.oracle_recall, 1.0);
        assert_eq!(report.acoustic_only_accuracy, 0.0);
        assert_eq!(report.context_only_accuracy, 1.0);
        assert_eq!(report.grammar_only_accuracy, 1.0);
        assert_eq!(report.combined_accuracy, 1.0);
    }

    #[test]
    fn external_lexicons_and_edit_operations_retain_provenance() {
        let mut index =
            ConfusablePronunciationIndex::from_words(Vec::<String>::new(), "en-US").unwrap();
        index.extend_resource_entries([
            PronunciationResourceEntry {
                spelling: "foo".into(),
                candidates: vec![vec!["f".into(), "u".into()]],
                resource: "wiktionary:test".into(),
                variety: "en-test".into(),
            },
            PronunciationResourceEntry {
                spelling: "phoo".into(),
                candidates: vec![vec!["f".into(), "u".into()]],
                resource: "wiktionary:test".into(),
                variety: "en-test".into(),
            },
        ]);
        let external = index
            .candidates("foo", false)
            .into_iter()
            .find(|candidate| candidate.lexeme.spelling == "phoo")
            .unwrap();
        assert_eq!(external.lexeme.provenance.resource, "wiktionary:test");
        assert_eq!(external.lexeme.provenance.variety, "en-test");

        let edits = augment_sentence(
            "edit-fixture",
            "ALPHA BRAVO CHARLIE DELTA",
            "train",
            &index,
            &HomophoneAugmentationConfig {
                max_candidates_per_sentence: 16,
                include_deletion: true,
                include_insertion: true,
                ..HomophoneAugmentationConfig::default()
            },
        );
        assert!(edits.iter().any(|candidate| {
            candidate.provenance.relation == ConfusableRelationKind::Deletion
        }));
        assert!(edits.iter().any(|candidate| {
            candidate.provenance.relation == ConfusableRelationKind::Insertion
        }));
    }
}
