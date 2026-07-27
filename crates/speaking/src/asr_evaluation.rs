//! Content-private, provider-neutral ASR quality and latency evaluation.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationWord {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrEvaluationCase {
    pub id: String,
    pub license: String,
    pub language: String,
    pub reference: String,
    pub words: Vec<EvaluationWord>,
    pub hypothesis: String,
    #[serde(default)]
    pub partials: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detected_language: Option<String>,
    #[serde(default)]
    pub predicted_words: Vec<EvaluationWord>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub audio_duration_ms: u64,
    pub recognition_duration_ms: u64,
    pub first_partial_ms: u64,
    pub endpoint_latency_ms: u64,
    pub peak_memory_bytes: u64,
    pub dropped_audio_chunks: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrCaseMetrics {
    pub id: String,
    pub word_error_rate: f64,
    pub character_error_rate: f64,
    pub language_id_correct: Option<bool>,
    pub diarization_error_rate: Option<f64>,
    pub mean_timestamp_error_ms: Option<f64>,
    pub partial_churn_rate: f64,
    pub endpoint_latency_ms: u64,
    pub first_partial_ms: u64,
    pub real_time_factor: f64,
    pub peak_memory_bytes: u64,
    pub dropped_audio_chunks: u64,
}

pub fn evaluate_asr_case(case: &AsrEvaluationCase) -> anyhow::Result<AsrCaseMetrics> {
    anyhow::ensure!(!case.id.is_empty(), "evaluation case ID is empty");
    anyhow::ensure!(
        !case.license.is_empty(),
        "evaluation fixture license is empty"
    );
    anyhow::ensure!(
        case.audio_duration_ms > 0,
        "evaluation audio duration is zero"
    );
    let reference_words = words(&case.reference);
    let hypothesis_words = words(&case.hypothesis);
    let reference_chars = characters(&case.reference);
    let hypothesis_chars = characters(&case.hypothesis);
    let partial_edits = case
        .partials
        .windows(2)
        .map(|pair| edit_distance(&characters(&pair[0]), &characters(&pair[1])))
        .sum::<usize>();
    let timestamp_errors = case
        .words
        .iter()
        .zip(&case.predicted_words)
        .flat_map(|(reference, predicted)| {
            [
                reference.start_ms.abs_diff(predicted.start_ms),
                reference.end_ms.abs_diff(predicted.end_ms),
            ]
        })
        .collect::<Vec<_>>();
    let speaker_pairs = case
        .words
        .iter()
        .zip(&case.predicted_words)
        .filter_map(|(reference, predicted)| {
            Some((reference.speaker.as_ref()?, predicted.speaker.as_ref()?))
        })
        .collect::<Vec<_>>();
    Ok(AsrCaseMetrics {
        id: case.id.clone(),
        word_error_rate: rate(
            edit_distance(&reference_words, &hypothesis_words),
            reference_words.len(),
        ),
        character_error_rate: rate(
            edit_distance(&reference_chars, &hypothesis_chars),
            reference_chars.len(),
        ),
        language_id_correct: case
            .detected_language
            .as_ref()
            .map(|detected| detected.eq_ignore_ascii_case(&case.language)),
        diarization_error_rate: (!speaker_pairs.is_empty()).then(|| {
            rate(
                speaker_pairs
                    .iter()
                    .filter(|(left, right)| left != right)
                    .count(),
                speaker_pairs.len(),
            )
        }),
        mean_timestamp_error_ms: (!timestamp_errors.is_empty())
            .then(|| timestamp_errors.iter().sum::<u64>() as f64 / timestamp_errors.len() as f64),
        partial_churn_rate: rate(partial_edits, reference_chars.len()),
        endpoint_latency_ms: case.endpoint_latency_ms,
        first_partial_ms: case.first_partial_ms,
        real_time_factor: case.recognition_duration_ms as f64 / case.audio_duration_ms as f64,
        peak_memory_bytes: case.peak_memory_bytes,
        dropped_audio_chunks: case.dropped_audio_chunks,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrEvaluationReport {
    pub schema_version: u16,
    pub fixture_count: usize,
    pub mean_word_error_rate: f64,
    pub mean_character_error_rate: f64,
    pub language_id_accuracy: Option<f64>,
    pub mean_partial_churn_rate: f64,
    pub mean_endpoint_latency_ms: f64,
    pub mean_real_time_factor: f64,
    pub maximum_peak_memory_bytes: u64,
    pub total_dropped_audio_chunks: u64,
    pub content_logged: bool,
    pub cases: Vec<AsrCaseMetrics>,
}

pub fn evaluate_asr_suite(cases: &[AsrEvaluationCase]) -> anyhow::Result<AsrEvaluationReport> {
    anyhow::ensure!(!cases.is_empty(), "ASR evaluation suite is empty");
    let metrics = cases
        .iter()
        .map(evaluate_asr_case)
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mean = |values: Vec<f64>| values.iter().sum::<f64>() / values.len() as f64;
    let language = metrics
        .iter()
        .filter_map(|metric| metric.language_id_correct)
        .collect::<Vec<_>>();
    Ok(AsrEvaluationReport {
        schema_version: 1,
        fixture_count: metrics.len(),
        mean_word_error_rate: mean(metrics.iter().map(|value| value.word_error_rate).collect()),
        mean_character_error_rate: mean(
            metrics
                .iter()
                .map(|value| value.character_error_rate)
                .collect(),
        ),
        language_id_accuracy: (!language.is_empty()).then(|| {
            language.iter().filter(|correct| **correct).count() as f64 / language.len() as f64
        }),
        mean_partial_churn_rate: mean(
            metrics
                .iter()
                .map(|value| value.partial_churn_rate)
                .collect(),
        ),
        mean_endpoint_latency_ms: mean(
            metrics
                .iter()
                .map(|value| value.endpoint_latency_ms as f64)
                .collect(),
        ),
        mean_real_time_factor: mean(metrics.iter().map(|value| value.real_time_factor).collect()),
        maximum_peak_memory_bytes: metrics
            .iter()
            .map(|value| value.peak_memory_bytes)
            .max()
            .unwrap_or(0),
        total_dropped_audio_chunks: metrics.iter().map(|value| value.dropped_audio_chunks).sum(),
        content_logged: false,
        cases: metrics,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivateAsrTrace {
    pub stage: String,
    pub event: String,
    pub session_id: String,
    pub sequence: u64,
    pub elapsed_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u64>,
    /// Always false in the default trace schema; content requires a separate,
    /// visible opt-in sink.
    pub content_logged: bool,
}

impl PrivateAsrTrace {
    pub fn new(
        stage: impl Into<String>,
        event: impl Into<String>,
        session_id: impl Into<String>,
        sequence: u64,
        elapsed_ms: u64,
    ) -> Self {
        Self {
            stage: stage.into(),
            event: event.into(),
            session_id: session_id.into(),
            sequence,
            elapsed_ms,
            count: None,
            content_logged: false,
        }
    }
}

fn words(value: &str) -> Vec<String> {
    value
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|character: char| !character.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn characters(value: &str) -> Vec<char> {
    value
        .chars()
        .filter(|character| !character.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn rate(errors: usize, total: usize) -> f64 {
    errors as f64 / total.max(1) as f64
}

fn edit_distance<T: Eq>(left: &[T], right: &[T]) -> usize {
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    for (left_index, left_item) in left.iter().enumerate() {
        let mut current = vec![left_index + 1; right.len() + 1];
        for (right_index, right_item) in right.iter().enumerate() {
            current[right_index + 1] = (previous[right_index + 1] + 1)
                .min(current[right_index] + 1)
                .min(previous[right_index] + usize::from(left_item != right_item));
        }
        previous = current;
    }
    previous[right.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct Suite {
        cases: Vec<AsrEvaluationCase>,
    }

    #[test]
    fn pinned_multilingual_baseline_is_reproducible_and_content_private() {
        let suite: Suite =
            serde_json::from_str(include_str!("../../../fixtures/asr/evaluation_v1.json")).unwrap();
        let first = evaluate_asr_suite(&suite.cases).unwrap();
        let second = evaluate_asr_suite(&suite.cases).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.fixture_count, 4);
        assert_eq!(first.language_id_accuracy, Some(0.75));
        assert!(!first.content_logged);
        assert_eq!(first.total_dropped_audio_chunks, 1);
        let encoded =
            serde_json::to_string(&PrivateAsrTrace::new("asr", "partial", "session:1", 2, 40))
                .unwrap();
        assert!(!encoded.contains("transcript"));
        assert!(!encoded.contains("audio"));
    }

    #[test]
    fn fixture_tags_cover_required_failure_dimensions() {
        let suite: Suite =
            serde_json::from_str(include_str!("../../../fixtures/asr/evaluation_v1.json")).unwrap();
        let tags = suite
            .cases
            .iter()
            .flat_map(|case| case.tags.iter().map(String::as_str))
            .collect::<std::collections::BTreeSet<_>>();
        for required in [
            "accent",
            "code_switching",
            "low_resource",
            "numbers_names",
            "noise",
            "clipping",
            "echo",
            "overlap",
            "long_silence",
            "long_speech",
            "malformed_stream",
        ] {
            assert!(tags.contains(required), "missing {required}");
        }
    }
}
