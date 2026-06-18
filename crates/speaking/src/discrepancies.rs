use serde::{Deserialize, Serialize};

use crate::data::lexicons::cmudict::{CmuPhoneme, CmuStress};
use crate::data::notation::arpabet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PronouncerResult {
    pub source: String,
    pub output: Option<String>,
    pub status: PronouncerStatus,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PronouncerStatus {
    Found,
    Missing,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PronunciationDiscrepancy {
    pub word: String,
    pub sources: Vec<PronouncerResult>,
    pub comparison_keys: Vec<String>,
    pub edit_distance_max: usize,
}

pub trait PronunciationProvider {
    fn name(&self) -> &str;
    fn pronounce(&mut self, word: &str) -> PronouncerResult;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscrepancyProgress {
    pub word_index: usize,
    pub words_total: usize,
    pub provider_index: usize,
    pub providers_total: usize,
    pub word: String,
    pub provider: String,
}

pub fn find_pronunciation_discrepancies(
    words: &[String],
    providers: &mut [&mut dyn PronunciationProvider],
) -> Vec<PronunciationDiscrepancy> {
    find_pronunciation_discrepancies_with_progress(words, providers, |_| {})
}

pub fn find_pronunciation_discrepancies_with_progress<F>(
    words: &[String],
    providers: &mut [&mut dyn PronunciationProvider],
    mut progress: F,
) -> Vec<PronunciationDiscrepancy>
where
    F: FnMut(DiscrepancyProgress),
{
    let mut records = Vec::new();
    let words_total = words.len();
    let providers_total = providers.len();

    for (word_index, word) in words.iter().enumerate() {
        let mut sources = Vec::with_capacity(providers_total);
        for (provider_index, provider) in providers.iter_mut().enumerate() {
            let provider_name = provider.name().to_string();
            let source = provider.pronounce(word);
            progress(DiscrepancyProgress {
                word_index: word_index + 1,
                words_total,
                provider_index: provider_index + 1,
                providers_total,
                word: word.clone(),
                provider: provider_name,
            });
            sources.push(source);
        }

        let mut comparison_keys = sources
            .iter()
            .filter_map(|source| source.output.as_deref())
            .map(pronunciation_comparison_key)
            .filter(|key| !key.is_empty())
            .collect::<Vec<_>>();
        comparison_keys.sort();
        comparison_keys.dedup();

        if comparison_keys.len() > 1 {
            let edit_distance_max = max_pairwise_edit_distance(&comparison_keys);
            records.push(PronunciationDiscrepancy {
                word: word.clone(),
                sources,
                comparison_keys,
                edit_distance_max,
            });
        }
    }

    records.sort_by(|left, right| {
        right
            .edit_distance_max
            .cmp(&left.edit_distance_max)
            .then_with(|| left.word.cmp(&right.word))
    });
    records
}

pub fn pronunciation_comparison_key(value: &str) -> String {
    let no_length = value.replace('ː', "");
    let no_syllable_marks = no_length.replace('.', "");
    no_syllable_marks
        .chars()
        .filter(|c| !matches!(c, 'ˈ' | 'ˌ' | '/' | '[' | ']' | ' '))
        .collect::<String>()
        .replace('ɝ', "ɚ")
        .replace("iə", "iɚ")
        .replace("uə", "uɚ")
        .replace("əɹ", "ɚ")
        .replace("lɹ", "lɚ")
}

pub fn edit_distance_chars(left: &str, right: &str) -> usize {
    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();
    let mut prev: Vec<usize> = (0..=right.len()).collect();
    let mut curr = vec![0; right.len() + 1];

    for (i, lc) in left.iter().enumerate() {
        curr[0] = i + 1;
        for (j, rc) in right.iter().enumerate() {
            let substitution = prev[j] + usize::from(lc != rc);
            let insertion = curr[j] + 1;
            let deletion = prev[j + 1] + 1;
            curr[j + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[right.len()]
}

pub fn max_pairwise_edit_distance(keys: &[String]) -> usize {
    let mut max_distance = 0usize;
    for (index, left) in keys.iter().enumerate() {
        for right in keys.iter().skip(index + 1) {
            max_distance = max_distance.max(edit_distance_chars(left, right));
        }
    }
    max_distance
}

pub fn cmu_phonemes_to_ipa(phonemes: &[CmuPhoneme]) -> String {
    let mut text = String::new();
    let mut has_seen_vowel = false;
    let mut consonant_cluster_start = 0usize;
    let mut consonant_cluster_len = 0usize;

    for phoneme in phonemes {
        let Some(entry) = arpabet::entry(&phoneme.base) else {
            continue;
        };
        if entry.syllabic {
            let stress_index = if has_seen_vowel {
                if consonant_cluster_len > 2 {
                    text.char_indices()
                        .nth(
                            text[..consonant_cluster_start].chars().count() + consonant_cluster_len
                                - 2,
                        )
                        .map(|(index, _)| index)
                        .unwrap_or(text.len())
                } else {
                    consonant_cluster_start
                }
            } else {
                0
            };
            match phoneme.stress {
                Some(CmuStress::Primary) => text.insert(stress_index, 'ˈ'),
                Some(CmuStress::Secondary) => text.insert(stress_index, 'ˌ'),
                Some(CmuStress::Unstressed) if has_seen_vowel && !text.is_empty() => {
                    text.insert(stress_index, '.')
                }
                _ if has_seen_vowel && !text.is_empty() => text.insert(stress_index, '.'),
                _ => {}
            }
            has_seen_vowel = true;
            consonant_cluster_start = text.len();
            consonant_cluster_len = 0;
        } else if consonant_cluster_len == 0 {
            consonant_cluster_start = text.len();
            consonant_cluster_len = 1;
        } else {
            consonant_cluster_len += 1;
        }

        if let Some(phone) = arpabet::reduced_phone_for_cmu(&phoneme.base, phoneme.stress) {
            text.push_str(
                phone
                    .as_str()
                    .strip_prefix("ipa.phone.")
                    .unwrap_or(phone.as_str()),
            );
        } else {
            text.push_str(entry.ipa);
        }
    }

    text
}

pub fn render_discrepancy_markdown(
    records: &[PronunciationDiscrepancy],
    provider_names: &[String],
    checked_count: usize,
) -> String {
    let mut out = String::new();
    out.push_str("# Pronunciation Discrepancies\n\n");
    out.push_str(&format!(
        "Checked {} words across {} pronouncers. Found {} substantive discrepancies after comparison-key normalization.\n\n",
        checked_count,
        provider_names.join(", "),
        records.len()
    ));
    out.push_str("| Word | Max edit | ");
    for name in provider_names {
        out.push_str(&format!("{} | ", markdown_escape(name)));
    }
    out.push_str("Compare keys |\n");
    out.push_str("| --- | ---: | ");
    for _ in provider_names {
        out.push_str("--- | ");
    }
    out.push_str("--- |\n");

    for record in records {
        out.push_str(&format!(
            "| {} | {} | ",
            markdown_escape(&record.word),
            record.edit_distance_max
        ));
        for name in provider_names {
            let cell = record
                .sources
                .iter()
                .find(|source| &source.source == name)
                .map(markdown_cell_for_result)
                .unwrap_or_else(|| "missing".to_string());
            out.push_str(&format!("{} | ", cell));
        }
        out.push_str(&format!(
            "{} |\n",
            markdown_escape(&record.comparison_keys.join("<br>"))
        ));
    }

    out
}

fn markdown_cell_for_result(result: &PronouncerResult) -> String {
    match result.status {
        PronouncerStatus::Found => result
            .output
            .as_deref()
            .map(markdown_escape)
            .unwrap_or_else(|| "found".to_string()),
        PronouncerStatus::Missing => "missing".to_string(),
        PronouncerStatus::Error => {
            let note = result.note.as_deref().unwrap_or("error");
            format!("error: {}", markdown_escape(note))
        }
    }
}

fn markdown_escape(value: &str) -> String {
    value
        .replace('|', "\\|")
        .replace('\n', "<br>")
        .replace('\r', "")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_key_ignores_common_notation_differences() {
        assert_eq!(
            pronunciation_comparison_key("ˈziː.ɡɚ"),
            pronunciation_comparison_key("/ˈziɡəɹ/")
        );
    }

    #[test]
    fn cmu_ipa_renders_stress_and_reduced_vowels() {
        let phonemes = ["L", "OW1", "D", "S", "T", "OW2", "N"]
            .into_iter()
            .map(CmuPhoneme::parse)
            .collect::<Vec<_>>();
        assert_eq!(cmu_phonemes_to_ipa(&phonemes), "ˈloʊdˌstoʊn");
    }
}
