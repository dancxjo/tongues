use serde::{Deserialize, Serialize};

use crate::asr::AudioFrame;
use crate::data::lexicons::cmudict::{
    self, CmuPhoneme, CmuStress, PronunciationEntry, PronunciationStatus as CmudictStatus,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WordStreamId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WordId(pub u64);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimedWordStream {
    pub id: WordStreamId,
    pub source: WordStreamSource,
    pub words: Vec<WordNode>,
}

impl TimedWordStream {
    pub fn new(id: WordStreamId, source: WordStreamSource) -> Self {
        Self {
            id,
            source,
            words: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WordStreamSource {
    RecordedAudio,
    LiveAsr,
    GeneratedText,
    SyntheticSpeech,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordNode {
    pub id: WordId,
    pub text: String,
    pub lexical_span: Option<WordTextSpan>,
    pub timing: Option<WordTiming>,
    pub timing_confidence: Option<f32>,
    pub commitment: WordCommitment,
    pub boundary_source: BoundarySource,
    pub audio_ref: Option<AudioRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pronunciation: Option<WordPronunciation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WordTextSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WordTiming {
    pub start_ms: u64,
    pub end_ms: u64,
}

impl WordTiming {
    pub fn new(start_ms: u64, end_ms: u64) -> Option<Self> {
        (end_ms >= start_ms).then_some(Self { start_ms, end_ms })
    }

    pub fn duration_ms(&self) -> u64 {
        self.end_ms - self.start_ms
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WordCommitment {
    Hypothetical,
    StableText,
    Prepared,
    Playable,
    Played,
    Final,
    Confirmed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundarySource {
    Whisper,
    RefinedAcoustic,
    Predicted,
    PlaybackCursor,
    Manual,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioRef {
    pub buffer_id: String,
    pub byte_offset: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PronunciationLookupStatus {
    Exact,
    Normalized,
    Guessed,
    Missing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WordPronunciation {
    pub source: String,
    pub lookup: String,
    pub phonemes: Vec<String>,
    pub stress_pattern: String,
    pub status: PronunciationLookupStatus,
}

impl WordPronunciation {
    pub fn from_cmudict_entry(entry: &PronunciationEntry) -> Self {
        let phonemes = entry
            .candidates
            .first()
            .map(|candidate| candidate.iter().map(cmu_phoneme_token).collect())
            .unwrap_or_default();
        let stress_pattern = entry
            .candidates
            .first()
            .map(|candidate| {
                candidate
                    .iter()
                    .filter_map(|phoneme| phoneme.stress.map(stress_digit))
                    .collect()
            })
            .unwrap_or_default();
        Self {
            source: entry.source.to_string(),
            lookup: entry.lookup.clone(),
            phonemes,
            stress_pattern,
            status: pronunciation_status_from_cmudict(entry.status),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptWord {
    pub text: String,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
    pub confidence: Option<f32>,
}

pub fn transcript_to_word_stream(id: WordStreamId, words: &[TranscriptWord]) -> TimedWordStream {
    let mut byte_offset = 0usize;
    let word_nodes = words
        .iter()
        .enumerate()
        .map(|(i, word)| {
            let span_start = byte_offset;
            let span_end = span_start + word.text.len();
            byte_offset = span_end + 1;
            let timing = match (word.start_ms, word.end_ms) {
                (Some(start_ms), Some(end_ms)) => WordTiming::new(start_ms, end_ms),
                _ => None,
            };
            WordNode {
                id: WordId(i as u64 + 1),
                text: word.text.clone(),
                lexical_span: Some(WordTextSpan {
                    start: span_start,
                    end: span_end,
                }),
                timing,
                timing_confidence: word.confidence,
                commitment: WordCommitment::Final,
                boundary_source: if timing.is_some() {
                    BoundarySource::Whisper
                } else {
                    BoundarySource::Predicted
                },
                audio_ref: None,
                pronunciation: None,
            }
        })
        .collect();
    let mut stream = TimedWordStream {
        id,
        source: WordStreamSource::RecordedAudio,
        words: word_nodes,
    };
    attach_cmudict_pronunciations(&mut stream);
    stream
}

pub fn transcript_to_energy_snapped_word_stream(
    id: WordStreamId,
    words: &[TranscriptWord],
    audio: &[AudioFrame],
) -> TimedWordStream {
    let mut stream = transcript_to_word_stream(id, words);
    HeuristicAcousticWordBoundaryRefiner.refine(audio, &mut stream);
    stream
}

pub trait WordBoundaryRefiner {
    fn refine(&self, audio: &[AudioFrame], stream: &mut TimedWordStream);
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoopWordBoundaryRefiner;

impl WordBoundaryRefiner for NoopWordBoundaryRefiner {
    fn refine(&self, _audio: &[AudioFrame], _stream: &mut TimedWordStream) {}
}

#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicAcousticWordBoundaryRefiner;

const MIN_REFINED_WORD_DURATION_MS: u64 = 20;
const WHISPER_BOUNDARY_TIE_BREAK_BIAS: f32 = 0.02;

impl WordBoundaryRefiner for HeuristicAcousticWordBoundaryRefiner {
    fn refine(&self, audio: &[AudioFrame], stream: &mut TimedWordStream) {
        let Some(energy_per_ms) = energy_profile_per_ms(audio) else {
            return;
        };
        if stream.words.len() < 2 {
            return;
        }

        for index in 0..stream.words.len().saturating_sub(1) {
            let left = &stream.words[index];
            let right = &stream.words[index + 1];
            if left.boundary_source != BoundarySource::Whisper
                || right.boundary_source != BoundarySource::Whisper
            {
                continue;
            }
            let (Some(left_timing), Some(right_timing)) = (left.timing, right.timing) else {
                continue;
            };
            let min_boundary_ms = left_timing
                .start_ms
                .saturating_add(MIN_REFINED_WORD_DURATION_MS);
            let max_boundary_ms = right_timing
                .end_ms
                .saturating_sub(MIN_REFINED_WORD_DURATION_MS);
            if min_boundary_ms > max_boundary_ms {
                continue;
            }

            let original_boundary_ms = left_timing.end_ms.min(right_timing.start_ms);
            let audio_end_ms = energy_per_ms.len().saturating_sub(1) as u64;
            let search_start = min_boundary_ms;
            let search_end = max_boundary_ms.min(audio_end_ms);
            if search_start > search_end {
                continue;
            }

            let whisper_anchor_ms = original_boundary_ms.clamp(search_start, search_end);
            let search_span_ms = search_end.saturating_sub(search_start).max(1) as f32;
            let mut best_boundary_ms = whisper_anchor_ms;
            let mut best_score = f32::INFINITY;
            for boundary_ms in search_start..=search_end {
                let energy = smoothed_energy_at_ms(&energy_per_ms, boundary_ms as usize);
                let whisper_bias = boundary_ms.abs_diff(whisper_anchor_ms) as f32 / search_span_ms
                    * WHISPER_BOUNDARY_TIE_BREAK_BIAS;
                let score = energy + whisper_bias;
                if score < best_score {
                    best_score = score;
                    best_boundary_ms = boundary_ms;
                }
            }

            let (left_slice, right_slice) = stream.words.split_at_mut(index + 1);
            let left_mut = &mut left_slice[index];
            let right_mut = &mut right_slice[0];
            left_mut.timing = WordTiming::new(left_timing.start_ms, best_boundary_ms);
            right_mut.timing = WordTiming::new(best_boundary_ms, right_timing.end_ms);
            left_mut.boundary_source = BoundarySource::RefinedAcoustic;
            right_mut.boundary_source = BoundarySource::RefinedAcoustic;
        }
    }
}

pub fn attach_cmudict_pronunciations(stream: &mut TimedWordStream) {
    let pronouncer = cmudict::bundled();
    for word in &mut stream.words {
        if word.pronunciation.is_none() {
            let entry = pronouncer.lookup_entry(&word.text);
            word.pronunciation = Some(WordPronunciation::from_cmudict_entry(&entry));
        }
    }
}

fn cmu_phoneme_token(phoneme: &CmuPhoneme) -> String {
    phoneme.raw_symbol()
}

fn stress_digit(stress: CmuStress) -> char {
    match stress {
        CmuStress::Primary => '1',
        CmuStress::Secondary => '2',
        CmuStress::Unstressed => '0',
    }
}

fn pronunciation_status_from_cmudict(status: CmudictStatus) -> PronunciationLookupStatus {
    match status {
        CmudictStatus::Exact => PronunciationLookupStatus::Exact,
        CmudictStatus::Normalized => PronunciationLookupStatus::Normalized,
        CmudictStatus::Guessed => PronunciationLookupStatus::Guessed,
        CmudictStatus::Missing => PronunciationLookupStatus::Missing,
    }
}

fn energy_profile_per_ms(audio: &[AudioFrame]) -> Option<Vec<f32>> {
    let mut ms_energies = Vec::<f32>::new();
    let mut ms_counts = Vec::<u32>::new();
    let mut frame_offset_ms = 0u64;

    for frame in audio {
        if frame.sample_rate_hz == 0 || frame.channels == 0 {
            continue;
        }
        let channels = frame.channels as usize;
        let per_channel_samples = frame.samples.len() / channels;
        if per_channel_samples == 0 {
            continue;
        }
        let frame_duration_ms = per_channel_samples as u64 * 1_000 / frame.sample_rate_hz as u64;
        for sample_idx in 0..per_channel_samples {
            let mut mono = 0.0f32;
            for channel in 0..channels {
                mono += frame.samples[sample_idx * channels + channel].abs();
            }
            mono /= channels as f32;
            let ms_idx =
                frame_offset_ms + (sample_idx as u64 * 1_000 / frame.sample_rate_hz as u64);
            let ms_idx = ms_idx as usize;
            if ms_idx >= ms_energies.len() {
                ms_energies.resize(ms_idx + 1, 0.0);
                ms_counts.resize(ms_idx + 1, 0);
            }
            ms_energies[ms_idx] += mono;
            ms_counts[ms_idx] += 1;
        }
        frame_offset_ms = frame_offset_ms.saturating_add(frame_duration_ms);
    }

    if ms_energies.is_empty() {
        return None;
    }
    for (energy, count) in ms_energies.iter_mut().zip(ms_counts.iter()) {
        if *count > 0 {
            *energy /= *count as f32;
        }
    }
    Some(ms_energies)
}

fn smoothed_energy_at_ms(energy_per_ms: &[f32], ms_idx: usize) -> f32 {
    let start = ms_idx.saturating_sub(5);
    let end = (ms_idx + 5).min(energy_per_ms.len().saturating_sub(1));
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for value in &energy_per_ms[start..=end] {
        sum += *value;
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_export_with_timing() {
        let words = vec![
            TranscriptWord {
                text: "hello".into(),
                start_ms: Some(100),
                end_ms: Some(500),
                confidence: Some(0.95),
            },
            TranscriptWord {
                text: "world".into(),
                start_ms: Some(550),
                end_ms: Some(900),
                confidence: Some(0.88),
            },
        ];
        let stream = transcript_to_word_stream(WordStreamId(1), &words);
        assert_eq!(stream.source, WordStreamSource::RecordedAudio);
        assert_eq!(stream.words.len(), 2);
        assert_eq!(
            stream.words[0].timing,
            Some(WordTiming {
                start_ms: 100,
                end_ms: 500
            })
        );
        assert!(
            stream
                .words
                .iter()
                .all(|word| word.boundary_source == BoundarySource::Whisper)
        );
    }

    #[test]
    fn missing_timing_uses_predicted_boundary() {
        let stream = transcript_to_word_stream(
            WordStreamId(1),
            &[TranscriptWord {
                text: "hello".into(),
                start_ms: None,
                end_ms: None,
                confidence: None,
            }],
        );
        assert_eq!(stream.words[0].boundary_source, BoundarySource::Predicted);
        assert!(stream.words[0].timing.is_none());
    }

    #[test]
    fn heuristic_refiner_moves_boundary_toward_local_silence() {
        let words = vec![
            TranscriptWord {
                text: "hello".into(),
                start_ms: Some(0),
                end_ms: Some(330),
                confidence: Some(0.9),
            },
            TranscriptWord {
                text: "world".into(),
                start_ms: Some(330),
                end_ms: Some(800),
                confidence: Some(0.9),
            },
        ];
        let mut stream = transcript_to_word_stream(WordStreamId(9), &words);
        let mut samples = vec![1.0f32; 900];
        samples[390..470].fill(0.0);
        let audio = vec![AudioFrame {
            sample_rate_hz: 1_000,
            channels: 1,
            samples,
        }];
        HeuristicAcousticWordBoundaryRefiner.refine(&audio, &mut stream);
        let left = stream.words[0].timing.expect("left timing");
        let right = stream.words[1].timing.expect("right timing");
        assert_eq!(left.end_ms, right.start_ms);
        assert!((390..=470).contains(&left.end_ms));
    }
}
