//! Data pipeline for pronlex sequence-to-sequence translation.
//!
//! Handles CMUdict parsing, parallelized IPA phonemicization, splitting,
//! vocabulary construction, and seq2seq batch collation.

use std::sync::{Arc, Mutex};
use std::thread;

use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

use pronlex_core::{Vocab, BOS_ID, EOS_ID, PAD_ID, PM2S_ID, S2PM_ID};
use speech::{EnglishPhonemicizer, PhonemicizeRequest, Phonemicizer, VarietyId};

// ── Lexeme ─────────────────────────────────────────────────────────────────

/// Multimodal pronunciation entry storing spelling, broad IPA phonemes, and narrow IPA phones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lexeme {
    /// Orthographic spelling of the base word.
    pub base_word: String,
    /// Broad IPA phoneme string.
    pub phonemes: String,
}

// ── CMUdict parsing and parallel IPA generation ────────────────────────────

/// Parse base words from a CMUdict `.dict` file, keeping only standard alphabetical words.
pub fn parse_cmudict(text: &str) -> Vec<String> {
    let mut base_words = std::collections::BTreeSet::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(";;;") {
            continue;
        }
        let mut tokens = line.split_ascii_whitespace();
        let raw_word = match tokens.next() {
            Some(w) => w,
            None => continue,
        };

        // Extract base word by removing alternate suffix like "(2)"
        let base_word = if let Some(open) = raw_word.find('(') {
            raw_word[..open].to_lowercase()
        } else {
            raw_word.to_lowercase()
        };

        // Only keep alphabetical base words with optional apostrophes/hyphens
        if !base_word.is_empty()
            && base_word
                .chars()
                .all(|c| c.is_alphabetic() || c == '\'' || c == '-')
        {
            base_words.insert(base_word);
        }
    }
    base_words.into_iter().collect()
}

/// Phonemicize a single base word into its broad and narrow IPA string representations.
pub fn phonemicize_word(base_word: &str) -> Option<(String, String)> {
    let phonemicizer = EnglishPhonemicizer;
    let phonemicized = phonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: base_word.to_string(),
            variety: VarietyId("en-US".to_string()),
            style: None,
        })
        .ok()?;

    if phonemicized
        .warnings
        .iter()
        .any(|w| w.kind == speech::PronunciationWarningKind::GuessedWord)
    {
        return None;
    }

    let mut words: Vec<(usize, Vec<speech::Syllable>)> = Vec::new();
    for syllable in phonemicized.syllables.iter() {
        if let Some(first_phone) = syllable.phones.first() {
            if let Some(word_idx) = token_word_index(&first_phone.features) {
                if let Some(last_word) = words.last_mut() {
                    if last_word.0 == word_idx {
                        last_word.1.push(syllable.clone());
                        continue;
                    }
                }
                words.push((word_idx, vec![syllable.clone()]));
            }
        }
    }

    let mut broad_words = Vec::new();
    let mut narrow_words = Vec::new();
    for (_, word_syllables) in words {
        let broad_ipa = syllables_to_phonemes_ipa(
            &word_syllables,
            &phonemicized.phonemes,
            &phonemicized.variety,
        );
        let narrow_ipa = syllables_to_ipa_formatted(&word_syllables);
        if !broad_ipa.is_empty() {
            broad_words.push(broad_ipa);
        }
        if !narrow_ipa.is_empty() {
            narrow_words.push(narrow_ipa);
        }
    }

    if broad_words.is_empty() || narrow_words.is_empty() {
        None
    } else {
        Some((broad_words.join(" "), narrow_words.join(" ")))
    }
}

/// Run multi-threaded parallel IPA phonemicization for a list of base words.
pub fn phonemicize_lexemes(base_words: Vec<String>) -> Vec<Lexeme> {
    let base_words = Arc::new(base_words);
    let results = Arc::new(Mutex::new(Vec::new()));
    let num_threads = 20;
    let mut handles = Vec::new();

    let chunk_size = (base_words.len() + num_threads - 1) / num_threads;

    for t in 0..num_threads {
        let base_words = Arc::clone(&base_words);
        let results = Arc::clone(&results);
        let start_idx = t * chunk_size;
        let end_idx = (start_idx + chunk_size).min(base_words.len());

        if start_idx >= base_words.len() {
            break;
        }

        let handle = thread::spawn(move || {
            let mut local_results = Vec::new();
            for i in start_idx..end_idx {
                let word = &base_words[i];
                if let Some((phonemes, _phones)) = phonemicize_word(word) {
                    local_results.push(Lexeme {
                        base_word: word.clone(),
                        phonemes,
                    });
                }
            }
            let mut guard = results.lock().unwrap();
            guard.extend(local_results);
        });
        handles.push(handle);
    }

    for h in handles {
        let _ = h.join();
    }

    let guard = results.lock().unwrap();
    guard.clone()
}

// ── Speech Crate IPA Formatter Helpers ─────────────────────────────────────

fn find_phoneme_for_phone(
    phone: &speech::PhoneToken,
    phonemes: &[speech::PhonemeToken],
) -> Option<speech::PhonemeId> {
    for phoneme_token in phonemes {
        for realized_phone in &phoneme_token.realized_as {
            if realized_phone.phone == phone.phone
                && realized_phone.features == phone.features
                && realized_phone.span == phone.span
            {
                if let speech::Spec::Known(ref id) = phoneme_token.phoneme {
                    return Some(id.clone());
                }
            }
        }
    }
    None
}

fn phone_ipa(phone: &speech::PhoneToken) -> &str {
    match &phone.phone {
        speech::Spec::Known(id) => id
            .as_str()
            .strip_prefix("ipa.phone.")
            .unwrap_or(id.as_str()),
        _ => "",
    }
}

fn syllables_to_phonemes_ipa(
    syllables: &[speech::Syllable],
    phonemes: &[speech::PhonemeToken],
    variety: &speech::VarietyId,
) -> String {
    syllables
        .iter()
        .enumerate()
        .map(|(index, syllable)| {
            let mut text = String::new();
            let mut has_stress_mark = false;
            let stress_char = match syllable.stress {
                speech::Spec::Known(speech::Stress::Primary) => {
                    has_stress_mark = true;
                    Some('ˈ')
                }
                speech::Spec::Known(speech::Stress::Secondary) => {
                    has_stress_mark = true;
                    Some('ˌ')
                }
                _ => None,
            };

            if index > 0 && !has_stress_mark {
                text.push('.');
            }
            if let Some(c) = stress_char {
                text.push(c);
            }
            for phone in &syllable.phones {
                if let Some(phoneme_id) = find_phoneme_for_phone(phone, phonemes) {
                    let symbol = speech::phoneme_default_phone_display_symbol(&phoneme_id, variety);
                    text.push_str(&symbol);
                } else {
                    text.push_str(phone_ipa(phone));
                }
            }
            text
        })
        .collect()
}

fn syllables_to_ipa_formatted(syllables: &[speech::Syllable]) -> String {
    syllables
        .iter()
        .enumerate()
        .map(|(index, syllable)| {
            let mut text = String::new();
            let mut has_stress_mark = false;
            let stress_char = match syllable.stress {
                speech::Spec::Known(speech::Stress::Primary) => {
                    has_stress_mark = true;
                    Some('ˈ')
                }
                speech::Spec::Known(speech::Stress::Secondary) => {
                    has_stress_mark = true;
                    Some('ˌ')
                }
                _ => None,
            };

            if index > 0 && !has_stress_mark {
                text.push('.');
            }
            if let Some(c) = stress_char {
                text.push(c);
            }
            for phone in &syllable.phones {
                text.push_str(phone_ipa(phone));
            }
            text
        })
        .collect()
}

fn token_word_index(features: &speech::FeatureBundle) -> Option<usize> {
    let value = features
        .values
        .get(&speech::FeatureId("orthography.word_index".into()))?;
    match value {
        speech::Spec::Known(speech::FeatureValue::Number(value))
            if value.is_finite() && *value >= 0.0 =>
        {
            Some(*value as usize)
        }
        _ => None,
    }
}

// ── Vocabulary builder ─────────────────────────────────────────────────────

/// Build the full unified vocabulary from a collection of lexemes.
pub fn build_vocab(lexemes: &[Lexeme]) -> Vocab {
    let mut words = Vec::new();
    let mut phonemes = Vec::new();

    for lex in lexemes {
        words.push(lex.base_word.clone());
        words.push(lex.base_word.to_uppercase());
        words.push(initial_cap(&lex.base_word));
        phonemes.push(lex.phonemes.clone());
    }

    Vocab::build(&words, &phonemes, &[])
}

fn initial_cap(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => first
            .to_uppercase()
            .chain(chars.flat_map(char::to_lowercase))
            .collect(),
        None => String::new(),
    }
}

// ── Data splitting ─────────────────────────────────────────────────────────

/// Split lexemes into train / valid / test sets.
pub fn split_by_base_word<R: Rng>(
    lexemes: &[Lexeme],
    train_frac: f64,
    valid_frac: f64,
    rng: &mut R,
) -> (Vec<Lexeme>, Vec<Lexeme>, Vec<Lexeme>) {
    let mut lexemes = lexemes.to_vec();
    lexemes.shuffle(rng);

    let n = lexemes.len();
    let train_end = (n as f64 * train_frac).round() as usize;
    let valid_end = train_end + (n as f64 * valid_frac).round() as usize;

    let mut train = Vec::new();
    let mut valid = Vec::new();
    let mut test = Vec::new();

    for (i, lex) in lexemes.into_iter().enumerate() {
        if i < train_end {
            train.push(lex);
        } else if i < valid_end {
            valid.push(lex);
        } else {
            test.push(lex);
        }
    }

    (train, valid, test)
}

/// No-op verification helper for backward compatibility.
pub fn check_split_leakage(_train: &[Lexeme], _valid: &[Lexeme], _test: &[Lexeme]) -> Vec<String> {
    Vec::new()
}

// ── Seq2Seq Task Representation & Collation ────────────────────────────────

/// Available translation directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Task {
    S2Pm,
    Pm2S,
}

impl Task {
    /// Get the vocabulary ID corresponding to this task's prefix token.
    pub fn get_prefix_id(&self) -> u32 {
        match self {
            Task::S2Pm => S2PM_ID,
            Task::Pm2S => PM2S_ID,
        }
    }

    /// Randomly sample a task from all available tasks.
    pub fn sample<R: Rng>(rng: &mut R) -> Self {
        let tasks = [Task::S2Pm, Task::Pm2S];
        *tasks.choose(rng).unwrap()
    }

    /// Parse a task direction from a string slice.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "s2pm" => Some(Task::S2Pm),
            "pm2s" => Some(Task::Pm2S),
            _ => None,
        }
    }
}

/// A single translation training example.
#[derive(Debug, Clone)]
pub struct Seq2SeqExample {
    /// Token IDs for source sequence (starts with Task Token).
    pub src_ids: Vec<u32>,
    /// Token IDs for target decoder input (starts with BOS).
    pub tgt_in_ids: Vec<u32>,
    /// Token IDs for target decoder loss output (ends with EOS).
    pub tgt_out_ids: Vec<u32>,
}

/// Convert a Lexeme to a translation example.
pub fn make_seq2seq_example(lexeme: &Lexeme, task: Task, vocab: &Vocab) -> Seq2SeqExample {
    let (src_str, tgt_str) = match task {
        Task::S2Pm => (&lexeme.base_word, &lexeme.phonemes),
        Task::Pm2S => (&lexeme.phonemes, &lexeme.base_word),
    };

    let mut src_ids = vec![task.get_prefix_id()];
    src_ids.extend(vocab.encode_string(src_str));

    let mut tgt_in_ids = vec![BOS_ID];
    tgt_in_ids.extend(vocab.encode_string(tgt_str));

    let mut tgt_out_ids = vocab.encode_string(tgt_str);
    tgt_out_ids.push(EOS_ID);

    Seq2SeqExample {
        src_ids,
        tgt_in_ids,
        tgt_out_ids,
    }
}

/// Padded batch ready for the sequence-to-sequence model.
#[derive(Debug, Clone)]
pub struct Batch {
    /// `[batch, max_src_len]` source token IDs.
    pub src_ids: Vec<Vec<i32>>,
    /// `[batch, max_tgt_len]` target input token IDs.
    pub tgt_in_ids: Vec<Vec<i32>>,
    /// `[batch, max_tgt_len]` target output token IDs.
    pub tgt_out_ids: Vec<Vec<i32>>,
    /// `[batch, max_src_len]` padding mask (true for padding).
    pub src_pad_mask: Vec<Vec<bool>>,
    /// `[batch, max_tgt_len]` padding mask (true for padding).
    pub tgt_pad_mask: Vec<Vec<bool>>,
    /// Number of examples in the batch.
    pub size: usize,
}

/// Collate sequence-to-sequence examples into a padded batch.
pub fn collate_batch(examples: &[Seq2SeqExample], max_src_len: usize, max_tgt_len: usize) -> Batch {
    let size = examples.len();
    let mut src_ids = vec![vec![PAD_ID as i32; max_src_len]; size];
    let mut tgt_in_ids = vec![vec![PAD_ID as i32; max_tgt_len]; size];
    let mut tgt_out_ids = vec![vec![PAD_ID as i32; max_tgt_len]; size];

    let mut src_pad_mask = vec![vec![true; max_src_len]; size];
    let mut tgt_pad_mask = vec![vec![true; max_tgt_len]; size];

    for (i, ex) in examples.iter().enumerate() {
        for (j, &id) in ex.src_ids.iter().enumerate().take(max_src_len) {
            src_ids[i][j] = id as i32;
            src_pad_mask[i][j] = false;
        }
        for (j, &id) in ex.tgt_in_ids.iter().enumerate().take(max_tgt_len) {
            tgt_in_ids[i][j] = id as i32;
            tgt_pad_mask[i][j] = false;
        }
        for (j, &id) in ex.tgt_out_ids.iter().enumerate().take(max_tgt_len) {
            tgt_out_ids[i][j] = id as i32;
        }
    }

    Batch {
        src_ids,
        tgt_in_ids,
        tgt_out_ids,
        src_pad_mask,
        tgt_pad_mask,
        size,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_cmu_parsing_base_words() {
        let text = ";;; comment\nHELLO H EH1 L OW0\nWORLD(2) W ER1 L D\n12345 NOPE\n";
        let base_words = parse_cmudict(text);
        assert_eq!(base_words, vec!["hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn test_split_no_leakage() {
        let lex = vec![
            Lexeme {
                base_word: "cat".into(),
                phonemes: "kæt".into(),
            },
            Lexeme {
                base_word: "dog".into(),
                phonemes: "dɔɡ".into(),
            },
        ];
        let mut rng = StdRng::seed_from_u64(42);
        let (train, valid, test) = split_by_base_word(&lex, 0.5, 0.5, &mut rng);
        assert_eq!(train.len() + valid.len() + test.len(), 2);
    }

    #[test]
    fn build_vocab_includes_spelling_case_variants() {
        let lex = vec![Lexeme {
            base_word: "farkle".into(),
            phonemes: "ˈfɑɹ.kəl".into(),
        }];

        let vocab = build_vocab(&lex);

        for spelling in ["farkle", "Farkle", "FARKLE"] {
            assert!(
                vocab
                    .encode_string(spelling)
                    .iter()
                    .all(|&id| id != pronlex_core::UNK_ID),
                "{spelling} should not encode to UNK"
            );
        }
    }
}
