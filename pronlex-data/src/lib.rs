//! Data pipeline for pronlex: CMUdict parsing, masking, splitting, batching.
//!
//! # Data flow
//! ```text
//! CMUdict file  →  parse_cmudict()  →  Vec<Lexeme>
//!                  build_vocab()    →  Vocab
//!                  split_data()     →  (train, valid, test) Vec<Lexeme>
//!   (saved to  train/valid/test.jsonl + vocab.json)
//!
//! During training:
//!   batch of Lexemes  →  apply_mask()  →  MaskedExample
//!   Vec<MaskedExample>  →  collate_batch()  →  Batch (padded arrays)
//! ```

use std::collections::{HashMap, HashSet};

use rand::Rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use pronlex_core::{CharVocab, MASK_ID, PAD_ID, PhoneVocab, Vocab};

// ── Lexeme ─────────────────────────────────────────────────────────────────

/// One pronunciation entry from CMUdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lexeme {
    /// Original CMUdict entry key, e.g. `"WORD(2)"`.
    pub raw_word: String,
    /// Base spelling without variant suffix, e.g. `"WORD"`.
    pub base_word: String,
    /// Variant index: 1 for the primary entry, 2+ for alternates.
    pub variant: usize,
    /// Phones exactly as they appear in CMUdict, e.g. `["AH0", "T"]`.
    pub phones: Vec<String>,
}

// ── CMUdict parser ─────────────────────────────────────────────────────────

/// Parse the contents of a CMUdict `.dict` file.
///
/// Rules:
/// - Lines beginning with `;;;` are comments and are skipped.
/// - Blank lines are skipped.
/// - The first whitespace-delimited token is the word (possibly `WORD(2)`).
/// - Remaining tokens are phones, preserved verbatim.
/// - Alternate pronunciations like `WORD(2)` store `base_word="WORD"`,
///   `variant=2`.
pub fn parse_cmudict(text: &str) -> Vec<Lexeme> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(";;;") {
            continue;
        }
        let mut tokens = line.split_ascii_whitespace();
        let raw_word = match tokens.next() {
            Some(w) => w.to_string(),
            None => continue,
        };
        let phones: Vec<String> = tokens.map(|t| t.to_string()).collect();
        if phones.is_empty() {
            continue;
        }
        let (base_word, variant) = parse_word_entry(&raw_word);
        out.push(Lexeme {
            raw_word,
            base_word,
            variant,
            phones,
        });
    }
    out
}

/// Split a raw CMUdict word key into `(base_word, variant)`.
/// `"WORD"` → `("word", 1)`,  `"WORD(2)"` → `("word", 2)`.
fn parse_word_entry(raw: &str) -> (String, usize) {
    if let Some(open) = raw.find('(') {
        let base = raw[..open].to_lowercase();
        let variant_str = &raw[open + 1..raw.len().saturating_sub(1)];
        let variant = variant_str.parse::<usize>().unwrap_or(2);
        (base, variant)
    } else {
        (raw.to_lowercase(), 1)
    }
}

// ── Vocabulary builder ─────────────────────────────────────────────────────

/// Build the full vocabulary from a parsed lexicon.
///
/// 1. Starts with the standard ARPABET phone set.
/// 2. Extends with any extra phones found in the data.
/// 3. Builds a char vocab from all unique lowercase chars in `base_word`s.
pub fn build_vocab(lexemes: &[Lexeme]) -> Vocab {
    let mut phones = PhoneVocab::build_standard();
    let all_phones: Vec<String> = lexemes
        .iter()
        .flat_map(|l| l.phones.iter().cloned())
        .collect();
    phones.extend_from_data(&all_phones);

    let words: Vec<String> = lexemes.iter().map(|l| l.base_word.clone()).collect();
    let chars = CharVocab::build(&words);

    Vocab::new(phones, chars)
}

// ── Masking policy ─────────────────────────────────────────────────────────

/// High-level masking strategy chosen from the CLI.
#[derive(Debug, Clone)]
pub enum MaskPolicy {
    /// Always mask exactly one phone (original single-mask behaviour).
    Single,
    /// Curriculum-aware variable masking.
    Variable {
        /// Maximum fraction of phones to mask in the random-pct mode (e.g. 0.4).
        max_mask_rate: f64,
        /// Probability weight given to span masking in each curriculum phase.
        span_mask_prob: f64,
    },
}

impl Default for MaskPolicy {
    fn default() -> Self {
        MaskPolicy::Variable {
            max_mask_rate: 0.4,
            span_mask_prob: 0.15,
        }
    }
}

/// The concrete masking strategy drawn for one training example.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaskSpec {
    /// Mask exactly one phone at a random position.
    Single,
    /// Mask exactly two randomly chosen (non-overlapping) positions.
    Double,
    /// Mask a contiguous span of `len` phones starting at `start`.
    Span { start: usize, len: usize },
    /// Mask a random fraction (`rate`) of phones, chosen independently.
    RandomPct { rate: f64 },
}

/// Curriculum phase weights: `(single, double, span, random_pct)` fractions.
///
/// | Epoch range | Single | Double | Span | RandomPct |
/// |-------------|--------|--------|------|-----------|
/// | 1–3         | 0.70   | 0.20   | 0.10 | 0.00      |
/// | 4–8         | 0.50   | 0.30   | 0.15 | 0.05      |
/// | 9+          | 0.40   | 0.30   | 0.20 | 0.10      |
pub fn curriculum_weights(epoch: usize) -> [f64; 4] {
    if epoch <= 3 {
        [0.70, 0.20, 0.10, 0.00]
    } else if epoch <= 8 {
        [0.50, 0.30, 0.15, 0.05]
    } else {
        [0.40, 0.30, 0.20, 0.10]
    }
}

/// Sample a `MaskSpec` for one example given the current epoch and policy.
pub fn sample_mask_spec<R: Rng>(
    policy: &MaskPolicy,
    epoch: usize,
    phone_len: usize,
    rng: &mut R,
) -> MaskSpec {
    if phone_len == 0 {
        return MaskSpec::Single;
    }
    match policy {
        MaskPolicy::Single => MaskSpec::Single,
        MaskPolicy::Variable {
            max_mask_rate,
            span_mask_prob: _,
        } => {
            let weights = curriculum_weights(epoch);
            let roll: f64 = rng.gen();
            let cumulative = [
                weights[0],
                weights[0] + weights[1],
                weights[0] + weights[1] + weights[2],
                1.0,
            ];
            if roll < cumulative[0] || phone_len < 2 {
                MaskSpec::Single
            } else if roll < cumulative[1] || phone_len < 3 {
                MaskSpec::Double
            } else if roll < cumulative[2] {
                // Span: length 1–4, clamped to phone_len
                let max_span = (phone_len / 2).max(1).min(4);
                let len = rng.gen_range(1..=max_span);
                let start = rng.gen_range(0..=(phone_len - len));
                MaskSpec::Span { start, len }
            } else {
                // Random percent (30–max_mask_rate)
                let low = 0.3f64.min(*max_mask_rate);
                let rate = rng.gen_range(low..=*max_mask_rate);
                MaskSpec::RandomPct { rate }
            }
        }
    }
}

// ── Masking application ─────────────────────────────────────────────────────

/// Result of masking one lexeme for training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskedExample {
    /// Char IDs for the base word spelling (lowercase).
    pub char_ids: Vec<u32>,
    /// Phone IDs with `MASK_ID` at masked positions, true phone IDs elsewhere.
    pub phone_ids: Vec<u32>,
    /// Target phone IDs: the true phone at masked positions, `PAD_ID` elsewhere.
    pub targets: Vec<u32>,
    /// Indices of the masked positions (for eval/diagnostics).
    pub mask_positions: Vec<usize>,
}

/// Apply a `MaskSpec` to a lexeme and encode it using `vocab`.
///
/// Returns `None` if the phone sequence is empty.
pub fn apply_mask<R: Rng>(
    lexeme: &Lexeme,
    spec: MaskSpec,
    vocab: &Vocab,
    rng: &mut R,
) -> Option<MaskedExample> {
    let n = lexeme.phones.len();
    if n == 0 {
        return None;
    }

    // Encode phones
    let phone_ids_orig: Vec<u32> = lexeme
        .phones
        .iter()
        .map(|p| vocab.phones.get_id(p))
        .collect();

    // Determine which positions to mask
    let positions: Vec<usize> = match spec {
        MaskSpec::Single => {
            vec![rng.gen_range(0..n)]
        }
        MaskSpec::Double => {
            let mut pos = vec![rng.gen_range(0..n)];
            if n >= 2 {
                let mut second = rng.gen_range(0..n - 1);
                if second >= pos[0] {
                    second += 1;
                }
                pos.push(second);
            }
            pos.sort_unstable();
            pos
        }
        MaskSpec::Span { start, len } => {
            let end = (start + len).min(n);
            (start..end).collect()
        }
        MaskSpec::RandomPct { rate } => {
            let mut pos: Vec<usize> = (0..n)
                .filter(|_| rng.gen::<f64>() < rate)
                .collect();
            if pos.is_empty() {
                pos.push(rng.gen_range(0..n));
            }
            pos
        }
    };

    let mask_set: HashSet<usize> = positions.iter().copied().collect();

    let mut phone_ids = phone_ids_orig.clone();
    let mut targets = vec![PAD_ID; n];
    for &pos in &positions {
        targets[pos] = phone_ids_orig[pos];
        phone_ids[pos] = MASK_ID;
    }

    // Char IDs
    let char_ids: Vec<u32> = lexeme
        .base_word
        .chars()
        .map(|c| vocab.chars.get_id(c))
        .collect();

    drop(mask_set); // silence unused warning

    Some(MaskedExample {
        char_ids,
        phone_ids,
        targets,
        mask_positions: positions,
    })
}

/// Generate one `MaskedExample` per phone position (single-mask) for eval/test sets.
pub fn generate_single_mask_examples(lexeme: &Lexeme, vocab: &Vocab) -> Vec<MaskedExample> {
    let n = lexeme.phones.len();
    let phone_ids_orig: Vec<u32> = lexeme
        .phones
        .iter()
        .map(|p| vocab.phones.get_id(p))
        .collect();
    let char_ids: Vec<u32> = lexeme
        .base_word
        .chars()
        .map(|c| vocab.chars.get_id(c))
        .collect();

    (0..n)
        .map(|pos| {
            let mut phone_ids = phone_ids_orig.clone();
            let mut targets = vec![PAD_ID; n];
            targets[pos] = phone_ids_orig[pos];
            phone_ids[pos] = MASK_ID;
            MaskedExample {
                char_ids: char_ids.clone(),
                phone_ids,
                targets,
                mask_positions: vec![pos],
            }
        })
        .collect()
}

// ── Batch collation ────────────────────────────────────────────────────────

/// Fixed-length padded batch ready for the model.
#[derive(Debug, Clone)]
pub struct Batch {
    /// `[batch, max_word_len]` char IDs, padded with `PAD_ID`.
    pub char_ids: Vec<Vec<i32>>,
    /// `[batch, max_phone_len]` phone IDs (with `MASK_ID` at masked positions).
    pub phone_ids: Vec<Vec<i32>>,
    /// `[batch, max_phone_len]` target IDs (`PAD_ID` at non-masked positions).
    pub targets: Vec<Vec<i32>>,
    /// Number of examples.
    pub size: usize,
}

/// Collate `MaskedExample`s into a `Batch`, padding to the longest sequence
/// (clamped to `max_word_len` / `max_phone_len`).
pub fn collate_batch(
    examples: &[MaskedExample],
    max_word_len: usize,
    max_phone_len: usize,
) -> Batch {
    let size = examples.len();
    let mut char_ids = vec![vec![PAD_ID as i32; max_word_len]; size];
    let mut phone_ids = vec![vec![PAD_ID as i32; max_phone_len]; size];
    let mut targets = vec![vec![PAD_ID as i32; max_phone_len]; size];

    for (i, ex) in examples.iter().enumerate() {
        for (j, &c) in ex.char_ids.iter().enumerate().take(max_word_len) {
            char_ids[i][j] = c as i32;
        }
        for (j, &p) in ex.phone_ids.iter().enumerate().take(max_phone_len) {
            phone_ids[i][j] = p as i32;
        }
        for (j, &t) in ex.targets.iter().enumerate().take(max_phone_len) {
            targets[i][j] = t as i32;
        }
    }

    Batch {
        char_ids,
        phone_ids,
        targets,
        size,
    }
}

// ── Data splitting ─────────────────────────────────────────────────────────

/// Split lexemes by `base_word` into train / valid / test sets.
///
/// All variants of a `base_word` always end up in the same split so there is
/// no leakage from `WORD(2)` across train/test.
///
/// Returns `(train, valid, test)`.
pub fn split_by_base_word<R: Rng>(
    lexemes: &[Lexeme],
    train_frac: f64,
    valid_frac: f64,
    rng: &mut R,
) -> (Vec<Lexeme>, Vec<Lexeme>, Vec<Lexeme>) {
    // Group by base_word
    let mut groups: HashMap<String, Vec<&Lexeme>> = HashMap::new();
    for lex in lexemes {
        groups
            .entry(lex.base_word.clone())
            .or_default()
            .push(lex);
    }

    let mut keys: Vec<String> = groups.keys().cloned().collect();
    keys.shuffle(rng);

    let n = keys.len();
    let train_end = (n as f64 * train_frac).round() as usize;
    let valid_end = train_end + (n as f64 * valid_frac).round() as usize;

    let mut train = Vec::new();
    let mut valid = Vec::new();
    let mut test = Vec::new();

    for (i, key) in keys.iter().enumerate() {
        let entries: Vec<Lexeme> = groups[key].iter().map(|&l| l.clone()).collect();
        if i < train_end {
            train.extend(entries);
        } else if i < valid_end {
            valid.extend(entries);
        } else {
            test.extend(entries);
        }
    }

    (train, valid, test)
}

/// Verify that no `base_word` appears in more than one split.
///
/// Returns a list of leaking words (should be empty for a correct split).
pub fn check_split_leakage(
    train: &[Lexeme],
    valid: &[Lexeme],
    test: &[Lexeme],
) -> Vec<String> {
    let train_words: HashSet<&str> = train.iter().map(|l| l.base_word.as_str()).collect();
    let valid_words: HashSet<&str> = valid.iter().map(|l| l.base_word.as_str()).collect();
    let test_words: HashSet<&str> = test.iter().map(|l| l.base_word.as_str()).collect();

    let mut leaking = Vec::new();
    for w in train_words.intersection(&valid_words) {
        leaking.push(w.to_string());
    }
    for w in train_words.intersection(&test_words) {
        leaking.push(w.to_string());
    }
    for w in valid_words.intersection(&test_words) {
        leaking.push(w.to_string());
    }
    leaking.sort();
    leaking.dedup();
    leaking
}

// ── Baselines ──────────────────────────────────────────────────────────────

/// Compute the most common phone ID in a set of lexemes.
pub fn most_common_phone(lexemes: &[Lexeme], vocab: &PhoneVocab) -> u32 {
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for lex in lexemes {
        for p in &lex.phones {
            *counts.entry(vocab.get_id(p)).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map(|(id, _)| id)
        .unwrap_or(pronlex_core::UNK_ID)
}

/// Compute the most common phone per position index, with overall fallback.
pub fn most_common_phone_by_position(
    lexemes: &[Lexeme],
    vocab: &PhoneVocab,
) -> HashMap<usize, u32> {
    let overall = most_common_phone(lexemes, vocab);
    let mut pos_counts: HashMap<usize, HashMap<u32, usize>> = HashMap::new();
    for lex in lexemes {
        for (pos, p) in lex.phones.iter().enumerate() {
            *pos_counts
                .entry(pos)
                .or_default()
                .entry(vocab.get_id(p))
                .or_default() += 1;
        }
    }
    pos_counts
        .into_iter()
        .map(|(pos, counts)| {
            let best = counts
                .into_iter()
                .max_by_key(|&(_, c)| c)
                .map(|(id, _)| id)
                .unwrap_or(overall);
            (pos, best)
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn tiny_dict() -> &'static str {
        ";;; CMUdict comment\n\
         CHARLOTTE  SH AA1 R L AH0 T\n\
         CHARLOTTE(2)  SH EH1 R L AH0 T\n\
         BUTTER  B AH1 T ER0\n\
         \n\
         CAT  K AE1 T\n"
    }

    #[test]
    fn parse_comment_and_blank_lines() {
        let lexemes = parse_cmudict(tiny_dict());
        assert_eq!(lexemes.len(), 4, "should parse 4 entries");
    }

    #[test]
    fn parse_alternate_pronunciations() {
        let lexemes = parse_cmudict(tiny_dict());
        let charlotte: Vec<&Lexeme> = lexemes
            .iter()
            .filter(|l| l.base_word == "charlotte")
            .collect();
        assert_eq!(charlotte.len(), 2);
        let variant2 = charlotte.iter().find(|l| l.variant == 2).unwrap();
        assert_eq!(variant2.raw_word, "CHARLOTTE(2)");
        assert_eq!(variant2.base_word, "charlotte");
    }

    #[test]
    fn parse_stress_digits_preserved() {
        let lexemes = parse_cmudict(tiny_dict());
        let butter = lexemes.iter().find(|l| l.base_word == "butter").unwrap();
        assert!(
            butter.phones.contains(&"AH1".to_string()),
            "AH1 should be preserved"
        );
        assert!(
            butter.phones.contains(&"ER0".to_string()),
            "ER0 should be preserved"
        );
    }

    #[test]
    fn split_no_base_word_leakage() {
        let lexemes = parse_cmudict(tiny_dict());
        let mut rng = StdRng::seed_from_u64(42);
        let (train, valid, test) = split_by_base_word(&lexemes, 0.6, 0.2, &mut rng);
        let leaking = check_split_leakage(&train, &valid, &test);
        assert!(
            leaking.is_empty(),
            "leaking base_words: {:?}",
            leaking
        );
        // All lexemes accounted for
        assert_eq!(train.len() + valid.len() + test.len(), lexemes.len());
    }

    #[test]
    fn alternate_variants_stay_together() {
        let lexemes = parse_cmudict(tiny_dict());
        let mut rng = StdRng::seed_from_u64(42);
        // Run 10 random splits; CHARLOTTE(1) and CHARLOTTE(2) must stay together
        for seed in 0..10u64 {
            let mut rng2 = StdRng::seed_from_u64(seed);
            let (train, valid, test) = split_by_base_word(&lexemes, 0.6, 0.2, &mut rng2);
            let _ = rng; // suppress warning

            let charlotte_sets: [Vec<_>; 3] = [
                train
                    .iter()
                    .filter(|l| l.base_word == "charlotte")
                    .collect(),
                valid
                    .iter()
                    .filter(|l| l.base_word == "charlotte")
                    .collect(),
                test.iter()
                    .filter(|l| l.base_word == "charlotte")
                    .collect(),
            ];
            let non_empty: usize = charlotte_sets.iter().filter(|s| !s.is_empty()).count();
            assert_eq!(
                non_empty, 1,
                "charlotte variants must be in exactly one split"
            );
        }
        let _ = rng;
    }

    #[test]
    fn masking_single_generates_one_mask() {
        let vocab = build_vocab(&parse_cmudict(tiny_dict()));
        let lex = Lexeme {
            raw_word: "BUTTER".into(),
            base_word: "butter".into(),
            variant: 1,
            phones: vec!["B".into(), "AH1".into(), "T".into(), "ER0".into()],
        };
        let mut rng = StdRng::seed_from_u64(0);
        let ex = apply_mask(&lex, MaskSpec::Single, &vocab, &mut rng).unwrap();
        let mask_count = ex.phone_ids.iter().filter(|&&id| id == MASK_ID).count();
        assert_eq!(mask_count, 1);
        let target_count = ex.targets.iter().filter(|&&id| id != PAD_ID).count();
        assert_eq!(target_count, 1);
    }

    #[test]
    fn masking_generates_one_per_phone_position() {
        let vocab = build_vocab(&parse_cmudict(tiny_dict()));
        let lex = Lexeme {
            raw_word: "BUTTER".into(),
            base_word: "butter".into(),
            variant: 1,
            phones: vec!["B".into(), "AH1".into(), "T".into(), "ER0".into()],
        };
        let examples = generate_single_mask_examples(&lex, &vocab);
        assert_eq!(examples.len(), 4, "one example per phone");
        for (pos, ex) in examples.iter().enumerate() {
            assert_eq!(ex.phone_ids[pos], MASK_ID);
            assert_ne!(ex.targets[pos], PAD_ID);
        }
    }

    #[test]
    fn masking_double_produces_two_masks() {
        let vocab = build_vocab(&parse_cmudict(tiny_dict()));
        let lex = Lexeme {
            raw_word: "BUTTER".into(),
            base_word: "butter".into(),
            variant: 1,
            phones: vec!["B".into(), "AH1".into(), "T".into(), "ER0".into()],
        };
        let mut rng = StdRng::seed_from_u64(1);
        let ex = apply_mask(&lex, MaskSpec::Double, &vocab, &mut rng).unwrap();
        let mask_count = ex.phone_ids.iter().filter(|&&id| id == MASK_ID).count();
        assert_eq!(mask_count, 2);
    }

    #[test]
    fn masking_span_contiguous() {
        let vocab = build_vocab(&parse_cmudict(tiny_dict()));
        let lex = Lexeme {
            raw_word: "CHARLOTTE".into(),
            base_word: "charlotte".into(),
            variant: 1,
            phones: vec![
                "SH".into(),
                "AA1".into(),
                "R".into(),
                "L".into(),
                "AH0".into(),
                "T".into(),
            ],
        };
        let mut rng = StdRng::seed_from_u64(2);
        let ex = apply_mask(
            &lex,
            MaskSpec::Span { start: 1, len: 3 },
            &vocab,
            &mut rng,
        )
        .unwrap();
        assert_eq!(ex.phone_ids[1], MASK_ID);
        assert_eq!(ex.phone_ids[2], MASK_ID);
        assert_eq!(ex.phone_ids[3], MASK_ID);
        assert_ne!(ex.phone_ids[0], MASK_ID);
        assert_ne!(ex.phone_ids[4], MASK_ID);
    }

    #[test]
    fn curriculum_weights_progression() {
        let w1 = curriculum_weights(1);
        let w5 = curriculum_weights(5);
        let w10 = curriculum_weights(10);
        // Phase 1 is most biased toward single masking
        assert!(w1[0] > w5[0]);
        assert!(w5[0] > w10[0]);
        // Later phases have more multi-mask probability
        assert!(w10[3] > w5[3]);
        assert!(w5[3] > w1[3]);
    }
}
