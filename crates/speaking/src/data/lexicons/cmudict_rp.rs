use crate::data::lexicons::cmudict::{self, CmuPhoneme, CmuStress};
use crate::data::lexicons::{CMUDICT_RP_ID, LexiconAdapter, LexiconLookup, PronunciationStatus};
use crate::data::notation::PronunciationNotation;

pub const REGISTRATIONS: &[LexiconAdapter] = &[LexiconAdapter {
    id: CMUDICT_RP_ID,
    notation: PronunciationNotation::Arpabet,
    lookup,
}];

pub(crate) const RP_LOT: &str = "RP_LOT";
pub(crate) const RP_KIT: &str = "RP_KIT";
pub(crate) const RP_TRAP: &str = "RP_TRAP";
pub(crate) const RP_STRUT: &str = "RP_STRUT";
pub(crate) const RP_FOOT: &str = "RP_FOOT";
pub(crate) const RP_DRESS: &str = "RP_DRESS";
pub(crate) const RP_FLEECE: &str = "RP_FLEECE";
pub(crate) const RP_HAPPY: &str = "RP_HAPPY";
pub(crate) const RP_FACE: &str = "RP_FACE";
pub(crate) const RP_PALM: &str = "RP_PALM";
pub(crate) const RP_THOUGHT: &str = "RP_THOUGHT";
pub(crate) const RP_GOAT: &str = "RP_GOAT";
pub(crate) const RP_GOOSE: &str = "RP_GOOSE";
pub(crate) const RP_PRICE: &str = "RP_PRICE";
pub(crate) const RP_CHOICE: &str = "RP_CHOICE";
pub(crate) const RP_MOUTH: &str = "RP_MOUTH";
pub(crate) const RP_NURSE: &str = "RP_NURSE";
pub(crate) const RP_NEAR: &str = "RP_NEAR";
pub(crate) const RP_SQUARE: &str = "RP_SQUARE";
pub(crate) const RP_CURE: &str = "RP_CURE";
pub(crate) const RP_SCHWA: &str = "RP_SCHWA";

pub fn lookup(word: &str) -> LexiconLookup {
    let entry = cmudict::lookup(word);
    if entry.status == PronunciationStatus::Missing {
        return entry;
    }

    LexiconLookup {
        lookup: entry.lookup,
        source: entry.source,
        status: entry.status,
        candidates: entry
            .candidates
            .into_iter()
            .map(|candidate| adapt_candidate(word, &candidate))
            .collect(),
    }
}

pub(crate) fn adapted_symbol_matches_source(adapted: &str, source: &str) -> bool {
    let adapted = CmuPhoneme::parse(adapted);
    let source = CmuPhoneme::parse(source);
    if adapted.stress != source.stress {
        return false;
    }

    match adapted.base.as_str() {
        RP_LOT => source.base == "AA",
        RP_KIT => source.base == "IH",
        RP_TRAP => source.base == "AE",
        RP_STRUT => source.base == "AH",
        RP_FOOT => source.base == "UH",
        RP_DRESS => source.base == "EH",
        RP_FLEECE | RP_HAPPY => source.base == "IY",
        RP_FACE => source.base == "EY",
        RP_PALM => matches!(source.base.as_str(), "AA" | "AE"),
        RP_THOUGHT => matches!(source.base.as_str(), "AO" | "OW"),
        RP_GOAT => source.base == "OW",
        RP_GOOSE => source.base == "UW",
        RP_PRICE => source.base == "AY",
        RP_CHOICE => source.base == "OY",
        RP_MOUTH => source.base == "AW",
        RP_NURSE => source.base == "ER",
        RP_SCHWA => matches!(source.base.as_str(), "AH" | "ER"),
        _ => adapted == source,
    }
}

fn adapt_candidate(word: &str, candidate: &[String]) -> Vec<String> {
    let parsed = candidate
        .iter()
        .map(|symbol| CmuPhoneme::parse(symbol))
        .collect::<Vec<_>>();
    let normalized_word = cmudict::normalize_for_lookup(word);
    let mut adapted = Vec::new();
    let mut index = 0;

    while index < parsed.len() {
        let current = &parsed[index];
        let next = parsed.get(index + 1);
        let after_next = parsed.get(index + 2);

        if is_vowel(&current.base)
            && next.is_some_and(|phoneme| phoneme.base == "R")
            && !after_next.is_some_and(|phoneme| is_vowel(&phoneme.base))
        {
            adapt_pre_rhotic_vowel(current, &mut adapted);
            index += 2;
            continue;
        }

        if current.base == "R"
            && index > 0
            && is_vowel(&parsed[index - 1].base)
            && !next.is_some_and(|phoneme| is_vowel(&phoneme.base))
        {
            index += 1;
            continue;
        }

        if current.base == "UW"
            && retains_yod(&normalized_word)
            && adapted.last().is_some_and(|symbol: &String| {
                matches!(
                    CmuPhoneme::parse(symbol).base.as_str(),
                    "T" | "D" | "N" | "S" | "Z" | "L"
                )
            })
        {
            adapted.push("Y".into());
        }

        adapted.push(adapt_segment(
            &normalized_word,
            current,
            index + 1 == parsed.len(),
        ));
        index += 1;
    }

    adapted
}

fn adapt_segment(word: &str, phoneme: &CmuPhoneme, word_final: bool) -> String {
    let symbol = match phoneme.base.as_str() {
        "AA" => {
            if is_palm_word(word) {
                RP_PALM
            } else {
                RP_LOT
            }
        }
        "AE" => {
            if is_bath_word(word) {
                RP_PALM
            } else {
                RP_TRAP
            }
        }
        "AH" if phoneme.stress == Some(CmuStress::Unstressed) => RP_SCHWA,
        "AH" => RP_STRUT,
        "AO" => RP_THOUGHT,
        "AW" => RP_MOUTH,
        "AY" => RP_PRICE,
        "EH" => RP_DRESS,
        "ER" if phoneme.stress == Some(CmuStress::Unstressed) => RP_SCHWA,
        "ER" => RP_NURSE,
        "EY" => RP_FACE,
        "IH" => RP_KIT,
        "IY" if word_final && phoneme.stress == Some(CmuStress::Unstressed) => RP_HAPPY,
        "IY" => RP_FLEECE,
        "OW" => RP_GOAT,
        "OY" => RP_CHOICE,
        "UH" => RP_FOOT,
        "UW" => RP_GOOSE,
        _ => return phoneme.raw_symbol(),
    };
    with_stress(symbol, phoneme.stress)
}

fn adapt_pre_rhotic_vowel(phoneme: &CmuPhoneme, adapted: &mut Vec<String>) {
    let replacement = match phoneme.base.as_str() {
        "IH" | "IY" => Some(RP_NEAR),
        "EH" | "AE" | "EY" => Some(RP_SQUARE),
        "UH" | "UW" => Some(RP_CURE),
        "AA" => Some(RP_PALM),
        "AO" | "OW" => Some(RP_THOUGHT),
        _ => None,
    };
    if let Some(replacement) = replacement {
        adapted.push(with_stress(replacement, phoneme.stress));
    } else {
        adapted.push(adapt_segment("", phoneme, false));
        adapted.push(with_stress(RP_SCHWA, Some(CmuStress::Unstressed)));
    }
}

fn with_stress(base: &str, stress: Option<CmuStress>) -> String {
    let mut symbol = base.to_string();
    match stress {
        Some(CmuStress::Primary) => symbol.push('1'),
        Some(CmuStress::Secondary) => symbol.push('2'),
        Some(CmuStress::Unstressed) => symbol.push('0'),
        None => {}
    }
    symbol
}

fn is_vowel(symbol: &str) -> bool {
    matches!(
        symbol,
        "AA" | "AE"
            | "AH"
            | "AO"
            | "AW"
            | "AY"
            | "EH"
            | "ER"
            | "EY"
            | "IH"
            | "IY"
            | "OW"
            | "OY"
            | "UH"
            | "UW"
    )
}

fn is_palm_word(word: &str) -> bool {
    has_lexical_stem(
        word,
        &[
            "balm", "calm", "father", "lager", "llama", "palm", "psalm", "qualm", "rather", "spa",
        ],
    )
}

fn is_bath_word(word: &str) -> bool {
    has_lexical_stem(
        word,
        &[
            "advance",
            "advantage",
            "after",
            "afternoon",
            "answer",
            "ask",
            "aunt",
            "bath",
            "blanch",
            "branch",
            "brass",
            "can't",
            "calf",
            "cast",
            "castle",
            "chance",
            "chant",
            "class",
            "command",
            "contrast",
            "dance",
            "demand",
            "draft",
            "example",
            "fast",
            "flask",
            "gasp",
            "glass",
            "grant",
            "grass",
            "half",
            "last",
            "laugh",
            "mask",
            "master",
            "pass",
            "past",
            "path",
            "plaster",
            "plant",
            "raspberry",
            "rather",
            "sample",
            "shan't",
            "staff",
            "task",
            "transfer",
        ],
    )
}

fn has_lexical_stem(word: &str, stems: &[&str]) -> bool {
    stems.iter().any(|stem| {
        word == *stem
            || word.strip_prefix(stem).is_some_and(|suffix| {
                matches!(
                    suffix,
                    "s" | "es" | "ed" | "ing" | "er" | "ers" | "ful" | "less"
                )
            })
    })
}

fn retains_yod(word: &str) -> bool {
    has_lexical_stem(
        word,
        &[
            "assume",
            "consume",
            "cube",
            "cue",
            "due",
            "duke",
            "dune",
            "duty",
            "enthuse",
            "enthusiasm",
            "lute",
            "new",
            "news",
            "nude",
            "numeral",
            "presume",
            "student",
            "stupid",
            "suit",
            "tuba",
            "tube",
            "tune",
            "tuesday",
            "tutor",
        ],
    ) && !matches!(word, "suit" | "suits" | "suited" | "suiting")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first(word: &str) -> Vec<String> {
        lookup(word).candidates.into_iter().next().unwrap()
    }

    #[test]
    fn adapts_core_rp_lexical_sets() {
        assert_eq!(first("lot"), ["L", "RP_LOT1", "T"]);
        assert_eq!(first("bath"), ["B", "RP_PALM1", "TH"]);
        assert_eq!(first("goat"), ["G", "RP_GOAT1", "T"]);
        assert_eq!(first("nurse"), ["N", "RP_NURSE1", "S"]);
    }

    #[test]
    fn removes_non_prevocalic_r_and_retains_yod() {
        assert_eq!(first("start"), ["S", "T", "RP_PALM1", "T"]);
        assert_eq!(first("near"), ["N", "RP_NEAR1"]);
        assert_eq!(first("tune"), ["T", "Y", "RP_GOOSE1", "N"]);
    }

    #[test]
    fn adapted_symbols_still_match_cmu_selection_rules() {
        assert!(adapted_symbol_matches_source("RP_DRESS1", "EH1"));
        assert!(adapted_symbol_matches_source("RP_FLEECE1", "IY1"));
        assert!(!adapted_symbol_matches_source("RP_DRESS1", "IY1"));
    }
}
