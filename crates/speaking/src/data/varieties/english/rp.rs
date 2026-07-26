use std::collections::HashMap;

use crate::data::lexicons::cmudict_rp::{
    RP_CHOICE, RP_CURE, RP_DRESS, RP_FACE, RP_FLEECE, RP_FOOT, RP_GOAT, RP_GOOSE, RP_HAPPY, RP_KIT,
    RP_LOT, RP_MOUTH, RP_NEAR, RP_NURSE, RP_PALM, RP_PRICE, RP_SCHWA, RP_SQUARE, RP_STRUT,
    RP_THOUGHT, RP_TRAP,
};
use crate::data::notation::arpabet;
use crate::feature::FeatureBundle;
use crate::ids::{PhoneId, PhonemeId};
use crate::phonetics::{Phone, PhoneInventory};
use crate::phonology::{Phoneme, PhonemeAllophone, PhonemeInventory};
use crate::rules::{AllophoneRule, Phonotactics};
use crate::segment::{SegmentStatus, SymbolAlias};
use crate::spec::Spec;
use crate::variety::{ConnectedSpeechRule, OrthographicUnitKind, OrthographicUnitPronunciation};

pub(super) const ID: &str = "en-GB-RP";

#[derive(Debug, Clone, Copy)]
struct RpVowel {
    source: &'static str,
    arpabet: Option<&'static str>,
    ipa: &'static str,
    feature_base: &'static str,
    height: &'static str,
    backness: &'static str,
    roundedness: &'static str,
    trajectory: &'static str,
    reduced: bool,
}

const VOWELS: &[RpVowel] = &[
    vowel(RP_KIT, Some("IH"), "ɪ", "IH", "high", "front", "unrounded"),
    vowel(RP_DRESS, Some("EH"), "e", "EH", "mid", "front", "unrounded"),
    vowel(RP_TRAP, Some("AE"), "æ", "AE", "low", "front", "unrounded"),
    vowel(RP_LOT, Some("AA"), "ɒ", "AA", "low", "back", "rounded"),
    vowel(
        RP_STRUT,
        Some("AH"),
        "ʌ",
        "AH",
        "mid",
        "central",
        "unrounded",
    ),
    vowel(RP_FOOT, Some("UH"), "ʊ", "UH", "high", "back", "rounded"),
    vowel(
        RP_FLEECE,
        Some("IY"),
        "iː",
        "IY",
        "high",
        "front",
        "unrounded",
    ),
    vowel(RP_HAPPY, None, "i", "IY", "high", "front", "unrounded"),
    front_closing_diphthong(RP_FACE, Some("EY"), "eɪ", "EY", "mid", "front", "unrounded"),
    vowel(RP_PALM, None, "ɑː", "AA", "low", "back", "unrounded"),
    vowel(RP_THOUGHT, Some("AO"), "ɔː", "AO", "low", "back", "rounded"),
    back_closing_diphthong(RP_GOAT, Some("OW"), "əʊ", "OW", "mid", "back", "rounded"),
    vowel(RP_GOOSE, Some("UW"), "uː", "UW", "high", "back", "rounded"),
    front_closing_diphthong(
        RP_PRICE,
        Some("AY"),
        "aɪ",
        "AY",
        "low",
        "front",
        "unrounded",
    ),
    front_closing_diphthong(RP_CHOICE, Some("OY"), "ɔɪ", "OY", "mid", "back", "rounded"),
    back_closing_diphthong(
        RP_MOUTH,
        Some("AW"),
        "aʊ",
        "AW",
        "low",
        "central",
        "unrounded",
    ),
    vowel(
        RP_NURSE,
        Some("ER"),
        "ɜː",
        "ER",
        "mid",
        "central",
        "unrounded",
    ),
    centering_diphthong(RP_NEAR, None, "ɪə", "IH", "mid", "front", "unrounded"),
    centering_diphthong(RP_SQUARE, None, "eə", "EH", "mid", "front", "unrounded"),
    centering_diphthong(RP_CURE, None, "ʊə", "UH", "high", "back", "rounded"),
    RpVowel {
        source: RP_SCHWA,
        arpabet: None,
        ipa: "ə",
        feature_base: "AH",
        height: "mid",
        backness: "central",
        roundedness: "unrounded",
        trajectory: "stable",
        reduced: true,
    },
];

const fn vowel(
    source: &'static str,
    arpabet: Option<&'static str>,
    ipa: &'static str,
    feature_base: &'static str,
    height: &'static str,
    backness: &'static str,
    roundedness: &'static str,
) -> RpVowel {
    RpVowel {
        source,
        arpabet,
        ipa,
        feature_base,
        height,
        backness,
        roundedness,
        trajectory: "stable",
        reduced: false,
    }
}

const fn front_closing_diphthong(
    source: &'static str,
    arpabet: Option<&'static str>,
    ipa: &'static str,
    feature_base: &'static str,
    height: &'static str,
    backness: &'static str,
    roundedness: &'static str,
) -> RpVowel {
    let mut vowel = vowel(
        source,
        arpabet,
        ipa,
        feature_base,
        height,
        backness,
        roundedness,
    );
    vowel.trajectory = "closing_front";
    vowel
}

const fn back_closing_diphthong(
    source: &'static str,
    arpabet: Option<&'static str>,
    ipa: &'static str,
    feature_base: &'static str,
    height: &'static str,
    backness: &'static str,
    roundedness: &'static str,
) -> RpVowel {
    let mut vowel = vowel(
        source,
        arpabet,
        ipa,
        feature_base,
        height,
        backness,
        roundedness,
    );
    vowel.trajectory = "closing_back";
    vowel
}

const fn centering_diphthong(
    source: &'static str,
    arpabet: Option<&'static str>,
    ipa: &'static str,
    feature_base: &'static str,
    height: &'static str,
    backness: &'static str,
    roundedness: &'static str,
) -> RpVowel {
    let mut vowel = vowel(
        source,
        arpabet,
        ipa,
        feature_base,
        height,
        backness,
        roundedness,
    );
    vowel.trajectory = "centering";
    vowel
}

pub(super) fn phoneme_id(symbol: &str) -> PhonemeId {
    let (base, _) = arpabet::split_stress(symbol);
    if let Some(vowel) = VOWELS
        .iter()
        .find(|vowel| vowel.source == base || vowel.arpabet.is_some_and(|alias| alias == base))
    {
        return PhonemeId(format!("{ID}.phoneme.{}", vowel.ipa));
    }
    arpabet::phoneme_id(ID, symbol)
}

pub(super) fn phoneme_inventory() -> PhonemeInventory {
    let mut phonemes = HashMap::new();
    for entry in arpabet::ARPABET
        .iter()
        .filter(|entry| entry.major == "consonant")
    {
        let mut phoneme = arpabet::phoneme_for_entry(ID, entry);
        super::enrich_english_inventory_features(&mut phoneme.features, entry);
        phonemes.insert(phoneme.id.clone(), phoneme);
    }
    for vowel in VOWELS {
        let phoneme = vowel_phoneme(*vowel);
        phonemes.insert(phoneme.id.clone(), phoneme);
    }

    let rules = allophone_rules();
    for rule in &rules {
        let Spec::Known(phoneme_id) = &rule.input.phoneme else {
            continue;
        };
        let Spec::Known(phone_id) = &rule.output.phone else {
            continue;
        };
        if let Some(phoneme) = phonemes.get_mut(phoneme_id) {
            if !phoneme.possible_phones.contains(phone_id) {
                phoneme.possible_phones.push(phone_id.clone());
            }
            phoneme.allophones.push(PhonemeAllophone {
                phone: phone_id.clone(),
                environment: rule.environment.clone(),
                conditions: rule.conditions.clone(),
                confidence: rule.confidence,
                status: rule.status.clone(),
                source_rule_id: Some(rule.id.clone()),
            });
        }
    }

    PhonemeInventory { phonemes }
}

fn vowel_phoneme(spec: RpVowel) -> Phoneme {
    let phone = PhoneId::from(format!("ipa.phone.{}", spec.ipa));
    Phoneme {
        id: PhonemeId(format!("{ID}.phoneme.{}", spec.ipa)),
        notation: format!("/{}/", spec.ipa),
        features: vowel_features(spec),
        default_phone: Some(phone.clone()),
        possible_phones: vec![phone],
        aliases: vowel_aliases(spec),
        allophones: Vec::new(),
        status: SegmentStatus::Core,
    }
}

fn vowel_aliases(spec: RpVowel) -> Vec<SymbolAlias> {
    let mut aliases = vec![SymbolAlias {
        system: "arpabet".into(),
        symbol: spec.source.into(),
    }];
    if let Some(symbol) = spec.arpabet {
        aliases.push(SymbolAlias {
            system: "arpabet".into(),
            symbol: symbol.into(),
        });
    }
    aliases
}

fn vowel_features(spec: RpVowel) -> FeatureBundle {
    let mut features = super::allophonic_features_from_arpabet(spec.feature_base);
    super::put_phonology_category(&mut features, "vowel_height", spec.height);
    super::put_phonology_category(&mut features, "vowel_backness", spec.backness);
    super::put_phonology_category(&mut features, "roundedness", spec.roundedness);
    super::put_phonology_category(&mut features, "formant_trajectory", spec.trajectory);
    super::put_phonology_bool(&mut features, "diphthong", spec.trajectory != "stable");
    super::put_phonology_bool(&mut features, "rhoticity", false);
    super::put_phonology_bool(&mut features, "reduced_vowel", spec.reduced);
    features
}

pub(super) fn phone_inventory() -> PhoneInventory {
    let mut inventory = super::phone_inventory();
    for vowel in VOWELS {
        let id = PhoneId::from(format!("ipa.phone.{}", vowel.ipa));
        inventory.phones.insert(
            id.clone(),
            Phone {
                id,
                ipa: vowel.ipa.into(),
                features: vowel_features(*vowel),
                aliases: Vec::new(),
                status: SegmentStatus::Core,
            },
        );
    }
    inventory
}

pub(super) fn allophone_rules() -> Vec<AllophoneRule> {
    super::allophone_rules(ID)
        .into_iter()
        .filter(|rule| {
            !rule.id.starts_with("american_english_intervocalic") && !rule.id.contains("_er_")
        })
        .map(|mut rule| {
            rule.id = rule
                .id
                .replacen("american_english", "received_pronunciation", 1);
            rule.name = rule
                .name
                .replacen("American English", "Received Pronunciation", 1);
            rule
        })
        .collect()
}

pub(super) fn phonotactics() -> Phonotactics {
    let mut phonotactics = super::phonotactics(false);
    phonotactics.constraints.retain(|constraint| {
        !(constraint.id.starts_with("english.legal_coda.")
            && constraint
                .id
                .rsplit('.')
                .next()
                .is_some_and(|cluster| cluster.split('_').any(|phone| phone == "ɹ")))
    });
    phonotactics
}

pub(super) fn connected_speech() -> Vec<ConnectedSpeechRule> {
    vec![ConnectedSpeechRule::LinkingR { phone: "ɹ".into() }]
}

pub(super) fn adapt_orthographic_units(units: &mut [OrthographicUnitPronunciation]) {
    for unit in units {
        let replacement = match (unit.kind, unit.unit.as_str()) {
            (OrthographicUnitKind::LetterName, "R") => Some(vec!["RP_PALM1"]),
            (OrthographicUnitKind::LetterName, "Z") => Some(vec!["Z", "EH1", "D"]),
            (OrthographicUnitKind::DigitName, "4") => Some(vec!["F", "RP_THOUGHT1"]),
            _ => None,
        };
        let Some(replacement) = replacement else {
            continue;
        };
        unit.source_pronunciation = replacement.into_iter().map(str::to_string).collect();
        unit.pronunciation = unit
            .source_pronunciation
            .iter()
            .map(|symbol| phoneme_id(symbol))
            .collect();
    }
}

pub(super) fn default_rate_syllables_per_second() -> f32 {
    4.7
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_has_non_rhotic_rp_vowels() {
        let inventory = phoneme_inventory();
        for symbol in ["ɒ", "ɑː", "ɔː", "əʊ", "ɜː", "ɪə", "eə", "ʊə"] {
            let id = PhonemeId(format!("{ID}.phoneme.{symbol}"));
            assert!(inventory.phonemes.contains_key(&id), "missing /{symbol}/");
        }
        assert!(
            !inventory
                .phonemes
                .contains_key(&PhonemeId(format!("{ID}.phoneme.ɝ")))
        );
    }

    #[test]
    fn rules_exclude_american_flapping() {
        assert!(
            !allophone_rules()
                .iter()
                .any(|rule| rule.id.contains("flapping"))
        );
    }
}
