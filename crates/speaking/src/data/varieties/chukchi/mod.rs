use std::collections::HashMap;

use crate::feature::{FeatureBundle, FeatureSystem, FeatureValue};
use crate::ids::{LanguageId, PhoneId, PhonemeId, VarietyId};
use crate::orthography::Orthography;
use crate::phonetics::{Phone, PhoneInventory};
use crate::phonology::{Phoneme, PhonemeInventory};
use crate::segment::{SegmentStatus, SymbolAlias};
use crate::spec::Spec;
use crate::syntax::GrammarRuleSet;
use crate::variety::{
    LinguisticVariety, NumberNameSet, OrthographyPronunciationRules, VarietyImplementationStatus,
    VarietyStatus,
};

pub const REGISTRATIONS: &[crate::data::varieties::VarietyRegistration] =
    &[crate::data::varieties::VarietyRegistration {
        canonical_id: "ckt",
        aliases: &["chukchi", "ckt-Cyrl", "ckt-RU"],
        language_tag: "ckt-Cyrl",
        load: |_| variety(),
    }];

const SEGMENTS: &[&str] = &[
    "a", "e", "i", "o", "u", "ə", "b", "d", "f", "v", "w", "ɣ", "h", "j", "k", "l", "ɬ", "m", "n",
    "ŋ", "p", "q", "r", "s", "t", "t͡s", "t͡ʃ", "z", "ʃ", "ʒ", "x", "ʔ",
];

pub fn variety() -> LinguisticVariety {
    LinguisticVariety {
        id: VarietyId("ckt".into()),
        language: LanguageId("ckt".into()),
        name: "Standard Chukchi".into(),
        feature_system: FeatureSystem::default(),
        phonemes: phoneme_inventory(),
        phones: phone_inventory(),
        allophone_rules: Vec::new(),
        epenthesis_rules: Vec::new(),
        weak_forms: Vec::new(),
        orthographic_unit_pronunciations: Vec::new(),
        pronunciation_lexicons: Vec::new(),
        pronunciation_selection_rules: Vec::new(),
        pronunciation_pipeline: Some(
            crate::data::varieties::PRONUNCIATION_PIPELINE_VARIETY_DATA.into(),
        ),
        text_normalization: crate::data::varieties::small_number_text_normalization_profile(),
        syntax_profile: Some(crate::data::varieties::SYNTAX_PROFILE_CHUKCHI.into()),
        syntax_analyzer: None,
        syntax_rules: Some(GrammarRuleSet::empty()),
        orthography_pronunciation: Some(OrthographyPronunciationRules {
            synthesize_ipa: Some(synthesize_ipa_for_orthography),
        }),
        number_names: Some(NumberNameSet {
            cardinal_0_to_20: [
                "ноль",
                "ыннэн",
                "ӈирэӄ",
                "ӈыроӄ",
                "ӈыраӄ",
                "мэтԓыӈэн",
                "ыннанмытԓыӈэн",
                "ӈэръамытԓыӈэн",
                "амӈырооткэн",
                "конъачгынкэн",
                "мынгыткэн",
                "мынгыткэн ыннэн пароԓ",
                "мынгыткэн ӈирэӄ пароԓ",
                "мынгыткэн ӈыроӄ пароԓ",
                "мынгыткэн ӈыраӄ пароԓ",
                "кыԓгынкэн",
                "кыԓгынкэн ыннэн пароԓ",
                "кыԓгынкэн ӈирэӄ пароԓ",
                "кыԓгынкэн ӈыроӄ пароԓ",
                "кыԓгынкэн ӈыраӄ пароԓ",
                "ӄԓиккин",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            ..Default::default()
        }),
        punctuation: Some(Default::default()),
        question_contours: Some(Default::default()),
        connected_speech: Vec::new(),
        phonotactics: None,
        orthography: Some(Orthography {
            name: "Standard Chukchi Cyrillic orthography".into(),
            pronunciation: Some(crate::data::varieties::ORTHOGRAPHY_PROFILE_CHUKCHI.into()),
            initialism_joiners: vec!["и".into()],
            sample_words: vec!["ԓыгъоравэтԓьэн".into(), "йиԓыйиԓ".into()],
            sample_letter_units: vec!["А".into(), "Ӄ".into(), "Ӈ".into(), "Ԓ".into()],
            ..Default::default()
        }),
        morphology: None,
        acoustic_profile: None,
        prosody_profile: Some(crate::data::varieties::prosody_profile(
            crate::data::varieties::PROSODY_RHYTHM_SYLLABLE_TIMED,
            4.5,
        )),
        status: VarietyStatus::Attested,
        // The sound inventory and Cyrillic G2P are useful, but this does not
        // pretend to be a complete polysynthetic morphology or sociolect model.
        implementation_status: VarietyImplementationStatus::PermissiveProfile,
    }
}

fn synthesize_ipa_for_orthography(
    word: &str,
    _variety: &LinguisticVariety,
    _part_of_speech: Option<crate::syntax::PartOfSpeech>,
) -> Option<String> {
    synthesize_ipa(word)
}

/// Pronounce surface Standard Chukchi Cyrillic.
///
/// The spelling already reflects the output of dominant-recessive vowel
/// harmony. Applying harmony here would require a morphological analysis and
/// would corrupt correctly spelled surface forms, so this rule maps the
/// attested spelling and supplies the usual penultimate, non-schwa stress.
pub fn synthesize_ipa(word: &str) -> Option<String> {
    let chars = normalize_word(word)?;
    let symbols = chars
        .iter()
        .enumerate()
        .filter_map(|(index, ch)| grapheme_symbol(&chars, index, *ch))
        .collect::<Vec<_>>();
    if symbols.is_empty() {
        return None;
    }

    let vowel_indices = symbols
        .iter()
        .enumerate()
        .filter_map(|(index, symbol)| is_vowel(symbol).then_some(index))
        .collect::<Vec<_>>();
    let stress_index = vowel_indices
        .iter()
        .rev()
        .copied()
        .find(|index| symbols[*index] != "ə")
        .or_else(|| vowel_indices.last().copied())
        .and_then(|last| {
            vowel_indices
                .iter()
                .copied()
                .filter(|index| *index < last && symbols[*index] != "ə")
                .next_back()
                .or(Some(last))
        });

    let mut ipa = String::new();
    for (index, symbol) in symbols.iter().enumerate() {
        if Some(index) == stress_index {
            ipa.push('ˈ');
        }
        ipa.push_str(symbol);
    }
    Some(format!("/{ipa}/"))
}

fn normalize_word(word: &str) -> Option<Vec<char>> {
    let normalized = word.trim().to_lowercase();
    if normalized.is_empty()
        || normalized.chars().count() > 96
        || normalized
            .chars()
            .any(|ch| !(ch.is_alphabetic() || matches!(ch, '-' | '\'' | '’')))
    {
        return None;
    }
    Some(normalized.chars().collect())
}

fn grapheme_symbol(chars: &[char], index: usize, ch: char) -> Option<&'static str> {
    let starts_syllable = index == 0
        || matches!(
            chars.get(index.wrapping_sub(1)),
            Some('а' | 'е' | 'ё' | 'и' | 'о' | 'у' | 'ы' | 'э' | 'ю' | 'я' | 'ъ' | 'ь')
        );
    Some(match ch {
        'а' => "a",
        'б' => "b",
        'в' => "w",
        'г' => "ɣ",
        'д' => "d",
        'е' if starts_syllable => "je",
        'е' => "e",
        'ё' => "jo",
        'ж' => "ʒ",
        'з' => "z",
        'и' => "i",
        'й' => "j",
        'к' => "k",
        'ӄ' | 'қ' => "q",
        'л' => "l",
        'ԓ' | 'ӆ' => "ɬ",
        'м' => "m",
        'н' => "n",
        'ӈ' | 'ң' => "ŋ",
        'о' => "o",
        'п' => "p",
        'р' => "r",
        'с' => "s",
        'т' => "t",
        'у' => "u",
        'ф' => "f",
        'х' => "x",
        'ц' => "t͡s",
        'ч' => "t͡ʃ",
        'ш' | 'щ' => "ʃ",
        'ы' => "ə",
        'э' => "e",
        'ю' => "ju",
        'я' => "ja",
        'ъ' | '\'' | '’' => "ʔ",
        'ь' | '-' => return None,
        _ => return None,
    })
}

fn phoneme_inventory() -> PhonemeInventory {
    let phonemes = SEGMENTS
        .iter()
        .map(|symbol| {
            let phone_id = PhoneId(format!("ipa.phone.{symbol}").into());
            let phoneme = Phoneme {
                id: PhonemeId(format!("ckt.phoneme.{symbol}")),
                notation: format!("/{symbol}/"),
                features: segment_features(symbol),
                default_phone: Some(phone_id.clone()),
                possible_phones: vec![phone_id],
                aliases: vec![SymbolAlias {
                    system: "ipa".into(),
                    symbol: (*symbol).into(),
                }],
                allophones: Vec::new(),
                status: SegmentStatus::Core,
            };
            (phoneme.id.clone(), phoneme)
        })
        .collect::<HashMap<_, _>>();
    PhonemeInventory { phonemes }
}

fn phone_inventory() -> PhoneInventory {
    let phones = SEGMENTS
        .iter()
        .map(|symbol| {
            let id = PhoneId(format!("ipa.phone.{symbol}").into());
            let phone = Phone {
                id: id.clone(),
                ipa: (*symbol).into(),
                features: segment_features(symbol),
                aliases: vec![SymbolAlias {
                    system: "ipa".into(),
                    symbol: (*symbol).into(),
                }],
                status: SegmentStatus::Core,
            };
            (id, phone)
        })
        .collect::<HashMap<_, _>>();
    PhoneInventory { phones }
}

fn segment_features(symbol: &str) -> FeatureBundle {
    let mut features = FeatureBundle::default();
    let vowel = is_vowel(symbol);
    features.values.insert(
        crate::ids::FeatureId("phonology.major".into()),
        Spec::Known(FeatureValue::Category(
            if vowel { "vowel" } else { "consonant" }.into(),
        )),
    );
    features.values.insert(
        crate::ids::FeatureId("phonology.syllabic".into()),
        Spec::Known(FeatureValue::Bool(vowel)),
    );
    features.values.insert(
        crate::ids::FeatureId("phonology.base_symbol".into()),
        Spec::Known(FeatureValue::Category(symbol.into())),
    );
    features
}

fn is_vowel(symbol: &str) -> bool {
    matches!(symbol, "a" | "e" | "i" | "o" | "u" | "ə")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_cyrillic_and_chukchi_letters_are_pronounced() {
        assert_eq!(
            synthesize_ipa("ԓыгъоравэтԓьэн").as_deref(),
            Some("/ɬəɣʔorawˈetɬen/")
        );
        assert_eq!(synthesize_ipa("ӄԓиккин").as_deref(), Some("/qɬˈikkin/"));
        assert_eq!(synthesize_ipa("ӈирэӄ").as_deref(), Some("/ŋˈireq/"));
    }

    #[test]
    fn chukchi_inventory_includes_distinctive_segments() {
        let variety = variety();
        for symbol in ["q", "ɬ", "ŋ", "ʔ", "ə"] {
            assert!(
                variety
                    .phonemes
                    .phonemes
                    .contains_key(&PhonemeId(format!("ckt.phoneme.{symbol}")))
            );
        }
    }
}
