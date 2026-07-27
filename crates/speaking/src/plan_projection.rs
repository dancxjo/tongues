//! Pure text projections of the canonical [`UtterancePlan`] IR.
//!
//! These functions are intentionally lossy display or model-target projections.
//! Callers should carry `UtterancePlan` itself between components and serialize
//! it with Serde when the complete typed representation is required.

use crate::{
    FeatureBundle, FeatureId, FeatureValue, PauseKind, PhoneToken, PhonemeId, PhonemeToken, Spec,
    SpeechBoundaryToken, Stress, Syllable, TerminalPunctuation, UtterancePlan, VarietyId,
    phone_display_symbol, phoneme_default_phone_display_symbol, token_stress,
};

/// Render the connected-speech model target, including word, pause,
/// punctuation, and intonation boundaries.
///
/// This stream follows target syllabification. Phone-only connected-speech
/// insertions are retained as phones; phones aligned to intended phonemes are
/// rendered broadly. It is not a serialization of the complete plan.
pub fn display_plan_connected_speech(plan: &UtterancePlan) -> String {
    if plan.target_syllables.is_empty() {
        return serialize_token_words(
            plan.intended_phonemes.iter().filter_map(|token| {
                let Spec::Known(id) = &token.phoneme else {
                    return None;
                };
                Some((
                    display_phoneme_token(token, id, &plan.variety),
                    token_word_index(&token.features),
                ))
            }),
            &plan.boundaries,
        );
    }
    serialize_utterance_parts(plan, |syllables, plan| {
        syllables_to_phonemes_ipa(syllables, &plan.intended_phonemes, &plan.variety)
    })
}

/// Render the plan's broad phonemic projection for diagnostics.
pub fn display_plan_phonemes(plan: &UtterancePlan) -> String {
    display_plan_phoneme_words(plan)
}

/// Render the plan's realized-phone projection without typed wrappers.
pub fn display_plan_phones(plan: &UtterancePlan) -> String {
    display_plan_phone_words(plan)
}

/// Render realized phones with connected-speech boundary markers.
pub fn display_plan_connected_phones(plan: &UtterancePlan) -> String {
    if plan.target_syllables.is_empty() {
        return serialize_token_words(
            plan.target_phones.iter().filter_map(|token| {
                let Spec::Known(id) = &token.phone else {
                    return None;
                };
                (!id.as_str().starts_with("boundary.")).then(|| {
                    (
                        phone_display_symbol(id).to_string(),
                        token_word_index(&token.features),
                    )
                })
            }),
            &plan.boundaries,
        );
    }
    serialize_utterance_parts(plan, |syllables, _| crate::syllables_to_ipa(syllables))
}

/// Render broad phoneme words without utterance boundary markers.
///
/// This is intended for lexical datasets whose rows are words rather than
/// utterances. Connected-speech model targets should use
/// [`display_plan_connected_speech`].
pub fn display_plan_phoneme_words(plan: &UtterancePlan) -> String {
    if plan.target_syllables.is_empty() {
        return display_token_words(plan.intended_phonemes.iter().filter_map(|token| {
            let Spec::Known(id) = &token.phoneme else {
                return None;
            };
            Some((
                display_phoneme_token(token, id, &plan.variety),
                token_word_index(&token.features),
            ))
        }));
    }
    word_syllables(&plan.target_syllables)
        .into_iter()
        .map(|(_, syllables)| {
            syllables_to_intended_phonemes_ipa(&syllables, &plan.intended_phonemes, &plan.variety)
        })
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Render realized phone words without utterance boundary markers.
pub fn display_plan_phone_words(plan: &UtterancePlan) -> String {
    if plan.target_syllables.is_empty() {
        return display_token_words(plan.target_phones.iter().filter_map(|token| {
            let Spec::Known(id) = &token.phone else {
                return None;
            };
            (!id.as_str().starts_with("boundary.")).then(|| {
                (
                    phone_display_symbol(id).to_string(),
                    token_word_index(&token.features),
                )
            })
        }));
    }
    word_syllables(&plan.target_syllables)
        .into_iter()
        .map(|(_, syllables)| crate::syllables_to_ipa(&syllables))
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn serialize_utterance_parts(
    plan: &UtterancePlan,
    format_word: impl Fn(&[Syllable], &UtterancePlan) -> String,
) -> String {
    let words = word_syllables(&plan.target_syllables);
    serialize_words(
        words
            .into_iter()
            .map(|(word_index, syllables)| (word_index, format_word(&syllables, plan)))
            .collect(),
        &plan.boundaries,
    )
}

fn serialize_token_words(
    symbols: impl IntoIterator<Item = (String, Option<usize>)>,
    boundaries: &[SpeechBoundaryToken],
) -> String {
    serialize_words(group_token_words(symbols), boundaries)
}

fn display_token_words(symbols: impl IntoIterator<Item = (String, Option<usize>)>) -> String {
    group_token_words(symbols)
        .into_iter()
        .map(|(_, word)| word)
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn group_token_words(
    symbols: impl IntoIterator<Item = (String, Option<usize>)>,
) -> Vec<(usize, String)> {
    let mut words: Vec<(usize, String)> = Vec::new();
    for (symbol, word_index) in symbols {
        let word_index = word_index.unwrap_or_else(|| words.last().map_or(0, |word| word.0));
        if let Some(last) = words.last_mut().filter(|word| word.0 == word_index) {
            last.1.push_str(&symbol);
        } else {
            words.push((word_index, symbol));
        }
    }
    words
}

fn serialize_words(words: Vec<(usize, String)>, boundaries: &[SpeechBoundaryToken]) -> String {
    let last_index = words.len().saturating_sub(1);
    let mut parts = Vec::new();
    for (position, (word_index, word)) in words.into_iter().enumerate() {
        if word.is_empty() {
            continue;
        }
        parts.push(word);
        let boundary_symbols = boundary_symbols_after_word(boundaries, word_index);
        if boundary_symbols.is_empty() {
            if position != last_index {
                parts.push("|".to_string());
            }
        } else {
            parts.extend(boundary_symbols.into_iter().map(str::to_string));
        }
    }
    parts.join(" ")
}

fn word_syllables(syllables: &[Syllable]) -> Vec<(usize, Vec<Syllable>)> {
    let mut words: Vec<(usize, Vec<Syllable>)> = Vec::new();
    for syllable in syllables {
        let Some(first_phone) = syllable.phones.first() else {
            continue;
        };
        let Some(word_index) = token_word_index(&first_phone.features) else {
            continue;
        };
        if let Some(last_word) = words.last_mut() {
            if last_word.0 == word_index {
                last_word.1.push(syllable.clone());
                continue;
            }
        }
        words.push((word_index, vec![syllable.clone()]));
    }
    words
}

fn boundary_symbols_after_word(
    boundaries: &[SpeechBoundaryToken],
    word_index: usize,
) -> Vec<&'static str> {
    let Some(boundary) = boundaries
        .iter()
        .filter(|boundary| boundary.terminal.is_some() || boundary.pause.is_some())
        .find(|boundary| boundary.after_grapheme_index == word_index)
    else {
        return Vec::new();
    };
    if let Some(terminal) = boundary.terminal {
        return match terminal {
            TerminalPunctuation::Question => vec!["↗", "?"],
            TerminalPunctuation::Period => vec!["↘", "."],
            TerminalPunctuation::Exclamation => vec!["↘", "!"],
        };
    }
    if let Some(pause) = boundary.pause {
        return match pause {
            PauseKind::Comma => vec!["→", ","],
            PauseKind::AlternativeQuestionRise => vec!["↗", ","],
        };
    }
    Vec::new()
}

fn syllables_to_phonemes_ipa(
    syllables: &[Syllable],
    phonemes: &[PhonemeToken],
    variety: &VarietyId,
) -> String {
    format_syllables(syllables, |phone| {
        find_phoneme_for_phone(phone, phonemes)
            .and_then(|token| match &token.phoneme {
                Spec::Known(id) => Some(phoneme_token_default_phone_display_symbol(
                    token, id, variety,
                )),
                _ => None,
            })
            .unwrap_or_else(|| match &phone.phone {
                Spec::Known(id) => phone_display_symbol(id).to_string(),
                _ => String::new(),
            })
    })
}

fn syllables_to_intended_phonemes_ipa(
    syllables: &[Syllable],
    phonemes: &[PhonemeToken],
    variety: &VarietyId,
) -> String {
    format_syllables(syllables, |phone| {
        find_phoneme_for_phone(phone, phonemes)
            .and_then(|token| match &token.phoneme {
                Spec::Known(id) => Some(phoneme_token_default_phone_display_symbol(
                    token, id, variety,
                )),
                _ => None,
            })
            .unwrap_or_default()
    })
}

fn display_phoneme_token(token: &PhonemeToken, id: &PhonemeId, variety: &VarietyId) -> String {
    let mut symbol = phoneme_token_default_phone_display_symbol(token, id, variety);
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
    symbol
}

fn phoneme_token_default_phone_display_symbol(
    token: &PhonemeToken,
    id: &PhonemeId,
    variety: &VarietyId,
) -> String {
    let merged_strut_about = ["phonology.canonical_base_symbol", "phonology.base_symbol"]
        .into_iter()
        .any(|feature| {
            matches!(
                token.features.values.get(&FeatureId(feature.into())),
                Some(Spec::Known(
                    FeatureValue::Category(value) | FeatureValue::Text(value)
                )) if value == "AH"
            )
        });
    if variety.0 == "en-GB-RP"
        && merged_strut_about
        && let Some(Spec::Known(FeatureValue::Category(phone) | FeatureValue::Text(phone))) = token
            .features
            .values
            .get(&FeatureId("phonology.default_phone".into()))
    {
        return phone_display_symbol(&crate::PhoneId::from(phone.clone())).to_string();
    }
    phoneme_default_phone_display_symbol(id, variety)
}

fn format_syllables(
    syllables: &[Syllable],
    mut display_phone: impl FnMut(&PhoneToken) -> String,
) -> String {
    syllables
        .iter()
        .enumerate()
        .map(|(index, syllable)| {
            let mut text = String::new();
            let stressed = match syllable.stress {
                Spec::Known(Stress::Primary) => {
                    text.push('ˈ');
                    true
                }
                Spec::Known(Stress::Secondary) => {
                    text.push('ˌ');
                    true
                }
                _ => false,
            };
            if index > 0 && !stressed {
                text.insert(0, '.');
            }
            for phone in &syllable.phones {
                text.push_str(&display_phone(phone));
            }
            text
        })
        .collect()
}

fn find_phoneme_for_phone<'a>(
    phone: &PhoneToken,
    phonemes: &'a [PhonemeToken],
) -> Option<&'a PhonemeToken> {
    phonemes.iter().find_map(|phoneme_token| {
        phoneme_token
            .realized_as
            .iter()
            .any(|realized| {
                realized.phone == phone.phone
                    && realized.features == phone.features
                    && realized.span == phone.span
            })
            .then_some(phoneme_token)
    })
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

#[cfg(test)]
mod tests {
    use crate::{PhonemicizeRequest, VarietyId, phonemicizer_for_variety};

    use super::*;

    fn plan(text: &str) -> UtterancePlan {
        let variety = VarietyId("en-US".into());
        let output = phonemicizer_for_variety(&variety)
            .unwrap()
            .phonemicize(&PhonemicizeRequest {
                text: text.into(),
                variety,
                style: None,
            })
            .unwrap();
        (&output).into()
    }

    #[test]
    fn connected_speech_projection_preserves_boundaries() {
        let plan = plan("Hello, world?");
        let serialized = display_plan_connected_speech(&plan);
        assert!(serialized.contains("→ ,"));
        assert!(serialized.ends_with("↗ ?"));
        assert!(display_plan_connected_phones(&plan).ends_with("↗ ?"));
    }

    #[test]
    fn broad_and_realized_views_are_distinct_plan_projections() {
        let plan = plan("atlas");
        assert_eq!(display_plan_phonemes(&plan), "ˈæt.ləs");
        assert_eq!(display_plan_phones(&plan), "ˈæt.ləs");
        assert!(!display_plan_phonemes(&plan).contains('↘'));
    }

    #[test]
    fn token_only_plans_use_the_same_connected_speech_projection() {
        let mut plan = plan("hello world");
        plan.target_syllables.clear();
        assert!(display_plan_connected_speech(&plan).contains('|'));
        assert!(!display_plan_phonemes(&plan).contains('|'));
    }

    #[test]
    fn intended_phonemes_do_not_absorb_intrusive_r() {
        let variety = VarietyId("en-GB-RP".into());
        let output = phonemicizer_for_variety(&variety)
            .unwrap()
            .phonemicize(&PhonemicizeRequest {
                text: "umbrella up".into(),
                variety,
                style: None,
            })
            .unwrap();
        let plan = UtterancePlan::from(output);

        assert_eq!(display_plan_phonemes(&plan), "əmˈbɹe.lə ˈʌp");
        assert_eq!(display_plan_connected_speech(&plan), "əmˈbɹe.ləɹ | ˈʌp ↘ .");
        assert_eq!(display_plan_connected_phones(&plan), "əmˈbɹe.ləɹ | ˈʌp ↘ .");
        assert_eq!(display_plan_phones(&plan), "əmˈbɹe.ləɹ ˈʌp");
    }

    #[test]
    fn typed_plan_json_round_trips_without_projection_loss() {
        let plan = plan("Hello, world?");
        let json = serde_json::to_string(&plan).expect("plan JSON");
        let restored: UtterancePlan = serde_json::from_str(&json).expect("round-trip plan");
        assert_eq!(restored, plan);
    }
}
