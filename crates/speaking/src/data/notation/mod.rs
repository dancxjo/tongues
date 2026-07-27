pub mod arpabet;
pub mod openepd;

use crate::feature::{FeatureBundle, FeatureValue};
use crate::ids::{FeatureId, PhonemeId};
use crate::phonology::Phoneme;
use crate::spec::Spec;
use crate::variety::LinguisticVariety;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PronunciationNotation {
    Arpabet,
    Cmudict,
    Ipa,
}

#[derive(Debug, Clone)]
pub struct ParsedPronunciationToken {
    pub phoneme: PhonemeId,
    pub features: FeatureBundle,
}

pub fn pronunciation_notation(name: Option<&str>) -> Option<PronunciationNotation> {
    match name {
        Some("arpabet") => Some(PronunciationNotation::Arpabet),
        Some("cmudict") => Some(PronunciationNotation::Cmudict),
        Some("ipa") => Some(PronunciationNotation::Ipa),
        _ => None,
    }
}

pub fn parse_pronunciation_candidate(
    variety: &LinguisticVariety,
    candidate: &[String],
    notation: PronunciationNotation,
) -> Vec<ParsedPronunciationToken> {
    match notation {
        PronunciationNotation::Arpabet | PronunciationNotation::Cmudict => candidate
            .iter()
            .map(|symbol| parsed_arpabet_token(variety, symbol, notation))
            .collect(),
        PronunciationNotation::Ipa => candidate
            .iter()
            .flat_map(|symbol| parse_variety_ipa(symbol, variety))
            .collect(),
    }
}

fn parsed_arpabet_token(
    variety: &LinguisticVariety,
    symbol: &str,
    notation: PronunciationNotation,
) -> ParsedPronunciationToken {
    let cmu = crate::data::lexicons::cmudict::CmuPhoneme::parse(symbol);
    let raw_symbol = cmu.raw_symbol();
    let canonical_base = arpabet::canonical_base_symbol(&cmu.base);
    let inventory_phoneme = variety.phonemes.phonemes.values().find(|phoneme| {
        phoneme.aliases.iter().any(|alias| {
            alias.system.eq_ignore_ascii_case("arpabet")
                && alias.symbol.eq_ignore_ascii_case(canonical_base)
        })
    });
    let mut token = ParsedPronunciationToken {
        phoneme: inventory_phoneme
            .map(|phoneme| phoneme.id.clone())
            .unwrap_or_else(|| arpabet::phoneme_id(&variety.id.0, &raw_symbol)),
        features: inventory_phoneme
            .map(|phoneme| phoneme.features.clone())
            .unwrap_or_default(),
    };
    let source_schema = match notation {
        PronunciationNotation::Cmudict => arpabet::SOURCE_SCHEMA_CMUDICT,
        PronunciationNotation::Arpabet => arpabet::SOURCE_SCHEMA_ARPABET,
        PronunciationNotation::Ipa => unreachable!("IPA uses its own parser"),
    };
    for (id, value) in arpabet::source_token_features(&cmu, source_schema).values {
        let is_token_metadata = matches!(
            id.0.as_str(),
            "phonology.source_schema"
                | "phonology.source_schema_version"
                | "phonology.source_notation"
                | "phonology.source_token"
                | "phonology.base_symbol"
                | "phonology.canonical_base_symbol"
                | "phonology.stress"
                | "phonology.reduction_source"
                | "phonology.default_phone"
        );
        let is_standard_reduction =
            id.0 == "phonology.reduced_vowel" && arpabet::entry(&cmu.base).is_some();
        if is_token_metadata || is_standard_reduction {
            token.features.values.insert(id, value);
        }
    }
    if let Some(phone) = arpabet::reduced_phone_for_cmu(canonical_base, cmu.stress) {
        token.features.values.insert(
            FeatureId("phonology.default_phone".into()),
            Spec::Known(FeatureValue::Text(phone.as_str().to_string())),
        );
    }
    token
}

fn parse_variety_ipa(ipa: &str, variety: &LinguisticVariety) -> Vec<ParsedPronunciationToken> {
    let aliases = phoneme_aliases_by_length(variety);
    let chars = ipa
        .trim_matches('/')
        .chars()
        .filter(|ch| *ch != '.')
        .collect::<Vec<_>>();
    let has_explicit_stress = chars.iter().any(|ch| matches!(ch, 'ˈ' | 'ˌ'));
    let mut index = 0usize;
    let mut stress_pending = None;
    let mut candidate = Vec::new();
    while index < chars.len() {
        if matches!(chars[index], 'ˈ' | 'ˌ') {
            stress_pending = Some(if chars[index] == 'ˈ' {
                "primary"
            } else {
                "secondary"
            });
            index += 1;
            continue;
        }
        let rest = chars[index..].iter().collect::<String>();
        if let Some((phoneme, consumed)) = aliases.iter().find_map(|(alias, phoneme)| {
            rest.starts_with(alias)
                .then_some((*phoneme, alias.chars().count()))
        }) {
            let mut token = ParsedPronunciationToken {
                phoneme: phoneme.id.clone(),
                features: phoneme.features.clone(),
            };
            put_category(&mut token.features, "source_schema", "ipa");
            put_category(&mut token.features, "source_schema_version", "1");
            put_category(&mut token.features, "source_notation", "ipa");
            let source_token = chars[index..index + consumed].iter().collect::<String>();
            put_category(&mut token.features, "source_token", &source_token);
            if parsed_token_is_syllabic(&token) {
                let stress = stress_pending.take().unwrap_or(if has_explicit_stress {
                    "unstressed"
                } else {
                    "unknown"
                });
                put_category(&mut token.features, "stress", stress);
                put_category(
                    &mut token.features,
                    "reduction_source",
                    if stress == "unstressed" {
                        "inferred_from_lexical_stress"
                    } else {
                        "unspecified"
                    },
                );
            }
            candidate.push(token);
            index += consumed;
        } else {
            index += 1;
        }
    }
    candidate
}

fn put_category(features: &mut FeatureBundle, name: &str, value: &str) {
    features.values.insert(
        FeatureId(format!("phonology.{name}")),
        Spec::Known(FeatureValue::Category(value.into())),
    );
}

pub(crate) fn phoneme_aliases_by_length(variety: &LinguisticVariety) -> Vec<(String, &Phoneme)> {
    let mut aliases = Vec::new();
    for phoneme in variety.phonemes.phonemes.values() {
        aliases.push((phoneme_display_symbol(&phoneme.id).to_lowercase(), phoneme));
        for alias in &phoneme.aliases {
            aliases.push((alias.symbol.to_lowercase(), phoneme));
        }
    }
    aliases.sort_by(|left, right| right.0.len().cmp(&left.0.len()));
    aliases.dedup_by(|left, right| left.0 == right.0);
    aliases
}

fn parsed_token_is_syllabic(token: &ParsedPronunciationToken) -> bool {
    token
        .features
        .values
        .get(&FeatureId("phonology.syllabic".into()))
        == Some(&Spec::Known(FeatureValue::Bool(true)))
}

fn phoneme_display_symbol(id: &PhonemeId) -> &str {
    id.0.rsplit('.').next().unwrap_or(&id.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::variety_by_code;

    fn category<'a>(token: &'a ParsedPronunciationToken, name: &str) -> Option<&'a str> {
        match token
            .features
            .values
            .get(&FeatureId(format!("phonology.{name}")))?
        {
            Spec::Known(FeatureValue::Category(value)) | Spec::Known(FeatureValue::Text(value)) => {
                Some(value)
            }
            _ => None,
        }
    }

    #[test]
    fn cmudict_stress_is_token_context_not_phoneme_identity() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let tokens = ["AH0", "AH1", "AH2"].map(|symbol| {
            parse_pronunciation_candidate(
                &variety,
                &[symbol.to_string()],
                PronunciationNotation::Cmudict,
            )
            .pop()
            .expect("parsed token")
        });

        assert_eq!(tokens[0].phoneme, tokens[1].phoneme);
        assert_eq!(tokens[1].phoneme, tokens[2].phoneme);
        assert_eq!(category(&tokens[0], "stress"), Some("unstressed"));
        assert_eq!(category(&tokens[1], "stress"), Some("primary"));
        assert_eq!(category(&tokens[2], "stress"), Some("secondary"));
        assert_eq!(category(&tokens[1], "source_token"), Some("AH1"));
        assert_eq!(category(&tokens[1], "source_schema"), Some("cmudict"));
        assert!(!tokens[0].phoneme.0.contains(".phoneme.AX"));
    }

    #[test]
    fn direct_arpabet_and_cmudict_keep_distinct_provenance() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let parse = |notation| {
            parse_pronunciation_candidate(&variety, &["AH1".into()], notation)
                .pop()
                .expect("parsed token")
        };
        let direct = parse(PronunciationNotation::Arpabet);
        let cmudict = parse(PronunciationNotation::Cmudict);

        assert_eq!(direct.phoneme, cmudict.phoneme);
        assert_eq!(category(&direct, "source_schema"), Some("arpabet"));
        assert_eq!(category(&cmudict, "source_schema"), Some("cmudict"));
        assert_eq!(category(&direct, "source_token"), Some("AH1"));
    }

    #[test]
    fn external_ax_is_an_alias_for_the_merged_phoneme() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let parse = |symbol: &str| {
            parse_pronunciation_candidate(
                &variety,
                &[symbol.into()],
                PronunciationNotation::Arpabet,
            )
            .pop()
            .expect("parsed token")
        };
        let ax = parse("AX");
        let ah = parse("AH0");

        assert_eq!(ax.phoneme, ah.phoneme);
        assert_eq!(category(&ax, "source_token"), Some("AX"));
        assert_eq!(
            category(&ax, "reduction_source"),
            Some("explicit_source_symbol")
        );
    }

    #[test]
    fn ipa_g2p_output_preserves_primary_secondary_and_unstressed_context() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let tokens = parse_pronunciation_candidate(
            &variety,
            &["/əˈbʌv ˌʌ/".into()],
            PronunciationNotation::Ipa,
        );
        let vowels = tokens
            .iter()
            .filter(|token| parsed_token_is_syllabic(token))
            .collect::<Vec<_>>();

        assert_eq!(vowels.len(), 3);
        assert_eq!(vowels[0].phoneme, vowels[1].phoneme);
        assert_eq!(vowels[1].phoneme, vowels[2].phoneme);
        assert_eq!(category(vowels[0], "stress"), Some("unstressed"));
        assert_eq!(category(vowels[1], "stress"), Some("primary"));
        assert_eq!(category(vowels[2], "stress"), Some("secondary"));
        assert_eq!(category(vowels[0], "source_token"), Some("ə"));
        assert_eq!(category(vowels[0], "source_schema"), Some("ipa"));
    }

    #[test]
    fn ipa_without_stress_uses_explicit_unknown_context() {
        let variety = variety_by_code("en-US-GA").expect("GA");
        let token =
            parse_pronunciation_candidate(&variety, &["/ə/".into()], PronunciationNotation::Ipa)
                .pop()
                .expect("schwa token");

        assert_eq!(category(&token, "stress"), Some("unknown"));
        assert_eq!(category(&token, "reduction_source"), Some("unspecified"));
    }
}
