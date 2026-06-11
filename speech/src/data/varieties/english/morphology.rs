use crate::data::lexicons::cmudict::{CmuPhoneme, CmuStress, bundled};
use crate::feature::FeatureBundle;
use crate::ids::MorphemeId;
use crate::morphology::{
    Morpheme, MorphemeKind, MorphemeToken, MorphologicalAction, MorphologicalRule,
    MorphologicalTrigger, Morphology, compose_morpheme_tokens, finalize_word_pronunciation,
};
use crate::phonology::PhonemeToken;
use crate::spec::Spec;
use crate::variety::LinguisticVariety;

// Helper function to build a pronunciation list of PhonemeTokens from CMU symbols.
fn make_pronunciation(variety_id: &str, cmu_symbols: &[&str]) -> Vec<PhonemeToken> {
    cmu_symbols
        .iter()
        .map(|s| {
            let cmu = CmuPhoneme::parse(s);
            let raw_symbol = cmu.raw_symbol();
            let features = crate::data::notation::arpabet::cmu_token_features(&cmu);
            PhonemeToken {
                phoneme: Spec::Known(crate::data::notation::arpabet::phoneme_id(
                    variety_id,
                    &raw_symbol,
                )),
                span: None,
                features,
                realized_as: Vec::new(),
                confidence: 1.0,
                provenance: crate::evidence::EvidenceProvenance {
                    source: crate::evidence::EvidenceSource::Lexicon,
                    method: "morphology lookup".into(),
                    version: None,
                },
            }
        })
        .collect()
}

pub fn english_morphology(variety_id: &str) -> Morphology {
    let mut morphemes = std::collections::HashMap::new();

    // Suffixes
    let suffixes = &[
        ("-ness", vec!["N", "AH0", "S"]),
        ("-less", vec!["L", "AH0", "S"]),
        ("-ly", vec!["L", "IY0"]),
        ("-able", vec!["AH0", "B", "AH0", "L"]),
        ("-ible", vec!["AH0", "B", "AH0", "L"]),
        ("-ative", vec!["AH0", "T", "IH0", "V"]),
        ("-ativity", vec!["IH0", "V", "AH0", "T", "IY0"]),
        ("-tion", vec!["SH", "AH0", "N"]),
        ("-sion", vec!["SH", "AH0", "N"]),
        ("-ity", vec!["AH0", "T", "IY0"]),
        ("-ology", vec!["AA1", "L", "AH0", "JH", "IY0"]),
        ("-graphy", vec!["G", "R", "AH0", "F", "IY0"]),
        ("-phobia", vec!["F", "OW1", "B", "IY0", "AH0"]),
        ("-rrhea", vec!["R", "IY1", "AH0"]),
        ("-ing", vec!["IH0", "NG"]),
    ];

    for &(form, ref cmu_symbols) in suffixes {
        let id = MorphemeId(form.to_string());
        let pronunciation = make_pronunciation(variety_id, cmu_symbols);
        let morpheme = Morpheme {
            id: id.clone(),
            form: form.to_string(),
            kind: MorphemeKind::Suffix,
            gloss: None,
            features: FeatureBundle::default(),
            pronunciation,
        };
        morphemes.insert(id, morpheme);
    }

    // Prefixes
    let prefixes = &[
        ("in-", vec!["IH2", "N"]),
        ("un-", vec!["AH2", "N"]),
        ("re-", vec!["R", "IY2"]),
        ("de-", vec!["D", "IY2"]),
        ("dis-", vec!["D", "IH2", "S"]),
        ("mis-", vec!["M", "IH2", "S"]),
        ("non-", vec!["N", "AA2", "N"]),
        ("pre-", vec!["P", "R", "IY2"]),
        ("anti-", vec!["AE2", "N", "T", "IY0"]),
        ("co-", vec!["K", "OW2"]),
        ("sub-", vec!["S", "AH2", "B"]),
    ];

    for &(form, ref cmu_symbols) in prefixes {
        let id = MorphemeId(form.to_string());
        let pronunciation = make_pronunciation(variety_id, cmu_symbols);
        let morpheme = Morpheme {
            id: id.clone(),
            form: form.to_string(),
            kind: MorphemeKind::Prefix,
            gloss: None,
            features: FeatureBundle::default(),
            pronunciation,
        };
        morphemes.insert(id, morpheme);
    }

    // Rules
    let mut rules = Vec::new();

    // 1. y -> i replacement rule
    rules.push(MorphologicalRule {
        id: "english_y_to_i".to_string(),
        name: "English y to i before suffixes".to_string(),
        triggers: vec![
            MorphologicalTrigger::LeftEndsWith("y".to_string()),
            MorphologicalTrigger::RightMorphemeKind(MorphemeKind::Suffix),
        ],
        actions: vec![MorphologicalAction::ReplaceLeftSuffix {
            find: "y".to_string(),
            replace: "i".to_string(),
        }],
    });

    // 2. drop final e before vowels
    for vowel in &["a", "e", "i", "o", "u"] {
        rules.push(MorphologicalRule {
            id: format!("english_drop_e_before_{vowel}"),
            name: format!("English drop final e before suffix starting with {vowel}"),
            triggers: vec![
                MorphologicalTrigger::LeftEndsWith("e".to_string()),
                MorphologicalTrigger::RightStartsWith(vowel.to_string()),
            ],
            actions: vec![MorphologicalAction::DropLeftFinalE],
        });
    }

    // 3. Stress shifts
    rules.push(MorphologicalRule {
        id: "stress_attraction_ity".to_string(),
        name: "Stress attraction for -ity".to_string(),
        triggers: vec![MorphologicalTrigger::RightMorphemeId("-ity".to_string())],
        actions: vec![MorphologicalAction::SetPrimaryStressOnLeftSyllableFromEnd(
            1,
        )],
    });

    rules.push(MorphologicalRule {
        id: "stress_attraction_ivity".to_string(),
        name: "Stress attraction for -ivity".to_string(),
        triggers: vec![MorphologicalTrigger::RightMorphemeId("-ivity".to_string())],
        actions: vec![MorphologicalAction::SetPrimaryStressOnLeftSyllableFromEnd(
            2,
        )],
    });

    rules.push(MorphologicalRule {
        id: "stress_attraction_ology".to_string(),
        name: "Stress attraction for -ology".to_string(),
        triggers: vec![MorphologicalTrigger::RightMorphemeId("-ology".to_string())],
        actions: vec![MorphologicalAction::SetPrimaryStressOnLeftSyllableFromEnd(
            3,
        )],
    });

    rules.push(MorphologicalRule {
        id: "stress_attraction_graphy".to_string(),
        name: "Stress attraction for -graphy".to_string(),
        triggers: vec![MorphologicalTrigger::RightMorphemeId("-graphy".to_string())],
        actions: vec![MorphologicalAction::SetPrimaryStressOnLeftSyllableFromEnd(
            3,
        )],
    });

    Morphology { morphemes, rules }
}

/// Recursively decomposes the word using the variety's morphology database.
pub fn decompose_word(variety: &LinguisticVariety, word: &str) -> Option<Vec<MorphemeToken>> {
    let word_lower = word.to_lowercase();
    let morph_db = variety.morphology.as_ref()?;

    // 1. Try Suffixes
    for (morpheme_id, morpheme) in &morph_db.morphemes {
        if morpheme.kind == MorphemeKind::Suffix {
            let suffix_trimmed = morpheme.form.trim_start_matches('-');
            if word_lower.ends_with(suffix_trimmed) && word_lower.len() > suffix_trimmed.len() {
                let stem = &word_lower[..word_lower.len() - suffix_trimmed.len()];

                // Try different spelling mutation patterns for the stem:
                let mut stem_candidates = vec![stem.to_string()];
                if stem.ends_with('i') {
                    let mut stem_y = stem[..stem.len() - 1].to_string();
                    stem_y.push('y');
                    stem_candidates.push(stem_y);
                }
                if !stem.ends_with('e') {
                    let mut stem_e = stem.to_string();
                    stem_e.push('e');
                    stem_candidates.push(stem_e);
                }
                if stem.len() > 1 {
                    let bytes = stem.as_bytes();
                    if bytes[bytes.len() - 1] == bytes[bytes.len() - 2] {
                        stem_candidates.push(stem[..stem.len() - 1].to_string());
                    }
                }

                for stem_cand in stem_candidates {
                    if let Some(mut parts) = decompose_word(variety, &stem_cand) {
                        parts.push(MorphemeToken {
                            morpheme: Spec::Known(morpheme_id.clone()),
                            surface: suffix_trimmed.to_string(),
                            span: None,
                            features: FeatureBundle::default(),
                            pronunciation: morpheme.pronunciation.clone(),
                            confidence: 1.0,
                        });
                        return Some(parts);
                    }
                }
            }
        }
    }

    // 2. Try Prefixes
    for (morpheme_id, morpheme) in &morph_db.morphemes {
        if morpheme.kind == MorphemeKind::Prefix {
            let prefix_trimmed = morpheme.form.trim_end_matches('-');
            if word_lower.starts_with(prefix_trimmed) && word_lower.len() > prefix_trimmed.len() {
                let stem = &word_lower[prefix_trimmed.len()..];
                if let Some(mut parts) = decompose_word(variety, stem) {
                    parts.insert(
                        0,
                        MorphemeToken {
                            morpheme: Spec::Known(morpheme_id.clone()),
                            surface: prefix_trimmed.to_string(),
                            span: None,
                            features: FeatureBundle::default(),
                            pronunciation: morpheme.pronunciation.clone(),
                            confidence: 1.0,
                        },
                    );
                    return Some(parts);
                }
            }
        }
    }

    // Check if the whole word is in the dictionary first (as a root/base morpheme).
    let entry = bundled().lookup_entry(&word_lower);
    if !entry.candidates.is_empty() {
        let morpheme_id = MorphemeId(word_lower.clone());
        let pr_symbols: Vec<String> = entry.candidates[0].iter().map(|p| p.raw_symbol()).collect();
        let pr_strs: Vec<&str> = pr_symbols.iter().map(|s: &String| s.as_str()).collect();
        let pronunciation = make_pronunciation(&variety.id.0, &pr_strs);
        return Some(vec![MorphemeToken {
            morpheme: Spec::Known(morpheme_id),
            surface: word_lower,
            span: None,
            features: FeatureBundle::default(),
            pronunciation,
            confidence: 1.0,
        }]);
    }

    None
}

/// Combines decomposed MorphemeTokens into a single coherent pronunciation, applying morphotactic rules.
pub fn compose_pronunciation(
    variety: &LinguisticVariety,
    parts: &[MorphemeToken],
) -> Vec<CmuPhoneme> {
    if parts.is_empty() {
        return Vec::new();
    }

    let mut tokens = parts.to_vec();
    let morph_db = match &variety.morphology {
        Some(db) => db,
        None => return Vec::new(),
    };

    // Apply morphological rules
    compose_morpheme_tokens(&mut tokens, &morph_db.morphemes, &morph_db.rules);

    // Concatenate all PhonemeTokens
    let mut all_phonemes = Vec::new();
    for token in &tokens {
        all_phonemes.extend(token.pronunciation.clone());
    }

    // Resolve stress conflicts (keeping the last primary stress)
    finalize_word_pronunciation(&mut all_phonemes);

    // Convert PhonemeTokens back to CmuPhonemes
    let stress_id = crate::ids::FeatureId("phonology.stress".to_string());
    let base_id = crate::ids::FeatureId("phonology.base_symbol".to_string());

    all_phonemes
        .into_iter()
        .map(|p| {
            let base = if let Some(Spec::Known(crate::feature::FeatureValue::Category(b))) =
                p.features.values.get(&base_id)
            {
                b.clone()
            } else {
                let phoneme_str = match &p.phoneme {
                    Spec::Known(id) => id.0.as_str(),
                    _ => "",
                };
                let parts: Vec<&str> = phoneme_str.split('.').collect();
                parts.last().cloned().unwrap_or("AH").to_string()
            };

            let stress = if let Some(Spec::Known(crate::feature::FeatureValue::Category(s))) =
                p.features.values.get(&stress_id)
            {
                match s.as_str() {
                    "primary" => Some(CmuStress::Primary),
                    "secondary" => Some(CmuStress::Secondary),
                    "unstressed" => Some(CmuStress::Unstressed),
                    _ => None,
                }
            } else {
                None
            };

            CmuPhoneme { base, stress }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::VarietyId;

    fn test_variety() -> LinguisticVariety {
        let mut var = LinguisticVariety {
            id: VarietyId("en-US".to_string()),
            language: crate::ids::LanguageId("en".to_string()),
            name: "American English".to_string(),
            feature_system: crate::feature::FeatureSystem::default(),
            phonemes: crate::phonology::PhonemeInventory::default(),
            phones: crate::phonetics::PhoneInventory::default(),
            allophone_rules: Vec::new(),
            epenthesis_rules: Vec::new(),
            weak_forms: Vec::new(),
            orthographic_unit_pronunciations: Vec::new(),
            phonotactics: None,
            orthography: None,
            morphology: None,
            acoustic_profile: None,
            prosody_profile: None,
            status: crate::variety::VarietyStatus::Experimental,
            implementation_status: crate::variety::VarietyImplementationStatus::Complete,
        };
        var.morphology = Some(english_morphology(&var.id.0));
        var
    }

    #[test]
    fn test_decompose_talkativeness() {
        let variety = test_variety();
        let parts = decompose_word(&variety, "talkativeness");
        assert!(parts.is_some(), "Should decompose talkativeness");
        let parts = parts.unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].surface, "talk");
        assert_eq!(parts[1].surface, "ative");
        assert_eq!(parts[2].surface, "ness");

        let pron = compose_pronunciation(&variety, &parts);
        let symbols: Vec<String> = pron.iter().map(|p| p.raw_symbol()).collect();
        assert_eq!(symbols[0], "T");
        assert!(symbols.contains(&"AH0".to_string()));
        assert!(symbols.contains(&"N".to_string()));
    }

    #[test]
    fn test_decompose_wordiness() {
        let variety = test_variety();
        let parts = decompose_word(&variety, "wordiness");
        assert!(parts.is_some(), "Should decompose wordiness");
        let parts = parts.unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].surface, "wordy");
        assert_eq!(parts[1].surface, "ness");

        let pron = compose_pronunciation(&variety, &parts);
        let symbols: Vec<String> = pron.iter().map(|p| p.raw_symbol()).collect();
        assert_eq!(symbols[0], "W");
        assert!(symbols.contains(&"N".to_string()));
    }

    #[test]
    fn test_decompose_unforgivingly() {
        let variety = test_variety();
        let parts = decompose_word(&variety, "unforgivingly");
        assert!(parts.is_some(), "Should decompose unforgivingly");
        let parts = parts.unwrap();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0].surface, "un");
        assert_eq!(parts[1].surface, "forgive");
        assert_eq!(parts[2].surface, "ing");
        assert_eq!(parts[3].surface, "ly");

        // Test spelling composition
        let mut tokens_for_spelling = parts.clone();
        let morph_db = variety.morphology.as_ref().unwrap();
        compose_morpheme_tokens(
            &mut tokens_for_spelling,
            &morph_db.morphemes,
            &morph_db.rules,
        );
        let spelling: String = tokens_for_spelling
            .iter()
            .map(|t| t.surface.as_str())
            .collect();
        assert_eq!(spelling, "unforgivingly");

        // Test pronunciation composition
        let pron = compose_pronunciation(&variety, &parts);
        let symbols: Vec<String> = pron.iter().map(|p| p.raw_symbol()).collect();
        assert_eq!(symbols[0], "AH2");
        assert_eq!(symbols[1], "N");
        assert_eq!(symbols[2], "F");
        assert_eq!(symbols[3], "ER0");
        assert_eq!(symbols[4], "G");
        assert_eq!(symbols[5], "IH1");
        assert_eq!(symbols[6], "V");
        assert_eq!(symbols[7], "IH0");
        assert_eq!(symbols[8], "NG");
        assert_eq!(symbols[9], "L");
        assert_eq!(symbols[10], "IY0");
    }
}
