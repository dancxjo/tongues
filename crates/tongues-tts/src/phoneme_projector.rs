use std::collections::{BTreeMap, BTreeSet};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use speaking::{
    phoneme_default_phone_display_symbol, FeatureBundle, FeatureId, FeatureValue, PauseKind, Spec,
    TerminalPunctuation, UtterancePlan,
};

use crate::{LinguisticInputKind, LinguisticIntent, LinguisticProjector, ModelInputContract};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonemeCharactersConfig {
    pub pad: Option<String>,
    pub eos: Option<String>,
    pub bos: Option<String>,
    pub blank: Option<String>,
    pub characters: String,
    pub punctuations: String,
    pub phonemes: Option<String>,
    #[serde(default)]
    pub is_unique: bool,
    #[serde(default = "default_true")]
    pub is_sorted: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonemeTokenizerConfig {
    #[serde(default)]
    pub use_phonemes: bool,
    pub phoneme_language: Option<String>,
    #[serde(default)]
    pub add_blank: bool,
    #[serde(default)]
    pub enable_eos_bos_chars: bool,
    pub characters: PhonemeCharactersConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhonemeTokenIds {
    pub ids: Vec<i64>,
    pub projected_symbols: String,
}

/// Terminal projection from Tongues' linguistic IR into one checkpoint's
/// private symbol table.
#[derive(Debug, Clone)]
pub struct PhonemeVocabularyProjector {
    config: PhonemeTokenizerConfig,
    vocabulary: Vec<char>,
    symbol_to_id: BTreeMap<char, i64>,
    contract: ModelInputContract,
}

impl PhonemeVocabularyProjector {
    pub fn from_json5_str(source: &str) -> Result<Self> {
        let config: PhonemeTokenizerConfig =
            json5::from_str(source).context("failed to parse phoneme tokenizer config")?;
        Self::from_config(config)
    }

    pub fn from_config(config: PhonemeTokenizerConfig) -> Result<Self> {
        ensure!(
            config.use_phonemes,
            "phoneme vocabulary projector currently requires a phoneme-input model"
        );
        let mut symbols = config
            .characters
            .phonemes
            .as_deref()
            .unwrap_or(&config.characters.characters)
            .chars()
            .collect::<Vec<_>>();
        if config.characters.is_unique {
            symbols.sort_unstable();
            symbols.dedup();
        } else if config.characters.is_sorted {
            symbols.sort_unstable();
        }

        prepend_special(&mut symbols, config.characters.blank.as_deref());
        prepend_special(&mut symbols, config.characters.bos.as_deref());
        prepend_special(&mut symbols, config.characters.eos.as_deref());
        prepend_special(&mut symbols, config.characters.pad.as_deref());
        symbols.extend(config.characters.punctuations.chars());

        let mut symbol_to_id = BTreeMap::new();
        for (id, symbol) in symbols.iter().copied().enumerate() {
            let id = i64::try_from(id).context("phoneme vocabulary is too large")?;
            ensure!(
                symbol_to_id.insert(symbol, id).is_none(),
                "phoneme vocabulary contains duplicate symbol {symbol:?}"
            );
        }
        ensure!(!symbols.is_empty(), "phoneme vocabulary must not be empty");

        let variety = config
            .phoneme_language
            .as_deref()
            .map(normalize_variety)
            .unwrap_or_else(|| "*".to_string());
        let vocabulary_fingerprint = format!(
            "phoneme-vocabulary-v1:{}",
            symbols.iter().collect::<String>().escape_unicode()
        );
        let contract = ModelInputContract {
            kind: LinguisticInputKind::Phonemes,
            vocabulary_fingerprint,
            supported_varieties: vec![variety],
            consumes: BTreeSet::from([
                LinguisticIntent::Phonemes,
                LinguisticIntent::Phones,
                LinguisticIntent::Boundaries,
                LinguisticIntent::ProsodicBreaks,
            ]),
        };
        contract.validate()?;

        Ok(Self {
            config,
            vocabulary: symbols,
            symbol_to_id,
            contract,
        })
    }

    pub fn vocabulary(&self) -> &[char] {
        &self.vocabulary
    }

    pub fn symbol_id(&self, symbol: char) -> Option<i64> {
        self.symbol_to_id.get(&symbol).copied()
    }

    fn project_symbols(&self, plan: &UtterancePlan) -> Result<String> {
        self.contract.ensure_supports(plan)?;
        project_plan_symbols(plan, push_model_phoneme, "phoneme")
    }

    fn encode_symbols(&self, symbols: &str) -> Result<Vec<i64>> {
        let mut ids = symbols
            .chars()
            .map(|symbol| {
                self.symbol_id(symbol).with_context(|| {
                    format!(
                        "symbol {symbol:?} is not in the model vocabulary; projected sequence: {symbols:?}"
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;

        if self.config.add_blank {
            let blank = special_char(self.config.characters.blank.as_deref())
                .or_else(|| special_char(self.config.characters.pad.as_deref()))
                .context("add_blank requires a blank or pad symbol")?;
            let blank_id = self
                .symbol_id(blank)
                .context("model blank symbol is absent from the vocabulary")?;
            let mut interspersed = vec![blank_id; ids.len() * 2 + 1];
            for (slot, id) in interspersed.iter_mut().skip(1).step_by(2).zip(ids) {
                *slot = id;
            }
            ids = interspersed;
        }
        if self.config.enable_eos_bos_chars {
            let bos = special_char(self.config.characters.bos.as_deref())
                .context("BOS/EOS mode requires a BOS symbol")?;
            let eos = special_char(self.config.characters.eos.as_deref())
                .context("BOS/EOS mode requires an EOS symbol")?;
            ids.insert(0, self.symbol_id(bos).context("BOS symbol is absent")?);
            ids.push(self.symbol_id(eos).context("EOS symbol is absent")?);
        }
        Ok(ids)
    }
}

impl LinguisticProjector for PhonemeVocabularyProjector {
    type ModelInput = PhonemeTokenIds;

    fn contract(&self) -> &ModelInputContract {
        &self.contract
    }

    fn project(&self, plan: &UtterancePlan) -> Result<Self::ModelInput> {
        let projected_symbols = self.project_symbols(plan)?;
        let ids = self.encode_symbols(&projected_symbols)?;
        Ok(PhonemeTokenIds {
            ids,
            projected_symbols,
        })
    }
}

fn prepend_special(vocabulary: &mut Vec<char>, symbol: Option<&str>) {
    if let Some(symbol) = special_char(symbol) {
        vocabulary.insert(0, symbol);
    }
}

fn special_char(symbol: Option<&str>) -> Option<char> {
    let mut chars = symbol?.chars();
    let first = chars.next()?;
    chars.next().is_none().then_some(first)
}

fn normalize_variety(language: &str) -> String {
    let mut parts = language.split('-');
    let language = parts.next().unwrap_or(language).to_ascii_lowercase();
    match parts.next() {
        Some(region) => format!("{language}-{}", region.to_ascii_uppercase()),
        None => language,
    }
}

fn push_space(output: &mut String) {
    if !output.is_empty() && !output.ends_with(' ') {
        output.push(' ');
    }
}

fn push_model_phoneme(output: &mut String, ipa: &str) {
    for symbol in ipa.chars() {
        match symbol {
            // These are realization details absent from the released model's
            // phoneme inventory. They are discarded only at this model edge.
            'ʰ' | '˭' | 'ˈ' | 'ˌ' | 'ː' | 'ˑ' => {}
            // The old LJSpeech inventory has one rhotic-vowel symbol.
            'ɝ' => output.push('ɚ'),
            symbol => output.push(symbol),
        }
    }
}

pub(crate) fn project_plan_symbols(
    plan: &UtterancePlan,
    mut push_phoneme: impl FnMut(&mut String, &str),
    model_label: &str,
) -> Result<String> {
    let mut output = String::new();

    if !plan.intended_phonemes.is_empty() {
        let mut previous_word = None;
        let mut saw_indexed_word = false;
        for token in &plan.intended_phonemes {
            let word_index = token_word_index(&token.features);
            if let (Some(previous), Some(current)) = (previous_word, word_index) {
                if previous != current {
                    push_boundary_punctuation(&mut output, plan, previous);
                    push_space(&mut output);
                }
            }
            let Spec::Known(id) = &token.phoneme else {
                continue;
            };
            let ipa = phoneme_default_phone_display_symbol(id, &plan.variety);
            push_phoneme(&mut output, &ipa);
            if word_index.is_some() {
                saw_indexed_word = true;
                previous_word = word_index;
            }
        }
        if let Some(word_index) = previous_word {
            push_boundary_punctuation(&mut output, plan, word_index);
        } else if !saw_indexed_word {
            push_trailing_punctuation(&mut output, plan);
        }
    } else {
        let mut word_index = 0;
        for token in &plan.target_phones {
            let Spec::Known(phone) = &token.phone else {
                continue;
            };
            match phone.as_str() {
                "boundary.word" => {
                    push_boundary_punctuation(&mut output, plan, word_index);
                    push_space(&mut output);
                    word_index += 1;
                }
                "boundary.letter" => {}
                id => {
                    let ipa = id.strip_prefix("ipa.phone.").with_context(|| {
                        format!("{model_label} projection requires an IPA phone, got `{id}`")
                    })?;
                    push_phoneme(&mut output, ipa);
                }
            }
        }
        push_boundary_punctuation(&mut output, plan, word_index);
    }

    while output.ends_with(' ') {
        output.pop();
    }
    ensure!(
        !output.is_empty(),
        "{model_label} projection produced no symbols"
    );
    Ok(output)
}

fn token_word_index(features: &FeatureBundle) -> Option<usize> {
    match features
        .values
        .get(&FeatureId("orthography.word_index".into()))?
    {
        Spec::Known(FeatureValue::Number(value)) if value.is_finite() && *value >= 0.0 => {
            Some(*value as usize)
        }
        _ => None,
    }
}

fn push_boundary_punctuation(output: &mut String, plan: &UtterancePlan, word_index: usize) {
    for boundary in plan
        .boundaries
        .iter()
        .filter(|boundary| boundary.after_grapheme_index == word_index)
    {
        if boundary.pause == Some(PauseKind::Comma) && !output.ends_with(',') {
            output.push(',');
        }
        if let Some(terminal) = boundary.terminal {
            let punctuation = terminal_punctuation(terminal);
            if !output.ends_with(punctuation) {
                output.push(punctuation);
            }
        }
    }
}

fn push_trailing_punctuation(output: &mut String, plan: &UtterancePlan) {
    if let Some(terminal) = plan
        .boundaries
        .iter()
        .rev()
        .find_map(|boundary| boundary.terminal)
    {
        output.push(terminal_punctuation(terminal));
    } else if plan
        .boundaries
        .iter()
        .any(|boundary| boundary.pause == Some(PauseKind::Comma))
    {
        output.push(',');
    }
}

fn terminal_punctuation(terminal: TerminalPunctuation) -> char {
    match terminal {
        TerminalPunctuation::Period => '.',
        TerminalPunctuation::Question => '?',
        TerminalPunctuation::Exclamation => '!',
    }
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use speaking::{
        BoundaryKind, EvidenceProvenance, EvidenceSource, FeatureBundle, FeatureId, FeatureValue,
        PhoneId, PhoneToken, PhonemeId, PhonemeToken, Spec, SpeechBoundaryToken, TextSpan,
        UtteranceId, VarietyId,
    };

    use super::*;

    const CONFIG: &str = r#"{
      "use_phonemes": true,
      "phoneme_language": "en-us",
      "add_blank": false,
      "enable_eos_bos_chars": false,
      "characters": {
        "pad": "_",
        "eos": "~",
        "bos": "^",
        "blank": null,
        "characters": "abc",
        "punctuations": "!?., ",
        "phonemes": "tkɚʃ"
      }
    }"#;

    fn phone(id: &str) -> PhoneToken {
        PhoneToken {
            phone: Spec::Known(PhoneId(id.to_string().into())),
            span: None,
            features: FeatureBundle::default(),
            acoustic_evidence: vec![],
            confidence: 1.0,
            provenance: EvidenceProvenance {
                source: EvidenceSource::Rule,
                method: "test".into(),
                version: None,
            },
        }
    }

    fn phoneme(id: &str, word_index: usize) -> PhonemeToken {
        let mut features = FeatureBundle::default();
        features.values.insert(
            FeatureId("orthography.word_index".into()),
            Spec::Known(FeatureValue::Number(word_index as f64)),
        );
        PhonemeToken {
            phoneme: Spec::Known(PhonemeId(format!("en-US.phoneme.{id}").into())),
            span: None,
            features,
            realized_as: vec![],
            confidence: 1.0,
            provenance: EvidenceProvenance {
                source: EvidenceSource::Rule,
                method: "test".into(),
                version: None,
            },
        }
    }

    fn plan() -> UtterancePlan {
        UtterancePlan {
            id: UtteranceId("coqui-projector-test".into()),
            variety: VarietyId("en-US".into()),
            speaker: None,
            intended_text: Some("tur".into()),
            intended_morphemes: vec![],
            intended_phonemes: vec![],
            target_phones: vec![
                phone("ipa.phone.tʰ"),
                phone("ipa.phone.ɝ"),
                phone("boundary.word"),
                phone("ipa.phone.ʃ"),
            ],
            target_syllables: vec![],
            boundaries: vec![],
            target_prosody: Default::default(),
            target_acoustics: vec![],
            speaker_reference: None,
            style: None,
            provenance: EvidenceProvenance {
                source: EvidenceSource::TtsPlan,
                method: "test".into(),
                version: None,
            },
        }
    }

    #[test]
    fn vocabulary_matches_coqui_special_and_sorted_order() {
        let projector = PhonemeVocabularyProjector::from_json5_str(CONFIG).expect("projector");

        assert_eq!(
            projector.vocabulary().iter().collect::<String>(),
            "_~^ktɚʃ!?., "
        );
        assert_eq!(projector.symbol_id('_'), Some(0));
        assert_eq!(projector.symbol_id('k'), Some(3));
    }

    #[test]
    fn projects_native_ipa_phones_only_at_the_model_boundary() {
        let projector = PhonemeVocabularyProjector::from_json5_str(CONFIG).expect("projector");
        let projected = projector.project(&plan()).expect("projection");

        assert_eq!(projected.projected_symbols, "tɚ ʃ");
        assert_eq!(
            projected.ids,
            projected
                .projected_symbols
                .chars()
                .map(|symbol| projector.symbol_id(symbol).unwrap())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn checkpoint_projection_prefers_phonemes_and_keeps_punctuation_in_place() {
        let projector = PhonemeVocabularyProjector::from_json5_str(CONFIG).expect("projector");
        let mut plan = plan();
        plan.intended_phonemes = vec![phoneme("t", 0), phoneme("ɝ", 0), phoneme("ʃ", 1)];
        plan.target_phones = vec![
            phone("ipa.phone.tʰ"),
            phone("ipa.phone.ɝ"),
            phone("ipa.phone.ʃ"),
        ];
        plan.boundaries = vec![
            SpeechBoundaryToken {
                kind: BoundaryKind::Word,
                after_grapheme_index: 0,
                span: Some(TextSpan {
                    start_char: 2,
                    end_char: 3,
                }),
                terminal: None,
                pause: Some(PauseKind::Comma),
            },
            SpeechBoundaryToken {
                kind: BoundaryKind::Phrase,
                after_grapheme_index: 1,
                span: Some(TextSpan {
                    start_char: 5,
                    end_char: 6,
                }),
                terminal: Some(TerminalPunctuation::Period),
                pause: None,
            },
        ];

        let projected = projector.project(&plan).expect("checkpoint projection");

        assert_eq!(projected.projected_symbols, "tɚ, ʃ.");
        assert_eq!(
            projected.ids,
            projected
                .projected_symbols
                .chars()
                .map(|symbol| projector.symbol_id(symbol).unwrap())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn rejects_symbols_the_checkpoint_does_not_know() {
        let projector = PhonemeVocabularyProjector::from_json5_str(CONFIG).expect("projector");
        let mut plan = plan();
        plan.target_phones = vec![phone("ipa.phone.θ")];

        let error = projector.project(&plan).expect_err("unknown symbol");

        assert!(error.to_string().contains("not in the model vocabulary"));
    }

    #[test]
    fn loads_published_speedy_speech_vocabulary_when_provided() {
        let Some(path) = std::env::var_os("TONGUES_TEST_COQUI_SPEEDY_CONFIG") else {
            return;
        };
        let source = std::fs::read_to_string(path).expect("published config");
        let projector =
            PhonemeVocabularyProjector::from_json5_str(&source).expect("published vocabulary");

        assert_eq!(projector.vocabulary().len(), 130);
        assert_eq!(projector.symbol_id('_'), Some(0));
        assert_eq!(
            projector.contract().supported_varieties,
            vec!["en-US".to_string()]
        );

        // Golden output from Coqui 0.6.1 TTSTokenizer using this config and
        // Gruut's normalized phoneme string for the same sentence.
        let sentence = crate::utterance_plan_from_text(crate::SpeechRequest {
            text: "Morning light while the kettle began to sing.".into(),
            variety: "en-US".into(),
        })
        .expect("native sentence plan");
        let projected = projector
            .project(&sentence)
            .expect("published-compatible sentence projection");
        assert_eq!(
            projected.projected_symbols,
            "mɔɹnɪŋ laɪt waɪl ðə kɛtəl bɪɡæn tə sɪŋ."
        );
        assert_eq!(
            projected.ids,
            vec![
                14, 43, 77, 15, 63, 33, 129, 13, 3, 63, 21, 129, 24, 3, 63, 13, 129, 30, 48, 129,
                12, 50, 21, 48, 13, 129, 4, 63, 55, 28, 15, 129, 21, 48, 129, 20, 63, 33, 125,
            ]
        );
    }
}
