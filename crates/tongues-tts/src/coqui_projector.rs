use std::collections::{BTreeMap, BTreeSet};

use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use speaking::{PauseKind, Spec, TerminalPunctuation, UtterancePlan};

use crate::{LinguisticInputKind, LinguisticIntent, LinguisticProjector, ModelInputContract};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CoquiCharactersConfig {
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

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CoquiTokenizerConfig {
    #[serde(default)]
    pub use_phonemes: bool,
    pub phoneme_language: Option<String>,
    #[serde(default)]
    pub add_blank: bool,
    #[serde(default)]
    pub enable_eos_bos_chars: bool,
    pub characters: CoquiCharactersConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoquiTokenIds {
    pub ids: Vec<i64>,
    pub projected_symbols: String,
}

/// Terminal projection from Tongues' linguistic IR into one Coqui checkpoint's
/// private symbol table.
#[derive(Debug, Clone)]
pub struct CoquiLinguisticProjector {
    config: CoquiTokenizerConfig,
    vocabulary: Vec<char>,
    symbol_to_id: BTreeMap<char, i64>,
    contract: ModelInputContract,
}

impl CoquiLinguisticProjector {
    pub fn from_json5_str(source: &str) -> Result<Self> {
        let config: CoquiTokenizerConfig =
            json5::from_str(source).context("failed to parse Coqui tokenizer config")?;
        Self::from_config(config)
    }

    pub fn from_config(config: CoquiTokenizerConfig) -> Result<Self> {
        ensure!(
            config.use_phonemes,
            "Coqui linguistic projector currently requires a phoneme-input model"
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
            let id = i64::try_from(id).context("Coqui vocabulary is too large")?;
            ensure!(
                symbol_to_id.insert(symbol, id).is_none(),
                "Coqui vocabulary contains duplicate symbol {symbol:?}"
            );
        }
        ensure!(!symbols.is_empty(), "Coqui vocabulary must not be empty");

        let variety = config
            .phoneme_language
            .as_deref()
            .map(normalize_variety)
            .unwrap_or_else(|| "*".to_string());
        let vocabulary_fingerprint = format!(
            "coqui-phonemes-v1:{}",
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
        let mut output = String::new();

        if !plan.target_phones.is_empty() {
            for token in &plan.target_phones {
                let Spec::Known(phone) = &token.phone else {
                    continue;
                };
                match phone.as_str() {
                    "boundary.word" => push_space(&mut output),
                    "boundary.letter" => {}
                    id => {
                        let ipa = id.strip_prefix("ipa.phone.").with_context(|| {
                            format!("Coqui phoneme projection requires an IPA phone, got `{id}`")
                        })?;
                        push_model_phoneme(&mut output, ipa);
                    }
                }
            }
        } else {
            for token in &plan.intended_phonemes {
                let realized = token.realized_as.iter().find_map(|phone| {
                    let Spec::Known(id) = &phone.phone else {
                        return None;
                    };
                    id.as_str().strip_prefix("ipa.phone.")
                });
                let realized = realized.with_context(|| {
                    let id = match &token.phoneme {
                        Spec::Known(id) => id.0.as_ref(),
                        _ => "<unknown>",
                    };
                    format!("phoneme `{id}` has no IPA realization for Coqui projection")
                })?;
                push_model_phoneme(&mut output, realized);
            }
        }

        while output.ends_with(' ') {
            output.pop();
        }
        if let Some(terminal) = plan
            .boundaries
            .iter()
            .rev()
            .find_map(|token| token.terminal)
        {
            output.push(match terminal {
                TerminalPunctuation::Period => '.',
                TerminalPunctuation::Question => '?',
                TerminalPunctuation::Exclamation => '!',
            });
        } else if plan
            .boundaries
            .iter()
            .any(|token| token.pause == Some(PauseKind::Comma))
        {
            output.push(',');
        }
        ensure!(!output.is_empty(), "Coqui projection produced no symbols");
        Ok(output)
    }

    fn encode_symbols(&self, symbols: &str) -> Result<Vec<i64>> {
        let mut ids = symbols
            .chars()
            .map(|symbol| {
                self.symbol_id(symbol).with_context(|| {
                    format!(
                        "symbol {symbol:?} is not in the Coqui model vocabulary; projected sequence: {symbols:?}"
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;

        if self.config.add_blank {
            let blank = special_char(self.config.characters.blank.as_deref())
                .or_else(|| special_char(self.config.characters.pad.as_deref()))
                .context("Coqui add_blank requires a blank or pad symbol")?;
            let blank_id = self
                .symbol_id(blank)
                .context("Coqui blank symbol is absent from the vocabulary")?;
            let mut interspersed = vec![blank_id; ids.len() * 2 + 1];
            for (slot, id) in interspersed.iter_mut().skip(1).step_by(2).zip(ids) {
                *slot = id;
            }
            ids = interspersed;
        }
        if self.config.enable_eos_bos_chars {
            let bos = special_char(self.config.characters.bos.as_deref())
                .context("Coqui BOS/EOS mode requires a BOS symbol")?;
            let eos = special_char(self.config.characters.eos.as_deref())
                .context("Coqui BOS/EOS mode requires an EOS symbol")?;
            ids.insert(0, self.symbol_id(bos).context("BOS symbol is absent")?);
            ids.push(self.symbol_id(eos).context("EOS symbol is absent")?);
        }
        Ok(ids)
    }
}

impl LinguisticProjector for CoquiLinguisticProjector {
    type ModelInput = CoquiTokenIds;

    fn contract(&self) -> &ModelInputContract {
        &self.contract
    }

    fn project(&self, plan: &UtterancePlan) -> Result<Self::ModelInput> {
        let projected_symbols = self.project_symbols(plan)?;
        let ids = self.encode_symbols(&projected_symbols)?;
        Ok(CoquiTokenIds {
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
            'ʰ' | '˭' => {}
            // The old LJSpeech inventory has one rhotic-vowel symbol.
            'ɝ' => output.push('ɚ'),
            symbol => output.push(symbol),
        }
    }
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use speaking::{
        EvidenceProvenance, EvidenceSource, FeatureBundle, PhoneId, PhoneToken, Spec, UtteranceId,
        VarietyId,
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
        "punctuations": "!? ",
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
        let projector = CoquiLinguisticProjector::from_json5_str(CONFIG).expect("projector");

        assert_eq!(
            projector.vocabulary().iter().collect::<String>(),
            "_~^ktɚʃ!? "
        );
        assert_eq!(projector.symbol_id('_'), Some(0));
        assert_eq!(projector.symbol_id('k'), Some(3));
    }

    #[test]
    fn projects_native_ipa_phones_only_at_the_model_boundary() {
        let projector = CoquiLinguisticProjector::from_json5_str(CONFIG).expect("projector");
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
    fn rejects_symbols_the_checkpoint_does_not_know() {
        let projector = CoquiLinguisticProjector::from_json5_str(CONFIG).expect("projector");
        let mut plan = plan();
        plan.target_phones = vec![phone("ipa.phone.θ")];

        let error = projector.project(&plan).expect_err("unknown symbol");

        assert!(error
            .to_string()
            .contains("not in the Coqui model vocabulary"));
    }

    #[test]
    fn loads_published_speedy_speech_vocabulary_when_provided() {
        let Some(path) = std::env::var_os("TONGUES_TEST_COQUI_SPEEDY_CONFIG") else {
            return;
        };
        let source = std::fs::read_to_string(path).expect("published config");
        let projector =
            CoquiLinguisticProjector::from_json5_str(&source).expect("published vocabulary");

        assert_eq!(projector.vocabulary().len(), 130);
        assert_eq!(projector.symbol_id('_'), Some(0));
        assert_eq!(
            projector.contract().supported_varieties,
            vec!["en-US".to_string()]
        );
    }
}
