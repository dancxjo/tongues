use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result};
use speaking::UtterancePlan;

use crate::{
    phoneme_projector::project_plan_symbols, vits_config::ImportedVitsConfig, LinguisticInputKind,
    LinguisticIntent, LinguisticProjector, ModelInputContract, PhonemeTokenIds,
};

/// Terminal projection from Tongues' linguistic IR into the vocabulary
/// embedded in a VITS checkpoint.
#[derive(Debug, Clone)]
pub(crate) struct VitsLinguisticProjector {
    add_blank: bool,
    vocabulary: Vec<String>,
    symbol_to_id: BTreeMap<String, i64>,
    blank_id: i64,
    contract: ModelInputContract,
}

impl VitsLinguisticProjector {
    pub(crate) fn from_json5_str(source: &str) -> Result<Self> {
        Self::from_config(ImportedVitsConfig::from_json5_str(source)?)
    }

    pub(crate) fn from_config(config: ImportedVitsConfig) -> Result<Self> {
        config.validate()?;
        let vocabulary = config.vocabulary();
        let mut symbol_to_id = BTreeMap::new();
        for (id, symbol) in vocabulary.iter().enumerate() {
            let id = i64::try_from(id).context("VITS vocabulary is too large")?;
            // VitsCharacters deliberately permits duplicates. Python's dict
            // construction resolves them to the final embedding row.
            symbol_to_id.insert(symbol.clone(), id);
        }
        let blank_id = symbol_to_id
            .get("<BLNK>")
            .copied()
            .context("VITS blank token is absent from the vocabulary")?;
        let variety = config
            .phoneme_language
            .clone()
            .unwrap_or_else(|| "*".to_string());
        let contract = ModelInputContract {
            kind: LinguisticInputKind::Phonemes,
            vocabulary_fingerprint: format!("vits-compat-v1:{vocabulary:?}"),
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
            add_blank: config.add_blank,
            vocabulary,
            symbol_to_id,
            blank_id,
            contract,
        })
    }

    pub(crate) fn vocabulary(&self) -> &[String] {
        &self.vocabulary
    }

    pub(crate) fn symbol_id(&self, symbol: &str) -> Option<i64> {
        self.symbol_to_id.get(symbol).copied()
    }

    pub(crate) fn blank_id(&self) -> i64 {
        self.blank_id
    }

    fn project_symbols(&self, plan: &UtterancePlan) -> Result<String> {
        self.contract.ensure_supports(plan)?;
        project_plan_symbols(plan, |output, ipa| self.push_phone(output, ipa), "VITS")
    }

    fn push_phone(&self, output: &mut String, ipa: &str) {
        for symbol in ipa.chars() {
            // Tongues distinguishes unaspirated stops with the IPA extension
            // `˭`. The released VCTK VITS inventory does not, so lower that
            // detail to the base stop at this checkpoint boundary. Preserve it
            // for any future checkpoint whose private vocabulary supports it.
            if symbol == '˭' && self.symbol_id("˭").is_none() {
                continue;
            }
            output.push(symbol);
        }
    }

    fn encode_symbols(&self, symbols: &str) -> Result<Vec<i64>> {
        let ids = symbols
            .chars()
            .map(|symbol| {
                self.symbol_id(&symbol.to_string()).with_context(|| {
                    format!(
                        "symbol {symbol:?} is not in the VITS vocabulary; projected sequence: {symbols:?}"
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if !self.add_blank {
            return Ok(ids);
        }

        let mut interspersed = vec![self.blank_id; ids.len() * 2 + 1];
        for (slot, id) in interspersed.iter_mut().skip(1).step_by(2).zip(ids) {
            *slot = id;
        }
        Ok(interspersed)
    }
}

impl LinguisticProjector for VitsLinguisticProjector {
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

#[cfg(test)]
mod tests {
    use speaking::{
        EvidenceProvenance, EvidenceSource, FeatureBundle, PhoneId, PhoneToken, Spec, UtteranceId,
        VarietyId,
    };

    use super::*;

    fn phone(id: &str) -> PhoneToken {
        PhoneToken {
            phone: Spec::Known(PhoneId(id.to_string().into())),
            span: None,
            features: FeatureBundle::default(),
            acoustic_evidence: vec![],
            confidence: 1.0,
            provenance: EvidenceProvenance {
                source: EvidenceSource::Rule,
                method: "vits-projector-test".into(),
                version: None,
            },
        }
    }

    fn plan() -> UtterancePlan {
        UtterancePlan {
            id: UtteranceId("vits-projector-test".into()),
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
                method: "vits-projector-test".into(),
                version: None,
            },
        }
    }

    #[test]
    fn projects_native_ipa_and_intersperses_the_model_blank() {
        let projector =
            VitsLinguisticProjector::from_config(crate::vits_config::test_imported_vits_config())
                .expect("VITS projector");

        assert_eq!(projector.symbol_id("'"), Some(5));
        assert_eq!(projector.blank_id(), 9);
        let projected = projector.project(&plan()).expect("native IPA projection");
        assert_eq!(projected.projected_symbols, "tʰɝ ʃ");
        assert_eq!(
            projected.ids.len(),
            2 * projected.projected_symbols.chars().count() + 1
        );
        assert!(projected
            .ids
            .iter()
            .step_by(2)
            .all(|id| *id == projector.blank_id()));
    }

    #[test]
    fn published_vocabulary_and_native_ipa_projection_match_when_available() {
        let Some(path) = std::env::var_os("TONGUES_TEST_COQUI_VITS_CONFIG") else {
            return;
        };
        let source = std::fs::read_to_string(path).expect("published VITS config");
        let projector = VitsLinguisticProjector::from_json5_str(&source).expect("VITS projector");

        assert_eq!(projector.vocabulary().len(), 179);
        assert_eq!(projector.symbol_id("'"), Some(176));
        assert_eq!(projector.symbol_id("A"), Some(17));
        assert_eq!(projector.symbol_id("a"), Some(43));
        assert_eq!(projector.symbol_id("ɑ"), Some(69));
        assert_eq!(projector.symbol_id("ɝ"), Some(88));
        assert_eq!(projector.symbol_id("ˈ"), Some(156));
        assert_eq!(projector.blank_id(), 178);

        let projected = projector.project(&plan()).expect("native IPA projection");
        assert_eq!(projected.projected_symbols, "tʰɝ ʃ");
        assert_eq!(
            projected.ids.len(),
            2 * projected.projected_symbols.chars().count() + 1
        );
        assert!(projected
            .ids
            .iter()
            .step_by(2)
            .all(|id| *id == projector.blank_id()));
        // Golden output from Coqui 0.6.1 TTSTokenizer for the same normalized
        // phoneme input, including its leading/trailing blank interspersion.
        assert_eq!(
            projected.ids,
            vec![178, 62, 178, 162, 178, 88, 178, 16, 178, 131, 178]
        );
        assert_eq!(
            projected.ids[1],
            projector.symbol_id("t").expect("t embedding")
        );
        assert_eq!(
            projected.ids[3],
            projector.symbol_id("ʰ").expect("aspiration embedding")
        );

        let sentence = crate::utterance_plan_from_text(crate::SpeechRequest {
            text: "Morning light rested on the cedar trees while the kettle began to sing.".into(),
            variety: "en-US".into(),
        })
        .expect("native sentence plan");
        let projected = projector
            .project(&sentence)
            .expect("published-compatible sentence projection");
        assert!(!projected.projected_symbols.contains('˭'));
        assert!(projected.projected_symbols.contains("ɹɛstəd"));
    }

    #[test]
    fn lowers_unaspirated_stops_when_the_model_has_no_extension_symbol() {
        let projector =
            VitsLinguisticProjector::from_config(crate::vits_config::test_imported_vits_config())
                .expect("VITS projector");
        let mut plan = plan();
        plan.target_phones = vec![phone("ipa.phone.t˭")];

        let projected = projector.project(&plan).expect("compatible projection");

        assert_eq!(projected.projected_symbols, "t");
        assert_eq!(
            projected.ids[1],
            projector.symbol_id("t").expect("t embedding")
        );
    }

    #[test]
    fn unknown_distal_symbol_fails_at_the_model_boundary() {
        let projector =
            VitsLinguisticProjector::from_config(crate::vits_config::test_imported_vits_config())
                .expect("VITS projector");
        let mut plan = plan();
        plan.target_phones = vec![phone("ipa.phone.θ")];

        let error = projector.project(&plan).expect_err("unsupported symbol");
        assert!(error.to_string().contains("not in the VITS vocabulary"));
    }
}
