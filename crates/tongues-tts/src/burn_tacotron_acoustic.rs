use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use serde_json::Value;
use speaking::{StyleSource, UtterancePlan};

use crate::{
    AcousticArtifact, AcousticModel, AcousticOutputContract, AudioFeatureConfig, ConditioningKind,
    EmbeddingContract, InferenceRuntime, LinguisticInputKind, LinguisticIntent,
    LinguisticProjector, ModelInputContract, PhonemeTokenIds, PhonemeTokenizerConfig,
    PhonemeVocabularyProjector, Spectrogram, SpectrogramContract, SpectrogramLayout,
    SpeechModelCapabilities, SpeechModelFamily, SpeechSynthesisRequest, Tacotron2,
    TacotronConditioning, TacotronInferenceConfig,
};

pub const CAPACITRON_LATENT_SPACE: &str = "capacitron-standard-normal-latent-v1";

#[derive(Debug, Clone)]
pub struct TacotronGraphemeProjector {
    vocabulary: Vec<char>,
    symbol_to_id: BTreeMap<char, i64>,
    contract: ModelInputContract,
}

impl TacotronGraphemeProjector {
    pub fn from_config(config: &PhonemeTokenizerConfig) -> Result<Self> {
        ensure!(
            !config.use_phonemes,
            "grapheme projector cannot load a phoneme-input checkpoint"
        );
        let mut vocabulary = config.characters.characters.chars().collect::<Vec<_>>();
        if config.characters.is_unique {
            vocabulary.sort_unstable();
            vocabulary.dedup();
        } else if config.characters.is_sorted {
            // Modern Coqui character configs request sorting explicitly.
            // Released legacy configs did not carry the field and their
            // literal alphabet must remain in checkpoint order.
            vocabulary.sort_unstable();
        }
        prepend_special(&mut vocabulary, config.characters.blank.as_deref());
        prepend_special(&mut vocabulary, config.characters.bos.as_deref());
        prepend_special(&mut vocabulary, config.characters.eos.as_deref());
        prepend_special(&mut vocabulary, config.characters.pad.as_deref());
        vocabulary.extend(config.characters.punctuations.chars());
        let mut symbol_to_id = BTreeMap::new();
        for (index, symbol) in vocabulary.iter().copied().enumerate() {
            ensure!(
                symbol_to_id.insert(symbol, i64::try_from(index)?).is_none(),
                "Tacotron grapheme vocabulary contains duplicate symbol {symbol:?}"
            );
        }
        ensure!(
            !vocabulary.is_empty(),
            "Tacotron grapheme vocabulary is empty"
        );
        let supported_variety = config
            .phoneme_language
            .as_deref()
            .map(normalize_variety)
            .unwrap_or_else(|| "*".into());
        let contract = ModelInputContract {
            kind: LinguisticInputKind::Graphemes,
            vocabulary_fingerprint: format!(
                "tacotron-grapheme-v1:{}",
                vocabulary.iter().collect::<String>().escape_unicode()
            ),
            supported_varieties: vec![supported_variety],
            consumes: BTreeSet::from([LinguisticIntent::Text]),
        };
        contract.validate()?;
        Ok(Self {
            vocabulary,
            symbol_to_id,
            contract,
        })
    }

    pub fn vocabulary(&self) -> &[char] {
        &self.vocabulary
    }
}

impl LinguisticProjector for TacotronGraphemeProjector {
    type ModelInput = PhonemeTokenIds;

    fn contract(&self) -> &ModelInputContract {
        &self.contract
    }

    fn project(&self, plan: &UtterancePlan) -> Result<Self::ModelInput> {
        self.contract.ensure_supports(plan)?;
        let intended = plan
            .intended_text
            .as_deref()
            .context("grapheme Tacotron checkpoint requires intended text")?;
        let projected_symbols = intended.split_whitespace().collect::<Vec<_>>().join(" ");
        ensure!(
            !projected_symbols.is_empty(),
            "Tacotron text is empty after whitespace normalization"
        );
        let ids = projected_symbols
            .chars()
            .map(|symbol| {
                self.symbol_to_id.get(&symbol).copied().with_context(|| {
                    format!(
                        "grapheme {symbol:?} is not in the Tacotron checkpoint vocabulary; \
                         normalize the text with the checkpoint's declared cleaner before synthesis"
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PhonemeTokenIds {
            ids,
            projected_symbols,
        })
    }
}

#[derive(Debug, Clone)]
enum TacotronProjector {
    Graphemes(TacotronGraphemeProjector),
    Phonemes(Box<PhonemeVocabularyProjector>),
}

impl TacotronProjector {
    fn contract(&self) -> &ModelInputContract {
        match self {
            Self::Graphemes(projector) => projector.contract(),
            Self::Phonemes(projector) => projector.contract(),
        }
    }

    fn vocabulary_len(&self) -> usize {
        match self {
            Self::Graphemes(projector) => projector.vocabulary().len(),
            Self::Phonemes(projector) => projector.vocabulary().len(),
        }
    }

    fn project(&self, plan: &UtterancePlan) -> Result<PhonemeTokenIds> {
        match self {
            Self::Graphemes(projector) => projector.project(plan),
            Self::Phonemes(projector) => projector.project(plan),
        }
    }
}

pub struct BurnTacotron2Acoustic<B: Backend> {
    model: Tacotron2<B>,
    projector: TacotronProjector,
    output_contract: SpectrogramContract,
    conditioning_contracts: Vec<EmbeddingContract>,
    device: B::Device,
}

impl<B: Backend> BurnTacotron2Acoustic<B> {
    pub fn load(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
    ) -> Result<Self> {
        let config_path = config_path.as_ref();
        let source = fs::read_to_string(config_path)
            .with_context(|| format!("failed to read model config {}", config_path.display()))?;
        let root: Value = json5::from_str(&source)
            .with_context(|| format!("invalid model config {}", config_path.display()))?;
        let model_config =
            TacotronInferenceConfig::from_json_value(&root).map_err(anyhow::Error::new)?;
        let mut tokenizer: PhonemeTokenizerConfig =
            json5::from_str(&source).context("invalid Tacotron tokenizer config")?;
        if root
            .get("characters")
            .and_then(|characters| characters.get("is_sorted"))
            .is_none()
        {
            // The old tokenizer preserved the alphabet exactly as written.
            // `PhonemeCharactersConfig` defaults this field for modern Coqui
            // configs, so recover the legacy behavior explicitly.
            tokenizer.characters.is_sorted = false;
        }
        let projector = if tokenizer.use_phonemes {
            TacotronProjector::Phonemes(Box::new(PhonemeVocabularyProjector::from_config(
                tokenizer,
            )?))
        } else {
            TacotronProjector::Graphemes(TacotronGraphemeProjector::from_config(&tokenizer)?)
        };
        ensure!(
            projector.vocabulary_len() == model_config.num_chars,
            "Tacotron vocabulary has {} entries but checkpoint expects {}",
            projector.vocabulary_len(),
            model_config.num_chars
        );
        let output_contract = AudioFeatureConfig::from_file(config_path)?.mel_contract()?;
        ensure!(
            output_contract.layout == SpectrogramLayout::FramesByBins,
            "Tacotron acoustic adapter requires frame-major shared spectrograms"
        );
        ensure!(
            output_contract.bins == model_config.out_channels,
            "audio config declares {} mel bins but Tacotron emits {}",
            output_contract.bins,
            model_config.out_channels
        );
        let conditioning_contracts = model_config
            .capacitron
            .as_ref()
            .map(|capacitron| {
                vec![EmbeddingContract {
                    kind: ConditioningKind::Style,
                    space: CAPACITRON_LATENT_SPACE.into(),
                    dimensions: capacitron.embedding_dim,
                    l2_normalized: false,
                }]
            })
            .unwrap_or_default();
        let model = model_config
            .init_tacotron2::<B>(&device)
            .map_err(anyhow::Error::new)?
            .load_checkpoint(checkpoint_path)
            .map_err(anyhow::Error::new)?;
        Ok(Self {
            model,
            projector,
            output_contract,
            conditioning_contracts,
            device,
        })
    }

    pub fn model(&self) -> &Tacotron2<B> {
        &self.model
    }

    fn conditioning(&self, request: &SpeechSynthesisRequest) -> Result<TacotronConditioning<B>> {
        let Some(style) = request.plan.style.as_ref() else {
            return Ok(TacotronConditioning::default());
        };
        let Some(contract) = self.conditioning_contracts.first() else {
            bail!("style conditioning was requested for a non-Capacitron Tacotron checkpoint");
        };
        let StyleSource::Embedding { kind, values } = &style.source else {
            bail!(
                "native Capacitron currently accepts an explicit `{}` latent embedding; \
                 reference-audio posterior encoding is not available",
                contract.space
            );
        };
        ensure!(
            kind == &contract.space,
            "Capacitron style embedding space `{kind}` does not match `{}`",
            contract.space
        );
        ensure!(
            values.len() == contract.dimensions,
            "Capacitron style embedding has {} values; expected {}",
            values.len(),
            contract.dimensions
        );
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "Capacitron style embedding contains non-finite values"
        );
        Ok(TacotronConditioning {
            style_embedding: Some(Tensor::from_data(
                TensorData::new(values.clone(), [1, values.len()]),
                &self.device,
            )),
        })
    }
}

impl<B: Backend> AcousticModel for BurnTacotron2Acoustic<B> {
    fn runtime(&self) -> InferenceRuntime {
        InferenceRuntime::Burn
    }

    fn capabilities(&self) -> SpeechModelCapabilities {
        SpeechModelCapabilities {
            family: SpeechModelFamily::AcousticModel,
            supports_named_speakers: false,
            supports_languages: false,
            // Explicit latent embeddings are supported, but the Capacitron
            // reference encoder is intentionally not claimed yet.
            supports_reference_audio: false,
            supports_voice_conversion: false,
            integrated_vocoder: false,
        }
    }

    fn input_contract(&self) -> &ModelInputContract {
        self.projector.contract()
    }

    fn conditioning_contracts(&self) -> &[EmbeddingContract] {
        &self.conditioning_contracts
    }

    fn output_contract(&self) -> AcousticOutputContract {
        AcousticOutputContract::Spectrogram(self.output_contract.clone())
    }

    fn synthesize(&mut self, request: &SpeechSynthesisRequest) -> Result<AcousticArtifact> {
        ensure!(
            request.plan.speaker.is_none() && request.options.speaker_id.is_none(),
            "native Tacotron 2 backend currently supports single-speaker checkpoints"
        );
        ensure!(
            request.plan.speaker_reference.is_none(),
            "native Tacotron 2 backend does not accept speaker reference audio"
        );
        let projected = self.projector.project(&request.plan)?;
        let token_count = projected.ids.len();
        let token_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(projected.ids, [1, token_count]),
            &self.device,
        );
        let conditioning = self.conditioning(request)?;
        let output = self
            .model
            .inference(token_ids, conditioning, None)
            .map_err(anyhow::Error::new)
            .context("native Tacotron 2 inference failed")?;
        let frames = output.mel.dims()[1];
        let values = output
            .mel
            .into_data()
            .to_vec::<f32>()
            .context("Tacotron output is not f32")?;
        let spectrogram = Spectrogram {
            contract: self.output_contract.clone(),
            frames,
            values,
        };
        spectrogram.validate()?;
        Ok(AcousticArtifact::Spectrogram(spectrogram))
    }
}

fn prepend_special(vocabulary: &mut Vec<char>, symbol: Option<&str>) {
    let Some(symbol) = symbol else {
        return;
    };
    let mut chars = symbol.chars();
    if let (Some(symbol), None) = (chars.next(), chars.next()) {
        vocabulary.insert(0, symbol);
    }
}

fn normalize_variety(language: &str) -> String {
    let mut parts = language.split('-');
    let language = parts.next().unwrap_or(language).to_ascii_lowercase();
    match parts.next() {
        Some(region) => format!("{language}-{}", region.to_ascii_uppercase()),
        None => language,
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    use super::*;
    use crate::{SpeechRequest, SynthesisOptions};

    type TestBackend = NdArray<f32>;

    fn legacy_grapheme_config() -> PhonemeTokenizerConfig {
        json5::from_str(
            r#"{
                use_phonemes: false,
                phoneme_language: "en-us",
                characters: {
                    pad: "", eos: "", bos: "",
                    characters: "_-!'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                    punctuations: "", phonemes: "",
                    is_sorted: false
                }
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn legacy_released_alphabet_preserves_checkpoint_order() {
        let projector = TacotronGraphemeProjector::from_config(&legacy_grapheme_config()).unwrap();
        assert_eq!(projector.vocabulary().len(), 64);
        assert_eq!(projector.vocabulary()[0], '_');
        assert_eq!(projector.vocabulary()[11], ' ');
        assert_eq!(projector.vocabulary()[63], 'z');
    }

    #[test]
    fn repeated_characters_remain_repeated_checkpoint_tokens() {
        let projector = TacotronGraphemeProjector::from_config(&legacy_grapheme_config()).unwrap();
        let mut plan = crate::utterance_plan_from_text(crate::SpeechRequest {
            text: "Bookkeeper".into(),
            variety: "en-US".into(),
        })
        .unwrap();
        plan.intended_text = Some("Bookkeeper".into());
        let projected = projector.project(&plan).unwrap();
        assert_eq!(projected.projected_symbols, "Bookkeeper");
        assert_eq!(projected.ids.len(), 10);
        assert_eq!(projected.ids[1], projected.ids[2]);
        assert_eq!(projected.ids[3], projected.ids[4]);
        assert_eq!(projected.ids[5], projected.ids[6]);
    }

    #[test]
    fn unsupported_grapheme_reports_the_exact_character() {
        let projector = TacotronGraphemeProjector::from_config(&legacy_grapheme_config()).unwrap();
        let mut plan = crate::utterance_plan_from_text(crate::SpeechRequest {
            text: "cafe".into(),
            variety: "en-US".into(),
        })
        .unwrap();
        plan.intended_text = Some("café".into());
        let error = projector.project(&plan).unwrap_err().to_string();
        assert!(error.contains("'é'"));
        assert!(error.contains("checkpoint vocabulary"));
    }

    #[test]
    fn released_ddc_checkpoint_synthesizes_native_mel_when_available() {
        let Some(config_path) = std::env::var_os("TONGUES_TEST_COQUI_TACOTRON2_CONFIG") else {
            return;
        };
        let checkpoint_path = std::env::var_os("TONGUES_TEST_COQUI_TACOTRON2_MODEL")
            .expect("TONGUES_TEST_COQUI_TACOTRON2_MODEL must accompany config");
        let mut backend = BurnTacotron2Acoustic::<TestBackend>::load(
            config_path,
            checkpoint_path,
            NdArrayDevice::Cpu,
        )
        .expect("released Tacotron2-DDC acoustic backend");
        let mut plan = crate::utterance_plan_from_text(SpeechRequest {
            text: "Hello.".into(),
            variety: "en-US".into(),
        })
        .unwrap();
        // The released grapheme checkpoint consumes its normalized text at
        // the terminal adapter rather than the plan's phone realization.
        plan.intended_text = Some("Hello.".into());
        let artifact = backend
            .synthesize(&SpeechSynthesisRequest {
                plan,
                options: SynthesisOptions::default(),
            })
            .expect("native Tacotron2 mel synthesis");
        let AcousticArtifact::Spectrogram(mel) = artifact else {
            panic!("Tacotron2 must emit a spectrogram");
        };
        assert_eq!(mel.contract.bins, 80);
        assert!(mel.frames > 0);
        assert!(mel.values.iter().all(|value| value.is_finite()));
    }
}
