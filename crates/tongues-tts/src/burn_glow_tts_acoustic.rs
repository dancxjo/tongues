//! Native acoustic-model adapter for Glow-TTS and SC-GlowTTS.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use speaking::SpeakerReferenceSource;

use crate::{
    AcousticArtifact, AcousticModel, AcousticOutputContract, ConditioningKind, EmbeddingContract,
    GlowTts, GlowTtsInferenceConfig, GlowTtsOutput, InferenceRuntime, LinguisticProjector,
    ModelInputContract, PhonemeVocabularyProjector, Spectrogram, SpectrogramContract,
    SpeechModelCapabilities, SpeechModelFamily, SpeechSynthesisRequest, StochasticGlowTts,
};

pub const COQUI_D_VECTOR_SPACE: &str = "coqui-speaker-encoder-d-vector-v1";

enum GlowDurationBackend<B: Backend> {
    Deterministic(GlowTts<B>),
    Stochastic(StochasticGlowTts<B>),
}

impl<B: Backend> GlowDurationBackend<B> {
    fn config(&self) -> &GlowTtsInferenceConfig {
        match self {
            Self::Deterministic(model) => model.config(),
            Self::Stochastic(model) => model.config(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn inference(
        &self,
        token_ids: Tensor<B, 2, Int>,
        lengths: Tensor<B, 1, Int>,
        conditioning: Option<Tensor<B, 3>>,
        explicit_durations: Option<Tensor<B, 2>>,
        length_scale: f64,
        acoustic_noise_scale: f64,
        duration_noise_scale: f64,
        seed: Option<u64>,
    ) -> Result<GlowTtsOutput<B>> {
        match self {
            Self::Deterministic(model) => {
                ensure!(
                    duration_noise_scale == 0.0,
                    "this deterministic-duration Glow-TTS checkpoint does not accept duration noise"
                );
                model
                    .inference(
                        token_ids,
                        lengths,
                        conditioning,
                        explicit_durations,
                        length_scale,
                        acoustic_noise_scale,
                        seed,
                    )
                    .map_err(anyhow::Error::new)
            }
            Self::Stochastic(model) => model
                .inference(
                    token_ids,
                    lengths,
                    conditioning,
                    explicit_durations,
                    length_scale,
                    acoustic_noise_scale,
                    duration_noise_scale,
                    seed,
                )
                .map_err(anyhow::Error::new),
        }
    }
}

/// Burn-native Glow-TTS acoustic backend.
///
/// SC-GlowTTS accepts a shared `SpeakerReferenceSource::Embedding`; it does not
/// introduce a provider-specific request field or execute a speaker encoder.
pub struct BurnGlowTtsAcoustic<B: Backend> {
    model: GlowDurationBackend<B>,
    projector: PhonemeVocabularyProjector,
    output_contract: SpectrogramContract,
    conditioning_contracts: Vec<EmbeddingContract>,
    device: B::Device,
}

impl<B: Backend> BurnGlowTtsAcoustic<B> {
    pub fn load(
        config_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        device: B::Device,
    ) -> Result<Self> {
        let config_path = config_path.as_ref();
        let config = GlowTtsInferenceConfig::from_file(config_path)?;
        let projector = PhonemeVocabularyProjector::from_legacy_config_with_duplicates(
            config.tokenizer.clone(),
        )?;
        let output_contract = config.output_contract()?;
        let conditioning_channels = config.network.speaker_conditioning_channels();
        let conditioning_contracts = if conditioning_channels == 0 {
            Vec::new()
        } else {
            vec![EmbeddingContract {
                kind: ConditioningKind::Speaker,
                space: COQUI_D_VECTOR_SPACE.into(),
                dimensions: conditioning_channels,
                l2_normalized: true,
            }]
        };
        let model = if config.network.use_sdp {
            GlowDurationBackend::Stochastic(
                StochasticGlowTts::load(config, checkpoint_path, device.clone())
                    .context("failed to load native stochastic-duration Glow-TTS model")?,
            )
        } else {
            GlowDurationBackend::Deterministic(
                GlowTts::load(config, checkpoint_path, device.clone())
                    .context("failed to load native Glow-TTS model")?,
            )
        };
        Ok(Self {
            model,
            projector,
            output_contract,
            conditioning_contracts,
            device,
        })
    }

    pub fn projector(&self) -> &PhonemeVocabularyProjector {
        &self.projector
    }

    pub fn synthesize_tensor(&self, request: &SpeechSynthesisRequest) -> Result<Tensor<B, 3>> {
        self.projector.contract().ensure_supports(&request.plan)?;
        let projected = self.projector.project(&request.plan)?;
        let token_count = projected.ids.len();
        ensure!(token_count > 0, "Glow-TTS projection produced no tokens");
        let token_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(projected.ids, [1, token_count]),
            &self.device,
        );
        let lengths = Tensor::<B, 1, Int>::from_data(
            TensorData::new(vec![token_count as i64], [1]),
            &self.device,
        );
        let conditioning = self.conditioning_from_request(request)?;
        let explicit_durations = request
            .options
            .durations
            .as_ref()
            .map(|durations| {
                ensure!(
                    durations.len() == token_count,
                    "Glow-TTS received {} explicit durations for {token_count} projected tokens",
                    durations.len()
                );
                Ok(Tensor::<B, 2>::from_data(
                    TensorData::new(
                        durations.iter().map(|value| *value as f32).collect(),
                        [1, token_count],
                    ),
                    &self.device,
                ))
            })
            .transpose()?;
        ensure!(
            request.options.pitch.is_none()
                && request.options.pitch_scale.is_none()
                && request.options.pitch_shift.is_none(),
            "Glow-TTS does not expose pitch conditioning"
        );
        let length_scale = request
            .options
            .length_scale
            .unwrap_or(self.model.config().network.length_scale);
        let noise_scale = request
            .options
            .noise_scale
            .unwrap_or(self.model.config().network.inference_noise_scale);
        let duration_noise_scale = request.options.noise_w.unwrap_or_else(|| {
            if self.model.config().network.use_sdp {
                self.model.config().network.inference_noise_scale_dp
            } else {
                0.0
            }
        });
        let output = self.model.inference(
            token_ids,
            lengths,
            conditioning,
            explicit_durations,
            f64::from(length_scale),
            f64::from(noise_scale),
            f64::from(duration_noise_scale),
            request.options.seed,
        )?;
        Ok(output.mel)
    }

    fn conditioning_from_request(
        &self,
        request: &SpeechSynthesisRequest,
    ) -> Result<Option<Tensor<B, 3>>> {
        let Some(contract) = self.conditioning_contracts.first() else {
            ensure!(
                request.plan.speaker.is_none() && request.options.speaker_id.is_none(),
                "this Glow-TTS checkpoint is single-speaker"
            );
            ensure!(
                request.plan.speaker_reference.is_none(),
                "this Glow-TTS checkpoint does not accept speaker conditioning"
            );
            return Ok(None);
        };
        ensure!(
            request.plan.speaker.is_none() && request.options.speaker_id.is_none(),
            "SC-GlowTTS consumes a speaker embedding, not a named or numeric speaker ID"
        );
        let reference = request
            .plan
            .speaker_reference
            .as_ref()
            .context("SC-GlowTTS requires a speaker-reference embedding")?;
        let SpeakerReferenceSource::Embedding { space, values } = &reference.source else {
            anyhow::bail!(
                "SC-GlowTTS requires a precomputed speaker embedding; reference audio needs a separate speaker encoder"
            );
        };
        ensure!(
            space == &contract.space,
            "SC-GlowTTS speaker embedding space `{space}` does not match `{}`",
            contract.space
        );
        ensure!(
            values.len() == contract.dimensions && values.iter().all(|value| value.is_finite()),
            "SC-GlowTTS speaker embedding must contain {} finite values",
            contract.dimensions
        );
        Ok(Some(Tensor::from_data(
            TensorData::new(values.clone(), [1, contract.dimensions, 1]),
            &self.device,
        )))
    }
}

impl<B: Backend> AcousticModel for BurnGlowTtsAcoustic<B> {
    fn runtime(&self) -> InferenceRuntime {
        InferenceRuntime::Burn
    }

    fn capabilities(&self) -> SpeechModelCapabilities {
        SpeechModelCapabilities {
            family: SpeechModelFamily::AcousticModel,
            supports_named_speakers: false,
            supports_languages: false,
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
        let mel = self.synthesize_tensor(request)?;
        let frames = mel.dims()[1];
        let values = mel
            .into_data()
            .to_vec::<f32>()
            .context("Glow-TTS output is not f32")?;
        let spectrogram = Spectrogram {
            contract: self.output_contract.clone(),
            frames,
            values,
        };
        spectrogram.validate()?;
        Ok(AcousticArtifact::Spectrogram(spectrogram))
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    use super::*;
    use crate::{utterance_plan_from_text, SpeechRequest, SynthesisOptions};

    type TestBackend = NdArray<f32>;

    #[test]
    #[ignore = "requires the checksum-pinned published Glow-TTS artifact"]
    fn published_acoustic_backend_emits_neutral_mel() {
        let config_path = std::env::var_os("TONGUES_TEST_GLOW_CONFIG")
            .expect("TONGUES_TEST_GLOW_CONFIG is required");
        let checkpoint_path = std::env::var_os("TONGUES_TEST_GLOW_CHECKPOINT")
            .expect("TONGUES_TEST_GLOW_CHECKPOINT is required");
        let device = NdArrayDevice::Cpu;
        let backend =
            BurnGlowTtsAcoustic::<TestBackend>::load(config_path, checkpoint_path, device)
                .expect("published acoustic backend");
        assert!(matches!(
            backend.output_contract(),
            AcousticOutputContract::Spectrogram(_)
        ));
        assert!(backend.conditioning_contracts().is_empty());
    }

    #[test]
    #[ignore = "requires the checksum-pinned published Glow-TTS artifact"]
    fn published_acoustic_backend_covers_input_matrix() {
        let config_path = std::env::var_os("TONGUES_TEST_GLOW_CONFIG")
            .expect("TONGUES_TEST_GLOW_CONFIG is required");
        let checkpoint_path = std::env::var_os("TONGUES_TEST_GLOW_CHECKPOINT")
            .expect("TONGUES_TEST_GLOW_CHECKPOINT is required");
        let mut backend = BurnGlowTtsAcoustic::<TestBackend>::load(
            config_path,
            checkpoint_path,
            NdArrayDevice::Cpu,
        )
        .expect("published acoustic backend");
        for (label, text) in [
            ("short", "Hi."),
            ("ordinary", "Morning light rested on cedar trees."),
            (
                "long",
                "Morning light rested on the cedar trees while the kettle began to sing, and beyond the windows the quiet neighborhood slowly woke beneath a pale summer sky.",
            ),
            ("repeated", "Never never never, very very quietly."),
            (
                "punctuation",
                "Wait—what? Yes: commas, pauses; questions, and exclamations!",
            ),
        ] {
            let request = SpeechSynthesisRequest {
                plan: utterance_plan_from_text(SpeechRequest {
                    text: text.into(),
                    variety: "en-US".into(),
                })
                .unwrap_or_else(|error| panic!("{label} plan: {error:#}")),
                options: SynthesisOptions {
                    seed: Some(27),
                    ..Default::default()
                },
            };
            let artifact = backend
                .synthesize(&request)
                .unwrap_or_else(|error| panic!("{label} inference: {error:#}"));
            let AcousticArtifact::Spectrogram(spectrogram) = artifact else {
                panic!("{label} did not emit a spectrogram");
            };
            assert!(spectrogram.frames > 0, "{label}");
            assert!(
                spectrogram.values.iter().all(|value| value.is_finite()),
                "{label}"
            );
        }
    }
}
