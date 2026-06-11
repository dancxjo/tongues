use serde::{Deserialize, Serialize};
use speech::{ProsodyTrack, SpeakerId, StyleRef, UtterancePlan};

use crate::plan::{BackendSynthesisPlan, StyleTts2PlanOptions, prepare_styletts2_plan};
use crate::symbols::styletts2_en_us_symbol_set;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StyleTts2SynthesisRequest {
    pub backend_plan: BackendSynthesisPlan,
    pub speaker: Option<SpeakerId>,
    pub style: Option<StyleRef>,
    pub speaker_reference_audio_uri: Option<String>,
    pub style_reference_audio_uri: Option<String>,
    pub prosody: ProsodyTrack,
}

impl StyleTts2SynthesisRequest {
    pub fn from_plan(utterance_plan: UtterancePlan) -> Self {
        let backend_plan = prepare_styletts2_plan(
            &utterance_plan,
            &styletts2_en_us_symbol_set(),
            StyleTts2PlanOptions::default(),
        )
        .expect("default StyleTTS2 plan preparation should succeed");
        Self::from_backend_plan(
            backend_plan,
            utterance_plan.speaker.clone(),
            utterance_plan.style.clone(),
            utterance_plan.target_prosody.clone(),
        )
    }

    pub fn from_backend_plan(
        backend_plan: BackendSynthesisPlan,
        speaker: Option<SpeakerId>,
        style: Option<StyleRef>,
        prosody: ProsodyTrack,
    ) -> Self {
        Self {
            backend_plan,
            speaker,
            style,
            speaker_reference_audio_uri: None,
            style_reference_audio_uri: None,
            prosody,
        }
    }

    pub fn with_speaker_reference_audio_uri(mut self, uri: impl Into<String>) -> Self {
        self.speaker_reference_audio_uri = Some(uri.into());
        self
    }

    pub fn with_style_reference_audio_uri(mut self, uri: impl Into<String>) -> Self {
        self.style_reference_audio_uri = Some(uri.into());
        self
    }

    pub fn is_empty(&self) -> bool {
        self.backend_plan.chunks.is_empty()
            || self
                .backend_plan
                .chunks
                .iter()
                .all(|chunk| chunk.symbols.is_empty())
    }
}
