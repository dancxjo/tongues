use serde::{Deserialize, Serialize};

use crate::acoustics::AcousticFrame;
use crate::evidence::EvidenceProvenance;
use crate::ids::{SpeakerId, UtteranceId, VarietyId};
use crate::morphology::MorphemeToken;
use crate::orthography::GraphemeToken;
use crate::phonemicize::PhonemicizeOutput;
use crate::phonology::{PhoneToken, PhonemeToken};
use crate::prosody::{ProsodyTrack, Syllable};
use crate::segment::SpeechBoundaryToken;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Utterance {
    pub id: UtteranceId,
    pub variety: VarietyId,
    pub speaker: Option<SpeakerId>,
    pub text: Option<String>,
    pub morphemes: Vec<MorphemeToken>,
    pub graphemes: Vec<GraphemeToken>,
    pub phonemes: Vec<PhonemeToken>,
    pub phones: Vec<PhoneToken>,
    pub syllables: Vec<Syllable>,
    #[serde(default)]
    pub boundaries: Vec<SpeechBoundaryToken>,
    pub acoustic_frames: Vec<AcousticFrame>,
    pub prosody: ProsodyTrack,
    pub provenance: EvidenceProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UtterancePlan {
    pub id: UtteranceId,
    pub variety: VarietyId,
    pub speaker: Option<SpeakerId>,
    pub intended_text: Option<String>,
    pub intended_morphemes: Vec<MorphemeToken>,
    pub intended_phonemes: Vec<PhonemeToken>,
    pub target_phones: Vec<PhoneToken>,
    #[serde(default)]
    pub target_syllables: Vec<Syllable>,
    #[serde(default)]
    pub boundaries: Vec<SpeechBoundaryToken>,
    pub target_prosody: ProsodyTrack,
    pub target_acoustics: Vec<AcousticFrame>,
    #[serde(default)]
    pub speaker_reference: Option<SpeakerReference>,
    pub style: Option<StyleRef>,
    pub provenance: EvidenceProvenance,
}

impl From<&PhonemicizeOutput> for UtterancePlan {
    fn from(output: &PhonemicizeOutput) -> Self {
        output.clone().into()
    }
}

impl From<PhonemicizeOutput> for UtterancePlan {
    fn from(output: PhonemicizeOutput) -> Self {
        let provenance = output.provenance.clone();
        Self::from_phonemicized(output, UtteranceId("speaking.plan".into()), provenance)
    }
}

impl UtterancePlan {
    /// Build a plan from phonemicizer output with caller-owned plan identity
    /// and provenance.
    ///
    /// This is the single field-mapping boundary between the pronunciation
    /// pipeline and the canonical speaking plan.
    pub fn from_phonemicized(
        output: PhonemicizeOutput,
        id: UtteranceId,
        provenance: EvidenceProvenance,
    ) -> Self {
        Self {
            id,
            variety: output.variety,
            speaker: None,
            intended_text: Some(output.text),
            intended_morphemes: Vec::new(),
            intended_phonemes: output.phonemes,
            target_phones: output.phones,
            target_syllables: output.syllables,
            boundaries: output.boundaries,
            target_prosody: output.prosody,
            target_acoustics: Vec::new(),
            speaker_reference: None,
            style: None,
            provenance,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerReference {
    pub description: Option<String>,
    pub source: SpeakerReferenceSource,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerReferenceSource {
    ReferenceAudio { uri: String },
    Embedding { space: String, values: Vec<f32> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StyleRef {
    pub description: Option<String>,
    pub source: StyleSource,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StyleSource {
    ReferenceAudio { uri: String },
    Embedding { kind: String, values: Vec<f32> },
    Manual,
    Inferred,
}
