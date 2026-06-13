use serde::{Deserialize, Serialize};

use crate::phonology::PhoneToken;
use crate::segment::{BoundaryKind, SyllablePosition};
use crate::spec::Spec;
use crate::time::TimeSpan;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stress {
    Primary,
    Secondary,
    Unstressed,
    Reduced,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Syllable {
    pub phones: Vec<PhoneToken>,
    pub stress: Spec<Stress>,
    #[serde(default)]
    pub phone_positions: Vec<SyllablePosition>,
    pub span: Option<TimeSpan>,
    pub nucleus_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ProsodyTrack {
    pub pitch: Curve,
    pub energy: Curve,
    pub speaking_rate: Curve,
    pub breaks: Vec<ProsodicBreak>,
    pub labels: Vec<ProsodicLabel>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Curve {
    pub points: Vec<CurvePoint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    pub time_s: f64,
    pub value: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProsodicBreak {
    pub after_s: f64,
    pub duration_s: Spec<f32>,
    pub boundary: BoundaryKind,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProsodicLabel {
    pub span: TimeSpan,
    pub kind: ProsodicLabelKind,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProsodicLabelKind {
    Emphasis,
    Focus,
    QuestionRise,
    AlternativeQuestionRise,
    AlternativeQuestionFall,
    ContinuationRise,
    FinalFall,
    Hesitation,
    Repair,
    Backchannel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProsodicContext {
    PhraseInitial,
    PhraseMedial,
    PhraseFinal,
    TurnInitial,
    TurnFinal,
    Emphasized,
    Deemphasized,
    FastSpeech,
    CarefulSpeech,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ProsodyProfile {
    pub default_pitch_hz: Option<f32>,
    pub default_rate_syllables_per_second: Option<f32>,
    pub rhythm_class: Option<String>,
}
