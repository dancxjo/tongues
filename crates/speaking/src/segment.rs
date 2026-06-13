use serde::{Deserialize, Serialize};

use crate::feature::FeatureBundle;
use crate::ids::{PhoneId, PhonemeId};
use crate::prosody::{ProsodicContext, Stress};
use crate::spec::Spec;
use crate::time::TextSpan;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentStatus {
    Core,
    Marginal,
    Borrowed,
    Allophonic,
    Archiphonemic,
    Reconstructed,
    Experimental,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolAlias {
    pub system: String,
    pub symbol: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Environment {
    pub before: Vec<SegmentMatcher>,
    pub after: Vec<SegmentMatcher>,
    pub word_position: Spec<WordPosition>,
    pub syllable_position: Spec<SyllablePosition>,
    pub stress_context: Spec<Stress>,
    pub prosodic_context: Spec<ProsodicContext>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentMatcher {
    Phone(PhoneId),
    Phoneme(PhonemeId),
    FeatureBundle(FeatureBundle),
    Boundary(BoundaryKind),
    Any,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WordPosition {
    Initial,
    Medial,
    Final,
    Isolated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyllablePosition {
    Onset,
    Nucleus,
    Coda,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryKind {
    Phone,
    Syllable,
    Morpheme,
    Word,
    Phrase,
    BreathGroup,
    Turn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalPunctuation {
    Period,
    Question,
    Exclamation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PauseKind {
    Comma,
    AlternativeQuestionRise,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechBoundaryToken {
    pub kind: BoundaryKind,
    pub after_grapheme_index: usize,
    pub span: Option<TextSpan>,
    pub terminal: Option<TerminalPunctuation>,
    pub pause: Option<PauseKind>,
}
