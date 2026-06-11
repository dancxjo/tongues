use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceProvenance {
    pub source: EvidenceSource,
    pub method: String,
    pub version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSource {
    Manual,
    Lexicon,
    Rule,
    AcousticModel,
    ForcedAlignment,
    G2p,
    Asr,
    TtsPlan,
    Memory,
    Inference,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PronunciationSource {
    Lexicon,
    MorphologicalComposition,
    LearnedSuffix,
    GraphemeToPhoneme,
    Unknown,
}
