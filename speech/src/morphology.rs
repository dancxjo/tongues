use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::feature::FeatureBundle;
use crate::ids::MorphemeId;
use crate::spec::Spec;
use crate::time::TextSpan;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Morpheme {
    pub id: MorphemeId,
    pub form: String,
    pub gloss: Option<String>,
    pub features: FeatureBundle,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MorphemeToken {
    pub morpheme: Spec<MorphemeId>,
    pub surface: String,
    pub span: Option<TextSpan>,
    pub features: FeatureBundle,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Morphology {
    pub morphemes: HashMap<MorphemeId, Morpheme>,
    pub rules: Vec<MorphologicalRule>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MorphologicalRule {
    pub id: String,
    pub name: String,
    pub input_features: FeatureBundle,
    pub output_features: FeatureBundle,
}
