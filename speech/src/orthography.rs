use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::feature::FeatureBundle;
use crate::ids::GraphemeId;
use crate::rules::PhonemePattern;
use crate::segment::Environment;
use crate::spec::Spec;
use crate::time::TextSpan;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Grapheme {
    pub id: GraphemeId,
    pub text: String,
    pub features: FeatureBundle,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphemeToken {
    pub grapheme: Spec<GraphemeId>,
    pub text: String,
    pub span: Option<TextSpan>,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Orthography {
    pub name: String,
    pub graphemes: HashMap<GraphemeId, Grapheme>,
    pub g2p_rules: Vec<GraphemeToPhonemeRule>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphemeToPhonemeRule {
    pub id: String,
    pub input: String,
    pub environment: Option<Environment>,
    pub output: Vec<PhonemePattern>,
    pub confidence: f32,
}
