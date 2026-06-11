use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ids::FeatureId;
use crate::spec::Spec;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureDomain {
    Phonological,
    Articulatory,
    Acoustic,
    Prosodic,
    Orthographic,
    Morphological,
    Semantic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureValueType {
    Boolean,
    Category { allowed: Vec<String> },
    Number { unit: Option<String> },
    Vector { dimensions: usize },
    Text,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureDef {
    pub id: FeatureId,
    pub name: String,
    pub domain: FeatureDomain,
    pub value_type: FeatureValueType,
    pub contrastive: bool,
    pub conditioned: bool,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureValue {
    Bool(bool),
    Category(String),
    Number(f64),
    Vector(Vec<f32>),
    Text(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct FeatureBundle {
    pub values: HashMap<FeatureId, Spec<FeatureValue>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct FeatureSystem {
    pub features: HashMap<FeatureId, FeatureDef>,
}
