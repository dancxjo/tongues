use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Spec<T> {
    Known(T),
    Unknown,
    Unspecified,
    NotApplicable,
    Variable(Vec<T>),
    Gradient { value: T, confidence: f32 },
}

impl<T> Default for Spec<T> {
    fn default() -> Self {
        Self::Unspecified
    }
}
