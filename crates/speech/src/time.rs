use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeSpan {
    pub start_s: f64,
    pub end_s: f64,
}

impl TimeSpan {
    pub fn duration_s(&self) -> f64 {
        (self.end_s - self.start_s).max(0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextSpan {
    pub start_char: usize,
    pub end_char: usize,
}

impl TextSpan {
    pub fn len_chars(&self) -> usize {
        self.end_char.saturating_sub(self.start_char)
    }
}
