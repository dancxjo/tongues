use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhoneTimedPlan {
    pub phones: Vec<MbrolaPhone>,
}

impl PhoneTimedPlan {
    pub fn new(phones: Vec<MbrolaPhone>) -> Self {
        Self { phones }
    }

    pub fn total_duration_ms(&self) -> u64 {
        self.phones
            .iter()
            .map(|phone| u64::from(phone.duration_ms))
            .sum()
    }

    pub fn validate(&self) -> Result<(), MbrolaPhoError> {
        for (index, phone) in self.phones.iter().enumerate() {
            validate_phone(phone, index + 1)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MbrolaPhone {
    pub symbol: String,
    pub duration_ms: u32,
    #[serde(default)]
    pub pitch_targets: Vec<MbrolaPitchTarget>,
}

impl MbrolaPhone {
    pub fn new(symbol: impl Into<String>, duration_ms: u32) -> Self {
        Self {
            symbol: symbol.into(),
            duration_ms,
            pitch_targets: Vec::new(),
        }
    }

    pub fn with_pitch_targets(mut self, pitch_targets: Vec<MbrolaPitchTarget>) -> Self {
        self.pitch_targets = pitch_targets;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MbrolaPitchTarget {
    pub percent: u8,
    pub hz: f32,
}

pub fn serialize_pho(plan: &PhoneTimedPlan) -> Result<String, MbrolaPhoError> {
    plan.validate()?;
    let mut out = String::new();
    for phone in &plan.phones {
        let _ = write!(out, "{} {}", phone.symbol, phone.duration_ms);
        for target in &phone.pitch_targets {
            let hz = if target.hz.fract().abs() < 0.005 {
                format!("{:.0}", target.hz)
            } else {
                format!("{:.2}", target.hz)
            };
            let _ = write!(out, " {} {}", target.percent, hz);
        }
        out.push('\n');
    }
    Ok(out)
}

pub fn parse_pho(text: &str) -> Result<PhoneTimedPlan, MbrolaPhoError> {
    let mut phones = Vec::new();
    for (line_index, raw_line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let content = raw_line
            .split_once('#')
            .map_or(raw_line, |(content, _)| content)
            .trim();
        if content.is_empty() {
            continue;
        }
        let parts = content.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 2 {
            return Err(MbrolaPhoError::MissingDuration { line: line_number });
        }
        let duration_ms = parts[1]
            .parse::<u32>()
            .map_err(|_| MbrolaPhoError::BadDuration {
                line: line_number,
                value: parts[1].to_string(),
            })?;
        if parts[2..].len() % 2 != 0 {
            return Err(MbrolaPhoError::OddPitchTargetCount { line: line_number });
        }
        let mut pitch_targets = Vec::new();
        for pair in parts[2..].chunks_exact(2) {
            let percent = pair[0]
                .parse::<u8>()
                .map_err(|_| MbrolaPhoError::BadPitchPercent {
                    line: line_number,
                    value: pair[0].to_string(),
                })?;
            let hz = pair[1]
                .parse::<f32>()
                .map_err(|_| MbrolaPhoError::BadPitchHz {
                    line: line_number,
                    value: pair[1].to_string(),
                })?;
            pitch_targets.push(MbrolaPitchTarget { percent, hz });
        }
        let phone = MbrolaPhone {
            symbol: parts[0].to_string(),
            duration_ms,
            pitch_targets,
        };
        validate_phone(&phone, line_number)?;
        phones.push(phone);
    }
    Ok(PhoneTimedPlan { phones })
}

fn validate_phone(phone: &MbrolaPhone, line: usize) -> Result<(), MbrolaPhoError> {
    if phone.symbol.trim().is_empty() || phone.symbol.chars().any(char::is_whitespace) {
        return Err(MbrolaPhoError::BadSymbol {
            line,
            value: phone.symbol.clone(),
        });
    }
    if phone.duration_ms == 0 {
        return Err(MbrolaPhoError::ZeroDuration { line });
    }
    let mut previous = None;
    for target in &phone.pitch_targets {
        if target.percent > 100 {
            return Err(MbrolaPhoError::PitchPercentOutOfRange {
                line,
                value: target.percent,
            });
        }
        if !target.hz.is_finite() || target.hz <= 0.0 {
            return Err(MbrolaPhoError::InvalidPitchHz {
                line,
                value: target.hz,
            });
        }
        if previous.is_some_and(|value| target.percent < value) {
            return Err(MbrolaPhoError::UnorderedPitchTargets { line });
        }
        previous = Some(target.percent);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum MbrolaPhoError {
    #[error("line {line}: missing duration")]
    MissingDuration { line: usize },
    #[error("line {line}: invalid duration `{value}`")]
    BadDuration { line: usize, value: String },
    #[error("line {line}: duration must be greater than zero")]
    ZeroDuration { line: usize },
    #[error("line {line}: invalid phone symbol `{value}`")]
    BadSymbol { line: usize, value: String },
    #[error("line {line}: pitch targets must be percent/Hz pairs")]
    OddPitchTargetCount { line: usize },
    #[error("line {line}: invalid pitch target percent `{value}`")]
    BadPitchPercent { line: usize, value: String },
    #[error("line {line}: pitch percent {value} is outside 0..=100")]
    PitchPercentOutOfRange { line: usize, value: u8 },
    #[error("line {line}: invalid pitch target Hz `{value}`")]
    BadPitchHz { line: usize, value: String },
    #[error("line {line}: pitch Hz must be finite and positive, got {value}")]
    InvalidPitchHz { line: usize, value: f32 },
    #[error("line {line}: pitch target percentages must be nondecreasing")]
    UnorderedPitchTargets { line: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pho_round_trip_preserves_all_targets_comments_and_silence() {
        let source = "# diagnostic\nh 80\n_ 40 # phrase break\n@ 120 0 120 50 130 100 125\n";
        let plan = parse_pho(source).unwrap();
        assert_eq!(plan.phones.len(), 3);
        let serialized = serialize_pho(&plan).unwrap();
        assert_eq!(parse_pho(&serialized).unwrap(), plan);
    }

    #[test]
    fn pho_rejects_invalid_values_deliberately() {
        assert!(matches!(
            parse_pho("a 0"),
            Err(MbrolaPhoError::ZeroDuration { .. })
        ));
        assert!(matches!(
            parse_pho("a 10 101 120"),
            Err(MbrolaPhoError::PitchPercentOutOfRange { .. })
        ));
        assert!(matches!(
            parse_pho("a 10 50 NaN"),
            Err(MbrolaPhoError::InvalidPitchHz { .. })
        ));
        assert!(matches!(
            parse_pho("a 10 50"),
            Err(MbrolaPhoError::OddPitchTargetCount { .. })
        ));
    }
}
