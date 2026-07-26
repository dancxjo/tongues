use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

pub const MAX_CUDA_DEVICE_INDEX: usize = u16::MAX as usize;

/// Backend-neutral policy requested by a speech caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpeechDeviceRequest {
    /// Prefer CUDA device 0, but fall back to CPU when it cannot be initialized.
    Auto,
    Cpu,
    Cuda {
        index: usize,
    },
}

impl FromStr for SpeechDeviceRequest {
    type Err = SpeechDeviceSpecError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if value.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        if value.eq_ignore_ascii_case("cuda") {
            return Ok(Self::Cuda { index: 0 });
        }
        if let Some((kind, index)) = value.split_once(':') {
            if kind.eq_ignore_ascii_case("cuda") {
                let index = index
                    .parse::<usize>()
                    .map_err(|_| SpeechDeviceSpecError(value.to_string()))?;
                return Ok(Self::Cuda { index });
            }
        }
        Err(SpeechDeviceSpecError(value.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeechDeviceSpecError(String);

impl fmt::Display for SpeechDeviceSpecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid speech device `{}`; expected `auto`, `cpu`, `cuda`, or `cuda:<index>`",
            self.0
        )
    }
}

impl std::error::Error for SpeechDeviceSpecError {}

/// The concrete execution device chosen before model loading starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResolvedSpeechDevice {
    Cpu,
    Cuda { index: usize },
}

impl ResolvedSpeechDevice {
    pub fn kind(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda { .. } => "cuda",
        }
    }

    pub fn index(self) -> Option<usize> {
        match self {
            Self::Cpu => None,
            Self::Cuda { index } => Some(index),
        }
    }

    pub fn display_name(self) -> String {
        match self {
            Self::Cpu => "CPU".to_string(),
            Self::Cuda { index } => format!("CUDA GPU {index}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeechDeviceSelection {
    pub requested: SpeechDeviceRequest,
    pub resolved: ResolvedSpeechDevice,
    pub fallback_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeechDeviceSelectionError {
    pub index: usize,
    pub reason: String,
}

impl fmt::Display for SpeechDeviceSelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "requested CUDA device index {} is unavailable: {}",
            self.index, self.reason
        )
    }
}

impl std::error::Error for SpeechDeviceSelectionError {}

/// Resolve a speech device without coupling policy to a specific tensor backend.
///
/// `probe_cuda` must initialize and execute a minimal operation on the requested
/// CUDA index. Explicit CUDA failures are returned; automatic selection falls
/// back to CPU and retains the reason for diagnostics.
pub fn resolve_speech_device(
    requested: SpeechDeviceRequest,
    mut probe_cuda: impl FnMut(usize) -> Result<(), String>,
) -> Result<SpeechDeviceSelection, SpeechDeviceSelectionError> {
    match requested {
        SpeechDeviceRequest::Cpu => Ok(SpeechDeviceSelection {
            requested,
            resolved: ResolvedSpeechDevice::Cpu,
            fallback_reason: None,
        }),
        SpeechDeviceRequest::Auto => match probe_cuda(0) {
            Ok(()) => Ok(SpeechDeviceSelection {
                requested,
                resolved: ResolvedSpeechDevice::Cuda { index: 0 },
                fallback_reason: None,
            }),
            Err(reason) => Ok(SpeechDeviceSelection {
                requested,
                resolved: ResolvedSpeechDevice::Cpu,
                fallback_reason: Some(reason),
            }),
        },
        SpeechDeviceRequest::Cuda { index } => {
            if index > MAX_CUDA_DEVICE_INDEX {
                return Err(SpeechDeviceSelectionError {
                    index,
                    reason: format!(
                        "index exceeds the supported maximum of {MAX_CUDA_DEVICE_INDEX}"
                    ),
                });
            }
            probe_cuda(index).map_err(|reason| SpeechDeviceSelectionError { index, reason })?;
            Ok(SpeechDeviceSelection {
                requested,
                resolved: ResolvedSpeechDevice::Cuda { index },
                fallback_reason: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_cuda_preserves_and_probes_the_requested_index() {
        let mut probed = Vec::new();
        let selection = resolve_speech_device(SpeechDeviceRequest::Cuda { index: 3 }, |index| {
            probed.push(index);
            Ok(())
        })
        .expect("explicit CUDA selection");

        assert_eq!(probed, vec![3]);
        assert_eq!(selection.resolved, ResolvedSpeechDevice::Cuda { index: 3 });
        assert_eq!(selection.resolved.index(), Some(3));
    }

    #[test]
    fn explicit_unavailable_cuda_is_an_error_without_fallback() {
        let error = resolve_speech_device(SpeechDeviceRequest::Cuda { index: 7 }, |_| {
            Err("only 2 CUDA devices were detected".into())
        })
        .expect_err("explicit unavailable CUDA must fail");

        assert_eq!(error.index, 7);
        assert_eq!(
            error.to_string(),
            "requested CUDA device index 7 is unavailable: only 2 CUDA devices were detected"
        );
    }

    #[test]
    fn explicit_cuda_rejects_indices_that_the_backend_cannot_represent() {
        let index = MAX_CUDA_DEVICE_INDEX + 1;
        let error = resolve_speech_device(SpeechDeviceRequest::Cuda { index }, |_| {
            panic!("an invalid index must fail before probing")
        })
        .expect_err("out-of-range CUDA index must fail");

        assert_eq!(error.index, index);
        assert_eq!(
            error.reason,
            format!("index exceeds the supported maximum of {MAX_CUDA_DEVICE_INDEX}")
        );
    }

    #[test]
    fn automatic_policy_probes_cuda_zero_then_falls_back_to_cpu() {
        let selection = resolve_speech_device(SpeechDeviceRequest::Auto, |index| {
            assert_eq!(index, 0);
            Err("driver unavailable".into())
        })
        .expect("automatic selection");

        assert_eq!(selection.resolved, ResolvedSpeechDevice::Cpu);
        assert_eq!(
            selection.fallback_reason.as_deref(),
            Some("driver unavailable")
        );
    }

    #[test]
    fn cpu_selection_does_not_probe_cuda() {
        let selection = resolve_speech_device(SpeechDeviceRequest::Cpu, |_| {
            panic!("CPU selection must not probe CUDA")
        })
        .expect("CPU selection");

        assert_eq!(selection.resolved, ResolvedSpeechDevice::Cpu);
    }

    #[test]
    fn device_specs_parse_explicit_indices_and_reject_invalid_values() {
        assert_eq!(
            "cuda:12".parse::<SpeechDeviceRequest>().unwrap(),
            SpeechDeviceRequest::Cuda { index: 12 }
        );
        assert_eq!(
            "cuda".parse::<SpeechDeviceRequest>().unwrap(),
            SpeechDeviceRequest::Cuda { index: 0 }
        );
        assert!("cuda:gpu0".parse::<SpeechDeviceRequest>().is_err());
        assert!("metal".parse::<SpeechDeviceRequest>().is_err());
    }
}
