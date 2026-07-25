use std::fmt;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

/// One measured native synthesis boundary.
///
/// Timings are emitted from the library so CLI and resident runtimes observe
/// the same model stages. GPU implementations synchronize only when profiling
/// is explicitly enabled, keeping the ordinary inference path asynchronous.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynthesisProfileEvent {
    pub stage: SynthesisStage,
    pub elapsed_ms: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dimensions: Vec<SynthesisDimension>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelLoadProfileEvent {
    pub stage: ModelLoadStage,
    pub elapsed_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component: Option<String>,
}

impl ModelLoadProfileEvent {
    pub fn new(stage: ModelLoadStage, elapsed: Duration, component: Option<&str>) -> Self {
        Self {
            stage,
            elapsed_ms: elapsed.as_secs_f64() * 1_000.0,
            component: component.map(str::to_owned),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLoadStage {
    ConfigCheckpointParsing,
    ModelConstruction,
    WeightUpload,
}

impl SynthesisProfileEvent {
    pub fn new(
        stage: SynthesisStage,
        elapsed: Duration,
        dimensions: impl IntoIterator<Item = SynthesisDimension>,
    ) -> Self {
        Self {
            stage,
            elapsed_ms: elapsed.as_secs_f64() * 1_000.0,
            dimensions: dimensions.into_iter().collect(),
            detail: None,
        }
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SynthesisDimension {
    pub name: String,
    pub value: usize,
}

impl SynthesisDimension {
    pub fn new(name: impl Into<String>, value: usize) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

/// Stable stage names used by CLI JSON output and benchmark tooling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynthesisStage {
    CheckpointProjection,
    HostToDevice,
    TextEncoder,
    DurationPrediction,
    DurationExpansion,
    AcousticDecoder,
    VitsFlow,
    WaveformDecoder,
    DeviceToHost,
    AudioSink,
}

impl SynthesisStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CheckpointProjection => "checkpoint_projection",
            Self::HostToDevice => "host_to_device",
            Self::TextEncoder => "text_encoder",
            Self::DurationPrediction => "duration_prediction",
            Self::DurationExpansion => "duration_expansion",
            Self::AcousticDecoder => "acoustic_decoder",
            Self::VitsFlow => "vits_flow",
            Self::WaveformDecoder => "waveform_decoder",
            Self::DeviceToHost => "device_to_host",
            Self::AudioSink => "audio_sink",
        }
    }
}

impl fmt::Display for SynthesisStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

pub trait SynthesisProfiler {
    fn record(&mut self, event: SynthesisProfileEvent);
}

pub(crate) fn reborrow_profiler<'a>(
    profiler: &'a mut Option<&mut dyn SynthesisProfiler>,
) -> Option<&'a mut dyn SynthesisProfiler> {
    match profiler {
        Some(profiler) => Some(&mut **profiler),
        None => None,
    }
}

impl<F> SynthesisProfiler for F
where
    F: FnMut(SynthesisProfileEvent),
{
    fn record(&mut self, event: SynthesisProfileEvent) {
        self(event);
    }
}

pub(crate) fn finish_backend_stage<B: Backend>(
    profiler: &mut Option<&mut dyn SynthesisProfiler>,
    device: &B::Device,
    stage: SynthesisStage,
    started: Instant,
    dimensions: impl IntoIterator<Item = SynthesisDimension>,
) -> Result<()> {
    let Some(profiler) = profiler.as_deref_mut() else {
        return Ok(());
    };
    B::sync(device)
        .map_err(anyhow::Error::new)
        .with_context(|| format!("failed to synchronize after {stage}"))?;
    profiler.record(SynthesisProfileEvent::new(
        stage,
        started.elapsed(),
        dimensions,
    ));
    Ok(())
}

pub(crate) fn finish_host_stage(
    profiler: &mut Option<&mut dyn SynthesisProfiler>,
    stage: SynthesisStage,
    started: Instant,
    dimensions: impl IntoIterator<Item = SynthesisDimension>,
) {
    if let Some(profiler) = profiler.as_deref_mut() {
        profiler.record(SynthesisProfileEvent::new(
            stage,
            started.elapsed(),
            dimensions,
        ));
    }
}

pub(crate) fn record_load_stage(
    profiler: &mut Option<&mut dyn FnMut(ModelLoadProfileEvent)>,
    stage: ModelLoadStage,
    started: Instant,
    component: Option<&str>,
) {
    if let Some(profiler) = profiler.as_deref_mut() {
        profiler(ModelLoadProfileEvent::new(
            stage,
            started.elapsed(),
            component,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_events_have_stable_structured_json() {
        let event = SynthesisProfileEvent::new(
            SynthesisStage::WaveformDecoder,
            Duration::from_millis(12),
            [
                SynthesisDimension::new("latent_frames", 20),
                SynthesisDimension::new("samples", 5_120),
            ],
        );

        let value = serde_json::to_value(event).expect("serialize profile event");

        assert_eq!(value["stage"], "waveform_decoder");
        assert_eq!(value["elapsed_ms"], 12.0);
        assert_eq!(value["dimensions"][0]["name"], "latent_frames");
        assert_eq!(value["dimensions"][1]["value"], 5_120);
    }
}
