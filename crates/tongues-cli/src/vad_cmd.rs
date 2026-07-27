use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{ArgGroup, Args, ValueEnum};
use serde_json::{json, Value};
use tongues_audio::{
    AudioSource, CpalAudioSource, EnergyVad, EnergyVadConfig, NormalizedAudioSource,
    SegmentationConfig, SegmentationEvent, UtteranceSegmenter, VadBackendKind, VadPipelineEvent,
    VadSegmentationPipeline, VoiceActivityDetector, WavAudioSource, WebRtcVad, WebRtcVadConfig,
};

const CPAL_QUEUE_CAPACITY_CHUNKS: usize = 32;
const VAD_SAMPLE_RATE_HZ: u32 = 16_000;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum VadBackendArg {
    WebRtc,
    Energy,
}

impl From<VadBackendArg> for VadBackendKind {
    fn from(value: VadBackendArg) -> Self {
        match value {
            VadBackendArg::WebRtc => Self::WebRtc,
            VadBackendArg::Energy => Self::Energy,
        }
    }
}

/// Detect speech and segment utterances in a WAV file or live microphone.
#[derive(Debug, Args)]
#[command(group(
    ArgGroup::new("source")
        .required(true)
        .args(["input", "microphone"])
))]
pub struct VadCommand {
    /// Input WAV file.
    pub input: Option<PathBuf>,

    /// Capture live microphone audio through CPAL.
    #[arg(long)]
    pub microphone: bool,

    /// Exact CPAL input device id from `common-phone listen-devices`.
    #[arg(long, requires = "microphone")]
    pub input_device: Option<String>,

    /// Voice activity detector implementation.
    #[arg(long, value_enum, default_value = "web-rtc")]
    pub backend: VadBackendArg,

    /// Consecutive speech required to open an utterance.
    #[arg(long, default_value_t = 30)]
    pub speech_start_ms: u64,

    /// Silence required to emit the low-latency acoustic endpoint.
    #[arg(long, default_value_t = 300)]
    pub acoustic_end_ms: u64,

    /// Silence required to close the conversational segment.
    #[arg(long, default_value_t = 800)]
    pub segment_end_ms: u64,

    /// Audio retained before the detected speech start.
    #[arg(long, default_value_t = 200)]
    pub pre_roll_ms: u64,

    /// Shorter speech regions are reported as dropped.
    #[arg(long, default_value_t = 250)]
    pub minimum_speech_ms: u64,

    /// Force-close a continuously active segment at this duration.
    #[arg(long, default_value_t = 30_000)]
    pub maximum_segment_ms: u64,

    /// RMS threshold used by the energy baseline.
    #[arg(long, default_value_t = 0.02)]
    pub energy_threshold_rms: f32,

    /// Minimum RMS accepted as speech by WebRTC VAD.
    #[arg(long, default_value_t = 0.025)]
    pub minimum_speech_rms: f32,

    /// Adaptive WebRTC noise-floor multiplier.
    #[arg(long, default_value_t = 1.8)]
    pub noise_gate_multiplier: f32,

    /// Emit one JSON object per lifecycle event.
    #[arg(long)]
    pub jsonl: bool,

    /// Include every 10 ms VAD decision. By default only boundaries and metrics are printed.
    #[arg(long)]
    pub show_frames: bool,
}

pub fn run(command: VadCommand) -> Result<()> {
    let config = SegmentationConfig {
        frame_ms: 10,
        speech_start_ms: command.speech_start_ms,
        acoustic_end_silence_ms: command.acoustic_end_ms,
        segment_end_silence_ms: command.segment_end_ms,
        minimum_speech_ms: command.minimum_speech_ms,
        pre_roll_ms: command.pre_roll_ms,
        maximum_segment_ms: command.maximum_segment_ms,
    };
    config.validate()?;

    if let Some(input) = &command.input {
        // A whole-file chunk avoids introducing artificial boundaries while
        // normalizing arbitrary WAV rates to WebRTC's canonical mono input.
        let source = WavAudioSource::open(input, usize::MAX)?;
        return run_source(source, input.display().to_string(), &command, config);
    }
    if command.microphone {
        let source =
            CpalAudioSource::open(command.input_device.as_deref(), CPAL_QUEUE_CAPACITY_CHUNKS)?;
        let source_id = source.descriptor().id.clone();
        eprintln!(
            "listening on {} with {:?} VAD (Ctrl-C to stop)",
            source_id, command.backend
        );
        return run_source(source, source_id, &command, config);
    }
    bail!("either a WAV input or --microphone is required")
}

fn run_source<S: AudioSource>(
    source: S,
    source_id: String,
    command: &VadCommand,
    config: SegmentationConfig,
) -> Result<()> {
    let source = NormalizedAudioSource::new(source, VAD_SAMPLE_RATE_HZ, 1)?;
    let detector: Box<dyn VoiceActivityDetector> = match command.backend {
        VadBackendArg::Energy => Box::new(EnergyVad::new(EnergyVadConfig {
            threshold_rms: command.energy_threshold_rms,
        })?),
        VadBackendArg::WebRtc => {
            let mut vad = WebRtcVadConfig::default();
            vad.minimum_speech_rms = command.minimum_speech_rms;
            vad.noise_gate_multiplier = command.noise_gate_multiplier;
            Box::new(WebRtcVad::new(vad)?)
        }
    };
    let segmenter = UtteranceSegmenter::new(source_id, config)?;
    let mut pipeline = VadSegmentationPipeline::new(source, detector, segmenter)?;

    while let Some(event) = pipeline.next_event()? {
        match event {
            VadPipelineEvent::VadDecision { frame, decision } if command.show_frames => emit(
                command.jsonl,
                json!({
                    "type": "vad_decision",
                    "frame_sequence": frame.sequence,
                    "start_frame": frame.start_frame,
                    "rms": decision.rms,
                    "speech_probability": decision.speech_probability,
                    "is_speech": decision.is_speech,
                    "backend": decision.backend,
                }),
            ),
            VadPipelineEvent::VadDecision { .. } => {}
            VadPipelineEvent::Segmentation(event) => {
                if let Some(value) = summarize_segment_event(event) {
                    emit(command.jsonl, value);
                }
            }
            VadPipelineEvent::SourceDiscontinuity(gap) => emit(
                command.jsonl,
                json!({
                    "type": "discontinuity",
                    "expected_chunk_sequence": gap.expected_chunk_sequence,
                    "received_chunk_sequence": gap.received_chunk_sequence,
                    "reason": gap.reason,
                }),
            ),
            VadPipelineEvent::EndOfStream { metrics } => emit(
                command.jsonl,
                json!({
                    "type": "metrics",
                    "speech_ratio": metrics.speech_ratio(),
                    "mean_endpoint_latency_ms": metrics.mean_endpoint_latency_ms(),
                    "metrics": metrics,
                }),
            ),
        }
    }
    Ok(())
}

fn summarize_segment_event(event: SegmentationEvent) -> Option<Value> {
    match event {
        SegmentationEvent::SegmentOpened {
            segment_id,
            pre_roll_frames,
            ..
        } => Some(json!({
            "type": "segment_opened",
            "segment_id": segment_id,
            "pre_roll_frames": pre_roll_frames,
        })),
        SegmentationEvent::SpeechEnded {
            segment_id,
            endpoint_latency_ms,
            ..
        } => Some(json!({
            "type": "speech_ended",
            "segment_id": segment_id,
            "endpoint_latency_ms": endpoint_latency_ms,
        })),
        SegmentationEvent::SegmentFinalized(segment) => Some(json!({
            "type": "segment_final",
            "segment_id": segment.id,
            "accepted": true,
            "reason": segment.close_reason,
            "speech_duration_ms": segment.speech_duration_ms,
            "total_duration_ms": segment.total_duration_ms,
            "pre_roll_frames": segment.pre_roll_frames,
            "post_roll_frames": segment.post_roll_frames,
        })),
        SegmentationEvent::SegmentDropped(segment) => Some(json!({
            "type": "segment_final",
            "segment_id": segment.id,
            "accepted": false,
            "reason": segment.close_reason,
            "speech_duration_ms": segment.speech_duration_ms,
            "total_duration_ms": segment.total_duration_ms,
            "pre_roll_frames": segment.pre_roll_frames,
            "post_roll_frames": segment.post_roll_frames,
        })),
        SegmentationEvent::SpeechStarted { .. }
        | SegmentationEvent::SpeechResumed { .. }
        | SegmentationEvent::SegmentUpdated { .. } => None,
    }
}

fn emit(jsonl: bool, value: Value) {
    if jsonl {
        println!("{value}");
        return;
    }
    match value.get("type").and_then(Value::as_str) {
        Some("segment_opened") => println!(
            "opened {} ({} pre-roll frames)",
            value["segment_id"], value["pre_roll_frames"]
        ),
        Some("speech_ended") => println!(
            "speech ended {} ({} ms endpoint latency)",
            value["segment_id"], value["endpoint_latency_ms"]
        ),
        Some("segment_final") => println!(
            "final {}: accepted={} speech={} ms total={} ms reason={}",
            value["segment_id"],
            value["accepted"],
            value["speech_duration_ms"],
            value["total_duration_ms"],
            value["reason"]
        ),
        Some("metrics") => println!(
            "metrics: speech_ratio={:.3} mean_endpoint_latency_ms={:.1} dropped_chunks={} forced_flushes={}",
            value["speech_ratio"].as_f64().unwrap_or_default(),
            value["mean_endpoint_latency_ms"].as_f64().unwrap_or_default(),
            value["metrics"]["dropped_source_chunks"],
            value["metrics"]["forced_flushes"],
        ),
        _ => println!("{value}"),
    }
}
