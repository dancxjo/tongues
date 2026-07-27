//! Host audio-device discovery and bounded CPAL capture.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::{Deserialize, Serialize};
use speaking::{AudioEncoding, AudioFormat, ChannelLayout, StreamSource};

use crate::{
    AudioBuffer, AudioSource, AudioSourceDescriptor, AudioSourceEvent, AudioSourceKind,
    BoundedAudioInput, BoundedAudioInputSender, PushedAudioChunk, Result, bounded_audio_input,
    invalid,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputDeviceInfo {
    /// Backend-provided selection key. Higher layers treat it as opaque.
    pub id: String,
    pub display_name: String,
    pub is_default: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioInputCapabilities {
    pub source_kinds: Vec<AudioSourceKind>,
    pub raw_pcm_encodings: Vec<crate::PcmEncoding>,
    pub devices: Vec<InputDeviceInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_discovery_error: Option<String>,
    pub bounded_queues: bool,
    pub explicit_discontinuities: bool,
    pub cleanup_stages: Vec<crate::CleanupCapability>,
}

pub fn input_device_inventory() -> Result<Vec<InputDeviceInfo>> {
    let host = cpal::default_host();
    let default_name = host.default_input_device().map(|device| device.to_string());
    let devices = host
        .input_devices()
        .map_err(|error| invalid(format!("failed to enumerate audio input devices: {error}")))?
        .map(|device| {
            let display_name = device.to_string();
            InputDeviceInfo {
                id: display_name.clone(),
                is_default: Some(display_name.as_str()) == default_name.as_deref(),
                display_name,
            }
        })
        .collect();
    Ok(devices)
}

pub fn input_capabilities() -> AudioInputCapabilities {
    let (devices, device_discovery_error) = match input_device_inventory() {
        Ok(devices) => (devices, None),
        Err(error) => (Vec::new(), Some(error.to_string())),
    };
    AudioInputCapabilities {
        source_kinds: vec![
            AudioSourceKind::File,
            AudioSourceKind::Stdin,
            AudioSourceKind::Microphone,
            AudioSourceKind::Tcp,
            AudioSourceKind::Unix,
            AudioSourceKind::Browser,
            AudioSourceKind::Server,
            AudioSourceKind::Fixture,
        ],
        raw_pcm_encodings: vec![
            crate::PcmEncoding::Signed16Le,
            crate::PcmEncoding::Float32Le,
        ],
        devices,
        device_discovery_error,
        bounded_queues: true,
        explicit_discontinuities: true,
        cleanup_stages: crate::CleanupPipeline::capabilities(),
    }
}

pub struct CpalAudioSource {
    inner: BoundedAudioInput,
    _stream: cpal::Stream,
}

impl CpalAudioSource {
    pub fn open(device_id: Option<&str>, queue_capacity: usize) -> Result<Self> {
        let host = cpal::default_host();
        let device = select_device(&host, device_id)?;
        let device_name = device.to_string();
        let supported = device
            .default_input_config()
            .map_err(|error| invalid(format!("failed to inspect `{device_name}`: {error}")))?;
        let sample_rate_hz = supported.sample_rate();
        let channels = supported.channels();
        let format = AudioFormat {
            encoding: AudioEncoding::PcmF32Le,
            sample_rate_hz,
            channels: channel_layout(channels),
        };
        let descriptor = AudioSourceDescriptor {
            id: format!("microphone:{device_name}"),
            kind: AudioSourceKind::Microphone,
            source: StreamSource::Live {
                device: Some(device_name.clone()),
            },
            decoded_format: format,
            live: true,
            seekable: false,
            metadata: BTreeMap::from([
                ("backend".into(), "cpal".into()),
                (
                    "native_sample_format".into(),
                    supported.sample_format().to_string(),
                ),
            ]),
        };
        let (sender, inner) = bounded_audio_input(descriptor, queue_capacity)?;
        let config = supported.config();
        let stream = match supported.sample_format() {
            cpal::SampleFormat::F32 => {
                build_f32_stream(&device, &config, channels, sender.clone())?
            }
            cpal::SampleFormat::I16 => {
                build_i16_stream(&device, &config, channels, sender.clone())?
            }
            cpal::SampleFormat::U16 => build_u16_stream(&device, &config, channels, sender)?,
            format => {
                return Err(invalid(format!(
                    "unsupported CPAL input sample format {format}"
                )));
            }
        };
        stream
            .play()
            .map_err(|error| invalid(format!("failed to start `{device_name}`: {error}")))?;
        Ok(Self {
            inner,
            _stream: stream,
        })
    }
}

impl AudioSource for CpalAudioSource {
    fn descriptor(&self) -> &AudioSourceDescriptor {
        self.inner.descriptor()
    }

    fn next_event(&mut self) -> Result<AudioSourceEvent> {
        self.inner.next_event()
    }

    fn cancel(&mut self) {
        self.inner.cancel();
    }
}

fn select_device(host: &cpal::Host, device_id: Option<&str>) -> Result<cpal::Device> {
    if let Some(device_id) = device_id {
        let mut devices = host
            .input_devices()
            .map_err(|error| invalid(format!("failed to enumerate input devices: {error}")))?;
        return devices
            .find(|device| device.to_string() == device_id)
            .ok_or_else(|| invalid(format!("no input device with id `{device_id}`")));
    }
    host.default_input_device()
        .ok_or_else(|| invalid("no default audio input device is available"))
}

fn build_f32_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: u16,
    sender: BoundedAudioInputSender,
) -> Result<cpal::Stream> {
    let sample_rate_hz = config.sample_rate;
    let sequence = Arc::new(AtomicU64::new(0));
    let start_frame = Arc::new(AtomicU64::new(0));
    let error_sender = sender.clone();
    device
        .build_input_stream(
            config.clone(),
            move |data: &[f32], _| {
                push_callback_chunk(
                    data.to_vec(),
                    sample_rate_hz,
                    channels,
                    &sender,
                    &sequence,
                    &start_frame,
                );
            },
            move |_| {
                let _ = error_sender.end();
            },
            None,
        )
        .map_err(|error| invalid(format!("failed to build f32 input stream: {error}")))
}

fn build_i16_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: u16,
    sender: BoundedAudioInputSender,
) -> Result<cpal::Stream> {
    let sample_rate_hz = config.sample_rate;
    let sequence = Arc::new(AtomicU64::new(0));
    let start_frame = Arc::new(AtomicU64::new(0));
    let error_sender = sender.clone();
    device
        .build_input_stream(
            config.clone(),
            move |data: &[i16], _| {
                push_callback_chunk(
                    data.iter()
                        .map(|sample| f32::from(*sample) / 32768.0)
                        .collect(),
                    sample_rate_hz,
                    channels,
                    &sender,
                    &sequence,
                    &start_frame,
                );
            },
            move |_| {
                let _ = error_sender.end();
            },
            None,
        )
        .map_err(|error| invalid(format!("failed to build i16 input stream: {error}")))
}

fn build_u16_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: u16,
    sender: BoundedAudioInputSender,
) -> Result<cpal::Stream> {
    let sample_rate_hz = config.sample_rate;
    let sequence = Arc::new(AtomicU64::new(0));
    let start_frame = Arc::new(AtomicU64::new(0));
    let error_sender = sender.clone();
    device
        .build_input_stream(
            config.clone(),
            move |data: &[u16], _| {
                push_callback_chunk(
                    data.iter()
                        .map(|sample| f32::from(*sample) / 32767.5 - 1.0)
                        .collect(),
                    sample_rate_hz,
                    channels,
                    &sender,
                    &sequence,
                    &start_frame,
                );
            },
            move |_| {
                let _ = error_sender.end();
            },
            None,
        )
        .map_err(|error| invalid(format!("failed to build u16 input stream: {error}")))
}

fn push_callback_chunk(
    samples: Vec<f32>,
    sample_rate_hz: u32,
    channels: u16,
    sender: &BoundedAudioInputSender,
    sequence: &AtomicU64,
    start_frame: &AtomicU64,
) {
    let sequence = sequence.fetch_add(1, Ordering::Relaxed);
    let frames = samples.len() / usize::from(channels.max(1));
    let start_frame = start_frame.fetch_add(frames as u64, Ordering::Relaxed);
    // A full queue drops this callback chunk. The incremented sequence makes
    // the loss explicit as a discontinuity when capture catches up.
    let _ = sender.try_send(PushedAudioChunk {
        sequence,
        start_frame: Some(start_frame),
        audio: AudioBuffer {
            samples,
            sample_rate_hz,
            channels,
        },
    });
}

fn channel_layout(channels: u16) -> ChannelLayout {
    match channels {
        1 => ChannelLayout::Mono,
        2 => ChannelLayout::Stereo,
        count => ChannelLayout::Interleaved {
            labels: (0..count).map(|index| format!("channel_{index}")).collect(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_queue_overflow_is_visible_as_a_later_sequence_gap() {
        let descriptor = AudioSourceDescriptor {
            id: "capture".into(),
            kind: AudioSourceKind::Microphone,
            source: StreamSource::Live {
                device: Some("fixture".into()),
            },
            decoded_format: AudioFormat {
                encoding: AudioEncoding::PcmF32Le,
                sample_rate_hz: 16_000,
                channels: ChannelLayout::Mono,
            },
            live: true,
            seekable: false,
            metadata: BTreeMap::new(),
        };
        let (sender, mut source) = bounded_audio_input(descriptor, 1).unwrap();
        let sequence = AtomicU64::new(0);
        let start_frame = AtomicU64::new(0);

        push_callback_chunk(vec![0.0; 160], 16_000, 1, &sender, &sequence, &start_frame);
        push_callback_chunk(vec![0.0; 160], 16_000, 1, &sender, &sequence, &start_frame);
        assert!(matches!(
            source.next_event().unwrap(),
            AudioSourceEvent::Audio(_)
        ));
        push_callback_chunk(vec![0.0; 160], 16_000, 1, &sender, &sequence, &start_frame);
        assert!(matches!(
            source.next_event().unwrap(),
            AudioSourceEvent::Discontinuity(crate::AudioDiscontinuity {
                expected_chunk_sequence: 1,
                received_chunk_sequence: 2,
                ..
            })
        ));
    }
}
