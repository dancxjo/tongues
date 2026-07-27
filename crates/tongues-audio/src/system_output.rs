//! Host audio-device discovery and explicit CPAL/WAV output sinks.

use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::{Deserialize, Serialize};

use crate::{invalid, AudioBuffer, Result};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputDeviceInfo {
    /// Backend-provided selection key. Higher layers treat it as opaque.
    pub id: String,
    pub display_name: String,
    pub is_default: bool,
}

pub fn output_device_inventory() -> Result<Vec<OutputDeviceInfo>> {
    let host = cpal::default_host();
    let default_name = host
        .default_output_device()
        .map(|device| device.to_string());
    let devices = host
        .output_devices()
        .map_err(|error| invalid(format!("failed to enumerate audio output devices: {error}")))?
        .map(|device| {
            let display_name = device.to_string();
            OutputDeviceInfo {
                id: display_name.clone(),
                is_default: Some(display_name.as_str()) == default_name.as_deref(),
                display_name,
            }
        })
        .collect();
    Ok(devices)
}

/// A blocking, bounded-lifetime CPAL sink for one normalized PCM buffer.
///
/// Opening and playback are separate so callers can report device selection
/// failures before claiming that any samples were submitted.
pub struct CpalAudioSink {
    device: cpal::Device,
    device_name: String,
    config: cpal::SupportedStreamConfig,
}

impl CpalAudioSink {
    pub fn open(device_id: Option<&str>) -> Result<Self> {
        let host = cpal::default_host();
        let device = select_output_device(&host, device_id)?;
        let device_name = device.to_string();
        let config = device
            .default_output_config()
            .map_err(|error| invalid(format!("failed to inspect `{device_name}`: {error}")))?;
        Ok(Self {
            device,
            device_name,
            config,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn play_blocking(&self, audio: &AudioBuffer) -> Result<()> {
        audio.validate()?;
        let channels = self.config.channels();
        let output_rate = self.config.sample_rate();
        let state = Arc::new((
            Mutex::new(PlaybackState {
                audio: audio.clone(),
                input_position: 0.0,
                output_rate,
                output_channels: channels,
                finished: false,
                stream_error: None,
            }),
            Condvar::new(),
        ));
        let config = self.config.config();
        let stream = match self.config.sample_format() {
            cpal::SampleFormat::F32 => self.device.build_output_stream(
                config.clone(),
                {
                    let callback_state = Arc::clone(&state);
                    move |output: &mut [f32], _| {
                        fill_output(output, &callback_state, |sample| sample)
                    }
                },
                output_error_callback(Arc::clone(&state)),
                None,
            ),
            cpal::SampleFormat::I16 => self.device.build_output_stream(
                config.clone(),
                {
                    let callback_state = Arc::clone(&state);
                    move |output: &mut [i16], _| {
                        fill_output(output, &callback_state, |sample| {
                            (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16
                        })
                    }
                },
                output_error_callback(Arc::clone(&state)),
                None,
            ),
            cpal::SampleFormat::U16 => self.device.build_output_stream(
                config,
                {
                    let callback_state = Arc::clone(&state);
                    move |output: &mut [u16], _| {
                        fill_output(output, &callback_state, |sample| {
                            ((sample.clamp(-1.0, 1.0) + 1.0) * 0.5 * u16::MAX as f32).round() as u16
                        })
                    }
                },
                output_error_callback(Arc::clone(&state)),
                None,
            ),
            format => {
                return Err(invalid(format!(
                    "unsupported CPAL output sample format {format}"
                )));
            }
        }
        .map_err(|error| {
            invalid(format!(
                "failed to build `{}` output: {error}",
                self.device_name
            ))
        })?;
        stream
            .play()
            .map_err(|error| invalid(format!("failed to start `{}`: {error}", self.device_name)))?;

        let (lock, wake) = &*state;
        let mut state = lock
            .lock()
            .map_err(|_| invalid("audio output state was poisoned"))?;
        while !state.finished {
            state = wake
                .wait(state)
                .map_err(|_| invalid("audio output state was poisoned"))?;
        }
        if let Some(error) = state.stream_error.take() {
            return Err(invalid(format!(
                "audio output stream `{}` failed: {error}",
                self.device_name
            )));
        }
        Ok(())
    }
}

fn output_error_callback(
    shared: Arc<(Mutex<PlaybackState>, Condvar)>,
) -> impl FnMut(cpal::Error) + Send + 'static {
    move |error| {
        let (lock, wake) = &*shared;
        if let Ok(mut state) = lock.lock() {
            state.stream_error = Some(error.to_string());
            state.finished = true;
            wake.notify_all();
        }
    }
}

pub fn write_wav_output(path: &Path, audio: &AudioBuffer) -> Result<()> {
    audio.validate()?;
    let mut part_name = path.as_os_str().to_os_string();
    part_name.push(".part");
    let part_path = std::path::PathBuf::from(part_name);
    let spec = hound::WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate_hz,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&part_path, spec)?;
    for sample in &audio.samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)?;
    }
    writer.finalize()?;
    std::fs::rename(part_path, path)?;
    Ok(())
}

fn select_output_device(host: &cpal::Host, device_id: Option<&str>) -> Result<cpal::Device> {
    if let Some(device_id) = device_id.filter(|id| *id != "default") {
        let mut devices = host
            .output_devices()
            .map_err(|error| invalid(format!("failed to enumerate output devices: {error}")))?;
        return devices
            .find(|device| device.to_string() == device_id)
            .ok_or_else(|| invalid(format!("no output device with id `{device_id}`")));
    }
    host.default_output_device()
        .ok_or_else(|| invalid("no default audio output device is available"))
}

struct PlaybackState {
    audio: AudioBuffer,
    input_position: f64,
    output_rate: u32,
    output_channels: u16,
    finished: bool,
    stream_error: Option<String>,
}

fn fill_output<T: Copy>(
    output: &mut [T],
    shared: &Arc<(Mutex<PlaybackState>, Condvar)>,
    convert: impl Fn(f32) -> T,
) {
    let (lock, wake) = &**shared;
    let Ok(mut state) = lock.lock() else {
        return;
    };
    let output_channels = usize::from(state.output_channels);
    for frame in output.chunks_mut(output_channels) {
        if state.finished {
            frame.fill(convert(0.0));
            continue;
        }
        let input_frame = state.input_position.floor() as usize;
        if input_frame >= state.audio.frames() {
            state.finished = true;
            frame.fill(convert(0.0));
            wake.notify_all();
            continue;
        }
        let input_channels = usize::from(state.audio.channels);
        for (channel, destination) in frame.iter_mut().enumerate() {
            let source_channel = channel.min(input_channels.saturating_sub(1));
            *destination =
                convert(state.audio.samples[input_frame * input_channels + source_channel]);
        }
        state.input_position +=
            f64::from(state.audio.sample_rate_hz) / f64::from(state.output_rate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_output_preserves_audio_geometry() {
        let path = std::env::temp_dir().join(format!(
            "tongues-audio-output-{}-{}.wav",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let audio = AudioBuffer {
            samples: vec![0.0, 0.5, -0.5, 0.0],
            sample_rate_hz: 16_000,
            channels: 2,
        };
        write_wav_output(&path, &audio).unwrap();
        let reader = hound::WavReader::open(&path).unwrap();
        assert_eq!(reader.spec().channels, 2);
        assert_eq!(reader.spec().sample_rate, 16_000);
        assert_eq!(reader.duration(), 2);
        std::fs::remove_file(path).unwrap();
    }
}
