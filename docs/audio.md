# Native audio and feature extraction

`tongues-audio` is the model-neutral CPU implementation for waveform loading,
channel conversion, resampling, signal conditioning, STFT/ISTFT, and
spectrogram features. Model crates may translate an imported configuration into
this API, but external framework names do not enter the shared crate.

## Conventions

- PCM is interleaved `f32`; integer WAV input is divided by `2^(bits - 1)`, so
  the most-negative integer maps to `-1.0`.
- Downmixing averages channels. Mono upmixing duplicates the mono sample.
- Linear resampling evaluates output frame `i` at
  `i * source_rate / target_rate` and rounds the output length to the nearest
  frame.
- STFT uses a periodic Hann window centered inside the FFT frame. Centered STFT
  pads by half the FFT size with explicit reflect or constant padding.
- Spectrogram tensors are frame-major. Configuration, shape, scale,
  normalization, padding, pre-emphasis, mel convention, and frequency limits
  serialize with each `Spectrogram`.
- Slaney mel filters use Slaney frequency conversion and area normalization.
  HTK frequency conversion and unnormalized filters are also explicit options.
- Logarithmic transforms use a recorded positive floor; Coqui compatibility
  uses `1e-8`.

`stft` retains complex phase and `istft` performs window-sum-square normalized
overlap-add. Pre-emphasis/de-emphasis, peak normalization, RMS normalization,
and silence trimming are available independently of feature extraction.

## Coqui compatibility boundary

`tongues-tts::AudioFeatureConfig` is the compatibility adapter for imported
Coqui checkpoint configuration. `native_spectrogram_config()` maps Coqui field
names and defaults into a `tongues_audio::SpectrogramConfig`, while
`extract_native_spectrogram()` exposes the same CPU path to inference and
native training code without Python.

The golden fixture at
`crates/tongues-audio/tests/fixtures/coqui-v0.22.0-mel.json` was generated with
the Coqui TTS v0.22.0 `AudioProcessor`/`numpy_transforms` algorithm using NumPy
1.26.4, SciPy 1.11.4, and librosa 0.10.1. It covers reflect-centered STFT,
pre-emphasis, Slaney mel projection, logarithmic scaling, and range
normalization. The native frame-major tensor is checked with an absolute
tolerance of `3e-5`.

## Shared consumers

- Interpretation and the emotion model built on it use the native FFT and
  Slaney mel bank for training and inference features.
- Common Phone uses the same WAV, channel, resampling, and peak-normalization
  primitives during dataset preparation and live feature input.
- StyleTTS2 reference conditioning uses the shared WAV, downmix, trimming, and
  resampling path.
- Native VITS/HiFi-GAN training can consume the checkpoint-derived adapter
  directly, preserving the full feature contract with cached tensors.

The focused verification command is:

```sh
cargo test -p tongues-audio
```

Boundary tests cover mono/stereo conversion, 44.1 kHz PCM input, resampling,
full-scale integer samples, short clips, silence, metadata serialization, and
STFT/ISTFT round trips.
