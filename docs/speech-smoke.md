# Speech Synthesis Smoke Measurements

The repository includes `scripts/speech-smoke.sh` for real, release-mode
waveform synthesis. It runs the native Burn component model, VITS with two
speakers, the ONNX compatibility voice, and StyleTTS2. Every successful case is
probed with `ffprobe` and `ffmpeg`; the JSON result records wall time, peak RSS,
audio format, duration, file size, mean and peak level, SHA-256, and real-time
factor (wall time divided by audio duration).

Run the standard sentence:

```sh
scripts/speech-smoke.sh
```

Override the sentence or force CPU execution:

```sh
SPEECH_SMOKE_TEXT="A sentence to synthesize." scripts/speech-smoke.sh
SPEECH_SMOKE_CPU=1 scripts/speech-smoke.sh
```

Outputs default to `target/speech-smoke/<UTC timestamp>/`. Model download time
is included when a registered bundle is not already installed.

## 2026-07-25 Release-Readiness Run

This run used the working tree based on revision
`68271c110a8d2be8696ed68d560bfb3908086be1`, an NVIDIA GeForce RTX 3050 Laptop
GPU, automatic device selection, preinstalled model bundles, and this sentence:

> Morning light rested on the cedar trees while the kettle began to sing.

The native Burn and VITS logs confirmed their CUDA implementations. ONNX
provider selection was not emitted, so these results do not infer an execution
provider from the presence of a GPU. StyleTTS2 used the `fast` preset with two
diffusion steps. Each row is one fresh CLI process, so wall time includes model
loading and process startup.

| Case | Speaker | Wall | Peak RSS | Audio | Rate | Mean | Peak | RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Burn SpeedySpeech + HiFi-GAN | single | 13.13 s | 360 MiB | 4.214 s | 22,050 Hz | -26.4 dBFS | -8.1 dBFS | 3.12 |
| VITS | p225 (ID 1) | 27.52 s | 433 MiB | 3.123 s | 22,050 Hz | -32.2 dBFS | -15.2 dBFS | 8.81 |
| VITS | p330 (ID 90) | 32.45 s | 433 MiB | 2.961 s | 22,050 Hz | -23.5 dBFS | -2.9 dBFS | 10.96 |
| ONNX LJSpeech high | single | 4.62 s | 463 MiB | 4.087 s | 22,050 Hz | -23.9 dBFS | -7.3 dBFS | 1.13 |
| StyleTTS2 fast | reference-conditioned | 7.81 s | 850 MiB | 5.300 s | 24,000 Hz | -25.9 dBFS | -5.3 dBFS | 1.47 |

All five outputs were mono, 16-bit little-endian PCM WAV files. None clipped:
the hottest output, p330, retained 2.9 dB of peak headroom. The two VITS
speakers produced measurably different duration and level, confirming that
p330 resolved to its own model-declared embedding rather than silently falling
back to p225.

These are smoke measurements, not a benchmark: one sample is useful for
detecting regressions in execution, format, silence, gross level, and speaker
selection, but not for comparing perceptual quality or steady-state throughput.
