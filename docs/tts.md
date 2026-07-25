# Text to Speech

Tongues has runnable text-to-speech paths behind the `speak` command. The
default backend is native Burn synthesis; TTS is no longer only a planned
front-end integration.

## Backends

| CLI backend | Implementation | Model shape | Status |
|---|---|---|---|
| `burn` | `tongues-tts` on Burn | SpeedySpeech acoustic model -> HiFi-GAN v2 vocoder | active |
| `vits` | `tongues-tts` on Burn | end-to-end VCTK VITS with named speakers | active |
| `onnx` | `tongues-tts` ONNX compatibility adapter | registered single- or multi-speaker voice bundle | active compatibility path |
| `styletts2` | `styletts2` ONNX backend | reference/style-conditioned synthesis | experimental |
| `mock` | deterministic local backend | synthetic waveform for integration tests | active test path |

The native Burn implementations import published checkpoint configuration and
weights, but inference runs through Tongues' Rust model components. The
registered SpeedySpeech, HiFi-GAN, and VITS bundles are published Coqui release
artifacts; the model registry records their sources, sizes, checksums, and
licenses.

## Usage

Write native component synthesis to a WAV file:

```sh
just speak \
  --backend burn \
  --output /tmp/tongues-burn.wav \
  "Morning light rested on the cedar trees."
```

Use a named speaker from the VCTK VITS model:

```sh
just speak \
  --backend vits \
  --speaker p225 \
  --output /tmp/tongues-vits.wav \
  "Morning light rested on the cedar trees."
```

List the VITS speakers:

```sh
just speak --backend vits --list-speakers
```

The local server exposes the same model-declared catalog at
`GET /api/speech/speakers?backend=vits`. The synthesis UI renders that response
as a speaker selector, including the checkpoint embedding ID; it does not carry
a separate hard-coded speaker list. Linguistic varieties are independently
enumerated from the shared variety data registry at
`GET /api/linguistic/varieties`.

Force CPU execution with the global CLI flag:

```sh
just speak --cpu --backend vits --speaker p225 "Hello."
```

If `--output` is omitted, the CLI opens the local audio output device. Registered
model bundles are downloaded, checksum-verified when a checksum is available,
and extracted on first use under the platform model-data directory.

Run every built-in backend:

```sh
just speech-demo
```

The demo uses optimized inference, selects CUDA automatically when it is
available, and assigns a shuffled sentence to each backend. Pass `--cpu` to
force the CPU path for backends that support both:

```sh
just speech-demo --cpu
```

For a repeatable five-case synthesis probe, measured output fields, and the
latest recorded run, see [Speech synthesis smoke measurements](speech-smoke.md).

## Linguistic Boundary

The `speaking` crate produces a backend-neutral utterance plan containing
phonemes, realized phones, boundaries, speaker identity, and prosodic
information. `tongues-tts` projects that plan into the private vocabulary and
input contract of the selected checkpoint:

```text
text
  -> speaking phonemicization and realization
  -> UtterancePlan
  -> checkpoint-specific linguistic projection
  -> acoustic model or end-to-end speech model
  -> optional neural vocoder
  -> mono waveform
```

Surface details remain in the shared linguistic plan until the model boundary.
For example, the VCTK VITS projector preserves aspiration because its vocabulary
contains `ʰ`, while lowering the unsupported unaspirated-stop extension `˭` to
the base stop. Unknown symbols without an explicit compatibility lowering still
fail at the boundary.

## Native Burn Components

`crates/tongues-tts` contains:

- model-neutral contracts for linguistic input, spectrograms, codecs,
  conditioning, vocoders, and waveforms;
- published-config import and validation;
- native SpeedySpeech text/acoustic inference;
- native HiFi-GAN waveform generation;
- native VITS text encoding, stochastic duration prediction, flow, and waveform
  decoding;
- named-speaker lookup for the VCTK model;
- streaming audio chunks for end-to-end synthesis;
- ONNX voice compatibility for registered voice bundles.

Burn synthesis uses CUDA when it is available and the command supports it, with
`--cpu` available for deterministic fallback. Release builds are strongly
recommended for real synthesis:

```sh
cargo run --release --bin tongues -- speak --backend vits --speaker p225 "Hello."
```

## Current Limits

- First use can spend time downloading and extracting model bundles.
- Debug builds are much slower than optimized inference.
- Native checkpoint compatibility is explicit rather than permissive; an
  unsupported topology or symbol produces an error.
- StyleTTS2 reference/style behavior and the neural streaming front end remain
  experimental even though their runnable integration paths exist.
