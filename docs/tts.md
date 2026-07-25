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

The demo is orchestrated by `xtask`: it builds the optimized binary once,
starts one speech process, and preloads each unique backend once. All engines
remain resident for the whole run, so the two VITS speakers share one
CUDA-resident checkpoint instead of reloading it. CUDA is selected
automatically when available, StyleTTS2 uses its fast quality preset by
default, and each backend receives a shuffled sentence. Pass `--cpu` to force
the CPU path for backends that support both:

```sh
just speech-demo --cpu
```

To validate without opening an audio device, write each result to a WAV:

```sh
just speech-demo --output-dir target/speech-demo
```

For a repeatable five-case synthesis probe, measured output fields, and the
latest recorded run, see [Speech synthesis smoke measurements](speech-smoke.md).

For native stage profiling and cold-versus-warm CPU/CUDA measurements, run:

```sh
scripts/speech-benchmark.sh
```

The latest measured matrix and CUDA kernel evidence are in
[Native speech performance](speech-performance.md).

The benchmark keeps each model resident for repeated inference. `--timings`
emits machine-readable `startup_profile_json` and `inference_profile_json`
records with token, frame, and sample dimensions. Each inference record reports
first playable audio latency, total synthesis time, audio duration, and
real-time factor independently. Playback drain time is emitted separately as
`playback_profile_json` when no WAV output is requested.

Use `nsys` to verify CUDA kernel execution and inspect synchronization:

```sh
nsys profile \
  --trace=cuda,osrt \
  --stats=true \
  --force-overwrite=true \
  --output=target/speech-benchmark/vits-cuda \
  target/release/tongues --verbose speak \
    --backend vits --speaker p225 --seed 27 \
    --benchmark-runs 2 --timings \
    --output target/speech-benchmark/vits.wav \
    "Morning light rested on the cedar trees."
```

The first run includes Burn fusion/CUDA kernel compilation. Runs two and later
are the representative warm measurements.

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
- synchronized, structured profiling at native model boundaries;
- ONNX voice compatibility for registered voice bundles.

Burn synthesis uses CUDA when it is available and the command supports it, with
`--cpu` available for deterministic fallback. Release builds are strongly
recommended for real synthesis:

```sh
cargo run --release --bin tongues -- speak --backend vits --speaker p225 "Hello."
```

The native SpeedySpeech pipeline keeps mel features on the Burn device between
the acoustic model and HiFi-GAN, avoiding a GPU-to-host-to-GPU round trip. VITS
decodes the complete latent sequence once and slices the resulting host
waveform into sink chunks; it does not repeatedly launch the decoder over
overlapping latent windows.

## Speech conformance

The normal CI workflow checks the default, no-default, and declared feature
combinations, runs the workspace tests, and compiles the speech CLI in release
mode. These jobs do not download multi-gigabyte speech checkpoints.

Run the opt-in full-model harness on a machine with the registered Coqui and
ONNX model bundles:

```sh
scripts/speech-conformance.sh
```

The harness builds a container pinned to Coqui TTS revision
`0cf3265a4686d7e856bd472cdaf1572d61cab2b8` and PyTorch 1.13.1 CPU, verifies
every model artifact by SHA-256, and atomically writes reference evidence under
`target/speech-conformance/`. Missing or mismatched large artifacts are hard
failures, never successful skips.

The reference and native implementations use the same multiword sentence and
zero acoustic/duration noise. Conformance covers:

- SpeedySpeech token IDs, predicted durations, encoder expansion, positional
  encoding, mel frames, and HiFi-GAN output;
- VITS token IDs, speaker rows p225, p330, and p376, duration expansion, text
  prior, reverse flow, and integrated waveform decoder;
- the registered ONNX voice through an actual CLI synthesis;
- mono/22.05 kHz validity, bounded duration and levels, non-silent RMS/mean,
  peak, sample count, deterministic probes, and wall-clock timing.

Stage probes span the whole utterance. They serve as the semantic/intelligibility
gate: a native backend cannot pass merely by producing finite audio with the
right shape after diverging from the pinned upstream text, duration, acoustic,
or latent path. The small tokenizer fixture in `fixtures/speech/` is committed
to the repository; model weights and generated audio are not.

## Resident server execution

`tongues-server` owns a resident registry for the `burn`, `vits`, `onnx`,
`styletts2`, and `mock` backends. The first request loads the selected model;
subsequent `POST /api/speak` requests reuse it. Speech requests do not invoke
`cargo run`.
Responses include `X-Tongues-Model-Loaded`, `X-Tongues-Speech-Engine`,
`X-Tongues-Real-Time-Factor`, and `Server-Timing` headers.

Inspect loading, ready, and failed state at:

```text
GET /api/speech/runtime
```

The response exposes explicit `idle`, `loading`, `ready`, `busy`, `reloading`,
and `failed` state plus active, queued, and capacity counts. The synthesis UI
shows this state and provides a Reload Models control.

`POST /api/speech/runtime/reload` enters the same bounded admission path, waits
for an admitted active synthesis to finish, then drops loaded engines and clears
cached load failures. The next request reloads the current model files
deliberately.

The resident runtime uses CUDA by default. Set
`TONGUES_SPEECH_DEVICE=cpu` before server startup to force CPU inference.
Mutable engines are serialized through one registry mutex behind a bounded FIFO
admission gate. The default permits one active request and one queued request;
additional requests receive HTTP 429 with `Retry-After: 1`. Set
`TONGUES_SPEECH_MAX_IN_FLIGHT` to a value from 1 through 32 before server startup
to change the total admitted request count.

## Current Limits

- First use can spend time downloading and extracting model bundles.
- Debug builds are much slower than optimized inference.
- Native checkpoint compatibility is explicit rather than permissive; an
  unsupported topology or symbol produces an error.
- StyleTTS2 reference/style behavior and the neural streaming front end remain
  experimental even though their runnable integration paths exist.
