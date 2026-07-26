# Text to Speech

Tongues has runnable text-to-speech paths behind the `speak` command. The
default backend is native Burn synthesis; TTS is no longer only a planned
front-end integration.

## Backends

| CLI backend | Implementation | Model shape | Status |
|---|---|---|---|
| `burn` | `tongues-tts` on Burn | SpeedySpeech acoustic model -> selectable native vocoder | active |
| `fastpitch` | `tongues-tts` on Burn | controllable FastPitch acoustic model -> selectable native vocoder | active |
| `glow` | `tongues-tts` on Burn | Glow-TTS -> pinned per-bin standardizer -> MultiBand-MelGAN | active |
| `vits` | `tongues-tts` on Burn | end-to-end VCTK VITS with named speakers | active |
| `xtts` | `tongues-tts` on Burn | XTTS v2 GPT -> native HiFi decoder with reference voice conditioning | active local-package path |
| `onnx` | `tongues-tts` ONNX compatibility adapter | registered single- or multi-speaker voice bundle | active compatibility path |
| `styletts2` | `styletts2` ONNX backend | reference/style-conditioned synthesis | experimental |
| `mock` | deterministic local backend | synthetic waveform for integration tests | active test path |

The native Burn implementations import published checkpoint configuration and
weights, but inference runs through Tongues' Rust model components. The
registered SpeedySpeech, FastPitch, Glow-TTS, HiFi-GAN, and VITS bundles are
published Coqui release artifacts; the model registry records their sources,
sizes, checksums, and current license evidence. It does not claim that Tongues
invented those architectures. See [Speech system provenance](provenance.md).

FastSpeech, FastSpeech 2, and DelightfulTTS are available as native,
model-neutral acoustic components and through the safe Coqui package importer.
They are currently import-only rather than named CLI backends because the
upstream Coqui releases do not provide a redistributable, checksum-pinned model
artifact for these configurations. FastSpeech uses duration prediction only;
FastSpeech 2 additionally exposes pitch and energy prediction; DelightfulTTS
adds Conformer encoding/decoding, explicit variance adaptation, implicit
utterance/phoneme prosody, and optional speaker conditioning. Any of these
acoustic components can be composed with a compatible native vocoder whose mel
contract matches the imported model.

Tacotron 2 is available as a native, import-only acoustic component. The safe
importer accepts plain, DDC, and Capacitron configurations, reloads converted
weights through the Burn graph, and records the exact variant in `model.json`.
DDC's coarse decoder is a training regularizer and is not used during upstream
inference; released DDC packages may therefore omit it. Capacitron packages
declare a style-embedding contract in
`capacitron-standard-normal-latent-v1`; callers may supply that latent or let
the model sample its standard-normal prior. Reference-mel posterior encoding is
not claimed yet. Tacotron 1 configs are identified, but import fails before
writing because the native CBHG/decoder graph is not implemented.

Autoregressive safety is explicit. Location-sensitive attention retains both
the previous and cumulative alignment, stop probabilities are returned with
the mel, and reaching `max_decoder_steps` is an actionable attention failure
rather than a successful truncated synthesis. Released configs that require
prenet dropout at inference keep that behavior even on non-autodiff Burn
backends.

XTTS v2 is a native `xtts_v2` architecture and an active `xtts` CLI backend.
The importer validates the checkpoint's GPT text/audio embeddings, transformer
layer count, conditioning perceiver, HiFi decoder, and speaker encoder, then
loads the converted checkpoint through the complete Burn graph before accepting
the package. It also validates `vocab.json` special and language tokens against
the declared text vocabulary, records the 24 kHz output contract and declared
model languages, and copies the tokenizer into the checksummed package. Import
and inference do not execute Python.

The runtime performs speaker and GPT reference conditioning, cached GPT-2
autoregressive generation, seeded top-k/top-p sampling, and HiFi waveform
decoding. Speaker-conditioning entries are cached by canonical reference path,
file size, and modification time. The engine implements the same
`SpeechSynthesisEngine` and `PlanEngineBackend` contracts as other native
backends; the unified request's `reference_audio.speaker` field becomes a
backend-neutral speaker reference rather than an XTTS-specific public type.

Use a locally imported package and an explicit checkpoint-local language:

```sh
cargo run --release --bin tongues -- speak \
  --backend xtts \
  --model /path/to/xtts-v2-package \
  --model-language en \
  --voice-wav /path/to/reference.wav \
  --seed 17 \
  --output xtts.wav \
  "Hello, my friend."
```

The native cleaner currently advertises the published Latin- and
Cyrillic-script language entries. Arabic, Chinese, Japanese, and Korean remain
unadvertised until their upstream-compatible transliterators are native, and
digits must be spelled out so language-specific number expansion cannot be
silently approximated.

Streaming consumes cumulative waveforms as GPT code latents grow, emits only
the newly stable suffix, crossfades the recomputed 1,024-sample overlap, and
flushes the withheld tail in a final ordered chunk. Sink errors stop generation
immediately. Profiling separates reference conditioning, first autoregressive
code, remaining autoregressive generation, each waveform decode, and stream
assembly; this keeps conditioning cost, first-playable latency, steady RTF, and
overlap recomputation observable independently.

Run the two-language, two-reference and streaming-equivalence gate with:

```sh
TONGUES_XTTS_PACKAGE=/path/to/xtts-v2-package \
TONGUES_XTTS_REFERENCE_A=/path/to/reference-a.wav \
TONGUES_XTTS_REFERENCE_B=/path/to/reference-b.wav \
TONGUES_XTTS_CUDA_DEVICE=0 \
  scripts/xtts-conformance.sh
```

Set `TONGUES_XTTS_CPU=1` for the retained CPU path. The harness validates
English and French with both clips, decodes each WAV, records process peak
resident memory and structured native timings, and requires the concatenated
streaming WAV to equal one-shot decoding for the same seed.

## Safe Coqui package import

Legacy Coqui artifacts can be converted once into a backend-neutral, versioned
Tongues model package:

```sh
cargo run --bin tongues -- models import-coqui \
  --config /path/to/config.json \
  --checkpoint /path/to/model_file.pth \
  --speakers /path/to/speaker_ids.json \
  --out /path/to/tongues-package \
  --license LicenseRef-Documented-Upstream-Terms \
  --source https://example.invalid/upstream-model \
  --coqui-version 0.6.1
```

Omit `--speakers` for single-speaker acoustic models and vocoders. Use
`--languages language_ids.json` when a multilingual artifact has a separate
language map. `--checkpoint-key` defaults to `model`.

XTTS v2 reads languages from `config.json` and requires the official tokenizer
as a separate input:

```sh
cargo run --bin tongues -- models import-coqui \
  --config /path/to/XTTS-v2/config.json \
  --checkpoint /path/to/XTTS-v2/model.pth \
  --tokenizer /path/to/XTTS-v2/vocab.json \
  --out /path/to/xtts-v2-package \
  --license LicenseRef-Coqui-Public-Model-License-1.0.0 \
  --source https://huggingface.co/coqui/XTTS-v2/tree/cae391834a3834328bfa5a7b1ad3d6d6a46144c0 \
  --coqui-version dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e
```

The `LicenseRef` in this example denotes the model repository's CPML 1.0.0
terms; it is not an SPDX license or permission for commercial use.

The package contains:

| File | Purpose |
|---|---|
| `manifest.json` | Schema version, architecture, audio contract, speakers, languages, symbols, license, provenance, and source/package checksums |
| `model.json` | Canonical backend-neutral inference configuration |
| `model.safetensors` | Deterministically ordered tensor names and weights |
| `tensors.json` | Exact dtype and shape index used during validation |
| `vocab.json` | XTTS tokenizer; present only in XTTS packages and covered by the package file checksums |

Inspect without writing anything:

```sh
cargo run --bin tongues -- models import-coqui \
  --config /path/to/config.json \
  --checkpoint /path/to/model_file.pth \
  --license LicenseRef-Documented-Upstream-Terms \
  --source https://example.invalid/upstream-model \
  --dry-run --json
```

The `LicenseRef` above is an explicit placeholder, not a license grant. Replace
it with the model artifact's documented SPDX expression or a locally defined
`LicenseRef` backed by retained evidence; do not copy the source repository's
license onto weights by assumption.

Validate an existing package and every recorded file checksum:

```sh
cargo run --bin tongues -- models inspect-package /path/to/tongues-package
```

The importer normally accepts modern ZIP-based PyTorch checkpoints. Before
tensor loading, it scans `data.pkl`, permits only the storage/tensor
reconstruction globals needed by published checkpoints, and rejects arbitrary
globals, `STACK_GLOBAL`, unapproved object-construction opcodes, unsafe archive
paths, unsupported protocols, and oversized metadata. The MelGAN-family adapter
also accepts the published protocol-2 legacy checkpoint: its data-only parser
allows the same constrained tensor types, and normalizes the inert
`collections.Counter` metadata global to `collections.OrderedDict` in a
temporary copy before tensor loading. Parsing and conversion are Rust-only:
Python is never started, modules are never imported, and pickle callables are
never executed. Unknown inference fields are errors; training-only fields are
sorted into `ignored_training_fields` in the manifest rather than silently
discarded.

The XTTS profile additionally recognizes exactly four inert Coqui
configuration dataclasses embedded by the official checkpoint (`XttsConfig`,
`XttsArgs`, `XttsAudioConfig`, and `BaseDatasetConfig`) and the protocol-2
`NEWOBJ` opcode that builds their data representation inside the Rust parser.
No Python class is imported or executed; other globals, extension opcodes,
dynamic `STACK_GLOBAL`, and unknown XTTS classes remain rejected.

Plain MelGAN also recognizes the root state-dictionary layout used by the
MIT-licensed Descript `melgan-neurips` checkpoints. The importer preserves
those source tensor names in SafeTensors, records
`descript-pytorch-legacy` provenance, and applies an explicit checked mapping
into the same native generator topology at load time. The committed Linda
Johnson configuration records its log10 mel contract, zero inference padding,
and the original non-centered STFT's explicit 384-sample reflection padding.

Schema v1 packages are deterministic: they contain no timestamps or local
source paths, JSON fields and tensor names have stable ordering, writes use
`.part` files followed by atomic rename, and importing identical inputs with
identical provenance produces byte-identical package members. The manifest
reader includes an explicit v0-to-v1 migration path and rejects future schema
versions it cannot safely interpret.

Package compatibility proves structure and safe loading. Cross-runtime
numerical and audio conformance remains the separate #4 responsibility.

For the licensed LJSpeech Tacotron2-DDC fixture, set
`TONGUES_TEST_COQUI_TACOTRON2_CONFIG` and
`TONGUES_TEST_COQUI_TACOTRON2_MODEL`, then run:

```sh
cargo test --release -p tongues-tts tacotron --lib -- --nocapture
```

The opt-in tests validate the original checkpoint, the converted SafeTensors
package path, native stop/attention behavior, and finite 80-bin mel synthesis.
The release profile is intentional: the unoptimized 1024-channel
autoregressive CPU graph is not a useful throughput measurement.

## Verified model catalog and offline cache

Tongues has a schema-v1 backend-neutral model catalog for the native
SpeedySpeech, FastPitch, Glow-TTS, HiFi-GAN, MultiBand-MelGAN, VITS,
YourTTS, FreeVC, StyleTTS2, and ONNX voice backends.
Entries record architecture, package version, languages and varieties,
speakers, sample rate, capabilities, compatibility contracts, source format,
provenance, license evidence, artifact sizes, and SHA-256 checksums.
`architecture` is the provider-neutral neural family (`vits`, not
`coqui-vits` or `piper-onnx`); provider/runtime identities remain under
`compatible_with`, `provenance`, and legacy aliases. Source formats are strings
rather than runtime enums, so a private catalog can describe Coqui, Fairseq,
ONNX, or organization-specific artifacts without changing synthesis APIs.

```sh
# Catalog metadata and installation state
cargo run --bin tongues -- models list
cargo run --bin tongues -- models search vits
cargo run --bin tongues -- models inspect vits-vctk

# Verified install, offline reuse, and removal
cargo run --bin tongues -- models install vits-vctk
cargo run --bin tongues -- models install vits-vctk --offline
cargo run --bin tongues -- models verify vits-vctk
cargo run --bin tongues -- models verify --all
cargo run --bin tongues -- models remove vits-vctk

# Non-commercial multilingual voice cloning
cargo run --bin tongues -- models inspect yourtts-multilingual
cargo run --bin tongues -- models install yourtts-multilingual
cargo run --bin tongues -- speak --backend yourtts \
  --model-language fr-fr --voice-wav reference.wav \
  "Bonjour tout le monde"

# Install an already converted private/local package
cargo run --bin tongues -- models install \
  --package /path/to/tongues-package --id private-voice
```

`models fetch` remains a compatibility alias for the default runtime bundles,
while `models fetch --all` installs every registered bundle, including every
Piper ONNX voice. `just fetch` uses the latter after fetching the pronunciation
lexicons. Cataloged speech models use the same verified installer. Downloads
are resumable in `*.part` files, checked for the exact size and SHA-256 before
an atomic cache rename, then installed atomically. Archive extraction is
limited to registered normalized members. Installed records checksum every
runtime file, and the CLI and server refuse corrupt or mismatched artifacts.

Paths and offline behavior are explicit:

| Setting | Meaning |
|---|---|
| `TONGUES_MODEL_HOME` | Model installation root; falls back to `MORTAR_SEA_HOME`, then the platform data directory |
| `TONGUES_MODEL_CACHE` | Verified download cache; defaults to `<model-home>/cache/model-downloads` |
| `TONGUES_OFFLINE=1` | Prohibit network access and use only verified installed/cached artifacts |
| `TONGUES_MODEL_CATALOGS` | Platform-separated list of private/local schema-v1 catalog JSON files |

Commands also accept repeatable `--catalog /path/catalog.json`. Private
catalogs may add ids but cannot replace official ids or install over another
entry's paths. Installation metadata requires both a license value and a stable
evidence location, plus a pinned hash. `NOASSERTION` explicitly means the
catalog has not established redistributable terms; executability is not
permission to redistribute. The current Coqui `v0.6.1` model registry leaves
the relevant licenses `TBD` or blank, so those entries use `NOASSERTION` rather
than incorrectly inheriting the source repository's license.

### Fairseq MMS VITS

The embedded catalog also contains the 1,143 language- and script-specific
Fairseq MMS VITS checkpoints in Meta's pinned source inventory. Every entry
records the original `G_100000.pth`, `config.json`, and `vocab.txt` as separate
size- and SHA-256-verified artifacts. These models are published under
CC-BY-NC-4.0, so inspect the license before use:

```sh
cargo run --bin tongues -- models inspect fairseq-mms-vits-eng
cargo run --bin tongues -- models install fairseq-mms-vits-eng

# The verified installation is sufficient; no Python or network is used.
TONGUES_OFFLINE=1 cargo run --bin tongues -- speak \
  --backend fairseq --model fairseq-mms-vits-eng \
  --output hello.wav "This is a test."
```

Coqui's `tts_models/eng/fairseq/vits` name is retained as a catalog alias.
Within the Tongues portfolio this backend is the **Tier B language-coverage
renderer**: prioritize fast synthesis of short committed phrases and broad
cross-script coverage.
Script-qualified upstream ids remain separate entries, and language ids do not
claim pronunciation-variety support that the source metadata does not
establish. Models whose original config names a `.uroman` training input
require the same preprocessing at inference time. Point
`TONGUES_UROMAN` at a local `uroman.pl` or compatible executable; if it is
absent or fails, synthesis stops with an explicit preprocessing error instead
of silently feeding the wrong alphabet.

Fairseq MMS VITS remains a whole-utterance renderer. Capability metadata and
runtime behavior do not claim revision-safe suffix replacement inside the
model itself; when an unplayed phrase must be replaced around committed audio,
that happens in the ledger/orchestration layer. FastPitch + HiFi-GAN remains
the canonical fine-grained revision renderer.

The server uses the same verified installation and model id:

```json
{
  "backend": "fairseq",
  "model": "fairseq-mms-vits-eng",
  "text": "This is a test.",
  "variety": "en"
}
```

Catalog maintenance is split into two deterministic stages. The source
snapshot records the upstream language/script inventory, license evidence, and
all three artifact hashes; the Rust generator validates that metadata and
writes the runtime catalog atomically:

```sh
scripts/generate-fairseq-mms-source.py \
  --checkout /path/to/facebook-mms-tts \
  --language-index /path/to/all-tts-languages.html \
  --out crates/tongues-tts/catalog/fairseq-mms-source-v1.json
cargo run --bin tongues -- models generate-fairseq-catalog \
  --source crates/tongues-tts/catalog/fairseq-mms-source-v1.json \
  --language-index /path/to/all-tts-languages.html \
  --out crates/tongues-tts/catalog/fairseq-mms-models-v1.json
```

Both stages detect additions, removals, and renamed language rows. Catalog
validation rejects missing license evidence, checksums, sizes, scripts, or
preprocessing metadata. CI tests the committed 1,143-entry snapshot and a
small cross-script fixture matrix; it does not download every checkpoint.

The server exposes the same metadata and verification state at
`GET /api/models/catalog`. Installation and import perform exhaustive checksum,
archive-member, package-manifest, and model-metadata verification, then write an
atomic record under `<model-home>/models/.verification`. The record includes
the artifact fingerprint, model manifest version, verifier/cache schema
versions, runtime compatibility version, verified checksums, and file
size/mtime snapshots.

Startup and catalog discovery never re-hash every registered model. They use
only file presence, the versioned verification record, and size/mtime checks to
classify each entry as `verified`, `pending_verification`,
`changed_since_verification`, `verification_failed`, or `unavailable`.
Synthesis requires a still-valid cached result for only the selected pipeline;
its tensor and audio port contracts are then compared in memory before model
construction. `tongues models verify MODEL` and Studio's **Verify model** or
**Verify pipeline** actions perform an explicit deep check. `tongues models
verify --all` exhaustively checks installed catalog models for CI/nightly use.
Previously verified unchanged artifacts therefore remain immediately usable
offline without waiting for unrelated catalog entries.

## Native YourTTS conditioning

The YourTTS path uses the checkpoint's grapheme vocabulary and requires an
explicit model language (`en`, `fr-fr`, or `pt-br`). The selected row is
concatenated into the text encoder and independently projected into the
stochastic duration predictor; it is never guessed from `--variety`.

Speaker conditioning accepts either one named enrollment from `speakers.json`
or `--voice-wav`, never both. Reference WAVs are downmixed, resampled, split
into the upstream ten evaluation crops, and encoded by the native ResNet
attentive-statistics speaker encoder. Only derived 512-value embeddings may be
retained in the bounded runtime cache; reference PCM is not cached.
Precomputed embeddings must declare the exact
`coqui-resnet-speaker-encoder-0cf3265a-v1` space.

## Native FreeVC voice conversion

The `freevc` backend accepts source content and target-speaker references
through the unified `ReferenceAudioRequest`: `source` is the utterance to
convert and `speaker` is the target enrollment WAV. It does not accept text as
content and does not expose a provider-specific HTTP endpoint.

Inference is fully native: the source is downmixed and resampled to 16 kHz,
WavLM-Large extracts content features, the FreeVC speaker encoder produces the
256-value target embedding, and the conditioned FreeVC flow and decoder emit
24 kHz mono audio. The response metadata retains the original sample rate,
channel count, frame count, and RMS level for both input files even though the
runtime converts their internal representation.

Install the checksum-pinned MIT-licensed artifact set with:

```sh
cargo run --bin tongues -- models install freevc24-vctk
```

The native artifact conformance test is opt-in because the three checkpoints
total roughly 2.7 GB:

```sh
TONGUES_FREEVC_MODEL_DIR=/path/to/freevc24 \
TONGUES_FREEVC_SOURCE_WAV=/path/to/source.wav \
TONGUES_FREEVC_TARGET_WAV=/path/to/target.wav \
cargo test -p tongues-tts --lib \
  freevc::tests::published_artifacts_convert_without_python \
  --no-default-features -- --ignored
```

## Usage

Write native component synthesis to a WAV file:

```sh
just speak \
  --backend burn \
  --output /tmp/tongues-burn.wav \
  "Morning light rested on the cedar trees."
```

Use FastPitch's backend-neutral duration and normalized pitch-conditioning
controls:

```sh
just speak \
  --backend fastpitch \
  --speed 0.9 \
  --pitch-scale 1.08 \
  --pitch-shift 0.15 \
  --output /tmp/tongues-fastpitch.wav \
  "Morning light rested on the cedar trees."
```

`--pitch`, `--energy`, and `--durations` accept comma-separated values with
exactly one value per checkpoint-projected token. These explicit controls are
intended for reproducible programmatic synthesis; verbose CLI output reports
the projected token IDs and count. Pitch and energy values and shifts use the
checkpoint's normalized conditioning space, not physical units. Capability
discovery gates these fields: original FastSpeech does not advertise or accept
pitch/energy controls, while FastSpeech 2 and DelightfulTTS do.

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

Model-declared learned languages are separate from `--variety`. The variety
selects Tongues' pronunciation and linguistic plan; `--model-language` (or the
low-level `--language-id`) selects an exact checkpoint embedding row. There is
no prefix-based inference between the two namespaces. Inspect the selected
VITS model with:

```sh
just speak --backend vits --list-model-languages
```

The cataloged VCTK checkpoint reports no learned model languages because it is
an English model. Imported multilingual VITS packages load
`language_ids.json` through `LanguageCatalog`; multi-language models require an
explicit named or numeric selection, while a one-language embedding may
default to row zero.

The local server exposes every registered model and executable component graph
through `GET /api/speech/models`. Schema v3 separates `components`,
directed `compatibility` edges, complete `compositions`, convenient `presets`,
and the deprecated `paths` compatibility view. Component edges carry exact
contract-match status and a user-facing mismatch reason. Each composition
includes backend/model identity, model family, varieties, learned model
languages, named and numeric speakers, style and reference-audio support,
speed/seed/device support, normalized output format, installation state, and
provenance. The synthesis UI renders the VITS speaker selector from this
contract, including the checkpoint embedding ID; it does not carry a separate
hard-coded speaker list. The older
`GET /api/speech/speakers?backend=vits` endpoint remains as a compatibility
view. Linguistic varieties are independently enumerated from the shared
variety data registry at `GET /api/linguistic/varieties`.

`POST /api/speak` accepts either the legacy `backend`/`model` pair or a
component-addressed `pipeline`:

```json
{
  "text": "Composable speech.",
  "variety": "en-US-GA",
  "pipeline": {
    "input": "text",
    "projector": "projector/fastpitch-ljspeech",
    "acoustic_model": "fastpitch-ljspeech",
    "conditioners": [],
    "vocoder": "hifigan-v2-ljspeech",
    "output": "wav"
  }
}
```

End-to-end graphs use `end_to_end` and omit `acoustic_model` and `vocoder`.
Supplying both pipeline and legacy selection is rejected as ambiguous. Legacy
requests are normalized to the same canonical pipeline before validation,
resident model lookup, and inference.

Discovery itself is lightweight. Its `verification_ids` list contains only
installed entries that are new, changed, previously failed, or lack a valid
cached result; unavailable entries are never queued. `POST
/api/speech/models/verify/{model_id}` deeply verifies one entry, persists the
result, and returns an updated discovery snapshot. Speech Studio processes
background checks one at a time and also exposes explicit **Verify model**,
**Verify pipeline**, and **Verify all changed models** actions. A verified,
unchanged pipeline is ready as soon as the first discovery response arrives.

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

## Unified synthesis contract

`tongues-tts::UnifiedSynthesisRequest` is the public orchestration boundary for
native component pipelines, end-to-end models, imported ONNX voices,
reference-conditioned models, and future voice-conversion backends. It carries
text, pronunciation variety, optional named or numeric checkpoint-language
selection, named or numeric speaker selection, speaker/style and source
reference audio, named or embedded style, speed, stochastic seed, device, and
chunking/streaming intent.

Every imported implementation exposes `BackendCapabilities` and implements
`SynthesizerBackend`. Capability validation happens before inference and
returns typed errors for unsupported features, unsupported catalog values,
missing required selections, and malformed controls. FastPitch advertises
pitch scaling, pitch shifting, explicit per-token pitch, and explicit
per-token duration support through this same capability object. The resident
server keeps boxed implementations of that trait, so its synthesis path does
not branch on VITS, SpeedySpeech + HiFi-GAN, FastPitch + HiFi-GAN, StyleTTS2,
or ONNX output shapes.

Audio from every backend is normalized to interleaved `f32` chunks with sample
rate, channel count, frame offset, chunk/final markers, and common completion
metadata. Completion metadata reports backend/model identity, format, frame
count, audio duration, streaming mode, and backend timing stages.

## Native Burn Components

`crates/tongues-tts` contains:

- model-neutral contracts for linguistic input, spectrograms, codecs,
  conditioning, vocoders, and waveforms;
- published-config import and validation;
- native SpeedySpeech text/acoustic inference;
- native FastPitch duration, pitch, and mel inference with explicit controls;
- native Align-TTS encoder, duration prediction/override, alignment expansion,
  and mel decoding through the model-neutral acoustic contract;
- native Glow-TTS text encoding, deterministic or stochastic duration
  prediction, monotonic expansion, seeded Gaussian latent sampling, and
  reverse flow;
- SC-GlowTTS speaker conditioning through the shared typed embedding contract;
- native HiFi-GAN waveform generation;
- native MelGAN waveform generation and MultiBand-MelGAN generation with PQMF
  synthesis;
- native VITS text encoding, stochastic duration prediction, flow, and waveform
  decoding;
- model-declared VITS language catalogs plus learned language conditioning in
  both the text encoder and stochastic duration predictor;
- named-speaker lookup for the VCTK model;
- streaming audio chunks for end-to-end synthesis;
- synchronized, structured profiling at native model boundaries;
- ONNX voice compatibility for registered voice bundles.

Burn synthesis uses one device policy across native model components. By
default, `auto` probes CUDA device 0 and falls back to CPU when CUDA cannot be
initialized. Use `--cpu` for an explicit CPU request, or `--cuda-device INDEX`
to require a particular CUDA device. Explicit CUDA requests never fall back:
an unavailable or invalid index exits with an error before model loading.
Release builds are strongly recommended for real synthesis:

```sh
cargo run --release --bin tongues -- speak --backend vits --speaker p225 "Hello."
cargo run --release --bin tongues -- --cuda-device 1 speak --backend vits --speaker p225 "Hello."
```

Before cache checks or model loading, verbose output reports the resolved device
and emits `device_metadata_json` with `kind` and nullable `index` fields. Native
startup profiles repeat those fields as `device` and `device_index`.

The native SpeedySpeech, FastPitch, and Glow-TTS pipelines are generic over a
Burn tensor vocoder and keep mel features on the device between components,
avoiding a GPU-to-host-to-GPU round trip. Select a cataloged CLI implementation
with `--vocoder hifigan` or `--vocoder multiband-melgan` where that acoustic
backend exposes the option. Construction compares the
complete spectrogram contract: layout, bins, hop size, frequency bounds, log
scale, mel filter bank, and normalization identity must all match. The
published MultiBand-MelGAN bundle requires its exact standardized feature
statistics, so the current LJSpeech acoustic bundles are rejected rather than
silently composed with it.

Align-TTS imports use the same safe, deterministic Coqui package path as the
other acoustic families. Runtime inference loads the encoder, duration
predictor, decoder, and the training-only MDN/modulation tensors without
executing Python or pickle. The acoustic adapter accepts the shared
`length_scale` and explicit-duration controls, emits the shared frame-major mel
contract, and therefore composes with any registered native vocoder whose
spectrogram contract matches exactly. `BurnAlignTtsPipeline` keeps those
features on the selected Burn device through a compatible native vocoder.

The model-neutral `BurnAcousticTrainingHooks` contract exposes Coqui's
alignment, decoder, acoustic, duration-predictor, and joint phases without
making inference depend on a trainer. Align-TTS imports retain the four phase
boundaries and SSIM, spectrogram, duration, and MDN loss weights in the neutral
package configuration. The hook returns hard monotonic alignment, aligned and
predicted duration logs, optional decoder mel output, and MDN alignment
likelihoods for trainer-owned losses and evaluation.

CPU CI uses the redistributable fixture under
`fixtures/speech/align-tts-mpl-fixture/`. It was generated by the pinned
MPL-2.0 Coqui implementation, is not a pretrained voice, and verifies exact
duration/alignment structure plus sparse mel probes at an absolute tolerance of
`3e-4`. The same test set imports the artifact through the safe package path and
synthesizes it through a small native HiFi-GAN on CPU.

Glow-TTS deliberately emits its unstandardized checkpoint mel contract. The
historical Coqui LJSpeech release names MultiBand-MelGAN as its default
vocoder, whose analysis geometry matches but whose inputs use the published
per-bin statistics in `scale_stats.npy`. The `glow` composition inserts the
named `coqui-ljspeech-multiband-melgan-standardize-v1` component. Its 80 means
and standard deviations are package data pinned to SHA-256
`8c4a45b935563157509ddbff09f59e4ffea35e1d07f3bbf87ec21484cb275c4a`;
construction rejects any other normalization identity or change in mel
geometry. No sample rate, log base, frequency range, layout, or vocabulary is
silently repaired.

Run the licensed CPU path with:

```sh
cargo run --release --bin tongues -- --cpu speak --backend glow \
  --timings --benchmark-runs 2 --seed 27 \
  "Morning light rested on the cedar trees."
```

The two benchmark runs report cold/warm total synthesis time, first playable
audio, real-time factor, and model-stage timings. The server and Speech Studio
discover the same composition from the shared pipeline registry; there is no
Glow-specific HTTP endpoint or UI branch.

The model historically published under the SC-GlowTTS name uses SC to mean
speaker conditioning: it consumes a 256-value Coqui speaker-encoder d-vector
and retains deterministic durations. Stochastic duration is a separate,
config-driven `use_sdp` mode. Reference audio must first be lowered by a
compatible speaker encoder into `SpeakerReferenceSource::Embedding`. VITS
decodes the complete latent sequence once and slices the resulting host
waveform into sink chunks; it does not repeatedly launch the decoder over
overlapping latent windows.

## Speech conformance

The normal CI workflow checks the default, no-default, and declared feature
combinations, runs the workspace tests, and compiles the speech CLI in release
mode. These jobs do not download multi-gigabyte speech checkpoints.

Run the opt-in full-model harness on a machine with the pinned Coqui, Descript,
and ONNX model artifacts:

```sh
scripts/speech-conformance.sh
```

The harness builds a container pinned to Coqui TTS revision
`0cf3265a4686d7e856bd472cdaf1572d61cab2b8`, Descript MelGAN revision
`6488045bfba1975602288de07a58570c7b4d66ea`, and PyTorch 1.13.1 CPU. It
verifies every model artifact by SHA-256 and atomically writes reference
evidence under `target/speech-conformance/`. Missing or mismatched large
artifacts are hard failures, never successful skips.

The reference and native implementations use the same multiword sentence and
zero acoustic/duration noise. Conformance covers:

- SpeedySpeech token IDs, predicted durations, encoder expansion, positional
  encoding, mel frames, and HiFi-GAN output;
- FastPitch token IDs, predicted durations, normalized pitch conditioning, and
  mel frames;
- plain MelGAN waveform samples from the licensed Descript Linda Johnson
  checkpoint and a deterministic mel input;
- MultiBand-MelGAN generator and PQMF waveform samples from a deterministic mel
  input;
- VITS token IDs, speaker rows p225, p330, and p376, duration expansion, text
  prior, reverse flow, and integrated waveform decoder;
- multilingual YourTTS across two synthesized language IDs and three speaker selections,
  including a real reference WAV through the native speaker encoder and full
  synthesis; the exact embedding and waveform limits are documented in
  [Published YourTTS conformance](yourtts-conformance.md);
- the registered ONNX voice through an actual CLI synthesis;
- mono/22.05 kHz validity, bounded duration and levels, non-silent RMS/mean,
  peak, sample count, deterministic probes, and wall-clock timing.

Stage probes span the whole utterance. They serve as the semantic/intelligibility
gate: a native backend cannot pass merely by producing finite audio with the
right shape after diverging from the pinned upstream text, duration, acoustic,
or latent path. The small tokenizer fixture in `fixtures/speech/` is committed
to the repository; model weights and generated audio are not.

## Resident server execution

`tongues-server` owns a resident registry for the `burn`, `fastpitch`, `vits`,
`onnx`, `styletts2`, and `mock` backends. The first request loads the selected
model; subsequent `POST /api/speak` requests reuse it. Speech requests do not
invoke `cargo run`.
Responses include `X-Tongues-Model-Loaded`, `X-Tongues-Speech-Engine`,
`X-Tongues-Real-Time-Factor`, and `Server-Timing` headers.

Inspect loading, ready, and failed state at:

```text
GET /api/speech/runtime
```

Inspect the backend-neutral model and capability contract at:

```text
GET /api/speech/models
```

The response exposes explicit `idle`, `loading`, `ready`, `busy`, `reloading`,
and `failed` state plus active, queued, and capacity counts. The synthesis UI
shows this state and provides a Reload Models control.

`POST /api/speech/runtime/reload` enters the same bounded admission path, waits
for an admitted active synthesis to finish, then drops loaded engines and clears
cached load failures. The next request reloads the current model files
deliberately.

The resident runtime uses the same automatic CUDA-0-then-CPU policy by default.
Set `TONGUES_SPEECH_DEVICE=cpu` before server startup to force CPU inference,
`TONGUES_SPEECH_DEVICE=cuda` to require CUDA device 0, or
`TONGUES_SPEECH_DEVICE=cuda:1` to require CUDA device 1. Invalid or unavailable
explicit CUDA selections stop server startup with a clear indexed error. The
`/api/speech/runtime` response includes `device` and `device_index`; individual
`/api/speak` requests may set `cuda_device` to require a different indexed
resident engine, or `cpu: true` to force CPU.
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
