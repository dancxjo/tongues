# tongues

Tongues is a Rust speech-systems workspace. It spans written and spoken
language: pronunciation modeling, phonemic and phonetic realization, streaming
text segmentation, acoustic interpretation, text-to-speech synthesis, and
runtime playback.

Streaming audio, recognition, generated text, and speech-output lifecycle
surfaces share one versioned event IR; see
[the streaming event contract](docs/streaming-event-contract.md).

Reversible mappings between written forms and pronunciation remain one active
model family:

```text
farkle     -> ˈfɑɹ.kəl
ˈfɑɹ.kəl  -> farkel
```

Unlike a static dictionary lookup, the G2P2G models learn from pronunciation
lexicons and generalize to unseen words.

```text
pneumocryptology -> ˌnuː.məˈkɹɪp.təˌloʊ.dʒiː
ˈzwɪ.kɚ.bɚɡ     -> zwickerberg
```

The same workspace now also runs speech synthesis end to end. The native Burn
paths include SpeedySpeech, FastPitch, and Glow-TTS acoustic models, plus a
speaker-conditioned VCTK VITS model. ONNX voice compatibility, StyleTTS2,
streaming front ends, ASR-oriented interpretation, common-phone decoding, and
emotion modeling share the same linguistic and artifact infrastructure.

## Features

- Rust workspace using Burn 0.21.
- Shared neural artifact metadata and vocabulary tooling.
- OpenEPD and Wiktionary pronunciation-data preparation.
- Spelling-to-phoneme, phoneme-to-spelling, and multilingual pronunciation inference.
- Interactive REPL for loaded-model prediction.
- Discrepancy and sight-word refinement workflows.
- Lexicon-backed and rule-based phonemicization/realization helpers in the local `speaking` crate.
- Native Burn TTS using SpeedySpeech + HiFi-GAN, controllable FastPitch + HiFi-GAN, or end-to-end multi-speaker VITS.
- ONNX-compatible voices, StyleTTS2 synthesis and style controls, and a deterministic mock backend.
- Trainable experimental sentence/head detection, LibriSpeech interpretation, common-phone CTC, and emotion models.

## Quick Start

Train the default pronunciation model:

```sh
just train
```

Run inference:

```sh
just infer "farkle"
just infer --task p2g "ˈfɑɹ.kəl"
```

Start the interactive REPL:

```sh
just g2p2g repl
```

Synthesize speech. Missing registered model bundles are downloaded and verified
on first use:

```sh
just speak --backend burn --output /tmp/tongues-burn.wav "Hello from Tongues."
just speak --backend vits --speaker p225 --output /tmp/tongues-vits.wav "Hello from Tongues."
just speech-demo
```

Inspect the same backend-neutral speaking plan without synthesizing it:

```sh
cargo run --bin tongues -- speaking "Hello, world?"
cargo run --bin tongues -- speaking --json "Hello, world?"
cargo run --bin tongues -- speaking --variety ckt --json "Ԓыгъоравэтԓьэн йиԓыйиԓ."
```

`--json` emits the complete typed `speaking::UtterancePlan`, which is the
canonical backend-neutral IR. The default text form is a deliberately lossy
connected-speech projection used by streaming training paths. `tongues
phonemes` and `tongues phones` expose separate broad-phoneme and realized-phone
views of the same plan; none of those text views is a second IR contract.
Pronunciation-analysis schema v2 likewise stores the plan once and computes
text projections through `PronunciationAnalysis` accessors.

Run tests:

```sh
cargo test
scripts/check-feature-matrix.sh all
scripts/speech-smoke.sh
```

Before submitting a change, run the same formatting and Clippy gates as CI:

```sh
just lint
```

The repository toolchain is pinned in `rust-toolchain.toml`; rustup selects it
automatically and installs the required `rustfmt` and `clippy` components.
Toolchain upgrades are explicit repository changes: update the channel there
and every `dtolnay/rust-toolchain@…` workflow pin to the same reviewed release,
then run `just lint` and the workspace tests in the same commit.

For detailed training, data preparation, and model-family documentation, see:

- [G2P2G](docs/g2p2g.md)
- [Wiktionary](docs/wiktionary.md)
- [Sentence parser](docs/sentence-parser.md)
- [Head2Phones exceptional cases](docs/head2phones-exceptional-cases.md)
- [Head2Phones v0 model card](docs/models/head2phones-v0.md)
- [Interpretation](docs/interpretation.md)
- [Common Phone](docs/common-phone.md)
- [Emotions](docs/emotions.md)
- [Native audio and feature extraction](docs/audio.md)
- [Native VITS training and fine-tuning](docs/vits-training.md)
- [Native vocoder training and fine-tuning](docs/vocoder-training.md)
- [Text to speech](docs/tts.md)
- [Speech system provenance](docs/provenance.md)
- [Speech synthesis smoke measurements](docs/speech-smoke.md)
- [StyleTTS2 emotion vectors](docs/styletts2-emotions.md)
- [Refinement](docs/refinement.md)
- [Architecture](docs/architecture.md)
- [Examples](docs/examples.md)
- [Licensing notes](docs/licensing.md)

## Workspace Layout

```text
crates/tongues-core              shared vocabulary and special token IDs
crates/tongues-audio             native PCM, DSP, STFT/ISTFT, and mel features
crates/tongues-data              lexicon parsing, IPA normalization, splits, collation
crates/tongues-neural            shared neural artifact metadata
crates/tongues-g2p2g             Burn seq2seq G2P/P2G model, training, evaluation, prediction
crates/tongues-wiktionary        Wiktionary pronunciation data, training, and inference
crates/tongues-sentence-parser   cursor-time sentence boundary, continuation, and repair
crates/tongues-head2phones       streaming head-chunk-to-phone model family
crates/tongues-common-phone      compact acoustic frame -> phone/feature CTC model
crates/tongues-interpretation    utterance-level Mel ASR with sentence/phoneme supervision
crates/tongues-emotions          pooled-log-mel audio emotion classifier
crates/tongues-tts               native Burn and ONNX-compatible speech synthesis
crates/speaking                  linguistic varieties, phonemicization, realization, and ASR runtime
crates/styletts2                 StyleTTS2 planning, ONNX inference, and style controls
crates/tongues-cli               command-line routing and model/data wiring
crates/tongues-server            local HTTP/UI server
xtask/                           repository maintenance tasks

configs/                         default family config files
datasets/                        prepared local datasets
runs/                            run-local scratch/output artifacts
models/                          trained local model artifacts
docs/                            reference documentation
```

The authoritative workspace is defined in `Cargo.toml`. Maintained Rust source
lives under `crates/` and `xtask/`; historical top-level `pronlex-*` and
`speech` source trees have been removed from the checkout and remain available
through Git history. Their language data was not discarded:
`crates/speaking/src/data/varieties/` is the sole active variety registry and
contains the maintained, expanded implementations. Burn workloads use
ndarray/autodiff with CUDA where the selected command and machine support it.

## Core Commands

| Command | Purpose |
|---|---|
| `just fetch` | Fetch pronunciation lexicons and every registered model/voice bundle, including English and Chukchi MMS. |
| `just prepare` | Prepare default OpenEPD G2P2G data. |
| `just run ...` | Forward directly to `cargo run --bin tongues -- ...`. |
| `just train` | Train the default `g2p2g` model. |
| `just infer "farkle"` | Run one G2P2G prediction. |
| `just sentence-parser train --training-set all` | Forward a model-family command to `tongues`. |
| `just emotions prepare --source-manifest datasets/emotions/labels.jsonl` | Prepare emotion classifier cuts from the shared emotion corpora. |
| `just sentence-parser clean --all` | Archive default sentence-parser data/model artifacts and recreate empty run directories. |
| `just g2p2g repl` | Start the G2P2G REPL. |
| `just g2p2g eval --model models/g2p2g/openepd-v0 --data datasets/g2p2g/openepd-v0` | Evaluate a trained model. |
| `just refine` | Fine-tune from validation/test discrepancies. |
| `just sight-words` | Fine-tune on built-in Dolch sight words. |
| `just phonemes "hello world"` | Run the rule-based phoneme helper. |
| `just phones "hello world"` | Run the rule-based phone helper. |
| `just serve` | Start the local web UI on loopback; set `TONGUES_ALLOW_INSECURE_REMOTE=1` only for explicit insecure remote development binds. |
| `cargo run --bin tongues -- pronunciation --json "hello world"` | Inspect the versioned pronunciation trace and canonical plan. |
| `cargo run --bin tongues -- speaking --json "hello world"` | Emit the complete backend-neutral `UtterancePlan`. |
| `just speak --backend burn "hello world"` | Synthesize with native Burn SpeedySpeech + HiFi-GAN. |
| `just speak --backend fastpitch "hello world"` | Synthesize with native Burn FastPitch + HiFi-GAN. |
| `just speak --backend vits --speaker p225 "hello world"` | Synthesize with native Burn multi-speaker VITS. |
| `just speak --backend fairseq --variety ckt "Ԓыгъоравэтԓьэн йиԓыйиԓ."` | Synthesize Chukchi with the default MMS Chukchi checkpoint. |
| `just speech-demo` | Run a shuffled sentence through every built-in speech backend. |
| `just race --cpu` | Run a compact smoke test across the active model families. |
| `just be [--mechanical]` | Stream Ollama text through sentence detection, pronunciation, ONNX speech playback, and the local audio queue. |

The model-family `just` recipes forward their arguments to the `tongues` CLI.

`just be` is an end-to-end streaming demo: generated
text is streamed into a speech front end, split into speakable chunks, converted
to pronunciations, synthesized, and queued for playback. It works in both modes:
without `--mechanical`, it uses the resident `head2phones` model; with
`--mechanical`, it uses the deterministic sentence detector and speaking
phonemicizer. Both paths are valid, and making that equivalence possible is the
point of the current architecture. The neural front-end loop remains slow and
experimental, but speech synthesis itself is implemented and runnable through
the `speak` command.

Each model-family namespace also has a `clean` subcommand:

```sh
just g2p2g clean --data
just wiktionary clean --model
just sentence-parser clean --all
just emotions clean --all
```

`clean` moves selected default artifacts under `archive/<run-id>/...` and recreates empty default directories for the next prepare/train run. With no selection flags it behaves like `--all`; pass `--no-create` to archive without recreating directories.

## Current Systems

| Family | Purpose | Status |
|---|---|---|
| `g2p2g` | spelling <-> broad IPA | active |
| `wiktionary` | multilingual orthography/phonology | active |
| `tts` | native Burn acoustic/vocoder and end-to-end synthesis plus ONNX compatibility | active inference |
| `speaking` | linguistic IR, lexicons, phonemicization, realization, and ASR runtime | active |
| `common-phone` | compact acoustic frames -> phones/features | experimental |
| `sentence-parser` | cursor-time sentence boundary, continuation, and repair | experimental |
| `head2phones` | streaming head chunk -> phones front end | experimental |
| `interpretation` | utterance-level ASR and multi-head acoustic interpretation | experimental |
| `emotions` | pooled-log-mel audio emotion classification | experimental |
| `styletts2` | reference/style-conditioned ONNX synthesis | experimental |

Legacy verb-first commands still work for now, but trainable model-family
commands use the family-first shape: `tongues g2p2g ...`,
`tongues wiktionary ...`, and so on. Runtime systems such as synthesis use
direct commands such as `tongues speak ...`.

Current focus:

- pronunciation modeling;
- multilingual pronunciation data;
- lexical refinement;
- phonology and realization plumbing;
- native speech-model compatibility and inference performance;
- streaming speech input and output.

Planned work:

- improve neural sentence/head quality and latency;
- deepen learned prosody and phonetic realization;
- mature the interpretation and common-phone families into stronger ASR paths;
- tighten streaming playback, repair, and barge-in behavior.

## Why This Exists

Tongues is one piece of a larger streaming speech system. A practical streaming TTS stack needs more than a synthesizer:

```text
incoming text stream
  -> safe prefix / sentence-boundary detector
  -> repair and rewind protocol for bad cuts
  -> normalization
  -> lexical pronunciation
  -> prosody-ready phonological form
  -> synthesis
  -> playback queue / barge-in control
```

Tongues contains active or experimental implementations for each layer in that
path. Its shared `UtterancePlan`-style linguistic representation keeps
phonology and phonetics independent of a checkpoint's private vocabulary, and
`tongues-tts` performs the final model-specific projection and waveform
synthesis. In the other direction, `interpretation`, `common-phone`, and the
`speaking` ASR runtime connect audio back to text, phones, phonemes, and related
features.

## License

Tongues-authored source is MIT licensed. Compatibility code, model weights,
voices, datasets, fixtures, and other third-party material may carry separate
terms; the repository license does not erase those terms.

Review the [third-party notices](THIRD_PARTY_NOTICES.md), [speech provenance
ledger](docs/provenance.md), and [licensing notes](docs/licensing.md) before
redistributing source, prepared data, models, or generated artifacts.

Rust dependency locking, advisory/license/source checks, exception ownership,
system-library boundaries, and release inventories are documented in the
[dependency audit policy](docs/dependencies.md).
