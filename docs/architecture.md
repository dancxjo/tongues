# Architecture

Tongues is a workspace of typed speech components and trainable model families.
The shared linguistic representation connects text-side analysis to synthesis,
while acoustic families connect audio back to text and phonological labels:

```text
text
  -> sentence/head analysis
  -> phonemicization and phonetic realization
  -> utterance plan
  -> TTS model and waveform

audio
  -> tongues-audio native PCM/DSP and compact acoustic features
  -> interpretation/common-phone models
  -> text, phones, phonemes, boundaries, syntax, and emotion labels
```

## G2P2G Model

The G2P2G family is a shared-vocabulary encoder-decoder Transformer:

```text
task token + source chars -> embedding + position -> Transformer encoder
BOS + target chars        -> embedding + position -> Transformer decoder
decoder states            -> linear -> vocabulary logits
```

## Default Model Shape

| Parameter | Default |
|---|---|
| `d_model` | 128 |
| `n_heads` | 4 |
| `n_layers` | 3 |
| `d_ff` | 512 |
| `dropout` | 0.1 |
| `max_seq_len` | 128 |

## Default Training

| Parameter | Default |
|---|---|
| `batch_size` | 64 |
| `epochs` | 20 |
| `learning_rate` | 3e-4 |
| `weight_decay` | 1e-4 |
| `patience` | 5 |
| `task` | `both` |

Optimizer: AdamW. Early stopping uses validation loss.

## Model-Family Shape

The CLI and crate layout are organized around model families and composable
speech-runtime components rather than one monolithic model:

- `g2p2g`: spelling <-> broad IPA;
- `wiktionary`: multilingual orthography/phonology tasks;
- `sentence-parser`: cursor-time sentence boundaries, continuations, and repair;
- `head2phones`: rolling text heads to speakable phone sequences;
- `interpretation`: LibriSpeech acoustic interpretation scaffold with compact
  audio features, streaming CTC-style heads, frame-level auxiliary heads, and a
  lightweight after-utterance transcript head;
- `common-phone`: compact acoustic frames to phones, phonemes, and feature axes;
- `emotions`: pooled-log-mel audio emotion classification;
- `tongues-audio`: model-neutral WAV/PCM, channel/resample, STFT/ISTFT, and
  mel/log-mel feature extraction shared by training and inference;
- `tongues-tts`: native Burn and ONNX-compatible waveform synthesis;
- `tongues-duplex`: provider-neutral completion beams, deterministic
  posterior-mass selection, commit-frontier policy, and replayable simulation;
- `speaking`: shared linguistic varieties, lexicons, phonemicization,
  realization, and speech-runtime types;
- `styletts2`: StyleTTS2 planning, ONNX inference, and style controls.

Each family can own its data preparation, task tags, training config, artifact metadata, and inference command while sharing common workspace infrastructure.

The shared feature tensor carries its full serializable preprocessing
configuration. Imported Coqui field names are translated only by the
`tongues-tts` compatibility adapter. See
[Native audio and feature extraction](audio.md).

## Language and Variety Boundary

Tongues treats lower-case-l languages as data, not as runtime subsystems.
Every consumer resolves a tag through the same `speaking` registry and receives
a `LinguisticVariety`. Interlinguistic differences—aliases, names, inventories,
orthography-to-pronunciation rules, normalization, morphology, syntax,
phonotactics, allophony, prosody, and external language tags—belong under
`crates/speaking/src/data/varieties/`.

```text
CLI / server / training / ASR / TTS
  -> variety or language identifier
  -> BCP 47 / ISO 639 parser
  -> speaking variety registry
  -> LinguisticVariety
  -> generic phonemicization, realization, syntax, and synthesis interfaces
```

`langtag` validates external RFC 5646/BCP 47 tags; `isolang` parses and converts
ISO 639-1 and ISO 639-3 identifiers. Tongues' internal variety IDs remain opaque
registry keys because descriptive internal suffixes are not necessarily valid
BCP 47 subtags. Runtime and UI code must not dispatch on a particular language
name or maintain parallel language/variety lists. The server projects its
options directly from the data registry, and the browser consumes generic
`{value, label}` records.

## Speech Synthesis

The synthesis boundary is a typed linguistic plan rather than raw text or a
checkpoint-specific symbol string:

```text
text
  -> speaking phonemicization and phonetic realization
  -> UtterancePlan
  -> checkpoint-specific linguistic projector
  -> SpeedySpeech or FastPitch -> HiFi-GAN
     or end-to-end VITS
     or YourTTS (language row + d-vector/reference encoder)
     or ONNX/StyleTTS2 compatibility backend
  -> streaming waveform chunks and playback/WAV output
```

`speaking::UtterancePlan` is the only backend-neutral linguistic IR carried
between components. `tongues speaking --json` serializes that typed plan for
inspection. The default `tongues speaking` output is a lossy connected-speech
model target; head2phones, data, and debug views use pure projections owned by
`speaking`. Broad phonemes come only from intended phonemes, realized phones
come only from target phones, and checkpoint tokens remain a terminal
projection. None of those text forms is an alternate IR contract.

The native Burn component path and end-to-end VITS path are active inference
implementations. Model-private vocabulary lowering occurs only at the final
projector, leaving the shared phonological and phonetic representation intact.
See [Text to Speech](tts.md) for backend and command details.

Named speakers are model data rather than linguistic varieties. VITS loads the
checkpoint's `speaker_ids.json` through `SpeakerCatalog`; the server exposes
that catalog through `/api/speech/speakers`, and the UI uses the returned names
and embedding IDs.

Learned model languages follow the same ownership boundary. Multilingual VITS
loads `language_ids.json` through `LanguageCatalog` and conditions its text and
duration networks with the selected row. These opaque checkpoint identities
remain separate from the shared `VarietyId`; callers select the mapping
explicitly instead of deriving a model row from a language-tag prefix.

YourTTS composes that language-conditioned VITS graph with a typed speaker
embedding boundary. Named enrollment vectors and reference-audio vectors share
one explicit 512-dimensional embedding-space contract. Reference audio is
processed by the native Coqui ResNet/attentive-statistics encoder; VITS then
passes the normalized vector to duration prediction, the flow, and the
waveform decoder. Named speaker selection, precomputed embeddings, and
reference audio have explicit precedence errors rather than silent fallback.

## Duplex Belief Contracts and Ownership

The shared duplex utterance-belief contract lives in `crates/speaking` as
provider-neutral, serializable types (`duplex.rs`). It carries evidence states,
typed deltas/actions, commit-frontier updates, withdrawals/repairs, delivery
status, and a versioned replayable journal.

```text
observed_text / observed_acoustics / linguistic_inference / predicted_completion
   -> BeliefAction events
   -> BeliefEventJournal (versioned)
   -> deterministic replay
   -> UtteranceBeliefState
```

```text
delivery state machine:
planned -> synthesized -> verified -> queued -> played
```

```text
ownership boundary:
speaking (contracts only)
  <- interpreted evidence from tongues-interpretation / ASR
  <- synthesis delivery updates from tongues-tts / runtime playback
  <- orchestration policy from tongues-duplex
```

`speaking` owns only the backend-neutral IR and invariants. Interpretation,
TTS, playback, and orchestration crates emit or consume typed events but do not
introduce reverse dependencies into `speaking`.

### Deterministic duplex simulator

`tongues-duplex` accepts a provider beam whose hypotheses carry normalized
morpheme sequences, syntax, prosody, evidence references, and provenance. It
sorts equal-probability branches by stable hypothesis ID, selects the smallest
highest-probability set covering the configured posterior mass, and computes
that set's longest common morpheme prefix.

The shared prefix is only a candidate frontier. Each selected branch must tie
each newly committed morpheme to direct text or acoustic evidence that supports
the same normalized key. Predicted suffixes remain provisional even when all
selected branches predict them. New evidence can withdraw or repair those
branches without mutating the append-only committed prefix.

```text
direct text / mock acoustic evidence
  -> CompletionProvider proposals
  -> normalized completion beam
  -> posterior-mass hypothesis set
  -> longest common morpheme prefix
  -> direct-support gate
  -> append-only commit frontier
```

Every evidence, proposal, withdrawal, repair, beam inference, and frontier
advance is written to a versioned simulator journal. Replay reconstructs the
same final state without invoking the provider:

```sh
cargo run -p tongues-cli -- duplex demo --fixture who-shot-john-f
cargo run -p tongues-cli -- duplex demo \
  --chunk "Who shot John F." --chunk "Kennedy?"
cargo run -p tongues-cli -- duplex demo \
  --mock-acoustic "turn left"
```

The built-in fixtures cover initials, abbreviations, heteronyms, stress
changes, garden paths, code switching, end-of-turn uncertainty, chunked
`Who shot John F.` / `Kennedy?`, and mock acoustic evidence. They use no
checkpoint, large model, audio device, or live service.

### Incremental morphology

`speaking::IncrementalMorphemeAnalyzer` accepts either valid strings or byte
deltas. It buffers an incomplete UTF-8 scalar without exposing corrupt text,
revises the last extended grapheme and unfinished word as more text arrives,
and flushes all remaining candidates explicitly at end of stream.

```text
UTF-8 bytes + per-delta variety
  -> provisional graphemes and word candidates
  -> variety-owned morphology and phonology
  -> stable MorphemeOccurrenceId values
  -> append / replace / split / merge / finalize deltas
  -> native morpheme journal or duplex BeliefAction journal
```

The English analyzer uses the English variety's lexical morphemes,
pronunciations, and cross-boundary composition rules. Hyphenated compounds are
analyzed component by component, apostrophe clitics remain linguistic
morphemes, punctuation finalizes the preceding word, and a variety change is a
word boundary even when no whitespace is present. A variety without an
incremental analyzer emits one `Spec::Unknown` whole-word occurrence with
`language-neutral-whole-word-fallback` provenance. It never exposes BPE pieces
or checkpoint-private symbols as morphemes.

## Build and Runtime Verification

`scripts/check-feature-matrix.sh` checks the workspace with default features,
without default features, and with all features, then checks every declared
optional-feature combination for `speaking`, `tongues-tts`, `tongues-cli`, and
the StyleTTS2 ONNX execution providers. CI runs those matrices, the full
workspace test suite, and a JavaScript syntax check.

`scripts/speech-smoke.sh` performs real release-mode synthesis with Burn, VITS
p225, VITS p330, ONNX, and StyleTTS2. FastPitch's full-model stage comparison
is part of `scripts/speech-conformance.sh`. The smoke harness records machine-readable latency,
memory, audio format, duration, level, hash, and real-time-factor evidence under
`target/speech-smoke/`. See
[Speech synthesis smoke measurements](speech-smoke.md).

## Authoritative and Legacy Trees

`Cargo.toml` is authoritative for maintained Rust packages. Source lives in
`crates/` and `xtask/`. In particular,
`crates/speaking/src/data/varieties/` is the sole maintained language-variety
data tree; it contains the actively registered English, Esperanto, French,
German, Greek, Latin, Sanskrit, and Spanish data.

The former top-level `speech/` and `pronlex-cli/` trees were incomplete,
non-workspace duplicates. They are removed from the working tree, not treated
as the archive or authority for language data. Their historical snapshots
remain recoverable through Git history. No active variety module is removed:
the maintained `crates/speaking` versions are equal to or newer than those
top-level copies.

## Interpretation Scaffold

```text
[log_mel, delta_mel, energy, vad, zcr, centroid, flux, f0, voiced_prob]
  -> shared frame encoder
       -> CTC-style transcript/phone/phoneme/word heads
       -> boundary, repair, syntax, and masked-audio heads
       -> after-utterance transcript head
```

The interpretation family is intentionally a scaffold rather than a finished
ASR system. The streaming heads are meant to learn monotonic partial output and
alignment. The after-utterance head is meant to learn correction with more
context. Both share the same cheap acoustic frontend and model artifact layout.

## Sentence Parser

```sh
cargo run --release -- sentence-parser prepare
cargo run --release -- sentence-parser train
cargo run --release -- sentence-parser eval --model models/sentence-parser/v0 --data datasets/sentence-parser/v0 --split test
cargo run --release -- sentence-parser parse --model models/sentence-parser/v0 "The quick brown fox jumps."
```

The sentence parser family trains a cursor-boundary seq2seq model. `eval` runs model inference on a prepared split and reports action accuracy, boundary precision/recall/F1, and repair edit distance. The `parse` command runs the separate rule-based `SentenceSyntaxAnalysis` utility. Syntax analysis goes through the uniform grammar parser API: UDPipe can provide parsed CoNLL-U when configured, and each variety owns a fallback rule profile that emits the same typed links plus raw parser metadata. The cursor-time boundary model is still separate from the syntax backend.

## Rule-Based Speech Helpers

```sh
just phonemes "hello world"
just phones "hello world"
```

Runs the rule-based speech pipeline directly.

```sh
just speak "hello world"
```

Synthesizes speech through the default native Burn backend. Use
`--backend vits`, `--backend onnx`, `--backend styletts2`, or `--backend mock`
to select another runtime.
