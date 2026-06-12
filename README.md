# tongues

Tongues is a Rust toolkit for neural lexical and speech-front-end models.

The project currently focuses on reversible mappings between written forms and pronunciation:

```text
farkle     -> ˈfɑɹ.kəl
ˈfɑɹ.kəl  -> farkel
```

Unlike a static dictionary lookup, Tongues learns from pronunciation lexicons and generalizes to unseen words.

```text
pneumocryptology -> ˌnuː.məˈkɹɪp.təˌloʊ.dʒiː
ˈzwɪ.kɚ.bɚɡ     -> zwickerberg
```

Tongues is also evolving into a broader speech research toolkit containing model families for:

- grapheme <-> phonology translation;
- multilingual pronunciation modeling;
- sentence parsing;
- multimodal speech representations;
- future TTS and ASR front-end work.

The codebase is organized around model families (`g2p2g`, `wiktionary`, `sentence-parser`, `speech-manifold`) that share common infrastructure while remaining independently trainable.

## Features

- Rust workspace using Burn 0.21.
- Shared neural artifact metadata and vocabulary tooling.
- OpenEPD-based pronunciation data preparation.
- Spelling-to-phoneme and phoneme-to-spelling inference.
- Interactive REPL for loaded-model prediction.
- Discrepancy and sight-word refinement workflows.
- Rule-based phonemicization and realization helpers in the local `speech` crate.
- Experimental StyleTTS2/Piper-adjacent speech plumbing.

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
cargo run --release -- g2p2g repl
```

Run tests:

```sh
cargo test
```

For detailed training, data preparation, and model-family documentation, see:

- [G2P2G](docs/g2p2g.md)
- [Wiktionary](docs/wiktionary.md)
- [Sentence parser](docs/sentence-parser.md)
- [Speech manifold](docs/speech-manifold.md)
- [Refinement](docs/refinement.md)
- [Architecture](docs/architecture.md)
- [Examples](docs/examples.md)
- [Licensing notes](docs/licensing.md)

## Workspace Layout

```text
crates/tongues-core              shared vocabulary and special token IDs
crates/tongues-data              lexicon parsing, IPA normalization, splits, collation
crates/tongues-neural            shared neural artifact metadata
crates/tongues-g2p2g             Burn seq2seq G2P/P2G model, training, evaluation, prediction
crates/tongues-wiktionary        Wiktionary pronunciation data and model-family scaffold
crates/tongues-speech-manifold   multimodal speech-manifold data/model family
crates/tongues-sentence-parser   cursor-boundary data and model-family code
crates/tongues-cli               command-line routing and model/data wiring
crates/speech                    rule-based phonemicization and realization pipeline
crates/styletts2                 StyleTTS2 symbol lowering and backend experiments

configs/                         default family config files
datasets/                        prepared local datasets
runs/                            run-local scratch/output artifacts
models/                          trained local model artifacts
docs/                            reference documentation
```

The workspace is defined in `Cargo.toml` and currently uses Burn with ndarray/autodiff plus optional CUDA support.

## Core Commands

| Command | Purpose |
|---|---|
| `just prepare` | Prepare default OpenEPD G2P2G data. |
| `just train` | Train the default `g2p2g` model. |
| `just infer "farkle"` | Run one G2P2G prediction. |
| `just sentence-parser train --training-set all` | Forward a model-family command to `tongues`. |
| `cargo run --release -- g2p2g repl` | Start the G2P2G REPL. |
| `cargo run --release -- g2p2g eval --model models/g2p2g/openepd-v0 --data datasets/g2p2g/openepd-v0` | Evaluate a trained model. |
| `just refine` | Fine-tune from validation/test discrepancies. |
| `just sight-words` | Fine-tune on built-in Dolch sight words. |
| `just phonemes "hello world"` | Run the rule-based phoneme helper. |
| `just phones "hello world"` | Run the rule-based phone helper. |
| `just race --cpu` | Run a compact smoke test across model families. |

Most commands also have direct `cargo run --release -- ...` forms documented in the model-family pages.

## Current Model Families

| Family | Purpose | Status |
|---|---|---|
| `g2p2g` | spelling <-> broad IPA | active |
| `wiktionary` | multilingual orthography/phonology | active |
| `sentence-parser` | cursor-time sentence boundary, continuation, and repair | experimental |
| `speech-manifold` | multimodal speech representations | experimental |

Legacy verb-first commands still work for now, but the active CLI shape is model-family first: `tongues g2p2g ...`, `tongues wiktionary ...`, and so on.

## Roadmap

Current focus:

- pronunciation modeling;
- multilingual pronunciation data;
- lexical refinement;
- phonology and realization plumbing.

Planned work:

- phonetic realization models;
- sentence boundary detection;
- streaming text chunk repair;
- prosody prediction;
- multimodal speech representations;
- ASR/TTS integration.

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

Tongues currently focuses on the lexical and phonological layer:

```text
orthography <-> phonology
```

Future sibling models may handle segmentation, lexical repair, phonetic realization, prosody, and ASR-adjacent phone/phoneme representations.

## License

MIT.

Generated datasets and audio may include or point to material with different terms. Review [licensing notes](docs/licensing.md) before redistributing prepared data or generated artifacts.
