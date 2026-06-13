# Architecture

The current G2P2G model is a shared-vocabulary encoder-decoder Transformer:

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

The CLI and crate layout are organized around model families rather than one monolithic model:

- `g2p2g`: spelling <-> broad IPA;
- `wiktionary`: multilingual orthography/phonology tasks;
- `sentence-parser`: `speaking::syntax::SentenceSyntaxAnalysis` scaffold;
- `interpretation`: LibriSpeech acoustic interpretation scaffold with compact
  audio features, streaming CTC-style heads, frame-level auxiliary heads, and a
  lightweight after-utterance transcript head;

Each family can own its data preparation, task tags, training config, artifact metadata, and inference command while sharing common workspace infrastructure.

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

## Sentence Parser Scaffold

```sh
cargo run --release -- sentence-parser prepare
cargo run --release -- sentence-parser train
cargo run --release -- sentence-parser eval --model models/sentence-parser/v0
cargo run --release -- sentence-parser parse --model models/sentence-parser/v0 "The quick brown fox jumps."
```

The parser scaffold writes the expected model-family artifact files and returns JSON shaped as `speaking::syntax::SentenceSyntaxAnalysis`. Its current parser backend delegates to the existing heuristic parser until a neural architecture is implemented.

## Rule-Based Speech Helpers

```sh
just phonemes "hello world"
just phones "hello world"
```

Runs the rule-based speech pipeline directly.

```sh
just speak "hello world"
```

Synthesizes speech through the configured backend.
