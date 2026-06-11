# pronlex

`pronlex` is a Rust toolkit for experimenting with neural lexical and speech-front-end models. Its first working blade is a small Burn-powered sequence-to-sequence model that learns a reversible-ish mapping between English spelling and broad IPA phonemic strings.

```text
spelling -> broad IPA phonemes
farkle   -> ˈfɑɹ.kəl

broad IPA phonemes -> spelling
ˈfɑɹ.kəl           -> farkel
```

It is not just a dictionary lookup. The model is trained from lexicons and then asked to generalize to words it has never seen:

```text
pneumocryptology -> ˌnuː.məˈkɹɪp.təˌloʊ.dʒiː
ˈzwɪ.kɚ.bɚɡ     -> zwickerberg
```

The longer-term goal is high-quality streaming TTS plumbing: text segmentation, pronunciation prediction, lexical repair, prosody-ready phonological forms, and eventually related ASR-side representation work.

---

## Current status

Pronlex currently includes:

- a Rust workspace using Burn 0.21;
- a seq2seq pronunciation model;
- spelling-to-phoneme (`g2p`) prediction;
- phoneme-to-spelling (`p2g`) prediction;
- a REPL that keeps the model loaded for interactive use;
- OpenEPD-based data preparation;
- OpenEPD-based discrepancy mining and refinement;
- sight-word refinement using the built-in Dolch list;
- a local `speech` crate for rule-based phonemicization/realization;
- StyleTTS2/Piper-adjacent speech plumbing that is still experimental.

This project is moving quickly. Some old command help may still refer to the earlier masked-phone prototype. The active model path is now sequence-to-sequence translation.

---

## Workspace layout

```text
pronlex-core   shared vocabulary and special token IDs
pronlex-data   Lexicon parsing, IPA normalization, splits, collation
pronlex-model  Burn seq2seq model, training, evaluation, prediction
pronlex-cli    command-line interface and model/data wiring
speech         rule-based phonemicization and realization pipeline
styletts2      StyleTTS2 symbol lowering and backend experiments
```

The workspace is defined in `Cargo.toml` and currently uses Burn with ndarray/autodiff plus optional CUDA support.

---

## Quick start

The easiest path is through the `just` recipes:

```sh
just train
```

That trains the default model at `models/cmudict-v0` using data in `runs/cmudict-v0`. If the prepared data is missing, training prepares it from the embedded OpenEPD corpus first.

Direct form:

```sh
cargo run --release -- train \
    --data runs/cmudict-v0 \
    --out models/cmudict-v0 \
    --task both
```

The default `both` task trains spelling-to-phoneme and phoneme-to-spelling directions together.

---

## Common commands

### Fetch CMUdict

```sh
just fetch
```

or:

```sh
cargo run --release -- fetch-cmudict --out data/cmudict.dict
```

CMUdict is fetched from:

```text
https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
```

### Prepare data

```sh
just prepare
```

or:

```sh
cargo run --release -- prepare \
    --out runs/cmudict-v0
```

Optional. Builds `runs/cmudict-v0` from the embedded OpenEPD corpus without starting training. Use this when you want to refresh or inspect the generated data, or pass custom prepare arguments.

`prepare` writes a prepared data directory containing:

| File | Purpose |
|------|---------|
| `train.jsonl` | Training lexemes |
| `valid.jsonl` | Validation lexemes |
| `test.jsonl` | Test lexemes |
| `train_words.txt` | Words assigned to train |
| `valid_words.txt` | Words assigned to validation |
| `test_words.txt` | Words assigned to test |
| `vocab.json` | Unified vocabulary and special tokens |

Each JSONL row looks like:

```json
{"base_word":"farkle","phonemes":"ˈfɑɹ.kəl","rarity":50000.0}
```

`rarity` is OpenEPD's 0-indexed wordfreq rank: lower means more frequent.

Splits are deterministic by base word. Alternate source entries for the same base word are collapsed before splitting, so a word cannot leak across train/validation/test.

### Train

```sh
just train
```

Trains `models/cmudict-v0`. By default it trains `--task both`, with an even mix of grapheme-to-phoneme and phoneme-to-grapheme examples. Training also applies frequency weighting from OpenEPD rarity ranks, repeating the most common words up to 8 times and leaving words at or beyond rank 50,000 unexpanded.

Direct form:

```sh
cargo run --release -- train \
    --data runs/cmudict-v0 \
    --out models/cmudict-v0 \
    --task both \
    --learning-rate 3e-4 \
    --weight-decay 1e-4 \
    --dropout 0.1 \
    --batch-size 64 \
    --epochs 20 \
    --patience 5
```

CUDA is used automatically when available. Pass global `--cpu` to force the CPU backend:

```sh
cargo run --release -- --cpu train --data runs/cmudict-v0
```

Tasks:

| Task | Meaning |
|------|---------|
| `g2p` | spelling/graphemes to broad IPA phonemes |
| `p2g` | broad IPA phonemes to spelling/graphemes |
| `both` | train both directions |

`both` is the default. In training, the default `both` path alternates task directions within each shuffled batch, giving an even mix for normal even-sized batches.

The train command still accepts legacy masking flags such as `--mask-policy`, `--max-mask-rate`, and `--span-mask-prob`. They are currently ignored by the seq2seq training path.

The model directory receives:

| File | Purpose |
|------|---------|
| `model.bin` | Best model weights |
| `model-epoch-N.bin` | Per-epoch checkpoints |
| `model_config.json` | Architecture config |
| `train_config.json` | Training config, including task direction |
| `train_state.json` | Resume state |
| `vocab.json` | Copied vocabulary for self-contained prediction |

Training resumes automatically when `train_state.json` and checkpoint files are present in the output directory.

### Predict

```sh
just infer "farkle"
just infer --task p2g "ˈfɑɹ.kəl"
just infer --cpu "ˈfɑɹ.kəl"
```

Runs one translation prediction.

Direct form:

```sh
cargo run --release -- predict \
    --model models/cmudict-v0 \
    "farkle"
```

Task detection is automatic by default. You can force a direction:

```sh
cargo run --release -- predict \
    --model models/cmudict-v0 \
    --task p2g \
    "ˈfɑɹ.kəl"
```

Prediction searches for `vocab.json` in:

1. the explicit `--data` directory;
2. the model directory;
3. the model directory's parent;
4. a sibling `runs/<model-name>/vocab.json`.

Training copies `vocab.json` into the model directory, so ordinary prediction should not require `--data`.

### REPL

```sh
cargo run --release -- repl --cpu
```

The REPL is also the default subcommand:

```sh
cargo run --release -- --cpu
```

The REPL loads vocabulary, device, config, and weights once, then accepts repeated inputs:

```text
pronlex> farkle
ˈfɑɹ.kəl

pronlex> ˈfɑɹ.kəl
farkel
```

Commands:

- `:quit`, `:q`, or `Ctrl-D` exits;
- `:task g2p` forces spelling-to-phoneme;
- `:task p2g` forces phoneme-to-spelling;
- `:auto` restores automatic task detection;
- `:timings` toggles timing output;
- `:help` prints available commands.

For these tiny models, CPU inference is often faster than CUDA for interactive use because CUDA launch/transfer overhead can dominate.

### Evaluate

```sh
cargo run --release -- eval \
    --model models/cmudict-v0 \
    --data runs/cmudict-v0 \
    --split test \
    --task auto
```

`--task auto` reads `train_config.json`. You can also force `g2p`, `p2g`, or `both`.

Metrics currently include:

- loss;
- exact-match accuracy;
- token accuracy.

Use exact match for strict whole-output correctness and token accuracy for “mostly got the pronunciation/spelling right.”

### Refine from discrepancies

```sh
just refine
```

Mines validation/test pronunciation discrepancies and fine-tunes a copy of the model on those failed examples. The recipe enables verbose output, so each exceptional word is printed as it is found.

Direct form:

```sh
cargo run --release -- refine \
    --model models/cmudict-v0 \
    --data runs/cmudict-v0 \
    --out models/cmudict-v0-refined \
    --splits valid,test \
    --task g2p \
    --verbose \
    --learning-rate 1e-4 \
    --epochs 5 \
    --patience 2
```

Refinement runs the model over held-out splits, looks up reference pronunciations in OpenEPD (`open-english-pronouncing-dictionary`), normalizes them through the `speech` notation and syllabification layer, compares each prediction with that gold target using a broad comparison key, computes character-level edit distance on that key, writes every substantive mismatch to `discrepancies.jsonl`, and fine-tunes from the source model weights using only the mismatched lexemes.

Example discrepancy:

```text
word : zweig
gold : ˈzwaɪɡ
pred : ˈzweɪɡ
```

The default task is `g2p`, grapheme to phoneme. Use `--task p2g` for phoneme-to-grapheme refinement, or `--task both` to mine and train both directions. The source model directory is left untouched; refinement requires a separate `--out` directory.

With `--verbose`, each discrepant word is printed with its split, task, edit distance, input, gold target, and prediction.

Length marks, syllable dots, stress mark placement, and common rhotic spellings are ignored for discrepancy detection so refinement does not train on merely notational differences.

OpenEPD entries containing IPA characters outside the existing model vocabulary are skipped, because the saved model cannot emit tokens that are not in its `vocab.json` without rebuilding the vocabulary and retraining.

Some discrepancies are regular patterns worth training. Others are sight-word exceptions and probably belong in an override table rather than in the productive model.

### Sight-word refinement

```sh
just sight-words
```

Fine-tunes a copy of the model on the built-in Dolch sight-word list using OpenEPD gold pronunciations. The default output is `models/cmudict-v0-sight-words`.

Pass refinement flags after the recipe:

```sh
just sight-words --epochs 8 --learning-rate 5e-5
```

Direct form:

```sh
cargo run --release -- refine \
    --model models/cmudict-v0 \
    --data runs/cmudict-v0 \
    --out models/cmudict-v0-sight-words \
    --source sight-words \
    --task both
```

Unlike the default discrepancy source, `--source sight-words` trains every usable sight-word lexeme after OpenEPD and vocabulary filtering. It still writes `discrepancies.jsonl` so current sight-word failures are visible before fine-tuning.

#### Why sight words?

Not every pronunciation pattern is productive.

Some words are best treated as lexical exceptions and memorized directly:

```text
one
two
yacht
colonel
choir
```

The sight-word source gives the system a way to reinforce high-frequency irregular forms without requiring the productive model to contort itself around every historical spelling accident.

### Rule-based speech helpers

```sh
just phonemes "hello world"
just phones "hello world"
```

Runs the rule-based speech pipeline directly.

```sh
just speak "hello world"
```

Synthesizes speech through the configured backend.

---

## Model architecture

The current model is a shared-vocabulary encoder-decoder Transformer:

```text
task token + source chars -> embedding + position -> Transformer encoder
BOS + target chars        -> embedding + position -> Transformer decoder
decoder states            -> linear -> vocabulary logits
```

Default architecture:

| Parameter | Default |
|-----------|---------|
| `d_model` | 128 |
| `n_heads` | 4 |
| `n_layers` | 3 |
| `d_ff` | 512 |
| `dropout` | 0.1 |
| `max_seq_len` | 128 |

Default training:

| Parameter | Default |
|-----------|---------|
| `batch_size` | 64 |
| `epochs` | 20 |
| `learning_rate` | 3e-4 |
| `weight_decay` | 1e-4 |
| `patience` | 5 |
| `task` | `both` |

Optimizer: AdamW. Early stopping uses validation loss.

---

## Why this exists

Pronlex is one piece of a larger streaming speech system. A practical streaming TTS stack needs more than a synthesizer:

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

Pronlex currently focuses on the lexical/phonological part:

```text
orthography <-> phonology
```

Future sibling models may handle:

- end-of-utterance detection;
- headless sentence/chunk detection;
- streaming chunk repair;
- sight-word exception routing;
- pronunciation discrepancy mining;
- surface realizations and allophony;
- ASR-adjacent phone/phoneme representations.

The CLI will likely be reshaped around multiple model families rather than one monolithic `pronlex` model.

---

## Tests

```sh
cargo test
```

For the model crate only:

```sh
cargo test -p pronlex-model
```

---

## Development notes

- Models and prepared datasets are intentionally local artifacts and are not expected to be committed.
- The current name, `pronlex`, may change. The project is becoming more of a neural lexical/speech toolkit than a single pronunciation lexicon.
- Outputs can be wrong in useful ways. For reverse spelling especially, the model often produces plausible spellings rather than dictionary spellings: `ˈhɛ.loʊ -> hellow`, `ˈfɑɹ.kəl -> farkel`.

---

## License

MIT
