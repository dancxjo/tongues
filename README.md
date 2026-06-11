# pronlex

A Rust/Burn sequence-to-sequence pronunciation model.

`pronlex` trains a Transformer to translate in both directions:

```text
spelling -> broad IPA phonemes
farkle   -> ˈfɑɹ.kəl

broad IPA phonemes -> spelling
ˈfɑɹ.kəl           -> farkle
```

The current data pipeline starts from CMUdict word entries, filters to base
words, phonemicizes those words through the local `speech` crate, and writes
training examples as spelling/broad-IPA pairs.

---

## Quick Start

The default path is:

```sh
just train
```

`just train` runs:

```sh
cargo run --bin pronlex -- train \
    --data runs/cmudict-v0 \
    --out models/cmudict-v0 \
    --task both
```

If `runs/cmudict-v0` is missing `vocab.json`, `train.jsonl`, or `valid.jsonl`,
training automatically prepares the data first. If `data/cmudict.dict` is also
missing, training automatically downloads CMUdict first.

So the usual first run is just:

```sh
just train
```

---

## Just Recipes

```sh
just fetch
```

Downloads CMUdict to `data/cmudict.dict`.

```sh
just prepare
```

Optional. Builds `runs/cmudict-v0` without starting training. Use this when you
want to refresh or inspect the generated data, or pass custom prepare arguments.

```sh
just train
```

Trains `models/cmudict-v0`. By default it trains `--task both`, with an even
mix of spelling-to-phonemes and phonemes-to-spelling examples.

```sh
just infer "farkle"
just infer --task pm2s "ˈfɑɹ.kəl"
```

Runs one translation prediction.

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

## Data Flow

`prepare` writes a prepared data directory containing:

| File | Purpose |
|------|---------|
| `train.jsonl` | Training lexemes |
| `valid.jsonl` | Validation lexemes |
| `test.jsonl` | Test lexemes |
| `train_words.txt` | Words assigned to train |
| `valid_words.txt` | Words assigned to validation |
| `test_words.txt` | Words assigned to test |
| `vocab.json` | Unified character vocabulary and special tokens |

Each JSONL row is a `Lexeme`:

```json
{"base_word":"farkle","phonemes":"ˈfɑɹ.kəl"}
```

The split is deterministic by base word. Alternate source entries for the same
base word are collapsed before splitting, so a word cannot appear in multiple
splits.

---

## Training

Useful direct CLI form:

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

CUDA is used automatically when available. Pass global `--cpu` to force the
CPU backend:

```sh
cargo run --release -- --cpu train --data runs/cmudict-v0
```

`--task` controls the direction:

| Task | Meaning |
|------|---------|
| `s2pm` | spelling to phonemes |
| `pm2s` | phonemes to spelling |
| `both` | train both directions |

`both` is the default. In training, the default `both` path alternates task
directions within each shuffled batch, giving an even mix for normal even-sized
batches.

The train command still accepts legacy masking flags such as `--mask-policy`,
`--max-mask-rate`, and `--span-mask-prob`. They are currently ignored by the
seq2seq training path.

The model directory receives:

| File | Purpose |
|------|---------|
| `model.bin` | Best model weights |
| `model-epoch-N.bin` | Per-epoch checkpoints |
| `model_config.json` | Architecture config |
| `train_config.json` | Training config, including task direction |
| `train_state.json` | Resume state |
| `vocab.json` | Copied vocabulary for self-contained prediction |

Training resumes automatically when `train_state.json` and checkpoint files are
present in the output directory.

---

## Evaluation

```sh
cargo run --release -- eval \
    --model models/cmudict-v0 \
    --data runs/cmudict-v0 \
    --split test \
    --task auto
```

`--task auto` reads `train_config.json`. You can also force `s2pm`, `pm2s`, or
`both`.

Metrics currently reported:

- loss
- exact match accuracy
- token accuracy

---

## Prediction

```sh
cargo run --release -- predict \
    --model models/cmudict-v0 \
    "farkle"
```

Task detection is automatic by default. Use `--task` when you want to force a
direction:

```sh
cargo run --release -- predict \
    --model models/cmudict-v0 \
    --task pm2s \
    "ˈfɑɹ.kəl"
```

Prediction looks for `vocab.json` in this order:

1. the explicit `--data` directory
2. the model directory
3. the model directory's parent
4. a sibling `runs/<model-name>/vocab.json`

Because training copies `vocab.json` into the model directory, normal prediction
does not need `--data`.

---

## REPL

```sh
cargo run --release -- repl --cpu
```

The REPL is also the default subcommand:

```sh
cargo run --release -- --cpu
```

REPL commands:

- `:quit`, `:q`, or `Ctrl-D` exits
- `:task s2pm` forces spelling-to-phonemes
- `:task pm2s` forces phonemes-to-spelling
- `:auto` restores automatic task detection
- `:timings` toggles timing output
- `:help` prints available commands

---

## Model Architecture

The model is a shared-vocabulary encoder-decoder Transformer:

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

## Crate Layout

| Crate | Contents |
|-------|----------|
| `pronlex-core` | Unified vocabulary and special token IDs |
| `pronlex-data` | CMUdict parsing, IPA phonemicization, splits, collation |
| `pronlex-model` | Burn seq2seq model, training loop, eval, predict |
| `pronlex-cli` | CLI commands and model/data wiring |
| `speech` | Rule-based phonemicization and realization pipeline |
| `styletts2` | StyleTTS2 symbol lowering and backend support |

---

## Running Tests

```sh
cargo test
```

For the model crate only:

```sh
cargo test -p pronlex-model
```

---

## Notes

Some CLI help strings still mention the older masked-phone predictor. The
runtime model and data pipeline are now sequence-to-sequence translation.
