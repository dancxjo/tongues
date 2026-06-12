# G2P2G

`g2p2g` is the active spelling/pronunciation model family. It trains a shared-vocabulary encoder-decoder Transformer for:

- `g2p`: spelling/graphemes to broad IPA phonemes;
- `p2g`: broad IPA phonemes to spelling/graphemes;
- `both`: an even mix of both directions.

## Prepare Data

```sh
just prepare
```

Direct form:

```sh
cargo run --release -- g2p2g prepare \
    --out datasets/g2p2g/openepd-v0
```

## Clean Start

```sh
cargo run --bin tongues -- g2p2g clean --all
```

`clean` archives selected default artifacts and recreates empty directories for the next run. Use `--data` or `--model` to archive only one side. Archives preserve the original relative path under `archive/<run-id>/...`; pass `--run-id NAME` for a named archive folder or `--no-create` to skip recreating empty defaults.

`prepare` builds `datasets/g2p2g/openepd-v0` from the embedded OpenEPD corpus. Splits are deterministic by base word, and alternate source entries for the same base word are collapsed before splitting.

The prepared directory contains:

| File | Purpose |
|---|---|
| `train.jsonl` | Training lexemes. |
| `valid.jsonl` | Validation lexemes. |
| `test.jsonl` | Test lexemes. |
| `train_words.txt` | Words assigned to train. |
| `valid_words.txt` | Words assigned to validation. |
| `test_words.txt` | Words assigned to test. |
| `vocab.json` | Unified vocabulary and special tokens. |

Each JSONL row looks like:

```json
{"base_word":"farkle","phonemes":"ˈfɑɹ.kəl","rarity":50000.0}
```

`rarity` is OpenEPD's 0-indexed wordfreq rank: lower means more frequent.

## Train

```sh
just train
```

Direct form:

```sh
cargo run --release -- g2p2g train \
    --data datasets/g2p2g/openepd-v0 \
    --out models/g2p2g/openepd-v0 \
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
cargo run --release -- --cpu g2p2g train --data datasets/g2p2g/openepd-v0
```

Training applies frequency weighting from OpenEPD rarity ranks, repeating the most common words up to 8 times and leaving words at or beyond rank 50,000 unexpanded. For `--task both`, training alternates task directions within each shuffled batch.

The train command still accepts legacy masking flags such as `--mask-policy`, `--max-mask-rate`, and `--span-mask-prob`. They are currently ignored by the seq2seq training path.

The model directory receives:

| File | Purpose |
|---|---|
| `model.bin` | Best model weights. |
| `model-epoch-N.bin` | Per-epoch checkpoints. |
| `model_config.json` | Architecture config. |
| `train_config.json` | Training config, including task direction. |
| `train_state.json` | Resume state. |
| `vocab.json` | Copied vocabulary for self-contained prediction. |
| `manifest.json` | Generic model-family artifact metadata. |

Training resumes automatically when `train_state.json` and checkpoint files are present in the output directory.

## Infer

```sh
just infer "farkle"
just infer --task p2g "ˈfɑɹ.kəl"
just infer --cpu "ˈfɑɹ.kəl"
```

Direct form:

```sh
cargo run --release -- g2p2g infer \
    --model models/g2p2g/openepd-v0 \
    "farkle"
```

Task detection is automatic by default. You can force a direction:

```sh
cargo run --release -- g2p2g infer \
    --model models/g2p2g/openepd-v0 \
    --task p2g \
    "ˈfɑɹ.kəl"
```

Prediction searches for `vocab.json` in:

1. the explicit `--data` directory;
2. the model directory;
3. the model directory's parent;
4. a sibling `runs/<model-name>/vocab.json` for legacy layouts.

Training copies `vocab.json` into the model directory, so ordinary prediction should not require `--data`.

## REPL

```sh
cargo run --release -- g2p2g repl --cpu
```

The REPL is also the default subcommand:

```sh
cargo run --release -- --cpu
```

The REPL loads vocabulary, device, config, and weights once, then accepts repeated inputs:

```text
tongues> farkle
ˈfɑɹ.kəl

tongues> ˈfɑɹ.kəl
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

## Evaluate

```sh
cargo run --release -- g2p2g eval \
    --model models/g2p2g/openepd-v0 \
    --data datasets/g2p2g/openepd-v0 \
    --split test \
    --task auto
```

`--task auto` reads `train_config.json`. You can also force `g2p`, `p2g`, or `both`.

Metrics currently include:

- loss;
- exact-match accuracy;
- token accuracy.

Use exact match for strict whole-output correctness and token accuracy for "mostly got the pronunciation/spelling right."

## CMUdict

```sh
just fetch
```

Direct form:

```sh
cargo run --release -- fetch-cmudict --out data/cmudict.dict
```

CMUdict is fetched from:

```text
https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
```
