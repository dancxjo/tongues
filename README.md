# pronlex

A Rust/Burn masked-phone predictor trained on raw CMUdict ARPABET data.

## What it does

`pronlex` trains a small Transformer model to predict missing phones in a
pronunciation sequence, given the word's spelling plus the surrounding phone
context.

```
spelling: "charlotte"
phones:   SH AA1 R L <MASK> T
target:   AH0
```

This is **v0**: ARPABET phones from CMUdict, CPU-only via the
[Burn](https://burn.dev) framework.  IPA support and a Listenbury integration
come later.

---

## Quick start

```sh
# 1. Fetch the dictionary
cargo run --release -- fetch-cmudict --out data/cmudict.dict

# 2. Build vocabularies and splits
cargo run --release -- prepare \
    --input data/cmudict.dict \
    --out   runs/cmudict-v0

# 3. Train
cargo run --release -- train \
    --data runs/cmudict-v0 \
    --out  models/cmudict-v0 \
    --mask-policy variable \
    --max-mask-rate 0.4 \
    --span-mask-prob 0.15 \
    --learning-rate 3e-4 \
    --weight-decay 1e-4 \
    --dropout 0.1 \
    --epochs 20 \
    --patience 5

# 4. Evaluate
cargo run --release -- eval \
    --model models/cmudict-v0 \
    --data  runs/cmudict-v0 \
    --split test

# 5. Predict
cargo run --release -- predict \
    --model  models/cmudict-v0 \
    --word   charlotte \
    --phones "SH AA1 R L MASK T"
```

---

## Data source

CMUdict is downloaded from the public CMUShinx GitHub mirror:

```
https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
```

A local path is accepted too (`--input /path/to/cmudict.dict`).

---

## Why ARPABET first?

CMUdict ships ARPABET phones that include stress digits (`AH0`, `AH1`, `AH2`).
These are preserved exactly — they are distinct tokens.  IPA conversion is a
separate later step; v0 focuses on the prediction task with the raw source data.

---

## Train / validation / test split

Splits are made by **base word**, not by individual pronunciation variant.

`WORD`, `WORD(2)`, `WORD(3)` always land in the same split.  This prevents the
model from memorising a word's pronunciation from one variant and "predicting"
another.

A warning is printed if any base word appears in more than one split (should
never happen, but the check is there).

---

## Variable masking with curriculum schedule

Training uses on-the-fly masking controlled by `--mask-policy`:

| Policy     | Description                                  |
|------------|----------------------------------------------|
| `single`   | Always mask exactly one phone per example    |
| `variable` | Curriculum schedule (see below)              |

**Variable schedule** (epochs → sampling weights):

| Epoch range | Single | Double | Span | Random % |
|-------------|--------|--------|------|----------|
| 1–3         | 70 %   | 20 %   | 10 % | 0 %      |
| 4–8         | 50 %   | 30 %   | 15 % | 5 %      |
| 9+          | 40 %   | 30 %   | 20 % | 10 %     |

Loss is computed at **all** masked positions, not just one.  The same
architecture handles one or many missing phones.

---

## Model architecture

```
char_ids  [B, W]  ──embedding + pos─→ masked-mean-pool ─────┐
                                                              │ broadcast-add
phone_ids [B, P]  ──embedding + pos────────────────────────→ [B, P, D]
                                                              │
                                              2-layer Transformer encoder
                                                              │
                                              Linear → [B, P, phone_vocab]
```

Default hyper-parameters (all configurable):

| Parameter        | Default |
|------------------|---------|
| `d_model`        | 64      |
| `n_heads`        | 2       |
| `n_layers`       | 2       |
| `d_ff`           | 256     |
| `dropout`        | 0.1     |
| `batch_size`     | 64      |
| `epochs`         | 20      |
| `learning_rate`  | 3e-4    |
| `weight_decay`   | 1e-4    |
| `patience`       | 5       |

Optimizer: **AdamW** with configurable lr and weight decay.
Early stopping on validation loss (patience = number of epochs without
improvement before stopping).

---

## Metrics reported by `eval`

- Top-1 accuracy
- Top-3 accuracy
- Validation loss
- Confusion matrix (masked output)
- Per-phone accuracy
- Baseline comparison (most-common-phone-overall and most-common-by-position)

---

## Crate layout

| Crate           | Contents                                          |
|-----------------|---------------------------------------------------|
| `pronlex-core`  | `PhoneVocab`, `CharVocab`, token IDs              |
| `pronlex-data`  | CMUdict parser, masking, curriculum, data splits  |
| `pronlex-model` | Burn model, training loop, eval, predict          |
| `pronlex-cli`   | `clap`-based CLI wiring all commands together     |

---

## Running tests

```sh
cargo test
```

16 unit tests cover:

- CMUdict parser (comments, blanks, stress digits, alternate variants)
- Split leakage prevention
- Masking strategies (single, double, span, random-pct)
- Curriculum weight schedule
- Model forward-pass shape
- Tiny training fixture (no panic)
- Predict returns ranked phones

---

## Next steps

- IPA tokenisation via `data/phones.toml` registry
- Surface-realisation modelling (latent phoneme bottleneck)
- GPU backend support (WGPU)
- Listenbury integration
