# Emotions

`emotions` is the audio emotion-classification model family. It reuses the
same labeled corpora fetched for StyleTTS2 emotion vectors and trains a compact
classifier that accepts a WAV file and returns ranked emotion labels.

## Data Source

Fetch or refresh the shared emotion corpora:

```sh
just fetch-corpora --out-dir datasets/emotions
```

That writes `datasets/emotions/labels.jsonl`, with absolute WAV paths plus
`emotion`, `speaker`, and `corpus`. The StyleTTS2 vector workflow consumes this
same manifest, so classifier preparation does not need a separate corpus list or
a completed `style_vectors.jsonl` file. `style_vectors.jsonl` still works as an
alternate `--source-manifest` because it carries the same `path` and `emotion`
fields.

## Prepare

Prepare compressed audio features:

```sh
just emotions prepare \
  --source-manifest datasets/emotions/labels.jsonl \
  --out datasets/emotions/v0
```

Preparation decodes each WAV, resamples it to 16 kHz mono, creates random cuts
plus one full-length cut by default, and writes pooled log-mel mean/std feature
vectors. The result is deliberately inspectable JSONL rather than opaque audio
blobs:

| File | Purpose |
|---|---|
| `examples.jsonl` | Recoverable stream of all generated cuts before shuffling. |
| `train.jsonl` | Training split. |
| `valid.jsonl` | Validation split. |
| `test.jsonl` | Test split. |
| `prepare_config.json` | Effective preparation config. |
| `prepare_state.json` | Durable prepare status and final counts. |
| `README.md` | Dataset summary. |

JSONL outputs are written through `.part` files and renamed after flush.

## Train

Train the classifier:

```sh
just emotions train \
  --data datasets/emotions/v0 \
  --out models/emotions/v0 \
  --epochs 100 \
  --patience 25
```

Before training starts the CLI prints the checkpoint paths:

| File | Purpose |
|---|---|
| `train_state.json` | Latest complete epoch, best metrics, patience, shuffle position, config, and data identity. |
| `train_state-epoch-N.json` | State paired with one durable epoch checkpoint for recovery. |
| `model-epoch-N.json` | Per-epoch checkpoint. |
| `model.json` | Best validation-loss model. |
| `model_config.json` | Labels and feature dimensions. |
| `manifest.json` | Generic model-family artifact metadata. |

Training uses precomputed pooled log-mel features from `prepare`, so each epoch
updates a compact linear classifier instead of decoding WAV files again. If the
run stops before `--epochs`, early stopping ended training after `--patience`
epochs without validation-loss improvement.

A directory with no training artifacts starts a new run. Continue an interrupted
compatible run explicitly:

```sh
just emotions train \
  --data datasets/emotions/v0 \
  --out models/emotions/v0 \
  --epochs 100 \
  --resume
```

Use `--restart` instead to deliberately replace the emotion training artifacts
in that output directory. Without either flag, existing artifacts are rejected
instead of being silently resumed or overwritten. Resume also rejects changes
to the prepared train/validation data, learning rate, weight decay, batch size,
patience, or seed with a migration/new-output-directory message.

Shuffling uses a documented seed-and-epoch derivation, so the next epoch does
not depend on an opaque in-memory RNG snapshot. For the same build, platform,
data, and effective config, resumed training is expected to match an
uninterrupted run exactly; the regression fixture uses zero tolerance. Epoch
weights and their paired state are atomically renamed after a file sync. Resume
scans backward for the latest complete pair, while `model.json` is copied from
the checkpoint named by `best_epoch`.

## Evaluate And Infer

Evaluate a split:

```sh
just emotions eval \
  --model models/emotions/v0 \
  --data datasets/emotions/v0 \
  --split test
```

Classify one WAV:

```sh
just emotions infer \
  --model models/emotions/v0 \
  path/to/audio.wav
```

Inference prints labels sorted by probability, one label per line.
