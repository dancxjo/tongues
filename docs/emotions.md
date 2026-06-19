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
  --out models/emotions/v0
```

Before training starts the CLI prints the checkpoint paths:

| File | Purpose |
|---|---|
| `train_state.json` | Current epoch, best loss, and metric state. |
| `model-epoch-N.json` | Per-epoch checkpoint. |
| `model.json` | Best validation-loss model. |
| `model_config.json` | Labels and feature dimensions. |
| `manifest.json` | Generic model-family artifact metadata. |

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
