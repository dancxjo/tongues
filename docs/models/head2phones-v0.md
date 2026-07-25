# Head2Phones v0 Streaming Front-End Model

Path: `models/head2phones/v0`

Release archive: `releases/head2phones-v0/head2phones-v0.tar.gz`

The repository's `models` path is a symlink to local generated artifacts. This
release archive follows that symlink and packages the dereferenced
`models/head2phones/v0` files into a normal tracked release directory.

Install or restore the packaged model with:

```sh
just release extract head2phones
```

Regenerate the release archive from the current symlinked local artifact with:

```sh
just release package head2phones
```

Archive SHA-256:

```text
f045838ba466f860f5c67106e1d73d0f433edacfced107c06bb293fdb4b60011  head2phones-v0.tar.gz
```

This is the first packaged `head2phones` seq2seq model. It predicts the first
complete TTS-speakable head chunk from a rolling UTF-8 text buffer, and emits
phones plus the split position for that chunk.

It is intended for the current `just be` / ONNX voice streaming path, where
the runtime can fall back to the original text when the model refuses, truncates,
or produces an incomplete control block.

## Versioned Artifact Policy

The repository tracks the minimal artifact set needed for inference and safe
resume:

- `model.bin`: best model weights
- `model-epoch-4.bin`: latest epoch checkpoint referenced by `train_state.json`
- `train_state.json`: resume state
- `model_config.json`, `train_config.json`, `head2phones_config.json`,
  `manifest.json`
- `vocab.json`
- `SHA256SUMS`

Earlier epoch checkpoints are intentionally ignored. They are useful locally for
inspection, but they are not necessary for inference or for resuming from the
current best checkpoint.

Current tracked checksums:

```text
18221861f68ac25a24b291082e76b9d3e12ea50c018fd27d0b293ee6af75876e  model.bin
18221861f68ac25a24b291082e76b9d3e12ea50c018fd27d0b293ee6af75876e  model-epoch-4.bin
a075ac63f28b27d2d1ab020dee609dd1d2b006defe4ba34830def02d98216e68  vocab.json
c65f196ebe58dbb75a1af68999ce38ebb6bf6f68c3d9f12be720f7c39f6f649b  head2phones_config.json
5054261a01e62e68d38e46e186c23e9b7e6efe461065469f052c33008dd1eb73  manifest.json
5f8be129a3027054c5f706b50fd6e266fd422770ed39f656b47d57618174085f  model_config.json
4aadb07d9e6eeb145589bb25ad90bd2c541a9b998d953620aba974b04a4f2bc7  train_config.json
6961eece38c1c4f7dfd3954496dd43f299e523fb8467d058ece906a1408afaa6  train_state.json
```

## Current State

`train_state.json`:

```json
{
  "current_epoch": 4,
  "best_val_loss": 0.0029459042,
  "best_epoch": 4,
  "best_exact_match": 0.894,
  "early_stop_metric": "val_loss"
}
```

`manifest.json`:

```json
{
  "schema_version": 1,
  "family": "head2phones",
  "architecture": "seq2seq-transformer",
  "created_by": "tongues",
  "data_id": "all-gutenberg-v0",
  "task": "head-chunk-to-phones"
}
```

Model shape:

```json
{
  "vocab_size": 403,
  "d_model": 128,
  "n_heads": 4,
  "n_layers": 3,
  "d_ff": 512,
  "dropout": 0.1,
  "max_seq_len": 256
}
```

Training config:

```json
{
  "learning_rate": 0.0003,
  "weight_decay": 0.0001,
  "dropout": 0.1,
  "batch_size": 8,
  "epochs": 20,
  "early_stopping_patience": 5,
  "max_seq_len": 256,
  "task": null,
  "max_frequency_repeat": 8,
  "frequency_rarity_cap": 50000.0
}
```

## Data Mix

The saved `head2phones_config.json` identifies the dataset as
`all-gutenberg-v0`. It combines synthetic buffers, exceptional examples,
naive-vs-seams discrepancy examples, and Project Gutenberg sources for these
varieties:

```text
en-US
fr-FR-Standard
de-DE-Standard
es-ES-Castilian
es-419-Standard
eo
la-Classical
la-Ecclesiastical
el-GR-Standard
grc-Attic
grc-Koine
sa-Deva-Standard
```

The packaged run has `verify_with_ollama=false`, so the training rows were not
preflight-audited by an Ollama verifier during preparation.

## Usage

CPU inference:

```sh
cargo run -q --bin tongues -- --cpu head2phones infer \
  --model models/head2phones/v0 \
  --variety en-US \
  "For years the mountain had been a place of legend."
```

Expected representative output:

```text
<HEAD_FOUND>
<HEAD_LENGTH> 50
<PHONES> ... </PHONES>
<SPLIT_AFTER> 50
```

Streaming story demo:

```sh
just be
```

The `be` path loads this model as:

```text
resident head2phones CPU model=models/head2phones/v0
```

## Caveats

This is an experimental release. It is good enough to exercise the streaming
front-end path, but it is not a production chunker or pronunciation oracle.

Observed from the June 20, 2026 `just be` run:

- It successfully emits complete phone blocks for many simple English prose
  chunks.
- It can return `<NO_HEAD>` for headings or decorated text such as markdown
  title lines. Runtime fallback to the original text is expected.
- It can truncate long outputs, including missing `</PHONES>` and later control
  fields. The runtime must validate the block before trusting it.
- It may split earlier than the full input and leave the rest of the buffer for
  later synthesis.
- It sometimes emits language-mismatch metadata for effectively English text,
  for example `DETECTED_LANG en` vs `EXPECTED_LANG en-US`.
- It is trained on multiple varieties, but the currently observed streaming
  path is primarily English. Non-English behavior should be treated as ungraded.
- The phones are broad IPA-ish `speaking` IR intended for downstream lowering.
  Backend-specific phoneme conversion, such as voice-model ARPABET lowering, remains
  a separate synthesis-time step.

The practical release contract is therefore:

1. Treat only a syntactically complete `<HEAD_FOUND>` block as model-positive.
2. Require plausible `<HEAD_LENGTH>`, `<PHONES>...</PHONES>`, and
   `<SPLIT_AFTER>` fields before consuming a model cut.
3. Fall back to deterministic text chunking or the original text on `<NO_HEAD>`,
   malformed output, language mismatch, or excessive latency.
4. Keep `model.bin`, `model-epoch-4.bin`, `train_state.json`, and `vocab.json`
   together if resuming training.

## Smoke Snapshot

Representative successful probe:

```text
Input:
For years the mountain had been a place of legend.

Output:
<HEAD_FOUND>
<HEAD_LENGTH> 50
<PHONES> ˈfɔɹ | ˈjɪɹz | ðə | ˈmaʊn.tən | ˈhæd | ˈbɪn | ə | ˈpleɪs | əv | ˈlɛ.dʒənd ↘ . </PHONES>
<SPLIT_AFTER> 50
```

Representative refusal:

```text
Input:
**The Library of the Silent Mountain**

Output:
<NO_HEAD>
```

Representative malformed long-output probe:

```text
Input:
Her days were filled with chores and the soft murmur of the stream that ran through the valley, but her heart beat to a different rhythm.

Output starts:
<HEAD_FOUND>
<HEAD_LENGTH> 95
<PHONES> ˈhɚ | ˈdeɪz | ˈwɚ | ...
```

That output was truncated before a closing `</PHONES>` tag in the observed CPU
probe, so it must be rejected by consumers that require a complete control
block.

## Resume

Resume training from the current artifact directory with the normal family
command:

```sh
cargo run --bin tongues -- head2phones train \
  --data datasets/head2phones/v0 \
  --out models/head2phones/v0
```

The shared seq2seq checkpointing path should resume from epoch 5 using
`model-epoch-4.bin` and continue updating `train_state.json` plus `model.bin`.
