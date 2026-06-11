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
- CMUdict-based data preparation;
- OpenEPD-based discrepancy mining and refinement;
- a local `speech` crate for rule-based phonemicization/realization;
- StyleTTS2/Piper-adjacent speech plumbing that is still experimental.

This project is moving quickly. Some old command help may still refer to the earlier masked-phone prototype. The active model path is now sequence-to-sequence translation.

---

## Workspace layout

```text
pronlex-core   shared vocabulary and special token IDs
pronlex-data   CMUdict parsing, IPA phonemicization, splits, collation
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

That trains the default model at `models/cmudict-v0` using data in `runs/cmudict-v0`. If the prepared data is missing, training prepares it. If `data/cmudict.dict` is missing, training downloads CMUdict first.

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
    --input data/cmudict.dict \
    --out runs/cmudict-v0
```

`prepare` writes:

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
{"base_word":"farkle","phonemes":"ˈfɑɹ.kəl"}
```

Splits are deterministic by base word. Alternate source entries for the same base word are collapsed before splitting, so a word cannot leak across train/validation/test.

### Train

```sh
just train
```

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

Tasks:

| Task | Meaning |
|------|---------|
| `g2p` | spelling/graphemes to broad IPA phonemes |
| `p2g` | broad IPA phonemes to spelling/graphemes |
| `both` | train both directions |

Training resumes automatically when a previous `train_state.json` and checkpoints are present.

### Predict

```sh
just infer "farkle"
just infer --cpu "ˈfɑɹ.kəl"
```

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

Metrics currently include:

- loss;
- exact-match accuracy;
- token accuracy.

Use exact match for strict whole-output correctness and token accuracy for “mostly got the pronunciation/spelling right.”

### Refine from discrepancies

```sh
just refine
```

or:

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

Refinement runs the model over held-out splits, looks up reference pronunciations in OpenEPD, normalizes them through the `speech` layer, compares predictions with gold targets, writes substantive mismatches to `discrepancies.jsonl`, and fine-tunes from the source model weights using those hard examples.

Example discrepancy:

```text
word : zweig
gold : ˈzwaɪɡ
pred : ˈzweɪɡ
```

Some discrepancies are regular patterns worth training. Others are sight-word exceptions and probably belong in an override table rather than in the productive model.

### Rule-based speech helpers

```sh
just phonemes "hello world"
just phones "hello world"
just speak "hello world"
```

These commands exercise the local speech pipeline directly.

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

TBD
