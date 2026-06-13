# Interpretation

`interpretation` is an utterance-level streaming ASR model family. It prepares
LibriSpeech FLAC/transcript pairs into durable log-Mel features, enriches each
transcript with `seams` sentence boundaries, and phonemicizes each detected
sentence with the `speaking` crate.

## Prepare

```sh
just interpretation prepare \
  --subset mini \
  --out datasets/interpretation/mini-v0
```

For the larger baseline:

```sh
just interpretation prepare \
  --subset train-clean-100 \
  --out datasets/interpretation/train-clean-100-v0
```

The prepare step downloads the selected OpenSLR archive, extracts transcripts
and FLAC audio, writes `features/*.mel.bin` through `.part` files, and then
writes `train.jsonl`, `valid.jsonl`, `test.jsonl`, `vocab.json`,
`phoneme_vocab.json`, `phone_vocab.json`, `word_vocab.json`,
`dataset_config.json`, and `README.md`.

Prepare is restartable. Completed Mel files are validated and reused, extraction
is guarded by `.extract-complete`, and `utterances.jsonl` is flushed after each
utterance so an interrupted run can rebuild final splits without starting over.

Rows include utterance metadata, the Mel feature path, transcript text, sentence
spans, approximate frame spans, boundary labels, rendered phonemes/phones, and
serialized phonemicizer output. Each row also includes `repair_examples`:
deterministic synthetic mishears such as homophone substitutions and dropped
middle words, paired with the corrected sentence and `<boundary:repair>`.
Rows also include word-level supervision and deterministic masked-word cloze
examples for predicting previous/current/next words and reconstructing a hidden
word plus phonemes.

Each sentence is also enriched with the built-in link-grammar parser output.
`syntax` contains parser words, POS-like tags, typed link labels, linked word
indices, relative head offsets, parse rank/cost, confidence weight, phrase-ish
boundary hints, and the raw serializable syntax analysis. Training adds low
weight auxiliary heads for POS, link label, head offset, parse acceptability,
and phrase boundary labels; failed parses leave syntax labels padded so syntax
loss is skipped for that sentence.

## Train

```sh
just interpretation train \
  --data datasets/interpretation/mini-v0 \
  --out models/interpretation/mini-v0 \
  --epochs 20 \
  --batch-size 8
```

Training prints checkpoint paths before starting:

| File | Purpose |
|---|---|
| `model.bin` | Best model weights. |
| `model-epoch-N.bin` | Per-epoch checkpoints. |
| `train_state.json` | Resume state. |
| `model_config.json` | Architecture config. |
| `train_config.json` | Training config and loss weights. |
| `vocab.json`, `phoneme_vocab.json`, `phone_vocab.json`, `word_vocab.json` | Self-contained decode vocabularies. |
| `manifest.json` | Generic model-family artifact metadata. |

The v1 model exposes streaming CTC-style greedy collapse, a sentence-boundary
head, a repair class, phoneme and phone heads, word-context heads, masked-word
cloze heads, and masked Mel reconstruction. The forward objectives train audio
to orthographic transcript characters, phonemes, phones, and previous/current/
next word labels. The masked cloze objective hides a word span and trains the
model to recover the word and its phonemes. The backward/audio objective masks
deterministic spans of input Mel frames and asks the model to reconstruct the
original Mel features, which is the first `back to audio` training path before
adding a vocoder.

The event head learns three frame labels: continue, sentence emit, and repair.
V1 trains word heads with Burn's native `CTCLoss`, using CTC-style blank tokens,
compact target word sequences, input/target lengths, log-probabilities, and
greedy-collapse decoding. The artifact architecture records this as
`streaming-mel-native-ctc`.

## Eval

```sh
just interpretation eval \
  --model models/interpretation/mini-v0 \
  --data datasets/interpretation/mini-v0 \
  --split test
```

Evaluation reports loss, token error rate, word error rate, boundary F1, repair
F1, phoneme token error rate, phone token error rate, and masked-audio
reconstruction MSE. It also reports previous/current/next word accuracy,
masked-word accuracy, and masked-word phoneme token error rate.

## Stream

```sh
just interpretation stream \
  --model models/interpretation/mini-v0 \
  --wav /path/to/mono-16khz.wav
```

The v1 stream command accepts a WAV file for repeatable smoke testing and emits
JSON containing the partial transcript and final sentence events with attached
phonemic supervision. It can also carry repair events using the same
`<boundary:repair>` semantics as `sentence-parser`, plus previous/current/next
word predictions with phoneme-side predictions. Live microphone chunking can
reuse the same `stream_from_samples` library path.
