# Interpretation

`interpretation` is an utterance-level ASR and interpretation scaffold. It
prepares LibriSpeech FLAC/transcript pairs into durable acoustic feature files,
enriches each transcript with `seams` sentence boundaries, and phonemicizes each
detected sentence with the `speaking` crate.

The intended shape is a shared audio encoder with several cheap supervised
heads:

```text
compact audio frames
  -> shared frame encoder
       -> streaming CTC-ish heads: transcript chars, phones, phonemes, words
       -> frame heads: sentence/repair boundary, syntax labels, masked audio
       -> lightweight seq-style transcript head for after-utterance correction
```

The CTC-style heads are for monotonic streaming and alignment. The seq-style
head is for post-utterance correction and richer interpretation once more
context is available. V1 keeps this deliberately small: the encoder is a linear
projection plus `tanh`/dropout, and the "seq2seq" transcript head is an
auxiliary per-position decoder head rather than a full autoregressive
Transformer decoder.

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
and FLAC audio, writes `features/*.mel.bin` through `.part` files, recases and
repunctuates transcripts with the default multilingual
`ggml-large-v3-turbo.bin` Whisper ASR model, and then writes
`train.jsonl`, `valid.jsonl`, `test.jsonl`, `vocab.json`,
`phoneme_vocab.json`, `phone_vocab.json`, `word_vocab.json`,
`syntax_pos_vocab.json`, `syntax_link_vocab.json`,
`syntax_head_offset_vocab.json`, `dataset_config.json`, and `README.md`.

By default, prepare also looks for the prepared Wiktionary dataset at
`datasets/wiktionary/enwiktionary-2026-06-01-v0`. When that dataset exists, it
reads `patterns.jsonl` for `{{audio}}` rows, joins them to `phonemes.jsonl` and
`phones.jsonl`, computes the direct Commons upload URL from the audio filename,
downloads a bounded set of pronunciation audio files, decodes them, writes
matching `features/*.mel.bin`, and adds those single-word pronunciation rows
beside the LibriSpeech sentence rows. Use `--no-wiktionary-audio` to skip this,
`--wiktionary-audio-data` to point at a different prepared Wiktionary dataset,
`--max-wiktionary-audio` to change the import limit, or
`--no-download-wiktionary-audio` to reuse only already-downloaded audio.

Despite the `.mel.bin` suffix, new feature files contain the compact acoustic
vector by default:

```text
[log_mel_80,
 delta_mel_80,
 energy,
 vad,
 zcr,
 spectral_centroid,
 spectral_flux,
 f0,
 voiced_prob]
```

That is 167 floats per frame with the default 80 Mel bins. The cheap scalar
features give the model rough loudness, silence/voicing, pitch, and timbre cues
without adding an expensive frontend. Existing legacy 80-bin log-Mel files can
be upgraded in place during recovery: prepare validates the file header, derives
delta/Mel-side scalar approximations from the saved log-Mel rows, writes the
upgraded feature file through a `.part` path, and then reuses it.

Whisper transcript refinement is enabled by default to turn the original
all-caps LibriSpeech text into sentence-like text. The default transcript model
is Whisper large-v3-turbo multilingual. Each Whisper transcript is compared
against the original transcript after case and punctuation are stripped;
obvious digit tokens and spelled-out number words are treated as comparable
numeric tokens so rows are not rejected only because Whisper wrote `24` where
LibriSpeech wrote `TWENTY FOUR`. Utterances above `--max-whisper-wer` are
omitted with a progress warning that shows the source and Whisper transcripts
instead of silently poisoning the dataset. Use `--no-whisper-transcripts` to
keep the original LibriSpeech text, `--whisper-model` to point at a specific
ggml model, or `--max-whisper-wer` to adjust the divergence threshold. The
default threshold is `0.70`.

Prepare is restartable. Completed feature files are validated and reused,
extraction is guarded by `.extract-complete`, and `utterances.jsonl` is flushed
after each utterance so an interrupted run can rebuild final splits without
starting over. Final split JSONL files are written through `.part` files and
renamed after a successful flush. When Whisper transcript refinement is enabled,
existing `utterances.jsonl` rows are rebuilt so older all-caps prepared data is
refreshed while reusable feature files stay in place.

Rows include utterance metadata, the acoustic feature path, transcript text,
sentence spans, approximate frame spans, boundary labels, rendered
phonemes/phones, and serialized phonemicizer output. Each row also includes
`repair_examples`:
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

For English, syntax analysis uses the upstream `link-parser` command when it is
available on `PATH`, preserving raw Link Grammar connector labels under
`raw_link_grammar_parses` and projecting known connector families into the
existing simplified labels. Set `LINK_GRAMMAR_PARSER` or
`LINK_GRAMMAR_EN_DICTIONARY` to override the executable or dictionary path. Set
`TONGUES_LINK_GRAMMAR_BACKEND=heuristic` to force the legacy heuristic fallback.

Vocabulary files are built from the recovered split rows. `vocab.json` is the
character/CTC vocabulary. `phoneme_vocab.json` is built from phonemicizer token
IDs, not whole rendered IPA strings. `phone_vocab.json` is built from phone
labels. `word_vocab.json` lowercases surface words, maps numeric forms to
`<NUM>`, keeps `<WORD_BLANK>` and `<WORD_UNK>`, and caps rare words so the word
heads stay bounded. Syntax vocabularies reserve padding/default labels and then
add parser labels from the dataset.

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
| `optim-epoch-N.bin` | Per-epoch optimizer checkpoints. |
| `model_config.json` | Architecture config. |
| `train_config.json` | Training config and loss weights. |
| `vocab.json`, `phoneme_vocab.json`, `phone_vocab.json`, `word_vocab.json`, `syntax_*_vocab.json` | Self-contained decode vocabularies. |
| `manifest.json` | Generic model-family artifact metadata. |

Training reads the first prepared feature file to set `model_config.mel_bins`
and `train_config.input_feature_bins`. A compact-feature dataset should produce
`167` there. If the model directory is stale, remove old model metadata and
checkpoints before training so the new model cannot resume with an incompatible
80-bin input layer.

If a resumed optimizer state drives an epoch to `NaN`, training preserves the
last best model and train state, quarantines the optimizer checkpoint as
`optim-epoch-N.nan.bin`, and leaves the next run to resume from the model
checkpoint with a fresh optimizer and reduced resume learning rate.

The v1 model exposes streaming CTC-style greedy collapse, a seq-style transcript
head, a sentence-boundary head, a repair class, phoneme and phone heads,
weakly supervised phonetic-feature CTC heads, word-context heads, masked-word
cloze heads, syntax heads, and masked feature reconstruction. The forward
objectives train audio to orthographic transcript characters, phonemes, phones,
ordered feature axes, and previous/current/next word labels. The
seq-style transcript objective trains a bounded after-utterance character target
over the first `max_seq2seq_tokens` positions. The masked cloze objective hides
a word span and trains the model to recover the word and its phonemes. The
backward/audio objective masks deterministic spans of input acoustic frames and
asks the model to reconstruct the original compact features, which is the first
`back to audio` training path before adding a vocoder.

The event head learns three frame labels: continue, sentence emit, and repair.
V1 trains the phoneme, phone, word, masked-word, and masked-word-phoneme heads
with Burn's native `CTCLoss`, using CTC-style blank tokens, compact target
sequences, input/target lengths, log-probabilities, and greedy-collapse
decoding. The phoneme and phone heads also keep a proportional frame-label CE
auxiliary loss as a stabilizer, but their ordered sequence objective is CTC so
the alignment can emerge from the audio. The feature heads train separate CTC
targets for place, manner, voicing, syllabic, vowel height, vowel backness, and
rounding. They only receive ordered symbolic supervision derived from the phone
sequence, not forced-alignment spans, so temporal feature ribbons can emerge
from the audio. The artifact architecture records this as
`streaming-mel-native-ctc`; current prepared inputs are compact acoustic
features rather than Mel-only frames.

## Eval

```sh
just interpretation eval \
  --model models/interpretation/mini-v0 \
  --data datasets/interpretation/mini-v0 \
  --split test
```

Evaluation reports loss, CTC/frame token error rate, word error rate, seq-style
token error rate, boundary F1, repair F1, phoneme token error rate, phone token
error rate, and masked-audio reconstruction MSE. It also reports
previous/current/next word accuracy, masked-word accuracy, and masked-word
phoneme token error rate.

## Stream

```sh
just interpretation stream \
  --model models/interpretation/mini-v0 \
  --wav /path/to/mono-16khz.wav
```

The v1 stream command accepts a WAV file for repeatable smoke testing and emits
JSON containing both `partial_transcript` from the streaming CTC-style transcript
head and `seq2seq_transcript` from the after-utterance head. It also emits final
sentence events with attached phonemic supervision, repair events using the same
`<boundary:repair>` semantics as `sentence-parser`, and previous/current/next
word predictions with phoneme-side predictions. Live microphone chunking can
reuse the same `stream_from_samples` library path.
