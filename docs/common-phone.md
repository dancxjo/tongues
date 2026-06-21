# Common Phone

`common-phone` is a focused model family for learning monotonic mappings from
mechanically measured audio frames to phones and phonetic feature ribbons.

```text
audio
  -> mechanical compact acoustic frames
  -> shared frame encoder
  -> CTC heads: phones, phonemes, feature axes
```

The canonical spelling is `common-phone`:

```sh
just common-phone prepare --input /path/to/common-phone --out datasets/common-phone/v0
just common-phone train --data datasets/common-phone/v0 --out models/common-phone/v0
just common-phone eval --data datasets/common-phone/v0 --model models/common-phone/v0 --split valid
```

The CLI also accepts `commonphone` as an alias, but docs and recipes use
`common-phone`.

## Dataset Layout

V0 ingests a local Common Phone checkout/export. The first supported layout is
an input directory containing one of:

- `metadata.jsonl`
- `metadata.csv`
- `metadata.tsv`

Each row must provide an audio path and phone targets:

```json
{
  "utterance_id": "eng-0001",
  "lang": "eng",
  "speaker_id": "speaker-a",
  "audio_path": "audio/eng-0001.wav",
  "phones": ["t", "ɪ", "p"],
  "phonemes": ["t", "ɪ", "p"]
}
```

`audio_path`, `path`, or `wav` may be used for the audio field. `phones` and
`phonemes` may be JSON arrays or whitespace-separated strings. WAV is the first
implemented audio decoder.

Useful smoke run:

```sh
just common-phone prepare \
  --input fixtures/common-phone-mini \
  --out datasets/common-phone/v0 \
  --lang eng \
  --max-utterances 16
```

## Compact Frames

Prepared features are written to `features/*.acf.bin`. These are compact
acoustic frames, not raw audio and not learned EnCodec-style tokens. Each file
contains a little-endian header:

```text
frames: u32
bins: u32
```

followed by `frames * bins` `f32` values. The default vector follows the
interpretation family’s compact-frame shape:

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

That is 167 floats per frame at the default 16 kHz sample rate and 100 Hz frame
rate. Feature files are written through `.part` files and reused after header
validation on resume.

## Targets

Prepare preserves phone targets and uses phoneme targets when available. If
phonemes are omitted, phones are reused as phonemes for v0.

Phone symbols are also mapped to ordered feature-axis targets:

- `manner`
- `place`
- `voicing`
- `syllabic`
- `height`
- `backness`
- `rounding`

The mapping is deliberately small and explicit. Unknown symbols map to `<UNK>`
axis labels and are counted in the generated dataset README and eval report.

Prepared output:

```text
datasets/common-phone/v0/
  train.jsonl
  valid.jsonl
  test.jsonl
  vocab.json
  phone_vocab.json
  phoneme_vocab.json
  feature_axis_vocabs.json
  dataset_config.json
  README.md
  features/*.acf.bin
```

Rows include `row_source`, `utterance_id`, `lang`, optional speaker/variety,
the original audio path, feature path, sample rate, frame rate, duration,
phones, phonemes, generated feature targets, and raw source metadata.

## Train

```sh
just common-phone train \
  --data datasets/common-phone/v0 \
  --out models/common-phone/v0 \
  --epochs 3 \
  --batch-size 8
```

Training prints checkpoint behavior before work starts:

```text
train_state.json
model-epoch-N.bin
model.bin
```

The v0 artifact records architecture
`common-phone-compact-frame-ctc-v0` and the intended CTC heads for phones,
phonemes, and feature axes. It is a CPU-friendly scaffold while the data path is
stabilized; a fuller Burn temporal encoder can replace the baseline artifact
without changing the prepared data format.

## Eval And Inspect

```sh
just common-phone eval \
  --data datasets/common-phone/v0 \
  --model models/common-phone/v0 \
  --split valid

just common-phone show-row \
  --data datasets/common-phone/v0 \
  --index 0
```

Eval reports phone token error rate, phoneme token error rate, per-axis feature
token error rate, aggregate feature token error rate, greedy decode samples,
unknown phone counts, and split-level language distribution.

`show-row` prints the utterance id, language, phone target, feature-axis
targets, compact feature dimensions, first frame values, and summary stats.

## Difference From Interpretation

`interpretation` is a broader ASR and utterance-understanding scaffold with
transcript, sentence repair, syntax, word-context, seq-style correction, and
masked-audio objectives.

`common-phone` v0 is deliberately narrower:

```text
mechanical audio frames -> phones
mechanical audio frames -> phonemes
mechanical audio frames -> feature axes
```

It does not train a custom audio tokenizer, add a vocoder, require forced
alignment, or try to solve all IPA feature mappings in the first version.
