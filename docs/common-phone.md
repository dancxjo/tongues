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
just common-phone train --data datasets/common-phone/v0 --model models/common-phone/v0 --task frames2phones
just common-phone eval --data datasets/common-phone/v0 --model models/common-phone/v0 --split valid --task frames2phones
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
  "wav_path": "audio/eng-0001.wav",
  "phones": ["t", "ɪ", "p"],
  "phonemes": ["t", "ɪ", "p"]
}
```

`wav_path`, `audio_path`, `path`, or `wav` may be used for the audio field.
`phones` and `phonemes` may be JSON arrays or whitespace-separated strings. WAV
is the first implemented audio decoder.

The family can also download and prepare the official Common Phone 1.0 archive
from Zenodo. The archive is large, about 13 GB, and expands to the original
language-directory layout with split CSVs, WAV files, and Praat TextGrids:

```sh
cargo run --bin tongues -- common-phone fetch \
  --out data/common-phone/raw \
  --source zenodo

cargo run --bin tongues -- common-phone prepare \
  --download \
  --input data/common-phone/raw \
  --out models/common-phone/common-phone-v0 \
  --lang eng \
  --max-utterances 1000
```

`prepare --download` downloads `cp-1-0.tgz` through a `.part` file, extracts it,
then reads the extracted Common Phone layout directly. It uses each language
directory’s `train`/`dev`/`test` CSVs, `wav/` audio, and `grids/` TextGrid phone
annotations. The `dev` split is normalized to `valid`.

Useful smoke run:

```sh
just common-phone prepare \
  --input fixtures/common-phone-mini \
  --out datasets/common-phone/v0 \
  --lang eng \
  --max-utterances 16
```

## Compact Frames

Prepared features are written to `frames/*.acf.bin`. These are compact
acoustic frames, not raw audio and not learned EnCodec-style tokens. Each file
contains a little-endian header:

```text
magic: ACF0
version: u32
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
  config.json
  manifest.jsonl
  train.jsonl
  valid.jsonl
  test.jsonl
  stats.json
  vocab.json
  phone_vocab.json
  phoneme_vocab.json
  feature_bundle_vocab.json
  feature_axis_vocabs.json
  vocabs/
    phones.json
    phonemes.json
    feature_bundles.json
  dataset_config.json
  README.md
  frames/*.acf.bin
```

Rows include `row_source`, `utterance_id`, `lang`, optional speaker/variety,
the original audio path, feature path, sample rate, frame rate, duration,
phones, phonemes, generated feature targets, and raw source metadata.

## Train

```sh
just common-phone train \
  --data datasets/common-phone/v0 \
  --model models/common-phone/v0 \
  --task frames2phones \
  --epochs 3 \
  --batch-frames 12000 \
  --lr 0.0003 \
  --device cpu
```

Training prints checkpoint behavior before work starts:

```text
train_state.json
model-epoch-N.bin
model.bin
```

The v0 artifact records architecture `common-phone-compact-frame-ctc-v0`.
Training uses a small Burn frame encoder:

```text
[batch, time, frame_dim]
  -> linear + tanh + linear + tanh + dropout
  -> CTC heads for phones, phonemes, and feature bundles
```

`frames2phones` is the primary task. `frames2features`, `frames2phonemes`, and
`multitask` are also accepted.

## Eval And Inspect

```sh
just common-phone eval \
  --data datasets/common-phone/v0 \
  --model models/common-phone/v0 \
  --split valid \
  --task frames2phones

just common-phone show \
  --data datasets/common-phone/v0 \
  --index 0
```

Eval reports token error rate, edit distance, exact sequence accuracy, blank
ratio, mean prediction/target length, greedy CTC samples, unknown phone counts,
and split-level language distribution.

`show-row` prints the utterance id, language, phone target, feature-axis
targets, compact feature dimensions, first frame values, and summary stats.

For an end-to-end generated fixture smoke test:

```sh
just common-phone-smoke
```

## Listen Demo

The live demo captures microphone audio with CPAL, keeps a rolling context
window, regenerates the same mechanical compact frames used by `prepare`, runs
the trained CTC model, and greedily collapses frame predictions into phones.

List input devices:

```sh
cargo run --bin tongues -- common-phone listen-devices
```

Dry-run the microphone and frame generator without loading a model:

```sh
cargo run --bin tongues -- common-phone listen \
  --dry-run \
  --debug-frames
```

Run a phone listener:

```sh
cargo run --bin tongues -- common-phone listen \
  --model models/common-phone/common-phone-v0-phone-ctc \
  --task frames2phones \
  --device cpu \
  --sample-rate 16000 \
  --chunk-ms 100 \
  --context-ms 1500 \
  --show-phones
```

Add feature bundles and frame diagnostics:

```sh
cargo run --bin tongues -- common-phone listen \
  --model models/common-phone/common-phone-v0-phone-ctc \
  --show-phones \
  --show-features \
  --debug-frames
```

Select an input device by name substring:

```sh
cargo run --bin tongues -- common-phone listen \
  --model models/common-phone/common-phone-v0-phone-ctc \
  --input-device "Scarlett"
```

The listener recomputes the full rolling window for v0. Silence is gated with a
simple RMS/VAD threshold so the terminal does not constantly print noise. Rough
phones-to-orthography text is not required for v0; `--phones2orth` is accepted
as a future hook but is not wired yet.

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
