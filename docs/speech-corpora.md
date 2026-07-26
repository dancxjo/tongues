# Native Speech Corpora

Tongues ingests LJSpeech, VCTK, and generic metadata without Python. The
normalized records and batching interfaces live in `tongues-data`; format names
stop at the ingestion boundary.

## Prepare LJSpeech

```bash
just speech-corpus prepare \
  --format ljspeech \
  --input datasets/raw/LJSpeech-1.1 \
  --out datasets/tts/ljspeech-v1
```

The reader requires the standard `metadata.csv` rows
`id|raw text|normalized text` and `wavs/<id>.wav`.

## Prepare VCTK

```bash
just speech-corpus prepare \
  --format vctk \
  --input datasets/raw/VCTK-Corpus-0.92 \
  --out datasets/tts/vctk-v1 \
  --language en-GB \
  --split-by-speaker
```

The reader discovers the VCTK `txt/` tree and WAV or FLAC audio recursively. It
maps files such as `p330_001_mic1.flac` to transcript `p330_001.txt`, preferring
mic 1 over mic 2 when both are present.

## Generic metadata

Generic JSONL accepts `id`, `audio_path`, and `text`, plus optional
`normalized_text`, `speaker`, `language`, `emotion`, and `style`. Delimited
metadata accepts `audio|text` or
`id|audio|text|speaker|language|emotion|style`.

```bash
just speech-corpus prepare \
  --format generic \
  --input datasets/raw/custom \
  --metadata metadata.jsonl \
  --out datasets/tts/custom-v1
```

## Output and recovery

Preparation writes:

- `manifest.jsonl` and deterministic `train.jsonl`, `valid.jsonl`, `test.jsonl`;
- `batches.json` with seeded, length-aware record-id batches;
- `validation.json` with source file, line, severity, and reason;
- `statistics.json`, `dataset_config.json`, and `README.md`.

WAV and FLAC headers are inspected natively for duration statistics. Duplicate
IDs/audio paths, malformed rows, missing files, and invalid audio fail
preparation after the durable validation report is written. Every final output
is flushed through a `.part` file and renamed only after a successful write.

The library also exposes:

- `collate_speech_batch` for padded token/acoustic tensors and masks;
- `write_cached_speech_features` / `read_cached_speech_features` for atomic,
  configuration-fingerprinted text, phoneme, and acoustic feature caches;
- a normalization callback at ingestion so dataset-specific cleanup does not
  leak into model APIs.
