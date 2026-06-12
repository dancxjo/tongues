# Speech Manifold

`speech-manifold` is an experimental multimodal model family for speech representation work.

It builds a dataset from the embedded OpenEPD corpus. Each example records spelling, broad IPA, narrow phones, stress, syllable structure, placeholder acoustic frames, source labels, and optional sampled audio provenance.

## Prepare

```sh
cargo run --release -- speech-manifold prepare \
    --out datasets/speech-manifold/openepd-synth-v0
```

Start a fresh default run by archiving the existing default dataset/model artifacts and recreating empty directories:

```sh
cargo run --bin tongues -- speech-manifold clean --all
```

Use `--data` or `--model` to archive only one side. Artifacts are moved under `archive/<run-id>/...`; pass `--run-id NAME` for a stable archive folder or `--no-create` if you do not want empty defaults recreated.

The audio stage is intentionally quota-based: it samples a small diversity of available voices/backends rather than generating a WAV for every word.

Network-backed audio fetches are conservative. The prepare step checks `robots.txt` before attempting each network audio URL. If a host disallows a path, that backend is skipped and the example falls back to local eSpeak/mock provenance. Dictionary.com and Wiktionary page URLs are recorded as reference metadata only; Dictionary.com pages are not fetched.

## External Audio Manifests

For real voice diversity, prefer permissioned local manifests over scraping. Add one or more JSONL manifests through `external_audio_manifests` in `configs/speech-manifold/default.toml` or a custom config:

```toml
external_audio_manifests = ["data/audio/wikimedia_pronunciations.jsonl"]
```

Each row must include rights metadata and a pronunciation assurance:

```json
{"word":"cat","audio_uri":"/data/audio/cat-us.ogg","broad_ipa":"kæt","source":"wikimedia-commons","license":"CC BY-SA 4.0","attribution":"Example Speaker / Wikimedia Commons","source_url":"https://commons.wikimedia.org/wiki/File:En-us-cat.ogg"}
```

Rows are accepted only when:

- `word` exists in the prepared OpenEPD-derived examples;
- `license` and `attribution` are non-empty;
- `broad_ipa` normalizes to the same IPA as OpenEPD, or `pronunciation_assurance` is one of `single-word-pronunciation`, `source-pronunciation-entry`, or `manually-verified`.

Good candidates include Wikimedia Commons/Wiktionary pronunciation audio with per-file licenses, curated classroom/dictionary recordings you have permission to use, public-domain or permissively licensed word-list recordings, and locally generated TTS audio whose model/output terms allow your use.

Sentence corpora such as Common Voice, LibriSpeech, CMU Arctic, or VoxPopuli should only be imported at the word level after segmentation/alignment and verification; the raw sentence audio does not by itself assure a specific isolated word pronunciation.

## License Notes

The source code in this repository is MIT licensed, but generated speech-manifold datasets may include or point to material with different terms. Treat prepared data directories as local artifacts and review their generated `README.md`, `dataset_config.json`, and per-row provenance before redistributing them.

See [licensing.md](licensing.md) for the source/backend table.
