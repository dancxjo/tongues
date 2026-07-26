# Speech conformance fixtures

`coqui-v0.6.1-tokenization.json` is a tiny, deterministic fixture derived from
the redistributable Coqui TTS v0.6.1 tokenizer output. It contains no model
weights or audio. The full conformance harness regenerates these IDs and
symbols inside a pinned reference container and fails if they drift. It also
generates the uncommitted YourTTS golden embeddings and waveform probes
described in [`docs/yourtts-conformance.md`](../../docs/yourtts-conformance.md).

Large upstream model archives remain outside the repository. Their SHA-256
digests are pinned in `scripts/coqui-reference.py`, and
`scripts/speech-conformance.sh` fails explicitly when any required artifact is
missing.
