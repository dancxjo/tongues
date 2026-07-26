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

The harness also requires the pinned Glow-TTS LJSpeech config/checkpoint and
the paired MultiBand-MelGAN package. It records projected IDs, encoder
statistics, durations, monotonic alignment, a runtime-independent fixed latent,
all 12 reverse-flow block probes, final mel probes, standardized-mel probes,
and a native waveform. It separately covers seeded reproducibility and the
short/ordinary/long/repeated/punctuation input matrix. SC-GlowTTS is not
counted as an available conformance artifact: the historical official registry
left both the VCTK acoustic model and its paired vocoder license fields empty.

`fairseq-mms-vits-conformance.json` pins the original English MMS checkpoint,
the exact Fairseq blank-insertion and vocabulary-filtering behavior for
English, pre-romanized Amharic, and native-script Thai, plus a seeded waveform
probe produced by the upstream Fairseq runtime. Regenerate the probe with
`scripts/fairseq-mms-reference.py`; the script records its Fairseq revision and
Torch version so reference drift is visible. The production adapter is native
Rust and never invokes Python.
