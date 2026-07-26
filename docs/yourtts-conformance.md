# Published YourTTS conformance

The opt-in speech conformance run treats the cataloged multilingual YourTTS
archive as one published artifact bundle. It does not substitute synthetic
weights, skip unavailable cases, or invoke Python from the native runtime.

`scripts/speech-conformance.sh` verifies the archive and every extracted member
by SHA-256, runs the upstream graph in the container pinned to Coqui TTS
revision `0cf3265a4686d7e856bd472cdaf1572d61cab2b8`, and then runs the same
fixtures through native Burn inference. The upstream process copies
`tests/inputs/example_1.wav` from that pinned source tree, resamples it to the
encoder's 16 kHz input rate with the pinned librosa 0.8.1 runtime, applies the
published -27 dB RMS normalization, and writes a deterministic PCM16 WAV to the
evidence directory. The source SHA-256 is
`6563390fa42121eeeab15f49fa91fd26afe000022bfdaaa882f06224ad549599`;
the exact 16 kHz fixture SHA-256 is
`d40c065b740317f9007ddca22ec076302ebb302a17236f0c20e7d92c21ea6629`.
Neither that reference WAV, the model weights, nor generated waveforms are
committed to this repository.

The shared matrix is:

| Case | Model language | Speaker source | Covered path |
|---|---|---|---|
| `named-male-en` | `en` | published `male-en-2` enrollment | stored upstream d-vector through full synthesis |
| `named-female-fr` | `fr-fr` | published `female-en-5` enrollment | second speaker and learned language row through full synthesis |
| `reference-ljspeech-en` | `en` | pinned real reference WAV | WAV preprocessing, native speaker encoder, and full synthesis |

This synthesizes two checkpoint-declared languages and three distinct reference
selections. The artifact's third declared language, `pt-br`, remains enumerated
and shape-validated, but is not mislabeled as end-to-end conformance because
Tongues does not yet have a registered `pt-BR` linguistic variety. The run also
selects two published clips for `male-en-2` and one for `female-en-5`,
recomputes their cosine scores through the neutral embedding API, and requires
the same-speaker score to exceed the different-speaker score.

## Numerical contract

The conformance test disables acoustic and duration noise in both runtimes.
Language IDs, checkpoint token IDs, embedding dimensions, sample rate, channel
count, and waveform sample count must match exactly.

Floating-point comparisons use these explicit absolute tolerances:

| Measurement | Tolerance |
|---|---:|
| Each of 512 upstream speaker-embedding values | `3e-4` |
| `1 - cosine(native, upstream)` for the reference-WAV embedding | `1e-4` |
| Waveform RMS | `5e-4` |
| Deterministic waveform probes spanning each utterance | `2e-3` |
| Stored-embedding same/different cosine fixtures | `1e-6` |

The full 512-value upstream embedding for each case is written to
`target/speech-conformance/coqui-reference.json`; waveform evidence records
RMS, sample count, extrema, and ten probes spanning the complete output. These
are semantic parity checks, not perceptual-quality claims.

Run:

```sh
scripts/speech-conformance.sh
```

The harness fails before inference if any published artifact is absent or has
the wrong digest. Successful evidence is committed atomically under
`target/speech-conformance/`.
