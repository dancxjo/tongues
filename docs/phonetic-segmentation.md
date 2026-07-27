# Phonetic segmentation

Tongues exposes deterministic, evidence-bound phone and phoneme segmentation:

```text
tongues phonetic-segment --wav input.wav --recipe alignment.json --out segments.json
```

The command reads a versioned `AlignmentRecipe`, loads the waveform, runs the
`tongues-audio` engine, and atomically writes a versioned
`PhoneticSegmentArtifact`. The typed pipeline catalog also exposes the
`phonetic_segmentation` node. Its graph component reports `unavailable` until a
model/runtime alignment adapter is registered; the catalog does not claim that
the deterministic core alone can infer boundaries.

## Transfer from Listenbury

The behavioral reference is Listenbury's `src/audio/lattice/sources.rs` and
`src/audio/lattice/engine.rs`. The transferable behavior is:

- sources produce competing timed hypotheses with independent confidence and
  optional evidence signals;
- evidence is normalized and combined by a deterministic weighted average;
- external evidence is blended with source confidence at a 3:1 ratio;
- the highest-scoring candidate wins, while weak evidence remains revisable.

Tongues adapts that behavior to expected phone/phoneme indices. Indexing keeps
repeated symbols distinct and allows multilingual inventories without a
hard-coded English symbol set. Candidate ties are resolved by confidence,
start frame, end frame, and stable source identity.

The Tongues contract is stricter than general hypothesis fusion: no source
interval means no segment interval. The engine does not divide an utterance
evenly among expected phones and does not label unaligned gaps as silence.

## Input contract

The WAV supplies interleaved PCM, sample rate, channel count, and the original
frame timebase. A recipe supplies:

- `audio_artifact_id` and optional `expected_audio_sha256`;
- optional transcript text;
- an ordered `expected` sequence, where every entry names its symbol, segment
  kind, language tag, inventory ID and membership, and pronunciation/G2P
  source;
- alignment candidates with an expected-sequence index, half-open frame
  interval, confidence, adapter identity, and optional evidence;
- graph ID/revision, recipe ID, execution-record ID, runtime, and runtime
  version.

`expected_audio_sha256` is checked against a deterministic digest of the loaded
PCM and geometry. A mismatch or empty waveform fails without an artifact.
Model-specific forced aligners, CTC decoders, DTW sources, or manual correction
tools implement `PhoneticAlignmentSource`; the core has no model dependency.

## Output and timebase

Intervals use half-open original-audio frames:

```text
[start_frame, end_frame)
seconds = frame / sample_rate_hz
```

Accepted intervals are monotonic and non-overlapping. Each segment carries its
symbol, kind, expected index, confidence, pronunciation provenance, language
and inventory IDs, and alignment-source identity. The artifact records the
audio digest and geometry, graph/execution context, source artifacts, algorithm
version, and original expected sequence.

Readiness is:

- `ready`: every expected segment has accepted evidence and no issues;
- `partial`: at least one interval is accepted, but another is clipped,
  unknown, weak, missing, or inconsistent;
- `unsupported`: no trustworthy interval can be emitted.

Unknown symbols, low-confidence candidates, missing candidates, and
non-monotonic candidates retain a segment row with no interval. A candidate
that extends past the loaded audio may be clipped explicitly. Gaps are emitted
as `unaligned_not_assumed_silence`; callers need separate VAD or acoustic
evidence before calling them silence or non-speech.

## Reference tolerance

`fixtures/phonetic-segmentation/listenbury-fusion-v1.json` records a
Listenbury-style competing-candidate case. Tongues must select the same
candidate and match fused confidence within `0.000001`. Other tests cover
repeated phones, unknown inventory members, missing and low-confidence
evidence, overlap, clipping, empty audio, and audio checksum mismatch.
