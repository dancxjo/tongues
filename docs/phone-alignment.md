# Evidence-preserving phone alignment

Tongues schema v2 represents phone timing as an inspectable hypothesis set. It
extends the schema-v1 phonetic-segmentation timebase and timeline attachment;
it does not introduce a competing timestamp format.

```text
tongues phone-align \
  --wav utterance.wav \
  --request alignment-request-v2.json \
  --posteriors common-phone-posteriors-v1.json \
  --out alignment-v2.json
```

For a locally trained native Common Phone model, replace `--posteriors` with
`--common-phone-model MODEL_DIR`. The Common Phone adapter returns every
frame/class probability before greedy CTC collapse. The shared alignment engine
then runs a bounded CTC trellis. A greedy CTC spike is never treated as a phone
boundary.

The server exposes the same engine at `POST /api/phone-align`. The JSON body is
`{"audio": AudioBuffer, "request": PhoneAlignmentRequest, "posteriors":
CtcPosteriorMatrix}`. Invalid or incompatible evidence returns HTTP 422 with
the backend identity and a failed readiness reason. The body and the request's
frames/symbols/lattice dimensions are independently bounded.

## Semantic separations

- Segmentation partitions audio. Alignment relates acoustic evidence to
  linguistic or recognized units.
- Forced alignment constrains supplied content. Recognition-derived alignment
  proposes symbols from acoustic decoding. Neither is synthesis timing,
  imported annotation, or manual correction.
- A phone is a realized surface hypothesis. A phoneme is a linguistic analysis
  linked through an explicit many-to-many projection.
- A boundary has an estimate, supported range, timebase, method, timing
  authority, and lifecycle. Scalar confidence is not a boundary range.

`AlignmentMode` records `unconstrained`, `transcript_constrained`,
`pronunciation_constrained`, `synthesis_known`, `imported`, or `hybrid`.
Every unit separately records `TimingAuthority`, so a hybrid artifact cannot
relabel a TTS plan as acoustic observation.

## Request, paths, and limits

`PhoneAlignmentRequest` schema 2 carries audio identity/checksum/channel,
selected regions, preprocessing, an optional multilingual transcript lattice,
pronunciation alternatives, structural links, hints, corrections, resource
limits, and graph/run provenance.

Before allocating a trellis, the engine checks waveform geometry, checksum,
posterior shape and probability mass, sample rate, blank index, language,
inventory, symbol vocabulary, and configured frame/state/total-lattice-cell
limits. The cell-product limit bounds actual trellis memory even when each
dimension is individually valid. Unsupported paths abstain.
No evenly divided fallback exists. `AlignmentCancellation` is checked between
paths and at bounded trellis intervals so long inputs can stop without waiting
for the whole matrix.

Each `AlignmentHypothesis` retains stable identity, rank, lifecycle,
selection/pruning reason, normalized within-lattice posterior, and units.
Units distinguish match, insertion, deletion, substitution, silence,
non-speech, and unknown material. Acoustic likelihood, pronunciation and
duration priors, insertion/deletion penalties, correction contribution, and
backend score remain separate fields.

CTC forward/backward occupancy supplies supported boundary ranges. Phone
presence probability is distinct from path posterior. Selection abstains when
the leading path misses the posterior or winner-margin policy. Strong competing
non-target evidence in blank regions becomes an inspectable
recognition-derived insertion; missing requested symbols remain deletion or
unaligned linguistic evidence.

## Streaming and correction

`StreamingPhoneAligner` emits `append`, `replace`, `withdraw`, and `commit`
deltas. Only the tail after the explicit commit frontier is revisable.
Committed identities and intervals remain frozen; changing them requires a
correction/repair request. Updates report evidence/frontier frames, revision
count/depth, provisional and committed counts, churn ratio, and mean
time-to-stability in frames.

`PhoneAlignmentArtifact::attach_to_timeline` stores the complete lattice as one
immutable schema-v2 `phonetic_segmentation` attachment for compatibility with
the #169/#170 surfaces. Only the selected path becomes timeline spans.
Alternatives stay inspectable in the attachment.

Tracks and WaveDeck show mode, selected path, posterior, alternatives, boundary
ranges, score breakdown, lifecycle, relation, and timing authority. WaveDeck
replay supports symbols, point boundaries, boundary ranges, and pronunciation
choices without mutating the original run.

## Schema-v1 migration

`PhoneAlignmentArtifact::migrate_v1` preserves v1 intervals, gaps, source
identity, graph context, and readiness. Because v1 did not carry posterior
uncertainty, migrated points are tagged
`legacy_v1_point_boundary_no_uncertainty`; migration never invents a range.

## Evaluation

`evaluate_alignment` reports selected path error rate, oracle top-k recall,
insertion/deletion/substitution counts, boundary mean absolute error, tolerance
accuracy, supported-interval coverage, phone-presence and path-selection Brier
scores, word boundary error/tolerance accuracy, unaligned reference units,
language, and variety.

```text
tongues phone-align ... \
  --evaluation-reference trusted-reference.json \
  --evaluation-out evaluation-report.json
```

References record annotator tolerance because phonetic boundaries are not
infinitely precise ground truth. Deterministic Rust and browser fixtures cover
competing pronunciations, abstention, repeated phones, insertions, streaming
revision/commit safety, many-to-many projection, migration, correction replay,
and metrics without requiring a model download.

`fixtures/phone-alignment/multilingual-synthetic-v1.json` is CC0-1.0 and
contains reproducible English repeated-phone, Japanese no-whitespace, and
Swahili/English code-switch cases. It is deliberately labeled synthetic: it is
a contract/metric regression corpus, not evidence of natural-speech quality.
Generate its per-language/per-variety report with one supported command:

```text
tongues phone-alignment-eval --out outputs/phone-alignment-evaluation.json
```

The graph catalog advertises `phone-alignment:ctc-lattice-v2` as ready for
supplied posteriors and declares CTC, alternatives, uncertainty, and streaming
capabilities. The legacy explicit-hint component remains unavailable for graph
execution. A Common Phone acoustic model is ready only when its model directory
is supplied; Tongues does not claim an absent model is installed.

Backends implement `PhoneAlignmentBackend` and return the same artifact.
`check_alignment_conformance` verifies schema and backend identity, stable
hypothesis/unit IDs, selected-path retention, and valid boundary supports. An
external importer or aligner remains an adapter; it does not redefine Tongues'
timing semantics.
