# Speech pipeline graph contract

`tongues-pipeline` owns Speech Studio's durable graph, typed ports, validation,
deterministic execution plan, lifecycle, cancellation, and provenance
contracts. `/speech-dataflow.html` consumes those backend contracts; it does not
carry a separate component or port inventory.

## Backend APIs

- `GET /api/pipeline/catalog` returns node kinds, typed input/output ports,
  configuration schemas/defaults, adapters, merge semantics, and the
  components currently resolved from the ASR, diarization, interpretation,
  response-generation, and TTS registries.
- `GET /api/pipeline/starters` returns currently executable TTS,
  transcription, diarized meeting transcription, spoken interpretation, and
  live-conversation graphs. A starter is omitted if one of its required
  components is not ready.
- `POST /api/pipeline/validate` returns diagnostics keyed to graph, node, port,
  and edge IDs.
- `POST /api/pipeline/compile` returns a deterministic execution plan or HTTP
  422 with the same structured diagnostics.
- `POST /api/pipeline/migrate` upgrades a saved graph document without resolving
  its runtime component IDs.

The friendly CLI and browser are expected to call this library-owned contract;
neither surface should reproduce routing rules.

## Schema 2 document

A graph records `graph_id`, monotonically increasing `revision`, metadata,
stable node and edge IDs, explicit port endpoints, positive bounded edge
capacities, and selected sink endpoints. A node stores a backend node-kind ID,
an optional registry component ID, and configuration. Component IDs are
resolved when validation/compilation runs, so a missing or newly unavailable
model produces a node-keyed diagnostic instead of silently changing providers.

Port values distinguish streaming audio, buffered audio, text,
partial/revised/committed transcripts, language, speaker assignments, utterance
plans, control, cancellation, artifacts, and errors. Incompatible values are
never coerced. The catalog currently exposes explicit buffered-audio-to-stream
and committed-transcript-to-text adapters.

Ordinary inputs accept one edge and fan-in inputs exist only on explicit merge
nodes. Outputs may fan out over one bounded channel per edge. The transcript
merge orders equal-time values by source edge ID and source sequence.

## Validation and planning

Validation rejects missing required inputs, unconsumed required outputs,
unknown nodes or ports, duplicate IDs, incompatible types, zero-capacity
channels, implicit fan-in, unsafe nodes without graph approval, invalid cycles,
missing capabilities, unavailable/unverified components, invalid required
configuration, and disconnected selected sinks.

Compilation topologically orders otherwise-independent nodes by stable node ID,
sorts edge-derived channels by saved edge order, and records:

- bounded capacity and producer-blocking backpressure for every edge;
- the node that owns each runtime resource and channel close;
- explicit merge strategy;
- cancellation propagation to every upstream/downstream resource owner;
- graph/catalog revisions, resolved provider/model identities, and every
  input-to-output derivation.

`execute_plan` is the provider-neutral dispatch boundary. Runtime adapters
implement `NodeRunner`, while the shared engine records started, output,
completed, cancelled, and failed events with elapsed timing and derivation
data. Cancellation closes all remaining owners/channels and is covered by a
mid-run fixture.

## Migration

Schema 1 graphs migrate to schema 2 by renaming `id` to `graph_id`, moving
`name` under `metadata.name`, and replacing selected output node IDs with
explicit sink endpoints. Future schemas fail with an instruction to upgrade
Tongues. Unknown older schemas fail with a precise missing-migration error.
After structural migration, normal validation reports any component or port
that the current registry can no longer resolve.

The browser stores schema-2 JSON locally and sends old saved JSON through the
backend migration endpoint before use. WaveDeck remains the sibling evidence
editor; graph configuration and human correction do not become one mutable
authority.
