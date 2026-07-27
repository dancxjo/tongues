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
- `GET /api/pipeline/graphs` and `GET`/`PUT
  /api/pipeline/graphs/{graph_id}` list, reopen, and atomically persist graph
  documents under `data/speech-graphs`.
- `POST /api/pipeline/run` compiles the current graph and streams deterministic
  lifecycle-contract events as NDJSON. It is the browser acceptance runner;
  provider adapters remain responsible for real microphone, model, and audio
  resources at the `NodeRunner` boundary.

## Patch canvas

`/speech-dataflow.html` uses Cytoscape from a pinned jsDelivr URL for pan, zoom,
touch, node movement, selection, and edge rendering. Tongues code owns graph
meaning, typed-port compatibility, adapter explanations, diagnostics,
persistence, configuration, and accessibility.

The palette and inspector are built only from `/api/pipeline/catalog`. Required
inputs without an edge are shown as dashed ghost ports and carry the
backend diagnostic in their accessible label. Selecting an output highlights
only compatible input targets; incompatible selections name both value types
and any registered adapter path. Output fan-out creates one bounded edge per
consumer, while fan-in remains available only through backend-declared merge
nodes.

The DOM graph outline provides keyboard selection, movement, and deletion
alongside the pointer/touch canvas. Port buttons, live diagnostics, validation,
and streamed execution changes use status/live regions for screen readers.
Starter graphs can replace the document or be inserted as reusable subgraphs.
Schema 3 stores layout, faceplates, titled/colorable frames, freeform notes,
cable routes/reroute points, emphasis, opacity, and collapsed-subpatch state
in versioned `presentation` metadata. None changes runtime routing. Disabled
nodes and structurally bypassed nodes
remain visible in the saved document but do not enter the execution plan;
their old edges are removed explicitly rather than silently rerouted.

Configuration-backed sources are catalog contracts too. Text sources expose a
streaming `out(text)` port; the port's `streaming` flag describes incremental
delivery, while its `many` cardinality continues to mean that the output may
fan out to multiple graph edges. `text_source` is the inline form: it exposes a
required multiline `text` value and emits that value as one event on the
stream. `text_file` reads a workspace-relative UTF-8 file incrementally,
preserving line endings. `text_url` incrementally reads a public HTTP(S) UTF-8
response, follows only revalidated redirects, and enforces timeout and bounded
size limits. Source failures produce a failed lifecycle event rather than a
fabricated output.

`audio_file` likewise requires a non-empty `path`. Applying source
configuration mutates only the node configuration, so saved edges, node
duplication, graph duplication, and JSON persistence keep their existing
relationships and values. Live microphone and control sources remain driven
by runtime events rather than pretending that live input is saved config.

The friendly CLI and browser are expected to call this library-owned contract;
neither surface should reproduce routing rules.

## Schema 3 document

A graph records `graph_id`, monotonically increasing `revision`, metadata,
stable node and edge IDs, explicit port endpoints, positive bounded edge
capacities, and selected sink endpoints. A node stores a backend node-kind ID,
an optional registry component ID, and configuration. Component IDs are
resolved when validation/compilation runs, so a missing or newly unavailable
model produces a node-keyed diagnostic instead of silently changing providers.

Port values distinguish streaming audio, buffered audio, text,
partial/revised/committed transcripts, language, speaker assignments, utterance
plans, control, cancellation, artifacts, and errors. Incompatible values are
never coerced. Streaming delivery is an independent port contract because a
stream carries a bounded sequence of typed values; it is not the same as edge
cardinality. The catalog currently exposes explicit buffered-audio-to-stream
and committed-transcript-to-text adapters.

Ordinary inputs accept one edge and fan-in inputs exist only on explicit merge
nodes. Outputs may fan out directly over one bounded channel per edge. The
explicit transcript splitter is a real pass-through node for cases where a
named branch improves readability; it remains distinct from direct fan-out,
the transcript merge, and type adapters. The transcript merge orders
equal-time values by source edge ID and source sequence.

## Frames, notes, cables, and subpatches

Frames and notes are presentation-only. Moving a frame can deliberately move
its listed nodes, but framing never creates an execution boundary. Cable
reroute points are coordinates on an existing edge, never runtime nodes.
Straight, curved, and orthogonal routes preserve the same endpoints and
capacity. Global opacity is bounded at 0.1 so connections remain discoverable.

A subpatch is created only from an explicit selection after reviewing every
edge and selected sink that crosses its boundary. Exposed ports map to real
backend-typed internal endpoints. Runtime nodes and edges remain top-level, so
collapsing, drilling, or reopening cannot change the compiled plan. Breadcrumb
links use `?subpatch={id}&node={id}` and work with pointer, keyboard, browser
history, and direct URLs.

Schema 3 uses **embedded copies**, not live linked definitions. Copying or
duplicating a complete subpatch mints fresh runtime IDs, a fresh subpatch ID,
and a fresh `definition_id`; it retains `definition_version` as provenance.
The copies update independently. Linked definitions require a future schema
with an explicit update policy. Parent chains are validated, recursion fails
closed, and nesting is capped at eight levels.

## Validation and planning

Validation uses the component schema for component-backed nodes and the
node-kind schema otherwise. It rejects missing required inputs, unconsumed required outputs,
unknown nodes or ports, duplicate IDs, incompatible types, zero-capacity
channels, implicit fan-in, unsafe nodes without graph approval, invalid cycles,
missing capabilities, unavailable/unverified components, missing, empty, or
mistyped required configuration, and disconnected selected sinks.

Compilation topologically orders otherwise-independent nodes by stable node ID,
sorts edge-derived channels by saved edge order, and records:

- bounded capacity, explicit stream delivery, and producer-blocking
  backpressure for every edge;
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

Route-level graph identity, durable execution records, tracks handoffs, and
the distinction between configuration drafts and execution evidence are
documented in [speech-workspace-navigation.md](speech-workspace-navigation.md).

## Migration

Schema 1 graphs migrate to schema 3 by renaming `id` to `graph_id`, moving
`name` under `metadata.name`, and replacing selected output node IDs with
explicit sink endpoints. Schema 2 graphs promote legacy layout and faceplate
label JSON into typed presentation metadata and begin with no subpatches.
Future schemas fail with an instruction to upgrade Tongues. Unknown older schemas fail with a precise missing-migration error.
After structural migration, normal validation reports any component or port
that the current registry can no longer resolve.

The browser stores schema-3 JSON, upgrades schema-2 drafts before editing, and
sends old saved JSON through the backend migration endpoint before use.
WaveDeck remains the sibling evidence
editor; graph configuration and human correction do not become one mutable
authority.
