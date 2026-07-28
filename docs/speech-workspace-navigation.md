# Speech workspace navigation

Tongues uses several independently loadable browser workspaces instead of one
global-state SPA. A route identifies the workspace and durable backend IDs
restore graph, execution-run, and session context after a reload. Browser
history is useful navigation, but it is never treated as evidence storage.

## Route map and page identity

| Route | Workspace | Authority shown on the page |
| --- | --- | --- |
| `/commands` and `/commands/{command-id}` | Command Workbench | a configuration draft for a browser-safe CLI workflow |
| `/speech`, `/speech/compose`, `/speech/compare`, `/speech/catalog`, `/speech/operate` | Speech Studio | speech configuration, comparison, discovery, or runtime operation |
| `/speech/live` | Live conversation | an execution in progress; committed turns become session evidence |
| `/studio/graphs/new?starter={starter-id}` | Graph Studio | an unsaved configuration draft seeded by a backend starter |
| `/studio/graphs/{graph-id}` | Graph Studio | a saved graph revision; `?subpatch={subpatch-id}&node={node-id}` drills into an embedded subpatch and selects provenance |
| `/runs` | Execution tracks | an index of durable execution records |
| `/runs/{run-id}/tracks` | Execution tracks | observed lifecycle evidence for one execution run |
| `/sessions/new/correct` | WaveDeck | an empty correction workspace awaiting observed evidence |
| `/sessions/{session-id}/correct` | WaveDeck | an editable projection; baseline evidence remains immutable |

The legacy `/speech-dataflow.html` and `/wavedeck.html` files remain valid
entrypoints, but links should use the route-level contracts above.

## Durable API contracts

- `GET/PUT /api/pipeline/graphs/{graph-id}` restores and atomically saves a
  versioned graph under `data/speech-graphs`.
- `GET /api/pipeline/runs` lists recent execution records.
- `GET /api/pipeline/runs/{run-id}` restores graph identity, revision, status,
  and lifecycle events from `data/speech-runs`.
- `GET/PUT /api/timeline/sessions/{session-id}` restores or atomically saves a
  validated `SpeechTimelineSession` plus optional `graph_id`, `run_id`, and
  human-readable source context under `data/speech-sessions`.

Identifiers permit only ASCII letters, digits, dash, underscore, colon, and
dot. The server hashes each validated ID for its storage filename and verifies
that the document identity still matches the requested identity when reading.

## Intentional handoffs

Command Workbench links a workflow with `studio_template` metadata to
`/studio/graphs/new?starter={starter-id}`. This carries a **configuration
draft**, not a past execution. The starter is resolved from the backend starter
catalog on every fresh page load. Saving gives the draft an independent graph
ID and replaces the URL with its durable graph route.

Starting a Workbench command creates an atomic backend record under
`data/command-jobs` before the process launches. Its `/commands/{command-id}`
URL preserves the schema-owned argument draft and adds the durable `job` ID.
Opening that URL restores status, bounded output, progress, artifacts, and the
exact validated argv after a browser or server restart. A job that was running
when the server stopped is restored as failed with an explicit
`Interrupted by server restart` phase; Tongues never presents it as completed.
Output files link through the bounded artifact download API. Commands classified
as destructive by the server-owned CLI schema show recovery guidance and
require confirmation immediately before submission.

Graph execution creates a run record before streaming lifecycle events.
Graph Studio exposes `/runs/{run-id}/tracks` as soon as the first event arrives.
This carries an **execution run**. The tracks page can return to the exact saved
graph and selected node using `graph_id` and `node_id`; it does not reconstruct
canvas state from browser memory.

When a run has a timeline `session_id`, tracks link to
`/sessions/{session-id}/correct`. This carries **immutable observed evidence**
into WaveDeck. WaveDeck appends provenance-bearing operations and saves the
result as an **editable projection**. Its graph/run return links come from the
session record, not from referrer history.

Execution tracks uses one reusable projector for durable sessions, graph runs,
and incremental microphone/file events. It places input audio levels, VAD
boundaries, anonymous diarization assignments, raw and normalized transcript,
language, generated tokens, TTS/playback, pipeline lifecycle, latency, and
errors on stable millisecond coordinates. Repeated speaker assignments remain
visible as revisions, and spans from different anonymous speakers are marked
when they overlap; neither state is presented as verified identity. An enrolled
display name is used only when the source provenance explicitly includes
identity consent.

Zoom, pan, follow-live, track filters, speaker filters, and transcript
projection controls are non-destructive. The renderer limits the number of DOM
spans in a visible window, so a long session does not continuously append every
historical event to the page. Selecting a span exposes its source event,
provider/model, graph node, upstream sources, and downstream consumers. The
WaveDeck handoff carries the same `span`, `start_ms`, and `end_ms`; WaveDeck
restores the matching evidence interval without changing it.

Phone and phoneme selections use the same query contract. Tracks and WaveDeck
write the focused evidence ID and interval back into the current URL whenever
selection changes, so refresh/bookmark/back-to-run restores the same span.
Timeline alignment edges highlight the selected phone's word, transcript,
anonymous speaker, and source-audio parents, or the related phones when a
parent is selected. The selection inspector shows the attached artifact,
algorithm/version, alignment provider/model/version, boundary origin,
confidence, recipe, graph, owning run, and source-audio identity.

If no `phonetic_segmentation` attachment exists, both surfaces say that no
alignment is claimed. Partial/unsupported artifacts report untimed issue rows
from their immutable payload and never draw them as authoritative spans.

Microphone and file audio stays in browser memory for monitoring and permitted
interval playback. Durable session writes remove raw audio payloads and
speaker/voice embeddings while retaining non-biometric levels, timestamps,
anonymous labels, transcript, and provenance. The tracks page always shows
capture, raw-audio retention, and biometric retention state. Completion,
cancellation, and failure are terminal, separately styled states.

The live conversation page links to the backend `live_conversation` starter.
After a completed turn, committed stream events are projected into a durable
timeline session and the page exposes its WaveDeck route. Unstable hypotheses
and browser-only playback state are not fabricated as historical evidence.

## History, errors, and accessibility

Within the Command Workbench and Speech Studio shell, ordinary unmodified links
use `pushState`; `popstate` restores the route and selected workflow. Route
changes update the document title, announce the new identity in a polite live
region, and move focus to the page or workflow heading. Modified clicks and
new-tab behavior remain normal browser navigation. The independent Graph
Studio, tracks, and WaveDeck pages use ordinary links, so reload and
back/forward behavior comes from their durable URLs.

Missing graph, node, run, or session IDs produce an explicit recovery message
with links to a new graph, recent runs, or a new WaveDeck import/live session.
The UI never invents a missing prior selection. Responsive layouts collapse
tracks and evidence panes to one column, and all transition targets, event
selections, errors, and correction status use keyboard-focusable headings or
live regions.
