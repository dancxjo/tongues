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
| `/studio/graphs/{graph-id}` | Graph Studio | a saved graph revision; `?node={node-id}` selects provenance |
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
