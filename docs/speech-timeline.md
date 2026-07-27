# Speech timeline and correction workbench

WaveDeck is available at `/wavedeck.html` and linked from Speech Studio as a
distinct sibling view. The node graph configures execution; WaveDeck inspects
and corrects session evidence. Both live ASR and saved files use
`SpeechTimelineSession` schema version 1.

The baseline `evidence` array is immutable. Transcript replacement, boundary
movement, annotation, segmentation, and audio-region edits append
`TimelineOperation` records containing a stable operation ID, actor, origin,
time, source span/event IDs, and optional reason. Undo and redo are themselves
serialized control operations. Replaying the log against the baseline produces
the edited projection deterministically.

The workbench presents original and edited projections side by side. It can
export raw evidence, corrected transcript/timing, the edit log, or a complete
bundle independently. Bundles label source evidence as `observed` and edits as
`corrected_interpretation`; a human correction never upgrades or overwrites the
authority of captured evidence.

Keyboard operation includes normal tab navigation, `E` to edit selected text,
`[` and `]` for boundary movement, and platform undo/redo shortcuts. Selection
and errors are announced through one live status region. Files with an unknown
schema version fail with the expected version instead of being silently
rewritten.

The reduced fixture
`fixtures/timeline/listenbury_user_interrupts_v1.json` records the Listenbury
playback → overlap → interruption → yield ordering. It replaces the much larger
viewer payload while retaining the acceptance-relevant spans and provenance.

WaveDeck's durable `/sessions/{session-id}/correct` route, graph/run return
context, recovery behavior, and cross-workspace authority boundaries are
documented in [speech-workspace-navigation.md](speech-workspace-navigation.md).
