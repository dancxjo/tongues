import test from "node:test";
import assert from "node:assert/strict";
import {appendOperation, projectSession, redo, sessionFromEvents, undo} from "./wavedeck-model.mjs";

function base() {
  return {
    schema_version: 1, session_id: "saved:1",
    evidence: [{id:"word:1",start_ms:10,end_ms:40,modality:"word",metadata:{text:"hello"}}],
    alignments: [], source_events: [], operations: [],
  };
}

test("corrections preserve original evidence and replay from JSON", () => {
  const session = base();
  appendOperation(session, "transcript_replace", {span_id:"word:1",text:"hullo"});
  appendOperation(session, "alignment_move_boundary", {span_id:"word:1",boundary:"end",new_time_ms:55});
  const projection = projectSession(JSON.parse(JSON.stringify(session)));
  assert.equal(session.evidence[0].metadata.text, "hello");
  assert.equal(projection.edited[0].metadata.text, "hullo");
  assert.equal(projection.edited[0].end_ms, 55);
});

test("undo and redo are replayable operations", () => {
  const session = base();
  appendOperation(session, "transcript_replace", {span_id:"word:1",text:"hullo"});
  assert.equal(undo(session), true);
  assert.equal(projectSession(session).edited[0].metadata.text, "hello");
  assert.equal(redo(session), true);
  assert.equal(projectSession(session).edited[0].metadata.text, "hullo");
});

test("live shared events become the same session schema", () => {
  const session = sessionFromEvents("live:1", [{
    received_at_ms: 100,
    event: {type:"committed_segment",data:{segment_id:"s1",text:"hello",words:[]}},
  }]);
  assert.equal(session.schema_version, 1);
  assert.equal(session.evidence[0].metadata.text, "hello");
  assert.equal(session.source_events[0].type, "committed_segment");
});

test("segment IDs reused by separate live turns remain distinct evidence", () => {
  const committed = (text, received_at_ms) => ({
    received_at_ms,
    event: {
      type: "committed_segment",
      data: {
        segment_id: "generation-segment-1",
        text,
        words: [{text, range:{start_ms:received_at_ms,end_ms:received_at_ms + 20}}],
      },
    },
  });
  const session = sessionFromEvents("live:multiple-turns", [
    committed("hello", 100),
    committed("again", 200),
  ]);

  assert.deepEqual(
    session.evidence.map(span => span.id),
    [
      "transcript:generation-segment-1",
      "word:generation-segment-1:0",
      "transcript:generation-segment-1:occurrence-2",
      "word:generation-segment-1:occurrence-2:0",
    ],
  );
});

test("unknown schema produces an actionable migration error", () => {
  assert.throws(() => projectSession({...base(),schema_version:99}), /unsupported; expected 1/);
});

test("Run Tracks handoff has a stable evidence interval to select", () => {
  const session = base();
  const selected = projectSession(session).edited.find(span =>
    span.id.endsWith(":1") || (span.start_ms < 40 && span.end_ms > 10));
  assert.equal(selected.id, "word:1");
});
