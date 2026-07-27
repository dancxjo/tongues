import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {
  appendOperation, focusedSessionSpan, projectSession, redo, relatedSpanIds, segmentationState,
  sessionFromEvents, undo, validateSession, waveformPolylinePoints,
} from "./wavedeck-model.mjs";

const phoneticFixture = JSON.parse(readFileSync(new URL(
  "../../../fixtures/timeline/phonetic-segmentation-inspection-v1.json",
  import.meta.url,
)));

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

test("segmentation correction journey preserves baseline and records who what and when", () => {
  const session = validateSession(structuredClone(phoneticFixture.session));
  const phoneId = "phonetic-segmentation:phones-v1:3";
  const original = session.evidence.find(span => span.id === phoneId);
  appendOperation(session, "phonetic_symbol_replace", {
    span_id:phoneId,symbol:"ɕ",reason:"multilingual fixture review",
  }, "fixture-reviewer");
  appendOperation(session, "alignment_move_boundary", {
    span_id:phoneId,boundary:"start",new_time_ms:365,reason:"waveform review",
  }, "fixture-reviewer");

  const replayed = projectSession(JSON.parse(JSON.stringify(session)));
  const edited = replayed.edited.find(span => span.id === phoneId);
  assert.equal(original.metadata.symbol, "ʃ");
  assert.equal(session.evidence.find(span => span.id === phoneId).start_ms, 370);
  assert.equal(edited.metadata.symbol, "ɕ");
  assert.equal(edited.start_ms, 365);
  assert.equal(edited.metadata.boundary_origin, "corrected");
  assert.equal(edited.metadata.correction_actor, "fixture-reviewer");
  assert.ok(Number.isFinite(edited.metadata.correction_at_ms));
  assert.match(edited.metadata.correction_operation_id, /^edit:/);
});

test("word and phone selections expose linked evidence in WaveDeck", () => {
  const session = validateSession(structuredClone(phoneticFixture.session));
  const phoneId = "phonetic-segmentation:phones-v1:1";
  assert.ok(relatedSpanIds(session, phoneId).has("word:utterance-1:0"));
  assert.ok(relatedSpanIds(session, "word:utterance-1:0").has(phoneId));
});

test("exact deep-link span wins over a broad overlapping audio interval", () => {
  const session = validateSession(structuredClone(phoneticFixture.session));
  const phoneId = "phonetic-segmentation:phones-v1:3";
  const span = focusedSessionSpan(
    session,
    `?span=${encodeURIComponent(phoneId)}&start_ms=370&end_ms=505`,
  );
  assert.equal(span.id, phoneId);
});

test("missing and partial segmentation states never imply authoritative timing", () => {
  assert.equal(segmentationState(base()).readiness, "missing");
  const state = segmentationState(phoneticFixture.session);
  assert.equal(state.readiness, "partial");
  assert.equal(state.artifacts[0].missing_segments.length, 2);
});

test("waveform summaries remain compact deterministic evidence", () => {
  const peaks = phoneticFixture.session.evidence[0].metadata.waveform_peaks;
  const points = waveformPolylinePoints(peaks);
  assert.equal(points.split(" ").length, peaks.length);
  assert.match(points, /^0\.00,/);
  assert.doesNotMatch(points, /NaN|Infinity/);
});
