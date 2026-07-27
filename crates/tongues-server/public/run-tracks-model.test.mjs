import test from "node:test";
import assert from "node:assert/strict";
import {boundedVisibleSpans, projectSessionTracks, selectionProvenance, waveDeckHandoff} from "./run-tracks-model.mjs";

const envelope = (sequence, offset_ms, event, provenance = {}) => ({
  schema_version: 1, stream_id: "stream:fixture", event_id: `event:${sequence}`, sequence,
  times: {occurred_at: {origin: {kind: "stream_start"}, offset_ms}, observed_at: {origin: {kind: "unix_epoch"}, offset_ms: 10_000 + offset_ms}},
  provenance: {kind: "derived", sources: [], ...provenance}, event,
});

function fixture(status = "completed") {
  const events = [
    envelope(0, 0, {type:"session_started",data:{purpose:"file transcription"}}),
    envelope(1, 10, {type:"audio_chunk",data:{direction:"input",chunk_sequence:0,frame_count:1600,format:{sample_rate_hz:16000},metadata:{level_dbfs:-18}}}),
    envelope(2, 20, {type:"speech_started",data:{segment_id:"s1"}}),
    envelope(3, 25, {type:"partial_hypothesis",data:{role:"recognition",segment_id:"s1",text:"good"}}),
    envelope(4, 40, {type:"revised_hypothesis",data:{role:"recognition",segment_id:"s1",replaces:{start:0,end:4},text:"hello"}}),
    envelope(5, 45, {type:"speaker_assigned",data:{segment_id:"s1",speaker_id:"cluster-9"}}),
    envelope(6, 50, {type:"speaker_assigned",data:{segment_id:"s1",speaker_id:"cluster-2"}}),
    envelope(7, 200, {type:"committed_segment",data:{role:"recognition",segment_id:"s1",text:"hello",words:[{text:"hello",range:{start_ms:20,end_ms:200}}],language:{language:"en"},speaker_id:"cluster-2"}},
      {provider:"fixture-asr",model:"tiny",attributes:{graph_node_id:"asr"}}),
    envelope(8, 150, {type:"committed_segment",data:{role:"recognition",segment_id:"s2",text:"yes",words:[{text:"yes",range:{start_ms:150,end_ms:260}}],speaker_id:"cluster-7"}}),
    envelope(9, 210, {type:"derived_artifact",data:{stage:"transcript_normalization",artifact_id:"n1",value:{text:"Hello."}}},
      {sources:[{stream_id:"stream:fixture",event_id:"event:7"}]}),
    envelope(10, 220, {type:"token_timing",data:{segment_id:"g1",token:{text:"Hi",range:{start_ms:220,end_ms:240}}}}),
    envelope(11, 240, {type:"audio_chunk",data:{direction:"output",chunk_sequence:0,frame_count:4800,format:{sample_rate_hz:24000}}}),
    envelope(12, 270, status === "cancelled" ? {type:"cancelled",data:{reason:"operator cancelled"}} : {type:"completed",data:{}}),
  ];
  return {
    status, context:{graph_id:"pipeline:demo",run_id:"run:1"},
    session:{schema_version:1,session_id:"session:1",evidence:[],alignments:[],source_events:events,operations:[]},
  };
}

test("offline projector aligns audio, transcript, language, generation, TTS, and terminal tracks", () => {
  const view = projectSessionTracks(fixture());
  assert.equal(view.session_id, "session:1");
  assert.equal(view.status, "completed");
  for (const id of ["audio_input","transcript_raw","transcript_normalized","language","generation","tts","pipeline"]) {
    assert.ok(view.tracks.find(track => track.id === id), id);
  }
  assert.equal(view.tracks.find(track => track.id === "audio_input").spans[0].start_ms, 10);
  assert.equal(view.privacy.raw_audio_retained, false);
});

test("partial/revised transcript and speaker revisions preserve segment order and show overlap", () => {
  const view = projectSessionTracks(fixture());
  const transcript = view.tracks.find(track => track.id === "transcript_raw").spans;
  assert.deepEqual(transcript.filter(span => span.segment_id === "s1").map(span => span.status), ["provisional","revised","committed"]);
  const speakers = view.tracks.find(track => track.id === "speakers").spans;
  const first = speakers.find(span => span.segment_id === "s1");
  assert.equal(first.revision, 2);
  assert.match(first.label, /^Speaker [A-Z]/);
  assert.equal(first.metadata.identity_authority, "diarization_only");
  assert.ok(speakers.some(span => span.overlap));
});

test("selection resolves source, provider, model, graph node, and downstream evidence", () => {
  const view = projectSessionTracks(fixture());
  const committed = view.tracks.find(track => track.id === "transcript_raw").spans.find(span => span.event_id === "event:7");
  const provenance = selectionProvenance(view, committed);
  assert.equal(provenance.provider, "fixture-asr");
  assert.equal(provenance.model, "tiny");
  assert.equal(provenance.graph_node_id, "asr");
  assert.deepEqual(provenance.downstream_event_ids, ["event:9"]);
  assert.equal(
    waveDeckHandoff(view.session_id, committed),
    "/sessions/session%3A1/correct?span=s1&start_ms=20&end_ms=200",
  );
});

test("cancelled stream is terminal and unambiguous", () => {
  const view = projectSessionTracks(fixture("cancelled"));
  assert.equal(view.status, "cancelled");
  assert.equal(view.tracks.find(track => track.id === "errors").spans.at(-1).status, "cancelled");
});

test("bounded rendering samples long visible ranges while retaining the tail", () => {
  const track = {spans:Array.from({length:10_000}, (_, index) => ({start_ms:index,end_ms:index + 1}))};
  const visible = boundedVisibleSpans(track, {start_ms:0,end_ms:20_000}, 500);
  assert.ok(visible.length <= 501);
  assert.equal(visible.at(-1), track.spans.at(-1));
});
