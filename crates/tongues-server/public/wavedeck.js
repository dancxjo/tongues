import {
  appendOperation, projectSession, redo, sessionFromEvents, undo, validateSession,
} from "./wavedeck-model.mjs";

let session = null;
let selected = null;
let socket = null;
let audio = null;
let media = null;
let source = null;
let processor = null;
const liveEvents = [];

const byId = id => document.getElementById(id);
const announce = text => { byId("status").textContent = text; };

function render() {
  const empty = !session;
  document.querySelectorAll("[data-needs-session]").forEach(node => node.disabled = empty);
  if (empty) {
    byId("original").replaceChildren();
    byId("edited").replaceChildren();
    announce("Open a saved session or start live recognition.");
    return;
  }
  let projection;
  try {
    projection = projectSession(session);
  } catch (error) {
    announce(`Projection failed: ${error.message}`);
    return;
  }
  renderLane(byId("original"), projection.original, false);
  renderLane(byId("edited"), projection.edited, true);
  byId("session-name").textContent = session.session_id;
  byId("operation-count").textContent = `${session.operations.length} replayable operations`;
  announce(`Timeline ready. ${projection.edited.length} spans; ${session.operations.length} operations.`);
}

function renderLane(target, spans, editable) {
  target.replaceChildren(...spans.map((span, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `span span-${span.modality}${selected === span.id ? " selected" : ""}`;
    button.dataset.spanId = span.id;
    button.dataset.index = index;
    button.innerHTML = `<strong>${escapeHtml(span.metadata?.text ?? span.modality)}</strong>
      <small>${span.start_ms}–${span.end_ms} ms · ${escapeHtml(span.id)}</small>`;
    button.onclick = () => { selected = span.id; render(); announce(`Selected ${span.id}.`); };
    button.ondblclick = () => editable && replaceSelected();
    return button;
  }));
}

function selectedSpan() {
  return projectSession(session).edited.find(span => span.id === selected);
}

function replaceSelected() {
  const span = selectedSpan();
  if (!span || !["transcript", "word"].includes(span.modality)) return announce("Select a transcript or word span.");
  const text = prompt("Corrected text", span.metadata?.text ?? "");
  if (text === null) return;
  appendOperation(session, "transcript_replace", {span_id: span.id, text, reason: "operator correction"});
  render();
}

function moveBoundary(boundary, amount) {
  const span = selectedSpan();
  if (!span) return announce("Select a span first.");
  const current = boundary === "start" ? span.start_ms : span.end_ms;
  appendOperation(session, "alignment_move_boundary", {
    span_id: span.id, boundary, new_time_ms: Math.max(0, current + amount),
    reason: "keyboard alignment adjustment",
  });
  render();
}

function annotateSelected() {
  const span = selectedSpan();
  if (!span) return announce("Select a span first.");
  const value = prompt("Annotation");
  if (value === null) return;
  appendOperation(session, "annotate", {span_id: span.id, key: "note", value});
  render();
}

async function openFile(file) {
  try {
    const document = JSON.parse(await file.text());
    session = validateSession(document.session ?? document);
    selected = null;
    render();
  } catch (error) {
    announce(`Cannot open session: ${error.message}`);
  }
}

function download(kind) {
  const projection = projectSession(session);
  const payload = kind === "evidence" ? session.evidence
    : kind === "corrected" ? projection.edited.filter(span => ["transcript", "word"].includes(span.modality))
    : kind === "operations" ? session.operations
    : {session, projection, evidence_authority: "observed", edit_authority: "corrected_interpretation"};
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type: "application/json"}));
  link.download = `${session.session_id.replaceAll(":", "-")}-${kind}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}

async function startLive() {
  if (socket) return;
  liveEvents.length = 0;
  media = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1}});
  audio = new AudioContext();
  source = audio.createMediaStreamSource(media);
  processor = audio.createScriptProcessor(4096, 1, 1);
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/asr/stream`);
  socket.binaryType = "arraybuffer";
  socket.onopen = () => socket.send(JSON.stringify({
    type: "open", schema_version: 1, provider: "fixture",
    sample_rate_hz: audio.sampleRate, channels: 1, language: "en",
  }));
  socket.onmessage = ({data}) => {
    const message = JSON.parse(data);
    if (message.type === "recognition") {
      liveEvents.push({event: message.event, received_at_ms: performance.now()});
      if (message.event.type === "committed_segment") {
        session = sessionFromEvents(`live:${Date.now()}`, liveEvents);
        render();
      }
    }
    if (message.type === "error") announce(`${message.code}: ${message.message}`);
    if (message.type === "ended") teardownLive();
  };
  processor.onaudioprocess = event => {
    if (socket?.readyState === WebSocket.OPEN) socket.send(event.inputBuffer.getChannelData(0).slice().buffer);
  };
  source.connect(processor);
  processor.connect(audio.destination);
  byId("start-live").disabled = true;
  byId("stop-live").disabled = false;
  announce("Microphone active. Recognition evidence is not written to disk.");
}

function stopLive() {
  if (socket?.readyState === WebSocket.OPEN) socket.send('{"type":"end"}');
  teardownLive();
}

function teardownLive() {
  processor?.disconnect();
  source?.disconnect();
  media?.getTracks().forEach(track => track.stop());
  audio?.close();
  socket = audio = media = source = processor = null;
  byId("start-live").disabled = false;
  byId("stop-live").disabled = true;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
}

byId("open-file").onchange = event => event.target.files[0] && openFile(event.target.files[0]);
byId("replace").onclick = replaceSelected;
byId("annotate").onclick = annotateSelected;
byId("start-earlier").onclick = () => moveBoundary("start", -10);
byId("start-later").onclick = () => moveBoundary("start", 10);
byId("end-earlier").onclick = () => moveBoundary("end", -10);
byId("end-later").onclick = () => moveBoundary("end", 10);
byId("undo").onclick = () => { if (undo(session)) render(); else announce("Nothing to undo."); };
byId("redo").onclick = () => { if (redo(session)) render(); else announce("Nothing to redo."); };
byId("start-live").onclick = () => startLive().catch(error => announce(`Microphone unavailable: ${error.message}`));
byId("stop-live").onclick = stopLive;
document.querySelectorAll("[data-export]").forEach(button => button.onclick = () => download(button.dataset.export));
document.onkeydown = event => {
  if (!session || /INPUT|TEXTAREA/.test(event.target.tagName)) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault(); (event.shiftKey ? redo(session) : undo(session)); render();
  } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
    event.preventDefault(); redo(session); render();
  } else if (event.key.toLowerCase() === "e") replaceSelected();
  else if (event.key === "[") moveBoundary("start", -10);
  else if (event.key === "]") moveBoundary("end", 10);
};
render();
