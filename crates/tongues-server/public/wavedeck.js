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
let context = {};
const liveEvents = [];

const byId = id => document.getElementById(id);
const announce = text => { byId("status").textContent = text; };
const sessionIdFromRoute = (pathname = location.pathname) => {
  const match = pathname.match(/^\/sessions\/([^/]+)\/correct\/?$/);
  if (!match || match[1] === "new") return null;
  try { return decodeURIComponent(match[1]); } catch { return null; }
};

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let value = {};
  try { value = text ? JSON.parse(text) : {}; } catch { value = {error: text}; }
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status}).`);
  return value;
}

function renderContext() {
  byId("context-run").hidden = !context.run_id;
  byId("context-graph").hidden = !context.graph_id;
  if (context.run_id) byId("context-run").href = `/runs/${encodeURIComponent(context.run_id)}/tracks`;
  if (context.graph_id) byId("context-graph").href = `/studio/graphs/${encodeURIComponent(context.graph_id)}`;
  byId("session-context").textContent = context.source
    ? `Source: ${context.source}. Original evidence is immutable; edits remain a replayable interpretation.`
    : "Original evidence remains immutable. Edits are a separate replayable interpretation.";
}

async function persistSession() {
  if (!session) return;
  const record = await request(`/api/timeline/sessions/${encodeURIComponent(session.session_id)}`, {
    method: "PUT",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({schema_version: 1, session, context}),
  });
  context = record.context || context;
  history.replaceState({session_id: session.session_id}, "", `/sessions/${encodeURIComponent(session.session_id)}/correct`);
  renderContext();
  announce(`Saved corrected projection for ${session.session_id}; original evidence remains unchanged.`);
}

async function loadDurableSession(sessionId) {
  const record = await request(`/api/timeline/sessions/${encodeURIComponent(sessionId)}`);
  session = validateSession(record.session);
  context = record.context || {};
  selected = null;
  renderContext();
  render();
  restoreRouteSelection();
  document.title = `${session.session_id} · WaveDeck · Tongues`;
}

function restoreRouteSelection(search = location.search) {
  if (!session) return;
  const params = new URLSearchParams(search);
  const requested = params.get("span");
  const start = Number(params.get("start_ms"));
  const end = Number(params.get("end_ms"));
  const projection = projectSession(session);
  const span = projection.edited.find(candidate =>
    candidate.id === requested
    || candidate.id.endsWith(`:${requested}`)
    || (Number.isFinite(start) && Number.isFinite(end)
      && candidate.start_ms < end && candidate.end_ms > start));
  if (!span) {
    if (requested || Number.isFinite(start) || Number.isFinite(end)) {
      byId("recovery").textContent = "The requested Run Tracks interval is no longer present; the full session is open.";
    }
    return;
  }
  selected = span.id;
  render();
  announce(`Opened ${formatInterval(span.start_ms, span.end_ms)} from Run Tracks; select an editing action when ready.`);
}

function formatInterval(start, end) {
  return `${(start / 1000).toFixed(2)}–${(end / 1000).toFixed(2)} seconds`;
}

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
  persistSession().catch(error => announce(`Correction is only in this page: ${error.message}`));
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
  persistSession().catch(error => announce(`Alignment is only in this page: ${error.message}`));
}

function annotateSelected() {
  const span = selectedSpan();
  if (!span) return announce("Select a span first.");
  const value = prompt("Annotation");
  if (value === null) return;
  appendOperation(session, "annotate", {span_id: span.id, key: "note", value});
  render();
  persistSession().catch(error => announce(`Annotation is only in this page: ${error.message}`));
}

async function openFile(file) {
  try {
    const document = JSON.parse(await file.text());
    session = validateSession(document.session ?? document);
    context = document.context || {source: "imported file"};
    selected = null;
    render();
    await persistSession();
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
        context = {source: "WaveDeck live recognition"};
        render();
        persistSession().catch(error => announce(`Live session is only in this page: ${error.message}`));
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
byId("undo").onclick = () => { if (undo(session)) { render(); persistSession().catch(error => announce(error.message)); } else announce("Nothing to undo."); };
byId("redo").onclick = () => { if (redo(session)) { render(); persistSession().catch(error => announce(error.message)); } else announce("Nothing to redo."); };
byId("start-live").onclick = () => startLive().catch(error => announce(`Microphone unavailable: ${error.message}`));
byId("stop-live").onclick = stopLive;
document.querySelectorAll("[data-export]").forEach(button => button.onclick = () => download(button.dataset.export));
document.onkeydown = event => {
  if (!session || /INPUT|TEXTAREA/.test(event.target.tagName)) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault(); (event.shiftKey ? redo(session) : undo(session)); render(); persistSession().catch(error => announce(error.message));
  } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
    event.preventDefault(); redo(session); render(); persistSession().catch(error => announce(error.message));
  } else if (event.key.toLowerCase() === "e") replaceSelected();
  else if (event.key === "[") moveBoundary("start", -10);
  else if (event.key === "]") moveBoundary("end", 10);
};
renderContext();
const requestedSession = sessionIdFromRoute();
if (requestedSession) {
  loadDurableSession(requestedSession).catch(error => {
    byId("recovery").innerHTML = `${escapeHtml(error.message)} <a href="/sessions/new/correct">Open a file or start live recognition</a> or <a href="/runs">return to execution tracks</a>.`;
    announce("Session context could not be restored. Recovery links are available.");
    byId("page-title").focus();
    render();
  });
} else {
  render();
}

export {restoreRouteSelection, sessionIdFromRoute};
