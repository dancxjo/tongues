import {
  appendOperation, focusedSessionSpan, projectSession, redo, relatedSpanIds, segmentationState,
  sessionFromEvents, undo, validateSession, waveformPolylinePoints,
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
  renderSegmentation();
}

function renderSegmentation() {
  const state = segmentationState(session);
  const panel = byId("segmentation-state");
  panel.dataset.readiness = state.readiness;
  byId("segmentation-message").textContent = state.message;
  const selectedArtifact = state.artifacts.find(artifact =>
    selectedSpan()?.metadata?.artifact_id === artifact.artifact_id) ?? state.artifacts[0];
  if (!selectedArtifact) {
    byId("segmentation-provenance").replaceChildren();
    return;
  }
  const rows = [
    ["Algorithm", selectedArtifact.algorithm_version],
    ["Artifact", selectedArtifact.artifact_id],
    ["Recipe", selectedArtifact.recipe_id ?? "Not linked"],
    ["Readiness", selectedArtifact.readiness],
    ["Mode", selectedArtifact.mode ?? "Legacy segmentation"],
    ["Selected path", selectedArtifact.selected_hypothesis?.id ?? "Abstained"],
    ["Path posterior", Number.isFinite(selectedArtifact.selected_hypothesis?.normalized_path_posterior)
      ? selectedArtifact.selected_hypothesis.normalized_path_posterior.toFixed(3) : "Not calibrated"],
    ["Retained alternatives", String(selectedArtifact.alternatives?.length ?? 0)],
    ["Units with boundary ranges", String(selectedArtifact.boundary_ranges?.length ?? 0)],
    ["Untimed rows", String(selectedArtifact.missing_segments.length)],
    ["Issues", String(selectedArtifact.issues.length)],
  ];
  byId("segmentation-provenance").replaceChildren(...rows.flatMap(([term, value]) => {
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = term; dd.textContent = value; return [dt, dd];
  }));
}

async function persistSession() {
  if (!session) return;
  const record = await request(`/api/timeline/sessions/${encodeURIComponent(session.session_id)}`, {
    method: "PUT",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({schema_version: 1, session, context}),
  });
  context = record.context || context;
  const focus = selectedSpan();
  const query = focus ? `?${new URLSearchParams({span:focus.id,start_ms:String(focus.start_ms),end_ms:String(focus.end_ms)})}` : "";
  history.replaceState({session_id: session.session_id,span_id:focus?.id}, "", `/sessions/${encodeURIComponent(session.session_id)}/correct${query}`);
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
  const span = focusedSessionSpan(session, search);
  if (!span) {
    if (new URLSearchParams(search).size) {
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
  renderSegmentation();
  byId("session-name").textContent = session.session_id;
  byId("operation-count").textContent = `${session.operations.length} replayable operations`;
  announce(`Timeline ready. ${projection.edited.length} spans; ${session.operations.length} operations.`);
}

function renderLane(target, spans, editable) {
  const related = relatedSpanIds(session, selected);
  target.replaceChildren(...spans.map((span, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `span span-${span.modality}${selected === span.id ? " selected" : ""}${selected !== span.id && related.has(span.id) ? " related" : ""}`;
    button.dataset.spanId = span.id;
    button.dataset.index = index;
    button.dataset.boundaryOrigin = span.metadata?.boundary_origin ?? "";
    const label = span.metadata?.symbol ?? span.metadata?.text ?? span.modality;
    const waveform = span.modality === "audio" && Array.isArray(span.metadata?.waveform_peaks)
      ? `<svg class="waveform" viewBox="0 0 240 36" role="img" aria-label="Waveform summary"><polyline points="${waveformPolylinePoints(span.metadata.waveform_peaks)}"></polyline><polyline transform="translate(0 36) scale(1 -1)" points="${waveformPolylinePoints(span.metadata.waveform_peaks)}"></polyline></svg>`
      : "";
    button.innerHTML = `<strong>${escapeHtml(label)}</strong>${waveform}
      <small>${span.start_ms}–${span.end_ms} ms · ${escapeHtml(span.id)}${span.metadata?.boundary_origin ? ` · ${escapeHtml(span.metadata.boundary_origin)}` : ""}</small>`;
    button.onclick = () => {
      selected = span.id;
      const url = new URL(location.href);
      url.search = new URLSearchParams({span:span.id,start_ms:String(span.start_ms),end_ms:String(span.end_ms)});
      history.replaceState({session_id:session.session_id,span_id:span.id}, "", url);
      render();
      announce(`Selected ${span.id}; ${relatedSpanIds(session, span.id).size - 1} linked evidence spans highlighted.`);
    };
    button.ondblclick = () => editable && replaceSelected();
    return button;
  }));
}

function selectedSpan() {
  return projectSession(session).edited.find(span => span.id === selected);
}

function replaceSelected() {
  const span = selectedSpan();
  if (!span || !["transcript", "word", "phone", "phoneme"].includes(span.modality)) return announce("Select a transcript, word, phone, or phoneme span.");
  const phonetic = ["phone", "phoneme"].includes(span.modality);
  const text = prompt(phonetic ? "Corrected phone/phoneme symbol" : "Corrected text", phonetic ? span.metadata?.symbol ?? "" : span.metadata?.text ?? "");
  if (text === null) return;
  appendOperation(session, phonetic ? "phonetic_symbol_replace" : "transcript_replace", {
    span_id: span.id,
    ...(phonetic ? {symbol:text} : {text}),
    reason: phonetic ? "operator phonetic correction proposal" : "operator correction",
  });
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

function setBoundaryRange() {
  const span = selectedSpan();
  if (!span || !["phone", "phoneme"].includes(span.modality)) {
    return announce("Select a phone or phoneme span first.");
  }
  const boundary = prompt("Boundary to range: start or end", "start");
  if (!["start", "end"].includes(boundary)) return;
  const current = boundary === "start" ? span.start_ms : span.end_ms;
  const lower = Number(prompt("Lower supported time (ms)", String(Math.max(0, current - 10))));
  const estimate = Number(prompt("Boundary estimate (ms)", String(current)));
  const upper = Number(prompt("Upper supported time (ms)", String(current + 10)));
  if (!(Number.isFinite(lower) && lower <= estimate && estimate <= upper)) {
    return announce("Boundary range must satisfy lower ≤ estimate ≤ upper.");
  }
  appendOperation(session, "alignment_set_boundary_range", {
    span_id: span.id, boundary, lower_time_ms: lower, estimate_time_ms: estimate,
    upper_time_ms: upper, reason: "operator boundary-range correction",
  });
  render();
  persistSession().catch(error => announce(`Boundary range is only in this page: ${error.message}`));
}

function choosePronunciationPath() {
  const span = selectedSpan();
  if (!span || !["phone", "phoneme", "word"].includes(span.modality)) {
    return announce("Select a phone, phoneme, or word span first.");
  }
  const state = segmentationState(session);
  const paths = state.artifacts.flatMap(artifact =>
    (artifact.hypotheses ?? []).map(path => path.id)).filter(Boolean);
  const choice = prompt(
    paths.length ? `Pronunciation path ID:\n${paths.join("\n")}` : "Pronunciation path ID",
    span.metadata?.pronunciation_path_id ?? paths[0] ?? "",
  );
  if (!choice?.trim()) return;
  appendOperation(session, "alignment_choose_pronunciation", {
    span_id: span.id, pronunciation_path_id: choice.trim(),
    reason: "operator pronunciation-path correction",
  });
  render();
  persistSession().catch(error => announce(`Pronunciation choice is only in this page: ${error.message}`));
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
    : kind === "corrected" ? projection.edited.filter(span => ["transcript", "word", "phone", "phoneme"].includes(span.modality))
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
byId("boundary-range").onclick = setBoundaryRange;
byId("pronunciation-path").onclick = choosePronunciationPath;
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
