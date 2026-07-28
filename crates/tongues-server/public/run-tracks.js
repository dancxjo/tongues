import {
  boundedVisibleSpans, focusQuery, focusedSpan, projectSessionTracks, relatedSelectionIds,
  selectionProvenance, spanDensity, waveDeckHandoff,
} from "./run-tracks-model.mjs";
import {sessionFromEvents} from "./wavedeck-model.mjs";

const byId = id => document.getElementById(id);
const state = {
  record:null, projected:null, selected:null, poll:null, follow:true, zoom:1, pan:0,
  hidden:new Set(), speaker:"", socket:null, media:null, audioContext:null, source:null,
  processor:null, liveEvents:[], liveSessionId:null, audioUrl:null, renderQueued:false,
};

export function runIdFromRoute(pathname = location.pathname) {
  const match = pathname.match(/^\/runs\/([^/]+)\/tracks\/?$/);
  if (!match) return null;
  try { return decodeURIComponent(match[1]); } catch { return null; }
}

export function graphRoute(graphId, nodeId = "") {
  if (!graphId) return "/studio/graphs/new";
  const route = `/studio/graphs/${encodeURIComponent(graphId)}`;
  return nodeId ? `${route}?node=${encodeURIComponent(nodeId)}` : route;
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let value = {};
  try { value = text ? JSON.parse(text) : {}; } catch { value = {error:text}; }
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status}).`);
  return value;
}

const announce = text => { byId("status").textContent = text; };

function renderIndex(runs) {
  byId("run-view").hidden = true;
  byId("run-index").hidden = false;
  byId("run-list").replaceChildren(...runs.map(run => {
    const link = document.createElement("a");
    link.href = `/runs/${encodeURIComponent(run.run_id)}/tracks`;
    link.innerHTML = `<strong>${escapeHtml(run.run_id)}</strong>
      <span>${escapeHtml(run.status)} · graph ${escapeHtml(run.graph_id)} revision ${run.graph_revision}</span>
      <small>${run.event_count ?? run.events?.length ?? 0} recorded events</small>`;
    return link;
  }));
  announce(runs.length ? `${runs.length} recent durable runs available.` : "No durable runs yet. Start a microphone or file session here, or run a graph.");
}

function renderRecord(record) {
  state.record = record;
  state.projected = projectSessionTracks(record);
  byId("run-index").hidden = true;
  byId("run-view").hidden = false;
  byId("run-name").textContent = state.projected.run_id ?? state.projected.session_id ?? "Live session";
  byId("status-badge").textContent = state.projected.status;
  byId("status-badge").dataset.state = state.projected.status;
  byId("cancel-live").hidden = !state.socket;
  const interpretationLink = safeInterpretationLink(record.context?.interpretation_deep_link);
  byId("interpretation-link").hidden = !interpretationLink;
  if (interpretationLink) byId("interpretation-link").href = interpretationLink;
  renderPrivacy();
  renderSegmentation();
  renderFilters();
  updateViewport(state.follow);
  scheduleRender();
  const terminal = ["completed","cancelled","failed"].includes(state.projected.status);
  announce(`${byId("run-name").textContent} is ${state.projected.status}; ${spanCount()} aligned spans are available.${terminal ? " Stream is terminal." : ""}`);
  if (!state.selected) {
    const focused = focusedSpan(state.projected, location.search);
    if (focused) selectSpan(focused, {updateUrl:false, focusHeading:false});
  }
}

function renderPrivacy() {
  const privacy = state.projected.privacy;
  byId("privacy").innerHTML = `<strong>Capture: ${escapeHtml(privacy.capture)}</strong>
    <span>Raw audio retained: ${privacy.raw_audio_retained ? "yes" : "no"}</span>
    <span>Biometric speaker data retained: ${privacy.biometric_speaker_data_retained ? "yes" : "no"}</span>
    <span>${escapeHtml(privacy.policy)}</span>`;
}

function renderSegmentation() {
  const segmentation = state.projected.segmentation;
  const panel = byId("segmentation-state");
  panel.dataset.readiness = segmentation.readiness;
  byId("segmentation-message").textContent = segmentation.message;
  byId("segmentation-artifacts").replaceChildren(...segmentation.artifacts.map(artifact => {
    const detail = document.createElement("p");
    const missing = artifact.missing_segments.length;
    const alternatives = artifact.alternatives?.length ?? 0;
    const ranges = artifact.boundary_ranges?.length ?? 0;
    detail.textContent = `${artifact.algorithm_version} · recipe ${artifact.recipe_id ?? "not linked"} · ${artifact.readiness}${artifact.mode ? ` · ${artifact.mode}` : ""}${alternatives ? ` · ${alternatives} alternative paths` : ""}${ranges ? ` · ${ranges} ranged units` : ""}${missing ? ` · ${missing} untimed rows` : ""}`;
    return detail;
  }));
}

function renderFilters() {
  if (!byId("track-filters").childElementCount) {
    byId("track-filters").replaceChildren(...state.projected.tracks.map(track => {
      const label = document.createElement("label");
      label.innerHTML = `<input type="checkbox" data-track="${track.id}" checked> ${escapeHtml(track.label)}`;
      label.querySelector("input").onchange = event => {
        event.target.checked ? state.hidden.delete(track.id) : state.hidden.add(track.id);
        scheduleRender();
      };
      return label;
    }));
  }
  const speakers = [...new Set(state.projected.tracks.find(track => track.id === "speakers").spans.map(span => span.speaker).filter(Boolean))];
  const selected = byId("speaker-filter").value;
  byId("speaker-filter").replaceChildren(new Option("All anonymous speakers", ""), ...speakers.map(value => new Option(value, value)));
  byId("speaker-filter").value = speakers.includes(selected) ? selected : "";
}

function updateViewport(jumpToEnd = false) {
  const duration = state.projected.duration_ms;
  const windowMs = Math.max(250, duration / state.zoom);
  const maxPan = Math.max(0, duration - windowMs);
  byId("pan").max = String(Math.ceil(maxPan));
  if (jumpToEnd) state.pan = maxPan;
  state.pan = Math.min(maxPan, Math.max(0, state.pan));
  byId("pan").value = String(state.pan);
}

function scheduleRender() {
  if (state.renderQueued) return;
  state.renderQueued = true;
  requestAnimationFrame(() => { state.renderQueued = false; renderTimeline(); });
}

function renderTimeline() {
  const duration = state.projected.duration_ms;
  const windowMs = Math.max(250, duration / state.zoom);
  const range = {start_ms:state.pan, end_ms:Math.min(duration, state.pan + windowMs)};
  renderRuler(range);
  const raw = byId("show-raw").checked, normalized = byId("show-normalized").checked;
  const tracks = state.projected.tracks.filter(track =>
    !state.hidden.has(track.id)
    && (track.id !== "transcript_raw" || raw)
    && (track.id !== "transcript_normalized" || normalized));
  byId("tracks").replaceChildren(...tracks.map(track => renderTrack(track, range)));
}

function renderRuler(range) {
  const ticks = [];
  for (let index = 0; index <= 10; index += 1) {
    const time = range.start_ms + (range.end_ms - range.start_ms) * index / 10;
    const tick = document.createElement("span");
    tick.className = "tick"; tick.style.left = `${index * 10}%`; tick.textContent = formatTime(time);
    ticks.push(tick);
  }
  byId("ruler").replaceChildren(...ticks);
}

function renderTrack(track, range) {
  const row = document.createElement("div");
  row.className = "track"; row.dataset.track = track.id;
  const label = document.createElement("div");
  label.className = "track-label";
  const allVisible = track.spans.filter(span => !state.speaker || !span.speaker || span.speaker === state.speaker);
  label.innerHTML = `<strong>${escapeHtml(track.label)}</strong><br><small>${allVisible.length} spans</small>`;
  const lane = document.createElement("div"); lane.className = "lane";
  const visible = boundedVisibleSpans({...track,spans:allVisible}, range);
  lane.replaceChildren(...visible.map(span => renderSpan(span, range)));
  if (!visible.length) { const empty = document.createElement("span"); empty.className = "empty"; empty.textContent = "No evidence in view"; lane.append(empty); }
  row.append(label, lane);
  return row;
}

function renderSpan(span, range) {
  const button = document.createElement("button");
  const width = range.end_ms - range.start_ms;
  button.className = "span"; button.type = "button";
  button.style.left = `${Math.max(0, (span.start_ms - range.start_ms) / width * 100)}%`;
  button.style.width = `${Math.max(.35, (Math.min(span.end_ms, range.end_ms) - Math.max(span.start_ms, range.start_ms)) / width * 100)}%`;
  button.dataset.status = span.status; button.dataset.overlap = String(span.overlap);
  button.dataset.boundaryOrigin = span.metadata?.boundary_origin ?? "";
  button.dataset.density = spanDensity(span, range);
  button.dataset.related = String(state.selected ? relatedSelectionIds(state.projected, state.selected).has(span.id) : false);
  button.textContent = button.dataset.density === "tick" ? "" : span.label;
  button.title = `${span.label} · ${formatTime(span.start_ms)}–${formatTime(span.end_ms)}`;
  button.setAttribute("aria-label", button.title);
  if (state.selected?.id === span.id && state.selected?.track === span.track) button.setAttribute("aria-current", "true");
  button.onclick = () => selectSpan(span);
  return button;
}

function selectSpan(span, {updateUrl = true, focusHeading = true} = {}) {
  state.selected = span;
  const provenance = selectionProvenance(state.projected, span);
  byId("selection-empty").hidden = true;
  byId("selection-details").hidden = false;
  const values = [
    ["Interval", `${formatTime(span.start_ms)}–${formatTime(span.end_ms)}`],
    ["State", span.status], ["Speaker", span.speaker ?? "—"], ["Source event", provenance.event_id ?? "—"],
    ["Provider / model / version", [provenance.provider, provenance.model, provenance.version].filter(Boolean).join(" / ") || "Not declared"],
    ["Boundary origin", provenance.boundary_origin ?? "Not a phonetic boundary"],
    ["Timing authority", provenance.timing_authority ?? "Not declared"],
    ["Lifecycle / relation", [provenance.lifecycle, provenance.relation].filter(Boolean).join(" / ") || "Not declared"],
    ["Hypothesis", provenance.hypothesis_id ?? "Not linked"],
    ["Path posterior", Number.isFinite(provenance.path_posterior) ? provenance.path_posterior.toFixed(3) : "Not calibrated"],
    ["Start boundary range", formatBoundary(provenance.start_boundary)],
    ["End boundary range", formatBoundary(provenance.end_boundary)],
    ["Score breakdown", provenance.score_breakdown ? JSON.stringify(provenance.score_breakdown) : "Not declared"],
    ["Confidence", Number.isFinite(provenance.confidence) ? provenance.confidence.toFixed(3) : "Not declared"],
    ["Algorithm", provenance.algorithm_version ?? "Not a segmentation span"],
    ["Artifact", provenance.artifact_id ?? "—"], ["Recipe", provenance.recipe_id ?? "—"],
    ["Owning execution", provenance.execution_record_id ?? state.projected.run_id ?? "—"],
    ["Source audio", provenance.audio_artifact_id ?? "—"],
    ["Graph node", provenance.graph_node_id ?? "Not linked"], ["Sources", provenance.sources.map(source => scalar(source.event_id)).join(", ") || "Direct"],
    ["Downstream", provenance.downstream_event_ids.join(", ") || "None recorded"], ["Authority", provenance.authority],
  ];
  byId("selection-details").replaceChildren(...values.flatMap(([term,value]) => {
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = term; dd.textContent = value; return [dt,dd];
  }));
  byId("graph-link").href = graphRoute(state.projected.graph_id, provenance.graph_node_id);
  const owningRun = provenance.execution_record_id ?? state.projected.run_id;
  byId("run-link").href = owningRun ? `/runs/${encodeURIComponent(owningRun)}/tracks?${focusQuery(span)}` : "/runs";
  const sessionId = state.projected.session_id;
  byId("session-link").hidden = !sessionId;
  if (sessionId) byId("session-link").href = waveDeckHandoff(sessionId, span);
  const interpretationLink = safeInterpretationLink(
    span.metadata?.interpretation_deep_link
      ?? state.record?.context?.interpretation_deep_link,
  );
  byId("interpretation-link").hidden = !interpretationLink;
  if (interpretationLink) byId("interpretation-link").href = interpretationLink;
  byId("play-selection").disabled = !state.audioUrl;
  if (updateUrl) {
    const url = new URL(location.href);
    url.search = focusQuery(span);
    history.replaceState({span_id:span.id}, "", url);
  }
  if (focusHeading) byId("selection-heading").focus();
  announce(`Selected ${span.label}; provenance and handoff controls are available.`);
  scheduleRender();
}

function safeInterpretationLink(link) {
  return typeof link === "string"
    && /^\/speech\/operate(?:\?|$)/.test(link) ? link : null;
}

function formatBoundary(boundary) {
  if (!boundary) return "Not declared";
  const lower = boundary.lower_frame ?? boundary.lower_ms;
  const estimate = boundary.estimate_frame ?? boundary.estimate_ms;
  const upper = boundary.upper_frame ?? boundary.upper_ms;
  return `${lower} ≤ ${estimate} ≤ ${upper} · ${boundary.method ?? "unknown method"}`;
}

async function load() {
  clearTimeout(state.poll);
  const runId = runIdFromRoute();
  try {
    if (!runId) {
      renderIndex((await request("/api/pipeline/runs")).runs);
      return;
    }
    let record;
    try {
      record = await request(`/api/pipeline/runs/${encodeURIComponent(runId)}`);
      if (record.session_id) {
        try { Object.assign(record, await request(`/api/timeline/sessions/${encodeURIComponent(record.session_id)}`)); } catch {}
      }
    } catch (runError) {
      const sessionRecord = await request(`/api/timeline/sessions/${encodeURIComponent(runId)}`).catch(() => { throw runError; });
      record = {...sessionRecord,status:"completed",run_id:sessionRecord.context?.run_id};
    }
    renderRecord(record);
    if (record.status === "running") state.poll = setTimeout(load, 750);
  } catch (error) {
    byId("run-view").hidden = true;
    byId("error").innerHTML = `${escapeHtml(error.message)} <a href="/runs">Open recent runs</a> or <a href="/studio/graphs/new">start from a graph</a>.`;
    announce("Run context could not be restored. Recovery links are available.");
    byId("page-title").focus();
  }
}

async function startCapture(file = null) {
  if (state.socket) return;
  state.liveEvents = []; state.liveSessionId = null;
  state.audioContext = new AudioContext();
  if (file) {
    state.audioUrl = URL.createObjectURL(file);
    const decoded = await state.audioContext.decodeAudioData(await file.arrayBuffer());
    const samples = decoded.getChannelData(0);
    openSocket(decoded.sampleRate, () => streamFileSamples(samples));
  } else {
    state.media = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1}});
    state.source = state.audioContext.createMediaStreamSource(state.media);
    state.processor = state.audioContext.createScriptProcessor(4096, 1, 1);
    openSocket(state.audioContext.sampleRate, () => {
      state.processor.onaudioprocess = event => {
        const samples = event.inputBuffer.getChannelData(0).slice();
        observeAudio(samples, state.audioContext.sampleRate);
        if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(samples.buffer);
      };
      state.source.connect(state.processor); state.processor.connect(state.audioContext.destination);
    });
  }
  byId("run-index").hidden = true;
  byId("run-view").hidden = false;
  renderRecord({status:"running",capture_state:"active",events:[],privacy:{audio_retained:false,biometric_speaker_data_retained:false}});
}

function openSocket(sampleRate, ready) {
  state.socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/asr/stream`);
  state.socket.binaryType = "arraybuffer";
  state.socket.onopen = () => state.socket.send(JSON.stringify({type:"open",schema_version:1,provider:"fixture",sample_rate_hz:sampleRate,channels:1,language:"en"}));
  state.socket.onmessage = async ({data}) => {
    const message = JSON.parse(data);
    if (message.type === "ready") {
      state.liveSessionId = message.session_id; ready();
      announce(`Capture active for ${message.session_id}. Raw audio is not retained.`);
    } else if (message.type === "recognition") {
      observeEvent(message.event);
    } else if (message.type === "error") {
      observeEvent({type:"error",data:{code:message.code,message:message.message,recoverable:false}});
      finishCapture("failed");
    } else if (message.type === "ended") {
      finishCapture(statusFromLiveEvents());
    }
  };
  state.socket.onerror = () => finishCapture("failed");
}

async function streamFileSamples(samples) {
  const chunk = 4096, rate = state.audioContext.sampleRate;
  for (let offset = 0; offset < samples.length && state.socket; offset += chunk) {
    const slice = samples.slice(offset, Math.min(samples.length, offset + chunk));
    observeAudio(slice, rate);
    state.socket.send(slice.buffer);
    await new Promise(resolve => setTimeout(resolve, Math.max(1, slice.length / rate * 250)));
  }
  if (state.socket?.readyState === WebSocket.OPEN) state.socket.send('{"type":"end"}');
}

function observeAudio(samples, rate) {
  const rms = Math.sqrt(samples.reduce((sum,value) => sum + value * value, 0) / Math.max(1,samples.length));
  observeEvent({type:"audio_chunk",data:{direction:"input",chunk_sequence:state.liveEvents.length,frame_count:samples.length,format:{sample_rate_hz:rate},metadata:{level_dbfs:20 * Math.log10(Math.max(rms,1e-6))}}});
}

function observeEvent(event) {
  state.liveEvents.push({event,received_at_ms:performance.now(),event_id:`live:${state.liveEvents.length}`});
  const session = sessionFromEvents(state.liveSessionId ?? `live:${Date.now()}`, state.liveEvents);
  const renderingSession = {...session,source_events:[]};
  state.record = {status:"running",capture_state:"active",session:renderingSession,events:state.liveEvents,context:{source:"Run Tracks live observation"},privacy:{audio_retained:false,biometric_speaker_data_retained:false}};
  state.projected = projectSessionTracks(state.record);
  updateViewport(state.follow); renderRecord(state.record);
}

async function finishCapture(status) {
  if (!state.liveSessionId && !state.socket) return;
  const sessionId = state.liveSessionId ?? `live:${Date.now()}`;
  const session = sessionFromEvents(sessionId, state.liveEvents);
  const record = {status,capture_state:"inactive",session,context:{source:"Run Tracks live observation"},privacy:{audio_retained:false,biometric_speaker_data_retained:false}};
  try {
    await request(`/api/timeline/sessions/${encodeURIComponent(sessionId)}`, {
      method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({schema_version:1,session,context:record.context}),
    });
  } catch (error) {
    announce(`Capture ended ${status}; the session could not be made durable: ${error.message}`);
  }
  teardownCapture(); renderRecord(record);
  history.replaceState({session_id:sessionId},"",`/runs/${encodeURIComponent(sessionId)}/tracks`);
}

function cancelCapture() {
  if (state.socket?.readyState === WebSocket.OPEN) state.socket.send('{"type":"cancel","reason":"operator cancelled Run Tracks capture"}');
  observeEvent({type:"cancelled",data:{reason:"operator cancelled"}});
}

function teardownCapture() {
  state.processor?.disconnect(); state.source?.disconnect(); state.media?.getTracks().forEach(track => track.stop());
  state.audioContext?.close(); state.socket?.close();
  state.socket = state.media = state.audioContext = state.source = state.processor = null;
}

function playSelection() {
  if (!state.audioUrl || !state.selected) return;
  const audio = byId("playback");
  audio.src = state.audioUrl; audio.currentTime = state.selected.start_ms / 1000; audio.loop = false;
  const stopAt = state.selected.end_ms / 1000;
  audio.ontimeupdate = () => {
    if (audio.currentTime >= stopAt) {
      if (byId("loop-selection").checked) audio.currentTime = state.selected.start_ms / 1000;
      else audio.pause();
    }
  };
  audio.play().catch(error => announce(`Playback unavailable: ${error.message}`));
}

function statusFromLiveEvents() {
  const type = state.liveEvents.at(-1)?.event?.type;
  return type === "cancelled" ? "cancelled" : type === "error" ? "failed" : "completed";
}
function spanCount(){ return state.projected.tracks.reduce((sum,track) => sum + track.spans.length,0); }
function formatTime(ms){ const seconds=Math.max(0,ms)/1000; return seconds<60?`${seconds.toFixed(seconds<10?2:1)}s`:`${Math.floor(seconds/60)}:${String(Math.floor(seconds%60)).padStart(2,"0")}`; }
function escapeHtml(value){ return String(value??"").replace(/[&<>"']/g,char=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char])); }
function scalar(value){ return typeof value==="object"&&value?String(Object.values(value)[0]):String(value??""); }

byId("start-microphone").onclick = () => startCapture().catch(error => announce(`Microphone unavailable: ${error.message}`));
byId("audio-file").onchange = event => event.target.files[0] && startCapture(event.target.files[0]).catch(error => announce(`Audio file unavailable: ${error.message}`));
byId("cancel-live").onclick = cancelCapture;
byId("follow").onclick = () => { state.follow=!state.follow;byId("follow").setAttribute("aria-pressed",String(state.follow));updateViewport(state.follow);scheduleRender(); };
byId("zoom").oninput = event => { state.zoom=Number(event.target.value);updateViewport(state.follow);scheduleRender(); };
byId("zoom-in").onclick = () => { byId("zoom").value=String(Math.min(40,state.zoom+1));byId("zoom").dispatchEvent(new Event("input")); };
byId("zoom-out").onclick = () => { byId("zoom").value=String(Math.max(1,state.zoom-1));byId("zoom").dispatchEvent(new Event("input")); };
byId("pan").oninput = event => { state.pan=Number(event.target.value);state.follow=false;byId("follow").setAttribute("aria-pressed","false");scheduleRender(); };
byId("speaker-filter").onchange = event => { state.speaker=event.target.value;scheduleRender(); };
byId("show-raw").onchange = scheduleRender; byId("show-normalized").onchange = scheduleRender;
byId("play-selection").onclick = playSelection;
addEventListener("pagehide", () => { clearTimeout(state.poll);teardownCapture();if(state.audioUrl)URL.revokeObjectURL(state.audioUrl); });
load();
