export const TRACK_ORDER = [
  "audio_input", "vad", "speakers", "transcript_raw", "words", "phones", "phonemes", "transcript_normalized",
  "language", "generation", "tts", "pipeline", "latency", "errors",
];

const TRACK_LABELS = {
  audio_input: "Input audio",
  vad: "Speech / VAD",
  speakers: "Speakers",
  transcript_raw: "Transcript · raw",
  words: "Words",
  phones: "Phones",
  phonemes: "Phonemes",
  transcript_normalized: "Transcript · normalized",
  language: "Language",
  generation: "Generated text",
  tts: "Synthesis / playback",
  pipeline: "Pipeline events",
  latency: "Stage latency",
  errors: "Errors",
};

export function projectSessionTracks(source) {
  const record = source ?? {};
  const session = record.session ?? null;
  const context = record.context ?? {};
  const events = eventEntries([...(record.events ?? []), ...(session?.source_events ?? [])]);
  const evidence = Array.isArray(session?.evidence) ? session.evidence : [];
  const attachments = Array.isArray(session?.attachments) ? session.attachments : [];
  const tracks = new Map(TRACK_ORDER.map(id => [id, {id, label: TRACK_LABELS[id], spans: []}]));
  const segmentState = new Map();
  const speakerHistory = new Map();
  const speakerLabels = new Map();
  const eventIndex = new Map();
  const sourceConsumers = new Map();
  let inferredCursor = 0;

  const ensureSegment = id => {
    const key = scalar(id) || `segment:${segmentState.size}`;
    if (!segmentState.has(key)) segmentState.set(key, {id: key, start_ms: null, end_ms: null});
    return segmentState.get(key);
  };
  const add = (track, span) => {
    const start = finite(span.start_ms, inferredCursor);
    const end = Math.max(start + 1, finite(span.end_ms, start + 1));
    inferredCursor = Math.max(inferredCursor, end);
    tracks.get(track).spans.push({
      id: span.id || `${track}:${tracks.get(track).spans.length}`,
      track, start_ms: start, end_ms: end, label: span.label || TRACK_LABELS[track],
      status: span.status || "observed", event_id: span.event_id ?? null,
      segment_id: span.segment_id ?? null, speaker: span.speaker ?? null,
      revision: span.revision ?? 0, overlap: Boolean(span.overlap),
      metadata: span.metadata ?? {}, provenance: span.provenance ?? {},
    });
  };

  for (const entry of events) {
    const event = entry.event;
    const data = event.data ?? event;
    const type = event.type ?? event.kind ?? data.type ?? "pipeline_event";
    const eventId = scalar(entry.event_id) || `event:${entry.sequence}`;
    const time = eventTime(entry, inferredCursor);
    eventIndex.set(eventId, entry);
    for (const sourceRef of entry.provenance?.sources ?? []) {
      const sourceId = scalar(sourceRef.event_id);
      if (sourceId) {
        const consumers = sourceConsumers.get(sourceId) ?? [];
        consumers.push(eventId);
        sourceConsumers.set(sourceId, consumers);
      }
    }
    const segmentId = scalar(data.segment_id);
    const segment = segmentId ? ensureSegment(segmentId) : null;

    if (type === "speech_started" && segment) {
      segment.start_ms = time;
      add("vad", evidenceSpan(entry, time, time + 1, "Speech started", {segment_id: segmentId}));
    } else if (type === "speech_ended" && segment) {
      segment.end_ms = Math.max(time, (segment.start_ms ?? time - 1) + 1);
      add("vad", evidenceSpan(entry, segment.start_ms ?? time - 1, segment.end_ms, `Speech · ${data.reason ?? "ended"}`, {segment_id: segmentId}));
    } else if (type === "audio_chunk") {
      const direction = String(data.direction ?? "input");
      const duration = audioDurationMs(data);
      add(direction === "output" ? "tts" : "audio_input", evidenceSpan(
        entry, time, time + duration,
        direction === "output" ? `TTS chunk ${data.chunk_sequence ?? ""}` : audioLabel(data),
        {segment_id: segmentId, metadata: data.metadata ?? {}},
      ));
    } else if (type === "partial_hypothesis" || type === "revised_hypothesis") {
      const track = textTrack(data.role);
      const revision = (segment?.revisions ?? 0) + (type === "revised_hypothesis" ? 1 : 0);
      if (segment) segment.revisions = revision;
      add(track, evidenceSpan(entry, segment?.start_ms ?? time, time + 1, data.text || "Partial text", {
        segment_id: segmentId, revision, status: type === "revised_hypothesis" ? "revised" : "provisional",
        metadata: {role: data.role, replaces: data.replaces ?? null},
      }));
    } else if (type === "committed_segment") {
      const [start, end] = wordBounds(data.words, segment?.start_ms ?? time, time + 1);
      if (segment) Object.assign(segment, {start_ms: start, end_ms: end, text: data.text});
      add(textTrack(data.role), evidenceSpan(entry, start, end, data.text || "Committed text", {
        segment_id: segmentId, status: "committed",
        metadata: {role: data.role, language: data.language?.language ?? null},
      }));
      if (data.language?.language) {
        add("language", evidenceSpan(entry, start, end, data.language.language, {segment_id: segmentId}));
      }
      if (data.speaker_id && segmentId) assignSpeaker(segmentId, data.speaker_id, entry, time);
    } else if (type === "language_hypothesis") {
      add("language", evidenceSpan(entry, segment?.start_ms ?? time, segment?.end_ms ?? time + 1,
        data.hypothesis?.language ?? "Unknown language", {segment_id: segmentId, status: "provisional"}));
    } else if (type === "speaker_assigned" && segmentId) {
      assignSpeaker(segmentId, data.speaker_id, entry, time);
    } else if (type === "derived_artifact") {
      const normalized = /normali[sz]|transcript/i.test(data.stage ?? "");
      const label = typeof data.value === "string" ? data.value : data.value?.text ?? data.stage;
      add(normalized ? "transcript_normalized" : "pipeline",
        evidenceSpan(entry, time, time + 1, label || "Derived artifact", {segment_id: segmentId}));
    } else if (type === "token_timing" || type === "text_completed") {
      const token = data.token;
      const [start, end] = token?.range ? [token.range.start_ms, token.range.end_ms] : [time, time + 1];
      add("generation", evidenceSpan(entry, start, end, token?.text ?? data.text ?? "Generated text", {segment_id: segmentId}));
    } else if (type.startsWith("output_")) {
      add("tts", evidenceSpan(entry, time, time + 1, type.replaceAll("_", " "), {status: terminalStatus(type)}));
    } else if (type === "warning" || type === "error" || type === "cancelled") {
      add("errors", evidenceSpan(entry, time, time + 1, data.message ?? data.reason ?? data.code ?? type, {
        status: type === "cancelled" ? "cancelled" : type,
      }));
    } else {
      const start = finite(entry.raw.elapsed_ms, time);
      const previous = tracks.get("pipeline").spans.at(-1);
      const latency = previous ? Math.max(0, start - previous.start_ms) : start;
      add("pipeline", evidenceSpan(entry, start, start + 1,
        `${entry.raw.node_id ?? "graph"} · ${entry.raw.kind ?? type}`, {
          status: terminalStatus(entry.raw.kind), metadata: {node_id: entry.raw.node_id, detail: entry.raw.detail},
        }));
      add("latency", evidenceSpan(entry, Math.max(0, start - latency), start || 1,
        `${entry.raw.node_id ?? "graph"} · ${latency} ms`, {metadata: {latency_ms: latency}}));
    }
  }

  function assignSpeaker(segmentId, speakerId, entry, time) {
    const segment = ensureSegment(segmentId);
    const history = speakerHistory.get(segmentId) ?? [];
    const label = consentedSpeakerLabel(speakerId, entry.provenance, speakerLabels);
    history.push({speaker_id: scalar(speakerId), label, event_id: scalar(entry.event_id), at_ms: time});
    speakerHistory.set(segmentId, history);
    segment.speaker = label;
  }

  for (const span of evidence) {
    const metadata = span.metadata ?? {};
    const track = evidenceTrack(span);
    if (!track) continue;
    const label = metadata.text ?? metadata.language ?? metadata.label ?? span.modality;
    if (tracks.get(track).spans.some(existing =>
      existing.start_ms === span.start_ms && existing.end_ms === span.end_ms && existing.label === label)) {
      continue;
    }
    const speaker = metadata.speaker_id ? consentedSpeakerLabel(metadata.speaker_id, {attributes: metadata}, speakerLabels) : null;
    add(track, {
      id: span.id, start_ms: span.start_ms, end_ms: span.end_ms,
      label,
      status: metadata.status ?? "committed", segment_id: metadata.segment_id ?? segmentFromSpanId(span.id),
      speaker, metadata, provenance: metadata.provenance ?? {},
    });
  }

  for (const [segmentId, history] of speakerHistory) {
    const segment = ensureSegment(segmentId);
    const current = history.at(-1);
    add("speakers", {
      id: `speaker:${segmentId}:${history.length - 1}`,
      start_ms: segment.start_ms ?? current.at_ms,
      end_ms: segment.end_ms ?? current.at_ms + 1,
      label: history.length > 1 ? `${current.label} · revised ${history.length - 1}×` : current.label,
      segment_id: segmentId, speaker: current.label, revision: history.length - 1,
      status: history.length > 1 ? "revised" : "observed",
      event_id: current.event_id, metadata: {history, identity_authority: "diarization_only"},
    });
  }
  markSpeakerOverlaps(tracks.get("speakers").spans);

  for (const track of tracks.values()) {
    track.spans.sort((a, b) => a.start_ms - b.start_ms || a.end_ms - b.end_ms || a.id.localeCompare(b.id));
  }
  const duration_ms = Math.max(1, ...[...tracks.values()].flatMap(track => track.spans.map(span => span.end_ms)));
  const status = record.status ?? statusFromEvents(events) ?? (session ? "completed" : "unknown");
  const relatedSpanIds = alignmentRelations(session?.alignments ?? []);
  const segmentation = segmentationState(attachments);
  return {
    schema_version: 1,
    run_id: record.run_id ?? context.run_id ?? null,
    session_id: session?.session_id ?? record.session_id ?? null,
    graph_id: record.graph_id ?? context.graph_id ?? null,
    status, duration_ms,
    privacy: privacyState(record, session),
    segmentation,
    tracks: TRACK_ORDER.map(id => tracks.get(id)),
    eventIndex, sourceConsumers, relatedSpanIds,
  };
}

export function selectionProvenance(projected, span) {
  const entry = span?.event_id ? projected.eventIndex.get(span.event_id) : null;
  const provenance = entry?.provenance ?? span?.provenance ?? {};
  return {
    event_id: span?.event_id ?? null,
    segment_id: span?.segment_id ?? null,
    graph_id: projected.graph_id,
    graph_node_id: span?.metadata?.node_id ?? provenance.attributes?.graph_node_id ?? null,
    provider: provenance.provider ?? span?.metadata?.alignment_provider ?? null,
    model: provenance.model ?? span?.metadata?.alignment_model ?? null,
    version: span?.metadata?.alignment_version ?? null,
    artifact_id: span?.metadata?.artifact_id ?? null,
    algorithm_version: span?.metadata?.algorithm_version ?? null,
    recipe_id: span?.metadata?.recipe_id ?? null,
    execution_record_id: span?.metadata?.execution_record_id ?? null,
    audio_artifact_id: span?.metadata?.audio_artifact_id ?? null,
    boundary_origin: span?.metadata?.boundary_origin ?? null,
    confidence: span?.metadata?.confidence ?? null,
    sources: provenance.sources ?? [],
    downstream_event_ids: span?.event_id ? projected.sourceConsumers.get(span.event_id) ?? [] : [],
    authority: span?.track === "speakers"
      ? "diarization label, not verified identity"
      : ["phones","phonemes"].includes(span?.track)
        ? `${span.metadata?.boundary_origin ?? "unknown"} boundary · ${span.status ?? "unknown"}`
        : provenance.kind ?? span?.metadata?.evidence_authority ?? "observed",
  };
}

export function relatedSelectionIds(projected, span) {
  if (!span) return new Set();
  return new Set([span.id, ...(projected.relatedSpanIds.get(span.id) ?? [])]);
}

export function boundedVisibleSpans(track, range, limit = 600) {
  const visible = track.spans.filter(span => span.end_ms >= range.start_ms && span.start_ms <= range.end_ms);
  if (visible.length <= limit) return visible;
  const stride = Math.ceil(visible.length / limit);
  return visible.filter((_, index) => index % stride === 0 || index === visible.length - 1);
}

export function spanDensity(span, range, viewportPx = 1_000) {
  const duration = Math.max(1, range.end_ms - range.start_ms);
  const pixels = Math.max(0, span.end_ms - span.start_ms) / duration * viewportPx;
  if (pixels < 8) return "tick";
  if (pixels < 28) return "symbol";
  return "label";
}

export function waveDeckHandoff(sessionId, span) {
  if (!sessionId || !span) return null;
  const query = new URLSearchParams({
    span: span.segment_id ?? span.id,
    start_ms: String(span.start_ms),
    end_ms: String(span.end_ms),
  });
  return `/sessions/${encodeURIComponent(sessionId)}/correct?${query}`;
}

function eventEntries(values) {
  return values.map((raw, sequence) => {
    const envelope = raw?.event && (raw.schema_version || raw.times || raw.provenance || raw.received_at_ms != null);
    return {
      raw, event: envelope ? raw.event : raw,
      event_id: envelope ? raw.event_id : raw.event_id,
      sequence: finite(raw.sequence, sequence),
      times: envelope ? raw.times : raw.times,
      provenance: raw.provenance ?? {},
      received_at_ms: raw.received_at_ms,
    };
  });
}

function eventTime(entry, fallback) {
  const occurred = entry.times?.occurred_at?.offset_ms;
  return Math.max(0, finite(occurred, finite(entry.received_at_ms, finite(entry.raw.elapsed_ms, fallback))));
}

function evidenceSpan(entry, start_ms, end_ms, label, extra = {}) {
  return {
    id: `${scalar(entry.event_id) || `event:${entry.sequence}`}:${extra.segment_id ?? ""}`,
    start_ms, end_ms, label, event_id: scalar(entry.event_id) || null,
    provenance: entry.provenance, ...extra,
  };
}

function audioDurationMs(data) {
  const rate = data.format?.sample_rate_hz ?? data.metadata?.sample_rate_hz;
  return rate ? Math.max(1, Math.round(finite(data.frame_count, 1) * 1_000 / rate)) : 20;
}

function audioLabel(data) {
  const level = data.metadata?.level_dbfs;
  return Number.isFinite(level) ? `${Math.round(level)} dBFS` : `Audio chunk ${data.chunk_sequence ?? ""}`;
}

function textTrack(role) {
  if (role === "generation") return "generation";
  if (role === "normalized") return "transcript_normalized";
  return "transcript_raw";
}

function wordBounds(words, fallbackStart, fallbackEnd) {
  const valid = (words ?? []).filter(word => Number.isFinite(word?.range?.start_ms) && Number.isFinite(word?.range?.end_ms));
  return valid.length ? [valid[0].range.start_ms, valid.at(-1).range.end_ms] : [fallbackStart, Math.max(fallbackStart + 1, fallbackEnd)];
}

function evidenceTrack(span) {
  if (span.modality === "audio") return span.metadata?.direction === "output" ? "tts" : "audio_input";
  if (span.modality === "transcript") return span.metadata?.role === "normalized" ? "transcript_normalized" : "transcript_raw";
  if (span.modality === "word") return "words";
  if (span.modality === "phone") return "phones";
  if (span.modality === "phoneme") return "phonemes";
  if (span.modality === "speaker") return "speakers";
  if (span.modality === "playback") return "tts";
  if (span.modality === "interruption" || span.modality === "breath_group") return "vad";
  return null;
}

function alignmentRelations(alignments) {
  const relations = new Map();
  const link = (left, right) => {
    if (!left || !right) return;
    const values = relations.get(left) ?? new Set();
    values.add(right);
    relations.set(left, values);
  };
  for (const alignment of alignments) {
    link(alignment.source_span_id, alignment.target_span_id);
    link(alignment.target_span_id, alignment.source_span_id);
  }
  return new Map([...relations].map(([id, values]) => [id, [...values]]));
}

function segmentationState(attachments) {
  const segmentations = attachments.filter(attachment => attachment?.kind === "phonetic_segmentation");
  if (!segmentations.length) {
    return {
      available: false,
      readiness: "missing",
      message: "No phone/phoneme segmentation artifact is attached; no alignment is implied.",
      artifacts: [],
    };
  }
  const artifacts = segmentations.map(attachment => ({
    artifact_id: attachment.artifact_id,
    readiness: attachment.payload?.readiness ?? "unsupported",
    algorithm_version: attachment.payload?.algorithm_version ?? "unknown",
    recipe_id: attachment.payload?.graph?.recipe_id ?? null,
    issues: attachment.payload?.issues ?? [],
    missing_segments: (attachment.payload?.segments ?? []).filter(segment => !segment.interval),
  }));
  const readiness = artifacts.some(artifact => artifact.readiness === "partial")
    ? "partial"
    : artifacts.every(artifact => artifact.readiness === "ready") ? "ready" : "unsupported";
  return {
    available: true,
    readiness,
    message: readiness === "ready"
      ? "Phone/phoneme timing is backed by attached alignment evidence."
      : `${readiness} segmentation: untimed or unsupported rows remain explicit and are not drawn as authoritative spans.`,
    artifacts,
  };
}

function consentedSpeakerLabel(rawId, provenance, labels) {
  const id = scalar(rawId) || "unknown";
  const attrs = provenance?.attributes ?? {};
  if (attrs.identity_consent === true && typeof attrs.enrolled_display_name === "string") return attrs.enrolled_display_name;
  if (!labels.has(id)) labels.set(id, `Speaker ${String.fromCharCode(65 + (labels.size % 26))}${labels.size >= 26 ? Math.floor(labels.size / 26) : ""}`);
  return labels.get(id);
}

function markSpeakerOverlaps(spans) {
  for (let left = 0; left < spans.length; left += 1) {
    for (let right = left + 1; right < spans.length; right += 1) {
      if (spans[right].start_ms >= spans[left].end_ms) break;
      if (spans[left].speaker !== spans[right].speaker && spans[left].start_ms < spans[right].end_ms) {
        spans[left].overlap = true;
        spans[right].overlap = true;
      }
    }
  }
}

function terminalStatus(type = "") {
  if (/cancel|abort|interrupt/.test(type)) return "cancelled";
  if (/complete|finish|succeed/.test(type)) return "completed";
  if (/error|fail/.test(type)) return "error";
  return "observed";
}

function statusFromEvents(entries) {
  for (const entry of [...entries].reverse()) {
    const type = entry.event?.type ?? entry.raw?.kind;
    if (type === "cancelled") return "cancelled";
    if (type === "error" || type === "failed") return "failed";
    if (type === "completed" || type === "succeeded") return "completed";
  }
  return entries.length ? "running" : null;
}

function privacyState(record, session) {
  const retention = record.retention ?? record.privacy ?? {};
  const rawAudioPresent = (session?.evidence ?? []).some(span => span.modality === "audio" && span.metadata?.audio_base64);
  return {
    capture: record.capture_state ?? (record.status === "running" ? "active" : "inactive"),
    raw_audio_retained: Boolean(retention.audio_retained ?? rawAudioPresent),
    biometric_speaker_data_retained: Boolean(retention.biometric_speaker_data_retained),
    policy: retention.policy ?? "No raw audio or biometric speaker data is retained by default.",
  };
}

function segmentFromSpanId(id = "") {
  const parts = String(id).split(":");
  return parts.length > 1 ? parts.slice(1).join(":") : null;
}

function scalar(value) {
  if (value == null) return null;
  if (typeof value === "string" || typeof value === "number") return String(value);
  if (typeof value === "object" && Object.keys(value).length === 1) return scalar(Object.values(value)[0]);
  return null;
}

function finite(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}
