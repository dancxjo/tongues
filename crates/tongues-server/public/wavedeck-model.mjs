export const SCHEMA_VERSION = 1;

export function validateSession(value) {
  if (!value || value.schema_version !== SCHEMA_VERSION) {
    throw new Error(`Timeline schema ${value?.schema_version ?? "missing"} is unsupported; expected ${SCHEMA_VERSION}.`);
  }
  if (!value.session_id || !Array.isArray(value.evidence) || !Array.isArray(value.operations ?? [])) {
    throw new Error("Timeline session is missing session_id, evidence, or operations.");
  }
  const ids = new Set();
  for (const span of value.evidence) {
    if (!span.id || !Number.isFinite(span.start_ms) || !Number.isFinite(span.end_ms) || span.end_ms <= span.start_ms) {
      throw new Error(`Invalid evidence span ${span?.id ?? "without ID"}.`);
    }
    if (ids.has(span.id)) throw new Error(`Duplicate evidence span ${span.id}.`);
    ids.add(span.id);
  }
  const attachmentIds = new Set();
  for (const attachment of value.attachments ?? []) {
    if (!attachment?.artifact_id || attachment.kind !== "phonetic_segmentation" || !attachment.payload) {
      throw new Error("Timeline contains an invalid phonetic segmentation attachment.");
    }
    if (attachmentIds.has(attachment.artifact_id)) {
      throw new Error(`Duplicate timeline attachment ${attachment.artifact_id}.`);
    }
    attachmentIds.add(attachment.artifact_id);
  }
  return value;
}

export function projectSession(session) {
  validateSession(session);
  const enabled = new Map();
  for (const op of session.operations ?? []) {
    if (op.kind === "undo") {
      if (!enabled.has(op.target_operation_id)) throw new Error(`Undo target ${op.target_operation_id} is unknown.`);
      enabled.set(op.target_operation_id, false);
    } else if (op.kind === "redo") {
      if (!enabled.has(op.target_operation_id)) throw new Error(`Redo target ${op.target_operation_id} is unknown.`);
      enabled.set(op.target_operation_id, true);
    } else {
      enabled.set(op.operation_id, true);
    }
  }
  const edited = new Map(session.evidence.map(span => [span.id, structuredClone(span)]));
  const audio_region_edits = [];
  const applied_operation_ids = [];
  for (const op of session.operations ?? []) {
    if (op.kind === "undo" || op.kind === "redo" || !enabled.get(op.operation_id)) continue;
    applyOperation(edited, op, audio_region_edits);
    applied_operation_ids.push(op.operation_id);
  }
  return {
    schema_version: session.schema_version,
    session_id: session.session_id,
    original: structuredClone(session.evidence),
    edited: [...edited.values()].sort((a, b) => a.start_ms - b.start_ms || a.id.localeCompare(b.id)),
    alignments: structuredClone(session.alignments ?? []),
    attachments: structuredClone(session.attachments ?? []),
    applied_operation_ids,
    audio_region_edits,
  };
}

function applyOperation(spans, op, audioEdits) {
  const span = op.span_id ? spans.get(op.span_id) : null;
  if (op.span_id && !span) throw new Error(`Edit target ${op.span_id} is unknown.`);
  switch (op.kind) {
    case "transcript_replace":
      if (!["transcript", "word"].includes(span.modality)) throw new Error("Transcript edits require text spans.");
      span.metadata = {...span.metadata, text: String(op.text ?? "")};
      markCorrected(span, op, "transcript");
      break;
    case "phonetic_symbol_replace":
      if (!["phone", "phoneme"].includes(span.modality)) throw new Error("Phonetic symbol edits require phone or phoneme spans.");
      if (!String(op.symbol ?? "").trim()) throw new Error("A corrected phone/phoneme symbol is required.");
      span.metadata = {...span.metadata, symbol: String(op.symbol)};
      markCorrected(span, op, "symbol");
      break;
    case "alignment_move_boundary":
      if (op.boundary === "start" && op.new_time_ms < span.end_ms) span.start_ms = op.new_time_ms;
      else if (op.boundary === "end" && op.new_time_ms > span.start_ms) span.end_ms = op.new_time_ms;
      else throw new Error("Boundary edit would create a zero-duration span.");
      markCorrected(span, op, "boundary");
      break;
    case "annotate":
      span.metadata = {...span.metadata, [`annotation:${op.key}`]: op.value};
      break;
    case "segment_split": {
      if (!(op.split_at_ms > span.start_ms && op.split_at_ms < span.end_ms)) throw new Error("Split is outside the span.");
      spans.delete(span.id);
      spans.set(op.left_span_id, {...structuredClone(span), id: op.left_span_id, end_ms: op.split_at_ms});
      spans.set(op.right_span_id, {...structuredClone(span), id: op.right_span_id, start_ms: op.split_at_ms});
      break;
    }
    case "audio_region":
      if (span.modality !== "audio") throw new Error("Audio-region edits require audio evidence.");
      audioEdits.push(structuredClone(op));
      break;
    default:
      throw new Error(`Unsupported operation ${op.kind}.`);
  }
}

function markCorrected(span, op, correctionKind) {
  span.metadata = {
    ...span.metadata,
    ...(correctionKind === "boundary" ? {boundary_origin: "corrected"} : {}),
    correction_kind: correctionKind,
    correction_operation_id: op.operation_id,
    correction_actor: op.provenance?.actor ?? "unknown",
    correction_at_ms: op.provenance?.at_ms ?? null,
  };
}

export function appendOperation(session, kind, fields, actor = "browser-operator") {
  const operation_id = `edit:${Date.now()}:${session.operations.length}`;
  const op = {
    operation_id,
    provenance: {
      origin: "manual",
      actor,
      at_ms: Date.now(),
      source_span_ids: fields.span_id ? [fields.span_id] : [],
      source_event_ids: [],
      reason: fields.reason ?? null,
    },
    kind,
    ...fields,
  };
  session.operations.push(op);
  projectSession(session);
  return op;
}

export function undo(session) {
  const projection = projectSession(session);
  const target = projection.applied_operation_ids.at(-1);
  if (!target) return false;
  appendOperation(session, "undo", {target_operation_id: target});
  return true;
}

export function redo(session) {
  const disabled = new Set();
  for (const op of session.operations) {
    if (op.kind === "undo") disabled.add(op.target_operation_id);
    if (op.kind === "redo") disabled.delete(op.target_operation_id);
  }
  const target = [...disabled].at(-1);
  if (!target) return false;
  appendOperation(session, "redo", {target_operation_id: target});
  return true;
}

export function sessionFromEvents(sessionId, events) {
  const evidence = [];
  const segmentOccurrences = new Map();
  for (const envelope of events) {
    const event = envelope.event ?? envelope;
    if (event.type !== "committed_segment") continue;
    const data = event.data ?? event;
    const segment = data.segment_id ?? `segment-${evidence.length}`;
    const occurrence = (segmentOccurrences.get(segment) ?? 0) + 1;
    segmentOccurrences.set(segment, occurrence);
    const evidenceSegment = occurrence === 1 ? segment : `${segment}:occurrence-${occurrence}`;
    const words = data.words ?? [];
    const start = words[0]?.range?.start_ms ?? envelope.received_at_ms ?? 0;
    const end = words.at(-1)?.range?.end_ms ?? Math.max(start + 1, envelope.received_at_ms ?? start + 1);
    const transcriptId = `transcript:${evidenceSegment}`;
    evidence.push({
      id: transcriptId, start_ms: start, end_ms: end, modality: "transcript",
      metadata: {text: data.text, language: data.language?.language ?? null, speaker_id: data.speaker_id ?? null},
    });
    words.forEach((word, index) => evidence.push({
      id: `word:${evidenceSegment}:${index}`, start_ms: word.range.start_ms, end_ms: word.range.end_ms,
      modality: "word", metadata: {text: word.text},
    }));
  }
  return validateSession({
    schema_version: SCHEMA_VERSION,
    session_id: sessionId,
    evidence,
    alignments: [],
    attachments: [],
    source_events: events.map(item => item.event ?? item),
    operations: [],
  });
}

export function relatedSpanIds(session, spanId) {
  const related = new Set([spanId]);
  for (const alignment of session?.alignments ?? []) {
    if (alignment.source_span_id === spanId) related.add(alignment.target_span_id);
    if (alignment.target_span_id === spanId) related.add(alignment.source_span_id);
  }
  return related;
}

export function focusedSessionSpan(session, search = "") {
  const params = new URLSearchParams(search);
  const requested = params.get("span");
  const start = Number(params.get("start_ms"));
  const end = Number(params.get("end_ms"));
  const spans = projectSession(session).edited;
  return spans.find(candidate =>
    candidate.id === requested || candidate.id.endsWith(`:${requested}`))
    ?? spans.find(candidate =>
      Number.isFinite(start) && Number.isFinite(end)
      && candidate.start_ms < end && candidate.end_ms > start)
    ?? null;
}

export function segmentationState(session) {
  const artifacts = (session?.attachments ?? [])
    .filter(attachment => attachment.kind === "phonetic_segmentation")
    .map(attachment => ({
      artifact_id: attachment.artifact_id,
      readiness: attachment.payload?.readiness ?? "unsupported",
      algorithm_version: attachment.payload?.algorithm_version ?? "unknown",
      recipe_id: attachment.payload?.graph?.recipe_id ?? null,
      issues: attachment.payload?.issues ?? [],
      missing_segments: (attachment.payload?.segments ?? []).filter(segment => !segment.interval),
    }));
  if (!artifacts.length) {
    return {
      available: false,
      readiness: "missing",
      message: "No segmentation artifact is attached; this session does not claim phone-level alignment.",
      artifacts,
    };
  }
  const readiness = artifacts.some(artifact => artifact.readiness === "partial")
    ? "partial"
    : artifacts.every(artifact => artifact.readiness === "ready") ? "ready" : "unsupported";
  return {
    available: true,
    readiness,
    message: readiness === "ready"
      ? "Attached phone/phoneme timing is inspectable as immutable evidence."
      : `${readiness} segmentation stays explicit; untimed rows are not rendered as authoritative boundaries.`,
    artifacts,
  };
}

export function waveformPolylinePoints(peaks, width = 240, height = 36) {
  if (!Array.isArray(peaks) || !peaks.length) return "";
  const center = height / 2;
  const step = peaks.length === 1 ? 0 : width / (peaks.length - 1);
  return peaks.map((peak, index) => {
    const amplitude = Math.max(0, Math.min(1, Number(peak) || 0));
    const y = center - amplitude * (center - 1);
    return `${(index * step).toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
}
