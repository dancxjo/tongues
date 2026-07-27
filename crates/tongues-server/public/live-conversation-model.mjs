export function classifyCommittedInput(text, spokenText) {
  const normalize = value => String(value ?? "").toLocaleLowerCase().replace(/[^\p{L}\p{N}]+/gu," ").trim();
  const input=normalize(text), output=normalize(spokenText);
  if (!input) return "empty";
  if (output && (output.includes(input) || input.includes(output.slice(-Math.min(output.length,64))))) return "likely_self_speech";
  return "likely_external_speech";
}

export function committedTurnAction({event, playbackActive, bargeIn, spokenText}) {
  if (event?.type !== "committed_segment" || event?.data?.role !== "recognition") return {action:"wait",reason:"unstable or non-recognition event"};
  const classification=classifyCommittedInput(event.data.text,spokenText);
  if (classification==="likely_self_speech") return {action:"ignore_echo",reason:"committed text matches current output"};
  if (playbackActive && !bargeIn) return {action:"wait",reason:"barge-in is disabled"};
  return {action:playbackActive?"cancel_and_restart":"start_turn",reason:"committed external speech"};
}

export function latencySnapshot(times) {
  const delta=(end,start)=>end != null && start != null ? Math.max(0,end-start):null;
  return {
    auditory_detection_ms:delta(times.speechStarted,times.captureStarted),
    segmentation_ms:delta(times.committed,times.speechStarted),
    asr_first_partial_ms:delta(times.firstPartial,times.speechStarted),
    llm_first_token_ms:delta(times.firstGenerated,times.committed),
    speech_planning_ms:delta(times.firstPlanned,times.firstGenerated),
    tts_first_audio_ms:delta(times.firstAudio,times.firstPlanned),
    playback_ms:delta(times.playbackCompleted,times.firstAudio),
  };
}
