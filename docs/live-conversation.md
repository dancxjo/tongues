# Live ASR → response → TTS conversation

Speech Studio’s Live workflow can accept typed turns or start a browser
microphone conversation. Microphone PCM streams through `/api/asr/stream`;
only `committed_segment` recognition events can become user messages. Partial
and revised text remains visibly provisional and never reaches the response
provider.

The credential-free path selects the fixture ASR provider, deterministic live
response provider, and any installed local speech recipe. Generated tokens are
shown immediately. The server commits safe sentence/clause chunks to TTS, and
the browser schedules each returned audio segment on one Web Audio clock before
generation completes.

Context is append-only between committed turns. If a committed external segment
arrives during an active response, the documented barge-in policy cancels all
generation/synthesis/playback stages and starts a new turn; it never mutates an
active prompt. Microphone activity alone cannot interrupt. Browser echo
cancellation is enabled, and committed text matching the active spoken output
is classified as likely self-speech and ignored. Ambiguous or unstable evidence
waits.

The turn journal distinguishes generation, planning, output requested/started,
playback acknowledged, interruption/cancellation, and completion. It reports
auditory detection, segmentation/final ASR, first partial, LLM first token,
speech planning, first TTS audio, and playback latency separately. Conservative
fillers/backchannels are absent and therefore disabled in deterministic tests.
