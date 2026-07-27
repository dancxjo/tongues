# ASR evaluation, observability, and privacy

`speaking::asr_evaluation` provides deterministic WER/CER, language-ID
accuracy, speaker-label error, timestamp error, partial churn, endpoint and
first-partial latency, real-time factor, peak memory, and dropped-audio metrics.
The CC0 fixture at `fixtures/asr/evaluation_v1.json` pins multilingual, accent,
code-switching, low-resource, names/numbers, noise, clipping, echo, overlap,
long-silence, long-speech, and malformed-stream cases.

Default structured traces contain only stage/event identity, opaque session and
sequence IDs, elapsed time, and counts. Raw audio, transcript content, speaker
features, prompts, and provider payloads are absent unless a separate visible
opt-in sink is configured. Browser microphone controls visibly change state;
CLI microphone capture prints an active-capture and no-retention notice.

| Data | Default retention | Remote boundary |
| --- | --- | --- |
| Microphone PCM | none | local ASR only |
| Transcript / speaker labels | in-memory session | sent only to the selected response provider after commit |
| Response prompt/context | in-memory turn history | Ollama receives it only when explicitly selected; deterministic mode stays local |
| Target/perceived audio features | active playback only | local verification only |

Whisper is offline-only in the current runtime and requires its separately
installed model/license. Fixture providers are deterministic test doubles, not
quality claims. Hardware performance numbers must be recorded with CPU/GPU,
model/checksum, sample geometry, cold/warm state, and fixture ID; scores from
different languages/providers are not normalized into a misleading universal
number.

## CI and recipe matrix

Workspace CI runs redistributable, credential-free tests for shared recognition
event ordering, file/stream ASR, multilingual routing, diarization, committed
normalization/parser integration, server transports, WaveDeck replay, Studio
templates, and deterministic conversation gating. Physical microphones, paid
APIs, and secrets are never required. Minimal CLI, advanced composition, server
clients, Studio recipes, the timeline workbench, and live conversation are
documented in `recognition-cli.md`, `speech-dataflow.md`, `asr-api.md`,
`speech-timeline.md`, and `live-conversation.md`.

## Fixture migration inventory

| Donor | Disposition |
| --- | --- |
| Listenbury VAD/breath-group semantics | ported into `tongues-audio` segmentation tests |
| Listenbury interruption viewer payload | replaced by reduced `listenbury_user_interrupts_v1.json` |
| Listenbury microphone transcription | replaced by bounded ASR WebSocket fixtures |
| WaveDeck alignment/edit operations | ported into schema-v1 timeline Rust/JS tests |
| Daringsby Whisper/speaker paths | replaced by backend ASR/diarization registries |
| Daringsby simulated inputs | represented by deterministic ASR/conversation providers |
| Mortar-Sea provenance/replay scenarios | ported into shared event and timeline provenance tests |
