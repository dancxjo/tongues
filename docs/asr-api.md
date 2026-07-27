# ASR HTTP and WebSocket API

Tongues exposes the provider-neutral recognition runtime through three versioned
routes:

- `GET /api/asr/capabilities` discovers providers, models, source adapters,
  transport policy, and exact limits.
- `POST /api/asr/transcriptions` accepts a WAV file for offline recognition.
- `GET /api/asr/stream` upgrades to a bidirectional WebSocket carrying mono
  float32 PCM and the shared `StreamEvent` recognition contract.

The machine-readable HTTP contract is in
[`asr-api-openapi.yaml`](asr-api-openapi.yaml). The WebSocket portion is
described here because OpenAPI 3.1 does not define bidirectional message
semantics.

## File transcription

```bash
curl --fail-with-body \
  -H 'Content-Type: audio/wav' \
  --data-binary @recording.wav \
  'http://127.0.0.1:3000/api/asr/transcriptions?provider=whisper.cpp&language=en'
```

The response contains the committed transcript and the complete ordered event
list. `provider=fixture` is a deterministic contract-test provider; it is not a
speech model. The Python client in
[`../examples/asr_http_client.py`](../examples/asr_http_client.py) is directly
runnable with only the standard library.

## Live WebSocket protocol

Connect to `ws://127.0.0.1:3000/api/asr/stream`. Browser requests must have the
same origin as the server. The first frame is JSON:

```json
{
  "type": "open",
  "schema_version": 1,
  "provider": "fixture",
  "sample_rate_hz": 16000,
  "channels": 1,
  "language": "en"
}
```

The server replies with `ready`, including its session ID and limits. Each
subsequent binary WebSocket frame is little-endian float32 PCM. Recognition
messages have `type: "recognition"`, a monotonically increasing `sequence`, and
an `event` from the shared schema. Finish or cancel explicitly:

```json
{"type":"end"}
{"type":"cancel","reason":"user stopped capture"}
```

`ended` is terminal. Disconnect, cancellation, invalid state, format failure,
duration exhaustion, or a 30-second idle timeout removes the runtime session
and releases its admission permit. Incoming frames are consumed one at a time;
the advertised application queue capacity is one and chunks are capped at
256 KiB, so a slow client cannot create an unbounded application queue.

Reconnect/resume is deliberately unsupported in schema version 1. Supplying
`resume_session_id` returns `invalid_state`; clients open a new session after a
disconnect. WebRTC was evaluated but is not enabled: it would add signaling and
ICE infrastructure without improving the local same-origin deployment. Browser
`getUserMedia` capture sends float32 PCM through this WebSocket fallback; see
[`../examples/asr_websocket_client.html`](../examples/asr_websocket_client.html).

Audio retention is always disabled. Neither endpoint writes submitted PCM or
WAV content to disk. The error envelope distinguishes:

| Code | Meaning |
| --- | --- |
| `unsupported_format` | MIME type, WAV, channel, sample, or chunk encoding is unsupported |
| `unavailable_model` | provider/model is unknown or not installed |
| `unsupported_configuration` | language/control or offline/streaming mode mismatch |
| `invalid_state` | session ordering or resume contract violation |
| `timeout` | request or idle deadline elapsed |
| `capacity_exhausted` | file, duration, chunk, or concurrent-session limit |
