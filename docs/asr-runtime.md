# Provider-neutral ASR runtime

The `speaking` crate owns the ASR provider, model, session, and event contracts.
CLI, server, and Speech Studio code can inspect `AsrProviderCapabilities` and
drive `AsrRuntime`; they do not need provider-specific conditionals.

Each capability record names the provider and exact model, installation state,
languages, streaming mode, decoding controls, session capacity, estimated
memory, and optional license/checksum metadata. Model catalogs can populate
those fields directly, keeping installation and license data authoritative
outside frontends.

Streaming support is explicit:

- `native` means the adapter emits genuinely incremental results.
- `chunked_simulation` includes its window and overlap and must not be presented
  as native streaming.
- `offline_only` buffers input until finalization.

The local Whisper.cpp adapter is deliberately advertised as `offline_only`.
It validates and loads the installed model, applies an optional language
constraint, emits the shared recognition event contract, and unloads cleanly.
The deterministic fixture provider exercises native partial, revision, commit,
completion, error, and cancellation behavior without a model download.

## Lifecycle and resource behavior

Providers are registered, loaded, used for one or more bounded sessions, and
unloaded. A provider cannot be unloaded while a session is active. Both
per-provider concurrent-session limits and runtime-wide session/estimated-memory
limits fail with `AsrRuntimeError::ResourceExhausted`.

Session configuration is validated against provider capabilities before
allocation. Unsupported languages and decoding controls are typed errors.
Current portable decoding controls cover beam width, temperature, prompt,
timestamps, punctuation, and vocabulary bias; an adapter advertises only those
it actually implements.

`push_audio` emits partial/revised/committed shared `StreamEvent` values.
`finish_session` emits the final commit and completion, while `cancel_session`
emits cancellation and releases the session. Offline transcription drives the
same session API and event sequence. If an offline push fails, cleanup is
attempted without replacing the primary provider error.

## Adding providers

Implement `AsrProvider` for model lifecycle and capability discovery, then
return an `AsrSession` for inference. Registering a second provider does not
change runtime, CLI, server, or Studio contracts. Provider-specific settings
remain capability-gated rather than becoming frontend branches.

Focused fixtures verify deterministic event ordering and transcript assembly,
load/unload behavior, cancellation, invalid state, unsupported controls and
languages, resource exhaustion, cleanup after failure, honest chunked-stream
labeling, offline/streaming contract reuse, and unchanged registration of a
second provider.
