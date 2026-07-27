# Audio input

`tongues-audio` owns the provider-neutral input boundary used before VAD and
recognition. Every adapter implements `AudioSource` and yields the same three
events:

- decoded interleaved `f32` PCM with a source sequence and optional absolute
  frame position;
- an explicit discontinuity for dropped, duplicated, out-of-order, reconnected,
  or frame-timeline input;
- end of stream.

`WavAudioSource` accepts files or in-memory fixtures.
`PcmReaderSource` accepts raw signed-16 or float-32 little-endian PCM from
files, stdin, `TcpStream`, `UnixStream`, or any other `Read` implementation.
`CpalAudioSource` provides exact device selection and bounded local microphone
capture. `bounded_audio_input` is the adapter for server-fed and browser-fed
chunks.

The bounded sender uses `try_send`: producers receive `AudioError::Backpressure`
instead of growing memory. A producer that reconnects calls `reconnect` before
the first resumed chunk. Sequence gaps and callback queue overflow are reported
to consumers as discontinuities. Cancellation is shared between the consumer
and all sender clones.

`NormalizedAudioSource` applies deterministic channel conversion and linear
resampling while retaining the original format in descriptor metadata. It also
checks absolute frame continuity so a numerically continuous chunk sequence
cannot hide a source-timeline gap.

The corresponding #115 contract events come from
`AudioSourceDescriptor::stream_opened_event` and
`AudioSourceEvent::stream_event`. Raw PCM is not serialized by default.
Transports may opt into carrying audio only under their own privacy and
retention policy.

## Operator checks

List the backend-owned microphone IDs:

```sh
just run common-phone listen-devices
```

Exercise capture without loading a recognition model:

```sh
just run common-phone listen --dry-run --debug-frames
```

Start the local server:

```sh
just serve
```

Then inspect `GET /api/audio-input/capabilities`. Speech Studio reads that same
response; selecting the Compose input stage shows source kinds and the actual
host device inventory without a browser-maintained device catalog.
