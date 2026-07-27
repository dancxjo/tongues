# Output verification and self-hearing

Tongues treats speaking as a chain of correlated claims:

```text
generated text -> speech plan -> synthesis request -> target PCM
               -> device playback -> microphone observation
```

Submitting PCM to an audio API proves only submission. It does not prove device
playback or a room-level acoustic event. `speaking::output_monitor` therefore
uses stable IDs for every link and emits distinct planned, requested, started,
interrupted, resumed, aborted, completed, and perceived lifecycle events.
Downstream agents can consume the append-only lifecycle without treating
intended speech as observed speech.

## Default privacy boundary

`AudioEvidenceFeatures` retains only eight coarse energy-shape buckets plus
duration, RMS, peak, zero-crossing rate, and clipping fraction. Target and
microphone PCM are not stored by the verifier. Feature state is bounded by
active playback session and explicitly released at teardown. Verification
results always report `raw_audio_retained: false`.

This is a deterministic baseline for correlation and policy, not a claim of
production acoustic echo cancellation. `EchoReferenceFrame` provides an
explicit live-only target PCM feed for a device AEC; it is consumed and
discarded, is not serializable, and never enters the feature verifier. A device
integration supplies:

- target features before playback;
- whether the device reports an active playback session;
- feature-only microphone observations;
- an external-speech probability from the input pipeline;
- explicit output discontinuity/dropout evidence.

The verifier distinguishes likely self-speech, likely external speech, overlap,
uncertain attribution, partial or missing output, playback failure, clipping,
and dropout. Barge-in interrupts active playback only for external-speech or
overlap evidence. Target-like echo continues playback; ambiguous evidence waits
for more input.

## Deterministic acceptance fixtures

[`../fixtures/output-verification/scenarios_v1.json`](../fixtures/output-verification/scenarios_v1.json)
is CC0 and covers normal playback, user interruption, echo overlap, missing
output, and partial output. Tests also cover clipping/dropout representation,
lifecycle ordering, stable correlation, terminal-state enforcement, feature
boundedness, and feature-state release. No physical microphone, speaker, model,
credential, transcript, or retained raw audio is required.
