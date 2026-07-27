# Voice activity detection and utterance segmentation

Tongues uses one streaming contract for decoded files, host microphones,
browser microphones, and pushed transport audio. `AudioSource` provides bounded
decoded chunks; `VadSegmentationPipeline` frames them into 10 ms windows and
feeds an `UtteranceSegmenter`. The segmenter does not depend on where the audio
came from.

## Defaults and controls

The default WebRTC detector combines the native WebRTC decision with an adaptive
noise floor. The energy detector is a deterministic baseline. Both produce a
speech probability, speech decision, and RMS measurement for every frame.

The default segment policy is:

- 30 ms of consecutive speech to open, retaining 200 ms of pre-roll.
- An acoustic `speech_ended` event after 300 ms of silence.
- A conversational segment close after 800 ms of silence.
- Speech shorter than 250 ms is reported as dropped.
- Continuous speech is force-closed at 30 seconds.

The acoustic endpoint and conversational close are deliberately separate. A
recognizer can react promptly to `speech_ended`, while a dialogue component can
keep the segment open across a short pause. Initial and final audio are retained
in the segment: opening includes pre-roll and closing includes silence through
the configured boundary.

Providers with authoritative boundaries can call
`UtteranceSegmenter::process_authoritative`. These boundaries bypass VAD
thresholds and are marked as authoritative evidence rather than inferred VAD
evidence.

## CLI

Inspect a WAV with the same pipeline used by live input:

```sh
tongues vad recording.wav
```

Machine-readable lifecycle output and tuning are available directly:

```sh
tongues vad recording.wav --jsonl \
  --backend web-rtc \
  --speech-start-ms 30 \
  --acoustic-end-ms 300 \
  --segment-end-ms 800 \
  --pre-roll-ms 200 \
  --minimum-speech-ms 250 \
  --maximum-segment-ms 30000
```

Use `--show-frames` when per-frame RMS, probability, and speech decisions are
needed. The energy baseline exposes `--energy-threshold-rms`; WebRTC exposes
`--minimum-speech-rms` and `--noise-gate-multiplier`.

In Speech Studio, open the input stage inspector and select **Test this
browser's microphone**. Its VAD and segmentation controls are sent in the
WebSocket open message. The meter shows current RMS/VAD, while the retained
event list shows segment opening, acoustic endpoint, and final close.

## Events and metrics

Segmentation lifecycle events project into the shared streaming event contract:

| Segmentation event | Streaming event |
| --- | --- |
| segment opened | `speech_started` |
| segment audio | `audio_chunk` |
| acoustic endpoint | `speech_ended` |
| accepted or dropped close | `derived_artifact` |

Metrics report observed and speech frames, speech ratio, opened/finalized/dropped
segments, dropped source chunks, actual forced flushes, and mean/maximum endpoint
latency. Source sequence or frame-timeline gaps explicitly close the active
segment with a discontinuity reason instead of silently joining unrelated audio.
At end of input, a partial 10 ms frame is zero-padded before the final boundary
is evaluated.

## Listenbury migration

This implementation ports the useful behavior of Listenbury's
`BreathGroupSegmenter` rather than importing its hearing types:

- Stable stream-scoped segment IDs are retained.
- The three-frame opening rule and 800 ms conversational close are retained as
  defaults.
- Silence and maximum-duration close reasons remain explicit.
- Tongues adds a distinct 300 ms acoustic endpoint, configurable pre-roll and
  minimum duration, source-discontinuity accounting, authoritative-boundary
  evidence, and bounded audio-source integration.
- Listenbury's timeout/overlap/cancel outcomes map to maximum duration,
  discontinuity, and cancelled/forced-flush reasons.

The donor-equivalent opening, silence-close, maximum-duration, and authoritative
boundary cases live beside the segmenter tests in `tongues-audio`.
