# Streaming audio cleanup

`tongues-audio` exposes optional cleanup as an ordered `CleanupPipeline`. Every
stage receives and returns the same interleaved floating-point PCM geometry.
The pipeline rejects a stage if it changes frame count, sample rate, or channel
layout.

Available stages are DC removal, adaptive gain control, low-pass filtering,
noise gating, delayed-echo suppression, and a conservative source-separation
floor baseline. Stages are independently selectable, reorderable, and
bypassable. Each processed chunk reports input/output RMS, bypass state, and
declared algorithmic latency. Causal built-in stages currently declare zero
look-ahead frames.

`CleanupAudioSource` applies the same pipeline to any file, microphone, browser,
or transport `AudioSource`. It preserves chunk sequence and source-frame
position and records the ordered server stages in descriptor metadata. A source
discontinuity resets bounded DSP history so state is never carried across a
gap.

The echo stage is a bounded delayed-self suppression baseline, not a claim of
full acoustic echo cancellation without a playback reference. The source
separation stage is likewise an optional floor baseline, not speaker
diarization. Both are pluggable contract points for stronger adapters.

## Speech Studio

The browser microphone inspector offers ordered server cleanup choices and sends
them in the same WebSocket open contract used by recognition. Raw and processed
RMS meters are computed together on the server for each identical input chunk,
so the comparison does not duplicate cleanup logic in JavaScript.

For this comparison, browser-native echo cancellation, noise suppression, and
automatic gain control are explicitly disabled in `getUserMedia`. The UI labels
the selected processing as server-side, keeping browser preprocessing distinct
from Tongues processing.

## Verification fixtures

Focused `tongues-audio` tests cover:

- geometry and ordered provenance on clean speech;
- steady-noise attenuation;
- finite, bounded output for impulsive noise;
- full-scale bounds for clipped input;
- suppression of a known delayed echo;
- independent bypass and stage ordering; and
- bounded, geometry-preserving capability declarations.
