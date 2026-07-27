# Native MBROLA synthesis

Tongues treats an MBROLA `.pho` file as an inspectable, lossy backend format,
not as its speech intermediate representation. The native path is:

```text
speaking::UtterancePlan
  -> voice-specific symbol and timing projection
  -> PhoneTimedPlan / optional .pho inspection
  -> native Rust diphone TD-PSOLA
  -> finite mono f32 waveform / WAV
```

No `mbrola` executable is invoked.

The database reader, half-diphone assembly, join smoothing, pitch-mark handling,
and TD-PSOLA implementation were adapted from Tongues' sibling Listenbury
project. Tongues has no runtime or build dependency on that repository.

## Voice setup and licensing

Tongues does not redistribute MBROLA voice databases. Supply a database whose
terms allow your intended use and affirm those terms explicitly:

```sh
export TONGUES_MBROLA_VOICE=/path/to/us3
export TONGUES_MBROLA_LICENSE='user-supplied; reviewed for this use'
export TONGUES_MBROLA_SYMBOL_MAP=/path/to/us3-symbol-map.json
```

The symbol map is a JSON object from Tongues phone display symbols to the
selected database inventory:

```json
{
  "HH": "h",
  "AH0": "@",
  "L": "l",
  "OW1": "@U"
}
```

An omitted map selects an inspectable identity mapping. Unknown mappings,
symbols absent from the selected voice, and missing diphones fail with errors
that name the phone, variety, voice/map, or exact diphone. Tongues never
substitutes an unrelated sound.

The server and Speech Studio read the environment variables above. Once the
voice and licensing assertion are present, the shared discovery endpoint
exposes `mbrola-user-voice`, its phone timing/F0/rate capabilities, the
`projector/mbrola-phone-timing` stage, and the native TD-PSOLA renderer.

## CLI

```sh
TONGUES_MBROLA_LICENSE='user-supplied; reviewed' \
cargo run --bin tongues -- speak 'Hello.' \
  --backend mbrola \
  --model /path/to/us3 \
  --mbrola-symbol-map /path/to/us3-symbol-map.json \
  --pho-output /tmp/hello.pho \
  --output /tmp/hello.wav
```

The library also exposes `parse_pho`, `serialize_pho`, and
`NativeMbrolaRenderer::render_pho`, so parsed `.pho` and typed plans enter the
same renderer.

## Timing and prosody

Valid phone spans are authoritative. Without spans, the configurable
`MbrolaTimingProfile` starts from 72 ms consonants and 110 ms vowels, adjusts
stressed/focused nuclei, samples the speaking-rate curve, and inserts bounded
20–800 ms silences for explicit prosodic breaks. These are documented
fallbacks, not measurements.

Absolute pitch points are sampled over voiced phone spans and override fallback
contours. When pitch is absent, the selected voice baseline/range or explicit
synthesis controls are used; only then does the documented 120 Hz baseline and
30 Hz range apply. Stress, focus, emphasis, question/continuation rise, and
final fall affect the conservative fallback contour. Silence and unvoiced
phones never receive F0 targets.

`.pho` cannot encode energy, style, or target acoustic frames. They remain on
the canonical `UtterancePlan`, and the lowering report states this limitation.
