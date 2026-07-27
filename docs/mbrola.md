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

## Voice artifacts and licensing

The shared artifact getter includes the upstream `us1`, `us3`, `en1`, and `nl2`
diphone databases and installs each database together with its license notice:

```sh
cargo run --bin tongues -- models fetch mbrola-us1
cargo run --bin tongues -- models fetch mbrola-us3
cargo run --bin tongues -- models fetch mbrola-en1
cargo run --bin tongues -- models fetch mbrola-nl2
```

The published voice terms permit no-charge distribution with the notice, but
restrict use of the stock databases to the official MBROLA program. Fetching a
database therefore does not authorize Tongues' native renderer. Native use
fails closed unless you have separate permission and attest to it:

```sh
export TONGUES_MBROLA_NATIVE_USE_AUTHORIZED=1
```

Set that variable only when your authorization covers native use. This gate
never triggers an executable fallback: Tongues does not probe for, install, or
invoke the MBROLA binary.

Each catalog database has an explicit `MbrolaVoiceConfig` recording the logical
voice, database artifact, database voice ID, variety, symbol map, and optional
pitch metadata. Multiple logical voices may share one database without
pretending they are the same language.

The catalog voices have built-in source-to-inventory phone maps adapted from
Listenbury. A user-supplied voice may use
`TONGUES_MBROLA_SYMBOL_MAP=/path/to/map.json`; the file is a JSON object:

```json
{
  "HH": "h",
  "AH0": "@",
  "L": "l",
  "OW1": "@U"
}
```

An unknown user voice without an explicit map selects an inspectable identity
mapping. Unknown mappings, symbols absent from the selected voice, and missing
diphones fail with errors that name the phone, variety, voice/map, or exact
diphone. Tongues never substitutes an unrelated sound.

The server and Speech Studio expose `mbrola-us3` as the default native path and
the catalog exposes all four downloadable databases. Once an artifact is
installed and native use is authorized, the shared discovery/runtime path
provides its phone timing/F0/rate capabilities, the
`projector/mbrola-phone-timing` stage, and the native TD-PSOLA renderer.

## CLI

```sh
TONGUES_MBROLA_NATIVE_USE_AUTHORIZED=1 \
cargo run --bin tongues -- speak 'Hello.' \
  --backend mbrola \
  --model mbrola-us3 \
  --pho-output /tmp/hello.pho \
  --output /tmp/hello.wav
```

`--model` also accepts a direct path for a separately supplied database.

## Esperanto through Dutch nl2

`mbrola-eo-nl2` is a logical Esperanto voice backed by the full Dutch `nl2`
diphone database. It selects Tongues' `eo` phonemicizer and a dedicated map for
all 28 Esperanto phonemes. The three affricates expand deliberately:

```text
t͡s -> t s
t͡ʃ -> t S
d͡ʒ -> d Z
```

The database contains the required adjacent diphones; durations and F0 targets
are partitioned across expanded phones rather than duplicating the requested
duration.

```sh
cargo run --bin tongues -- models fetch mbrola-nl2
TONGUES_MBROLA_NATIVE_USE_AUTHORIZED=1 \
cargo run --bin tongues -- speak 'Saluton, mondo!' \
  --backend mbrola \
  --model mbrola-eo-nl2 \
  --output /tmp/saluton.wav
```

Speech Studio exposes `mbrola-eo-nl2` as an Esperanto path while installation,
verification, and license provenance continue to point to the single
`mbrola-nl2` database artifact.

Classical Latin and Sanskrit are exposed the same way:

```sh
cargo run --bin tongues -- models fetch mbrola-la1
cargo run --bin tongues -- models fetch mbrola-in1
cargo run --bin tongues -- models fetch mbrola-in2
```

`mbrola-la-la1` uses the purpose-built Classical Latin `la1` database.
`mbrola-sa-in1` and `mbrola-sa-in2` use the Hindi databases, whose stop,
retroflex, aspiration, vowel, and sibilant inventories cover Tongues' Sanskrit
variety. The database has no syllabic-r unit, so Sanskrit `r̩` is explicitly
projected to `r ii`; velar and palatal nasals use the database's dental nasal,
and both Sanskrit postalveolar sibilants use `sh`. These are documented
database limitations, not claims of exact phonetic identity.

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
