# Recognition and conversation CLI

Tongues exposes five friendly verbs whose stage definitions live in the
`speaking` library:

| Verb | Result |
| --- | --- |
| `listen` | normalized audio source events or float32 PCM |
| `transcribe` | committed recognized text |
| `recognize` | normalized transcript plus sentence structure |
| `interpret` | recognition plus semantic interpretation |
| `converse` | interpreted input, response-provider text, and synthesized WAV |

Print the exact library-owned stages without opening a source:

```sh
tongues transcribe --describe
tongues converse --describe
```

Every verb accepts a WAV path, `--microphone`, headerless PCM `--stdin`, a
`--tcp HOST:PORT` stream, or a `--unix PATH` stream. Raw inputs declare
`--pcm-encoding`, `--sample-rate`, and `--channels`; all sources pass through
the same deterministic mono 16 kHz normalization boundary. Microphone selection
uses `--input-device`. The process exits nonzero for invalid geometry,
discontinuity, unavailable models, unsupported languages, and provider errors.
`--maximum-audio-ms` provides a noninteractive cancellation bound and releases
live capture or socket resources when reached; Ctrl-C remains the interactive
operator cancellation path.

## Transcription and structured output

```sh
tongues transcribe recording.wav
tongues transcribe recording.wav --language en --partials
tongues recognize recording.wav --output json
tongues interpret recording.wav --output jsonl
```

Text mode writes only committed transcript text to stdout. With `--partials`,
unstable text is visibly prefixed with `~` on stderr and revisions with
`~ [revision]`; it cannot be mistaken for pipeable committed output. JSON and
JSONL use the shared recognition and downstream artifact contracts.
`--no-timestamps` and `--no-speaker-labels` remove those projections when a
caller does not want them.

The deterministic fixture provider makes examples and CI credential-free:

```sh
tongues transcribe recording.wav --provider fixture
```

Installed Whisper is the default and is currently advertised honestly as an
offline-only adapter.

## Unix composition

`listen --emit-pcm` writes only mono 16 kHz float32 little-endian PCM, making
process boundaries unambiguous:

```sh
tongues listen recording.wav --emit-pcm \
  | tongues transcribe --stdin --pcm-encoding f32le --provider fixture
```

Advanced stage commands remain available, including `vad`,
`language-routing`, `duplex`, `sentence-boundary`, `grammar-parser`, and
`interpretation`.

## Conversation

`converse` routes committed recognition through normalization, parsing, and
interpretation, formats response-provider text, and synthesizes a WAV through
the deterministic checkpoint-free TTS renderer:

```sh
tongues converse recording.wav --provider fixture \
  --response-template 'You said: {text}' \
  --response-wav response.wav --output json
```

This deterministic response provider proves the CLI ASR-to-response-to-TTS
composition without credentials. The genuinely token-streamed LLM, live
playback, and barge-in demo remain the separate #126 acceptance boundary.

Help and completions come from the real Clap tree:

```sh
tongues transcribe --help
tongues completions bash > tongues.bash
```
