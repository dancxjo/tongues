# Streaming event contract

Tongues has one streaming intermediate representation: `speaking::StreamEventEnvelope`.
Recognizers, predictive duplex, interpretation, the CLI, the server, and Speech
Studio use this type directly. A provider may add provenance attributes, but it
must not introduce a parallel chunk or transcript event schema.

The current wire version is `schema_version: 1`. JSON uses a stable adjacent
tag:

```json
{
  "schema_version": 1,
  "stream_id": "recognition:example",
  "event_id": "recognition:example:4",
  "sequence": 4,
  "times": {
    "occurred_at": {"origin": {"kind": "stream_start"}, "offset_ms": 320},
    "observed_at": {"origin": {"kind": "unix_epoch"}, "offset_ms": 1785100000000}
  },
  "provenance": {
    "kind": "derived",
    "sources": [{"stream_id": "recognition:example", "event_id": "recognition:example:2"}]
  },
  "event": {
    "type": "revised_hypothesis",
    "data": {
      "role": "recognition",
      "segment_id": "segment-1",
      "replaces": {"start": 6, "end": 9},
      "text": "world"
    }
  }
}
```

## Text stability and revisions

`partial_hypothesis` is explicitly unstable. `revised_hypothesis` retains the
same stable `segment_id` and carries an exact half-open replacement range.
Ranges count Unicode scalar values, not UTF-8 bytes or UTF-16 code units.
`committed_segment` is immutable; a later partial or revision for that ID is
malformed and must be rejected. Clients therefore never infer replacement
boundaries by comparing strings.

The `role` field lets the same IR represent recognition, generated text,
normalization, parsing, and interpretation without conflating their authority.
Structured downstream products use `derived_artifact` and must link to their
source event IDs through envelope provenance.

## Audio, clocks, and provenance

`stream_opened` owns the audio encoding, sample rate, channel layout, source,
and clock. `audio_chunk` carries a monotonic chunk sequence and frame count.
Input and synthesized output chunks differ only by `direction`. File, live, and
replayed audio use the same event model; only `StreamSource` changes.

`occurred_at` is media or real-world event time. `observed_at` is arrival time.
They must not be substituted for one another during buffering, replay, delayed
transport, recognition, parsing, or interpretation.

Direct events name their provider/model when relevant. Derived and recalled
events retain source `stream_id` plus `event_id` links. Confidence includes its
scale. `provider_native` scores are not presented as probabilities and are not
compared across providers unless an explicit calibration identifier is present.

Speech output uses stable utterance IDs for requested, started, interrupted,
resumed, aborted, and finished events. `output_requested.caused_by` correlates
speech with the committed or generated text that caused it.

## Ordering, buffering, cancellation, and failures

Sequences are contiguous within one `stream_id`. Consumers reject a stream ID
change, duplicate, gap, or backward sequence unless a transport layer explicitly
repairs it before contract validation. Audio loss is represented explicitly by
`discontinuity`; it does not authorize out-of-order event envelopes.

The default library policy bounds a queue at 64 events and applies producer
backpressure when full. Transports may choose explicit rejection instead, but
must not silently discard recognition, commit, discontinuity, error, or terminal
events. Cancellation clears buffered work and prevents subsequent producer
writes. `cancelled`, `completed`, and non-recoverable `error` are terminal;
events after them are malformed.

Malformed JSON, an unsupported version, missing derived provenance, an invalid
replacement range, and mutation of committed text fail closed. A boundary may
instead be configured to emit a warning and drop malformed/out-of-order input,
but the warning itself uses this contract and no invalid state is applied.

## Compatibility rules

- Version 1 field meanings, enum tags, ID scope, sequence rules, and Unicode
  range units are stable.
- Optional fields may be added within version 1. Consumers must ignore unknown
  object fields.
- New event variants, changed required fields, changed units, or changed
  ordering/terminal semantics require a new schema version because exhaustive
  Rust and fail-closed browser consumers cannot safely guess their meaning.
- Producers emit one version per stream. Version changes require a new
  `stream_id`; in-band version switching is invalid.
- JSONL is one complete envelope per line. The Rust JSON API and JSONL helpers
  serialize the same type.

The conformance fixtures live in
`fixtures/streaming/recognition_scenarios_v1.json` and cover silence, overlap,
language change, speaker change, discontinuity, cancellation, and provider
failure. Rust round-trip tests cover every event variant; Speech Studio runs the
same ordering and revision checks before applying server events.
