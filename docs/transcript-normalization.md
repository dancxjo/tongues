# Transcript normalization and downstream routing

`CommittedTranscriptPipeline` is the library-owned boundary between recognition
and linguistic/semantic consumers. It accepts the shared streaming event
contract one event at a time and invokes normalization, sentence parsing, and
interpretation only for `CommittedSegment` events whose role is `recognition`.
Partial hypotheses, revisions, generated text, warnings, and transport events
cannot trigger downstream interpretation or irreversible actions.

Every `NormalizedTranscriptSegment` keeps three text projections:

- `raw_text` is immutable provider evidence.
- `display_text` removes provider control tokens and applies readable spacing,
  casing, and punctuation.
- `downstream_text` applies configured disfluency, non-speech annotation, and
  inverse-text-normalization policy for parsers and interpreters.

Words and their timing/confidence, segment ID, language evidence, speaker ID,
segment confidence, source event reference, dual event times, and provenance
remain aligned with all three projections. `transcript_export_jsonl` serializes
that complete record, so verbatim and normalized transcripts can be exported
together rather than overwriting one another.

## Language-aware rules

The deterministic baseline selects filled-pause and digit-word rules by the
recognized language's primary subtag. English, French, German, and Spanish
digit sequences are supported; filled pauses likewise use language-specific
sets. The sentence parser receives the matching built-in linguistic variety
where available. Unknown languages retain common cleanup and use the default
parser rather than pretending that language-specific inverse normalization was
available.

Applications can replace `TranscriptNormalizer` and
`CommittedTranscriptInterpreter` independently. Interpretation adapters receive
both the normalized segment and syntax analysis, while the shared output stays
provider-neutral.

## Commit behavior

A segment ID is processed at most once. A later attempt to recommit different
punctuation or segmentation for the same ID fails with `DuplicateCommit`.
Hypothesis revisions after commit are ignored by this pipeline and rejected by
the upstream stream-contract validator. Correcting committed evidence requires
a new explicitly derived artifact or edit operation; it never silently rewrites
the raw transcript or repeats a downstream action.

Focused fixtures cover provider-token cleanup, raw/display/downstream
distinction, language-aware disfluency and inverse normalization, partial-event
isolation, incremental parser/interpretation output, duplicate-commit
protection, aligned timing/speaker/confidence metadata, provenance preservation,
and combined verbatim/normalized JSONL export.
