# Speaker diarization and voice familiarity

The `speaking` crate owns provider-neutral diarization contracts for both
streaming and offline recognition. `SpeakerDiarizer::process` consumes one
ordered segment observation at a time; `diarize_offline` drives the same
interface over a finite collection. `NoopSpeakerDiarizer` makes bypass explicit
without changing the recognition event contract.

`AnonymousSpeakerClusterer` is a deterministic baseline. It assigns stable
session-local cluster IDs, reports unknown speakers when no embedding is
available or cluster capacity is exhausted, emits overlap when an adapter
supplies simultaneous embeddings, and can revise earlier assignments after a
cluster merge. Speaker revisions retain the original segment ID and sequence;
they never revise transcript text or ordering.

`DiarizationProjection` attaches the latest cluster assignment to a committed
recognition segment. Normalization, parsing, interpretation, and export stages
can therefore retain the same `speaker_id` while continuing to distinguish
transcript evidence from later speaker revisions.

## Identity and privacy boundary

Anonymous clusters, familiar voices, and enrolled people are separate
capabilities:

- A `SpeakerClusterId` means only that observations sound similar within the
  diarizer's current scope.
- `VoiceFamiliarityEvidence` preserves the two signature IDs, similarity score,
  embedding model, and retention scope. It supports statements such as "this
  voice sounds familiar"; it does not name a person.
- `EnrolledSpeakerMapping` is a separate, explicit record with an enrollment ID
  and consent record. No clustering or similarity operation creates one.

`InMemoryVoiceFamiliarityMatcher` compares embeddings without enrollment or
stored names. Segment-scoped embeddings are never retained. Session-scoped
embeddings are discarded by `clear_session`. Persistent retention is rejected
by default and must be enabled explicitly with
`VoiceRetentionPolicy::retain_persistently`; applications providing persistent
storage remain responsible for consent, deletion, encryption, and retention
limits.

Embeddings from different named models are never compared, and a diarization
stream rejects a silent model or dimensionality change. Confidence values are
similarity evidence from that named model, not universally calibrated identity
probabilities.

## Fixture coverage

Focused tests cover anonymous operation, speaker changes and returns, overlap,
unknown speakers, merge revisions without transcript reordering, downstream
speaker projection, familiarity without enrollment, model provenance, explicit
persistent-retention opt-in, and session clearing.
