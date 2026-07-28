# Interpretation Evidence Inspection

Tongues exposes one inspection contract for Duplex CLI output, the server API,
and Speech Studio's Operate workflow. The contract answers the same questions
without giving any display surface permission to reinterpret model state:

- which claim won and which alternatives remained;
- what source and authority class each claim used;
- which claims supported or conflicted with it;
- whether confidence was calibrated;
- which lifecycle transitions occurred;
- which hypothesis, score components, commit blocks, and generated output were
  affected;
- whether grammar backends were unavailable, failed, or partially projected;
- whether verification accepted a known projection loss.

## Versions

`DuplexStudioProjection` uses `schema_version = 2`.
`InterpretationInspectionPage` nested at `interpretation` uses
`schema_version = 1`. The simulator journal and the underlying
`LinguisticEvidenceArtifact` retain their independent versions.

Projection v2 adds the bounded interpretation page and an optional durable deep
link. Existing projection v1 readers must reject the new version rather than
guessing at evidence semantics. The server still accepts legacy syntax aliases
at deserialization boundaries; when a saved journal actually contains one, the
inspection page emits a `legacy_syntax_alias_migrated` warning and serializes
canonical names.

## Bounded pages and stable targets

The default page contains 20 targets. Clients may request 1–100 with a cursor.
Resolution targets use `resolution:<resolution-id>`; unresolved or historical
claims use `claim:<claim-id>`. A requested target that no longer exists returns
an empty page with `target_not_found`, rather than silently selecting a nearby
claim.

Each target bounds alternatives, linked claims, backend reports, and lifecycle
history. Truncation is explicit in boolean fields and warnings. Missing
linguistic evidence sets `evidence_status = "missing"` and explains that
alternatives and confidence are unknown; it is never rendered as confidence
zero.

Audio links preserve their actual precision. Current Duplex observations carry
utterance/chunk frame spans, so `alignment = "utterance_or_chunk_span"` is
shown instead of implying token-exact alignment. Claim values and linked claim
IDs connect text ranges to syntax links, pronunciation phonemes/phones, stress,
and boundaries. Consequence records connect them to hypotheses, score
components, commit state, and synthesis delivery records.

## CLI

The default timeline remains concise:

```sh
cargo run -q -p tongues-cli -- duplex demo --fixture who-shot-john-f
```

Explain evidence in text:

```sh
cargo run -q -p tongues-cli -- duplex demo \
  --fixture who-shot-john-f \
  --explain \
  --evidence-limit 20
```

Emit the same projection contract returned by the server:

```sh
cargo run -q -p tongues-cli -- duplex demo \
  --fixture who-shot-john-f \
  --json
```

Use `--evidence-cursor` for the next page and `--evidence-target` for one stable
resolution or claim target. Use `--explain` rather than the repository-wide
`--verbose` flag so the evidence mode is explicit.

## Server and Operate

`POST /api/duplex/project` returns the complete versioned projection and accepts
`evidence_cursor`, `evidence_limit`, and `evidence_target_id`.

`GET /api/duplex/evidence` returns only the shared inspection page. It accepts
`fixture` or `journal_path`, plus `cursor`, `limit`, and `target_id`.

For fixtures and saved journals the server returns a `/speech/operate` deep
link containing the durable source and optional target/page identity. Operate
restores that link after refresh. Ad hoc unsaved chunks deliberately have no
permanent link; the UI says that a saved journal is required.

Operate keeps the event timeline concise. Branch scores and evidence IDs,
target alternatives, conflict explanations, audio spans, backend reports, and
output consequences are progressively disclosed with native keyboard-operable
`details` controls. Corrections remain lifecycle entries and repaired
hypotheses in the immutable journal; the inspection view never overwrites
original observed evidence.

Durable timeline-session context may carry `interpretation_deep_link` and
`interpretation_target_id`. Execution Tracks and WaveDeck expose that local
Operate handoff when present, while continuing to treat their original
execution/session evidence as immutable. The server rejects non-local
`interpretation_deep_link` values.
