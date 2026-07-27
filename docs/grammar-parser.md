# Grammar Parser

`grammar-parser` converts sentence text into the backend-neutral
`speaking::syntax::GrammarAnalysis` contract. It is separate from the
cursor-time [sentence-boundary](sentence-boundary.md) model and from downstream
interpretation or resolution.

```sh
just grammar-parser parse "The quick brown fox jumps."
just grammar-parser parse --variety fr-FR-Standard --backend tongues-rules \
  "Je vois la maison."
```

The canonical JSON shape uses:

- `ranked_parses` for ranked, projected grammar parses;
- `backend_parses` for backend-native diagnostic metadata;
- `backend = "tongues_rules"` for native variety-owned rules;
- `backend = "ud_pipe"` for a UDPipe projection.

`auto` tries a configured UDPipe model and falls back to native Tongues rules.
`tongues-rules` selects only the native rules. `ud-pipe` selects UDPipe and
returns `status = "failed"` with a diagnostic if that explicitly requested
backend is unavailable; it does not claim a native parse came from UDPipe.

## Ranked alternatives

`ranked_parses` preserves a backend primary parse and any rule-supported
alternatives. Each entry has:

- a stable `id` derived from backend, backend-parse index, ambiguity family,
  and stable token anchors;
- typed links;
- a normalized `rank` and `confidence`;
- `complete` or `partial` status;
- provenance naming the backend parse and the transformation that produced the
  alternative.

The bounded ambiguity layer covers prepositional attachment, coordination
scope, context-supported noun/verb POS, complement attachment, phrasal
particle/adposition readings, relative-clause attachment, and parenthetical
islands. For example:

```sh
just grammar-parser parse "I saw the man with the telescope."
```

retains the backend primary graph plus noun-attachment and verb-attachment
variants. An unambiguous fixture such as `I saw the man.` keeps one parse.
Alternative generation is capped at eight total parses and disabled beyond 128
tokens, preventing combinatorial growth during streaming or on large inputs.

`GrammarAnalysis::identity_delta_from` compares stable IDs across an extension
or repaired transcript and reports retained, invalidated, and introduced
parses. Rank changes do not change identity.

## Rank, confidence, and backend cost

The three score surfaces have deliberately different meanings:

| Field | Meaning |
|---|---|
| `RankedGrammarParse.rank` | Backend-neutral score in `[0, 1]`; higher is better. The primary score is 65% mean normalized link confidence, 25% linked-token coverage, and 10% completeness. A deterministic ambiguity-family penalty is applied to generated alternatives. |
| `RankedGrammarParse.confidence` | Mean normalized confidence of the typed links in that parse. It is validated/consumed only as a `[0, 1]` value. |
| `BackendParse.cost` | Unmodified backend diagnostics such as unused-token, disjunct, or link-length cost. These values are not treated as normalized confidence and are not compared directly across backends. |

`BackendParse.accepted` is true only for a complete backend result. A nonempty
but incomplete projection is `partial`; unavailable input/backend/rules are
`failed` with a diagnostic. Empty link vectors are therefore not presented as
accepted success for multi-token input.

Native and UDPipe projections both pass through the same normalized scoring
stage. Generated alternatives retain the source backend and
`backend_parse_index`; `backend_parses` remains the untouched diagnostic
record.

## Conservative interpretation

Consumers configure `GrammarRankingPolicy.close_rank_gap` and
`confidence_floor`. Defaults are a `0.08` rank gap and `0.60` confidence floor.
`best_parse`, `alternatives`, `ambiguity_margin`, `close_alternatives`, and
`token_facts_for_parse` expose the complete ranked set.

When the best/runner-up margin is close, the best confidence is below the
floor, or analysis is partial, `conservative_facts` intersects POS/link/prosody
facts across the close parses. Variant-only facts disappear from that view but
remain on their parse. `rule_context` uses the conservative view, and
phonemicization avoids grammar-dependent POS, reduction, stress, or inserted
boundary decisions until the policy becomes decisive.

`to_linguistic_evidence` projects every parse, link, and POS assertion into the
shared lifecycle-aware claim artifact. Parse candidates share one selection
target, conflict explicitly, retain per-parse identity/rationale, and are
supported by their component claims. Resolution uses normalized parse rank
without erasing alternatives.

Link Grammar remains an acknowledged architectural influence on some English
connector rules. It is not the name of the generic parser contract or of native
Tongues and UDPipe output.

See the [terminology migration](terminology-migration.md) for compatibility
fields and their removal date.
