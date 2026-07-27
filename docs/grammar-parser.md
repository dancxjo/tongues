# Grammar Parser

`grammar-parser` converts sentence text into the backend-neutral
`speaking::syntax::GrammarAnalysis` contract. It is separate from the
cursor-time [sentence-boundary](sentence-boundary.md) model and from downstream
interpretation or resolution.

```sh
just grammar-parser parse "The quick brown fox jumps."
just grammar-parser parse --variety fr-FR-Standard --backend tongues-rules \
  "Je vois la maison."
just grammar-parser health --variety en-US
just grammar-parser compare "The quick brown fox jumps."
```

The canonical JSON shape uses:

- `ranked_parses` for ranked, projected grammar parses;
- `backend_parses` for backend-native diagnostic metadata;
- `backend = "tongues_rules"` for native variety-owned rules;
- `backend = "ud_pipe"` for a UDPipe projection;
- `backend = "link_grammar_oracle"` only for the optional comparison oracle.

`auto` tries a configured UDPipe model and falls back to native Tongues rules.
`tongues-rules` selects only the native rules. `ud-pipe` selects UDPipe and
returns `status = "failed"` with a diagnostic if that explicitly requested
backend is unavailable; it does not claim a native parse came from UDPipe.

Every analysis includes a `backend_report`:

- `requested` records `auto`, `tongues_rules`, `ud_pipe`, or an explicit
  `link_grammar_oracle` comparison request;
- `selected` names the backend whose analysis was returned;
- `attempts` retain each backend identity, terminal state, bounded diagnostic,
  elapsed time, process exit code, projection loss, and native coverage;
- `fallback_reason` explains why `auto` selected native rules after an external
  attempt.

Terminal backend states distinguish readiness, unsupported variety, unavailable
model, spawn failure, timeout, cancellation, malformed output, oversized input
or output, token-alignment loss, partial projection, accepted output, and
backend rejection. A failed forced backend is therefore different from a
grammatical fragment that produced a partial native analysis.

## Configuration and readiness

The native backend is ready when the requested variety declares a
`syntax_analyzer` or `syntax_rules` profile. Every built-in v1 variety has an
honest native profile; a missing profile is reported as unsupported rather than
silently treated as a successful empty parse.

UDPipe discovery checks these variables:

| Variable | Meaning |
|---|---|
| `TONGUES_UDPIPE_MODEL_<VARIETY>` | Model path explicitly scoped to one normalized variety name, for example `TONGUES_UDPIPE_MODEL_EN_US_GA`. |
| `TONGUES_UDPIPE_MODEL` | Shared model path. A shared model must also declare compatible varieties. |
| `TONGUES_UDPIPE_MODEL_VARIETIES` | Comma-separated exact variety codes supported by the shared model. |
| `TONGUES_UDPIPE_COMMAND` | UDPipe executable; defaults to `udpipe`. |

`grammar-parser health` validates variety support, model readability, a bounded
command version probe, and reports the command, version, model path, SHA-256,
and declared varieties without running a parse or synthesis pipeline.
Programmatic callers can use `grammar_backend_catalog` for the same serializable
readiness contract. The catalog also reports the optional Link Grammar oracle
as `feature_disabled`, `unavailable_executable`, `unavailable_dictionary`,
`unsupported_variety`, or `ready`; it is never selected by `auto`.

`auto` uses UDPipe only when it is configured for the exact requested variety
and returns a complete projection. Otherwise the external attempt remains in
`backend_report.attempts`, native rules run, and `fallback_reason` records
whether configuration, compatibility, execution, rejection, or projection was
responsible.

## Bounded execution and projection

UDPipe execution defaults to a 2-second deadline, 64 KiB input, 2 MiB stdout,
and 16 KiB stderr. Stdout and stderr are drained concurrently so the child
cannot deadlock on a full pipe. Timeout and cancellation kill and reap the
child. Captured stderr is bounded, whitespace-normalized, and redacts the model
path and tokens shaped like keys, passwords, secrets, or access tokens.
`UdPipeExecutionLimits` allows stricter caller-specific limits.

CoNLL-U projection aligns normalized token forms instead of truncating both
token arrays to their shorter length. `GrammarProjectionReport` records input
and backend token counts, aligned tokens, unmatched input indices, unmatched
backend token IDs/forms, and links that could not be projected. All input
tokens remain in `GrammarAnalysis.tokens`; unmatched tokens carry unknown facts.
Raw backend links remain in `backend_parses`, while generic typed links use only
successfully aligned input indices.

`backend_report` is an additive serialization field. Legacy grammar JSON that
omits it still decodes with an empty `auto` report; newly produced analyses
always populate the report at the `VarietyGrammarParser` or UDPipe boundary.
Backend-native links and costs remain inside the explicit `backend_parses`
envelope rather than being copied into normalized rank or confidence fields.

## Optional Link Grammar parity oracle

Link Grammar is a development-time comparison oracle, not a production parser
dependency or source of ground truth. The adapter is absent from default
features and cannot be selected by `grammar-parser parse` or `auto`. A default
build still accepts:

```sh
just grammar-parser compare "I saw the man with the telescope."
```

Its JSON includes native and UDPipe results and says that Link Grammar is
`feature_disabled`. Omitting the text runs five bounded curated fixtures and
labels the resulting agreement numbers as diagnostic parity:

```sh
just grammar-parser compare
```

To use an installed Link Grammar 5.12 or 5.13 `link-parser`:

```sh
cargo run -p tongues-cli --features link-grammar-oracle -- \
  grammar-parser compare "I saw the man with the telescope."
```

The adapter discovers:

| Variable | Meaning |
|---|---|
| `TONGUES_LINK_GRAMMAR_COMMAND` | `link-parser` executable; defaults to `link-parser`. |
| `TONGUES_LINK_GRAMMAR_DICTIONARY` | Installed language code such as `en`, or an explicit dictionary path. English varieties default to `en`; other varieties require this setting. |

The enabled adapter probes `--version`, invokes the executable as a separate
bounded process, and asks for complete link rows with at most eight linkage
alternatives. It uses the same 2-second, 64 KiB input, 2 MiB stdout, 16 KiB
stderr, concurrent pipe-draining, cancellation, kill/reap, and secret-redaction
controls as UDPipe. A missing executable, missing configured dictionary,
timeout, oversized output, malformed protocol, parser rejection, partial
linkage, and token-alignment loss remain distinct states.

One comparison report contains:

- the exact bounded Link Grammar stdout and redacted stderr;
- executable version, command, dictionary selector/path, optional file
  checksum, protocol, upstream URL, and license provenance;
- every linkage cost vector and backend-native link endpoint/label;
- the projected `GrammarAnalysis` and projection-loss report;
- unknown Link Grammar labels in both the raw link list and a top-level
  `unknown_labels` list;
- pairwise typed-link and endpoint attachment precision/recall/F1;
- parse acceptance, alternative count/top-rank comparison, and per-token
  pronunciation-context/prosodic-role agreement.

The conservative projection recognizes subject (`S*`), object (`O*`),
determiner (`D*`), modifier (`A*`, `M*`, `R*`), infinitival (`I*`, `TO*`),
auxiliary (`SI*`, `PP*`, `PG*`), preposition (`J*`, `IN*`), coordination
(`CO*`), and selected complement (`B*`, `C*`) families. Unrecognized labels
are not coerced to a generic kind or discarded; they stay inspectable in the
raw envelope and `BackendParse`.

Tongues does not link or redistribute Link Grammar code, English dictionaries,
or rules. The separately installed upstream project is LGPL-2.1, and dictionary
data remains part of that installation or an operator-supplied path. The
runtime provenance records what was actually invoked. See
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) and the official
[Link Grammar repository](https://github.com/opencog/link-grammar).

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
