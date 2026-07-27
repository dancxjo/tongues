# Boundary and Grammar Terminology Migration

Tongues v1 uses responsibility-specific names:

| Responsibility | Canonical vocabulary |
|---|---|
| Cursor-time emit/continue/repair decision | sentence-boundary detector/model; `sentence-boundary` |
| Syntactic or dependency analysis | grammar parser; `grammar-parser` |
| Projected result and alternatives | `GrammarAnalysis`; `RankedGrammarParse`; `ranked_parses` |
| Backend-native diagnostic metadata | `BackendParse`; `BackendLink`; `BackendCost`; `backend_parses` |
| Backend identity | `GrammarBackend`; `tongues_rules`; `ud_pipe` |
| Evidence combination and resolution | interpretation |

## V1 compatibility

New artifacts and payloads always serialize canonical names. Existing v1
payloads decode automatically as follows:

| Previous v1 name | Canonical name |
|---|---|
| `link_parses` | `ranked_parses` |
| `raw_link_grammar_parses` | `backend_parses` |
| `tongues_rule_grammar` or `tongues_link_grammar` | `tongues_rules` |
| `link_grammar_rule` | `grammar_rule` |
| `link_grammar_projection` | `grammar_projection` |
| model manifest family `sentence-parser` | `sentence-boundary` |
| `sentence_parser_config.json` | `sentence_boundary_config.json` |

Sentence-boundary config files have `schema_version = 1`. Missing versions from
pre-versioned v1 configs decode as v1. An unsupported version fails with a
migration message naming the found and expected versions. Model manifests
continue to use the shared artifact `schema_version = 1`; the previous family
value is accepted with a warning.

The CLI spelling `sentence-parser` remains an alias for `sentence-boundary`.
Its hidden `parse` subcommand warns and forwards to the default grammar parser
only for source compatibility. New scripts must use `grammar-parser parse`.

Deprecated Rust source aliases live only in
`speaking::syntax_compatibility`; new code must import canonical names from
`speaking::syntax`.

## Removal plan

The compatibility command, Rust aliases, JSON field aliases, backend aliases,
legacy manifest family, and legacy config filename are scheduled for removal on
2027-07-27. Before that date, rewrite durable artifacts by loading them with a
current Tongues build and serializing the canonical shape, or directly change
the manifest/config names listed above. Historical Link Grammar attribution
remains in architecture history after compatibility removal.
