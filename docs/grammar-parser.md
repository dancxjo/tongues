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
returns an empty analysis if that explicitly requested backend is unavailable;
it does not claim a native parse came from UDPipe.

Link Grammar remains an acknowledged architectural influence on some English
connector rules. It is not the name of the generic parser contract or of native
Tongues and UDPipe output.

See the [terminology migration](terminology-migration.md) for compatibility
fields and their removal date.
