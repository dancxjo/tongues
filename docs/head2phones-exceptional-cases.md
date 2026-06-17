# Head2Phones Exceptional Cases

`head2phones` data prep depends on sentence-boundary detection and the shared
`speaking` phonemicizer. Small pronunciation or segmentation mistakes can poison
many generated rows, so observed edge cases live in a checked catalog:

```text
docs/head2phones-exceptional-cases.json
```

The Rust tests load that JSON file directly. When a new bad row is found, add a
minimal case there first, then fix the shared mechanism that should cover it.

Current categories:

- `phones`: the first complete head must serialize the expected phone string and
  avoid known-bad alternatives.
- `head`: `first_complete_head` must return the expected text, or `null` for
  incomplete prefixes.
- `flush`: end-of-text flush examples must become explicit `<HEAD_FOUND>`
  speakable rows.
- `repair`: false early emissions must produce a low-confidence repair row with
  `<ERROR_REPAIR>` and `<ROLLBACK_GRAPHEMES>` before the corrected
  `<HEAD_FOUND>` block.
- Synthetic multilingual buffers scope phone rows to the head language. Other
  requested languages should get `<LANG_MISMATCH>` rather than phones.
- Synthetic guess-mode rows omit the input variety tag and include
  `<DETECTED_LANG>` before phones or language spans.

Important cases currently covered include:

- `Loadstone` pronounced as `lodestone`;
- `St.` before a proper name as `Saint`;
- address-final `St.` as `Street`;
- `Dr.`, `No.`, decimals, `p.m.`, and political/state abbreviations not causing
  false head splits;
- incomplete prefixes remaining `<NO_HEAD>`;
- title-like end-of-text fragments flushing into phone rows;
- false abbreviation splits, such as `Who shot John F.` before `Kennedy?`,
  producing rollback repair rows.
- code-switching contexts where the completed head and following text can use
  different languages.
- completed heads with internal code switches, marked with plain
  `<lang id="...">...</lang>` spans.

## Verifier false-positive workflow

Ollama verification is a passive audit over prepared `head2phones` rows. It is
useful for surfacing suspicious chunks, but a verifier report is not proof that
the generated row is wrong. When a report appears, first decide whether the
problem is in the data generator, the verifier prompt, or the model response
parser.

The code-switching `<PHONES>` false positive is the current example. The
prepared rows were allowed by the dataset contract: a normal `<HEAD_FOUND>`
phone block contains `<HEAD_LENGTH>`, `<PHONES>...</PHONES>`, and
`<SPLIT_AFTER>`, while a code-switch block contains `<LANGUAGE_SPANS>` markup,
`<HEAD_LENGTH>`, and `<SPLIT_AFTER>` and intentionally omits `<PHONES>`. The
old verifier prompt collapsed those cases and said every `<HEAD_FOUND>` block
needed `<PHONES>`, which caused the local model to report valid language-span
rows as malformed.

The fix process was:

1. Read the failing `ollama_verification_chunks.jsonl.part` reports and group
   repeated complaints by row shape, not by the model's wording.
2. Compare the reported row shape against the actual dataset contract in
   `dataset_readme`, the formatter helpers, and the exceptional-case tests.
3. If the generated row matches the contract, fix the verifier instructions
   instead of changing the data generator.
4. Add a focused regression assertion to the verifier prompt test so the prompt
   keeps distinguishing normal phone blocks from language-mismatch and
   language-span diagnostic blocks.
5. Run `cargo test -p tongues-head2phones` before trusting the change.

Use the same split for future reports:

- Add a JSON exceptional case when a real prepared row is bad.
- Patch the verifier prompt when the row is valid but the audit instructions
  make it look invalid.
- Patch response parsing only when Ollama returned a usable judgement that the
  current parser handles incorrectly.
