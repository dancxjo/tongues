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
- `flush`: end-of-text flush examples must become speakable rows.
- `repair`: false early emissions must produce a low-confidence repair row with
  `<ERROR_REPAIR>` and `<ROLLBACK_GRAPHEMES>` before corrected phones.

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
