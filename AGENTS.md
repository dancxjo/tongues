# Agent Notes

## Status Updates For Long-Running Commands

Long-running prepare/train paths should make progress visible and durable. Do not leave users staring at one spinner message while work accumulates only in memory.

Preferred pattern:

- Put progress events in the library crate, not only in the CLI. Use an enum such as `PrepareProgress` with variants for stages, counts, writes, downloads, parsing, expansion, etc.
- Keep the existing simple API as a wrapper, for example `prepare_dataset(...)` should call `prepare_dataset_with_progress(..., |_| {})`.
- In the CLI, translate progress events into concise spinner messages with concrete counts and paths.
- Report progress at bounded intervals. Good defaults are first few rows/files and then every N rows, pages, files, or batches.
- Include the active output path in progress messages whenever work is being written.

Durability expectations:

- For expensive expansion/build phases, write intermediate artifacts as work is produced, using a `.part` suffix.
- Flush writers after expensive phases.
- Write final JSONL outputs through `.part` files and `rename` them into place after a successful flush.
- Remove temporary `.part` files only after final outputs, config, README, and vocab/model metadata are written.
- If a phase still needs an in-memory vector for shuffling or vocab construction, also stream a recoverable/debuggable copy to disk as it is generated.

Training expectations:

- Make checkpoint behavior explicit before training starts: print `train_state.json`, epoch checkpoint pattern, and best-model path.
- Shared seq2seq training already writes `train_state.json`, per-epoch `model-epoch-N.bin`, and best `model.bin`; new trainers should reuse that path or match its behavior.
- If a trainer cannot checkpoint until epoch end, say so in the startup/status output.

Recent examples to follow:

- `crates/tongues-wiktionary/src/lib.rs`: `PrepareProgress::Expand` and `expanded.jsonl.part`.
- `crates/tongues-sentence-parser/src/lib.rs`: `prepare_dataset_with_progress`, `sentences.jsonl.part`, `examples.jsonl.part`, and atomic final JSONL writes.
- `crates/tongues-cli/src/main.rs`: progress formatter functions that convert library progress events into spinner messages.
