# Sentence Parser

`sentence-boundary` is a seq2seq cursor-boundary model family. It is trained from Project Gutenberg-style plain text using the `seams` sentence detector as the teacher.

Prepare data with the default sources:

```sh
just sentence-boundary prepare \
  --out datasets/sentence-boundary/v0
```

Start a fresh default run by archiving the existing default dataset/model artifacts and recreating empty directories:

```sh
just sentence-boundary clean --all
```

Use `--data` or `--model` to archive only one side. Artifacts are moved under `archive/<run-id>/...`; pass `--run-id NAME` for a stable archive folder or `--no-create` if you do not want empty defaults recreated.

With the default config, preparation downloads a small Project Gutenberg cache and generates deterministic synthetic sentence-boundary cases. Local text files or directories can still override those defaults:

```sh
just sentence-boundary prepare \
  --input /path/to/gutenberg_texts \
  --out datasets/sentence-boundary/v0
```

Train:

```sh
just sentence-boundary train \
  --data datasets/sentence-boundary/v0 \
  --out models/sentence-boundary/v0
```

Prepare and train in one command:

```sh
just sentence-boundary train --prepare \
  --input /path/to/gutenberg_texts \
  --data datasets/sentence-boundary/v0 \
  --out models/sentence-boundary/v0
```

Preparation also runs a deliberately naive punctuation splitter and compares it to `seams`. Useful over-split disagreements are saved to `naive_discrepancies.jsonl` and folded into the default training splits as `row_source = "naive-discrepancy"` correction rows.

Train only the clean `seams` rows, only mined corrections, or both:

```sh
just sentence-boundary train --training-set seams
just sentence-boundary train --training-set naive-discrepancy
just sentence-boundary train --training-set all
```

Cursor inference:

```sh
just sentence-boundary infer \
  --model models/sentence-boundary/v0 \
  --previous "Who shot John F." \
  "Kennedy?"
```

Evaluate a trained model on the test split:

```sh
just sentence-boundary eval \
  --model models/sentence-boundary/v0 \
  --data datasets/sentence-boundary/v0 \
  --split test
```

Evaluation first validates the artifact manifest, then loads the requested split and
runs model inference over each sampled example using the same input/output projection
as `infer`.  Reported metrics include:

- exact action accuracy (how often the predicted action token matches gold);
- boundary precision, recall, and F1 (treating Emit, Repair, and MissingHead as the
  positive class);
- no-boundary precision, recall, and F1 (treating Continue as the positive class);
- mean character-level edit distance for Repair examples;
- invalid output rate (predictions that cannot be parsed as any known action token);
- a deterministic sample of up to eight disagreements between gold and predicted output.

Use `--limit` and `--seed` for quick bounded checks:

```sh
just sentence-boundary eval \
  --model models/sentence-boundary/v0 \
  --data datasets/sentence-boundary/v0 \
  --split test \
  --limit 50 \
  --seed 7
```

Write machine-readable metrics to a JSON file:

```sh
just sentence-boundary eval \
  --model models/sentence-boundary/v0 \
  --data datasets/sentence-boundary/v0 \
  --split test \
  --report eval_metrics.json
```

Stream stdin into newline-delimited sentences:

```sh
printf 'Who shot John F.\nKennedy? Another sentence.\n' | just parse --model models/sentence-boundary/v0
```

`just parse` uses `sentence-boundary stream`. On `<boundary:repair>`, it writes the configured repair control sequence before the repaired sentence; the default is ANSI cursor-up plus erase-line (`ESC[1A ESC[2K`) so terminal consumers can replace the prior emitted line.

The model sees only:

```text
<task:sentence_boundary><ctx:previous>...<ctx:cursor>...
```

It does not receive the next sentence. Targets use these control tokens:

```text
<boundary:continue>
<boundary:emit><sentence>\n
<boundary:missing_head><tail fragment>
<boundary:repair><repaired sentence>
```

The repair class covers bad prior cuts such as:

```text
previous = "Who shot John F."
cursor   = "Kennedy?"
target   = "<boundary:repair>Who shot John F. Kennedy?"
```

The legacy `sentence-boundary parse` command still emits the existing rule-based `speaking::syntax::GrammarAnalysis` shape for compatibility.
