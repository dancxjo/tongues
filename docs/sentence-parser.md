# Sentence Parser

`sentence-parser` is a seq2seq cursor-boundary model family. It is trained from Project Gutenberg-style plain text using the `seams` sentence detector as the teacher.

Prepare data with the default sources:

```sh
just sentence-parser prepare \
  --out datasets/sentence-parser/v0
```

Start a fresh default run by archiving the existing default dataset/model artifacts and recreating empty directories:

```sh
just sentence-parser clean --all
```

Use `--data` or `--model` to archive only one side. Artifacts are moved under `archive/<run-id>/...`; pass `--run-id NAME` for a stable archive folder or `--no-create` if you do not want empty defaults recreated.

With the default config, preparation downloads a small Project Gutenberg cache and generates deterministic synthetic sentence-boundary cases. Local text files or directories can still override those defaults:

```sh
just sentence-parser prepare \
  --input /path/to/gutenberg_texts \
  --out datasets/sentence-parser/v0
```

Train:

```sh
just sentence-parser train \
  --data datasets/sentence-parser/v0 \
  --out models/sentence-parser/v0
```

Prepare and train in one command:

```sh
just sentence-parser train --prepare \
  --input /path/to/gutenberg_texts \
  --data datasets/sentence-parser/v0 \
  --out models/sentence-parser/v0
```

Preparation also runs a deliberately naive punctuation splitter and compares it to `seams`. Useful over-split disagreements are saved to `naive_discrepancies.jsonl` and folded into the default training splits as `row_source = "naive-discrepancy"` correction rows.

Train only the clean `seams` rows, only mined corrections, or both:

```sh
just sentence-parser train --training-set seams
just sentence-parser train --training-set naive-discrepancy
just sentence-parser train --training-set all
```

Cursor inference:

```sh
just sentence-parser infer \
  --model models/sentence-parser/v0 \
  --previous "Who shot John F." \
  "Kennedy?"
```

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

The legacy `sentence-parser parse` command still emits the existing rule-based `speech::syntax::SentenceSyntaxAnalysis` shape for compatibility.
