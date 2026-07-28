# Multilingual Interpretation Acceptance Corpus

Tongues keeps its v1 ambiguity and interpretation acceptance set in
`fixtures/interpretation/ambiguity-acceptance-v1.json`. It is an
evaluation-only corpus: the native grammar parser, phonemicizer and linguistic
claim ledger, and referenced Duplex streaming fixtures produce the actual
results. The JSON stores expectations and never substitutes expected values for
runtime output.

## Coverage

The corpus includes:

- English homophones, contextual heteronyms, articles selected from phonetic
  onset, weak/strong function-word intent, PP and complement attachment,
  coordination, infinitival/prepositional/citation `to`, contrast, vocative,
  apposition, parenthetical material, questions, fragments, and streaming
  garden-path repair;
- French as a Romance profile;
- Sanskrit as a case-rich/free-word-order profile;
- Chukchi as an existing lower-resource profile with grammar explicitly
  reported as partial instead of projecting English link categories;
- native rules, an actual optional UDPipe probe, an actual optional Link
  Grammar probe, and deterministic timeout, malformed-output, and token-mismatch
  degradation probes.

Every child of epic #175 is named by at least one end-to-end case. Corpus
validation rejects missing child coverage, missing Romance/case-rich/lower-
resource coverage, duplicate IDs, unknown sources, missing license or
provenance metadata, invalid streaming fixture references, and required link
categories on a case marked explicitly unsupported.

## Commands

The bounded CI profile needs no model download:

```sh
just interpretation-acceptance
```

It writes a complete diffable report to
`target/interpretation/ambiguity-acceptance-report.json` through a flushed
`.part` file and atomic rename. CI also invokes the same command explicitly
after building the release CLI.

Run every offline case:

```sh
just interpretation-acceptance full
```

Supply a trained Duplex checkpoint when learned contribution evidence is
available:

```sh
just interpretation-acceptance full \
  --learned-model models/duplex/prefix-transducer
```

Without a checkpoint, learned cases skip with an explicit reason. UDPipe and
Link Grammar likewise either report their actual accepted/partial state or skip
with the discovered readiness diagnostic. Optional backends never gate the
native CI path.

Use `--json` to emit the report to stdout as well as the durable report file.
The CLI prints bounded per-case/backend progress before evaluation completes.

## Report contract

`InterpretationAcceptanceReport` schema 1 contains:

- the canonical corpus SHA-256, selected profile, pass/fail totals, structured
  per-field diffs, and leakage-policy result;
- actual ranked parse IDs/ranks/variants, typed links, backend attempts, claim
  and resolution IDs, conflicts, lifecycle counts, selected and alternative
  pronunciation sequences, stress/boundary counts, provenance sources, and
  streaming output/repair/withdrawal/abstention/frontier evidence;
- parse-link agreement, ambiguity recall, top-k lexical accuracy,
  homophone/heteronym accuracy, selected-positive Brier score, repair
  precision/recall, pronunciation-selection accuracy, boundary/stress
  accuracy, and mean/p95 latency;
- separate native-rule, learned, external-backend, and combined contribution
  summaries.

The Brier value is explicitly the selected-positive score on this curated
acceptance set, not a claim of population calibration. Failure reports retain
the expected and actual values at stable paths such as
`grammar.required_links`, `claims.total`, and
`streaming.final_committed_text`.

## Provenance and leakage

Each source declares its license, provenance, evaluation-only status, and a
`training_exclusion_key`. The split policy lists every exclusion key and groups
related cases by `leakage_group`; corpus validation fails if this boundary is
incomplete. These fixtures must not be copied into model training or
augmentation inputs.

## V1 demo handoff

Issue #167 can use the existing ambiguity/repair journey without bespoke
setup:

```sh
just serve
```

Then open:

```text
/speech/operate?duplex_fixture=homophones
```

The same durable fixture exercised by the acceptance harness opens in Operate
with its repair, withdrawal, alternatives, conflicts, and evidence chain.
