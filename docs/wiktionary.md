# Wiktionary Model Family

The `wiktionary` model family prepares multilingual pronunciation rows from English Wiktionary dumps and trains orthography/phonology tasks over phonemic and phonetic representations.

The default model artifact is documented in [models/wiktionary-default.md](models/wiktionary-default.md).

## Prepare

```sh
cargo run --release -- wiktionary prepare \
    --out datasets/wiktionary/enwiktionary-2026-06-01-v0
```

Start a fresh default run by archiving the existing default dataset/model artifacts and recreating empty directories:

```sh
cargo run --bin tongues -- wiktionary clean --all
```

Use `--data` or `--model` to archive only one side. Artifacts are moved under `archive/<run-id>/...`; pass `--run-id NAME` for a stable archive folder or `--no-create` if you do not want empty defaults recreated.

This downloads the English Wiktionary MediaWiki XML bzip2 dump from the configured Wikimedia dump index:

```text
https://dumps.wikimedia.org/other/mediawiki_content_current/enwiktionary/2026-06-01/xml/bzip2/
```

The parser streams a decompressed MediaWiki XML dump and extracts `{{IPA}}`, `{{audio}}`, `{{homophones}}`, and `{{rhymes}}` pronunciation-section patterns for `eng`, `fra`, `deu`, `spa`, `cym`, `lat`, `ell`, `grc`, and `san`.

Slash-delimited `/phonemes/` are written to `phonemes.jsonl`; bracket-delimited `[phones]` are written separately to `phones.jsonl`. Entry etymology templates from Etymology sections are written to `etymologies.jsonl`. Model-facing rows normalize orthography and pronunciation payloads with Unicode NFC, then expand into orthography-to-phonology, phonology-to-orthography, phonetic-realization, find-etymology, and language-guessing tasks.

Wiktionary uses a stable family vocabulary rather than deriving model shape from whichever shard happens to arrive first. The stable vocab seeds Wiktionary task/control tokens, cleanup tags, etymology relation/source tags, ASCII, IPA, major modern writing systems, CJK/Kana/Hangul, combining marks, punctuation, symbols, and many historic scripts. This makes `--while-preparing` viable because the model can start before final splits exist without growing its embedding table later.

Preparation writes durable checkpoints while it parses and expands:

| Path | When written | Resume behavior |
|---|---|---|
| `prepare-checkpoints/parse-pronunciation/pages-START-END.json` | During XML parsing, for the first few pages and then every 1,000 pages, plus the final tail. | If final parse artifacts are missing, prepare reloads these shards, skips already checkpointed XML pages, and continues from the next page. |
| `patterns.jsonl`, `phonemes.jsonl`, `phones.jsonl`, `etymologies.jsonl`, `supplemental_terms.jsonl` | After parsing completes, written through `.writing.part` files and renamed atomically. | If all five exist, prepare resumes from them without reparsing the dump. |
| `expanded.jsonl.writing.part` | While model-facing rows are expanded. | If `expanded.jsonl` exists with the current schema marker, prepare resumes from it. Interrupted partials are archived before rebuilding. |
| `train.jsonl`, `valid.jsonl`, `test.jsonl`, `vocab.json`, `dataset_config.json`, `README.md` | Final split and metadata phase, written atomically. | Existing completed parse/expanded artifacts are reused if final split files need to be rebuilt. |

For the default Just recipe, pass the model-family command through Just:

```sh
just wiktionary prepare --out datasets/wiktionary/enwiktionary-2026-06-01-cleanup-v0
```

To start training while another process is preparing the same dataset:

```sh
just wiktionary train \
  --data datasets/wiktionary/enwiktionary-2026-06-01-cleanup-v0 \
  --out models/wiktionary/enwiktionary-2026-06-01-cleanup-v0 \
  --while-preparing \
  --patience 1000000
```

`--while-preparing` watches `expanded.jsonl.writing.part` and `expanded.jsonl`, reads only complete JSONL rows, and advances one training epoch whenever enough new expanded rows are available. Once `train.jsonl`, `valid.jsonl`, `test.jsonl`, and `vocab.json` are complete, normal prepared training resumes from the same `train_state.json` and checkpoints.

`wiktionary train` now applies OpenEPD rarity weighting and English Dolch sight-word oversampling by default. This happens in training memory and is also written to `train.augmented.jsonl` for inspection/reuse, while leaving `train.jsonl`, `valid.jsonl`, `test.jsonl`, and `vocab.json` unchanged. Pass `--sight-words=false` to disable the sight-word oversampling layer.

Spanish page titles with a Spanish section also get synthetic phonemic rows when `synthesize_spanish = true` in the Wiktionary config, which is the default. The generator emits Castilian Spanish and standard Latin American Spanish variants from regular orthography, including `c/z` seseo-vs-`θ`, `ll/y`, silent `h`, `qu/gu`, contextual `c/g`, and `r/rr`.

Supplemental Wiktionary collation is enabled by default with `include_wiktionary_supplements = true`. It writes `supplemental_terms.jsonl` and duplicates matching pronunciation rows with domain variety tags for English Greek-derived names, Latin, neo-Latin/scientific names, and legal Latin. Terms without a pronunciation row are preserved in `supplemental_terms.jsonl` for review but are not fabricated into pronunciation examples.

English cleanup rows are enabled by default with `include_cleanup_corpus = true`. These pinned synthetic examples target recurring Wiktionary-model attractors:

| Bucket | Purpose |
|---|---|
| `cleanup:core-function-words` | High-frequency English function-word pronunciations and contexts such as `do`, `to-do`, and `how-do-you-do`. |
| `cleanup:hyphenated-compounds` | Compound pronunciation plus `<task:segment_compound>` and `<task:pronounce_segments>` auxiliary rows. |
| `cleanup:letter-symbol-word-disambiguation` | Explicit `<WORD>`, `<LETTER>`, and `<PHONEME>` tags for forms like `a`, `i`, and `s`. |
| `cleanup:spelling-hallucination-negatives` | Gold pronunciations and `<task:verify_pronunciation>` GOOD/BAD contrastive rows for known bad outputs. |
| `cleanup:dialect-tagged-variants` | Explicit `<en-US>` and `<en-UK>` rows for common dialect splits such as `work`, `world`, `also`, and `both`. |
| `cleanup:broad-vs-narrow-equivalence` | `<task:normalize_phonology>` broad targets plus narrow allowed realizations for aspiration, dark L, offglides, and syllabic consonants. |

## Focused Language Runs

```sh
cargo run --release -- wiktionary prepare \
    --lang spa,fra \
    --out datasets/wiktionary/es-fr-focused-v0

cargo run --release -- wiktionary train \
    --data datasets/wiktionary/es-fr-focused-v0 \
    --out models/wiktionary/es-fr-focused-v0 \
    --lang spa,fra \
    --notation phonemes \
    --task all
```

When continuing an existing Wiktionary model, training reuses the saved `vocab.json`. Newly prepared examples containing tokens outside that vocabulary are skipped with a count instead of being silently encoded as `<UNK>`. Use a fresh `--out` directory when you want to train the full expanded language set with a rebuilt vocabulary.

This means `just wiktionary train ...` is safe to rerun against existing prepared datasets: it refreshes `train.augmented.jsonl` and keeps the baseline dataset files and model vocabulary stable.

## Inference

The default Wiktionary inference command is:

```sh
cargo run --release -- wiktionary infer \
    --model models/wiktionary/enwiktionary-2026-06-01-v0-phones \
    --task orthography-to-phones \
    --lang eng \
    --notation phones \
    "cat"
```

Inference options:

| Option | Default | Notes |
|---|---|---|
| `--model` | `models/wiktionary/enwiktionary-2026-06-01-v0-phones` | model directory |
| `--task` | `orthography-to-phones` | task selector |
| `--lang` | `eng` | Wiktionary language code for tagged tasks |
| `--notation` | `phones` | `phones` or `phonemes`; inference rejects `all` |
| `--variety` | unset | optional pronunciation variety control |
| `--raw` | unset | treat input as the exact tagged model source |
| positional `INPUT` | required | orthography, phoneme/phone sequence, combined language-guessing input, or raw source |

Supported `--task` values:

| Task | Example |
|---|---|
| `orthography-to-phonemes` | `cargo run --release -- wiktionary infer --task orthography-to-phonemes --lang eng --notation phonemes "cat"` |
| `orthography-to-phones` | `cargo run --release -- wiktionary infer --task orthography-to-phones --lang eng --notation phones "cat"` |
| `phonemes-to-orthography` | `cargo run --release -- wiktionary infer --task phonemes-to-orthography --lang eng --notation phonemes "kæt"` |
| `phones-to-orthography` | `cargo run --release -- wiktionary infer --task phones-to-orthography --lang eng --notation phones "ˈkʰæt"` |
| `phonetic-realization` | `cargo run --release -- wiktionary infer --task phonetic-realization --lang eng --variety en-US.GenAm --notation phonemes "kæt"` |
| `find-etymology` | `cargo run --release -- wiktionary infer --task find-etymology --lang eng "thorp"` |
| `segment-compound` | `cargo run --release -- wiktionary infer --task segment-compound --lang eng "how-do-you-do"` |
| `pronounce-segments` | `cargo run --release -- wiktionary infer --task pronounce-segments --lang eng "how | do | you | do"` |
| `verify-pronunciation` | `cargo run --release -- wiktionary infer --task verify-pronunciation --lang eng "get || d͡ʒɛt"` |
| `normalize-phonology` | `cargo run --release -- wiktionary infer --task normalize-phonology --lang eng "tʰuː"` |
| `normalize` | `cargo run --release -- wiktionary infer --task normalize --lang eng "Cat!"` |
| `guess-lang-from-orthography` | `cargo run --release -- wiktionary infer --task guess-lang-from-orthography --notation phones "cat"` |
| `guess-lang-from-phonology` | `cargo run --release -- wiktionary infer --task guess-lang-from-phonology --notation phones "ˈkʰæt"` |
| `guess-lang-from-orthography-and-phonology` | `cargo run --release -- wiktionary infer --task guess-lang-from-orthography-and-phonology --notation phones "cat => ˈkʰæt"` |

Variety and raw-source examples:

```sh
cargo run --release -- wiktionary infer \
    --task orthography-to-phones \
    --lang eng \
    --notation phones \
    --variety en-GB.RP \
    "cat"

cargo run --release -- wiktionary infer \
    --raw \
    "<task:orthography_to_phonology> <lang:eng> <repr:phones> cat"
```

## Race Smoke Test

`just race` is the compact smoke test for the active model families. It keeps the
Wiktionary family honest while `head2phones` and `interpretation` are still
training in parallel.

```sh
just race --cpu
just race --skip-build through brötchen mañana धर्मक्षेत्र

just wiktionary prepare --out datasets/wiktionary/enwiktionary-2026-06-01-v0
just wiktionary train --data datasets/wiktionary/enwiktionary-2026-06-01-v0 --out models/wiktionary/enwiktionary-2026-06-01-v0-phones
just wiktionary eval --data datasets/wiktionary/enwiktionary-2026-06-01-v0 --model models/wiktionary/enwiktionary-2026-06-01-v0-phones --split valid
just wiktionary infer --model models/wiktionary/enwiktionary-2026-06-01-v0-phones --task orthography-to-phones --lang eng --notation phones "cat"
just wiktionary clean --all

just head2phones prepare --out datasets/head2phones/v0
just head2phones train --data datasets/head2phones/v0 --out models/head2phones/v0 --prepare --wait-for-prepare

just interpretation prepare --subset mini --out datasets/interpretation/mini-v0
just interpretation train --data datasets/interpretation/mini-v0 --out models/interpretation/mini-v0 --wait-for-prepare
```

The default list is deliberately jagged: sight words, common English irregulars,
regular multi-morphemic English words, plausible nonce words, dinosaur and
taxonomic names, and Unicode-heavy forms such as `mañana`, `brötchen`, `Łódź`,
`Dvořák`, `ἄνθρωπος`, and `कर्म`. The Wiktionary round-trip sample also includes
real and imaginary-looking probes across the default target languages:
English, Spanish, French, German, Latin, Greek, and Sanskrit.

The race output prints abbreviated counts up front, for example `g2p2g=54 rt`,
`wiktionary=29 rt`, and `wiktionary task demos=9 + raw`. "Successful" means
the inference command completed; it is not an exact-match score.

The run is useful mostly as a smoke test and terminology check. It exercises
phonemes and phones as distinct representations, runs phonetic realization from
phonemes to phones, uses task-specific probes for language guessing, keeps the
raw-control example visible so vocabulary/control-token regressions are easy to
spot, and gives the nearby head2phones/interpretation training loops a
consistent companion example set.
