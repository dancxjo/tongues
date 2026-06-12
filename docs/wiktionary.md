# Wiktionary Model Family

The `wiktionary` model family prepares multilingual pronunciation rows from English Wiktionary dumps and trains orthography/phonology tasks over phonemic and phonetic representations.

The default model artifact is documented in [models/wiktionary-default.md](models/wiktionary-default.md).

## Prepare

```sh
cargo run --release -- wiktionary prepare \
    --out datasets/wiktionary/enwiktionary-2026-06-01-v0
```

This downloads the English Wiktionary MediaWiki XML bzip2 dump from the configured Wikimedia dump index:

```text
https://dumps.wikimedia.org/other/mediawiki_content_current/enwiktionary/2026-06-01/xml/bzip2/
```

The parser streams a decompressed MediaWiki XML dump and extracts `{{IPA}}`, `{{audio}}`, `{{homophones}}`, and `{{rhymes}}` pronunciation-section patterns for `eng`, `fra`, `deu`, `spa`, `lat`, `ell`, `grc`, and `san`.

Slash-delimited `/phonemes/` are written to `phonemes.jsonl`; bracket-delimited `[phones]` are written separately to `phones.jsonl`. Model-facing rows normalize orthography and pronunciation payloads with Unicode NFC, then expand into orthography-to-phonology, phonology-to-orthography, phonetic-realization, and language-guessing tasks.

Spanish page titles with a Spanish section also get synthetic phonemic rows when `synthesize_spanish = true` in the Wiktionary config, which is the default. The generator emits Castilian Spanish and standard Latin American Spanish variants from regular orthography, including `c/z` seseo-vs-`θ`, `ll/y`, silent `h`, `qu/gu`, contextual `c/g`, and `r/rr`.

Supplemental Wiktionary collation is enabled by default with `include_wiktionary_supplements = true`. It writes `supplemental_terms.jsonl` and duplicates matching pronunciation rows with domain variety tags for English Greek-derived names, Latin, neo-Latin/scientific names, and legal Latin. Terms without a pronunciation row are preserved in `supplemental_terms.jsonl` for review but are not fabricated into pronunciation examples.

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

`just race` demonstrates the G2P2G and Wiktionary defaults without trying every word against every language.

```sh
just race --cpu
just race --skip-build Archaeopteryx Quetzalcoatlus mañana brötchen
```

The default list is deliberately short and jagged: common English irregulars, dinosaur and taxonomic names, and Unicode-heavy forms such as `mañana`, `brötchen`, `Łódź`, `Dvořák`, `ἄνθρωπος`, and `कर्म`.

The race output prints abbreviated counts up front, for example `g2p2g=23 rt`, `wiktionary=11 rt`, and `wiktionary task demos=9 + raw`. "Successful" means the inference command completed; it is not an exact-match score.

The run is useful mostly as a smoke test and terminology check. It exercises phonemes and phones as distinct representations, runs phonetic realization from phonemes to phones, and keeps the raw-control example visible so vocabulary/control-token regressions are easy to spot.
