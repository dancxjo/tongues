# Refinement

Tongues includes two G2P2G refinement workflows:

- discrepancy refinement from validation/test failures;
- sight-word refinement from the built-in Dolch list.

Both fine-tune a copy of an existing model. The source model directory is left untouched.

## Discrepancy Refinement

```sh
just refine
```

Direct form:

```sh
cargo run --release -- g2p2g refine \
    --model models/g2p2g/openepd-v0 \
    --data datasets/g2p2g/openepd-v0 \
    --out models/g2p2g/openepd-v0-refined \
    --splits valid,test \
    --task g2p \
    --verbose \
    --learning-rate 1e-4 \
    --epochs 5 \
    --patience 2
```

Refinement runs the model over held-out splits, looks up reference pronunciations in OpenEPD, normalizes them through the `speaking` notation and syllabification layer, compares each prediction with that gold target using a broad comparison key, computes character-level edit distance on that key, writes every substantive mismatch to `discrepancies.jsonl`, and fine-tunes from the source model weights using only the mismatched lexemes.

Example discrepancy:

```text
word : zweig
gold : ˈzwaɪɡ
pred : ˈzweɪɡ
```

The default task is `g2p`. Use `--task p2g` for phoneme-to-grapheme refinement, or `--task both` to mine and train both directions.

With `--verbose`, each discrepant word is printed with its split, task, edit distance, input, gold target, and prediction.

Length marks, syllable dots, stress mark placement, and common rhotic spellings are ignored for discrepancy detection so refinement does not train on merely notational differences.

OpenEPD entries containing IPA characters outside the existing model vocabulary are skipped, because the saved model cannot emit tokens that are not in its `vocab.json` without rebuilding the vocabulary and retraining.

Some discrepancies are regular patterns worth training. Others are sight-word exceptions and probably belong in an override table rather than in the productive model.

## Sight-Word Refinement

```sh
just sight-words
```

The default output is `models/g2p2g/openepd-v0-sight-words`.

Pass refinement flags after the recipe:

```sh
just sight-words --epochs 8 --learning-rate 5e-5
```

Direct form:

```sh
cargo run --release -- g2p2g refine \
    --model models/g2p2g/openepd-v0 \
    --data datasets/g2p2g/openepd-v0 \
    --out models/g2p2g/openepd-v0-sight-words \
    --source sight-words \
    --task both
```

Unlike the default discrepancy source, `--source sight-words` trains every usable sight-word lexeme after OpenEPD and vocabulary filtering. It still writes `discrepancies.jsonl` so current sight-word failures are visible before fine-tuning.

Sight-word refinement is meant for high-frequency irregular forms such as:

```text
one
two
yacht
colonel
choir
```

Not every pronunciation pattern is productive. The sight-word source gives the system a way to reinforce high-frequency irregular forms without requiring the productive model to contort itself around every historical spelling accident.
