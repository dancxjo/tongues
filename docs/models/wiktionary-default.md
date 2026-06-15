# Wiktionary Default Pronunciation Model

Path: `models/wiktionary/enwiktionary-2026-06-01-v0-phones`

This is the default `wiktionary` seq2seq pronunciation model used by:

```sh
cargo run --bin tongues -- wiktionary train
cargo run --bin tongues -- wiktionary infer --model models/wiktionary/enwiktionary-2026-06-01-v0-phones ...
```

## Versioned artifact policy

The repository tracks only the minimal artifact set needed for inference and safe resume:

- `model.bin`: best model weights
- `model-epoch-11.bin`: latest epoch checkpoint referenced by `train_state.json`
- `train_state.json`: resume state
- `model_config.json`, `train_config.json`, `wiktionary_config.json`, `manifest.json`
- `vocab.json`

Older epoch checkpoints are intentionally left ignored. They are useful locally but not necessary to avoid losing the current trained model.

Current tracked binary checksums:

```text
e6977aadfe79be4df91a255ccd2b4e403d1fcfcbbb84bde3dc7d527451f3c1e6  model.bin
e6977aadfe79be4df91a255ccd2b4e403d1fcfcbbb84bde3dc7d527451f3c1e6  model-epoch-11.bin
1ada0512cadd646fd0faa5d732ef2df50060005758c871bb7f6344ade8cbfbb3  vocab.json
```

## Current state

`train_state.json`:

```json
{
  "current_epoch": 11,
  "best_val_loss": 0.08179155
}
```

`manifest.json`:

```json
{
  "schema_version": 1,
  "family": "wiktionary",
  "architecture": "seq2seq-transformer",
  "created_by": "tongues",
  "data_id": "enwiktionary-2026-06-01-v0",
  "task": "phonemic+phonetic:all"
}
```

Model shape:

```json
{
  "vocab_size": 2733,
  "d_model": 128,
  "n_heads": 4,
  "n_layers": 3,
  "d_ff": 512,
  "dropout": 0.1,
  "max_seq_len": 128
}
```

## Data and task mix

The current checked-in model was trained before the later default-language expansion that adds Latin, Greek, Ancient Greek, Sanskrit, Spanish synthetic rows, and supplemental Greek-name/legal/scientific collation. Its saved `wiktionary_config.json` has:

```json
{
  "languages": ["eng", "fra", "deu", "spa"],
  "train_notations": ["phonemic", "phonetic"],
  "train_task": "all",
  "include_reverse": true,
  "include_language_guessing": true,
  "seed": 777
}
```

The next `cargo run --bin tongues -- wiktionary train --prepare` will rebuild the prepared dataset from the current config and include the expanded language/supplement data before training. Because this checked-in model has the older `vocab_size=2733`, in-place continuation filters out prepared examples containing tokens outside the existing vocab and reports the skipped counts. To train every expanded Latin/Greek/Sanskrit/script row, use a fresh `--out` directory so the trainer can build a new vocabulary and initialize a compatible model.

## Training history

Captured from the interrupted local run on June 12, 2026:

```text
cargo run --bin tongues -- wiktionary train --prepare
Parsing dump: 36000 pages, 43846 patterns, 20811 phonemes, 3459 phones, 0 PIE roots

cargo run --bin tongues -- wiktionary train
Loaded 281052 rows for phonemic+phonetic
Selected 1580700 Wiktionary examples for task=all
Encoded 1264560 train / 158070 valid examples with vocab size 2733
Starting Wiktionary training...
  lr=0.0003 wd=0.0001 dropout=0.1 epochs=20 patience=5 batch_size=64
  device: CUDA GPU
Resuming training from epoch 1 checkpoint: models/wiktionary/enwiktionary-2026-06-01-v0-phones/model-epoch-1.bin

Epoch 2: checkpoint saved, new best val_loss=0.1010
Epoch 3: train_loss=0.1125 val_loss=0.0969 val_exact_match=0.733 val_token_acc=0.911
Epoch 4: train_loss=0.1007 val_loss=0.0909 val_exact_match=0.724 val_token_acc=0.914
Epoch 5: train_loss=0.0945 val_loss=0.0904 val_exact_match=0.738 val_token_acc=0.917
Epoch 6: train_loss=0.0905 val_loss=0.0876 val_exact_match=0.736 val_token_acc=0.918
Epoch 7: train_loss=0.0876 val_loss=0.0878 val_exact_match=0.743 val_token_acc=0.919
Epoch 8: train_loss=0.0854 val_loss=0.0857 val_exact_match=0.742 val_token_acc=0.921
Epoch 9: train_loss=0.0820 val_loss=0.0849 val_exact_match=0.739 val_token_acc=0.920
Epoch 10: train_loss=0.0807 val_loss=0.0823 val_exact_match=0.741 val_token_acc=0.920
Epoch 11: train_loss=0.0796 val_loss=0.0818 val_exact_match=0.740 val_token_acc=0.921
Epoch 12: interrupted at 12008/19759 batches
```

Resume command:

```sh
cargo run --bin tongues -- wiktionary train
```

The trainer should resume from epoch 12 using `model-epoch-11.bin`.

## Race progress snapshot: June 13, 2026

After one roughly five-hour Wiktionary epoch, the `just race` smoke test looked promising: the model completed all 44 inference demos without failures, and the remaining pronunciation/spelling errors are mostly plausible native-speaker spellings or approximations rather than random output.

```text
race: 23 forms, 8 configured Wiktionary languages, compact task coverage
race plan: g2p2g=23 rt, wiktionary=11 rt, wiktionary task demos=9 + raw
race: done in 43993ms wall; 44 successful inference demos, 0 failures, 43990ms summed inference time
```

Representative rows:

```text
G2P2G:
  have           -> hæv                -> have
  children       -> ˈtʃɪl.dɹən         -> children
  through        -> ˈθɹu               -> thru
  queue          -> ˈkju               -> cue
  Tyrannosaurus  -> tɪɹ.æ.nə.sɔɹ.əs    -> tyranosorous

Wiktionary:
  spa/phonemes mañana       -> maˈɲana              -> mañana
  spa/phones   jalapeño     -> xalaˈpeɲo            -> jalapeño
  deu/phonemes brötchen     -> ˈbʁøːtçən            -> brötchen
  grc/phonemes ἄνθρωπος     -> ˈanθropos            -> άνθροπος
  san/phonemes कर्म         -> ˈʔa.fi.n             -> άφfηn
```

## Expanded training snapshot: June 14, 2026

A fresh expanded run rebuilt the Wiktionary task set with 8 configured languages and a larger vocabulary:

```text
Loaded 5,732,554 train / 716,569 valid prepared rows
Encoded 5,732,554 train / 716,569 valid examples with vocab size 4,945
lr=0.0003 wd=0.0001 dropout=0.1 epochs=20 patience=5 batch_size=64
```

Validation improved strongly through epoch 6, then began to flatten:

```text
Epoch 1 | train_loss=0.0603  val_loss=0.0397  val_exact_match=0.890  val_token_acc=0.970
Epoch 2 | train_loss=0.0364  val_loss=0.0373  val_exact_match=0.899  val_token_acc=0.973
Epoch 3 | train_loss=0.0334  val_loss=0.0351  val_exact_match=0.900  val_token_acc=0.974
Epoch 4 | train_loss=0.0318  val_loss=0.0340  val_exact_match=0.900  val_token_acc=0.974
Epoch 5 | train_loss=0.0307  val_loss=0.0322  val_exact_match=0.902  val_token_acc=0.975
Epoch 6 | train_loss=0.0300  val_loss=0.0320  val_exact_match=0.906  val_token_acc=0.977
Epoch 7 | train_loss=0.0294  val_loss=0.0323  val_exact_match=0.904  val_token_acc=0.975
Epoch 8 | train_loss=0.0290  val_loss=0.0324  val_exact_match=0.905  val_token_acc=0.976
```

Epoch 6 is the best checkpoint reported so far. Later checkpoints keep reducing training loss but show the first signs of a validation shelf.

## Expanded race snapshot: June 15, 2026

After expanding `just race --cpu` to 54 G2P2G forms and 29 curated Wiktionary round-trip cases, the smoke test completed all 93 inference demos without a runtime failure:

```text
race: 54 forms, 8 configured Wiktionary languages, compact task coverage
race plan: g2p2g=54 rt, wiktionary=29 rt, wiktionary task demos=9 + raw
race: done in ~102s wall; 93 successful inference demos, 0 failures
```

### G2P2G: English sight-word and nonce-word strength

The dedicated G2P2G model remains notably adept at English. Common sight words and compact irregulars round-trip cleanly:

```text
the    -> ði     -> the
and    -> ænd    -> and
said   -> ˈsɛd   -> said
one    -> ˈwʌn   -> one
two    -> ˈtu    -> two
have   -> hæv    -> have
come   -> ˈkʌm   -> come
where  -> ˌwɛɹ   -> where
laugh  -> ˈlæf   -> laugh
```

Longer compositional forms also hold together:

```text
unhelpfulness        -> ʌnˈhɛlp.fəl.nəs       -> unhelpfulness
rediscovering        -> ˌɹi.dɪˈskʌ.vəɹ.ɪŋ     -> rediscovering
reclassification     -> ɹəˌklæ.sə.fəˈkeɪ.ʃ... -> reclassification
microbiological      -> ˌmaɪ.kɹoʊˌbaɪ.əˈlɑ... -> microbiological
internationalization -> ˌɪn.təɹˌnæ.ʃə.nə.l... -> internationalizati...
hyperconnected       -> ˌhaɪ.pəɹ.kəˈnɛk.tɪ... -> hyperconnected
```

English nonce probes are especially encouraging:

```text
glimmerthorn  -> ˈɡlɪ.məɹˌθɔɹn   -> glimmerthorn
brindlewise   -> ˈbɹɪn.dəlˌwaɪz  -> brindlewise
sprockleton   -> ˈspɹɑ.kəl.tən   -> sprockleton
mindlecrate   -> ˈmɪn.dəlˌkɹeɪt  -> mindlecrate
```

The sight-word behavior is striking: high-frequency English forms are recovered cleanly, while unfamiliar or invented words are sounded out according to learned English regularities. This looks more like early literacy than dictionary lookup.

### Wiktionary: English is improving, but generalization is visible

The Wiktionary model shows a different, more diagnostic behavior. Sight-word probes expose regularization pressure, while later runs also show recovery on several English cases:

```text
eng/phonemes said                    -> seɪd                         -> sayed
eng/phones   where                   -> ˈʍɛɹ̩                        -> where
eng/phonemes unhelpfulness           -> ʌnˈhɛlpfəlnəs                -> unhelpfulness
eng/phones   internationalization    -> ˌɪn.tɚˈneɪ.ʃnə.lɪˈze...      -> internationalization
eng/phones   brindlewise             -> ˈbɹɪndɫ̩ˌwaɪz                -> brindlewise
eng/phones   Archaeopteryx           -> ˌɑɹ.kiˈɑp.tə.ɹɪks            -> archiopterics
```

`said -> seɪd -> sayed` is useful rather than merely bad: it shows the model applying a regular sound-to-spelling rule where the dedicated G2P2G model has already memorized the irregular English word. `where`, `internationalization`, and `brindlewise` show that the Wiktionary task model can still recover exact English spellings when the task conditioning and representation line up.

### Phoneme/phone distinction

The phoneme/phone distinction is visibly being observed. Phone rows carry extra surface detail such as aspiration, syllabicity, labialization, rhotic coloring, tie bars, glottal onsets, nasalization, and offglides:

```text
eng/phones where              -> ˈʍɛɹ̩
eng/phones brindlewise        -> ˈbɹɪndɫ̩ˌwaɪz
eng/phones Quetzalcoatlus     -> kʰɛˈtsɑːlkoʊ̯tʰləs
fra/phones rendezvous         -> ʁã.de.d͡zvu
deu/phones Wiedervereinigung  -> ˈviːdɐfɛɐ̯ˌʔaɪ̯nɪɡʊŋ
lat/phones praefulgeo         -> pʁefʊlˈd͡ʒio
```

The phonemic rows are usually broader and cleaner:

```text
spa/phonemes mañana             -> maˈɲana
spa/phonemes desafortunadamente -> desafɔɾtunadaˈmente
deu/phonemes Sonnenklangerei    -> ˈzɔnənˌklaŋəʁaɪ̯
lat/phonemes ventoribus         -> vɛnˈtoːɾibʊs
grc/phonemes ἄνθρωπος           -> ˈanθɾopos
san/phonemes कर्म               -> kɐɾm
```

### Productive multilingual successes

Several non-English examples round-trip impressively or almost so:

```text
spa/phonemes mañana             -> maˈɲana              -> mañana
spa/phonemes desafortunadamente -> desafɔɾtunadaˈmente  -> desafortunadamente
fra/phones   lumivrage          -> ly.mi.vʁaʒ           -> lumivrage
deu/phones   Wiedervereinigung  -> ˈviːdɐfɛɐ̯ˌʔaɪ̯nɪɡʊŋ -> Wiedervereinigung
deu/phonemes Sonnenklangerei    -> ˈzɔnənˌklaŋəʁaɪ̯     -> Sonnenklangerei
lat/phonemes Velociraptor       -> velosirapˈtoːr       -> velosiraptor
lat/phonemes ventoribus         -> vɛnˈtoːɾibʊs         -> ventoribus
grc/phonemes ἄνθρωπος           -> ˈanθɾopos            -> άνθροπος
grc/phones   φιλοσοφία          -> fi.lo.soˈfi.a        -> φυλοσοφία
grc/phonemes νεφελόφως          -> ne.feˈlo.fos         -> νεφελόφος
```

These are not all exact or ideal outputs, but the regularities are clearly language-shaped. German compounds and Greek phonology look especially promising.

### Structured failures remain valuable

The remaining failures are no longer just noise. They show language leakage, script drift, casing artifacts, and overgeneralized phonology:

```text
spa/phones   jalapeño       -> xalaˈpeɲo            -> Jalapeño
spa/phones   clarolumbre    -> klaʁoˈlumbɾe         -> Klarolumbre
fra/phones   rendezvous     -> ʁã.de.d͡zvu          -> randeju
deu/phonemes brötchen       -> ˈbʁœtçən             -> Bröttchen
lat/phones   praefulgeo     -> pʁefʊlˈd͡ʒio         -> prefulgio
san/phonemes कर्म           -> kɐɾm                 -> क्रम
san/phones   धर्मक्षेत्र    -> juː.ɐ́.mɐ.ki.ɡɐ.t͡sɐ -> URक्GACATSA
san/phonemes सुगमनिका       -> sú.ɡɐ.ni.ko          -> स्ूनκo
normalize Archaeopteryx!    -> archaeopteryxá
```

These failures suggest that the model has learned a shared multilingual orthography/phonology space, but still needs stronger task boundaries, script locking, and casing policy.

### Current interpretation

- G2P2G is already a strong English orthography-to-phonology-to-orthography model.
- The expanded Wiktionary model is learning cross-task structure rather than merely memorizing rows.
- The phone/phoneme distinction is visible in qualitative outputs.
- Validation performance is strong: best reported exact match 90.6%, token accuracy 97.7% over 716,569 validation examples.
- The most useful next improvements are likely curriculum and scoping changes: language batches, explicit script tags, explicit casing tags, and separate normalized-orthography vs entry-title reconstruction tasks.

The headline: this run shows structured linguistic behavior. The model is wrong in increasingly informative ways, which is exactly the kind of failure pattern worth studying.
