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

After one roughly five-hour Wiktionary epoch, the `just race` smoke test looked
promising: the model completed all 44 inference demos without failures, and the
remaining pronunciation/spelling errors are mostly plausible native-speaker
spellings or approximations rather than random output.

```text
race: building tongues binary
race: 23 forms, 8 configured Wiktionary languages, compact task coverage
race: g2p2g=models/g2p2g/openepd-v0, wiktionary=models/wiktionary/enwiktionary-2026-06-01-v0-phones
race plan: g2p2g=23 rt, wiktionary=11 rt, wiktionary task demos=9 + raw

G2P2G round trips (compact stress sample)
  ok  176ms +  186ms  have           -> hæv                -> have
  ok  414ms +  336ms  children       -> ˈtʃɪl.dɹən         -> children
  ok  197ms +  196ms  through        -> ˈθɹu               -> thru
  ok  201ms +  153ms  queue          -> ˈkju               -> cue
  ok  711ms +  578ms  Tyrannosaurus  -> tɪɹ.æ.nə.sɔɹ.əs    -> tyranosorous
  ok  820ms +  674ms  Archaeopteryx  -> ˌɑɹ.kiˈɑp.təɹ.ɪks  -> archaeopterics
  ok  787ms +  589ms  Velociraptor   -> vəˈlɑ.səɹˌæp.təɹ   -> velocerapter
  ok  839ms +  743ms  Quetzalcoatlus -> kwɛt.sæl.koʊt.ləs  -> quetsalcoatless
  ok  959ms +  705ms  Parasaurolophu... -> ˌpɛɹ.ə.sɔɹˈɑ.lə.fə... -> parasorolifous
  ok 1203ms + 1053ms  Pachycephalosa... -> pæ.kɪ.sɛ.fə.lə.sɔɹ... -> packissephalosorou...
  ok 1751ms +  919ms  Micropachyceph... -> ˌmaɪ.kɹoʊ.pəˌkaɪ.s... -> micropocycifolors
  ok  610ms +  477ms  Coelophysis    -> ˌsi.ləˈfɪ.zɪs      -> sealifisis
  ok  166ms +  173ms  Yi             -> ˈji                -> yee
  ok  346ms +  290ms  mañana         -> məˈhɑ.nə           -> mahana
  ok  565ms +  314ms  jalapeño       -> dʒɑ.lɑˈpɛ.noʊ      -> jalapeno
  ok  376ms +  418ms  brötchen       -> ˈbɹʌ.tʃən          -> brutcheon
  ok  464ms +  413ms  Kraftwerk      -> ˈkɹæf.twɚk         -> craftwork
  ok  639ms +  430ms  Pteranodon     -> təɹ.æ.nə.ʊ.dɑn     -> teranodon
  ok  267ms +  215ms  Łódź           -> ˈɛˈdeɪ             -> eday
  ok  275ms +  240ms  Dvořák         -> dvəˈʊk             -> dvoke
  ok  446ms +  310ms  São Paulo      -> ˌsu.pɔˈloʊ         -> supallo
  ok  398ms +  276ms  ἄνθρωπος       -> ˈɛ.tʃəˌɡɑ          -> echiga
  ok  257ms +  180ms  कर्म           -> ˈju.ɡə             -> uga

Wiktionary orthography/phonology round trips (11 curated cases)
  ok  852ms +  702ms  eng/phonemes Tyrannosaurus      -> taɪˈɹænəsɔːɹəs       -> tyranosorous
  ok 1117ms +  830ms  eng/phones   Archaeopteryx      -> ɑɹˈt͡ʃɛə̯.ə.ptə.ɹɪks -> archaropterics
  ok  884ms +  677ms  lat/phonemes Velociraptor       -> vɛloˈsiʁaptoːɐ̯      -> velosiraptor
  ok  894ms +  759ms  eng/phones   Quetzalcoatlus     -> ˈkwɛtsəlˌkoʊtləs     -> quetzlecotless
  ok 1117ms +  824ms  lat/phonemes Parasaurolophus    -> pa.ʁa.zaʊ̯.ʁoˈlo.fus -> parazaurolofus
  ok  404ms +  372ms  spa/phonemes mañana             -> maˈɲana              -> mañana
  ok  453ms +  435ms  spa/phones   jalapeño           -> xalaˈpeɲo            -> jalapeño
  ok  506ms +  458ms  fra/phones   rendezvous         -> ʁɛnde.zvo            -> rendesvo
  ok  511ms +  456ms  deu/phonemes brötchen           -> ˈbʁøːtçən            -> brötchen
  ok  499ms +  455ms  grc/phonemes ἄνθρωπος           -> ˈanθropos            -> άνθροπος
  ok  439ms +  311ms  san/phonemes कर्म               -> ˈʔa.fi.n             -> άφfηn

Wiktionary task demos
  ok 1051ms  orthography-to-phones --variety en-GB.RP Archaeopteryx -> ɑːˈtʃɛə̯.ə.ptə.ɹɪks
  ok 1004ms  orthography-to-phonemes                Archaeopteryx -> ɑː(ɹ)ˈtʃiːəptəɹɪks
  ok  710ms  phonemes-to-orthography                ɑː(ɹ)ˈtʃiːəptəɹɪks -> archioptorics
  ok  783ms  phones-to-orthography                  ɑːˈtʃɛə̯.ə.ptə.ɹɪks -> archaropterics
  ok  987ms  phonetic-realization                   ɑː(ɹ)ˈtʃiːəptəɹɪks -> ɑːɹˈtʃiːəptʰɹɪks
  ok  788ms  normalize                              Archaeopteryx! -> archaeopteryxe
  ok  254ms  guess-lang-from-orthography            Archaeopteryx -> deu
  ok  241ms  guess-lang-from-phonology              ɑːˈtʃɛə̯.ə.ptə.ɹɪks -> eng
  ok  287ms  guess-lang-from-orthography-and-phonology Archaeopteryx => ɑːˈtʃɛə̯.ə.... -> eng
  ok 1157ms  --raw tagged source                    <task:orthography_to_phonolo... -> ɑɹˈt͡ʃɛə̯.ə.ptə.ɹɪks

race: done in 43993ms wall; 44 successful inference demos, 0 failures, 43990ms summed inference time
```
