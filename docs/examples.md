# Examples

Tongues models are trained from lexicon rows, but prediction is generative. For reverse spelling especially, the model often produces plausible spellings rather than dictionary spellings.

## Basic Inference

```sh
just infer "farkle"
just infer --task p2g "ˈfɑɹ.kəl"
```

Example outputs:

```text
farkle     -> ˈfɑɹ.kəl
ˈfɑɹ.kəl  -> farkel
```

## Example Generalization

These fake, real-sounding words were run through forced spelling-to-phoneme prediction:

| Input | Output |
|---|---|
| `quadract` | `ˈkwɑˌdɹækt` |
| `listotheria` | `ˌlɪ.stəˈθɪɹ.iə` |
| `velliptor` | `ˈvɛ.ləp.təɹ` |

Some strings survive a full G2P/P2G round trip exactly; others come back as plausible alternate spellings. That is expected for P2G because a phoneme string usually has several credible English spellings.

## Invented Word Round Trips

```sh
just infer --cpu --task g2p "quadract"
just infer --cpu --task p2g "ˈkwɑˌdɹækt"
```

| Invented spelling | G2P prediction | P2G from predicted IPA |
|---|---|---|
| `quadract` | `ˈkwɑˌdɹækt` | `quadract` |
| `listotheria` | `ˌlɪ.stəˈθɪɹ.iə` | `listotheria` |
| `velliptor` | `ˈvɛ.ləp.təɹ` | `velopter` |
| `morvane` | `ˈmɔɹˌveɪn` | `morvane` |
| `glastifer` | `ˈɡlæ.stɪ.fɚ` | `glastifer` |
| `perulance` | `ˈpɛɹ.jə.ləns` | `parulance` |
| `dravoline` | `ˈdɹæ.vəˌlaɪn` | `dravaline` |
| `selquorin` | `ˈsɛl.kəɹ.ən` | `selkeren` |
| `brenthic` | `ˈbɹɛn.θɪk` | `brenthic` |
| `caldovar` | `ˈkæl.dəˌvɑɹ` | `caldavar` |
| `threnomy` | `θɹɛˈnɑ.mi` | `threnami` |
| `pluvaster` | `ˈpluˌvæ.stəɹ` | `pluvaster` |
| `nordelith` | `ˈnɔɹ.də.lɪθ` | `nordalith` |
| `cormivane` | `ˈkɔɹ.məˌveɪn` | `cormivane` |
| `astralon` | `ˈæ.stɹəˌlɑn` | `astralon` |
| `velquatic` | `vɛlˈkwɑ.tɪk` | `velquatic` |
| `grendolith` | `ˈɡɹɛn.dəˌlɪθ` | `grendalith` |
| `marispen` | `ˈmɛɹ.ə.spən` | `maraspen` |
| `torvellan` | `tɔɹˈvɛ.lən` | `torvellan` |
| `quoridance` | `ˈkwɔɹ.ə.dəns` | `quaridence` |
| `splinterax` | `ˈsplɪn.təɹˌæks` | `splinterax` |
| `avirenth` | `ˈæ.vəɹ.ənθ` | `averanth` |
| `clastoria` | `klæˈstɔɹ.iə` | `clastoria` |
| `mendriful` | `ˈmɛn.dɹə.fəl` | `mendriful` |
| `opterane` | `ˈɑp.təɹˌeɪn` | `opterain` |
| `zenthoria` | `zɛnˈθɔɹ.iə` | `zenthoria` |
| `draluvian` | `dɹəˈlu.viən` | `dralluvian` |
| `kestavorn` | `ˈkɛ.stəˌvɔɹn` | `kestivorn` |
| `florithium` | `flɔɹ.ɪ.θiəm` | `florithium` |
| `praxaline` | `ˈpɹæk.səˌlaɪn` | `praxoline` |
| `morthelion` | `mɔɹˈθi.liən` | `morthelian` |
| `vundricate` | `ˈvʌn.dɹəˌkeɪt` | `vundricate` |
