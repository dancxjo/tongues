# Licensing Notes

Tongues-authored source code is MIT licensed. That repository-level license
does not relicense third-party source expression, model weights, voice packages,
generated datasets, downloaded lexicons, synthesized audio, or referenced
media. See the [third-party notices](../THIRD_PARTY_NOTICES.md) and
[speech provenance ledger](provenance.md) for the known Coqui/Piper
relationships and unresolved source-audit work.

Treat prepared data directories as local artifacts unless you have reviewed their generated `README.md`, `dataset_config.json`, and per-row provenance.

## Data And Audio Sources

| Source/backend | Use | License/terms note |
|---|---|---|
| OpenEPD (`open-english-pronouncing-dictionary`) | Primary lexical source for spelling, IPA variants, rarity, and source labels. | OpenEPD is documented upstream as CC-BY-SA 4.0 because it includes WikiPron/Wiktionary-derived data. |
| CMUdict 0.7b | Embedded English ARPAbet lexicon supplied by the checksum-pinned `arpabet_cmudict` crate. | CMUdict's BSD-style license is included in the dependency package; preserve its copyright and conditions in source/binary redistributions. |
| Lexique383 seed | Repository-bundled French lexicon seed for `speaking` French lookup. | Lexique383 is documented by its maintainers as CC BY-SA 4.0; preserve attribution and share-alike notes when redistributing full derived data. |
| WikiPron/Wiktionary-derived labels | Preserved through OpenEPD source labels and used to add Wiktionary reference URLs. | WikiPron/Wiktionary material is share-alike; preserve attribution and license notes when redistributing generated data. |
| `speaking` crate phonemicizer | Derives narrow phones, syllables, stress, and placeholder acoustic features locally. | Project-local code under this repository's license. |
| eSpeak NG | Optional local WAV generation with a small rotating voice set. | eSpeak NG is GPL-3-or-later; some data/docs mention CC-BY-SA components. Review eSpeak NG terms before redistributing generated audio. |
| Google Translate TTS URL support (`tts-urls`) | Optional network audio backend; skipped when robots policy disallows the TTS path. | URL helper crate is MIT, but Google service output/access is governed by Google's terms and robots policy; this project is not affiliated with Google. |
| Wiktionary/Wikimedia audio | Optional best-effort audio lookup through public file metadata/audio URLs, only when robots policy allows. | Individual media files may have their own licenses; keep source URLs/provenance with any redistributed audio. |
| Wikimedia Commons pronunciation audio | Optional real-human pronunciation audio lookup from allowed Commons file pages and direct media URLs. | Individual Commons files carry their own licenses; prepare preserves source URL, license label, and attribution in provenance. |
| AnySpeak | Optional local MP3 generation through an AnySpeak checkout (`anyspeak_dir` or `ANYSPEAK_DIR`). | AnySpeak is AGPL-3 and Qwen3-TTS-based; review AnySpeak and model/output terms before redistributing generated audio. |
| Dictionary.com | Reference URL metadata only. | Pages are not fetched by prepare; respect Dictionary.com's terms if using those links manually. |
| Local speech synthesis | Opportunistic synthesis through installed local models and compatible runtimes. | Model/audio asset terms depend on the specific installed assets. |

## Speech Source And Model Licenses

Source code, model weights, model configuration, datasets, and generated audio
are distinct licensable layers. Do not infer a model's license from the source
repository that published it.

The Coqui TTS `v0.6.1` source tag is MPL-2.0. Its model registry, however,
labels the cataloged LJSpeech SpeedySpeech and FastPitch models `TBD` and leaves
the relevant HiFi-GAN v2 and VCTK VITS license values blank. The downloaded
archives inspected by Tongues contain weights/configuration without a license
file. Those catalog entries are therefore `NOASSERTION`, not Apache-2.0, until
stable upstream evidence establishes redistributable terms. The published
Tacotron2-DDC, Glow-TTS, and MultiBand-MelGAN entries are exceptions: the
registry labels the LJSpeech Tacotron2-DDC artifact `Apache 2.0` and the latter
two artifacts `MPL`. An import must record that artifact evidence explicitly;
it must not infer it from the source license. The native Tacotron 2 and
Glow-TTS source files separately remain MPL-covered modifications because they
adapt Coqui inference graphs.

The plain MelGAN conformance checkpoint is Descript's `linda_johnson.pt` from
`descriptinc/melgan-neurips`, whose repository declares MIT. Its revision and
artifact checksum are pinned independently from the Coqui artifacts.

The published multilingual YourTTS entry is different from the unresolved
artifacts above: Coqui's model registry identifies its weights as
CC BY-NC-ND 4.0. The official catalog records `CC-BY-NC-ND-4.0`, so installing
or running the model does not grant commercial use or permission to distribute
modified weights. This model license is independent from both Tongues' MIT
code and Coqui's MPL-2.0 source implementation.

The Coqui `freevc24` voice-conversion registry entry declares MIT for the
published FreeVC weights. Its two required auxiliary artifacts are tracked
separately: WavLM-Large follows Microsoft's MIT-licensed WavLM implementation,
and the FreeVC speaker encoder follows the MIT-licensed FreeVC distribution.
The catalog pins all three download URLs, byte sizes, SHA-256 digests, and
license evidence. Installing the main archive alone is intentionally
insufficient for execution.

The official XTTS v2 repository publishes its model, tokenizer, conditioning
statistics, and examples under the Coqui Public Model License 1.0.0
(`LicenseRef-Coqui-Public-Model-License-1.0.0`). CPML 1.0.0 restricts the model
and its outputs to the license's non-commercial purposes and requires
recipients of the model, modifications, or outputs to receive the terms or
their URL. Local conversion to SafeTensors does not remove those restrictions.
Tongues must not catalog or redistribute converted XTTS artifacts under its MIT
source license.

Piper source is MIT, with its notice preserved in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). Piper-distributed voice
weights keep their individual cataloged license and evidence; Piper's MIT
source license does not apply automatically to those voices.

A complete reference copy of Apache-2.0 is stored under
[`LICENSES/`](../LICENSES/). The Coqui `v0.6.1`
[MPL-2.0 source license](https://github.com/coqui-ai/TTS/blob/v0.6.1/LICENSE.txt)
is linked from the notice; if an audit identifies MPL-covered files, the full
MPL text must accompany their distribution. Including a license text is not
itself evidence that the license applies to an artifact.

## Redistribution Checklist

- Preserve source URLs and provenance metadata.
- Preserve attribution and license labels for any copied audio.
- Review share-alike obligations before combining generated data with other datasets.
- Keep scraped or robots-disallowed resources out of redistributed artifacts.
- Review terms for local synthesis engines and model checkpoints before publishing generated audio.
- Do not redistribute a `NOASSERTION` model merely because Tongues can download
  or execute it; obtain and retain affirmative license evidence first.
- When source adaptation is confirmed, preserve the upstream file notice,
  revision, and license instead of relying only on the root MIT declaration.

## Linguistic Asset Updates

The machine-readable provenance record is
[`crates/speaking/assets/linguistic-assets.json`](../crates/speaking/assets/linguistic-assets.json).
Run `just prepare-assets` after changing an asset or dependency. That command
fetches the exact CMUdict package under Cargo registry checksum verification,
then rebuilds `speaking` with `CARGO_NET_OFFLINE=true` and verifies both the
pinned input and deterministic generated-artifact hashes.

For an update:

1. Pin the exact asset or crate version; never use a moving branch or URL.
2. Recompute the source and generated-artifact SHA-256 values in the manifest.
3. Review the upstream license and update this page if its obligations changed.
4. Run `just prepare-assets` twice and compare the generated artifact checksums.
