# Licensing Notes

The source code in this repository is MIT licensed. Generated datasets, downloaded lexicons, synthesized audio, and referenced media may include or point to material with different terms.

Treat prepared data directories as local artifacts unless you have reviewed their generated `README.md`, `dataset_config.json`, and per-row provenance.

## Data And Audio Sources

| Source/backend | Use | License/terms note |
|---|---|---|
| OpenEPD (`open-english-pronouncing-dictionary`) | Primary lexical source for spelling, IPA variants, rarity, and source labels. | OpenEPD is documented upstream as CC-BY-SA 4.0 because it includes WikiPron/Wiktionary-derived data. |
| WikiPron/Wiktionary-derived labels | Preserved through OpenEPD source labels and used to add Wiktionary reference URLs. | WikiPron/Wiktionary material is share-alike; preserve attribution and license notes when redistributing generated data. |
| `speaking` crate phonemicizer | Derives narrow phones, syllables, stress, and placeholder acoustic features locally. | Project-local code under this repository's license. |
| eSpeak NG | Optional local WAV generation with a small rotating voice set. | eSpeak NG is GPL-3-or-later; some data/docs mention CC-BY-SA components. Review eSpeak NG terms before redistributing generated audio. |
| Google Translate TTS URL support (`tts-urls`) | Optional network audio backend; skipped when robots policy disallows the TTS path. | URL helper crate is MIT, but Google service output/access is governed by Google's terms and robots policy; this project is not affiliated with Google. |
| Wiktionary/Wikimedia audio | Optional best-effort audio lookup through public file metadata/audio URLs, only when robots policy allows. | Individual media files may have their own licenses; keep source URLs/provenance with any redistributed audio. |
| Wikimedia Commons pronunciation audio | Optional real-human pronunciation audio lookup from allowed Commons file pages and direct media URLs. | Individual Commons files carry their own licenses; prepare preserves source URL, license label, and attribution in provenance. |
| AnySpeak | Optional local MP3 generation through an AnySpeak checkout (`anyspeak_dir` or `ANYSPEAK_DIR`). | AnySpeak is AGPL-3 and Qwen3-TTS-based; review AnySpeak and model/output terms before redistributing generated audio. |
| Dictionary.com | Reference URL metadata only. | Pages are not fetched by prepare; respect Dictionary.com's terms if using those links manually. |
| StyleTTS2/Piper | Opportunistic local synthesis backends through installed local models. | Model/audio asset terms depend on the specific installed assets. |

## Redistribution Checklist

- Preserve source URLs and provenance metadata.
- Preserve attribution and license labels for any copied audio.
- Review share-alike obligations before combining generated data with other datasets.
- Keep scraped or robots-disallowed resources out of redistributed artifacts.
- Review terms for local synthesis engines and model checkpoints before publishing generated audio.
