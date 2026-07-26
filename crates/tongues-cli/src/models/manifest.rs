#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Llm,
    Face,
    Asr,
    StyleTts2,
    VoiceModel,
    AcousticModel,
    NeuralVocoder,
    EndToEndSpeech,
    VoiceConversion,
    Lexicon,
    Phonemicizer,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelAsset {
    pub id: &'static str,
    pub filename: &'static str,
    pub relative_path: &'static str,
    pub url: &'static str,
    pub sha256: Option<&'static str>,
    pub size_bytes: Option<u64>,
    pub license: Option<&'static str>,
    pub source: Option<&'static str>,
    pub notes: Option<&'static str>,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelBundle {
    pub id: &'static str,
    pub display_name: &'static str,
    pub kind: ModelKind,
    pub primary_asset_id: &'static str,
    pub required_asset_ids: &'static [&'static str],
    pub aliases: &'static [&'static str],
}

#[derive(Debug, Clone, Copy)]
pub struct ArchiveMember {
    pub member_path: &'static str,
    pub relative_path: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelArchive {
    pub asset_id: &'static str,
    pub primary_member_path: &'static str,
    pub members: &'static [ArchiveMember],
}

pub const DEFAULT_LLM_MODEL_ID: &str = "gemma-4-e4b-it-q4-k-m";
pub const DEFAULT_FACE_MODEL_ID: &str = "face-insightface-buffalo-l";
pub const DEFAULT_ASR_MODEL_ID: &str = "whisper-large-v3-turbo";
pub const DEFAULT_STYLETTS2_MODEL_ID: &str = "styletts2-en-us";
pub const DEFAULT_VOICE_MODEL_ID: &str = "voice-ljspeech-high";
pub const DEFAULT_ACOUSTIC_MODEL_ID: &str = "speedy-speech-ljspeech";
pub const FASTPITCH_ACOUSTIC_MODEL_ID: &str = "fastpitch-ljspeech";
pub const GLOW_TTS_ACOUSTIC_MODEL_ID: &str = "glow-tts-ljspeech";
pub const DEFAULT_NEURAL_VOCODER_ID: &str = "hifigan-v2-ljspeech";
pub const MULTIBAND_MELGAN_VOCODER_ID: &str = "multiband-melgan-ljspeech";
pub const DEFAULT_END_TO_END_SPEECH_MODEL_ID: &str = "vits-vctk";
pub const YOURTTS_MODEL_ID: &str = "yourtts-multilingual";

pub const NON_SPEECH_MODEL_ASSETS: &[ModelAsset] = &[
    ModelAsset {
        id: "gemma-4-e4b-it-q4-k-m",
        filename: "gemma-4-E4B-it-Q4_K_M.gguf",
        relative_path: "models/gemma/gemma-4-E4B-it-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf",
        sha256: None,
        size_bytes: None,
        license: Some("LicenseRef-Gemma"),
        source: Some("https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF"),
        notes: None,
    },
    ModelAsset {
        id: "gemma-4-e4b-it-mmproj-bf16",
        filename: "mmproj-BF16.gguf",
        relative_path: "models/gemma/mmproj-BF16.gguf",
        url: "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-BF16.gguf",
        sha256: Some("ee01cba03fd9c71ea2ea722225d24a84f72e7197714367e550ef705ef8851bc6"),
        size_bytes: None,
        license: Some("LicenseRef-Gemma"),
        source: Some("https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF"),
        notes: None,
    },
    ModelAsset {
        id: "gemma-3-4b-it-q4-k-m",
        filename: "gemma-3-4b-it-Q4_K_M.gguf",
        relative_path: "models/gemma/gemma-3-4b-it-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
        sha256: None,
        size_bytes: None,
        license: Some("LicenseRef-Gemma"),
        source: Some("https://huggingface.co/unsloth/gemma-3-4b-it-GGUF"),
        notes: None,
    },
    ModelAsset {
        id: "face-scrfd-34g-gnkps",
        filename: "34g_gnkps.onnx",
        relative_path: "models/face/scrfd/34g_gnkps.onnx",
        url: "https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX/resolve/main/34g_gnkps.onnx",
        sha256: None,
        size_bytes: None,
        license: None,
        source: Some("https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX"),
        notes: None,
    },
    ModelAsset {
        id: "face-buffalo-l-w600k-r50",
        filename: "w600k_r50.onnx",
        relative_path: "models/face/buffalo_l/w600k_r50.onnx",
        url: "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
        sha256: None,
        size_bytes: None,
        license: None,
        source: Some("https://huggingface.co/public-data/insightface"),
        notes: None,
    },
    ModelAsset {
        id: "face-buffalo-l-genderage",
        filename: "genderage.onnx",
        relative_path: "models/face/buffalo_l/genderage.onnx",
        url: "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/genderage.onnx",
        sha256: None,
        size_bytes: None,
        license: None,
        source: Some("https://huggingface.co/public-data/insightface"),
        notes: None,
    },
    ModelAsset {
        id: "whisper-base-en",
        filename: "ggml-base.en.bin",
        relative_path: "models/whisper/ggml-base.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        sha256: None,
        size_bytes: None,
        license: Some("MIT"),
        source: Some("https://huggingface.co/ggerganov/whisper.cpp"),
        notes: None,
    },
    ModelAsset {
        id: "whisper-large-v3-turbo",
        filename: "ggml-large-v3-turbo.bin",
        relative_path: "models/whisper/ggml-large-v3-turbo.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
        sha256: None,
        size_bytes: Some(1_624_555_275),
        license: Some("MIT"),
        source: Some("https://huggingface.co/ggerganov/whisper.cpp"),
        notes: Some("Multilingual Whisper large-v3-turbo ggml model for transcript refinement."),
    },
    ModelAsset {
        id: "phonemicizer-en-us-builtin",
        filename: "phonemicizer-en-us-builtin.json",
        relative_path: "models/speaking/en-us/phonemicizer-en-us-builtin.json",
        url: "builtin://mortar-sea/en-us-phonemicizer",
        sha256: None,
        size_bytes: None,
        license: Some("MIT"),
        source: Some("speaking::data::varieties::english"),
        notes: Some("Built-in CMUdict-style seed lexicon plus explicit unknown-word fallback."),
    },
    ModelAsset {
        id: "lexicon-en-us-builtin",
        filename: "lexicon-en-us-builtin.json",
        relative_path: "models/speaking/en-us/lexicon-en-us-builtin.json",
        url: "builtin://mortar-sea/en-us-lexicon",
        sha256: None,
        size_bytes: None,
        license: Some("MIT"),
        source: Some("speaking::data::varieties::english"),
        notes: Some("Small deterministic built-in lexicon used by default tests and smoke runs."),
    },
    ModelAsset {
        id: "cmudict-base",
        filename: "cmudict.dict",
        relative_path: "models/speaking/en-us/cmudict.dict",
        url: "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict",
        sha256: None,
        size_bytes: None,
        license: Some("BSD-3-Clause"),
        source: Some("https://github.com/cmusphinx/cmudict"),
        notes: Some("CMU US English Pronouncing Dictionary."),
    },
    ModelAsset {
        id: "cmudict-vp",
        filename: "cmudict.vp",
        relative_path: "models/speaking/en-us/cmudict.vp",
        url: "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.vp",
        sha256: None,
        size_bytes: None,
        license: Some("BSD-3-Clause"),
        source: Some("https://github.com/cmusphinx/cmudict"),
        notes: Some("CMU US English Pronouncing Dictionary Verbal Pronunciations."),
    },
];

pub const NON_SPEECH_MODEL_ARCHIVES: &[ModelArchive] = &[];

pub const NON_SPEECH_MODEL_BUNDLES: &[ModelBundle] = &[
    ModelBundle {
        id: "gemma-4-e4b-it-q4-k-m",
        display_name: "Gemma 4 E4B IT Q4_K_M",
        kind: ModelKind::Llm,
        primary_asset_id: "gemma-4-e4b-it-q4-k-m",
        required_asset_ids: &["gemma-4-e4b-it-q4-k-m", "gemma-4-e4b-it-mmproj-bf16"],
        aliases: &["gemma4", "gemma-4", "gemma-4-e4b", "gemma"],
    },
    ModelBundle {
        id: "gemma-3-4b-it-q4-k-m",
        display_name: "Gemma 3 4B IT Q4_K_M",
        kind: ModelKind::Llm,
        primary_asset_id: "gemma-3-4b-it-q4-k-m",
        required_asset_ids: &["gemma-3-4b-it-q4-k-m"],
        aliases: &["gemma3", "gemma-3", "gemma-3-4b"],
    },
    ModelBundle {
        id: DEFAULT_FACE_MODEL_ID,
        display_name: "InsightFace Buffalo_L Face Stack",
        kind: ModelKind::Face,
        primary_asset_id: "face-scrfd-34g-gnkps",
        required_asset_ids: &[
            "face-scrfd-34g-gnkps",
            "face-buffalo-l-w600k-r50",
            "face-buffalo-l-genderage",
        ],
        aliases: &["face", "faces", "insightface", "buffalo-l"],
    },
    ModelBundle {
        id: DEFAULT_ASR_MODEL_ID,
        display_name: "Whisper Large v3 Turbo Multilingual",
        kind: ModelKind::Asr,
        primary_asset_id: "whisper-large-v3-turbo",
        required_asset_ids: &["whisper-large-v3-turbo"],
        aliases: &[
            "asr",
            "whisper",
            "whisper-turbo",
            "large-v3-turbo",
            "whisper-large-v3-turbo",
        ],
    },
    ModelBundle {
        id: "whisper-base-en",
        display_name: "Whisper Base English",
        kind: ModelKind::Asr,
        primary_asset_id: "whisper-base-en",
        required_asset_ids: &["whisper-base-en"],
        aliases: &["whisper-base", "base-en", "whisper-base-en"],
    },
    ModelBundle {
        id: "phonemicizer-en-us",
        display_name: "Built-in en-US Phonemicizer",
        kind: ModelKind::Phonemicizer,
        primary_asset_id: "phonemicizer-en-us-builtin",
        required_asset_ids: &["phonemicizer-en-us-builtin"],
        aliases: &["phonemicizer", "g2p", "en-us-phonemicizer"],
    },
    ModelBundle {
        id: "lexicon-en-us",
        display_name: "Built-in en-US Lexicon",
        kind: ModelKind::Lexicon,
        primary_asset_id: "lexicon-en-us-builtin",
        required_asset_ids: &["lexicon-en-us-builtin"],
        aliases: &["lexicon", "en-us-lexicon"],
    },
];

include!(concat!(env!("OUT_DIR"), "/speech-model-catalog.rs"));

pub static MODEL_ASSETS: std::sync::LazyLock<Vec<ModelAsset>> = std::sync::LazyLock::new(|| {
    NON_SPEECH_MODEL_ASSETS
        .iter()
        .chain(GENERATED_SPEECH_MODEL_ASSETS)
        .copied()
        .collect()
});
pub static MODEL_ARCHIVES: std::sync::LazyLock<Vec<ModelArchive>> =
    std::sync::LazyLock::new(|| {
        NON_SPEECH_MODEL_ARCHIVES
            .iter()
            .chain(GENERATED_SPEECH_MODEL_ARCHIVES)
            .copied()
            .collect()
    });
pub static MODEL_BUNDLES: std::sync::LazyLock<Vec<ModelBundle>> = std::sync::LazyLock::new(|| {
    NON_SPEECH_MODEL_BUNDLES
        .iter()
        .chain(GENERATED_SPEECH_MODEL_BUNDLES)
        .copied()
        .collect()
});

pub fn find_bundle(name: &str) -> Option<&'static ModelBundle> {
    let normalized = normalize_model_name(name);
    MODEL_BUNDLES.iter().find(|bundle| {
        normalize_model_name(bundle.id) == normalized
            || bundle
                .aliases
                .iter()
                .any(|alias| normalize_model_name(alias) == normalized)
    })
}

pub fn bundle_primary_asset(bundle: &ModelBundle) -> anyhow::Result<&'static ModelAsset> {
    find_asset(bundle.primary_asset_id)
        .ok_or_else(|| anyhow::anyhow!("bundle `{}` references unknown primary asset", bundle.id))
}

pub fn bundle_required_assets(bundle: &ModelBundle) -> anyhow::Result<Vec<&'static ModelAsset>> {
    bundle
        .required_asset_ids
        .iter()
        .map(|asset_id| {
            find_asset(asset_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "bundle `{}` references unknown asset `{asset_id}`",
                    bundle.id
                )
            })
        })
        .collect()
}

pub fn bundle_multimodal_projector_asset(
    bundle: &ModelBundle,
) -> anyhow::Result<Option<&'static ModelAsset>> {
    bundle
        .required_asset_ids
        .iter()
        .copied()
        .find(|asset_id| asset_id.contains("mmproj"))
        .map(|asset_id| {
            find_asset(asset_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "bundle `{}` references unknown multimodal projector `{asset_id}`",
                    bundle.id
                )
            })
        })
        .transpose()
}

pub fn find_asset(asset_id: &str) -> Option<&'static ModelAsset> {
    MODEL_ASSETS.iter().find(|asset| asset.id == asset_id)
}

pub fn find_archive(asset_id: &str) -> Option<&'static ModelArchive> {
    MODEL_ARCHIVES
        .iter()
        .find(|archive| archive.asset_id == asset_id)
}

pub fn bundle_entrypoint_relative_path(bundle: &ModelBundle) -> anyhow::Result<&'static str> {
    let primary = bundle_primary_asset(bundle)?;
    Ok(find_archive(primary.id)
        .map(|archive| archive.primary_member_path)
        .unwrap_or(primary.relative_path))
}

pub fn bundle_archive_members(bundle: &ModelBundle) -> anyhow::Result<Vec<&'static ArchiveMember>> {
    Ok(bundle_required_assets(bundle)?
        .into_iter()
        .filter_map(|asset| find_archive(asset.id))
        .flat_map(|archive| archive.members)
        .collect())
}

fn normalize_model_name(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_aliases_resolve() {
        assert_eq!(find_bundle("gemma4").unwrap().id, DEFAULT_LLM_MODEL_ID);
        assert_eq!(find_bundle("gemma-4").unwrap().id, DEFAULT_LLM_MODEL_ID);
    }

    #[test]
    fn gemma4_bundle_includes_multimodal_projector() {
        let bundle = find_bundle("gemma4").unwrap();
        assert_eq!(
            bundle_multimodal_projector_asset(bundle)
                .unwrap()
                .unwrap()
                .id,
            "gemma-4-e4b-it-mmproj-bf16"
        );
    }

    #[test]
    fn registry_lists_styletts2_and_speech_assets() {
        assert_eq!(
            find_bundle("styletts2-en-us").unwrap().kind,
            ModelKind::StyleTts2
        );
        assert_eq!(find_bundle("voice").unwrap().kind, ModelKind::VoiceModel);
        assert_eq!(
            find_bundle("coqui-default").unwrap().id,
            DEFAULT_VOICE_MODEL_ID
        );
        assert_eq!(
            find_bundle("phonemicizer-en-us").unwrap().kind,
            ModelKind::Phonemicizer
        );
        assert_eq!(
            find_bundle("lexicon-en-us").unwrap().kind,
            ModelKind::Lexicon
        );
        assert_eq!(
            find_bundle("speedy-speech").unwrap().kind,
            ModelKind::AcousticModel
        );
        assert_eq!(
            find_bundle("fastpitch").unwrap().kind,
            ModelKind::AcousticModel
        );
        assert_eq!(
            find_bundle("hifigan").unwrap().kind,
            ModelKind::NeuralVocoder
        );
        assert_eq!(
            find_bundle("vctk-vits").unwrap().kind,
            ModelKind::EndToEndSpeech
        );
    }

    #[test]
    fn generated_speech_projection_matches_authoritative_catalog() {
        let catalog = tongues_tts::ModelCatalog::from_json(
            tongues_tts::model_catalog::EMBEDDED_MODEL_CATALOG,
        )
        .expect("embedded speech catalog");
        let generated_ids = GENERATED_SPEECH_MODEL_BUNDLES
            .iter()
            .map(|bundle| bundle.id)
            .collect::<std::collections::BTreeSet<_>>();
        let catalog_ids = catalog
            .entries
            .iter()
            .map(|entry| entry.id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(generated_ids, catalog_ids);
        for entry in catalog.entries {
            let bundle = find_bundle(&entry.id).expect("generated bundle");
            assert_eq!(bundle.display_name, entry.display_name);
            assert_eq!(
                bundle_entrypoint_relative_path(bundle).expect("entrypoint"),
                entry.runtime_path
            );
            for alias in entry.aliases {
                assert_eq!(find_bundle(&alias).map(|bundle| bundle.id), Some(bundle.id));
            }
        }
    }

    #[test]
    fn coqui_archives_have_integrity_and_registered_entrypoints() {
        for (bundle_id, member_count) in [
            ("fastpitch-ljspeech", 2),
            ("speedy-speech-ljspeech", 2),
            ("hifigan-v2-ljspeech", 2),
            ("vits-vctk", 3),
        ] {
            let bundle = find_bundle(bundle_id).expect("bundle");
            let asset = bundle_primary_asset(bundle).expect("asset");
            let sha256 = asset.sha256.expect("pinned SHA-256");
            assert_eq!(sha256.len(), 64);
            assert!(sha256.bytes().all(|byte| byte.is_ascii_hexdigit()));
            assert!(bundle_entrypoint_relative_path(bundle)
                .expect("entrypoint")
                .ends_with("model_file.pth"));
            assert_eq!(bundle_archive_members(bundle).unwrap().len(), member_count);
        }
    }
}
