#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Llm,
    Face,
    Asr,
    StyleTts2,
    PiperVoice,
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

pub const DEFAULT_LLM_MODEL_ID: &str = "gemma-4-e4b-it-q4-k-m";
pub const DEFAULT_FACE_MODEL_ID: &str = "face-insightface-buffalo-l";
pub const DEFAULT_ASR_MODEL_ID: &str = "whisper-base-en";
pub const DEFAULT_STYLETTS2_MODEL_ID: &str = "styletts2-en-us";
pub const DEFAULT_PIPER_VOICE_MODEL_ID: &str = "piper-ryan-medium";

pub const MODEL_ASSETS: &[ModelAsset] = &[
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
        id: "phonemicizer-en-us-builtin",
        filename: "phonemicizer-en-us-builtin.json",
        relative_path: "models/speech/en-us/phonemicizer-en-us-builtin.json",
        url: "builtin://mortar-sea/en-us-phonemicizer",
        sha256: None,
        size_bytes: None,
        license: Some("MIT"),
        source: Some("speech::EnglishPhonemicizer"),
        notes: Some("Built-in CMUdict-style seed lexicon plus explicit unknown-word fallback."),
    },
    ModelAsset {
        id: "lexicon-en-us-builtin",
        filename: "lexicon-en-us-builtin.json",
        relative_path: "models/speech/en-us/lexicon-en-us-builtin.json",
        url: "builtin://mortar-sea/en-us-lexicon",
        sha256: None,
        size_bytes: None,
        license: Some("MIT"),
        source: Some("speech::EnglishPhonemicizer"),
        notes: Some("Small deterministic built-in lexicon used by default tests and smoke runs."),
    },
    ModelAsset {
        id: "cmudict-base",
        filename: "cmudict.dict",
        relative_path: "models/speech/en-us/cmudict.dict",
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
        relative_path: "models/speech/en-us/cmudict.vp",
        url: "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.vp",
        sha256: None,
        size_bytes: None,
        license: Some("BSD-3-Clause"),
        source: Some("https://github.com/cmusphinx/cmudict"),
        notes: Some("CMU US English Pronouncing Dictionary Verbal Pronunciations."),
    },
    ModelAsset {
        id: "styletts2-en-us-onnx-14b6dd",
        filename: "14b6dd78237d223f172f8af702ed8aeb4a2c51fd0ff7e3ca03a4967d33fa13bc.onnx",
        relative_path: "models/styletts2/en-us/14b6dd78237d223f172f8af702ed8aeb4a2c51fd0ff7e3ca03a4967d33fa13bc.onnx",
        url: "https://huggingface.co/hexgrad/styletts2/resolve/main/14b6dd78237d223f172f8af702ed8aeb4a2c51fd0ff7e3ca03a4967d33fa13bc.onnx",
        sha256: Some("14b6dd78237d223f172f8af702ed8aeb4a2c51fd0ff7e3ca03a4967d33fa13bc"),
        size_bytes: Some(102_000_000),
        license: Some("MIT"),
        source: Some("https://huggingface.co/hexgrad/styletts2"),
        notes: Some(
            "Public ONNX conversion of StyleTTS2-LibriTTS; native inference uses the token encoder and decoder path.",
        ),
    },
    ModelAsset {
        id: "styletts2-en-us-onnx-4612a9",
        filename: "4612a9dc0c0e142468f361e8e901bdccfdca45a2ae1145e5452bc98c7915302d.onnx",
        relative_path: "models/styletts2/en-us/4612a9dc0c0e142468f361e8e901bdccfdca45a2ae1145e5452bc98c7915302d.onnx",
        url: "https://huggingface.co/hexgrad/styletts2/resolve/main/4612a9dc0c0e142468f361e8e901bdccfdca45a2ae1145e5452bc98c7915302d.onnx",
        sha256: Some("4612a9dc0c0e142468f361e8e901bdccfdca45a2ae1145e5452bc98c7915302d"),
        size_bytes: Some(238_000_000),
        license: Some("MIT"),
        source: Some("https://huggingface.co/hexgrad/styletts2"),
        notes: Some(
            "Public ONNX conversion of StyleTTS2-LibriTTS; reserved for style/diffusion wiring.",
        ),
    },
    ModelAsset {
        id: "styletts2-en-us-onnx-91473d",
        filename: "91473db52725b0c3b8387537979a2f42f0da82836e50902503a877c610864ad6.onnx",
        relative_path: "models/styletts2/en-us/91473db52725b0c3b8387537979a2f42f0da82836e50902503a877c610864ad6.onnx",
        url: "https://huggingface.co/hexgrad/styletts2/resolve/main/91473db52725b0c3b8387537979a2f42f0da82836e50902503a877c610864ad6.onnx",
        sha256: Some("91473db52725b0c3b8387537979a2f42f0da82836e50902503a877c610864ad6"),
        size_bytes: Some(23_100_000),
        license: Some("MIT"),
        source: Some("https://huggingface.co/hexgrad/styletts2"),
        notes: Some("Public ONNX conversion of StyleTTS2-LibriTTS token encoder."),
    },
    ModelAsset {
        id: "styletts2-en-us-onnx-99e40b",
        filename: "99e40b35027e96a247c8e1f359d2f99d3cd6e93afec2e0f4a15f72dd7b79d457.onnx",
        relative_path: "models/styletts2/en-us/99e40b35027e96a247c8e1f359d2f99d3cd6e93afec2e0f4a15f72dd7b79d457.onnx",
        url: "https://huggingface.co/hexgrad/styletts2/resolve/main/99e40b35027e96a247c8e1f359d2f99d3cd6e93afec2e0f4a15f72dd7b79d457.onnx",
        sha256: Some("99e40b35027e96a247c8e1f359d2f99d3cd6e93afec2e0f4a15f72dd7b79d457"),
        size_bytes: Some(307_000_000),
        license: Some("MIT"),
        source: Some("https://huggingface.co/hexgrad/styletts2"),
        notes: Some("Public ONNX conversion of StyleTTS2-LibriTTS waveform decoder."),
    },
    ModelAsset {
        id: "styletts2-libritts-reference-audio",
        filename: "reference_audio.zip",
        relative_path: "models/styletts2/en-us/reference_audio.zip",
        url: "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/reference_audio.zip",
        sha256: Some("d25b4950ec39cec5a00f5061491ad0b3606edc6618a54adc59663bfd6e6ab55e"),
        size_bytes: Some(2_918_087),
        license: Some("CC-BY-4.0"),
        source: Some("https://huggingface.co/yl4579/StyleTTS2-LibriTTS"),
        notes: Some(
            "Short LibriTTS-derived StyleTTS2 reference WAVs used for default voice and intonation references.",
        ),
    },
    ModelAsset {
        id: "piper-ryan-medium-onnx",
        filename: "en_US-ryan-medium.onnx",
        relative_path: "models/piper/en_US-ryan-medium.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        sha256: None,
        size_bytes: None,
        license: Some("CC-BY-4.0"),
        source: Some("https://huggingface.co/rhasspy/piper-voices"),
        notes: Some("Piper voice ONNX model; Mortar runs it directly without the Piper binary."),
    },
    ModelAsset {
        id: "piper-ryan-medium-config",
        filename: "en_US-ryan-medium.onnx.json",
        relative_path: "models/piper/en_US-ryan-medium.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
        sha256: None,
        size_bytes: None,
        license: Some("CC-BY-4.0"),
        source: Some("https://huggingface.co/rhasspy/piper-voices"),
        notes: Some("Piper voice phoneme map and inference defaults."),
    },
    ModelAsset {
        id: "piper-amy-medium-onnx",
        filename: "en_US-amy-medium.onnx",
        relative_path: "models/piper/en_US-amy-medium.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
        sha256: None,
        size_bytes: None,
        license: Some("CC-BY-4.0"),
        source: Some("https://huggingface.co/rhasspy/piper-voices"),
        notes: Some("Piper voice ONNX model; Mortar runs it directly without the Piper binary."),
    },
    ModelAsset {
        id: "piper-amy-medium-config",
        filename: "en_US-amy-medium.onnx.json",
        relative_path: "models/piper/en_US-amy-medium.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
        sha256: None,
        size_bytes: None,
        license: Some("CC-BY-4.0"),
        source: Some("https://huggingface.co/rhasspy/piper-voices"),
        notes: Some("Piper voice phoneme map and inference defaults."),
    },
    ModelAsset {
        id: "piper-ljspeech-high-onnx",
        filename: "en_US-ljspeech-high.onnx",
        relative_path: "models/piper/en_US-ljspeech-high.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx",
        sha256: None,
        size_bytes: None,
        license: Some("CC0-1.0"),
        source: Some("https://huggingface.co/rhasspy/piper-voices"),
        notes: Some("Piper voice ONNX model; Mortar runs it directly without the Piper binary."),
    },
    ModelAsset {
        id: "piper-ljspeech-high-config",
        filename: "en_US-ljspeech-high.onnx.json",
        relative_path: "models/piper/en_US-ljspeech-high.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx.json",
        sha256: None,
        size_bytes: None,
        license: Some("CC0-1.0"),
        source: Some("https://huggingface.co/rhasspy/piper-voices"),
        notes: Some("Piper voice phoneme map and inference defaults."),
    },
];

pub const MODEL_BUNDLES: &[ModelBundle] = &[
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
        display_name: "Whisper Base English",
        kind: ModelKind::Asr,
        primary_asset_id: "whisper-base-en",
        required_asset_ids: &["whisper-base-en"],
        aliases: &["asr", "whisper", "whisper-base", "base-en"],
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
    ModelBundle {
        id: DEFAULT_STYLETTS2_MODEL_ID,
        display_name: "StyleTTS2 en-US ONNX",
        kind: ModelKind::StyleTts2,
        primary_asset_id: "styletts2-en-us-onnx-14b6dd",
        required_asset_ids: &[
            "phonemicizer-en-us-builtin",
            "lexicon-en-us-builtin",
            "styletts2-en-us-onnx-14b6dd",
            "styletts2-en-us-onnx-4612a9",
            "styletts2-en-us-onnx-91473d",
            "styletts2-en-us-onnx-99e40b",
            "styletts2-libritts-reference-audio",
            "cmudict-base",
            "cmudict-vp",
        ],
        aliases: &["styletts2", "styletts2-en", "tts", "speech"],
    },
    ModelBundle {
        id: DEFAULT_PIPER_VOICE_MODEL_ID,
        display_name: "Piper Ryan Medium",
        kind: ModelKind::PiperVoice,
        primary_asset_id: "piper-ryan-medium-onnx",
        required_asset_ids: &["piper-ryan-medium-onnx", "piper-ryan-medium-config"],
        aliases: &["ryan", "piper", "piper-ryan", "voice"],
    },
    ModelBundle {
        id: "piper-amy-medium",
        display_name: "Piper Amy Medium",
        kind: ModelKind::PiperVoice,
        primary_asset_id: "piper-amy-medium-onnx",
        required_asset_ids: &["piper-amy-medium-onnx", "piper-amy-medium-config"],
        aliases: &["amy", "piper-amy"],
    },
    ModelBundle {
        id: "piper-ljspeech-high",
        display_name: "Piper LJSpeech High",
        kind: ModelKind::PiperVoice,
        primary_asset_id: "piper-ljspeech-high-onnx",
        required_asset_ids: &["piper-ljspeech-high-onnx", "piper-ljspeech-high-config"],
        aliases: &["ljspeech", "lj", "piper-ljspeech"],
    },
];

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
        assert_eq!(find_bundle("piper").unwrap().kind, ModelKind::PiperVoice);
        assert_eq!(
            find_bundle("phonemicizer-en-us").unwrap().kind,
            ModelKind::Phonemicizer
        );
        assert_eq!(
            find_bundle("lexicon-en-us").unwrap().kind,
            ModelKind::Lexicon
        );
    }
}
