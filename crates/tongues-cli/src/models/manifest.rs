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
pub const DEFAULT_ACOUSTIC_MODEL_ID: &str = "coqui-speedy-speech-ljspeech";
pub const DEFAULT_NEURAL_VOCODER_ID: &str = "coqui-hifigan-v2-ljspeech";

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
        id: "voice-ryan-medium-onnx",
        filename: "en_US-ryan-medium.onnx",
        relative_path: "models/voices/en_US-ryan-medium.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        sha256: None,
        size_bytes: None,
        license: Some("CC-BY-4.0"),
        source: Some("https://huggingface.co/rhasspy/piper-voices"),
        notes: Some("ONNX voice model; Mortar runs it directly without the source runtime."),
    },
    ModelAsset {
        id: "voice-ryan-medium-config",
        filename: "en_US-ryan-medium.onnx.json",
        relative_path: "models/voices/en_US-ryan-medium.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
        sha256: None,
        size_bytes: None,
        license: Some("CC-BY-4.0"),
        source: Some("https://huggingface.co/rhasspy/piper-voices"),
        notes: Some("Voice model phoneme map and inference defaults."),
    },
    ModelAsset {
        id: "voice-amy-medium-onnx",
        filename: "en_US-amy-medium.onnx",
        relative_path: "models/voices/en_US-amy-medium.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
        sha256: None,
        size_bytes: None,
        license: Some("CC-BY-4.0"),
        source: Some("https://huggingface.co/rhasspy/piper-voices"),
        notes: Some("ONNX voice model; Mortar runs it directly without the source runtime."),
    },
    ModelAsset {
        id: "voice-amy-medium-config",
        filename: "en_US-amy-medium.onnx.json",
        relative_path: "models/voices/en_US-amy-medium.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
        sha256: None,
        size_bytes: None,
        license: Some("CC-BY-4.0"),
        source: Some("https://huggingface.co/rhasspy/piper-voices"),
        notes: Some("Voice model phoneme map and inference defaults."),
    },
    ModelAsset {
        id: "voice-ljspeech-high-onnx",
        filename: "en_US-ljspeech-high.onnx",
        relative_path: "models/voices/en_US-ljspeech-high.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx",
        sha256: None,
        size_bytes: None,
        license: Some("CC0-1.0"),
        source: Some("https://huggingface.co/rhasspy/piper-voices"),
        notes: Some("Closest runnable ONNX voice to Coqui's default LJSpeech English model; Mortar runs it directly without the source runtime."),
    },
    ModelAsset {
        id: "voice-ljspeech-high-config",
        filename: "en_US-ljspeech-high.onnx.json",
        relative_path: "models/voices/en_US-ljspeech-high.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx.json",
        sha256: None,
        size_bytes: None,
        license: Some("CC0-1.0"),
        source: Some("https://huggingface.co/rhasspy/piper-voices"),
        notes: Some("Voice model phoneme map and inference defaults."),
    },
    ModelAsset {
        id: "coqui-speedy-speech-ljspeech-release",
        filename: "tts_models--en--ljspeech--speedy-speech.zip",
        relative_path:
            "models/speech/coqui/en/ljspeech/speedy-speech/tts_models--en--ljspeech--speedy-speech.zip",
        url: "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--speedy-speech.zip",
        sha256: Some("ae772bc84c6b4d8fe97234f4d2a1282b925bf325659f319f03610edc4eb8023a"),
        size_bytes: Some(53_437_190),
        license: Some("Apache-2.0"),
        source: Some("https://github.com/coqui-ai/TTS/releases/tag/v0.6.1_models"),
        notes: Some("Published Coqui SpeedySpeech acoustic-model release archive."),
    },
    ModelAsset {
        id: "coqui-hifigan-v2-ljspeech-release",
        filename: "vocoder_models--en--ljspeech--hifigan_v2.zip",
        relative_path:
            "models/speech/coqui/en/ljspeech/hifigan-v2/vocoder_models--en--ljspeech--hifigan_v2.zip",
        url: "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--en--ljspeech--hifigan_v2.zip",
        sha256: Some("4378dc0afb12ae3f50c1614cb143c4084d066b28fde35b58f45fc6b56b1c75f3"),
        size_bytes: Some(3_802_006),
        license: Some("Apache-2.0"),
        source: Some("https://github.com/coqui-ai/TTS/releases/tag/v0.6.1_models"),
        notes: Some("Published Coqui HiFi-GAN v2 neural-vocoder release archive."),
    },
    ModelAsset {
        id: "coqui-vits-vctk-release",
        filename: "tts_models--en--vctk--vits.zip",
        relative_path: "models/speech/coqui/en/vctk/vits/tts_models--en--vctk--vits.zip",
        url: "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--vctk--vits.zip",
        sha256: Some("ad753e5200614907627495b6177a08d755df1dbbde30eb6e31264caa2a3f3eaa"),
        size_bytes: Some(147_691_678),
        license: Some("Apache-2.0"),
        source: Some("https://github.com/coqui-ai/TTS/releases/tag/v0.6.1_models"),
        notes: Some("Published Coqui VCTK VITS release with 109 learned speaker embeddings."),
    },
];

pub const MODEL_ARCHIVES: &[ModelArchive] = &[
    ModelArchive {
        asset_id: "coqui-speedy-speech-ljspeech-release",
        primary_member_path: "models/speech/coqui/en/ljspeech/speedy-speech/model_file.pth",
        members: &[
            ArchiveMember {
                member_path: "tts_models--en--ljspeech--speedy-speech/model_file.pth",
                relative_path: "models/speech/coqui/en/ljspeech/speedy-speech/model_file.pth",
            },
            ArchiveMember {
                member_path: "tts_models--en--ljspeech--speedy-speech/config.json",
                relative_path: "models/speech/coqui/en/ljspeech/speedy-speech/config.json",
            },
        ],
    },
    ModelArchive {
        asset_id: "coqui-hifigan-v2-ljspeech-release",
        primary_member_path: "models/speech/coqui/en/ljspeech/hifigan-v2/model_file.pth",
        members: &[
            ArchiveMember {
                member_path: "vocoder_models--en--ljspeech--hifigan_v2/model_file.pth",
                relative_path: "models/speech/coqui/en/ljspeech/hifigan-v2/model_file.pth",
            },
            ArchiveMember {
                member_path: "vocoder_models--en--ljspeech--hifigan_v2/config.json",
                relative_path: "models/speech/coqui/en/ljspeech/hifigan-v2/config.json",
            },
        ],
    },
    ModelArchive {
        asset_id: "coqui-vits-vctk-release",
        primary_member_path: "models/speech/coqui/en/vctk/vits/model_file.pth",
        members: &[
            ArchiveMember {
                member_path: "tts_models--en--vctk--vits/model_file.pth",
                relative_path: "models/speech/coqui/en/vctk/vits/model_file.pth",
            },
            ArchiveMember {
                member_path: "tts_models--en--vctk--vits/config.json",
                relative_path: "models/speech/coqui/en/vctk/vits/config.json",
            },
            ArchiveMember {
                member_path: "tts_models--en--vctk--vits/speaker_ids.json",
                relative_path: "models/speech/coqui/en/vctk/vits/speaker_ids.json",
            },
        ],
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
        id: "voice-ryan-medium",
        display_name: "Ryan Medium Voice",
        kind: ModelKind::VoiceModel,
        primary_asset_id: "voice-ryan-medium-onnx",
        required_asset_ids: &["voice-ryan-medium-onnx", "voice-ryan-medium-config"],
        aliases: &["ryan", "voice-ryan"],
    },
    ModelBundle {
        id: "voice-amy-medium",
        display_name: "Amy Medium Voice",
        kind: ModelKind::VoiceModel,
        primary_asset_id: "voice-amy-medium-onnx",
        required_asset_ids: &["voice-amy-medium-onnx", "voice-amy-medium-config"],
        aliases: &["amy", "voice-amy"],
    },
    ModelBundle {
        id: DEFAULT_VOICE_MODEL_ID,
        display_name: "LJSpeech High Voice",
        kind: ModelKind::VoiceModel,
        primary_asset_id: "voice-ljspeech-high-onnx",
        required_asset_ids: &["voice-ljspeech-high-onnx", "voice-ljspeech-high-config"],
        aliases: &["ljspeech", "lj", "voice", "voice-ljspeech", "coqui-default"],
    },
    ModelBundle {
        id: DEFAULT_ACOUSTIC_MODEL_ID,
        display_name: "Coqui SpeedySpeech LJSpeech",
        kind: ModelKind::AcousticModel,
        primary_asset_id: "coqui-speedy-speech-ljspeech-release",
        required_asset_ids: &["coqui-speedy-speech-ljspeech-release"],
        aliases: &["speedy-speech", "speedyspeech", "coqui-speedy-speech"],
    },
    ModelBundle {
        id: DEFAULT_NEURAL_VOCODER_ID,
        display_name: "Coqui HiFi-GAN v2 LJSpeech",
        kind: ModelKind::NeuralVocoder,
        primary_asset_id: "coqui-hifigan-v2-ljspeech-release",
        required_asset_ids: &["coqui-hifigan-v2-ljspeech-release"],
        aliases: &["hifigan-v2", "hifigan", "coqui-hifigan"],
    },
    ModelBundle {
        id: "coqui-vits-vctk",
        display_name: "Coqui VITS VCTK",
        kind: ModelKind::EndToEndSpeech,
        primary_asset_id: "coqui-vits-vctk-release",
        required_asset_ids: &["coqui-vits-vctk-release"],
        aliases: &["vctk-vits", "vits-vctk", "coqui-vctk"],
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
            find_bundle("hifigan").unwrap().kind,
            ModelKind::NeuralVocoder
        );
        assert_eq!(
            find_bundle("vctk-vits").unwrap().kind,
            ModelKind::EndToEndSpeech
        );
    }

    #[test]
    fn coqui_archives_have_integrity_and_registered_entrypoints() {
        for (bundle_id, member_count) in [
            ("coqui-speedy-speech-ljspeech", 2),
            ("coqui-hifigan-v2-ljspeech", 2),
            ("coqui-vits-vctk", 3),
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
