use std::path::Path;

use speaking::{
    AsrLanguageCapability, LanguageId, LanguageIdentifierCapability, LanguageRoutingCapabilities,
};

fn language(id: &str) -> LanguageId {
    LanguageId(id.into())
}

pub fn capabilities() -> LanguageRoutingCapabilities {
    let whisper_model = std::env::var_os("TONGUES_WHISPER_MODEL");
    let whisper_installed = whisper_model
        .as_deref()
        .is_some_and(|path| Path::new(path).exists());
    let whisper_languages = ["en", "es", "fr", "de", "it", "pt", "ja", "zh"]
        .into_iter()
        .map(language)
        .collect::<Vec<_>>();
    LanguageRoutingCapabilities::new(
        vec![LanguageIdentifierCapability {
            detector_id: "whisper-language-id".into(),
            model_id: whisper_model
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_else(|| "TONGUES_WHISPER_MODEL".into()),
            installed: whisper_installed,
            languages: whisper_languages.clone(),
        }],
        vec![
            AsrLanguageCapability {
                provider_id: "whisper".into(),
                model_id: whisper_model
                    .as_ref()
                    .map(|path| path.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "TONGUES_WHISPER_MODEL".into()),
                installed: whisper_installed,
                languages: whisper_languages,
            },
            AsrLanguageCapability {
                provider_id: "common-phone".into(),
                model_id: "models/common-phone/v0/model-latest.bin".into(),
                installed: Path::new("models/common-phone/v0/model-latest.bin").exists(),
                languages: vec![language("en")],
            },
            AsrLanguageCapability {
                provider_id: "interpretation".into(),
                model_id: "models/interpretation/mini-v0/model-latest.bin".into(),
                installed: Path::new("models/interpretation/mini-v0/model-latest.bin").exists(),
                languages: vec![language("en")],
            },
        ],
    )
}

pub fn run() -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(&capabilities())?);
    Ok(())
}
