use std::path::{Path, PathBuf};

use clap::Args;
use speaking::{
    AsrLanguageCapability, LanguageId, LanguageIdentifier, LanguageIdentifierCapability,
    LanguageRoutingCapabilities, WhisperLanguageIdentifier,
};

fn language(id: &str) -> LanguageId {
    LanguageId(id.into())
}

pub fn capabilities() -> LanguageRoutingCapabilities {
    let whisper_model = crate::models::asr_whisper_model_path().ok();
    let whisper_installed = whisper_model
        .as_ref()
        .and_then(|path| path.metadata().ok())
        .is_some_and(|metadata| metadata.is_file() && metadata.len() > 0);
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

#[derive(Debug, Args)]
pub struct LanguageRoutingCommand {
    /// Run the installed Whisper detector on a WAV file.
    #[arg(long)]
    detect_wav: Option<PathBuf>,
}

pub fn run(command: LanguageRoutingCommand) -> anyhow::Result<()> {
    if let Some(path) = command.detect_wav {
        let model = crate::models::asr_whisper_model_path()?;
        let audio = tongues_audio::read_wav(&path)?
            .convert_channels(1)?
            .resample_linear(16_000)?;
        let mut detector = WhisperLanguageIdentifier::new_quiet(model)?;
        let detection = detector.detect(
            &path.display().to_string(),
            0,
            &speaking::AudioFrame {
                sample_rate_hz: audio.sample_rate_hz,
                channels: audio.channels,
                samples: audio.samples,
            },
        )?;
        println!("{}", serde_json::to_string_pretty(&detection)?);
        return Ok(());
    }
    println!("{}", serde_json::to_string_pretty(&capabilities())?);
    Ok(())
}
