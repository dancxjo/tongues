use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::models::manifest::{
    bundle_archive_members, bundle_multimodal_projector_asset, bundle_primary_asset,
    bundle_required_assets, find_bundle, ModelAsset, ModelBundle, ModelKind, DEFAULT_ASR_MODEL_ID,
    DEFAULT_LLM_MODEL_ID, DEFAULT_VOICE_MODEL_ID,
};

#[derive(Debug, Serialize, Deserialize, Default)]
struct ModelSelection {
    llm: Option<String>,
    voice_model: Option<String>,
}

pub fn asr_whisper_model_path() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("TONGUES_WHISPER_MODEL")
        .or_else(|| std::env::var_os("MORTAR_ASR_WHISPER_MODEL"))
    {
        return Ok(PathBuf::from(path));
    }
    let bundle =
        find_bundle(DEFAULT_ASR_MODEL_ID).context("default ASR model bundle is not registered")?;
    let asset = bundle_primary_asset(bundle)?;
    Ok(asset_path(&resolve_mortar_home()?, asset))
}

pub fn selected_llm_model_path() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("MORTAR_LLM_MODEL") {
        return Ok(PathBuf::from(path));
    }
    let bundle = selected_bundle()?;
    let asset = bundle_primary_asset(bundle)?;
    Ok(asset_path(&resolve_mortar_home()?, asset))
}

pub fn selected_llm_projector_path() -> Result<Option<PathBuf>> {
    if let Some(path) = std::env::var_os("MORTAR_LLM_MMPROJ") {
        return Ok(Some(PathBuf::from(path)));
    }
    if std::env::var_os("MORTAR_LLM_MODEL").is_some() {
        return Ok(None);
    }

    let bundle = selected_bundle()?;
    let Some(asset) = bundle_multimodal_projector_asset(bundle)? else {
        return Ok(None);
    };
    Ok(Some(asset_path(&resolve_mortar_home()?, asset)))
}

pub fn selected_llm_model_label() -> Result<&'static str> {
    Ok(selected_bundle()?.display_name)
}

pub fn selected_bundle() -> Result<&'static ModelBundle> {
    selected_bundle_for_kind(ModelKind::Llm)
}

pub fn selected_voice_model_bundle() -> Result<&'static ModelBundle> {
    selected_bundle_for_kind(ModelKind::VoiceModel)
}

pub fn selected_bundle_for_kind(kind: ModelKind) -> Result<&'static ModelBundle> {
    let selection = read_selection()?;
    let selected = match kind {
        ModelKind::Llm => selection.llm.as_deref().unwrap_or(DEFAULT_LLM_MODEL_ID),
        ModelKind::VoiceModel => selection
            .voice_model
            .as_deref()
            .unwrap_or(DEFAULT_VOICE_MODEL_ID),
        _ => default_bundle_id_for_kind(kind)?,
    };
    let bundle = find_bundle(selected)
        .with_context(|| format!("selected model `{selected}` is not registered"))?;
    if bundle.kind != kind {
        bail!(
            "selected model `{selected}` is not a {} bundle",
            model_kind_name(kind)
        );
    }
    Ok(bundle)
}

pub fn write_selected_model(model_id: &str) -> Result<()> {
    write_selected_model_for_kind(ModelKind::Llm, model_id)
}

pub fn write_selected_model_for_kind(kind: ModelKind, model_id: &str) -> Result<()> {
    let bundle = find_bundle(model_id)
        .with_context(|| format!("selected model `{model_id}` is not registered"))?;
    if bundle.kind != kind {
        bail!(
            "selected model `{model_id}` is not a {} bundle",
            model_kind_name(kind)
        );
    }

    let path = model_selection_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut selection = read_selection()?;
    match kind {
        ModelKind::Llm => selection.llm = Some(model_id.to_string()),
        ModelKind::VoiceModel => selection.voice_model = Some(model_id.to_string()),
        _ => bail!("{} selections are not stored yet", model_kind_name(kind)),
    }
    fs::write(&path, serde_json::to_vec_pretty(&selection)?)?;
    Ok(())
}

pub fn resolve_mortar_home() -> Result<PathBuf> {
    if let Some(home) = std::env::var_os("MORTAR_SEA_HOME") {
        let home = PathBuf::from(home);
        if home.as_os_str().is_empty() {
            bail!("MORTAR_SEA_HOME is set but empty");
        }
        ensure_not_vendor_home(&home)?;
        return Ok(home);
    }

    let base = dirs::data_local_dir().context("failed to resolve local data directory")?;
    let home = base.join("mortar-sea");
    ensure_not_vendor_home(&home)?;
    Ok(home)
}

pub fn model_selection_path() -> Result<PathBuf> {
    Ok(resolve_mortar_home()?.join("model-selection.json"))
}

pub fn asset_path(home: &Path, asset: &ModelAsset) -> PathBuf {
    home.join(asset.relative_path)
}

pub fn bundle_present(bundle: &ModelBundle) -> Result<bool> {
    let home = resolve_mortar_home()?;
    let assets_present = bundle_required_assets(bundle)?
        .iter()
        .all(|asset| is_non_empty_file(&asset_path(&home, asset)));
    let members_present = bundle_archive_members(bundle)?
        .iter()
        .all(|member| is_non_empty_file(&home.join(member.relative_path)));
    Ok(assets_present && members_present)
}

pub fn is_non_empty_file(path: &Path) -> bool {
    path.metadata()
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
}

fn default_bundle_id_for_kind(kind: ModelKind) -> Result<&'static str> {
    match kind {
        ModelKind::Llm => Ok(DEFAULT_LLM_MODEL_ID),
        ModelKind::VoiceModel => Ok(DEFAULT_VOICE_MODEL_ID),
        _ => missing_default_for_kind(kind),
    }
}

fn missing_default_for_kind(kind: ModelKind) -> Result<&'static str> {
    bail!("{} selections are not stored yet", model_kind_name(kind))
}

fn model_kind_name(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Llm => "LLM",
        ModelKind::Face => "face",
        ModelKind::Asr => "ASR",
        ModelKind::StyleTts2 => "StyleTTS2",
        ModelKind::VoiceModel => "Voice model",
        ModelKind::AcousticModel => "acoustic model",
        ModelKind::NeuralVocoder => "neural vocoder",
        ModelKind::EndToEndSpeech => "end-to-end speech model",
        ModelKind::VoiceConversion => "voice-conversion model",
        ModelKind::Lexicon => "lexicon",
        ModelKind::Phonemicizer => "phonemicizer",
    }
}

fn ensure_not_vendor_home(path: &Path) -> Result<()> {
    if path_has_vendor_component(path) {
        bail!(
            "refusing to use {} as a model home because Tongues does not create or manage .vendor directories",
            path.display()
        );
    }
    Ok(())
}

fn path_has_vendor_component(path: &Path) -> bool {
    path.components()
        .any(|component| matches!(component, Component::Normal(name) if name == ".vendor"))
}

fn read_selection() -> Result<ModelSelection> {
    let path = model_selection_path()?;
    if !path.exists() {
        return Ok(ModelSelection::default());
    }
    let bytes = fs::read(&path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_paths_are_rejected_as_model_home() {
        assert!(ensure_not_vendor_home(Path::new(".vendor")).is_err());
        assert!(ensure_not_vendor_home(Path::new("/tmp/tongues/.vendor/models")).is_err());
        assert!(ensure_not_vendor_home(Path::new("/tmp/tongues/vendor/models")).is_ok());
        assert!(ensure_not_vendor_home(Path::new("/tmp/tongues/models")).is_ok());
    }
}
