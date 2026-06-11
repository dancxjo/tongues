use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use owo_colors::OwoColorize;
use sha2::{Digest, Sha256};

use crate::models::manifest::{
    bundle_primary_asset, bundle_required_assets, find_asset, find_bundle, ModelAsset, ModelBundle,
    ModelKind, DEFAULT_ASR_MODEL_ID, DEFAULT_FACE_MODEL_ID, DEFAULT_PIPER_VOICE_MODEL_ID,
    DEFAULT_STYLETTS2_MODEL_ID,
};
use crate::models::selection::{
    asset_path, is_non_empty_file, resolve_mortar_home, selected_bundle, selected_llm_model_path,
    selected_llm_projector_path, selected_piper_voice_bundle, write_selected_model,
};

#[derive(Debug, Clone)]
pub struct FaceModelPaths {
    pub detector: PathBuf,
    pub recognizer: PathBuf,
    pub attributes: PathBuf,
}

#[derive(Debug, Clone)]
pub struct RuntimeModelPaths {
    pub llm: PathBuf,
    pub llm_projector: Option<PathBuf>,
    pub face: FaceModelPaths,
    pub piper_voice: PathBuf,
}

#[derive(Debug, Clone)]
pub struct StyleTts2ReferenceAudioPaths {
    pub voice: PathBuf,
    pub style: PathBuf,
}

const DEFAULT_STYLETTS2_VOICE_REFERENCE: &str = "reference_audio/1221-135767-0014.wav";
const DEFAULT_STYLETTS2_STYLE_REFERENCE: &str = "reference_audio/amused.wav";

pub fn ensure_selected_llm_available() -> Result<PathBuf> {
    let path = selected_llm_model_path()?;
    if std::env::var_os("MORTAR_LLM_MODEL").is_some() {
        if is_non_empty_file(&path) {
            return Ok(path);
        }
        anyhow::bail!(
            "MORTAR_LLM_MODEL points to a missing or empty file: {}",
            path.display()
        );
    }

    let bundle = selected_bundle()?;
    ensure_bundle_available(bundle)?;
    selected_llm_model_path()
}

pub fn ensure_selected_llm_projector_available() -> Result<Option<PathBuf>> {
    let Some(path) = selected_llm_projector_path()? else {
        return Ok(None);
    };

    if std::env::var_os("MORTAR_LLM_MMPROJ").is_some() {
        if is_non_empty_file(&path) {
            return Ok(Some(path));
        }
        anyhow::bail!(
            "MORTAR_LLM_MMPROJ points to a missing or empty file: {}",
            path.display()
        );
    }

    Ok(Some(path))
}

pub fn ensure_face_models_available() -> Result<FaceModelPaths> {
    let bundle = find_bundle(DEFAULT_FACE_MODEL_ID)
        .context("default face model bundle is not registered")?;
    ensure_bundle_available(bundle)?;
    face_model_paths()
}

pub fn ensure_asr_whisper_model_available() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("MORTAR_ASR_WHISPER_MODEL") {
        let path = PathBuf::from(path);
        if is_non_empty_file(&path) {
            return Ok(path);
        }
        anyhow::bail!(
            "MORTAR_ASR_WHISPER_MODEL points to a missing or empty file: {}",
            path.display()
        );
    }

    let bundle =
        find_bundle(DEFAULT_ASR_MODEL_ID).context("default ASR model bundle is not registered")?;
    ensure_bundle_available(bundle)?;
    let primary = bundle_primary_asset(bundle)?;
    Ok(asset_path(&resolve_mortar_home()?, primary))
}

pub fn ensure_styletts2_model_available() -> Result<PathBuf> {
    let path = ensure_model_available(DEFAULT_STYLETTS2_MODEL_ID)?;
    ensure_styletts2_reference_audio_extracted()?;
    Ok(path)
}

pub fn ensure_styletts2_default_reference_audio_available() -> Result<StyleTts2ReferenceAudioPaths>
{
    ensure_styletts2_reference_audio_extracted()?;
    styletts2_default_reference_audio_paths()
}

pub fn styletts2_default_reference_audio_paths() -> Result<StyleTts2ReferenceAudioPaths> {
    let home = resolve_mortar_home()?;
    let archive = find_asset("styletts2-libritts-reference-audio")
        .context("StyleTTS2 reference audio asset is not registered")?;
    let reference_dir = asset_path(&home, archive)
        .parent()
        .context("StyleTTS2 reference audio archive path has no parent")?
        .to_path_buf();
    Ok(StyleTts2ReferenceAudioPaths {
        voice: reference_dir.join(DEFAULT_STYLETTS2_VOICE_REFERENCE),
        style: reference_dir.join(DEFAULT_STYLETTS2_STYLE_REFERENCE),
    })
}

pub fn ensure_piper_voice_model_available() -> Result<PathBuf> {
    let bundle = selected_piper_voice_bundle()?;
    ensure_bundle_available(bundle)?;
    let primary = bundle_primary_asset(bundle)?;
    Ok(asset_path(&resolve_mortar_home()?, primary))
}

pub fn ensure_model_available(model: &str) -> Result<PathBuf> {
    let bundle = find_bundle(model).with_context(|| format!("unknown model `{model}`"))?;
    ensure_bundle_available(bundle)?;
    let primary = bundle_primary_asset(bundle)?;
    Ok(asset_path(&resolve_mortar_home()?, primary))
}

pub fn ensure_runtime_models_available() -> Result<RuntimeModelPaths> {
    Ok(RuntimeModelPaths {
        llm: ensure_selected_llm_available()?,
        llm_projector: ensure_selected_llm_projector_available()?,
        face: ensure_face_models_available()?,
        piper_voice: ensure_piper_voice_model_available()?,
    })
}

pub fn missing_model_asset_paths(model: &str) -> Result<Vec<PathBuf>> {
    let bundle = find_bundle(model).with_context(|| format!("unknown model `{model}`"))?;
    let home = resolve_mortar_home()?;
    Ok(bundle_required_assets(bundle)?
        .into_iter()
        .map(|asset| asset_path(&home, asset))
        .filter(|path| !is_non_empty_file(path))
        .collect())
}

pub fn fetch_model(model: Option<&str>, force: bool) -> Result<PathBuf> {
    if let Some(model) = model {
        let bundle = find_bundle(model).with_context(|| format!("unknown model `{model}`"))?;
        if bundle.kind == ModelKind::Llm {
            write_selected_model(bundle.id)?;
        }
        fetch_bundle(bundle, force)?;
        if bundle.kind == ModelKind::StyleTts2 {
            ensure_styletts2_reference_audio_extracted()?;
        }
        if bundle.kind == ModelKind::Llm {
            println!("{} {}", "selected".green(), bundle.display_name.bold());
            return selected_llm_model_path();
        }

        let primary = bundle_primary_asset(bundle)?;
        return Ok(asset_path(&resolve_mortar_home()?, primary));
    }

    fetch_all_runtime_bundles(force)?;
    selected_llm_model_path()
}

fn fetch_all_runtime_bundles(force: bool) -> Result<()> {
    for bundle in default_runtime_bundles()? {
        fetch_bundle(bundle, force)?;
        if bundle.kind == ModelKind::StyleTts2 {
            ensure_styletts2_reference_audio_extracted()?;
        }
    }
    Ok(())
}

fn default_runtime_bundles() -> Result<Vec<&'static ModelBundle>> {
    Ok(vec![
        selected_bundle()?,
        find_bundle(DEFAULT_FACE_MODEL_ID)
            .context("default face model bundle is not registered")?,
        find_bundle(DEFAULT_ASR_MODEL_ID).context("default ASR model bundle is not registered")?,
        find_bundle(DEFAULT_STYLETTS2_MODEL_ID)
            .context("default StyleTTS2 model bundle is not registered")?,
        find_bundle(DEFAULT_PIPER_VOICE_MODEL_ID)
            .context("default Piper voice model bundle is not registered")?,
    ])
}

fn fetch_bundle(bundle: &ModelBundle, force: bool) -> Result<()> {
    if bundle.kind == ModelKind::Llm {
        write_selected_model(bundle.id)?;
    }
    for asset in bundle_required_assets(bundle)? {
        fetch_asset(asset, force)?;
    }
    Ok(())
}

fn ensure_bundle_available(bundle: &ModelBundle) -> Result<()> {
    let home = resolve_mortar_home()?;
    let assets = bundle_required_assets(bundle)?;
    let missing = assets
        .iter()
        .any(|asset| !is_non_empty_file(&asset_path(&home, asset)));

    if missing {
        eprintln!(
            "model bundle `{}` is missing locally; downloading it now. This can take a while...",
            bundle.display_name
        );
    }

    for asset in assets {
        ensure_asset_available(asset)?;
    }
    Ok(())
}

fn ensure_asset_available(asset: &ModelAsset) -> Result<()> {
    let home = resolve_mortar_home()?;
    let path = asset_path(&home, asset);
    if is_non_empty_file(&path) {
        println!("{} {}", "already present".green(), path.display());
        return Ok(());
    }

    fetch_asset(asset, false)
}

fn ensure_styletts2_reference_audio_extracted() -> Result<()> {
    let paths = styletts2_default_reference_audio_paths()?;
    if is_non_empty_file(&paths.voice) && is_non_empty_file(&paths.style) {
        return Ok(());
    }

    let home = resolve_mortar_home()?;
    let archive = find_asset("styletts2-libritts-reference-audio")
        .context("StyleTTS2 reference audio asset is not registered")?;
    let archive_path = asset_path(&home, archive);
    if !is_non_empty_file(&archive_path) {
        fetch_asset(archive, false)?;
    }
    extract_zip_asset(&archive_path)?;

    anyhow::ensure!(
        is_non_empty_file(&paths.voice) && is_non_empty_file(&paths.style),
        "StyleTTS2 reference audio archive did not contain default references"
    );
    Ok(())
}

fn extract_zip_asset(path: &Path) -> Result<()> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("failed to read ZIP archive {}", path.display()))?;
    let target_dir = path
        .parent()
        .context("ZIP archive path has no parent directory")?;

    for index in 0..archive.len() {
        let mut member = archive
            .by_index(index)
            .with_context(|| format!("failed to read ZIP member {index} in {}", path.display()))?;
        if member.is_dir() {
            continue;
        }
        let Some(enclosed_name) = member.enclosed_name() else {
            continue;
        };
        let output_path = target_dir.join(enclosed_name);
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut output = File::create(&output_path)
            .with_context(|| format!("failed to create {}", output_path.display()))?;
        std::io::copy(&mut member, &mut output)
            .with_context(|| format!("failed to extract {}", output_path.display()))?;
    }
    Ok(())
}

fn fetch_asset(asset: &ModelAsset, force: bool) -> Result<()> {
    let home = resolve_mortar_home()?;
    let path = asset_path(&home, asset);
    if asset.url.starts_with("builtin://") {
        return fetch_builtin_asset(asset, &path, force);
    }
    let metadata = remote_metadata(asset).unwrap_or_default();
    let expected_sha256 = asset.sha256.map(str::to_string);

    if is_non_empty_file(&path) && !force {
        verify_existing_asset(&path, expected_sha256.as_deref())?;
        println!("{} {}", "already present".green(), path.display());
        return Ok(());
    }

    fs::create_dir_all(path.parent().context("model path has no parent")?)?;
    let part_path = path.with_file_name(format!("{}.part", asset.filename));
    if force {
        let _ = fs::remove_file(&part_path);
    }

    let mut resume_from = file_len(&part_path).unwrap_or(0);
    let mut request = ureq::get(asset.url);
    if resume_from > 0 {
        request = request.header("Range", &format!("bytes={resume_from}-"));
    }

    println!(
        "{} {}",
        "fetching".cyan(),
        format!("{} -> {}", asset.url, path.display()).dimmed()
    );

    let response = request
        .call()
        .with_context(|| format!("failed to download {}", asset.url))?;
    if resume_from > 0 && response.status().as_u16() != 206 {
        resume_from = 0;
        let _ = fs::remove_file(&part_path);
    }

    let total = metadata.content_length.or_else(|| {
        response
            .body()
            .content_length()
            .map(|length| length + resume_from)
    });
    let mut body = response.into_body();
    let mut reader = body.as_reader();
    let mut file = OpenOptions::new()
        .create(true)
        .append(resume_from > 0)
        .write(true)
        .truncate(resume_from == 0)
        .open(&part_path)
        .with_context(|| format!("failed to open {}", part_path.display()))?;
    let mut buffer = [0_u8; 128 * 1024];
    let mut downloaded = resume_from;

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])?;
        downloaded += read as u64;
        print_progress(downloaded, total);
    }
    println!();
    file.flush()?;
    drop(file);
    fs::rename(&part_path, &path).with_context(|| {
        format!(
            "failed to move {} to {}",
            part_path.display(),
            path.display()
        )
    })?;

    let sha256 = sha256_file(&path)?;
    if let Some(expected) = expected_sha256.as_deref() {
        anyhow::ensure!(
            sha256.eq_ignore_ascii_case(expected),
            "checksum mismatch for {}: expected {}, got {}",
            path.display(),
            expected,
            sha256
        );
    }
    write_checksum_sidecar(&path, &sha256)?;

    println!("{} {}", "downloaded".green(), path.display());
    println!("{} {}", "sha256".cyan(), sha256);
    Ok(())
}

fn fetch_builtin_asset(asset: &ModelAsset, path: &Path, force: bool) -> Result<()> {
    if is_non_empty_file(path) && !force {
        println!("{} {}", "already present".green(), path.display());
        return Ok(());
    }

    fs::create_dir_all(path.parent().context("model path has no parent")?)?;
    let body = serde_json::json!({
        "id": asset.id,
        "kind": "builtin",
        "source": asset.source,
        "license": asset.license,
        "notes": asset.notes,
    });
    fs::write(path, serde_json::to_vec_pretty(&body)?)?;
    println!("{} {}", "registered builtin".green(), path.display());
    Ok(())
}

#[derive(Default)]
struct RemoteMetadata {
    content_length: Option<u64>,
}

fn remote_metadata(asset: &ModelAsset) -> Result<RemoteMetadata> {
    let response = ureq::head(asset.url)
        .call()
        .with_context(|| format!("failed to inspect {}", asset.url))?;
    let content_length = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    Ok(RemoteMetadata { content_length })
}

fn verify_existing_asset(path: &Path, expected_sha256: Option<&str>) -> Result<()> {
    let sidecar_sha256 = read_checksum_sidecar(path)?;
    if let Some(expected) = expected_sha256.or(sidecar_sha256.as_deref()) {
        let actual = sha256_file(path)?;
        anyhow::ensure!(
            actual.eq_ignore_ascii_case(expected),
            "checksum mismatch for {}; rerun `cargo run models fetch -- --force`",
            path.display()
        );
    }
    Ok(())
}

fn file_len(path: &Path) -> Result<u64> {
    Ok(path.metadata()?.len())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut buffer = [0_u8; 128 * 1024];
    let mut hasher = Sha256::new();
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn checksum_sidecar_path(path: &Path) -> PathBuf {
    path.with_file_name(format!(
        "{}.sha256",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model")
    ))
}

fn write_checksum_sidecar(path: &Path, sha256: &str) -> Result<()> {
    fs::write(checksum_sidecar_path(path), format!("{sha256}\n"))?;
    Ok(())
}

fn read_checksum_sidecar(path: &Path) -> Result<Option<String>> {
    let sidecar = checksum_sidecar_path(path);
    if !sidecar.exists() {
        return Ok(None);
    }
    Ok(Some(fs::read_to_string(sidecar)?.trim().to_string()))
}

fn face_model_paths() -> Result<FaceModelPaths> {
    let home = resolve_mortar_home()?;
    let detector = find_asset("face-scrfd-34g-gnkps").context("missing face detector asset")?;
    let recognizer =
        find_asset("face-buffalo-l-w600k-r50").context("missing face recognizer asset")?;
    let attributes =
        find_asset("face-buffalo-l-genderage").context("missing face attributes asset")?;

    Ok(FaceModelPaths {
        detector: asset_path(&home, detector),
        recognizer: asset_path(&home, recognizer),
        attributes: asset_path(&home, attributes),
    })
}

fn print_progress(downloaded: u64, total: Option<u64>) {
    match total {
        Some(total) if total > 0 => {
            let pct = (downloaded as f64 / total as f64 * 100.0).min(100.0);
            print!(
                "\r{} {pct:5.1}% ({downloaded}/{total} bytes)",
                "downloading".cyan()
            );
        }
        _ => print!("\r{} {downloaded} bytes", "downloading".cyan()),
    }
    let _ = std::io::stdout().flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_runtime_fetch_includes_styletts2_bundle() {
        let ids = default_runtime_bundles()
            .expect("default runtime bundles")
            .into_iter()
            .map(|bundle| bundle.id)
            .collect::<Vec<_>>();

        assert!(ids.contains(&DEFAULT_STYLETTS2_MODEL_ID));
        assert!(ids.contains(&DEFAULT_PIPER_VOICE_MODEL_ID));
    }
}
