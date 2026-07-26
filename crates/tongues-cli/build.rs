use std::{env, fs, path::PathBuf};

use serde_json::Value;

fn string(value: &Value, field: &str) -> String {
    serde_json::to_string(
        value
            .get(field)
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("catalog field `{field}` is missing")),
    )
    .expect("serialize catalog string")
}

fn optional_string(value: Option<&str>) -> String {
    value
        .map(|value| format!("Some({})", serde_json::to_string(value).unwrap()))
        .unwrap_or_else(|| "None".into())
}

fn main() {
    let catalog_path = PathBuf::from("../tongues-tts/catalog/models-v2.json");
    println!("cargo:rerun-if-changed={}", catalog_path.display());
    let catalog: Value =
        serde_json::from_slice(&fs::read(&catalog_path).expect("read speech model catalog"))
            .expect("parse speech model catalog");
    let entries = catalog["entries"]
        .as_array()
        .expect("speech catalog entries");

    let mut assets = String::from("pub const GENERATED_SPEECH_MODEL_ASSETS: &[ModelAsset] = &[\n");
    let mut archives =
        String::from("pub const GENERATED_SPEECH_MODEL_ARCHIVES: &[ModelArchive] = &[\n");
    let mut bundles =
        String::from("pub const GENERATED_SPEECH_MODEL_BUNDLES: &[ModelBundle] = &[\n");

    for entry in entries {
        let id = entry["id"].as_str().expect("entry id");
        let runtime_path = entry["runtime_path"].as_str().expect("runtime path");
        let artifacts = entry["artifacts"].as_array().expect("entry artifacts");
        let mut asset_ids = Vec::new();
        let mut primary_asset_id = None;
        for (index, artifact) in artifacts.iter().enumerate() {
            let asset_id = format!("{id}-artifact-{}", index + 1);
            let install_path = artifact["install_path"].as_str().expect("install path");
            let members = artifact["members"].as_array();
            if install_path == runtime_path
                || members.is_some_and(|members| {
                    members
                        .iter()
                        .any(|member| member["install_path"].as_str() == Some(runtime_path))
                })
            {
                primary_asset_id = Some(asset_id.clone());
            }
            let filename = install_path.rsplit('/').next().expect("artifact filename");
            let license = artifact
                .get("license")
                .and_then(|license| license.get("expression"))
                .and_then(Value::as_str)
                .or_else(|| entry["license"]["expression"].as_str());
            assets.push_str(&format!(
                "ModelAsset {{ id: {asset_id:?}, filename: {filename:?}, relative_path: {install_path:?}, url: {}, sha256: {}, size_bytes: Some({}), license: {}, source: Some({}), notes: None }},\n",
                string(artifact, "url"),
                optional_string(artifact["sha256"].as_str()),
                artifact["size_bytes"].as_u64().expect("artifact size"),
                optional_string(license),
                string(&entry["provenance"], "source"),
            ));
            if let Some(members) = members.filter(|members| !members.is_empty()) {
                archives.push_str(&format!(
                    "ModelArchive {{ asset_id: {asset_id:?}, primary_member_path: {runtime_path:?}, members: &[\n"
                ));
                for member in members {
                    archives.push_str(&format!(
                        "ArchiveMember {{ member_path: {}, relative_path: {} }},\n",
                        string(member, "archive_path"),
                        string(member, "install_path"),
                    ));
                }
                archives.push_str("] },\n");
            }
            asset_ids.push(asset_id);
        }
        let primary_asset_id =
            primary_asset_id.unwrap_or_else(|| panic!("{id} runtime path has no artifact"));
        let kind = match entry["kind"].as_str().expect("model kind") {
            "acoustic_model" => "ModelKind::AcousticModel",
            "neural_vocoder" => "ModelKind::NeuralVocoder",
            "end_to_end_speech" => "ModelKind::EndToEndSpeech",
            "voice_conversion" => "ModelKind::VoiceConversion",
            "style_tts2" => "ModelKind::StyleTts2",
            "voice_model" => "ModelKind::VoiceModel",
            kind => panic!("unsupported speech model kind `{kind}`"),
        };
        let required = asset_ids
            .iter()
            .map(|id| format!("{id:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        let aliases = entry["aliases"]
            .as_array()
            .expect("aliases")
            .iter()
            .map(|alias| format!("{:?}", alias.as_str().expect("alias string")))
            .collect::<Vec<_>>()
            .join(", ");
        bundles.push_str(&format!(
            "ModelBundle {{ id: {id:?}, display_name: {}, kind: {kind}, primary_asset_id: {primary_asset_id:?}, required_asset_ids: &[{required}], aliases: &[{aliases}] }},\n",
            string(entry, "display_name"),
        ));
    }
    assets.push_str("];\n");
    archives.push_str("];\n");
    bundles.push_str("];\n");
    let output = format!("{assets}\n{archives}\n{bundles}");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    fs::write(out_dir.join("speech-model-catalog.rs"), output)
        .expect("write generated speech model projection");
}
