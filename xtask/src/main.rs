use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("new-family") => {
            let family = args
                .next()
                .ok_or_else(|| format!("missing family slug\n\n{}", new_family_usage()))?;
            if args.next().is_some() {
                return Err(format!(
                    "new-family accepts exactly one family slug\n\n{}",
                    new_family_usage()
                ));
            }
            new_family(&family)
        }
        Some("-h") | Some("--help") | None => {
            print!("{}", usage());
            Ok(())
        }
        Some(command) => Err(format!("unknown xtask command `{command}`\n\n{}", usage())),
    }
}

fn usage() -> &'static str {
    "Usage: cargo xtask <command>\n\nCommands:\n  new-family <family-slug>  Create a model-family scaffold\n"
}

fn new_family_usage() -> &'static str {
    "Usage: cargo xtask new-family <family-slug>\n\nThe family slug must be lowercase kebab-case, for example:\n  sentence-boundary\n  allophone-realizer\n"
}

fn new_family(family: &str) -> Result<(), String> {
    validate_family_slug(family)?;

    let crate_name = format!("tongues-{family}");
    let crate_dir = PathBuf::from("crates").join(&crate_name);
    let config_dir = PathBuf::from("configs").join(family);
    let dataset_dir = PathBuf::from("datasets").join(family);
    let run_dir = PathBuf::from("runs").join(family);
    let model_dir = PathBuf::from("models").join(family);

    ensure_missing(&crate_dir)?;
    ensure_missing(&config_dir)?;

    fs::create_dir_all(crate_dir.join("src"))
        .map_err(|error| format!("creating {}: {error}", crate_dir.join("src").display()))?;
    fs::create_dir_all(&config_dir)
        .map_err(|error| format!("creating {}: {error}", config_dir.display()))?;
    create_placeholder_dir(&dataset_dir)?;
    create_placeholder_dir(&run_dir)?;
    create_placeholder_dir(&model_dir)?;

    write_file(&crate_dir.join("Cargo.toml"), &crate_manifest(&crate_name))?;
    write_file(&crate_dir.join("src/lib.rs"), &crate_lib_rs(family))?;
    write_file(&config_dir.join("default.toml"), "dataset_id = \"v0\"\n")?;
    add_workspace_member(&crate_dir)?;

    println!("Created {family} model family scaffold:");
    println!("  {}", crate_dir.display());
    println!("  {}", config_dir.join("default.toml").display());
    println!("  {}", dataset_dir.join(".gitkeep").display());
    println!("  {}", run_dir.join(".gitkeep").display());
    println!("  {}", model_dir.join(".gitkeep").display());
    println!();
    println!("Next steps:");
    println!("  cargo test -p {crate_name}");
    println!("  wire {family} into crates/tongues-cli when its CLI semantics are clear");

    Ok(())
}

fn validate_family_slug(family: &str) -> Result<(), String> {
    if family.is_empty() {
        return Err(format!("missing family slug\n\n{}", new_family_usage()));
    }
    let bytes = family.as_bytes();
    let starts_ok = bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit();
    let ends_ok =
        bytes[bytes.len() - 1].is_ascii_lowercase() || bytes[bytes.len() - 1].is_ascii_digit();
    let chars_ok = bytes
        .iter()
        .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-');
    if starts_ok && ends_ok && chars_ok {
        Ok(())
    } else {
        Err(format!(
            "family slug must be lowercase kebab-case: {family}\n\n{}",
            new_family_usage()
        ))
    }
}

fn ensure_missing(path: &Path) -> Result<(), String> {
    if path.exists() {
        Err(format!("{} already exists", path.display()))
    } else {
        Ok(())
    }
}

fn create_placeholder_dir(path: &Path) -> Result<(), String> {
    fs::create_dir_all(path).map_err(|error| format!("creating {}: {error}", path.display()))?;
    write_file(&path.join(".gitkeep"), "")
}

fn write_file(path: &Path, contents: &str) -> Result<(), String> {
    fs::write(path, contents).map_err(|error| format!("writing {}: {error}", path.display()))
}

fn add_workspace_member(crate_dir: &Path) -> Result<(), String> {
    let cargo_toml = Path::new("Cargo.toml");
    let text = fs::read_to_string(cargo_toml)
        .map_err(|error| format!("reading {}: {error}", cargo_toml.display()))?;
    let member = crate_dir.to_str().ok_or_else(|| {
        format!(
            "workspace member path is not UTF-8: {}",
            crate_dir.display()
        )
    })?;
    let entry = format!("    \"{member}\",\n");
    if text.contains(&entry) {
        return Ok(());
    }

    let anchor = "    \"crates/tongues-cli\",\n";
    let updated = text.replacen(anchor, &(entry + anchor), 1);
    if updated == text {
        return Err(format!(
            "workspace member anchor not found in {}",
            cargo_toml.display()
        ));
    }
    write_file(cargo_toml, &updated)
}

fn crate_manifest(crate_name: &str) -> String {
    format!(
        r#"[package]
name = "{crate_name}"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = {{ workspace = true }}
serde = {{ workspace = true }}
serde_json = {{ workspace = true }}
tongues-neural = {{ path = "../tongues-neural" }}
"#
    )
}

fn crate_lib_rs(family: &str) -> String {
    format!(
        r#"//! {family} model-family scaffold.

use std::fs;
use std::path::Path;

use anyhow::{{Context, Result}};
use serde::{{Deserialize, Serialize}};
use tongues_neural::{{write_manifest, ModelArtifactManifest}};

pub const FAMILY: &str = "{family}";
pub const ARCHITECTURE: &str = "scaffold";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FamilyConfig {{
    pub dataset_id: String,
}}

impl Default for FamilyConfig {{
    fn default() -> Self {{
        Self {{
            dataset_id: "v0".to_string(),
        }}
    }}
}}

pub fn prepare_dataset(out: &Path, config: &FamilyConfig) -> Result<()> {{
    fs::create_dir_all(out).with_context(|| format!("creating {{}}", out.display()))?;
    fs::write(out.join("dataset_config.json"), serde_json::to_string_pretty(config)?)?;
    fs::write(
        out.join("README.md"),
        format!(
            "{{}} dataset scaffold. Add train/valid/test data here.\n",
            FAMILY
        ),
    )?;
    Ok(())
}}

pub fn write_scaffold_model(out: &Path, config: &FamilyConfig) -> Result<()> {{
    fs::create_dir_all(out).with_context(|| format!("creating {{}}", out.display()))?;
    fs::write(out.join("model.bin"), format!("{{}} scaffold\n", FAMILY))?;
    fs::write(
        out.join("model_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    fs::write(
        out.join("train_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    fs::write(
        out.join("train_state.json"),
        serde_json::to_string_pretty(&serde_json::json!({{
            "status": "scaffold",
            "epochs": 0
        }}))?,
    )?;
    write_manifest(
        out,
        &ModelArtifactManifest::new(FAMILY, ARCHITECTURE, &config.dataset_id),
    )
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn default_config_names_v0_dataset() {{
        assert_eq!(FamilyConfig::default().dataset_id, "v0");
    }}
}}
"#
    )
}
