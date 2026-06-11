#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/new-model-family.sh <family-slug>

Creates a new model-family scaffold:
  crates/tongues-<family-slug>/
  configs/<family-slug>/default.toml
  datasets/<family-slug>/.gitkeep
  runs/<family-slug>/.gitkeep
  models/<family-slug>/.gitkeep

The family slug must be lowercase kebab-case, for example:
  sentence-boundary
  allophone-realizer
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

family="${1:-}"
if [[ -z "$family" ]]; then
    usage >&2
    exit 2
fi

if [[ ! "$family" =~ ^[a-z0-9][a-z0-9-]*[a-z0-9]$ && ! "$family" =~ ^[a-z0-9]$ ]]; then
    echo "error: family slug must be lowercase kebab-case: $family" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

crate_name="tongues-${family}"
crate_dir="crates/${crate_name}"
config_dir="configs/${family}"
dataset_dir="datasets/${family}"
run_dir="runs/${family}"
model_dir="models/${family}"

for path in "$crate_dir" "$config_dir"; do
    if [[ -e "$path" ]]; then
        echo "error: $path already exists" >&2
        exit 1
    fi
done

mkdir -p "$crate_dir/src" "$config_dir" "$dataset_dir" "$run_dir" "$model_dir"
touch "$dataset_dir/.gitkeep" "$run_dir/.gitkeep" "$model_dir/.gitkeep"

cat >"$crate_dir/Cargo.toml" <<EOF
[package]
name = "${crate_name}"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
tongues-neural = { path = "../tongues-neural" }
EOF

cat >"$crate_dir/src/lib.rs" <<EOF
//! ${family} model-family scaffold.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tongues_neural::{write_manifest, ModelArtifactManifest};

pub const FAMILY: &str = "${family}";
pub const ARCHITECTURE: &str = "scaffold";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FamilyConfig {
    pub dataset_id: String,
}

impl Default for FamilyConfig {
    fn default() -> Self {
        Self {
            dataset_id: "v0".to_string(),
        }
    }
}

pub fn prepare_dataset(out: &Path, config: &FamilyConfig) -> Result<()> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    fs::write(out.join("dataset_config.json"), serde_json::to_string_pretty(config)?)?;
    fs::write(
        out.join("README.md"),
        format!(
            "{} dataset scaffold. Add train/valid/test data here.\\n",
            FAMILY
        ),
    )?;
    Ok(())
}

pub fn write_scaffold_model(out: &Path, config: &FamilyConfig) -> Result<()> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    fs::write(out.join("model.bin"), format!("{} scaffold\\n", FAMILY))?;
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
        serde_json::to_string_pretty(&serde_json::json!({
            "status": "scaffold",
            "epochs": 0
        }))?,
    )?;
    write_manifest(
        out,
        &ModelArtifactManifest::new(FAMILY, ARCHITECTURE, &config.dataset_id),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_names_v0_dataset() {
        assert_eq!(FamilyConfig::default().dataset_id, "v0");
    }
}
EOF

cat >"$config_dir/default.toml" <<'EOF'
dataset_id = "v0"
EOF

python3 - "$crate_dir" <<'PY'
from pathlib import Path
import sys

member = sys.argv[1]
path = Path("Cargo.toml")
text = path.read_text()
entry = f'    "{member}",\n'
if entry in text:
    raise SystemExit(0)

anchor = '    "crates/tongues-cli",\n'
if anchor not in text:
    raise SystemExit("error: workspace member anchor not found in Cargo.toml")
text = text.replace(anchor, entry + anchor, 1)
path.write_text(text)
PY

echo "Created ${family} model family scaffold:"
echo "  ${crate_dir}"
echo "  ${config_dir}/default.toml"
echo "  ${dataset_dir}/.gitkeep"
echo "  ${run_dir}/.gitkeep"
echo "  ${model_dir}/.gitkeep"
echo
echo "Next steps:"
echo "  cargo test -p ${crate_name}"
echo "  wire ${family} into crates/tongues-cli when its CLI semantics are clear"
