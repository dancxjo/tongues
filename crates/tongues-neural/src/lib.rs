//! Shared neural-model artifact metadata.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const ARTIFACT_MANIFEST_FILE: &str = "manifest.json";
pub const ARTIFACT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArtifactManifest {
    pub schema_version: u32,
    pub family: String,
    pub architecture: String,
    pub created_by: String,
    pub data_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
}

impl ModelArtifactManifest {
    pub fn new(
        family: impl Into<String>,
        architecture: impl Into<String>,
        data_id: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            family: family.into(),
            architecture: architecture.into(),
            created_by: "tongues".to_string(),
            data_id: data_id.into(),
            task: None,
        }
    }

    pub fn with_task(mut self, task: impl Into<String>) -> Self {
        self.task = Some(task.into());
        self
    }
}

pub fn write_manifest(dir: &Path, manifest: &ModelArtifactManifest) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(ARTIFACT_MANIFEST_FILE);
    fs::write(&path, serde_json::to_string_pretty(manifest)?)
        .with_context(|| format!("writing {}", path.display()))
}

pub fn read_manifest(path: &Path) -> Result<ModelArtifactManifest> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_expected_contract() {
        let manifest = ModelArtifactManifest::new("g2p2g", "seq2seq-transformer", "openepd-v0")
            .with_task("both");
        let json = serde_json::to_value(&manifest).unwrap();

        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["family"], "g2p2g");
        assert_eq!(json["architecture"], "seq2seq-transformer");
        assert_eq!(json["created_by"], "tongues");
        assert_eq!(json["data_id"], "openepd-v0");
        assert_eq!(json["task"], "both");
    }
}
