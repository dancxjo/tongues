//! Deterministic Fairseq MMS catalog generation and upstream drift checks.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    CatalogArtifact, CatalogLicense, CatalogProvenance, CatalogSpeakers, ModelCatalog,
    ModelCatalogEntry, FAIRSEQ_MMS_CHECKPOINT, FAIRSEQ_MMS_CONFIG, FAIRSEQ_MMS_SOURCE,
    FAIRSEQ_MMS_VOCAB, MODEL_CATALOG_SCHEMA_VERSION,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FairseqCatalogSource {
    pub schema_version: u32,
    pub id: String,
    /// Immutable upstream repository revision used in generated URLs.
    pub revision: String,
    pub entries: Vec<FairseqCatalogSourceEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FairseqCatalogSourceEntry {
    /// Published MMS identifier, including any `-script_*` suffix.
    pub model_id: String,
    pub language_name: String,
    /// ISO 639-3 identity without a script suffix.
    pub language: String,
    pub script: Option<String>,
    #[serde(default)]
    pub varieties: Vec<String>,
    #[serde(default)]
    pub preprocessing: Vec<String>,
    pub sample_rate_hz: u32,
    pub license: CatalogLicense,
    pub checkpoint: FairseqCatalogFile,
    pub config: FairseqCatalogFile,
    pub vocab: FairseqCatalogFile,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FairseqCatalogFile {
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct FairseqCatalogDrift {
    /// Present in the live language index but absent from the source snapshot.
    pub additions: Vec<String>,
    /// Present in the source snapshot but absent from the live language index.
    pub removals: Vec<String>,
    /// Same model id but a different upstream display name.
    pub renamed: Vec<FairseqCatalogRename>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FairseqCatalogRename {
    pub model_id: String,
    pub catalog_name: String,
    pub upstream_name: String,
}

impl FairseqCatalogDrift {
    pub fn is_empty(&self) -> bool {
        self.additions.is_empty() && self.removals.is_empty() && self.renamed.is_empty()
    }
}

impl FairseqCatalogSource {
    pub fn from_json(source: &str) -> Result<Self> {
        let source: Self =
            serde_json::from_str(source).context("invalid Fairseq MMS catalog source JSON")?;
        source.validate()?;
        Ok(source)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1,
            "unsupported Fairseq MMS catalog source schema {}",
            self.schema_version
        );
        ensure!(
            !self.id.trim().is_empty(),
            "Fairseq catalog source id is empty"
        );
        ensure!(
            self.revision.len() == 40
                && self.revision.chars().all(|value| value.is_ascii_hexdigit()),
            "Fairseq catalog source revision must be a 40-character Git commit"
        );
        ensure!(
            !self.entries.is_empty(),
            "Fairseq catalog source contains no entries"
        );
        let mut ids = BTreeSet::new();
        for entry in &self.entries {
            ensure!(
                ids.insert(entry.model_id.as_str()),
                "Fairseq catalog source repeats model `{}`",
                entry.model_id
            );
            ensure!(
                !entry.model_id.trim().is_empty()
                    && !entry.language_name.trim().is_empty()
                    && !entry.language.trim().is_empty(),
                "Fairseq catalog source entry has empty identity metadata"
            );
            ensure!(
                entry.model_id == entry.language
                    || entry
                        .model_id
                        .strip_prefix(&format!("{}-", entry.language))
                        .is_some(),
                "Fairseq model `{}` does not preserve language identity `{}`",
                entry.model_id,
                entry.language
            );
            match (&entry.script, entry.model_id.split_once("-script_")) {
                (Some(script), Some((_, suffix))) => ensure!(
                    script == suffix,
                    "Fairseq model `{}` declares script `{script}` but id suffix is `{suffix}`",
                    entry.model_id
                ),
                (None, None) => {}
                (Some(_), None) => anyhow::bail!(
                    "Fairseq model `{}` declares a script without a script-qualified id",
                    entry.model_id
                ),
                (None, Some(_)) => anyhow::bail!(
                    "Fairseq model `{}` has a script-qualified id but no script metadata",
                    entry.model_id
                ),
            }
            ensure!(
                entry.sample_rate_hz > 0,
                "Fairseq model `{}` has no sample rate",
                entry.model_id
            );
            ensure!(
                !entry.license.expression.trim().is_empty()
                    && !entry.license.evidence.trim().is_empty(),
                "Fairseq model `{}` lacks license metadata or evidence",
                entry.model_id
            );
            for (role, file) in [
                ("checkpoint", &entry.checkpoint),
                ("config", &entry.config),
                ("vocab", &entry.vocab),
            ] {
                ensure!(
                    file.sha256.len() == 64
                        && file.sha256.chars().all(|value| value.is_ascii_hexdigit()),
                    "Fairseq model `{}` {role} lacks a SHA-256 checksum",
                    entry.model_id
                );
                ensure!(
                    file.size_bytes > 0,
                    "Fairseq model `{}` {role} has no expected size",
                    entry.model_id
                );
            }
            ensure!(
                entry
                    .preprocessing
                    .iter()
                    .all(|requirement| !requirement.trim().is_empty()),
                "Fairseq model `{}` contains an empty preprocessing requirement",
                entry.model_id
            );
        }
        Ok(())
    }

    pub fn drift_from_language_index(&self, html: &str) -> Result<FairseqCatalogDrift> {
        let upstream = parse_fairseq_language_index(html)?;
        let snapshot = self
            .entries
            .iter()
            .map(|entry| (entry.model_id.clone(), entry.language_name.clone()))
            .collect::<BTreeMap<_, _>>();
        let additions = upstream
            .keys()
            .filter(|id| !snapshot.contains_key(*id))
            .cloned()
            .collect();
        let removals = snapshot
            .keys()
            .filter(|id| !upstream.contains_key(*id))
            .cloned()
            .collect();
        let renamed = snapshot
            .iter()
            .filter_map(|(model_id, catalog_name)| {
                let upstream_name = upstream.get(model_id)?;
                (catalog_name != upstream_name).then(|| FairseqCatalogRename {
                    model_id: model_id.clone(),
                    catalog_name: catalog_name.clone(),
                    upstream_name: upstream_name.clone(),
                })
            })
            .collect();
        Ok(FairseqCatalogDrift {
            additions,
            removals,
            renamed,
        })
    }

    pub fn ensure_matches_language_index(&self, html: &str) -> Result<()> {
        let drift = self.drift_from_language_index(html)?;
        ensure!(
            drift.is_empty(),
            "Fairseq MMS catalog drift detected: {} additions, {} removals, {} renamed entries",
            drift.additions.len(),
            drift.removals.len(),
            drift.renamed.len()
        );
        Ok(())
    }
}

pub fn generate_fairseq_catalog(source: &FairseqCatalogSource) -> Result<ModelCatalog> {
    source.validate()?;
    let mut entries = source
        .entries
        .iter()
        .map(|entry| {
            let directory = format!("models/speech/fairseq-mms/{}", entry.model_id);
            let catalog_model_id = safe_catalog_model_id(&entry.model_id);
            let source_model_id = url_path_segment(&entry.model_id);
            let artifact = |filename: &str, file: &FairseqCatalogFile| CatalogArtifact {
                url: format!(
                    "{FAIRSEQ_MMS_SOURCE}/resolve/{}/models/{}/{}",
                    source.revision, source_model_id, filename
                ),
                sha256: file.sha256.clone(),
                size_bytes: file.size_bytes,
                install_path: format!("{directory}/{filename}"),
                license: None,
                members: Vec::new(),
            };
            ModelCatalogEntry {
                id: format!("fairseq-mms-vits-{catalog_model_id}"),
                display_name: format!("MMS VITS {} ({})", entry.language_name, entry.model_id),
                aliases: vec![format!("tts_models/{}/fairseq/vits", entry.model_id)],
                architecture: "vits".into(),
                compatible_with: vec!["fairseq-mms-vits".into(), "coqui-fairseq-vits".into()],
                package_version: 1,
                languages: vec![entry.language.clone()],
                script: entry.script.clone(),
                varieties: entry.varieties.clone(),
                speakers: CatalogSpeakers {
                    count: 1,
                    names: Vec::new(),
                    names_file: None,
                },
                sample_rate_hz: Some(entry.sample_rate_hz),
                capabilities: vec![
                    "end-to-end-speech".into(),
                    "graphemes".into(),
                    "speed".into(),
                    "seed".into(),
                ],
                preprocessing: entry.preprocessing.clone(),
                license: entry.license.clone(),
                provenance: CatalogProvenance {
                    format: "fairseq-mms-vits".into(),
                    source: format!(
                        "{FAIRSEQ_MMS_SOURCE}/tree/{}/models/{}",
                        source.revision, source_model_id
                    ),
                },
                artifacts: vec![
                    artifact(FAIRSEQ_MMS_CHECKPOINT, &entry.checkpoint),
                    artifact(FAIRSEQ_MMS_CONFIG, &entry.config),
                    artifact(FAIRSEQ_MMS_VOCAB, &entry.vocab),
                ],
            }
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.id.cmp(&right.id));
    let catalog = ModelCatalog {
        schema_version: MODEL_CATALOG_SCHEMA_VERSION,
        id: source.id.clone(),
        entries,
    };
    catalog.validate()?;
    Ok(catalog)
}

fn safe_catalog_model_id(model_id: &str) -> String {
    let mut output = String::with_capacity(model_id.len());
    for character in model_id.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
            output.push(character);
        } else {
            output.push_str(&format!("_u{:x}_", u32::from(character)));
        }
    }
    output
}

fn url_path_segment(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            output.push(char::from(byte));
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

/// Parse Meta's intentionally simple `<p> code &emsp; name </p>` index.
pub fn parse_fairseq_language_index(html: &str) -> Result<BTreeMap<String, String>> {
    let mut entries = BTreeMap::new();
    for line in html.lines() {
        let Some(content) = line
            .trim()
            .strip_prefix("<p>")
            .and_then(|line| line.strip_suffix("</p>"))
        else {
            continue;
        };
        let Some((model_id, language_name)) = content.split_once("&emsp;") else {
            continue;
        };
        let model_id = model_id.trim();
        let language_name = decode_minimal_html(language_name.trim());
        if model_id.eq_ignore_ascii_case("iso code") {
            continue;
        }
        ensure!(
            !model_id.is_empty() && !language_name.is_empty(),
            "invalid Fairseq language-index row {line:?}"
        );
        ensure!(
            entries
                .insert(model_id.to_string(), language_name)
                .is_none(),
            "Fairseq language index repeats `{model_id}`"
        );
    }
    ensure!(
        !entries.is_empty(),
        "Fairseq language index contains no model rows"
    );
    Ok(entries)
}

fn decode_minimal_html(value: &str) -> String {
    value
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(byte: char, size_bytes: u64) -> FairseqCatalogFile {
        FairseqCatalogFile {
            sha256: std::iter::repeat_n(byte, 64).collect(),
            size_bytes,
        }
    }

    fn source() -> FairseqCatalogSource {
        FairseqCatalogSource {
            schema_version: 1,
            id: "fairseq-mms-fixture".into(),
            revision: "44cc7fb408064ef9ea6e7c59130d88cac1274671".into(),
            entries: vec![
                FairseqCatalogSourceEntry {
                    model_id: "eng".into(),
                    language_name: "English".into(),
                    language: "eng".into(),
                    script: None,
                    varieties: vec!["en-US-GA".into(), "en-GB-RP".into()],
                    preprocessing: vec!["lowercase-and-filter-vocab".into()],
                    sample_rate_hz: 16_000,
                    license: CatalogLicense {
                        expression: "CC-BY-NC-4.0".into(),
                        evidence: "https://example.invalid/license".into(),
                    },
                    checkpoint: file('a', 145_484_625),
                    config: file('b', 1_887),
                    vocab: file('c', 78),
                },
                FairseqCatalogSourceEntry {
                    model_id: "azj-script_cyrillic".into(),
                    language_name: "Azerbaijani, North".into(),
                    language: "azj".into(),
                    script: Some("cyrillic".into()),
                    varieties: Vec::new(),
                    preprocessing: vec!["uroman".into()],
                    sample_rate_hz: 16_000,
                    license: CatalogLicense {
                        expression: "CC-BY-NC-4.0".into(),
                        evidence: "https://example.invalid/license".into(),
                    },
                    checkpoint: file('d', 145_000_000),
                    config: file('e', 1_900),
                    vocab: file('f', 100),
                },
            ],
        }
    }

    #[test]
    fn generated_entries_preserve_script_and_variety_boundaries() {
        let catalog = generate_fairseq_catalog(&source()).unwrap();
        let english = catalog.find("tts_models/eng/fairseq/vits").unwrap();
        assert_eq!(english.languages, ["eng"]);
        assert_eq!(english.varieties, ["en-US-GA", "en-GB-RP"]);
        assert_eq!(english.artifacts.len(), 3);
        assert_eq!(english.provenance.format, "fairseq-mms-vits");
        assert!(
            !english.capabilities.iter().any(|value| value == "streaming"),
            "MMS VITS synthesis is whole-utterance; revision-safe suffix replacement is handled by the ledger layer"
        );

        let azj = catalog
            .find("fairseq-mms-vits-azj-script_cyrillic")
            .unwrap();
        assert_eq!(azj.languages, ["azj"]);
        assert_eq!(azj.script.as_deref(), Some("cyrillic"));
        assert!(azj.varieties.is_empty());
        assert_eq!(azj.preprocessing, ["uroman"]);
    }

    #[test]
    fn drift_check_detects_additions_removals_and_renames() {
        let html = r#"<!doctype html>
<p> Iso Code &emsp; Language Name </p>
<p> eng &emsp; English, Modern </p>
<p> amh &emsp; Amharic </p>
"#;
        let drift = source().drift_from_language_index(html).unwrap();
        assert_eq!(drift.additions, ["amh"]);
        assert_eq!(drift.removals, ["azj-script_cyrillic"]);
        assert_eq!(drift.renamed.len(), 1);
        assert_eq!(drift.renamed[0].model_id, "eng");
    }

    #[test]
    fn source_validation_rejects_license_and_checksum_gaps() {
        let mut missing_license = source();
        missing_license.entries[0].license.evidence.clear();
        assert!(missing_license
            .validate()
            .unwrap_err()
            .to_string()
            .contains("license"));

        let mut missing_checksum = source();
        missing_checksum.entries[0].vocab.sha256.clear();
        assert!(missing_checksum
            .validate()
            .unwrap_err()
            .to_string()
            .contains("checksum"));
    }

    #[test]
    fn committed_catalog_is_the_deterministic_output_of_the_source_snapshot() {
        let source =
            FairseqCatalogSource::from_json(include_str!("../catalog/fairseq-mms-source-v1.json"))
                .expect("committed source snapshot");
        let generated = generate_fairseq_catalog(&source).expect("generated catalog");
        let committed =
            ModelCatalog::from_json(include_str!("../catalog/fairseq-mms-models-v1.json"))
                .expect("committed generated catalog");

        assert_eq!(source.entries.len(), 1_143);
        assert_eq!(generated, committed);
    }
}
