//! Licensed, backend-neutral model catalog and verified artifact lifecycle.
//!
//! Catalog metadata is deliberately independent of any runtime implementation.
//! Installers verify pinned artifacts before making them visible, record every
//! installed file, and can validate the same files while fully offline.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zip::ZipArchive;

use crate::open_model_package;

pub const MODEL_CATALOG_SCHEMA_VERSION: u32 = 2;
pub const INSTALLED_MODEL_SCHEMA_VERSION: u32 = 1;
pub const MODEL_VERIFICATION_CACHE_SCHEMA_VERSION: u32 = 1;
pub const MODEL_VERIFIER_VERSION: u32 = 1;
pub const MODEL_RUNTIME_COMPATIBILITY_VERSION: u32 = 1;
pub const EMBEDDED_MODEL_CATALOG: &str = include_str!("../catalog/models-v2.json");
pub const EMBEDDED_FAIRSEQ_MODEL_CATALOG: &str =
    include_str!("../catalog/fairseq-mms-models-v2.json");

// Parsing and merging the MMS inventory used to happen on every discovery
// request. The merge alone is large enough to stall the speech studio for
// tens of seconds, even though this compile-time catalog cannot change while
// the process is running. Cache the validated source and hand callers an
// independent clone so existing mutation-oriented APIs remain safe.
static EMBEDDED_CATALOG: OnceLock<Result<ModelCatalog, String>> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCatalog {
    pub schema_version: u32,
    pub id: String,
    pub entries: Vec<ModelCatalogEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCatalogEntry {
    pub id: String,
    pub display_name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    /// A provider-neutral neural architecture identifier. Serialization and
    /// distribution ancestry belong in `provenance` and `compatible_with`.
    pub architecture: String,
    /// Stable join from this distributable artifact to the implementation
    /// inventory. `None` means the artifact is intentionally catalog-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component_id: Option<String>,
    /// User-facing/runtime classification projected by CLI and server
    /// consumers instead of repeated in their own registries.
    #[serde(default)]
    pub kind: CatalogModelKind,
    /// Runtime provider that can load this artifact. `None` keeps an entry
    /// visible and installable without claiming that it can run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// Canonical installed file used to open the model. This may identify an
    /// extracted archive member.
    #[serde(default)]
    pub runtime_path: String,
    /// External checkpoint, configuration, or runtime contracts accepted by
    /// this entry. Values are searchable compatibility labels, not Tongues
    /// architecture identities.
    #[serde(default)]
    pub compatible_with: Vec<String>,
    pub package_version: u32,
    #[serde(default)]
    pub languages: Vec<String>,
    /// Source-declared writing system for script-qualified checkpoints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
    #[serde(default)]
    pub varieties: Vec<String>,
    pub speakers: CatalogSpeakers,
    pub sample_rate_hz: Option<u32>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Ordered preprocessing contracts required before model tokenization.
    #[serde(default)]
    pub preprocessing: Vec<String>,
    pub license: CatalogLicense,
    pub provenance: CatalogProvenance,
    #[serde(default)]
    pub artifacts: Vec<CatalogArtifact>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CatalogModelKind {
    AcousticModel,
    NeuralVocoder,
    #[default]
    EndToEndSpeech,
    VoiceConversion,
    StyleTts2,
    VoiceModel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CatalogSpeakers {
    pub count: usize,
    #[serde(default)]
    pub names: Vec<String>,
    pub names_file: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogLicense {
    pub expression: String,
    /// Stable upstream evidence for the declared license or `NOASSERTION`
    /// status. A value without evidence is not sufficient for installation.
    pub evidence: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogProvenance {
    /// Examples include `coqui-pytorch-zip`, `fairseq`, `onnx`, or a private
    /// organization's own source format.
    pub format: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogArtifact {
    pub url: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub install_path: String,
    /// Optional artifact-specific license when a bundle contains material
    /// under a different license than the model weights.
    #[serde(default)]
    pub license: Option<CatalogLicense>,
    #[serde(default)]
    pub members: Vec<CatalogArchiveMember>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogArchiveMember {
    pub archive_path: String,
    pub install_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledModelRecord {
    pub schema_version: u32,
    pub entry: ModelCatalogEntry,
    pub files: Vec<InstalledModelFile>,
    pub package_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledModelFile {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedModel {
    pub id: String,
    pub package_version: u32,
    pub files: Vec<PathBuf>,
    pub package_path: Option<PathBuf>,
    fingerprints: Vec<InstalledModelFile>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelVerificationStatus {
    Verified,
    PendingVerification,
    ChangedSinceVerification,
    VerificationFailed,
    Unavailable,
}

impl ModelVerificationStatus {
    pub fn needs_deep_verification(self) -> bool {
        matches!(
            self,
            Self::PendingVerification | Self::ChangedSinceVerification | Self::VerificationFailed
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelVerificationState {
    pub status: ModelVerificationStatus,
    pub verified_files: usize,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ModelVerificationCache {
    schema_version: u32,
    verifier_version: u32,
    runtime_compatibility_version: u32,
    catalog_schema_version: u32,
    manifest_version: u32,
    artifact_fingerprint: String,
    verified_at_unix_seconds: u64,
    success: bool,
    error: Option<String>,
    files: Vec<CachedVerificationFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CachedVerificationFile {
    path: String,
    size_bytes: u64,
    modified_unix_nanos: Option<u64>,
    sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelInstallProgress {
    CheckingCache {
        path: PathBuf,
    },
    Downloading {
        url: String,
        downloaded: u64,
        total: u64,
        part_path: PathBuf,
    },
    Verifying {
        path: PathBuf,
    },
    Installing {
        path: PathBuf,
    },
    Complete {
        id: String,
        version: u32,
    },
}

#[derive(Debug, Clone)]
pub struct ModelStore {
    root: PathBuf,
    cache: PathBuf,
    offline: bool,
}

impl ModelCatalog {
    pub fn embedded() -> Result<Self> {
        match EMBEDDED_CATALOG.get_or_init(|| {
            let mut catalog =
                Self::from_json(EMBEDDED_MODEL_CATALOG).map_err(|error| format!("{error:#}"))?;
            catalog
                .merge(
                    Self::from_json(EMBEDDED_FAIRSEQ_MODEL_CATALOG)
                        .map_err(|error| format!("{error:#}"))?,
                )
                .map_err(|error| format!("{error:#}"))?;
            Ok(catalog)
        }) {
            Ok(catalog) => Ok(catalog.clone()),
            Err(error) => anyhow::bail!("{error}"),
        }
    }

    pub fn from_json(source: &str) -> Result<Self> {
        let catalog: Self = serde_json::from_str(source).context("invalid model catalog JSON")?;
        catalog.validate()?;
        Ok(catalog)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read model catalog {}", path.display()))?;
        Self::from_json(&source)
            .with_context(|| format!("invalid model catalog {}", path.display()))
    }

    pub fn with_private_catalogs(paths: &[PathBuf]) -> Result<Self> {
        let mut catalog = Self::embedded()?;
        for path in paths {
            catalog.merge(Self::from_file(path)?)?;
        }
        Ok(catalog)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == MODEL_CATALOG_SCHEMA_VERSION,
            "unsupported model catalog schema {}; supported schema is {}",
            self.schema_version,
            MODEL_CATALOG_SCHEMA_VERSION
        );
        ensure!(!self.id.trim().is_empty(), "model catalog id is empty");
        let mut ids = BTreeMap::<String, &str>::new();
        let mut install_paths = BTreeMap::<&str, &str>::new();
        for entry in &self.entries {
            entry.validate()?;
            if let Some(component_id) = &entry.component_id {
                ensure!(
                    crate::native_speech_components()
                        .iter()
                        .any(|component| component.id == component_id),
                    "catalog model `{}` references unknown speech component `{component_id}`",
                    entry.id
                );
            }
            register_catalog_name(&mut ids, &entry.id, &entry.id)?;
            for alias in &entry.aliases {
                register_catalog_name(&mut ids, alias, &entry.id)?;
            }
            for artifact in &entry.artifacts {
                register_install_path(&mut install_paths, &artifact.install_path, &entry.id)?;
                for member in &artifact.members {
                    register_install_path(&mut install_paths, &member.install_path, &entry.id)?;
                }
            }
        }
        Ok(())
    }

    pub fn merge(&mut self, other: Self) -> Result<()> {
        ensure!(
            other.schema_version == self.schema_version,
            "cannot merge model catalog schema {} into schema {}",
            other.schema_version,
            self.schema_version
        );
        let mut ids = self
            .entries
            .iter()
            .map(|entry| normalize_id(&entry.id))
            .collect::<BTreeSet<_>>();
        for entry in other.entries {
            ensure!(
                ids.insert(normalize_id(&entry.id)),
                "private catalog attempts to replace existing model `{}`",
                entry.id
            );
            self.entries.push(entry);
        }
        self.entries.sort_by(|left, right| left.id.cmp(&right.id));
        self.validate()
    }

    pub fn find(&self, name: &str) -> Option<&ModelCatalogEntry> {
        let wanted = normalize_id(name);
        self.entries.iter().find(|entry| {
            normalize_id(&entry.id) == wanted
                || entry
                    .aliases
                    .iter()
                    .any(|alias| normalize_id(alias) == wanted)
        })
    }

    pub fn search(&self, query: &str) -> Vec<&ModelCatalogEntry> {
        let query = query.to_ascii_lowercase();
        let mut matches = self
            .entries
            .iter()
            .filter(|entry| {
                [
                    entry.id.as_str(),
                    entry.display_name.as_str(),
                    entry.architecture.as_str(),
                    entry.provenance.format.as_str(),
                ]
                .into_iter()
                .chain(entry.aliases.iter().map(String::as_str))
                .chain(entry.compatible_with.iter().map(String::as_str))
                .chain(entry.languages.iter().map(String::as_str))
                .chain(entry.varieties.iter().map(String::as_str))
                .chain(entry.capabilities.iter().map(String::as_str))
                .chain(entry.preprocessing.iter().map(String::as_str))
                .any(|value| value.to_ascii_lowercase().contains(&query))
            })
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| left.id.cmp(&right.id));
        matches
    }
}

impl ModelCatalogEntry {
    pub fn validate(&self) -> Result<()> {
        ensure_safe_id(&self.id)?;
        ensure!(
            !self.display_name.trim().is_empty(),
            "catalog model `{}` has an empty display name",
            self.id
        );
        ensure!(
            !self.architecture.trim().is_empty(),
            "catalog model `{}` has no architecture",
            self.id
        );
        if let Some(component_id) = &self.component_id {
            ensure_safe_id(component_id)?;
        }
        if let Some(backend) = &self.backend {
            ensure_safe_id(backend)?;
        }
        ensure_relative_path(&self.runtime_path)?;
        ensure!(
            self.compatible_with
                .iter()
                .all(|value| !value.trim().is_empty()),
            "catalog model `{}` has an empty compatibility identifier",
            self.id
        );
        ensure!(
            self.preprocessing
                .iter()
                .all(|value| !value.trim().is_empty()),
            "catalog model `{}` has an empty preprocessing requirement",
            self.id
        );
        ensure!(
            self.package_version > 0,
            "catalog model `{}` has package version zero",
            self.id
        );
        ensure!(
            !self.license.expression.trim().is_empty() && !self.license.evidence.trim().is_empty(),
            "catalog model `{}` lacks recorded license evidence",
            self.id
        );
        ensure!(
            !self.provenance.format.trim().is_empty() && !self.provenance.source.trim().is_empty(),
            "catalog model `{}` lacks provenance",
            self.id
        );
        ensure!(
            !self.artifacts.is_empty(),
            "catalog model `{}` contains no artifacts",
            self.id
        );
        for artifact in &self.artifacts {
            ensure!(
                artifact.url.starts_with("https://") || artifact.url.starts_with("file://"),
                "catalog model `{}` has unsupported artifact URL `{}`",
                self.id,
                artifact.url
            );
            validate_sha256(&artifact.sha256)?;
            ensure!(
                artifact.size_bytes > 0,
                "catalog artifact `{}` has no expected size",
                artifact.install_path
            );
            ensure_relative_path(&artifact.install_path)?;
            if let Some(license) = &artifact.license {
                ensure!(
                    !license.expression.trim().is_empty() && !license.evidence.trim().is_empty(),
                    "catalog artifact `{}` lacks recorded license evidence",
                    artifact.install_path
                );
            }
            for member in &artifact.members {
                ensure_relative_path(&member.archive_path)?;
                ensure_relative_path(&member.install_path)?;
            }
        }
        ensure!(
            self.artifacts.iter().any(|artifact| {
                artifact.install_path == self.runtime_path
                    || artifact
                        .members
                        .iter()
                        .any(|member| member.install_path == self.runtime_path)
            }),
            "catalog model `{}` runtime path `{}` is not declared by an artifact or member",
            self.id,
            self.runtime_path
        );
        if let Some(path) = &self.speakers.names_file {
            ensure_relative_path(path)?;
        }
        Ok(())
    }
}

impl ModelStore {
    pub fn from_environment() -> Result<Self> {
        let root = default_model_home()?;
        let cache = default_model_cache(&root)?;
        Ok(Self {
            root,
            cache,
            offline: environment_offline(),
        })
    }

    pub fn new(root: impl Into<PathBuf>, cache: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            cache: cache.into(),
            offline: false,
        }
    }

    pub fn with_offline(mut self, offline: bool) -> Self {
        self.offline = offline;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn cache(&self) -> &Path {
        &self.cache
    }

    pub fn offline(&self) -> bool {
        self.offline
    }

    pub fn install(&self, entry: &ModelCatalogEntry, force: bool) -> Result<VerifiedModel> {
        self.install_with_progress(entry, force, |_| {})
    }

    pub fn install_with_progress(
        &self,
        entry: &ModelCatalogEntry,
        force: bool,
        mut progress: impl FnMut(ModelInstallProgress),
    ) -> Result<VerifiedModel> {
        entry.validate()?;
        if !force {
            if let Ok(verified) = self.verify(entry) {
                return Ok(verified);
            }
        }
        fs::create_dir_all(&self.root)
            .with_context(|| format!("failed to create model home {}", self.root.display()))?;
        fs::create_dir_all(&self.cache)
            .with_context(|| format!("failed to create model cache {}", self.cache.display()))?;

        for artifact in &entry.artifacts {
            let cached = self.cached_artifact(artifact, &mut progress)?;
            let destination = checked_join(&self.root, &artifact.install_path)?;
            progress(ModelInstallProgress::Installing {
                path: destination.clone(),
            });
            install_file_atomic(&cached, &destination)?;
            if !artifact.members.is_empty() {
                extract_registered_members(&destination, artifact, &self.root, &mut progress)?;
            }
        }
        let record = self.build_record(entry, None)?;
        self.write_record(&record)?;
        let verified = self.verify(entry)?;
        progress(ModelInstallProgress::Complete {
            id: entry.id.clone(),
            version: entry.package_version,
        });
        Ok(verified)
    }

    pub fn install_local_package(
        &self,
        id: &str,
        source: impl AsRef<Path>,
        force: bool,
    ) -> Result<VerifiedModel> {
        let source = source.as_ref();
        ensure_safe_id(id)?;
        let package = open_model_package(source)
            .with_context(|| format!("invalid local Tongues package {}", source.display()))?;
        let version = package.manifest.schema_version;
        let destination = self
            .root
            .join("models/packages")
            .join(id)
            .join(format!("v{version}"));
        let staging = destination.with_extension("part");
        if destination.exists() {
            ensure!(
                force,
                "local package `{id}` version {version} is already installed; use --force to replace it"
            );
        }
        if staging.exists() {
            fs::remove_dir_all(&staging)
                .with_context(|| format!("failed to remove stale {}", staging.display()))?;
        }
        fs::create_dir_all(&staging)?;
        for name in [
            crate::MODEL_PACKAGE_MANIFEST,
            crate::MODEL_PACKAGE_CONFIG,
            crate::MODEL_PACKAGE_WEIGHTS,
            crate::MODEL_PACKAGE_TENSORS,
        ] {
            let from = package.directory.join(name);
            let to = staging.join(name);
            install_file_atomic(&from, &to)?;
        }
        open_model_package(&staging).context("copied local package failed verification")?;
        fs::create_dir_all(
            destination
                .parent()
                .context("local package destination has no parent")?,
        )?;
        if destination.exists() {
            let backup = destination.with_extension("old");
            if backup.exists() {
                fs::remove_dir_all(&backup)
                    .with_context(|| format!("failed to remove stale {}", backup.display()))?;
            }
            fs::rename(&destination, &backup).with_context(|| {
                format!(
                    "failed to preserve existing local package {}",
                    destination.display()
                )
            })?;
            if let Err(error) = fs::rename(&staging, &destination) {
                let _ = fs::rename(&backup, &destination);
                return Err(error).with_context(|| {
                    format!(
                        "failed to install local package {} at {}",
                        source.display(),
                        destination.display()
                    )
                });
            }
            fs::remove_dir_all(&backup)
                .with_context(|| format!("failed to remove replaced {}", backup.display()))?;
        } else {
            fs::rename(&staging, &destination).with_context(|| {
                format!(
                    "failed to install local package {} at {}",
                    source.display(),
                    destination.display()
                )
            })?;
        }

        let entry = ModelCatalogEntry {
            id: id.into(),
            display_name: id.into(),
            aliases: Vec::new(),
            architecture: package.manifest.architecture.as_str().into(),
            component_id: None,
            kind: CatalogModelKind::EndToEndSpeech,
            backend: None,
            runtime_path: destination
                .strip_prefix(&self.root)
                .context("local package escaped model home")?
                .to_string_lossy()
                .into_owned(),
            compatible_with: Vec::new(),
            package_version: version,
            languages: package
                .manifest
                .languages
                .iter()
                .map(|language| language.tag.clone())
                .collect(),
            script: None,
            varieties: Vec::new(),
            speakers: CatalogSpeakers {
                count: package.manifest.speakers.len(),
                names: package
                    .manifest
                    .speakers
                    .iter()
                    .map(|speaker| speaker.name.clone())
                    .collect(),
                names_file: None,
            },
            sample_rate_hz: package
                .manifest
                .audio
                .as_ref()
                .map(|audio| audio.sample_rate_hz),
            capabilities: vec!["tongues-model-package".into()],
            preprocessing: Vec::new(),
            license: CatalogLicense {
                expression: package.manifest.license.expression.clone(),
                evidence: package.manifest.provenance.source.clone(),
            },
            provenance: CatalogProvenance {
                format: package.manifest.provenance.source_format.clone(),
                source: package.manifest.provenance.source.clone(),
            },
            artifacts: vec![CatalogArtifact {
                url: format!("file://{}", package.directory.display()),
                sha256: sha256_file(&package.directory.join(crate::MODEL_PACKAGE_MANIFEST))?,
                size_bytes: fs::metadata(package.directory.join(crate::MODEL_PACKAGE_MANIFEST))?
                    .len(),
                install_path: destination
                    .strip_prefix(&self.root)
                    .context("local package escaped model home")?
                    .to_string_lossy()
                    .into_owned(),
                license: None,
                members: Vec::new(),
            }],
        };
        entry.validate()?;
        let record = self.build_record(&entry, Some(&destination))?;
        self.write_record(&record)?;
        self.verify(&entry)
    }

    /// Perform exhaustive artifact and package verification and persist the
    /// result. Normal application startup and synthesis should use
    /// [`Self::verification_state`] and [`Self::require_cached_verification`]
    /// instead.
    pub fn verify(&self, entry: &ModelCatalogEntry) -> Result<VerifiedModel> {
        let result = self.verify_uncached(entry);
        match result {
            Ok(verified) => {
                self.write_verification_cache(entry, &verified, None)?;
                Ok(verified)
            }
            Err(error) => {
                let message = format!("{error:#}");
                let _ = self.write_failed_verification_cache(entry, &message);
                Err(error)
            }
        }
    }

    /// Return verification state using only manifest/cache reads and file
    /// metadata. This never hashes an artifact or constructs a model.
    pub fn verification_state(&self, entry: &ModelCatalogEntry) -> ModelVerificationState {
        match self.verification_state_checked(entry) {
            Ok(state) => state,
            Err(error) => ModelVerificationState {
                status: ModelVerificationStatus::ChangedSinceVerification,
                verified_files: 0,
                error: Some(format!("{error:#}")),
            },
        }
    }

    /// Require a still-valid cached deep verification result without reading
    /// artifact contents.
    pub fn require_cached_verification(&self, entry: &ModelCatalogEntry) -> Result<()> {
        let state = self.verification_state_checked(entry)?;
        match state.status {
            ModelVerificationStatus::Verified => Ok(()),
            ModelVerificationStatus::PendingVerification => {
                anyhow::bail!("model `{}` is pending verification", entry.id)
            }
            ModelVerificationStatus::ChangedSinceVerification => anyhow::bail!(
                "model `{}` changed since verification{}",
                entry.id,
                state
                    .error
                    .as_deref()
                    .map(|error| format!(": {error}"))
                    .unwrap_or_default()
            ),
            ModelVerificationStatus::VerificationFailed => anyhow::bail!(
                "model `{}` failed verification{}",
                entry.id,
                state
                    .error
                    .as_deref()
                    .map(|error| format!(": {error}"))
                    .unwrap_or_default()
            ),
            ModelVerificationStatus::Unavailable => {
                anyhow::bail!("model `{}` is unavailable", entry.id)
            }
        }
    }

    fn verify_uncached(&self, entry: &ModelCatalogEntry) -> Result<VerifiedModel> {
        entry.validate()?;
        let record_path = self.record_path(&entry.id, entry.package_version);
        if record_path.is_file() {
            let source = fs::read_to_string(&record_path)
                .with_context(|| format!("failed to read {}", record_path.display()))?;
            let record: InstalledModelRecord =
                serde_json::from_str(&source).with_context(|| {
                    format!("invalid installed model record {}", record_path.display())
                })?;
            ensure!(
                installed_record_matches_entry(&record, entry)?,
                "installed model record does not match catalog entry {} v{}",
                entry.id,
                entry.package_version
            );
            return if record.package_path.is_some() {
                self.verify_record(&record)
            } else {
                // The record is inventory, not trust authority. Catalog pins
                // and verified archive contents remain authoritative.
                self.verify_configured_artifacts(entry)
            };
        }

        // Existing pre-catalog installations can be used offline only after
        // re-verifying every pinned artifact and extracted archive member.
        self.verify_configured_artifacts(entry)
    }

    fn verification_state_checked(
        &self,
        entry: &ModelCatalogEntry,
    ) -> Result<ModelVerificationState> {
        entry.validate()?;
        let expected_files = self.quick_verification_files(entry)?;
        for (path, _) in &expected_files {
            match fs::metadata(path) {
                Ok(metadata) if metadata.is_file() => {}
                Ok(_) => {
                    return Ok(ModelVerificationState {
                        status: ModelVerificationStatus::Unavailable,
                        verified_files: 0,
                        error: Some(format!("{} is not a file", path.display())),
                    });
                }
                Err(_) => {
                    return Ok(ModelVerificationState {
                        status: ModelVerificationStatus::Unavailable,
                        verified_files: 0,
                        error: Some(format!("model artifact is missing: {}", path.display())),
                    });
                }
            }
        }

        let cache_path = self.verification_cache_path(&entry.id, entry.package_version);
        if !cache_path.is_file() {
            return Ok(ModelVerificationState {
                status: ModelVerificationStatus::PendingVerification,
                verified_files: 0,
                error: None,
            });
        }
        let source = fs::read_to_string(&cache_path)
            .with_context(|| format!("failed to read {}", cache_path.display()))?;
        let cache: ModelVerificationCache = match serde_json::from_str(&source) {
            Ok(cache) => cache,
            Err(error) => {
                return Ok(ModelVerificationState {
                    status: ModelVerificationStatus::ChangedSinceVerification,
                    verified_files: 0,
                    error: Some(format!("cached verification metadata is invalid: {error}")),
                });
            }
        };
        let fingerprint = catalog_entry_fingerprint(entry)?;
        let context_error = if cache.schema_version != MODEL_VERIFICATION_CACHE_SCHEMA_VERSION {
            Some(format!(
                "verification cache schema changed from {} to {}",
                cache.schema_version, MODEL_VERIFICATION_CACHE_SCHEMA_VERSION
            ))
        } else if cache.verifier_version != MODEL_VERIFIER_VERSION {
            Some(format!(
                "verifier changed from {} to {}",
                cache.verifier_version, MODEL_VERIFIER_VERSION
            ))
        } else if cache.runtime_compatibility_version != MODEL_RUNTIME_COMPATIBILITY_VERSION {
            Some(format!(
                "runtime compatibility changed from {} to {}",
                cache.runtime_compatibility_version, MODEL_RUNTIME_COMPATIBILITY_VERSION
            ))
        } else if cache.catalog_schema_version != MODEL_CATALOG_SCHEMA_VERSION {
            Some(format!(
                "model manifest schema changed from {} to {}",
                cache.catalog_schema_version, MODEL_CATALOG_SCHEMA_VERSION
            ))
        } else if cache.manifest_version != entry.package_version {
            Some(format!(
                "model manifest changed from {} to {}",
                cache.manifest_version, entry.package_version
            ))
        } else if cache.artifact_fingerprint != fingerprint {
            Some("artifact fingerprint changed".into())
        } else {
            None
        };
        if let Some(error) = context_error {
            return Ok(ModelVerificationState {
                status: ModelVerificationStatus::ChangedSinceVerification,
                verified_files: cache.files.len(),
                error: Some(error),
            });
        }
        for file in &cache.files {
            let path = checked_join(&self.root, &file.path)?;
            let metadata = match fs::metadata(&path) {
                Ok(metadata) if metadata.is_file() => metadata,
                _ => {
                    return Ok(ModelVerificationState {
                        status: ModelVerificationStatus::Unavailable,
                        verified_files: cache.files.len(),
                        error: Some(format!("model artifact is missing: {}", path.display())),
                    });
                }
            };
            if metadata.len() != file.size_bytes
                || modified_unix_nanos(&metadata) != file.modified_unix_nanos
            {
                return Ok(ModelVerificationState {
                    status: ModelVerificationStatus::ChangedSinceVerification,
                    verified_files: cache.files.len(),
                    error: Some(format!("{} changed since verification", path.display())),
                });
            }
        }
        Ok(ModelVerificationState {
            status: if cache.success {
                ModelVerificationStatus::Verified
            } else {
                ModelVerificationStatus::VerificationFailed
            },
            verified_files: cache.files.len(),
            error: cache.error,
        })
    }

    fn quick_verification_files(
        &self,
        entry: &ModelCatalogEntry,
    ) -> Result<Vec<(PathBuf, Option<u64>)>> {
        let record_path = self.record_path(&entry.id, entry.package_version);
        if record_path.is_file() {
            if let Ok(record) =
                serde_json::from_slice::<InstalledModelRecord>(&fs::read(&record_path)?)
            {
                if installed_record_matches_entry(&record, entry).unwrap_or(false) {
                    return record
                        .files
                        .iter()
                        .map(|file| {
                            Ok((checked_join(&self.root, &file.path)?, Some(file.size_bytes)))
                        })
                        .collect();
                }
            }
        }
        entry
            .artifacts
            .iter()
            .flat_map(|artifact| {
                std::iter::once((artifact.install_path.as_str(), Some(artifact.size_bytes))).chain(
                    artifact
                        .members
                        .iter()
                        .map(|member| (member.install_path.as_str(), None)),
                )
            })
            .map(|(path, size)| Ok((checked_join(&self.root, path)?, size)))
            .collect()
    }

    fn write_verification_cache(
        &self,
        entry: &ModelCatalogEntry,
        verified: &VerifiedModel,
        error: Option<String>,
    ) -> Result<()> {
        let mut files = Vec::with_capacity(verified.fingerprints.len());
        for fingerprint in &verified.fingerprints {
            let path = checked_join(&self.root, &fingerprint.path)?;
            let metadata = fs::metadata(&path)
                .with_context(|| format!("failed to stat {}", path.display()))?;
            files.push(CachedVerificationFile {
                path: fingerprint.path.clone(),
                size_bytes: fingerprint.size_bytes,
                modified_unix_nanos: modified_unix_nanos(&metadata),
                sha256: Some(fingerprint.sha256.clone()),
            });
        }
        self.write_verification_cache_record(entry, error, files)
    }

    fn write_failed_verification_cache(
        &self,
        entry: &ModelCatalogEntry,
        error: &str,
    ) -> Result<()> {
        let files = self
            .quick_verification_files(entry)?
            .into_iter()
            .filter_map(|(path, _)| {
                let metadata = fs::metadata(&path).ok()?;
                let relative = path.strip_prefix(&self.root).ok()?;
                Some(CachedVerificationFile {
                    path: relative.to_string_lossy().into_owned(),
                    size_bytes: metadata.len(),
                    modified_unix_nanos: modified_unix_nanos(&metadata),
                    sha256: None,
                })
            })
            .collect();
        self.write_verification_cache_record(entry, Some(error.into()), files)
    }

    fn write_verification_cache_record(
        &self,
        entry: &ModelCatalogEntry,
        error: Option<String>,
        files: Vec<CachedVerificationFile>,
    ) -> Result<()> {
        let record = ModelVerificationCache {
            schema_version: MODEL_VERIFICATION_CACHE_SCHEMA_VERSION,
            verifier_version: MODEL_VERIFIER_VERSION,
            runtime_compatibility_version: MODEL_RUNTIME_COMPATIBILITY_VERSION,
            catalog_schema_version: MODEL_CATALOG_SCHEMA_VERSION,
            manifest_version: entry.package_version,
            artifact_fingerprint: catalog_entry_fingerprint(entry)?,
            verified_at_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            success: error.is_none(),
            error,
            files,
        };
        write_json_atomic(
            &self.verification_cache_path(&entry.id, entry.package_version),
            &record,
        )
    }

    pub fn installed_records(&self) -> Result<Vec<InstalledModelRecord>> {
        let directory = self.root.join("models/.catalog");
        if !directory.is_dir() {
            return Ok(Vec::new());
        }
        let mut records = Vec::new();
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let source = fs::read_to_string(entry.path())?;
            let record: InstalledModelRecord = serde_json::from_str(&source)?;
            records.push(record);
        }
        records.sort_by(|left, right| {
            (&left.entry.id, left.entry.package_version)
                .cmp(&(&right.entry.id, right.entry.package_version))
        });
        Ok(records)
    }

    pub fn remove(&self, entry: &ModelCatalogEntry, purge_cache: bool) -> Result<()> {
        entry.validate()?;
        let record_path = self.record_path(&entry.id, entry.package_version);
        if record_path.is_file() {
            let record: InstalledModelRecord = serde_json::from_slice(&fs::read(&record_path)?)?;
            if let Some(package_path) = record.package_path {
                let package_path = checked_join(&self.root, &package_path)?;
                if package_path.is_dir() {
                    fs::remove_dir_all(&package_path).with_context(|| {
                        format!("failed to remove package {}", package_path.display())
                    })?;
                }
            } else {
                for file in record.files {
                    let path = checked_join(&self.root, &file.path)?;
                    if path.is_file() {
                        fs::remove_file(&path)
                            .with_context(|| format!("failed to remove {}", path.display()))?;
                    }
                }
            }
            fs::remove_file(&record_path)
                .with_context(|| format!("failed to remove {}", record_path.display()))?;
        } else {
            for artifact in &entry.artifacts {
                for path in std::iter::once(artifact.install_path.as_str()).chain(
                    artifact
                        .members
                        .iter()
                        .map(|member| member.install_path.as_str()),
                ) {
                    let path = checked_join(&self.root, path)?;
                    if path.is_file() {
                        fs::remove_file(&path)
                            .with_context(|| format!("failed to remove {}", path.display()))?;
                    }
                }
            }
        }
        if purge_cache {
            for artifact in &entry.artifacts {
                let path = self.cache_path(artifact);
                if path.is_file() {
                    fs::remove_file(&path)
                        .with_context(|| format!("failed to remove {}", path.display()))?;
                }
            }
        }
        let verification_cache = self.verification_cache_path(&entry.id, entry.package_version);
        if verification_cache.is_file() {
            fs::remove_file(&verification_cache)
                .with_context(|| format!("failed to remove {}", verification_cache.display()))?;
        }
        Ok(())
    }

    fn verify_record(&self, record: &InstalledModelRecord) -> Result<VerifiedModel> {
        ensure!(
            record.schema_version == INSTALLED_MODEL_SCHEMA_VERSION,
            "unsupported installed-model record schema {}",
            record.schema_version
        );
        let mut files = Vec::with_capacity(record.files.len());
        for expected in &record.files {
            let path = checked_join(&self.root, &expected.path)?;
            verify_file(&path, expected.size_bytes, &expected.sha256)?;
            files.push(path);
        }
        let package_path = record
            .package_path
            .as_deref()
            .map(|path| checked_join(&self.root, path))
            .transpose()?;
        if let Some(path) = &package_path {
            open_model_package(path)
                .with_context(|| format!("installed package {} is invalid", path.display()))?;
        }
        Ok(VerifiedModel {
            id: record.entry.id.clone(),
            package_version: record.entry.package_version,
            files,
            package_path,
            fingerprints: record.files.clone(),
        })
    }

    fn verify_configured_artifacts(&self, entry: &ModelCatalogEntry) -> Result<VerifiedModel> {
        let mut files = Vec::new();
        let mut fingerprints = Vec::new();
        for artifact in &entry.artifacts {
            let path = checked_join(&self.root, &artifact.install_path)?;
            verify_file(&path, artifact.size_bytes, &artifact.sha256)?;
            files.push(path.clone());
            fingerprints.push(InstalledModelFile {
                path: artifact.install_path.clone(),
                size_bytes: artifact.size_bytes,
                sha256: artifact.sha256.to_ascii_lowercase(),
            });
            if !artifact.members.is_empty() {
                let archive_file = File::open(&path)?;
                let mut archive = ZipArchive::new(archive_file)
                    .with_context(|| format!("invalid archive {}", path.display()))?;
                for member in &artifact.members {
                    let installed = checked_join(&self.root, &member.install_path)?;
                    let mut source = archive.by_name(&member.archive_path).with_context(|| {
                        format!(
                            "archive {} is missing registered member {}",
                            path.display(),
                            member.archive_path
                        )
                    })?;
                    ensure!(!source.is_dir(), "{} is a directory", member.archive_path);
                    let (size, sha256) = hash_reader(&mut source)?;
                    verify_file(&installed, size, &sha256)?;
                    files.push(installed);
                    fingerprints.push(InstalledModelFile {
                        path: member.install_path.clone(),
                        size_bytes: size,
                        sha256,
                    });
                }
            }
        }
        Ok(VerifiedModel {
            id: entry.id.clone(),
            package_version: entry.package_version,
            files,
            package_path: None,
            fingerprints,
        })
    }

    fn cached_artifact(
        &self,
        artifact: &CatalogArtifact,
        progress: &mut impl FnMut(ModelInstallProgress),
    ) -> Result<PathBuf> {
        let path = self.cache_path(artifact);
        progress(ModelInstallProgress::CheckingCache { path: path.clone() });
        if path.is_file() && verify_file(&path, artifact.size_bytes, &artifact.sha256).is_ok() {
            return Ok(path);
        }
        fs::create_dir_all(&self.cache)?;
        let part = path.with_extension("part");
        let resumed = part.metadata().is_ok_and(|metadata| metadata.len() > 0);
        if resumed && verify_file(&part, artifact.size_bytes, &artifact.sha256).is_ok() {
            if path.exists() {
                fs::remove_file(&path)?;
            }
            fs::rename(&part, &path)?;
            return Ok(path);
        }
        ensure!(
            !self.offline,
            "offline mode is enabled and verified artifact {} is not cached at {}",
            artifact.sha256,
            path.display()
        );
        if part
            .metadata()
            .is_ok_and(|metadata| metadata.len() >= artifact.size_bytes)
        {
            fs::remove_file(&part)?;
        }
        if artifact.url.starts_with("file://") {
            let source = PathBuf::from(artifact.url.trim_start_matches("file://"));
            install_file_atomic(&source, &part)?;
        } else {
            download_resumable(artifact, &part, progress)?;
        }
        progress(ModelInstallProgress::Verifying { path: part.clone() });
        if let Err(first_error) = verify_file(&part, artifact.size_bytes, &artifact.sha256) {
            ensure!(
                resumed && artifact.url.starts_with("https://"),
                "{first_error:#}"
            );
            fs::remove_file(&part)?;
            download_resumable(artifact, &part, progress)?;
            verify_file(&part, artifact.size_bytes, &artifact.sha256)
                .context("full retry after an invalid partial download failed")?;
        }
        if path.exists() {
            fs::remove_file(&path)?;
        }
        fs::rename(&part, &path).with_context(|| {
            format!(
                "failed to atomically cache {} at {}",
                artifact.url,
                path.display()
            )
        })?;
        Ok(path)
    }

    fn cache_path(&self, artifact: &CatalogArtifact) -> PathBuf {
        self.cache.join(format!("{}.artifact", artifact.sha256))
    }

    fn build_record(
        &self,
        entry: &ModelCatalogEntry,
        package_path: Option<&Path>,
    ) -> Result<InstalledModelRecord> {
        let paths = if let Some(package_path) = package_path {
            let package = open_model_package(package_path)?;
            [
                crate::MODEL_PACKAGE_MANIFEST,
                crate::MODEL_PACKAGE_CONFIG,
                crate::MODEL_PACKAGE_WEIGHTS,
                crate::MODEL_PACKAGE_TENSORS,
            ]
            .into_iter()
            .map(|name| package.directory.join(name))
            .collect::<Vec<_>>()
        } else {
            entry
                .artifacts
                .iter()
                .flat_map(|artifact| {
                    std::iter::once(artifact.install_path.as_str()).chain(
                        artifact
                            .members
                            .iter()
                            .map(|member| member.install_path.as_str()),
                    )
                })
                .map(|path| checked_join(&self.root, path))
                .collect::<Result<Vec<_>>>()?
        };
        let mut files = paths
            .iter()
            .map(|path| installed_file(&self.root, path))
            .collect::<Result<Vec<_>>>()?;
        files.sort_by(|left, right| left.path.cmp(&right.path));
        files.dedup_by(|left, right| left.path == right.path);
        Ok(InstalledModelRecord {
            schema_version: INSTALLED_MODEL_SCHEMA_VERSION,
            entry: entry.clone(),
            files,
            package_path: package_path
                .map(|path| {
                    path.strip_prefix(&self.root)
                        .context("installed package escaped model home")
                        .map(|path| path.to_string_lossy().into_owned())
                })
                .transpose()?,
        })
    }

    fn write_record(&self, record: &InstalledModelRecord) -> Result<()> {
        let path = self.record_path(&record.entry.id, record.entry.package_version);
        write_json_atomic(&path, record)
    }

    fn record_path(&self, id: &str, version: u32) -> PathBuf {
        self.root
            .join("models/.catalog")
            .join(format!("{id}-v{version}.json"))
    }

    fn verification_cache_path(&self, id: &str, version: u32) -> PathBuf {
        self.root
            .join("models/.verification")
            .join(format!("{id}-v{version}.json"))
    }
}

pub fn private_catalog_paths_from_environment() -> Vec<PathBuf> {
    std::env::var_os("TONGUES_MODEL_CATALOGS")
        .map(|paths| std::env::split_paths(&paths).collect())
        .unwrap_or_default()
}

pub fn default_model_home() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("TONGUES_MODEL_HOME") {
        let path = PathBuf::from(path);
        ensure!(!path.as_os_str().is_empty(), "TONGUES_MODEL_HOME is empty");
        return Ok(path);
    }
    if let Some(path) = std::env::var_os("MORTAR_SEA_HOME") {
        let path = PathBuf::from(path);
        ensure!(!path.as_os_str().is_empty(), "MORTAR_SEA_HOME is empty");
        return Ok(path);
    }
    Ok(dirs::data_local_dir()
        .context("failed to resolve local data directory")?
        .join("mortar-sea"))
}

pub fn default_model_cache(root: &Path) -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("TONGUES_MODEL_CACHE") {
        let path = PathBuf::from(path);
        ensure!(!path.as_os_str().is_empty(), "TONGUES_MODEL_CACHE is empty");
        return Ok(path);
    }
    Ok(root.join("cache/model-downloads"))
}

pub fn environment_offline() -> bool {
    std::env::var("TONGUES_OFFLINE")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
}

fn register_install_path<'a>(
    paths: &mut BTreeMap<&'a str, &'a str>,
    path: &'a str,
    entry_id: &'a str,
) -> Result<()> {
    if let Some(existing) = paths.insert(path, entry_id) {
        ensure!(
            existing == entry_id,
            "catalog models `{existing}` and `{entry_id}` both install `{path}`"
        );
    }
    Ok(())
}

fn register_catalog_name<'a>(
    names: &mut BTreeMap<String, &'a str>,
    name: &str,
    entry_id: &'a str,
) -> Result<()> {
    if let Some(existing) = names.insert(normalize_id(name), entry_id) {
        ensure!(
            existing == entry_id,
            "catalog name `{name}` is shared by models `{existing}` and `{entry_id}`"
        );
    }
    Ok(())
}

fn normalize_id(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .map(|character| character.to_ascii_lowercase())
        .collect()
}

fn ensure_safe_id(id: &str) -> Result<()> {
    ensure!(
        !id.is_empty()
            && id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
        "unsafe model id `{id}`"
    );
    Ok(())
}

fn validate_sha256(value: &str) -> Result<()> {
    ensure!(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "invalid SHA-256 `{value}`"
    );
    Ok(())
}

fn ensure_relative_path(path: &str) -> Result<()> {
    let path = Path::new(path);
    ensure!(
        !path.as_os_str().is_empty()
            && path
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
        "path must be relative and normalized: {}",
        path.display()
    );
    Ok(())
}

fn checked_join(root: &Path, relative: &str) -> Result<PathBuf> {
    ensure_relative_path(relative)?;
    let path = root.join(relative);
    ensure!(
        path.starts_with(root),
        "model path escapes model home: {relative}"
    );
    Ok(path)
}

fn install_file_atomic(source: &Path, destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .context("model destination has no parent")?;
    fs::create_dir_all(parent)?;
    let part = destination.with_file_name(format!(
        "{}.part",
        destination
            .file_name()
            .and_then(|name| name.to_str())
            .context("model destination filename is not UTF-8")?
    ));
    let mut input =
        File::open(source).with_context(|| format!("failed to open {}", source.display()))?;
    let mut output =
        File::create(&part).with_context(|| format!("failed to create {}", part.display()))?;
    std::io::copy(&mut input, &mut output)?;
    output.flush()?;
    output.sync_all()?;
    fs::rename(&part, destination).with_context(|| {
        format!(
            "failed to atomically install {} at {}",
            source.display(),
            destination.display()
        )
    })?;
    Ok(())
}

fn extract_registered_members(
    archive_path: &Path,
    artifact: &CatalogArtifact,
    root: &Path,
    progress: &mut impl FnMut(ModelInstallProgress),
) -> Result<()> {
    let file = File::open(archive_path)
        .with_context(|| format!("failed to open archive {}", archive_path.display()))?;
    let mut archive = ZipArchive::new(file)
        .with_context(|| format!("invalid ZIP archive {}", archive_path.display()))?;
    for member in &artifact.members {
        let mut source = archive.by_name(&member.archive_path).with_context(|| {
            format!(
                "archive {} is missing registered member {}",
                archive_path.display(),
                member.archive_path
            )
        })?;
        ensure!(!source.is_dir(), "{} is a directory", member.archive_path);
        ensure!(
            source.enclosed_name().is_some(),
            "unsafe archive member {}",
            member.archive_path
        );
        let destination = checked_join(root, &member.install_path)?;
        progress(ModelInstallProgress::Installing {
            path: destination.clone(),
        });
        let parent = destination
            .parent()
            .context("archive member destination has no parent")?;
        fs::create_dir_all(parent)?;
        let part = destination.with_extension("part");
        let mut output =
            File::create(&part).with_context(|| format!("failed to create {}", part.display()))?;
        std::io::copy(&mut source, &mut output)?;
        output.flush()?;
        output.sync_all()?;
        fs::rename(&part, &destination).with_context(|| {
            format!(
                "failed to atomically install archive member {}",
                destination.display()
            )
        })?;
    }
    Ok(())
}

fn download_resumable(
    artifact: &CatalogArtifact,
    part: &Path,
    progress: &mut impl FnMut(ModelInstallProgress),
) -> Result<()> {
    let mut resume_from = part.metadata().map(|metadata| metadata.len()).unwrap_or(0);
    ensure!(
        resume_from <= artifact.size_bytes,
        "partial download {} is larger than expected",
        part.display()
    );
    let mut request = ureq::get(&artifact.url);
    if resume_from > 0 {
        request = request.header("Range", &format!("bytes={resume_from}-"));
    }
    let response = request
        .call()
        .with_context(|| format!("failed to download {}", artifact.url))?;
    if resume_from > 0 && response.status().as_u16() != 206 {
        resume_from = 0;
    }
    let mut body = response.into_body();
    let mut reader = body.as_reader();
    let mut output = OpenOptions::new()
        .create(true)
        .write(true)
        .append(resume_from > 0)
        .truncate(resume_from == 0)
        .open(part)
        .with_context(|| format!("failed to open {}", part.display()))?;
    let mut downloaded = resume_from;
    let mut next_report = downloaded;
    let mut buffer = [0u8; 128 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        output.write_all(&buffer[..read])?;
        downloaded = downloaded
            .checked_add(read as u64)
            .context("download size overflow")?;
        ensure!(
            downloaded <= artifact.size_bytes,
            "downloaded artifact exceeds expected size {}",
            artifact.size_bytes
        );
        if downloaded >= next_report || downloaded == artifact.size_bytes {
            progress(ModelInstallProgress::Downloading {
                url: artifact.url.clone(),
                downloaded,
                total: artifact.size_bytes,
                part_path: part.to_path_buf(),
            });
            next_report = downloaded.saturating_add(8 * 1024 * 1024);
        }
    }
    output.flush()?;
    output.sync_all()?;
    Ok(())
}

fn verify_file(path: &Path, expected_size: u64, expected_sha256: &str) -> Result<()> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("model artifact is missing: {}", path.display()))?;
    ensure!(metadata.is_file(), "{} is not a file", path.display());
    ensure!(
        metadata.len() == expected_size,
        "size mismatch for {}: expected {}, got {}",
        path.display(),
        expected_size,
        metadata.len()
    );
    let actual = sha256_file(path)?;
    ensure!(
        actual.eq_ignore_ascii_case(expected_sha256),
        "checksum mismatch for {}: expected {}, got {}",
        path.display(),
        expected_sha256,
        actual
    );
    Ok(())
}

fn installed_file(root: &Path, path: &Path) -> Result<InstalledModelFile> {
    let relative = path
        .strip_prefix(root)
        .with_context(|| format!("installed file {} escaped model home", path.display()))?
        .to_string_lossy()
        .into_owned();
    let metadata = fs::metadata(path)?;
    Ok(InstalledModelFile {
        path: relative,
        size_bytes: metadata.len(),
        sha256: sha256_file(path)?,
    })
}

fn hash_reader(reader: &mut impl Read) -> Result<(u64, String)> {
    let mut digest = Sha256::new();
    let mut size = 0u64;
    let mut buffer = [0u8; 128 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        size = size
            .checked_add(read as u64)
            .context("file size overflow")?;
        digest.update(&buffer[..read]);
    }
    Ok((
        size,
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    ))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = BufReader::new(
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?,
    );
    Ok(hash_reader(&mut file)?.1)
}

fn catalog_entry_fingerprint(entry: &ModelCatalogEntry) -> Result<String> {
    let artifacts = entry
        .artifacts
        .iter()
        .map(|artifact| {
            serde_json::json!({
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "install_path": artifact.install_path,
                "members": artifact.members,
            })
        })
        .collect::<Vec<_>>();
    let source =
        serde_json::to_vec(&artifacts).context("failed to serialize artifact fingerprints")?;
    Ok(Sha256::digest(source)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn installed_record_matches_entry(
    record: &InstalledModelRecord,
    entry: &ModelCatalogEntry,
) -> Result<bool> {
    if record.schema_version != INSTALLED_MODEL_SCHEMA_VERSION
        || record.entry.id != entry.id
        || record.entry.package_version != entry.package_version
    {
        return Ok(false);
    }

    // Schema-v1 installation records embedded the complete catalog entry.
    // Catalog schema v2 added runtime-only metadata to that entry without
    // changing the pinned artifacts. Older records therefore deserialize with
    // an empty runtime path and should remain usable when their artifact
    // fingerprints still match the current authoritative catalog.
    if record.entry.runtime_path.is_empty() {
        return Ok(catalog_entry_fingerprint(&record.entry)? == catalog_entry_fingerprint(entry)?);
    }

    Ok(record.entry == *entry)
}

fn modified_unix_nanos(metadata: &fs::Metadata) -> Option<u64> {
    let nanos = metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos();
    Some(nanos.min(u64::MAX as u128) as u64)
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path.parent().context("JSON path has no parent")?;
    fs::create_dir_all(parent)?;
    let part = path.with_extension("part");
    let file =
        File::create(&part).with_context(|| format!("failed to create {}", part.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&part, path)
        .with_context(|| format!("failed to atomically install {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_directory(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "tongues-model-catalog-{name}-{}",
            std::process::id()
        ));
        if path.exists() {
            fs::remove_dir_all(&path).expect("remove prior test directory");
        }
        fs::create_dir_all(&path).expect("create test directory");
        path
    }

    #[test]
    fn embedded_catalog_is_valid_and_covers_native_speech_models() {
        let catalog = ModelCatalog::embedded().expect("embedded catalog");
        for id in [
            "speedy-speech-ljspeech",
            "fastpitch-ljspeech",
            "hifigan-v2-ljspeech",
            "vits-vctk",
            "styletts2-en-us",
            "supertonic-3-multilingual",
            "voice-ljspeech-high",
            "voice-ryan-medium",
            "voice-amy-medium",
        ] {
            assert!(catalog.find(id).is_some(), "missing {id}");
        }
        assert_eq!(catalog.find("vits-vctk").unwrap().speakers.count, 109);
    }

    #[test]
    fn embedded_catalog_cache_returns_independent_catalogs() {
        let mut first = ModelCatalog::embedded().expect("first embedded catalog");
        let expected_entries = first.entries.len();
        first.entries.clear();

        let second = ModelCatalog::embedded().expect("cached embedded catalog");
        assert_eq!(second.entries.len(), expected_entries);
        assert!(second.find("styletts2-en-us").is_some());
    }

    #[test]
    fn catalog_requires_license_evidence_and_pinned_hashes() {
        let mut catalog = ModelCatalog::embedded().expect("embedded catalog");
        catalog.entries[0].license.evidence.clear();
        assert!(catalog
            .validate()
            .unwrap_err()
            .to_string()
            .contains("license"));

        let mut catalog = ModelCatalog::embedded().expect("embedded catalog");
        catalog.entries[0].artifacts[0].sha256 = "unpinned".into();
        assert!(catalog
            .validate()
            .unwrap_err()
            .to_string()
            .contains("SHA-256"));
    }

    #[test]
    fn catalog_rejects_duplicate_aliases_mismatched_paths_and_dangling_components() {
        let mut catalog = ModelCatalog::embedded().expect("embedded catalog");
        let duplicate_alias = catalog.entries[0].id.clone();
        catalog.entries[1].aliases.push(duplicate_alias);
        assert!(catalog
            .validate()
            .expect_err("duplicate alias")
            .to_string()
            .contains("is shared"));

        let mut catalog = ModelCatalog::embedded().expect("embedded catalog");
        catalog.entries[0].runtime_path = "models/not-declared.bin".into();
        assert!(catalog
            .validate()
            .expect_err("mismatched runtime path")
            .to_string()
            .contains("is not declared"));

        let mut catalog = ModelCatalog::embedded().expect("embedded catalog");
        catalog.entries[0].component_id = Some("missing-speech-component".into());
        assert!(catalog
            .validate()
            .expect_err("dangling component")
            .to_string()
            .contains("unknown speech component"));
    }

    #[test]
    fn private_catalog_cannot_replace_an_official_entry() {
        let mut catalog = ModelCatalog::embedded().expect("embedded catalog");
        let other = ModelCatalog {
            schema_version: MODEL_CATALOG_SCHEMA_VERSION,
            id: "private".into(),
            entries: vec![catalog.entries[0].clone()],
        };
        assert!(catalog
            .merge(other)
            .unwrap_err()
            .to_string()
            .contains("replace"));
    }

    #[test]
    fn offline_verification_rejects_corruption() {
        let root = temporary_directory("corruption");
        let cache = root.join("cache");
        let artifact_path = root.join("models/test/model.bin");
        fs::create_dir_all(artifact_path.parent().unwrap()).unwrap();
        fs::write(&artifact_path, b"valid").unwrap();
        let hash = sha256_file(&artifact_path).unwrap();
        let entry = ModelCatalogEntry {
            id: "fixture".into(),
            display_name: "Fixture".into(),
            aliases: Vec::new(),
            architecture: "fixture".into(),
            component_id: None,
            kind: CatalogModelKind::EndToEndSpeech,
            backend: None,
            runtime_path: "models/test/model.bin".into(),
            compatible_with: Vec::new(),
            package_version: 1,
            languages: Vec::new(),
            script: None,
            varieties: Vec::new(),
            speakers: CatalogSpeakers::default(),
            sample_rate_hz: None,
            capabilities: Vec::new(),
            preprocessing: Vec::new(),
            license: CatalogLicense {
                expression: "MIT".into(),
                evidence: "https://example.invalid/license".into(),
            },
            provenance: CatalogProvenance {
                format: "fixture".into(),
                source: "https://example.invalid/source".into(),
            },
            artifacts: vec![CatalogArtifact {
                url: "https://example.invalid/model".into(),
                sha256: hash,
                size_bytes: 5,
                install_path: "models/test/model.bin".into(),
                license: None,
                members: Vec::new(),
            }],
        };
        let store = ModelStore::new(&root, &cache).with_offline(true);
        assert_eq!(
            store.verification_state(&entry).status,
            ModelVerificationStatus::PendingVerification
        );
        store
            .write_record(&store.build_record(&entry, None).unwrap())
            .unwrap();
        store.verify(&entry).expect("valid offline artifact");
        let installed_record_path = store.record_path(&entry.id, entry.package_version);
        let mut legacy_record: serde_json::Value =
            serde_json::from_slice(&fs::read(&installed_record_path).unwrap()).unwrap();
        let legacy_entry = legacy_record["entry"].as_object_mut().unwrap();
        legacy_entry.remove("component_id");
        legacy_entry.remove("kind");
        legacy_entry.remove("backend");
        legacy_entry.remove("runtime_path");
        write_json_atomic(&installed_record_path, &legacy_record).unwrap();

        let restarted = ModelStore::new(&root, &cache).with_offline(true);
        assert_eq!(
            restarted.installed_records().unwrap().len(),
            1,
            "catalog-v1 installation records should remain readable"
        );
        restarted
            .verify(&entry)
            .expect("catalog-v1 installation record should verify against unchanged artifacts");
        assert_eq!(
            restarted.verification_state(&entry).status,
            ModelVerificationStatus::Verified,
            "a new process should trust unchanged cached verification metadata"
        );
        let mut display_only_change = entry.clone();
        display_only_change.display_name = "Renamed fixture".into();
        assert_eq!(
            restarted.verification_state(&display_only_change).status,
            ModelVerificationStatus::Verified,
            "display metadata is not an artifact or runtime compatibility change"
        );
        let verification_path = restarted.verification_cache_path(&entry.id, entry.package_version);
        let mut stale_cache: serde_json::Value =
            serde_json::from_slice(&fs::read(&verification_path).unwrap()).unwrap();
        stale_cache["verifier_version"] = serde_json::json!(MODEL_VERIFIER_VERSION + 1);
        write_json_atomic(&verification_path, &stale_cache).unwrap();
        assert_eq!(
            restarted.verification_state(&entry).status,
            ModelVerificationStatus::ChangedSinceVerification
        );
        restarted.verify(&entry).expect("refresh verifier cache");
        fs::write(&artifact_path, b"changed").unwrap();
        assert_eq!(
            restarted.verification_state(&entry).status,
            ModelVerificationStatus::ChangedSinceVerification
        );
        assert!(store
            .verify(&entry)
            .unwrap_err()
            .to_string()
            .contains("size mismatch"));
        assert_eq!(
            restarted.verification_state(&entry).status,
            ModelVerificationStatus::VerificationFailed
        );
        fs::remove_file(&artifact_path).unwrap();
        assert_eq!(
            restarted.verification_state(&entry).status,
            ModelVerificationStatus::Unavailable
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn local_artifact_install_is_atomic_cached_and_repairable_offline() {
        let root = temporary_directory("local-install");
        let source = root.join("source.bin");
        fs::write(&source, b"verified fixture").unwrap();
        let hash = sha256_file(&source).unwrap();
        let entry = ModelCatalogEntry {
            id: "local-fixture".into(),
            display_name: "Local Fixture".into(),
            aliases: Vec::new(),
            architecture: "fairseq".into(),
            component_id: None,
            kind: CatalogModelKind::EndToEndSpeech,
            backend: None,
            runtime_path: "models/private/model.bin".into(),
            compatible_with: Vec::new(),
            package_version: 1,
            languages: vec!["en".into()],
            script: None,
            varieties: Vec::new(),
            speakers: CatalogSpeakers::default(),
            sample_rate_hz: None,
            capabilities: vec!["fixture".into()],
            preprocessing: Vec::new(),
            license: CatalogLicense {
                expression: "LicenseRef-Private".into(),
                evidence: "https://example.invalid/private-license".into(),
            },
            provenance: CatalogProvenance {
                format: "fairseq".into(),
                source: "https://example.invalid/private-model".into(),
            },
            artifacts: vec![CatalogArtifact {
                url: format!("file://{}", source.display()),
                sha256: hash,
                size_bytes: 16,
                install_path: "models/private/model.bin".into(),
                license: None,
                members: Vec::new(),
            }],
        };
        let model_home = root.join("home");
        let cache = root.join("cache");
        let store = ModelStore::new(&model_home, &cache);
        let installed = store.install(&entry, false).expect("local install");
        assert_eq!(installed.files.len(), 1);
        assert!(!model_home.join("models/private/model.bin.part").exists());
        assert!(!cache
            .join(format!("{}.part", entry.artifacts[0].sha256))
            .exists());

        fs::write(
            model_home.join("models/private/model.bin"),
            b"corrupt fixture!",
        )
        .unwrap();
        assert!(store.verify(&entry).is_err());
        let offline = store.with_offline(true);
        offline
            .install(&entry, true)
            .expect("repair from verified offline cache");
        offline.verify(&entry).expect("repaired install");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn search_is_backend_neutral() {
        let catalog = ModelCatalog::embedded().expect("embedded catalog");
        let coqui = catalog.search("coqui");
        assert!(coqui.iter().any(|entry| entry.id == "vits-vctk"));
        assert!(
            coqui.iter().any(|entry| entry.id == "fairseq-mms-vits-eng"),
            "compatibility metadata should make Fairseq models discoverable by Coqui ancestry"
        );
        assert_eq!(catalog.search("onnx").len(), 5);
        assert_eq!(catalog.search("109").len(), 0);
        assert!(!catalog.search("uroman").is_empty());
    }

    #[test]
    fn embedded_supertonic_catalog_entry_preserves_pinned_multilingual_snapshot() {
        let catalog = ModelCatalog::embedded().expect("embedded catalog");
        let supertonic = catalog.find("supertonic-3-multilingual").unwrap();
        assert_eq!(supertonic.display_name, "Supertonic 3 Multilingual ONNX");
        assert_eq!(supertonic.sample_rate_hz, Some(44_100));
        assert_eq!(supertonic.speakers.count, 10);
        assert_eq!(supertonic.languages.len(), 32);
        assert!(supertonic.languages.iter().any(|lang| lang == "na"));
        assert_eq!(supertonic.artifacts.len(), 16);
        assert!(supertonic
            .compatible_with
            .iter()
            .any(|value| value == "upstream-archived-2026-07-23"));
        assert!(supertonic
            .provenance
            .source
            .contains("service-and-repository-notice-july-23-2026"));
        let style_hashes = supertonic
            .artifacts
            .iter()
            .filter(|artifact| artifact.install_path.contains("/voice_styles/"))
            .count();
        assert_eq!(style_hashes, 10);
    }

    #[test]
    fn embedded_fairseq_catalog_has_the_pinned_upstream_inventory() {
        let catalog = ModelCatalog::embedded().expect("embedded catalog");
        let fairseq = catalog
            .entries
            .iter()
            .filter(|entry| entry.provenance.format == "fairseq-mms-vits")
            .collect::<Vec<_>>();
        assert_eq!(fairseq.len(), 1_143);

        let english = catalog.find("tts_models/eng/fairseq/vits").unwrap();
        assert_eq!(english.id, "fairseq-mms-vits-eng");
        assert_eq!(english.languages, ["eng"]);
        assert!(
            english.varieties.is_empty(),
            "a language id is not evidence for pronunciation-variety coverage"
        );
        assert_eq!(english.artifacts.len(), 3);
    }

    #[test]
    fn embedded_architectures_are_provider_neutral_and_compatibility_is_explicit() {
        let catalog = ModelCatalog::embedded().expect("embedded catalog");
        for entry in &catalog.entries {
            assert!(
                !entry.architecture.starts_with("coqui-")
                    && !entry.architecture.starts_with("piper-"),
                "{} leaks provider identity into architecture {}",
                entry.id,
                entry.architecture
            );
        }

        let speedy = catalog.find("speedy-speech-ljspeech").unwrap();
        assert_eq!(speedy.architecture, "speedy-speech");
        assert!(speedy
            .compatible_with
            .iter()
            .any(|value| value == "coqui-speedy-speech"));

        let piper_voice = catalog.find("voice-ljspeech-high").unwrap();
        assert_eq!(piper_voice.architecture, "vits");
        assert!(piper_voice
            .compatible_with
            .iter()
            .any(|value| value == "piper-onnx"));
    }
}
