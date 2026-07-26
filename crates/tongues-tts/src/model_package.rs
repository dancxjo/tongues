//! Safe, deterministic import of legacy Coqui model artifacts.
//!
//! The compatibility boundary deliberately ends at import time. PyTorch ZIP
//! checkpoints are parsed by Rust code after a restrictive pickle opcode and
//! callable scan, then rewritten as SafeTensors. Runtime packages never need
//! Python, pickle, or Coqui.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::DType;
use burn_store::pytorch::PytorchReader;
use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use zip::ZipArchive;

use crate::vits_config::ImportedVitsConfig;
use crate::{
    AudioFeatureConfig, BurnVitsSpeech, FastPitchConfig, HifiganBundleConfig,
    PhonemeTokenizerConfig, PhonemeVocabularyProjector, SpeakerCatalog, SpeedySpeechConfig,
    VitsInferenceConfig,
};

pub const MODEL_PACKAGE_SCHEMA_VERSION: u32 = 1;
pub const MODEL_PACKAGE_FORMAT: &str = "tongues-model-package";
pub const MODEL_PACKAGE_MANIFEST: &str = "manifest.json";
pub const MODEL_PACKAGE_CONFIG: &str = "model.json";
pub const MODEL_PACKAGE_WEIGHTS: &str = "model.safetensors";
pub const MODEL_PACKAGE_TENSORS: &str = "tensors.json";

const MAX_PICKLE_METADATA_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TENSOR_COUNT: usize = 100_000;
const MAX_TENSOR_ELEMENTS: usize = 1 << 34;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelPackageArchitecture {
    FastPitch,
    SpeedySpeech,
    HifiGan,
    Vits,
    SpeakerEncoder,
}

impl ModelPackageArchitecture {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FastPitch => "fast_pitch",
            Self::SpeedySpeech => "speedy_speech",
            Self::HifiGan => "hifi_gan",
            Self::Vits => "vits",
            Self::SpeakerEncoder => "speaker_encoder",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageSpeaker {
    pub id: u32,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageLanguage {
    pub id: Option<u32>,
    pub tag: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageLicense {
    pub expression: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageProvenance {
    pub source: String,
    pub source_format: String,
    pub importer: String,
    pub importer_version: String,
    pub coqui_version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageArtifact {
    pub role: String,
    pub filename: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageFile {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageAudio {
    pub sample_rate_hz: u32,
    pub fft_size: Option<usize>,
    pub window_size: Option<usize>,
    pub hop_size: Option<usize>,
    pub mel_bins: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub elements: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeutralModelConfig {
    pub schema_version: u32,
    pub architecture: ModelPackageArchitecture,
    pub parameters: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelPackageManifest {
    pub schema_version: u32,
    pub package_format: String,
    pub architecture: ModelPackageArchitecture,
    pub runtime: String,
    pub config: String,
    pub weights: String,
    pub tensor_index: String,
    pub tensor_count: usize,
    pub audio: Option<PackageAudio>,
    pub speakers: Vec<PackageSpeaker>,
    pub languages: Vec<PackageLanguage>,
    pub symbols: Vec<String>,
    pub license: PackageLicense,
    pub provenance: PackageProvenance,
    pub source_artifacts: Vec<PackageArtifact>,
    pub files: Vec<PackageFile>,
    /// Config fields intentionally excluded from inference packages. Keeping
    /// this list makes the conversion explicit instead of silently dropping
    /// Coqui training state.
    pub ignored_training_fields: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ModelPackage {
    pub directory: PathBuf,
    pub manifest: ModelPackageManifest,
    pub config: NeutralModelConfig,
    pub tensors: Vec<TensorMetadata>,
}

impl ModelPackage {
    pub fn weights_path(&self) -> PathBuf {
        self.directory.join(&self.manifest.weights)
    }
}

impl ModelPackageManifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == MODEL_PACKAGE_SCHEMA_VERSION,
            "unsupported Tongues model package schema {}; current schema is {}",
            self.schema_version,
            MODEL_PACKAGE_SCHEMA_VERSION
        );
        ensure!(
            self.package_format == MODEL_PACKAGE_FORMAT,
            "unsupported model package format `{}`",
            self.package_format
        );
        ensure!(
            self.config == MODEL_PACKAGE_CONFIG
                && self.weights == MODEL_PACKAGE_WEIGHTS
                && self.tensor_index == MODEL_PACKAGE_TENSORS,
            "model package uses non-canonical member names"
        );
        ensure!(self.tensor_count > 0, "model package contains no tensors");
        ensure!(
            !self.license.expression.trim().is_empty(),
            "model package license expression is empty"
        );
        ensure!(
            !self.provenance.source.trim().is_empty(),
            "model package provenance source is empty"
        );
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CoquiImportOptions {
    pub config_path: PathBuf,
    pub checkpoint_path: PathBuf,
    pub output_dir: PathBuf,
    pub speaker_map_path: Option<PathBuf>,
    pub language_map_path: Option<PathBuf>,
    pub checkpoint_key: String,
    pub license: String,
    pub source: String,
    pub coqui_version: Option<String>,
}

impl CoquiImportOptions {
    pub fn new(
        config_path: impl Into<PathBuf>,
        checkpoint_path: impl Into<PathBuf>,
        output_dir: impl Into<PathBuf>,
        license: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            config_path: config_path.into(),
            checkpoint_path: checkpoint_path.into(),
            output_dir: output_dir.into(),
            speaker_map_path: None,
            language_map_path: None,
            checkpoint_key: "model".into(),
            license: license.into(),
            source: source.into(),
            coqui_version: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelImportProgress {
    ReadingConfig {
        path: PathBuf,
    },
    ScanningCheckpoint {
        path: PathBuf,
    },
    ValidatingShapes {
        architecture: ModelPackageArchitecture,
    },
    ValidatingConvertedWeights {
        architecture: ModelPackageArchitecture,
        path: PathBuf,
    },
    ConvertingTensor {
        current: usize,
        total: usize,
        name: String,
        output: PathBuf,
    },
    WritingMetadata {
        path: PathBuf,
    },
    Complete {
        path: PathBuf,
        sha256: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoquiImportInspection {
    pub architecture: ModelPackageArchitecture,
    pub tensor_count: usize,
    pub tensors: Vec<TensorMetadata>,
    pub audio: Option<PackageAudio>,
    pub speakers: Vec<PackageSpeaker>,
    pub languages: Vec<PackageLanguage>,
    pub symbols: Vec<String>,
    pub ignored_training_fields: Vec<String>,
    pub source_artifacts: Vec<PackageArtifact>,
}

#[derive(Debug)]
enum ParsedConfig {
    FastPitch {
        model: FastPitchConfig,
        audio: AudioFeatureConfig,
        parameters: Value,
        symbols: Vec<String>,
    },
    Speedy {
        model: SpeedySpeechConfig,
        audio: AudioFeatureConfig,
        parameters: Value,
        symbols: Vec<String>,
    },
    HifiGan {
        model: HifiganBundleConfig,
        parameters: Value,
    },
    Vits {
        inference: VitsInferenceConfig,
        parameters: Value,
        symbols: Vec<String>,
    },
    SpeakerEncoder {
        model: SpeakerEncoderPackageConfig,
        parameters: Value,
    },
}

impl ParsedConfig {
    fn architecture(&self) -> ModelPackageArchitecture {
        match self {
            Self::FastPitch { .. } => ModelPackageArchitecture::FastPitch,
            Self::Speedy { .. } => ModelPackageArchitecture::SpeedySpeech,
            Self::HifiGan { .. } => ModelPackageArchitecture::HifiGan,
            Self::Vits { .. } => ModelPackageArchitecture::Vits,
            Self::SpeakerEncoder { .. } => ModelPackageArchitecture::SpeakerEncoder,
        }
    }

    fn parameters(&self) -> &Value {
        match self {
            Self::FastPitch { parameters, .. }
            | Self::Speedy { parameters, .. }
            | Self::HifiGan { parameters, .. }
            | Self::Vits { parameters, .. }
            | Self::SpeakerEncoder { parameters, .. } => parameters,
        }
    }

    fn audio(&self) -> Option<PackageAudio> {
        let audio = match self {
            Self::FastPitch { audio, .. } | Self::Speedy { audio, .. } => audio,
            Self::HifiGan { model, .. } => &model.audio,
            Self::Vits { inference, .. } => &inference.audio,
            Self::SpeakerEncoder { model, .. } => {
                return Some(PackageAudio {
                    sample_rate_hz: model.sample_rate_hz,
                    fft_size: model.fft_size,
                    window_size: model.window_size,
                    hop_size: model.hop_size,
                    mel_bins: Some(model.input_dim),
                });
            }
        };
        Some(PackageAudio {
            sample_rate_hz: audio.sample_rate,
            fft_size: Some(audio.fft_size),
            window_size: Some(audio.win_length),
            hop_size: Some(audio.hop_length),
            mel_bins: Some(audio.num_mels),
        })
    }

    fn symbols(&self) -> Vec<String> {
        match self {
            Self::FastPitch { symbols, .. }
            | Self::Speedy { symbols, .. }
            | Self::Vits { symbols, .. } => symbols.clone(),
            Self::HifiGan { .. } | Self::SpeakerEncoder { .. } => Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerEncoderPackageConfig {
    pub model_name: String,
    pub input_dim: usize,
    pub projection_dim: usize,
    pub lstm_dim: usize,
    pub num_lstm_layers: usize,
    pub use_lstm_with_projection: bool,
    pub sample_rate_hz: u32,
    pub fft_size: Option<usize>,
    pub window_size: Option<usize>,
    pub hop_size: Option<usize>,
}

impl SpeakerEncoderPackageConfig {
    fn from_root(root: &Value) -> Result<Self> {
        let params = root
            .get("model_params")
            .or_else(|| root.get("model_args"))
            .and_then(Value::as_object)
            .context("speaker encoder config requires `model_params` or `model_args`")?;
        reject_unknown_keys(
            params,
            &[
                "model_name",
                "input_dim",
                "num_mels",
                "proj_dim",
                "projection_dim",
                "lstm_dim",
                "hidden_dim",
                "num_lstm_layers",
                "num_layers",
                "use_lstm_with_projection",
            ],
            "model_params",
        )?;
        let audio = root.get("audio").and_then(Value::as_object);
        let config = Self {
            model_name: optional_string(params, &["model_name"]).unwrap_or_else(|| "lstm".into()),
            input_dim: required_usize_alias(params, &["input_dim", "num_mels"])?,
            projection_dim: required_usize_alias(params, &["proj_dim", "projection_dim"])?,
            lstm_dim: required_usize_alias(params, &["lstm_dim", "hidden_dim"])?,
            num_lstm_layers: required_usize_alias(params, &["num_lstm_layers", "num_layers"])?,
            use_lstm_with_projection: params
                .get("use_lstm_with_projection")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            sample_rate_hz: audio
                .and_then(|value| optional_u64(value, &["sample_rate"]))
                .unwrap_or(16_000)
                .try_into()
                .context("speaker encoder sample rate does not fit u32")?,
            fft_size: audio
                .and_then(|value| optional_u64(value, &["fft_size", "n_fft"]))
                .map(usize::try_from)
                .transpose()
                .context("speaker encoder FFT size does not fit usize")?,
            window_size: audio
                .and_then(|value| optional_u64(value, &["win_length"]))
                .map(usize::try_from)
                .transpose()
                .context("speaker encoder window size does not fit usize")?,
            hop_size: audio
                .and_then(|value| optional_u64(value, &["hop_length"]))
                .map(usize::try_from)
                .transpose()
                .context("speaker encoder hop size does not fit usize")?,
        };
        ensure!(
            matches!(config.model_name.as_str(), "lstm" | "speaker_encoder"),
            "unsupported speaker encoder model_name `{}`",
            config.model_name
        );
        ensure!(
            config.input_dim > 0
                && config.projection_dim > 0
                && config.lstm_dim > 0
                && config.num_lstm_layers > 0,
            "speaker encoder dimensions must be positive"
        );
        Ok(config)
    }
}

pub fn inspect_coqui_import(options: &CoquiImportOptions) -> Result<CoquiImportInspection> {
    inspect_coqui_import_with_progress(options, |_| {})
}

pub fn inspect_coqui_import_with_progress(
    options: &CoquiImportOptions,
    mut progress: impl FnMut(ModelImportProgress),
) -> Result<CoquiImportInspection> {
    validate_options(options)?;
    progress(ModelImportProgress::ReadingConfig {
        path: options.config_path.clone(),
    });
    let (root, parsed, ignored_training_fields) = parse_config(&options.config_path)?;
    progress(ModelImportProgress::ScanningCheckpoint {
        path: options.checkpoint_path.clone(),
    });
    scan_safe_pytorch_checkpoint(&options.checkpoint_path)?;
    let reader = checkpoint_reader(options)?;
    let tensors = tensor_metadata(&reader)?;
    let speakers = load_speakers(options, &parsed)?;
    let languages = load_languages(options, &root)?;
    progress(ModelImportProgress::ValidatingShapes {
        architecture: parsed.architecture(),
    });
    validate_runtime_shapes(options, &parsed, &tensors, &options.checkpoint_path)?;
    let source_artifacts = source_artifacts(options)?;
    Ok(CoquiImportInspection {
        architecture: parsed.architecture(),
        tensor_count: tensors.len(),
        tensors,
        audio: parsed.audio(),
        speakers,
        languages,
        symbols: parsed.symbols(),
        ignored_training_fields,
        source_artifacts,
    })
}

pub fn import_coqui_model(options: &CoquiImportOptions) -> Result<ModelPackageManifest> {
    import_coqui_model_with_progress(options, |_| {})
}

pub fn import_coqui_model_with_progress(
    options: &CoquiImportOptions,
    mut progress: impl FnMut(ModelImportProgress),
) -> Result<ModelPackageManifest> {
    let inspection = inspect_coqui_import_with_progress(options, &mut progress)?;
    let (_, parsed, _) = parse_config(&options.config_path)?;
    fs::create_dir_all(&options.output_dir).with_context(|| {
        format!(
            "failed to create model package directory {}",
            options.output_dir.display()
        )
    })?;

    let neutral_config = NeutralModelConfig {
        schema_version: MODEL_PACKAGE_SCHEMA_VERSION,
        architecture: inspection.architecture,
        parameters: parsed.parameters().clone(),
    };
    let config_path = options.output_dir.join(MODEL_PACKAGE_CONFIG);
    progress(ModelImportProgress::WritingMetadata {
        path: config_path.clone(),
    });
    write_json_atomic(&config_path, &neutral_config)?;

    let tensor_path = options.output_dir.join(MODEL_PACKAGE_TENSORS);
    progress(ModelImportProgress::WritingMetadata {
        path: tensor_path.clone(),
    });
    write_json_atomic(&tensor_path, &inspection.tensors)?;

    let weights_path = options.output_dir.join(MODEL_PACKAGE_WEIGHTS);
    convert_checkpoint(options, &weights_path, &inspection.tensors, &mut progress)?;
    verify_safetensors(&weights_path, &inspection.tensors)?;
    progress(ModelImportProgress::ValidatingConvertedWeights {
        architecture: inspection.architecture,
        path: weights_path.clone(),
    });
    validate_runtime_shapes(options, &parsed, &inspection.tensors, &weights_path)?;

    let files = [
        MODEL_PACKAGE_CONFIG,
        MODEL_PACKAGE_WEIGHTS,
        MODEL_PACKAGE_TENSORS,
    ]
    .into_iter()
    .map(|name| package_file(&options.output_dir.join(name), name))
    .collect::<Result<Vec<_>>>()?;
    let manifest = ModelPackageManifest {
        schema_version: MODEL_PACKAGE_SCHEMA_VERSION,
        package_format: MODEL_PACKAGE_FORMAT.into(),
        architecture: inspection.architecture,
        runtime: "backend-neutral".into(),
        config: MODEL_PACKAGE_CONFIG.into(),
        weights: MODEL_PACKAGE_WEIGHTS.into(),
        tensor_index: MODEL_PACKAGE_TENSORS.into(),
        tensor_count: inspection.tensor_count,
        audio: inspection.audio,
        speakers: inspection.speakers,
        languages: inspection.languages,
        symbols: inspection.symbols,
        license: PackageLicense {
            expression: options.license.trim().into(),
        },
        provenance: PackageProvenance {
            source: options.source.trim().into(),
            source_format: "coqui-pytorch-zip".into(),
            importer: "tongues-tts".into(),
            importer_version: env!("CARGO_PKG_VERSION").into(),
            coqui_version: options.coqui_version.clone(),
        },
        source_artifacts: inspection.source_artifacts,
        files,
        ignored_training_fields: inspection.ignored_training_fields,
    };
    manifest.validate()?;
    let manifest_path = options.output_dir.join(MODEL_PACKAGE_MANIFEST);
    progress(ModelImportProgress::WritingMetadata {
        path: manifest_path.clone(),
    });
    write_json_atomic(&manifest_path, &manifest)?;
    let manifest_sha256 = sha256_file(&manifest_path)?;
    progress(ModelImportProgress::Complete {
        path: options.output_dir.clone(),
        sha256: manifest_sha256,
    });
    Ok(manifest)
}

pub fn read_model_package(path: impl AsRef<Path>) -> Result<ModelPackageManifest> {
    let path = path.as_ref();
    let package_dir = if path.is_dir() {
        path
    } else {
        path.parent()
            .context("model package manifest has no parent directory")?
    };
    let manifest_path = if path.is_dir() {
        path.join(MODEL_PACKAGE_MANIFEST)
    } else {
        path.to_path_buf()
    };
    let source = fs::read_to_string(&manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    let value: Value = serde_json::from_str(&source)
        .with_context(|| format!("invalid model package manifest {}", manifest_path.display()))?;
    let migrated = migrate_model_package_manifest(value)?;
    let manifest: ModelPackageManifest =
        serde_json::from_value(migrated).context("invalid migrated model package manifest")?;
    manifest.validate()?;
    for file in &manifest.files {
        ensure_safe_member_name(&file.path)?;
        let file_path = package_dir.join(&file.path);
        let metadata = fs::metadata(&file_path)
            .with_context(|| format!("model package file is missing: {}", file_path.display()))?;
        ensure!(
            metadata.len() == file.size_bytes,
            "model package file {} has size {}, expected {}",
            file.path,
            metadata.len(),
            file.size_bytes
        );
        let actual = sha256_file(&file_path)?;
        ensure!(
            actual == file.sha256,
            "model package file {} checksum mismatch: expected {}, got {}",
            file.path,
            file.sha256,
            actual
        );
    }
    Ok(manifest)
}

pub fn open_model_package(path: impl AsRef<Path>) -> Result<ModelPackage> {
    let path = path.as_ref();
    let directory = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .context("model package manifest has no parent directory")?
            .to_path_buf()
    };
    let manifest = read_model_package(path)?;
    let config_path = directory.join(&manifest.config);
    let config: NeutralModelConfig = serde_json::from_slice(
        &fs::read(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?,
    )
    .with_context(|| format!("invalid neutral model config {}", config_path.display()))?;
    ensure!(
        config.schema_version == MODEL_PACKAGE_SCHEMA_VERSION,
        "neutral model config schema {} does not match package schema {}",
        config.schema_version,
        MODEL_PACKAGE_SCHEMA_VERSION
    );
    ensure!(
        config.architecture == manifest.architecture,
        "neutral model config architecture {:?} does not match manifest {:?}",
        config.architecture,
        manifest.architecture
    );
    let tensor_path = directory.join(&manifest.tensor_index);
    let tensors: Vec<TensorMetadata> = serde_json::from_slice(
        &fs::read(&tensor_path)
            .with_context(|| format!("failed to read {}", tensor_path.display()))?,
    )
    .with_context(|| format!("invalid tensor index {}", tensor_path.display()))?;
    ensure!(
        tensors.len() == manifest.tensor_count,
        "tensor index contains {} tensors, manifest declares {}",
        tensors.len(),
        manifest.tensor_count
    );
    ensure!(
        tensors.windows(2).all(|pair| pair[0].name < pair[1].name),
        "tensor index names are not strictly sorted"
    );
    for tensor in &tensors {
        ensure_safe_tensor_name(&tensor.name)?;
        ensure!(
            checked_elements(&tensor.shape)? == tensor.elements,
            "tensor index element count mismatch for `{}`",
            tensor.name
        );
    }
    verify_safetensors(&directory.join(&manifest.weights), &tensors)?;
    Ok(ModelPackage {
        directory,
        manifest,
        config,
        tensors,
    })
}

pub fn migrate_model_package_manifest(mut value: Value) -> Result<Value> {
    let object = value
        .as_object_mut()
        .context("model package manifest must be a JSON object")?;
    let version = object
        .get("schema_version")
        .or_else(|| object.get("version"))
        .and_then(Value::as_u64)
        .context("model package manifest has no numeric schema version")?;
    match version {
        0 => {
            object.remove("version");
            object.insert(
                "schema_version".into(),
                Value::from(MODEL_PACKAGE_SCHEMA_VERSION),
            );
            object
                .entry("package_format")
                .or_insert_with(|| Value::from(MODEL_PACKAGE_FORMAT));
            object
                .entry("runtime")
                .or_insert_with(|| Value::from("backend-neutral"));
            Ok(value)
        }
        1 => Ok(value),
        other => bail!(
            "model package schema {other} is newer than supported schema {}",
            MODEL_PACKAGE_SCHEMA_VERSION
        ),
    }
}

fn validate_options(options: &CoquiImportOptions) -> Result<()> {
    ensure!(
        options.config_path.is_file(),
        "Coqui config does not exist: {}",
        options.config_path.display()
    );
    ensure!(
        options.checkpoint_path.is_file(),
        "Coqui checkpoint does not exist: {}",
        options.checkpoint_path.display()
    );
    ensure!(
        !options.checkpoint_key.trim().is_empty(),
        "checkpoint key must not be empty"
    );
    ensure!(
        !options.license.trim().is_empty(),
        "a license expression is required"
    );
    ensure!(
        !options.source.trim().is_empty(),
        "a provenance source is required"
    );
    Ok(())
}

fn parse_config(path: &Path) -> Result<(Value, ParsedConfig, Vec<String>)> {
    let source = fs::read_to_string(path)
        .with_context(|| format!("failed to read Coqui config {}", path.display()))?;
    let root: Value = json5::from_str(&source)
        .with_context(|| format!("invalid Coqui JSON/JSON5 config {}", path.display()))?;
    let object = root
        .as_object()
        .context("Coqui config root must be an object")?;
    let architecture = detect_architecture(object)?;
    let parsed = match architecture {
        ModelPackageArchitecture::FastPitch => {
            reject_model_args(
                object,
                &[
                    "d_vector_dim",
                    "decoder_params",
                    "decoder_type",
                    "detach_duration_predictor",
                    "duration_predictor_dropout_p",
                    "duration_predictor_hidden_channels",
                    "duration_predictor_kernel_size",
                    "encoder_params",
                    "encoder_type",
                    "hidden_channels",
                    "length_scale",
                    "max_duration",
                    "num_chars",
                    "num_speakers",
                    "out_channels",
                    "pitch_embedding_kernel_size",
                    "pitch_predictor_dropout_p",
                    "pitch_predictor_hidden_channels",
                    "pitch_predictor_kernel_size",
                    "poisitonal_encoding_use_scale",
                    "positional_encoding",
                    "use_aligner",
                    "use_d_vector",
                ],
            )?;
            let model = FastPitchConfig::from_json_value(&root).map_err(anyhow::Error::new)?;
            let audio = AudioFeatureConfig::from_json5_str(&source)?;
            let tokenizer: PhonemeTokenizerConfig =
                json5::from_str(&source).context("invalid FastPitch tokenizer config")?;
            let projector = PhonemeVocabularyProjector::from_config(tokenizer.clone())?;
            ensure!(
                projector.vocabulary().len() == model.num_chars,
                "FastPitch symbol count {} does not match num_chars {}",
                projector.vocabulary().len(),
                model.num_chars
            );
            let parameters = json!({
                "model": model,
                "audio": audio,
                "tokenizer": tokenizer,
            });
            ParsedConfig::FastPitch {
                model,
                audio,
                parameters,
                symbols: projector.vocabulary().iter().map(char::to_string).collect(),
            }
        }
        ModelPackageArchitecture::SpeedySpeech => {
            reject_model_args(
                object,
                &[
                    "d_vector_dim",
                    "decoder_params",
                    "decoder_type",
                    "detach_duration_predictor",
                    "duration_predictor_dropout_p",
                    "duration_predictor_hidden_channels",
                    "duration_predictor_kernel_size",
                    "encoder_params",
                    "encoder_type",
                    "hidden_channels",
                    "length_scale",
                    "max_duration",
                    "num_chars",
                    "num_speakers",
                    "out_channels",
                    "pitch_embedding_kernel_size",
                    "pitch_predictor_dropout_p",
                    "pitch_predictor_hidden_channels",
                    "pitch_predictor_kernel_size",
                    "poisitonal_encoding_use_scale",
                    "positional_encoding",
                    "use_aligner",
                    "use_d_vector",
                    "use_pitch",
                ],
            )?;
            let model = SpeedySpeechConfig::from_json_value(&root).map_err(anyhow::Error::new)?;
            let audio = AudioFeatureConfig::from_json5_str(&source)?;
            let tokenizer: PhonemeTokenizerConfig =
                json5::from_str(&source).context("invalid SpeedySpeech tokenizer config")?;
            let projector = PhonemeVocabularyProjector::from_config(tokenizer.clone())?;
            ensure!(
                projector.vocabulary().len() == model.num_chars,
                "SpeedySpeech symbol count {} does not match num_chars {}",
                projector.vocabulary().len(),
                model.num_chars
            );
            let parameters = json!({
                "model": model,
                "audio": audio,
                "tokenizer": tokenizer,
            });
            ParsedConfig::Speedy {
                model,
                audio,
                parameters,
                symbols: projector.vocabulary().iter().map(char::to_string).collect(),
            }
        }
        ModelPackageArchitecture::HifiGan => {
            let params = object
                .get("generator_model_params")
                .and_then(Value::as_object)
                .context("HiFi-GAN config requires generator_model_params")?;
            reject_unknown_keys(
                params,
                &[
                    "resblock_dilation_sizes",
                    "resblock_kernel_sizes",
                    "resblock_type",
                    "upsample_factors",
                    "upsample_initial_channel",
                    "upsample_kernel_sizes",
                ],
                "generator_model_params",
            )?;
            ensure!(
                !object
                    .get("use_pqmf")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                "unsupported Coqui config field `use_pqmf=true`"
            );
            let model = HifiganBundleConfig::from_json5_str(&source)?;
            let parameters = serde_json::to_value(&model)?;
            ParsedConfig::HifiGan { model, parameters }
        }
        ModelPackageArchitecture::Vits => {
            reject_model_args(
                object,
                &[
                    "condition_dp_on_speaker",
                    "d_vector_dim",
                    "d_vector_file",
                    "detach_dp_input",
                    "dilation_rate_flow",
                    "dilation_rate_posterior_encoder",
                    "dropout_p_duration_predictor",
                    "dropout_p_text_encoder",
                    "embedded_language_dim",
                    "freeze_DP",
                    "freeze_PE",
                    "freeze_encoder",
                    "freeze_flow_decoder",
                    "freeze_waveform_decoder",
                    "hidden_channels",
                    "hidden_channels_ffn_text_encoder",
                    "inference_noise_scale",
                    "inference_noise_scale_dp",
                    "init_discriminator",
                    "kernel_size_flow",
                    "kernel_size_posterior_encoder",
                    "kernel_size_text_encoder",
                    "language_ids_file",
                    "length_scale",
                    "max_inference_len",
                    "noise_scale",
                    "noise_scale_dp",
                    "num_chars",
                    "num_heads_text_encoder",
                    "num_languages",
                    "num_layers_flow",
                    "num_layers_posterior_encoder",
                    "num_layers_text_encoder",
                    "num_speakers",
                    "out_channels",
                    "resblock_dilation_sizes_decoder",
                    "resblock_kernel_sizes_decoder",
                    "resblock_type_decoder",
                    "speaker_embedding_channels",
                    "speaker_encoder_config_path",
                    "speaker_encoder_model_path",
                    "speakers_file",
                    "spec_segment_size",
                    "upsample_initial_channel_decoder",
                    "upsample_kernel_sizes_decoder",
                    "upsample_rates_decoder",
                    "use_d_vector_file",
                    "use_language_embedding",
                    "use_sdp",
                    "use_speaker_embedding",
                    "use_speaker_encoder_as_loss",
                    "use_spectral_norm_disriminator",
                ],
            )?;
            let imported = ImportedVitsConfig::from_json5_str(&source)?;
            let symbols = imported.vocabulary();
            let inference = imported.inference_config();
            let parameters = json!({
                "model": inference,
                "tokenizer": imported,
            });
            ParsedConfig::Vits {
                inference,
                parameters,
                symbols,
            }
        }
        ModelPackageArchitecture::SpeakerEncoder => {
            let model = SpeakerEncoderPackageConfig::from_root(&root)?;
            let parameters = serde_json::to_value(&model)?;
            ParsedConfig::SpeakerEncoder { model, parameters }
        }
    };
    let ignored = ignored_training_fields(object, architecture);
    Ok((root, parsed, ignored))
}

fn detect_architecture(object: &Map<String, Value>) -> Result<ModelPackageArchitecture> {
    if object
        .get("generator_model")
        .and_then(Value::as_str)
        .is_some_and(|name| name == "hifigan_generator")
    {
        return Ok(ModelPackageArchitecture::HifiGan);
    }
    match object.get("model").and_then(Value::as_str) {
        Some("fast_pitch") => Ok(ModelPackageArchitecture::FastPitch),
        Some("speedy_speech") => Ok(ModelPackageArchitecture::SpeedySpeech),
        Some(value) if value.eq_ignore_ascii_case("vits") => Ok(ModelPackageArchitecture::Vits),
        Some("speaker_encoder") | Some("speaker-encoder") => {
            Ok(ModelPackageArchitecture::SpeakerEncoder)
        }
        Some(other) => bail!("unsupported Coqui model architecture `{other}`"),
        None if object.contains_key("model_params") => Ok(ModelPackageArchitecture::SpeakerEncoder),
        None => bail!("unable to determine Coqui model architecture"),
    }
}

fn reject_model_args(object: &Map<String, Value>, allowed: &[&str]) -> Result<()> {
    let args = object
        .get("model_args")
        .and_then(Value::as_object)
        .context("Coqui config requires model_args")?;
    reject_unknown_keys(args, allowed, "model_args")
}

fn reject_unknown_keys(object: &Map<String, Value>, allowed: &[&str], prefix: &str) -> Result<()> {
    let allowed = allowed.iter().copied().collect::<BTreeSet<_>>();
    let unsupported = object
        .keys()
        .filter(|key| !allowed.contains(key.as_str()))
        .map(|key| format!("{prefix}.{key}"))
        .collect::<Vec<_>>();
    ensure!(
        unsupported.is_empty(),
        "unsupported Coqui config field(s): {}",
        unsupported.join(", ")
    );
    Ok(())
}

fn ignored_training_fields(
    object: &Map<String, Value>,
    architecture: ModelPackageArchitecture,
) -> Vec<String> {
    let retained = match architecture {
        ModelPackageArchitecture::FastPitch
        | ModelPackageArchitecture::SpeedySpeech
        | ModelPackageArchitecture::Vits => [
            "model",
            "model_args",
            "audio",
            "characters",
            "use_phonemes",
            "phoneme_language",
            "phonemizer",
            "text_cleaner",
            "add_blank",
            "enable_eos_bos_chars",
            "speakers_file",
            "language_ids_file",
            "num_speakers",
            "use_speaker_embedding",
            "use_language_embedding",
            "speaker_embedding_channels",
        ]
        .as_slice(),
        ModelPackageArchitecture::HifiGan => [
            "generator_model",
            "generator_model_params",
            "audio",
            "use_pqmf",
        ]
        .as_slice(),
        ModelPackageArchitecture::SpeakerEncoder => {
            ["model", "model_params", "model_args", "audio"].as_slice()
        }
    };
    let retained = retained.iter().copied().collect::<BTreeSet<_>>();
    let mut ignored = object
        .keys()
        .filter(|key| !retained.contains(key.as_str()))
        .map(|key| key.to_string())
        .collect::<Vec<_>>();
    match architecture {
        ModelPackageArchitecture::FastPitch => ignored.extend(
            [
                "model_args.detach_duration_predictor",
                "model_args.poisitonal_encoding_use_scale",
            ]
            .into_iter()
            .filter(|path| json_path_exists(object, path))
            .map(str::to_string),
        ),
        ModelPackageArchitecture::SpeedySpeech => ignored.extend(
            [
                "model_args.detach_duration_predictor",
                "model_args.pitch_embedding_kernel_size",
                "model_args.pitch_predictor_dropout_p",
                "model_args.pitch_predictor_hidden_channels",
                "model_args.pitch_predictor_kernel_size",
                "model_args.poisitonal_encoding_use_scale",
            ]
            .into_iter()
            .filter(|path| json_path_exists(object, path))
            .map(str::to_string),
        ),
        ModelPackageArchitecture::Vits => ignored.extend(
            [
                "model_args.d_vector_file",
                "model_args.detach_dp_input",
                "model_args.freeze_DP",
                "model_args.freeze_PE",
                "model_args.freeze_encoder",
                "model_args.freeze_flow_decoder",
                "model_args.freeze_waveform_decoder",
                "model_args.init_discriminator",
                "model_args.language_ids_file",
                "model_args.noise_scale",
                "model_args.noise_scale_dp",
                "model_args.speaker_encoder_config_path",
                "model_args.speaker_encoder_model_path",
                "model_args.speakers_file",
                "model_args.use_speaker_encoder_as_loss",
                "model_args.use_spectral_norm_disriminator",
            ]
            .into_iter()
            .filter(|path| json_path_exists(object, path))
            .map(str::to_string),
        ),
        ModelPackageArchitecture::HifiGan | ModelPackageArchitecture::SpeakerEncoder => {}
    }
    ignored.sort();
    ignored.dedup();
    ignored
}

fn json_path_exists(object: &Map<String, Value>, path: &str) -> bool {
    let mut segments = path.split('.');
    let Some(first) = segments.next() else {
        return false;
    };
    let mut current = object.get(first);
    for segment in segments {
        current = current
            .and_then(Value::as_object)
            .and_then(|value| value.get(segment));
    }
    current.is_some()
}

fn load_speakers(
    options: &CoquiImportOptions,
    parsed: &ParsedConfig,
) -> Result<Vec<PackageSpeaker>> {
    let expected = match parsed {
        ParsedConfig::Vits { inference, .. } if inference.network.use_speaker_embedding => {
            Some(inference.network.num_speakers)
        }
        _ => None,
    };
    let Some(path) = options.speaker_map_path.as_deref() else {
        ensure!(
            expected.is_none_or(|count| count <= 1),
            "speaker_ids.json is required for this {}-speaker model",
            expected.unwrap_or_default()
        );
        return Ok(Vec::new());
    };
    let source = fs::read_to_string(path)
        .with_context(|| format!("failed to read speaker map {}", path.display()))?;
    let map: BTreeMap<String, u32> = serde_json::from_str(&source)
        .with_context(|| format!("invalid speaker map {}", path.display()))?;
    if let Some(expected) = expected {
        SpeakerCatalog::new(map.clone(), expected)?;
    }
    let mut speakers = map
        .into_iter()
        .map(|(name, id)| PackageSpeaker { id, name })
        .collect::<Vec<_>>();
    speakers.sort_by(|left, right| (left.id, &left.name).cmp(&(right.id, &right.name)));
    Ok(speakers)
}

fn load_languages(options: &CoquiImportOptions, root: &Value) -> Result<Vec<PackageLanguage>> {
    let mut languages = if let Some(path) = options.language_map_path.as_deref() {
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read language map {}", path.display()))?;
        let map: BTreeMap<String, u32> = serde_json::from_str(&source)
            .with_context(|| format!("invalid language map {}", path.display()))?;
        map.into_iter()
            .map(|(tag, id)| PackageLanguage { id: Some(id), tag })
            .collect::<Vec<_>>()
    } else {
        root.get("phoneme_language")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(|tag| {
                vec![PackageLanguage {
                    id: None,
                    tag: tag.to_string(),
                }]
            })
            .unwrap_or_default()
    };
    languages.sort_by(|left, right| (left.id, &left.tag).cmp(&(right.id, &right.tag)));
    languages.dedup_by(|left, right| left.id == right.id && left.tag == right.tag);
    Ok(languages)
}

fn checkpoint_reader(options: &CoquiImportOptions) -> Result<PytorchReader> {
    PytorchReader::with_top_level_key(&options.checkpoint_path, &options.checkpoint_key)
        .with_context(|| {
            format!(
                "failed to read tensor-only checkpoint key `{}` from {}",
                options.checkpoint_key,
                options.checkpoint_path.display()
            )
        })
}

fn tensor_metadata(reader: &PytorchReader) -> Result<Vec<TensorMetadata>> {
    ensure!(
        !reader.is_empty(),
        "checkpoint contains no tensors under the selected key"
    );
    ensure!(
        reader.len() <= MAX_TENSOR_COUNT,
        "checkpoint tensor count {} exceeds safety limit {}",
        reader.len(),
        MAX_TENSOR_COUNT
    );
    let mut tensors = reader
        .tensors()
        .iter()
        .map(|(name, tensor)| {
            ensure_safe_tensor_name(name)?;
            let shape = tensor.shape.as_slice().to_vec();
            let elements = checked_elements(&shape)?;
            Ok(TensorMetadata {
                name: name.clone(),
                dtype: dtype_name(tensor.dtype)?.into(),
                shape,
                elements,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(tensors)
}

fn checked_elements(shape: &[usize]) -> Result<usize> {
    let elements = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .context("tensor shape element count overflow")?;
    ensure!(
        elements <= MAX_TENSOR_ELEMENTS,
        "tensor contains {elements} elements, exceeding safety limit {MAX_TENSOR_ELEMENTS}"
    );
    Ok(elements)
}

fn validate_runtime_shapes(
    options: &CoquiImportOptions,
    parsed: &ParsedConfig,
    tensors: &[TensorMetadata],
    checkpoint_path: &Path,
) -> Result<()> {
    type Cpu = NdArray<f32>;
    let device = NdArrayDevice::Cpu;
    match parsed {
        ParsedConfig::FastPitch { model, .. } => {
            model
                .clone()
                .init::<Cpu>(&device)
                .map_err(anyhow::Error::new)?
                .load_checkpoint(checkpoint_path)
                .map_err(anyhow::Error::new)?;
        }
        ParsedConfig::Speedy { model, .. } => {
            model
                .clone()
                .init::<Cpu>(&device)
                .map_err(anyhow::Error::new)?
                .load_checkpoint(checkpoint_path)
                .map_err(anyhow::Error::new)?;
        }
        ParsedConfig::HifiGan { model, .. } => {
            model.load_burn_generator::<Cpu>(checkpoint_path, &device)?;
        }
        ParsedConfig::Vits { .. } => {
            let speakers = options
                .speaker_map_path
                .as_deref()
                .context("VITS shape validation requires speaker_ids.json")?;
            BurnVitsSpeech::<Cpu>::load(&options.config_path, checkpoint_path, speakers, device)
                .context("VITS checkpoint shape validation failed")?;
        }
        ParsedConfig::SpeakerEncoder { model, .. } => {
            validate_speaker_encoder_shapes(model, tensors)?;
        }
    }
    Ok(())
}

fn validate_speaker_encoder_shapes(
    config: &SpeakerEncoderPackageConfig,
    tensors: &[TensorMetadata],
) -> Result<()> {
    let by_name = tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect::<BTreeMap<_, _>>();
    let projection = by_name
        .iter()
        .find(|(name, _)| {
            name.ends_with("linear.weight")
                || name.ends_with("projection.weight")
                || name.ends_with("proj.weight")
        })
        .map(|(_, tensor)| *tensor)
        .context("speaker encoder checkpoint has no projection weight")?;
    ensure!(
        projection.shape
            == [config.projection_dim, config.lstm_dim]
            || projection.shape == [config.lstm_dim, config.projection_dim],
        "speaker encoder projection weight has shape {:?}; expected [{}, {}] (PyTorch) or its runtime transpose",
        projection.shape,
        config.projection_dim,
        config.lstm_dim
    );
    let lstm_weights = tensors
        .iter()
        .filter(|tensor| tensor.name.contains("lstm") && tensor.name.contains("weight"))
        .count();
    ensure!(
        lstm_weights >= config.num_lstm_layers * 2,
        "speaker encoder checkpoint has {lstm_weights} LSTM weights; expected at least {} for {} layers",
        config.num_lstm_layers * 2,
        config.num_lstm_layers
    );
    Ok(())
}

fn convert_checkpoint(
    options: &CoquiImportOptions,
    output: &Path,
    expected: &[TensorMetadata],
    progress: &mut impl FnMut(ModelImportProgress),
) -> Result<()> {
    let reader = checkpoint_reader(options)?;
    let mut tensors = reader.into_tensors().into_iter().collect::<Vec<_>>();
    tensors.sort_by(|left, right| left.0.cmp(&right.0));
    ensure!(
        tensors.len() == expected.len(),
        "checkpoint tensor count changed during conversion"
    );
    let total = tensors.len();
    let mut materialized = Vec::with_capacity(total);
    for (index, (name, snapshot)) in tensors.into_iter().enumerate() {
        if index < 3 || index + 1 == total || (index + 1).is_multiple_of(100) {
            progress(ModelImportProgress::ConvertingTensor {
                current: index + 1,
                total,
                name: name.clone(),
                output: output.to_path_buf(),
            });
        }
        let data = snapshot
            .to_data()
            .with_context(|| format!("failed to materialize tensor `{name}`"))?;
        ensure!(
            data.shape.as_slice() == expected[index].shape,
            "tensor `{name}` changed shape during conversion"
        );
        materialized.push((name, data));
    }

    let part = part_path(output);
    let views = materialized
        .iter()
        .map(|(name, data)| {
            let bytes: &[u8] = data.bytes.as_ref();
            let view = TensorView::new(
                safe_dtype(data.dtype)?,
                data.shape.as_slice().to_vec(),
                bytes,
            )
            .with_context(|| format!("invalid tensor view for `{name}`"))?;
            Ok((name.as_str(), view))
        })
        .collect::<Result<Vec<_>>>()?;
    serialize_to_file(views, None, &part)
        .with_context(|| format!("failed to write SafeTensors {}", part.display()))?;
    File::open(&part)?.sync_all()?;
    fs::rename(&part, output).with_context(|| {
        format!(
            "failed to atomically install SafeTensors {}",
            output.display()
        )
    })?;
    Ok(())
}

fn verify_safetensors(path: &Path, expected: &[TensorMetadata]) -> Result<()> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to verify SafeTensors {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("invalid generated SafeTensors {}", path.display()))?;
    let mut actual = tensors
        .tensors()
        .into_iter()
        .map(|(name, tensor)| (name, tensor.shape().to_vec()))
        .collect::<Vec<_>>();
    actual.sort_by(|left, right| left.0.cmp(&right.0));
    ensure!(
        actual.len() == expected.len(),
        "generated SafeTensors contains {} tensors, expected {}",
        actual.len(),
        expected.len()
    );
    for ((name, shape), expected) in actual.iter().zip(expected) {
        ensure!(
            name == &expected.name && shape == &expected.shape,
            "generated SafeTensors tensor mismatch: found `{name}` {shape:?}, expected `{}` {:?}",
            expected.name,
            expected.shape
        );
    }
    Ok(())
}

fn source_artifacts(options: &CoquiImportOptions) -> Result<Vec<PackageArtifact>> {
    let mut artifacts = vec![
        source_artifact("config", &options.config_path)?,
        source_artifact("checkpoint", &options.checkpoint_path)?,
    ];
    if let Some(path) = &options.speaker_map_path {
        artifacts.push(source_artifact("speakers", path)?);
    }
    if let Some(path) = &options.language_map_path {
        artifacts.push(source_artifact("languages", path)?);
    }
    artifacts.sort_by(|left, right| left.role.cmp(&right.role));
    Ok(artifacts)
}

fn source_artifact(role: &str, path: &Path) -> Result<PackageArtifact> {
    let metadata =
        fs::metadata(path).with_context(|| format!("failed to stat {}", path.display()))?;
    Ok(PackageArtifact {
        role: role.into(),
        filename: path
            .file_name()
            .and_then(|name| name.to_str())
            .context("source artifact filename is not valid UTF-8")?
            .into(),
        size_bytes: metadata.len(),
        sha256: sha256_file(path)?,
    })
}

fn package_file(path: &Path, name: &str) -> Result<PackageFile> {
    let metadata =
        fs::metadata(path).with_context(|| format!("failed to stat {}", path.display()))?;
    Ok(PackageFile {
        path: name.into(),
        size_bytes: metadata.len(),
        sha256: sha256_file(path)?,
    })
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let part = part_path(path);
    let file =
        File::create(&part).with_context(|| format!("failed to create {}", part.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .with_context(|| format!("failed to serialize {}", path.display()))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&part, path)
        .with_context(|| format!("failed to atomically install {}", path.display()))?;
    Ok(())
}

fn part_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".part");
    PathBuf::from(name)
}

fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 128 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn ensure_safe_member_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    ensure!(
        path.components().count() == 1
            && path.file_name().and_then(|value| value.to_str()) == Some(name),
        "unsafe model package member name `{name}`"
    );
    Ok(())
}

fn ensure_safe_tensor_name(name: &str) -> Result<()> {
    ensure!(!name.is_empty(), "checkpoint contains an empty tensor name");
    ensure!(
        !name.starts_with('.') && !name.contains('/') && !name.contains('\\'),
        "unsafe checkpoint tensor name `{name}`"
    );
    ensure!(
        name.bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte)),
        "checkpoint tensor name contains unsupported characters: `{name}`"
    );
    Ok(())
}

fn dtype_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F64 => Ok("f64"),
        DType::F32 | DType::Flex32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        DType::I64 => Ok("i64"),
        DType::I32 => Ok("i32"),
        DType::I16 => Ok("i16"),
        DType::I8 => Ok("i8"),
        DType::U64 => Ok("u64"),
        DType::U32 => Ok("u32"),
        DType::U16 => bail!("u16 tensors are not supported by SafeTensors"),
        DType::U8 => Ok("u8"),
        DType::Bool(_) => Ok("bool"),
        DType::QFloat(_) => bail!("quantized PyTorch tensors are not supported by this importer"),
    }
}

fn safe_dtype(dtype: DType) -> Result<SafeDtype> {
    match dtype {
        DType::F64 => Ok(SafeDtype::F64),
        DType::F32 | DType::Flex32 => Ok(SafeDtype::F32),
        DType::F16 => Ok(SafeDtype::F16),
        DType::BF16 => Ok(SafeDtype::BF16),
        DType::I64 => Ok(SafeDtype::I64),
        DType::I32 => Ok(SafeDtype::I32),
        DType::I16 => Ok(SafeDtype::I16),
        DType::I8 => Ok(SafeDtype::I8),
        DType::U64 => Ok(SafeDtype::U64),
        DType::U32 => Ok(SafeDtype::U32),
        DType::U16 => bail!("u16 tensors are not supported by SafeTensors"),
        DType::U8 => Ok(SafeDtype::U8),
        DType::Bool(_) => Ok(SafeDtype::BOOL),
        DType::QFloat(_) => bail!("quantized PyTorch tensors are not supported by this importer"),
    }
}

/// Verify that a checkpoint is a modern PyTorch ZIP and that its pickle
/// metadata contains only the small callable vocabulary required to rebuild
/// tensor storage references. Parsing remains data-only; none of these
/// callables are imported or executed.
pub fn scan_safe_pytorch_checkpoint(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let file = File::open(path)
        .with_context(|| format!("failed to open PyTorch checkpoint {}", path.display()))?;
    let mut archive = ZipArchive::new(file).with_context(|| {
        format!(
            "unsafe/unsupported checkpoint {}: only modern ZIP-based PyTorch checkpoints are accepted",
            path.display()
        )
    })?;
    let mut pickle_index = None;
    for index in 0..archive.len() {
        let entry = archive.by_index(index)?;
        let name = entry.name();
        ensure!(
            entry.enclosed_name().is_some(),
            "checkpoint contains unsafe archive member `{name}`"
        );
        if name.ends_with("/data.pkl") || name == "data.pkl" {
            ensure!(
                pickle_index.replace(index).is_none(),
                "checkpoint contains multiple data.pkl members"
            );
        }
    }
    let index = pickle_index.context("checkpoint ZIP has no data.pkl tensor index")?;
    let mut entry = archive.by_index(index)?;
    ensure!(
        entry.size() <= MAX_PICKLE_METADATA_BYTES,
        "checkpoint pickle metadata is {} bytes, exceeding safety limit {}",
        entry.size(),
        MAX_PICKLE_METADATA_BYTES
    );
    let mut pickle = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut pickle)?;
    scan_pickle_program(&pickle)
}

fn scan_pickle_program(bytes: &[u8]) -> Result<()> {
    let mut cursor = 0usize;
    let mut protocol = None;
    let mut stopped = false;
    while cursor < bytes.len() {
        let opcode = bytes[cursor];
        cursor += 1;
        match opcode {
            0x80 => {
                let version = take_u8(bytes, &mut cursor, "PROTO")?;
                ensure!(
                    version == 2,
                    "unsupported pickle protocol {version}; Coqui imports require protocol 2 tensor checkpoints"
                );
                protocol = Some(version);
            }
            b'c' => {
                let module = take_line(bytes, &mut cursor, "GLOBAL module")?;
                let name = take_line(bytes, &mut cursor, "GLOBAL name")?;
                ensure!(
                    allowed_pickle_global(module, name),
                    "unsafe/unsupported pickle GLOBAL `{module}.{name}`; arbitrary Python callables are rejected"
                );
            }
            b'X' | b'T' | b'B' => {
                let length = take_u32(bytes, &mut cursor, "length")? as usize;
                take(bytes, &mut cursor, length, "byte/string payload")?;
            }
            b'U' | b'C' => {
                let length = take_u8(bytes, &mut cursor, "short length")? as usize;
                take(bytes, &mut cursor, length, "short byte/string payload")?;
            }
            b'J' => {
                take(bytes, &mut cursor, 4, "BININT")?;
            }
            b'K' | b'q' | b'h' => {
                take(bytes, &mut cursor, 1, "one-byte pickle argument")?;
            }
            b'M' => {
                take(bytes, &mut cursor, 2, "two-byte pickle argument")?;
            }
            b'r' | b'j' => {
                take(bytes, &mut cursor, 4, "four-byte pickle argument")?;
            }
            b'G' => {
                take(bytes, &mut cursor, 8, "BINFLOAT")?;
            }
            0x8a => {
                let length = take_u8(bytes, &mut cursor, "LONG1 length")? as usize;
                take(bytes, &mut cursor, length, "LONG1 payload")?;
            }
            b'I' | b'F' | b'S' | b'V' | b'P' => {
                take_line(bytes, &mut cursor, "line pickle argument")?;
            }
            b'.' => {
                stopped = true;
                break;
            }
            b')' | b'R' | b'(' | b't' | b'Q' | b'N' | 0x85 | 0x86 | 0x87 | 0x88 | 0x89 | b's'
            | b'u' | b'}' | b'd' | b'b' | b']' | b'l' | b'a' | b'e' | 0x94 => {}
            0x81 | 0x82 | 0x83 | 0x84 | 0x91 | 0x92 | 0x93 => {
                bail!(
                    "unsafe/unsupported pickle construction opcode 0x{opcode:02x}; arbitrary objects are rejected"
                )
            }
            other => bail!(
                "unsupported pickle opcode 0x{other:02x} at byte {}",
                cursor - 1
            ),
        }
    }
    ensure!(
        protocol == Some(2),
        "checkpoint pickle has no protocol 2 header"
    );
    ensure!(stopped, "checkpoint pickle has no STOP opcode");
    Ok(())
}

fn allowed_pickle_global(module: &str, name: &str) -> bool {
    matches!(
        (module, name),
        ("collections", "OrderedDict")
            | ("torch._utils", "_rebuild_tensor")
            | ("torch._utils", "_rebuild_tensor_v2")
            | ("torch._utils", "_rebuild_parameter")
            | ("torch", "FloatStorage")
            | ("torch", "DoubleStorage")
            | ("torch", "HalfStorage")
            | ("torch", "BFloat16Storage")
            | ("torch", "LongStorage")
            | ("torch", "IntStorage")
            | ("torch", "ShortStorage")
            | ("torch", "CharStorage")
            | ("torch", "ByteStorage")
            | ("torch", "BoolStorage")
    )
}

fn take<'a>(bytes: &'a [u8], cursor: &mut usize, length: usize, label: &str) -> Result<&'a [u8]> {
    let end = cursor
        .checked_add(length)
        .context("pickle cursor overflow")?;
    let value = bytes
        .get(*cursor..end)
        .with_context(|| format!("truncated pickle {label}"))?;
    *cursor = end;
    Ok(value)
}

fn take_u8(bytes: &[u8], cursor: &mut usize, label: &str) -> Result<u8> {
    Ok(*take(bytes, cursor, 1, label)?
        .first()
        .context("missing pickle byte")?)
}

fn take_u32(bytes: &[u8], cursor: &mut usize, label: &str) -> Result<u32> {
    let value: [u8; 4] = take(bytes, cursor, 4, label)?
        .try_into()
        .context("invalid pickle u32")?;
    Ok(u32::from_le_bytes(value))
}

fn take_line<'a>(bytes: &'a [u8], cursor: &mut usize, label: &str) -> Result<&'a str> {
    let rest = bytes
        .get(*cursor..)
        .with_context(|| format!("truncated pickle {label}"))?;
    let length = rest
        .iter()
        .position(|byte| *byte == b'\n')
        .with_context(|| format!("unterminated pickle {label}"))?;
    let line = std::str::from_utf8(&rest[..length])
        .with_context(|| format!("pickle {label} is not UTF-8"))?;
    *cursor += length + 1;
    Ok(line.trim_end_matches('\r'))
}

fn optional_string(object: &Map<String, Value>, aliases: &[&str]) -> Option<String> {
    aliases
        .iter()
        .find_map(|key| object.get(*key).and_then(Value::as_str))
        .map(str::to_string)
}

fn optional_u64(object: &Map<String, Value>, aliases: &[&str]) -> Option<u64> {
    aliases
        .iter()
        .find_map(|key| object.get(*key).and_then(Value::as_u64))
}

fn required_usize_alias(object: &Map<String, Value>, aliases: &[&str]) -> Result<usize> {
    optional_u64(object, aliases)
        .with_context(|| format!("missing numeric config field `{}`", aliases.join("` or `")))?
        .try_into()
        .with_context(|| {
            format!(
                "config field `{}` does not fit usize",
                aliases.join("` or `")
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_pickle_scanner_accepts_tensor_rebuild_vocabulary() {
        let pickle = b"\x80\x02ccollections\nOrderedDict\nctorch._utils\n_rebuild_tensor_v2\nctorch\nFloatStorage\n.";
        scan_pickle_program(pickle).expect("tensor-only pickle vocabulary");
    }

    #[test]
    fn safe_pickle_scanner_rejects_arbitrary_global() {
        let pickle = b"\x80\x02cos\nsystem\n.";
        let error = scan_pickle_program(pickle).expect_err("os.system must be rejected");
        assert!(error.to_string().contains("os.system"));
        assert!(error.to_string().contains("arbitrary Python"));
    }

    #[test]
    fn safe_pickle_scanner_rejects_stack_global() {
        let pickle = b"\x80\x02\x93.";
        let error = scan_pickle_program(pickle).expect_err("STACK_GLOBAL must be rejected");
        assert!(error.to_string().contains("opcode 0x93"));
    }

    #[test]
    fn speaker_encoder_config_rejects_unknown_model_field() {
        let root = json!({
            "model": "speaker_encoder",
            "model_params": {
                "model_name": "lstm",
                "input_dim": 80,
                "proj_dim": 256,
                "lstm_dim": 768,
                "num_lstm_layers": 3,
                "python_callback": "oops"
            }
        });
        let error = SpeakerEncoderPackageConfig::from_root(&root)
            .expect_err("unknown inference field must fail");
        assert!(error.to_string().contains("model_params.python_callback"));
    }

    #[test]
    fn migrates_v0_manifest_without_losing_metadata() {
        let value = json!({
            "version": 0,
            "architecture": "speaker_encoder",
            "config": "model.json",
            "weights": "model.safetensors",
            "tensor_index": "tensors.json",
            "tensor_count": 1,
            "audio": null,
            "speakers": [],
            "languages": [],
            "symbols": [],
            "license": {"expression": "Apache-2.0"},
            "provenance": {
                "source": "fixture",
                "source_format": "coqui-pytorch-zip",
                "importer": "tongues-tts",
                "importer_version": "0.0.0",
                "coqui_version": null
            },
            "source_artifacts": [],
            "files": [],
            "ignored_training_fields": []
        });
        let migrated = migrate_model_package_manifest(value).expect("migration");
        assert_eq!(migrated["schema_version"], 1);
        assert_eq!(migrated["package_format"], MODEL_PACKAGE_FORMAT);
        assert_eq!(migrated["runtime"], "backend-neutral");
        let manifest: ModelPackageManifest =
            serde_json::from_value(migrated).expect("current manifest");
        manifest.validate().expect("valid manifest");
    }

    #[test]
    fn tensor_names_cannot_escape_the_package_contract() {
        assert!(ensure_safe_tensor_name("encoder.layers.0.weight").is_ok());
        assert!(ensure_safe_tensor_name("../weight").is_err());
        assert!(ensure_safe_tensor_name("encoder/weight").is_err());
    }

    #[test]
    fn speaker_encoder_shapes_are_checked() {
        let config = SpeakerEncoderPackageConfig {
            model_name: "lstm".into(),
            input_dim: 80,
            projection_dim: 256,
            lstm_dim: 768,
            num_lstm_layers: 2,
            use_lstm_with_projection: true,
            sample_rate_hz: 16_000,
            fft_size: Some(512),
            window_size: Some(400),
            hop_size: Some(160),
        };
        let tensor = |name: &str, shape: Vec<usize>| TensorMetadata {
            name: name.into(),
            dtype: "f32".into(),
            elements: shape.iter().product(),
            shape,
        };
        let tensors = vec![
            tensor("lstm.weight_ih_l0", vec![3072, 80]),
            tensor("lstm.weight_hh_l0", vec![3072, 768]),
            tensor("lstm.weight_ih_l1", vec![3072, 768]),
            tensor("lstm.weight_hh_l1", vec![3072, 768]),
            tensor("projection.weight", vec![256, 768]),
        ];
        validate_speaker_encoder_shapes(&config, &tensors).expect("valid shapes");
    }

    fn fixture_options(config_env: &str, checkpoint_env: &str) -> Option<CoquiImportOptions> {
        let config = std::env::var_os(config_env)?;
        let checkpoint = std::env::var_os(checkpoint_env)?;
        Some(CoquiImportOptions::new(
            config,
            checkpoint,
            "target/coqui-import-fixture",
            "Apache-2.0",
            "fixture",
        ))
    }

    #[test]
    fn published_speedy_speech_fixture_uses_common_importer_when_available() {
        let Some(options) = fixture_options(
            "TONGUES_TEST_COQUI_SPEEDY_CONFIG",
            "TONGUES_TEST_COQUI_SPEEDY_MODEL",
        ) else {
            return;
        };
        let inspection = inspect_coqui_import(&options).expect("SpeedySpeech import inspection");
        assert_eq!(
            inspection.architecture,
            ModelPackageArchitecture::SpeedySpeech
        );
        assert!(!inspection.tensors.is_empty());
    }

    #[test]
    fn published_fast_pitch_fixture_uses_common_importer_when_available() {
        let Some(options) = fixture_options(
            "TONGUES_TEST_COQUI_FASTPITCH_CONFIG",
            "TONGUES_TEST_COQUI_FASTPITCH_MODEL",
        ) else {
            return;
        };
        let inspection = inspect_coqui_import(&options).expect("FastPitch import inspection");
        assert_eq!(inspection.architecture, ModelPackageArchitecture::FastPitch);
        assert!(!inspection.tensors.is_empty());
    }

    #[test]
    fn published_hifigan_fixture_uses_common_importer_when_available() {
        let Some(options) = fixture_options(
            "TONGUES_TEST_COQUI_HIFIGAN_CONFIG",
            "TONGUES_TEST_COQUI_HIFIGAN_MODEL",
        ) else {
            return;
        };
        let inspection = inspect_coqui_import(&options).expect("HiFi-GAN import inspection");
        assert_eq!(inspection.architecture, ModelPackageArchitecture::HifiGan);
        assert!(!inspection.tensors.is_empty());
    }

    #[test]
    fn published_vits_fixture_uses_common_importer_when_available() {
        let Some(mut options) = fixture_options(
            "TONGUES_TEST_COQUI_VITS_CONFIG",
            "TONGUES_TEST_COQUI_VITS_CHECKPOINT",
        ) else {
            return;
        };
        let Some(speakers) = std::env::var_os("TONGUES_TEST_COQUI_VITS_SPEAKERS") else {
            return;
        };
        options.speaker_map_path = Some(speakers.into());
        let inspection = inspect_coqui_import(&options).expect("VITS import inspection");
        assert_eq!(inspection.architecture, ModelPackageArchitecture::Vits);
        assert!(!inspection.speakers.is_empty());
    }

    #[test]
    fn speaker_encoder_fixture_uses_common_importer_when_available() {
        let Some(options) = fixture_options(
            "TONGUES_TEST_COQUI_SPEAKER_ENCODER_CONFIG",
            "TONGUES_TEST_COQUI_SPEAKER_ENCODER_CHECKPOINT",
        ) else {
            return;
        };
        let inspection = inspect_coqui_import(&options).expect("speaker encoder import inspection");
        assert_eq!(
            inspection.architecture,
            ModelPackageArchitecture::SpeakerEncoder
        );
        assert!(!inspection.tensors.is_empty());
    }
}
