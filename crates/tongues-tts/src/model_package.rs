//! Safe, deterministic import of legacy Coqui model artifacts.
//!
//! The compatibility boundary deliberately ends at import time. PyTorch ZIP
//! checkpoints are parsed by Rust code after a restrictive pickle opcode and
//! callable scan, then rewritten as SafeTensors. Runtime packages never need
//! Python, pickle, or Coqui.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::ops::Deref;
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

use crate::speaker_encoder::CoquiResNetSpeakerEncoder;
use crate::vits_config::ImportedVitsConfig;
use crate::{
    AlignTtsConfig, AudioFeatureConfig, BurnVitsSpeech, DVectorCatalog, DelightfulTtsConfig,
    FastPitchConfig, FastSpeechConfig, FastSpeechVariant, GlowTts, GlowTtsInferenceConfig,
    HifiganBundleConfig, LanguageCatalog, MelganBundleConfig, MelganVariant,
    PhonemeTokenizerConfig, PhonemeVocabularyProjector, SpeakerCatalog, SpeedySpeechConfig,
    StochasticGlowTts, TacotronArchitecture, TacotronGraphemeProjector, TacotronInferenceConfig,
    VitsInferenceConfig, XttsV2Config, COQUI_RESNET_SPEAKER_EMBEDDING_SPACE,
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
    AlignTts,
    FastPitch,
    FastSpeech,
    FastSpeech2,
    SpeedySpeech,
    Tacotron,
    Tacotron2,
    DelightfulTts,
    HifiGan,
    MelGan,
    MultibandMelGan,
    GlowTts,
    Vits,
    XttsV2,
    SpeakerEncoder,
}

impl ModelPackageArchitecture {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AlignTts => "align_tts",
            Self::FastPitch => "fast_pitch",
            Self::FastSpeech => "fast_speech",
            Self::FastSpeech2 => "fast_speech_2",
            Self::SpeedySpeech => "speedy_speech",
            Self::Tacotron => "tacotron",
            Self::Tacotron2 => "tacotron2",
            Self::DelightfulTts => "delightful_tts",
            Self::HifiGan => "hifi_gan",
            Self::MelGan => "mel_gan",
            Self::MultibandMelGan => "multiband_mel_gan",
            Self::GlowTts => "glow_tts",
            Self::Vits => "vits",
            Self::XttsV2 => "xtts_v2",
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
    pub tokenizer_path: Option<PathBuf>,
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
            tokenizer_path: None,
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
    AlignTts {
        model: AlignTtsConfig,
        audio: AudioFeatureConfig,
        parameters: Value,
        symbols: Vec<String>,
    },
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
    Tacotron {
        inference: TacotronInferenceConfig,
        audio: AudioFeatureConfig,
        parameters: Value,
        symbols: Vec<String>,
    },
    FastSpeech {
        model: FastSpeechConfig,
        audio: AudioFeatureConfig,
        parameters: Value,
        symbols: Vec<String>,
    },
    DelightfulTts {
        model: DelightfulTtsConfig,
        parameters: Value,
        symbols: Vec<String>,
        padding_id: usize,
    },
    HifiGan {
        model: HifiganBundleConfig,
        parameters: Value,
    },
    MelGan {
        model: MelganBundleConfig,
        parameters: Value,
    },
    GlowTts {
        inference: GlowTtsInferenceConfig,
        parameters: Value,
        symbols: Vec<String>,
    },
    Vits {
        inference: VitsInferenceConfig,
        parameters: Value,
        symbols: Vec<String>,
    },
    Xtts {
        model: XttsV2Config,
        parameters: Value,
    },
    SpeakerEncoder {
        model: SpeakerEncoderPackageConfig,
        parameters: Value,
    },
}

impl ParsedConfig {
    fn architecture(&self) -> ModelPackageArchitecture {
        match self {
            Self::AlignTts { .. } => ModelPackageArchitecture::AlignTts,
            Self::FastPitch { .. } => ModelPackageArchitecture::FastPitch,
            Self::Speedy { .. } => ModelPackageArchitecture::SpeedySpeech,
            Self::Tacotron { inference, .. } => match inference.architecture {
                TacotronArchitecture::Tacotron => ModelPackageArchitecture::Tacotron,
                TacotronArchitecture::Tacotron2 => ModelPackageArchitecture::Tacotron2,
            },
            Self::FastSpeech { model, .. } => match model.variant {
                FastSpeechVariant::FastSpeech => ModelPackageArchitecture::FastSpeech,
                FastSpeechVariant::FastSpeech2 => ModelPackageArchitecture::FastSpeech2,
            },
            Self::DelightfulTts { .. } => ModelPackageArchitecture::DelightfulTts,
            Self::HifiGan { .. } => ModelPackageArchitecture::HifiGan,
            Self::MelGan { model, .. } => match model.variant().expect("validated MelGAN config") {
                MelganVariant::Melgan => ModelPackageArchitecture::MelGan,
                MelganVariant::Multiband => ModelPackageArchitecture::MultibandMelGan,
            },
            Self::GlowTts { .. } => ModelPackageArchitecture::GlowTts,
            Self::Vits { .. } => ModelPackageArchitecture::Vits,
            Self::Xtts { .. } => ModelPackageArchitecture::XttsV2,
            Self::SpeakerEncoder { .. } => ModelPackageArchitecture::SpeakerEncoder,
        }
    }

    fn parameters(&self) -> &Value {
        match self {
            Self::AlignTts { parameters, .. }
            | Self::FastPitch { parameters, .. }
            | Self::Speedy { parameters, .. }
            | Self::Tacotron { parameters, .. }
            | Self::FastSpeech { parameters, .. }
            | Self::DelightfulTts { parameters, .. }
            | Self::HifiGan { parameters, .. }
            | Self::MelGan { parameters, .. }
            | Self::GlowTts { parameters, .. }
            | Self::Vits { parameters, .. }
            | Self::Xtts { parameters, .. }
            | Self::SpeakerEncoder { parameters, .. } => parameters,
        }
    }

    fn audio(&self) -> Option<PackageAudio> {
        let audio = match self {
            Self::AlignTts { audio, .. }
            | Self::FastPitch { audio, .. }
            | Self::Speedy { audio, .. }
            | Self::Tacotron { audio, .. }
            | Self::FastSpeech { audio, .. } => audio,
            Self::DelightfulTts { model, .. } => {
                return Some(PackageAudio {
                    sample_rate_hz: model.audio.sample_rate,
                    fft_size: Some(model.audio.fft_size),
                    window_size: Some(model.audio.win_length),
                    hop_size: Some(model.audio.hop_length),
                    mel_bins: Some(model.audio.num_mels),
                });
            }
            Self::HifiGan { model, .. } => &model.audio,
            Self::MelGan { model, .. } => &model.audio,
            Self::GlowTts { inference, .. } => &inference.audio,
            Self::Vits { inference, .. } => &inference.audio,
            Self::Xtts { model, .. } => {
                return Some(PackageAudio {
                    sample_rate_hz: model.audio.output_sample_rate,
                    fft_size: Some(2_048),
                    window_size: Some(1_024),
                    hop_size: Some(256),
                    mel_bins: Some(crate::XTTS_V2_CONDITIONING_MEL_BINS),
                });
            }
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
            Self::AlignTts { symbols, .. }
            | Self::FastPitch { symbols, .. }
            | Self::Speedy { symbols, .. }
            | Self::Tacotron { symbols, .. }
            | Self::FastSpeech { symbols, .. }
            | Self::DelightfulTts { symbols, .. }
            | Self::GlowTts { symbols, .. }
            | Self::Vits { symbols, .. } => symbols.clone(),
            Self::HifiGan { .. }
            | Self::MelGan { .. }
            | Self::Xtts { .. }
            | Self::SpeakerEncoder { .. } => Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerEncoderPackageConfig {
    pub model_name: String,
    pub input_dim: usize,
    pub projection_dim: usize,
    pub lstm_dim: Option<usize>,
    pub num_lstm_layers: Option<usize>,
    pub use_lstm_with_projection: bool,
    pub use_torch_spec: bool,
    pub log_input: bool,
    pub encoder_type: String,
    pub layers: Vec<usize>,
    pub num_filters: Vec<usize>,
    pub sample_rate_hz: u32,
    pub fft_size: Option<usize>,
    pub window_size: Option<usize>,
    pub hop_size: Option<usize>,
}

impl SpeakerEncoderPackageConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read speaker encoder config {}", path.display()))?;
        let root: Value = json5::from_str(&source).with_context(|| {
            format!("failed to parse speaker encoder config {}", path.display())
        })?;
        Self::from_root(&root)
            .with_context(|| format!("invalid speaker encoder config {}", path.display()))
    }

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
                "use_torch_spec",
                "log_input",
                "encoder_type",
                "layers",
                "num_filters",
            ],
            "model_params",
        )?;
        let audio = root.get("audio").and_then(Value::as_object);
        let config = Self {
            model_name: optional_string(params, &["model_name"]).unwrap_or_else(|| "lstm".into()),
            input_dim: required_usize_alias(params, &["input_dim", "num_mels"])?,
            projection_dim: required_usize_alias(params, &["proj_dim", "projection_dim"])?,
            lstm_dim: optional_u64(params, &["lstm_dim", "hidden_dim"])
                .map(usize::try_from)
                .transpose()
                .context("speaker encoder LSTM dimension does not fit usize")?,
            num_lstm_layers: optional_u64(params, &["num_lstm_layers", "num_layers"])
                .map(usize::try_from)
                .transpose()
                .context("speaker encoder LSTM layer count does not fit usize")?,
            use_lstm_with_projection: params
                .get("use_lstm_with_projection")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            use_torch_spec: params
                .get("use_torch_spec")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            log_input: params
                .get("log_input")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            encoder_type: optional_string(params, &["encoder_type"])
                .unwrap_or_else(|| "ASP".into()),
            layers: optional_usize_array(params, "layers")?.unwrap_or_else(|| vec![3, 4, 6, 3]),
            num_filters: optional_usize_array(params, "num_filters")?
                .unwrap_or_else(|| vec![32, 64, 128, 256]),
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
            matches!(
                config.model_name.to_ascii_lowercase().as_str(),
                "lstm" | "speaker_encoder" | "resnet"
            ),
            "unsupported speaker encoder model_name `{}`",
            config.model_name
        );
        ensure!(
            config.input_dim > 0 && config.projection_dim > 0,
            "speaker encoder dimensions must be positive"
        );
        if config.model_name.eq_ignore_ascii_case("resnet") {
            ensure!(
                config.encoder_type.eq_ignore_ascii_case("asp"),
                "unsupported ResNet speaker encoder pooling `{}`; expected ASP",
                config.encoder_type
            );
            ensure!(
                config.layers.len() == 4
                    && config.num_filters.len() == 4
                    && config.layers.iter().all(|value| *value > 0)
                    && config.num_filters.iter().all(|value| *value > 0),
                "ResNet speaker encoder requires four positive layer and filter counts"
            );
        } else {
            ensure!(
                config.lstm_dim.is_some_and(|value| value > 0)
                    && config.num_lstm_layers.is_some_and(|value| value > 0),
                "LSTM speaker encoder requires positive `lstm_dim` and `num_lstm_layers`"
            );
        }
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
    let (root, parsed, ignored_training_fields) =
        parse_config(&options.config_path, options.tokenizer_path.as_deref())?;
    progress(ModelImportProgress::ScanningCheckpoint {
        path: options.checkpoint_path.clone(),
    });
    if matches!(parsed, ParsedConfig::MelGan { .. }) {
        scan_safe_melgan_checkpoint(options)?;
    } else if matches!(parsed, ParsedConfig::Xtts { .. }) {
        scan_safe_xtts_pytorch_checkpoint(&options.checkpoint_path)?;
    } else {
        scan_safe_pytorch_checkpoint(&options.checkpoint_path)?;
    }
    let reader = checkpoint_reader(options)?;
    let tensors = tensor_metadata(&reader)?;
    let speakers = load_speakers(options, &parsed)?;
    let languages = load_languages(options, &root, &parsed)?;
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
    let (_, parsed, _) = parse_config(&options.config_path, options.tokenizer_path.as_deref())?;
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

    let mut package_members = vec![
        MODEL_PACKAGE_CONFIG,
        MODEL_PACKAGE_WEIGHTS,
        MODEL_PACKAGE_TENSORS,
    ];
    if matches!(parsed, ParsedConfig::Xtts { .. }) {
        let tokenizer = options
            .tokenizer_path
            .as_deref()
            .context("XTTS import requires vocab.json")?;
        let destination = options.output_dir.join("vocab.json");
        progress(ModelImportProgress::WritingMetadata {
            path: destination.clone(),
        });
        copy_file_atomic(tokenizer, &destination)?;
        package_members.push("vocab.json");
    }
    let files = package_members
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
            source_format: checkpoint_source_format(&options.checkpoint_path)?.into(),
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

fn checkpoint_source_format(path: &Path) -> Result<&'static str> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to inspect checkpoint format {}", path.display()))?;
    let mut magic = [0u8; 4];
    let read = file.read(&mut magic)?;
    if read == magic.len() && magic == *b"PK\x03\x04" {
        return Ok("coqui-pytorch-zip");
    }
    let descript_layout = PytorchReader::new(path).is_ok_and(|reader| {
        reader.tensors().keys().any(|name| {
            name.strip_prefix("model.")
                .and_then(|suffix| suffix.split('.').next())
                .is_some_and(|index| index.parse::<usize>().is_ok())
        })
    });
    Ok(if descript_layout {
        "descript-pytorch-legacy"
    } else {
        "coqui-pytorch-legacy"
    })
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

fn parse_config(
    path: &Path,
    tokenizer_path: Option<&Path>,
) -> Result<(Value, ParsedConfig, Vec<String>)> {
    let source = fs::read_to_string(path)
        .with_context(|| format!("failed to read Coqui config {}", path.display()))?;
    let root: Value = json5::from_str(&source)
        .with_context(|| format!("invalid Coqui JSON/JSON5 config {}", path.display()))?;
    let object = root
        .as_object()
        .context("Coqui config root must be an object")?;
    let architecture = detect_architecture(object)?;
    let parsed = match architecture {
        ModelPackageArchitecture::AlignTts => {
            reject_model_args(
                object,
                &[
                    "d_vector_dim",
                    "decoder_params",
                    "decoder_type",
                    "encoder_params",
                    "encoder_type",
                    "hidden_channels",
                    "hidden_channels_dp",
                    "length_scale",
                    "num_chars",
                    "num_speakers",
                    "out_channels",
                    "use_d_vector_file",
                    "use_speaker_embedding",
                ],
            )?;
            let model = AlignTtsConfig::from_json_value(&root).map_err(anyhow::Error::new)?;
            let audio = AudioFeatureConfig::from_json5_str(&source)?;
            let tokenizer: PhonemeTokenizerConfig =
                json5::from_str(&source).context("invalid Align-TTS tokenizer config")?;
            let projector = PhonemeVocabularyProjector::from_config(tokenizer.clone())?;
            ensure!(
                projector.vocabulary().len() == model.num_chars,
                "Align-TTS symbol count {} does not match num_chars {}",
                projector.vocabulary().len(),
                model.num_chars
            );
            let parameters = json!({ "model": model, "audio": audio, "tokenizer": tokenizer });
            ParsedConfig::AlignTts {
                model,
                audio,
                parameters,
                symbols: projector.vocabulary().iter().map(char::to_string).collect(),
            }
        }
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
        ModelPackageArchitecture::FastSpeech | ModelPackageArchitecture::FastSpeech2 => {
            reject_model_args(
                object,
                &[
                    "d_vector_dim",
                    "d_vector_file",
                    "decoder_params",
                    "decoder_type",
                    "detach_duration_predictor",
                    "duration_predictor_dropout_p",
                    "duration_predictor_hidden_channels",
                    "duration_predictor_kernel_size",
                    "encoder_params",
                    "encoder_type",
                    "energy_embedding_kernel_size",
                    "energy_predictor_dropout_p",
                    "energy_predictor_hidden_channels",
                    "energy_predictor_kernel_size",
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
                    "speakers_file",
                    "use_aligner",
                    "use_d_vector_file",
                    "use_energy",
                    "use_pitch",
                    "use_speaker_embedding",
                ],
            )?;
            let model = FastSpeechConfig::from_json_value(&root).map_err(anyhow::Error::new)?;
            ensure!(
                matches!(
                    (architecture, model.variant),
                    (
                        ModelPackageArchitecture::FastSpeech,
                        FastSpeechVariant::FastSpeech
                    ) | (
                        ModelPackageArchitecture::FastSpeech2,
                        FastSpeechVariant::FastSpeech2
                    )
                ),
                "detected FastSpeech architecture disagrees with model config"
            );
            let audio = AudioFeatureConfig::from_json5_str(&source)?;
            let tokenizer: PhonemeTokenizerConfig =
                json5::from_str(&source).context("invalid FastSpeech tokenizer config")?;
            let projector = PhonemeVocabularyProjector::from_config(tokenizer.clone())?;
            ensure!(
                projector.vocabulary().len() == model.num_chars,
                "FastSpeech symbol count {} does not match num_chars {}",
                projector.vocabulary().len(),
                model.num_chars
            );
            let parameters = json!({
                "model": model,
                "audio": audio,
                "tokenizer": tokenizer,
            });
            ParsedConfig::FastSpeech {
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
        ModelPackageArchitecture::Tacotron | ModelPackageArchitecture::Tacotron2 => {
            let inference =
                TacotronInferenceConfig::from_json_value(&root).map_err(anyhow::Error::new)?;
            ensure!(
                matches!(
                    (architecture, inference.architecture),
                    (
                        ModelPackageArchitecture::Tacotron,
                        TacotronArchitecture::Tacotron
                    ) | (
                        ModelPackageArchitecture::Tacotron2,
                        TacotronArchitecture::Tacotron2
                    )
                ),
                "detected Tacotron architecture disagrees with model config"
            );
            let audio = AudioFeatureConfig::from_file(path)?;
            let mut tokenizer: PhonemeTokenizerConfig =
                json5::from_str(&source).context("invalid Tacotron tokenizer config")?;
            if root
                .get("characters")
                .and_then(|characters| characters.get("is_sorted"))
                .is_none()
            {
                tokenizer.characters.is_sorted = false;
            }
            let symbols = if tokenizer.use_phonemes {
                PhonemeVocabularyProjector::from_config(tokenizer.clone())?
                    .vocabulary()
                    .iter()
                    .map(char::to_string)
                    .collect::<Vec<_>>()
            } else {
                TacotronGraphemeProjector::from_config(&tokenizer)?
                    .vocabulary()
                    .iter()
                    .map(char::to_string)
                    .collect::<Vec<_>>()
            };
            ensure!(
                symbols.len() == inference.num_chars,
                "Tacotron symbol count {} does not match num_chars {}",
                symbols.len(),
                inference.num_chars
            );
            let parameters = json!({
                "model": inference,
                "audio": audio,
                "tokenizer": tokenizer,
            });
            ParsedConfig::Tacotron {
                inference,
                audio,
                parameters,
                symbols,
            }
        }
        ModelPackageArchitecture::DelightfulTts => {
            ensure!(
                !object
                    .get("use_language_embedding")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                "native DelightfulTTS does not support language embeddings"
            );
            let model = DelightfulTtsConfig::from_json_value(&root)?;
            let tokenizer: PhonemeTokenizerConfig =
                json5::from_str(&source).context("invalid DelightfulTTS tokenizer config")?;
            let projector = PhonemeVocabularyProjector::from_config(tokenizer.clone())?;
            ensure!(
                projector.vocabulary().len() == model.num_chars,
                "DelightfulTTS symbol count {} does not match num_chars {}",
                projector.vocabulary().len(),
                model.num_chars
            );
            let padding_symbol = root
                .get("characters")
                .and_then(|value| value.get("pad"))
                .and_then(Value::as_str)
                .and_then(|value| value.chars().next())
                .context("DelightfulTTS config requires a one-character padding symbol")?;
            let padding_id = projector
                .symbol_id(padding_symbol)
                .context("DelightfulTTS padding symbol is absent from its vocabulary")?;
            let padding_id = usize::try_from(padding_id)
                .context("DelightfulTTS padding ID must be non-negative")?;
            let parameters = json!({
                "model": model,
                "tokenizer": tokenizer,
            });
            ParsedConfig::DelightfulTts {
                model,
                parameters,
                symbols: projector.vocabulary().iter().map(char::to_string).collect(),
                padding_id,
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
            let model = HifiganBundleConfig::from_file(path)?;
            let parameters = serde_json::to_value(&model)?;
            ParsedConfig::HifiGan { model, parameters }
        }
        ModelPackageArchitecture::MelGan | ModelPackageArchitecture::MultibandMelGan => {
            let params = object
                .get("generator_model_params")
                .and_then(Value::as_object)
                .context("MelGAN config requires generator_model_params")?;
            reject_unknown_keys(
                params,
                &[
                    "in_channels",
                    "out_channels",
                    "proj_kernel",
                    "base_channels",
                    "upsample_factors",
                    "res_kernel",
                    "num_res_blocks",
                    "inference_padding",
                ],
                "generator_model_params",
            )?;
            let model = MelganBundleConfig::from_file(path)?;
            ensure!(
                matches!(
                    (architecture, model.variant()?),
                    (ModelPackageArchitecture::MelGan, MelganVariant::Melgan)
                        | (
                            ModelPackageArchitecture::MultibandMelGan,
                            MelganVariant::Multiband
                        )
                ),
                "detected MelGAN architecture disagrees with generator config"
            );
            let parameters = serde_json::to_value(&model)?;
            ParsedConfig::MelGan { model, parameters }
        }
        ModelPackageArchitecture::GlowTts => {
            let inference = GlowTtsInferenceConfig::from_json5_str(&source)?;
            let projector = PhonemeVocabularyProjector::from_legacy_config_with_duplicates(
                inference.tokenizer.clone(),
            )?;
            let symbols = projector.vocabulary().iter().map(char::to_string).collect();
            let parameters = json!({
                "model": inference,
                "tokenizer": inference.tokenizer,
            });
            ParsedConfig::GlowTts {
                inference,
                parameters,
                symbols,
            }
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
        ModelPackageArchitecture::XttsV2 => {
            let tokenizer_path =
                tokenizer_path.context("XTTS import requires --tokenizer /path/to/vocab.json")?;
            let model = XttsV2Config::from_json_value(&root, tokenizer_path)?;
            let parameters = serde_json::to_value(&model)?;
            ParsedConfig::Xtts { model, parameters }
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
    if let Some(generator) = object.get("generator_model").and_then(Value::as_str) {
        return match generator {
            "hifigan_generator" => Ok(ModelPackageArchitecture::HifiGan),
            "melgan_generator" => Ok(ModelPackageArchitecture::MelGan),
            "multiband_melgan_generator" => Ok(ModelPackageArchitecture::MultibandMelGan),
            other => bail!("unsupported Coqui generator architecture `{other}`"),
        };
    }
    match object.get("model").and_then(Value::as_str) {
        Some("align_tts") | Some("align-tts") => Ok(ModelPackageArchitecture::AlignTts),
        Some("fast_pitch") => Ok(ModelPackageArchitecture::FastPitch),
        Some("fastspeech") | Some("fast_speech") => Ok(ModelPackageArchitecture::FastSpeech),
        Some("fastspeech2") | Some("fast_speech2") => Ok(ModelPackageArchitecture::FastSpeech2),
        Some("speedy_speech") => Ok(ModelPackageArchitecture::SpeedySpeech),
        Some(value) if value.eq_ignore_ascii_case("tacotron") => {
            Ok(ModelPackageArchitecture::Tacotron)
        }
        Some(value)
            if value.eq_ignore_ascii_case("tacotron2")
                || value.eq_ignore_ascii_case("tacotron_2")
                || value.eq_ignore_ascii_case("tacotron-2") =>
        {
            Ok(ModelPackageArchitecture::Tacotron2)
        }
        Some("delightful_tts") => Ok(ModelPackageArchitecture::DelightfulTts),
        Some(value)
            if value.eq_ignore_ascii_case("glow_tts") || value.eq_ignore_ascii_case("glow-tts") =>
        {
            Ok(ModelPackageArchitecture::GlowTts)
        }
        Some(value) if value.eq_ignore_ascii_case("vits") => Ok(ModelPackageArchitecture::Vits),
        Some(value) if value.eq_ignore_ascii_case("xtts") => Ok(ModelPackageArchitecture::XttsV2),
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
        ModelPackageArchitecture::AlignTts
        | ModelPackageArchitecture::FastPitch
        | ModelPackageArchitecture::FastSpeech
        | ModelPackageArchitecture::FastSpeech2
        | ModelPackageArchitecture::SpeedySpeech
        | ModelPackageArchitecture::Tacotron
        | ModelPackageArchitecture::Tacotron2
        | ModelPackageArchitecture::DelightfulTts
        | ModelPackageArchitecture::GlowTts
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
            "out_channels",
            "hidden_channels_enc",
            "hidden_channels_encoder",
            "hidden_channels_dec",
            "hidden_channels_decoder",
            "hidden_channels_dp",
            "hidden_channels_duration_predictor",
            "dropout_p_dp",
            "dropout_p_dec",
            "mean_only",
            "num_flow_blocks_dec",
            "kernel_size_dec",
            "dilation_rate",
            "num_block_layers",
            "num_splits",
            "num_squeeze",
            "sigmoid_scale",
            "encoder_type",
            "encoder_params",
            "use_encoder_prenet",
            "use_d_vector_file",
            "use_external_speaker_embedding_file",
            "d_vector_dim",
            "use_sdp",
            "inference_noise_scale",
            "inference_noise_scale_dp",
            "length_scale",
            "run_name",
            "run_description",
            "r",
            "memory_size",
            "prenet_type",
            "prenet_dropout",
            "prenet_dropout_at_inference",
            "stopnet",
            "separate_stopnet",
            "max_decoder_steps",
            "encoder_in_features",
            "decoder_in_features",
            "attention_type",
            "attention_heads",
            "attention_norm",
            "attention_win",
            "windowing",
            "use_forward_attn",
            "forward_attn_mask",
            "transition_agent",
            "location_attn",
            "bidirectional_decoder",
            "double_decoder_consistency",
            "ddc_r",
            "use_gst",
            "gst",
            "use_capacitron_vae",
            "capacitron_vae",
        ]
        .as_slice(),
        ModelPackageArchitecture::HifiGan
        | ModelPackageArchitecture::MelGan
        | ModelPackageArchitecture::MultibandMelGan => [
            "generator_model",
            "generator_model_params",
            "audio",
            "use_pqmf",
        ]
        .as_slice(),
        ModelPackageArchitecture::XttsV2 => [
            "model",
            "model_args",
            "audio",
            "languages",
            "temperature",
            "length_penalty",
            "repetition_penalty",
            "top_k",
            "top_p",
            "gpt_cond_len",
            "gpt_cond_chunk_len",
            "max_ref_len",
            "sound_norm_refs",
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
        ModelPackageArchitecture::AlignTts => {}
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
        ModelPackageArchitecture::FastSpeech | ModelPackageArchitecture::FastSpeech2 => {
            ignored.extend(
                [
                    "model_args.detach_duration_predictor",
                    "model_args.poisitonal_encoding_use_scale",
                ]
                .into_iter()
                .filter(|path| json_path_exists(object, path))
                .map(str::to_string),
            );
        }
        ModelPackageArchitecture::DelightfulTts => ignored.extend(
            [
                "model_args.freeze_basis_vectors_predictor",
                "model_args.freeze_decoder",
                "model_args.freeze_duration_predictor",
                "model_args.freeze_energy_predictor",
                "model_args.freeze_pitch_predictor",
                "model_args.freeze_text_encoder",
                "model_args.freeze_vocoder",
                "model_args.spec_segment_size",
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
        ModelPackageArchitecture::XttsV2 => ignored.extend(
            [
                "model_args.gpt_batch_size",
                "model_args.enable_redaction",
                "model_args.gpt_checkpoint",
                "model_args.clvp_checkpoint",
                "model_args.decoder_checkpoint",
                "model_args.num_chars",
                "model_args.tokenizer_file",
            ]
            .into_iter()
            .filter(|path| json_path_exists(object, path))
            .map(str::to_string),
        ),
        ModelPackageArchitecture::GlowTts => {}
        ModelPackageArchitecture::Tacotron | ModelPackageArchitecture::Tacotron2 => {}
        ModelPackageArchitecture::HifiGan
        | ModelPackageArchitecture::MelGan
        | ModelPackageArchitecture::MultibandMelGan
        | ModelPackageArchitecture::SpeakerEncoder => {}
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
        ParsedConfig::DelightfulTts { model, .. } if model.speakers.use_speaker_embedding => Some(
            u32::try_from(model.speakers.num_speakers)
                .context("DelightfulTTS speaker count does not fit u32")?,
        ),
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
    if let ParsedConfig::Vits { inference, .. } = parsed {
        if inference.network.use_d_vector_file {
            let catalog = DVectorCatalog::from_file(
                path,
                inference.network.d_vector_dim,
                COQUI_RESNET_SPEAKER_EMBEDDING_SPACE,
            )?;
            return catalog
                .speaker_names()
                .into_iter()
                .enumerate()
                .map(|(id, name)| {
                    Ok(PackageSpeaker {
                        id: id
                            .try_into()
                            .context("d-vector speaker ID does not fit u32")?,
                        name: name.into(),
                    })
                })
                .collect();
        }
    }
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

fn load_languages(
    options: &CoquiImportOptions,
    root: &Value,
    parsed: &ParsedConfig,
) -> Result<Vec<PackageLanguage>> {
    let expected = match parsed {
        ParsedConfig::Vits { inference, .. } if inference.network.use_language_embedding => {
            Some(inference.network.num_languages)
        }
        _ => None,
    };
    let mut languages = if let ParsedConfig::Xtts { model, .. } = parsed {
        ensure!(
            options.language_map_path.is_none(),
            "XTTS languages come from config.json; do not pass --languages"
        );
        model
            .languages
            .iter()
            .map(|tag| PackageLanguage {
                id: None,
                tag: tag.clone(),
            })
            .collect::<Vec<_>>()
    } else if let Some(path) = options.language_map_path.as_deref() {
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read language map {}", path.display()))?;
        let map: BTreeMap<String, u32> = serde_json::from_str(&source)
            .with_context(|| format!("invalid language map {}", path.display()))?;
        if let Some(expected) = expected {
            LanguageCatalog::new(map.clone(), expected)?;
        }
        map.into_iter()
            .map(|(tag, id)| PackageLanguage { id: Some(id), tag })
            .collect::<Vec<_>>()
    } else {
        ensure!(
            expected.is_none(),
            "language_ids.json is required for this {}-language model",
            expected.unwrap_or_default()
        );
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

struct CheckpointReader {
    reader: PytorchReader,
    _sanitized: Option<crate::model_config::SanitizedLegacyCheckpoint>,
}

impl Deref for CheckpointReader {
    type Target = PytorchReader;

    fn deref(&self) -> &Self::Target {
        &self.reader
    }
}

fn checkpoint_reader(options: &CoquiImportOptions) -> Result<CheckpointReader> {
    match PytorchReader::with_top_level_key(&options.checkpoint_path, &options.checkpoint_key) {
        Ok(reader) => Ok(CheckpointReader {
            reader,
            _sanitized: None,
        }),
        Err(error) if error.to_string().contains("collections.Counter") => {
            let sanitized =
                crate::model_config::SanitizedLegacyCheckpoint::create(&options.checkpoint_path)?;
            let reader =
                PytorchReader::with_top_level_key(&sanitized.path, &options.checkpoint_key)
                    .with_context(|| {
                        format!(
                            "failed to read safely sanitized legacy tensor key `{}` from {}",
                            options.checkpoint_key,
                            options.checkpoint_path.display()
                        )
                    })?;
            Ok(CheckpointReader {
                reader,
                _sanitized: Some(sanitized),
            })
        }
        Err(nested_error) => match PytorchReader::new(&options.checkpoint_path) {
            Ok(reader) => Ok(CheckpointReader {
                reader,
                _sanitized: None,
            }),
            Err(root_error) => Err(root_error).with_context(|| {
                format!(
                    "failed to read tensor-only checkpoint {} either at root or key `{}`; nested-key error: {nested_error}",
                    options.checkpoint_path.display(),
                    options.checkpoint_key
                )
            }),
        },
    }
}

fn scan_safe_melgan_checkpoint(options: &CoquiImportOptions) -> Result<()> {
    let reader = checkpoint_reader(options).with_context(|| {
        format!(
            "unsafe/unsupported MelGAN checkpoint {}: legacy files may contain only the data types accepted by Burn's non-executing pickle reader",
            options.checkpoint_path.display()
        )
    })?;
    tensor_metadata(&reader)?;
    Ok(())
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
        ParsedConfig::AlignTts { model, .. } => {
            model
                .clone()
                .init::<Cpu>(&device)
                .map_err(anyhow::Error::new)?
                .load_checkpoint(checkpoint_path)
                .map_err(anyhow::Error::new)?;
        }
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
        ParsedConfig::Tacotron { inference, .. } => {
            ensure!(
                inference.architecture == TacotronArchitecture::Tacotron2,
                "Tacotron 1 checkpoint import is not yet shippable: its CBHG encoder and \
                 autoregressive decoder topology differ from the native Tacotron 2 graph"
            );
            inference
                .init_tacotron2::<Cpu>(&device)
                .map_err(anyhow::Error::new)?
                .load_checkpoint(checkpoint_path)
                .map_err(anyhow::Error::new)?;
        }
        ParsedConfig::FastSpeech { model, .. } => {
            model
                .clone()
                .init::<Cpu>(&device)
                .map_err(anyhow::Error::new)?
                .load_checkpoint(checkpoint_path)
                .map_err(anyhow::Error::new)?;
        }
        ParsedConfig::DelightfulTts {
            model, padding_id, ..
        } => {
            model
                .clone()
                .init::<Cpu>(*padding_id, &device)
                .map_err(anyhow::Error::new)?
                .load_checkpoint(checkpoint_path)
                .map_err(anyhow::Error::new)?;
        }
        ParsedConfig::HifiGan { model, .. } => {
            model.load_burn_generator::<Cpu>(checkpoint_path, &device)?;
        }
        ParsedConfig::MelGan { model, .. } => match model.variant()? {
            MelganVariant::Melgan => {
                model.load_burn_generator::<Cpu>(checkpoint_path, &device)?;
            }
            MelganVariant::Multiband => {
                model.load_burn_multiband_generator::<Cpu>(checkpoint_path, &device)?;
            }
        },
        ParsedConfig::Vits { inference, .. } => {
            let speakers = options
                .speaker_map_path
                .as_deref()
                .context("VITS shape validation requires speaker_ids.json")?;
            if inference.network.use_language_embedding {
                let languages = options.language_map_path.as_deref().context(
                    "language-conditioned VITS shape validation requires language_ids.json",
                )?;
                BurnVitsSpeech::<Cpu>::load_with_languages(
                    &options.config_path,
                    checkpoint_path,
                    speakers,
                    languages,
                    device,
                )
                .context("language-conditioned VITS checkpoint shape validation failed")?;
            } else {
                BurnVitsSpeech::<Cpu>::load(
                    &options.config_path,
                    checkpoint_path,
                    speakers,
                    device,
                )
                .context("VITS checkpoint shape validation failed")?;
            }
        }
        ParsedConfig::GlowTts { inference, .. } => {
            if inference.network.use_sdp {
                StochasticGlowTts::<Cpu>::load(inference.clone(), checkpoint_path, device)
                    .context("stochastic Glow-TTS checkpoint shape validation failed")?;
            } else {
                GlowTts::<Cpu>::load(inference.clone(), checkpoint_path, device)
                    .context("Glow-TTS checkpoint shape validation failed")?;
            }
        }
        ParsedConfig::SpeakerEncoder { model, .. } => {
            validate_speaker_encoder_shapes(model, tensors)?;
            if model.model_name.eq_ignore_ascii_case("resnet") {
                CoquiResNetSpeakerEncoder::<Cpu>::load(model, checkpoint_path, &device)
                    .context("ResNet speaker encoder checkpoint shape validation failed")?;
            }
        }
        ParsedConfig::Xtts { model, .. } => validate_xtts_shapes(model, tensors)?,
    }
    Ok(())
}

fn validate_xtts_shapes(config: &XttsV2Config, tensors: &[TensorMetadata]) -> Result<()> {
    for required in [
        "gpt.text_embedding.weight",
        "gpt.mel_embedding.weight",
        "gpt.mel_head.weight",
        "gpt.final_norm.weight",
        "hifigan_decoder.speaker_encoder.fc.weight",
        "hifigan_decoder.waveform_decoder.conv_pre.weight",
    ] {
        ensure!(
            tensors.iter().any(|tensor| tensor.name.ends_with(required)),
            "XTTS checkpoint is missing `{required}`"
        );
    }
    ensure!(
        tensors
            .iter()
            .any(|tensor| tensor.name.contains("gpt.conditioning_perceiver")),
        "XTTS v2 checkpoint is missing the conditioning perceiver"
    );
    let text_embedding = tensors
        .iter()
        .find(|tensor| tensor.name.ends_with("gpt.text_embedding.weight"))
        .context("XTTS checkpoint has no text embedding")?;
    ensure!(
        text_embedding.shape
            == [
                config.model_args.gpt_number_text_tokens,
                config.model_args.gpt_n_model_channels,
            ],
        "XTTS text embedding has shape {:?}, expected [{}, {}]",
        text_embedding.shape,
        config.model_args.gpt_number_text_tokens,
        config.model_args.gpt_n_model_channels
    );
    let mel_embedding = tensors
        .iter()
        .find(|tensor| tensor.name.ends_with("gpt.mel_embedding.weight"))
        .context("XTTS checkpoint has no audio-code embedding")?;
    ensure!(
        mel_embedding.shape
            == [
                config.model_args.gpt_num_audio_tokens,
                config.model_args.gpt_n_model_channels,
            ],
        "XTTS audio-code embedding has shape {:?}, expected [{}, {}]",
        mel_embedding.shape,
        config.model_args.gpt_num_audio_tokens,
        config.model_args.gpt_n_model_channels
    );
    let transformer_layers = tensors
        .iter()
        .filter_map(|tensor| {
            tensor
                .name
                .strip_prefix("gpt.gpt.h.")
                .or_else(|| tensor.name.strip_prefix("xtts.gpt.gpt.h."))
                .and_then(|suffix| suffix.split('.').next())
                .and_then(|index| index.parse::<usize>().ok())
        })
        .collect::<BTreeSet<_>>();
    ensure!(
        transformer_layers.len() == config.model_args.gpt_layers,
        "XTTS checkpoint exposes {} GPT layers, expected {}",
        transformer_layers.len(),
        config.model_args.gpt_layers
    );
    Ok(())
}

fn validate_speaker_encoder_shapes(
    config: &SpeakerEncoderPackageConfig,
    tensors: &[TensorMetadata],
) -> Result<()> {
    if config.model_name.eq_ignore_ascii_case("resnet") {
        let projection = tensors
            .iter()
            .find(|tensor| tensor.name.ends_with("fc.weight"))
            .context("ResNet speaker encoder checkpoint has no `fc.weight` projection")?;
        let final_frequency = config.input_dim.div_ceil(8);
        let pooled = config.num_filters[3] * final_frequency * 2;
        ensure!(
            projection.shape == [config.projection_dim, pooled]
                || projection.shape == [pooled, config.projection_dim],
            "ResNet speaker encoder projection weight has shape {:?}; expected [{}, {}] (PyTorch) or its runtime transpose",
            projection.shape,
            config.projection_dim,
            pooled
        );
        for required in ["conv1.weight", "attention.0.weight", "attention.3.weight"] {
            ensure!(
                tensors.iter().any(|tensor| tensor.name.ends_with(required)),
                "ResNet speaker encoder checkpoint is missing `{required}`"
            );
        }
        return Ok(());
    }

    let lstm_dim = config
        .lstm_dim
        .context("LSTM speaker encoder is missing `lstm_dim`")?;
    let num_lstm_layers = config
        .num_lstm_layers
        .context("LSTM speaker encoder is missing `num_lstm_layers`")?;
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
        projection.shape == [config.projection_dim, lstm_dim]
            || projection.shape == [lstm_dim, config.projection_dim],
        "speaker encoder projection weight has shape {:?}; expected [{}, {}] (PyTorch) or its runtime transpose",
        projection.shape,
        config.projection_dim,
        lstm_dim
    );
    let lstm_weights = tensors
        .iter()
        .filter(|tensor| tensor.name.contains("lstm") && tensor.name.contains("weight"))
        .count();
    ensure!(
        lstm_weights >= num_lstm_layers * 2,
        "speaker encoder checkpoint has {lstm_weights} LSTM weights; expected at least {} for {} layers",
        num_lstm_layers * 2,
        num_lstm_layers
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
    let mut tensors = reader
        .tensors()
        .iter()
        .map(|(name, snapshot)| (name.clone(), snapshot.clone()))
        .collect::<Vec<_>>();
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
    if let Some(path) = &options.tokenizer_path {
        artifacts.push(source_artifact("tokenizer", path)?);
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

fn copy_file_atomic(source: &Path, destination: &Path) -> Result<()> {
    let part = part_path(destination);
    let mut input = BufReader::new(
        File::open(source).with_context(|| format!("failed to open {}", source.display()))?,
    );
    let mut output = BufWriter::new(
        File::create(&part).with_context(|| format!("failed to create {}", part.display()))?,
    );
    std::io::copy(&mut input, &mut output)?;
    output.flush()?;
    output.get_ref().sync_all()?;
    fs::rename(&part, destination)
        .with_context(|| format!("failed to atomically install {}", destination.display()))?;
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
    scan_safe_pytorch_checkpoint_profile(path.as_ref(), false)
}

fn scan_safe_xtts_pytorch_checkpoint(path: impl AsRef<Path>) -> Result<()> {
    scan_safe_pytorch_checkpoint_profile(path.as_ref(), true)
}

fn scan_safe_pytorch_checkpoint_profile(path: &Path, allow_xtts_config: bool) -> Result<()> {
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
    scan_pickle_program(&pickle, allow_xtts_config)
}

fn scan_pickle_program(bytes: &[u8], allow_xtts_config: bool) -> Result<()> {
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
                    allowed_pickle_global(module, name, allow_xtts_config),
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
            0x81 if allow_xtts_config => {}
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

fn allowed_pickle_global(module: &str, name: &str, allow_xtts_config: bool) -> bool {
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
    ) || (allow_xtts_config
        && matches!(
            (module, name),
            ("TTS.tts.models.xtts", "XttsAudioConfig")
                | ("TTS.tts.models.xtts", "XttsArgs")
                | ("TTS.tts.configs.xtts_config", "XttsConfig")
                | ("TTS.tts.configs.shared_configs", "BaseDatasetConfig")
        ))
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

fn optional_usize_array(object: &Map<String, Value>, key: &str) -> Result<Option<Vec<usize>>> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    let values = value
        .as_array()
        .with_context(|| format!("config field `{key}` must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let value = value
                .as_u64()
                .with_context(|| format!("config field `{key}[{index}]` must be numeric"))?;
            usize::try_from(value)
                .with_context(|| format!("config field `{key}[{index}]` does not fit usize"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(values))
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
        scan_pickle_program(pickle, false).expect("tensor-only pickle vocabulary");
    }

    #[test]
    fn safe_pickle_scanner_rejects_arbitrary_global() {
        let pickle = b"\x80\x02cos\nsystem\n.";
        let error = scan_pickle_program(pickle, false).expect_err("os.system must be rejected");
        assert!(error.to_string().contains("os.system"));
        assert!(error.to_string().contains("arbitrary Python"));
    }

    #[test]
    fn safe_pickle_scanner_rejects_stack_global() {
        let pickle = b"\x80\x02\x93.";
        let error = scan_pickle_program(pickle, false).expect_err("STACK_GLOBAL must be rejected");
        assert!(error.to_string().contains("opcode 0x93"));
    }

    #[test]
    fn xtts_pickle_profile_allows_only_known_inert_config_classes() {
        let pickle = b"\x80\x02cTTS.tts.models.xtts\nXttsArgs\n)\x81.";
        scan_pickle_program(pickle, true).expect("known XTTS config dataclass");
        let error =
            scan_pickle_program(pickle, false).expect_err("generic tensor scan must stay strict");
        assert!(error.to_string().contains("XttsArgs"));

        let unknown = b"\x80\x02cTTS.tts.models.xtts\nRunMe\n)\x81.";
        let error =
            scan_pickle_program(unknown, true).expect_err("unknown XTTS global must be rejected");
        assert!(error.to_string().contains("RunMe"));
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
    fn detects_all_duration_based_acoustic_architectures() {
        let detect = |model| {
            let root = json!({"model": model});
            detect_architecture(root.as_object().expect("object"))
        };
        assert_eq!(
            detect("align_tts").expect("Align-TTS"),
            ModelPackageArchitecture::AlignTts
        );
        assert_eq!(
            detect("fastspeech").expect("FastSpeech"),
            ModelPackageArchitecture::FastSpeech
        );
        assert_eq!(
            detect("fastspeech2").expect("FastSpeech 2"),
            ModelPackageArchitecture::FastSpeech2
        );
        assert_eq!(
            detect("delightful_tts").expect("DelightfulTTS"),
            ModelPackageArchitecture::DelightfulTts
        );
    }

    #[test]
    fn detects_legacy_mixed_case_tacotron_architectures() {
        let detect = |model| {
            let root = json!({"model": model});
            detect_architecture(root.as_object().expect("object"))
        };
        assert_eq!(
            detect("Tacotron").expect("Tacotron"),
            ModelPackageArchitecture::Tacotron
        );
        assert_eq!(
            detect("Tacotron2").expect("Tacotron 2"),
            ModelPackageArchitecture::Tacotron2
        );
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
            lstm_dim: Some(768),
            num_lstm_layers: Some(2),
            use_lstm_with_projection: true,
            use_torch_spec: false,
            log_input: false,
            encoder_type: "ASP".into(),
            layers: vec![3, 4, 6, 3],
            num_filters: vec![32, 64, 128, 256],
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
    fn published_tacotron2_ddc_fixture_uses_common_importer_when_available() {
        let Some(options) = fixture_options(
            "TONGUES_TEST_COQUI_TACOTRON2_CONFIG",
            "TONGUES_TEST_COQUI_TACOTRON2_MODEL",
        ) else {
            return;
        };
        let inspection = inspect_coqui_import(&options).expect("Tacotron2-DDC import inspection");
        assert_eq!(inspection.architecture, ModelPackageArchitecture::Tacotron2);
        assert_eq!(inspection.audio.expect("audio").mel_bins, Some(80));
        assert_eq!(inspection.symbols.len(), 64);
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
    fn fastspeech_fixture_uses_common_importer_when_available() {
        let Some(options) = fixture_options(
            "TONGUES_TEST_COQUI_FASTSPEECH_CONFIG",
            "TONGUES_TEST_COQUI_FASTSPEECH_MODEL",
        ) else {
            return;
        };
        let inspection = inspect_coqui_import(&options).expect("FastSpeech import inspection");
        assert_eq!(
            inspection.architecture,
            ModelPackageArchitecture::FastSpeech
        );
        assert!(!inspection.tensors.is_empty());
    }

    #[test]
    fn fastspeech2_fixture_uses_common_importer_when_available() {
        let Some(options) = fixture_options(
            "TONGUES_TEST_COQUI_FASTSPEECH2_CONFIG",
            "TONGUES_TEST_COQUI_FASTSPEECH2_MODEL",
        ) else {
            return;
        };
        let inspection = inspect_coqui_import(&options).expect("FastSpeech 2 import inspection");
        assert_eq!(
            inspection.architecture,
            ModelPackageArchitecture::FastSpeech2
        );
        assert!(!inspection.tensors.is_empty());
    }

    #[test]
    fn delightful_tts_fixture_uses_common_importer_when_available() {
        let Some(options) = fixture_options(
            "TONGUES_TEST_COQUI_DELIGHTFUL_TTS_CONFIG",
            "TONGUES_TEST_COQUI_DELIGHTFUL_TTS_MODEL",
        ) else {
            return;
        };
        let inspection = inspect_coqui_import(&options).expect("DelightfulTTS import inspection");
        assert_eq!(
            inspection.architecture,
            ModelPackageArchitecture::DelightfulTts
        );
        assert!(!inspection.tensors.is_empty());
    }

    #[test]
    #[ignore = "requires the checksum-pinned published Glow-TTS artifact"]
    fn published_glow_tts_fixture_uses_common_importer() {
        let options = fixture_options("TONGUES_TEST_GLOW_CONFIG", "TONGUES_TEST_GLOW_CHECKPOINT")
            .expect("TONGUES_TEST_GLOW_CONFIG and TONGUES_TEST_GLOW_CHECKPOINT are required");
        let inspection = inspect_coqui_import(&options).expect("Glow-TTS import inspection");
        assert_eq!(inspection.architecture, ModelPackageArchitecture::GlowTts);
        assert_eq!(inspection.audio.expect("audio").mel_bins, Some(80));
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
    fn published_melgan_fixture_uses_common_importer_when_available() {
        let Some(options) = fixture_options(
            "TONGUES_TEST_DESCRIPT_MELGAN_CONFIG",
            "TONGUES_TEST_DESCRIPT_MELGAN_MODEL",
        ) else {
            return;
        };
        let inspection = inspect_coqui_import(&options).expect("MelGAN import inspection");
        assert_eq!(inspection.architecture, ModelPackageArchitecture::MelGan);
        assert!(!inspection.tensors.is_empty());
    }

    #[test]
    fn published_multiband_melgan_fixture_uses_common_importer_when_available() {
        let Some(options) = fixture_options(
            "TONGUES_TEST_COQUI_MULTIBAND_MELGAN_CONFIG",
            "TONGUES_TEST_COQUI_MULTIBAND_MELGAN_MODEL",
        ) else {
            return;
        };
        let inspection =
            inspect_coqui_import(&options).expect("MultiBand-MelGAN import inspection");
        assert_eq!(
            inspection.architecture,
            ModelPackageArchitecture::MultibandMelGan
        );
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
