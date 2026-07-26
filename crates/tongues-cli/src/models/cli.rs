use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use inquire::Select;
use owo_colors::OwoColorize;
use std::path::PathBuf;

use crate::models::download::fetch_model;
use crate::models::manifest::{
    bundle_required_assets, find_bundle, ModelKind, MODEL_ASSETS, MODEL_BUNDLES,
};
use crate::models::selection::{
    asset_path, bundle_present, is_non_empty_file, model_selection_path, resolve_mortar_home,
    selected_bundle, selected_bundle_for_kind, selected_llm_model_path, write_selected_model,
    write_selected_model_for_kind,
};

#[derive(Debug, Subcommand)]
pub enum ModelsCommand {
    #[command(about = "Choose the active LLM model")]
    Menu,
    #[command(about = "List known model bundles")]
    List,
    #[command(about = "Print model paths and current selection")]
    Path(ModelsPathCommand),
    #[command(about = "Show selected model and file presence")]
    Status,
    #[command(about = "Select the active LLM model")]
    Use(ModelsUseCommand),
    #[command(about = "Fetch default runtime models, or a named model")]
    Fetch(ModelsFetchCommand),
    #[command(
        name = "import-coqui",
        about = "Safely import a Coqui config/checkpoint into a versioned Tongues package"
    )]
    ImportCoqui(ModelsImportCoquiCommand),
    #[command(
        name = "inspect-package",
        about = "Validate and print a Tongues model package manifest"
    )]
    InspectPackage(ModelsInspectPackageCommand),
}

#[derive(Debug, Args)]
pub struct ModelsUseCommand {
    #[arg(default_value = "gemma4")]
    model: String,
}

#[derive(Debug, Args)]
pub struct ModelsFetchCommand {
    model: Option<String>,
    #[arg(long)]
    force: bool,
}

#[derive(Debug, Args)]
pub struct ModelsPathCommand {
    model: Option<String>,
}

#[derive(Debug, Args)]
pub struct ModelsImportCoquiCommand {
    /// Coqui JSON or JSON5 configuration
    #[arg(long)]
    config: PathBuf,
    /// Modern ZIP-based PyTorch checkpoint
    #[arg(long)]
    checkpoint: PathBuf,
    /// Destination directory for manifest, neutral config, tensor index, and SafeTensors
    #[arg(long)]
    out: Option<PathBuf>,
    /// Optional Coqui speaker_ids.json
    #[arg(long = "speakers")]
    speaker_map: Option<PathBuf>,
    /// Optional Coqui language_ids.json
    #[arg(long = "languages")]
    language_map: Option<PathBuf>,
    /// Tensor dictionary key inside the checkpoint
    #[arg(long, default_value = "model")]
    checkpoint_key: String,
    /// SPDX license expression or explicit LicenseRef
    #[arg(long)]
    license: String,
    /// Stable upstream URL or provenance identifier
    #[arg(long)]
    source: String,
    /// Upstream Coqui version/revision
    #[arg(long)]
    coqui_version: Option<String>,
    /// Validate and inspect without writing a package
    #[arg(long)]
    dry_run: bool,
    /// Print the inspection/manifest as JSON
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
pub struct ModelsInspectPackageCommand {
    /// Package directory or manifest.json
    package: PathBuf,
}

pub fn run(command: Option<ModelsCommand>) -> Result<()> {
    match command.unwrap_or(ModelsCommand::Menu) {
        ModelsCommand::Menu => model_menu(),
        ModelsCommand::List => list_models(),
        ModelsCommand::Path(command) => print_paths(command.model.as_deref()),
        ModelsCommand::Status => print_status(),
        ModelsCommand::Use(command) => select_model(&command.model),
        ModelsCommand::Fetch(command) => {
            fetch_model(command.model.as_deref(), command.force)?;
            Ok(())
        }
        ModelsCommand::ImportCoqui(command) => import_coqui(command),
        ModelsCommand::InspectPackage(command) => inspect_package(command),
    }
}

fn import_coqui(command: ModelsImportCoquiCommand) -> Result<()> {
    let output = command
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from("model-package"));
    anyhow::ensure!(
        command.dry_run || command.out.is_some(),
        "--out is required unless --dry-run is used"
    );
    let mut options = tongues_tts::CoquiImportOptions::new(
        command.config,
        command.checkpoint,
        output,
        command.license,
        command.source,
    );
    options.speaker_map_path = command.speaker_map;
    options.language_map_path = command.language_map;
    options.checkpoint_key = command.checkpoint_key;
    options.coqui_version = command.coqui_version;

    let mut report_progress = |event| match event {
        tongues_tts::ModelImportProgress::ReadingConfig { path } => {
            eprintln!("import: reading config {}", path.display());
        }
        tongues_tts::ModelImportProgress::ScanningCheckpoint { path } => {
            eprintln!(
                "import: scanning safe tensor-only checkpoint {}",
                path.display()
            );
        }
        tongues_tts::ModelImportProgress::ValidatingShapes { architecture } => {
            eprintln!(
                "import: validating {:?} tensor names and shapes",
                architecture
            );
        }
        tongues_tts::ModelImportProgress::ValidatingConvertedWeights { architecture, path } => {
            eprintln!(
                "import: loading converted {:?} weights through the native runtime from {}",
                architecture,
                path.display()
            );
        }
        tongues_tts::ModelImportProgress::ConvertingTensor {
            current,
            total,
            name,
            output,
        } => {
            eprintln!(
                "import: converting tensor {current}/{total} {name} -> {}",
                output.display()
            );
        }
        tongues_tts::ModelImportProgress::WritingMetadata { path } => {
            eprintln!("import: writing {}", path.display());
        }
        tongues_tts::ModelImportProgress::Complete { path, sha256 } => {
            eprintln!(
                "import: complete {} manifest-sha256={sha256}",
                path.display()
            );
        }
    };
    if command.dry_run {
        let inspection =
            tongues_tts::inspect_coqui_import_with_progress(&options, &mut report_progress)?;
        if command.json {
            println!("{}", serde_json::to_string_pretty(&inspection)?);
        } else {
            println!(
                "compatible {:?} checkpoint: {} tensors, {} speakers, {} languages, {} symbols",
                inspection.architecture,
                inspection.tensor_count,
                inspection.speakers.len(),
                inspection.languages.len(),
                inspection.symbols.len()
            );
            if !inspection.ignored_training_fields.is_empty() {
                println!(
                    "reported training-only fields: {}",
                    inspection.ignored_training_fields.join(", ")
                );
            }
        }
    } else {
        let manifest =
            tongues_tts::import_coqui_model_with_progress(&options, &mut report_progress)?;
        if command.json {
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        } else {
            println!(
                "wrote schema-v{} {:?} package to {} ({} tensors)",
                manifest.schema_version,
                manifest.architecture,
                options.output_dir.display(),
                manifest.tensor_count
            );
        }
    }
    Ok(())
}

fn inspect_package(command: ModelsInspectPackageCommand) -> Result<()> {
    let package = tongues_tts::open_model_package(&command.package)?;
    println!("{}", serde_json::to_string_pretty(&package.manifest)?);
    Ok(())
}

fn model_menu() -> Result<()> {
    let category = Select::new(
        "Model category",
        vec![
            CategoryChoice::new(ModelKind::Llm)?,
            CategoryChoice::new(ModelKind::VoiceModel)?,
        ],
    )
    .prompt()
    .context("model menu was cancelled")?;
    let selected = selected_bundle_for_kind(category.kind)?;
    let choices = MODEL_BUNDLES
        .iter()
        .filter(|bundle| bundle.kind == category.kind)
        .map(|bundle| {
            let state = if bundle_present(bundle)? {
                "present".green().to_string()
            } else {
                "missing".red().to_string()
            };
            let current = if bundle.id == selected.id {
                " current".cyan().to_string()
            } else {
                String::new()
            };
            Ok(ModelChoice {
                bundle,
                label: format!("{:<28} {}{}", bundle.display_name, state, current),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let cursor = choices
        .iter()
        .position(|choice| choice.bundle.id == selected.id)
        .unwrap_or(0);
    let choice = Select::new(&format!("{} model", category.name), choices)
        .with_starting_cursor(cursor)
        .prompt()
        .context("model menu was cancelled")?;

    write_selected_model_for_kind(category.kind, choice.bundle.id)?;
    println!(
        "{} {} {}",
        "selected".green(),
        model_kind_label(category.kind),
        choice.bundle.display_name.bold()
    );
    Ok(())
}

#[derive(Clone)]
struct CategoryChoice {
    kind: ModelKind,
    name: &'static str,
    label: String,
}

impl CategoryChoice {
    fn new(kind: ModelKind) -> Result<Self> {
        let name = match kind {
            ModelKind::Llm => "LLM",
            ModelKind::VoiceModel => "Voice model",
            _ => model_kind_label(kind),
        };
        let selected = selected_bundle_for_kind(kind)?;
        Ok(Self {
            kind,
            name,
            label: format!(
                "{name:<12} {}",
                format!("current: {}", selected.display_name).dimmed()
            ),
        })
    }
}

impl std::fmt::Display for CategoryChoice {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.label)
    }
}

#[derive(Clone)]
struct ModelChoice {
    bundle: &'static crate::models::manifest::ModelBundle,
    label: String,
}

impl std::fmt::Display for ModelChoice {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.label)
    }
}

fn list_models() -> Result<()> {
    let selected = selected_bundle()?;
    let selected_voice = selected_bundle_for_kind(ModelKind::VoiceModel)?;
    println!("{}", "Models".bold());
    for bundle in MODEL_BUNDLES {
        let marker = if (bundle.kind == ModelKind::Llm && bundle.id == selected.id)
            || (bundle.kind == ModelKind::VoiceModel && bundle.id == selected_voice.id)
        {
            "*"
        } else {
            " "
        };
        let state = if bundle_present(bundle)? {
            "present".green().to_string()
        } else {
            "missing".red().to_string()
        };
        println!(
            "{} {:<4} {} {:<32} {}",
            marker,
            model_kind_label(bundle.kind),
            bundle.id.bold(),
            bundle.display_name,
            state
        );
    }
    Ok(())
}

fn print_paths(model: Option<&str>) -> Result<()> {
    let home = resolve_mortar_home()?;
    println!("{}={}", "mortar_home".cyan(), home.display());
    println!("{}={}", "models_dir".cyan(), home.join("models").display());
    println!(
        "{}={}",
        "selection".cyan(),
        model_selection_path()?.display()
    );
    if let Some(model) = model {
        let bundle = find_bundle(model).with_context(|| format!("unknown model `{model}`"))?;
        println!(
            "{}={} ({})",
            "bundle".cyan(),
            bundle.id,
            bundle.display_name
        );
        for asset in bundle_required_assets(bundle)? {
            println!("{}={}", asset.id.cyan(), asset_path(&home, asset).display());
        }
    } else {
        for asset in MODEL_ASSETS {
            println!("{}={}", asset.id.cyan(), asset_path(&home, asset).display());
        }
    }
    Ok(())
}

fn print_status() -> Result<()> {
    let bundle = selected_bundle()?;
    println!(
        "{} {} ({})",
        "selected".cyan(),
        bundle.display_name.bold(),
        bundle.id
    );
    let home = resolve_mortar_home()?;
    let selected_path = selected_llm_model_path()?;
    let mut missing = !is_non_empty_file(&selected_path);
    for asset in bundle_required_assets(bundle)? {
        let path = asset_path(&home, asset);
        let state = if is_non_empty_file(&path) {
            "present".green().to_string()
        } else {
            missing = true;
            "missing".red().to_string()
        };
        println!("{} {:<30} {}", state, asset.id, path.display());
    }
    if missing {
        println!("{} cargo run models fetch", "fetch with:".dimmed());
    }

    println!();
    println!("{}", "Face".bold());
    for bundle in MODEL_BUNDLES
        .iter()
        .filter(|bundle| bundle.kind == ModelKind::Face)
    {
        let state = if bundle_present(bundle)? {
            "present".green().to_string()
        } else {
            "missing".red().to_string()
        };
        println!("{} {} ({})", state, bundle.display_name.bold(), bundle.id);
        if !bundle_present(bundle)? {
            println!("{} cargo run models fetch", "fetch with:".dimmed());
        }
    }

    println!();
    println!("{}", "ASR".bold());
    for bundle in MODEL_BUNDLES
        .iter()
        .filter(|bundle| bundle.kind == ModelKind::Asr)
    {
        let state = if bundle_present(bundle)? {
            "present".green().to_string()
        } else {
            "missing".red().to_string()
        };
        println!("{} {} ({})", state, bundle.display_name.bold(), bundle.id);
        if !bundle_present(bundle)? {
            println!("{} cargo run models fetch asr", "fetch with:".dimmed());
        }
    }

    println!();
    println!("{}", "Speech".bold());
    for bundle in MODEL_BUNDLES.iter().filter(|bundle| {
        matches!(
            bundle.kind,
            ModelKind::StyleTts2
                | ModelKind::VoiceModel
                | ModelKind::AcousticModel
                | ModelKind::NeuralVocoder
                | ModelKind::EndToEndSpeech
                | ModelKind::Lexicon
                | ModelKind::Phonemicizer
        )
    }) {
        let selected_marker = if bundle.kind == ModelKind::VoiceModel
            && bundle.id == selected_bundle_for_kind(ModelKind::VoiceModel)?.id
        {
            "* "
        } else {
            "  "
        };
        let state = if bundle_present(bundle)? {
            "present".green().to_string()
        } else {
            "missing".red().to_string()
        };
        println!(
            "{}{} {:<12} {} ({})",
            selected_marker,
            state,
            model_kind_label(bundle.kind),
            bundle.display_name.bold(),
            bundle.id
        );
        if !bundle_present(bundle)? {
            println!(
                "{} cargo run models fetch {}",
                "fetch with:".dimmed(),
                bundle.id
            );
        }
    }
    Ok(())
}

fn select_model(model: &str) -> Result<()> {
    let bundle = find_bundle(model).with_context(|| format!("unknown model `{model}`"))?;
    match bundle.kind {
        ModelKind::Llm => {
            write_selected_model(bundle.id)?;
            println!("{} LLM {}", "selected".green(), bundle.display_name.bold());
        }
        ModelKind::VoiceModel => {
            write_selected_model_for_kind(ModelKind::VoiceModel, bundle.id)?;
            println!(
                "{} voice-model {}",
                "selected".green(),
                bundle.display_name.bold()
            );
        }
        _ => {
            anyhow::bail!(
                "`{model}` is not a selectable model; use `cargo run models fetch {}`",
                bundle.id
            );
        }
    }
    Ok(())
}

fn model_kind_label(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Llm => "llm",
        ModelKind::Face => "face",
        ModelKind::Asr => "asr",
        ModelKind::StyleTts2 => "styletts2",
        ModelKind::VoiceModel => "voice-model",
        ModelKind::AcousticModel => "acoustic-model",
        ModelKind::NeuralVocoder => "neural-vocoder",
        ModelKind::EndToEndSpeech => "end-to-end-speech",
        ModelKind::Lexicon => "lexicon",
        ModelKind::Phonemicizer => "phonemicizer",
    }
}
