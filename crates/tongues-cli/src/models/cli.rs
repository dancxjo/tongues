use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use inquire::Select;
use owo_colors::OwoColorize;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use crate::models::download::{fetch_all_models, fetch_model};
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
    #[command(about = "List verified catalog metadata and installation state")]
    List(ModelsListCommand),
    #[command(about = "Search model catalog metadata")]
    Search(ModelsSearchCommand),
    #[command(about = "Install a verified catalog model or local Tongues package")]
    Install(ModelsInstallCommand),
    #[command(about = "Inspect and verify a catalog model or local package")]
    Inspect(ModelsInspectCommand),
    #[command(about = "Deeply verify one installed model or every installed catalog model")]
    Verify(ModelsVerifyCommand),
    #[command(about = "Remove an installed catalog model")]
    Remove(ModelsRemoveCommand),
    #[command(about = "Print model paths and current selection")]
    Path(ModelsPathCommand),
    #[command(about = "Show selected model and file presence")]
    Status,
    #[command(about = "Select the active LLM model")]
    Use(ModelsUseCommand),
    #[command(about = "Fetch default runtime models, every catalog model, or a named model")]
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
    #[command(
        name = "generate-fairseq-catalog",
        about = "Generate and drift-check a checksum-pinned Fairseq MMS catalog"
    )]
    GenerateFairseqCatalog(ModelsGenerateFairseqCatalogCommand),
}

#[derive(Debug, Args)]
pub struct ModelsUseCommand {
    #[arg(default_value = "gemma4")]
    model: String,
}

#[derive(Debug, Args)]
pub struct ModelsFetchCommand {
    /// Fetch every manifest bundle and every model in the default/configured catalogs
    #[arg(long, conflicts_with = "model")]
    all: bool,
    model: Option<String>,
    #[arg(long)]
    force: bool,
}

#[derive(Debug, Args)]
pub struct ModelsPathCommand {
    model: Option<String>,
}

#[derive(Debug, Args, Default)]
pub struct ModelsListCommand {
    /// Additional private/local catalog JSON files
    #[arg(long = "catalog")]
    catalogs: Vec<PathBuf>,
    /// Emit machine-readable JSON
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
pub struct ModelsSearchCommand {
    query: String,
    /// Additional private/local catalog JSON files
    #[arg(long = "catalog")]
    catalogs: Vec<PathBuf>,
    /// Emit machine-readable JSON
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
pub struct ModelsInstallCommand {
    /// Catalog model id or alias
    model: Option<String>,
    /// Install an already converted local Tongues package directory
    #[arg(long, conflicts_with = "model", requires = "id")]
    package: Option<PathBuf>,
    /// Stable id for --package
    #[arg(long)]
    id: Option<String>,
    /// Additional private/local catalog JSON files
    #[arg(long = "catalog")]
    catalogs: Vec<PathBuf>,
    /// Never contact the network; use verified cache/installations only
    #[arg(long)]
    offline: bool,
    /// Replace an existing installation
    #[arg(long)]
    force: bool,
}

#[derive(Debug, Args)]
pub struct ModelsInspectCommand {
    /// Catalog id/alias or local Tongues package path
    target: String,
    /// Additional private/local catalog JSON files
    #[arg(long = "catalog")]
    catalogs: Vec<PathBuf>,
}

#[derive(Debug, Args)]
pub struct ModelsVerifyCommand {
    /// Catalog model id or alias
    #[arg(required_unless_present = "all", conflicts_with = "all")]
    model: Option<String>,
    /// Deeply verify every installed catalog model
    #[arg(long)]
    all: bool,
    /// Additional private/local catalog JSON files
    #[arg(long = "catalog")]
    catalogs: Vec<PathBuf>,
}

#[derive(Debug, Args)]
pub struct ModelsRemoveCommand {
    /// Catalog model id or alias
    model: String,
    /// Additional private/local catalog JSON files
    #[arg(long = "catalog")]
    catalogs: Vec<PathBuf>,
    /// Also remove the verified download cache
    #[arg(long)]
    purge_cache: bool,
}

#[derive(Debug, Args)]
pub struct ModelsImportCoquiCommand {
    /// Coqui JSON or JSON5 configuration
    #[arg(long)]
    config: PathBuf,
    /// Modern ZIP-based PyTorch checkpoint, or a supported legacy MelGAN checkpoint
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
    /// XTTS tokenizer vocab.json (required when model=xtts)
    #[arg(long)]
    tokenizer: Option<PathBuf>,
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

#[derive(Debug, Args)]
pub struct ModelsGenerateFairseqCatalogCommand {
    /// Checksum/source metadata JSON for all MMS model files
    #[arg(long)]
    source: PathBuf,
    /// Optional downloaded all-tts-languages.html used to detect drift
    #[arg(long = "language-index")]
    language_index: Option<PathBuf>,
    /// Destination catalog JSON
    #[arg(long)]
    out: PathBuf,
}

pub fn run(command: Option<ModelsCommand>) -> Result<()> {
    match command.unwrap_or(ModelsCommand::Menu) {
        ModelsCommand::Menu => model_menu(),
        ModelsCommand::List(command) => list_models(command),
        ModelsCommand::Search(command) => search_models(command),
        ModelsCommand::Install(command) => install_model(command),
        ModelsCommand::Inspect(command) => inspect_model(command),
        ModelsCommand::Verify(command) => verify_models(command),
        ModelsCommand::Remove(command) => remove_model(command),
        ModelsCommand::Path(command) => print_paths(command.model.as_deref()),
        ModelsCommand::Status => print_status(),
        ModelsCommand::Use(command) => select_model(&command.model),
        ModelsCommand::Fetch(command) => {
            if command.all {
                fetch_all_models(command.force)?;
            } else {
                fetch_model(command.model.as_deref(), command.force)?;
            }
            Ok(())
        }
        ModelsCommand::ImportCoqui(command) => import_coqui(command),
        ModelsCommand::InspectPackage(command) => inspect_package(command),
        ModelsCommand::GenerateFairseqCatalog(command) => generate_fairseq_catalog(command),
    }
}

fn generate_fairseq_catalog(command: ModelsGenerateFairseqCatalogCommand) -> Result<()> {
    eprintln!("fairseq-catalog: reading {}", command.source.display());
    let source_text = fs::read_to_string(&command.source)
        .with_context(|| format!("failed to read {}", command.source.display()))?;
    let source = tongues_tts::FairseqCatalogSource::from_json(&source_text)
        .with_context(|| format!("invalid source snapshot {}", command.source.display()))?;
    if let Some(language_index) = command.language_index {
        eprintln!(
            "fairseq-catalog: checking upstream language ids against {}",
            language_index.display()
        );
        let html = fs::read_to_string(&language_index)
            .with_context(|| format!("failed to read {}", language_index.display()))?;
        source.ensure_matches_language_index(&html)?;
    }
    let catalog = tongues_tts::generate_fairseq_catalog(&source)?;
    let part = command.out.with_extension(
        command
            .out
            .extension()
            .and_then(|extension| extension.to_str())
            .map_or_else(
                || "part".to_string(),
                |extension| format!("{extension}.part"),
            ),
    );
    if let Some(parent) = command.out.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    eprintln!(
        "fairseq-catalog: writing {} entries to {}",
        catalog.entries.len(),
        part.display()
    );
    let file =
        File::create(&part).with_context(|| format!("failed to create {}", part.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &catalog)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&part, &command.out).with_context(|| {
        format!(
            "failed to finalize {} from {}",
            command.out.display(),
            part.display()
        )
    })?;
    eprintln!(
        "fairseq-catalog: complete {} entries at {}",
        catalog.entries.len(),
        command.out.display()
    );
    Ok(())
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
    options.tokenizer_path = command.tokenizer;
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

fn load_catalog(extra: Vec<PathBuf>) -> Result<tongues_tts::ModelCatalog> {
    let mut paths = tongues_tts::private_catalog_paths_from_environment();
    paths.extend(extra);
    tongues_tts::ModelCatalog::with_private_catalogs(&paths)
}

fn model_store(offline: bool) -> Result<tongues_tts::ModelStore> {
    let store = tongues_tts::ModelStore::from_environment()?;
    let offline = offline || store.offline();
    Ok(store.with_offline(offline))
}

fn report_install_progress(event: tongues_tts::ModelInstallProgress) {
    match event {
        tongues_tts::ModelInstallProgress::CheckingCache { path } => {
            eprintln!("install: checking cache {}", path.display());
        }
        tongues_tts::ModelInstallProgress::Downloading {
            url,
            downloaded,
            total,
            part_path,
        } => {
            eprintln!(
                "install: {downloaded}/{total} bytes from {url} -> {}",
                part_path.display()
            );
        }
        tongues_tts::ModelInstallProgress::Verifying { path } => {
            eprintln!("install: verifying {}", path.display());
        }
        tongues_tts::ModelInstallProgress::Installing { path } => {
            eprintln!("install: writing {}", path.display());
        }
        tongues_tts::ModelInstallProgress::Complete { id, version } => {
            eprintln!("install: complete {id} package-version={version}");
        }
    }
}

fn install_model(command: ModelsInstallCommand) -> Result<()> {
    let store = model_store(command.offline)?;
    if let Some(package) = command.package {
        let id = command.id.context("--id is required with --package")?;
        let verified = store.install_local_package(&id, package, command.force)?;
        println!(
            "installed {} v{} at {}",
            verified.id,
            verified.package_version,
            verified
                .package_path
                .as_deref()
                .context("local package installation has no package path")?
                .display()
        );
        return Ok(());
    }

    let model = command
        .model
        .context("provide a catalog model id or use --package PATH --id ID")?;
    let catalog = load_catalog(command.catalogs)?;
    let entry = catalog
        .find(&model)
        .with_context(|| format!("unknown catalog model `{model}`"))?;
    let verified = store.install_with_progress(entry, command.force, report_install_progress)?;
    println!(
        "installed {} v{} ({} verified files)",
        verified.id,
        verified.package_version,
        verified.files.len()
    );
    Ok(())
}

fn inspect_model(command: ModelsInspectCommand) -> Result<()> {
    let path = PathBuf::from(&command.target);
    if path.exists() {
        let package = tongues_tts::open_model_package(path)?;
        println!("{}", serde_json::to_string_pretty(&package.manifest)?);
        return Ok(());
    }
    let catalog = load_catalog(command.catalogs)?;
    let store = model_store(true)?;
    let entry = catalog
        .find(&command.target)
        .cloned()
        .or_else(|| {
            store
                .installed_records()
                .ok()?
                .into_iter()
                .find(|record| record.entry.id == command.target)
                .map(|record| record.entry)
        })
        .with_context(|| format!("unknown catalog or installed model `{}`", command.target))?;
    let (installed, verification_error) = match store.verify(&entry) {
        Ok(_) => (true, None),
        Err(error) => (false, Some(format!("{error:#}"))),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "entry": entry,
            "installed": installed,
            "verification_error": verification_error,
            "model_home": store.root(),
            "cache": store.cache(),
            "offline": store.offline(),
        }))?
    );
    Ok(())
}

fn verify_models(command: ModelsVerifyCommand) -> Result<()> {
    let catalog = load_catalog(command.catalogs)?;
    let store = model_store(true)?;
    let entries = if command.all {
        let recorded = store.installed_records()?;
        recorded
            .into_iter()
            .map(|record| {
                catalog
                    .find(&record.entry.id)
                    .cloned()
                    .unwrap_or(record.entry)
            })
            .collect::<Vec<_>>()
    } else {
        let model = command.model.context("provide a model id or use --all")?;
        vec![catalog
            .find(&model)
            .cloned()
            .or_else(|| {
                store
                    .installed_records()
                    .ok()?
                    .into_iter()
                    .find(|record| record.entry.id == model)
                    .map(|record| record.entry)
            })
            .with_context(|| format!("unknown catalog or installed model `{model}`"))?]
    };
    if entries.is_empty() {
        println!("no installed catalog models to verify");
        return Ok(());
    }
    let mut failures = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        eprintln!(
            "verify: {}/{} {} v{}",
            index + 1,
            entries.len(),
            entry.id,
            entry.package_version
        );
        match store.verify(entry) {
            Ok(verified) => println!(
                "verified {} v{} ({} files)",
                verified.id,
                verified.package_version,
                verified.files.len()
            ),
            Err(error) => failures.push(format!("{}: {error:#}", entry.id)),
        }
    }
    anyhow::ensure!(
        failures.is_empty(),
        "model verification failed:\n{}",
        failures.join("\n")
    );
    Ok(())
}

fn remove_model(command: ModelsRemoveCommand) -> Result<()> {
    let catalog = load_catalog(command.catalogs)?;
    let store = model_store(true)?;
    let entry = catalog
        .find(&command.model)
        .cloned()
        .or_else(|| {
            store
                .installed_records()
                .ok()?
                .into_iter()
                .find(|record| record.entry.id == command.model)
                .map(|record| record.entry)
        })
        .with_context(|| format!("unknown catalog or installed model `{}`", command.model))?;
    store.remove(&entry, command.purge_cache)?;
    println!(
        "removed {} v{}{}",
        entry.id,
        entry.package_version,
        if command.purge_cache {
            " and cached artifacts"
        } else {
            ""
        }
    );
    Ok(())
}

fn search_models(command: ModelsSearchCommand) -> Result<()> {
    let catalog = load_catalog(command.catalogs)?;
    let matches = catalog.search(&command.query);
    if command.json {
        println!("{}", serde_json::to_string_pretty(&matches)?);
    } else {
        for entry in matches {
            println!(
                "{:<28} {:<24} {} [{}]",
                entry.id, entry.architecture, entry.display_name, entry.license.expression
            );
        }
    }
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

fn list_models(command: ModelsListCommand) -> Result<()> {
    let catalog = load_catalog(command.catalogs)?;
    let store = model_store(true)?;
    if command.json {
        let entries = catalog
            .entries
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "entry": entry,
                    "verification": store.verification_state(entry),
                })
            })
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": catalog.schema_version,
                "catalog": catalog.id,
                "entries": entries,
            }))?
        );
        return Ok(());
    }
    println!(
        "{} (home: {}, cache: {}, offline: {})",
        "Model catalog".bold(),
        store.root().display(),
        store.cache().display(),
        store.offline()
    );
    for entry in &catalog.entries {
        let state = match store.verification_state(entry).status {
            tongues_tts::ModelVerificationStatus::Verified => "verified".green().to_string(),
            tongues_tts::ModelVerificationStatus::PendingVerification => {
                "pending verification".yellow().to_string()
            }
            tongues_tts::ModelVerificationStatus::ChangedSinceVerification => {
                "changed".yellow().to_string()
            }
            tongues_tts::ModelVerificationStatus::VerificationFailed => "failed".red().to_string(),
            tongues_tts::ModelVerificationStatus::Unavailable => "unavailable".dimmed().to_string(),
        };
        println!(
            "{:<28} {:<24} v{:<3} {:<20} {} [{}]",
            entry.id.bold(),
            entry.architecture,
            entry.package_version,
            state,
            entry.display_name,
            entry.license.expression
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
        for asset in MODEL_ASSETS.iter() {
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
                | ModelKind::VoiceConversion
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
        ModelKind::VoiceConversion => "voice-conversion",
        ModelKind::Lexicon => "lexicon",
        ModelKind::Phonemicizer => "phonemicizer",
    }
}
