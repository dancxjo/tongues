use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant};

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
        Some("race") => {
            let args = args.collect::<Vec<_>>();
            if args.iter().any(|arg| arg == "-h" || arg == "--help") {
                print!("{}", race_usage());
                Ok(())
            } else {
                race(args)
            }
        }
        Some("-h") | Some("--help") | None => {
            print!("{}", usage());
            Ok(())
        }
        Some(command) => Err(format!("unknown xtask command `{command}`\n\n{}", usage())),
    }
}

fn usage() -> &'static str {
    "Usage: cargo xtask <command>\n\nCommands:\n  new-family <family-slug>  Create a model-family scaffold\n  race [options] [words...] Run round-trip inference benchmarks\n"
}

fn race_usage() -> &'static str {
    "Usage: cargo xtask race [options] [words...]\n\nOptions:\n  --cpu                         Force CPU inference\n  --skip-build                  Do not build the tongues binary first\n  --g2p2g-model <path>          G2P2G model dir (default: models/g2p2g/openepd-v0)\n  --wiktionary-model <path>     Wiktionary model dir (default: models/wiktionary/enwiktionary-2026-06-01-v0-phones)\n  --wiktionary-config <path>    Wiktionary config (default: configs/wiktionary/default.toml)\n"
}

#[derive(Debug)]
struct RaceConfig {
    cpu: bool,
    skip_build: bool,
    g2p2g_model: PathBuf,
    wiktionary_model: PathBuf,
    wiktionary_config: PathBuf,
    words: Vec<String>,
}

#[derive(Debug)]
struct RaceResult {
    output: String,
    elapsed: Duration,
}

#[derive(Debug)]
struct RaceStats {
    runs: usize,
    failures: usize,
    total: Duration,
}

struct WiktionaryInferDemo<'a> {
    label: &'a str,
    task: &'a str,
    lang: &'a str,
    notation: &'a str,
    accent: Option<&'a str>,
    raw: bool,
    input: String,
}

impl RaceStats {
    fn new() -> Self {
        Self {
            runs: 0,
            failures: 0,
            total: Duration::ZERO,
        }
    }

    fn record(&mut self, elapsed: Duration) {
        self.runs += 1;
        self.total += elapsed;
    }

    fn fail(&mut self) {
        self.failures += 1;
    }
}

fn race(raw_args: Vec<String>) -> Result<(), String> {
    let config = parse_race_args(raw_args)?;
    let languages = read_wiktionary_languages(&config.wiktionary_config)?;
    let words = if config.words.is_empty() {
        default_race_words()
    } else {
        config.words.clone()
    };

    if !config.skip_build {
        println!("race: building tongues binary");
        run_build()?;
    }

    let tongues = tongues_bin_path();
    if !tongues.exists() {
        return Err(format!(
            "{} does not exist; run without --skip-build first",
            tongues.display()
        ));
    }

    println!(
        "race: {} words, {} Wiktionary languages, phones+phonemes",
        words.len(),
        languages.len()
    );
    println!(
        "race: g2p2g={}, wiktionary={}",
        config.g2p2g_model.display(),
        config.wiktionary_model.display()
    );

    let total_start = Instant::now();
    let mut stats = RaceStats::new();

    println!();
    println!("G2P2G round trips");
    for word in &words {
        match round_trip_g2p2g(&tongues, &config, word) {
            Ok((forward, reverse)) => {
                stats.record(forward.elapsed + reverse.elapsed);
                println!(
                    "  ok {:>6} + {:>6}  {:<14} -> {:<18} -> {}",
                    fmt_ms(forward.elapsed),
                    fmt_ms(reverse.elapsed),
                    clip(word, 14),
                    clip(&forward.output, 18),
                    clip(&reverse.output, 18)
                );
            }
            Err(error) => {
                stats.fail();
                println!("  fail {:<14} {}", clip(word, 14), error);
            }
        }
    }

    println!();
    println!("Wiktionary round trips");
    for word in &words {
        for lang in &languages {
            for notation in ["phones", "phonemes"] {
                match round_trip_wiktionary(&tongues, &config, word, lang, notation) {
                    Ok((forward, reverse)) => {
                        stats.record(forward.elapsed + reverse.elapsed);
                        println!(
                            "  ok {:>6} + {:>6}  {:<3}/{:<8} {:<14} -> {:<18} -> {}",
                            fmt_ms(forward.elapsed),
                            fmt_ms(reverse.elapsed),
                            lang,
                            notation,
                            clip(word, 14),
                            clip(&forward.output, 18),
                            clip(&reverse.output, 18)
                        );
                    }
                    Err(error) => {
                        stats.fail();
                        println!(
                            "  fail {:<3}/{:<8} {:<14} {}",
                            lang,
                            notation,
                            clip(word, 14),
                            error
                        );
                    }
                }
            }
        }
    }

    println!();
    println!("Wiktionary task demos");
    let demo_word = words
        .iter()
        .find(|word| word.as_str() == "cat")
        .unwrap_or(&words[0]);
    let demo_lang = languages
        .iter()
        .find(|lang| lang.as_str() == "eng")
        .unwrap_or(&languages[0]);
    match run_wiktionary_infer(
        &tongues,
        &config,
        "spelling-to-ipa",
        demo_lang,
        "phones",
        Some("RP"),
        false,
        demo_word,
    ) {
        Ok(pronunciation) => {
            stats.record(pronunciation.elapsed);
            println!(
                "  ok {:>6}  {:<38} {} -> {}",
                fmt_ms(pronunciation.elapsed),
                "spelling-to-ipa --accent RP",
                clip(demo_word, 14),
                clip(&pronunciation.output, 28)
            );

            for demo in [
                WiktionaryInferDemo {
                    label: "normalize",
                    task: "normalize",
                    lang: demo_lang,
                    notation: "phones",
                    accent: None,
                    raw: false,
                    input: demo_word.to_string(),
                },
                WiktionaryInferDemo {
                    label: "guess-lang-from-spelling",
                    task: "guess-lang-from-spelling",
                    lang: demo_lang,
                    notation: "phones",
                    accent: None,
                    raw: false,
                    input: demo_word.to_string(),
                },
                WiktionaryInferDemo {
                    label: "guess-lang-from-ipa",
                    task: "guess-lang-from-ipa",
                    lang: demo_lang,
                    notation: "phones",
                    accent: None,
                    raw: false,
                    input: pronunciation.output.clone(),
                },
                WiktionaryInferDemo {
                    label: "guess-lang-from-spelling-and-ipa",
                    task: "guess-lang-from-spelling-and-ipa",
                    lang: demo_lang,
                    notation: "phones",
                    accent: None,
                    raw: false,
                    input: format!("{demo_word} => {}", pronunciation.output),
                },
                WiktionaryInferDemo {
                    label: "--raw tagged source",
                    task: "spelling-to-ipa",
                    lang: demo_lang,
                    notation: "phones",
                    accent: None,
                    raw: true,
                    input: format!("<task:g2p> <lang:{demo_lang}> <N_PHONE> {demo_word}"),
                },
            ] {
                match run_wiktionary_infer(
                    &tongues,
                    &config,
                    demo.task,
                    demo.lang,
                    demo.notation,
                    demo.accent,
                    demo.raw,
                    &demo.input,
                ) {
                    Ok(result) => {
                        stats.record(result.elapsed);
                        println!(
                            "  ok {:>6}  {:<38} {} -> {}",
                            fmt_ms(result.elapsed),
                            demo.label,
                            clip(&demo.input, 28),
                            clip(&result.output, 28)
                        );
                    }
                    Err(error) => {
                        stats.fail();
                        println!("  fail {:<38} {}", demo.label, error);
                    }
                }
            }
        }
        Err(error) => {
            stats.fail();
            println!("  fail {:<38} {}", "spelling-to-ipa --accent RP", error);
        }
    }

    println!();
    println!(
        "race: done in {} wall; {} successful inference demos, {} failures, {} summed inference time",
        fmt_ms(total_start.elapsed()),
        stats.runs,
        stats.failures,
        fmt_ms(stats.total)
    );

    Ok(())
}

fn parse_race_args(args: Vec<String>) -> Result<RaceConfig, String> {
    let mut config = RaceConfig {
        cpu: false,
        skip_build: false,
        g2p2g_model: PathBuf::from("models/g2p2g/openepd-v0"),
        wiktionary_model: PathBuf::from("models/wiktionary/enwiktionary-2026-06-01-v0-phones"),
        wiktionary_config: PathBuf::from("configs/wiktionary/default.toml"),
        words: Vec::new(),
    };

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--cpu" => config.cpu = true,
            "--skip-build" => config.skip_build = true,
            "--g2p2g-model" => {
                config.g2p2g_model = PathBuf::from(next_race_value(&mut iter, "--g2p2g-model")?);
            }
            "--wiktionary-model" => {
                config.wiktionary_model =
                    PathBuf::from(next_race_value(&mut iter, "--wiktionary-model")?);
            }
            "--wiktionary-config" => {
                config.wiktionary_config =
                    PathBuf::from(next_race_value(&mut iter, "--wiktionary-config")?);
            }
            _ if arg.starts_with("--") => {
                return Err(format!("unknown race option `{arg}`\n\n{}", race_usage()));
            }
            _ => config.words.push(arg),
        }
    }

    Ok(config)
}

fn next_race_value(
    iter: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("{option} requires a value\n\n{}", race_usage()))
}

fn run_build() -> Result<(), String> {
    let status = Command::new("cargo")
        .args(["build", "--quiet", "--bin", "tongues"])
        .status()
        .map_err(|error| format!("starting cargo build: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("cargo build failed with {status}"))
    }
}

fn round_trip_g2p2g(
    tongues: &Path,
    config: &RaceConfig,
    word: &str,
) -> Result<(RaceResult, RaceResult), String> {
    let mut args = base_tongues_args(config);
    args.extend([
        "g2p2g".to_string(),
        "infer".to_string(),
        "--task".to_string(),
        "g2p".to_string(),
        "--model".to_string(),
        config.g2p2g_model.display().to_string(),
        "--".to_string(),
        word.to_string(),
    ]);
    let forward = run_infer(tongues, &args)?;

    let mut reverse_args = base_tongues_args(config);
    reverse_args.extend([
        "g2p2g".to_string(),
        "infer".to_string(),
        "--task".to_string(),
        "p2g".to_string(),
        "--model".to_string(),
        config.g2p2g_model.display().to_string(),
        "--".to_string(),
        forward.output.clone(),
    ]);
    let reverse = run_infer(tongues, &reverse_args)?;
    Ok((forward, reverse))
}

fn round_trip_wiktionary(
    tongues: &Path,
    config: &RaceConfig,
    word: &str,
    lang: &str,
    notation: &str,
) -> Result<(RaceResult, RaceResult), String> {
    let mut args = base_tongues_args(config);
    args.extend([
        "wiktionary".to_string(),
        "infer".to_string(),
        "--model".to_string(),
        config.wiktionary_model.display().to_string(),
        "--task".to_string(),
        "spelling-to-ipa".to_string(),
        "--lang".to_string(),
        lang.to_string(),
        "--notation".to_string(),
        notation.to_string(),
        "--".to_string(),
        word.to_string(),
    ]);
    let forward = run_infer(tongues, &args)?;

    let mut reverse_args = base_tongues_args(config);
    reverse_args.extend([
        "wiktionary".to_string(),
        "infer".to_string(),
        "--model".to_string(),
        config.wiktionary_model.display().to_string(),
        "--task".to_string(),
        "ipa-to-spelling".to_string(),
        "--lang".to_string(),
        lang.to_string(),
        "--notation".to_string(),
        notation.to_string(),
        "--".to_string(),
        forward.output.clone(),
    ]);
    let reverse = run_infer(tongues, &reverse_args)?;
    Ok((forward, reverse))
}

#[allow(clippy::too_many_arguments)]
fn run_wiktionary_infer(
    tongues: &Path,
    config: &RaceConfig,
    task: &str,
    lang: &str,
    notation: &str,
    accent: Option<&str>,
    raw: bool,
    input: &str,
) -> Result<RaceResult, String> {
    let mut args = base_tongues_args(config);
    args.extend([
        "wiktionary".to_string(),
        "infer".to_string(),
        "--model".to_string(),
        config.wiktionary_model.display().to_string(),
        "--task".to_string(),
        task.to_string(),
        "--lang".to_string(),
        lang.to_string(),
        "--notation".to_string(),
        notation.to_string(),
    ]);
    if let Some(accent) = accent {
        args.extend(["--accent".to_string(), accent.to_string()]);
    }
    if raw {
        args.push("--raw".to_string());
    }
    args.extend(["--".to_string(), input.to_string()]);
    run_infer(tongues, &args)
}

fn base_tongues_args(config: &RaceConfig) -> Vec<String> {
    if config.cpu {
        vec!["--cpu".to_string()]
    } else {
        Vec::new()
    }
}

fn run_infer(tongues: &Path, args: &[String]) -> Result<RaceResult, String> {
    let start = Instant::now();
    let output = Command::new(tongues)
        .args(args)
        .output()
        .map_err(|error| format!("starting {}: {error}", tongues.display()))?;
    let elapsed = start.elapsed();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "exited {}: {}",
            output.status,
            clip(stderr.trim(), 80)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let prediction = extract_prediction(&stdout)
        .ok_or_else(|| format!("prediction output not found: {}", clip(stdout.trim(), 80)))?;
    Ok(RaceResult {
        output: prediction,
        elapsed,
    })
}

fn extract_prediction(stdout: &str) -> Option<String> {
    let mut lines = stdout.lines();
    while let Some(line) = lines.next() {
        if line.trim() == "Prediction output:" {
            return lines.next().map(|value| value.trim().to_string());
        }
    }
    None
}

fn read_wiktionary_languages(path: &Path) -> Result<Vec<String>, String> {
    let raw =
        fs::read_to_string(path).map_err(|error| format!("reading {}: {error}", path.display()))?;
    for line in raw.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("languages") {
            let Some((_, value)) = rest.split_once('=') else {
                continue;
            };
            return parse_toml_string_array(value)
                .ok_or_else(|| format!("could not parse languages in {}", path.display()));
        }
    }
    Ok(vec!["eng".to_string()])
}

fn parse_toml_string_array(value: &str) -> Option<Vec<String>> {
    let start = value.find('[')?;
    let end = value.rfind(']')?;
    let inner = &value[start + 1..end];
    let mut out = Vec::new();
    for item in inner.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        out.push(item.trim_matches('"').to_string());
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn default_race_words() -> Vec<String> {
    [
        "have",
        "cat",
        "cats",
        "walked",
        "running",
        "children",
        "read",
        "lead",
        "wind",
        "record",
        "through",
        "tough",
        "queue",
        "knight",
        "psychology",
        "xylophone",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn tongues_bin_path() -> PathBuf {
    PathBuf::from("target")
        .join("debug")
        .join(format!("tongues{}", env::consts::EXE_SUFFIX))
}

fn fmt_ms(duration: Duration) -> String {
    format!("{}ms", duration.as_millis())
}

fn clip(value: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (index, ch) in value.chars().enumerate() {
        if index >= max_chars {
            out.push_str("...");
            return out;
        }
        out.push(ch);
    }
    out
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
