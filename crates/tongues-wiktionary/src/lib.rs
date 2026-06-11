//! Wiktionary pronunciation model-family data preparation.
//!
//! This family downloads the English Wiktionary MediaWiki XML dump and expands
//! extracted spelling/IPA pairs into multilingual seq2seq-style training rows.
//! The XML/wikitext extraction itself is intentionally stubbed until the parser
//! policy for Wiktionary pronunciation templates is implemented.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use tongues_core::Vocab;
use tongues_neural::{write_manifest, ModelArtifactManifest};

pub const FAMILY: &str = "wiktionary";
pub const ARCHITECTURE: &str = "wiktionary-pronunciation-seq2seq-scaffold";
pub const DEFAULT_DATASET_ID: &str = "enwiktionary-2026-06-01-v0";
pub const DEFAULT_DUMP_INDEX_URL: &str =
    "https://dumps.wikimedia.org/other/mediawiki_content_current/enwiktionary/2026-06-01/xml/bzip2/";
const USER_AGENT: &str = "tongues-wiktionary/0.1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WiktionaryConfig {
    pub dataset_id: String,
    pub dump_index_url: String,
    #[serde(default)]
    pub dump_file_url: Option<String>,
    pub train_frac: f64,
    pub valid_frac: f64,
    pub seed: u64,
    pub languages: Vec<String>,
    pub include_reverse: bool,
    pub include_language_guessing: bool,
    #[serde(default)]
    pub max_pages: Option<usize>,
}

impl Default for WiktionaryConfig {
    fn default() -> Self {
        Self {
            dataset_id: DEFAULT_DATASET_ID.to_string(),
            dump_index_url: DEFAULT_DUMP_INDEX_URL.to_string(),
            dump_file_url: None,
            train_frac: 0.8,
            valid_frac: 0.1,
            seed: 42,
            languages: ["eng", "fra", "deu", "spa"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            include_reverse: true,
            include_language_guessing: true,
            max_pages: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PronunciationEntry {
    pub lang: String,
    pub spelling: String,
    pub ipa: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingExample {
    pub task: WiktionaryTask,
    pub lang: Option<String>,
    pub input: String,
    pub output: String,
    pub source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WiktionaryTask {
    SpellingToIpa,
    IpaToSpelling,
    GuessLangFromSpelling,
    GuessLangFromIpa,
    GuessLangFromSpellingAndIpa,
}

impl WiktionaryTask {
    pub fn token(self) -> &'static str {
        match self {
            Self::SpellingToIpa => "<WIKT_SPELLING_TO_IPA>",
            Self::IpaToSpelling => "<WIKT_IPA_TO_SPELLING>",
            Self::GuessLangFromSpelling => "<WIKT_GUESS_LANG_FROM_SPELLING>",
            Self::GuessLangFromIpa => "<WIKT_GUESS_LANG_FROM_IPA>",
            Self::GuessLangFromSpellingAndIpa => "<WIKT_GUESS_LANG_FROM_SPELLING_AND_IPA>",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrepareReport {
    pub dump_path: String,
    pub parsed_entries: usize,
    pub train_examples: usize,
    pub valid_examples: usize,
    pub test_examples: usize,
}

pub fn read_config(path: &Path) -> Result<WiktionaryConfig> {
    if !path.exists() {
        return Ok(WiktionaryConfig::default());
    }
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

pub fn prepare_dataset(
    out: &Path,
    cache_dir: &Path,
    config: &WiktionaryConfig,
) -> Result<PrepareReport> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    fs::create_dir_all(cache_dir).with_context(|| format!("creating {}", cache_dir.display()))?;

    let dump_path = download_dump(cache_dir, config)?;
    let entries = parse_dump_stub(&dump_path, config)?;
    let examples = expand_training_examples(&entries, config);
    let (train, valid, test) =
        split_examples(examples, config.train_frac, config.valid_frac, config.seed);

    write_jsonl(&out.join("train.jsonl"), &train)?;
    write_jsonl(&out.join("valid.jsonl"), &valid)?;
    write_jsonl(&out.join("test.jsonl"), &test)?;
    write_vocab(out, [&train[..], &valid[..], &test[..]].concat().as_slice())?;
    fs::write(
        out.join("dataset_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    fs::write(out.join("README.md"), dataset_readme(config, &dump_path))?;

    Ok(PrepareReport {
        dump_path: dump_path.display().to_string(),
        parsed_entries: entries.len(),
        train_examples: train.len(),
        valid_examples: valid.len(),
        test_examples: test.len(),
    })
}

pub fn download_dump(cache_dir: &Path, config: &WiktionaryConfig) -> Result<PathBuf> {
    let dump_url = match &config.dump_file_url {
        Some(url) => url.clone(),
        None => resolve_dump_file_url(&config.dump_index_url)?,
    };
    let filename = dump_url
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .context("dump URL has no filename")?;
    let path = cache_dir.join(filename);
    if path.exists() && path.metadata()?.len() > 0 {
        return Ok(path);
    }
    download_to_file(&dump_url, &path)?;
    Ok(path)
}

pub fn resolve_dump_file_url(index_url: &str) -> Result<String> {
    let response = ureq::get(index_url)
        .header("User-Agent", USER_AGENT)
        .call()
        .with_context(|| format!("GET {index_url}"))?;
    let index = response
        .into_body()
        .read_to_string()
        .with_context(|| format!("reading dump index {index_url}"))?;
    let href = find_dump_href(&index).context("no enwiktionary XML bzip2 dump found in index")?;
    Ok(join_url(index_url, href))
}

fn find_dump_href(index: &str) -> Option<&str> {
    let mut best = None;
    for marker in ["href=\"", "href='"] {
        for chunk in index.split(marker).skip(1) {
            let quote = marker.as_bytes()[5] as char;
            let href = chunk.split(quote).next()?;
            if href.ends_with(".xml.bz2") && href.contains("enwiktionary") {
                best = Some(href);
                if href.contains("pages-articles") || href.contains("pages-meta-current") {
                    return Some(href);
                }
            }
        }
    }
    best
}

fn join_url(base: &str, href: &str) -> String {
    if href.starts_with("http://") || href.starts_with("https://") {
        href.to_string()
    } else {
        format!(
            "{}/{}",
            base.trim_end_matches('/'),
            href.trim_start_matches('/')
        )
    }
}

fn download_to_file(url: &str, path: &Path) -> Result<()> {
    let part_path = path.with_extension("part");
    let response = ureq::get(url)
        .header("User-Agent", USER_AGENT)
        .call()
        .with_context(|| format!("GET {url}"))?;
    let mut body = response.into_body();
    let mut reader = body.as_reader();
    let mut file =
        File::create(&part_path).with_context(|| format!("creating {}", part_path.display()))?;
    let mut buffer = [0_u8; 1024 * 64];
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        file.write_all(&buffer[..n])?;
    }
    file.flush()?;
    anyhow::ensure!(part_path.metadata()?.len() > 0, "empty dump response");
    fs::rename(&part_path, path).with_context(|| {
        format!(
            "moving downloaded dump {} to {}",
            part_path.display(),
            path.display()
        )
    })
}

pub fn parse_dump_stub(
    _dump_path: &Path,
    _config: &WiktionaryConfig,
) -> Result<Vec<PronunciationEntry>> {
    Ok(Vec::new())
}

pub fn expand_training_examples(
    entries: &[PronunciationEntry],
    config: &WiktionaryConfig,
) -> Vec<TrainingExample> {
    let allowed: BTreeSet<&str> = config.languages.iter().map(String::as_str).collect();
    let mut examples = Vec::new();
    for entry in entries {
        if !allowed.contains(entry.lang.as_str()) {
            continue;
        }
        examples.push(TrainingExample {
            task: WiktionaryTask::SpellingToIpa,
            lang: Some(entry.lang.clone()),
            input: format!("lang={} | {}", entry.lang, entry.spelling),
            output: entry.ipa.clone(),
            source: "enwiktionary".to_string(),
        });
        if config.include_reverse {
            examples.push(TrainingExample {
                task: WiktionaryTask::IpaToSpelling,
                lang: Some(entry.lang.clone()),
                input: format!("lang={} | {}", entry.lang, entry.ipa),
                output: entry.spelling.clone(),
                source: "enwiktionary".to_string(),
            });
        }
        if config.include_language_guessing {
            examples.push(TrainingExample {
                task: WiktionaryTask::GuessLangFromSpelling,
                lang: None,
                input: format!("lang=<MASK> | spelling={}", entry.spelling),
                output: entry.lang.clone(),
                source: "enwiktionary".to_string(),
            });
            examples.push(TrainingExample {
                task: WiktionaryTask::GuessLangFromIpa,
                lang: None,
                input: format!("lang=<MASK> | ipa={}", entry.ipa),
                output: entry.lang.clone(),
                source: "enwiktionary".to_string(),
            });
            examples.push(TrainingExample {
                task: WiktionaryTask::GuessLangFromSpellingAndIpa,
                lang: None,
                input: format!(
                    "lang=<MASK> | spelling={} | ipa={}",
                    entry.spelling, entry.ipa
                ),
                output: entry.lang.clone(),
                source: "enwiktionary".to_string(),
            });
        }
    }
    examples
}

fn split_examples(
    mut examples: Vec<TrainingExample>,
    train_frac: f64,
    valid_frac: f64,
    seed: u64,
) -> (
    Vec<TrainingExample>,
    Vec<TrainingExample>,
    Vec<TrainingExample>,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    examples.shuffle(&mut rng);
    let train_len = ((examples.len() as f64) * train_frac).round() as usize;
    let valid_len = ((examples.len() as f64) * valid_frac).round() as usize;
    let train_end = train_len.min(examples.len());
    let valid_end = (train_end + valid_len).min(examples.len());
    let test = examples.split_off(valid_end);
    let valid = examples.split_off(train_end);
    (examples, valid, test)
}

fn write_jsonl(path: &Path, examples: &[TrainingExample]) -> Result<()> {
    let mut file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    for example in examples {
        writeln!(file, "{}", serde_json::to_string(example)?)?;
    }
    Ok(())
}

fn write_vocab(out: &Path, examples: &[TrainingExample]) -> Result<()> {
    let inputs = examples
        .iter()
        .map(|example| format!("{} {}", example.task.token(), example.input))
        .collect::<Vec<_>>();
    let outputs = examples
        .iter()
        .map(|example| example.output.clone())
        .collect::<Vec<_>>();
    let vocab = Vocab::build(&inputs, &outputs, &[]);
    fs::write(
        out.join("vocab.json"),
        serde_json::to_string_pretty(&vocab)?,
    )?;
    Ok(())
}

fn dataset_readme(config: &WiktionaryConfig, dump_path: &Path) -> String {
    format!(
        "# Wiktionary pronunciation dataset\n\nSource dump: `{}`\n\nConfigured languages: {}\n\nThe MediaWiki XML/wikitext pronunciation parser is currently stubbed, so prepare creates the split files and vocabulary schema but emits zero examples until parsing is implemented.\n\nTraining row shape:\n\n```json\n{{\"task\":\"spelling-to-ipa\",\"lang\":\"eng\",\"input\":\"lang=eng | champ\",\"output\":\"/ʃɑ̃/\",\"source\":\"enwiktionary\"}}\n```\n\nReverse and language-guessing rows are controlled by `include_reverse` and `include_language_guessing`.\n",
        dump_path.display(),
        config.languages.join(", ")
    )
}

pub fn write_scaffold_model(out: &Path, config: &WiktionaryConfig) -> Result<()> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    fs::write(out.join("model.bin"), b"wiktionary scaffold\n")?;
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
    fn default_config_targets_requested_dump_and_languages() {
        let config = WiktionaryConfig::default();
        assert_eq!(config.dataset_id, DEFAULT_DATASET_ID);
        assert_eq!(config.dump_index_url, DEFAULT_DUMP_INDEX_URL);
        assert_eq!(config.languages, ["eng", "fra", "deu", "spa"]);
    }

    #[test]
    fn expands_forward_reverse_and_language_guessing_tasks() {
        let config = WiktionaryConfig::default();
        let examples = expand_training_examples(
            &[PronunciationEntry {
                lang: "deu".to_string(),
                spelling: "schief".to_string(),
                ipa: "/ʃiːf/".to_string(),
            }],
            &config,
        );
        assert_eq!(examples.len(), 5);
        assert!(examples.iter().any(|example| {
            example.task == WiktionaryTask::SpellingToIpa
                && example.input == "lang=deu | schief"
                && example.output == "/ʃiːf/"
        }));
        assert!(examples.iter().any(|example| {
            example.task == WiktionaryTask::IpaToSpelling
                && example.input == "lang=deu | /ʃiːf/"
                && example.output == "schief"
        }));
    }

    #[test]
    fn finds_dump_href_from_index() {
        let index = r#"<a href="enwiktionary-20260601-pages-meta-current.xml.bz2">dump</a>"#;
        assert_eq!(
            find_dump_href(index),
            Some("enwiktionary-20260601-pages-meta-current.xml.bz2")
        );
    }
}
